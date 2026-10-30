//! Controller side: the node list in nodes.yaml and the ssh tunnel to each node's daemon

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{bail, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::{sleep, timeout, Duration, Instant};

use crate::executor::remote::RemoteExecutor;
use crate::executor::shell::ShellExecutor;
use crate::executor::Executor;
use crate::ipc::client;
use crate::ipc::protocol::{NodeInfo, Request, Response};
use crate::paths::node_socket_path;
use crate::resource::discovery::SystemResources;
use crate::resource::pool::ResourcePool;

/// One entry of nodes.yaml, written by `roster node add`; file order is first-fit order
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ssh: Option<String>, // ssh destination, e.g. user@host or a ~/.ssh/config alias; None = this machine
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub socket: Option<String>, // the node daemon's socket, from `roster info`
}

/// A node the scheduler can place on
pub struct Node {
    pub name:     String,
    pub ssh:      Option<String>,
    pub executor: Arc<dyn Executor>,
    pub socket:   Option<PathBuf>,    // local end of the tunnel, for requests other than jobs; None for this machine
    online:       Arc<AtomicBool>,    // false while the tunnel or the node daemon is down
    health:       Arc<Mutex<Health>>, // written by the tunnel task
}

impl Node {
    /// Send a request other than a job operation to this node's daemon: through its tunnel,
    /// or this machine's own socket
    pub async fn request(&self, request: Request) -> anyhow::Result<Response> {
        match &self.socket {
            Some(socket) => client::exchange(UnixStream::connect(socket).await?, request).await,
            None => client::send(request).await,
        }
    }

    pub fn is_online(&self) -> bool {
        self.online.load(Ordering::Relaxed)
    }

    pub async fn health(&self) -> Health {
        self.health.lock().await.clone()
    }
}

/// Where a node stands; each tunnel event handler returns the next state
#[derive(Debug, Clone)]
pub enum NodeState {
    Online,
    TunnelDown(String),      // ssh exited or its local socket is missing; ssh's own error is in daemon.log
    DaemonDown,              // the node is reachable but nothing answers on the daemon's socket
    NoReply,                 // a request went out and nothing came back in 3 s: network stalled or daemon hung
    VersionMismatch(String), // the node daemon runs another roster version, so nothing is launched there
}

impl NodeState {
    pub fn is_online(&self) -> bool {
        matches!(self, NodeState::Online)
    }
}

impl fmt::Display for NodeState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NodeState::Online                  => write!(f, "online"),
            NodeState::TunnelDown(reason)      => write!(f, "ssh tunnel down: {}", reason),
            NodeState::DaemonDown              => write!(f, "node reachable, roster daemon not running"),
            NodeState::NoReply                 => write!(f, "no reply in 3 s (network stalled or daemon hung)"),
            NodeState::VersionMismatch(reason) => write!(f, "version mismatch: {}", reason),
        }
    }
}

/// A node's state and when it began, for `node ls` and `info NODE`
/// the scheduler reads `online` instead, which a failed request also clears at once
#[derive(Debug, Clone)]
pub struct Health {
    pub state: NodeState,
    pub since: DateTime<Utc>,    // when the node entered this kind of state
    pub info:  Option<NodeInfo>, // last answer from the node daemon; None for this machine
}

impl Health {
    /// Take the next state; since moves only when the kind of state changes, not its reason
    /// returns whether it changed
    fn enter(&mut self, state: NodeState, info: Option<NodeInfo>) -> bool {
        if info.is_some() {
            self.info = info;
        }
        let changed = std::mem::discriminant(&self.state) != std::mem::discriminant(&state);
        if changed {
            self.since = Utc::now();
        }
        self.state = state;
        changed
    }
}

fn local() -> NodeConfig {
    NodeConfig { name: "local".into(), ssh: None, socket: None }
}

/// Read nodes.yaml; without the file, this machine is the only node
pub fn load(path: &Path) -> anyhow::Result<Vec<NodeConfig>> {
    let yaml = match std::fs::read_to_string(path) {
        Ok(yaml) => yaml,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![local()]),
        Err(error) => return Err(error).with_context(|| format!("failed to read {}", path.display())),
    };
    let configs: Vec<NodeConfig> = serde_yaml::from_str(&yaml)
        .with_context(|| format!("invalid nodes file {}", path.display()))?;

    if configs.is_empty() {
        bail!("{} lists no nodes", path.display());
    }
    let mut names = HashSet::new();
    for config in &configs {
        if !names.insert(&config.name) {
            bail!("{} lists node '{}' twice", path.display(), config.name);
        }
    }
    Ok(configs)
}

/// Write nodes.yaml through a temp file so a crash never leaves half a file
pub fn save(path: &Path, configs: &[NodeConfig]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("yaml.tmp");
    std::fs::write(&temp, serde_yaml::to_string(configs)?)?;
    std::fs::rename(&temp, path)?;
    Ok(())
}

/// `roster node add`: enable the node's daemon and read its info over ssh, then append it to nodes.yaml
/// the controller daemon picks it up on restart
pub async fn add(path: &Path, target: &str, name: Option<String>) -> anyhow::Result<NodeInfo> {
    let mut configs = load(path)?;
    let name = name.unwrap_or_else(|| target.to_string());

    // the name becomes a socket file name
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
        bail!("node name '{}' may only use letters, digits, '-', '_' and '.'", name);
    }
    if configs.iter().any(|config| config.name == name || config.ssh.as_deref() == Some(target)) {
        bail!("node '{}' ({}) is already in {}", name, target, path.display());
    }

    let info = info_over_ssh(target).await?;
    configs.push(NodeConfig { name, ssh: Some(target.into()), socket: Some(info.socket.clone()) });
    save(path, &configs)?;
    Ok(info)
}

/// `roster node rm`
pub fn remove(path: &Path, name: &str) -> anyhow::Result<()> {
    let mut configs = load(path)?;
    let before = configs.len();
    configs.retain(|config| config.name != name);

    if configs.len() == before {
        bail!("no node '{}' in {}", name, path.display());
    }
    if configs.is_empty() {
        bail!("'{}' is the last node", name);
    }
    save(path, &configs)
}

/// Enable the node's daemon (a no-op if already enabled) and read its info; needs roster on the node's PATH
/// one ssh session; enable's own output is dropped so stdout carries only the JSON
async fn info_over_ssh(target: &str) -> anyhow::Result<NodeInfo> {
    let remote = "roster enable > /dev/null && roster info --json";
    let output = Command::new("ssh")
        .args(["-o", "LogLevel=ERROR", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote])
        .stdin(Stdio::null())
        .output()
        .await
        .context("failed to run ssh")?;

    if !output.status.success() {
        bail!("`ssh {} '{}'` failed ({}): {}",
              target, remote, output.status, String::from_utf8_lossy(&output.stderr).trim());
    }
    let info: NodeInfo = serde_json::from_slice(&output.stdout)
        .context("unreadable `roster info --json` output")?;
    check_version(&info)?;
    Ok(info)
}

/// The protocol has no version number yet, so both ends must run the same build version
fn check_version(info: &NodeInfo) -> anyhow::Result<()> {
    if info.version != env!("CARGO_PKG_VERSION") {
        bail!("node runs roster {}, this machine runs {}", info.version, env!("CARGO_PKG_VERSION"));
    }
    Ok(())
}

/// Open every node in file order; a node that can't be reached aborts startup
/// returns nodes and their pools at the same index
pub async fn connect(configs: Vec<NodeConfig>, local: &SystemResources)
        -> anyhow::Result<(Vec<Node>, Vec<ResourcePool>)> {
    let mut nodes = Vec::new();
    let mut pools = Vec::new();

    for config in configs {
        let online = Arc::new(AtomicBool::new(true));
        let health = Arc::new(Mutex::new(Health { state: NodeState::Online, since: Utc::now(), info: None }));
        let mut socket = None;

        let (executor, resources): (Arc<dyn Executor>, SystemResources) = match (&config.ssh, &config.socket) {
            (None, _) => (Arc::new(ShellExecutor), local.clone()),
            (Some(target), Some(remote)) => {
                let tunnel = Tunnel {
                    name:   config.name.clone(),
                    target: target.clone(),
                    remote: remote.clone(),
                    local:  node_socket_path(&config.name),
                };
                let (child, info) = tunnel.open().await
                    .with_context(|| format!("node '{}'", config.name))?;
                health.lock().await.info = Some(info.clone());
                let executor = RemoteExecutor::new(tunnel.local.clone(), online.clone());
                socket = Some(tunnel.local.clone());
                tokio::spawn(tunnel.supervise(child, online.clone(), health.clone()));
                (Arc::new(executor), info.resources)
            }
            (Some(_), None) => bail!("node '{}' has no socket, add it with `roster node add`", config.name),
        };
        tracing::info!(node = %config.name, cpu_cores = resources.cpu_cores, memory_mb = resources.memory_mb,
                       gpu_count = resources.gpus.len(), "node ready");

        pools.push(ResourcePool::new(resources));
        nodes.push(Node { name: config.name, ssh: config.ssh, executor, socket, online, health });
    }
    Ok((nodes, pools))
}

/// `ssh -N -L local:remote` to one node; the controller reaches the node daemon through `local`
struct Tunnel {
    name:   String,
    target: String,
    remote: String,  // the node daemon's socket
    local:  PathBuf, // this end, under the runtime dir
}

impl Tunnel {
    /// Start ssh; at LogLevel=ERROR it writes only why it failed (connect, auth, host key) to daemon.log
    fn spawn(&self) -> anyhow::Result<Child> {
        if let Some(parent) = self.local.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let forward = format!("{}:{}", self.local.display(), self.remote);
        let mut command = Command::new("ssh");
        command
            .args(["-N", "-o", "LogLevel=ERROR", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"])
            .args(["-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindUnlink=yes"])
            .args(["-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3"])
            .args(["-L", &forward, &self.target])
            .stdin(Stdio::null())
            .kill_on_drop(true);
        // also gone when this daemon is killed outright, so a restarted one finds no stale tunnels
        unsafe {
            command.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }
        Ok(command.spawn()?)
    }

    /// Event: the health check; asks the node daemon for its info through the tunnel
    /// the daemon's answer comes back even when its version does not match
    async fn check(&self) -> (NodeState, Option<NodeInfo>) {
        let stream = match UnixStream::connect(&self.local).await {
            Ok(stream) => stream,
            Err(error) => return (NodeState::TunnelDown(error.to_string()), None),
        };
        let info = match timeout(Duration::from_secs(3), client::exchange(stream, Request::Info)).await {
            Ok(Ok(Response::Info(info))) => info,
            // ssh resets the connection at once when nothing listens on the node's socket
            Ok(Ok(_)) | Ok(Err(_)) => return (NodeState::DaemonDown, None),
            Err(_elapsed) => return (NodeState::NoReply, None),
        };
        match check_version(&info) {
            Ok(()) => (NodeState::Online, Some(info)),
            Err(error) => (NodeState::VersionMismatch(error.to_string()), Some(info)),
        }
    }

    /// Where every event's state takes effect: the scheduler's flag, the health record, a log line on change
    async fn set_state(&self, online: &AtomicBool, health: &Mutex<Health>, state: NodeState, info: Option<NodeInfo>) {
        online.store(state.is_online(), Ordering::Relaxed);
        let mut health = health.lock().await;
        if health.enter(state, info) {
            tracing::info!(node = %self.name, state = %health.state, "node state changed");
        }
    }

    /// Start ssh and wait up to 10 s for the node daemon to answer through it
    async fn open(&self) -> anyhow::Result<(Child, NodeInfo)> {
        let mut child = self.spawn()?;
        let deadline = Instant::now() + Duration::from_secs(10);

        loop {
            let state = match self.check().await {
                (NodeState::Online, Some(info)) => return Ok((child, info)),
                (state @ NodeState::VersionMismatch(_), _) => bail!("{}", state),
                (state, _) => state,
            };
            if Instant::now() >= deadline {
                bail!("no answer through the ssh tunnel to {}: {}", self.target, state);
            }
            if let Some(status) = child.try_wait()? {
                bail!("ssh to {} exited: {}", self.target, status);
            }
            sleep(Duration::from_millis(200)).await;
        }
    }

    /// Keep the tunnel up: check the node daemon every 2 s, restart ssh when it exits
    async fn supervise(self, mut child: Child, online: Arc<AtomicBool>, health: Arc<Mutex<Health>>) {
        let mut backoff = Duration::from_secs(1);

        loop {
            tokio::select! {
                status = child.wait() => {
                    self.set_state(&online, &health, ssh_exited(status), None).await;
                    sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                    match self.spawn() {
                        Ok(restarted) => child = restarted,
                        Err(error) => tracing::error!(node = %self.name, %error, "ssh restart failed"),
                    }
                }
                _ = sleep(Duration::from_secs(2)) => {
                    let (state, info) = self.check().await;
                    if state.is_online() {
                        backoff = Duration::from_secs(1);
                    }
                    self.set_state(&online, &health, state, info).await;
                }
            }
        }
    }
}

/// Event: the tunnel's ssh process exited
fn ssh_exited(status: std::io::Result<ExitStatus>) -> NodeState {
    match status {
        Ok(status) => NodeState::TunnelDown(format!("ssh exited ({})", status)),
        Err(error) => NodeState::TunnelDown(format!("ssh wait failed: {}", error)),
    }
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path() -> PathBuf {
        std::env::temp_dir().join(format!("roster-nodes-{}.yaml", uuid::Uuid::new_v4()))
    }

    #[test]
    fn missing_file_means_local_only_and_save_round_trips() {
        let path = temp_path();
        let mut configs = load(&path).unwrap();
        assert_eq!(configs.len(), 1);
        assert!(configs[0].ssh.is_none());

        configs.push(NodeConfig { name: "sim-1".into(), ssh: Some("sim-1".into()), socket: Some("/run/r.socket".into()) });
        save(&path, &configs).unwrap();
        assert_eq!(load(&path).unwrap()[1].socket.as_deref(), Some("/run/r.socket"));

        remove(&path, "sim-1").unwrap();
        assert!(remove(&path, "local").is_err()); // last node stays
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_empty_duplicate_and_misspelled() {
        for yaml in ["[]", "- name: a\n- name: a\n", "- name: a\n  sshh: host\n"] {
            let path = temp_path();
            std::fs::write(&path, yaml).unwrap();
            assert!(load(&path).is_err(), "{}", yaml);
            std::fs::remove_file(&path).unwrap();
        }
    }

    #[test]
    fn health_moves_since_only_when_the_kind_of_state_changes() {
        let mut health = Health { state: NodeState::Online, since: Utc::now(), info: None };

        assert!(health.enter(NodeState::TunnelDown("ssh exited (exit status: 255)".into()), None));
        let since = health.since;
        assert!(!health.enter(NodeState::TunnelDown("No such file or directory (os error 2)".into()), None)); // same state, newest reason kept
        assert_eq!(health.state.to_string(), "ssh tunnel down: No such file or directory (os error 2)");
        assert_eq!(health.since, since);

        let resources = SystemResources { cpu_cores: 1, memory_mb: 1024, gpus: vec![] };
        let info = NodeInfo { version: "0.0.0".into(), pid: 7, socket: "/r.socket".into(), running_jobs: 0, resources, gpu_health: vec![], jobs_survive_restart: false };
        assert!(health.enter(NodeState::VersionMismatch("node runs roster 0.0.0".into()), Some(info)));
        assert!(!health.state.is_online());
        assert!(health.enter(NodeState::Online, None));
        assert_eq!(health.info.map(|info| info.pid), Some(7)); // a state without an answer keeps the last one
    }
}
