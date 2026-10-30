use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::Arc; // atomic reference counting for shared ownership across threads
use std::sync::atomic::AtomicU64;

use tokio::net::UnixListener;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::{timeout, Duration};

use crate::broadcast::{self as spmc, SpmcSubscriber};
use crate::error::RosterError;
use crate::event::JobEvent;
use crate::ipc::server::handle_connection;
use crate::node::Node;
use crate::paths::db_path;
use crate::paths::{pid_path, socket_path};
use crate::resource::discovery::SystemResources;
use crate::resource::health::GpuMonitor;
use crate::resource::pool::ResourcePool;
use crate::scheduler;
use crate::store::RunStore;
use crate::worker::Worker;
use crate::workflow::model::WorkflowRun;

/// Shared daemon state, accessed behind Arch<DaemonState>
pub struct DaemonState {
    pub runs:        Mutex<HashMap<String, WorkflowRun>>, // run_id: WorkflowRun
    pub nodes:       Vec<Node>,                // nodes.yaml order = first-fit order
    pub pools:       Mutex<Vec<ResourcePool>>, // same index as nodes
    pub resources:   SystemResources,          // this machine's inventory, reported by Info
    pub gpu_monitor: GpuMonitor,               // this machine's GPU readings, reported by Info
    pub worker:      Worker,                   // runs jobs for controllers that use this machine as a node
    pub store:       RunStore,
    pub events:      SpmcSubscriber<JobEvent>, // Sync, Clone - IPC handlers call .subscribe()
    pub run_counter: AtomicU64, // fetch_add(1, Relaxed) at submit -> run_seq
    pub job_counter: AtomicU64, // fetch_add(n, Relaxed) at submit -> job_seq_start
}

impl DaemonState {
    /// Returns the shared state plus the SpmcSender (must be moved into)
    /// SpmcSender is !Sync - cannot live in Arc<DaemonState>
    pub fn new(resources: SystemResources, nodes: Vec<Node>, pools: Vec<ResourcePool>, store: RunStore)
            -> (Arc<Self>, crate::broadcast::SpmcSender<JobEvent>) {
        let (sender, events) = spmc::channel(256);
        let gpu_monitor = GpuMonitor::open(&resources.gpus);
        let state = Arc::new(Self {
            runs:        Mutex::new(HashMap::new()),
            nodes,
            pools:       Mutex::new(pools),
            resources,
            gpu_monitor,
            worker:      Worker::default(),
            store,
            events,
            run_counter: AtomicU64::new(0),
            job_counter: AtomicU64::new(0),
        });
        (state, sender)
    }
}

/// Entry point for `roster start --foreground`
/// enforce single instance, write PID file, bind socket, run until signal
/// `resources` is this machine's inventory, `nodes` and `pools` what the scheduler places on
pub async fn run(resources: SystemResources, nodes: Vec<Node>, pools: Vec<ResourcePool>) -> anyhow::Result<()> {
    let _pid_lock = lock_pid_file(&pid_path())?; // held until run returns
    cleanup_stale_socket();
    crate::executor::shell::limit_jobs(resources.memory_mb);

    let store = RunStore::open(&db_path()).await?;
    tracing::info!(path = %db_path().display(), "databased opened");
    store.reconcile_interrupted().await?;
    let (state, event_sender) = DaemonState::new(resources, nodes, pools, store);

    let result = tokio::select! {
        r = listen(state.clone()) => r,
        _ = scheduler::run(state.clone(), event_sender) => Ok(()),
    };
    cleanup();
    result
}

/// Single instance: an exclusive flock on the pid file, held for the daemon's life;
/// the kernel drops it on any exit, even SIGKILL, so a file left by a crash never blocks a restart
fn lock_pid_file(path: &Path) -> anyhow::Result<fs::File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new().read(true).write(true).create(true).mode(0o600).open(path)?;

    // LOCK_NB: fail at once instead of waiting for the running daemon to exit
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let pid = fs::read_to_string(path).ok().and_then(|text| text.trim().parse().ok()).unwrap_or(0);
        return Err(RosterError::AlreadyRunning(pid).into());
    }
    file.set_len(0)?;
    write!(file, "{}", std::process::id())?;
    Ok(file)
}

/// Remove remain socket file from prev crash
fn cleanup_stale_socket() {
    let _ = fs::remove_file(socket_path()); // ignore err
}

/// Remove PID file and socket on clean shutdown
fn cleanup() {
    let _ = fs::remove_file(pid_path());
    let _ = fs::remove_file(socket_path());
}

/// Bind socket and accept conn until SIGTERM or SIGINT
/// spawned conn tasks are tracked in JoinSet and drained on shutdown
async fn listen(state: Arc<DaemonState>) -> anyhow::Result<()> {
    let socket = socket_path();

    if let Some(parent) = socket.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // umask(0o077) before bind, socket inherits 0600
    unsafe { libc::umask(0o077) };

    let listener = UnixListener::bind(&socket)?;
    tracing::info!(?socket, "roster daemon listening");

    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;
    let mut join_set: JoinSet<()> = JoinSet::new();

    loop {
        tokio::select! {
            result = listener.accept() => {
                match result {
                    Ok((conn, _addr)) => {
                        let state = state.clone();
                        join_set.spawn(handle_connection(conn, state));
                    }
                    Err(error) => {
                        tracing::error!(?error, "accept failed");
                    }
                }
            }

            _ = sigterm.recv() => {
                tracing::info!("received SIGTERM, shutting down");
                break;
            }
            _ = sigint.recv() => {
                tracing::info!("received SIGINT, shutting down");
                break;
            }
        }
    }

    // drain in-flight conn with 5s timeout
    tracing::info!("draining in-flight connections");
    timeout(Duration::from_secs(5), async {
        while join_set.join_next().await.is_some() {}
    })
    .await
    .ok();

    Ok(())
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_lock_is_exclusive_and_ignores_a_stale_file() {
        let path = std::env::temp_dir().join(format!("roster-{}.pid", uuid::Uuid::new_v4()));
        fs::write(&path, "1").unwrap(); // left by a crash; pid 1 is alive but not a roster daemon
        let held = lock_pid_file(&path).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), std::process::id().to_string());
        assert!(lock_pid_file(&path).is_err()); // a second daemon is refused
        drop(held);
        assert!(lock_pid_file(&path).is_ok()); // the lock goes away with its holder
        fs::remove_file(&path).unwrap();
    }
}
