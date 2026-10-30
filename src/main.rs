use std::io::IsTerminal;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{bail, Context};
use chrono::Local;
use clap::{Parser, Subcommand};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use tokio::net::UnixStream;
use tokio::time::{sleep, Duration, Instant};
use tracing_subscriber::EnvFilter;

use roster::ipc::client;
use roster::ipc::protocol::{NodeInfo, NodeStatus, Request, Response};
use roster::node;
use roster::paths::{daemon_log_path, nodes_path, socket_path, unit_path};
use roster::resource::discovery::{self, SystemResources};

#[derive(Parser)]
#[command(name = "roster", about = "GPU-aware workflow scheduler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start this machine's daemon in the background
    Start {
        #[arg(long, help = "declared inventory (YAML or JSON) instead of discovering this machine's")]
        resources: Option<PathBuf>,
        #[arg(long, help = "run in this process, e.g. under systemd or in a container")]
        foreground: bool,
    },
    /// Stop this machine's daemon; refuses while jobs are running
    Stop {
        #[arg(long, help = "stop anyway; running jobs keep running untracked")]
        force: bool,
    },
    /// Stop and start this machine's daemon
    Restart {
        #[arg(long, help = "declared inventory (YAML or JSON) instead of discovering this machine's")]
        resources: Option<PathBuf>,
        #[arg(long, help = "restart anyway; running jobs keep running untracked")]
        force: bool,
    },
    /// Run this machine's daemon under systemd: now, at boot, and after logout
    Enable {
        #[arg(long, help = "declared inventory (YAML or JSON) instead of discovering this machine's")]
        resources: Option<PathBuf>,
    },
    /// Undo `enable`: stop the daemon and remove its systemd unit; refuses while jobs are running
    Disable {
        #[arg(long, help = "disable anyway; running jobs keep running untracked")]
        force: bool,
    },
    /// Show this machine's daemon, or one of its nodes: state, running jobs, free resources
    Info {
        #[arg(help = "node name, see `roster node ls`")]
        node: Option<String>,
        #[arg(long, help = "print JSON; for this machine, as read by `node add`")]
        json: bool,
    },
    /// Manage the nodes this daemon places jobs on
    Node {
        #[command(subcommand)]
        command: NodeCommand,
    },
    /// Submit a workflow YAML
    Submit {
        #[arg(help = "path to workflow YAML")]
        file: String,
    },
    /// List active workflow runs
    Ps,
    /// Show status of a run
    Status {
        #[arg(help = "run id")]
        id: String,
    },
    /// Tail logs for a job
    Logs {
        #[arg(help = "job id")]
        id: String,
    },
    /// Cancel a run
    Cancel {
        #[arg(help = "run id")]
        id: String,
    },
}

#[derive(Subcommand)]
enum NodeCommand {
    /// Start roster on a machine you reach over ssh and add it; takes effect on `roster restart`
    Add {
        #[arg(help = "ssh destination, e.g. user@host or a ~/.ssh/config alias")]
        target: String,
        #[arg(long, help = "node name, defaults to the ssh destination")]
        name: Option<String>,
    },
    /// List nodes in first-fit order with their state
    Ls,
    /// Remove a node; takes effect on `roster restart`
    Rm {
        #[arg(help = "node name")]
        name: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr) // stdout carries command output, e.g. `info --json`
        .with_ansi(std::io::stderr().is_terminal()) // plain text in daemon.log
        .init();
    let cli = Cli::parse();

    match cli.command {
        Command::Start { resources, foreground: true } => {
            // before opening tunnels, which would take over a running daemon's tunnel sockets
            if let Some(info) = daemon_info().await? {
                bail!("a daemon is already running here (pid {})", info.pid);
            }
            let resources = load_resources(resources)?;
            let (nodes, pools) = node::connect(node::load(&nodes_path())?, &resources).await?;
            roster::daemon::run(resources, nodes, pools).await?;
        },
        Command::Start { resources, foreground: false } => start(resources).await?,
        Command::Enable { resources } => enable(resources).await?,
        Command::Disable { force } => disable(force).await?,
        Command::Stop { force } => stop(force).await?,
        Command::Restart { resources, force } => {
            stop(force).await?;
            start(resources).await?;
        },
        Command::Info { node: None, json } => {
            let info = daemon_info().await?
                .context("no daemon is running here, start it with `roster start`")?;
            if json {
                println!("{}", serde_json::to_string(&info)?);
            } else {
                print_info(&info);
            }
        },
        Command::Info { node: Some(name), json } => {
            let status = node_statuses().await?.into_iter().find(|status| status.name == name)
                .with_context(|| format!("no node '{}' in this daemon, see `roster node ls`", name))?;
            if json {
                println!("{}", serde_json::to_string(&status)?);
            } else {
                print_node(&status);
            }
        },
        Command::Node { command: NodeCommand::Add { target, name } } => {
            let info = node::add(&nodes_path(), &target, name).await?;
            println!("added {}: {} cores, {} MB, {} GPUs; run `roster restart` to use it",
                     target, info.resources.cpu_cores, info.resources.memory_mb, info.resources.gpus.len());
        },
        Command::Node { command: NodeCommand::Ls } => list_nodes().await?,
        Command::Node { command: NodeCommand::Rm { name } } => {
            node::remove(&nodes_path(), &name)?;
            println!("removed {}; run `roster restart` to apply", name);
        },
        Command::Submit { file } => {
            let spec_yaml = tokio::fs::read_to_string(&file).await
                .map_err(|error| anyhow::anyhow!("failed to read {}: {}", file, error))?;
            let response = roster::ipc::client::send(
                roster::ipc::protocol::Request::Submit { spec_yaml }
            ).await?;
            println!("{:?}", response);
        },
        Command::Ps => {
            let response = roster::ipc::client::send(
                roster::ipc::protocol::Request::Ps
            ).await?;
            println!("{:?}", response);
        },
        Command::Status { id } => {
            let response = roster::ipc::client::send(
                roster::ipc::protocol::Request::Status { run_id: id }
            ).await?;
            println!("{:?}", response);
        },
        Command::Logs { id } => {
            // id format: <run_id>/<job_id>
            let parts: Vec<&str> = id.splitn(2, '/').collect();
            if parts.len() != 2 {
                anyhow::bail!("logs requires <run_id>/<job_id>");
            }
            let response = roster::ipc::client::send(
                roster::ipc::protocol::Request::Logs {
                    run_id: parts[0].into(),
                    job_id: parts[1].into()
                }
            ).await?;
            println!("{:?}", response);
        },
        Command::Cancel { .. } => { unimplemented!("cancel") },
    }

    Ok(())
}

/// A declared inventory file, or this machine's discovered one
fn load_resources(path: Option<PathBuf>) -> anyhow::Result<SystemResources> {
    let Some(path) = path else { return Ok(discovery::discover()) };
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("invalid resources file {}", path.display()))
}

/// Ask this machine's daemon about itself; None when nothing listens on the socket
async fn daemon_info() -> anyhow::Result<Option<NodeInfo>> {
    let Ok(stream) = UnixStream::connect(socket_path()).await else { return Ok(None) };
    match client::exchange(stream, Request::Info).await? {
        Response::Info(info) => Ok(Some(info)),
        other => bail!("unexpected reply: {:?}", other),
    }
}

/// Run this binary again as `start --foreground` in a new session, logging to daemon.log,
/// and return once it answers; a daemon already running counts as started
/// once enabled, systemd starts it instead
async fn start(resources: Option<PathBuf>) -> anyhow::Result<()> {
    if let Some(info) = daemon_info().await? {
        println!("already running (pid {})", info.pid);
        return Ok(());
    }
    if unit_path().exists() {
        if resources.is_some() {
            bail!("the daemon runs under systemd; change its inventory with `roster enable --resources FILE`");
        }
        run("systemctl", &["--user", "start", "roster"])?;
        let info = wait_for_daemon(None, JOURNAL).await?;
        println!("started (pid {}), log `{}`", info.pid, JOURNAL);
        return Ok(());
    }

    let log_path = daemon_log_path();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;

    let mut command = std::process::Command::new(std::env::current_exe()?);
    command.args(["start", "--foreground"]);
    if let Some(path) = &resources {
        command.arg("--resources").arg(path);
    }
    if std::env::var_os("RUST_LOG").is_none() {
        command.env("RUST_LOG", "info"); // node and shutdown events belong in the log
    }
    command.stdin(Stdio::null()).stdout(log.try_clone()?).stderr(log);

    // setsid() runs in the child after fork before exec
    // a new session has no controlling terminal, so closing the shell or ssh session doesn't stop it
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command.spawn().context("failed to start the daemon")?;

    let log = log_path.display().to_string();
    let info = wait_for_daemon(Some(child), &log).await?;
    println!("started (pid {}), log {}", info.pid, log);
    Ok(())
}

/// Where a daemon run by systemd logs
const JOURNAL: &str = "journalctl --user -u roster";

/// Wait until the daemon answers; `child` is the process `start` spawned, to notice it exit early
async fn wait_for_daemon(mut child: Option<std::process::Child>, log: &str) -> anyhow::Result<NodeInfo> {
    // a controller first opens a tunnel to each node, up to 10 s each
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(info) = daemon_info().await? {
            return Ok(info);
        }
        if let Some(child) = child.as_mut() {
            if let Some(status) = child.try_wait()? {
                bail!("daemon exited ({}), see {}", status, log);
            }
        }
        if Instant::now() >= deadline {
            bail!("daemon did not answer within 60 s, see {}", log);
        }
        sleep(Duration::from_millis(100)).await;
    }
}

/// Run the daemon as a systemd user service, with linger so it starts at boot and survives logout;
/// replaces a daemon started by `roster start`; without --resources an existing unit is kept,
/// so `node add` never changes a node's declared inventory
async fn enable(resources: Option<PathBuf>) -> anyhow::Result<()> {
    run("loginctl", &["enable-linger"])
        .context("run `sudo loginctl enable-linger $USER` once, then `roster enable` again")?;

    let path = unit_path();
    let enabled = path.exists();
    if !enabled && daemon_info().await?.is_some() {
        stop(false).await?; // started by `roster start`; refuses while it runs jobs
    }
    if !enabled || resources.is_some() {
        let unit = unit_file(resources)?;
        if std::fs::read_to_string(&path).ok().as_deref() != Some(unit.as_str()) {
            std::fs::create_dir_all(path.parent().expect("unit path has a parent"))?;
            std::fs::write(&path, unit)?;
            run("systemctl", &["--user", "daemon-reload"])?;
            if enabled {
                println!("unit updated, run `roster restart` to apply");
            }
        }
    }
    run("systemctl", &["--user", "enable", "--now", "roster"])?;

    let info = wait_for_daemon(None, JOURNAL).await?;
    println!("enabled, running (pid {}), log `{}`", info.pid, JOURNAL);
    Ok(())
}

/// The unit `enable` writes; KillMode=process so stopping the daemon leaves its jobs running, as `roster stop` does
fn unit_file(resources: Option<PathBuf>) -> anyhow::Result<String> {
    let mut exec = format!("\"{}\" start --foreground", std::env::current_exe()?.display());
    if let Some(path) = resources {
        let path = std::fs::canonicalize(&path).with_context(|| format!("no resources file {}", path.display()))?;
        exec.push_str(&format!(" --resources \"{}\"", path.display()));
    }
    Ok(format!("[Unit]\nDescription=roster daemon\n\n\
                [Service]\nExecStart={}\nEnvironment=RUST_LOG=info\nKillMode=process\nRestart=on-failure\n\n\
                [Install]\nWantedBy=default.target\n", exec))
}

/// Undo `enable`: stop the daemon and remove its unit; linger stays, other user services may need it
async fn disable(force: bool) -> anyhow::Result<()> {
    if !unit_path().exists() {
        println!("not enabled");
        return Ok(());
    }
    stop(force).await?;
    run("systemctl", &["--user", "disable", "roster"])?;
    std::fs::remove_file(unit_path())?;
    run("systemctl", &["--user", "daemon-reload"])?;
    println!("disabled");
    Ok(())
}

/// Run a command to completion, failing with its error output
fn run(program: &str, args: &[&str]) -> anyhow::Result<()> {
    let output = std::process::Command::new(program).args(args).output()
        .with_context(|| format!("failed to run {}", program))?;
    if !output.status.success() {
        bail!("`{} {}` failed: {}", program, args.join(" "), String::from_utf8_lossy(&output.stderr).trim());
    }
    Ok(())
}

/// SIGTERM the daemon and wait for it to exit; refuse while it runs jobs unless forced,
/// since nothing tracks them after the stop (no journal yet)
async fn stop(force: bool) -> anyhow::Result<()> {
    let Some(info) = daemon_info().await? else {
        println!("not running");
        return Ok(());
    };
    if info.running_jobs > 0 && !force {
        bail!("{} jobs are running and would keep running untracked; wait for them or use --force",
              info.running_jobs);
    }
    if unit_path().exists() {
        run("systemctl", &["--user", "stop", "roster"])?; // returns once the daemon has exited
        println!("stopped (pid {})", info.pid);
        return Ok(());
    }

    let pid = Pid::from_raw(info.pid as i32);
    kill(pid, Signal::SIGTERM).context("failed to signal the daemon")?;

    // kill(pid, 0) checks the process still exists, no signal sent
    let deadline = Instant::now() + Duration::from_secs(10);
    while kill(pid, None).is_ok() {
        if Instant::now() >= deadline {
            bail!("daemon (pid {}) still running 10 s after SIGTERM", info.pid);
        }
        sleep(Duration::from_millis(100)).await;
    }
    println!("stopped (pid {})", info.pid);
    Ok(())
}

fn print_info(info: &NodeInfo) {
    println!("roster {}, daemon pid {}, {} jobs running", info.version, info.pid, info.running_jobs);
    println!("socket  {}", info.socket);
    println!("cpu     {} cores", info.resources.cpu_cores);
    println!("memory  {} MB", info.resources.memory_mb);
    if info.resources.gpus.is_empty() {
        println!("gpu     none");
    }
    for gpu in &info.resources.gpus {
        let health = info.gpu_health.iter().find(|health| health.index == gpu.index);
        println!("gpu {:<3} {}, {} MB{}", gpu.index, gpu.name, gpu.vram_mb,
                 health.map(|health| health.to_string()).unwrap_or_default());
    }
}

/// Ask this machine's daemon for the nodes it places jobs on
async fn node_statuses() -> anyhow::Result<Vec<NodeStatus>> {
    let response = client::send(Request::Nodes).await
        .context("no daemon is running here, start it with `roster start`")?;
    match response {
        Response::Nodes { nodes } => Ok(nodes),
        other => bail!("unexpected reply: {:?}", other),
    }
}

/// The daemon's nodes with their state; without a daemon, what nodes.yaml lists
async fn list_nodes() -> anyhow::Result<()> {
    if daemon_info().await?.is_none() {
        for config in node::load(&nodes_path())? {
            println!("{:<12} {:<20} daemon not running", config.name, config.ssh.as_deref().unwrap_or("-"));
        }
        return Ok(());
    }
    println!("{:<12} {:<20} {:<8} {:<12} {}", "NAME", "SSH", "RUNNING", "SINCE", "STATE");
    for status in node_statuses().await? {
        println!("{:<12} {:<20} {:<8} {:<12} {}", status.name, status.ssh.as_deref().unwrap_or("-"),
                 status.running_jobs, status.since.with_timezone(&Local).format("%m-%d %H:%M"), status.state);
    }
    Ok(())
}

fn print_node(status: &NodeStatus) {
    println!("node    {} ({})", status.name, status.ssh.as_deref().unwrap_or("this machine"));
    println!("state   {}, since {}", status.state, status.since.with_timezone(&Local).format("%Y-%m-%d %H:%M:%S"));
    let held = if status.online || status.running_jobs == 0 { "" } else { ", held until the node answers" };
    println!("jobs    {} running{}", status.running_jobs, held);
    println!("cpu     {}/{} cores free", status.free_cpu, status.total.cpu_cores);
    println!("memory  {}/{} MB free", status.free_memory_mb, status.total.memory_mb);
    if status.total.gpus.is_empty() {
        println!("gpu     none");
    }
    for (gpu, free) in status.total.gpus.iter().zip(&status.free_vram_mb) {
        let health = status.gpu_health.iter().find(|health| health.index == gpu.index);
        let outside = *free == gpu.vram_mb && health.is_some_and(|health| health.busy());
        println!("gpu {:<3} {}, {}/{} MB free{}{}", gpu.index, gpu.name, free, gpu.vram_mb,
                 health.map(|health| health.to_string()).unwrap_or_default(),
                 if outside { ", in use outside roster" } else { "" });
    }
}
