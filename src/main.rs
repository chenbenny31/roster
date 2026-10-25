use std::io::IsTerminal;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use tokio::net::UnixStream;
use tokio::time::{sleep, Duration, Instant};
use tracing_subscriber::EnvFilter;

use roster::ipc::client;
use roster::ipc::protocol::{NodeInfo, Request, Response};
use roster::paths::{daemon_log_path, socket_path};
use roster::resource::discovery::{self, SystemResources};
use roster::resource::pool::ResourcePool;

#[derive(Parser)]
#[command(name = "roster", about = "GPU-aware single-node workflow scheduler")]
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
    /// Show this machine's daemon: version, pid, running jobs, inventory
    Info {
        #[arg(long, help = "print JSON, as read by `node add`")]
        json: bool,
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
            let resources = load_resources(resources)?;
            let pool = ResourcePool::new(resources.clone());
            roster::daemon::run(resources, pool).await?;
        },
        Command::Start { resources, foreground: false } => start(resources).await?,
        Command::Stop { force } => stop(force).await?,
        Command::Restart { resources, force } => {
            stop(force).await?;
            start(resources).await?;
        },
        Command::Info { json } => {
            let info = daemon_info().await?
                .context("no daemon is running here, start it with `roster start`")?;
            if json {
                println!("{}", serde_json::to_string(&info)?);
            } else {
                print_info(&info);
            }
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
async fn start(resources: Option<PathBuf>) -> anyhow::Result<()> {
    if let Some(info) = daemon_info().await? {
        println!("already running (pid {})", info.pid);
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
    let mut child = command.spawn().context("failed to start the daemon")?;

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(info) = daemon_info().await? {
            println!("started (pid {}), log {}", info.pid, log_path.display());
            return Ok(());
        }
        if let Some(status) = child.try_wait()? {
            bail!("daemon exited ({}), see {}", status, log_path.display());
        }
        if Instant::now() >= deadline {
            bail!("daemon did not answer within 10 s, see {}", log_path.display());
        }
        sleep(Duration::from_millis(100)).await;
    }
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
        println!("gpu {:<3} {}, {} MB", gpu.index, gpu.name, gpu.vram_mb);
    }
}
