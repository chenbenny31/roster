use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc; // atomic reference counting for shared ownership across threads

use tokio::net::UnixListener;
use tokio::signal::unix::{signal, SignalKind};
use tokio::task::JoinSet;
use tokio::time::{timeout, Duration};

use crate::error::RosterError;
use crate::ipc::server::handle_connection;
use crate::paths::{pid_path, socket_path};

/// Shared daemon state, accessed behind Arch<DaemonState>
pub struct DaemonState {}

impl DaemonState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

/// Entry point for `roster daemon`
/// enforce single instance, write PID file, bind socket, run until signal
pub async fn run(state: Arc<DaemonState>) -> anyhow::Result<()> {
    check_not_running()?;
    write_pid_file()?;
    cleanup_stale_socket();

    let result = listen(state).await;
    cleanup();
    result
}

/// Refuse to start if another roster aemon is running
fn check_not_running() -> Result<(), RosterError> {
    let path = pid_path();

    let contents = match fs::read_to_string(&path) {
        Ok(contents) => contents,
        Err(_) => return Ok(()), // no PID file means no running daemon
    };

    let pid: u32 = match contents.trim().parse() {
        Ok(pid) => pid,
        Err(_) => return Ok(()), // corrupt PID file
    };

    // kill (pid, 0) checks if proc exists (no signal)
    let pid = nix::unistd::Pid::from_raw(pid as i32);
    match nix::sys::signal::kill(pid, None) {
        Ok(_) => Err(RosterError::AlreadyRunning(pid.as_raw() as u32)),
        Err(_) => Ok(()), // proc gone, stale PID file
    }
}

/// Write current PID to pid file atomically (O_CREAT|O_EXCL)
fn write_pid_file() -> anyhow::Result<()> {
    let path = pid_path();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true) // O_CREAT | O_EXCL, fails if existing
        .open(&path)?;

    // chmod 0600, only owner can rw
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    write!(file, "{}", std::process::id())?;
    Ok(())
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