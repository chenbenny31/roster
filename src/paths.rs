use std::path::PathBuf;

/// Returns the runtime dir for roster socket and PID file
/// $XDG_RUNTIME_DIR falls back to ~/.local/run for non-systemd
pub fn runtime_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XRD_RUNTIME_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").expect("HOME not set");
    PathBuf::from(home).join(".local").join("run")
}

/// Unix socket path for roster IPC
pub fn socket_path() -> PathBuf {
    runtime_dir().join("roster.socket")
}

/// PID file path for daemon single-instance enforcement
pub fn pid_path() -> PathBuf {
    runtime_dir().join("roster.pid")
}

/// SQLite database path: ~/.local/share/roster/roster.db
pub fn db_path() -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME not set");
    PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("roster")
        .join("roster.db")
}

/// Output of a daemon started by `roster start`: ~/.local/share/roster/daemon.log
pub fn daemon_log_path() -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME not set");
    PathBuf::from(home)
        .join(".local")
        .join("share")
        .join("roster")
        .join("daemon.log")
}
/// Log file path for a job: ~/.local/share/roster/runs/<run_id>/<job_id>.log
pub fn job_log_path(run_id: &str, job_id: &str) -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME not set");
    PathBuf::from(home).join(job_log_relpath(run_id, job_id))
}

/// Job log path relative to HOME, the same on every node
pub fn job_log_relpath(run_id: &str, job_id: &str) -> PathBuf {
    PathBuf::from(".local")
        .join("share")
        .join("roster")
        .join("runs")
        .join(run_id)
        .join(format!("{}.log", job_id))
}

/// Controller's node list, written by `roster node add`: ~/.config/roster/nodes.yaml
pub fn nodes_path() -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME not set");
    PathBuf::from(home)
        .join(".config")
        .join("roster")
        .join("nodes.yaml")
}

/// Local end of the ssh tunnel to a node's daemon socket
pub fn node_socket_path(name: &str) -> PathBuf {
    runtime_dir().join("roster-nodes").join(format!("{}.socket", name))
}