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