use std::path::PathBuf; // owned + mutable filesystem path

pub mod protocol;
pub mod server;
pub mod client;

/// Returns the UNIX socket path for roster IPC
/// Prefers XDG_RUNTIME_DIR (/run/user/<uid> on systemd)
/// Falls back to ~/.local/run/roster.sock for non-systemd
pub fn socket_path() -> PathBuf {
    if let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime_dir).join("roster.socket");
    }
    let home = std::env::var_os("HOME").expect("HOME not set");
    PathBuf::from(home).join(".local").join("run").join("roster.socket")
}