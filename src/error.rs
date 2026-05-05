use thiserror::Error;

#[derive(Debug, Error)]
pub enum RosterError {
    #[error("daemon is already running (pid {0})")]
    AlreadyRunning(u32),

    #[error("daemon is not running")]
    NotRunning,

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}