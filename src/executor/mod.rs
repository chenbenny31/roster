use async_trait::async_trait;

use crate::workflow::model::JobInstance;

pub mod shell;

/// Result of non-blocking process poll
#[derive(Debug)]
pub enum PollResult {
    Running,
    Exited { exit_code: i32 },
}

/// Errors produced by executor operations
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("spawn failed: {0}")]
    SpawnFailed(std::io::Error),

    #[error("wait failed: {0}")]
    WaitFailed(std::io::Error),

    #[error("kill failed: {0}")]
    KillFailed(std::io::Error),

    #[error("log setup failed: {0}")]
    LogSetupFailed(std::io::Error),
}

/// Trait for job executors, MVP implements ShellExecutor
#[async_trait]
pub trait Executor: Send + Sync {
    /// Spawn the job subprocess, return the PID
    async fn launch(&self, run_id: &str, job: &JobInstance) -> Result<u32, ExecutorError>;

    /// Non-blocking check, return Running or Exited
    async fn poll(&self, pid: u32) -> Result<PollResult, ExecutorError>;

    /// Send SIGTERM, wait 5s, send SIGKILL, return when signal is sent
    async fn cancel(&self, pid: u32) -> Result<(), ExecutorError>;
}
