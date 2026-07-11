use async_trait::async_trait;

use crate::workflow::model::JobInstance;
use crate::resource::pool::Allocation;

pub mod shell;

// Opaque handle to a launched job, return by `launch` and passed back to `poll`/`cancel`
#[derive(Debug, Clone)]
pub struct JobHandle {
    pub host_pid: u32,
    #[allow(dead_code)]
    backend: Backend,
}

impl JobHandle {
    /// Construct a handle for a plain host process (ShellExecutor)
    pub(crate) fn process(host_pid: u32) -> Self {
        Self { host_pid, backend: Backend::Process }
    }
}

/// Concrete executor produced a handle and specific identity to operate job later
#[derive(Debug, Clone)]
enum Backend {
    Process,
    #[allow(dead_code)]
    Container(String),
}

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
    async fn launch(&self, run_id: &str, job: &JobInstance, placement: &Allocation)
        -> Result<JobHandle, ExecutorError>;

    /// Non-blocking check, return Running or Exited
    async fn poll(&self, handle: &JobHandle) -> Result<PollResult, ExecutorError>;

    /// Send SIGTERM, wait 5s, send SIGKILL, return when signal is sent
    async fn cancel(&self, handle: &JobHandle) -> Result<(), ExecutorError>;
}
