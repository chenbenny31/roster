use std::path::PathBuf;

use async_trait::async_trait;

use crate::workflow::model::JobInstance;
use crate::resource::pool::Allocation;

pub mod remote;
pub mod shell;

// Opaque handle to a launched job, return by `launch` and passed back to `poll`/`cancel`
#[derive(Debug, Clone)]
pub struct JobHandle {
    pub host_pid: u32,
    backend: Backend,
}

impl JobHandle {
    /// Construct a handle for a host process (ShellExecutor); `scope` is the systemd scope holding it and
    /// `exit_file` where its wrapper writes the exit code, both None without systemd
    pub(crate) fn process(host_pid: u32, scope: Option<String>, exit_file: Option<PathBuf>) -> Self {
        Self { host_pid, backend: Backend::Process { scope, exit_file } }
    }

    /// Where the job's wrapper writes its exit code, if it runs in a scope
    pub(crate) fn exit_file(&self) -> Option<&PathBuf> {
        match &self.backend {
            Backend::Process { exit_file, .. } => exit_file.as_ref(),
            _ => None,
        }
    }

    /// The systemd scope unit holding a local job's processes, if it runs in one
    pub(crate) fn scope(&self) -> Option<&str> {
        match &self.backend {
            Backend::Process { scope, .. } => scope.as_deref(),
            _ => None,
        }
    }

    /// Construct a handle for an execution on another node (RemoteExecutor)
    /// host_pid is the node's pid, 0 when the launch reply was lost
    pub(crate) fn remote(host_pid: u32, run_id: String, job_id: String) -> Self {
        Self { host_pid, backend: Backend::Remote { run_id, job_id } }
    }
}

/// Concrete executor produced a handle and specific identity to operate job later
#[derive(Debug, Clone)]
enum Backend {
    Process { scope: Option<String>, exit_file: Option<PathBuf> }, // the scope is a cgroup holding the job and everything it starts
    #[allow(dead_code)]
    Container(String),
    Remote { run_id: String, job_id: String }, // the node's name for the execution
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

    #[error("log read failed: {0}")]
    LogReadFailed(std::io::Error),

    /// The node did not answer; nothing is known about the job, so its resources stay held
    #[error("node unreachable: {0}")]
    Unreachable(String),

    /// The node answered with an error
    #[error("node error: {0}")]
    Remote(String),

    /// The node turned the job down before starting it, e.g. its program isn't there; another node may run it
    #[error("rejected: {0}")]
    Rejected(String),
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

    /// The last `bytes` of a job's log, read where the job ran; works while it runs and after it ended
    async fn tail(&self, run_id: &str, job_id: &str, bytes: u64) -> Result<String, ExecutorError>;
}
