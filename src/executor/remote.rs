//! Controller side: run executions on another node through its daemon's tunnel

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::net::UnixStream;
use tokio::time::{timeout, Duration};

use crate::executor::{Backend, Executor, ExecutorError, JobHandle, PollResult};
use crate::ipc::client;
use crate::ipc::protocol::{Request, Response};
use crate::resource::pool::Allocation;
use crate::workflow::model::JobInstance;

/// Sends worker operations to one node's daemon through the local end of its ssh tunnel
/// a failed or timed-out request marks the node offline until the tunnel's health check passes
pub struct RemoteExecutor {
    socket: PathBuf,
    online: Arc<AtomicBool>, // shared with the node's tunnel task
}

impl RemoteExecutor {
    pub fn new(socket: PathBuf, online: Arc<AtomicBool>) -> Self {
        Self { socket, online }
    }

    /// Connect to the tunnel; failing here means nothing was sent
    async fn connect(&self) -> Result<UnixStream, ExecutorError> {
        if !self.online.load(Ordering::Relaxed) {
            return Err(ExecutorError::Unreachable("node offline".into()));
        }
        UnixStream::connect(&self.socket).await.map_err(|error| self.lost(error.to_string()))
    }

    /// Send one request on a connected stream; an error reply becomes ExecutorError::Remote
    async fn exchange(&self, stream: UnixStream, request: Request, limit: Duration)
            -> Result<Response, ExecutorError> {
        match timeout(limit, client::exchange(stream, request)).await {
            Ok(Ok(Response::Error { message })) => Err(ExecutorError::Remote(message)),
            Ok(Ok(response)) => Ok(response),
            Ok(Err(error)) => Err(self.lost(error.to_string())),
            Err(_elapsed) => Err(self.lost("no reply".into())),
        }
    }

    fn lost(&self, reason: String) -> ExecutorError {
        self.online.store(false, Ordering::Relaxed);
        ExecutorError::Unreachable(reason)
    }
}

/// The node's name for an execution, taken from a handle this executor made
fn key(handle: &JobHandle) -> Result<(String, String), ExecutorError> {
    match &handle.backend {
        Backend::Remote { run_id, job_id } => Ok((run_id.clone(), job_id.clone())),
        _ => Err(ExecutorError::Remote("not a remote handle".into())),
    }
}

#[async_trait]
impl Executor for RemoteExecutor {
    /// If the request went out but no reply came back, the job may be running,
    /// so return a handle anyway and let the next poll find out
    async fn launch(&self, run_id: &str, job: &JobInstance, placement: &Allocation)
            -> Result<JobHandle, ExecutorError> {
        let stream = self.connect().await?;
        let request = Request::Launch {
            run_id:     run_id.into(),
            job:        job.spec.clone(),
            allocation: placement.clone(),
        };

        let host_pid = match self.exchange(stream, request, Duration::from_secs(10)).await {
            Ok(Response::Launched { host_pid }) => host_pid,
            Ok(other) => return Err(ExecutorError::Remote(format!("unexpected reply: {:?}", other))),
            Err(ExecutorError::Unreachable(reason)) => {
                tracing::warn!(%run_id, job_id = %job.job_id, %reason, "launch reply lost, next poll decides");
                0
            }
            Err(error) => return Err(error),
        };
        Ok(JobHandle::remote(host_pid, run_id.into(), job.job_id.clone()))
    }

    async fn poll(&self, handle: &JobHandle) -> Result<PollResult, ExecutorError> {
        let (run_id, job_id) = key(handle)?;
        let stream = self.connect().await?;

        match self.exchange(stream, Request::Poll { run_id, job_id }, Duration::from_secs(5)).await? {
            Response::Polled { exit_code: None } => Ok(PollResult::Running),
            Response::Polled { exit_code: Some(exit_code) } => Ok(PollResult::Exited { exit_code }),
            other => Err(ExecutorError::Remote(format!("unexpected reply: {:?}", other))),
        }
    }

    /// The node waits 5 s between SIGTERM and SIGKILL before it replies
    async fn cancel(&self, handle: &JobHandle) -> Result<(), ExecutorError> {
        let (run_id, job_id) = key(handle)?;
        let stream = self.connect().await?;

        match self.exchange(stream, Request::Kill { run_id, job_id }, Duration::from_secs(15)).await? {
            Response::Killed => Ok(()),
            other => Err(ExecutorError::Remote(format!("unexpected reply: {:?}", other))),
        }
    }
}
