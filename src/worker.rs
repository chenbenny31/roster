//! Worker side: run the jobs a controller places on this machine

use std::collections::HashMap;

use tokio::sync::Mutex;

use crate::executor::shell::ShellExecutor;
use crate::executor::{Executor, ExecutorError, JobHandle, PollResult};
use crate::ipc::protocol::Response;
use crate::resource::pool::Allocation;
use crate::workflow::model::JobInstance;
use crate::workflow::spec::JobSpec;

/// The job this node started, and its exit code once known
struct JobProcess {
    handle:    JobHandle,   // its scope and exit file, or without systemd the pid of `sh -c` and its process group
    exit_code: Option<i32>, // kept after exit so a repeated Poll gets the same answer
}

impl JobProcess {
    /// Check the process unless its exit is already known; None while it runs
    async fn exit_code(&mut self) -> Result<Option<i32>, ExecutorError> {
        if self.exit_code.is_none() {
            if let PollResult::Exited { exit_code } = ShellExecutor.poll(&self.handle).await? {
                self.exit_code = Some(exit_code);
            }
        }
        Ok(self.exit_code)
    }
}

/// This node's side of the worker operations; a job started before this daemon restarted is found
/// again through its scope or exit file the first time the controller asks about it
#[derive(Default)]
pub struct Worker {
    processes: Mutex<HashMap<(String, String), JobProcess>>, // (run_id, job_id)
}

impl Worker {
    /// Start the job once; a repeated Launch returns the process already started
    pub async fn launch(&self, run_id: String, job: JobSpec, allocation: Allocation) -> Response {
        let key = (run_id, job.id.clone());
        let mut processes = self.processes.lock().await;
        if let Some(existing) = processes.get(&key) {
            return Response::Launched { host_pid: existing.handle.host_pid };
        }

        match ShellExecutor.launch(&key.0, &JobInstance::new(job, 0), &allocation).await {
            Ok(handle) => {
                let host_pid = handle.host_pid;
                processes.insert(key, JobProcess { handle, exit_code: None });
                Response::Launched { host_pid }
            }
            Err(error) => Response::Error { message: format!("launch failed: {}", error) },
        }
    }

    pub async fn poll(&self, run_id: String, job_id: String) -> Response {
        let mut processes = self.processes.lock().await;
        let Some(process) = known(&mut processes, run_id, job_id).await else {
            return Response::Error { message: "unknown execution".into() };
        };

        match process.exit_code().await {
            Ok(exit_code) => Response::Polled { exit_code },
            Err(error) => Response::Error { message: format!("poll failed: {}", error) },
        }
    }

    /// SIGTERM the process group, SIGKILL after 5 s; the exit shows up in the next Poll
    pub async fn kill(&self, run_id: String, job_id: String) -> Response {
        let handle = match known(&mut *self.processes.lock().await, run_id, job_id).await {
            None => return Response::Error { message: "unknown execution".into() },
            Some(process) if process.exit_code.is_some() => return Response::Killed,
            Some(process) => process.handle.clone(),
        }; // lock released before the 5 s cancel

        match ShellExecutor.cancel(&handle).await {
            Ok(()) => Response::Killed,
            Err(error) => Response::Error { message: format!("kill failed: {}", error) },
        }
    }

    /// Processes started here that have not exited, for `roster info` and `roster stop`
    /// a process that can't be checked counts as running
    pub async fn running(&self) -> usize {
        let mut processes = self.processes.lock().await;
        let mut running = 0;
        for process in processes.values_mut() {
            if !matches!(process.exit_code().await, Ok(Some(_))) {
                running += 1;
            }
        }
        running
    }
}

/// The job's process: one this daemon started, or one started before it restarted, found by its scope or exit file
async fn known(processes: &mut HashMap<(String, String), JobProcess>, run_id: String, job_id: String)
        -> Option<&mut JobProcess> {
    let key = (run_id, job_id);
    if !processes.contains_key(&key) {
        let handle = ShellExecutor::find(&key.0, &key.1).await?;
        tracing::info!(run_id = %key.0, job_id = %key.1, "found a job started before this daemon restarted");
        processes.insert(key.clone(), JobProcess { handle, exit_code: None });
    }
    processes.get_mut(&key)
}

// tests

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    fn allocation() -> Allocation {
        Allocation { cpu: 1, memory_mb: 512, gpus: vec![] }
    }

    #[tokio::test]
    async fn launch_once_and_keep_exit_code() {
        let worker = Worker::default();
        let run_id = format!("worker-test-{}", uuid::Uuid::new_v4());
        let job = JobSpec { id: "exit3".into(), command: "exit 3".into(), ..Default::default() };

        let Response::Launched { host_pid } = worker.launch(run_id.clone(), job.clone(), allocation()).await
            else { panic!() };
        let Response::Launched { host_pid: again } = worker.launch(run_id.clone(), job, allocation()).await
            else { panic!() };
        assert_eq!(host_pid, again); // not started twice

        let exit_code = loop {
            match worker.poll(run_id.clone(), "exit3".into()).await {
                Response::Polled { exit_code: Some(code) } => break code,
                Response::Polled { exit_code: None } => sleep(Duration::from_millis(20)).await,
                other => panic!("{:?}", other),
            }
        };
        assert_eq!(exit_code, 3);
        assert!(matches!(worker.poll(run_id.clone(), "exit3".into()).await, Response::Polled { exit_code: Some(3) }));
        assert!(matches!(worker.poll(run_id, "missing".into()).await, Response::Error { .. }));
    }

    #[tokio::test]
    async fn running_counts_until_killed() {
        let worker = Worker::default();
        let run_id = format!("worker-test-{}", uuid::Uuid::new_v4());
        let job = JobSpec { id: "sleeper".into(), command: "sleep 30".into(), ..Default::default() };

        worker.launch(run_id.clone(), job, allocation()).await;
        assert_eq!(worker.running().await, 1);
        assert!(matches!(worker.kill(run_id.clone(), "sleeper".into()).await, Response::Killed));
        assert_eq!(worker.running().await, 0);
        assert!(matches!(worker.poll(run_id, "sleeper".into()).await, Response::Polled { exit_code: Some(-1) }));
    }
}
