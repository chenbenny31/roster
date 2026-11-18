//! Worker side: run the jobs a controller places on this machine

use std::collections::HashMap;

use tokio::sync::Mutex;

use crate::daemon::DaemonState;
use crate::executor::shell::ShellExecutor;
use crate::executor::{Executor, ExecutorError, JobHandle, PollResult};
use crate::ipc::protocol::Response;
use crate::resource::pool::{Allocation, Holder};
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
            Err(ExecutorError::Rejected(reason)) => Response::Rejected { reason },
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

    /// The end of a job's log here, whether or not this daemon started the job
    pub async fn tail(&self, run_id: String, job_id: String, bytes: u64) -> Response {
        match ShellExecutor.tail(&run_id, &job_id, bytes).await {
            Ok(text) => Response::Log { text },
            Err(error) => Response::Error { message: error.to_string() },
        }
    }

    /// The jobs started here that have exited, by id
    pub async fn ended(&self) -> Vec<Holder> {
        let mut processes = self.processes.lock().await;
        let mut ended = Vec::new();
        for ((run_id, job_id), process) in processes.iter_mut() {
            if matches!(process.exit_code().await, Ok(Some(_))) {
                ended.push(Holder::new(run_id, job_id));
            }
        }
        ended
    }

    /// A job found again after this daemon restarted, so its exit is seen without a controller asking
    pub async fn adopt(&self, holder: &Holder, handle: JobHandle) {
        self.processes.lock().await
            .entry((holder.run_id.clone(), holder.job_id.clone()))
            .or_insert(JobProcess { handle, exit_code: None });
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

/// A controller's job or card test on this machine: book its cards here under its id, so this machine's own
/// scheduler and other controllers see them, then start it; a card another id holds answers Busy, after
/// giving back the cards of jobs that already ended
pub async fn book_and_launch(state: &DaemonState, run_id: String, job: JobSpec, allocation: Allocation) -> Response {
    let Some(local) = state.local_node() else {
        return state.worker.launch(run_id, job, allocation).await; // nothing here books this machine's cards
    };
    let holder = Holder::new(&run_id, &job.id);
    let mut booked = state.pools.lock().await[local].try_book(&holder, &allocation);
    if booked.is_err() {
        free_ended(state).await; // the job holding it may have just ended
        booked = state.pools.lock().await[local].try_book(&holder, &allocation);
    }
    if let Err((gpu, other)) = booked {
        return Response::Busy { reason: format!("GPU {} is held by {}", gpu, other) };
    }
    if let Err(error) = state.store.save_hold(&holder, &allocation).await {
        tracing::warn!(%holder, %error, "could not save what a job holds; a restart here would forget it");
    }

    let response = state.worker.launch(run_id, job, allocation).await;
    if !matches!(response, Response::Launched { .. }) {
        unbook(state, local, &holder).await;
    }
    response
}

/// Give back here the cards of controllers' jobs that ended; every 2 s, and when a launch finds a card held
pub async fn free_ended(state: &DaemonState) {
    let Some(local) = state.local_node() else { return };
    for holder in state.worker.ended().await {
        if state.pools.lock().await[local].holds(&holder) {
            unbook(state, local, &holder).await;
        }
    }
}

async fn unbook(state: &DaemonState, local: usize, holder: &Holder) {
    state.pools.lock().await[local].release(holder);
    if let Err(error) = state.store.drop_hold(holder).await {
        tracing::warn!(%holder, %error, "could not forget what an ended job held");
    }
}

/// After a restart, before the scheduler or a launch can hand them out: book again the cards of
/// controllers' jobs still here, running or ended while the daemon was down; one that left nothing is forgotten
pub async fn take_back_holds(state: &DaemonState) -> anyhow::Result<()> {
    let Some(local) = state.local_node() else { return Ok(()) };
    for (holder, allocation) in state.store.holds().await? {
        match ShellExecutor::find(&holder.run_id, &holder.job_id).await {
            Some(handle) => {
                state.worker.adopt(&holder, handle).await;
                state.pools.lock().await[local].book(&holder, &allocation);
                tracing::info!(%holder, "took back what a controller's job holds here");
            }
            None => state.store.drop_hold(&holder).await?,
        }
    }
    Ok(())
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
    async fn ended_lists_jobs_that_exited() {
        let worker = Worker::default();
        let run_id = format!("worker-test-{}", uuid::Uuid::new_v4());
        let quick = JobSpec { id: "quick".into(), command: "true".into(), ..Default::default() };
        let slow = JobSpec { id: "slow".into(), command: "sleep 30".into(), ..Default::default() };
        worker.launch(run_id.clone(), quick, allocation()).await;
        worker.launch(run_id.clone(), slow, allocation()).await;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        while worker.ended().await.is_empty() && tokio::time::Instant::now() < deadline {
            sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(worker.ended().await, vec![Holder::new(&run_id, "quick")]);
        worker.kill(run_id.clone(), "slow".into()).await;
        assert_eq!(worker.ended().await.len(), 2);
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
