use std::sync::Arc;

use chrono::Utc;
use tokio::time::{interval, Duration};

use crate::daemon::DaemonState;
use crate::executor::{Executor, PollResult};
use crate::resource::pool::Allocation;
use crate::workflow::model::{JobRun, JobState};

const TICK_MS: u64 = 100;

/// Run the scheduler loop alongside the IPC listener, return only process exits
pub async fn run(state: Arc<DaemonState>) {
    let mut ticker = interval(Duration::from_millis(TICK_MS));
    loop {
        ticker.tick().await;
        tick(&state).await;
    }
}

/// One scheduler iteration, four pahses in order
async fn tick(state: &Arc<DaemonState>) {
    advance_pending(state).await;
    cascade_skipped(state).await;
    advance_queued(state).await;
    advance_running(state).await;
}


/// Pending -> Queued: transition jobs with succeeded dependencies
async fn advance_pending(state: &Arc<DaemonState>) {
    let mut runs = state.runs.lock().await;

    for run in runs.values_mut() {
        let to_queue: Vec<String> = run.jobs.values()
            .filter(|job| matches!(job.state, JobState::Pending))
            .filter(|job| {
                job.spec.depends_on.iter().all(|dep_id| {
                    matches!(
                        run.jobs.get(dep_id).map(|j| &j.state),
                        Some(JobState::Succeeded)
                    )
                })
            })
            .map(|job| job.job_id.clone())
            .collect();

        for job_id in to_queue {
            if let Some(job) = run.jobs.get_mut(&job_id) {
                job.state = JobState::Queued;
                tracing::info!(run_id = %run.run_id, %job_id, "job -> Queued");
            }
        }
    }
}

/// Mark Pending/Queued jobs Skipped if missing any dependency
async fn cascade_skipped(state: &Arc<DaemonState>) {
    let mut runs = state.runs.lock().await;

    for run in runs.values_mut() {
        let to_skip: Vec<String> = run.jobs.values()
            .filter(|job| matches!(job.state, JobState::Pending | JobState::Queued))
            .filter(|job| {
                job.spec.depends_on.iter().any(|dep_id| {
                    matches!(
                        run.jobs.get(dep_id).map(|j| &j.state),
                        Some(
                            JobState::Failed
                            | JobState::Skipped
                            | JobState::Cancelled
                            | JobState::TimedOut
                            | JobState::Interrupted
                        )
                    )
                })
            })
            .map(|job| job.job_id.clone())
            .collect();

        for job_id in to_skip {
            if let Some(job) = run.jobs.get_mut(&job_id) {
                job.state = JobState::Skipped;
                job.ended_at = Some(Utc::now());
                tracing::info!(run_id = %run.run_id, %job_id, "job -> Skipped");
            }
        }
    }
}

/// Queued -> Running: reserve resource and launch admitted jobs
async fn advance_queued(state: &Arc<DaemonState>) {
    // phase 1: reserve under lock, collect launch tasks
    struct AdmitTask {
        run_id: String,
        job_run: JobRun,
        alloc: Allocation,
    }

    let mut tasks: Vec<AdmitTask> = Vec::new();

    {
        let runs = state.runs.lock().await;
        let mut pool = state.pool.lock().await;

        for run in runs.values() {
            for job in run.jobs.values() {
                if !matches!(job.state, JobState::Queued) {
                    continue;
                }
                if let Some(alloc) = pool.try_reserve(&job.spec.resources) {
                    tasks.push(AdmitTask {
                        run_id: run.run_id.clone(),
                        job_run: job.clone(),
                        alloc,
                    });
                }
            }
        }
    } // both locks released

    // phase 2: launch async, no lock held
    struct LaunchOutcome {
        run_id: String,
        job_id: String,
        alloc: Allocation,
        result: Result<u32, crate::executor::ExecutorError>,
    }

    let mut outcomes: Vec<LaunchOutcome> = Vec::new();

    for task in tasks {
        let result = state.executor.launch(&task.run_id, &task.job_run).await;
        outcomes.push(LaunchOutcome {
            run_id: task.run_id,
            job_id: task.job_run.job_id,
            alloc: task.alloc,
            result,
        });
    }

    // phase 3: apply results under lock
    {
        let mut runs = state.runs.lock().await;
        let mut pool = state.pool.lock().await;

        for outcome in outcomes {
            let run = match runs.get_mut(&outcome.run_id) {
                Some(run) => run,
                None => { pool.release(&outcome.alloc); continue; }
            };
            let job = match run.jobs.get_mut(&outcome.job_id) {
                Some(job) => job,
                None => { pool.release(&outcome.alloc); continue; }
            };

            match outcome.result {
                Ok(pid) => {
                    job.state = JobState::Running;
                    job.pid = Some(pid);
                    job.allocation = Some(outcome.alloc);
                    job.started_at = Some(Utc::now());
                    tracing::info!(run_id = %outcome.run_id, job_id = %outcome.job_id, "job -> Running");
                }
                Err(error) => {
                    tracing::error!(run_id = %outcome.run_id, job_id = %outcome.job_id, %error, "launch failed -> Failed");
                    pool.release(&outcome.alloc);
                    job.state = JobState::Failed;
                    job.ended_at = Some(Utc::now());
                }
            }
        }
    }
}

// Running -> terminal: poll each running job and apply exit results
async fn advance_running(state: &Arc<DaemonState>) {
    // phase 1: collect running jobs snapshot
    struct RunningJob {
        run_id: String,
        job_id: String,
        pid: u32,
        cancelling: bool,
    }

    let running: Vec<RunningJob> = {
        let runs = state.runs.lock().await;
        runs.values()
            .flat_map(|run| {
                run.jobs.values()
                    .filter(|job| matches!(job.state, JobState::Running))
                    .filter_map(|job| job.pid.map(|pid| RunningJob {
                        run_id: run.run_id.clone(),
                        job_id: job.job_id.clone(),
                        pid,
                        cancelling: job.cancelling,
                    }))
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    // phase 2: poll async, no lock
    struct PollEntry {
        run_id: String,
        job_id: String,
        cancelling: bool,
        result: Result<PollResult, crate::executor::ExecutorError>,
    }

    let mut entries: Vec<PollEntry> = Vec::new();

    for job in running {
        let result = state.executor.poll(job.pid).await;
        entries.push(PollEntry {
            run_id: job.run_id,
            job_id: job.job_id,
            cancelling: job.cancelling,
            result,
        });
    }

    // phase 3: apply result under lock
    {
        let mut runs = state.runs.lock().await;
        let mut pool = state.pool.lock().await;

        for entry in entries {
            let run = match runs.get_mut(&entry.run_id) {
                Some(run) => run,
                None => continue,
            };
            let job = match run.jobs.get_mut(&entry.job_id) {
                Some(job) => job,
                None => continue,
            };

            match entry.result {
                Err(error) => {
                    tracing::error!(run_id = %entry.run_id, job_id = %entry.job_id, %error, "poll failed -> Failed");
                    if let Some(alloc) = job.allocation.take() { pool.release(&alloc); }
                    job.state = JobState::Failed;
                    job.ended_at = Some(Utc::now());
                    job.pid = None;
                }
                Ok(PollResult::Running) => {} // no chance
                Ok(PollResult::Exited { exit_code }) => {
                    if let Some(alloc) = job.allocation.take() { pool.release(&alloc); }

                    job.exit_code = Some(exit_code);
                    job.ended_at = Some(Utc::now());
                    job.pid = None;

                    // cancelling flag prevents non-zero exit, failed when kill the job
                    job.state = if entry.cancelling {
                        tracing::info!(run_id = %entry.run_id, job_id = %entry.job_id, "job -> Cancelled");
                        JobState::Cancelled
                    } else if exit_code == 0 {
                        tracing::info!(run_id = %entry.run_id, job_id = %entry.job_id, "job -> Succeeded");
                        JobState::Succeeded
                    } else {
                        tracing::info!(run_id = %entry.run_id, job_id = %entry.job_id, "job -> Failed");
                        JobState::Failed
                    };
                }
            }
        }
    }
}

