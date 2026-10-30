use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use tokio::time::{interval, Duration, Instant};

use crate::broadcast::SpmcSender;
use crate::daemon::DaemonState;
use crate::event::{monotonic_raw_ns, JobEvent};
use crate::executor::{ExecutorError, JobHandle, PollResult};
use crate::node::Node;
use crate::paths::{job_log_path, job_log_relpath};
use crate::resource::placement::place;
use crate::resource::pool::{Allocation, ResourcePool};
use crate::workflow::model::{Execution, JobInstance, JobState};

const TICK_MS: u64 = 100;

/// Cards are checked every 20 ticks (2 s, as often as a node's health check);
/// one that stays out of placement this long gets a warning
const GPU_CHECK_TICKS: u64 = 20;
const GPU_BLOCKED_WARN: Duration = Duration::from_secs(30);

/// Return each execution's allocation to its own node's pool
fn release(pools: &mut [ResourcePool], executions: &[Execution]) {
    for execution in executions {
        pools[execution.node].release(&execution.allocation);
    }
}

/// Where a job's log lives: a local path, or `ssh-target:path-under-home` on a remote node
fn log_path(node: &Node, run_id: &str, job_id: &str) -> PathBuf {
    match &node.ssh {
        None => job_log_path(run_id, job_id),
        Some(target) => PathBuf::from(format!("{}:{}", target, job_log_relpath(run_id, job_id).display())),
    }
}

/// Run the scheduler loop alongside the IPC listener, return only process exits
/// owns the SpmcSender - the single writer to the event ring buffer
pub async fn run(state: Arc<DaemonState>, event_sender: SpmcSender<JobEvent>) {
    let mut ticker = interval(Duration::from_millis(TICK_MS));
    let mut blocked_since = HashMap::new();
    for count in 0u64.. {
        ticker.tick().await;
        if count % GPU_CHECK_TICKS == 0 {
            check_gpus(&state, &mut blocked_since).await;
        }
        tick(&state, &event_sender).await;
    }
}

/// Keep cards out of placement for as long as `GpuHealth::blocked` gives a reason: quarantined by
/// their node, or, with nothing booked, busy with processes roster didn't start or throttled;
/// readings come from each node's last health check, or NVML on this machine
/// `blocked_since` holds when each blocked card was first seen and whether it was warned about
async fn check_gpus(state: &Arc<DaemonState>, blocked_since: &mut HashMap<(usize, u32), (Instant, bool)>) {
    // read before taking the pools lock
    let mut readings = Vec::new();
    for node in &state.nodes {
        readings.push(match &node.ssh {
            None => state.gpu_monitor.read(&state.resources.gpus),
            Some(_) => node.health().await.info.map(|info| info.gpu_health).unwrap_or_default(),
        });
    }

    let mut pools = state.pools.lock().await;
    for (index, (node, reading)) in state.nodes.iter().zip(readings).enumerate() {
        let pool = &mut pools[index];
        for gpu in &pool.total.gpus {
            let health = reading.iter().find(|health| health.index == gpu.index);
            let unbooked = pool.available_vram_mb[gpu.index as usize] == gpu.vram_mb;
            let reason = health.and_then(|health| health.blocked(unbooked));
            pool.blocked[gpu.index as usize] = reason.is_some();

            let key = (index, gpu.index);
            match reason {
                Some(reason) => {
                    let (since, warned) = blocked_since.entry(key).or_insert((Instant::now(), false));
                    if !*warned && since.elapsed() >= GPU_BLOCKED_WARN {
                        tracing::warn!(node = %node.name, gpu = gpu.index, %reason, "GPU not handed out");
                        *warned = true;
                    }
                }
                None => {
                    if let Some((_, true)) = blocked_since.remove(&key) {
                        tracing::info!(node = %node.name, gpu = gpu.index, "GPU free again");
                    }
                }
            }
        }
    }
}

/// One scheduler iteration, four pahses in order
async fn tick(state: &Arc<DaemonState>, event_sender: &SpmcSender<JobEvent>) {
    advance_pending(state, event_sender).await;
    cascade_skipped(state, event_sender).await;
    advance_queued(state, event_sender).await;
    advance_running(state, event_sender).await;
}

/// Persist a job's current state to SQLite and emit a JobEvent
async fn flush_transitions(state: &Arc<DaemonState>,
                           event_sender: &SpmcSender<JobEvent>,
                           to_write: &[(String, String)]) {
    for (run_id, job_id) in to_write {
        let runs = state.runs.lock().await;
        if let Some(run) = runs.get(run_id) {
            if let Some(job) = run.jobs.get(job_id) {
                let _ = state.store.upsert_job(run_id, job).await;
                event_sender.send(JobEvent::new(
                    run.run_seq,
                    job.job_seq,
                    job.state,
                    monotonic_raw_ns(),
                ));
            }
        }
    }
}


/// Pending -> Queued: transition jobs with succeeded dependencies
async fn advance_pending(state: &Arc<DaemonState>, event_sender: &SpmcSender<JobEvent>) {
    // collect transitions first, write to SQLite after releasing lock
    let mut to_write: Vec<(String, String)> = Vec::new(); // (run_id, job_id)

    {
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
                    to_write.push((run.run_id.clone(), job_id));
                }
            }
        }
    } // lock release

    flush_transitions(state, event_sender, &to_write).await;
}

/// Mark Pending/Queued jobs Skipped if any dep is in a terminal non-success state
async fn cascade_skipped(state: &Arc<DaemonState>, event_sender: &SpmcSender<JobEvent>) {
    let mut to_write: Vec<(String, String)> = Vec::new();

    {
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
                    job.state    = JobState::Skipped;
                    job.ended_at = Some(Utc::now());
                    tracing::info!(run_id = %run.run_id, %job_id, "job -> Skipped");
                    to_write.push((run.run_id.clone(), job_id));
                }
            }
        }
    }

    flush_transitions(state, event_sender, &to_write).await;
}

/// Queued -> Running: reserve resource and launch admitted jobs
async fn advance_queued(state: &Arc<DaemonState>, event_sender: &SpmcSender<JobEvent>) {
    // phase 1: reserve under lock, collect launch tasks
    struct AdmitTask {
        run_id:  String,
        job_instance: JobInstance,
        placement: Vec<(usize, Allocation)>, // (node index, allocation) per member
    }

    let mut tasks: Vec<AdmitTask> = Vec::new();

    {
        let runs = state.runs.lock().await;
        let mut pools = state.pools.lock().await;
        let online: Vec<bool> = state.nodes.iter().map(|node| node.is_online()).collect();

        for run in runs.values() {
            for job in run.jobs.values() {
                if !matches!(job.state, JobState::Queued) {
                    continue;
                }
                // one member per job until coordinated jobs add a member count
                if let Some(placement) = place(&mut pools, &online, std::slice::from_ref(&job.spec.resources)) {
                    tasks.push(AdmitTask {
                        run_id: run.run_id.clone(),
                        job_instance: job.clone(),
                        placement,
                    });
                }
            }
        }
    } // both locks released

    // phase 2: launch async, no lock held
    struct LaunchOutcome {
        run_id: String,
        job_id: String,
        executions: Vec<Execution>,
        result: Result<(), crate::executor::ExecutorError>,
    }

    let mut outcomes: Vec<LaunchOutcome> = Vec::new();

    for task in tasks {
        let mut executions = Vec::new();
        let mut result = Ok(());
        for (node, allocation) in task.placement {
            let launched = state.nodes[node].executor.launch(&task.run_id, &task.job_instance, &allocation).await;
            let handle = match launched {
                Ok(handle) => Some(handle),
                Err(error) => { result = Err(error); None }
            };
            executions.push(Execution { node, allocation, handle });
        }
        outcomes.push(LaunchOutcome {
            run_id: task.run_id,
            job_id: task.job_instance.job_id,
            executions,
            result,
        });
    }

    let mut to_write: Vec<(String, String)> = Vec::new();

    // phase 3: apply results under lock
    {
        let mut runs = state.runs.lock().await;
        let mut pools = state.pools.lock().await;

        for outcome in outcomes {
            let run = match runs.get_mut(&outcome.run_id) {
                Some(run) => run,
                None      => { release(&mut pools, &outcome.executions); continue; }
            };
            let job = match run.jobs.get_mut(&outcome.job_id) {
                Some(job) => job,
                None      => { release(&mut pools, &outcome.executions); continue; }
            };

            match outcome.result {
                Ok(()) => {
                    // one execution per job for now, its node keeps the log
                    let log = log_path(&state.nodes[outcome.executions[0].node], &outcome.run_id, &outcome.job_id);
                    job.state       = JobState::Running;
                    job.executions  = outcome.executions;
                    job.started_at  = Some(Utc::now());
                    job.log_path    = Some(log);
                    tracing::info!(run_id = %outcome.run_id, job_id = %outcome.job_id, "job -> Running");
                    to_write.push((outcome.run_id, outcome.job_id));
                }
                // the request never left, so the job can wait for another node
                Err(ExecutorError::Unreachable(reason)) => {
                    tracing::debug!(run_id = %outcome.run_id, job_id = %outcome.job_id, %reason, "node unreachable, job stays Queued");
                    release(&mut pools, &outcome.executions);
                }
                Err(error) => {
                    tracing::error!(run_id = %outcome.run_id, job_id = %outcome.job_id, %error, "launch failed -> Failed");
                    // with several members, the ones already launched must be stopped first
                    release(&mut pools, &outcome.executions);
                    job.state    = JobState::Failed;
                    job.ended_at = Some(Utc::now());
                    to_write.push((outcome.run_id, outcome.job_id));
                }
            }
        }
    }

    flush_transitions(state, event_sender, &to_write).await;
}

// Running -> terminal: poll each running job and apply exit results
async fn advance_running(state: &Arc<DaemonState>, event_sender: &SpmcSender<JobEvent>) {
    // phase 1: collect running jobs snapshot
    struct RunningJob {
        run_id:     String,
        job_id:     String,
        node:       usize,
        handle:     JobHandle,
        cancelling: bool,
    }

    let running: Vec<RunningJob> = {
        let runs = state.runs.lock().await;
        runs.values()
            .flat_map(|run| {
                run.jobs.values()
                    .filter(|job| matches!(job.state, JobState::Running))
                    .filter_map(|job| {
                        // one execution per job for now; polling several and combining
                        // their outcomes comes with coordinated jobs
                        let execution = job.executions.first()?;
                        execution.handle.clone().map(|handle| RunningJob {
                            run_id:     run.run_id.clone(),
                            job_id:     job.job_id.clone(),
                            node:       execution.node,
                            handle,
                            cancelling: job.cancelling,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };

    // phase 2: poll async, no lock
    struct PollEntry {
        run_id:     String,
        job_id:     String,
        cancelling: bool,
        result:     Result<PollResult, crate::executor::ExecutorError>,
    }

    let mut entries: Vec<PollEntry> = Vec::new();

    for job in running {
        let result = state.nodes[job.node].executor.poll(&job.handle).await;
        entries.push(PollEntry {
            run_id:     job.run_id,
            job_id:     job.job_id,
            cancelling: job.cancelling,
            result,
        });
    }

    let mut to_write: Vec<(String, String)> = Vec::new();

    // phase 3: apply result under lock
    {
        let mut runs = state.runs.lock().await;
        let mut pools = state.pools.lock().await;

        for entry in entries {
            let run = match runs.get_mut(&entry.run_id) {
                Some(run) => run,
                None      => continue,
            };
            let job = match run.jobs.get_mut(&entry.job_id) {
                Some(job) => job,
                None      => continue,
            };

            match entry.result {
                // nothing is known, so the job stays Running and keeps its resources until the node answers
                Err(ExecutorError::Unreachable(_)) => {}
                Err(error) => {
                    tracing::error!(run_id = %entry.run_id, job_id = %entry.job_id, %error, "poll failed -> Failed");
                    release(&mut pools, &std::mem::take(&mut job.executions));
                    job.state    = JobState::Failed;
                    job.ended_at = Some(Utc::now());
                    to_write.push((entry.run_id, entry.job_id));
                }
                Ok(PollResult::Running) => {} // no change
                Ok(PollResult::Exited { exit_code }) => {
                    release(&mut pools, &std::mem::take(&mut job.executions));

                    job.exit_code = Some(exit_code);
                    job.ended_at  = Some(Utc::now());

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

                    to_write.push((entry.run_id, entry.job_id));
                }
            }
        }
    }

    flush_transitions(state, event_sender, &to_write).await;
}