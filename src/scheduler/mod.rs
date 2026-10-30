use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use tokio::time::{interval, Duration, Instant};

use crate::broadcast::SpmcSender;
use crate::daemon::DaemonState;
use crate::event::{monotonic_raw_ns, JobEvent};
use crate::executor::{ExecutorError, JobHandle, PollResult};
use crate::ipc::protocol::Request;
use crate::node::Node;
use crate::paths::{job_log_path, job_log_relpath};
use crate::resource::discovery::GpuInfo;
use crate::resource::placement::place;
use crate::resource::pool::{Allocation, ResourcePool};
use crate::workflow::model::{Execution, JobInstance, JobState};
use crate::workflow::spec::JobSpec;

const TICK_MS: u64 = 100;

/// Cards are checked every 20 ticks (2 s, as often as a node's health check);
/// one that stays out of placement this long gets a warning
const GPU_CHECK_TICKS: u64 = 20;
const GPU_BLOCKED_WARN: Duration = Duration::from_secs(30);

/// Qualification: how often an idle card is tested again, how long a test may take (DCGM's level 2
/// takes minutes), and how long to wait for the node's readings to show a finished test
const QUALIFY_EVERY: chrono::TimeDelta = chrono::TimeDelta::hours(24);
const QUALIFY_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const QUALIFY_SETTLE: Duration = Duration::from_secs(60);

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
    let mut checks = GpuChecks::default();
    for count in 0u64.. {
        ticker.tick().await;
        if count % GPU_CHECK_TICKS == 0 {
            check_gpus(&state, &mut checks).await;
        }
        tick(&state, &event_sender).await;
    }
}

/// What the 2 s GPU check carries from one run to the next
#[derive(Default)]
struct GpuChecks {
    blocked_since:  HashMap<(usize, u32), (Instant, bool)>, // when a card was first seen blocked, warned yet
    qualifying:     Vec<Qualification>,
    last_qualified: HashMap<(usize, u32), Instant>, // so a reading the node hasn't updated yet starts no second run
}

/// `roster qualify` running as an internal job that holds a whole card
struct Qualification {
    node:       usize,
    gpu:        u32,
    allocation: Allocation,
    handle:     JobHandle,
    started:    Instant,
}

/// Keep cards out of placement for as long as `GpuHealth::blocked` gives a reason: quarantined by
/// their node, or, with nothing booked, busy with processes roster didn't start or throttled;
/// readings come from each node's last health check, or NVML on this machine
/// then qualify idle cards that need it
async fn check_gpus(state: &Arc<DaemonState>, checks: &mut GpuChecks) {
    finish_qualifications(state, checks).await;

    // read before taking the pools lock
    let mut readings = Vec::new();
    for node in &state.nodes {
        readings.push(match &node.ssh {
            None => state.gpu_monitor.read(&state.resources.gpus),
            Some(_) => node.health().await.info.map(|info| info.gpu_health).unwrap_or_default(),
        });
    }

    let mut to_qualify = Vec::new();
    {
        let mut pools = state.pools.lock().await;
        let mut requalify = state.requalify.lock().unwrap();
        for (index, (node, reading)) in state.nodes.iter().zip(readings).enumerate() {
            let pool = &mut pools[index];
            for gpu in pool.total.gpus.clone() {
                let health = reading.iter().find(|health| health.index == gpu.index);
                let unbooked = pool.available_vram_mb[gpu.index as usize] == gpu.vram_mb;
                let reason = health.and_then(|health| health.blocked(unbooked));
                pool.blocked[gpu.index as usize] = reason.is_some();

                let key = (index, gpu.index);
                match reason {
                    Some(reason) => {
                        let (since, warned) = checks.blocked_since.entry(key).or_insert((Instant::now(), false));
                        if !*warned && since.elapsed() >= GPU_BLOCKED_WARN {
                            tracing::warn!(node = %node.name, gpu = gpu.index, %reason, "GPU not handed out");
                            *warned = true;
                        }
                    }
                    None => {
                        if let Some((_, true)) = checks.blocked_since.remove(&key) {
                            tracing::info!(node = %node.name, gpu = gpu.index, "GPU free again");
                        }
                    }
                }

                // an idle card, untested since its node started, a day ago, after a new fault,
                // or after a job on it was killed; a quarantined card is tested again too
                let Some(health) = health else { continue };
                let due = requalify.contains(&key)
                    || health.tested_at.is_none_or(|tested| Utc::now() - tested > QUALIFY_EVERY);
                let recent = checks.last_qualified.get(&key).is_some_and(|last| last.elapsed() < QUALIFY_SETTLE);
                let idle = !health.busy() && health.throttled.is_empty();
                if due && !recent && idle && node.is_online() {
                    if let Some(allocation) = pool.reserve_card(gpu.index) {
                        requalify.remove(&key);
                        to_qualify.push((index, gpu, health.quarantined.is_some(), allocation));
                    }
                }
            }
        }
    } // locks released before launching

    for (node, gpu, retest, allocation) in to_qualify {
        checks.last_qualified.insert((node, gpu.index), Instant::now());
        match start_qualification(state, node, &gpu, retest, &allocation).await {
            Some(handle) => checks.qualifying.push(Qualification { node, gpu: gpu.index, allocation, handle, started: Instant::now() }),
            None => state.pools.lock().await[node].release(&allocation),
        }
    }
}

/// Launch `roster qualify` on the node, holding the card; `retest` asks for DCGM's level 2 where installed
async fn start_qualification(state: &Arc<DaemonState>, node: usize, gpu: &GpuInfo, retest: bool, allocation: &Allocation)
        -> Option<JobHandle> {
    let mut command = format!("\"$ROSTER_BIN\" qualify --index {}", gpu.index);
    if !gpu.uuid.is_empty() {
        command.push_str(&format!(" --uuid {}", gpu.uuid));
    }
    if retest {
        command.push_str(" --retest");
    }
    let job = JobInstance::new(JobSpec { id: format!("gpu{}", gpu.index), command, ..Default::default() }, 0);
    let run_id = format!("qualify-{}", uuid::Uuid::new_v4().simple());

    let name = &state.nodes[node].name;
    match state.nodes[node].executor.launch(&run_id, &job, allocation).await {
        Ok(handle) => {
            tracing::info!(node = %name, gpu = gpu.index, retest, %run_id, "qualifying GPU");
            Some(handle)
        }
        Err(error) => {
            tracing::warn!(node = %name, gpu = gpu.index, %error, "could not start GPU qualification");
            None
        }
    }
}

/// Release the cards of qualifications that ended; one past its timeout is stopped and counts
/// as a failure, since a hung test is no pass; the node records the result either way
async fn finish_qualifications(state: &Arc<DaemonState>, checks: &mut GpuChecks) {
    let mut ended = Vec::new();
    for qualification in std::mem::take(&mut checks.qualifying) {
        let node = &state.nodes[qualification.node];
        match node.executor.poll(&qualification.handle).await {
            Ok(PollResult::Running) if qualification.started.elapsed() > QUALIFY_TIMEOUT => {
                let _ = node.executor.cancel(&qualification.handle).await;
                let reason = format!("no result within {} minutes", QUALIFY_TIMEOUT.as_secs() / 60);
                let request = Request::GpuQualified { index: qualification.gpu, passed: false, reason };
                if let Err(error) = node.request(request).await {
                    tracing::warn!(node = %node.name, gpu = qualification.gpu, %error, "could not report a timed-out qualification");
                }
                ended.push(qualification);
            }
            // still running, or the node can't be asked now: keep holding the card
            Ok(PollResult::Running) | Err(ExecutorError::Unreachable(_)) => checks.qualifying.push(qualification),
            Ok(PollResult::Exited { .. }) | Err(_) => ended.push(qualification),
        }
    }
    if !ended.is_empty() {
        let mut pools = state.pools.lock().await;
        for qualification in &ended {
            pools[qualification.node].release(&qualification.allocation);
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
                    let executions = std::mem::take(&mut job.executions);
                    // killed by a signal it wasn't sent by a cancel, e.g. the GPU faulted: test its cards
                    if exit_code == -1 && !entry.cancelling {
                        let mut requalify = state.requalify.lock().unwrap();
                        for execution in &executions {
                            requalify.extend(execution.allocation.gpus.iter().map(|gpu| (execution.node, gpu.index)));
                        }
                    }
                    release(&mut pools, &executions);

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