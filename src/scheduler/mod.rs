use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use tokio::time::{interval, Duration, Instant};

use crate::broadcast::SpmcSender;
use crate::daemon::DaemonState;
use crate::event::{monotonic_raw_ns, JobEvent};
use crate::executor::shell::CUT_NOTICE;
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

/// How much of a failed job's log is read for its reason; the last lines usually name the error
const REASON_TAIL_BYTES: u64 = 4096;
const REASON_LINES: usize = 3;

/// Return each execution's allocation to its own node's pool
fn release(pools: &mut [ResourcePool], executions: &[Execution]) {
    for execution in executions {
        pools[execution.node].release(&execution.allocation);
    }
}

/// Why a job ended badly, from the last lines of its log, e.g.
/// "exit 1: File "train.py", line 1, in <module> | import torch | ModuleNotFoundError: No module named 'torch'"
fn exit_reason(exit_code: i32, tail: Option<&str>) -> String {
    let ended = match exit_code {
        -1   => "killed by a signal".to_string(),
        code => format!("exit {}", code),
    };
    // progress bars redraw with \r, so each redraw counts as a line; Python's ^^^ and ~~~ lines only point
    // at the line above
    let lines: Vec<&str> = tail.unwrap_or_default()
        .split(['\n', '\r'])
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != CUT_NOTICE && !line.chars().all(|c| matches!(c, '^' | '~')))
        .collect();
    match lines.len() {
        0 => ended,
        n => format!("{}: {}", ended, lines[n.saturating_sub(REASON_LINES)..].join(" | ")),
    }
}

/// A job keeps every reason it collected, e.g. the nodes that turned it down and then its exit
fn add_reason(job: &mut JobInstance, reason: String) {
    job.reason = Some(match job.reason.take() {
        Some(earlier) => format!("{}; {}", earlier, reason),
        None          => reason,
    });
}

/// Where a job's log lives: a local path, or `ssh-target:path-under-home` on a remote node
pub fn log_path(node: &Node, run_id: &str, job_id: &str) -> PathBuf {
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
            None => Some(state.gpu_monitor.read(&state.resources.gpus)),
            Some(_) => node.health().await.info.map(|info| info.gpu_health), // None until its first health check
        });
    }

    let mut to_qualify = Vec::new();
    {
        let mut pools = state.pools.lock().await;
        let mut requalify = state.requalify.lock().unwrap();
        for (index, (node, reading)) in state.nodes.iter().zip(readings).enumerate() {
            let pool = &mut pools[index];
            // an offline node gets no work anyway; until a reading arrives, at startup or once it is back,
            // its cards wait, since a node daemon that just started hasn't qualified them yet
            let Some(reading) = reading.filter(|_| node.is_online()) else {
                pool.blocked.fill(true);
                continue;
            };
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
                let node = job.executions.first().map(|execution| state.nodes[execution.node].name.as_str());
                let _ = state.store.upsert_job(run_id, job, node).await;
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
                let usable: Vec<bool> = online.iter().enumerate()
                    .map(|(node, &up)| up && !job.rejected_by(node))
                    .collect();
                // one member per job until coordinated jobs add a member count
                if let Some(placement) = place(&mut pools, &usable, std::slice::from_ref(&job.spec.resources)) {
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
                // the node won't run it, so the job waits for another node, or fails once no other node could take it;
                // either way `roster status` says which nodes turned it down and why
                Err(ExecutorError::Rejected(reason)) => {
                    let node = outcome.executions[0].node; // one member per job for now
                    tracing::warn!(run_id = %outcome.run_id, job_id = %outcome.job_id, node = %state.nodes[node].name, %reason, "node turned the job down");
                    release(&mut pools, &outcome.executions);
                    job.rejected.push((node, reason));
                    let reasons: Vec<String> = job.rejected.iter()
                        .map(|(node, reason)| format!("{}: {}", state.nodes[*node].name, reason))
                        .collect();
                    job.reason = Some(format!("turned down by {}", reasons.join("; ")));

                    let options = job.spec.resources.gpu_options();
                    let could_go_elsewhere = pools.iter().enumerate()
                        .any(|(node, pool)| !job.rejected_by(node) && options.iter().any(|option| pool.could_fit(option)));
                    if !could_go_elsewhere {
                        tracing::error!(run_id = %outcome.run_id, job_id = %outcome.job_id, "every node that fits turned the job down -> Failed");
                        job.state    = JobState::Failed;
                        job.ended_at = Some(Utc::now());
                    }
                    to_write.push((outcome.run_id, outcome.job_id));
                }
                Err(error) => {
                    tracing::error!(run_id = %outcome.run_id, job_id = %outcome.job_id, %error, "launch failed -> Failed");
                    // with several members, the ones already launched must be stopped first
                    release(&mut pools, &outcome.executions);
                    job.state    = JobState::Failed;
                    job.ended_at = Some(Utc::now());
                    add_reason(job, error.to_string());
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
        tail:       Option<String>, // end of the log of a job that failed on its own
    }

    let mut entries: Vec<PollEntry> = Vec::new();

    for job in running {
        let executor = &state.nodes[job.node].executor;
        let result = executor.poll(&job.handle).await;
        let tail = match &result {
            Ok(PollResult::Exited { exit_code }) if *exit_code != 0 && !job.cancelling =>
                executor.tail(&job.run_id, &job.job_id, REASON_TAIL_BYTES).await.ok(),
            _ => None,
        };
        entries.push(PollEntry {
            run_id:     job.run_id,
            job_id:     job.job_id,
            cancelling: job.cancelling,
            result,
            tail,
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
                    add_reason(job, format!("poll failed: {}", error));
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
                        let reason = exit_reason(exit_code, entry.tail.as_deref());
                        tracing::warn!(run_id = %entry.run_id, job_id = %entry.job_id, %reason, "job -> Failed");
                        add_reason(job, reason);
                        JobState::Failed
                    };

                    to_write.push((entry.run_id, entry.job_id));
                }
            }
        }
    }

    flush_transitions(state, event_sender, &to_write).await;
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_job_is_explained_by_the_end_of_its_log() {
        let traceback = "epoch 0\nTraceback (most recent call last):\n  File \"train.py\", line 1, in <module>\n    import torch\nModuleNotFoundError: No module named 'torch'\n";
        assert_eq!(exit_reason(1, Some(traceback)),
                   "exit 1: File \"train.py\", line 1, in <module> | import torch | ModuleNotFoundError: No module named 'torch'");
        assert_eq!(exit_reason(-1, Some("step 10\r step 11\r")), "killed by a signal: step 10 | step 11");
        assert_eq!(exit_reason(2, Some("[earlier output cut]\n\n")), "exit 2");
        let marked = "  File \"<string>\", line 1, in <module>\n    import torch\n    ^^^^^^^^^^^^\nModuleNotFoundError: No module named 'torch'\n";
        assert_eq!(exit_reason(1, Some(marked)),
                   "exit 1: File \"<string>\", line 1, in <module> | import torch | ModuleNotFoundError: No module named 'torch'");
        assert_eq!(exit_reason(137, None), "exit 137");
    }
}
