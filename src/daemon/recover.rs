//! After a restart the controller takes back the runs that hadn't ended: pending and queued jobs carry on,
//! running jobs keep what they hold and are polled where they run, and one that can't be found again
//! ends as Interrupted

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::daemon::DaemonState;
use crate::executor::shell::ShellExecutor;
use crate::executor::JobHandle;
use crate::resource::pool::Allocation;
use crate::store::{JobRow, SavedJob};
use crate::workflow::model::{Execution, JobInstance, JobState, WorkflowRun};
use crate::workflow::spec::WorkflowSpec;

/// Runs before the listener and the scheduler start, so placement already counts what running jobs hold
pub async fn take_back_runs(state: &Arc<DaemonState>) -> anyhow::Result<()> {
    for saved in state.store.unfinished_runs().await? {
        let spec: Option<WorkflowSpec> = saved.spec_json.as_deref().and_then(|json| serde_json::from_str(json).ok());
        let Some(spec) = spec else {
            // stored by an earlier release, without its spec: nothing to rebuild it from
            for job in &saved.jobs {
                if JobState::from_str(&job.row.state).is_some_and(|job_state| !job_state.is_terminal()) {
                    state.store.interrupt(&saved.run_id, &job.row.job_id).await?;
                }
            }
            continue;
        };

        let run_seq = state.run_counter.fetch_add(1, Ordering::Relaxed);
        let job_seq = state.job_counter.fetch_add(spec.jobs.len() as u64, Ordering::Relaxed);
        let mut run = WorkflowRun::new(saved.run_id.clone(), run_seq, spec, job_seq);
        if let Some(created_at) = parse_time(&saved.created_at) {
            run.created_at = created_at;
        }

        let mut pools = state.pools.lock().await;
        for saved_job in &saved.jobs {
            let Some(job) = run.jobs.get_mut(&saved_job.row.job_id) else { continue };
            restore(job, &saved_job.row);
            if job.state != JobState::Running {
                continue;
            }
            match reattach(state, &saved.run_id, saved_job).await {
                Some(execution) => {
                    pools[execution.node].book(&execution.allocation);
                    tracing::info!(run_id = %saved.run_id, job_id = %job.job_id, node = %state.nodes[execution.node].name,
                                   "took back a running job");
                    job.executions = vec![execution];
                }
                None => {
                    tracing::warn!(run_id = %saved.run_id, job_id = %job.job_id, "running job not found again -> Interrupted");
                    job.state = JobState::Interrupted;
                    job.ended_at = Some(Utc::now());
                    state.store.interrupt(&saved.run_id, &job.job_id).await?;
                }
            }
        }
        drop(pools);
        state.runs.lock().await.insert(saved.run_id.clone(), run);
    }
    Ok(())
}

/// A running job's execution again: its node by name, what it held there, and a handle to poll it by;
/// None when its node is no longer in nodes.yaml, or on this machine nothing of the job is left
async fn reattach(state: &DaemonState, run_id: &str, saved: &SavedJob) -> Option<Execution> {
    let node = state.nodes.iter().position(|node| Some(node.name.as_str()) == saved.node.as_deref())?;
    let allocation: Allocation = serde_json::from_str(saved.allocation.as_deref()?).ok()?;
    let job_id = &saved.row.job_id;
    let handle = match state.nodes[node].ssh {
        None => ShellExecutor::find(run_id, job_id).await?, // its scope, or its exit file
        Some(_) => JobHandle::remote(0, run_id.to_string(), job_id.clone()), // the node finds it by the same ids
    };
    Some(Execution { node, allocation, handle: Some(handle) })
}

/// The job's last stored state over the fresh instance built from the spec
fn restore(job: &mut JobInstance, row: &JobRow) {
    job.state      = JobState::from_str(&row.state).unwrap_or(JobState::Pending);
    job.exit_code  = row.exit_code.map(|code| code as i32);
    job.started_at = row.started_at.as_deref().and_then(parse_time);
    job.ended_at   = row.ended_at.as_deref().and_then(parse_time);
    job.log_path   = row.log_path.as_ref().map(PathBuf::from);
    job.reason     = row.reason.clone();
}

fn parse_time(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text).ok().map(|time| time.with_timezone(&Utc))
}
