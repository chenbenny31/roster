use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

use crate::workflow::spec::{JobSpec, WorkflowSpec};
use crate::resource::pool::Allocation;

/// Run level state, mirrors JobState but no Skipped/Interrupted
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RunState {
    Pending, // no jobs started
    Running, // at least one job active
    Succeeded, // all jobs terminal, no failure
    Failed, // any job ended in Failed/TimeOut/Interrupted
    Cancelled, // user cancelled, no failures during cancel
}

/// Full 9 variant job state machine
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JobState {
    Pending, // waiting for deps to finish
    Queued, // deps satisfied, waiting for resources
    Running, // executing as subprocess
    Succeeded, // exit code 0
    Failed, // exist != 0 or executor error
    Skipped, // a dep failed, neve run
    Cancelled, // user-fired stop
    TimedOut, // exceeded declared timeout
    Interrupted, // daemon shut down mid-job, no auto-resume
}

impl JobState {
    /// Returns true if no further transitions are possible
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobState::Succeeded
            | JobState::Failed
            | JobState::Skipped
            | JobState::Cancelled
            | JobState::TimedOut
            | JobState::Interrupted
        )
    }
}

/// Runtime state of a single job within a workflow run
#[derive(Debug, Clone)]
pub struct JobRun {
    pub job_id: String,
    pub spec: JobSpec,
    pub state: JobState,
    pub pid: Option<u32>, // set when Running, cleared on termin
    pub allocation: Option<Allocation>,
    pub cancelling: bool, // set before killpg, prevents non-zero exit
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub exit_code: Option<u32>,
    pub log_path: Option<PathBuf>,
}

impl JobRun {
    pub fn new(spec: JobSpec) -> Self {
        Self {
            job_id: spec.id.clone(),
            spec,
            state: JobState::Pending,
            pid: None,
            allocation: None,
            cancelling: false,
            started_at: None,
            ended_at: None,
            exit_code: None,
            log_path: None,
        }
    }
}

/// Runtime state of an entire workflow run
#[derive(Debug)]
pub struct WorkflowRun {
    pub run_id: String,
    pub workflow_name: String,
    pub spec: WorkflowSpec,
    pub jobs: HashMap<String, JobRun>,
    pub created_at: DateTime<Utc>,
}

impl WorkflowRun {
    pub fn new(run_id: String, spec: WorkflowSpec) -> Self {
        let jobs = spec.jobs
            .iter()
            .map(|job_spec| (job_spec.id.clone(), JobRun::new(job_spec.clone())))
            .collect();

        let workflow_name = spec.name.clone();

        Self {
            run_id,
            workflow_name,
            spec,
            jobs,
            created_at: Utc::now(),
        }
    }

    /// Derive run state from job states, computed on demand
    /// Precedence: Failed/TimeOut/Interrupted > Cancelled > Running > Succeeded > Pending
    pub fn status(&self) -> RunState {
        if self.jobs.values().any(|j| matches!(j.state, JobState::Failed | JobState::TimedOut | JobState::Interrupted)) {
            return RunState::Failed;
        }
        if self.jobs.values().any(|j| matches!(j.state, JobState::Cancelled)) {
            return RunState::Cancelled;
        }
        if self.jobs.values().any(|j| matches!(j.state, JobState::Running | JobState::Queued)) {
            return RunState::Running;
        }
        if self.jobs.values().all(|j| matches!(j.state, JobState::Succeeded | JobState::Skipped)) {
            return RunState::Succeeded;
        }
        RunState::Pending
    }
}

// tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::spec::parse;

    const EXAMPLE: &str = r#"
name: train-and-eval
jobs:
  - id: preprocess
    command: python preprocess.py
    resources:
      cpu: 4
      memory_mb: 4096
  - id: train
    depends_on: [preprocess]
    command: python train.py
    resources:
      cpu: 8
      gpu: 0
      vram_mb: 12000
  - id: eval
    depends_on: [train]
    command: python eval.py
    resources:
      cpu: 2
      gpu: 0
      vram_mb: 4000
"#;

    #[test]
    fn all_jobs_start_pending() {
        let spec = parse(EXAMPLE).unwrap();
        let run = WorkflowRun::new("run-001".into(), spec);

        for job in run.jobs.values() {
            assert_eq!(job.state, JobState::Pending);
        }
    }

    #[test]
    fn run_starts_pending() {
        let spec = parse(EXAMPLE).unwrap();
        let run = WorkflowRun::new("run-001".into(), spec);

        assert_eq!(run.status(), RunState::Pending);
    }

    #[test]
    fn terminal_states_correct() {
        assert!(JobState::Succeeded.is_terminal());
        assert!(JobState::Failed.is_terminal());
        assert!(JobState::Skipped.is_terminal());
        assert!(JobState::Interrupted.is_terminal());
        assert!(!JobState::Pending.is_terminal());
        assert!(!JobState::Running.is_terminal());
        assert!(!JobState::Queued.is_terminal());
    }

    #[test]
    fn failed_job_makes_run_failed() {
        let spec = parse(EXAMPLE).unwrap();
        let mut run = WorkflowRun::new("r1".into(), spec);
        run.jobs.get_mut("preprocess").unwrap().state = JobState::Failed;
        run.jobs.get_mut("train").unwrap().state = JobState::Running;
        assert_eq!(run.status(), RunState::Failed);
    }

    #[test]
    fn failed_bests_cancelled() {
        let spec = parse(EXAMPLE).unwrap();
        let mut run = WorkflowRun::new("r1".into(), spec);
        run.jobs.get_mut("preprocess").unwrap().state = JobState::Failed;
        run.jobs.get_mut("train").unwrap().state = JobState::Cancelled;
        assert_eq!(run.status(), RunState::Failed);
    }

    #[test]
    fn all_succeeded_or_skipped_is_succeeded() {
        let spec = parse(EXAMPLE).unwrap();
        let mut run = WorkflowRun::new("r1".into(), spec);
        run.jobs.get_mut("preprocess").unwrap().state = JobState::Succeeded;
        run.jobs.get_mut("train").unwrap().state = JobState::Skipped;
        run.jobs.get_mut("eval").unwrap().state = JobState::Skipped;
        assert_eq!(run.status(), RunState::Succeeded);
    }
}