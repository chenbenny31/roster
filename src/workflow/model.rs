use std::collections::HashMap; // job_id -> JobRun lookup

use crate::workflow::spec::{JobSpec, WorkflowSpec};

/// Full 9-variant job state machine
#[derive(Debug, Clone, PartialEq)]
pub enum JobState {
    Pending, // wait for deps to finish
    Queued, // deps satisfied, wait for resources
    Running, // exec as a sub proc
    Succeeded, // exit code 0
    Failed, // exit code != 0 or error
    Skipped, // one dep failed, never run
    Cancelled, // user-init stop
    TimedOut, // exceed declared timeout
    Interrupted, // daemon shut down, no auto-resume
}

impl JobState {
    /// Returns true if not further transitions are possible
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
    pub pid: Option<u32>, // set when running, cleared on terminal state
}

impl JobRun {
    pub fn new(spec: JobSpec) -> Self {
        Self {
            job_id: spec.id.clone(),
            spec,
            state: JobState::Pending,
            pid: None,
        }
    }
}

// Runtime state of entire workflow run
#[derive(Debug)]
pub struct WorkflowRun {
    pub run_id: String,
    pub spec: WorkflowSpec,
    pub jobs: HashMap<String, JobRun>,
}

impl WorkflowRun {
    pub fn new(run_id: String, spec: WorkflowSpec) -> Self {
        let jobs = spec.jobs
            .iter()
            .map(|job_spec| (job_spec.id.clone(), JobRun::new(job_spec.clone())))
            .collect();

        Self { run_id, spec, jobs }
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
    fn job_count_matches_spec() {
        let spec = parse(EXAMPLE).unwrap();
        let run = WorkflowRun::new("run-001".into(), spec);
        assert_eq!(run.jobs.len(), 3);
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
}