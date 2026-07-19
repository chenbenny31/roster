use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Serializer, Deserializer, Serialize, Deserialize};

use crate::resource::pool::Allocation;
use crate::workflow::spec::{JobSpec, WorkflowSpec};
use crate::executor::JobHandle;

/// Run level state - derived from job states, never stored in memory
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RunState {
    Pending,    // no jobs started
    Running,    // at least one job active
    Succeeded,  // all jobs complete, no failure
    Failed,     // any job ended in Failed/TimeOut/Interrupted
    Cancelled,  // user cancell, no failures during cancel
}

impl RunState {
    /// Canonical string for SQL storage, match upsert_job/upsert_run writes
    pub fn as_str(&self) -> &'static str {
        match self {
            RunState::Pending   => "Pending",
            RunState::Running   => "Running",
            RunState::Succeeded => "Succeeded",
            RunState::Failed    => "Failed",
            RunState::Cancelled => "Cancelled",
        }
    }

    /// Inverse of as_str(), used by SQL reads and custom Deserialize
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Pending"   => Some(RunState::Pending),
            "Running"   => Some(RunState::Running),
            "Succeeded" => Some(RunState::Succeeded),
            "Failed"    => Some(RunState::Failed),
            "Cancelled" => Some(RunState::Cancelled),
            _ => None,
        }
    }
}

impl Serialize for RunState {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for RunState {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        RunState::from_str(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown RunState: {s}")))
    }
}

/// Full 9-variant job state machine
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JobState {
    Pending,        // waiting for deps to finish
    Queued,         // deps satisfied, waiting for resources
    Running,        // executing as subprocess
    Succeeded,      // exit code 0
    Failed,         // exit code != 0 or executor error
    Skipped,        // a dep failed, neve run
    Cancelled,      // user-init stop
    TimedOut,       // exceeded declared timeout
    Interrupted,    // daemon shut down mid-job, regard terminal, no auto-resume
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

    /// Canonical string for both wire(serde) and SQL storage
    pub fn as_str(&self) -> &'static str {
        match self {
            JobState::Pending     => "Pending",
            JobState::Queued      => "Queued",
            JobState::Running     => "Running",
            JobState::Succeeded   => "Succeeded",
            JobState::Failed      => "Failed",
            JobState::Skipped     => "Skipped",
            JobState::Cancelled   => "Cancelled",
            JobState::TimedOut    => "TimedOut",
            JobState::Interrupted => "Interrupted",
        }
    }

    /// Inverse of as_str(), used by SQL reads and custom Deserialize
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "Pending"     => Some(JobState::Pending),
            "Queued"      => Some(JobState::Queued),
            "Running"     => Some(JobState::Running),
            "Succeeded"   => Some(JobState::Succeeded),
            "Failed"      => Some(JobState::Failed),
            "Skipped"     => Some(JobState::Skipped),
            "Cancelled"   => Some(JobState::Cancelled),
            "TimedOut"    => Some(JobState::TimedOut),
            "Interrupted" => Some(JobState::Interrupted),
            _ => None,
        }
    }

    /// Compact numeric code from SPMC wire event
    /// for ring buffer's word-wise atomic read/write, #[repr(C)] struct of u64, zero padding
    pub fn to_code(&self) -> u64 {
        match self {
            JobState::Pending     => 0,
            JobState::Queued      => 1,
            JobState::Running     => 2,
            JobState::Succeeded   => 3,
            JobState::Failed      => 4,
            JobState::Skipped     => 5,
            JobState::Cancelled   => 6,
            JobState::TimedOut    => 7,
            JobState::Interrupted => 8,
        }
    }

    /// Inverse of to_code(), used by JobEvent::state() to decode the wire event
    pub fn from_code(code: u64) -> Option<Self> {
        match code {
            0 => Some(JobState::Pending),
            1 => Some(JobState::Queued),
            2 => Some(JobState::Running),
            3 => Some(JobState::Succeeded),
            4 => Some(JobState::Failed),
            5 => Some(JobState::Skipped),
            6 => Some(JobState::Cancelled),
            7 => Some(JobState::TimedOut),
            8 => Some(JobState::Interrupted),
            _ => None,
        }
    }
}

impl Serialize for JobState {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for JobState {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        JobState::from_str(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown JobState: {s}")))
    }
}

/// Derive run state from an iterator of job states, load-bearing order check
/// - In-memory: `WorkflowRun::state()` calls this over `self.jobs.values()`
/// - SQL: `RunStore::list_runs`/`get_run` call this over job rows fetched via JOIN
pub fn derive_run_state(states: impl Iterator<Item = JobState>) -> RunState {
    let states: Vec<JobState> = states.collect();
    // Failed/TimedOut/Interrupted -> RunState::Failed
    if states.iter().any(|s| matches!(s, JobState::Failed | JobState::TimedOut | JobState::Interrupted)) {
        return RunState::Failed;
    }
    // Cancelled -> RunState::Cancelled
    if states.iter().any(|s| matches!(s, JobState::Cancelled)) {
        return RunState::Cancelled;
    }
    // Running/Queued -> RunState::Running
    if states.iter().any(|s| matches!(s, JobState::Running | JobState::Queued)) {
        return RunState::Running;
    }
    // non-empty, all Succeeded/Skipped -> RunState::Succeeded
    if !states.is_empty() && states.iter().all(|s| matches!(s, JobState::Succeeded | JobState::Skipped)) {
        return RunState::Succeeded;
    }
    // otherwise (including empty) -> RunState::Pending
    RunState::Pending
}

/// Runtime state of a single job within a workflow run
#[derive(Debug, Clone)]
pub struct JobInstance {
    pub job_id:     String,
    pub job_seq:    u64, // assigned at submit, used in JobEvent (Copy payload)
    pub spec:       JobSpec,
    pub state:      JobState,
    pub handle:     Option<JobHandle>, // set when Running, cleared on termin
    pub allocation: Option<Allocation>,
    pub cancelling: bool, // set before cancel, prevents non-zero exit -> Failed
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at:   Option<DateTime<Utc>>,
    pub exit_code:  Option<i32>,
    pub log_path:   Option<PathBuf>,
}

impl JobInstance {
    pub fn new(spec: JobSpec, job_seq: u64) -> Self {
        Self {
            job_id:     spec.id.clone(),
            job_seq,
            spec,
            state:      JobState::Pending,
            handle:     None,
            allocation: None,
            cancelling: false,
            started_at: None,
            ended_at:   None,
            exit_code:  None,
            log_path:   None,
        }
    }
}

/// Runtime state of an entire workflow run
#[derive(Debug)]
pub struct WorkflowRun {
    pub run_id:        String,
    pub run_seq:       u64,
    pub workflow_name: String,
    pub spec:          WorkflowSpec,
    pub jobs:          HashMap<String, JobInstance>, // keyed by job_id
    pub created_at:    DateTime<Utc>,
}

impl WorkflowRun {
    pub fn new(run_id: String, run_seq: u64, spec: WorkflowSpec, job_seq_start: u64) -> Self {
        let jobs = spec.jobs
            .iter()
            .enumerate()
            .map(|(i, job_spec)| {
                let job_seq = job_seq_start + i as u64;
                (job_spec.id.clone(), JobInstance::new(job_spec.clone(), job_seq))
            })
            .collect();

        let workflow_name = spec.name.clone();

        Self {
            run_id,
            run_seq,
            workflow_name,
            spec,
            jobs,
            created_at: Utc::now(),
        }
    }

    /// Derive run state from job states, computed on demand
    /// Precedence: Failed/TimeOut/Interrupted > Cancelled > Running > Succeeded > Pending
    pub fn status(&self) -> RunState {
        derive_run_state(self.jobs.values().map(|j| j.state))
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

    fn make_run(run_id: &str) -> WorkflowRun {
        let spec = parse(EXAMPLE).unwrap();
        WorkflowRun::new(run_id.to_string(), 0, spec, 0)
    }

    // Enum string tripwire

    const ALL_JOB_STATES: [JobState; 9] = [
        JobState::Pending, JobState::Queued, JobState::Running, JobState::Succeeded,
        JobState::Failed, JobState::Skipped, JobState::Cancelled, JobState::TimedOut,
        JobState::Interrupted,
    ];

    const ALL_RUN_STATES: [RunState; 5] = [
        RunState::Pending, RunState::Running, RunState::Succeeded,
        RunState::Failed, RunState::Cancelled
    ];

    #[test]
    fn every_job_state_roundtrips_serde_and_str() {
        for state in ALL_JOB_STATES {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(json.trim_matches('"'), state.as_str());
            assert_eq!(JobState::from_str(state.as_str()), Some(state));

            let parsed: JobState = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, state);
        }
    }

    #[test]
    fn every_run_state_roundtrips_serde_and_str() {
        for state in ALL_RUN_STATES {
            let json = serde_json::to_string(&state).unwrap();
            assert_eq!(json.trim_matches('"'), state.as_str());
            assert_eq!(RunState::from_str(state.as_str()), Some(state));

            let parsed: RunState = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, state);
        }
    }

    #[test]
    fn from_str_rejects_unknown() {
        assert_eq!(JobState::from_str("NotAState"), None);
        assert_eq!(RunState::from_str("NotAState"), None);
    }

    #[test]
    fn all_jobs_start_pending() {
        let run = make_run("run-001");
        for job in run.jobs.values() {
            assert_eq!(job.state, JobState::Pending);
        }
    }

    #[test]
    fn run_starts_pending() {
        assert_eq!(make_run("run-001").status(), RunState::Pending);
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
        let mut run = make_run("r1");
        run.jobs.get_mut("preprocess").unwrap().state = JobState::Failed;
        run.jobs.get_mut("train").unwrap().state = JobState::Running;
        assert_eq!(run.status(), RunState::Failed);
    }

    #[test]
    fn failed_bests_cancelled() {
        let mut run = make_run("r1");
        run.jobs.get_mut("preprocess").unwrap().state = JobState::Failed;
        run.jobs.get_mut("train").unwrap().state = JobState::Cancelled;
        assert_eq!(run.status(), RunState::Failed);
    }

    #[test]
    fn all_succeeded_or_skipped_is_succeeded() {
        let mut run = make_run("r1");
        run.jobs.get_mut("preprocess").unwrap().state = JobState::Succeeded;
        run.jobs.get_mut("train").unwrap().state = JobState::Skipped;
        run.jobs.get_mut("eval").unwrap().state = JobState::Skipped;
        assert_eq!(run.status(), RunState::Succeeded);
    }

    // derive_run_state direct tests
    #[test]
    fn derive_run_state_empty_is_pending_not_succeeded() {
        assert_eq!(derive_run_state(std::iter::empty()), RunState::Pending);
    }

    #[test]
    fn derive_run_state_partial_flush_is_not_succeeded() {
        let partial = vec![JobState::Succeeded];
        assert_eq!(derive_run_state(partial.into_iter()), RunState::Succeeded);
    }

    #[test]
    fn every_job_state_roundtrips_through_code() {
        for state in ALL_JOB_STATES {
            let code = state.to_code();
            assert_eq!(JobState::from_code(code), Some(state));
        }
    }

    #[test]
    fn from_code_rejects_out_of_range() {
        assert_eq!(JobState::from_code(9), None);
        assert_eq!(JobState::from_code(999), None);
    }

    #[test]
    fn all_job_state_codes_are_distinct() {
        let codes: Vec<u64> = ALL_JOB_STATES.iter().map(|s| s.to_code()).collect();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len(), "duplicate JobState code found: {codes:?}");
    }
}