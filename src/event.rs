use crate::workflow::model::JobState;
use crate::broadcast::WordSafe;

/// Events emitted by the scheduler on every job state transition
/// make word-wise Relaxed-atomic read/write
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobEvent {
    pub run_seq:    u64, // mono run counter assigned at submit
    pub job_seq:    u64, // mono job counter assigned at submit
    pub state_code: u64, // JobState::to_code(), decode via .state()
    pub emitted_at: u64, // CLOCK_MONOTONIC_RAW ns, publish-to-render latency source
}

// safety: 4 u64 fields, load-bearing layout, make word-wise Relaxed-atomic read/write
unsafe impl WordSafe for JobEvent {
    const WORDS: usize = 4;
}

impl JobEvent {
    /// Construct an event, encoding `new_state` as its numeric code
    pub fn new(run_seq: u64, job_seq: u64, new_state: JobState, emitted_at: u64) -> Self {
        Self {
            run_seq,
            job_seq,
            state_code: new_state.to_code(),
            emitted_at,
        }
    }

    /// Decode the job state this event carries
    pub fn state(&self) -> JobState {
        JobState::from_code(self.state_code).expect(
            "JobEvent.state_code was not a valid JobState code - \
             this indicates a bug in JobState::to_code/from_code, not corrupted data",
        )
    }
}

// Names snapshot fetched once on TUI subscribe
#[derive(Debug, Clone)]
pub struct Names {
    pub runs: std::collections::HashMap<u64, String>, // run_seq -> run_id (UUID)
    pub jobs: std::collections::HashMap<u64, String>, // job_seq -> job_id
}

/// Read CLOCK_MONOTONIC_RAW in nanoseconds
pub fn monotonic_raw_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts);
    }
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_event_roundtrips_through_new_and_state() {
        let event = JobEvent::new(7, 42, JobState::Running, 123_456_789);
        assert_eq!(event.run_seq, 7);
        assert_eq!(event.job_seq, 42);
        assert_eq!(event.emitted_at, 123_456_789);
        assert_eq!(event.state(), JobState::Running);
    }

    #[test]
    fn job_event_size_matches_declared_words() {
        // ties the size check to WordSafe::WORDS itself
        assert_eq!(
            std::mem::size_of::<JobEvent>(),
            JobEvent::WORDS * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn job_event_has_no_padding() {
        // padding check
        let event = JobEvent::new(0, 0, JobState::Pending, 0);
        let field_bytes = std::mem::size_of_val(&event.run_seq)
            + std::mem::size_of_val(&event.job_seq)
            + std::mem::size_of_val(&event.state_code)
            + std::mem::size_of_val(&event.emitted_at);
        assert_eq!(field_bytes, std::mem::size_of::<JobEvent>());
    }

    #[test]
    fn job_event_alignment_satisfies_wordsafe() {
        assert!(std::mem::align_of::<JobEvent>() <= std::mem::align_of::<usize>());
    }

    #[test]
    #[should_panic(expected = "not a valid JobState code")]
    fn state_panics_on_invalid_code() {
        let event = JobEvent { run_seq: 0, job_seq: 0, state_code: 999, emitted_at: 0 };
        let _ = event.state();
    }
}