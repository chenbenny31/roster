use crate::workflow::model::JobState;

/// Events emitted by the scheduler on every job state transition
/// consumed by roster top via broadcast channel
#[derive(Debug, Clone)]
pub enum JobEvent {
    StateChanged {
        run_id: String,
        job_id: String,
        new_state: JobState,
        emitted_at: u64, // CLOCK_MONOTONIC_RAW nanoseconds
    }
}

/// Read CLOCK_MONOTONIC_RAW in nanoseconds
pub fn monotonic_raw_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts);
    }
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}