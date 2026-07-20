//! Lock-free SPMC broadcast ring buffer
//! fan scheduler `JobEvent`s out to dashboard subscribers (`roster top`)
//!
//! Protocol (per-slot seqlock, position-encoded)
//! each slot carries an `AtomicUsize`, `version that encodes position:
//!   * `2*p + 1` (odd) - producer is mid-write at position `p`
//!   * `2*p + 2` (even) - slot hold the completed write from position `p`
//! slot `i` is initialized to `2*i`, first writing at position `i`
//!
//! Consumer at read position `p` wants version `2*p + 2`:
//!   * `version < want`: not yet written or writing right now
//!   * `version == want`: required event, stable as of load
//!   * `version > want`: have lost event, producer lapped consumer, need resync
//!
//! Guarantees and non-guarantees:
//!   * Lock-free, wait-free on producer, obstruction-free read
//!   * Lossy: a consumer a full lap behind drops events cleanly, never torn
//!   * Single producer only: `SpmcSender` is `!Sync` to enforce this
//!
//! Payload storage: per-word atomics, not `UnsafeCell`
//! Payloads are stored as `WordSafe::WORDS` plain `AtomicUsize`s per slot
//! this is standard fix for racing on the same `UnsafeCell` is UB even with discard
//!
//! Overflow bound:
//! Positions are `usize` and the version stores `2*p + {1,2}`
//! scheme is correct until `2*p` overflows `usize`: 9.2e18 events on 64-bit, unreachable
//!
//! Loom model checking
//! LOOM_MAX_PREEMPTIONS=3 cargo test --release --features loom --lib broadcast::loom_tests

#[cfg(feature = "loom")]
use loom::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(feature = "loom")]
use loom::sync::Arc;

#[cfg(not(feature = "loom"))]
use std::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(not(feature = "loom"))]
use std::sync::Arc;

// Marker trait: `T` can be soundly read/written word-by-word via `Relaxed` `AtomicUsize`
// fix the loom violation on `UnsafeCell`: a race on the same cell is UB even if discard
// safety guarantee:
//   1. `size_of::<Self>()` is a whole multiple of `size_of::<usize>()`
//   2. `align_of::<Self>() <= align_of::<usize>()`, storage is guaranteed `usize`-aligned
//   3. No padding
//   4. Any bit pattern from re-assembling two diff complete writes is a valid `Self`
pub unsafe trait WordSafe: Copy {
    const WORDS: usize; // word count, silently round down for non-whole-word size
}

const MAX_STAGING_WORDS: usize = 16;

#[repr(align(64))]
struct PaddedVersion {
    value: AtomicUsize,
}

#[repr(align(64))]
struct PaddedCursor {
    value: AtomicUsize,
}

struct RingBuffer<T: Copy + WordSafe> {
    mask:     usize,
    versions: Box<[PaddedVersion]>,
    words:    Box<[AtomicUsize]>, // capacity * stride
    stride:   usize, // cache-line-rounded words per slot
    producer: PaddedCursor,
    _marker:  std::marker::PhantomData<T>,
}

// Safety: payload access is mediated by the per-slot seqlock, racing Relaxed atomic word ops
unsafe impl<T: Copy + WordSafe + Send> Send for RingBuffer<T> {}
unsafe impl<T: Copy + WordSafe + Send> Sync for RingBuffer<T> {}

/// The single producer, concurrent `send` is a compile error
pub struct SpmcSender<T: Copy + WordSafe> {
    buffer:     Arc<RingBuffer<T>>,
    _not_sync:  std::marker::PhantomData<std::cell::Cell<()>>,
}

impl<T: Copy + WordSafe + Send> SpmcSender<T> {
    /// Publish one event, wait-free, the oldest slot if overwritten (lossy)
    pub fn send(&self, value: T) {
        let buffer = &*self.buffer;
        let pos = buffer.producer.value.fetch_add(1, Ordering::Relaxed);
        let slot = pos & buffer.mask;
        let word_base = slot * buffer.stride;

        buffer.versions[slot].value.store(2 * pos + 1, Ordering::Relaxed);
        fence(Ordering::Release); // prevent payload store from being reordered

        // Safety: u8-granularity copy needs no alignment on either side
        let mut staged = [0usize; MAX_STAGING_WORDS];
        unsafe {
            std::ptr::copy_nonoverlapping(
                &value as *const T as *const u8,
                staged.as_mut_ptr() as *mut u8,
                std::mem::size_of::<T>(),
            )
        }

        for i in 0..T::WORDS {
            buffer.words[word_base + i].store(staged[i], Ordering::Relaxed)
        }
        buffer.versions[slot].value.store(2 * pos + 2, Ordering::Release);
    }

    /// Current producer position
    pub fn producer_pos(&self) -> usize {
        self.buffer.producer.value.load(Ordering::Relaxed)
    }
}

/// Sync, Clone factory for receivers, lives in `DaemonState`, call `subscribe()` per consumer
pub struct SpmcSubscriber<T: Copy + WordSafe> {
    buffer: Arc<RingBuffer<T>>,
}

impl<T: Copy + WordSafe> Clone for SpmcSubscriber<T> {
    fn clone(&self) -> Self {
        Self { buffer: Arc::clone(&self.buffer) }
    }
}

impl<T: Copy + WordSafe + Send> SpmcSubscriber<T> {
    /// New receiver pos at current producer cursor
    pub fn subscribe(&self) -> SpmcReceiver<T> {
        let pos = self.buffer.producer.value.load(Ordering::Relaxed);
        SpmcReceiver { buffer: Arc::clone(&self.buffer), pos, laps: 0 }
    }
}

/// A single consumer with its own read cursor
pub struct SpmcReceiver<T: Copy + WordSafe> {
    buffer: Arc<RingBuffer<T>>,
    pos:    usize,
    laps:   u64,
}

impl<T: Copy + WordSafe + Send> SpmcReceiver<T> {
    /// Return the next event if ready, else `None`, non-blocking
    pub fn try_recv(&mut self) -> Option<T> {
        let buffer = &*self.buffer;
        let slot = self.pos & buffer.mask;
        let word_base = slot * buffer.stride;
        let want = 2 * self.pos + 2;

        let v1 = buffer.versions[slot].value.load(Ordering::Acquire);
        if v1 < want {
            return None;
        }
        if v1 > want {
            self.catch_up(); // producer lapped, loss event, re-sync
            return None;
        }

        let mut staged = [0usize; MAX_STAGING_WORDS];
        for i in 0..T::WORDS {
            staged[i] = buffer.words[word_base + i].load(Ordering::Relaxed);
        }

        fence(Ordering::Acquire); // prevent v2 load from being reordered
        let v2 = buffer.versions[slot].value.load(Ordering::Relaxed);
        if v1 != v2 {
            return None; // overwritten mid-read, discard
        }

        let data: T = unsafe { std::ptr::read(staged.as_ptr() as *const T) };

        self.pos += 1;
        Some(data)
    }

    /// Re-sync after being lapped, jump half a buffer back from producer
    fn catch_up(&mut self) {
        self.laps += 1;
        let producer_pos = self.buffer.producer.value.load(Ordering::Relaxed);
        let capacity = self.buffer.mask + 1;
        self.pos = producer_pos.saturating_sub(capacity / 2);
    }

    pub fn lap_count(&self) -> u64 {
        self.laps
    }
}

/// Build an SPMC broadcast channel, `capacity` round up to next power of 2 (min 2)
pub fn channel<T: Copy + WordSafe>(capacity: usize) -> (SpmcSender<T>, SpmcSubscriber<T>) {
    assert!(T::WORDS <= MAX_STAGING_WORDS);

    let capacity = capacity.next_power_of_two().max(2);

    const CACHE_LINE_BYTES: usize = 64;
    let slot_bytes = T::WORDS * std::mem::size_of::<usize>();
    let stride_bytes = slot_bytes.next_multiple_of(CACHE_LINE_BYTES);
    let stride = stride_bytes / std::mem::size_of::<usize>();

    let versions: Box<[PaddedVersion]> = (0..capacity)
        .map(|i| PaddedVersion { value: AtomicUsize::new(2 * i) })
        .collect();

    let words: Box<[AtomicUsize]> = (0..capacity * stride)
        .map(|_| AtomicUsize::new(0))
        .collect();

    let buffer = Arc::new(RingBuffer {
        mask:     capacity - 1,
        versions,
        words,
        stride,
        producer: PaddedCursor { value: AtomicUsize::new(0) },
        _marker:  std::marker::PhantomData,
    });

    let sender = SpmcSender {
        buffer:     Arc::clone(&buffer),
        _not_sync:  std::marker::PhantomData,
    };
    let subscriber = SpmcSubscriber { buffer };

    (sender, subscriber)
}

// normal tests (not run under loom)
#[cfg(all(test, not(feature = "loom")))]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct TestWord(u64);

    unsafe impl WordSafe for TestWord {
        const WORDS: usize = 1;
    }

    #[test]
    fn empty_returns_none() {
        let (_tx, sub) = channel::<TestWord>(4);
        let mut rx = sub.subscribe();
        assert_eq!(rx.try_recv(), None);
    }

    #[test]
    fn fifo_within_capacity() {
        let (tx, sub) = channel::<TestWord>(8);
        let mut rx = sub.subscribe();
        for i in 0..5_u64 {
            tx.send(TestWord(i));
        }
        for i in 0..5_u64 {
            assert_eq!(rx.try_recv(), Some(TestWord(i)));
        }
        assert_eq!(rx.try_recv(), None);
    }

    #[test]
    fn no_replay_for_late_subscriber() {
        let (tx, sub) = channel::<TestWord>(8);
        tx.send(TestWord(1));
        tx.send(TestWord(2));
        let mut rx = sub.subscribe(); // subscribe at pos 2
        assert_eq!(rx.try_recv(), None);
        tx.send(TestWord(3));
        assert_eq!(rx.try_recv(), Some(TestWord(3)));
    }

    #[test]
    fn lapped_consumer_resyncs() {
        let (tx, sub) = channel::<TestWord>(4);
        let mut rx = sub.subscribe();
        for i in 0..10_u64 {
            tx.send(TestWord(i));
        }
        assert_eq!(rx.try_recv(), None);

        let mut got = Vec::new();
        while let Some(v) = rx.try_recv() {
            got.push(v.0);
        }
        for w in got.windows(2) {
            assert!(w[0] < w[1]);
        }
        assert!(got.iter().all(|&v| v >= 8), "got: {got:?}");
    }

    #[test]
    fn subscribe_past_last_write_is_none() {
        let (tx, sub) = channel::<TestWord>(4);
        tx.send(TestWord(10));
        tx.send(TestWord(20));
        tx.send(TestWord(30));

        let mut rx = sub.subscribe(); // at pos = 3, never written
        assert_eq!(rx.try_recv(), None);
    }

    #[test]
    fn stride_is_cache_line_multiple() {
        let (tx, _sub) = channel::<TestWord>(4);
        let stride_bytes = tx.buffer.stride * std::mem::size_of::<usize>();
        assert_eq!(stride_bytes % 64, 0);
    }

    #[test]
    fn concurrent_order_preserved() {
        let (tx, sub) = channel::<TestWord>(1024);
        let mut rx = sub.subscribe();
        let n = 200_000_u64;
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_p = Arc::clone(&done);

        let producer = std::thread::spawn(move || {
            for i in 0..n { tx.send(TestWord(i)); }
            done_p.store(true, Ordering::Release);
        });

        let consumer = std::thread::spawn(move || {
            let mut seen = Vec::new();
            loop {
                match rx.try_recv() {
                    Some(v) => seen.push(v.0),
                    None => {
                        if done.load(Ordering::Acquire) {
                            while let Some(v) = rx.try_recv() { seen.push(v.0); }
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
            }
            seen
        });

        producer.join().unwrap();
        let seen = consumer.join().unwrap();

        for w in seen.windows(2) {
            assert!(w[0] < w[1], "order violated: {} then {}", w[0], w[1]);
        }
        assert!(seen.iter().all(|&v| v < n));
    }
}

// loom model-checked tests (ONLY run under --cfg loom)
// Invocation:
//   LOOM_MAX_PREEMPTIONS=3 cargo test --release --features loom --lib broadcast::loom_tests
#[cfg(all(test, feature = "loom"))]
mod loom_tests {
    use super::*;
    use crate::event::JobEvent;
    use loom::thread;

    const SENTINEL_A: u64 = 0x1111_1111_1111_1111;
    const SENTINEL_B: u64 = 0x2222_2222_2222_2222;
    const SENTINEL_C: u64 = 0x3333_3333_3333_3333;

    fn matched(sentinel: u64) -> JobEvent {
        JobEvent {
            run_seq:    sentinel,
            job_seq:    sentinel,
            state_code: sentinel,
            emitted_at: sentinel,
        }
    }

    #[test]
    fn seqlock_never_returns_torn_or_stale_data() {
        loom::model(|| {
            let (tx, sub) = channel::<JobEvent>(2);
            let mut rx = sub.subscribe();

            let producer = thread::spawn(move || {
                tx.send(matched(SENTINEL_A));
                tx.send(matched(SENTINEL_B));
                tx.send(matched(SENTINEL_C));
            });

            let mut seen = Vec::new();
            for _ in 0..3 {
                if let Some(v) = rx.try_recv() {
                    seen.push(v);
                }
            }

            producer.join().unwrap();

            let valid = [SENTINEL_A, SENTINEL_B, SENTINEL_C];
            for event in &seen {
                assert_eq!(event.run_seq, event.job_seq, "torn: {event:x?}");
                assert_eq!(event.job_seq, event.state_code, "torn: {event:x?}");
                assert_eq!(event.state_code, event.emitted_at, "torn: {event:x?}");
                assert!(valid.contains(&event.run_seq), "invalid: {event:x?}");
            }
            for pair in seen.windows(2) {
                assert!(pair[0].run_seq < pair[1].run_seq, "order violated");
            }
        });
    }
}