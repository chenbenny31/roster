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
//!   * `version < want`: not yet written
//!   * `version == want`: required event, stable as of load
//!   * `version > want`: have lost event, producer lapped consumer, need resync
//!
//! Guarantees and non-guarantees:
//!   * Lock-free, wait-free on producer, obstruction-free read
//!   * Lossy by default: a consumer slower than producer by a full ring drops event cleanly (skip)
//!   * Single producer only: `SpmcSender` is `!Sync` to enforce this
//!
//! Overflow bound:
//! Positions are `usize` and the version stores `2*p + {1,2}`
//! scheme is correct until `2*p` overflows `usize`: 9.2e18 events on 64-bit, unreachable

use std::mem::MaybeUninit;

#[cfg(feature = "loom")]
use loom::cell::UnsafeCell;
#[cfg(feature = "loom")]
use loom::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(feature = "loom")]
use loom::sync::Arc;

#[cfg(not(feature = "loom"))]
use std::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(not(feature = "loom"))]
use std::sync::Arc;

#[cfg(not(feature = "loom"))]
mod cell {
    use std::cell::UnsafeCell as StdUnsafeCell;

    pub(crate) struct UnsafeCell<T>(StdUnsafeCell<T>);

    impl<T> UnsafeCell<T> {
        pub(crate) fn new(data: T) -> Self {
            UnsafeCell(StdUnsafeCell::new(data))
        }

        pub(crate) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
            f(self.0.get())
        }

        pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            f(self.0.get())
        }
    }
}

#[cfg(not(feature = "loom"))]
use cell::UnsafeCell;

/// One ring slot, cache-line aligned to prevent false sharing between adjacent
#[repr(align(64))]
struct Slot<T: Copy> {
    version: AtomicUsize,
    data: UnsafeCell<MaybeUninit<T>>,
}

/// Producer cursor on its own cache line to prevent false sharing between slots
#[repr(align(64))]
struct PaddedCursor {
    value: AtomicUsize,
}

struct RingBuffer<T: Copy> {
    mask:     usize, // capacity - 1; index = pos & mask
    slots:    Box<[Slot<T>]>, // read-only after construction
    producer: PaddedCursor, // single-producer cursor; own cache line
}

// all access to non-atomic data cells is mediated by the per-slot version seqlock
unsafe impl<T: Copy + Send> Send for RingBuffer<T> {}
unsafe impl<T: Copy + Send> Sync for RingBuffer<T> {}

/// The single producer, moved into scheduler task
pub struct SpmcSender<T: Copy> {
    buffer:     Arc<RingBuffer<T>>,
    _not_sync:  std::marker::PhantomData<std::cell::Cell<()>>,
}

impl<T: Copy + Send> SpmcSender<T> {
    /// Publish one event, wait-free, the oldest slot if overwritten (lossy)
    pub fn send(&self, value: T) {
        let buffer = &*self.buffer;

        let pos = buffer.producer.value.fetch_add(1, Ordering::Relaxed);
        let slot = &buffer.slots[pos & buffer.mask];

        slot.version.store(2 * pos + 1, Ordering::Relaxed); // marking "writing"

        fence(Ordering::Release); // prevent payload store from being reordered

        slot.data.with_mut(|ptr| unsafe { // single producer, no concurrent writers
            (*ptr).write(value);
        });

        slot.version.store(2 * pos + 2, Ordering::Release); // publish in even
    }

    /// Current producer position
    pub fn producer_pos(&self) -> usize {
        self.buffer.producer.value.load(Ordering::Relaxed)
    }
}

/// Sync, Clone factory for receivers, lives in `DaemonState`, call `subscribe()`
pub struct SpmcSubscriber<T: Copy> {
    buffer: Arc<RingBuffer<T>>,
}

impl<T: Copy> Clone for SpmcSubscriber<T> {
    fn clone(&self) -> Self {
        Self { buffer: Arc::clone(&self.buffer) }
    }
}

impl<T: Copy + Send> SpmcSubscriber<T> {
    /// New receiver pos at current producer cursor
    pub fn subscribe(&self) -> SpmcReceiver<T> {
        let pos = self.buffer.producer.value.load(Ordering::Relaxed);
        SpmcReceiver { buffer: Arc::clone(&self.buffer), pos }
    }
}

/// A single consumer with its own read cursor
pub struct SpmcReceiver<T: Copy> {
    buffer: Arc<RingBuffer<T>>,
    pos:    usize,
}

impl<T: Copy> SpmcReceiver<T> {
    /// Return the next event if ready, else `None`, non-blocking
    pub fn try_recv(&mut self) -> Option<T> {
        let buffer = &*self.buffer;
        let slot = &buffer.slots[self.pos & buffer.mask];

        let want = 2 * self.pos + 2; // completed at even version

        let v1 = slot.version.load(Ordering::Acquire); // sync with producer's closing Release store
        if v1 < want {
            return None;
        }
        if v1 > want {
            self.catch_up(); // producer lapped, loss event, re-sync
            return None;
        }

        let data = slot.data.with(|ptr| unsafe { (*ptr).assume_init() }); // v1 == want

        fence(Ordering::Acquire); // prevent v2 load from being reordered
        let v2 = slot.version.load(Ordering::Relaxed);
        if v1 != v2 {
            return None;
        }

        self.pos += 1;
        Some(data)
    }

    /// Re-sync after being lapped, jump half a buffer back from producer
    fn catch_up(&mut self) {
        let producer_pos = self.buffer.producer.value.load(Ordering::Relaxed);
        let capacity = self.buffer.mask + 1;
        self.pos = producer_pos.saturating_sub(capacity / 2);
    }
}

/// Build an SPMC broadcast channel, `capacity` round up to next power of 2
pub fn channel<T: Copy>(capacity: usize) -> (SpmcSender<T>, SpmcSubscriber<T>) {
    let capacity = capacity.next_power_of_two().max(2);

    let slots: Box<[Slot<T>]> = (0..capacity)
        .map(|i| Slot {
            version: AtomicUsize::new(2 * i),
            data:    UnsafeCell::new(MaybeUninit::uninit()),
        })
        .collect();

    let buffer = Arc::new(RingBuffer {
        mask:     capacity - 1,
        slots,
        producer: PaddedCursor { value: AtomicUsize::new(0) },
    });

    let sender = SpmcSender {
        buffer:     Arc::clone(&buffer),
        _not_sync:  std::marker::PhantomData,
    };
    let subscriber = SpmcSubscriber { buffer };

    (sender, subscriber)
}

// normal tests

#[cfg(all(test, not(feature = "loom")))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::thread;

    #[test]
    fn empty_returns_none() {
        let (_tx, sub) = channel::<u64>(4);
        let mut rx = sub.subscribe();
        assert_eq!(rx.try_recv(), None);
    }

    #[test]
    fn fifo_within_capacity() {
        let (tx, sub) = channel::<u64>(8);
        let mut rx = sub.subscribe();
        for i in 0..5_u64 {
            tx.send(i);
        }
        for i in 0..5_u64 {
            assert_eq!(rx.try_recv(), Some(i));
        }
        assert_eq!(rx.try_recv(), None);
    }

    #[test]
    fn no_replay_for_late_subscriber() {
        let (tx, sub) = channel::<u64>(8);
        tx.send(1);
        tx.send(2);
        let mut rx = sub.subscribe(); // subscribe at pos 2
        assert_eq!(rx.try_recv(), None);
        tx.send(3);
        assert_eq!(rx.try_recv(), Some(3));
    }

    #[test]
    fn lapped_consumer_resyncs() {
        let (tx, sub) = channel::<u64>(4);
        let mut rx = sub.subscribe();
        for i in 0..10_u64 {
            tx.send(i);
        }
        assert_eq!(rx.try_recv(), None);

        let mut got = Vec::new();
        while let Some(v) = rx.try_recv() {
            got.push(v);
        }
        for w in got.windows(2) {
            assert!(w[0] < w[1]);
        }
        assert!(got.iter().all(|&v| v >= 8), "got: {got:?}");
    }

    #[test]
    fn subscribe_pas_last_write_is_none() {
        let (tx, sub) = channel::<u64>(4);
        tx.send(10);
        tx.send(20);
        tx.send(30);

        let mut tx = sub.subscribe(); // at pos = 3, never written
        assert_eq!(tx.try_recv(), None);
    }

    #[test]
    fn concurrent_order_preserved() {
        let (tx, sub) = channel::<u64>(1024);
        let mut rx = sub.subscribe();
        let n = 200_000_u64;
        let done = Arc::new(AtomicBool::new(false));
        let done_p = Arc::clone(&done);

        let producer = thread::spawn(move || {
            for i in 0..n { tx.send(i); }
            done_p.store(true, Ordering::Release);
        });

        let consumer = thread::spawn(move || {
            let mut seen = Vec::new();
            loop {
                match rx.try_recv() {
                    Some(v) => seen.push(v),
                    None => {
                        if done.load(Ordering::Acquire) {
                            while let Some(v) = rx.try_recv() { seen.push(v); }
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
    use loom::thread;

    const SENTINEL_A: u64 = 0x1111_1111_1111_1111;
    const SENTINEL_B: u64 = 0x2222_2222_2222_2222;
    const SENTINEL_C: u64 = 0x3333_3333_3333_3333;

    #[test]
    fn seqlock_never_returns_torn_or_stale_data() {
        loom::model(|| {
            let (tx, sub) = channel::<u64>(2);
            let mut rx = sub.subscribe();

            let producer = thread::spawn(move || {
                tx.send(SENTINEL_A);
                tx.send(SENTINEL_B);
                tx.send(SENTINEL_C);
            });

            let mut seen = Vec::new();
            for _ in 0..3 {
                if let Some(v) = rx.try_recv() {
                    seen.push(v);
                }
            }

            producer.join().unwrap();

            let valid = [SENTINEL_A, SENTINEL_B, SENTINEL_C];
            assert!(
                seen.iter().all(|v| valid.contains(v)),
                "torn or invalid value present: {seen:x?}"
            );
            for pair in seen.windows(2) {
                assert!(pair[0] < pair[1], "order violated: {:x} then {:x}", pair[0], pair[1]);
            }
        });
    }
}