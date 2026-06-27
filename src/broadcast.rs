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

use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{fence, AtomicUsize, Ordering};
use std::cell::Cell;

/// One ring slot, cache-line aligned to prevent false sharing between adjacent slots
#[repr(align(64))]
struct Slot<T: Copy> {
    version: AtomicUsize, // position-encoded seqlock version
    data: UnsafeCell<MaybeUninit<T>>, // payload written by producer between version bump
}

/// Producer cursor on its own cache line to prevent false sharing with slots
#[repr(align(64))]
struct PaddedCursor {
    value: AtomicUsize,
}

struct RingBuffer<T: Copy> {
    mask: usize, // index = pos & mask
    slots: Box<[Slot<T>]>, // read-only after construction
    producer: PaddedCursor, // single-producer cursor, own cache line
}

// All access to non-atomic data cells is mediated by per-slot version seqlock
unsafe impl<T: Copy + Send> Send for RingBuffer<T> {}
unsafe impl<T: Copy + Send> Sync for RingBuffer<T> {}

/// Single producer, moved into scheduler task
pub struct SpmcSender<T: Copy> {
    buffer: Arc<RingBuffer<T>>,
    _not_sync: PhantomData<Cell<()>>,
}

impl<T: Copy + Send> SpmcSender<T> {
    /// Publish one event, wait-free, overwritten the oldest on a full ring, single producer
    pub fn send(&self, value: T) {
        let buffer = &*self.buffer;

        // claim position, relaxed: cursor carries not data dependency
        let pos = buffer.producer.value.fetch_add(1, Ordering::Relaxed);
        let slot = &buffer.slots[pos & buffer.mask];

        // mark "writing" (odd), relaxed
        slot.version.store(2 * pos + 1, Ordering::Relaxed);

        // release fence: prevent payload store being re-ordered
        fence(Ordering::Release);

        // single produce writes, concurrent readers
        unsafe {
            (*slot.data.get()).write(value);
        }

        // Publish (even), release: consumer loads with acquire
        slot.version.store(2 * pos + 2, Ordering::Release);
    }

    /// Current producer position
    pub fn producer_pos(&self) -> usize {
        self.buffer.producer.value.load(Ordering::Relaxed)
    }
}

/// Sync, clone factory for receivers in `DaemonState`
pub struct SpmcSubscriber<T: Copy> {
    buffer: Arc<RingBuffer<T>>,
}

impl<T: Copy> Clone for SpmcSubscriber<T> {
    fn clone(&self) -> Self {
        Self { buffer: Arc::clone(&self.buffer) }
    }
}

impl<T: Copy + Send> SpmcSubscriber<T> {
    /// New receiver positioned at the current producer cursor, see only events published after
    pub fn subscribe(&self) -> SpmcReceiver<T> {
        let pos = self.buffer.producer.value.load(Ordering::Relaxed);
        SpmcReceiver { buffer: Arc::clone(&self.buffer), pos }
    }
}

/// Single consumer with own read cursor
pub struct SpmcReceiver<T: Copy> {
    buffer: Arc<RingBuffer<T>>,
    pos: usize,
}

impl<T: Copy + Send> SpmcReceiver<T> {
    /// Return the next event if read, else `None`, no blocking
    pub fn try_recv(&mut self) -> Option<T> {
        let buffer = &*self.buffer;
        let slot = &buffer.slots[self.pos & buffer.mask];
        let want = 2 * self.pos + 2;

        // acquire: sync with producer's closing relase store
        let v1 = slot.version.load(Ordering::Acquire);
        if v1 < want {
            return None;
        }
        if v1 > want {
            self.catch_up();
            return None;
        }

        let data = unsafe { (*slot.data.get()).assume_init() };

        // acquire fence: prevents v2 load from being re-ordered before the payload copy
        fence(Ordering::Acquire);
        let v2 = slot.version.load(Ordering::Relaxed);

        if v1 != v2 { // producer overwrote the slot during copy, discard
            return None;
        }

        self.pos += 1;
        Some(data)
    }

    /// Resync after being lapped, jump half a buffer back from producer
    fn catch_up(&mut self) {
        let producer_pos = self.buffer.producer.value.load(Ordering::Relaxed);
        let capacity = self.buffer.mask + 1;
        self.pos = producer_pos.saturating_sub(capacity / 2);
    }
}

/// Build a SPMC broadcast channel, `capacity` is rounded up to next power of 2
pub fn channel<T: Copy>(capacity: usize) -> (SpmcSender<T>, SpmcSubscriber<T>) {
    let capacity = capacity.next_power_of_two().max(2);

    let slots: Box<[Slot<T>]> = (0..capacity)
        .map(|i| Slot {
            version: AtomicUsize::new(2 * i),
            data: UnsafeCell::new(MaybeUninit::uninit()),
        })
        .collect();

    let buffer = Arc::new(RingBuffer {
        mask: capacity - 1,
        slots,
        producer: PaddedCursor { value: AtomicUsize::new(0) },
    });

    let sender = SpmcSender {
        buffer: Arc::clone(&buffer),
        _not_sync: PhantomData
    };

    let subscriber = SpmcSubscriber { buffer };

    (sender, subscriber)
}

// tests
#[cfg(test)]
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
        let mut rx = sub.subscribe();
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
        // first read sees v1 > want, trigger catch_up to 10 - 4/2 = 8
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