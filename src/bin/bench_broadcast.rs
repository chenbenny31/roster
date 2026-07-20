//! Broadcast ring buffer latency benchmark
//!
//! Measures publish-to-observe latency of `roster::broadcast`
//! 100K events/sec is designed stress load
//! Usage: `cargo run --release --bin bench_broadcast -- [--pin]`

use clap::Parser;
use hdrhistogram::Histogram;
use std::sync::{Arc, Barrier};
use std::thread;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use roster::broadcast;
use roster::event::{monotonic_raw_ns, JobEvent};

const CHANNEL_CAPACITY: usize = 4096;
const WARMUP_CAP: u64 = 10_000;
const MEASURED_CAP: u64 = 1_000_000;
const TARGET_SECONDS: u64 = 15; // wall-clock budget per config

#[derive(Parser)]
struct Args {
    // Pin producer/consumer threads to distinct logical cores
    #[arg(long)]
    pin: bool,
}

struct RunConfig {
    subscriber_count: usize,
    rate_per_sec:     u64,
    label:            String,
}

/// Scale (warmup, measured) to the offered rate to make all config capped at fixed max
fn samples_for_rate(rate_per_sec: u64) -> (u64, u64) {
    let measured = (rate_per_sec * TARGET_SECONDS).min(MEASURED_CAP).max(1_000);
    let warmup = rate_per_sec.min(WARMUP_CAP).max(100);
    (warmup, measured)
}

fn main() {
    let args = Args::parse();

    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let logical_cores = core_ids.len().max(1);

    println!("roster broadcast latency benchmark");
    println!("logical cores detected: {logical_cores}");
    println!("core pinning: {}", if args.pin { "ON" } else { "off (use --pin to enable)" });
    println!("100K/s runs are a deliberate stress load, ~3-5 orders of magnitude above \
              realistic scheduler rates (~1-100/s); production sits far below this.");
    println!();

    let sweep_cap = logical_cores.saturating_sub(1).max(1);
    let mut sweep: Vec<usize> = [1usize, 2, 4, 7].into_iter().filter(|&n| n <= sweep_cap).collect();
    if sweep.last().copied() != Some(sweep_cap) {
        sweep.push(sweep_cap);
    }
    sweep.dedup();

    let mut configs = vec![
        RunConfig { subscriber_count: 1,
                    rate_per_sec: 100_000,
                    label: "1 subscriber, stress load (100K/s)".into() },
        RunConfig { subscriber_count: 1,
                    rate_per_sec: 1_000,
                    label: "1 subscriber, sparse load (1K/s)".into() },
    ];
    for &n in sweep.iter().filter(|&&n| n != 1) {
        configs.push(RunConfig { subscriber_count: n,
                                 rate_per_sec: 100_000,
                                 label: format!("{n} subscribers") });
    }
    configs.push(RunConfig {
        subscriber_count: 64,
        rate_per_sec:     100_000,
        label:            "64 subscribers [OVERSUBSCRIBED - measures OS scheduling, not ring latency]".into(),
    });

    for config in &configs {
        run_config(config, args.pin, &core_ids);
    }
}

fn run_config(config: &RunConfig, pin: bool, core_ids: &[core_affinity::CoreId]) {
    let (warmup, measured) = samples_for_rate(config.rate_per_sec);
    let total_events = warmup + measured;

    println!("--- {} ---", config.label);
    println!("offered load: {} events/sec (warmup: {warmup}, measured: {measured}, est. {}s)",
              config.rate_per_sec,
              total_events / config.rate_per_sec.max(1),
    );

    // Heartbeat watcher - separate thread
    let heartbeat_done = Arc::new(AtomicBool::new(false));
    let heartbeat_done_w = Arc::clone(&heartbeat_done);
    let watcher = thread::spawn(move || {
        let start = Instant::now();
        while !heartbeat_done_w.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(5));
            if !heartbeat_done_w.load(Ordering::Relaxed) {
                eprintln!(" ... still running ({}s elapsed)", start.elapsed().as_secs());
            }
        }
    });

    let (tx, sub) = broadcast::channel::<JobEvent>(CHANNEL_CAPACITY);

    // subscribe every consumer before the producer starts
    let receivers: Vec<_> = (0..config.subscriber_count).map(|_| sub.subscribe()).collect();

    let barrier = Arc::new(Barrier::new(config.subscriber_count + 1));

    // consumers must NOT terminate by counting up to total_events
    let producer_done = Arc::new(AtomicBool::new(false));

    let mut consumer_handles = Vec::new();
    for (idx, mut rx) in receivers.into_iter().enumerate() {
        let barrier = Arc::clone(&barrier);
        let producer_done = Arc::clone(&producer_done);
        let core_id = pin.then(|| core_ids.get(idx).copied()).flatten();

        consumer_handles.push(thread::spawn(move || {
            if let Some(core_id) = core_id {
                core_affinity::set_for_current(core_id);
            }
            barrier.wait();

            let mut histogram = Histogram::<u64>::new_with_bounds(1, 10_000_000_000, 3).unwrap();
            let mut seen = 0u64;

            let record = |event: JobEvent, seen: &mut u64, histogram: &mut Histogram<u64>| {
                *seen += 1;
                if *seen > warmup {
                    let latency_ns = monotonic_raw_ns().saturating_sub(event.emitted_at);
                    let _ = histogram.record(latency_ns);
                }
            };

            loop {
                match rx.try_recv() {
                    Some(event) => record(event, &mut seen, &mut histogram),
                    None => {
                        if producer_done.load(Ordering::Acquire) {
                            while let Some(event) = rx.try_recv() {
                                record(event, &mut seen, &mut histogram);
                            }
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
            }

            (histogram, rx.lap_count())
        }));
    }

    let producer_core = pin.then(|| core_ids.last().copied()).flatten();
    let rate = config.rate_per_sec;
    let barrier_p = Arc::clone(&barrier);
    let producer_done_p = Arc::clone(&producer_done);

    let producer_handle = thread::spawn(move || {
        if let Some(core_id) = producer_core {
            core_affinity::set_for_current(core_id);
        }
        barrier_p.wait();

        let interval_ns = 1_000_000_000 / rate;
        let mut next_send = monotonic_raw_ns();

        for i in 0..total_events {
            while monotonic_raw_ns() < next_send {
                std::hint::spin_loop();
            }
            let now = monotonic_raw_ns();
            tx.send(JobEvent { run_seq: i, job_seq: i, state_code: 0, emitted_at: now });
            next_send += interval_ns;
        }

        producer_done_p.store(true, Ordering::Release);
    });

    producer_handle.join().unwrap();

    let mut histograms = Vec::new();
    let mut any_lapped = false;
    for handle in consumer_handles {
        let (hist, laps) = handle.join().unwrap();
        if laps > 0 {
            any_lapped = true;
            eprintln!("WARNING: a consumer lapped {laps} time(s) - this run is invalid");
        }
        histograms.push(hist);
    }

    heartbeat_done.store(true, Ordering::Relaxed);
    watcher.join().unwrap();

    let mut merged = Histogram::<u64>::new_with_bounds(1, 10_000_000_000, 3).unwrap();
    for h in &histograms {
        merged.add(h).unwrap();
    }

    if any_lapped {
        println!("*** INVALID RUN: a consumer lapped the ring ***");
    }
    println!("samples recorded: {}", merged.len());
    println!("p50:   {:>8} ns", merged.value_at_quantile(0.5));
    println!("p99:   {:>8} ns", merged.value_at_quantile(0.99));
    println!("p99.9: {:>8} ns", merged.value_at_quantile(0.999));

    if histograms.len() > 1 {
        let p99s: Vec<u64> = histograms.iter().map(|hist| hist.value_at_quantile(0.99)).collect();
        println!(
            "per-consumer p99 spread: min={} ns, max={} ns",
            p99s.iter().min().unwrap(),
            p99s.iter().max().unwrap()
        );
    }
    println!();
}