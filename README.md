# roster

Single-node resource-aware execution runtime. Submit a DAG of jobs as a YAML
spec; the daemon schedules them in dependency order, observes and reconciles
CPU/RAM/VRAM usage against declared values, and persists run state across
restarts.

## Quickstart

```bash
cargo build --release

# start the daemon
roster daemon

# submit a workflow
roster submit examples/cpu-only.yaml

# watch active runs
roster ps

# show job-level detail for a run
roster status <run-id>

# get log path for a job
roster logs <run-id>/<job-id>

# cancel a run
roster cancel <run-id>
```

## Workflow spec

```yaml
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
      gpu: 1
      vram_mb: 12000

  - id: eval
    depends_on: [train]
    command: python eval.py
    resources:
      cpu: 2
      gpu: 1
      vram_mb: 4000
```

`gpu` is a count (number of GPUs required), not a device index. `vram_mb` is
per-GPU. Jobs with `gpu: 0` are CPU-only. The scheduler places jobs on the
first available GPUs with sufficient free VRAM (first-fit).

Jobs run as soon as their dependencies finish and sufficient resources are
available. The scheduler blocks rather than evicts.

## Architecture

- **Daemon** — tokio async runtime, Unix socket IPC, graceful SIGTERM/SIGINT shutdown with in-flight drain
- **IPC** — newline-delimited JSON over `$XDG_RUNTIME_DIR/roster.sock` (fallback `~/.local/run/`)
- **Scheduler** — 100ms tick loop: Kahn's DAG traversal, resource admission, failure cascade to Skipped
- **Executor** — `sh -c` subprocess with `setpgid(0,0)`; cancel via `killpg(SIGTERM)` → 5s → `SIGKILL`; stdout/stderr captured to per-job log file
- **Resource discovery** — sysinfo (CPU/RAM), NVML (per-GPU VRAM, graceful fallback if no driver)
- **Resource accounting** — user-declared, conservative; `try_reserve` → `Allocation` → `release` on every terminal transition
- **Storage** — SQLite via sqlx; run/job state persisted on every transition; restart reconciliation marks interrupted jobs terminal
- **Broadcast** — lock-free SPMC ring buffer: seqlock with position-encoded versions over per-word `Relaxed` atomics (not `UnsafeCell` — see below), cache-line-isolated slots, `!Sync` sender enforcing single-producer at compile time; loom model-checked (exhaustive for the bounded modeled scenario) for memory-model soundness; feeds `roster top` (v0.2, not yet built — see Status)

## Broadcast latency

The SPMC ring buffer is benchmarked directly — a synthetic producer/consumer
against `broadcast::channel`, with no scheduler or IPC in the path. This
isolates ring latency from scheduler-tick and IPC cost, which are separate,
later measurements.

**Why per-word atomics, not `UnsafeCell`:** the original implementation
stored each slot's payload in `UnsafeCell<MaybeUninit<T>>`. Loom model
checking found a genuine causality violation — a race between the producer's
write and a consumer's read of the same cell is undefined behavior the
instant it occurs, even if the seqlock's version recheck later discards the
result. The fix (Boehm, "Can Seqlocks Get Along with Programming Language
Memory Models?", MSPC 2012) moves the payload into real per-word `Relaxed`
atomics instead: racing atomic operations are *defined* by construction, so
the same recheck-and-discard pattern is now sound under the formal memory
model, not just correct on real hardware. Verified via exhaustive
interleaving search:

```bash
LOOM_MAX_PREEMPTIONS=3 cargo test --release --features loom --lib broadcast::loom_tests
```

**Machine:** 8-core Intel Lunar Lake (Core Ultra 200V) — 4 P-cores + 4
E-cores, no SMT, per Intel's published specification (not independently
verified via topology inspection on this unit).

### Headline numbers (unpinned, most recent of three runs)

| Config | p50 | p99 | p99.9 |
|---|---|---|---|
| 1 subscriber, 100K events/sec | 110 ns | 131 ns | 140 ns |
| 1 subscriber, 1K events/sec (sparse) | 117 ns | 249 ns | 494 ns |
| 2 subscribers, 100K events/sec | 124 ns | 159 ns | 264 ns |
| 4 subscribers, 100K events/sec | 383 ns | 1,175 ns | 1.6–11.6 μs\* |

\* p99.9 at 4 subscribers varied 1.6–11.6 μs across three runs; p50/p99 were
stable. 4 subscribers means 5 busy threads (4 consumers + 1 producer)
against 4 P-cores — right at the edge of the P-core budget, where placement
effects start to appear in the tail.

**Reproducibility** (p50 / p99 range across three unpinned runs):

| Config | p50 range | p99 range |
|---|---|---|
| 1 sub, stress | 110–112 ns | 129–131 ns |
| 1 sub, sparse | 117–135 ns | 249–354 ns |
| 2 subs | 121–124 ns | 154–159 ns |
| 4 subs | 383–586 ns | 976–1,215 ns |

100K events/sec is a deliberate stress load — 3–5 orders of magnitude above
realistic scheduler transition rates (~1–100/sec) — chosen to characterize
the ring under sustained traffic; production operates far below this, where
the ring is trivially uncontended. The 1-subscriber sparse row (1K/sec) is
run specifically to surface a counterintuitive result: idle latency is
*worse* than under load. After a millisecond of silence, the consumer's
cached copy of the slot's cache line is stale, so the read pays a coherency
fetch it wouldn't incur under sustained traffic.

### Saturation boundary: 7 subscribers

7 subscribers means 8 busy-spinning threads (7 consumers + 1 producer) —
exactly the machine's total core count, P and E combined, leaving zero
scheduling headroom. Three unpinned runs all show the same failure mode:
tail latency governed by OS scheduling quanta, not the ring.

| Run | Lapped? | p99 | p99.9 | Per-consumer p99 spread |
|---|---|---|---|---|
| 1 | Once | 1,318 ns | 2.76 ms | 980 ns – 1,367 ns |
| 2 | No | 5,807 ns | 3.06 ms | 1,441 ns – 194,431 ns |
| 3 | No | 262,399 ns | 7.47 ms | 1,713 ns – 1,781,759 ns |

Whether a given run technically laps the 4096-slot buffer is close to
timing luck; the underlying contention is present every time. p99.9 in the
low-single-digit milliseconds is consistent with a full CFS scheduling
quantum being stolen from a spinning consumer thread. This row is excluded
from the headline numbers not because it "fails," but because at this point
the benchmark is measuring OS scheduler behavior rather than ring latency —
which is itself the useful, diagnosed boundary of where these results stop
describing the data structure.

### Oversubscription: 64 subscribers

65 busy threads on 8 cores. Included deliberately as a demonstration of the
boundary, not a result: every run laps on every consumer (23–81 times
each), p50 in the 8-millisecond range, consistently and reproducibly
invalid. This confirms the harness's lap-detection correctly flags an
invalid run rather than silently reporting corrupted percentiles.

### Core pinning: attempted, rejected

`--pin` assigns each thread to a distinct logical core via `core_affinity`,
by array index. One pinned run was measured against the unpinned baseline:
pinning regressed p50/p99 by roughly 3–12× across every stable
configuration (1/2/4 subscribers), and did not fix the 7-subscriber
saturation case.

Most likely cause: naive index-based core assignment has no awareness of
the P-core/E-core split. Pinning removes the OS scheduler's ability to
migrate a thread away from a momentarily busy or less-capable core — on a
heterogeneous chip, that can permanently seat a latency-sensitive thread on
a worse core with no escape. **Unpinned is the default and the reported
configuration** — this was a measured outcome, not an assumption.

### Methodology

- **Clock**: `CLOCK_MONOTONIC_RAW` — unaffected by NTP slewing, monotonic across suspend/resume.
- **Histogram**: HDR histogram (`hdrhistogram` crate), 1 ns – 10 s range, 3 significant figures.
- **Sample counts**: scaled to offered rate, capped at 15s wall-clock per configuration (`min(rate × 15, 1,000,000)` measured samples, `min(rate, 10,000)` warmup samples discarded per-consumer) — a fixed sample count that's ~10s at 100K/sec becomes ~17 minutes at 1K/sec otherwise.
- **Termination**: producer-done signal plus drain-to-empty, not a send-count comparison — a lapped consumer permanently skips events by design, so count-based termination hangs forever on any lapped run.
- **Validity**: any consumer lapping the ring invalidates that run; flagged automatically, not filtered silently.
- **Reproducibility**: three unpinned runs per configuration; ranges reported above.

Run it yourself: `cargo run --release --bin bench_broadcast [-- --pin]`

## Resource observation (v0.3, planned)

The daemon will sample actual CPU/RAM/VRAM usage at 1 Hz via NVML and sysinfo,
track per-job peaks, and reconcile declared vs actual on completion. Over runs
this builds a workload profile for autosuggest.

## Non-goals

- Multi-node / distributed scheduling (permanent)
- Preemption or job migration
- Auto-resume of interrupted jobs (`Interrupted` is terminal)
- CUDA dependency (NVML only — observes GPU resources, never executes GPU work)
- Retry policy (implement in your job command)

## Designed to compose

Roster manages one node and has no network listener — IPC is `AF_UNIX`,
local-machine only, and multi-node scheduling is a permanent non-goal (see
above). The single-node design is nonetheless shaped for composability: an
external global scheduler could treat one `roster` daemon per node as a
local admission and execution agent, querying resource state and submitting
jobs over the same local socket API a same-machine client uses today.
Roster does not implement, and has no plan to implement, any cross-node
transport itself — that responsibility stays entirely outside this repo.

## Status

v0.1.0 shipped. v0.2 in progress: lock-free SPMC event broadcast is
implemented, loom-verified, and benchmarked (see Broadcast latency above).
`roster top` — the TUI that consumes this event stream — is not yet built.

## License

GPL-3.0