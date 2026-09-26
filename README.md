# Roster

**Compute infrastructure and workload scheduling in Rust.**

Roster is a Rust execution runtime being developed to make CPU and GPU workloads
easy to run across available machines. Its design centers on a lightweight daemon:
submit an ordinary command, let Roster choose when and where it runs, and receive
actionable notifications when something goes wrong. Training, batch inference,
data processing, and other command-line workloads share the same scheduling core.

**Status:** the repository contains a single-node workflow prototype and a
benchmarked event-broadcast component. The controller/worker architecture,
multi-node scheduling, and command-first interface below are **proposed, not
implemented**. The sections below describe the intended architecture and delivery
plan; the current prototype and its limitations are documented separately below.

## Intended experience

Proposed CLI, not commands available in the current prototype:

```bash
roster run -- python train.py
roster run -- python infer.py --input batch-a.jsonl
roster jobs
roster show JOB_ID
roster next JOB_ID
roster cancel JOB_ID
```

No SDK or mandatory workflow file. A default execution profile supplies resource
and environment settings, with explicit overrides when needed. Roster places
each job on a compatible node and selects its GPUs. Users can inspect the command,
placement, logs, and known failure facts without watching every successful event.

`next` promotes waiting work; it does not interrupt a running job. Cancellation
stops managed work and waits for verified cleanup before reclaiming resources.
Genuine workload failures require a user-directed retry by default. Roster does
not automatically parallelize a command, install its environment, or provision
machines. See the [product contract](docs/roster-v2-design.md#product-contract).

## Core design

```text
CLI → controller: durable queue → first-fit placement → identified allocation
                                                      │
                         ┌────────────────────────────┴──────────────┐
                         ▼                                           ▼
                   worker: node A                              worker: node B
                   managed processes                           managed processes
                   CPU / RAM / GPUs                            CPU / RAM / GPUs
```

- **One queue, deterministic placement.** Queue policy chooses eligible work;
  first-fit selects the first compatible node with the complete requested resource
  set. Protection for older waiting jobs bounds bypass while allowing other work
  to use surplus resources. Unreachable capacity is not treated as free capacity.
- **One worker per node, many concurrent jobs.** Local and remote workers use the
  same execution contract. A worker validates assignments, manages processes,
  reports observations, and proves cleanup; it does not independently reorder the
  global queue or implement GPU kernel scheduling.
- **Explicit ownership.** Requirements describe what a job needs; durable claims
  identify its reserved resources; backend controls describe what is actually
  enforced. CPU/RAM budgets are not hard limits, and a VRAM reservation is not a
  guarantee of GPU compute fairness. Initial GPU delivery uses exclusive devices.
- **Recoverable lifecycle.** Prepare reserves and validates without starting user
  code. Activation permits launch once; cancellation closes that permission.
  Resources are released only after workload cleanup and device readiness are
  established. A lost heartbeat alone never permits release or duplicate launch.
- **Durable controller and worker state.** The proposed SQLite-backed controller
  records job intent, placement, and pending commands. Each worker keeps a local
  SQLite-backed journal of accepted allocations, launch permission, stop intent,
  and cleanup evidence. Recovery reconciles these records with managed processes;
  database transactions alone cannot make process launch atomic. See
  [persistence and journals](docs/roster-v2-design.md#persistence).
- **Useful observability.** Durable state explains ownership and decisions;
  logs, metrics, and events explain behavior. The broadcast ring carries telemetry,
  not authoritative state. Process launch and application readiness are separate
  facts, especially for inference servers.

The [command core](docs/roster-v2-design.md#command-core),
[worker contract](docs/roster-v2-design.md#cluster-worker), and
[resource contract](docs/roster-v2-design.md#resource-contract) define these
boundaries, including how future backends preserve resource ownership and recovery.

### From submission to cleanup

A job is the unit the user submits, inspects, cancels, and retries. Workflow
dependencies can order several jobs, but an ordinary command does not need a
workflow wrapper. The controller separates queue order from placement: it first
chooses eligible work, then finds a node that can satisfy the complete request.

The planned execution lifecycle is:

1. **Reserve:** record the selected node and resources durably; the worker checks
   local capacity and accepts ownership of the allocation.
2. **Prepare:** establish the managed process boundary and execution environment
   without running the user's command.
3. **Activate:** authorize that prepared command to start once, then report its
   observed state and capture logs locally.
4. **Reconcile:** after a restart or disconnect, compare durable intent with local
   execution evidence before deciding what can run next.
5. **Release:** confirm that managed descendants have stopped and the resource
   backend is ready before making the allocation available again.

Cancellation closes launch permission before cleanup. Missing contact leaves
ownership unresolved; it does not mean the job stopped. These rules apply to
local and remote execution alike. Application readiness, such as an inference
server becoming ready to accept requests, is a separate future capability.

### Adding machines

The planned enrollment flow uses an existing SSH connection to a prepared machine.
Each worker reports its identity, available resources, and supported execution
profiles. A profile describes the runtime and workspace needed by a command;
placement must check compatibility as well as free capacity.

A single-machine installation uses the same controller/worker structure as a
larger installation. Adding workers expands the pool for waiting jobs. Draining a
node prevents new assignments while allowing accepted work to finish. During a
controller disconnect, workers retain local process management and recording;
new placement waits for the controller. Enrollment does not install drivers or
silently change the machine's environment.

## Workload scope

| Scenario | Design path |
| --- | --- |
| CPU jobs, training, batch inference | Ordinary command on one selected node |
| One job using several GPUs | Reserve its complete GPU set on one node; the application uses those devices |
| Many jobs across machines | Shared queue assigns independent jobs to workers |
| Long-running inference server | Managed command initially; readiness, routing, and replica ownership are later extensions |
| One coordinated job spanning nodes | Later group preparation, launcher/rendezvous integration, and whole-group recovery |
| Several jobs sharing a GPU | Separate extension with explicit admission, enforcement, fairness, and cleanup validation |

The first multi-node milestone distributes independent jobs, with each job fitting
entirely on one node. A command that already supports multiple local GPUs can use
a complete device allocation after that path is validated.

Coordinated workloads are a later extension: one job requests a fixed group of
nodes, all members prepare before activation, and a launcher configures their
communication. A member failure triggers group cleanup; a retry waits until the
previous group is fully reclaimed. Independent inference replicas instead belong
to separate jobs, with future service policy managing their readiness and count.

CUDA execution belongs to the workload; NVML supplies GPU discovery and health
evidence. NCCL, RDMA, and NUMA are optional integration concerns in later stages,
not requirements for distributing independent jobs. Multiple GPUs do not imply
RDMA, and memory on different devices is not automatically pooled. Hardware paths
need their own validation; see [AI infrastructure coverage](docs/roster-v2-design.md#ai-infrastructure).

## Roadmap

No implementation stage below is complete. Stage IDs match the canonical design;
**S5 precedes S4** because multi-node placement comes before optional preemption.

| Order | Stage | Delivery gate |
| --- | --- | --- |
| 1 | S0 — validation preparation | Finalize core contracts and establish reproducible builds |
| 2 | S1 — scheduling core | Durable job/allocation model, typed resources, first-fit, and bounded queue bypass |
| 3 | S2 — reliable local execution | CPU process ownership, cancellation, cleanup, and restart reconciliation |
| 4 | S3 — GPU execution | NVML/UUID identity and exclusive GPU ownership; validate local multi-GPU separately |
| 5 | S5 — shared multi-node queue | Node enrollment, compatible execution profiles, reusable workers, and partition recovery |
| 6 | S6 — user experience | Simple onboarding, job explanations, log following, and actionable error notifications |

User experience is part of every stage. CPU-only multi-node work can follow S2
without waiting for GPU hardware validation.

Later branches extend the same ownership and lifecycle contracts:

- **S4:** optional interruption/restart of explicitly eligible jobs.
- **After S3:** validated shared-GPU modes, independent of distributed training.
- **S7:** fixed coordinated groups, beginning with a selected training launcher.
- **S8:** verified topology, NUMA, and RDMA/GPUDirect paths on supported hardware.
- **S9:** service/replica ownership, roles, checkpointing, and elasticity as
  separately scoped extensions. Serving features need not wait for S7 or RDMA.

See the [full roadmap and extension gates](docs/roster-v2-design.md#roadmap).

## Current prototype

The existing code accepts YAML workflows and schedules a dependency DAG locally.
It includes CPU/RAM budget accounting, per-GPU VRAM admission and first-fit
selection, shell subprocesses with process groups and file logs, SQLite status
records, and Unix-domain IPC. These are foundations, not the proposed worker's
recovery and enforcement guarantees.

Known limitations:

- Default discovery returns CPU/RAM inventory and no GPUs. The optional NVML
  code path references a dependency absent from the manifest.
- CLI `cancel` is unimplemented. `logs` returns a file path, not a live stream.
- Restart handling marks recorded running jobs interrupted; it does not recover
  durable resource ownership or reconstruct the proposed queue.
- Process-group execution and declared budgets do not establish reliable
  descendant cleanup or enforced resource isolation.

See [implementation findings](docs/roster-v2-design.md#findings) for details.

### Existing interface

With a prepared prototype binary, the existing interface is:

```bash
roster daemon
roster submit examples/cpu-only.yaml
roster ps
roster status <run-id>
roster logs <run-id>/<job-id>
```

The [CPU-only example](examples/cpu-only.yaml) shows the current YAML format.
`gpu` is a device count and `vram_mb` is a per-device admission budget; CPU-only
jobs use `gpu: 0`. These fields describe the prototype, not the proposed typed
resource contract.

Reproducible build setup is still pending. See the
[development validation plan](docs/roster-v2-design.md#validation) for lockfile
setup and build commands.

<a id="recorded-benchmarks"></a>

## Broadcast latency

Historical measurements of the single-producer, multiple-consumer (SPMC) ring:
**131 ns p99 with one subscriber; 159 ns with two**, at 100K events/sec.
The benchmark measures synthetic publication and consumption through
`broadcast::channel`; scheduling, IPC, SQL, and process launch are outside the
measurement. See the [benchmark evidence](docs/roster-v2-design.md#recorded-benchmarks)
for provenance and reproduction status.

**Why per-word atomics, not `UnsafeCell`:** the original implementation
stored each slot's payload in `UnsafeCell<MaybeUninit<T>>`. Loom model
checking found a genuine causality violation — a race between the producer's
write and a consumer's read of the same cell is undefined behavior the
instant it occurs, even if the seqlock's version recheck later discards the
result. The fix (Boehm, "Can Seqlocks Get Along with Programming Language
Memory Models?", MSPC 2012) moves the payload into real per-word `Relaxed`
atomics instead, under an unsafe `WordSafe` payload contract. The implementation
also uses cache-line-isolated slots and a `!Sync` sender. Historical Loom checking
covered a bounded modeled scenario (`LOOM_MAX_PREEMPTIONS=3`); it is not a universal
soundness proof. See [broadcast.rs](src/broadcast.rs) and the
[design notes](docs/roster-v2-design.md#recorded-benchmarks).

**Recorded benchmark machine:** 8-core Intel Lunar Lake (Core Ultra 200V).

### Headline numbers (unpinned, latest recorded of three runs)

| Config | p50 | p99 | p99.9 |
|---|---|---|---|
| 1 subscriber, 100K events/sec | 110 ns | 131 ns | 140 ns |
| 1 subscriber, 1K events/sec (sparse) | 117 ns | 249 ns | 494 ns |
| 2 subscribers, 100K events/sec | 124 ns | 159 ns | 264 ns |
| 4 subscribers, 100K events/sec | 383 ns | 1,175 ns | 1.6–11.6 μs\* |

\* p99.9 at 4 subscribers varied 1.6–11.6 μs across three runs. Four subscribers
means five busy threads (four consumers and one producer); the measurements alone
do not establish which scheduling or topology effects caused the variation.

**Reproducibility** (p50 / p99 range across three unpinned runs):

| Config | p50 range | p99 range |
|---|---|---|
| 1 sub, stress | 110–112 ns | 129–131 ns |
| 1 sub, sparse | 117–135 ns | 249–354 ns |
| 2 subs | 121–124 ns | 154–159 ns |
| 4 subs | 383–586 ns | 976–1,215 ns |

100K events/sec characterizes sustained synthetic traffic, not a measured
production event rate. The sparse 1K/sec case has higher reported latency;
cache/coherency explanations remain unverified without supporting traces.

### Saturation boundary: 7 subscribers

Seven subscribers means eight busy-spinning threads. The three recorded unpinned
runs show substantial tail variation and a lap in one run:

| Run | Lapped? | p99 | p99.9 | Per-consumer p99 spread |
|---|---|---|---|---|
| 1 | Once | 1,318 ns | 2.76 ms | 980 ns – 1,367 ns |
| 2 | No | 5,807 ns | 3.06 ms | 1,441 ns – 194,431 ns |
| 3 | No | 262,399 ns | 7.47 ms | 1,713 ns – 1,781,759 ns |

The lapped run is invalid under the harness's validity rule. The other two runs
show that avoiding laps does not ensure low or consistent tail latency. OS
scheduling is a possible contributor, not a diagnosed cause from these tables.

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

Heterogeneous core placement is an unverified explanation for this regression;
the recorded experiment does not establish its cause. **Unpinned is the default
and the reported configuration.**

### Methodology

- **Clock**: `CLOCK_MONOTONIC_RAW` for local elapsed-time measurements.
- **Histogram**: HDR histogram (`hdrhistogram` crate), 1 ns – 10 s range, 3 significant figures.
- **Sample counts**: scaled to offered rate, capped at 15s wall-clock per configuration (`min(rate × 15, 1,000,000)` measured samples, `min(rate, 10,000)` warmup samples discarded per-consumer) — a fixed sample count that's ~10s at 100K/sec becomes ~17 minutes at 1K/sec otherwise.
- **Termination**: producer-done signal plus drain-to-empty, not a send-count comparison — a lapped consumer permanently skips events by design, so count-based termination hangs forever on any lapped run.
- **Validity**: any consumer lapping the ring invalidates that run; flagged automatically, not filtered silently.
- **Reproducibility**: three unpinned runs per configuration; ranges reported above.

After completing the build setup, run
`cargo run --frozen --release --bin bench_broadcast` (append `-- --pin` for the
pinning experiment).

## License

GPL-3.0
