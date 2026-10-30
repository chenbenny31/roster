# Roster

Roster runs your commands on a small pool of Linux machines with GPUs. You submit an ordinary command such as `python train.py`. Roster queues it, picks a machine and GPUs that are free, runs it, keeps its logs, and tells you when something needs your attention. Training, batch inference, evaluation, and data processing all use the same path.

**Why Roster.** With a handful of GPU machines, the usual options sit at two extremes. Running jobs by hand over SSH means checking `nvidia-smi`, picking free GPUs yourself, and noticing failures late. Cluster schedulers such as Slurm or Kubernetes handle placement but expect cluster administration: the same accounts and config on every node, job scripts, an accounting database, and a separate monitoring stack to see what each GPU is doing. Roster fills the gap between them:

- **One binary per machine.** Every machine runs the same daemon, which finds its own CPUs, memory, and GPUs. `roster node add` registers a machine you already reach over SSH and starts its daemon there. There is no shared filesystem or database server to set up.
- **One command per job.** `roster run -- python train.py` queues the command as it is. When every node is busy, jobs wait in order and start as soon as resources free up, and a large job can't be starved by a stream of small ones.
- **Every job visible live.** Each node streams job events and resource use to the controller, so `roster watch` shows which job holds which GPU on which node within milliseconds. A node that goes quiet shows as stale, not idle.
- **Waiting explained.** `roster explain` reads the scheduler's own decision instead of a reason code.
- **Safe handover.** A GPU goes to the next job only after the previous job's processes are verified gone, and a job is never started twice after a crash or a lost message.

Roster is in early development; the [roadmap](#roadmap) shows what works today.

## Usage

```bash
roster run -- python train.py               # returns a job ID once the job is stored
roster run --gpus 4 -- torchrun --standalone --nproc-per-node=4 train.py
roster run --vram 140G -- 'vllm serve MODEL --tensor-parallel-size $ROSTER_GPUS'   # fewest GPUs that hold it
roster jobs [--json]
roster watch                                # live jobs and nodes
roster show JOB_ID
roster explain JOB_ID                       # why it waits, from the scheduler's own decision
roster logs JOB_ID [--attempt N] [--follow]
roster next JOB_ID                          # move to the front; never kills running work
roster cancel JOB_ID                        # returns once the stop is stored, completes after cleanup
roster retry JOB_ID                         # failed jobs wait for this
roster start                                # on every machine; also stop, restart
roster enable                               # start at boot and keep running after logout; also disable
roster info [NODE]                          # this machine, or a node as the controller sees it
roster node add SSH_ALIAS                   # on the controller
roster node ls                              # also node rm
roster node drain NODE_ID
```

A default profile per queue or workspace supplies resources, environment, and paths, and flags such as `--gpus` override it. Commands run as argv without a shell, with the working directory mapped through the node's profile, stdin from `/dev/null`, and only selected environment variables. Every request carries an ID, so resending after a lost reply returns the original job. Workflow files with dependencies stay available as optional input.

Normal events are recorded without alerts. A failure that needs action becomes an issue with the known facts and suggested actions.

## Design

```text
CLI → controller: durable queue → first-fit placement → identified allocation
                                                      │
                         ┌────────────────────────────┴──────────────┐
                         ▼                                           ▼
                   worker: node A                              worker: node B
                   managed processes                           managed processes
                   CPU / RAM / GPUs                            CPU / RAM / GPUs
```

Every machine runs the same daemon. As a **worker**, it finds its own resources at startup and owns local admission, processes, logs, and cleanup, running many jobs at once. A daemon with registered nodes is also the **controller**, which owns the queue and placement. On a single machine both roles live in one daemon. The controller reaches each worker through one SSH connection that forwards the worker's Unix socket, so there is no network listener to secure, and closing the connection doesn't stop jobs.

**Live state.** Each worker streams its job events and resource samples to the controller: a snapshot first, then batches every 10 ms and a heartbeat every second. A stream that falls behind gets a fresh snapshot. The stream is telemetry only. An exit event makes the controller ask the worker at once, and the worker's reply is what counts. `roster watch` shows every node on the controller and one node on a worker.

**Identities.** A **job** is what you submit, inspect, cancel, and retry; its spec never changes. An **attempt** is one try of the whole job, and a retry creates a new one. An **execution** is one node's part of an attempt. An **allocation** is the exact CPU/RAM budget and GPU UUIDs an execution holds. Because messages name their allocation, a late "exited" report can't free resources that now belong to another job. An attempt holds a list of executions: one for now, several for coordinated jobs later.

**Launch once.** A database commit and a process start can't be atomic, so starting takes four steps:

1. The controller stores the attempt and its allocations in one transaction.
2. **Prepare:** the worker rechecks its inventory, records the holds with a closed launch gate, and starts a runner inside a systemd service (cgroup v2). No user code runs yet.
3. **Activate:** once every execution reports Prepared, the controller commits activation and the worker opens the gate.
4. The runner gets permission exactly once. The worker records it as issued before handing it over and never issues it again. If delivery is uncertain, the outcome is `outcome_unknown` instead of a second launch.

A stop closes the gate on disk first, even when it arrives before Prepare.

**Release only with proof.**

```text
may_release(allocation) =
    launch gate closed
    AND managed process tree empty           (cgroup populated=0)
    AND GPU checks pass                      (same UUIDs, NVML queries OK, no stray contexts)
    AND release recorded by the worker
```

Child exit, `ECHILD`, a lost heartbeat, or a timeout never count as proof. If cleanup times out, the allocation is quarantined.

**Durable state.** The controller and every worker keep SQLite with WAL and `synchronous=FULL`. Nothing is acknowledged before it commits. A failed write stops new admission and keeps known allocations held. Workers keep their journal on local disk and keep managing their jobs while the controller is down.

**Queue and placement.** There is one FIFO queue. `next` promotes a job, and a bound keeps promoted jobs from starving the oldest waiting one. Placement is first-fit over a fixed node order: the first node that fits the whole request (CPU, RAM, and the full GPU set) wins. A job never combines GPUs from two machines. Placement runs again as soon as a worker reports an exit, so freed resources don't sit idle. An older job that can't fit yet gets a **ticket** on a node, so younger jobs only use spare resources there until it fits. Busy or disconnected nodes mean the job waits; Roster rejects a job only when known capacity could never fit it. `explain` reads the same decision the scheduler made:

```text
Job 42 is waiting for a GPU.
Job 17 is stopping and has not released it yet.
Roster will start this job when cleanup is confirmed.
```

**Resources.** A requirement ("one GPU with at least 24 GB"), the claim placement makes (that GPU's UUID), and what the backend enforces are kept separate. GPUs start as exclusive whole devices, and `CUDA_VISIBLE_DEVICES` gets their UUIDs. A job asks for a GPU count, a total VRAM, or both. With only VRAM, Roster picks the fewest GPUs on one node (1, 2, 4, or 8, the tensor-parallel sizes) that each hold an even share, and `ROSTER_GPUS` tells the job how many it got.

**Unknown stays unknown.** A disconnected node isn't treated as empty, a missing result isn't success, and uncertain work is never started again elsewhere. After any restart, Roster loads unreleased allocations before computing free capacity and then reconciles each worker. Retries are explicit; Roster re-places a job on its own only when it can prove no execution ever got launch permission.

## APIs

Each worker exposes the same operations:

| Operation | Contract |
| --- | --- |
| Info | Node identity, allocatable CPU/RAM, GPU UUIDs and health, runtime profiles |
| Prepare | Reserve all local resources or reject; the gate stays closed |
| Activate | Open the gate for a matching preparation |
| Stop | Close the gate on disk, clean up, report release separately |
| Sync | Versioned snapshot after reconnect |
| Logs | Stream by execution and offset; recording continues while disconnected |

Messages carry a protocol version, request ID, and the IDs they act on. A worker rejects an unknown required field or mode instead of ignoring it, so a multi-node job can't silently run on one node.

A job request separates the resource shape from how the program is launched:

```rust
struct JobSpec {
    shape: JobShape,
    launcher: LauncherSpec,            // Command now, Torchrun later
    environment: ProfileRef,
    placement: PlacementRequirements,
}

struct JobShape {
    nodes: NonZeroU16,
    per_node: LocalRequirements,       // CPU/RAM budgets, GPU count and minimum memory
}
```

1×0 is a CPU job, 1×4 is four GPUs on one node, and 2×4 is eight GPUs across two nodes. The first release accepts only `nodes = 1`.

## Event broadcast

The scheduler publishes job events through an in-process single-producer, multi-consumer ring ([broadcast.rs](src/broadcast.rs)) that IPC handlers subscribe to. It carries telemetry only: control messages never use it, and a consumer that falls a full lap behind reloads a snapshot. Across machines the ring stays local. Each worker has its own, and its stream to the controller is one more subscriber, so a slow or broken link never makes a worker wait. Over a network, latency comes from the link and the 10 ms batch, not the ring.

Each slot is a seqlock whose payload sits in per-word `Relaxed` atomics. The first version stored it in `UnsafeCell<MaybeUninit<T>>`, and loom showed that a reader racing the writer on that cell is undefined behavior even when the version check throws the value away (Boehm, "Can Seqlocks Get Along with Programming Language Memory Models?", MSPC 2012). Slots are cache-line aligned and the sender is `!Sync`.

p99 latency at 100K events/s, three unpinned runs:

| Subscribers | Intel Lunar Lake, 8 cores | AMD Ryzen 9 9900X, 24 threads |
|---|---|---|
| 1 | 129–131 ns | 80–90 ns |
| 2 | 154–159 ns | 210–311 ns |
| 4 | 976–1,215 ns | 170–180 ns |
| 7 | one run lapped; others 5.8 μs and 262 μs | 481–722 ns |

- Any lap invalidates a run. 64 subscribers lapped in every run on both machines, and 23 subscribers lapped in two of three runs on the Ryzen.
- Pinning threads to cores (`--pin`) made p50 and p99 3–12× worse on Lunar Lake, so unpinned is the default.
- 100K events/s is a stress load, far above a scheduler's real rate of about 1–100 events/s. The numbers cover the ring alone, without scheduling, IPC, or SQL.

Timing uses `CLOCK_MONOTONIC_RAW` and an HDR histogram with up to 1M samples per configuration after warmup. Reproduce with `cargo run --locked --release --bin bench_broadcast`, adding `-- --pin` for the pinning run.

## Scope

Supported: 64-bit Linux, NVIDIA GPUs with drivers and runtimes already installed, and one user or mutually trusted users per installation. One controller runs the cluster, with no HA.

Roster won't:

- Parallelize a program, split datasets, or merge outputs.
- Copy code or data to nodes, install environments, or provision machines. A profile maps paths per node and is checked before launch.
- Enforce per-job CPU, RAM, or VRAM limits. Those numbers are admission budgets. All jobs on a node share one memory cap that leaves headroom for the system.
- Clean up work a job hands off outside its cgroup, for example over SSH or to a container daemon.

Not planned for the first release: checkpoint or migration, MIG/MPS, isolation between untrusted users, container orchestration, an RDMA runtime, and a web dashboard.

## Roadmap

S5 comes before S4 because multi-node placement matters more than preemption.

| Stage | Delivers | Status |
| --- | --- | --- |
| S0 | Single-node prototype: YAML workflows with dependencies, CPU/RAM/VRAM budgets, first-fit GPU selection, shell processes, SQLite status, locked build | Done |
| **P1** | **Multi-node prototype ahead of S1–S5: a daemon on every node, `node add`, one queue across nodes, live node streams, `roster run` and `watch`; no launch gate or restart recovery yet** | **Current** |
| S1 | Job/attempt/execution/allocation model, FIFO with promotion and tickets, first-fit, typed claims, request IDs | Next |
| S2 | Reliable local execution: runner, launch gate, systemd service, cancel, deadlines, restart recovery | Planned |
| S3 | GPUs by UUID, exclusive allocation, release checks on real hardware | Planned |
| S5 | Shared queue across nodes: enrollment, drain, profiles, reconciliation; CPU-only can follow S2 | Planned |
| S4 | Opt-in preemption of restartable GPU jobs (`preempt=restart`, minimum runtime, once per job) | Planned |
| S6 | CLI polish, notifications, log following | Planned |
| S7 | Coordinated multi-node jobs with Torchrun: whole-bundle reservation, group start, stop, and retry, rendezvous per attempt | Planned |
| S8 | RDMA, GPUDirect, and NUMA/topology placement on tested hardware | Planned |
| S9 | Checkpoint resume, elasticity, service replicas, and roles, each with its own design | Planned |

The current prototype uses its own commands:

```bash
cargo build --locked                 # add --features nvml for NVIDIA discovery and GPU readings
roster start                         # also stop, restart, info; enable and disable use systemd
roster node add SSH_ALIAS            # on the controller, then roster restart; also node ls, node rm
roster info NODE                     # state, running jobs, free resources, GPU temperature and power
roster submit examples/cpu-only.yaml
roster ps
roster status <run-id>
roster logs <run-id>/<job-id>
```

Each job already runs in its own systemd scope under a memory-capped `roster.slice`, gets whole GPUs, and a GPU busy with work Roster didn't start is skipped until it frees.

Shared GPUs are a separate branch after S3, for example an 8 GB and a 2 GB job on one 16 GB card. They need per-job claims and a validated backend (budgets, MPS, or MIG). A memory split says nothing about compute share.

Open questions: systemd and cgroup support on target machines (S2), whether the launch gate holds at every crash point (S2), how soon a GPU is safe to reuse after cancel (S3), the notification channel (S6), controller capacity (S5), and how group bundles interact with tickets and who owns the rendezvous endpoint (S7).

## License

GPL-3.0
