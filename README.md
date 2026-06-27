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
- **Broadcast** — lock-free SPMC ring buffer (seqlock, position-encoded versions, `!Sync` sender); feeds `roster top` (v0.2)

## Resource observation (v0.3)

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

Roster manages one node. For multi-node workloads, Roster is designed to act
as the per-node agent in a two-level cluster scheduler — exposing resource
availability and accepting remote submissions while a global scheduler handles
placement across nodes.

## Status

v0.1.0 shipped. v0.2 in progress: `roster top` TUI with lock-free SPMC event
broadcast and HDR-histogram latency measurement.

## License

GPL-3.0