# roster

GPU-aware single-node workflow scheduler. Submit a DAG of jobs as a YAML spec;
the daemon schedules them in dependency order, tracks CPU/memory/VRAM as
first-class resources, and persists run state across restarts.

## Quickstart

```bash
cargo build --release

# start the daemon
roster daemon

# submit a workflow
roster submit examples/train-and-eval.yaml

# watch active runs
roster ps

# tail logs for a job
roster logs <job-id>

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

Jobs run as soon as their dependencies finish and sufficient resources are
available. The scheduler blocks a job rather than evicting a running one.

## Architecture

- **Daemon** — tokio async runtime, Unix socket IPC, graceful SIGTERM shutdown
- **IPC** — newline-delimited JSON over `~/.local/run/roster.sock`
- **Scheduler** — Kahn's algorithm for topological ordering, resource accounting
- **Executor** — shell command executor; captures stdout/stderr to disk
- **Storage** — SQLite for run/job metadata; log files written directly, no daemon involvement
- **Resource discovery** — sysinfo (CPU/RAM), nvml (per-GPU VRAM)

## Non-goals

- Multi-node / distributed scheduling
- Preemption or job migration
- Python/container executors (v1 scope)
- Retry policy (v1 scope)

## Status

Under active development. API and wire protocol unstable.