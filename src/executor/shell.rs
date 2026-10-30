use std::fs;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use async_trait::async_trait;
use nix::sys::signal::{killpg, Signal};
use nix::sys::wait::{waitpid, WaitPidFlag, WaitStatus};
use nix::unistd::Pid;
use nix::errno::Errno;
use tokio::time::{sleep, Duration};

use crate::executor::{Executor, ExecutorError, PollResult, JobHandle};
use crate::workflow::model::JobInstance;
use crate::paths::job_log_path;
use crate::resource::pool::Allocation;

pub struct ShellExecutor;

#[async_trait]
impl Executor for ShellExecutor {
    /// Spawn `sh -c <command>` with stdout + stderr -> log file
    /// child gets its own process groups via setpgid(0,0), killpg cancels all descendants
    async fn launch(&self, run_id: &str, job: &JobInstance, placement: &Allocation)
            -> Result<JobHandle, ExecutorError> {
        // a launch whose reply was lost, or one from before this daemon restarted, is taken over
        if let Some(handle) = ShellExecutor::find(run_id, &job.job_id).await {
            return Ok(handle);
        }
        let log_path = job_log_path(run_id, &job.job_id);

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent).map_err(ExecutorError::LogSetupFailed)?;
        }

        let log_file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&log_path)
            .map_err(ExecutorError::LogSetupFailed)?;

        let stderr_file = log_file.try_clone().map_err(ExecutorError::LogSetupFailed)?;

        let gpu_indices: String = placement.gpus
            .iter()
            .map(|gpu_alloc| gpu_alloc.index.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // each job gets a scope, a cgroup that holds everything it starts, inside roster.slice, named after
        // the job so a restarted daemon finds it again; systemd-run execs a wrapper that runs the job and writes
        // its exit code next to the log, since once this daemon is gone nobody else would keep it
        let scope = scopes_available().then(|| unit_name(run_id, &job.job_id));
        let exit_file = scope.as_ref().map(|_| log_path.with_extension("exit"));
        let mut command = match (&scope, &exit_file) {
            (Some(unit), Some(exit_file)) => {
                let mut command = Command::new("systemd-run");
                command.args(["--user", "--scope", "--quiet", "--collect", "--slice", SLICE, "--unit", unit]);
                command.args(["-p", "TimeoutStopSec=5", "sh", "-c", EXIT_WRAPPER, "roster", &job.spec.command]);
                command.arg(exit_file);
                command
            }
            _ => {
                let mut command = Command::new("sh");
                command.arg("-c").arg(&job.spec.command);
                command
            }
        };
        command
            .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID") // number GPUs as NVML does, so the indices name the booked cards
            .env("CUDA_VISIBLE_DEVICES", &gpu_indices)
            .env("ROSTER_GPUS", placement.gpus.len().to_string()) // how many cards it got, e.g. for --tensor-parallel-size
            .env("ROSTER_BIN", std::env::current_exe().unwrap_or_default()) // this roster, e.g. for `roster qualify`
            .stdout(log_file)
            .stderr(stderr_file);

        // setpgid(0, 0) runs in child after fork before exec
        // give the child its own process group (pgid == pid), cancel use killpg(pgid, sig) group
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }

        let child = command.spawn().map_err(ExecutorError::SpawnFailed)?;
        let host_pid = child.id();

        // Drop child handle - process continues running independently
        // poll() via waitpid reaps the zombie when it exits
        drop(child);

        tracing::info!(host_pid, job_id = %job.job_id, cuda_visible_devices = %gpu_indices, ?scope, "job launched");
        Ok(JobHandle::process(host_pid, scope, exit_file))
    }

    async fn poll(&self, handle: &JobHandle) -> Result<PollResult, ExecutorError> {
        match (handle.scope(), handle.exit_file()) {
            (Some(unit), Some(exit_file)) => Ok(poll_scope(handle.host_pid, unit, exit_file).await),
            _ => poll_child(handle.host_pid),
        }
    }

    /// A scope: systemd sends SIGTERM to everything in it and SIGKILL after 5 s;
    /// without one: SIGTERM to the process group, SIGKILL after 5 s, pgid == pid
    async fn cancel(&self, handle: &JobHandle) -> Result<(), ExecutorError> {
        if let Some(unit) = handle.scope() {
            stop_scope(unit).await;
            return Ok(());
        }
        let pgid = Pid::from_raw(handle.host_pid as i32);

        killpg(pgid, Signal::SIGTERM)
            .map_err(|error| ExecutorError::KillFailed(error.into()))?;

        sleep(Duration::from_secs(5)).await;

        // SIGKILL, force kill if still alive, ignore error
        let _ = killpg(pgid, Signal::SIGKILL);
        Ok(())
    }
}

impl ShellExecutor {
    /// A job started earlier, e.g. before this daemon restarted: its scope while it runs, or its exit file
    /// once it ended; None when neither exists, or without systemd
    pub async fn find(run_id: &str, job_id: &str) -> Option<JobHandle> {
        if !scopes_available() {
            return None;
        }
        let unit = unit_name(run_id, job_id);
        let exit_file = job_log_path(run_id, job_id).with_extension("exit");
        let pid = scope_pids(&unit).await.into_iter().min(); // the oldest, normally the wrapper
        if pid.is_none() && !exit_file.exists() {
            return None;
        }
        Some(JobHandle::process(pid.unwrap_or(0), Some(unit), Some(exit_file)))
    }
}

/// Runs the job, then writes its exit code through a temp file, so a reader never sees half of it
const EXIT_WRAPPER: &str = r#"sh -c "$1"; code=$?; echo $code > "$2.tmp" && mv "$2.tmp" "$2"; exit $code"#;

/// Running while our own child is alive: systemd-run still setting the scope up, then the wrapper;
/// done once the wrapper wrote the exit code, and stopping the scope then ends whatever the job left behind;
/// after a restart the job is no child of ours, so running while any process of its scope is alive;
/// otherwise ended without a code (a cancel, a kill of the wrapper, or a reboot)
async fn poll_scope(host_pid: u32, unit: &str, exit_file: &Path) -> PollResult {
    if host_pid > 0 && matches!(waitpid(Pid::from_raw(host_pid as i32), Some(WaitPidFlag::WNOHANG)), Ok(WaitStatus::StillAlive)) {
        return PollResult::Running;
    }
    if let Some(exit_code) = read_exit_code(exit_file) {
        stop_scope(unit).await;
        return PollResult::Exited { exit_code };
    }
    if in_scope(host_pid, unit) || !scope_pids(unit).await.is_empty() {
        return PollResult::Running;
    }
    PollResult::Exited { exit_code: -1 }
}

/// The code the wrapper wrote; 128 + N, how a shell reports death by signal N, becomes -1 as before
fn read_exit_code(exit_file: &Path) -> Option<i32> {
    let code: i32 = fs::read_to_string(exit_file).ok()?.trim().parse().ok()?;
    Some(if (129..=192).contains(&code) { -1 } else { code })
}

/// Cheap while the job runs: the process still lives in the job's scope
fn in_scope(pid: u32, unit: &str) -> bool {
    let scope = format!("/{}.scope", unit);
    pid > 0 && fs::read_to_string(format!("/proc/{}/cgroup", pid)).is_ok_and(|text| text.contains(&scope))
}

/// The processes in a job's scope; empty once the scope is gone
async fn scope_pids(unit: &str) -> Vec<u32> {
    let output = tokio::process::Command::new("systemctl")
        .args(["--user", "show", "-p", "ControlGroup", "--value", &format!("{}.scope", unit)])
        .output()
        .await;
    let Ok(output) = output else { return Vec::new() };
    let group = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if group.is_empty() {
        return Vec::new();
    }
    fs::read_to_string(format!("/sys/fs/cgroup{}/cgroup.procs", group))
        .unwrap_or_default()
        .lines()
        .filter_map(|pid| pid.parse().ok())
        .collect()
}

/// Without systemd: non-blocking check of our child via waitpid(WNOHANG)
fn poll_child(host_pid: u32) -> Result<PollResult, ExecutorError> {
    match waitpid(Pid::from_raw(host_pid as i32), Some(WaitPidFlag::WNOHANG)) {
        Ok(WaitStatus::StillAlive) => Ok(PollResult::Running),
        Ok(WaitStatus::Exited(_, exit_code)) => Ok(PollResult::Exited { exit_code }),
        Ok(WaitStatus::Signaled(_, _, _)) => Ok(PollResult::Exited { exit_code: -1 }),
        Ok(_) => Ok(PollResult::Running),
        Err(Errno::ECHILD) => Ok(PollResult::Exited { exit_code: 0 }), // already reaped by tokio
        Err(nix_error) => Err(ExecutorError::WaitFailed(nix_error.into())),
    }
}

/// The same name for the same job on every call, so a restarted daemon finds it again;
/// bytes a unit name can't hold become _XX, which keeps different ids apart
fn unit_name(run_id: &str, job_id: &str) -> String {
    let escape = |id: &str| id.bytes().map(|byte| {
        if byte.is_ascii_alphanumeric() || byte == b'-' { (byte as char).to_string() } else { format!("_{:02x}", byte) }
    }).collect::<String>();
    format!("roster-{}-{}", escape(run_id), escape(job_id))
}

/// Parent of every job's scope; its memory cap keeps all jobs together within what the pool books
const SLICE: &str = "roster.slice";

/// Cap the RAM all jobs on this machine may use together, so the system and the daemon keep the rest;
/// no swap either, since Fedora's zram swap would take that RAM anyway; no per-job caps
pub fn limit_jobs(memory_mb: u64) {
    if !scopes_available() {
        return;
    }
    let memory = format!("MemoryMax={}M", memory_mb);
    match Command::new("systemctl")
        .args(["--user", "set-property", "--runtime", SLICE, &memory, "MemorySwapMax=0"])
        .status()
    {
        Ok(status) if status.success() => tracing::info!(memory_mb, "jobs limited together in {}", SLICE),
        other => tracing::warn!(?other, "could not limit {}, jobs run without a memory cap", SLICE),
    }
}

/// Whether jobs can run in systemd scopes, which outlive this daemon; probed once by running `true` in one
pub fn scopes_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let available = Command::new("systemd-run")
            .args(["--user", "--scope", "--quiet", "true"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if !available {
            tracing::warn!("no systemd user manager: jobs run without cgroups");
        }
        available
    })
}

/// Stop a job's scope, killing whatever it left running (SIGTERM, SIGKILL after 5 s);
/// returns once the scope is gone, so its resources can be reused; a scope already gone is fine
async fn stop_scope(unit: &str) {
    let _ = tokio::process::Command::new("systemctl")
        .args(["--user", "stop", &format!("{}.scope", unit)])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await;
}

// tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::spec::JobSpec;
    use crate::resource::pool::GpuAllocation;

    fn make_job(id: &str, command: &str) -> JobInstance {
        JobInstance::new(
            JobSpec {
                id:         id.into(),
                command:    command.into(),
                ..Default::default()
            },
            0, // job_seq
        )
    }

    fn cpu_only_placement() -> Allocation {
        Allocation { cpu: 1, memory_mb: 512, gpus: vec![] }
    }

    /// A fresh run id per test run: a job's exit file outlives it, and a launch takes over a finished job of the same ids
    fn run_id(name: &str) -> String {
        format!("{}-{}", name, uuid::Uuid::new_v4().simple())
    }

    #[tokio::test]
    async fn launch_and_poll_sleep() {
        let executor = ShellExecutor;
        let job = make_job("sleep_job", "sleep 2");
        let run = run_id("test-run-001");
        let handle = executor.launch(&run, &job, &cpu_only_placement()).await.unwrap();

        assert!(handle.host_pid > 0);

        let result = executor.poll(&handle).await.unwrap();
        assert!(matches!(result, PollResult::Running));

        tokio::time::sleep(Duration::from_secs(4)).await; // wait for finish

        let result = executor.poll(&handle).await.unwrap();
        assert!(matches!(result, PollResult::Exited { .. }));
    }

    #[tokio::test]
    async fn launch_echo_writes_log() {
        let executor = ShellExecutor;
        let job = make_job("echo-job", "echo hello_roster");
        let run = run_id("test-run-002");
        let handle = executor.launch(&run, &job, &cpu_only_placement()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = executor.poll(&handle).await; // reap zombie

        let log_path = job_log_path(&run, "echo-job");
        let contents = fs::read_to_string(&log_path).unwrap();
        assert!(contents.contains("hello_roster"));
    }

    #[tokio::test]
    async fn cancel_kills_process() {
        let executor = ShellExecutor;
        let job = make_job("cancel-job", "sleep 60");
        let run = run_id("test-run-003");
        let handle = executor.launch(&run, &job, &cpu_only_placement()).await.unwrap();

        executor.cancel(&handle).await.unwrap();

        let result = executor.poll(&handle).await.unwrap();
        assert!(matches!(result, PollResult::Exited { .. }));
    }

    #[tokio::test]
    async fn cuda_visible_devices_set_from_placement() {
        let executor = ShellExecutor;
        let job = make_job("gpu-echo-job", "echo $CUDA_VISIBLE_DEVICES");
        let placement = Allocation {
            cpu: 1,
            memory_mb: 512,
            gpus: vec![GpuAllocation { index: 1, vram_mb: 4000 }],
        };
        let run = run_id("test-run-004");
        let handle = executor.launch(&run, &job, &placement).await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = executor.poll(&handle).await.unwrap();

        let log_path = job_log_path(&run, "gpu-echo-job");
        let contents = fs::read_to_string(&log_path).unwrap();
        assert!(contents.contains("1"));
    }

    // the tests below need a systemd user manager; without one they pass trivially

    async fn wait_exit(executor: &ShellExecutor, handle: &JobHandle) -> i32 {
        loop {
            if let PollResult::Exited { exit_code } = executor.poll(handle).await.unwrap() {
                return exit_code;
            }
            sleep(Duration::from_millis(100)).await;
        }
    }

    #[tokio::test]
    async fn the_exit_code_outlives_the_daemon() {
        if !scopes_available() {
            return;
        }
        let executor = ShellExecutor;
        let run = run_id("test-run-exit");
        let handle = executor.launch(&run, &make_job("seven", "sleep 1; exit 7"), &cpu_only_placement()).await.unwrap();
        assert!(handle.scope().is_some());
        assert_eq!(wait_exit(&executor, &handle).await, 7);
        // what a restarted daemon finds: the scope is gone, the exit file is not
        let found = ShellExecutor::find(&run, "seven").await.unwrap();
        assert_eq!(wait_exit(&executor, &found).await, 7);
    }

    #[tokio::test]
    async fn a_restarted_daemon_finds_its_running_job() {
        if !scopes_available() {
            return;
        }
        let executor = ShellExecutor;
        let job = make_job("long", "sleep 30");
        let run = run_id("test-run-find");
        executor.launch(&run, &job, &cpu_only_placement()).await.unwrap();

        // all a restarted daemon has are the run and job ids; the scope exists once systemd-run set it up
        let mut found = None;
        for _ in 0..50 {
            found = ShellExecutor::find(&run, "long").await;
            if found.is_some() {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
        let handle = found.unwrap();
        assert!(matches!(executor.poll(&handle).await.unwrap(), PollResult::Running));
        let again = executor.launch(&run, &job, &cpu_only_placement()).await.unwrap();
        assert_eq!(again.host_pid, handle.host_pid); // taken over, not started twice

        executor.cancel(&handle).await.unwrap();
        assert_eq!(wait_exit(&executor, &handle).await, -1);
        assert!(ShellExecutor::find(&run, "long").await.is_none());
    }

    #[tokio::test]
    async fn exit_stops_what_the_job_left_running() {
        if !scopes_available() {
            return;
        }
        let executor = ShellExecutor;
        let pid_file = std::env::temp_dir().join(format!("roster-{}.pid", uuid::Uuid::new_v4()));
        // setsid leaves the process group, so killpg alone would miss it
        let job = make_job("leaver", &format!("setsid sleep 300 & echo $! > {}", pid_file.display()));
        let run = run_id("test-run-leaver");
        let handle = executor.launch(&run, &job, &cpu_only_placement()).await.unwrap();
        assert!(handle.scope().is_some());
        assert_eq!(wait_exit(&executor, &handle).await, 0);

        let pid: i32 = fs::read_to_string(&pid_file).unwrap().trim().parse().unwrap();
        fs::remove_file(&pid_file).unwrap();
        for _ in 0..20 {
            if nix::sys::signal::kill(Pid::from_raw(pid), None).is_err() {
                return;
            }
            sleep(Duration::from_millis(100)).await;
        }
        panic!("pid {} outlived its job", pid);
    }
}
