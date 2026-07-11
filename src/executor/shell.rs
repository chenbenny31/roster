use std::fs;
use std::os::unix::process::CommandExt;
use std::process::Command;

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

        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(&job.spec.command)
            .env("CUDA_VISIBLE_DEVICES", &gpu_indices)
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

        tracing::info!(host_pid, job_id = %job.job_id, cuda_visible_devices = %gpu_indices, "job launched");
        Ok(JobHandle::process(host_pid))
    }

    /// Non-blocking process check via waitpid(WNOHANG)
    async fn poll(&self, handle: &JobHandle) -> Result<PollResult, ExecutorError> {
        let pid = Pid::from_raw(handle.host_pid as i32);

        match waitpid(pid, Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => Ok(PollResult::Running),
            Ok(WaitStatus::Exited(_, exit_code)) => Ok(PollResult::Exited { exit_code }),
            Ok(WaitStatus::Signaled(_, _, _)) => Ok(PollResult::Exited { exit_code: -1 }),
            Ok(_) => Ok(PollResult::Running),
            Err(Errno::ECHILD) => Ok(PollResult::Exited { exit_code: 0 }), // already reaped by tokio
            Err(nix_error) => Err(ExecutorError::WaitFailed(nix_error.into())),
        }
    }

    /// Send SIGTERM to process group, wait 5s, send SIGKILL, pgid == pid
    async fn cancel(&self, handle: &JobHandle) -> Result<(), ExecutorError> {
        let pgid = Pid::from_raw(handle.host_pid as i32);

        killpg(pgid, Signal::SIGTERM)
            .map_err(|error| ExecutorError::KillFailed(error.into()))?;

        sleep(Duration::from_secs(5)).await;

        // SIGKILL, force kill if still alive, ignore error
        let _ = killpg(pgid, Signal::SIGKILL);

        Ok(())
    }
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

    #[tokio::test]
    async fn launch_and_poll_sleep() {
        let executor = ShellExecutor;
        let job = make_job("sleep_job", "sleep 2");
        let handle = executor.launch("test-run-001", &job, &cpu_only_placement()).await.unwrap();

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
        let handle = executor.launch("test-run-002", &job, &cpu_only_placement()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = executor.poll(&handle).await; // reap zombie

        let log_path = job_log_path("test-run-002", "echo-job");
        let contents = fs::read_to_string(&log_path).unwrap();
        assert!(contents.contains("hello_roster"));
    }

    #[tokio::test]
    async fn cancel_kills_process() {
        let executor = ShellExecutor;
        let job = make_job("cancel-job", "sleep 60");
        let handle = executor.launch("test-run-003", &job, &cpu_only_placement()).await.unwrap();

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
        let handle = executor.launch("test-run-004", &job, &placement).await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = executor.poll(&handle).await.unwrap();

        let log_path = job_log_path("test-run-004", "gpu-echo-job");
        let contents = fs::read_to_string(&log_path).unwrap();
        assert!(contents.contains("1"));
    }
}