//! Wire protocol for roster IPC
//!
//! Format: one JSON object per line, \n-terminated
//! Max message size: 1MB, enforced by server
//! One request -> one response -> close, except for subscribe

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::resource::discovery::SystemResources;
use crate::resource::health::GpuHealth;
use crate::resource::pool::Allocation;
use crate::workflow::model::JobState;
use crate::workflow::spec::JobSpec;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Request {
    Ping,
    Submit { spec_yaml: String }, // client read file and send content
    Ps,
    Status { run_id: String },
    Logs { run_id: String, job_id: String },
    Cancel { run_id: String },
    Info, // this machine's daemon, for `roster info`/`stop`/`node add` and the tunnel health check
    Nodes, // the nodes this daemon places jobs on, for `roster node ls` and `roster info NODE`
    ClearGpu { node: Option<String>, gpu: u32 }, // `roster gpu clear`; a named node gets it with node None
    GpuQualified { index: u32, passed: bool, reason: String }, // from `roster qualify` to its own daemon

    // worker operations, sent by a controller through the node's tunnel
    // an execution is named by (run_id, job_id); one per job for now
    Launch { run_id: String, job: JobSpec, allocation: Allocation },
    Poll   { run_id: String, job_id: String },
    Kill   { run_id: String, job_id: String },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Response {
    Pong,
    Submitted { run_id: String },
    PsResult { runs: Vec<RunSummary>},
    StatusResult { run: RunDetail },
    LogPath { path: String, status: JobState },
    Cancelled { run_id: String },
    Error { message: String },
    Info(NodeInfo),
    Nodes { nodes: Vec<NodeStatus> },
    GpuCleared,
    Recorded,

    Launched { host_pid: u32 },
    Polled   { exit_code: Option<i32> }, // None while running
    Killed,
}

/// What a machine's daemon reports about itself, printed by `roster info`
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeInfo {
    pub version:      String, // crate version, controller and node must match
    pub pid:          u32,    // `roster stop` signals it
    pub socket:       String, // absolute path of the daemon's socket
    pub running_jobs: usize,  // as controller or worker; `roster stop` refuses while nonzero unless they survive it
    #[serde(default)]
    pub jobs_survive_restart: bool, // jobs run in systemd scopes, and this daemon takes them back when it starts
    pub resources:    SystemResources,
    #[serde(default)]
    pub gpu_health:   Vec<GpuHealth>, // live readings, empty without NVML
}

/// One node as this daemon sees it, printed by `roster node ls` and `roster info NODE`
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeStatus {
    pub name:           String,
    pub ssh:            Option<String>, // None = this machine
    pub online:         bool,
    pub state:          String,         // node::NodeState as text, e.g. "ssh tunnel down: ..."
    pub since:          DateTime<Utc>,  // when the node entered that state
    pub running_jobs:   usize,          // jobs placed there; held while the node is offline
    pub total:          SystemResources,
    pub free_cpu:       u32,
    pub free_memory_mb: u64,
    pub free_vram_mb:   Vec<u64>,       // per GPU, same order as total.gpus
    pub gpu_health:     Vec<GpuHealth>, // from the node's last health check; empty without NVML
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub struct RunSummary {
    pub run_id:        String,
    pub workflow_name: String,
    pub status:        String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub struct JobDetail {
    pub job_id:     String,
    pub state:      String,
    pub exit_code:  Option<i64>,
    pub started_at: Option<String>,
    pub ended_at:   Option<String>,
    pub log_path:   Option<String>,
}


#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(default)]
pub struct RunDetail {
    pub run_id:        String,
    pub workflow_name: String,
    pub status:        String,
    pub created_at:    String,
    pub jobs:          Vec<JobDetail>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip_request(res: Request) -> Request {
        let line = serde_json::to_string(&res).unwrap();
        serde_json::from_str(&line).unwrap()
    }

    fn round_trip_response(res: Response) -> Response {
        let line = serde_json::to_string(&res).unwrap();
        serde_json::from_str(&line).unwrap()
    }

    #[test]
    fn ping_pong_round_trip() {
        assert!(matches!(round_trip_request(Request::Ping), Request::Ping));
        assert!(matches!(round_trip_response(Response::Pong), Response::Pong));
    }

    #[test]
    fn submit_round_trip() {
        let req = Request::Submit { spec_yaml: "name: test\njobs: []".into() };
        let Request::Submit { spec_yaml } = round_trip_request(req) else { panic!() };
        assert_eq!(spec_yaml, "name: test\njobs: []");
    }

    #[test]
    fn worker_messages_round_trip() {
        let launch = Request::Launch {
            run_id: "r1".into(),
            job: JobSpec { id: "train".into(), command: "true".into(), ..Default::default() },
            allocation: Allocation { cpu: 1, memory_mb: 512, gpus: vec![] },
        };
        let Request::Launch { job, .. } = round_trip_request(launch) else { panic!() };
        assert_eq!(job.id, "train");

        let info = Response::Info(NodeInfo {
            version: "0.1.0".into(),
            pid: 4242,
            socket: "/run/roster.socket".into(),
            running_jobs: 2,
            resources: SystemResources { cpu_cores: 4, memory_mb: 8_000, gpus: vec![] },
            gpu_health: vec![], jobs_survive_restart: false,
        });
        let Response::Info(node) = round_trip_response(info) else { panic!() };
        assert_eq!(node.running_jobs, 2);
        assert_eq!(node.resources.cpu_cores, 4);
    }

    #[test]
    fn ps_result_empty_round_trip() {
        let res = Response::PsResult { runs: vec![] };
        let Response::PsResult { runs } = round_trip_response(res) else { panic!() };
        assert!(runs.is_empty());
    }

    #[test]
    fn log_path_round_trip() {
        let home = std::env::var("HOME").unwrap();
        let path = format!("{}/.local/share/roster/runs/abc/preprocess.log", home);
        let res = Response::LogPath {
            path: path.clone(),
            status: JobState::Running,
        };
        let Response::LogPath { path: got, status } = round_trip_response(res) else { panic!() };
        assert!(got.contains("/preprocess.log"));
        assert!(matches!(status, JobState::Running));
    }

    #[test]
    fn error_round_trip() {
        let res = Response::Error { message: "malformed request".into() };
        let Response::Error { message } = round_trip_response(res) else { panic!() };
        assert_eq!(message, "malformed request");
    }

    #[test]
    fn run_summary_missing_field_defaults_instead_of_failing() {
        let json = r#"{"run_id": "abc-123", "workflow_name": "train"}"#;
        let summary: RunDetail = serde_json::from_str(json).unwrap();
        assert_eq!(summary.run_id, "abc-123");
        assert_eq!(summary.workflow_name, "train");
        assert_eq!(summary.status, "");
    }

    #[test]
    fn job_default_missing_optional_fields_defaults_to_none() {
        let json = r#"{"job_id": "train", "state": "Running"}"#;
        let detail: JobDetail = serde_json::from_str(json).unwrap();
        assert_eq!(detail.job_id, "train");
        assert_eq!(detail.state, "Running");
        assert_eq!(detail.exit_code, None);
        assert_eq!(detail.log_path, None);
    }

    #[test]
    fn run_default_unknown_extra_field_is_ignored_not_rejected() {
        let json = r#"{
            "run_id": "abc-123",
            "workflow_name": "train",
            "status": "Running",
            "created_at": "2026-07-12T00:00:00Z",
            "jobs": [],
            "future_field_this_struct_does_not_know_about": 42
        }"#;
        let detail: RunDetail = serde_json::from_str(json).unwrap();
        assert_eq!(detail.run_id, "abc-123");
        assert_eq!(detail.status, "Running");
    }
}