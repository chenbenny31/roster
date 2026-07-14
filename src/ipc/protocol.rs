//! Wire protocol for roster IPC
//!
//! Format: one JSON object per line, \n-terminated
//! Max message size: 1MB, enforced by server
//! One request -> one response -> close, except for subscribe

use serde::{Deserialize, Serialize};

use crate::workflow::model::JobState;

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Request {
    Ping,
    Submit { spec_yaml: String }, // client read file and send content
    Ps,
    Status { run_id: String },
    Logs { run_id: String, job_id: String },
    Cancel { run_id: String },
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