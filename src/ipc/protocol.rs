//! Wire protocol for roster IPC
//!
//! Format: one JSON object per line, \n-terminated
//! Max message size: 1MB, enforced by server
//! One request -> one response -> close, except for subscribe

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Request {
    Ping,
    Submit { file: String },
    Ps,
    Status { run_id: String },
    Logs { job_id: String },
    Cancel { run_id: String },
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Response {
    Pong,
    Submitted { run_id: String },
    PsResult { runs: Vec<RunSummary>},
    StatusResult { run: String },
    LogPath { path: String, status: JobState },
    Cancelled { run_id: String },
    Error { message: String },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RunSummary {
    pub run_id: String,
    pub workflow_name: String,
    pub status: RunState,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RunDetail {
    pub run_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RunState {
    Pending,
    Running,
    Succeeded,
    Failed,
    Canceled,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum JobState {
    Pending,
    Running,
    Succeeded,
    Failed,
    Canceled,
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
        let req = Request::Submit { file: "/tmp/w.yaml".into() };
        let Request::Submit { file } = round_trip_request(req) else { panic!() };
        assert_eq!(file, "/tmp/w.yaml");
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
}