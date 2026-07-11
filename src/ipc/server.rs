use std::sync::Arc; // shared ownership across tasks
use std::time::Duration; // for read timeout
use std::sync::atomic::Ordering;

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader, BufWriter}; // read_line, write_all, flush
use tokio::net::UnixStream; // UNIX socket type
use tokio::time::timeout;

use crate::daemon::DaemonState;
use crate::executor::JobHandle;
use crate::ipc::protocol::{JobDetail, RunDetail, RunSummary, Request, Response};
use crate::workflow::dag;
use crate::workflow::model::{JobState, WorkflowRun};
use crate::workflow::spec;

/// Handle one connection: read one req, dispatch, write one res, close
pub async fn handle_connection(mut socket: UnixStream, state: Arc<DaemonState>) {
    let (read_half, write_half) = socket.split();
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    let response = match read_request(&mut reader).await {
        Ok(request) => dispatch(request, state).await,
        Err(message) => Response::Error { message },
    };

    write_response(&mut writer, response).await;
}

/// Read one \n-sep JSON line with 5sec timeout
async fn read_request<R>(reader: &mut BufReader<R>) -> Result<Request, String>
where
    R: AsyncRead + Unpin,
{
    let mut line = String::new();
    let result = timeout(Duration::from_secs(5), reader.read_line(&mut line)).await;

    match result {
        Err(_elapsed) => Err("read timeout".into()),
        Ok(Err(io_error)) => Err(format!("io error: {}", io_error)),
        Ok(Ok(0)) => Err("client disconnected".into()),
        Ok(Ok(_)) => serde_json::from_str(line.trim())
            .map_err(|e| format!("deserialization error: {}", e)),
    }
}

/// Write one \n-sep JSON response and flush
async fn write_response<W>(writer: &mut BufWriter<W>, response: Response)
where
    W: AsyncWriteExt + Unpin,
{
    let Ok(mut line) = serde_json::to_string(&response) else { return };
    line.push('\n');

    if let Err(error) = writer.write_all(line.as_bytes()).await {
        tracing::error!(?error, "write failed");
        return;
    }
    if let Err(error) = writer.flush().await {
        tracing::error!(?error, "flush failed");
    }
}

async fn handle_submit(spec_yaml: String, state: Arc<DaemonState>) -> Response {
    let spec = match spec::parse(&spec_yaml) {
        Ok(spec) => spec,
        Err(error) => return Response::Error { message: format!("yaml parse error: {}", error) },
    };


    if let Err(error) = dag::validate(&spec) {
        return Response::Error { message: format!("dag validate error: {}", error) };
    }

    // Fail fast if any job requires GPUs but none are available
    {
        let pool = state.pool.lock().await;
        for job in &spec.jobs {
            if job.resources.gpu > 0 && pool.total.gpus.is_empty() {
                return Response::Error {
                    message: format!(
                        "job '{}' requires {} GPU(s) but this node has none",
                        job.id, job.resources.gpu
                    ),
                };
            }
        }
    } // lock released here before WorkflowRun created

    let run_id = uuid::Uuid::new_v4().to_string();
    let run_seq = state.run_counter.fetch_add(1, Ordering::Relaxed);
    let job_seq_start = state.job_counter.fetch_add(spec.jobs.len() as u64, Ordering::Relaxed);
    let run = WorkflowRun::new(run_id.clone(), run_seq, spec, job_seq_start);

    state.runs.lock().await.insert(run_id.clone(), run);

    Response::Submitted { run_id }
}

/// Dispatch a parsed request to appropriate handler
async fn dispatch(request: Request, state: Arc<DaemonState>) -> Response {
    match request {
        Request::Ping => Response::Pong,
        Request::Submit { spec_yaml } => handle_submit(spec_yaml, state).await,
        Request::Ps => {
            match state.store.list_runs().await {
                Err(error) => Response::Error { message: format!("db error: {}", error) },
                Ok(rows) => Response::PsResult {
                    runs: rows.into_iter().map(|row| RunSummary {
                        run_id: row.run_id,
                        workflow_name: row.workflow_name,
                        status: row.status,
                    }).collect(),
                },
            }
        },
        Request::Status { run_id } => {
            let run_row = match state.store.get_run(&run_id).await {
                Err(error) => return Response::Error { message: format!("db error: {}", error) },
                Ok(None) => return Response::Error { message: format!("run '{}' not found", run_id) },
                Ok(Some(r)) => r,
            };
            let job_rows = match state.store.list_jobs(&run_id).await {
                Err(error) => return Response::Error { message: format!("db error: {}", error) },
                Ok(rows) => rows,
            };
            Response::StatusResult {
                run: RunDetail {
                    run_id: run_row.run_id,
                    workflow_name: run_row.workflow_name,
                    status: run_row.status,
                    created_at: run_row.created_at,
                    jobs: job_rows.into_iter().map(|j| JobDetail {
                        job_id: j.job_id,
                        state: j.state,
                        exit_code: j.exit_code,
                        started_at: j.started_at,
                        ended_at: j.ended_at,
                        log_path: j.log_path,
                    }).collect(),
                },
            }
        },
        Request::Logs { run_id, job_id } => {
            let job_rows = match state.store.list_jobs(&run_id).await {
                Err(error) => return Response::Error { message: format!("db error: {}", error) },
                Ok(rows) => rows,
            };
            match job_rows.into_iter().find(|j| j.job_id == job_id) {
                None => Response::Error { message: format!("run '{}' not found in run '{}'", job_id, run_id) },
                Some(job) => match job.log_path {
                    None => Response::Error { message: "log not available yet".into() },
                    Some(path) => Response::LogPath {
                        path,
                        status: serde_json::from_str(&format!("\"{}\"", job.state))
                            .unwrap_or(crate::workflow::model::JobState::Pending),
                    },
                },
            }
        },
        Request::Cancel { run_id } => {
            // collect Running jobs's handles under lock
            let handles: Vec<(String, JobHandle)> = {
                let mut runs = state.runs.lock().await;
                match runs.get_mut(&run_id) {
                    None => return Response::Error { message: format!("run '{}' not found or not active", run_id) },
                    Some(run) => {
                        run.jobs.values_mut()
                            .filter(|job| matches!(job.state, JobState::Running))
                            .filter_map(|job| {
                                job.cancelling = true;
                                job.handle.clone().map(|handle| (job.job_id.clone(), handle))
                            })
                            .collect()
                    }
                }
            };

            // cancel each running job outside the lock
            for (_job_id, handle) in handles {
                let _ = state.executor.cancel(&handle).await;
            }

            Response::Cancelled { run_id }
        },
    }
}