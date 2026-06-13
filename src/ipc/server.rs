use std::sync::Arc; // shared ownership across tasks
use std::time::Duration; // for read timeout

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader, BufWriter}; // read_line, write_all, flush
use tokio::net::UnixStream; // UNIX socket type
use tokio::time::timeout;

use crate::daemon::DaemonState;
use crate::ipc::protocol::{Request, Response};
use crate::workflow::dag;
use crate::workflow::model::WorkflowRun;
use crate::workflow::spec;
use crate::ipc::protocol::RunSummary;

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
    let run = WorkflowRun::new(run_id.clone(), spec);

    state.runs.lock().await.insert(run_id.clone(), run);

    Response::Submitted { run_id }
}

/// Dispatch a parsed request to appropriate handler
async fn dispatch(request: Request, state: Arc<DaemonState>) -> Response {
    match request {
        Request::Ping => Response::Pong,
        Request::Submit { spec_yaml } => handle_submit(spec_yaml, state).await,
        Request::Ps => {
            let runs = state.runs.lock().await;
            let summaries = runs.values()
                .map(|run| RunSummary {
                    run_id: run.run_id.clone(),
                    workflow_name: run.workflow_name.clone(),
                    status: run.status(),
                })
                .collect();
            Response::PsResult { runs: summaries }
        },
        Request::Status { .. } => Response::Error { message: "not implemented".into() },
        Request::Logs { .. } => Response::Error { message: "not implemented".into() },
        Request::Cancel { .. } => Response::Error { message: "not implemented".into() },
    }
}