use std::sync::Arc; // shared ownership across tasks
use std::time::Duration; // for read timeout
use std::sync::atomic::Ordering;

use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader, BufWriter}; // read_line, write_all, flush
use tokio::net::UnixStream; // UNIX socket type
use tokio::time::timeout;

use crate::daemon::DaemonState;
use crate::executor::JobHandle;
use crate::ipc::protocol::{JobDetail, NodeInfo, NodeStatus, RunDetail, RunSummary, Request, Response};
use crate::paths::socket_path;
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

    // Fail fast if a job could never fit on any node, it would otherwise wait forever
    {
        let pools = state.pools.lock().await;
        for job in &spec.jobs {
            let options = job.resources.gpu_options();
            if !pools.iter().any(|pool| options.iter().any(|option| pool.could_fit(option))) {
                return Response::Error {
                    message: format!("job '{}' does not fit on any node: {:?}", job.id, job.resources),
                };
            }
        }
    } // lock released here before WorkflowRun created

    let run_id = uuid::Uuid::new_v4().to_string();
    let run_seq = state.run_counter.fetch_add(1, Ordering::Relaxed);
    let job_seq_start = state.job_counter.fetch_add(spec.jobs.len() as u64, Ordering::Relaxed);
    let run = WorkflowRun::new(run_id.clone(), run_seq, spec, job_seq_start);

    let _ = state.store.upsert_run(&run).await;
    for job in run.jobs.values() {
        let _ = state.store.upsert_job(&run_id, job, None).await;
    }

    state.runs.lock().await.insert(run_id.clone(), run);

    Response::Submitted { run_id }
}

/// This machine's daemon: identity, inventory, and the jobs it runs as controller or worker
async fn handle_info(state: Arc<DaemonState>) -> Response {
    let controlled = state.runs.lock().await
        .values()
        .flat_map(|run| run.jobs.values())
        .filter(|job| job.state == JobState::Running)
        .count();

    Response::Info(NodeInfo {
        version:      env!("CARGO_PKG_VERSION").into(),
        pid:          std::process::id(),
        socket:       socket_path().display().to_string(),
        running_jobs: controlled + state.worker.running().await,
        jobs_survive_restart: crate::executor::shell::scopes_available(),
        resources:    state.resources.clone(),
        gpu_health:   state.gpu_monitor.read(&state.resources.gpus),
    })
}

/// The nodes this daemon places jobs on: each tunnel's last health check, the jobs placed there, free resources
async fn handle_nodes(state: Arc<DaemonState>) -> Response {
    // GPU readings: this machine's read now, other nodes' from their last health check
    let mut health = Vec::new();
    for node in &state.nodes {
        let node_health = node.health().await;
        let gpu_health = match &node.ssh {
            None => state.gpu_monitor.read(&state.resources.gpus),
            Some(_) => node_health.info.as_ref().map(|info| info.gpu_health.clone()).unwrap_or_default(),
        };
        health.push((node_health, gpu_health));
    }
    let runs = state.runs.lock().await;
    let pools = state.pools.lock().await;

    let nodes = state.nodes.iter().zip(health).enumerate().map(|(index, (node, (health, gpu_health)))| {
        let running_jobs = runs.values()
            .flat_map(|run| run.jobs.values())
            .filter(|job| job.state == JobState::Running)
            .filter(|job| job.executions.iter().any(|execution| execution.node == index))
            .count();
        let pool = &pools[index];
        NodeStatus {
            name:           node.name.clone(),
            ssh:            node.ssh.clone(),
            online:         health.state.is_online(),
            state:          health.state.to_string(),
            since:          health.since,
            running_jobs,
            total:          pool.total.clone(),
            free_cpu:       pool.available_cpu,
            free_memory_mb: pool.available_memory_mb,
            free_vram_mb:   pool.available_vram_mb.clone(),
            gpu_health,
        }
    }).collect();
    Response::Nodes { nodes }
}

/// `roster gpu clear NODE GPU`: a card of this machine, or passed on through the named node's tunnel
async fn handle_clear_gpu(state: Arc<DaemonState>, node: Option<String>, gpu: u32) -> Response {
    if let Some(name) = node {
        let Some(target) = state.nodes.iter().find(|candidate| candidate.name == name) else {
            return Response::Error { message: format!("no node '{}', see `roster node ls`", name) };
        };
        if target.socket.is_some() {
            return target.request(Request::ClearGpu { node: None, gpu }).await
                .unwrap_or_else(|error| Response::Error { message: format!("node '{}': {}", name, error) });
        }
    }
    match state.resources.gpus.iter().find(|card| card.index == gpu) {
        Some(card) => {
            state.gpu_monitor.clear(card);
            Response::GpuCleared
        }
        None => Response::Error { message: format!("no GPU {} on this machine", gpu) },
    }
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
        Request::Info => handle_info(state).await,
        Request::Nodes => handle_nodes(state).await,
        Request::ClearGpu { node, gpu } => handle_clear_gpu(state, node, gpu).await,
        Request::GpuQualified { index, passed, reason } => match state.resources.gpus.iter().find(|card| card.index == index) {
            Some(card) => {
                state.gpu_monitor.record_test(card, passed, reason);
                Response::Recorded
            }
            None => Response::Error { message: format!("no GPU {} on this machine", index) },
        },
        Request::Launch { run_id, job, allocation } => state.worker.launch(run_id, job, allocation).await,
        Request::Poll { run_id, job_id } => state.worker.poll(run_id, job_id).await,
        Request::Kill { run_id, job_id } => state.worker.kill(run_id, job_id).await,
        Request::Cancel { run_id } => {
            // collect Running jobs' (node, handle) pairs under lock
            let handles: Vec<(usize, JobHandle)> = {
                let mut runs = state.runs.lock().await;
                match runs.get_mut(&run_id) {
                    None => return Response::Error { message: format!("run '{}' not found or not active", run_id) },
                    Some(run) => {
                        run.jobs.values_mut()
                            .filter(|job| matches!(job.state, JobState::Running))
                            .flat_map(|job| {
                                job.cancelling = true;
                                job.executions.iter()
                                    .filter_map(|execution| execution.handle.clone().map(|handle| (execution.node, handle)))
                                    .collect::<Vec<_>>()
                            })
                            .collect()
                    }
                }
            };

            // cancel each running execution outside the lock, through its node's executor
            for (node, handle) in handles {
                let _ = state.nodes[node].executor.cancel(&handle).await;
            }

            Response::Cancelled { run_id }
        },
    }
}