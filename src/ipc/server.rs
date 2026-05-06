use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter}; // read_line, write_all, flush
use tokio::net::{UnixListener, UnixStream}; // UNIX socket type
use tokio::time::timeout;

use crate::daemon::DaemonState;
use crate::ipc::protocol::{Request, Response};
use crate::ipc::socket_path;

/// Bind the UNIX socket and accept connections in a loop
/// Accept errors: logged and skipped, daemon never exits on accept failure
pub async fn listen(state: Arc<DaemonState>) -> anyhow::Result<()> {
    let path = socket_path();

    if let Some(parent) = path.parent() { // Parent dir must exit
        tokio::fs::create_dir_all(parent).await?;
    }

    let listener = UnixListener::bind(&path)?;
    tracing::info!(?path, "roster daemon listening");

    loop {
        match listener.accept().await {
            Ok((socket, _addr)) => {
                let state = state.clone();
                tokio::spawn(handle_connection(socket, state));
            }
            Err(error) => {
                tracing::error!(?error, "accept failed");
            }
        }
    }
}

/// Handle one connection: read one request, dispatch, write one response, close
async fn handle_connection(mut socket: UnixStream, state: Arc<DaemonState>) {
    // socket.split() borrows the socket, tied to its lifetime
    let (read_half, write_half) = socket.split();
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    let response = match read_request(&mut reader).await {
        Ok(request) => dispatch(request, state).await,
        Err(message) => Response::Error { message },
    };

    write_response(&mut writer, response).await;
}

/// Read one JSON line (\n-termination) with a 5-sec timeout
async fn read_request<R>(reader: &mut BufReader<R>) -> Result<Request, String>
where R: AsyncReadExt + Unpin {
    let mut line = String::new();

    let result = timeout(Duration::from_secs(5), reader.read_line(&mut line)).await;

    match result {
        Err(_elapsed) => Err("read timeout".into()),
        Ok(Err(io_error)) => Err(format!("io error: {}", io_error)),
        Ok(Ok(0)) => Err("client disconnected".into()),
        Ok(Ok(_bytes_read)) => serde_json::from_str(line.trim())
            .map_err(|e| format!("json deserialization error: {}", e)),
    }
}

/// Write one line of JSON (\n-termination) response and flush
async fn write_response<W>(writer: &mut BufWriter<W>, response: Response)
where W: AsyncWriteExt + Unpin {
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

/// Dispatch a parsed request to corresponding handler, returns stubs until gate lands
async fn dispatch(request: Request, _state: Arc<DaemonState>) -> Response {
    match request {
        Request::Ping => Response::Pong,
        Request::Submit { .. } => Response::Error { message: "not implemented".into() },
        Request::Ps => Response::Error { message: "not implemented".into() },
        Request::Status { .. }=> Response::Error { message: "not implemented".into() },
        Request::Logs { .. } => Response::Error { message: "not implemented".into() },
        Request::Cancel { .. } => Response::Error { message: "not implemented".into() },
    }
}