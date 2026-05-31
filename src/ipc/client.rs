use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;

use crate::ipc::protocol::{Request, Response};
use crate::paths::socket_path;

/// Send one request to the daemon and return the response
/// Connection is closed on drop after response is read
pub async fn send(request: Request) -> anyhow::Result<Response> {
    let socket = UnixStream::connect(socket_path()).await?;

    // tokio::io::split(socket) consumes the socket with no lifetime tied
    let (read_half, write_half) = tokio::io::split(socket);
    let mut reader = BufReader::new(read_half);
    let mut writer = BufWriter::new(write_half);

    // send request
    let mut line = serde_json::to_string(&request)?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await?;
    writer.flush().await?;

    // read response
    let mut response_line = String::new();
    reader.read_line(&mut response_line).await?;
    let response = serde_json::from_str(&response_line.trim())?;

    Ok(response)
}