use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;

use crate::ipc::protocol::{Request, Response};
use crate::paths::socket_path;

/// Send one request to the local daemon and return the response
pub async fn send(request: Request) -> anyhow::Result<Response> {
    let stream = UnixStream::connect(socket_path()).await?;
    exchange(stream, request).await
}

/// Write one request on a connected stream and read one response
/// callers that connect themselves can tell a failed connect (nothing sent) from a lost reply
/// Connection is closed on drop after response is read
pub async fn exchange(stream: UnixStream, request: Request) -> anyhow::Result<Response> {
    // tokio::io::split(stream) consumes the stream with no lifetime tied
    let (read_half, write_half) = tokio::io::split(stream);
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
