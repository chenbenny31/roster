use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use roster::resource::discovery;
use roster::resource::pool::ResourcePool;

#[derive(Parser)]
#[command(name = "roster", about = "GPU-aware single-node workflow scheduler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Start the roster daemon
    Daemon,
    /// Submit a workflow YAML
    Submit {
        #[arg(help = "path to workflow YAML")]
        file: String,
    },
    /// List active workflow runs
    Ps,
    /// Show status of a run
    Status {
        #[arg(help = "run id")]
        id: String,
    },
    /// Tail logs for a job
    Logs {
        #[arg(help = "job id")]
        id: String,
    },
    /// Cancel a run
    Cancel {
        #[arg(help = "run id")]
        id: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();

    match cli.command {
        Command::Daemon => {
            let resources = discovery::discover();
            let pool = ResourcePool::new(resources);
            let state = roster::daemon::DaemonState::new(pool);
            roster::daemon::run(state).await?;
        },
        Command::Submit { file } => {
            let spec_yaml = tokio::fs::read_to_string(&file).await
                .map_err(|error| anyhow::anyhow!("failed to read {}: {}", file, error))?;
            let response = roster::ipc::client::send(
                roster::ipc::protocol::Request::Submit { spec_yaml }
            ).await?;
            println!("{:?}", response);
        },
        Command::Ps => {
            let response = roster::ipc::client::send(
                roster::ipc::protocol::Request::Ps
            ).await?;
            println!("{:?}", response);
        },
        Command::Status { .. } => { unimplemented!("status") },
        Command::Logs { .. } => { unimplemented!("logs") },
        Command::Cancel { .. } => { unimplemented!("cancel") },
    }

    Ok(())
}