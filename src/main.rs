use clap::{Parser, Subcommand};

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
    let cli = Cli::parse();

    match cli.command {
        Command::Daemon => { unimplemented!("daemon") },
        Command::Submit { .. } => { unimplemented!("submit") },
        Command::Ps => { unimplemented!("ps") },
        Command::Status { .. } => { unimplemented!("status") },
        Command::Logs { .. } => { unimplemented!("logs") },
        Command::Cancel { .. } => { unimplemented!("cancel") },
    }

    Ok(())
}