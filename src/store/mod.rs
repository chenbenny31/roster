use std::path::Path;

use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};

use crate::workflow::model::{JobRun, WorkflowRun};

/// persistent store for run and job state
/// SQLite is the CLI read source, in-memory DaemonState is scheduler's working set
pub struct RunStore {
    pool: SqlitePool,
}

/// Flatten run row returned from SELECT queries
#[derive(Debug)]
pub struct RunRow {
    pub run_id: String,
    pub workflow_name: String,
    pub status: String,
    pub created_at: String,
}

/// Flatten job row return from SELECT queries
#[derive(Debug)]
pub struct JobRow {
    pub run_id: String,
    pub job_id: String,
    pub state: String,
    pub exit_code: Option<i64>,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
    pub log_path: Option<String>,
}

impl RunStore {
    /// Open or create SQLite database and apply schema migrations
    pub async fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true);

        let pool = SqlitePool::connect_with(options).await?;

        // schema migrations, idempotent, safe to run on every startup
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY NOT NULL,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS jobs (
                run_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                status TEXT NOT NULL,
                exit_code INTEGER,
                started_at TEXT,
                ended_at TEXT,
                log_path TEXT,
                PRIMARY KEY (run_id, job_id)
            )",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    /// Write or overwrite the run's current derived status to SQLite
    pub async fn upsert_run(&self, run: &WorkflowRun) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO runs (run_id, workflow_name, status, created_at)
             VALUES (?, ?, ?, ?)",
        )
        .bind(&run.run_id)
        .bind(&run.workflow_name)
        .bind(run.status().as_str())
        .bind(run.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Write or overwrite a job's current state to SQLite
    pub async fn upsert_job(&self, run_id: &str, job: &JobRun) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO jobs
             (run_id, job_id, state, exit_code, started_at, ended_at, log_path)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(&run_id)
        .bind(&job.job_id)
        .bind(job.state.as_str())
        .bind(job.exit_code.map(|c| c as i64)) .bind(job.started_at.as_ref().map(|t| t.to_rfc3339()))
        .bind(job.ended_at.as_ref().map(|t| t.to_rfc3339()))
        .bind(job.log_path.as_ref().map(|p| p.to_string_lossy().into_owned()))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Mark Running jobs as Interrupted and filp to Failed, on daemon startup
    pub async fn reconcile_interrupted(&self) -> anyhow::Result<()> {
        let mut tx = self.pool.begin().await?;

        sqlx::query(
            "UPDATE runs SET state = 'Interrupted' WHERE state = 'Running'",
        )
        .execute(&mut *tx)
        .await?;

        sqlx::query(
            "UPDATE runs SET status = 'Failed'
             WHERE status IN ('Pending', 'Running')
             AND run_id IN (
                 SELECT DISTINCT run_id FROM jobs WHERE state = 'Interrupted'
            )",
        )
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
        Ok(())
    }

    /// List all runs ordered by creation time descending
    pub async fn list_runs(&self) -> anyhow::Result<Vec<RunRow>> {
        let rows = sqlx::query(
            "SELECT run_id, worflow_name, status, created_at
             FROM runs ORDER BY created_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|row| RunRow {
            run_id: row.get("run_id"),
            workflow_name: row.get("workflow_name"),
            status: row.get("status"),
            created_at: row.get("created_at"),
        }).collect())
    }

    /// Get a single run by ID, return None if not found
    pub async fn get_run(&self, run_id: &str) -> anyhow::Result<Option<RunRow>> {
        let row = sqlx::query(
            "SELECT run_id, workflow_name, status, created_at
             FROM runs WHERE run_id = ?",
        )
        .bind(run_id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row.map(|row| RunRow {
            run_id: row.get("run_id"),
            workflow_name: row.get("workflow_name"),
            status: row.get("status"),
            created_at: row.get("created_at"),
        }))
    }

    /// List all jobs for a run ordered by job_id
    pub async fn list_jobs(&self, run_id: &str) -> anyhow::Result<Vec<JobRow>> {
        let rows = sqlx::query(
            "SELECT run_id, workflow_name, status, created_at
             FROM runs WHERE run_id = ?",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|row| JobRow {
            run_id: row.get("run_id"),
            job_id: row.get("job_id"),
            state: row.get("state"),
            exit_code: row.get("exit_code"),
            started_at: row.get("started_at"),
            ended_at: row.get("ended_at"),
            log_path: row.get("log_path"),
        }).collect())
    }
}