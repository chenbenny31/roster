use std::path::Path;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};

use crate::workflow::model::{derive_run_state, JobInstance, JobState, WorkflowRun};

/// Persistent store for run and job state
/// SQLite is the CLI read source, in-memory DaemonState is scheduler's working set
pub struct RunStore {
    pool: SqlitePool,
}

/// Flatten run row returned from SELECT queries, `status` is computed at read time
#[derive(Debug)]
pub struct RunRow {
    pub run_id:        String,
    pub workflow_name: String,
    pub status:        String,
    pub created_at:    String,
}

/// Flatten job row return from SELECT queries
#[derive(Debug)]
pub struct JobRow {
    pub run_id:     String,
    pub job_id:     String,
    pub state:      String,
    pub exit_code:  Option<i64>,
    pub started_at: Option<String>,
    pub ended_at:   Option<String>,
    pub log_path:   Option<String>,
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
                run_id        TEXT PRIMARY KEY NOT NULL,
                workflow_name TEXT NOT NULL,
                created_at    TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS jobs (
                run_id     TEXT NOT NULL,
                job_id     TEXT NOT NULL,
                state      TEXT NOT NULL,
                exit_code  INTEGER,
                started_at TEXT,
                ended_at   TEXT,
                log_path   TEXT,
                PRIMARY KEY (run_id, job_id)
            )",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    /// Write or overwrite the run's identity columns
    pub async fn upsert_run(&self, run: &WorkflowRun) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO runs (run_id, workflow_name, created_at)
             VALUES (?, ?, ?)",
        )
        .bind(&run.run_id)
        .bind(&run.workflow_name)
        .bind(run.created_at.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Write or overwrite a job's current state to SQLite
    pub async fn upsert_job(&self, run_id: &str, job: &JobInstance) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO jobs
             (run_id, job_id, state, exit_code, started_at, ended_at, log_path)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(run_id)
        .bind(&job.job_id)
        .bind(job.state.as_str())
        .bind(job.exit_code.map(|c| c as i64))
        .bind(job.started_at.as_ref().map(|t| t.to_rfc3339()))
        .bind(job.ended_at.as_ref().map(|t| t.to_rfc3339()))
        .bind(job.log_path.as_ref().map(|p| p.to_string_lossy().into_owned()))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// On daemon startup: mark any job still Running (from before) as Interrupted
    pub async fn reconcile_interrupted(&self) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE jobs SET state = 'Interrupted' WHERE state = 'Running'",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// List all runs ordered by creation time descending, with status derived from each run's job
    pub async fn list_runs(&self) -> anyhow::Result<Vec<RunRow>> {
        let rows = sqlx::query(
            "SELECT r.run_id, r.workflow_name, r.created_at, j.state AS job_state
             FROM runs r
             LEFT JOIN jobs j on j.run_id = r.run_id
             ORDER BY r.created_at DESC, r.run_id",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut result: Vec<RunRow> = Vec::new();
        let mut current: Option<RunAccumulator> = None;

        for row in rows {
            let run_id: String = row.get("run_id");
            let workflow_name: String = row.get("workflow_name");
            let created_at: String = row.get("created_at");
            let job_state: Option<String> = row.get("job_state");

            let is_new_run = current.as_ref().map(|acc| acc.run_id != run_id).unwrap_or(true);

            if is_new_run {
                if let Some(finished) = current.take() {
                    result.push(finished.into_run_row());
                }
                current = Some(RunAccumulator { run_id, workflow_name, created_at, job_states: Vec::new() });
            }

            if let Some(state_str) = job_state {
                current.as_mut().unwrap().push_state(&state_str);
            }
        }

        if let Some(finished) = current.take() {
            result.push(finished.into_run_row());
        }

        Ok(result)
    }

    /// Get a single run by ID, return None if not found
    pub async fn get_run(&self, run_id: &str) -> anyhow::Result<Option<RunRow>> {
        let rows = sqlx::query(
            "SELECT r.run_id, r.workflow_name, r.created_at, j.state AS job_state
             FROM runs r
             LEFT JOIN jobs j on j.run_id = r.run_id
             WHERE r.run_id = ?",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            return Ok(None)
        }

        let workflow_name: String = rows[0].get("workflow_name");
        let created_at: String = rows[0].get("created_at");

        let mut accumulator = RunAccumulator {
            run_id: run_id.to_string(),
            workflow_name,
            created_at,
            job_states: Vec::new(),
        };

        for row in &rows {
            let job_state: Option<String> = row.get("job_state");
            if let Some(state_str) = job_state {
                accumulator.push_state(&state_str);
            }
        }

        Ok(Some(accumulator.into_run_row()))
    }

    /// List all jobs for a run ordered by job_id
    pub async fn list_jobs(&self, run_id: &str) -> anyhow::Result<Vec<JobRow>> {
        let rows = sqlx::query(
            "SELECT run_id, job_id, state, exit_code, started_at, ended_at, log_path
             FROM jobs WHERE run_id = ? ORDER BY job_id",
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

/// Accumulates one run's identity plus its jobs' states
struct RunAccumulator {
    run_id:         String,
    workflow_name:  String,
    created_at:     String,
    job_states:     Vec<JobState>,
}

impl RunAccumulator {
    /// Parse and record one job's state string
    fn push_state(&mut self, state_str: &str) {
        match JobState::from_str(state_str) {
            Some(state) => self.job_states.push(state),
            None => {
                tracing::warn!(
                    run_id = %self.run_id,
                    state = %state_str,
                    "unparseable job state in SQL row, treating as Failed"
                );
                self.job_states.push(JobState::Failed);
            }
        }
    }

    fn into_run_row(self) -> RunRow {
        let status = derive_run_state(self.job_states.into_iter()).as_str().to_string();
        RunRow {
            run_id:        self.run_id,
            workflow_name: self.workflow_name,
            status,
            created_at:    self.created_at,
        }
    }
}