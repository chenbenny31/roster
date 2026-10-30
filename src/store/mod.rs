use std::path::Path;
use chrono::Utc;
use sqlx::sqlite::SqliteConnectOptions;
use sqlx::{Row, SqlitePool};

use crate::workflow::model::{derive_run_state, JobInstance, JobState, WorkflowRun};

/// A run whose jobs hadn't all ended when the daemon stopped
pub struct SavedRun {
    pub run_id:     String,
    pub created_at: String,
    pub spec_json:  Option<String>, // None for a run stored before specs were kept
    pub jobs:       Vec<SavedJob>,
}

/// A job's last stored state, and where it ran
pub struct SavedJob {
    pub row:        JobRow,
    pub node:       Option<String>, // node name while Running
    pub allocation: Option<String>, // JSON of what it held there
}

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
    pub reason:     Option<String>,
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

        // columns added later: a run's spec, and a running job's node and allocation, so a restarted
        // controller can take the run back; which nodes turned a job down, or why it failed without running
        for (table, column) in [("runs", "spec_json"), ("jobs", "node"), ("jobs", "allocation"), ("jobs", "reason")] {
            add_column(&pool, table, column).await?;
        }

        Ok(Self { pool })
    }

    /// Write or overwrite the run's identity columns and its spec
    pub async fn upsert_run(&self, run: &WorkflowRun) -> anyhow::Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO runs (run_id, workflow_name, created_at, spec_json)
             VALUES (?, ?, ?, ?)",
        )
        .bind(&run.run_id)
        .bind(&run.workflow_name)
        .bind(run.created_at.to_rfc3339())
        .bind(serde_json::to_string(&run.spec)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Write or overwrite a job's current state to SQLite; `node` names where it runs, and its allocation
    /// is kept with it, both empty once it ended
    pub async fn upsert_job(&self, run_id: &str, job: &JobInstance, node: Option<&str>) -> anyhow::Result<()> {
        let allocation = match job.executions.first() {
            Some(execution) => Some(serde_json::to_string(&execution.allocation)?),
            None => None,
        };
        sqlx::query(
            "INSERT OR REPLACE INTO jobs
             (run_id, job_id, state, exit_code, started_at, ended_at, log_path, node, allocation, reason)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(run_id)
        .bind(&job.job_id)
        .bind(job.state.as_str())
        .bind(job.exit_code.map(|c| c as i64))
        .bind(job.started_at.as_ref().map(|t| t.to_rfc3339()))
        .bind(job.ended_at.as_ref().map(|t| t.to_rfc3339()))
        .bind(job.log_path.as_ref().map(|p| p.to_string_lossy().into_owned()))
        .bind(node)
        .bind(allocation)
        .bind(&job.reason)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// A job that can't be taken back after a restart ends as Interrupted
    pub async fn interrupt(&self, run_id: &str, job_id: &str) -> anyhow::Result<()> {
        sqlx::query(
            "UPDATE jobs SET state = 'Interrupted', ended_at = ?, node = NULL, allocation = NULL
             WHERE run_id = ? AND job_id = ?",
        )
        .bind(Utc::now().to_rfc3339())
        .bind(run_id)
        .bind(job_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Runs with a job that hasn't ended, for the controller to take back after a restart
    pub async fn unfinished_runs(&self) -> anyhow::Result<Vec<SavedRun>> {
        let runs = sqlx::query(
            "SELECT run_id, created_at, spec_json FROM runs r
             WHERE EXISTS (SELECT 1 FROM jobs j WHERE j.run_id = r.run_id
                           AND j.state IN ('Pending', 'Queued', 'Running'))
             ORDER BY created_at",
        )
        .fetch_all(&self.pool)
        .await?;

        let mut saved = Vec::new();
        for run in runs {
            let run_id: String = run.get("run_id");
            let jobs = sqlx::query(
                "SELECT run_id, job_id, state, exit_code, started_at, ended_at, log_path, node, allocation, reason
                 FROM jobs WHERE run_id = ?",
            )
            .bind(&run_id)
            .fetch_all(&self.pool)
            .await?;
            saved.push(SavedRun {
                run_id,
                created_at: run.get("created_at"),
                spec_json:  run.get("spec_json"),
                jobs: jobs.into_iter().map(|row| SavedJob {
                    node:       row.get("node"),
                    allocation: row.get("allocation"),
                    row: JobRow {
                        run_id:     row.get("run_id"),
                        job_id:     row.get("job_id"),
                        state:      row.get("state"),
                        exit_code:  row.get("exit_code"),
                        started_at: row.get("started_at"),
                        ended_at:   row.get("ended_at"),
                        log_path:   row.get("log_path"),
                        reason:     row.get("reason"),
                    },
                }).collect(),
            });
        }
        Ok(saved)
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
            "SELECT run_id, job_id, state, exit_code, started_at, ended_at, log_path, reason
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
            reason: row.get("reason"),
        }).collect())
    }
}

/// Add a TEXT column unless the table has it already; a database from an earlier release lacks it
async fn add_column(pool: &SqlitePool, table: &str, column: &str) -> anyhow::Result<()> {
    let present = sqlx::query(&format!("PRAGMA table_info({})", table))
        .fetch_all(pool)
        .await?
        .iter()
        .any(|row| row.get::<String, _>("name") == column);
    if !present {
        sqlx::query(&format!("ALTER TABLE {} ADD COLUMN {} TEXT", table, column)).execute(pool).await?;
    }
    Ok(())
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

// tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::pool::{Allocation, GpuAllocation};
    use crate::workflow::model::Execution;
    use crate::workflow::spec::{self, WorkflowSpec};

    #[tokio::test]
    async fn an_unfinished_run_keeps_its_spec_and_where_its_job_runs() {
        let path = std::env::temp_dir().join(format!("roster-store-{}.db", uuid::Uuid::new_v4()));
        let store = RunStore::open(&path).await.unwrap();
        let spec = spec::parse("name: t\njobs:\n  - id: train\n    command: python train.py\n  - id: done\n    command: 'true'\n").unwrap();
        let mut run = WorkflowRun::new("run-1".into(), 0, spec, 0);
        store.upsert_run(&run).await.unwrap();

        let train = run.jobs.get_mut("train").unwrap();
        train.state = JobState::Running;
        let gpus = vec![GpuAllocation { index: 1, vram_mb: 24_000 }];
        train.executions = vec![Execution { node: 2, allocation: Allocation { cpu: 1, memory_mb: 512, gpus }, handle: None }];
        store.upsert_job("run-1", train, Some("sim-2")).await.unwrap();
        let done = run.jobs.get_mut("done").unwrap();
        done.state = JobState::Succeeded;
        store.upsert_job("run-1", done, None).await.unwrap();

        let saved = store.unfinished_runs().await.unwrap();
        assert_eq!(saved.len(), 1);
        let spec: WorkflowSpec = serde_json::from_str(saved[0].spec_json.as_deref().unwrap()).unwrap();
        assert_eq!(spec.jobs.len(), 2);
        let train = saved[0].jobs.iter().find(|job| job.row.job_id == "train").unwrap();
        assert_eq!(train.node.as_deref(), Some("sim-2"));
        let allocation: Allocation = serde_json::from_str(train.allocation.as_deref().unwrap()).unwrap();
        assert_eq!(allocation.gpus[0].index, 1);

        store.interrupt("run-1", "train").await.unwrap();
        assert!(store.unfinished_runs().await.unwrap().is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn a_job_that_failed_without_running_keeps_why() {
        let path = std::env::temp_dir().join(format!("roster-store-{}.db", uuid::Uuid::new_v4()));
        let store = RunStore::open(&path).await.unwrap();
        let spec = spec::parse("name: t\njobs:\n  - id: typo\n    command: pythn train.py\n").unwrap();
        let mut run = WorkflowRun::new("run-1".into(), 0, spec, 0);
        store.upsert_run(&run).await.unwrap();

        let typo = run.jobs.get_mut("typo").unwrap();
        typo.state  = JobState::Failed;
        typo.reason = Some("turned down by local: 'pythn' not found on PATH /usr/bin".into());
        store.upsert_job("run-1", typo, None).await.unwrap();

        let jobs = store.list_jobs("run-1").await.unwrap();
        assert_eq!(jobs[0].reason.as_deref(), Some("turned down by local: 'pythn' not found on PATH /usr/bin"));
        std::fs::remove_file(&path).ok();
    }
}
