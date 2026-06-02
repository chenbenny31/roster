use serde::{Deserialize, Serialize};

/// Top-level workflow YAML
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkflowSpec {
    pub name: String,
    pub jobs: Vec<JobSpec>,
}

/// Single job declaration inside a workflow YAML
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct JobSpec {
    pub id: String,
    pub command: String,

    #[serde(default)] // omitted in YAML, to empty vec
    pub depends_on: Vec<String>,

    #[serde(default)]
    pub resources: ResourceSpec,
}

/// Declared resource requirements for a job, held during job's lifetime
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResourceSpec {
    #[serde(default = "default_cpu")]
    pub cpu: u32,

    #[serde(default = "default_memory_mb")]
    pub memory_mb: u64,

    pub gpu: Option<u32>, // device index, None means CPU-only job
    pub vram_mb: Option<u64>, // None means no VARM reservation
}

impl Default for ResourceSpec {
    fn default() -> Self {
        Self {
            cpu: default_cpu(),
            memory_mb: default_memory_mb(),
            gpu: None,
            vram_mb: None,
        }
    }
}

fn default_cpu() -> u32 { 1 }
fn default_memory_mb() -> u64 { 512 }

/// Parse a workflow spec from YAML
pub fn parse(yaml: &str) -> Result<WorkflowSpec, serde_yaml::Error> {
    serde_yaml::from_str(yaml)
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"
name: train-and-eval
jobs:
  - id: preprocess
    command: python preprocess.py
    resources:
      cpu: 4
      memory_mb: 4096

  - id: train
    depends_on: [preprocess]
    command: python train.py
    resources:
      cpu: 8
      gpu: 0
      vram_mb: 12000

  - id: eval
    depends_on: [train]
    command: python eval.py
    resources:
      cpu: 2
      gpu: 0
      vram_mb: 4000
"#;

    #[test]
    fn parse_example_workflow() {
        let spec = parse(EXAMPLE).unwrap();
        assert_eq!(spec.name, "train-and-eval");
        assert_eq!(spec.jobs.len(), 3);
    }

    #[test]
    fn depends_on_defaults_to_empty() {
        let spec = parse(EXAMPLE).unwrap();
        assert!(spec.jobs[0].depends_on.is_empty());
    }

    #[test]
    fn resource_defaults_apply() {
        let yaml = r#"
name: minimal
jobs:
  - id: job1
    command: echo hello
"#;
        let spec = parse(yaml).unwrap();
        assert_eq!(spec.jobs[0].resources.cpu, 1);
        assert_eq!(spec.jobs[0].resources.memory_mb, 512);
        assert!(spec.jobs[0].resources.gpu.is_none());
    }

    #[test]
    fn gpu_job_parses_correctly() {
        let spec = parse(EXAMPLE).unwrap();
        let train = &spec.jobs[1];
        assert_eq!(train.resources.gpu, Some(0));
        assert_eq!(train.resources.vram_mb, Some(12000));
    }
}