use std::collections::{HashMap, HashSet, VecDeque};

use crate::workflow::spec::WorkflowSpec;

/// Errors produced during DAG validation
#[derive(Debug, PartialEq)]
pub enum DagError {
    DuplicateJobId { job_id: String },
    UnknownDependency { job_id: String, dep_id: String },
    Cycle { job_id: String },
}

impl std::fmt::Display for DagError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DagError::DuplicateJobId { job_id } =>
                write!(formatter, "duplicate job id: {}", job_id),
            DagError::UnknownDependency { job_id, dep_id } =>
                write!(formatter, "job '{}' depends on unknown job '{}'", job_id, dep_id),
            DagError::Cycle { job_id } =>
                write!(formatter, "cycle detected involving job '{}'", job_id),
        }
    }
}

/// Validate a workflow spec and return job IDs in topological execution order
pub fn validate(spec: &WorkflowSpec) -> Result<Vec<String>, DagError> {
    check_duplicate_ids(spec)?;
    check_unknown_deps(spec)?;
    topo_sort(spec)
}

/// Reject any workflow where two jobs share the same ID
fn check_duplicate_ids(spec: &WorkflowSpec) -> Result<(), DagError> {
    let mut seen: HashSet<&str> = HashSet::new();

    for job in &spec.jobs {
        if seen.contains(job.id.as_str()) == false {
            seen.insert(&job.id);
        } else {
            return Err(DagError::DuplicateJobId { job_id: job.id.clone() });
        }
    }

    Ok(())
}

/// Reject any job whose depends_on references a non-existence job ID
fn check_unknown_deps(spec: &WorkflowSpec) -> Result<(), DagError> {
    let known: HashSet<&str> = spec.jobs.iter().map(|job| job.id.as_str()).collect();

    for job in &spec.jobs {
        for dep_id in &job.depends_on {
            if known.contains(dep_id.as_str()) == false {
                return Err(DagError::UnknownDependency {
                    job_id: job.id.clone(),
                    dep_id: dep_id.clone(),
                });
            }
        }
    }

    Ok(())
}

/// Kahn's Algorithm, returns job IDs in topologic order, or Cycle on failure
fn topo_sort(spec: &WorkflowSpec) -> Result<Vec<String>, DagError> {
    let mut in_degree: HashMap<&str, usize> = spec.jobs
        .iter()
        .map(|job| (job.id.as_str(), job.depends_on.len()))
        .collect();

    let mut dependents: HashMap<&str, Vec<&str>> = spec.jobs
        .iter()
        .map(|job| (job.id.as_str(), vec![]))
        .collect();

    for job in &spec.jobs {
        for dep_id in &job.depends_on {
            dependents.entry(dep_id.as_str()).or_default().push(job.id.as_str());
        }
    }

    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter(|(_, &degree)| degree == 0)
        .map(|(&job_id, _)| job_id)
        .collect();

    let mut order: Vec<String> = Vec::with_capacity(spec.jobs.len());

    while let Some(job_id) = queue.pop_front() {
        order.push(job_id.to_string());

        for &dependent in dependents.get(job_id).unwrap_or(&vec![]) {
            let degree = in_degree.entry(dependent).or_default();
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if order.len() != spec.jobs.len() {
        let cycled = in_degree
            .into_iter()
            .find(|(_, degree)| *degree > 0)
            .map(|(job_id, _)| job_id.to_string())
            .unwrap_or_else(|| "unknown".into());

        return Err(DagError::Cycle { job_id: cycled });
    }

    Ok(order)
}

/// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::spec::parse;

    #[test]
    fn linear_chain_correct_order() {
        let spec = parse(r#"
name: linear
jobs:
  - id: a
    command: echo a
  - id: b
    command: echo b
    depends_on: [a]
  - id: c
    command: echo c
    depends_on: [b]
"#).unwrap();

        let order = validate(&spec).unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn diamond_dag_valid() {
        let spec = parse(r#"
name: diamond
jobs:
  - id: root
    command: echo root
  - id: left
    command: echo left
    depends_on: [root]
  - id: right
    command: echo right
    depends_on: [root]
  - id: merge
    command: echo merge
    depends_on: [left, right]
"#).unwrap();

        let order = validate(&spec).unwrap();
        // root must be first, merge must be last
        assert_eq!(order.first().unwrap(), "root");
        assert_eq!(order.last().unwrap(), "merge");
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn cycle_detected() {
        let spec = parse(r#"
name: cyclic
jobs:
  - id: a
    command: echo a
    depends_on: [b]
  - id: b
    command: echo b
    depends_on: [a]
"#).unwrap();

        assert!(matches!(validate(&spec), Err(DagError::Cycle { .. })));
    }

    #[test]
    fn unknown_dependency_rejected() {
        let spec = parse(r#"
name: broken
jobs:
  - id: a
    command: echo a
    depends_on: [nonexistent]
"#).unwrap();

        assert!(matches!(
            validate(&spec),
            Err(DagError::UnknownDependency { .. })
        ));
    }

    #[test]
    fn duplicate_job_id_rejected() {
        let spec = parse(r#"
name: dupes
jobs:
  - id: a
    command: echo a
  - id: a
    command: echo a again
"#).unwrap();

        assert!(matches!(
            validate(&spec),
            Err(DagError::DuplicateJobId { .. })
        ));
    }
}