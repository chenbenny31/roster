//! `roster qualify`: test one card and report the result to this machine's daemon; the controller
//! runs it as an internal job that holds the whole card, so no job lands on the card meanwhile

use std::process::Command;

use crate::ipc::client;
use crate::ipc::protocol::{Request, Response};
use crate::resource::cuda;

/// Test the card and record the result with this machine's daemon; an error when the card failed
/// `retest` asks for DCGM's longer level 2 diagnostic, required before a quarantined card returns
pub async fn run(index: u32, uuid: Option<String>, retest: bool) -> anyhow::Result<()> {
    let result = test(index, uuid.as_deref(), retest);
    let (passed, reason) = match &result {
        Ok(tests) => (true, tests.clone()),
        Err(reason) => (false, reason.clone()),
    };
    match client::send(Request::GpuQualified { index, passed, reason: reason.clone() }).await? {
        Response::Recorded => {}
        other => anyhow::bail!("unexpected reply: {:?}", other),
    }
    if !passed {
        anyhow::bail!("GPU {} failed qualification: {}", index, reason);
    }
    println!("GPU {} qualified: {}", index, reason);
    Ok(())
}

/// The kernel test, then DCGM where it is installed; Ok names the tests that ran
fn test(index: u32, uuid: Option<&str>, retest: bool) -> Result<String, String> {
    let mut passed = Vec::new();
    match cuda::kernel_test() {
        Some(Ok(())) => passed.push("kernel test".to_string()),
        Some(Err(reason)) => return Err(format!("kernel test: {}", reason)),
        None => {} // no NVIDIA driver here, e.g. a simulated node
    }
    let level = if retest { 2 } else { 1 };
    if let Some(id) = dcgm_gpu_id(index, uuid) {
        dcgm_diag(id, level)?;
        passed.push(format!("DCGM level {}", level));
    }
    if passed.is_empty() {
        return Ok("no test available here".to_string());
    }
    Ok(format!("{} passed", passed.join(" and ")))
}

/// DCGM's id for the card: the row of `dcgmi discovery -l` that lists its UUID, or its index when
/// the inventory has no UUID; None when dcgmi isn't installed or doesn't list the card
fn dcgm_gpu_id(index: u32, uuid: Option<&str>) -> Option<u32> {
    let output = Command::new("dcgmi").args(["discovery", "-l"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    match uuid {
        Some(uuid) => discovery_id(&String::from_utf8_lossy(&output.stdout), uuid),
        None => Some(index),
    }
}

/// In the discovery table a card's id opens its first row and its UUID follows a few rows below
fn discovery_id(table: &str, uuid: &str) -> Option<u32> {
    let mut current = None;
    for line in table.lines() {
        let first = line.trim().trim_start_matches('|').split('|').next().unwrap_or("").trim();
        if let Ok(id) = first.parse() {
            current = Some(id);
        }
        if line.contains(uuid) {
            return current;
        }
    }
    None
}

/// `dcgmi diag -r LEVEL -i ID -j`: any test reported "Fail", or a failed run, fails the card
fn dcgm_diag(id: u32, level: u8) -> Result<(), String> {
    let output = Command::new("dcgmi")
        .args(["diag", "-r", &level.to_string(), "-i", &id.to_string(), "-j"])
        .output()
        .map_err(|error| format!("dcgmi: {}", error))?;
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap_or(serde_json::Value::Null);
    let failed = failed_tests(&report);
    if !failed.is_empty() {
        return Err(format!("DCGM level {} failed: {}", level, failed.join(", ")));
    }
    if !output.status.success() {
        return Err(format!("DCGM level {} exited with {}", level, output.status));
    }
    Ok(())
}

/// Names of tests whose status is "Fail"; DCGM's JSON layout changes between versions, so walk all of it
/// and name a failure after the nearest enclosing "name"
fn failed_tests(report: &serde_json::Value) -> Vec<String> {
    fn walk(value: &serde_json::Value, name: Option<&str>, failed: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(object) => {
                let name = object.get("name").and_then(|name| name.as_str()).or(name);
                let status = object.get("status").and_then(|status| status.as_str());
                if status.is_some_and(|status| status.eq_ignore_ascii_case("fail")) {
                    let name = name.unwrap_or("unnamed test").to_string();
                    if !failed.contains(&name) {
                        failed.push(name);
                    }
                }
                object.values().for_each(|child| walk(child, name, failed));
            }
            serde_json::Value::Array(items) => items.iter().for_each(|child| walk(child, name, failed)),
            _ => {}
        }
    }
    let mut failed = Vec::new();
    walk(report, None, &mut failed);
    failed
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_finds_the_id_above_the_uuid() {
        let table = "\
2 GPUs found.
+--------+----------------------------------------------------------------------+
| GPU ID | Device Information                                                   |
+--------+----------------------------------------------------------------------+
| 0      | Name: NVIDIA A100-SXM4-80GB                                          |
|        | PCI Bus ID: 00000000:07:00.0                                         |
|        | Device UUID: GPU-aaaa                                                |
+--------+----------------------------------------------------------------------+
| 1      | Name: NVIDIA A100-SXM4-80GB                                          |
|        | PCI Bus ID: 00000000:0F:00.0                                         |
|        | Device UUID: GPU-bbbb                                                |
+--------+----------------------------------------------------------------------+";
        assert_eq!(discovery_id(table, "GPU-bbbb"), Some(1));
        assert_eq!(discovery_id(table, "GPU-cccc"), None);
    }

    #[test]
    fn failed_tests_are_named_whatever_the_layout() {
        let report: serde_json::Value = serde_json::from_str(r#"{"DCGM GPU Diagnostic": {"test_categories": [
            {"category": "Deployment", "tests": [{"name": "Denylist", "results": [{"gpu_id": "0", "status": "Pass"}]}]},
            {"category": "Hardware", "tests": [
                {"name": "Memory", "results": [{"gpu_id": "0", "status": "Fail", "info": "ECC errors"}]},
                {"name": "Diagnostic", "results": [{"gpu_id": "0", "status": "Skip"}]}]}]}}"#).unwrap();
        assert_eq!(failed_tests(&report), vec!["Memory"]);
        assert!(failed_tests(&serde_json::json!({"results": [{"status": "Pass"}]})).is_empty());
    }
}
