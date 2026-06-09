use sysinfo::System;

pub struct SystemResources {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub gpus: Vec<GpuInfo>, // empty if no NVIDIA driver
}

pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vram_mb: u64,
}

/// Discover available system resource, log warning on NVML failure
pub fn discover() -> SystemResources {
    let mut system = System::new_all();
    system.refresh_all();

    let cpu_cores = system.cpus().len() as u32;
    let memory_mb = system.total_memory() / 1024 / 1024;
    let gpus = discover_gpus();

    tracing::info!(cpu_cores, memory_mb, gpu_count = gpus.len(), "system resource discovered");

    SystemResources { cpu_cores, memory_mb, gpus }
}

/// Attempt NVML GPU discovery, return empty vec on failure
fn discover_gpus() -> Vec<GpuInfo> {
    #[cfg(feature = "nvml")]
    {
        match nvml_wrapper::Nvml::init() {
            Err(error) => {
                tracing::warn!(?error, "NVML initialization failed - run without GPU support");
                return vec![];
            }
            Ok(nvml) => {
                let count = match nvml.device_count() {
                    Ok(count) => count,
                    Err(error) => {
                        tracing::warn!(?error, "NVML device count failed");
                        return vec![];
                    }
                };

                let mut gpus = Vec::new();
                for index in 0..count {
                    match nvml.device_by_index(index) {
                        Err(error) => {
                            tracing::warn!(?error, index, "failed to query GPU");
                        }
                        Ok(device) => {
                            let name = device.name().unwrap_or_else(|| "unknown".into());
                            let vram_mb = device.memory_info()
                                .map(|info| info.total / 1024 / 1024)
                                .unwrap_or(0);
                            gpus.push(GpuInfo { index, name, vram_mb });
                        }
                    }
                }
                return gpus;
            }
        }
    }

    #[cfg(not(feature = "nvml"))]
    {
        tracing::info!("NVML feature not enabled - run without GPU support");
        vec![]
    }
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_succeeds() {
        let resources = discover();
        assert!(resources.cpu_cores > 0);
        assert!(resources.memory_mb > 0);
        // skip GPU test
    }

    #[test]
    fn cpu_count_is_sane() {
        let resources = discover();
        assert!(resources.cpu_cores <= 1024);
    }
}