//! Live GPU readings through NVML, shown by `roster info`; without the nvml feature or a driver it reads nothing

use std::fmt;

use serde::{Deserialize, Serialize};

#[cfg(feature = "nvml")]
use nvml_wrapper::{bitmasks::device::ThrottleReasons, enum_wrappers::device::TemperatureSensor, enums::device::UsedGpuMemory};

use crate::resource::discovery::GpuInfo;

/// One GPU's readings at the moment of asking; a reading the GPU does not report stays None
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct GpuHealth {
    pub index:         u32,
    pub temperature_c: Option<u32>,
    pub power_w:       Option<u32>,
    pub power_limit_w: Option<u32>,
    pub used_vram_mb:  Option<u64>, // measured, unlike the pool's booked VRAM
    pub busy_percent:  Option<u32>,
    pub throttled:     Vec<String>, // why clocks are held down: "thermal", "power", "hardware"; empty when not
    #[serde(default)]
    pub processes:       Option<u32>, // compute processes on the card, from any user or program
    #[serde(default)]
    pub process_vram_mb: Option<u64>, // VRAM those processes hold together
}

/// Compute processes holding less than this together don't make a card busy: desktop apps such as
/// a GTK terminal show up as compute processes with a few hundred MB
pub const BUSY_VRAM_MB: u64 = 1024;

impl GpuHealth {
    /// Whether processes on the card hold enough VRAM to keep a job off it
    pub fn busy(&self) -> bool {
        self.process_vram_mb.unwrap_or(0) >= BUSY_VRAM_MB
    }
}

/// NVML kept open for the daemon's life
pub struct GpuMonitor {
    #[cfg(feature = "nvml")]
    nvml: Option<nvml_wrapper::Nvml>,
}

impl GpuMonitor {
    pub fn open() -> Self {
        #[cfg(feature = "nvml")]
        return Self { nvml: nvml_wrapper::Nvml::init().ok() };

        #[cfg(not(feature = "nvml"))]
        return Self {};
    }

    /// Read the GPUs of this machine's inventory; one NVML can't find is left out
    #[cfg_attr(not(feature = "nvml"), allow(unused_variables))]
    pub fn read(&self, gpus: &[GpuInfo]) -> Vec<GpuHealth> {
        #[cfg(feature = "nvml")]
        if let Some(nvml) = &self.nvml {
            return gpus.iter().filter_map(|gpu| {
                let device = nvml.device_by_index(gpu.index).ok()?;
                let processes = device.running_compute_processes().ok();
                Some(GpuHealth {
                    index:         gpu.index,
                    temperature_c: device.temperature(TemperatureSensor::Gpu).ok(),
                    power_w:       device.power_usage().ok().map(|milliwatts| milliwatts / 1000),
                    power_limit_w: device.enforced_power_limit().ok().map(|milliwatts| milliwatts / 1000),
                    used_vram_mb:  device.memory_info().ok().map(|memory| memory.used / 1024 / 1024),
                    busy_percent:  device.utilization_rates().ok().map(|rates| rates.gpu),
                    throttled:     device.current_throttle_reasons().map(throttle_names).unwrap_or_default(),
                    processes:       processes.as_ref().map(|list| list.len() as u32),
                    process_vram_mb: processes.as_ref().map(|list| list.iter().map(|process| match process.used_gpu_memory {
                        UsedGpuMemory::Used(bytes) => bytes / 1024 / 1024,
                        UsedGpuMemory::Unavailable => 0,
                    }).sum()),
                })
            }).collect();
        }
        Vec::new()
    }
}

/// Only the reasons that mean trouble; idle and clock settings are left out
#[cfg(feature = "nvml")]
fn throttle_names(reasons: ThrottleReasons) -> Vec<String> {
    let mut names = Vec::new();
    if reasons.intersects(ThrottleReasons::SW_THERMAL_SLOWDOWN | ThrottleReasons::HW_THERMAL_SLOWDOWN) {
        names.push("thermal".to_string());
    }
    if reasons.intersects(ThrottleReasons::SW_POWER_CAP | ThrottleReasons::HW_POWER_BRAKE_SLOWDOWN) {
        names.push("power".to_string());
    }
    if reasons.contains(ThrottleReasons::HW_SLOWDOWN) {
        names.push("hardware".to_string());
    }
    names
}

/// Appended to a GPU's line in `roster info`, e.g. ", 52 C, 41/300 W, 3% busy, 1200 MB used"
impl fmt::Display for GpuHealth {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(temperature) = self.temperature_c {
            write!(f, ", {} C", temperature)?;
        }
        match (self.power_w, self.power_limit_w) {
            (Some(power), Some(limit)) => write!(f, ", {}/{} W", power, limit)?,
            (Some(power), None) => write!(f, ", {} W", power)?,
            _ => {}
        }
        if let Some(busy) = self.busy_percent {
            write!(f, ", {}% busy", busy)?;
        }
        if let Some(used) = self.used_vram_mb {
            write!(f, ", {} MB used", used)?;
        }
        if !self.throttled.is_empty() {
            write!(f, ", throttled: {}", self.throttled.join(", "))?;
        }
        let vram = self.process_vram_mb.unwrap_or(0);
        match self.processes {
            Some(1) => write!(f, ", 1 process ({} MB)", vram)?,
            Some(count) if count > 1 => write!(f, ", {} processes ({} MB)", count, vram)?,
            _ => {}
        }
        Ok(())
    }
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_only_the_readings_a_gpu_reports() {
        let full = GpuHealth {
            index: 0, temperature_c: Some(52), power_w: Some(41), power_limit_w: Some(300),
            used_vram_mb: Some(1200), busy_percent: Some(3), throttled: vec!["thermal".into()],
            processes: Some(2), process_vram_mb: Some(1100),
        };
        assert_eq!(full.to_string(), ", 52 C, 41/300 W, 3% busy, 1200 MB used, throttled: thermal, 2 processes (1100 MB)");

        let partial = GpuHealth { index: 1, temperature_c: Some(40), ..Default::default() };
        assert_eq!(partial.to_string(), ", 40 C");
    }

    #[test]
    fn a_desktop_app_does_not_make_a_card_busy() {
        let terminal = GpuHealth { processes: Some(1), process_vram_mb: Some(376), ..Default::default() };
        assert!(!terminal.busy());
        let leftover = GpuHealth { processes: Some(2), process_vram_mb: Some(6_000), ..Default::default() };
        assert!(leftover.busy());
    }

    #[cfg(feature = "nvml")]
    #[test]
    fn throttle_names_skip_idle_and_clock_settings() {
        assert!(throttle_names(ThrottleReasons::GPU_IDLE | ThrottleReasons::APPLICATIONS_CLOCKS_SETTING).is_empty());
        assert_eq!(throttle_names(ThrottleReasons::HW_THERMAL_SLOWDOWN | ThrottleReasons::SW_POWER_CAP),
                   vec!["thermal", "power"]);
    }
}
