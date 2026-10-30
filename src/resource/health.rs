//! Live GPU readings through NVML, shown by `roster info`; without the nvml feature or a driver it reads nothing

use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[cfg(feature = "nvml")]
use nvml_wrapper::{bitmasks::device::ThrottleReasons, enum_wrappers::device::TemperatureSensor, enums::device::UsedGpuMemory};
#[cfg(feature = "nvml")]
use nvml_wrapper::{bitmasks::event::EventTypes, enums::event::XidError, error::NvmlError, Nvml};

use crate::paths::quarantine_path;
use crate::resource::discovery::GpuInfo;
use crate::resource::quarantine::{CardEvent, Quarantines};

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
    #[serde(default)]
    pub quarantined:     Option<String>, // why the card is out of service, until a test passes or `roster gpu clear`
    #[serde(default)]
    pub tested_at:       Option<DateTime<Utc>>, // last qualification since this node's daemon started
}

/// Compute processes holding less than this together don't make a card busy: desktop apps such as
/// a GTK terminal show up as compute processes with a few hundred MB
pub const BUSY_VRAM_MB: u64 = 1024;

impl GpuHealth {
    /// Whether processes on the card hold enough VRAM to keep a job off it
    pub fn busy(&self) -> bool {
        self.process_vram_mb.unwrap_or(0) >= BUSY_VRAM_MB
    }

    /// Why a job may not get this card now: quarantined, or, while nothing is booked on it,
    /// busy with processes roster didn't start or held down for heat or power
    pub fn blocked(&self, unbooked: bool) -> Option<String> {
        if let Some(reason) = &self.quarantined {
            return Some(format!("quarantined: {}", reason));
        }
        if unbooked && self.busy() {
            return Some(format!("busy outside roster ({} MB)", self.process_vram_mb.unwrap_or(0)));
        }
        if unbooked && !self.throttled.is_empty() {
            return Some(format!("throttled while idle: {}", self.throttled.join(", ")));
        }
        None
    }
}

/// NVML kept open for the daemon's life, and this machine's quarantined cards
pub struct GpuMonitor {
    #[cfg(feature = "nvml")]
    nvml:        Option<Arc<Nvml>>,
    quarantines: Arc<Quarantines>,
}

impl GpuMonitor {
    /// Load saved quarantines; with NVML, also check for memory pages pending retirement
    /// and start listening for Xid errors
    #[cfg_attr(not(feature = "nvml"), allow(unused_variables))]
    pub fn open(gpus: &[GpuInfo]) -> Self {
        let quarantines = Arc::new(Quarantines::load(quarantine_path()));

        #[cfg(feature = "nvml")]
        {
            let nvml = Nvml::init().ok().map(Arc::new);
            if let Some(nvml) = &nvml {
                for gpu in gpus {
                    if matches!(nvml.device_by_index(gpu.index).and_then(|device| device.are_pages_pending_retired()), Ok(true)) {
                        quarantines.handle(gpu, CardEvent::PagesPendingRetirement);
                    }
                }
                listen_for_xids(nvml.clone(), gpus.to_vec(), quarantines.clone());
            }
            return Self { nvml, quarantines };
        }

        #[cfg(not(feature = "nvml"))]
        return Self { quarantines };
    }

    /// Readings for the GPUs of this machine's inventory; a quarantine shows even when NVML can't reach the card
    #[cfg_attr(not(feature = "nvml"), allow(unused_mut))]
    pub fn read(&self, gpus: &[GpuInfo]) -> Vec<GpuHealth> {
        gpus.iter().map(|gpu| {
            let mut health = GpuHealth {
                index:       gpu.index,
                quarantined: self.quarantines.get(gpu).map(|quarantine| quarantine.reason),
                tested_at:   self.quarantines.tested_at(gpu),
                ..Default::default()
            };
            #[cfg(feature = "nvml")]
            if let Some(device) = self.nvml.as_ref().and_then(|nvml| nvml.device_by_index(gpu.index).ok()) {
                let processes = device.running_compute_processes().ok();
                health.temperature_c   = device.temperature(TemperatureSensor::Gpu).ok();
                health.power_w         = device.power_usage().ok().map(|milliwatts| milliwatts / 1000);
                health.power_limit_w   = device.enforced_power_limit().ok().map(|milliwatts| milliwatts / 1000);
                health.used_vram_mb    = device.memory_info().ok().map(|memory| memory.used / 1024 / 1024);
                health.busy_percent    = device.utilization_rates().ok().map(|rates| rates.gpu);
                health.throttled       = device.current_throttle_reasons().map(throttle_names).unwrap_or_default();
                health.processes       = processes.as_ref().map(|list| list.len() as u32);
                health.process_vram_mb = processes.as_ref().map(|list| list.iter().map(|process| match process.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => bytes / 1024 / 1024,
                    UsedGpuMemory::Unavailable => 0,
                }).sum());
            }
            health
        }).collect()
    }

    /// `roster gpu clear`: put a quarantined card back in service
    pub fn clear(&self, gpu: &GpuInfo) {
        self.quarantines.handle(gpu, CardEvent::Cleared);
    }

    /// What `roster qualify` reports for one of this machine's cards
    pub fn record_test(&self, gpu: &GpuInfo, passed: bool, reason: String) {
        self.quarantines.record_test(gpu, passed, reason);
    }
}

/// Xids arrive as events and NVML's wait blocks, so a thread of its own waits for them and hands each
/// to the quarantines, which decide whether the card or the job is at fault
#[cfg(feature = "nvml")]
fn listen_for_xids(nvml: Arc<Nvml>, gpus: Vec<GpuInfo>, quarantines: Arc<Quarantines>) {
    std::thread::spawn(move || {
        let mut set = match nvml.create_event_set() {
            Ok(set) => set,
            Err(error) => return tracing::warn!(%error, "Xid errors unavailable"),
        };
        let mut watched = Vec::new();
        for gpu in &gpus {
            let Ok(device) = nvml.device_by_index(gpu.index) else { continue };
            // registering an unsupported event type would drop the whole set
            if !device.supported_event_types().is_ok_and(|types| types.contains(EventTypes::CRITICAL_XID_ERROR)) {
                continue;
            }
            set = match device.register_events(EventTypes::CRITICAL_XID_ERROR, set) {
                Ok(set) => set,
                Err(error) => return tracing::warn!(error = %error.error, "Xid errors unavailable"),
            };
            watched.push(gpu.clone());
        }
        if watched.is_empty() {
            return;
        }
        tracing::info!(gpus = watched.len(), "listening for Xid errors");

        loop {
            match set.wait(1000) {
                Ok(event) => {
                    let Some(XidError::Value(code)) = event.event_data else { continue };
                    let uuid = event.device.uuid().unwrap_or_default();
                    let Some(gpu) = watched.iter().find(|gpu| gpu.uuid == uuid) else { continue };
                    tracing::warn!(gpu = gpu.index, xid = code, "GPU reported an Xid error");
                    quarantines.handle(gpu, CardEvent::Xid(code));
                }
                Err(NvmlError::Timeout) => {}
                Err(error) => return tracing::warn!(%error, "stopped listening for Xid errors"),
            }
        }
    });
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
            processes: Some(2), process_vram_mb: Some(1100), quarantined: None, tested_at: None,
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

    #[test]
    fn blocked_says_why_a_card_is_not_handed_out() {
        let quarantined = GpuHealth { quarantined: Some("Xid 48".into()), ..Default::default() };
        assert_eq!(quarantined.blocked(false).as_deref(), Some("quarantined: Xid 48")); // even while booked
        let hot = GpuHealth { throttled: vec!["thermal".into()], ..Default::default() };
        assert!(hot.blocked(false).is_none()); // a job running on it heats it up
        assert_eq!(hot.blocked(true).as_deref(), Some("throttled while idle: thermal"));
    }

    #[cfg(feature = "nvml")]
    #[test]
    fn throttle_names_skip_idle_and_clock_settings() {
        assert!(throttle_names(ThrottleReasons::GPU_IDLE | ThrottleReasons::APPLICATIONS_CLOCKS_SETTING).is_empty());
        assert_eq!(throttle_names(ThrottleReasons::HW_THERMAL_SLOWDOWN | ThrottleReasons::SW_POWER_CAP),
                   vec!["thermal", "power"]);
    }
}
