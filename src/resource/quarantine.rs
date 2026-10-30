//! Cards taken out of service after a hardware fault, saved on this machine by UUID
//! so a restart never hands a faulty card out again

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::resource::discovery::GpuInfo;

/// Xids that point at the card rather than the job: ECC (48, 63, 64, 92, 94, 95), NVLink (74),
/// fallen off the bus (79), GSP (119, 120); others such as 13, 31, 43 or 45 usually mean the job crashed
const HARDWARE_XIDS: [u64; 10] = [48, 63, 64, 74, 79, 92, 94, 95, 119, 120];

/// Why and since when a card is out of service
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Quarantine {
    pub reason: String,
    pub since:  DateTime<Utc>,
}

/// What can happen to a card: an Xid, something the startup scan finds, a qualification's
/// failure, or a clear by a passing qualification or an operator
#[derive(Debug)]
pub enum CardEvent {
    Xid(u64),
    PagesPendingRetirement,
    TestFailed(String),
    Cleared,
}

impl CardEvent {
    /// The card's quarantine after this event; one already in place keeps its first reason
    fn apply(self, current: Option<Quarantine>) -> Option<Quarantine> {
        let reason = match self {
            CardEvent::Cleared => return None,
            CardEvent::Xid(code) if HARDWARE_XIDS.contains(&code) => format!("Xid {}", code),
            CardEvent::Xid(_) => return current, // usually the job crashed, not the card
            CardEvent::PagesPendingRetirement => "memory pages pending retirement".to_string(),
            CardEvent::TestFailed(reason) => format!("qualification failed: {}", reason),
        };
        current.or(Some(Quarantine { reason, since: Utc::now() }))
    }
}

/// Quarantined cards on this machine, saved to a file on every change, and when each was last tested
pub struct Quarantines {
    path:   PathBuf,
    cards:  Mutex<HashMap<String, Quarantine>>, // by card UUID
    tested: Mutex<HashMap<String, DateTime<Utc>>>, // in memory: after a restart every card is tested again
}

impl Quarantines {
    /// Read the saved quarantines; a missing file means none
    pub fn load(path: PathBuf) -> Self {
        let cards = match std::fs::read_to_string(&path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_else(|error| {
                tracing::error!(%error, path = %path.display(), "unreadable quarantine file, starting without it");
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        };
        Self { path, cards: Mutex::new(cards), tested: Mutex::new(HashMap::new()) }
    }

    /// Apply an event to a card; save and log when its quarantine starts or ends
    pub fn handle(&self, gpu: &GpuInfo, event: CardEvent) {
        let mut cards = self.cards.lock().unwrap();
        let before = cards.get(&key(gpu)).cloned();
        let after = event.apply(before.clone());
        if after == before {
            return;
        }
        match after {
            Some(quarantine) => {
                tracing::warn!(gpu = gpu.index, uuid = %gpu.uuid, reason = %quarantine.reason, "GPU quarantined");
                cards.insert(key(gpu), quarantine);
                self.tested.lock().unwrap().remove(&key(gpu)); // a new fault: test it again
            }
            None => {
                tracing::info!(gpu = gpu.index, uuid = %gpu.uuid, "GPU quarantine cleared");
                cards.remove(&key(gpu));
            }
        }
        if let Err(error) = self.save(&cards) {
            tracing::error!(%error, "could not save quarantines");
        }
    }

    pub fn get(&self, gpu: &GpuInfo) -> Option<Quarantine> {
        self.cards.lock().unwrap().get(&key(gpu)).cloned()
    }

    /// A qualification's result: a pass ends a quarantine, a failure starts one
    pub fn record_test(&self, gpu: &GpuInfo, passed: bool, reason: String) {
        if passed {
            tracing::info!(gpu = gpu.index, uuid = %gpu.uuid, %reason, "GPU qualified");
            self.handle(gpu, CardEvent::Cleared);
        } else {
            self.handle(gpu, CardEvent::TestFailed(reason));
        }
        self.tested.lock().unwrap().insert(key(gpu), Utc::now());
    }

    pub fn tested_at(&self, gpu: &GpuInfo) -> Option<DateTime<Utc>> {
        self.tested.lock().unwrap().get(&key(gpu)).copied()
    }

    /// Through a temp file, so a crash never leaves half a file
    fn save(&self, cards: &HashMap<String, Quarantine>) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let temp = self.path.with_extension("json.tmp");
        std::fs::write(&temp, serde_json::to_string_pretty(cards)?)?;
        std::fs::rename(&temp, &self.path)?;
        Ok(())
    }
}

/// A card's identity: its UUID, or its index when the inventory has none
fn key(gpu: &GpuInfo) -> String {
    if gpu.uuid.is_empty() { format!("index-{}", gpu.index) } else { gpu.uuid.clone() }
}

// tests

#[cfg(test)]
mod tests {
    use super::*;

    fn card(index: u32, uuid: &str) -> GpuInfo {
        GpuInfo { index, name: "fake".into(), vram_mb: 16_000, uuid: uuid.into() }
    }

    #[test]
    fn hardware_xids_quarantine_and_the_first_reason_stays() {
        assert!(CardEvent::Xid(13).apply(None).is_none()); // the job's own crash
        let quarantine = CardEvent::Xid(79).apply(None).unwrap();
        assert_eq!(quarantine.reason, "Xid 79");
        assert_eq!(CardEvent::Xid(48).apply(Some(quarantine.clone())), Some(quarantine.clone()));
        assert!(CardEvent::Cleared.apply(Some(quarantine)).is_none());
    }

    #[test]
    fn a_failed_test_quarantines_and_a_new_fault_asks_for_another() {
        let path = std::env::temp_dir().join(format!("roster-quarantine-{}.json", uuid::Uuid::new_v4()));
        let quarantines = Quarantines::load(path.clone());
        let gpu = card(0, "GPU-aaaa");
        quarantines.record_test(&gpu, false, "kernel test: 3 of 4096 hash chains wrong".into());
        assert_eq!(quarantines.get(&gpu).unwrap().reason, "qualification failed: kernel test: 3 of 4096 hash chains wrong");
        assert!(quarantines.tested_at(&gpu).is_some());

        quarantines.record_test(&gpu, true, "kernel test passed".into());
        assert!(quarantines.get(&gpu).is_none());
        quarantines.handle(&gpu, CardEvent::Xid(79));
        assert!(quarantines.tested_at(&gpu).is_none()); // tested again after the new fault
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn quarantines_survive_a_restart_by_uuid() {
        let path = std::env::temp_dir().join(format!("roster-quarantine-{}.json", uuid::Uuid::new_v4()));
        let quarantines = Quarantines::load(path.clone());
        quarantines.handle(&card(0, "GPU-aaaa"), CardEvent::Xid(48));
        drop(quarantines);

        let reloaded = Quarantines::load(path.clone());
        assert_eq!(reloaded.get(&card(3, "GPU-aaaa")).unwrap().reason, "Xid 48"); // index changed, same card
        assert!(reloaded.get(&card(0, "GPU-bbbb")).is_none());
        reloaded.handle(&card(3, "GPU-aaaa"), CardEvent::Cleared);
        assert!(Quarantines::load(path.clone()).get(&card(3, "GPU-aaaa")).is_none());
        std::fs::remove_file(&path).unwrap();
    }
}
