use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::resource::discovery::SystemResources;
use crate::workflow::spec::ResourceSpec;

/// The run and job a booking is for: a user's job, or one of roster's own card tests;
/// the same ids on the controller and on the node
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Holder {
    pub run_id: String,
    pub job_id: String,
}

impl Holder {
    pub fn new(run_id: &str, job_id: &str) -> Self {
        Self { run_id: run_id.to_string(), job_id: job_id.to_string() }
    }
}

impl fmt::Display for Holder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}/{}", self.run_id, self.job_id)
    }
}

/// Concrete allocation produced by try_reserve
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Allocation {
    pub cpu:       u32,
    pub memory_mb: u64,
    pub gpus:      Vec<GpuAllocation>, // empty for CPU-only jobs
}

/// One whole GPU within an allocation: its index and all of its VRAM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    pub index:   u32,
    pub vram_mb: u64,
}

/// What a machine has free, initialized from discovered hardware; every booking is kept under the
/// id of the job or card test holding it, so each card names its holder and a release needs only the id
pub struct ResourcePool {
    pub total:               SystemResources,
    pub available_cpu:       u32,
    pub available_memory_mb: u64,
    pub holders:             Vec<Option<Holder>>, // per GPU, indexed by GpuInfo.index; kept by book/reserve/release
    pub blocked:             Vec<bool>, // per GPU: quarantined, or busy or throttled while idle; not handed out
    bookings:                HashMap<Holder, Allocation>,
}

impl ResourcePool {
    /// initialize pool from discovered hardware, available starts equal to total
    pub fn new(resources: SystemResources) -> Self {
        Self {
            available_cpu:       resources.cpu_cores,
            available_memory_mb: resources.memory_mb,
            holders:             vec![None; resources.gpus.len()],
            blocked:             vec![false; resources.gpus.len()],
            bookings:            HashMap::new(),
            total:               resources,
        }
    }

    /// The one place bookings are written: counts, cards and the holder's entry, which grows when
    /// the same holder books again here
    fn add(&mut self, holder: &Holder, alloc: Allocation) {
        self.available_cpu = self.available_cpu.saturating_sub(alloc.cpu);
        self.available_memory_mb = self.available_memory_mb.saturating_sub(alloc.memory_mb);
        for gpu in &alloc.gpus {
            if let Some(slot) = self.holders.get_mut(gpu.index as usize) {
                *slot = Some(holder.clone());
            }
        }
        let booking = self.bookings.entry(holder.clone()).or_default();
        booking.cpu       += alloc.cpu;
        booking.memory_mb += alloc.memory_mb;
        booking.gpus.extend(alloc.gpus);
    }

    /// Count an allocation already in use, such as a running job taken back after a restart;
    /// unlike try_reserve it can't refuse
    pub fn book(&mut self, holder: &Holder, alloc: &Allocation) {
        self.add(holder, alloc.clone());
    }

    /// Book an allocation chosen elsewhere, such as a controller's job on this machine; refused with the
    /// card and its holder when another holder has one of its cards; booking the same holder again changes nothing.
    /// CPU and memory are counted but never refuse: the controller checked them, and only cards are exclusive
    pub fn try_book(&mut self, holder: &Holder, alloc: &Allocation) -> Result<(), (u32, Holder)> {
        if self.holds(holder) {
            return Ok(());
        }
        for gpu in &alloc.gpus {
            if let Some(Some(other)) = self.holders.get(gpu.index as usize) {
                return Err((gpu.index, other.clone()));
            }
        }
        self.add(holder, alloc.clone());
        Ok(())
    }

    /// Whether `holder` has anything booked here
    pub fn holds(&self, holder: &Holder) -> bool {
        self.bookings.contains_key(holder)
    }

    /// Book one whole card for roster's own use, such as a qualification; None when it's held or reports no VRAM
    pub fn reserve_card(&mut self, holder: &Holder, index: u32) -> Option<Allocation> {
        let gpu = self.total.gpus.iter().find(|gpu| gpu.index == index)?;
        if gpu.vram_mb == 0 || self.holders[index as usize].is_some() {
            return None;
        }
        let alloc = Allocation { cpu: 0, memory_mb: 0, gpus: vec![GpuAllocation { index, vram_mb: gpu.vram_mb }] };
        self.add(holder, alloc.clone());
        Some(alloc)
    }

    /// Read-only admission check, used by `roster status` to explain Queued job
    pub fn can_admit(&self, spec: &ResourceSpec) -> bool {
        self.try_find_gpus(spec).is_some()
            && self.available_cpu       >= spec.cpu
            && self.available_memory_mb >= spec.memory_mb
    }

    /// Whether the job could ever fit here, compared against totals rather than free resources
    pub fn could_fit(&self, spec: &ResourceSpec) -> bool {
        let matching_gpus = self.total.gpus
            .iter()
            .filter(|gpu| gpu.vram_mb > 0 && gpu.vram_mb >= spec.vram_per_gpu())
            .count();

        self.total.cpu_cores >= spec.cpu
            && self.total.memory_mb >= spec.memory_mb
            && matching_gpus >= spec.gpu as usize
    }

    /// Atomically check and reserve resource for `holder`, returns None if admission fails
    /// on success `release(holder)` gives it back when the job ends
    pub fn try_reserve(&mut self, holder: &Holder, spec: &ResourceSpec) -> Option<Allocation> {
        if self.available_cpu < spec.cpu {
            return None;
        }
        if self.available_memory_mb < spec.memory_mb {
            return None;
        }

        let gpu_allocs = if spec.gpu == 0 {
            vec![]
        } else  {
            self.try_find_gpus(spec)?
        };

        // commit as all check passed
        let alloc = Allocation {
            cpu:       spec.cpu,
            memory_mb: spec.memory_mb,
            gpus:      gpu_allocs,
        };
        self.add(holder, alloc.clone());
        Some(alloc)
    }

    /// Give back everything `holder` booked here; a holder with nothing booked changes nothing,
    /// so a second release is harmless, and a card goes free only while this holder holds it
    pub fn release(&mut self, holder: &Holder) {
        let Some(alloc) = self.bookings.remove(holder) else { return };
        self.available_cpu       += alloc.cpu;
        self.available_memory_mb += alloc.memory_mb;
        for gpu in &alloc.gpus {
            if let Some(slot) = self.holders.get_mut(gpu.index as usize) {
                if slot.as_ref() == Some(holder) {
                    *slot = None;
                }
            }
        }
    }

    /// First-fit GPU selection, find lowest indices with enough VRAM, else return None
    /// Whole GPUs, first-fit by index, never shared: a card is free when nothing holds it,
    /// and each must hold its even share of the job's VRAM; cards that report no VRAM are never handed out
    fn try_find_gpus(&self, spec: &ResourceSpec) -> Option<Vec<GpuAllocation>> {
        if spec.gpu == 0 {
            return Some(vec![]);
        }

        let selected: Vec<GpuAllocation> = self.total.gpus
            .iter()
            .filter(|gpu| gpu.vram_mb > 0 && self.holders[gpu.index as usize].is_none())
            .filter(|gpu| !self.blocked[gpu.index as usize])
            .filter(|gpu| gpu.vram_mb >= spec.vram_per_gpu())
            .take(spec.gpu as usize)
            .map(|gpu| GpuAllocation { index: gpu.index, vram_mb: gpu.vram_mb }) // the whole card
            .collect();

        if selected.len() == spec.gpu as usize {
            Some(selected)
        } else {
            None
        }
    }
}

/// tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::discovery::{SystemResources, GpuInfo};

    fn make_pool(cpu: u32, memory_mb: u64, gpus: Vec<GpuInfo>) -> ResourcePool {
        ResourcePool::new(SystemResources { cpu_cores: cpu, memory_mb, gpus })
    }

    fn cpu_spec(cpu: u32, memory_mb: u64) -> ResourceSpec {
        ResourceSpec { cpu, memory_mb, gpu: 0, vram_mb: 0 }
    }

    fn gpu_spec(cpu: u32, memory_mb: u64, gpu: u32, vram_mb: u64) -> ResourceSpec {
        ResourceSpec { cpu, memory_mb, gpu, vram_mb }
    }

    fn gpu(index: u32, vram_mb: u64) -> GpuInfo {
        GpuInfo { index, name: format!("GPU{}", index), vram_mb, uuid: String::new() }
    }

    fn job(id: &str) -> Holder {
        Holder::new("run", id)
    }

    #[test]
    fn cpu_only_job_admitted() {
        let mut pool = make_pool(16, 32_000, vec![]);
        let alloc = pool.try_reserve(&job("a"), &cpu_spec(4, 8_000));
        assert!(alloc.is_some());
    }

    #[test]
    fn cpu_only_job_rejected_insufficient_cpu() {
        let mut pool = make_pool(2, 32_000, vec![]);
        assert!(pool.try_reserve(&job("a"), &cpu_spec(4, 8_000)).is_none());
    }

    #[test]
    fn cpu_only_job_rejected_insufficient_memory() {
        let mut pool = make_pool(16, 4_000, vec![]);
        assert!(pool.try_reserve(&job("a"), &cpu_spec(4, 8_000)).is_none());
    }

    #[test]
    fn gpu_job_admitted_first_fit() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 16_000), gpu(1, 16_000)]);
        let alloc = pool.try_reserve(&job("a"), &gpu_spec(4, 8_000, 1, 12_000)).unwrap();
        assert_eq!(alloc.gpus.len(), 1);
        assert_eq!(alloc.gpus[0].index, 0);  // first-fit picks GPU 0
    }

    #[test]
    fn gpu_job_rejected_insufficient_vram() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 8_000)]);
        assert!(pool.try_reserve(&job("a"), &gpu_spec(4, 8_000, 1, 12_000)).is_none());
    }

    #[test]
    fn a_gpu_is_never_shared_even_without_vram() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 16_000)]);
        let alloc = pool.try_reserve(&job("a"), &gpu_spec(1, 512, 1, 0)).unwrap();
        assert_eq!(alloc.gpus[0].vram_mb, 16_000); // the whole card
        assert!(pool.try_reserve(&job("b"), &gpu_spec(1, 512, 1, 0)).is_none());
        pool.release(&job("a"));
        assert!(pool.try_reserve(&job("b"), &gpu_spec(1, 512, 1, 0)).is_some());
    }

    #[test]
    fn each_gpu_holds_an_even_share_of_vram() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 8_000), gpu(1, 16_000), gpu(2, 16_000)]);
        assert!(pool.try_reserve(&job("a"), &gpu_spec(1, 512, 2, 40_000)).is_none()); // 20 GB each, no card has it
        let alloc = pool.try_reserve(&job("a"), &gpu_spec(1, 512, 2, 24_000)).unwrap(); // 12 GB each
        assert_eq!(alloc.gpus.iter().map(|gpu| gpu.index).collect::<Vec<_>>(), vec![1, 2]); // GPU 0 too small
    }

    #[test]
    fn a_blocked_gpu_is_skipped() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 16_000), gpu(1, 16_000)]);
        pool.blocked[0] = true;
        assert_eq!(pool.try_reserve(&job("a"), &gpu_spec(1, 512, 1, 0)).unwrap().gpus[0].index, 1);
        assert!(pool.try_reserve(&job("b"), &gpu_spec(1, 512, 1, 0)).is_none());
        assert!(pool.could_fit(&gpu_spec(1, 512, 2, 0))); // it may free up later
    }

    #[test]
    fn release_restores_resources() {
        let mut pool  = make_pool(16, 32_000, vec![gpu(0, 16_000)]);
        pool.try_reserve(&job("a"), &gpu_spec(4, 8_000, 1, 12_000)).unwrap();
        pool.release(&job("a"));
        assert_eq!(pool.available_cpu, 16);
        assert_eq!(pool.available_memory_mb, 32_000);
        assert_eq!(pool.holders[0], None);
        pool.release(&job("a")); // a second release changes nothing
        assert_eq!(pool.available_cpu, 16);
    }

    #[test]
    fn each_card_names_its_holder() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 16_000), gpu(1, 16_000)]);
        pool.try_reserve(&job("train"), &gpu_spec(1, 512, 1, 0)).unwrap();
        assert_eq!(pool.holders[0], Some(job("train")));
        assert!(pool.reserve_card(&job("test"), 0).is_none()); // held by train
        pool.reserve_card(&job("test"), 1).unwrap();
        assert_eq!(pool.holders[1], Some(job("test")));
        pool.release(&job("other")); // holds nothing here
        assert_eq!(pool.holders, vec![Some(job("train")), Some(job("test"))]);
        pool.release(&job("train"));
        assert_eq!(pool.holders, vec![None, Some(job("test"))]);
        assert_eq!(pool.available_cpu, 16);
    }

    #[test]
    fn a_booking_from_elsewhere_is_refused_only_for_a_held_card() {
        let mut pool = make_pool(4, 8_000, vec![gpu(0, 16_000), gpu(1, 16_000)]);
        let card1 = Allocation { cpu: 1, memory_mb: 512, gpus: vec![GpuAllocation { index: 1, vram_mb: 16_000 }] };
        pool.reserve_card(&job("test"), 1).unwrap();
        assert_eq!(pool.try_book(&job("remote"), &card1), Err((1, job("test"))));
        assert!(!pool.holds(&job("remote")));
        pool.release(&job("test"));
        assert_eq!(pool.try_book(&job("remote"), &card1), Ok(()));
        assert_eq!(pool.try_book(&job("remote"), &card1), Ok(())); // the same launch again
        assert_eq!((pool.available_cpu, pool.holders[1].clone()), (3, Some(job("remote")))); // counted once
        let cpu_only = Allocation { cpu: 8, memory_mb: 512, gpus: vec![] };
        assert_eq!(pool.try_book(&job("big"), &cpu_only), Ok(())); // more cores than free: counted, not refused
        assert_eq!(pool.available_cpu, 0);
    }

    #[test]
    fn a_job_taken_back_after_a_restart_holds_its_cards() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 16_000), gpu(1, 16_000)]);
        let alloc = Allocation { cpu: 2, memory_mb: 1_000, gpus: vec![GpuAllocation { index: 1, vram_mb: 16_000 }] };
        pool.book(&job("train"), &alloc);
        assert_eq!(pool.holders[1], Some(job("train")));
        assert_eq!(pool.try_reserve(&job("next"), &gpu_spec(1, 512, 1, 0)).unwrap().gpus[0].index, 0);
        pool.release(&job("train"));
        assert_eq!((pool.available_cpu, pool.holders[1].clone()), (15, None)); // next still holds 1 core
    }

    #[test]
    fn multi_gpu_first_fit() {
        let mut pool = make_pool(32, 64_000, vec![gpu(0, 16_000), gpu(1, 16_000), gpu(2, 16_000)]);
        let alloc = pool.try_reserve(&job("a"), &gpu_spec(4, 8_000, 2, 8_000)).unwrap();
        assert_eq!(alloc.gpus.len(), 2);
        assert_eq!(alloc.gpus[0].index, 0);
        assert_eq!(alloc.gpus[1].index, 1);  // first-fit picks 0 and 1
    }

    #[test]
    fn could_fit_compares_totals_not_free_resources() {
        let mut pool = make_pool(4, 8_000, vec![gpu(0, 8_000), gpu(1, 16_000)]);
        pool.try_reserve(&job("a"), &gpu_spec(4, 8_000, 1, 16_000)).unwrap();
        assert!(pool.could_fit(&gpu_spec(4, 8_000, 1, 16_000))); // busy now, fits later
        assert!(!pool.could_fit(&gpu_spec(1, 1_000, 2, 24_000))); // 12 GB each, only one GPU has that
        assert!(!pool.could_fit(&cpu_spec(8, 1_000)));
    }
}