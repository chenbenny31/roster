use crate::resource::discovery::SystemResources;
use crate::workflow::spec::ResourceSpec;

/// Concrete allocation produced by try_reserve, passed back to release
#[derive(Debug, Clone)]
pub struct Allocation {
    pub cpu:       u32,
    pub memory_mb: u64,
    pub gpus:      Vec<GpuAllocation>, // empty for CPU-only jobs
}

/// A single GPU slow within an allocation, records index and VRAM reserved
#[derive(Debug, Clone)]
pub struct GpuAllocation {
    pub index:   u32,
    pub vram_mb: u64,
}

/// Tracks available resources, initialized from discovered hardware, mutated by reserve/release
pub struct ResourcePool {
    pub total:               SystemResources,
    pub available_cpu:       u32,
    pub available_memory_mb: u64,
    pub available_vram_mb:   Vec<u64>, // per GPU, indexed by GpuInfo.index
}

impl ResourcePool {
    /// initialize pool from discovered hardware, available starts equal to total
    pub fn new(resources: SystemResources) -> Self {
        let available_vram_mb = resources.gpus.iter().map(|gpu| gpu.vram_mb).collect();
        Self {
            available_cpu:       resources.cpu_cores,
            available_memory_mb: resources.memory_mb,
            available_vram_mb,
            total:               resources,
        }
    }

    /// Read-only admission check, used by `roster status` to explain Queued job
    pub fn can_admit(&self, spec: &ResourceSpec) -> bool {
        self.try_find_gpus(spec).is_some()
            && self.available_cpu       >= spec.cpu
            && self.available_memory_mb >= spec.memory_mb
    }

    /// Atomically check and reserve resource, returns None if admission fails
    /// on success returns an Allocation must be passed to release when job ends
    pub fn try_reserve(&mut self, spec: &ResourceSpec) -> Option<Allocation> {
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
        self.available_cpu       -= spec.cpu;
        self.available_memory_mb -= spec.memory_mb;

        for gpu_alloc in &gpu_allocs {
            self.available_vram_mb[gpu_alloc.index as usize] -= gpu_alloc.vram_mb;
        }

        Some(Allocation {
            cpu:       spec.cpu,
            memory_mb: spec.memory_mb,
            gpus:      gpu_allocs,
        })
    }

    /// Release a reserved allocation back to pool
    pub fn release(&mut self, alloc: &Allocation) {
        self.available_cpu       += alloc.cpu;
        self.available_memory_mb += alloc.memory_mb;

        for gpu_alloc in &alloc.gpus {
            self.available_vram_mb[gpu_alloc.index as usize] += gpu_alloc.vram_mb;
        }
    }

    /// First-fit GPU selection, find lowest indices with enough free VRAM, else return None
    fn try_find_gpus(&self, spec: &ResourceSpec) -> Option<Vec<GpuAllocation>> {
        if spec.gpu == 0 {
            return Some(vec![]);
        }

        let selected: Vec<GpuAllocation> = self.total.gpus
            .iter()
            .filter(|gpu| self.available_vram_mb[gpu.index as usize] >= spec.vram_mb)
            .take(spec.gpu as usize)
            .map(|gpu| GpuAllocation { index: gpu.index, vram_mb: spec.vram_mb })
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
        GpuInfo { index, name: format!("GPU{}", index), vram_mb }
    }

    #[test]
    fn cpu_only_job_admitted() {
        let mut pool = make_pool(16, 32_000, vec![]);
        let alloc = pool.try_reserve(&cpu_spec(4, 8_000));
        assert!(alloc.is_some());
    }

    #[test]
    fn cpu_only_job_rejected_insufficient_cpu() {
        let mut pool = make_pool(2, 32_000, vec![]);
        assert!(pool.try_reserve(&cpu_spec(4, 8_000)).is_none());
    }

    #[test]
    fn cpu_only_job_rejected_insufficient_memory() {
        let mut pool = make_pool(16, 4_000, vec![]);
        assert!(pool.try_reserve(&cpu_spec(4, 8_000)).is_none());
    }

    #[test]
    fn gpu_job_admitted_first_fit() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 16_000), gpu(1, 16_000)]);
        let alloc = pool.try_reserve(&gpu_spec(4, 8_000, 1, 12_000)).unwrap();
        assert_eq!(alloc.gpus.len(), 1);
        assert_eq!(alloc.gpus[0].index, 0);  // first-fit picks GPU 0
    }

    #[test]
    fn gpu_job_rejected_insufficient_vram() {
        let mut pool = make_pool(16, 32_000, vec![gpu(0, 8_000)]);
        assert!(pool.try_reserve(&gpu_spec(4, 8_000, 1, 12_000)).is_none());
    }

    #[test]
    fn release_restores_resources() {
        let mut pool  = make_pool(16, 32_000, vec![gpu(0, 16_000)]);
        let alloc = pool.try_reserve(&gpu_spec(4, 8_000, 1, 12_000)).unwrap();
        pool.release(&alloc);
        assert_eq!(pool.available_cpu, 16);
        assert_eq!(pool.available_memory_mb, 32_000);
        assert_eq!(pool.available_vram_mb[0], 16_000);
    }

    #[test]
    fn multi_gpu_first_fit() {
        let mut pool = make_pool(32, 64_000, vec![gpu(0, 16_000), gpu(1, 16_000), gpu(2, 16_000)]);
        let alloc = pool.try_reserve(&gpu_spec(4, 8_000, 2, 8_000)).unwrap();
        assert_eq!(alloc.gpus.len(), 2);
        assert_eq!(alloc.gpus[0].index, 0);
        assert_eq!(alloc.gpus[1].index, 1);  // first-fit picks 0 and 1
    }
}