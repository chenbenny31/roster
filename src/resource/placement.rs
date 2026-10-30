use crate::resource::pool::{Allocation, ResourcePool};
use crate::workflow::spec::ResourceSpec;

/// First-fit each member over the online nodes in order, all or nothing
/// `online[i]` says whether node i can take work; returns (node index, allocation) per member
/// on a miss, undoes this call's reservations
pub fn place(pools: &mut [ResourcePool], online: &[bool], members: &[ResourceSpec]) -> Option<Vec<(usize, Allocation)>> {
    let mut placed: Vec<(usize, Allocation)> = Vec::new();

    for spec in members {
        // a job that gives only VRAM tries 1, 2, 4, then 8 cards, each over every node, so the fewest cards win
        // try_reserve leaves a pool unchanged when it returns None
        let found = spec.gpu_options().iter().find_map(|option| {
            pools.iter_mut()
                .enumerate()
                .filter(|(node, _)| online[*node])
                .find_map(|(node, pool)| pool.try_reserve(option).map(|alloc| (node, alloc)))
        });

        match found {
            Some(member) => placed.push(member),
            None => {
                for (node, alloc) in &placed {
                    pools[*node].release(alloc);
                }
                return None;
            }
        }
    }
    Some(placed)
}

// tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::discovery::{GpuInfo, SystemResources};

    fn node(cpu: u32, gpu_vram_mb: &[u64]) -> ResourcePool {
        let gpus = gpu_vram_mb.iter().enumerate()
            .map(|(index, &vram_mb)| GpuInfo { index: index as u32, name: "fake".into(), vram_mb, uuid: String::new() })
            .collect();
        ResourcePool::new(SystemResources { cpu_cores: cpu, memory_mb: 8_000, gpus })
    }

    fn spec(cpu: u32, gpu: u32) -> ResourceSpec {
        ResourceSpec { cpu, memory_mb: 512, gpu, vram_mb: if gpu > 0 { 16_000 } else { 0 } }
    }

    #[test]
    fn first_node_that_fits_wins() {
        let mut pools = vec![node(2, &[16_000]), node(4, &[16_000])];
        let online = [true, true];
        assert_eq!(place(&mut pools, &online, &[spec(3, 0)]).unwrap()[0].0, 1); // node 0 too small
        assert_eq!(place(&mut pools, &online, &[spec(1, 1)]).unwrap()[0].0, 0);
        assert_eq!(place(&mut pools, &online, &[spec(1, 1)]).unwrap()[0].0, 1); // node 0 GPU taken
    }

    #[test]
    fn offline_nodes_are_skipped() {
        let mut pools = vec![node(4, &[]), node(4, &[])];
        assert_eq!(place(&mut pools, &[false, true], &[spec(1, 0)]).unwrap()[0].0, 1);
        assert!(place(&mut pools, &[false, false], &[spec(1, 0)]).is_none());
        assert_eq!(pools[0].available_cpu, 4);
    }

    #[test]
    fn gpus_never_combine_across_nodes() {
        let mut pools = vec![node(8, &[16_000]), node(8, &[16_000])];
        assert!(place(&mut pools, &[true, true], &[spec(1, 2)]).is_none());
    }

    #[test]
    fn vram_only_job_takes_the_fewest_cards() {
        let mut pools = vec![node(8, &[24_000, 24_000]), node(8, &[80_000])];
        let online = [true, true];
        let serve = ResourceSpec { cpu: 1, memory_mb: 512, gpu: 0, vram_mb: 40_000 };
        let first = place(&mut pools, &online, &[serve.clone()]).unwrap();
        assert_eq!((first[0].0, first[0].1.gpus.len()), (1, 1)); // one card on node 1 beats two on node 0
        let second = place(&mut pools, &online, &[serve.clone()]).unwrap();
        assert_eq!((second[0].0, second[0].1.gpus.len()), (0, 2)); // then two 24 GB cards, 20 GB each
        assert!(place(&mut pools, &online, &[serve]).is_none());
    }

    #[test]
    fn miss_undoes_earlier_members() {
        let mut pools = vec![node(4, &[]), node(4, &[])];
        assert!(place(&mut pools, &[true, true], &[spec(4, 0), spec(4, 0), spec(4, 0)]).is_none());
        assert_eq!(pools[0].available_cpu, 4);
        assert_eq!(pools[1].available_cpu, 4);
    }
}
