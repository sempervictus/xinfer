//! The centralized GPU memory budget (the MMU).
//!
//! Every GPU memory pool (the KV cache, the GDN/mamba slots, the MoE-offload
//! expert slots, the LoRA adapter bank) is capped by a single measured free-VRAM
//! pool, in priority order. Without this, each allocator independently reads
//! "free VRAM" at its own moment and the sum of their claims exceeds the actual
//! free memory, producing a deferred OOM (the mamba-snapshot OOM in the
//! production log).
//!
//! The priority order (the CUDA best practice: the models stay resident, the
//! swappable pools are sized by importance):
//!
//! 1. CUDA graph workspace (measured, reserved first — the graphs must always fit).
//! 2. The model-aware transient workspace (GEMM scratch, attention workspaces,
//!    MoE activation pool, MLA per-token tensors) — measured, not a percentage.
//! 3. The MoE-offload expert slots (if `XINFER_MOE_OFFLOAD` is on).
//! 4. The LoRA adapter bank (if LoRA is configured).
//! 5. The GDN/mamba recurrent slots (if the model is hybrid).
//! 6. The KV cache pool (the elastic remainder — the unified-memory demotion
//!    governor can shrink this at runtime).
//!
//! The model: each pool sizes itself with its existing logic, but is capped at
//! the running remainder of the free-VRAM pool. The KV cache (the last pool)
//! takes whatever is left, so it is the elastic pool. The invariant: the sum of
//! all pool allocations is at most the measured free VRAM.
//!
//! The transient workspace is the single source of truth shared by the master
//! (the engine plan) + the runner (the model-load plan), so they agree by
//! construction (the `NumBlocksReport` min-clamp remains as defense-in-depth).

use crate::utils::config::EngineConfig;
use crate::utils::kvcache_allocator::KVCacheAllocator;

/// The measured GPU reserve (the single source of truth for the transient
/// workspace + the graph slice). Computed once from the allocator (the
/// model-aware `compute_workspace_budget` + the `cuda_graph_bytes`), then
/// shared by the master + the runner so they carve the same bytes.
#[derive(Debug, Clone, Copy)]
pub struct GpuReserve {
    /// The CUDA graph capture cost (the measured per-batch metadata + hidden +
    /// logits buffers). Zero for non-graph builds (no VRAM idled).
    pub graph_bytes: u64,
    /// The model-aware transient workspace (FlashInfer + CUTLASS + MoE pool +
    /// split-K + transient activations + MLA per-token), floored at the
    /// minimum activation reserve. Scales with the actual prefill chunk + model
    /// width, so it is large when a large step actually needs it.
    pub workspace_bytes: u64,
}

impl GpuReserve {
    /// Compute the measured reserve from the allocator (the single source of
    /// truth). The `graph_batch` is the largest decode batch the graphs are
    /// captured for (the `hybrid_mamba_graph_capture_max_batch` for hybrid
    /// models, the `max_num_parallel_reqs` otherwise); the `max_model_len` is
    /// the configured context bound. Both are already finalized by the
    /// allocator's `plan`, so this reads the model-aware values, not a
    /// percentage of free VRAM.
    pub fn compute(allocator: &KVCacheAllocator, graph_batch: usize, max_model_len: usize) -> Self {
        let workspace = allocator.compute_workspace_budget();
        Self {
            graph_bytes: allocator.cuda_graph_bytes(graph_batch, max_model_len),
            workspace_bytes: workspace.total_bytes,
        }
    }

    /// The total measured reserve (the graph slice + the transient workspace).
    /// This is what the pools must not consume (so the NVFP4 GEMM
    /// workspace does not OOM, the failure the prior static 10% headroom band-aided).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.graph_bytes.saturating_add(self.workspace_bytes)
    }
}

/// The running remainder of the GPU memory budget. Each pool caps its
/// allocation at the current remainder, then reduces it by what it took.
#[derive(Debug, Clone, Copy)]
pub struct PoolBudget {
    /// The total free VRAM the budget is carved from (bytes).
    pub total_free: u64,
    /// The running remainder (the bytes still available to lower-priority pools).
    remainder: u64,
}

impl PoolBudget {
    /// Create a budget from the measured free VRAM, reserving the measured GPU
    /// reserve (the graph slice + the model-aware transient workspace, the
    /// highest-priority, non-elastic bytes). The pools are capped at the
    /// running remainder so they never consume the reserve (so the GEMM
    /// workspace does not OOM).
    #[must_use]
    pub fn new(total_free: u64, reserve: &GpuReserve) -> Self {
        let remainder = total_free.saturating_sub(reserve.total_bytes());
        Self { total_free, remainder }
    }

    /// Cap a pool's requested allocation at the running remainder, and reduce
    /// the remainder by what the pool actually took. Returns the capped
    /// allocation. The pool should allocate `min(requested, cap)` bytes.
    ///
    /// This is the priority-cap: the pool sizes itself with its own logic (the
    /// `requested`), but is never allowed to exceed the running remainder (the
    /// `cap`). The remainder is reduced by the actual allocation, so
    /// lower-priority pools see the reduced pool.
    #[must_use]
    pub fn cap(&mut self, requested: u64) -> u64 {
        let granted = requested.min(self.remainder);
        self.remainder = self.remainder.saturating_sub(granted);
        granted
    }

    /// The remaining bytes available to lower-priority pools (the KV cache, the
    /// elastic last pool, takes this).
    #[must_use]
    pub fn remainder(&self) -> u64 {
        self.remainder
    }

    /// The total free VRAM the budget was carved from.
    #[must_use]
    pub fn total_free(&self) -> u64 {
        self.total_free
    }

    /// The operator-visible budget report (the startup log line).
    #[must_use]
    pub fn report(&self, allocations: &[(&str, u64)]) -> String {
        let gb = |b: u64| format!("{:.2}", b as f64 / (1024.0 * 1024.0 * 1024.0));
        let parts: Vec<String> = allocations
            .iter()
            .map(|(name, bytes)| format!("{name}={}", gb(*bytes)))
            .collect();
        format!(
            "GPU memory budget ({} total free, {} remaining after caps): {}",
            gb(self.total_free),
            gb(self.remainder),
            parts.join(" "),
        )
    }
}

/// Apply the MMU KV-cache cap to `econfig.num_blocks` (the elastic last pool).
/// The KV cache is capped at `budget.remainder()` (the bytes left after the
/// graph/workspace/MoE/LoRA/GDN pools). The per-block byte size is derived from
/// the planned `kvcache_memory_bytes / num_blocks`. This is the single source of
/// the cap, called by both the master (the engine plan) + the runner (the
/// model-load plan) so they agree on `num_blocks` (avoiding the block-table
/// out-of-bounds error from a master/runner `num_blocks` divergence).
///
/// The admission-demand floor is computed against the FULL remainder (before any
/// MoE half-reserve), so the MoE carve cannot nullify it: the floor is the
/// worst-case concurrent demand (the `max_num_seqs` × the per-seq blocks + the
/// GDN block-equivalent), and the pool is raised to that floor (capped at the
/// byte ceiling) so the preemption/swap churn the floor was meant to prevent
/// cannot return.
pub fn apply_kv_cache_cap(
    budget: &PoolBudget,
    econfig: &mut EngineConfig,
) {
    let kv_cache_cap = budget.remainder();
    if kv_cache_cap == 0 {
        return;
    }
    let per_block_bytes = econfig
        .kvcache_memory_bytes
        .checked_div(econfig.num_blocks.max(1))
        .unwrap_or(0);
    if per_block_bytes == 0 {
        return;
    }
    let max_blocks = (kv_cache_cap / per_block_bytes as u64) as usize;
    // The admission-demand floor: the KV pool should hold the worst-case
    // concurrent demand (the max_num_seqs sequences each at their full
    // max_model_len + max_tokens, + the GDN slot when hybrid). The prefix hits
    // reduce the per-request demand at admission time (the required_blocks
    // nets them out), so the pool sized to the no-prefix worst case is the safe
    // floor. The demand is in blocks; the GDN cost is converted to block-equivalents
    // (the mamba_slot_bytes / the per_block_bytes).
    let per_seq_blocks = {
        let ctx = econfig.max_model_len.unwrap_or(0);
        let dec = econfig.max_tokens.unwrap_or(0);
        (ctx + dec).div_ceil(econfig.block_size.max(1))
    };
    let gdn_equiv_blocks = if econfig.mamba_slot_bytes > 0 {
        econfig.mamba_slot_bytes.div_euclid(per_block_bytes as usize).max(1)
    } else {
        0
    };
    let admission_demand_blocks =
        admission_demand(econfig.max_num_seqs, per_seq_blocks, gdn_equiv_blocks);
    // The R4: the floor is computed against the FULL remainder (before any MoE
    // half-reserve), so the MoE carve cannot nullify it. The pool is capped at
    // the byte ceiling + raised to the floor (the pure decision, the testable
    // without an EngineConfig).
    let resolved = resolve_kv_block_count(
        econfig.num_blocks,
        admission_demand_blocks,
        max_blocks,
    );
    if resolved < econfig.num_blocks {
        crate::log_info!(
            "KV cache capped by the memory budget: {} -> {} blocks",
            econfig.num_blocks,
            resolved
        );
    } else if resolved > econfig.num_blocks {
        crate::log_info!(
            "KV cache raised to the admission-demand floor: {} -> {} blocks (the {} seqs × {} blocks/seq)",
            econfig.num_blocks,
            resolved,
            econfig.max_num_seqs,
            per_seq_blocks + gdn_equiv_blocks
        );
    }
    econfig.num_blocks = resolved;
}

/// The admission-demand floor (the pure computation, the testable without an
/// `EngineConfig`): the worst-case concurrent KV block demand = the
/// `max_num_seqs` × the per-seq blocks (the context + the decode) + the GDN
/// block-equivalent. The prefix hits reduce the per-request demand at admission
/// time (the `required_blocks` nets them out), so this no-prefix worst case is
/// the safe floor for the pool size.
#[must_use]
pub fn admission_demand(max_num_seqs: usize, per_seq_blocks: usize, gdn_equiv_blocks: usize) -> usize {
    max_num_seqs.saturating_mul(per_seq_blocks + gdn_equiv_blocks)
}

/// The KV block count after applying the admission-demand floor + the byte
/// ceiling (the pure computation, the testable without an `EngineConfig`). The
/// `current_blocks` is the pool's prior size; the `floor_blocks` is the
/// admission-demand (the worst-case concurrent demand); the `ceiling_blocks` is
/// the byte cap (the `budget.remainder()` divided by the per-block bytes). The result is
/// the `max(min(current, ceiling), min(floor, ceiling))`: the pool is capped at
/// the byte ceiling (without over-committing the GPU) but raised to the floor
/// (without the preemption/swap churn) — the floor is always satisfiable because
/// it is computed against the full remainder (before any MoE half-reserve, the
/// R4 fix). The `ceiling` is the binding constraint when the floor exceeds it
/// (so there is no negative pool).
#[must_use]
pub fn resolve_kv_block_count(
    current_blocks: usize,
    floor_blocks: usize,
    ceiling_blocks: usize,
) -> usize {
    // The cap: the pool never exceeds the byte ceiling.
    let capped = current_blocks.min(ceiling_blocks);
    // The floor: the pool is raised to the admission-demand (capped at the
    // ceiling, so there is no negative pool when the demand exceeds the bytes).
    let floored = floor_blocks.min(ceiling_blocks);
    capped.max(floored)
}

/// The kind of a physical memory region (the kernel-MMU classification).
/// `Pinned` regions (the model weights, the CUDA graph pool, the transient
/// workspace) are allocated once and never reclaimed; they are the fixed
/// baseline. `Elastic` regions (the KV cache, the GDN/mamba slots, the CPU swap
/// mirror) can be grown from idle VRAM or shrunk back to it under pressure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    Pinned,
    Elastic,
}

/// A named physical memory region owned by a tier. The `bytes` is the current
/// allocation; the `base` is the address (the `0` when the caller does not
/// track addresses, the region is accounted by size only).
#[derive(Debug, Clone)]
pub struct Region {
    pub name: &'static str,
    pub bytes: u64,
    pub kind: RegionKind,
}

impl Region {
    #[must_use]
    pub fn new(name: &'static str, bytes: u64, kind: RegionKind) -> Self {
        Self { name, bytes, kind }
    }
}

/// The live physical memory map (the kernel-MMU core). It is the single source
/// of truth for what is allocated on one device (a rank's VRAM, or the host
/// DRAM), replacing the boot-time `PoolBudget` snapshot with a *computed* view:
/// `free` is always `total - sum(regions)`, and the live probe (the
/// `refresh_probe`) reconciles the computed free against the actual driver free
/// so the map detects drift (the allocations the map did not register, the CUDA
/// context overhead, the graph pool growth).
///
/// The map is plain owned data (the no internal locks, the no Arc, the no Box,
/// the no dyn wrapping the hot path). It is a field on the engine (the host,
/// rank 0), owned by value, and mutated under the engine's existing write lock.
/// The per-rank maps (the N-rank TP) are a `Vec<PhysicalMemoryMap>` on the
/// engine; rank 0 is probed directly, the other ranks report their live free
/// over the existing IPC (the `NumBlocksReport`) and the engine writes it into
/// their map. The probe closure is a plain field (the `None` probe for the
/// non-CUDA and the test builds), called under `&mut self` (the no a lock held
/// across the driver call).
pub struct PhysicalMemoryMap {
    /// The label (the "rank 0", the "host", the "device 1").
    pub label: &'static str,
    /// The total physical memory (the `cuMemGetInfo` total, the fixed).
    pub total: u64,
    /// The registered regions (the pinned baseline + the elastic tiers).
    pub regions: Vec<Region>,
    /// The cached live free (the last probe result, the `u64::MAX` sentinel means
    /// "not yet probed", the computed free is used until the first probe).
    pub live_free: u64,
    /// The live probe (the driver query, the `None` for the non-CUDA and the test
    /// builds). Returns the `(free, total)` pair (the cuMemGetInfo gives both).
    /// Called under `&mut self` by `refresh_probe`, which updates the cached
    /// live_free + the total (the physical VRAM, the no the boot-time free).
    pub probe: Option<Box<dyn Fn() -> Option<(u64, u64)> + Send + Sync>>,
}

impl std::fmt::Debug for PhysicalMemoryMap {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("PhysicalMemoryMap")
            .field("label", &self.label)
            .field("total", &self.total)
            .field("regions", &self.regions)
            .field("live_free", &self.live_free)
            .field("probe", &self.probe.is_some())
            .finish()
    }
}

impl PhysicalMemoryMap {
    /// Create the map from the total + the initial region set (the boot-time
    /// carve, the `PoolBudget` result). The `probe` is the live driver query (the
    /// `None` for the non-CUDA and the test builds).
    #[must_use]
    pub fn new(
        label: &'static str,
        total: u64,
        initial_regions: Vec<Region>,
        probe: Option<Box<dyn Fn() -> Option<(u64, u64)> + Send + Sync>>,
    ) -> Self {
        Self {
            label,
            total,
            regions: initial_regions,
            live_free: u64::MAX,
            probe,
        }
    }

    /// The sum of the registered region bytes (the allocated, the map's view).
    fn allocated(&self) -> u64 {
        self.regions.iter().map(|r| r.bytes).sum()
    }

    /// The computed free (the `total - allocated`, the map's view). A plain read.
    #[must_use]
    pub fn computed_free(&self) -> u64 {
        self.total.saturating_sub(self.allocated())
    }

    /// The live free (the cached probe, the ground truth). The computed free when
    /// the probe has not run (the `u64::MAX` sentinel). A plain read.
    #[must_use]
    pub fn live_free(&self) -> u64 {
        if self.live_free == u64::MAX {
            self.computed_free()
        } else {
            self.live_free
        }
    }

    /// The physical pressure (the `1 - live_free / total`, the 0..1 fraction of the
    /// physical memory in use). This is the signal the QoS controller reads (the no
    /// the block occupancy, which is the logical signal and is blind to the idle
    /// physical memory). A high block-occupancy + a low physical pressure is the
    /// "GBs free but token-starved" divergence (the static pool too small, the idle
    /// physical memory untracked). A plain read.
    #[must_use]
    pub fn pressure(&self) -> f32 {
        if self.total == 0 {
            return 1.0;
        }
        let used = self.total.saturating_sub(self.live_free());
        used as f32 / self.total as f32
    }

    /// The idle headroom (the `live_free`, the bytes available to grow an elastic
    /// region). A plain read.
    #[must_use]
    pub fn idle(&self) -> u64 {
        self.live_free()
    }

    /// Run the live probe (the expensive driver call, the run on a timer or before
    /// a grow, the no every pressure read). Takes `&mut self` (the engine write
    /// lock is held by the caller, the no an internal lock). Stores the result in
    /// the `live_free` cache. When the probe is absent (the non-CUDA build), it
    /// falls back to the computed free (the no a stale driver value).
pub fn refresh_probe(&mut self) {
        match self.probe.as_ref() {
            Some(probe) => match probe() {
                Some((free, total)) => {
                    self.live_free = free;
                    // The total is the physical VRAM (the cuMemGetInfo total, the
                    // fixed). Update it from the probe (the no the boot-time free,
                    // which was the prior mislabel). The 0 total (the probe did not
                    // report it) leaves the existing value.
                    if total > 0 {
                        self.total = total;
                    }
                }
                None => self.live_free = self.computed_free(),
            },
            None => self.live_free = self.computed_free(),
        }
    }

    /// Register (or update) a region by name. A new name adds the region; an
    /// existing name updates its bytes (the grow, the shrink, the boot carve).
    /// Takes `&mut self` (the engine write lock is held by the caller). Returns
    /// the region's new byte count.
    pub fn set_region(&mut self, name: &'static str, bytes: u64, kind: RegionKind) -> u64 {
        if let Some(r) = self.regions.iter_mut().find(|r| r.name == name) {
            r.bytes = bytes;
            r.kind = kind;
        } else {
            self.regions.push(Region::new(name, bytes, kind));
        }
        bytes
    }

    /// Grow an elastic region by `delta` bytes (the reclaim from idle physical
    /// memory). The `None` when the region is pinned (the weights, the graphs
    /// never grow) or when the map has no idle headroom (the `computed_free <
    /// delta`). Takes `&mut self`. Returns the new byte count on success.
    pub fn grow(&mut self, name: &'static str, delta: u64) -> Option<u64> {
        if delta == 0 {
            return None;
        }
        // The headroom check is the computed free (the conservative bound, the no a
        // probe round-trip under the mutation). Read it, then write.
        if self.computed_free() < delta {
            return None;
        }
        let region = self.regions.iter_mut().find(|r| r.name == name)?;
        if region.kind == RegionKind::Pinned {
            return None; // the pinned regions never grow (the baseline is fixed)
        }
        region.bytes = region.bytes.saturating_add(delta);
        Some(region.bytes)
    }

    /// Reclaim `delta` bytes from an elastic region (the shrink back to idle
    /// physical memory). The `None` when the region is pinned or holds fewer than
    /// `delta` bytes (the floor, the region cannot shrink below its minimum).
    /// Takes `&mut self`. Returns the new byte count on success.
    pub fn reclaim(&mut self, name: &'static str, delta: u64) -> Option<u64> {
        if delta == 0 {
            return None;
        }
        let region = self.regions.iter_mut().find(|r| r.name == name)?;
        if region.kind == RegionKind::Pinned {
            return None; // the pinned regions never shrink (the baseline is fixed)
        }
        if region.bytes < delta {
            return None; // the floor: cannot reclaim more than the region holds
        }
        region.bytes -= delta;
        Some(region.bytes)
    }

    /// The current byte count of a region (the `0` when absent). A plain read.
    #[must_use]
    pub fn region_bytes(&self, name: &'static str) -> u64 {
        self.regions.iter().find(|r| r.name == name).map_or(0, |r| r.bytes)
    }

    /// Whether `grow(name, delta)` would succeed (the feasibility check, the no
    /// mutation). The `false` when the region is absent, pinned, or the computed
    /// free is below the delta. Used by the uniform `grow_all` to check every rank
    /// before mutating any (the all-or-nothing contract, the no a partial grow that
    /// desyncs the block counts).
    #[must_use]
    pub fn can_grow(&self, name: &'static str, delta: u64) -> bool {
        if delta == 0 {
            return false;
        }
        if self.computed_free() < delta {
            return false;
        }
        match self.regions.iter().find(|r| r.name == name) {
            Some(region) => region.kind == RegionKind::Elastic,
            None => false,
        }
    }

    /// Whether `reclaim(name, delta)` would succeed (the feasibility check, the no
    /// mutation). The `false` when the region is absent, pinned, or holds fewer than
    /// the delta. Used by the uniform `reclaim_all` to check every rank before
    /// mutating any.
    #[must_use]
    pub fn can_reclaim(&self, name: &'static str, delta: u64) -> bool {
        if delta == 0 {
            return false;
        }
        match self.regions.iter().find(|r| r.name == name) {
            Some(region) => region.kind == RegionKind::Elastic && region.bytes >= delta,
            None => false,
        }
    }

    /// The operator-visible map report (the startup + the periodic log line, the
    /// cold path). A plain read (the no a probe call, the cached live_free is used
    /// as-is).
    #[must_use]
    pub fn report(&self) -> String {
        let gb = |b: u64| format!("{:.2}", b as f64 / (1024.0 * 1024.0 * 1024.0));
        let parts: Vec<String> = self
            .regions
            .iter()
            .map(|r| {
                let tag = if r.kind == RegionKind::Pinned { "pinned" } else { "elastic" };
                format!("{}({}:{})", r.name, tag, gb(r.bytes))
            })
            .collect();
        let computed = self.computed_free();
        let live = self.live_free();
        let drift = if live > computed { live - computed } else { computed - live };
        format!(
            "{} (total {}, computed free {}, live free {}, drift {}, pressure {:.0}%): {}",
            self.label,
            gb(self.total),
            gb(computed),
            gb(live),
            gb(drift),
            self.pressure() * 100.0,
            parts.join(" ")
        )
    }
}

/// The set of physical maps for a deployment: one device map per rank (the N-rank
/// TP, the N-host at M-rank) plus one host map (the CPU swap, the GDN spill, the
/// PLE table, the disagg-PD transfer buffer). The set is plain owned data (the no
/// a HashMap of Arcs, the no a Mutex). It is a field on the engine, owned by
/// value, and mutated under the engine's existing write lock. The per-rank device
/// maps are a `Vec` indexed by rank (the rank 0 is probed directly, the ranks
/// 1..N are filled by their IPC report). The host map is an `Option` (the absent
/// on the no-host-swap build).
#[derive(Debug)]
pub struct MemoryMapSet {
    /// The per-rank device maps (the index is the rank, the 0 is the master).
    pub devices: Vec<PhysicalMemoryMap>,
    /// The shared host map (the CPU tiers). The `None` when the deployment has no
    /// host swap (the no-CPU build, the single-device no-swap case).
    pub host: Option<PhysicalMemoryMap>,
}

impl MemoryMapSet {
    /// Create an empty set (the boot, before the maps are registered).
    #[must_use]
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
            host: None,
        }
    }

    /// Register the device map for a rank (the boot carve, the replace on a grow).
    /// The rank index must be within the `devices` length (the engine sizes the
    /// vec to the rank count at boot). Takes `&mut self`.
    pub fn set_device(&mut self, rank: usize, map: PhysicalMemoryMap) {
        self.devices[rank] = map;
    }

    /// Register the host map (the boot, the `None` clears it). Takes `&mut self`.
    pub fn set_host(&mut self, map: Option<PhysicalMemoryMap>) {
        self.host = map;
    }

    /// The device map for a rank (the `None` when the rank is out of range). A
    /// plain read.
    #[must_use]
    pub fn device(&self, rank: usize) -> Option<&PhysicalMemoryMap> {
        self.devices.get(rank)
    }

    /// The host map (the `None` when absent). A plain read.
    #[must_use]
    pub fn host(&self) -> Option<&PhysicalMemoryMap> {
        self.host.as_ref()
    }

    /// The rank count (the `devices` length). A plain read.
    #[must_use]
    pub fn rank_count(&self) -> usize {
        self.devices.len()
    }

    /// The aggregate device pressure (the max across ranks, the tightest rank
    /// bounds the deployment). The `0.0` when no rank is registered. This is the
    /// signal the QoS controller reads (the no the per-rank, the scheduler is rank
    /// 0 and acts on the worst rank). A plain read.
    #[must_use]
    pub fn device_pressure(&self) -> f32 {
        self.devices.iter().map(|m| m.pressure()).fold(0.0f32, f32::max)
    }

    /// The aggregate idle device physical memory (the min across ranks, the
    /// tightest rank bounds a grow, the no growing one rank while another is
    /// tight). The `0` when no rank is registered. A plain read.
    #[must_use]
    pub fn idle_device(&self) -> u64 {
        self.devices.iter().map(|m| m.idle()).min().unwrap_or(0)
    }

    /// Grow the named elastic region on every rank by `delta` (the uniform grow,
    /// the no a per-rank divergence). The `false` when any rank cannot grow (the
    /// tightest rank bounds, the no a partial grow that desyncs the block counts).
    /// This is the frame-table sync: a grow only happens when all ranks agree.
    /// Takes `&mut self` (the engine write lock is held by the caller).
    pub fn grow_all(&mut self, name: &'static str, delta: u64) -> bool {
        if self.devices.is_empty() {
            return false;
        }
        // The all-or-nothing contract: check every rank's feasibility first (the no
        // mutation), then apply. A partial grow (rank 0 grown, rank 1 not) would
        // desync the block counts across the TP group (the flashinfer validation
        // bound, the NumBlocksReport). The tightest rank bounds the whole grow.
        if !self.devices.iter().all(|m| m.can_grow(name, delta)) {
            return false;
        }
        for map in self.devices.iter_mut() {
            // The feasibility was checked above, so this cannot fail (the no a
            // partial mutation). The expect documents the invariant.
            map.grow(name, delta).expect("can_grow passed, the grow must succeed");
        }
        true
    }

    /// Reclaim the named elastic region on every rank by `delta` (the uniform
    /// demote, the no a per-rank divergence). The `false` when any rank cannot
    /// reclaim (the region floor, the no a partial reclaim). Takes `&mut self`.
    pub fn reclaim_all(&mut self, name: &'static str, delta: u64) -> bool {
        if self.devices.is_empty() {
            return false;
        }
        // The all-or-nothing contract: check every rank's feasibility first (the no
        // mutation), then apply. A partial reclaim would desync the block counts
        // across the TP group. The tightest rank (the smallest region) bounds.
        if !self.devices.iter().all(|m| m.can_reclaim(name, delta)) {
            return false;
        }
        for map in self.devices.iter_mut() {
            map.reclaim(name, delta).expect("can_reclaim passed, the reclaim must succeed");
        }
        true
    }

    /// Refresh every rank's live probe (the driver call, the run on a timer or
    /// before a grow). Rank 0 is probed directly (the `probe` field); the other
    /// ranks are filled by their IPC report (the engine writes the reported value
    /// into `live_free` before calling this, the no a per-rank driver call from
    /// rank 0). Takes `&mut self`.
    pub fn refresh_all(&mut self) {
        for map in self.devices.iter_mut() {
            map.refresh_probe();
        }
        if let Some(host) = self.host.as_mut() {
            host.refresh_probe();
        }
    }

    /// The operator-visible report across the whole set (the every rank + the host,
    /// the single startup + periodic line). A plain read (the no a probe call, the
    /// cached live_free is used as-is).
    #[must_use]
    pub fn report(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (rank, map) in self.devices.iter().enumerate() {
            parts.push(format!("rank{rank}: {}", map.report()));
        }
        if let Some(host) = self.host.as_ref() {
            parts.push(host.report());
        }
        parts.join(" | ")
    }

    /// Build the map set from the finalized engine config + the measured total VRAM
    /// (the boot carve, the C3 region registration). The device maps are one per
    /// rank (the `device_count`, the 1 for the single-device, the N for the N-rank
    /// TP). Each device map registers the pinned baseline (the weights, the graph
    /// pool, the workspace) + the elastic tiers (the KV, the GDN slots). The host
    /// map registers the CPU swap + the GDN spill + the PLE table (the `None` host
    /// map when the deployment has no host swap, the no-CPU build).
    ///
    /// The `weights_bytes`, `graph_bytes`, `workspace_bytes` are the pinned baseline
    /// (the measured, the no a guess). The `kv_bytes` is the `num_blocks` times the
    /// `per_block_bytes` (the dtype-correct, the section 8). The `gdn_bytes` is the
    /// `mamba_memory_bytes` (the workload-sized, the no a fixed carve). The probe is
    /// the live driver query (the `None` for the test builds, the the computed-free
    /// fallback).
    #[must_use]
    pub fn from_engine_config(
        device_count: usize,
        total_vram: u64,
        weights_bytes: u64,
        graph_bytes: u64,
        workspace_bytes: u64,
        kv_bytes: u64,
        gdn_bytes: u64,
        gdn_device_prefix_bytes: u64,
        cpu_swap_bytes: u64,
        gdn_cpu_spill_bytes: u64,
        ple_bytes: u64,
        probe: Option<Box<dyn Fn() -> Option<(u64, u64)> + Send + Sync>>,
        host_probe: Option<Box<dyn Fn() -> Option<(u64, u64)> + Send + Sync>>,
    ) -> Self {
        let device_regions = vec![
            Region::new("weights", weights_bytes, RegionKind::Pinned),
            Region::new("graph_pool", graph_bytes, RegionKind::Pinned),
            Region::new("workspace", workspace_bytes, RegionKind::Pinned),
            Region::new("kv_cache", kv_bytes, RegionKind::Elastic),
            Region::new("gdn_slots", gdn_bytes, RegionKind::Elastic),
            // The GDN device prefix LRU (the C4 co-sizing): the hot turn snapshots
            // that must stay on device for the prefix match to be GDN-aligned. The
            // elastic (the the LRU can spill to the CPU tier under pressure).
            Region::new("gdn_device_prefix", gdn_device_prefix_bytes, RegionKind::Elastic),
        ];
        // The total VRAM is the driver's physical total (the cuMemGetInfo total, the
        // 96GB), not the regions sum. The idle headroom is the total minus the
        // regions (the grow actuator's source). Each rank gets the same total
        // (the per-rank VRAM, the TP shards the data not the memory cap).
        //
        // The live probe is rank 0 only (the master process holds the CUDA
        // context for the driver query; the other ranks report their live free
        // over the NumBlocksReport IPC, the C7). The probe is moved into rank 0
        // (the no a clone, the Box<dyn Fn> is not Clone); the other ranks get
        // None (the computed-free fallback until their IPC report arrives).
        let rank0 = PhysicalMemoryMap::new("rank0", total_vram, device_regions.clone(), probe);
        let mut devices = vec![rank0];
        // The non-zero ranks (the 1..device_count). The label is the static
        // "rankN" (the cosmetic; the rank identity is the vec index, the no a
        // dynamic label string). Each gets the same total + regions, the no probe
        // (the live free arrives over the NumBlocksReport IPC, the C7).
        for _ in 1..device_count.max(1) {
            devices.push(PhysicalMemoryMap::new(
                "rankN",
                total_vram,
                device_regions.clone(),
                None,
            ));
        }
        let host_regions = vec![
            Region::new("cpu_swap", cpu_swap_bytes, RegionKind::Elastic),
            Region::new("gdn_cpu_spill", gdn_cpu_spill_bytes, RegionKind::Elastic),
            Region::new("ple_table", ple_bytes, RegionKind::Pinned),
        ];
        // The host map's total is the physical host RAM (the sysinfo total, the
        // fixed), not the regions sum (the no the device bug). The host_probe
        // fills it on the first refresh. The 0 initial total (the unknown). The
        // host_total is the regions sum (the no-host-regions check, the no map).
        let host_total = host_regions.iter().map(|r| r.bytes).sum::<u64>();
        let host = if host_total > 0 {
            Some(PhysicalMemoryMap::new("host", 0, host_regions, host_probe))
        } else {
            None
        };
        Self { devices, host }
    }
}

impl Default for MemoryMapSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The priority-cap invariant: each pool is capped at the running remainder,
    /// so the sum of all allocations is at most the total free VRAM. This is the
    /// property that was violated before the budget existed (each allocator read
    /// raw free VRAM independently and over-committed).
    #[test]
    fn budget_caps_sum_within_free_vram() {
        let total_free = 1000;
        let reserve = GpuReserve {
            graph_bytes: 100,
            workspace_bytes: 50,
        };
        let mut budget = PoolBudget::new(total_free, &reserve);
        // The remainder after the measured reserve: 1000 - 100 - 50 = 850.
        assert_eq!(budget.remainder(), 850);
        // Pool 1 (MoE experts) requests 400, gets min(400, 850) = 400.
        let moe = budget.cap(400);
        assert_eq!(moe, 400);
        assert_eq!(budget.remainder(), 450);
        // Pool 2 (LoRA bank) requests 300, gets min(300, 450) = 300.
        let lora = budget.cap(300);
        assert_eq!(lora, 300);
        assert_eq!(budget.remainder(), 150);
        // Pool 3 (GDN slots) requests 150, gets min(150, 150) = 150.
        let gdn = budget.cap(150);
        assert_eq!(gdn, 150);
        assert_eq!(budget.remainder(), 0);
        // Pool 4 (KV cache, the elastic last) takes the remainder: 0.
        let kv = budget.remainder();
        assert_eq!(kv, 0);
        // The invariant: the sum of all allocations + the measured reserve is at
        // most the total free VRAM.
        let total_allocated = reserve.total_bytes() + moe + lora + gdn + kv;
        assert!(total_allocated <= total_free, "the allocations must fit within the free VRAM");
        assert_eq!(total_allocated, total_free, "the budget is fully allocated (the KV takes the remainder)");
    }

    /// When a high-priority pool requests more than the remainder, it is capped
    /// at the remainder (so there is no negative remainder). The lower-priority pools
    /// then see the reduced pool.
    #[test]
    fn budget_caps_oversized_request_at_remainder() {
        let total_free = 1000;
        let reserve = GpuReserve {
            graph_bytes: 100,
            workspace_bytes: 0,
        };
        let mut budget = PoolBudget::new(total_free, &reserve);
        // The remainder is 900. A request for 5000 is capped at 900.
        let granted = budget.cap(5000);
        assert_eq!(granted, 900, "the oversized request is capped at the remainder");
        assert_eq!(budget.remainder(), 0, "the remainder is exhausted");
        // A subsequent request gets 0 (the remainder is exhausted).
        let next = budget.cap(100);
        assert_eq!(next, 0);
    }

    /// A zero free VRAM yields a zero budget (so there is no negative remainder).
    #[test]
    fn budget_zero_free_yields_zero_remainder() {
        let reserve = GpuReserve {
            graph_bytes: 100,
            workspace_bytes: 50,
        };
        let budget = PoolBudget::new(0, &reserve);
        assert_eq!(budget.remainder(), 0, "the remainder floors at 0");
    }

    /// The measured reserve: the graph slice + the model-aware workspace. The
    /// total is what the pools must not consume (so the GEMM workspace does not OOM).
    #[test]
    fn reserve_total_is_graph_plus_workspace() {
        let reserve = GpuReserve {
            graph_bytes: 256,
            workspace_bytes: 512,
        };
        assert_eq!(reserve.total_bytes(), 768);
        // Zero graph (non-graph build) leaves only the workspace.
        let no_graph = GpuReserve {
            graph_bytes: 0,
            workspace_bytes: 512,
        };
        assert_eq!(no_graph.total_bytes(), 512);
    }

    /// The admission-demand floor: the worst-case concurrent KV block demand is
    /// the max_num_seqs × the per-seq blocks (the context + the decode) + the
    /// GDN block-equivalent. The prefix hits reduce the per-request demand at
    /// admission time, so this no-prefix worst case is the safe pool floor.
    #[test]
    fn admission_demand_computes_the_worst_case_concurrent_blocks() {
        // The 4 seqs, each 16 blocks (the 1024 ctx + 256 decode at block 64), no GDN.
        assert_eq!(admission_demand(4, 16, 0), 64);
        // The GDN equivalent adds per-seq (the 2 GDN blocks/seq).
        assert_eq!(admission_demand(4, 16, 2), 72);
        // The saturating mul (no overflow at large max_num_seqs).
        assert_eq!(admission_demand(usize::MAX, 1, 0), usize::MAX);
    }

    /// The R4 oracle: the KV block count is capped at the byte ceiling + raised
    /// to the admission floor, and the floor is always satisfiable (so the MoE
    /// half-reserve does not nullify it). The pure decision, testable without
    /// an EngineConfig.
    #[test]
    fn resolve_kv_block_count_caps_at_ceiling() {
        // The pool (100 blocks) exceeds the byte ceiling (60): capped to 60.
        assert_eq!(resolve_kv_block_count(100, 40, 60), 60);
        // The pool (40) is under the ceiling (60) + under the floor (50): raised
        // to the floor.
        assert_eq!(resolve_kv_block_count(40, 50, 60), 50);
        // The pool (55) is within the floor (50) + ceiling (60): unchanged.
        assert_eq!(resolve_kv_block_count(55, 50, 60), 55);
    }

    #[test]
    fn resolve_kv_block_count_floor_is_satisfiable_when_below_ceiling() {
        // The R4 guarantee: when the admission floor (40) is below the byte
        // ceiling (60), the pool is raised to the floor even if the prior size
        // was smaller (10). The MoE carve (which reduces the ceiling) cannot
        // nullify the floor because the floor is computed against the FULL
        // remainder.
        assert_eq!(resolve_kv_block_count(10, 40, 60), 40);
        // The floor exactly at the ceiling: the pool is raised to the ceiling.
        assert_eq!(resolve_kv_block_count(10, 60, 60), 60);
    }

    #[test]
    fn resolve_kv_block_count_floor_exceeding_ceiling_is_capped() {
        // The admission floor (80) exceeds the byte ceiling (60): the pool is
        // capped at the ceiling (no negative pool, no over-committing
        // the GPU). This is the degenerate case where the demand cannot be met
        // within the bytes — the ceiling wins.
        assert_eq!(resolve_kv_block_count(10, 80, 60), 60);
        assert_eq!(resolve_kv_block_count(100, 80, 60), 60);
    }

    #[test]
    fn resolve_kv_block_count_zero_ceiling_yields_zero() {
        // The byte ceiling is 0 (no free VRAM for KV): the pool is 0, the
        // floor is moot.
        assert_eq!(resolve_kv_block_count(50, 40, 0), 0);
        assert_eq!(resolve_kv_block_count(0, 0, 0), 0);
    }

    #[test]
    fn resolve_kv_block_count_is_monotonic_in_pool_size() {
        // A larger prior pool never yields a smaller result (the cap is
        // monotonic). The floor + ceiling are fixed.
        let (floor, ceiling) = (50, 80);
        let a = resolve_kv_block_count(10, floor, ceiling);
        let b = resolve_kv_block_count(40, floor, ceiling);
        let c = resolve_kv_block_count(70, floor, ceiling);
        let d = resolve_kv_block_count(120, floor, ceiling);
        assert!(a <= b && b <= c && c <= d, "the result is monotonic in the prior pool size: {a} {b} {c} {d}");
        // The 120 is capped at the ceiling (80).
        assert_eq!(d, 80);
    }

    // The PhysicalMemoryMap + the MemoryMapSet (the C2 core, the live physical
    // accounting). These tests run without a GPU (the probe is injected, the
    // computed free is the no-probe fallback).

    #[test]
    fn map_computed_free_is_total_minus_regions() {
        let map = PhysicalMemoryMap::new("rank0", 1000, vec![Region::new("weights", 400, RegionKind::Pinned)], None);
        assert_eq!(map.computed_free(), 600);
        assert_eq!(map.live_free(), 600, "no probe: the live free is the computed free");
    }

    #[test]
    fn map_live_probe_overrides_computed() {
        // The driver says 500 free (the no the computed 600, the drift is the
        // unregistered allocation). The live probe wins after a refresh.
        let mut map = PhysicalMemoryMap::new(
            "rank0",
            1000,
            vec![Region::new("weights", 400, RegionKind::Pinned)],
            Some(Box::new(|| Some((500, 1000)))),
        );
        map.refresh_probe();
        assert_eq!(map.computed_free(), 600);
        assert_eq!(map.live_free(), 500);
        assert_eq!(map.idle(), 500);
    }

    #[test]
    fn map_pressure_is_physical_not_logical() {
        // The 500 live free of 1000 total is a 0.5 pressure (the physical signal,
        // the no the block occupancy).
        let mut map = PhysicalMemoryMap::new(
            "rank0",
            1000,
            vec![Region::new("kv", 500, RegionKind::Elastic)],
            Some(Box::new(|| Some((500, 1000)))),
        );
        map.refresh_probe();
        assert!((map.pressure() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn map_grow_elastic_only_from_computed_free() {
        let mut map = PhysicalMemoryMap::new("rank0", 1000, vec![Region::new("kv", 500, RegionKind::Elastic)], None);
        // The grow 100 fits in the computed free (500).
        assert_eq!(map.grow("kv", 100), Some(600));
        // The grow 500 does not fit (the computed free is now 400).
        assert_eq!(map.grow("kv", 500), None);
        // The pinned region never grows.
        let mut pinned = PhysicalMemoryMap::new("rank0", 1000, vec![Region::new("weights", 100, RegionKind::Pinned)], None);
        assert_eq!(pinned.grow("weights", 10), None);
    }

    #[test]
    fn map_reclaim_elastic_only_to_floor() {
        let mut map = PhysicalMemoryMap::new("rank0", 1000, vec![Region::new("kv", 500, RegionKind::Elastic)], None);
        assert_eq!(map.reclaim("kv", 200), Some(300));
        // The reclaim below the floor (the 300 held, the 400 requested) fails.
        assert_eq!(map.reclaim("kv", 400), None);
        // The pinned region never reclaims.
        let mut pinned = PhysicalMemoryMap::new("rank0", 1000, vec![Region::new("weights", 100, RegionKind::Pinned)], None);
        assert_eq!(pinned.reclaim("weights", 10), None);
    }

    #[test]
    fn set_aggregates_ranks_and_grows_uniformly() {
        let mut set = MemoryMapSet::new();
        // The two ranks (the 2-way TP). The rank 1 is tighter (the less idle):
        // rank 0 has kv=100 (the 900 computed free), rank 1 has kv=900 (the 100
        // computed free).
        set.devices.push(PhysicalMemoryMap::new("rank0", 1000, vec![Region::new("kv", 100, RegionKind::Elastic)], None));
        set.devices.push(PhysicalMemoryMap::new("rank1", 1000, vec![Region::new("kv", 900, RegionKind::Elastic)], None));
        assert_eq!(set.rank_count(), 2);
        // The grow 200 fits on rank 0 (the 900 free) but not rank 1 (the 100
        // free). The uniform grow fails (the tightest rank bounds).
        assert!(!set.grow_all("kv", 200));
        // The grow 50 fits on both (the rank 1 has 100 free).
        assert!(set.grow_all("kv", 50));
        assert_eq!(set.device(0).unwrap().region_bytes("kv"), 150);
        assert_eq!(set.device(1).unwrap().region_bytes("kv"), 950);
        // The device pressure is the max across ranks (the rank 1, the 950/1000).
        assert!((set.device_pressure() - 0.95).abs() < 1e-6);
        // The idle device VRAM is the min across ranks (the rank 1, the 50).
        assert_eq!(set.idle_device(), 50);
    }

    #[test]
    fn set_host_map_and_report() {
        let mut set = MemoryMapSet::new();
        set.host = Some(PhysicalMemoryMap::new("host", 8000, vec![Region::new("cpu_swap", 1000, RegionKind::Elastic)], None));
        assert!(set.host().is_some());
        assert!(set.report().contains("host"));
        set.host = None;
        assert!(set.host().is_none());
    }
}