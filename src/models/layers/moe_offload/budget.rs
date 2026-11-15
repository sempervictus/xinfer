//! Splits free VRAM between the expert slot cache and the KV cache. With
//! `XINFER_MOE_OFFLOAD=auto` every offloaded layer starts with the minimum number of
//! slots; once all weights are loaded (before the KV cache is planned and CUDA graphs
//! are captured) the slots grow evenly across layers to fill the free VRAM minus
//! `XINFER_MOE_OFFLOAD_RESERVE_GB`, which is left to the KV cache and workspaces.

use candle_core::cuda_backend::cudarc::driver::result;
use candle_core::{CudaDevice, Result};
use std::sync::{Arc, Mutex};

/// One layer's resizable expert slots.
pub trait SlotPool: Send + Sync {
    fn device(&self) -> &CudaDevice;
    fn num_experts(&self) -> usize;
    fn num_slots(&self) -> usize;
    /// VRAM taken by one slot (all expert tensors of one expert).
    fn slot_bytes(&self) -> usize;
    /// Reallocates the slots (empty) with a new count; only before graph capture.
    fn resize(&self, num_slots: usize) -> Result<()>;
}

static POOLS: Mutex<Vec<Arc<dyn SlotPool>>> = Mutex::new(Vec::new());

/// Registers a layer for auto sizing (no-op unless `XINFER_MOE_OFFLOAD=auto`).
pub fn register(pool: Arc<dyn SlotPool>) {
    if crate::utils::env::moe_offload_auto() {
        POOLS.lock().unwrap().push(pool);
    }
}

/// Even water-filling: one more slot per not-yet-full layer while the budget allows.
fn targets(pools: &[(usize, usize, usize)], budget: usize) -> Vec<usize> {
    // (current slots, max slots, bytes per slot)
    let mut slots: Vec<usize> = pools.iter().map(|p| p.0).collect();
    let mut left = budget.saturating_sub(pools.iter().map(|p| p.0 * p.2).sum());
    loop {
        let cost: usize = pools
            .iter()
            .zip(&slots)
            .filter(|(p, s)| **s < p.1)
            .map(|(p, _)| p.2)
            .sum();
        if cost == 0 || cost > left {
            return slots;
        }
        for (p, s) in pools.iter().zip(slots.iter_mut()) {
            if *s < p.1 {
                *s += 1;
            }
        }
        left -= cost;
    }
}

/// Grows every registered layer's slots into the free VRAM; call once after model load.
pub fn grow_to_free_vram() -> Result<()> {
    let pools = std::mem::take(&mut *POOLS.lock().unwrap());
    let Some(first) = pools.first() else {
        return Ok(());
    };
    first
        .device()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let (free, _) = result::mem_get_info().map_err(candle_core::Error::wrap)?;
    let reserve = crate::utils::env::moe_offload_reserve_bytes();
    let used: usize = pools.iter().map(|p| p.num_slots() * p.slot_bytes()).sum();
    let budget = (free + used).saturating_sub(reserve);
    let spec: Vec<_> = pools
        .iter()
        .map(|p| (p.num_slots(), p.num_experts(), p.slot_bytes()))
        .collect();
    let new = targets(&spec, budget);
    for (p, &n) in pools.iter().zip(&new) {
        if n != p.num_slots() {
            p.resize(n)?;
        }
    }
    // Freed slots return to the driver only at a sync; the KV planner must see them.
    first
        .device()
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    let (free_after, _) = result::mem_get_info().map_err(candle_core::Error::wrap)?;
    let (min, max) = (new.iter().min().unwrap(), new.iter().max().unwrap());
    let experts: usize = spec.iter().map(|s| s.1).sum();
    crate::log_info!(
        "MoE offload auto: {}..{} slots per layer ({:.1}% of experts, {:.2} GB), {:.2} GB free for KV (reserve {:.2} GB)",
        min,
        max,
        100.0 * new.iter().sum::<usize>() as f64 / experts as f64,
        new.iter().zip(&spec).map(|(n, s)| n * s.2).sum::<usize>() as f64 / 1e9,
        free_after as f64 / (1u64 << 30) as f64,
        reserve as f64 / (1u64 << 30) as f64
    );
    Ok(())
}

/// Grow the expert slots to fit within a byte budget (the MMU cap). Unlike
/// `grow_to_free_vram` (which reads raw free VRAM), this takes an explicit
/// byte cap from the centralized memory budget so the MoE-offload pool
/// cannot over-commit the GPU alongside the KV + GDN + LoRA pools.
pub fn grow_to_budget(cap_bytes: usize) -> Result<()> {
    let pools = std::mem::take(&mut *POOLS.lock().unwrap());
    let Some(first) = pools.first() else {
        return Ok(());
    };
    first
        .device()
        .bind_to_thread()
        .map_err(candle_core::Error::wrap)?;
    let used: usize = pools.iter().map(|p| p.num_slots() * p.slot_bytes()).sum();
    // The budget is the cap (the MMU slice), not the free VRAM. The water-fill
    // distributes the cap across the pools, respecting each pool's expert count.
    let spec: Vec<_> = pools
        .iter()
        .map(|p| (p.num_slots(), p.num_experts(), p.slot_bytes()))
        .collect();
    let new = targets(&spec, cap_bytes.saturating_sub(used));
    for (p, &n) in pools.iter().zip(&new) {
        if n != p.num_slots() {
            p.resize(n)?;
        }
    }
    // The freed slots return to the driver only at a sync; the KV planner must see them.
    first
        .device()
        .synchronize()
        .map_err(candle_core::Error::wrap)?;
    let (min, max) = (new.iter().min().unwrap(), new.iter().max().unwrap());
    let experts: usize = spec.iter().map(|s| s.1).sum();
    crate::log_info!(
        "MoE offload (budgeted): {}..{} slots per layer (cap {} bytes, {} experts)",
        min,
        max,
        cap_bytes,
        experts
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::targets;

    #[test]
    fn water_fills_evenly_and_respects_caps() {
        // Two layers of 10 bytes/slot; one caps at 3 experts.
        assert_eq!(targets(&[(1, 8, 10), (1, 3, 10)], 20), vec![1, 1]);
        assert_eq!(targets(&[(1, 8, 10), (1, 3, 10)], 60), vec![3, 3]);
        assert_eq!(targets(&[(1, 8, 10), (1, 3, 10)], 75), vec![4, 3]);
        assert_eq!(targets(&[(1, 8, 10), (1, 3, 10)], 1000), vec![8, 3]);
    }
}
