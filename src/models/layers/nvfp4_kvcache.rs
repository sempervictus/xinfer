//! NVFP4 (UltraQuant-style) paged KV cache - xInfer engine integration.
//!
//! Wraps the attention-rs `flash_nvfp4_kv_{store,decode,prefill}` kernels
//! (branch `kv/fp4`; see `.attention-rs/nvfp4-kvcache-handoff.md`). The engine
//! allocates the paged NVFP4 buffers (`k_fp4`/`k_sf`/`v_fp4`/`v_sf`), stores new
//! KV tokens via the store kernel, and runs decode/prefill attention over the
//! 4-bit cache.
//!
//! NVFP4 layout (per token, per KV head, head_dim = D):
//!   * `*_fp4`: D/2 bytes (2 E2M1 codes/byte)
//!   * `*_sf`:  D/16 bytes (one E4M3 scale per 16-element group)
//!
//! Citations:
//!   * UltraQuant: Chakrabarti et al., "UltraQuant: 4-bit KV Caching for
//!     Context-Heavy Agents", arXiv:2606.20474 (2026)44 KV tensors,
//!     Walsh-Hadamard rotation, group scales, asymmetric K/V.
//!   * NVFP4 = E2M1 (4-bit) + per-16 E4M3 scale factor (CUTLASS Blackwell
//!     block-scaled FP4, `nv_float4_t`).
//!   * Handoff: `.attention-rs/nvfp4-kvcache-handoff.md`.

#[cfg(feature = "nvfp4-kvcache")]
use attention_rs::flash;
use candle_core::{DType, Device, Result, Tensor};

/// Paged NVFP4 KV cache owned by the engine.
#[cfg(feature = "nvfp4-kvcache")]
pub struct Nvfp4KvCache {
    pub k_fp4: Tensor, // [num_blocks, block_size, H, D/2] U8
    pub k_sf: Tensor, // [num_blocks, block_size, H, D/16] U8
    pub v_fp4: Tensor,
    pub v_sf: Tensor,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub c_k: f32,
    pub c_v: f32,
    pub rotate: bool,
    pub device: Device,
}

#[cfg(feature = "nvfp4-kvcache")]
impl Nvfp4KvCache {
    /// Allocate the paged NVFP4 buffers. `c_k`/`c_v` are the E4M3 block-scale
    /// constants (1.0 = standard NVFP4; <1.0 shrinks the grid per UltraQuant);
    /// `rotate` enables Walsh-Hadamard rotation on K (store) and Q (attention).
    pub fn new(
        device: &Device,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        c_k: f32,
        c_v: f32,
        rotate: bool,
    ) -> Result<Self> {
        if head_dim % 16 != 0 {
            candle_core::bail!("Nvfp4KvCache: head_dim {head_dim} must be a multiple of 16");
        }
        let b = num_blocks;
        let s = block_size;
        let h = num_kv_heads;
        let d = head_dim;
        let k_fp4 = Tensor::zeros((b, s, h, d / 2), DType::U8, device)?;
        let k_sf = Tensor::zeros((b, s, h, d / 16), DType::U8, device)?;
        let v_fp4 = Tensor::zeros((b, s, h, d / 2), DType::U8, device)?;
        let v_sf = Tensor::zeros((b, s, h, d / 16), DType::U8, device)?;
        Ok(Self {
            k_fp4,
            k_sf,
            v_fp4,
            v_sf,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            c_k,
            c_v,
            rotate,
            device: device.clone(),
        })
    }

    /// Create a cache with the recommended **stable-precision** config:
/// Walsh-Hadamard rotation on (`rotate=true`) and standard NVFP4 block
/// scales (`c_k = c_v = 1.0`). The WHT rotation spreads per-channel outliers
/// across the coarse 4-bit grid so it is used efficiently, stabilizing
/// accuracy on real model activations (UltraQuant, arXiv:2606.20474).
    pub fn stable(
        device: &Device,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        Self::new(device, num_blocks, block_size, num_kv_heads, head_dim, 1.0, 1.0, true)
    }

    /// Store new KV tokens into the paged NVFP4 cache at `slot_mapping`.
    /// `key`/`value` are `[num_tokens, H, D]` BF16/FP16; `slot_mapping` is
    /// `[num_tokens]` I64 (absolute slot = block*block_size + entry).
    pub fn store(&self, key: &Tensor, value: &Tensor, slot_mapping: &Tensor) -> Result<()> {
        flash::flash_nvfp4_kv_store(
            key,
            value,
            &self.k_fp4,
            &self.k_sf,
            &self.v_fp4,
            &self.v_sf,
            slot_mapping,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            self.c_k,
            self.c_v,
            self.rotate,
        )
    }

    /// Single-token (generation) attention over the NVFP4 cache.
    pub fn decode(
        &self,
        query: &Tensor,
        block_tables: &Tensor,
        context_lens: &Tensor,
        output: &Tensor,
        max_context_len: usize,
        num_q_heads: usize,
        scale: f32,
        softcap: f32,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        flash::flash_nvfp4_kv_decode(
            query,
            &self.k_fp4,
            &self.k_sf,
            &self.v_fp4,
            &self.v_sf,
            block_tables,
            context_lens,
            output,
            max_context_len,
            num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            scale,
            softcap,
            sliding_window,
            self.rotate,
        )
    }

    /// Multi-token (prompt) attention over the NVFP4 cache.
    pub fn prefill(
        &self,
        query: &Tensor,
        block_tables: &Tensor,
        context_lens: &Tensor,
        cu_seqlens_q: &Tensor,
        output: &Tensor,
        max_context_len: usize,
        num_q_heads: usize,
        scale: f32,
        softcap: f32,
        sliding_window: Option<usize>,
    ) -> Result<Tensor> {
        flash::flash_nvfp4_kv_prefill(
            query,
            &self.k_fp4,
            &self.k_sf,
            &self.v_fp4,
            &self.v_sf,
            block_tables,
            context_lens,
            cu_seqlens_q,
            output,
            max_context_len,
            num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            scale,
            softcap,
            sliding_window,
            self.rotate,
        )
    }

    /// Actual NVFP4 cache footprint in bytes (K + V, data + SF).
    pub fn nvfp4_bytes(&self) -> usize {
        let numel = |t: &Tensor| t.dims().iter().product::<usize>();
        numel(&self.k_fp4) + numel(&self.k_sf) + numel(&self.v_fp4) + numel(&self.v_sf)
    }

    /// Equivalent BF16/FP16 paged-cache footprint in bytes (K + V).
    pub fn bf16_bytes(&self) -> usize {
        let n = self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim;
        n * 2 * 2 // K + V, 2 bytes each
    }

    /// Equivalent FP8 (E4M3) paged-cache footprint in bytes (K + V).
    pub fn fp8_bytes(&self) -> usize {
        let n = self.num_blocks * self.block_size * self.num_kv_heads * self.head_dim;
        n * 2 // K + V, 1 byte each
    }
}