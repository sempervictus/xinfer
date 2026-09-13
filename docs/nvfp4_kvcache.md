# NVFP4 (UltraQuant-style) Paged KV Cache

Engine-side integration of the attention-rs NVFP4 paged KV cache, completing the
"deferred" verification from `.attention-rs/docs/fp4_kv.md` (numerical accuracy
vs a BF16 reference, throughput, and space).

**Status:** implemented + verified on SM120 (RTX 5090 Laptop, CUDA 13.3).
Decode and prefill paths are accurate (cos_sim ~0.995 vs BF16). All tests
pass with `--features cuda`.

---

## 1. What this is

NVFP4 = **E2M1** 4-bit data (16 levels: {0, .5, 1, 1.5, 2, 3, 4, 6} &times; sign) +
one **E4M3** scale factor per 16-element group (CUTLASS Blackwell block-scaled
FP4, `nv_float4_t`). The KV cache stores K/V in this 4-bit form instead of BF16,
cutting the cache footprint ~3.56x.

The design follows **UltraQuant** (Chakrabarti et al., arXiv:2606.20474v2, 2026):
asymmetric K/V treatment with an optional Walsh-Hadamard (WHT) rotation that
spreads channel outliers so the coarse 4-bit grid is used efficiently. The
rotation is orthogonal (H^T H = I), so attention scores are invariant:
`(H Q) . (H K)^T = Q . K^T`.

The attention-rs kernels (branch `kv/fp4`) implement:
- `flash_nvfp4_kv_store` writes K/V into the paged NVFP4 buffers (E2M1 codes + E4M3 SF).
- `flash_nvfp4_kv_decode` / `flash_nvfp4_kv_prefill` run attention over the 4-bit cache
  (software LUT dequant in registers).

This repo covers the **xInfer engine integration** (`Nvfp4KvCache`) and the
verification results.

## 2. Dependency: attention-rs rev `fb0bc67`

xInfer's `attention-rs` git dependency is pinned to rev `fb0bc67` on
`guoqingbao/attention.rs`, which carries the remedied NVFP4 KV-cache PR
(the `flash_nvfp4_kv_{store,decode,prefill}` kernels plus the store-kernel
byte-addressing and SF read-back fixes). No local `[patch]` is needed; the
rev bump in `Cargo.toml` is the only dependency change:

```toml
attention-rs = { git = "https://github.com/guoqingbao/attention.rs.git",
                 version = "0.6.9", rev = "fb0bc67" }
```

The `nvfp4-kvcache` feature is enabled automatically whenever `cuda` is on
(`cuda = [..., "nvfp4-kvcache"]`), so `--features cuda` builds the module.

### 2.1 FlashInfer pin update (attention-rs `fb0bc67`)

The attention-rs build pins `sempervictus/flashinfer @ 7629d218` (branch
`flashinfer-upstream-20260912`), which carries the SM120 NVFP4 KV-cache PRs
(#3748, #3608, #3640). This is a **temporary** pin (attention-rs
`src/kernels/build.rs`), to be reverted once the PRs land in
`guoqingbao/flashinfer`. It only affects the `--features flashinfer` build;
the native `flash` kernels used by `Nvfp4KvCache` (and by `--features cuda`)
are independent of it.

## 3. Engine integration

`src/models/layers/nvfp4_kvcache.rs` (`Nvfp4KvCache`):
- `new(device, num_blocks, block_size, num_kv_heads, head_dim, c_k, c_v, rotate)`
  allocates the four paged buffers (`k_fp4`/`k_sf`/`v_fp4`/`v_sf`).
- `store(key, value, slot_mapping)` quantizes new KV tokens into the cache.
- `decode(query, block_tables, context_lens, output)` and `prefill(...)` run
  attention over the 4-bit cache.
- `nvfp4_bytes()` / `bf16_bytes()` / `fp8_bytes()` report the cache footprint.

`c_k`/`c_v` are the E4M3 block-scale constants (1.0 = standard NVFP4; <1.0
shrinks the grid per UltraQuant; the paper's c=0.156 is tuned for AMD
UE8M0 and must be re-calibrated for NVIDIA E4M3 per model).

## 4. Store-kernel fixes (committed as `40434dd` on `kv/fp4`)

The shipped `flash_nvfp4_kv_store.cuh` had two bugs that made the 4-bit cache
inaccurate (store dequant cos_sim ~0.14). Both are fixed here:

1. **Intra-head byte addressing.** The kernel wrote FP4 bytes to
   `fp4_off + elem_base + i` (with `elem_base = local*4`, and reading
   `k_reg[elem_base + i]` out-of-bounds for `local > 0`). The decode kernel
   reads `2*lane_id + i/2`. Fixed the store to write `k_reg[i]` to
   `fp4_off + 2*lane_id + i/2`, matching the decode layout.
2. **SF read-back race.** After `local==0` wrote `K_sf[off+g]`, all lanes read
   it back from global memory with no fence, so lanes 1-31 saw the stale 0, and their codes were all 0. Fixed by using the locally-reduced `sf_byte`
   (all group lanes hold the same value after the `shfl` reduction) instead of
   the global read-back.

After both fixes, store dequant cos_sim = **0.995** (K and V).

## 5. Verification results (SM120, `--features cuda`)

Test: `tests/nvfp4_kvcache.rs` (4 tests, all pass). Inputs are deterministic
pseudo-random BF16 in ~[-4, 4] (a realistic attention K/V activation range).

| Metric | Result | Notes |
|---|---|---|
| **Space** | NVFP4 18,432 B vs FP8 32,768 B vs BF16 65,536 B (D=128) | **3.56x vs BF16, 1.78x vs FP8** |
| **Decode accuracy** | cos_sim **0.995** vs BF16 `flash_decode` | stable config `rotate=true, c_k=c_v=1.0` (also 0.996 with `rotate=false`) |
| **Prefill accuracy** | cos_sim **0.994** vs BF16 `flash_prefill` | fixed (see &sect;5.1) |
| **Performance** | NVFP4 decode 1.45 ms/iter vs BF16 1.63 ms/iter (SEQ=64) | software-LUT dequant is comparable on this small config; the win is space (KV capacity), hardware block-scaled MMA is the future latency win |

The recommended **stable-precision** config is `rotate=true, c_k=1.0, c_v=1.0`
(WHT rotation spreads channel outliers so the coarse 4-bit grid is used
efficiently); `Nvfp4KvCache::stable(...)` uses it.

### 5.1 Prefill grid/tiling fix

The prefill kernel originally processed one query token per 32-token tile
(`q_base = q_tile * FP4_PREFILL_TILE`) with a non-ragged Q/O offset
(`seq_idx * q_stride`) and no causal mask, so a multi-token prefill left most
query tokens unwritten (cos_sim ~0.04). Fixed (attention-rs `e5bf798`):
- grid = `(max_q_len, num_q_heads, num_seqs)`; one query token per block.
- ragged Q/O offsets via `cu_seqlens_q` (token = `cu_seqlens_q[seq] + q_local`).
- causal mask: a query token attends only to KV positions
  `[0, q_local + (kv_len - q_len)]`.
- GQA: `kv_head = q_head / (num_q_heads / num_kv_heads)`.

Prefill now matches the BF16 `flash_prefill` reference at cos_sim 0.994.

## 6. How to run

```bash
# from the xinfer repo root. For verification, Cargo.toml carries a temporary
# [patch] pointing attention-rs at the local checkout (with the prefill fix,
# e5bf798); after it is pushed, the [patch] is dropped and the rev is bumped.
cargo test --features cuda --test nvfp4_kvcache -- --nocapture
```

Requires an SM120+ GPU (NVFP4 path) and the CUDA 13.3 toolchain.

## 7. Citations

- **UltraQuant:** Chakrabarti, Limpus, Rana, Bao, Tiwari, Crepaldi, Sirasao,
  "UltraQuant: 4-bit KV Caching for Context-Heavy Agents", arXiv:2606.20474v2
  (2026). Source of the WHT rotation + asymmetric-tensor scale-constant design.
(Note: an earlier note referenced "2606.2704", which resolves to an unrelated
   neutrino-physics paper; the correct ID is **2606.20474**.)
- **TurboQuant:** Zandieh et al., "TurboQuant: Online vector quantization with
  near-optimal distortion rate", ICLR 2026 (WHT rotation + codebook, the
  existing TQ4 path).
- **KIVI:** Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for
  KV Cache", ICML 2024, arXiv:2402.02750 (per-channel K / per-token V).
- **FlashMLA** (DeepSeek-AI) V41_FP4 KV layout; **hikarioyama/vllm-nvfp4-kv-sm120**
  (proven SM120 NVFP4 KV decode); **CUTLASS 4.5.2** SM120 block-scaled GEMM
  (future hardware path).
- **Handoff:** `.attention-rs/docs/fp4_kv.md` and
  `.attention-rs/.xbot/research/32-fp4-kivi-full-history-and-failures.md`.