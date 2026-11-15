use std::env;
use std::sync::OnceLock;

pub const MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV: &str = "XINFER_MAMBA_SNAPSHOT_STRIDE_BLOCKS";

pub const STREAM_AS_REASONING_CONTENT_ENV: &str = "XINFER_STREAM_AS_REASONING_CONTENT";

pub const SM90_LOWER_PRECISION_GDN_PREFILL_ENV: &str = "SM90_LOWER_PRECISION_GDN_PREFILL";
/// Select the GDN prefill algorithm: `auto`/`fast` uses the FlashInfer SM90
/// chunked kernel when the build and device support it, `exact` forces the
/// reference sequential recurrence.
pub const GDN_PREFILL_ENV: &str = "XINFER_GDN_PREFILL";

pub const MOE_OFFLOAD_ENV: &str = "XINFER_MOE_OFFLOAD";

/// ` number of finished sequences' per-seq stats are retained in the
/// `/metrics` export (the `XINFER_METRICS_SEQ_HISTORY`, the default 16). Bounds
/// the per-seq gauge history so the metrics endpoint doesn't grow unbounded
/// under high request rates.
pub const METRICS_SEQ_HISTORY_ENV: &str = "XINFER_METRICS_SEQ_HISTORY";
pub const METRICS_SEQ_HISTORY_DEFAULT: usize = 16;

static STREAM_AS_REASONING_CONTENT: OnceLock<bool> = OnceLock::new();
static MOE_OFFLOAD: OnceLock<Option<f32>> = OnceLock::new();

pub const MOE_OFFLOAD_PROFILE_ENV: &str = "XINFER_MOE_OFFLOAD_PROFILE";
static MOE_OFFLOAD_PROFILE: OnceLock<bool> = OnceLock::new();

/// Per-call timing breakdown for MoE expert offload (adds device syncs).
pub fn moe_offload_profile() -> bool {
    *MOE_OFFLOAD_PROFILE.get_or_init(|| {
        env::var(MOE_OFFLOAD_PROFILE_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// The number of finished sequences whose per-seq stats are retained in the
/// `/metrics` export (the `XINFER_METRICS_SEQ_HISTORY`, the default 16).
pub fn metrics_seq_history() -> usize {
    env::var(METRICS_SEQ_HISTORY_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(METRICS_SEQ_HISTORY_DEFAULT)
}

/// Fraction (0, 1) of each MoE layer's experts kept on GPU; the rest stay in host RAM.
/// `auto` returns 0.0: layers start with the minimum slots and grow to fill free VRAM
/// after loading (see `moe_offload::budget`).
pub fn moe_offload_rate() -> Option<f32> {
    *MOE_OFFLOAD.get_or_init(|| {
        let raw = env::var(MOE_OFFLOAD_ENV).ok()?;
        match raw.trim().parse::<f32>() {
            _ if raw.trim().eq_ignore_ascii_case("auto") => Some(0.0),
            Ok(r) if r > 0.0 && r < 1.0 => Some(r),
            _ => {
                crate::log_warn!(
                    "Invalid {MOE_OFFLOAD_ENV}='{raw}', expected auto or 0 < value < 1. Offload disabled."
                );
                None
            }
        }
    })
}

pub fn moe_offload_auto() -> bool {
    moe_offload_rate() == Some(0.0)
}

pub const MOE_OFFLOAD_RESERVE_ENV: &str = "XINFER_MOE_OFFLOAD_RESERVE_GB";

/// VRAM (bytes) that `XINFER_MOE_OFFLOAD=auto` leaves free for the KV cache and workspaces.
pub fn moe_offload_reserve_bytes() -> usize {
    let gb = env::var(MOE_OFFLOAD_RESERVE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v >= 0.0)
        .unwrap_or(2.5);
    (gb * (1u64 << 30) as f64) as usize
}
static SM90_LOWER_PRECISION_GDN_PREFILL: OnceLock<bool> = OnceLock::new();

pub fn sm90_lower_precision_gdn_prefill() -> bool {
    *SM90_LOWER_PRECISION_GDN_PREFILL.get_or_init(|| {
        // New, explicit control wins. `auto`/`fast` enable the chunked
        // FlashInfer kernel (the launcher itself falls back when the device
        // or build lacks SM90 support). `exact` keeps the reference
        // recurrence. The legacy variable still overrides the default, so
        // existing deployments that set it keep their behavior.
        if let Ok(v) = env::var(GDN_PREFILL_ENV) {
            let v = v.trim().to_ascii_lowercase();
            return matches!(v.as_str(), "1" | "true" | "yes" | "fast" | "chunked");
        }
        // The chunked FlashInfer kernel is SM90-only and uses lower-precision
        // accumulation than the reference recurrence, so it stays opt-in.
        // It matches the reference to BF16 rounding on typical inputs (p50
        // diff 0, p99 ~0.3% of range) but the differential probe shows
        // non-exact states for some input distributions.
        env::var(SM90_LOWER_PRECISION_GDN_PREFILL_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "TRUE" | "YES"))
            .unwrap_or(false)
    })
}

pub fn stream_as_reasoning_content() -> bool {
    *STREAM_AS_REASONING_CONTENT.get_or_init(|| {
        env::var(STREAM_AS_REASONING_CONTENT_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(true)
    })
}

pub const QOS_ENV: &str = "XINFER_QOS";
static QOS: OnceLock<bool> = OnceLock::new();

/// Master switch for class-aware QoS scheduling (priority admission, adaptive
/// prefill chunking, KV-headroom budgeting). Default off = prior operation.
pub fn qos_enabled() -> bool {
    *QOS.get_or_init(|| {
        env::var(QOS_ENV)
            .map(|v| matches!(v.trim(), "1" | "true" | "yes" | "TRUE" | "YES"))
            .unwrap_or(false)
    })
}

pub const STREAM_RESERVOIR_ENV: &str = "XINFER_STREAM_RESERVOIR";
static STREAM_RESERVOIR: OnceLock<bool> = OnceLock::new();

/// Master switch for the per-stream output reservoir (smooths bursty token
/// delivery). Default on; set XINFER_STREAM_RESERVOIR=0 to restore the legacy
/// immediate-push path.
pub fn stream_reservoir() -> bool {
    *STREAM_RESERVOIR.get_or_init(|| {
        env::var(STREAM_RESERVOIR_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(true)
    })
}

/// Cap on the DFlash projected-hidden context window kept per sequence (in rows).
/// `0` means unbounded full history; set e.g. `XINFER_SPEC_CONTEXT_WINDOW=512` to
/// bound memory on very long generations.
pub const SPEC_CONTEXT_WINDOW_ENV: &str = "XINFER_SPEC_CONTEXT_WINDOW";
static SPEC_CONTEXT_WINDOW: OnceLock<usize> = OnceLock::new();

pub fn spec_context_window() -> usize {
    *SPEC_CONTEXT_WINDOW.get_or_init(|| {
        env::var(SPEC_CONTEXT_WINDOW_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4096) // Default: 4096 (matches DFlash2 training window)
    })
}

/// Opt-out: capture the DFlash draft transformer into a CUDA graph (replayed when
/// the context window is full). Default ON; set `XINFER_SPEC_GRAPH=0` to force the
/// eager draft.
pub const SPEC_GRAPH_ENV: &str = "XINFER_SPEC_GRAPH";
static SPEC_GRAPH: OnceLock<bool> = OnceLock::new();

pub fn spec_graph() -> bool {
    *SPEC_GRAPH.get_or_init(|| {
        !matches!(
            env::var(SPEC_GRAPH_ENV)
                .ok()
                .as_deref()
                .map(|v| v.trim().eq_ignore_ascii_case("0") || v.trim().eq_ignore_ascii_case("false")),
            Some(true)
        )
    })
}

/// Optional override for how many concurrent sequences may draft in parallel
/// (the shared MTP + DFlash parity). When `None` (the default, the env unset),
/// the cap is the admission capacity (`max_num_seqs`) — so if admission is
/// uncapped + parallel drafts are enabled, drafting is uncapped too. When set
/// to `N`, drafting is capped at `N` sequences (the verify batch is `N x (K+1)`
/// tokens, and the MTP + DFlash verify buffers are sized to match, runner.rs).
///
/// Back-compat: `XINFER_SPEC_PARALLEL_DRAFT=1`/`true`/`yes` parses as `Some(1)`
/// (single-seq); a numeric value (e.g. `2`) is `Some(2)`. This replaces the old
/// bool `spec_parallel_draft` + the separate `dflash_parallel_slots` (the latter
/// removed as a dead deprecated alias).
/// Whether grammar VOB masking is offloaded to the CUDA sampler (the mask is passed to
/// `sample_cuda_masked` and applied inside the fused top-k stage) instead of biasing the
/// logits on the CPU via `where_cond`. Default: offload on CUDA builds. Set
/// `XINFER_SPEC_MASK_OFFLOAD=0` to force the CPU (where_cond) path.
pub const SPEC_MASK_OFFLOAD_ENV: &str = "XINFER_SPEC_MASK_OFFLOAD";
static SPEC_MASK_OFFLOAD: OnceLock<bool> = OnceLock::new();

pub fn spec_mask_offload() -> bool {
    *SPEC_MASK_OFFLOAD.get_or_init(|| {
        cfg!(feature = "cuda")
            && !matches!(
                env::var(SPEC_MASK_OFFLOAD_ENV)
                    .ok()
                    .as_deref()
                    .map(|v| v.trim().eq_ignore_ascii_case("0") || v.trim().eq_ignore_ascii_case("false")),
                Some(true)
            )
    })
}

/// Use the VOB bitset path for grammar sampling (8x less data than F32 mask).
/// Requires `XINFER_SPEC_MASK_OFFLOAD=1` (default on CUDA) AND this flag.
pub const VOB_SAMPLING_ENV: &str = "XINFER_VOB_SAMPLING";
static VOB_SAMPLING: OnceLock<bool> = OnceLock::new();

pub fn vob_sampling() -> bool {
    *VOB_SAMPLING.get_or_init(|| {
        cfg!(feature = "cuda")
            && env::var(VOB_SAMPLING_ENV)
                .ok()
                .map(|v| v.trim() == "1" || v.trim().eq_ignore_ascii_case("true"))
                .unwrap_or(false)
    })
}

pub const SPEC_PARALLEL_DRAFT_ENV: &str = "XINFER_SPEC_PARALLEL_DRAFT";
static SPEC_PARALLEL_DRAFT: OnceLock<Option<usize>> = OnceLock::new();

pub fn spec_parallel_draft() -> Option<usize> {
    *SPEC_PARALLEL_DRAFT.get_or_init(|| {
        env::var(SPEC_PARALLEL_DRAFT_ENV)
            .ok()
            .map(|v| {
                let t = v.trim().to_lowercase();
                match t.as_str() {
                    "1" | "true" | "yes" => Some(1), // the legacy bool form = single-seq
                    _ => t.parse::<usize>().ok(),
                }
            })
            .flatten()
    })
}

/// Speculative fast-forward decode: after the sampled base token, append the full
/// grammar-forced (ff) run instead of sampling it one token at a time. Set
/// `XINFER_SPEC_FF=1` to enable (default off).
pub const SPEC_FF_ENV: &str = "XINFER_SPEC_FF";
static SPEC_FF: OnceLock<bool> = OnceLock::new();

pub fn spec_ff() -> bool {
    *SPEC_FF.get_or_init(|| {
        env::var(SPEC_FF_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// The ff-span KV reserve (the positions past the base that the ff fast-forward
/// pre-allocates). The default is 32 (the typical tool-call header, the 10-20
/// forced tokens, with headroom). A user grammar with a longer forced literal
/// prefix (the >32 ff run) can raise this via the env (the no the static cap
/// truncating the ff span). The 0 disables the ff pre-alloc (the ff forward
/// falls back to the on-demand block allocation, the no the reserved span).
pub const FF_RESERVE_ENV: &str = "XINFER_FF_RESERVE";
static FF_RESERVE: OnceLock<Option<usize>> = OnceLock::new();

pub fn ff_reserve() -> Option<usize> {
    *FF_RESERVE.get_or_init(|| {
        env::var(FF_RESERVE_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
    })
}

/// `XINFER_SPEC_REJECTION_SAMPLING=1` enables lossless rejection-sampling verify
/// for speculative decode (the Leviathan algorithm: accept a draft token with
/// probability `min(1, p_target/p_draft)`, on rejection sample the residual
/// `max(0, p_target - p_draft)`). Default OFF: the greedy verify (the argmax
/// accept, the no distribution correction) is used. When ON + the target sampling
/// is non-greedy (temperature > 0), the verify honors the target's sampled
/// distribution (the no biased output). The drafter's per-position logits (the
/// `p_draft`) are required -- the MTP head's `draft_tokens_gpu` returns them.
pub const SPEC_REJECTION_SAMPLING_ENV: &str = "XINFER_SPEC_REJECTION_SAMPLING";
static SPEC_REJECTION_SAMPLING: OnceLock<bool> = OnceLock::new();

pub fn spec_rejection_sampling() -> bool {
    *SPEC_REJECTION_SAMPLING.get_or_init(|| {
        env::var(SPEC_REJECTION_SAMPLING_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_SPEC_DRAFT_MASK=1` enables the draft-side grammar MaskStream: the
/// drafter's per-position proposals are masked by the grammar (the `p_draft`
/// shares support with the masked `p_target`, the high-acceptance lossless
/// verify) + the infallible ff literals are extracted for free (no draft
/// forward spent on them). Default OFF: the prior unmasked draft behavior (full
/// back-compat, the production toggle for QA on the GPU servers). When ON, the
/// MTP draft loop + the DFlash burst are driven by the MaskStream (the single
/// cloned-FSM trajectory, the two consumers: the drafter + the verify).
pub const SPEC_DRAFT_MASK_ENV: &str = "XINFER_SPEC_DRAFT_MASK";
static SPEC_DRAFT_MASK: OnceLock<bool> = OnceLock::new();

pub fn spec_draft_mask() -> bool {
    *SPEC_DRAFT_MASK.get_or_init(|| {
        env::var(SPEC_DRAFT_MASK_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_SPEC_DYNAMIC_K=1` enables the per-sequence, per-step adaptive draft
/// length (the rolling prefix-acceptance-curve knee). Default OFF: the draft
/// length is the fixed `--num-speculative-tokens` ceiling (the prior behavior,
/// the no surprise for existing deployments). When ON, each sequence's K tracks
/// its own drafter reliability (a weak drafter drops K, a strong one holds the
/// ceiling), so the verify-prefill cost is no longer paid for draft positions the
/// drafter can't fill.
pub const SPEC_DYNAMIC_K_ENV: &str = "XINFER_SPEC_DYNAMIC_K";
static SPEC_DYNAMIC_K: OnceLock<bool> = OnceLock::new();

pub fn spec_dynamic_k() -> bool {
    *SPEC_DYNAMIC_K.get_or_init(|| {
        env::var(SPEC_DYNAMIC_K_ENV)
            .ok()
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// When set to "1" or "true", the grammar constrains all generation including
/// reasoning. Default is "0" (two-phase reasoning: unconstrained reasoning,
/// constrained response).
pub const LLG_FULL_ENV: &str = "XINFER_LLG_FULL";
static LLG_FULL: OnceLock<bool> = OnceLock::new();

pub fn llg_full_enabled() -> bool {
    *LLG_FULL.get_or_init(|| {
        env::var(LLG_FULL_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

/// `XINFER_MAX_REASONING_TOKENS=N` sets the reasoning token budget. The budget
/// drives the sampling bias in `apply_reasoning_bias` (the runner): as the
/// per-sequence reasoning token count approaches N, a positive logit bias is
/// added to the end-of-reasoning token, ramping to a strong push at N so the
/// model is steered to end reasoning. This is a sampling intervention, not a
/// grammar constraint — it works with or without a grammar active. The
/// `None` when unset (no budget, no steering).
pub const MAX_REASONING_TOKENS_ENV: &str = "XINFER_MAX_REASONING_TOKENS";
static MAX_REASONING_TOKENS: OnceLock<Option<usize>> = OnceLock::new();

pub fn max_reasoning_tokens() -> Option<usize> {
    *MAX_REASONING_TOKENS.get_or_init(|| {
        env::var(MAX_REASONING_TOKENS_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

/// `XINFER_DEBUG_LLG=1` enables verbose llg (grammar) debug logging (the grammar
/// resyncs, the anti-loop kicks, the grammar init). Default off.
pub const DEBUG_LLG_ENV: &str = "XINFER_DEBUG_LLG";
static DEBUG_LLG: OnceLock<bool> = OnceLock::new();

pub fn debug_llg() -> bool {
    *DEBUG_LLG.get_or_init(|| {
        env::var(DEBUG_LLG_ENV)
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

// the anti-loop detector (the loop-defense)
// The master switch (default OFF = zero overhead, the detector is not even
// constructed) + the probe-forward depth (the confirmation horizon for a stable
// loop). The parsing is factored into pure helpers (the testable primitives);
// the OnceLock getters are the process-wide caching wrappers.

/// `XINFER_ANTI_LOOP=1` enables the in-flight closed-loop detector + the
/// anti-loop kick. Default OFF (the overhead: the detector is not constructed,
/// the OBSERVE/APPLY points are no-ops).
pub const ANTI_LOOP_ENV: &str = "XINFER_ANTI_LOOP";

/// `XINFER_REPETITION_PROBE_DEPTH` (default 512): the probe-forward horizon that
/// confirms a candidate loop period is STABLE (a true closed loop keeps
/// repeating across the depth; a one-off phrase repeat does not).
pub const REPETITION_PROBE_DEPTH_ENV: &str = "XINFER_REPETITION_PROBE_DEPTH";
pub const DEFAULT_REPETITION_PROBE_DEPTH: usize = 512;

static ANTI_LOOP: OnceLock<bool> = OnceLock::new();
static REPETITION_PROBE_DEPTH: OnceLock<usize> = OnceLock::new();

/// Pure parsing primitive (the testable core): the anti-loop flag from the raw
/// env string. Inclusive: "1"/"true"/"yes" (case-insensitive) are on. Exclusive:
/// anything else (including unset) is off.
fn parse_anti_loop_flag(raw: Option<String>) -> bool {
    raw.map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// Pure parsing primitive: the probe depth from the raw env string. Inclusive: a
/// positive integer is used. Exclusive: unset / non-numeric / zero all fall back
/// to the default (512) (a zero depth would confirm nothing).
fn parse_probe_depth(raw: Option<String>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_REPETITION_PROBE_DEPTH)
}

pub fn anti_loop_enabled() -> bool {
    *ANTI_LOOP.get_or_init(|| parse_anti_loop_flag(env::var(ANTI_LOOP_ENV).ok()))
}

pub fn repetition_probe_depth() -> usize {
    *REPETITION_PROBE_DEPTH
        .get_or_init(|| parse_probe_depth(env::var(REPETITION_PROBE_DEPTH_ENV).ok()))
}

pub fn mamba_snapshot_block_stride_blocks(default: usize) -> usize {
    let default = default.max(1);
    let Ok(raw) = env::var(MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV) else {
        return default;
    };
    match raw.trim().parse::<usize>() {
        Ok(0) => {
            crate::log_warn!(
                "{} must be >= 1, got 0. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                default
            );
            default
        }
        Ok(v) => v,
        Err(_) => {
            crate::log_warn!(
                "Invalid {}='{}'. Falling back to default {}.",
                MAMBA_SNAPSHOT_BLOCK_STRIDE_ENV,
                raw,
                default
            );
            default
        }
    }
}

/// Environment variable to disable soft masking for gradient smoothing.
/// When NOT set: soft masking is ENABLED (default behavior).
/// When set to "1", "true", or "yes": soft masking is DISABLED (hard -inf masking).
/// When set to "0", "false", or "no": soft masking is ENABLED.
pub const SOFT_MASK_DISABLED_ENV: &str = "XINFER_SOFT_MASK_DISABLED";

static SOFT_MASK_DISABLED: OnceLock<bool> = OnceLock::new();

pub fn soft_mask_disabled() -> bool {
    *SOFT_MASK_DISABLED.get_or_init(|| {
        env::var(SOFT_MASK_DISABLED_ENV)
            .map(|v| !matches!(v.trim().to_lowercase().as_str(), "0" | "false" | "no"))
            .unwrap_or(false)
    })
}
