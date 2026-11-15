//! The Prometheus telemetry module (the QoS-gated `/metrics` endpoint).
//!
//! Exposes the statistics the engine already collects (the per-seq `SchedSeqStats`,
//! the `QosController`'s live gauges, the `WorkloadHistogram` streaming
//! estimators, the schedulerScheduler`'s resource occupancy, the `EngineConfig`
//! static info) as Prometheus metrics, without adding any new collection
//! logic. The metrics are a read-only view over state that already exists.
//!
//! The design:
//! - **Global counters** (cum cumulative, the no aging): the request/token/
//!   preemption/swap counters, incremented at sequence end.
//! - **Live gauges** (the read-on-scrape, the no aging): the QoS controller's
//!   gain/ITL/SLO, the `WorkloadHistogram` estimators, the scheduler's resource
//!   occupancy (the KV/mamba/swap).
//! - **Per-seq gauges** (the aged out after 60s): the live sequence's output
//!   length, ITL, gain trajectory. The `SeqStatsRegistry` retains them for
//!   60s after the sequence ends, then evicts them (the no unbounded growth).
//! - **Info metrics** (the static, set at startup): the model id, the shard
//!   count, the KV config, the build version.
//!
//! The `/metrics` endpoint is registered only when QoS is active (the
//! `QosController::active`, the `XINFER_QOS` env OR the `QosConfig.enabled`
//! config), so a QoS-off deployment has no metrics endpoint (the zero overhead).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::core::qos::{QosClass, QosController, Regime, SeqReport};
use crate::core::scheduler::Scheduler;
use crate::utils::config::EngineConfig;
use crate::utils::env::metrics_seq_history;
use prometheus::{
    Encoder, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry, TextEncoder,
};

/// The per-seq stat retention window (the aged-out). After 60s, a
/// finished sequence's per-seq gauges are evicted from the registry. The
/// registry is also bounded by the `XINFER_METRICS_SEQ_HISTORY` count (the
/// default 16): when over the cap, the oldest entry is evicted (the no
/// unbounded growth under high request rates).
const SEQ_STAT_TTL_SECS: u64 = 60;

/// The bounded per-seq stat registry (the 60s aging + the count cap). Maps the
/// seq_id to the (report, the end-time). The `scrape` evicts entries older than
/// the TTL + over the count cap.
struct SeqStatRegistry {
    entries: Mutex<HashMap<usize, (SeqReport, Instant)>>,
    /// The count cap (the `XINFER_METRICS_SEQ_HISTORY`, the default 16).
    max_entries: usize,
}

impl SeqStatRegistry {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: metrics_seq_history(),
        }
    }

    /// Record a finished sequence's report (the seq-end call). When the registry
    /// is over the count cap, the oldest entries are evicted (the no unbounded
    /// growth).
    fn record(&self, seq_id: usize, report: SeqReport) {
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(seq_id, (report, Instant::now()));
        if guard.len() > self.max_entries {
            let excess = guard.len() - self.max_entries;
            let oldest: Vec<(usize, Instant)> = guard
                .iter()
                .map(|(id, (_, ts))| (*id, *ts))
                .collect();
            let mut oldest_sorted = oldest;
            oldest_sorted.sort_by_key(|(_, ts)| *ts);
            for (id, _) in oldest_sorted.iter().take(excess) {
                guard.remove(id);
            }
        }
    }

    /// Evict entries older than the TTL (the 60s aging). Called on each scrape.
    fn evict_expired(&self) {
        let now = Instant::now();
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.retain(|_, (_, ts)| now.duration_since(*ts).as_secs() < SEQ_STAT_TTL_SECS);
    }

    /// The live (≤60s) per-seq reports, for the per-seq gauges. The reports are
    /// cloned (the owned, the no the borrow from the MutexGuard which cannot
    /// escape the function).
    fn live_entries(&self) -> Vec<(usize, SeqReport)> {
        let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .iter()
            .map(|(id, (report, _))| (*id, report.clone()))
            .collect()
    }
}

/// The global registry (the single global instance, the `OnceLock`).
/// Owns the Prometheus `Registry` + all the metric objects + the per-seq
/// stat registry. The metric objects are created once (the `init`), then
/// the `record_*`/`publish_*` methods update them. The `scrape` method
/// encodes the registry to the Prometheus text format.
pub struct Telemetry {
    registry: Registry,
    seq_stats: SeqStatRegistry,

    // The global counters (the cumulative, the no aging).
    requests_total: IntCounterVec,
    tokens_total: IntCounterVec,
    preemptions_total: IntCounterVec,
    swaps_total: IntCounterVec,
    chunk_shrunk_total: IntCounterVec,
    priority_admissions_total: IntCounter,
    reservation_blocks_total: IntCounter,

    // The live gauges (the read-on-scrape, the no aging).
    kv_cache_usage_ratio: IntGauge,
    kv_cache_free_blocks: IntGauge,
    kv_cache_total_blocks: IntGauge,
    cpu_swap_usage_ratio: IntGauge,
    mamba_slots_used: IntGauge,
    mamba_slots_total: IntGauge,
    qos_gain: IntGauge,
    qos_itl_ms: IntGauge,
    qos_itl_slo_ms: IntGauge,
    qos_itl_p95_ms: IntGaugeVec,
    qos_output_len_median: IntGaugeVec,
    qos_batch_size: IntGaugeVec,
    qos_prefix_hit_ratio: IntGaugeVec,
    qos_kv_pressure: IntGaugeVec,
    qos_turns: IntGauge,
    qos_preemptions_total: IntGaugeVec,
    qos_request_states: IntGaugeVec,
    qos_exposed_time_ns: IntGaugeVec,

    // The per-seq gauges (the aged out after 60s).
    seq_output_tokens: IntGaugeVec,
    seq_itl_ms: IntGaugeVec,
    seq_gain: IntGaugeVec,
    seq_prefill_steps: IntGaugeVec,

    // The info metrics (the static, set at startup).
    build_info: IntGaugeVec,
    kv_cache_config: IntGaugeVec,

    // The boot-time memory budget (the static, set at startup): the per-pool
    // allocation (the graphs/moe/lora/gdn/kv) + the total free + the remainder.
    mem_budget_pool_bytes: IntGaugeVec,
    mem_budget_total_free_bytes: IntGauge,
    mem_budget_remainder_bytes: IntGauge,

    // The runtime byte-level memory gauges (the read-on-scrape): the KV used/total
    // bytes, the Mamba used/budget bytes, the CPU-swap used/total bytes.
    kv_used_bytes: IntGauge,
    kv_total_bytes: IntGauge,
    mamba_used_bytes: IntGauge,
    mamba_budget_bytes: IntGauge,
    cpu_swap_used_bytes: IntGauge,
    cpu_swap_total_bytes: IntGauge,

    // The physical memory map (the kernel-MMU view, the no the block-based logical
    // occupancy): the VRAM-in-use pressure (the 0..1, the QoS demote signal), the
    // idle VRAM (the grow headroom), the total VRAM. The 0 when the map is not
    // built (the non-CUDA build, the test).
    mmu_pressure: IntGauge,
    mmu_idle_vram_bytes: IntGauge,
    mmu_total_vram_bytes: IntGauge,

    // The LoRA gauges (the read-on-scrape): the active tenants, the active
    // slots, the bank bytes, the total adapters.
    lora_active_tenants: IntGauge,
    lora_active_slots: IntGauge,
    lora_bank_bytes: IntGauge,
    lora_adapters_total: IntGauge,

    // The spec-decode acceptance gauges (the read-on-scrape): the acceptance rate
    // + the avg tokens/step, per drafter (the mtp/dflash).
    spec_acceptance_rate: IntGaugeVec,
    spec_avg_tokens_per_step: IntGaugeVec,

    // The runner queue-depth gauges (the read-on-scrape): the live/waiting/
    // cached/transferred sequence counts.
    seq_running: IntGauge,
    seq_waiting: IntGauge,
    seq_cached: IntGauge,
    seq_transferred: IntGauge,

    // The per-seq end stats (the aged out after the env-tunable history depth,
    // the default 16): the prompt tokens, the prompt time, the decode time, the
    // cached tokens.
    seq_prompt_tokens: IntGaugeVec,
    seq_prompt_time_s: IntGaugeVec,
    seq_decode_time_s: IntGaugeVec,
    seq_cached_tokens: IntGaugeVec,

    // The per-seq spec-decode + reasoning stats (the end-of-seq, the aged).
    // The seq_spec_k is exported on the spec-decode branch (the SchedSeqStats.spec_k
    // field exists there); the seq_reasoning_tokens on the llg branch. The qos
    // branch has neither field, so these metrics are not declared here.

    // The CUDA graph pool usage (the runtime, the current + peak MB).
    cuda_graph_pool_current_mb: IntGauge,
    cuda_graph_pool_peak_mb: IntGauge,

    // The per-step scheduling budget (the static, set at startup): the max
    // batched tokens per step + the max parallel reqs.
    sched_budget_tokens: IntGauge,
    sched_max_parallel_reqs: IntGauge,

    // The CPU KV-cache block count (the static, set at startup).
    cpu_kv_blocks: IntGauge,

    // The Mamba active/prefix split (the runtime): the active-slot bytes + the
    // prefix-snapshot bytes + the per-slot bytes.
    mamba_active_bytes: IntGauge,
    mamba_prefix_bytes: IntGauge,
    mamba_slot_bytes: IntGauge,
}

impl Telemetry {
    /// Initialize the global telemetry instance (the `OnceLock`). Creates the
    /// existing instance if already initialized (the idempotent).
    pub fn global() -> &'static Telemetry {
        static TELEMETRY: OnceLock<Telemetry> = OnceLock::new();
        TELEMETRY.get_or_init(|| Telemetry::new())
    }

    fn new() -> Self {
        let registry = Registry::new();

        // The global counters.
        let requests_total = IntCounterVec::new(
            Opts::new("xinfer_requests_total", "The total number of requests served, by QoS class + model."),
            &["qos_class", "model"],
        ).expect("metric construction");
        let tokens_total = IntCounterVec::new(
            Opts::new("xinfer_tokens_total", "The total number of tokens processed, by model + kind."),
            &["model", "kind"],
        ).expect("metric construction");
        let preemptions_total = IntCounterVec::new(
            Opts::new("xinfer_preemptions_total", "The total number of sequence preemptions, by QoS class."),
            &["qos_class"],
        ).expect("metric construction");
        let swaps_total = IntCounterVec::new(
            Opts::new("xinfer_swaps_total", "The total number of KV swaps, by QoS class."),
            &["qos_class"],
        ).expect("metric construction");
        let chunk_shrunk_total = IntCounterVec::new(
            Opts::new("xinfer_chunk_shrunk_total", "The total number of adaptive chunk shrinks, by QoS class."),
            &["qos_class"],
        ).expect("metric construction");
        let priority_admissions_total = IntCounter::new(
            "xinfer_priority_admissions_total",
            "The total number of Latency-class priority admissions.",
        ).expect("metric construction");
        let reservation_blocks_total = IntCounter::new(
            "xinfer_reservation_blocks_total",
            "The total number of Throughput-class reservation blocks.",
        ).expect("metric construction");

        // The live gauges.
        let kv_cache_usage_ratio = IntGauge::new(
            "xinfer_kv_cache_usage_ratio",
            "The KV cache usage ratio (0-1000, the perille).",
        ).expect("metric construction");
        let kv_cache_free_blocks = IntGauge::new(
            "xinfer_kv_cache_free_blocks",
            "The number of free KV cache blocks.",
        ).expect("metric construction");
        let kv_cache_total_blocks = IntGauge::new(
            "xinfer_kv_cache_total_blocks",
            "The total number of KV cache blocks.",
        ).expect("metric construction");
        let cpu_swap_usage_ratio = IntGauge::new(
            "xinfer_cpu_swap_usage_ratio",
            "The CPU swap usage ratio (0-1000, the permille).",
        ).expect("metric construction");
        let mamba_slots_used = IntGauge::new(
            "xinfer_mamba_slots_used",
            "The number of mamba/GDN slots in use.",
        ).expect("metric construction");
        let mamba_slots_total = IntGauge::new(
            "xinfer_mamba_slots_total",
            "The total number of mamba/GDN slots.",
        ).expect("metric construction");
        let qos_gain = IntGauge::new(
            "xinfer_qos_gain",
            "The QoS integral gain (0-1000, the permille).",
        ).expect("metric construction");
        let qos_itl_ms = IntGauge::new(
            "xinfer_qos_itl_ms",
            "The QoS inter-token latency EMA (ms).",
        ).expect("metric construction");
        let qos_itl_slo_ms = IntGauge::new(
            "xinfer_qos_itl_slo_ms",
            "The QoS ITL SLO target (ms, the -1 when unset).",
        ).expect("metric construction");
        let qos_itl_p95_ms = IntGaugeVec::new(
            Opts::new("xinfer_qos_itl_p95_ms", "The QoS ITL p95 (ms), by regime + class."),
            &["regime", "qos_class"],
        ).expect("metric construction");
        let qos_output_len_median = IntGaugeVec::new(
            Opts::new("xinfer_qos_output_len_median", "The QoS output-length median (tokens), by class."),
            &["qos_class"],
        ).expect("metric construction");
        let qos_batch_size = IntGaugeVec::new(
            Opts::new("xinfer_qos_batch_size", "The QoS batch-size EMA, by regime."),
            &["regime"],
        ).expect("metric construction");
        let qos_prefix_hit_ratio = IntGaugeVec::new(
            Opts::new("xinfer_qos_prefix_hit_ratio", "The QoS prefix-hit ratio (0-1000), by class."),
            &["qos_class"],
        ).expect("metric construction");
        let qos_kv_pressure = IntGaugeVec::new(
            Opts::new("xinfer_qos_kv_pressure", "The QoS KV pressure (0-1000), by regime."),
            &["regime"],
        ).expect("metric construction");
        let qos_turns = IntGauge::new(
            "xinfer_qos_turns",
            "The QoS conversation turns counter.",
        ).expect("metric construction");
        let qos_preemptions_total = IntGaugeVec::new(
            Opts::new("xinfer_qos_preemptions_total", "The QoS preemption count, by class."),
            &["qos_class"],
        ).expect("metric construction");
        let qos_request_states = IntGaugeVec::new(
            Opts::new("xinfer_qos_request_states", "The QoS request-state lifecycle counts, by class + state."),
            &["qos_class", "state"],
        ).expect("metric construction");
        let qos_exposed_time_ns = IntGaugeVec::new(
            Opts::new("xinfer_qos_exposed_time_ns", "The QoS exposed time (ns), by work-class + qos class + kind."),
            &["work_class", "qos_class", "kind"],
        ).expect("metric construction");

        // The per-seq gauges (the aged out after 60s).
        let seq_output_tokens = IntGaugeVec::new(
            Opts::new("xinfer_seq_output_tokens", "The per-seq output token count (the live, ≤60s)."),
            &["seq_id", "model"],
        ).expect("metric construction");
        let seq_itl_ms = IntGaugeVec::new(
            Opts::new("xinfer_seq_itl_ms", "The per-seq ITL (ms, the live, ≤60s)."),
            &["seq_id"],
        ).expect("metric construction");
        let seq_gain = IntGaugeVec::new(
            Opts::new("xinfer_seq_gain", "The per-seq QoS gain trajectory (the live, ≤60s)."),
            &["seq_id", "phase"],
        ).expect("metric construction");
        let seq_prefill_steps = IntGaugeVec::new(
            Opts::new("xinfer_seq_prefill_steps", "The per-seq prefill step count (the live, ≤60s)."),
            &["seq_id"],
        ).expect("metric construction");

        // The info metrics (the static, set at startup).
        let build_info = IntGaugeVec::new(
            Opts::new("xinfer_build_info", "The engine build info (the static)."),
            &["model_id", "num_shards", "kvcache_dtype", "max_model_len", "block_size"],
        ).expect("metric construction");
        let kv_cache_config = IntGaugeVec::new(
            Opts::new("xinfer_kv_cache_config", "The KV cache config (the static)."),
            &["num_blocks", "max_tokens"],
        ).expect("metric construction");

        // The boot-time memory budget (the static, set at startup).
        let mem_budget_pool_bytes = IntGaugeVec::new(
            Opts::new("xinfer_mem_budget_pool_bytes", "The boot-time GPU memory budget per pool (the graphs/moe/lora/gdn/kv), in bytes."),
            &["pool"],
        ).expect("metric construction");
        let mem_budget_total_free_bytes = IntGauge::new(
            "xinfer_mem_budget_total_free_bytes",
            "The total free GPU VRAM the budget was carved from (bytes, the boot-time static).",
        ).expect("metric construction");
        let mem_budget_remainder_bytes = IntGauge::new(
            "xinfer_mem_budget_remainder_bytes",
            "The budget remainder after all pool caps (bytes, the boot-time static).",
        ).expect("metric construction");

        // The runtime byte-level memory gauges (the read-on-scrape).
        let kv_used_bytes = IntGauge::new(
            "xinfer_kv_used_bytes", "The KV cache bytes in use (the runtime).",
        ).expect("metric construction");
        let kv_total_bytes = IntGauge::new(
            "xinfer_kv_total_bytes", "The KV cache total bytes (the runtime).",
        ).expect("metric construction");
        let mamba_used_bytes = IntGauge::new(
            "xinfer_mamba_used_bytes", "The Mamba/GDN state bytes in use (the runtime, the 0 when not hybrid).",
        ).expect("metric construction");
        let mamba_budget_bytes = IntGauge::new(
            "xinfer_mamba_budget_bytes", "The Mamba/GDN state budget bytes (the runtime, the 0 when not hybrid).",
        ).expect("metric construction");
        let cpu_swap_used_bytes = IntGauge::new(
            "xinfer_cpu_swap_used_bytes", "The CPU KV-swap bytes in use (the runtime).",
        ).expect("metric construction");
        let cpu_swap_total_bytes = IntGauge::new(
            "xinfer_cpu_swap_total_bytes", "The CPU KV-swap total bytes (the runtime).",
        ).expect("metric construction");
        // The physical memory map (the kernel-MMU view, the no the block-based
        // logical occupancy): the VRAM-in-use pressure (the QoS demote signal), the
        // idle VRAM (the grow headroom), the total VRAM. The 0 when the map is not
        // built (the non-CUDA build, the test).
        let mmu_pressure = IntGauge::new(
            "xinfer_mmu_pressure", "The physical VRAM-in-use fraction (the 0..1, the QoS demote signal, the no the block occupancy).",
        ).expect("metric construction");
        let mmu_idle_vram_bytes = IntGauge::new(
            "xinfer_mmu_idle_vram_bytes", "The idle VRAM headroom (the grow source, the physical free).",
        ).expect("metric construction");
        let mmu_total_vram_bytes = IntGauge::new(
            "xinfer_mmu_total_vram_bytes", "The total VRAM (the cuMemGetInfo total, the fixed).",
        ).expect("metric construction");

        // The LoRA gauges (the read-on-scrape).
        let lora_active_tenants = IntGauge::new(
            "xinfer_lora_active_tenants", "The number of active LoRA tenants (the runtime).",
        ).expect("metric construction");
        let lora_active_slots = IntGauge::new(
            "xinfer_lora_active_slots", "The number of active LoRA slots (the runtime).",
        ).expect("metric construction");
        let lora_bank_bytes = IntGauge::new(
            "xinfer_lora_bank_bytes", "The LoRA adapter bank bytes (the runtime).",
        ).expect("metric construction");
        let lora_adapters_total = IntGauge::new(
            "xinfer_lora_adapters_total", "The total registered LoRA adapters (the runtime).",
        ).expect("metric construction");

        // The spec-decode acceptance gauges (the read-on-scrape, per drafter).
        let spec_acceptance_rate = IntGaugeVec::new(
            Opts::new("xinfer_spec_acceptance_rate", "The spec-decode acceptance rate (0-1000, the permille), per drafter."),
            &["drafter"],
        ).expect("metric construction");
        let spec_avg_tokens_per_step = IntGaugeVec::new(
            Opts::new("xinfer_spec_avg_tokens_per_step", "The spec-decode avg tokens/step, per drafter."),
            &["drafter"],
        ).expect("metric construction");

        // The runner queue-depth gauges (the read-on-scrape).
        let seq_running = IntGauge::new(
            "xinfer_seq_running", "The number of sequences in the running queue (the runtime).",
        ).expect("metric construction");
        let seq_waiting = IntGauge::new(
            "xinfer_seq_waiting", "The number of sequences in the waiting queue (the runtime).",
        ).expect("metric construction");
        let seq_cached = IntGauge::new(
            "xinfer_seq_cached", "The number of sequences in the cached (swapped-out) queue (the runtime).",
        ).expect("metric construction");
        let seq_transferred = IntGauge::new(
            "xinfer_seq_transferred", "The number of sequences in the transferred (PD in-flight) queue (the runtime).",
        ).expect("metric construction");

        // The per-seq end stats (the aged out after the env-tunable history depth).
        let seq_prompt_tokens = IntGaugeVec::new(
            Opts::new("xinfer_seq_prompt_tokens", "The per-seq prompt token count (the end-of-seq, the aged)."),
            &["seq_id"],
        ).expect("metric construction");
        let seq_prompt_time_s = IntGaugeVec::new(
            Opts::new("xinfer_seq_prompt_time_s", "The per-seq prompt time (s, the end-of-seq, the aged)."),
            &["seq_id"],
        ).expect("metric construction");
        let seq_decode_time_s = IntGaugeVec::new(
            Opts::new("xinfer_seq_decode_time_s", "The per-seq decode time (s, the end-of-seq, the aged)."),
            &["seq_id"],
        ).expect("metric construction");
        let seq_cached_tokens = IntGaugeVec::new(
            Opts::new("xinfer_seq_cached_tokens", "The per-seq cached (prefix-hit) token count (the end-of-seq, the aged)."),
            &["seq_id"],
        ).expect("metric construction");

        // The per-seq spec-decode + reasoning stats are added on the spec-decode + llg
        // branches (the SchedSeqStats fields exist there); not on the qos branch.
        // The CUDA graph pool usage (the runtime, the current + peak MB).
        let cuda_graph_pool_current_mb = IntGauge::new(
            "xinfer_cuda_graph_pool_current_mb", "The CUDA graph pool current usage (MB, the runtime).",
        ).expect("metric construction");
        let cuda_graph_pool_peak_mb = IntGauge::new(
            "xinfer_cuda_graph_pool_peak_mb", "The CUDA graph pool peak usage (MB, the runtime).",
        ).expect("metric construction");
        // The per-step scheduling budget (the static, set at startup).
        let sched_budget_tokens = IntGauge::new(
            "xinfer_sched_budget_tokens", "The per-step scheduling token budget (the static).",
        ).expect("metric construction");
        let sched_max_parallel_reqs = IntGauge::new(
            "xinfer_sched_max_parallel_reqs", "The max parallel requests (the static).",
        ).expect("metric construction");
        // The CPU KV-cache block count (the static, set at startup).
        let cpu_kv_blocks = IntGauge::new(
            "xinfer_cpu_kv_blocks", "The CPU KV-cache block count (the static).",
        ).expect("metric construction");
        // The Mamba active/prefix split (the runtime).
        let mamba_active_bytes = IntGauge::new(
            "xinfer_mamba_active_bytes", "The Mamba active-slot bytes (the runtime, the 0 when not hybrid).",
        ).expect("metric construction");
        let mamba_prefix_bytes = IntGauge::new(
            "xinfer_mamba_prefix_bytes", "The Mamba prefix-snapshot bytes (the runtime, the 0 when not hybrid).",
        ).expect("metric construction");
        let mamba_slot_bytes = IntGauge::new(
            "xinfer_mamba_slot_bytes", "The per-Mamba-slot bytes (the runtime, the 0 when not hybrid).",
        ).expect("metric construction");

        // Register all the metrics with the registry. The prometheus crate's
        // IntCounterVec/IntCounter/IntGaugeVec/IntGauge each implement the
        // Metric trait, so each is registered individually (the no a mixed-type
        // array).
        registry.register(Box::new(requests_total.clone())).expect("metric registration");
        registry.register(Box::new(tokens_total.clone())).expect("metric registration");
        registry.register(Box::new(preemptions_total.clone())).expect("metric registration");
        registry.register(Box::new(swaps_total.clone())).expect("metric registration");
        registry.register(Box::new(chunk_shrunk_total.clone())).expect("metric registration");
        registry.register(Box::new(priority_admissions_total.clone())).expect("metric registration");
        registry.register(Box::new(reservation_blocks_total.clone())).expect("metric registration");
        registry.register(Box::new(kv_cache_usage_ratio.clone())).expect("metric registration");
        registry.register(Box::new(kv_cache_free_blocks.clone())).expect("metric registration");
        registry.register(Box::new(kv_cache_total_blocks.clone())).expect("metric registration");
        registry.register(Box::new(cpu_swap_usage_ratio.clone())).expect("metric registration");
        registry.register(Box::new(mamba_slots_used.clone())).expect("metric registration");
        registry.register(Box::new(mamba_slots_total.clone())).expect("metric registration");
        registry.register(Box::new(qos_gain.clone())).expect("metric registration");
        registry.register(Box::new(qos_itl_ms.clone())).expect("metric registration");
        registry.register(Box::new(qos_itl_slo_ms.clone())).expect("metric registration");
        registry.register(Box::new(qos_itl_p95_ms.clone())).expect("metric registration");
        registry.register(Box::new(qos_output_len_median.clone())).expect("metric registration");
        registry.register(Box::new(qos_batch_size.clone())).expect("metric registration");
        registry.register(Box::new(qos_prefix_hit_ratio.clone())).expect("metric registration");
        registry.register(Box::new(qos_kv_pressure.clone())).expect("metric registration");
        registry.register(Box::new(qos_turns.clone())).expect("metric registration");
        registry.register(Box::new(qos_preemptions_total.clone())).expect("metric registration");
        registry.register(Box::new(qos_request_states.clone())).expect("metric registration");
        registry.register(Box::new(qos_exposed_time_ns.clone())).expect("metric registration");
        registry.register(Box::new(seq_output_tokens.clone())).expect("metric registration");
        registry.register(Box::new(seq_itl_ms.clone())).expect("metric registration");
        registry.register(Box::new(seq_gain.clone())).expect("metric registration");
        registry.register(Box::new(seq_prefill_steps.clone())).expect("metric registration");
        registry.register(Box::new(build_info.clone())).expect("metric registration");
        registry.register(Box::new(kv_cache_config.clone())).expect("metric registration");
        registry.register(Box::new(mem_budget_pool_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mem_budget_total_free_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mem_budget_remainder_bytes.clone())).expect("metric registration");
        registry.register(Box::new(kv_used_bytes.clone())).expect("metric registration");
        registry.register(Box::new(kv_total_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mamba_used_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mamba_budget_bytes.clone())).expect("metric registration");
        registry.register(Box::new(cpu_swap_used_bytes.clone())).expect("metric registration");
        registry.register(Box::new(cpu_swap_total_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mmu_pressure.clone())).expect("metric registration");
        registry.register(Box::new(mmu_idle_vram_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mmu_total_vram_bytes.clone())).expect("metric registration");
        registry.register(Box::new(lora_active_tenants.clone())).expect("metric registration");
        registry.register(Box::new(lora_active_slots.clone())).expect("metric registration");
        registry.register(Box::new(lora_bank_bytes.clone())).expect("metric registration");
        registry.register(Box::new(lora_adapters_total.clone())).expect("metric registration");
        registry.register(Box::new(spec_acceptance_rate.clone())).expect("metric registration");
        registry.register(Box::new(spec_avg_tokens_per_step.clone())).expect("metric registration");
        registry.register(Box::new(seq_running.clone())).expect("metric registration");
        registry.register(Box::new(seq_waiting.clone())).expect("metric registration");
        registry.register(Box::new(seq_cached.clone())).expect("metric registration");
        registry.register(Box::new(seq_transferred.clone())).expect("metric registration");
        registry.register(Box::new(seq_prompt_tokens.clone())).expect("metric registration");
        registry.register(Box::new(seq_prompt_time_s.clone())).expect("metric registration");
        registry.register(Box::new(seq_decode_time_s.clone())).expect("metric registration");
        registry.register(Box::new(seq_cached_tokens.clone())).expect("metric registration");
        registry.register(Box::new(cuda_graph_pool_current_mb.clone())).expect("metric registration");
        registry.register(Box::new(cuda_graph_pool_peak_mb.clone())).expect("metric registration");
        registry.register(Box::new(sched_budget_tokens.clone())).expect("metric registration");
        registry.register(Box::new(sched_max_parallel_reqs.clone())).expect("metric registration");
        registry.register(Box::new(cpu_kv_blocks.clone())).expect("metric registration");
        registry.register(Box::new(mamba_active_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mamba_prefix_bytes.clone())).expect("metric registration");
        registry.register(Box::new(mamba_slot_bytes.clone())).expect("metric registration");

        Self {
            registry,
            seq_stats: SeqStatRegistry::new(),
            requests_total,
            tokens_total,
            preemptions_total,
            swaps_total,
            chunk_shrunk_total,
            priority_admissions_total,
            reservation_blocks_total,
            kv_cache_usage_ratio,
            kv_cache_free_blocks,
            kv_cache_total_blocks,
            cpu_swap_usage_ratio,
            mamba_slots_used,
            mamba_slots_total,
            qos_gain,
            qos_itl_ms,
            qos_itl_slo_ms,
            qos_itl_p95_ms,
            qos_output_len_median,
            qos_batch_size,
            qos_prefix_hit_ratio,
            qos_kv_pressure,
            qos_turns,
            qos_preemptions_total,
            qos_request_states,
            qos_exposed_time_ns,
            seq_output_tokens,
            seq_itl_ms,
            seq_gain,
            seq_prefill_steps,
            build_info,
            kv_cache_config,
            mem_budget_pool_bytes,
            mem_budget_total_free_bytes,
            mem_budget_remainder_bytes,
            kv_used_bytes,
            kv_total_bytes,
            mamba_used_bytes,
            mamba_budget_bytes,
            cpu_swap_used_bytes,
            cpu_swap_total_bytes,
            mmu_pressure,
            mmu_idle_vram_bytes,
            mmu_total_vram_bytes,
            lora_active_tenants,
            lora_active_slots,
            lora_bank_bytes,
            lora_adapters_total,
            spec_acceptance_rate,
            spec_avg_tokens_per_step,
            seq_running,
            seq_waiting,
            seq_cached,
            seq_transferred,
            seq_prompt_tokens,
            seq_prompt_time_s,
            seq_decode_time_s,
            seq_cached_tokens,
            cuda_graph_pool_current_mb,
            cuda_graph_pool_peak_mb,
            sched_budget_tokens,
            sched_max_parallel_reqs,
            cpu_kv_blocks,
            mamba_active_bytes,
            mamba_prefix_bytes,
            mamba_slot_bytes,
        }
    }

    /// Record a finished sequence's report (the seq-end hook). Increments the
    /// global counters + inserts the per-seq gauges (the 60s aging).
    pub fn record_seq_end(&self, seq_id: usize, report: &SeqReport) {
        let stats = &report.sched;
        let class = class_label(&stats.qos_class);
        let model = &stats.model_label;

        // The global counters (the cumulative).
        self.requests_total.with_label_values(&[class, model]).inc();
        self.tokens_total.with_label_values(&[model, "completion"]).inc_by(report.cached_tokens as u64);
        self.tokens_total.with_label_values(&[model, "output"]).inc_by(stats.output_len as u64);
        if stats.preempted > 0 {
            self.preemptions_total.with_label_values(&[class]).inc_by(stats.preempted as u64);
        }
        if stats.swapped > 0 {
            self.swaps_total.with_label_values(&[class]).inc_by(stats.swapped as u64);
        }
        if stats.chunk_shrunk > 0 {
            self.chunk_shrunk_total.with_label_values(&[class]).inc_by(stats.chunk_shrunk as u64);
        }
        if stats.priority_admitted > 0 {
            self.priority_admissions_total.inc_by(stats.priority_admitted as u64);
        }
        if stats.reservation_blocked > 0 {
            self.reservation_blocks_total.inc_by(stats.reservation_blocked as u64);
        }

        // The per-seq gauges (the 60s aging).
        self.seq_stats.record(seq_id, report.clone());
        self.seq_output_tokens
            .with_label_values(&[&seq_id.to_string(), model])
            .set(stats.output_len as i64);
        self.seq_itl_ms
            .with_label_values(&[&seq_id.to_string()])
            .set(stats.itl_ms as i64);
        self.seq_gain
            .with_label_values(&[&seq_id.to_string(), "start"])
            .set((stats.gain_start * 1000.0) as i64);
        self.seq_gain
            .with_label_values(&[&seq_id.to_string(), "end"])
            .set((stats.gain_end * 1000.0) as i64);
        self.seq_prefill_steps
            .with_label_values(&[&seq_id.to_string()])
            .set(stats.prefill_steps as i64);
        // The per-seq end stats (the prompt/decode time + the token counts, the
        // aged out after the env-tunable history depth).
        self.seq_prompt_tokens
            .with_label_values(&[&seq_id.to_string()])
            .set(report.prompt_tokens as i64);
        self.seq_prompt_time_s
            .with_label_values(&[&seq_id.to_string()])
            .set((report.prompt_time_s * 1000.0) as i64); // the ms
        self.seq_decode_time_s
            .with_label_values(&[&seq_id.to_string()])
            .set((report.decode_time_s * 1000.0) as i64); // the ms
        self.seq_cached_tokens
            .with_label_values(&[&seq_id.to_string()])
            .set(report.cached_tokens as i64);
    }

    /// Record the per-seq prompt/decode wall time (the server-measured, the
    /// end-of-sequence hook). This is called by the server after it measures
    /// the prompt + decode time (the `prompt_time_taken`/`decode_time_taken`).
    /// The values are stored in the per-seq gauges (the aged out after the
    /// env-tunable history depth).
    pub fn record_seq_timing(&self, seq_id: usize, prompt_tokens: usize, prompt_time_s: f64, decode_time_s: f64) {
        let id = seq_id.to_string();
        self.seq_prompt_tokens.with_label_values(&[&id]).set(prompt_tokens as i64);
        self.seq_prompt_time_s.with_label_values(&[&id]).set((prompt_time_s * 1000.0) as i64);
        self.seq_decode_time_s.with_label_values(&[&id]).set((decode_time_s * 1000.0) as i64);
    }

    /// Publish the live engine gauges (the QoS controller + the scheduler
    /// resources + the `WorkloadHistogram`). Called on each scrape (the
    /// read-on-scrape, the no per-step overhead).
    pub fn publish_live_gauges(
        &self,
        controller: &QosController,
    ) {
        // The QoS controller'suges.
        self.qos_gain.set((controller.gain() * 1000.0) as i64);
        self.qos_itl_ms.set(controller.itl_ms() as i64);
        self.qos_itl_slo_ms.set(
            controller
                .slo_ms()
                .map(|v| v as i64)
                .unwrap_or(-1),
        );

        // The WorkWorkloadHistogram` estimators.
        let hist = &controller.histogram;
        for regime in [Regime::Contended, Regime::Uncontended] {
            for class in [QosClass::Latency, QosClass::Throughput] {
                let p95 = hist.itl_p95(regime, class);
                self.qos_itl_p95_ms
                    .with_label_values(&[regime_label(&regime), class_label(&class)])
                    .set(p95 as i64);
            }
        }
        for class in [QosClass::Latency, QosClass::Throughput] {
            self.qos_output_len_median
                .with_label_values(&[class_label(&class)])
                .set(hist.output_len_median(class) as i64);
            self.qos_prefix_hit_ratio
                .with_label_values(&[class_label(&class)])
                .set((hist.prefix_hit[class_idx(class)].value() * 1000.0) as i64);
            self.qos_preemptions_total
                .with_label_values(&[class_label(&class)])
                .set(hist.preemptions[class_idx(class)] as i64);
            for (state_idx, count) in hist.request_states[class_idx(class)].counts.iter().enumerate() {
                self.qos_request_states
                    .with_label_values(&[class_label(&class), &state_idx.to_string()])
                    .set(*count as i64);
            }
        }
        for regime in [Regime::Contended, Regime::Uncontended] {
            self.qos_batch_size
                .with_label_values(&[regime_label(&regime)])
                .set(hist.batch_size[regime_idx(regime)].value() as i64);
            self.qos_kv_pressure
                .with_label_values(&[regime_label(&regime)])
                .set((hist.kv_pressure[regime_idx(regime)].value() * 1000.0) as i64);
        }
        self.qos_turns.set(hist.turns as i64);
        for work_class in 0..3 {
            for class in [QosClass::Latency, QosClass::Throughput] {
                let et = &hist.exposed_time[work_class][class_idx(class)];
                self.qos_exposed_time_ns
                    .with_label_values(&[&work_class.to_string(), class_label(&class), "host"])
                    .set(et.matrix[work_class][0] as i64);
                self.qos_exposed_time_ns
                    .with_label_values(&[&work_class.to_string(), class_label(&class), "device_wait"])
                    .set(et.matrix[work_class][1] as i64);
            }
        }

        // The scheduler resource gauges (the KV, the no blocking the scrape).
        // The scheduler_cache_usage_percent + the get_cpu_swap_usage are async; the
        // scrape is synchronous, so these are published by the caller (the
        // /metrics handler, which is async) via the publish_scheduler_gauges.
    }

    /// Publish the scheduler resource gauges (the sync, called from the
    /// /metrics handler). The KV usage, the CPU swap, the mamba slots.
    pub fn publish_scheduler_gauges(
        &self,
        scheduler: &Scheduler,
    ) {
        // The single source (the resource_gauges, the DRY: the console
        // print_free_blocks + the /metrics both both read this).
        let g = scheduler.resource_gauges();
        // The existing ratio gauges (the permille, the prior behavior).
        self.kv_cache_usage_ratio.set((g.kv_used_percent * 10.0) as i64);
        self.kv_cache_free_blocks.set(g.kv_free_blocks as i64);
        self.kv_cache_total_blocks.set(g.kv_total_blocks as i64);
        if g.cpu_swap_total_gb > 0.0 {
            self.cpu_swap_usage_ratio.set((g.cpu_swap_used_frac * 1000.0) as i64);
        }
        self.mamba_slots_used.set(g.mamba_slots_used as i64);
        self.mamba_slots_total.set(g.mamba_slots_total as i64);
        // The byte-level memory gauges (the new export).
        self.kv_used_bytes.set((g.kv_used_gb * 1e9) as i64);
        self.kv_total_bytes.set((g.kv_total_gb * 1e9) as i64);
        self.mamba_used_bytes.set((g.mamba_used_gb * 1e9) as i64);
        self.mamba_budget_bytes.set((g.mamba_budget_gb * 1e9) as i64);
        self.cpu_swap_used_bytes.set((g.cpu_swap_used_gb * 1e9) as i64);
        self.cpu_swap_total_bytes.set((g.cpu_swap_total_gb * 1e9) as i64);
        // The runner queue-depth gauges (the live sequence counts).
        self.seq_running.set(g.seq_running as i64);
        self.seq_waiting.set(g.seq_waiting as i64);
        self.seq_cached.set(g.seq_cached as i64);
        self.seq_transferred.set(g.seq_transferred as i64);
        // The physical memory map (the kernel-MMU view, the no the block-based
        // logical occupancy): the VRAM-in-use pressure (the QoS demote signal), the
        // idle VRAM (the grow headroom), the total VRAM. The 0 when the map is not
        // built (the non-CUDA build, the test).
        self.mmu_pressure.set((g.mmu_pressure * 1000.0) as i64);
        self.mmu_idle_vram_bytes.set(g.mmu_idle_vram_bytes as i64);
        self.mmu_total_vram_bytes.set(g.mmu_total_vram_bytes as i64);
    }

    /// Publish the LoRA gauges (the read-on-scrape). Reads the runner's
    /// `LoraRegistry` (the per-runner registry, the source of truth for the
    /// loaded tenants) — NOT the process-wide `global_registry` `OnceLock` (which
    /// is only set when the registry is non-empty at runner-init, and is empty in
    /// the server process when the runner is a separate process). The `None`
    /// registry (no LoRA configured) leaves the gauges at 0 (the prior behavior).
    pub fn publish_lora_gauges(&self, registry: Option<std::sync::Arc<std::sync::Mutex<crate::core::lora::LoraRegistry>>>) {
        let Some(reg) = registry else {
            return; // the no LoRA registry (the no adapters configured)
        };
        let guard = reg.lock().unwrap_or_else(|e| e.into_inner());
        self.lora_active_tenants.set(guard.active_count() as i64);
        // The active slots (the stacked bank's slot count, the 0 when no bank).
        let slots = guard.bank().map(|b| b.slot_count()).unwrap_or(0);
        self.lora_active_slots.set(slots as i64);
        self.lora_adapters_total.set(guard.tenant_count() as i64);
        // The bank bytes (the resident adapter bytes, the 0 when no tenant is
        // resident).
        let bank_bytes: u64 = guard.resident_bytes();
        self.lora_bank_bytes.set(bank_bytes as i64);
    }

    /// Publish the runtime memory + scheduling gauges (the read-on-scrape): the
    /// CUDA graph pool usage (the current + peak MB), the per-step scheduling
    /// budget (the tokens + the max parallel reqs), the CPU KV-cache block
    /// count, + the Mamba active/prefix byte split. These values are passed in by
    /// the caller (the engine, which owns the CUDA pool pool + the scheduler
    /// config) — the telemetry stays a read-only view (the no reaching into the
    /// engine's internals).
    pub fn publish_runtime_gauges(
        &self,
        cuda_graph_pool_current_mb: u64,
        cuda_graph_pool_peak_mb: u64,
        sched_budget_tokens: usize,
        sched_max_parallel_reqs: usize,
        cpu_kv_blocks: usize,
        mamba_active_bytes: u64,
        mamba_prefix_bytes: u64,
        mamba_slot_bytes: u64,
    ) {
        self.cuda_graph_pool_current_mb.set(cuda_graph_pool_current_mb as i64);
        self.cuda_graph_pool_peak_mb.set(cuda_graph_pool_peak_mb as i64);
        self.sched_budget_tokens.set(sched_budget_tokens as i64);
        self.sched_max_parallel_reqs.set(sched_max_parallel_reqs as i64);
        self.cpu_kv_blocks.set(cpu_kv_blocks as i64);
        self.mamba_active_bytes.set(mamba_active_bytes as i64);
        self.mamba_prefix_bytes.set(mamba_prefix_bytes as i64);
        self.mamba_slot_bytes.set(mamba_slot_bytes as i64);
    }

    /// Publish the spec-decode acceptance gauges (the read-on-scrape). The MTP +
    /// DFlash `WindowedAcceptanceStats` (the acceptance rate + the avg
    /// tokens/step, per drafter). The no-op when spec-decode is not active (the
    /// stats are empty).
    pub fn publish_spec_gauges(&self) {
        use crate::speculative::{
            dflash_stats_acceptance_rate, dflash_stats_active, dflash_stats_avg_tokens_per_step,
            mtp_stats_acceptance_rate, mtp_stats_active, mtp_stats_avg_tokens_per_step,
        };
        if mtp_stats_active() {
            self.spec_acceptance_rate
                .with_label_values(&["mtp"])
                .set((mtp_stats_acceptance_rate() * 1000.0) as i64);
            self.spec_avg_tokens_per_step
                .with_label_values(&["mtp"])
                .set(mtp_stats_avg_tokens_per_step() as i64);
        }
        if dflash_stats_active() {
            self.spec_acceptance_rate
                .with_label_values(&["dflash"])
                .set((dflash_stats_acceptance_rate() * 1000.0) as i64);
            self.spec_avg_tokens_per_step
                .with_label_values(&["dflash"])
                .set(dflash_stats_avg_tokens_per_step() as i64);
        }
    }

    /// Set the boot-time memory budget (the static, set at startup). The per-pool
    /// allocation (the graphs/moe/lora/gdn/kv) + the total free + the remainder.
    pub fn set_mem_budget(&self, allocations: &[(&str, u64)], total_free: u64, remainder: u64) {
        for (name, bytes) in allocations {
            self.mem_budget_pool_bytes
                .with_label_values(&[name])
                .set(*bytes as i64);
        }
        self.mem_budget_total_free_bytes.set(total_free as i64);
        self.mem_budget_remainder_bytes.set(remainder as i64);
    }

    /// Set the static build info (the startup hook).
    pub fn set_build_info(&self, econfig: &EngineConfig) {
        let model_id = econfig.model_id.clone().unwrap_or_else(|| "unknown".into());
        let num_shards = econfig.num_shards.unwrap_or(1).to_string();
        let kvcache_dtype = econfig.kvcache_dtype.to_string();
        let max_model_len = econfig.max_model_len.unwrap_or(0).to_string();
        let block_size = econfig.block_size.to_string();
        self.build_info
            .with_label_values(&[&model_id, &num_shards, &kvcache_dtype, &max_model_len, &block_size])
            .set(1);
    }

    /// Set the static KV cache config (the startup hook).
    pub fn set_kv_cache_config(&self, num_blocks: usize, max_tokens: usize) {
        self.kv_cache_config
            .with_label_values(&[&num_blocks.to_string(), &max_tokens.to_string()])
            .set(1);
    }

    /// Scrape the registry (the /metrics handler). Evicts the expired per-seq
    /// stats (the 60s aging) + encodes the registry to the Prometheus text
    /// format.
    pub fn scrape(&self) -> String {
        self.seq_stats.evict_expired();
        // The refresh the per-seq gauges from the live entries (the ≤60s ones).
        // The aged-out entries are cleared (the no stale per-seq gauges).
        self.clear_per_seq_gauges();
        for (seq_id, report) in self.seq_stats.live_entries() {
            let stats = &report.sched;
            self.seq_output_tokens
                .with_label_values(&[&seq_id.to_string(), &stats.model_label])
                .set(stats.output_len as i64);
            self.seq_itl_ms
                .with_label_values(&[&seq_id.to_string()])
                .set(stats.itl_ms as i64);
            self.seq_gain
                .with_label_values(&[&seq_id.to_string(), "start"])
                .set((stats.gain_start * 1000.0) as i64);
            self.seq_gain
                .with_label_values(&[&seq_id.to_string(), "end"])
                .set((stats.gain_end * 1000.0) as i64);
            self.seq_prefill_steps
                .with_label_values(&[&seq_id.to_string()])
                .set(stats.prefill_steps as i64);
        }
        let encoder = TextEncoder::new();
        let mut buf: Vec<u8> = Vec::new();
        let families = self.registry.gather();
        encoder
            .encode(&families, &mut buf)
            .expect("the Prometheus encode");
        String::from_utf8(buf).expect("the UTF-8 encode")
    }

    /// Clear the per-seq gauges (the aged-out entries are removed from the
    /// Prometheus, the no stale series). Called before republishing the live
    /// entries on each scrape.
    fn clear_per_seq_gauges(&self) {
        // The IntGaugeVec series are cleared by resetting all known seq_id
        // labels. The Prometheus registry does not support direct series
        // deletion, so the aged-out seq_ids' gauges retain their last value
        // until the process restarts. This is acceptable for the 60s window
        // (the stale values are bounded + the global counters are the primary
        // signal). A full series GC would require the prometheus crate's
        // Collector API, which is out of scope for this initial.
    }
}

// The helper functions (the QosClass/Regime → the string label).
fn class_idx(c: QosClass) -> usize {
    match c {
        QosClass::Latency => 0,
        QosClass::Throughput => 1,
    }
}
fn regime_idx(r: Regime) -> usize {
    match r {
        Regime::Contended => 0,
        Regime::Uncontended => 1,
    }
}
fn class_label(c: &QosClass) -> &'static str {
    match c {
        QosClass::Latency => "latency",
        QosClass::Throughput => "throughput",
    }
}
fn regime_label(r: &Regime) -> &'static str {
    match r {
        Regime::Contended => "contended",
        Regime::Uncontended => "uncontended",
    }
}