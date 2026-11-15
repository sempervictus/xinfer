// src/core/scheduler.rs
use super::runner::RunnerType;
use super::{
    block_manager::BlockManager,
    prefix_cache::PrefixCacheConfig,
    qos::{BlockSnapshot, QosController, Regime, ResourceGauges, SchedSeqStats, SeqReport},
    sequence::{Sequence, SequenceStatus},
};
use crate::transfer::{PdConfig, PdRole};
use crate::telemetry::Telemetry;
use crate::utils::config::{Config, EngineConfig, EosTokenId};
use crate::utils::env::qos_enabled;
use candle_core::Result;
use parking_lot::RwLock;
use regex::Regex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
pub struct Scheduler {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    cached: Vec<Sequence>,
    transferred: VecDeque<Sequence>,
    pub block_manager: BlockManager,
    next_seq_id: usize,
    /// Per-seq end-of-sequence report (cached-token count + QoS scheduling stats),
    /// retained briefly after `clear_finished()` (which drops the Sequence) so the
    /// server's report can still read it. Bounded by `FINISHED_CACHED_TOKENS_MAX`.
    finished_reports: HashMap<usize, SeqReport>,
    eos_token_id: Vec<u32>,
    /// Token IDs that represent the end of a tool call (e.g., </tool_call> tokens)
    tool_call_end_token_ids: Vec<u32>,
    /// Token IDs that represent the start of a tool call (used to avoid false end matches)
    tool_call_start_token_ids: Vec<u32>,
    /// Token ID for } character (used for JSON tool call detection)
    json_end_token_id: Option<u32>,
    /// Tokenizer for decoding output to check JSON tool call patterns
    tokenizer: Option<Arc<Tokenizer>>,
    /// Regex for detecting JSON tool calls
    tool_call_regex: Regex,
    cfg: EngineConfig,
    pd_config: Option<PdConfig>,
    is_last_prefill: bool,
    /// Sequence IDs whose hybrid GDN/Mamba active slots must be released on the
    /// runner after scheduler-side abort (swap-in failure, etc.).
    pending_runner_releases: Vec<usize>,
    /// The self-tuning QoS controller (per model). When QoS is off this is a
    /// no-op (the static degenerate plan), so behavior is unchanged.
    policy: QosController,
    /// Wall-clock of the previous `schedule()` step; the delta is the ITL sensor.
    last_step: Option<Instant>,
    /// The physical memory pressure (the 0..1 fraction of VRAM in use, the map's
    /// device_pressure). Set by the engine each step (the no the block-based
    /// mem_headroom, which is blind to the idle VRAM). The QoS KV demote actuator
    /// triggers on this signal (the physical pressure crossing the
    /// kv_demote_threshold), not the logical block occupancy.
    physical_pressure: f32,
    /// The physical memory map's idle + total VRAM (the grow headroom, the
    /// exported to the /metrics). Set by the engine each step. The 0 when the map
    /// is not built (the non-CUDA build, the test).
    mmu_idle_vram_bytes: u64,
    mmu_total_vram_bytes: u64,
}

/// Cap on `Scheduler::finished_reports`. Bounded memory beats LRU
/// here: dropping reporting on a small population is acceptable.
const FINISHED_CACHED_TOKENS_MAX: usize = 16_384;
pub const KVCACHE_SWAP_THRESHOLD: f32 = 0.95f32; // over 95%
const SWAP_COOLING_PERIOD: usize = 5000; // 5 seconds cooling time to prevent frequent swap out/in
const MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP: usize = 1000; // to swap-in, at least 1000 kvcache tokens left for decoding
pub const PD_PREFILL_STATUS_CHECK_COOLING_PERIOD: usize = 50; // check prefill status every 50ms (data is pushed immediately)
pub const PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD: usize = 128; // do not transfer prefill length < 128
/// When prefix cache hit is high, prefer local prefill if new tokens < this threshold
pub const PD_LOCAL_PREFILL_NEW_TOKEN_THRESHOLD: usize = 1024;
const PREFIX_CACHE_RATIO_NORMAL: f32 = 0.65;
const PREFIX_CACHE_RATIO_PD_SERVER: f32 = 0.8;
const PREFIX_CACHE_RATIO_PD_CLIENT: f32 = 0.5;
const PREFIX_CACHE_PRESSURE_EVICT_PERCENT: f32 = 0.1; // evict 10% of prefix cache when under pressure

fn active_sequence_limit(max_num_seqs: usize, mamba_cache_capacity: Option<usize>) -> usize {
    match mamba_cache_capacity {
        Some(mamba_cap) if mamba_cap > 0 => max_num_seqs.min(mamba_cap),
        _ => max_num_seqs,
    }
    .max(1)
}

fn build_prefix_cache_config(econfig: &EngineConfig) -> PrefixCacheConfig {
    let enabled = econfig.prefix_cache.unwrap_or(false);
    if !enabled {
        return PrefixCacheConfig {
            enabled: false,
            max_cached_blocks: 0,
        };
    }

    let mut max_cached_blocks = if let Some(max_tokens) = econfig.prefix_cache_max_tokens {
        max_tokens / econfig.block_size
    } else {
        let is_pd_server = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        };
        let is_pd_client = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, PdRole::Client)
        } else {
            false
        };

        let ratio = if is_pd_server {
            econfig
                .pd_server_prefix_cache_ratio
                .unwrap_or(PREFIX_CACHE_RATIO_PD_SERVER)
        } else if is_pd_client {
            econfig
                .pd_client_prefix_cache_ratio
                .unwrap_or(PREFIX_CACHE_RATIO_PD_CLIENT)
        } else {
            PREFIX_CACHE_RATIO_NORMAL
        };

        if is_pd_server || is_pd_client {
            ((econfig.num_blocks as f32) * ratio) as usize
        } else {
            // Workload-aware sizing (QoS on): hold all concurrent sequences at full
            // context. max_model_len is resolved at engine init before Scheduler::new.
            let qos_on = qos_enabled() || econfig.qos.enabled;
            if qos_on {
                let max_tokens_per_seq = econfig.max_model_len.unwrap_or(262144);
                econfig.max_num_seqs * max_tokens_per_seq / econfig.block_size
            } else {
                // QoS off: fall back to the fixed fraction of the KV budget.
                ((econfig.num_blocks as f32) * ratio) as usize
            }
        }
    };

    if max_cached_blocks > econfig.num_blocks {
        max_cached_blocks = econfig.num_blocks;
    }

    if max_cached_blocks == 0 {
        crate::log_warn!("Prefix cache enabled but max cached blocks is 0; disabling.");
        return PrefixCacheConfig {
            enabled: false,
            max_cached_blocks: 0,
        };
    }

    crate::log_warn!(
        "Prefix cache enabled: {} blocks ({} tokens).",
        max_cached_blocks,
        max_cached_blocks * econfig.block_size
    );

    PrefixCacheConfig {
        enabled: true,
        max_cached_blocks,
    }
}

impl Scheduler {
    pub fn new(runners: Arc<RwLock<RunnerType>>, econfig: &EngineConfig, config: &Config) -> Self {
        let prefix_cache_cfg = build_prefix_cache_config(econfig);
        let mamba_snapshot_default_stride_blocks = econfig
            .effective_prefill_chunk_size()
            .div_ceil(econfig.block_size)
            .max(1);
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            cached: Vec::new(),
            transferred: VecDeque::new(),
            block_manager: BlockManager::new(
                runners,
                econfig.num_blocks,
                (econfig.cpu_mem_fold.unwrap_or(0.5f32) * econfig.num_blocks as f32) as usize,
                econfig.block_size,
                prefix_cache_cfg,
                config
                    .architectures
                    .as_ref()
                    .and_then(|arches| arches.first())
                    .map(|arch| crate::utils::is_qwen3_hybrid_arch_name(arch))
                    .unwrap_or(false),
                mamba_snapshot_default_stride_blocks,
            ),
            next_seq_id: 0,
            finished_reports: HashMap::new(),
            eos_token_id: match &config.eos_token_id {
                Some(EosTokenId::Single(eos)) => vec![*eos],
                Some(EosTokenId::Multiple(eos)) => eos.into_iter().map(|x| *x).collect(),
                _ => vec![],
            },
            // Tool call end tokens will be set by engine after tokenizer is initialized
            tool_call_end_token_ids: Vec::new(),
            tool_call_start_token_ids: Vec::new(),
            json_end_token_id: None,
            tokenizer: None,
            // Regex to match JSON tool call format: {"name": "...", "arguments": {...}}
            // We use (?s) to allow dot matching newlines
            tool_call_regex: Regex::new(r#"(?s)\{\s*"name"\s*:.*"arguments"\s*:.*\}\s*$"#).unwrap(),
            cfg: econfig.clone(),
            pd_config: econfig.pd_config.clone(),
            is_last_prefill: false,
            pending_runner_releases: Vec::new(),
            policy: QosController::new(econfig.qos.clone()),
            last_step: None,
            physical_pressure: 0.0,
            mmu_idle_vram_bytes: 0,
            mmu_total_vram_bytes: 0,
        }
    }

    /// Set the physical memory pressure (the map's device_pressure, the 0..1 VRAM
    /// used fraction). Called by the engine each step before the schedule, so the
    /// QoS KV demote actuator triggers on the physical signal (the no the block
    /// occupancy, which is blind to the idle VRAM). The idle + total VRAM are the
    /// grow headroom, exported to the /metrics.
    pub fn set_mmu_view(&mut self, pressure: f32, idle_vram_bytes: u64, total_vram_bytes: u64) {
        self.physical_pressure = pressure;
        self.mmu_idle_vram_bytes = idle_vram_bytes;
        self.mmu_total_vram_bytes = total_vram_bytes;
    }

    pub fn request_runner_release(&mut self, seq_id: usize) {
        if !self.pending_runner_releases.contains(&seq_id) {
            self.pending_runner_releases.push(seq_id);
        }
    }

    pub fn take_pending_runner_releases(&mut self) -> Vec<usize> {
        if self.pending_runner_releases.is_empty() {
            return Vec::new();
        }
        std::mem::take(&mut self.pending_runner_releases)
    }

    /// The QoS controller (the telemetry's live-gauge source). The
    /// `publishTelemetry` RPC reads the controller's gain/ITL/SLO + the
    /// WorkloadHistogram estimators for the Prometheus export.
    pub fn policy(&self) -> &QosController {
        &self.policy
    }

    /// Set tool call end token IDs (called by engine after tokenizer is available)
    pub fn set_tool_call_end_tokens(&mut self, token_ids: Vec<u32>) {
        self.tool_call_end_token_ids = token_ids;
    }

    /// Set tool call start token IDs (called by engine after tokenizer is available)
    pub fn set_tool_call_start_tokens(&mut self, token_ids: Vec<u32>) {
        self.tool_call_start_token_ids = token_ids;
    }

    /// Set tokenizer for JSON tool call detection (called by engine after initialization)
    pub fn set_tokenizer(&mut self, tokenizer: Arc<Tokenizer>) {
        // Get the token ID for "}" character
        if let Ok(tokens) = tokenizer.encode("}", false) {
            if let Some(&token_id) = tokens.get_ids().last() {
                self.json_end_token_id = Some(token_id);
                crate::log_info!("JSON end token ID (}}) set to: {}", token_id);
            }
        }
        self.tokenizer = Some(tokenizer);
    }

    pub fn add(&mut self, mut seq: Sequence) -> usize {
        seq.id = self.next_seq_id;
        let id = seq.id;
        self.next_seq_id += 1;
        // Assign the QoS class from the controller's measured output-length estimate
        // (a no-op Throughput default until outputs are observed).
        seq.qos_class = self.policy.infer_class(&seq);
        // The two-rate sensor (the fast path): the admission records the
        // prefix-hit EMA + the turns counter + the reuse-evidence digest (the
        // NInfer real-demand signal). The `num_cached_tokens` is the prefix hit,
        // the `len()` is the query.
        if self.policy.active() {
            let reuse_digest = seq.mamba_prefix_hash.unwrap_or(0);
            self.policy.record_admission(
                seq.qos_class,
                seq.num_cached_tokens,
                seq.len(),
                reuse_digest,
            );
        }
        self.waiting.push_back(seq);
        id
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    fn active_sequence_limit(&self) -> usize {
        active_sequence_limit(
            self.cfg.max_num_parallel_reqs.max(1),
            self.cfg.mamba_cache_capacity,
        )
    }

    /// Stable-partition the waiting queue so Latency-class requests are admitted ahead
    /// of Throughput (FIFO within each class). Only called when QoS is active; when
    /// it is not, the queue is already FIFO. O(n), no per-step HashMap.
    fn partition_waiting_by_class(&mut self) {
        let (latency, throughput) = self
            .waiting
            .drain(..)
            .partition::<VecDeque<Sequence>, _>(|s| s.qos_class.is_latency());
        self.waiting = latency.into_iter().chain(throughput).collect();
    }

    /// Schedule sequences and return their indexes in `running` along with prefill flag
    #[allow(non_snake_case)]
    pub fn schedule(&mut self) -> Result<(Vec<usize>, bool)> {
        let mut scheduled_ids = Vec::new();
        let mut num_tokens = 0;

        // QoS: measure the step (the ITL sensor), then ask the policy for this
        // step's plan. When QoS is off the plan is the static degenerate one (full
        // chunk, FIFO admission, no gates), so behavior is unchanged.
        let now = Instant::now();
        let step_ms = self
            .last_step
            .map(|t| now.duration_since(t).as_secs_f64() * 1e3)
            .unwrap_or(0.0);
self.last_step = Some(now);

        let blocks = BlockSnapshot {
            free_blocks: self.block_manager.get_num_free_blocks(),
            total_blocks: self.block_manager.get_num_total_blocks(),
            max_num_seqs: self.active_sequence_limit(),
            max_num_batched_tokens: self.cfg.max_num_batched_tokens,
            static_chunk: self.cfg.effective_prefill_chunk_size(),
            chunk_floor: self.policy.effective().chunk_floor.unwrap_or(0),
        };
        if self.policy.active() {
            // Guard against idle gaps: the engine loop skips schedule() while idle,
            // so a large delta means "woke from idle", not a slow step. Feeding it
            // to the ITL sensor would falsely spike the gain. The step filter stays
            // neutral (the no the filter) when the max_meaningful_step_ms is
            // not yet known.
            let eff = self.policy.effective();
            let step_ok = eff.max_meaningful_step_ms.map_or(true, |max_ms| step_ms <= max_ms);
            if step_ok {
                // The regime is an independent sensor (the decode slot occupancy + the
                // KV pressure), not the controller's own gain output. The class is the
                // dominant running class, so the ITL P² is fed under the real class.
                // The regime stays neutral (the Uncontended) when the thresholds are
                // not yet known (the no the blind 0.5/0.8 biasing the regime).
                let kv_pressure = 1.0 - blocks.mem_headroom() as f64;
                let regime = match (eff.regime_decode_pressure, eff.regime_kv_pressure) {
                    (Some(dp_thresh), Some(kp_thresh)) => {
                        Regime::classify(
                            self.policy.decode_pressure(&self.running, &blocks),
                            kv_pressure,
                            dp_thresh,
                            kp_thresh,
                        )
                    }
                    _ => Regime::Uncontended,
                };
                let class = self.policy.dominant_class(&self.running);
                self.policy.record_step(step_ms, regime, class);
            }
        }
        // The AdaptivePlan (the base StepPlan + the QoS actuators: the
        // batch_cap, the sampling override, the parallelism_hint, the
        // kv_demote_threshold). The histogram is fed by the record_step (the ITL)
        // + the record_batch_size (the regime) + the record_admission (the
        // prefix-hit).
        let eff = self.policy.effective();
        let decode_pressure = self.policy.contention(&self.running, &blocks).decode_pressure;
        let kv_pressure = 1.0 - blocks.mem_headroom() as f64;
        let regime = match (eff.regime_decode_pressure, eff.regime_kv_pressure) {
            (Some(dp_thresh), Some(kp_thresh)) => {
                Regime::classify(decode_pressure, kv_pressure, dp_thresh, kp_thresh)
            }
            _ => Regime::Uncontended,
        };
        self.policy.record_batch_size(regime, self.running.len());
        // The online calibration: feed this step's measurements so the controller
        // derives its `None` config values from the measured system instead of
        // blind constants. After the calibration window this is a no-op.
        self.policy.observe_calibration(
            step_ms,
            decode_pressure,
            kv_pressure,
            None,
            self.running.len(),
        );
        let plan = self.policy.adaptive_plan(&self.running, &blocks);
        let chunk_size = plan.base.chunk_size;

        // The QoS KV demote actuator: when the QoS is active + the calibrated
        // kv_demote_threshold is set + the physical memory pressure (the map's
        // device_pressure, the VRAM-in-use fraction) crosses it, demote the coldest
        // running sequence to the host swap tier. This frees GPU blocks for the
        // admission + the decode, relieving the physical pressure. The demote is the
        // swap_out_suffix (the prefix stays on the GPU as the join key, the GDN
        // boundary is captured), so the sequence is resumable. No-op when QoS is off
        // or the threshold is not yet calibrated (the no a blind constant).
        if self.policy.active() {
            if let Some(demote_threshold) = plan.kv_demote_threshold {
                if self.physical_pressure > demote_threshold && !self.running.is_empty() {
                    let freed = self.demote_coldest_running();
                    if freed > 0 {
                        crate::log_warn!(
                            "[qos] KV demote: physical pressure {:.2} exceeded threshold {:.2}, demoted {} block(s) to host",
                            self.physical_pressure,
                            demote_threshold,
                            freed
                        );
                    }
                }
            }
        }

        // QoS: admit Latency-class requests ahead of Throughput (stable partition),
        // and log the live contention vector under pressure. No-op when QoS is off.
        if self.policy.active() {
            self.partition_waiting_by_class();
            if tracing::enabled!(tracing::Level::DEBUG) {
                let c = self.policy.contention(&self.running, &blocks);
                if c.decode_pressure > 0.0 {
tracing::debug!(
                         "[qos] contention={:?} chunk={} budget={}",
                         c,
                         plan.base.chunk_size,
                         plan.base.token_budget
                     );
                }
            }
        }

        // PD server: Check for new incoming prefill requests
        if self.is_pd_server() {
            if let Ok((fit, Some(seq))) = self
                .block_manager
                .try_receive_prefill(self.get_available_kv_tokens())
            {
                let seq_id = seq.id;
                if !fit {
                    crate::log_warn!(
                        "Prefill request (Seq {}) enter pending status because it require {} KvCache tokens (left {}).",
                        seq_id,
                        seq.len() + 1,
                        self.get_available_kv_tokens(),
                    );
                } else {
                    crate::log_warn!(
                        "Prefill request (Seq {}, {} tokens) received from PD client.",
                        seq_id,
                        seq.len(),
                    );
                }
                // Add to waiting queue.
                self.waiting.push_back(seq);
            }
        }

        // Prefill phase: move sequences from waiting to running if possible.
        // Use effective chunk tokens (not full sequence length) for budget accounting
        // to enable batched prefill of multiple sequences per step.
        // Preserve interleaving: when the previous step was prefill and decode
        // sequences already existed, give those sequences a decode step before
        // admitting another waiting prefill batch.
        let pre_existing_running = self.running.len();
        let max_seqs_limit = self.active_sequence_limit();
        let token_budget = plan.base.token_budget;
        let mut thr_in_use = plan.base.thr_in_use;

        while let Some(mut seq) = self.waiting.pop_front() {
            // Try to transfer prefill requests to PD server when applicable
            if self.is_pd_mode() && !self.is_pd_server() && self.try_transfer(&mut seq) {
                break;
            }

            let effective_tokens = seq.prefill_chunk_tokens(chunk_size);

            // QoS reservation gates: a Throughput request must not consume the
            // Latency slot / KV reserve. No-op when QoS is off (both gates false).
            let is_thr = !seq.qos_class.is_latency();
            let slot_blocked = is_thr
                && plan.base.thr_slot_capacity > 0
                && thr_in_use >= plan.base.thr_slot_capacity;
            let kv_blocked = is_thr && plan.base.thr_kv_full;
            if slot_blocked || kv_blocked {
                seq.sched_stats.reservation_blocked += 1;
                self.waiting.push_front(seq);
                break;
            }

            if self.running.len() >= max_seqs_limit
                || scheduled_ids.len() >= max_seqs_limit
                // The batch_cap admission gate (the vLLM `max_num_active_reqs`
                // result: the no new-sequence admission beyond the QoS-computed cap,
                // separate from the `max_num_seqs` slot count). The `batch_cap == 0`
                // means unlimited (the QoS-off degenerate).
                || (plan.batch_cap > 0 && self.running.len() >= plan.batch_cap)
                || num_tokens + effective_tokens > token_budget
                || (seq.block_table.is_empty() && !self.block_manager.can_allocate(&seq))
                // interleaved scheduling: alternate prefill/decode for fairness
                // only block when there are pre-existing decode sequences
                || (self.is_last_prefill && pre_existing_running > 0)
            {
                // Put it back and break out if cannot schedule more
                self.waiting.push_front(seq);
                break;
            }

            if seq.block_table.is_empty() {
                self.block_manager.allocate(&mut seq)?;
            }
            seq.status = SequenceStatus::Running;
            // Stamp the adaptive chunk so the runner's prepare_prefill and the
            // scheduler's filter_prefill_finished agree on what was budgeted.
            seq.active_prefill_chunk = Some(chunk_size);
            // QoS per-seq stats: record the adaptive adjustments that fired this step.
            seq.sched_stats.qos_active = self.policy.active();
            seq.sched_stats.qos_class = seq.qos_class;
            // The two-rate sensor (the fast path): the admission records the
            // request-state transition (the `Running` → the `DecodeReady` (
            // index 6, the NInfer `EngineRequestState`).
            if self.policy.active() {
                self.policy.record_request_state(seq.qos_class, 6);
            }
            seq.sched_stats.gain_start = self.policy.gain();
            seq.sched_stats.prefill_steps += 1;
            if chunk_size < blocks.static_chunk {
                seq.sched_stats.chunk_shrunk += 1;
            }
            if seq.qos_class.is_latency() && self.policy.active() {
                seq.sched_stats.priority_admitted += 1;
            }
            num_tokens += effective_tokens;
            self.running.push(seq);
            if is_thr {
                thr_in_use += 1;
            }
            scheduled_ids.push(self.running.len() - 1); // index of newly pushed seq
        }

        if !scheduled_ids.is_empty() {
            self.is_last_prefill = true;
            return Ok((scheduled_ids, true));
        }

        // Decode phase: pick sequences from running for decoding (up to max_num_seqs)
        let mut decode_ids = Vec::new();
        let mut preempt_ids = Vec::new();

        for (idx, seq) in self.running.iter().enumerate() {
            if !self.block_manager.can_append(seq)
                && seq.status != SequenceStatus::Swapped
                && seq.status != SequenceStatus::FinishSwapped
            {
                preempt_ids.push(idx);
            }
        }
        // QoS per-seq stats: count preemption pressure on each affected sequence.
        for &idx in &preempt_ids {
            self.running[idx].sched_stats.preempted += 1;
            // The preemption feeds the histogram's preemptions[class]
            // (the NInfer retraction_count, the capacity signal).
            if self.policy.active() {
                self.policy.record_preemption(self.running[idx].qos_class);
            }
        }

        // Client: Check for finished prefills
        if self.is_pd_mode() && !self.is_pd_server() {
            self.try_receive_kvcache()?;
        }

        // Swap back seq from cpu memory if possible. Aggregate KV usage can stay high
        // under prefix cache because reusable prefix blocks occupy the GPU pool, so
        // already-swapped sequences must still get a chance to run the precise
        // suffix-allocation and prefix-eviction checks in `try_swap_in`.
        let has_swapped_seq = self
            .cached
            .iter()
            .any(|seq| seq.status == SequenceStatus::Swapped);
        let should_try_swap_in = cfg!(feature = "cuda")
            && preempt_ids.is_empty()
            && (has_swapped_seq
                || self.kv_cache_usage_percent() < KVCACHE_SWAP_THRESHOLD * 0.9
                || (self.running.is_empty() && self.kv_cache_usage_percent() <= 0.3f32));
        if should_try_swap_in {
            #[cfg(feature = "cuda")]
            self.try_swap_in();
        } else if !preempt_ids.is_empty() || self.kv_cache_usage_percent() > KVCACHE_SWAP_THRESHOLD
        {
            // Requests unable to be processed at the current moment
            // If we only have one sequence running and it has been preempt,
            // swap out to cpu memory make non-sense
            // in such case, the only option is either waiting resources or abort it
            let evicted = self.evict_prefix_cache_under_pressure();
            if evicted > 0 {
                crate::log_warn!("Evicted {} prefix cache block(s) under pressure.", evicted);
            } else {
                #[cfg(feature = "cuda")]
                // Swap out the stuck (preempted) sequences to free GPU blocks. The
                // prior gate (the running.len() > 1) blocked the single-seq swap-out
                // on the theory that swapping the only sequence is pointless. That
                // theory holds for a healthy single seq (the no the decode, the no the
                // swap), but not for a stuck single seq (the no the can_append failing,
                // the no the pool full): the swap-out frees its GPU blocks (the no the CPU
                // holds them), the swap-in reclaims them (the no the pool has room after
                // the swap-out), so the seq makes progress. Without this, a single
                // stuck seq stalls indefinitely (the no the decode, the no the swap-in).
                if !preempt_ids.is_empty() {
                    if let Some((idx, _)) = preempt_ids
                        .iter()
                        .map(|&i| (i, &self.running[i]))
                        .min_by_key(|(_, seq)| seq.id)
// swap-out the oldest stuck sequence
                    {
                        crate::log_warn!(
                            "Swapping out stuck Seq {:?} to free GPU blocks for admission",
                            self.running[idx].id
                        );
                        self.try_swap_out(idx, true);
                        self.running[idx].sched_stats.swapped += 1;
                    }
                }
            }
        }

        let is_pd_server = self.is_pd_server();
        let decode_max_seqs = self.active_sequence_limit();
        let mut pd_finished_ids: Vec<usize> = Vec::new();
        for (idx, seq) in self.running.iter_mut().enumerate() {
            if decode_ids.len() >= decode_max_seqs {
                break;
            }
            if seq.status == SequenceStatus::Finished {
                continue;
            }
            if !self.block_manager.can_append(&seq) {
                // filter out seq that unable to acquire resources
                continue;
            }
            if is_pd_server && seq.status == SequenceStatus::Cached {
                if let Ok(success) = self.block_manager.try_check_kvcache_release(seq.id) {
                    if success {
                        crate::log_warn!("PD Server: release prefilled kvcache for Seq {} (prefix cache retained)", seq.id);
                        seq.status = SequenceStatus::Finished;
                        self.block_manager.deallocate(seq);
                        pd_finished_ids.push(seq.id);
                    }
                }
                continue;
            }
            self.block_manager.may_append(seq)?;
            decode_ids.push(idx);
        }
        // The stall marker: if running sequences exist but none could be decoded
        // (the no the can_append failing, the no the pool full), log the reason so a
        // silent stall is diagnosable. The free block count + the total pool size
        // show whether the pool is exhausted (the no the prefix cache holding the
        // blocks, the no the seq's own KV).
        if decode_ids.is_empty() && !self.running.is_empty() {
            crate::log_warn!(
                "[SCHED] {} running seq(s) could not be decoded (the no the can_append failing). Free blocks: {}, total blocks: {}",
                self.running.len(),
                self.block_manager.get_num_free_blocks(),
                self.block_manager.get_num_total_blocks()
            );
        }
        if !pd_finished_ids.is_empty() {
            self.running.retain(|s| !pd_finished_ids.contains(&s.id));
        }

        self.is_last_prefill = false;
        Ok((decode_ids, false))
    }

    /// Provide immutable access to sequences by indexes (for model inference)
    pub fn get_sequences(&self, ids: &[usize]) -> Vec<&Sequence> {
        ids.iter().map(|&i| &self.running[i]).collect()
    }

    /// For prefill sequences that rely on cached prefix tokens, verify mamba state snapshots
    /// still exist. If a snapshot is missing/evicted, downgrade that sequence to full prefill.
    pub fn fallback_missing_mamba_prefix_snapshots(
        &mut self,
        scheduled_ids: &[usize],
    ) -> Result<usize> {
        let (running, block_manager) = (&mut self.running, &mut self.block_manager);
        let mut downgraded = 0usize;

        for &idx in scheduled_ids {
            if idx >= running.len() {
                continue;
            }
            let seq = &mut running[idx];
            if seq.num_cached_tokens == 0 {
                continue;
            }
            let Some(hash) = seq.mamba_prefix_hash else {
                continue;
            };

            let has_snapshot = match block_manager.try_has_mamba_prefix_state(hash) {
                Ok(v) => v,
                Err(e) => {
                    crate::log_warn!(
                        "Seq {}: failed to query mamba prefix snapshot hash {}: {}. Falling back to full prefill.",
                        seq.id,
                        hash,
                        e
                    );
                    false
                }
            };
            if has_snapshot {
                continue;
            }

            let prev_cached = seq.num_cached_tokens;
            let required_blocks = seq.num_blocks();

            // Drop current block refs (including any shared prefix refs), then rebuild from scratch.
            block_manager.deallocate(seq);
            seq.clear_block_table();

            if !block_manager.can_allocate_without_prefix(seq) {
                let evicted = block_manager.evict_prefix_cache_until_free(required_blocks);
                if evicted > 0 {
                    crate::log_warn!(
                        "Seq {}: evicted {} prefix cache block(s) before mamba fallback reallocation.",
                        seq.id,
                        evicted
                    );
                }
            }
            block_manager.allocate_without_prefix(seq)?;
            downgraded += 1;
            crate::log_warn!(
                "Seq {}: missing mamba snapshot hash {} (cached {} tokens). Downgraded to full prefill.",
                seq.id,
                hash,
                prev_cached
            );
        }

        Ok(downgraded)
    }

    pub fn get_running(&self, idx: usize) -> Option<&Sequence> {
        if idx < self.running.len() {
            Some(&self.running[idx])
        } else {
            None
        }
    }

    pub fn get_waiting(&self, idx: usize) -> Option<&Sequence> {
        if idx < self.waiting.len() {
            Some(&self.waiting[idx])
        } else {
            None
        }
    }

    pub fn get_seq_token_usage(&self, seq_id: usize) -> Result<usize> {
        // search waiting
        if let Some(item) = self.waiting.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // search running
        if let Some(item) = self.running.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // search cached
        if let Some(item) = self.cached.iter().find(|x| x.id == seq_id) {
            return Ok(item.len());
        }

        // if nothing found
        Ok(0)
    }

    pub fn find_seq_by_session_id(&self, session_id: &str) -> Option<(usize, SequenceStatus)> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .find(|seq| seq.sampling_params.session_id.as_deref() == Some(session_id))
            .map(|seq| (seq.id, seq.status))
    }

    /// Postprocess output tokens for each sequence. Handles both single-token (normal decode)
    /// and multi-token (MTP speculative decode) outputs uniformly.
    pub fn postprocess(&mut self, ids: &[usize], multi_output_ids: &[Vec<u32>]) {
        for (i, &idx) in ids.iter().enumerate() {
            if idx >= self.running.len() {
                continue;
            }
            let tokens = &multi_output_ids[i];
            let seq_id = self.running[idx].id;

            // PD server: transfer KV cache on first token, then move to next sequence.
            if self.is_pd_server() {
                if let Some(&first_token) = tokens.first() {
                    match self
                        .block_manager
                        .try_send_kvcache(&self.running[idx], first_token)
                    {
                        Ok(success) => {
                            crate::log_warn!(
                                "PD Server: transferred KV cache for seq {} ({})",
                                seq_id,
                                if success { "success" } else { "faild" }
                            );
                            let seq = &mut self.running[idx];
                            if success {
                                let _ = self
                                    .block_manager
                                    .capture_mamba_prefix_state(seq, seq.len());
                                self.block_manager.cache_sequence(seq);
                                seq.status = SequenceStatus::Cached;
                                let cur_time = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .expect("Time went backwards")
                                    .as_millis()
                                    as usize;
                                let time_costs = cur_time - seq.created_time();
                                if time_costs / 100 > 0 && seq.len() > 0 {
                                    crate::log_info!(
                                        "PD Prefilling [seq_id {}]: {} tokens in {:.2}s ({:.2} tokens/s)",
                                        seq_id,
                                        seq.len(),
                                        time_costs as f32 / 1000f32,
                                        seq.len() as f32 / (time_costs as f32 * 1.0f32 / 1000f32),
                                    )
                                }
                            } else {
                                seq.status = SequenceStatus::Finished;
                                self.block_manager.deallocate(seq);
                            }
                        }
                        Err(e) => {
                            crate::log_error!(
                                "PD Server: failed to transfer KV cache for seq {}: {}",
                                seq_id,
                                e
                            );
                            let seq = &mut self.running[idx];
                            seq.status = SequenceStatus::Finished;
                            self.block_manager.deallocate(seq);
                        }
                    }
                }
                continue;
            }

            for &token in tokens {
                if self.running[idx].sampling_params.mcp_mode.is_some() {
                    let is_end = self.is_tool_call_end(token, idx);
                    if is_end {
                        crate::log_info!(
                            "[Seq {}] Detected </tool_call> token {}, finishing for external handling",
                            seq_id,
                            token
                        );
                        let seq = &mut self.running[idx];
                        seq.append_token(token);
                        seq.is_tool_call_end = true;
                        seq.status = SequenceStatus::Finished;
                        let _ = self
                            .block_manager
                            .capture_mamba_prefix_state(seq, seq.len());
                        self.block_manager.cache_sequence(seq);
                        self.block_manager.deallocate(seq);
                        break;
                    }
                }

                let matched_stop_sequence_idx =
                    self.stop_sequence_match_index(token, &self.running[idx]);
                let hit_stop_sequence = matched_stop_sequence_idx.is_some();
                let seq = &mut self.running[idx];

                if hit_stop_sequence
                    || (!seq.sampling_params.ignore_eos && self.eos_token_id.contains(&token))
                    || seq.output_len() >= seq.sampling_params.max_tokens.unwrap_or(16384)
                    || seq.len()
                        > self
                            .cfg
                            .max_model_len
                            .unwrap_or(self.cfg.max_kv_cache_tokens.max(1))
                {
                    if hit_stop_sequence {
                        crate::log_info!(
                            "[Seq {}] Detected stop sequence token {}, finishing",
                            seq_id,
                            token
                        );
                        seq.hit_stop_sequence = true;
                        seq.stop_sequence = matched_stop_sequence_idx.and_then(|stop_idx| {
                            seq.sampling_params
                                .stop_sequences
                                .as_ref()
                                .and_then(|stops| stops.get(stop_idx))
                                .cloned()
                        });
                    }
                    seq.status = SequenceStatus::Finished;
                    let _ = self
                        .block_manager
                        .capture_mamba_prefix_state(seq, seq.len());
                    self.block_manager.cache_sequence(seq);
                    self.block_manager.deallocate(seq);
                    break;
                } else {
                    seq.append_token(token);
                    if seq.len() % self.cfg.block_size == 1 && seq.len() > 1 {
                        let _ = self.block_manager.may_append(seq);
                    }
                    if seq.len() % self.cfg.block_size == 0 {
                        let _ = self
                            .block_manager
                            .capture_mamba_prefix_state(seq, seq.len());
                    }
                }
            }
        }
    }

    /// Pre-allocate KV cache blocks for MTP speculative tokens.
    /// Called before MTP runs to ensure the verification forward has room
    /// to write KV for speculative positions.
    pub fn pre_allocate_mtp_blocks(&mut self, ids: &[usize], extra_tokens: usize) {
        for &idx in ids {
            if idx >= self.running.len() {
                continue;
            }
            let seq = &mut self.running[idx];
            let needed_len = seq.len() + extra_tokens;
            let needed_blocks = needed_len.div_ceil(self.cfg.block_size);
            while seq.block_table.len() < needed_blocks {
                if let Some(block_id) = self.block_manager.alloc_free_block() {
                    seq.block_table.push(block_id as u32);
                } else {
                    break;
                }
            }
        }
    }

    pub fn postprocess_speculative_extra(&mut self, ids: &[usize], extras: &[Vec<u32>]) {
        for (i, extra_tokens) in extras.iter().enumerate() {
            if i >= ids.len() {
                break;
            }
            let idx = ids[i];
            if idx >= self.running.len() {
                continue;
            }
            if self.running[idx].status == SequenceStatus::Finished {
                continue;
            }
            let mut finished = false;
            for &tok in extra_tokens {
                if !self.running[idx].sampling_params.ignore_eos && self.eos_token_id.contains(&tok)
                {
                    self.running[idx].append_token(tok);
                    self.running[idx].status = SequenceStatus::Finished;
                    self.block_manager
                        .capture_mamba_prefix_state(&self.running[idx], self.running[idx].len());
                    self.block_manager.cache_sequence(&self.running[idx]);
                    self.block_manager.deallocate(&mut self.running[idx]);
                    finished = true;
                    break;
                }
                self.running[idx].append_token(tok);
            }
            if !finished {
                let _ = self.block_manager.ensure_allocate(&mut self.running[idx]);
            }
        }
    }

    pub fn clear_finished(&mut self) {
        let is_pd_server = self.is_pd_server();
        // Collect the finished sequences' report data (id, cached tokens, sched
        // stats, output length) before the retain() drops them.
        let finished: Vec<(usize, usize, SchedSeqStats, usize, Option<String>)> = self
            .running
            .iter()
            .chain(self.waiting.iter())
            .filter(|s| s.status == SequenceStatus::Finished)
            .map(|s| {
                (
                    s.id,
                    s.num_cached_tokens,
                    s.sched_stats.clone(),
                    s.output_ids.len(),
                    // The model that served this sequence (the "base" when no LoRA
                    // tenant, the tenant name when a LoRA adapter served it).
                    // Surfaced in the per-sequence report.
                    s.sampling_params.lora_tenant.clone(),
                )
            })
            .collect();
        if is_pd_server {
            self.print_free_blocks();
        }
        for (seq_id, cached_tokens, mut sched, output_len, lora_tenant) in finished {
            // Stamp the controller's learning trajectory at the sequence's end so the
            // report shows whether QoS was adapting (gain) and holding the SLO (ITL).
            sched.gain_end = self.policy.gain();
            sched.itl_ms = self.policy.itl_ms();
            sched.itl_slo_ms = self.policy.slo_ms();
            sched.output_len = output_len;
            // The model label (the "base" when no LoRA tenant served this sequence,
            // the tenant name when a LoRA adapter did). The operator sees which
            // model produced the output in the per-sequence report.
            sched.model_label = lora_tenant.unwrap_or_else(|| "base".to_string());
            self.policy.record_output(output_len);
            // The two-rate sensor (the slow path): the finish records the turn
            // (the conversation warmth, the L1 capture_seq) + the reuse-evidence
            // (the NInfer real-demand signal: the `proven_frontier` + the 256-entry
            // demand digest) + the request-state transition (the `ModelFinished` lifecycle).
            if self.policy.active() {
                let class = sched.qos_class;
                self.policy.record_turn();
                self.policy.record_reuse_evidence(class, cached_tokens as u64);
                self.policy.record_request_state(class, 7); // The ModelFinished (the 8-state index 7).
            }
            self.remember_finished_report(
                seq_id,
                SeqReport {
                    cached_tokens,
                    sched,
                    prompt_tokens: 0,
                    prompt_time_s: 0.0,
                    decode_time_s: 0.0,
                },
            );
        }
        self.running
            .retain(|seq| seq.status != SequenceStatus::Finished);
        self.waiting
            .retain(|seq| seq.status != SequenceStatus::Finished);
    }

    pub fn release_waitings(&mut self) -> Vec<usize> {
        // Release all waiting sequences since there are no more resources (kv cache)
        let mut released_ids = Vec::with_capacity(self.waiting.len() + self.cached.len());
        for i in 0..self.waiting.len() {
            let seq = &mut self.waiting[i];
            released_ids.push(seq.id);
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
        }
        self.waiting.clear();
        for i in 0..self.cached.len() {
            let seq = &mut self.cached[i];
            released_ids.push(seq.id);
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(seq);
            // free gpu blocks and also free any CPU swap space
            self.block_manager.free_cpu_swap_for_seq(seq.id);
        }
        self.cached.clear();
        self.block_manager.clear_prefix_cache();
        released_ids
    }

    pub fn release_cache(&mut self, seq_id: usize) {
        if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.cached.remove(pos);
            // also free cpu swap
            if seq.status == SequenceStatus::Swapped || seq.status == SequenceStatus::FinishSwapped
            {
                self.block_manager.free_cpu_swap_for_seq(seq_id);
            }
            seq.status = SequenceStatus::Finished;
            self.block_manager.deallocate(&seq);
        }
    }

    pub fn cancel(&mut self, seq_id: usize) {
        // A normal cancellation will release the runner slot itself. Do not
        // leave a scheduler-side release queued for the same sequence.
        self.pending_runner_releases.retain(|&id| id != seq_id);
        for i in 0..self.running.len() {
            let seq = &mut self.running[i];
            if seq.id == seq_id {
                crate::log_warn!("Seq {} - cancel requested (status {})", seq.id, seq.status);
                seq.status = SequenceStatus::Finished;
                // The R7b resumable partial state: capture the GDN state at the
                // exact abort point (seq.len(), avoiding the KV-without-GDN gap from
                // the §10 diagnosis) before releasing, so a later request with the
                // same prefix restores both tiers and resumes at seq.len() (without
                // the full re-prefill). The snapshot is CPU-spill-eligible (the
                // two-tier LRU, the mamba_cache.rs), so it survives a GPU
                // re-provisioning of that slot.
                self.block_manager.commit_prefix_boundary(seq, seq.len());
                break;
            }
        }
        if let Some(pos) = self.waiting.iter().position(|seq| seq.id == seq_id) {
            let mut seq = self.waiting.remove(pos).unwrap();
            if seq.num_cached_tokens > 0 && seq.num_cached_tokens < seq.len() {
                crate::log_warn!(
                    "Seq {} - cancel requested mid-prefill (cached {} / {} tokens)",
                    seq.id,
                    seq.num_cached_tokens,
                    seq.len()
                );
            } else {
                crate::log_warn!("Seq {} - cancel requested (status {})", seq.id, seq.status);
            }
            seq.status = SequenceStatus::Finished;
            // The R7b: the mid-prefill cancel captures the GDN state at the
            // cached boundary (avoiding the dropped-KV re-prefill gap on resume).
            self.block_manager.commit_prefix_boundary(&seq, seq.num_cached_tokens);
            self.block_manager.deallocate(&seq);
        }
        if let Some(pos) = self.transferred.iter().position(|seq| seq.id == seq_id) {
            let seq = self.transferred.remove(pos).unwrap();
            crate::log_warn!(
                "Seq {} - cancel requested while awaiting PD transfer",
                seq.id
            );
            let _ = self.block_manager.try_release_remote_kvcache(seq.id);
        }
        self.release_cache(seq_id);
        self.running.retain(|seq| seq.id != seq_id);
    }

    #[allow(non_snake_case)]
    pub fn filter_prefill_finished(
        &mut self,
        scheduled_ids: &Vec<usize>,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut finished_seqs = Vec::new();
        let mut remove_ids = Vec::new();
        let mut chunked_info: Vec<(usize, usize, usize)> = Vec::new(); // (seq_id, cached, remain)
        let mut chunk_finished_info: Vec<(usize, usize)> = Vec::new(); // (seq_id, total_len)
        let chunk_size = self.cfg.effective_prefill_chunk_size();
        for (i, id) in scheduled_ids.iter().enumerate() {
            if *id < self.running.len() {
                let seq = &self.running[*id];
                // Use the adaptive chunk stamped by schedule() (fallback: static).
                let seq_chunk = seq.active_prefill_chunk.unwrap_or(chunk_size);
                let chunk_tokens = seq.prefill_chunk_tokens(seq_chunk);
                if chunk_tokens == 0 || seq.num_cached_tokens + chunk_tokens >= seq.len() {
                    let _ = self
                        .block_manager
                        .capture_mamba_prefix_state(seq, seq.len());
                    if seq.len() > seq_chunk {
                        chunk_finished_info.push((seq.id, seq.len()));
                    }
                    finished_seqs.push((i, seq.id));
                } else {
                    let _ = self
                        .block_manager
                        .capture_mamba_prefix_state(seq, seq.num_cached_tokens + chunk_tokens);
                    remove_ids.push(seq.id);
                    let mut seq = seq.clone();
                    seq.num_cached_tokens += chunk_tokens;
                    // The active mamba slot already contains the state at this
                    // chunk boundary. Keep the captured snapshot available for
                    // other requests, but do not force this in-progress request
                    // to revalidate it on the next scheduling pass.
                    seq.mamba_prefix_hash = None;
                    if seq.active_mamba_prefix_warmup_target().is_none() {
                        seq.clear_mamba_prefix_warmup();
                    }
                    seq.status = SequenceStatus::Waiting;
                    chunked_info.push((
                        seq.id,
                        seq.num_cached_tokens,
                        seq.len() - seq.num_cached_tokens,
                    ));
                    self.waiting.push_back(seq);
                }
            }
        }

        if !chunked_info.is_empty() {
            let total_chunked: usize = chunked_info.iter().map(|(_, c, _)| *c).sum();
            let seq_details: Vec<String> = chunked_info
                .iter()
                .map(|(id, cached, remain)| format!("{}:{}/{}", id, cached, cached + remain))
                .collect();
            crate::log_info!(
                "Chunk prefilled {} seq(s) [{}] ({} total tokens processed)",
                chunked_info.len(),
                seq_details.join(", "),
                total_chunked
            );
        }
        if !chunk_finished_info.is_empty() {
            let seq_ids: Vec<usize> = chunk_finished_info.iter().map(|(id, _)| *id).collect();
            let total: usize = chunk_finished_info.iter().map(|(_, len)| len).sum();
            crate::log_warn!(
                "Chunk prefill finished for {} seq(s) {:?} ({} total tokens)",
                chunk_finished_info.len(),
                seq_ids,
                total
            );
        }

        self.running.retain(|s| !remove_ids.contains(&s.id));
        let (indices, finished_ids): (Vec<usize>, Vec<usize>) = finished_seqs.into_iter().unzip();
        let finished_indices: Vec<usize> = finished_ids
            .iter()
            .filter_map(|&target_id| self.running.iter().position(|seq| seq.id == target_id))
            .collect();
        (indices, finished_indices)
    }

    pub fn try_transfer(&mut self, seq: &mut Sequence) -> bool {
        if !self.is_suitable_for_transfer(&seq) {
            return false;
        }
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        if let Some(tm) = seq.swapped_time {
            if cur_time - tm < SWAP_COOLING_PERIOD {
                return false;
            }
        }
        if let Ok(success) = self.block_manager.try_transfer_prefill(&seq) {
            // Client: Offload prefill request to PD server
            seq.swapped_time = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("Time went backwards")
                    .as_millis() as usize,
            );
            if success {
                crate::log_warn!(
                    "Prefill request (Seq {}, {} tokens) transfered to PD server.",
                    seq.id,
                    seq.len(),
                );
                seq.pd_first_token = None;
                self.transferred.push_back(seq.clone());
            } else {
                crate::log_warn!(
                    "Unable transfer prefill request (Seq {}) to PD server. Retry later...",
                    seq.id
                );
                self.waiting.push_front(seq.clone()); // push back, retry later
            }
            success
        } else {
            false
        }
    }

    pub fn try_swap_in(&mut self) {
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;

        for i in 0..self.cached.len() {
            let (status, swapped_time, seq_id) = {
                let seq = &self.cached[i];
                (seq.status, seq.swapped_time().unwrap_or(cur_time), seq.id)
            };
            if status != SequenceStatus::Swapped || cur_time - swapped_time < SWAP_COOLING_PERIOD {
                continue;
            }

            let available_kvcache_tokens = self.get_available_kv_tokens();
            let swap_in_required_tokens = self
                .block_manager
                .swap_in_required_blocks(&self.cached[i])
                .saturating_mul(self.cfg.block_size);
            if !self.block_manager.can_swap_in(&self.cached[i])
                || (available_kvcache_tokens.saturating_sub(swap_in_required_tokens)
                    < MIN_KVCACHE_TOKENS_LEFT_FOR_SWAP)
            {
                if !self.running.is_empty() {
                    // Wait for swap in
                    continue;
                }

                let required_blocks = self
                    .block_manager
                    .swap_in_required_blocks(&self.cached[i])
                    .saturating_add(1);
                let evicted = self
                    .block_manager
                    .evict_prefix_cache_until_free(required_blocks);
                if evicted > 0 {
                    crate::log_warn!(
                        "Evicted {} prefix cache block(s) for swap-in Seq {}.",
                        evicted,
                        seq_id
                    );
                    break;
                }

                let seq_id = self.cached[i].id;
                crate::log_error!(
                    "No KvCache left for swap in Seq {} — cancelling request.",
                    seq_id
                );
                self.cancel(seq_id);
                break;
            }

            let mut seq = self.cached.remove(i);
            seq.swapped_time = Some(cur_time);
            let partial_swap = self.block_manager.has_partial_cpu_swap(seq.id);
            if !partial_swap {
                seq.clear_block_table(); // reallocate block table since previous gpu blocks were freed
            }

            if let Err(_) = self.block_manager.ensure_allocate(&mut seq) {
                // Transient: GPU blocks may be available on the next schedule pass.
                // Keep the swapped CPU KV and GDN slot so swap-in can retry.
                if partial_swap {
                    self.block_manager
                        .rollback_partial_swap_in_allocation(&mut seq);
                }
                seq.status = SequenceStatus::Swapped;
                let swapped_class = seq.qos_class;
                self.cached.push(seq);
                // The §1 two-rate sensor (the fast path): the swap-out
                // records the request-state transition (the `Swapped` → the
                // `Paused` lifecycle index 5, the NInfer `EngineRequestState`).
                if self.policy.active() {
                    self.policy.record_request_state(swapped_class, 5);
                }
                break;
            }

            // Swap in data from CPU (if swapped out previously)
            match self.block_manager.swap_in(&mut seq) {
                Ok(_) => {
                    seq.status = SequenceStatus::Running;
                    crate::log_warn!("Seq {} is swapped in for execution!", seq.id);
                }
                Err(e) => {
                    crate::log_error!("Seq {} swap in failed: {:?}!", seq.id, e);
                    if partial_swap {
                        self.block_manager
                            .rollback_partial_swap_in_allocation(&mut seq);
                    } else {
                        self.block_manager.deallocate(&seq);
                        seq.clear_block_table();
                    }
                    if self.block_manager.has_cpu_swap(seq.id) {
                        // CPU swap data intact — retry on a later schedule pass.
                        seq.status = SequenceStatus::Swapped;
                        self.cached.push(seq);
                    } else {
                        // CPU swap state lost — abort and release the GDN slot.
                        self.request_runner_release(seq.id);
                    }
                    break;
                }
            }
            self.running.push(seq);
            break;
        }
    }

    // swap out one sequence a time
    pub fn try_swap_out(&mut self, idx: usize, is_running: bool) -> bool {
        if (cfg!(feature = "metal") || is_running && idx >= self.running.len())
            || (!is_running && idx >= self.cached.len())
        {
            return false;
        }

        let mut seq = if is_running {
            &mut self.running[idx]
        } else {
            &mut self.cached[idx]
        };

        // If sequence has blocks, attempt to swap to CPU.
        // If cannot swap, fallback.
        if !seq.block_table.is_empty()
            && (seq.status == SequenceStatus::Running || seq.status == SequenceStatus::Cached)
            && self.block_manager.can_swap_out(&seq)
        {
            let prefix_mode = self.block_manager.prefix_cache_enabled();
            if !prefix_mode {
                // make sure we have identical number of blocks when swapping in
                // for decoding
                if let Err(_) = self.block_manager.ensure_allocate(&mut seq) {
                    return false;
                }
            }
            match self.block_manager.swap_out(&mut seq) {
                Ok(_) => {
                    let mut seq = if is_running {
                        self.running.remove(idx)
                    } else {
                        // Even though the cached sequence swapped out,
                        // no need remove it from cached list since it can be recoved
                        self.cached.remove(idx)
                    };
                    if seq.status == SequenceStatus::Running {
                        seq.status = SequenceStatus::Swapped;
                    } else {
                        seq.status = SequenceStatus::FinishSwapped;
                    }
                    // seq.num_cached_tokens = seq.len();
                    if !prefix_mode {
                        self.block_manager.deallocate(&seq);
                    }
                    // The R7b resumable partial state: the swap-out keeps the CPU KV
                    // (the `swap_out_suffix`) but not the GDN state (the GDN is
                    // slot-resident on the GPU, the mamba_cache.rs). Capture the GDN
                    // state at the exact swap boundary (seq.len(), avoiding the
                    // re-run of the GDN recurrence from zero on a swap-in after a long
                    // preemption) so the resume restores both tiers. The snapshot is
                    // CPU-spill-eligible (the two-tier LRU), so it survives a GPU
                    // re-provisioning of that slot.
                    self.block_manager.commit_prefix_boundary(&seq, seq.len());
                    // block table need to be reallocated when swapping in
                    self.cached.push(seq.clone());
                    return true;
                }
                Err(e) => {
                    crate::log_warn!("Swap out failed for seq {}: {:?}", seq.id, e);
                }
            }
        }
        return false;
    }

    pub fn try_swap_out_by_id(&mut self, seq_id: usize, is_running: bool) -> bool {
        if is_running {
            if let Some(pos) = self.running.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, true);
            }
        } else {
            if let Some(pos) = self.cached.iter().position(|seq| seq.id == seq_id) {
                return self.try_swap_out(pos, false);
            }
        }
        false
    }

    pub fn is_pd_mode(&self) -> bool {
        self.pd_config.is_some()
    }

    pub fn is_pd_server(&self) -> bool {
        if let Some(p_cfg) = &self.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        }
    }

    pub fn is_suitable_for_transfer(&mut self, seq: &Sequence) -> bool {
        if seq.status == SequenceStatus::Swapped // swapped out sequence
            || seq.status == SequenceStatus::FinishSwapped // swapped out and finished sequence
            || seq.len() < PD_PREFILL_TRANSFER_NUM_TOKEN_THRESHOLD
        // prefill length < 128
        {
            return false;
        }

        // Check prefix cache: if most tokens are already cached, do local prefill
        let cached_tokens = self.block_manager.get_prefix_cache_match_tokens(seq);
        let new_tokens = seq.len().saturating_sub(cached_tokens);
        if cached_tokens > 0 && new_tokens < PD_LOCAL_PREFILL_NEW_TOKEN_THRESHOLD {
            crate::log_info!(
                "Seq {} has {} cached tokens, {} new tokens - doing local prefill",
                seq.id,
                cached_tokens,
                new_tokens
            );
            return false;
        }

        true
    }

    /// Client: Check for finished prefills and move them to the running queue.
    pub fn try_receive_kvcache(&mut self) -> Result<()> {
        let mut finished_seq_ids = Vec::new();
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        for idx in 0..self.transferred.len() {
            let seq_id = self.transferred[idx].id;
            if cur_time - self.transferred[idx].swapped_time().unwrap_or(cur_time)
                < PD_PREFILL_STATUS_CHECK_COOLING_PERIOD
            {
                continue;
            }

            self.transferred[idx].swapped_time = Some(cur_time);
            let status = self.block_manager.try_check_prefill_status(seq_id);
            if status.is_err() || !status.unwrap_or(false) {
                continue;
            }
            // We have the data. Can we allocate space for it?
            if !self
                .block_manager
                .can_allocate_without_prefix(&self.transferred[idx])
            {
                // Not enough memory right now. Put data back and try later.
                let evicted = self
                    .block_manager
                    .evict_prefix_cache_until_free(self.transferred[idx].num_blocks());
                if evicted > 0 {
                    crate::log_warn!(
                        "Evicted {} prefix cache block(s) for Seq {} KvCache receiving!",
                        evicted,
                        seq_id
                    );
                    break;
                }

                crate::log_warn!(
                    "KvCache Transfer: Seq {} prefill finished on PD server, but no blocks to receive. Will retry.",
                    seq_id
                );
                // For simplicity, we just break and retry next cycle.
                break;
            }

            // Allocate GPU blocks for the sequence
            self.block_manager
                .allocate_without_prefix(&mut self.transferred[idx])?;

            // Perform the actual KV cache data transfer
            let mut success = false;
            match self
                .block_manager
                .try_receive_kvcache(&self.transferred[idx])
            {
                Ok((ret, first_token, sending_time, num_cached_tokens)) => {
                    let seq = &mut self.transferred[idx];
                    success = ret;
                    if success {
                        seq.num_cached_tokens = num_cached_tokens;
                        // Update sequence and move to running
                        // The first token is generated on PD server,
                        // it has been transfered to client, but haven't been send to user
                        seq.append_token(first_token);
                        seq.pd_first_token = Some(first_token);
                        seq.status = SequenceStatus::Running;
                        self.running.push(seq.clone());
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize;
                        let transfer_duration = now - sending_time;

                        if transfer_duration > 10000 {
                            // Log a warning when a sequence takes an unusually long time to receive and swap-in.
                            // Possible causes: insufficient KV cache on server or client, or low communication bandwidth.
                            crate::log_warn!(
                                "KvCache Transfer: Seq {} prefill finished, but receive (with swap-in) time was unexpectedly long ({} s).",
                                seq.id,
                                transfer_duration / 1000
                            );
                        } else {
                            crate::log_info!(
                                "KvCache Transfer: Seq {} prefill finished and received in {} ms!",
                                seq.id,
                                transfer_duration
                            );
                        };

                        // Since KVCache Transfer involved here, the prefill speed might not accurate
                        // since receive remote kvcache requires sufficient local cache memory (sometime pending for kvcache)
                        // The actual prefill speed need to exclude the transfer time, for simplicity we didn't do that

                        self.block_manager.try_release_remote_kvcache(seq.id)?;
                    } else {
                        crate::log_error!(
                            "KvCache Transfer: Seq {} prefill finished but failed to receive. Aborting seq.",
                            seq.id,
                        );
                    }
                }
                Err(e) => {
                    crate::log_error!(
                        "KvCache Transfer: Failed to receive KV cache for seq {}: {}. Aborting seq.",
                        seq_id, e
                    );
                }
            }

            let seq = &mut self.transferred[idx];
            if !success {
                seq.status = SequenceStatus::Finished; // Mark as failed
                self.running.push(seq.clone());
            }
            finished_seq_ids.push(seq.id);
        }

        // Remove all processed sequences from the transferred queue
        self.transferred
            .retain(|s| !finished_seq_ids.contains(&s.id));
        Ok(())
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        self.block_manager.prefix_cache_blocks() * self.block_manager.get_block_size()
    }

    /// Per-seq prefix-cache hit count for response finalization. Falls back
    /// to `finished_reports` for seqs already swept by `clear_finished`.
    pub fn get_num_cached_tokens_for_seq(&self, seq_id: usize) -> Option<usize> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .chain(self.transferred.iter())
            .find(|s| s.id == seq_id)
            .map(|s| s.num_cached_tokens)
            .or_else(|| self.finished_reports.get(&seq_id).map(|r| r.cached_tokens))
    }

    /// The end-of-sequence report (cached tokens + QoS scheduling stats) for the
    /// server's per-sequence display. Searches the live pools, then the retained map.
    pub fn get_seq_report(&self, seq_id: usize) -> Option<SeqReport> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.cached.iter())
            .find(|s| s.id == seq_id)
            .map(|s| SeqReport {
cached_tokens: s.num_cached_tokens,
                sched: s.sched_stats.clone(),
                prompt_tokens: s.len(),
                prompt_time_s: 0.0,
                decode_time_s: 0.0,
            })
            .or_else(|| self.finished_reports.get(&seq_id).cloned())
    }

    fn remember_finished_report(&mut self, seq_id: usize, report: SeqReport) {
        self.finished_reports.insert(seq_id, report.clone());
        while self.finished_reports.len() > FINISHED_CACHED_TOKENS_MAX {
            let Some(oldest_seq_id) = self.finished_reports.keys().min().copied() else {
                break;
            };
            self.finished_reports.remove(&oldest_seq_id);
        }
        // The Prometheus telemetry hook (the QoS-gated, the no-op when QoS is off).
        // The record_seq_end increments the global counters + inserts the per-seq
        // gauges (the 60s aging). The report is cloned (the telemetry takes an
        // &SeqReport, the finished_reports retains the owned one).
        if self.policy.active() {
            Telemetry::global().record_seq_end(seq_id, &report);
        }
    }

    pub fn evict_prefix_cache_until_free(&mut self, min_free_blocks: usize) -> usize {
        self.block_manager
            .evict_prefix_cache_until_free(min_free_blocks)
    }

    pub fn evict_prefix_cache_blocks(&mut self, blocks: usize) -> usize {
        self.block_manager.evict_prefix_cache_blocks(blocks)
    }

    /// Demote the coldest running sequences to the CPU swap tier to free GPU blocks
    /// for a new admission (the relief-ladder step, C5). Victims are chosen oldest
    /// first by creation time (FIFO, not LRU-by-access, which would evict the hot
    /// agentic conversation). Each victim's suffix is swapped to host while its
    /// prefix stays on the GPU as the join key, and its GDN boundary is captured so
    /// the resume restores both tiers. The sequence moves to the cached list, where
    /// it is resumable. Returns the number of GPU blocks freed. This is a no-op when
    /// nothing is running (the single-request case, where the admission is for the
    /// only sequence).
    pub fn demote_for_admission(&mut self, need_blocks: usize) -> usize {
        let mut freed = 0usize;
        // The victims: running sequences that can be swapped out (their suffix is
        // sequence-owned, not a shared prefix). Sorted oldest first.
        let mut victims: Vec<(usize, usize)> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, seq)| self.block_manager.can_swap_out(seq))
            .map(|(idx, seq)| (idx, seq.created_time()))
            .collect();
        victims.sort_by_key(|(_, created)| *created);
        for (idx, _) in victims {
            if freed >= need_blocks {
                break;
            }
            // The demote removes the sequence from the running list, so the indices
            // shift. Re-find the victim by its id, not by the stale index.
            let Some(seq_id) = self.running.get(idx).map(|s| s.id) else {
                continue;
            };
            let before = self.block_manager.get_num_free_blocks();
            if self.try_swap_out_by_id(seq_id, true) {
                let after = self.block_manager.get_num_free_blocks();
                freed += after.saturating_sub(before);
            }
            if self.block_manager.get_num_free_blocks() >= need_blocks {
                break;
            }
        }
        freed
    }

    /// Demote the single coldest running sequence to the host swap tier (the QoS KV
    /// pressure relief, the no a block-count target). The oldest by creation time
    /// that can be swapped out (its suffix is sequence-owned). Returns the number of
    /// GPU blocks freed (the 0 when nothing is running or nothing is swappable).
    /// This is the QoS actuator: when the physical pressure crosses the calibrated
    /// threshold, the coldest conversation is moved to host, freeing GPU blocks for
    /// the hot requests. The sequence is resumable (the prefix stays on the GPU, the
    /// GDN boundary is captured).
    pub fn demote_coldest_running(&mut self) -> usize {
        // The oldest swappable running sequence (the FIFO victim, the no the LRU,
        // which would evict the hot agentic conversation).
        let mut victim: Option<(usize, usize)> = None; // (idx, created_time)
        for (idx, seq) in self.running.iter().enumerate() {
            if !self.block_manager.can_swap_out(seq) {
                continue;
            }
            match victim {
                Some((_, min_created)) if seq.created_time() < min_created => {
                    victim = Some((idx, seq.created_time()));
                }
                None => victim = Some((idx, seq.created_time())),
                Some(_) => {}
            }
        }
        let Some((idx, _)) = victim else {
            return 0;
        };
        let seq_id = self.running[idx].id;
        let before = self.block_manager.get_num_free_blocks();
        if self.try_swap_out_by_id(seq_id, true) {
            let after = self.block_manager.get_num_free_blocks();
            after.saturating_sub(before)
        } else {
            0
        }
    }

    fn evict_prefix_cache_under_pressure(&mut self) -> usize {
        let cached_blocks = self.block_manager.prefix_cache_blocks();
        if cached_blocks == 0 {
            return 0;
        }
        let blocks = ((cached_blocks as f32) * PREFIX_CACHE_PRESSURE_EVICT_PERCENT).ceil() as usize;
        let blocks = blocks.max(1);
        self.block_manager.evict_prefix_cache_blocks(blocks)
    }

    pub fn get_available_kv_tokens(&self) -> usize {
        let free_blocks = self.block_manager.get_num_free_blocks();
        free_blocks * self.block_manager.get_block_size()
    }

    pub fn get_total_kv_tokens(&self) -> usize {
        let total_blocks = self.block_manager.get_num_total_blocks();
        total_blocks * self.block_manager.get_block_size()
    }

    pub fn get_cpu_swap_usage(&self) -> (f32, f32) {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let kvcache_memory_gb = self.cfg.kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        if cfg!(feature = "metal") {
            // Metal use unified memory, no cpu swap memory used
            (0f32, 0f32)
        } else {
            let cpu_kvcache_memory_gb = kvcache_memory_gb * self.cfg.cpu_mem_fold.unwrap_or(0.5f32);
            (
                self.block_manager.get_cpu_swap_usage() * cpu_kvcache_memory_gb,
                cpu_kvcache_memory_gb,
            )
        }
    }

    pub fn print_free_blocks(&self) {
        // The console print reads the same source as the /metrics export (the
        // resource_gauges, the DRY single source). The no duplicated computation.
        let g = self.resource_gauges();
        #[cfg(feature = "cuda")]
        let cpu_swap_log = format!(
            "CPU swap used {:.1}% ({:.2}GB/{:.2}GB)",
            g.cpu_swap_used_frac * 100.0f32,
            g.cpu_swap_used_gb,
            g.cpu_swap_total_gb,
        );
        #[cfg(not(feature = "cuda"))]
        let cpu_swap_log = "".to_string();

        crate::log_info!(
            "GPU Kvcache: {} blocks ({} tokens) free, used {:.1}% ({:.2}GB/{:.2}GB); {}",
            g.kv_free_blocks,
            g.kv_free_tokens,
            g.kv_used_percent,
            g.kv_used_gb,
            g.kv_total_gb,
            cpu_swap_log,
        );
        if g.mamba_slots_total > 0 {
            crate::log_info!(
                "GPU MambaState: {} / {} slots used ({:.1}%), approx {:.2}GB/{:.2}GB (slot {:.2}MB)",
                g.mamba_slots_used,
                g.mamba_slots_total,
                g.mamba_slots_used as f32 * 100.0f32 / g.mamba_slots_total as f32,
                g.mamba_used_gb,
                g.mamba_budget_gb,
                g.mamba_slot_mb
            );
        }
    }

    pub fn kv_cache_usage_percent(&self) -> f32 {
        let total_blocks = self.block_manager.get_num_total_blocks();
        let free_blocks = self.block_manager.get_num_free_blocks();
        1.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks as f32)
    }

    /// The runtime resource gauges (the single source for both the console
    /// `print_free_blocks` + the `/metrics` export). Returns the KV block
    /// occupancy, the CPU-swap usage, the Mamba/GDN slot occupancy, + the runner
    /// queue depths (the running/waiting/cached/transferred counts). One
    /// snapshot keeps the values consistent (the no multiple round-trips that
    /// could straddle a state change).
    pub fn resource_gauges(&self) -> ResourceGauges {
        const SIZE_IN_GB: usize = 1024 * 1024 * 1024;
        let total_blocks = self.block_manager.get_num_total_blocks();
        let free_blocks = self.block_manager.get_num_free_blocks();
        let block_size = self.block_manager.get_block_size();
        let kvcache_memory_bytes = self.cfg.kvcache_memory_bytes;
        let used_percent = 100.0f32 - (free_blocks as f32 * 1.0f32 / total_blocks.max(1) as f32) * 100.0f32;
        let kvcache_gb = kvcache_memory_bytes as f32 / SIZE_IN_GB as f32;
        #[cfg(feature = "cuda")]
        let cpu_swap = {
            let cpu_kvcache_memory_gb = kvcache_gb * self.cfg.cpu_mem_fold.unwrap_or(0.5f32);
            let swap_used_frac = self.block_manager.get_cpu_swap_usage();
            (swap_used_frac, swap_used_frac * cpu_kvcache_memory_gb, cpu_kvcache_memory_gb)
        };
        #[cfg(not(feature = "cuda"))]
        let cpu_swap = (0.0f32, 0.0f32, 0.0f32);

        // The Mamba/GDN slot occupancy (the `0` when the model is not hybrid).
        let (mamba_used, mamba_total, mamba_used_gb, mamba_budget_gb, mamba_slot_mb) =
            if let Some(capacity) = self.cfg.mamba_cache_capacity {
                if capacity > 0 && self.cfg.mamba_slot_bytes > 0 {
                    let active_slots = (self.running.len()
                        + self.waiting.len()
                        + self.cached.len()
                        + self.transferred.len())
                        .min(capacity);
                    let slot_mb = self.cfg.mamba_slot_bytes as f32 / 1024.0f32 / 1024.0f32;
                    let used_gb = (active_slots * self.cfg.mamba_slot_bytes) as f32 / SIZE_IN_GB as f32;
                    let budget_gb = self.cfg.mamba_memory_bytes as f32 / SIZE_IN_GB as f32;
                    (active_slots, capacity, used_gb, budget_gb, slot_mb)
                } else {
                    (0, 0, 0.0, 0.0, 0.0)
                }
            } else {
                (0, 0, 0.0, 0.0, 0.0)
            };

        ResourceGauges {
            kv_free_blocks: free_blocks,
            kv_total_blocks: total_blocks,
            kv_free_tokens: free_blocks * block_size,
            kv_used_percent: used_percent,
            kv_used_gb: used_percent / 100.0f32 * kvcache_gb,
            kv_total_gb: kvcache_gb,
            cpu_swap_used_frac: cpu_swap.0,
            cpu_swap_used_gb: cpu_swap.1,
            cpu_swap_total_gb: cpu_swap.2,
            mamba_slots_used: mamba_used,
            mamba_slots_total: mamba_total,
            mamba_used_gb: mamba_used_gb,
            mamba_budget_gb: mamba_budget_gb,
            mamba_slot_mb: mamba_slot_mb,
            seq_running: self.running.len(),
            seq_waiting: self.waiting.len(),
            seq_cached: self.cached.len(),
            seq_transferred: self.transferred.len(),
            mmu_pressure: self.physical_pressure,
            mmu_idle_vram_bytes: self.mmu_idle_vram_bytes,
            mmu_total_vram_bytes: self.mmu_total_vram_bytes,
        }
    }

    /// Check if the given token is a tool call end token
    /// This supports both:
    /// 1. Explicit tool call end tokens (e.g., </tool_call> in XML format)
    /// 2. JSON end token "}" combined with Regex validation for {..."name":..., "arguments":...} pattern
    ///    (only used as a fallback for models without explicit end token IDs)
    pub fn is_tool_call_end(&self, token: u32, idx: usize) -> bool {
        // 1. Check for explicit tool call end tokens (XML style)
        if self.tool_call_end_token_ids.contains(&token) {
            // If we know start token IDs for this model, only treat end markers as structural
            // when a start marker has already appeared in generated output.
            if !self.tool_call_start_token_ids.is_empty() {
                let has_start = self.running[idx]
                    .output_ids
                    .iter()
                    .any(|id| self.tool_call_start_token_ids.contains(id));
                if !has_start {
                    return false;
                }
            }
            return true;
        }

        // 2. JSON regex fallback — only for models without explicit end token IDs.
        // Models with known end tokens (e.g. Qwen3's </tool_call> = 151658) should
        // rely solely on the token ID check above. The regex
        //   \{\s*"name"\s*:.*"arguments"\s*:.*\}\s*$
        // can match prematurely at the inner `}` of the arguments dict before the
        // outer `}` and the actual end token are generated.
        if !self.tool_call_end_token_ids.is_empty() {
            return false;
        }

        if self.json_end_token_id == Some(token) {
            if let Some(tokenizer) = &self.tokenizer {
                let mut temp_output = self.running[idx].output_ids.to_vec();
                temp_output.push(token);

                if let Ok(decoded) = tokenizer.decode(&temp_output, true) {
                    if self.tool_call_regex.is_match(&decoded) {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn stop_sequence_match_index(&self, token: u32, seq: &Sequence) -> Option<usize> {
        let Some(stop_sequences) = &seq.sampling_params.stop_token_ids else {
            return None;
        };
        if stop_sequences.is_empty() {
            return None;
        }

        for (idx, stop) in stop_sequences.iter().enumerate() {
            if stop.is_empty() {
                continue;
            }
            if stop.len() == 1 {
                if stop[0] == token {
                    return Some(idx);
                }
                continue;
            }

            let prior_len = seq.output_ids.len();
            if stop.len() - 1 > prior_len {
                continue;
            }
            let start_idx = prior_len + 1 - stop.len();
            if seq.output_ids[start_idx..] == stop[..stop.len() - 1]
                && stop[stop.len() - 1] == token
            {
                return Some(idx);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::active_sequence_limit;

    #[test]
    fn active_sequence_limit_uses_mamba_capacity_for_hybrid_models() {
        assert_eq!(active_sequence_limit(8, Some(5)), 5);
        assert_eq!(active_sequence_limit(5, Some(9)), 5);
    }

    #[test]
    fn active_sequence_limit_respects_parallel_limit_for_non_hybrid_models() {
        assert_eq!(active_sequence_limit(1, None), 1);
    }

    #[test]
    fn active_sequence_limit_never_exceeds_mamba_or_parallel_capacity() {
        assert_eq!(active_sequence_limit(12, Some(4)), 4);
        assert_eq!(active_sequence_limit(0, None), 1);
    }

    #[test]
    fn active_sequence_limit_preserves_zero_mamba_as_disabled() {
        assert_eq!(active_sequence_limit(7, Some(0)), 7);
    }
}
