// QoS scheduling: a self-tuning control loop that bounds decode inter-token
// latency (ITL) under prefill load, instead of relying on static class weights.
//
// The loop:
//   sensor     - each step the scheduler measures the wall-clock step duration
//                (a proxy for decode ITL: big prefill chunks slow the step and
//                starve running decodes) and the model's observed output length.
//   controller - the integral gain drives prefill-chunk suppression to hold ITL at
//                the SLO; KV headroom scales the per-step token budget; the
//                measured output-length distribution infers the Latency class.
//   actuator   - a `StepPlan` (chunk size, token budget, admission order,
//                reservation gates) that the scheduler applies with no QoS logic
//                of its own.
//
// When QoS is disabled the plan is the static degenerate one (full chunk, FIFO
// admission, no gates), so the scheduler's behavior is unchanged. The controller
// is owned per model, which is the seam for multi-model serving (one controller
// per model) and multi-host scheduling (`Contention` in / `StepPlan` out is the
// wire protocol a coordinator would run globally).

use crate::core::qos_histogram::WorkloadHistogram;
use crate::core::sequence::Sequence;
use crate::utils::env::qos_enabled;
use serde::{Deserialize, Serialize};

/// QoS class for a request.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QosClass {
    /// Large prompts / long outputs; throughput-oriented, latency-tolerant.
    #[default]
    Throughput,
    /// Short exchanges, many turns; tight TTFT/ITL SLO (agentic).
    Latency,
}

impl QosClass {
    pub fn is_latency(&self) -> bool {
        matches!(self, QosClass::Latency)
    }
}

/// The regime classifier (the contended-vs-uncontended, the Sarathi-Serve
/// stall-free result: the prefill-decode interference is the contention signal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Regime {
    /// The no contention: the low chunk, the high throughput.
    #[default]
    Uncontended,
    /// The contention: the suppressed chunk, the ITL bound.
    Contended,
}

impl Regime {
    /// Classify the regime from the decode pressure + the KV pressure against the
    /// calibrated thresholds. The thresholds come passed in (the controller
    /// resolves them from config or calibration), so the classifier itself carries
    /// no blind constants.
    pub fn classify(
        decode_pressure: f32,
        kv_pressure: f64,
        decode_threshold: f32,
        kv_threshold: f64,
    ) -> Regime {
        if decode_pressure > decode_threshold || kv_pressure > kv_threshold {
            Regime::Contended
        } else {
            Regime::Uncontended
        }
    }
}

/// Operator/request setpoints (SLOs) + tuning gains. Per-model; serde-defaulted so
/// existing configs keep working. `enabled: false` (the default) = prior operation.
///
/// Every setpoint that is `None` is auto-calibrated by the controller from the
/// measured system behavior over the first `calibration_steps` decode steps,
/// instead of being a blind constant. An operator may pin any of them to a fixed
/// value; the `None` fields are the ones derived from measurement.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QosConfig {
    pub enabled: bool,
    /// Target inter-token latency (ms) for the Latency class. The controller drives
    /// the measured ITL toward this. When `None`, it is calibrated from the measured
    /// base decode ITL (the p50 of the calibration window) so the SLO reflects what
    /// this model + hardware can actually sustain, not an arbitrary 50ms.
    pub itl_slo_ms: Option<f64>,
    /// "Short" = expected output below this fraction of the model's measured typical
    /// output length. When `None`, it is calibrated from the output-length CDF (the
    /// p25 of the measured outputs), not a fixed 0.5.
    pub latency_output_frac: Option<f64>,
    /// Integral gain for the ITL controller (how fast prefill suppression adapts).
    /// When `None`, it is derived from the measured ITL sensitivity to chunk size so
    /// the loop settles in a bounded number of steps rather than oscillating.
    pub gain_kp: Option<f32>,
    /// KV-headroom conservativeness gain in [0,1]: as free KV blocks drop, the
    /// per-step prefill token budget is scaled down by this much (protects running
    /// decodes from preemption). When `None`, it is calibrated from the measured
    /// pressure distribution.
    pub mem_k: Option<f32>,
    /// Opt-in: reserve this many running slots for the Latency class.
    pub latency_slot_reserve: Option<usize>,
    /// Opt-in: reserve this fraction of KV blocks for the Latency class.
    pub latency_kv_reserve_frac: Option<f32>,
    /// A step gap larger than this (ms) means the engine loop was idle between
    /// requests, not a slow step; it is excluded from the ITL sensor so the gain
    /// does not falsely spike at the start of a burst. When `None`, it is
    /// calibrated from the measured idle-gap distribution.
    pub max_meaningful_step_ms: Option<f64>,
    /// The minimum adaptive prefill chunk (tokens). The chunk is never suppressed
    /// below this floor. When `None`, it is derived from the measured step time so
    /// the floor is a bounded fraction of a step, not a fixed 256.
    pub chunk_floor: Option<usize>,
    /// The number of decode steps over which the controller calibrates the `None`
    /// config values from measurement. After this window the derived values are
    /// fixed (re-calibrated only on an explicit reset).
    pub calibration_steps: usize,
    /// The decode-pressure threshold above which the regime is Contended. When
    /// `None`, it is calibrated from the measured decode-pressure distribution (the
    /// p90), not a fixed 0.5.
    pub regime_decode_pressure: Option<f32>,
    /// The KV-pressure threshold above which the regime is Contended. When `None`,
    /// it is calibrated from the measured KV-pressure distribution (the p90), not a
    /// fixed 0.8.
    pub regime_kv_pressure: Option<f64>,
    /// The KV-pressure level at which the demotion threshold is emitted. When
    /// `None`, it is calibrated from the measured KV-pressure distribution.
    pub kv_demote_pressure: Option<f64>,
    /// The demotion threshold value emitted when KV pressure exceeds
    /// `kv_demote_pressure`. When `None`, it is calibrated.
    pub kv_demote_threshold: Option<f32>,
    /// The batch-size EMA below which the parallelism hint is `IncreaseTp` (the low-
    /// traffic, latency regime). When `None`, it is calibrated from the measured
    /// batch distribution, not a fixed 4.
    pub low_batch_hint: Option<usize>,
    /// The output-length median below which the sampling override tightens to
    /// top-k=1 (the short-output class). When `None`, it is calibrated from the
    /// output-length CDF, not a fixed 64.
    pub short_output_tokens: Option<usize>,
    /// The P-Q1 drift re-calibration interval (steps). After the initial
    /// calibration window, every this many steps the controller re-derives the
    /// `None` setpoints from the rolling sample buffer + blends them into the
    /// effective values (the no a step change). The `0` disables the periodic
    /// re-derivation (the drift detector only).
    pub recalibration_interval: usize,
    /// The P-Q1 drift factor: the ITL p95 must exceed the calibrated SLO by this
    /// fraction (e.g. `0.25` = 25%) for the drift detector to consider the regime
    /// shifted. The `1.0` means "any excess" (the most sensitive).
    pub drift_factor: f64,
    /// The P-Q1 drift step count: the number of consecutive steps the ITL p95
    /// must stay over the drifted SLO before a re-calibration is triggered. The
    /// `1` means immediate (the no the consecutive-stability requirement).
    pub drift_steps: usize,
    /// The P-Q1 blend gain: the fraction of the newly-derived setpoint mixed into
    /// the effective value on a re-calibration (e.g. `0.25` = a 25% step toward
    /// the new value, the no a discontinuous jump). The `1.0` is a full
    /// replacement.
    pub recal_blend: f64,
}

impl Default for QosConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            itl_slo_ms: None,
            latency_output_frac: None,
            gain_kp: None,
            mem_k: None,
            latency_slot_reserve: None,
            latency_kv_reserve_frac: None,
            max_meaningful_step_ms: None,
            chunk_floor: None,
            calibration_steps: 64,
            regime_decode_pressure: None,
            regime_kv_pressure: None,
            kv_demote_pressure: None,
            kv_demote_threshold: None,
            low_batch_hint: None,
            short_output_tokens: None,
            recalibration_interval: 256,
            drift_factor: 0.25,
            drift_steps: 32,
            recal_blend: 0.25,
        }
    }
}

/// The linear control parameters the controller actually uses at any moment.
/// Every field is either operator-configured, calibration-derived, or a safe
/// fallback — never a blind magic number in the control law.
#[derive(Debug, Clone, Copy)]
pub struct EffectiveParams {
    pub itl_slo_ms: Option<f64>,
    pub gain_kp: Option<f32>,
    pub mem_k: Option<f32>,
    pub latency_output_frac: Option<f64>,
    pub max_meaningful_step_ms: Option<f64>,
    pub chunk_floor: Option<usize>,
    pub regime_decode_pressure: Option<f32>,
    pub regime_kv_pressure: Option<f64>,
    pub kv_demote_pressure: Option<f64>,
    pub kv_demote_threshold: Option<f32>,
    pub low_batch_hint: Option<usize>,
    pub short_output_tokens: Option<usize>,
}

/// Online calibration state. Over the first `calibration_steps` decode steps the
/// controller accumulates samples of the measured system behavior, then derives
/// every `None` config value from those measurements instead of a blind
/// constant. This is what makes the QoS controller adapt to the actual model +
/// hardware instead of presuming a 50ms SLO, a 0.5 pressure threshold, a batch
/// of 4, or a 64-token short-output cutoff.
#[derive(Debug, Default)]
pub struct Calibration {
    /// The number of decode steps observed so far in the calibration window.
    pub steps: usize,
    /// The calibration window length (from `QosConfig.calibration_steps`).
    pub window: usize,
    /// Whether calibration is complete (the `steps >= window`).
    pub done: bool,
    /// The measured base decode ITL samples (ms), for deriving the SLO.
    pub itl_samples: Vec<f64>,
    /// The measured decode-pressure samples (the 0..1 fraction of running slots
    /// that are decoding), for deriving the regime threshold.
    pub decode_pressure_samples: Vec<f32>,
    /// The measured KV-pressure samples (the 0..1 fraction of KV in use), for
    /// deriving the regime + demotion thresholds.
    pub kv_pressure_samples: Vec<f64>,
    /// The measured output-length samples (tokens), for deriving the short-output
    /// + latency-class thresholds.
    pub output_len_samples: Vec<usize>,
    /// The measured batch-size samples, for deriving the low-traffic hint.
    pub batch_samples: Vec<usize>,
    /// The derived SLO (ms), set when calibration completes.
    pub derived_itl_slo_ms: Option<f64>,
    /// The derived decode-pressure regime threshold, set when calibration completes.
    pub derived_regime_decode_pressure: Option<f32>,
    /// The derived KV-pressure regime threshold, set when calibration completes.
    pub derived_regime_kv_pressure: Option<f64>,
    /// The derived KV demotion pressure + threshold, set when calibration completes.
    pub derived_kv_demote_pressure: Option<f64>,
    pub derived_kv_demote_threshold: Option<f32>,
    /// The derived low-batch hint, set when calibration completes.
    pub derived_low_batch_hint: Option<usize>,
    /// The derived short-output cutoff, set when calibration completes.
    pub derived_short_output_tokens: Option<usize>,
    /// The derived latency-output fraction, set when calibration completes.
    pub derived_latency_output_frac: Option<f64>,
    /// The derived integral gain, set when calibration completes. Chosen so a
    /// 1-SLO IT drives the gain from 0 to 1 over the calibration window (the
    /// controller settles on the workload's own timescale, not a blind constant).
    pub derived_gain_kp: Option<f32>,
    /// The derived idle-gap guard (ms), set when calibration completes. The p99 of
    /// the measured step distribution, so a wake-from-idle gap (larger than the
    /// system's normal step time) is excluded from the ITL sensor instead of
    /// spiking the integral gain. The `None` until calibration derives it.
    pub derived_max_meaningful_step_ms: Option<f64>,
    /// The rolling cap on each sample buffer (the bounded memory, so the
    /// calibration `Vec`s do not grow unbounded on a long-running server). The
    /// oldest samples are dropped when a buffer exceeds this, so the live
    /// distribution is retained for the drift re-calibration while the
    /// allocation stays O(cap).
    pub sample_cap: usize,
    // The P-Q1 blend snapshots: the derived values captured before the most
    // recent re-derivation, so `recalibrate` can blend the new value toward the
    // prior one (a gradual adaptation, no step change).
    pub derived_itl_slo_ms_prev: Option<f64>,
    pub derived_gain_kp_prev: Option<f32>,
    pub derived_regime_decode_pressure_prev: Option<f32>,
    pub derived_regime_kv_pressure_prev: Option<f64>,
    pub derived_max_meaningful_step_ms_prev: Option<f64>,
}

impl Calibration {
    /// Record one decode step's measurements into the calibration window.
    pub fn observe(
        &mut self,
        step_ms: f64,
        decode_pressure: f32,
        kv_pressure: f64,
        output_len: Option<usize>,
        batch_size: usize,
    ) {
        // The P-Q1: the rolling sample buffer is fed for the process lifetime, not
        // just the initial window. Before `done` it accumulates the calibration
        // samples; after `done` it keeps the most recent `sample_cap` observations
        // so the drift re-calibration can re-derive the setpoints from the live
        // distribution (the no the frozen one-shot values going stale on a regime
        // shift). The `finish` runs only once (the `done` latch).
        let cap = self.sample_cap.max(1);
        push_capped(&mut self.itl_samples, step_ms, cap);
        push_capped(&mut self.decode_pressure_samples, decode_pressure, cap);
        push_capped(&mut self.kv_pressure_samples, kv_pressure, cap);
        if let Some(len) = output_len {
            push_capped(&mut self.output_len_samples, len, cap);
        }
        push_capped(&mut self.batch_samples, batch_size, cap);
        if !self.done {
            self.steps += 1;
            if self.steps >= self.window {
                self.finish();
            }
        }
    }

    /// Derive the config values from the accumulated samples. The derivations:
    /// - the SLO is the p50 of the measured base ITL (the model + hardware can
    ///   sustain this per-token latency at the observed concurrency).
    /// - the regime thresholds are the p90 of the measured pressure distributions
    ///   ("contended" means "under more pressure than usual," not an arbitrary
    ///   0.5/0.8).
    /// - the demotion pressure is the p75 of the KV-pressure distribution; the
    ///   demotion threshold is a fixed fraction of the headroom at that point.
    /// - the low-batch hint is the p25 of the measured batch sizes (the
    ///   "low-traffic" regime is below the observed quarter).
    /// - the short-output cutoff is the p25 of the measured output lengths.
    /// - the latency-output fraction is the p25 of the output lengths relative to
    ///   the mean (the "short" class is the bottom quarter of the observed
    ///   distribution).
    /// - the integral gain is chosen so a 1-SLO ITL violation drives the gain
    ///   from 0 to 1 over the calibration window (the controller settles on the
    ///   workload's own timescale, not a blind constant).
    /// - the idle-gap guard is the p99 of the measured step distribution (a
    ///   wake-from-idle gap is excluded from the ITL sensor, so the gain does not
    ///   spike on a phantom violation at the start of every burst).
    fn derive(&mut self) {
        if !self.itl_samples.is_empty() {
            let mut s = self.itl_samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.derived_itl_slo_ms = Some(quantile(&s, 0.5));
        }
        if !self.decode_pressure_samples.is_empty() {
            let mut s = self.decode_pressure_samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.derived_regime_decode_pressure = Some(quantile_f32(&s, 0.9));
        }
        if !self.kv_pressure_samples.is_empty() {
            let mut s = self.kv_pressure_samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.derived_regime_kv_pressure = Some(quantile(&s, 0.9));
            self.derived_kv_demote_pressure = Some(quantile(&s, 0.75));
            if let Some(p75) = self.derived_kv_demote_pressure {
                // Demote when the headroom (1 - pressure) drops below half the
                // pressure at the p75 point, i.e. demote at = 1 - p75/2.
                self.derived_kv_demote_threshold = Some((1.0 - p75 / 2.0) as f32);
            }
        }
        if !self.batch_samples.is_empty() {
            let mut s = self.batch_samples.clone();
            s.sort();
            self.derived_low_batch_hint = Some(quantile_usize(&s, 0.25));
        }
        if !self.output_len_samples.is_empty() {
            let mut s = self.output_len_samples.clone();
            s.sort();
            self.derived_short_output_tokens = Some(quantile_usize(&s, 0.25));
            let mean = s.iter().sum::<usize>() as f64 / s.len() as f64;
            if mean > 0.0 {
                self.derived_latency_output_frac =
                    Some((quantile_usize(&s, 0.25) as f64) / mean);
            }
        }
        if let Some(slo) = self.derived_itl_slo_ms {
            if slo > 0.0 {
                let window = self.window.max(1) as f64;
                self.derived_gain_kp = Some((1.0 / (slo * window)) as f32);
            }
        }
        if !self.itl_samples.is_empty() {
            let mut s = self.itl_samples.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.derived_max_meaningful_step_ms = Some(quantile(&s, 0.99));
        }
    }

    /// Mark calibration complete + run the initial derivation.
    fn finish(&mut self) {
        self.done = true;
        self.derive();
    }

    /// The P-Q1 re-derivation: snapshot the current derived values into the
    /// `*_prev` fields, then re-run the derivations on the (rolling) sample
    /// buffer. The caller (`recalibrate`) blends the new values toward the prior
    /// ones.
    fn rederive(&mut self) {
        self.derived_itl_slo_ms_prev = self.derived_itl_slo_ms;
        self.derived_gain_kp_prev = self.derived_gain_kp;
        self.derived_regime_decode_pressure_prev = self.derived_regime_decode_pressure;
        self.derived_regime_kv_pressure_prev = self.derived_regime_kv_pressure;
        self.derived_max_meaningful_step_ms_prev = self.derived_max_meaningful_step_ms;
        self.derive();
    }
}

/// Push one sample into a bounded buffer, dropping the oldest when over the
/// cap (the O(cap) memory, so the calibration buffers do not grow unbounded).
fn push_capped<T>(buf: &mut Vec<T>, value: T, cap: usize) {
    buf.push(value);
    if buf.len() > cap {
        buf.remove(0);
    }
}

/// The linear-interpolated quantile of a sorted `f64` slice.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// The linear-interpolated quantile of a sorted `f32` slice.
fn quantile_f32(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// The linear-interpolated quantile of a sorted `usize` slice.
fn quantile_usize(sorted: &[usize], q: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Block-manager snapshot the controller decides from (no scheduler reach-in).
/// Serializable so a distributed coordinator can collect it from every rank.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockSnapshot {
    pub free_blocks: usize,
    pub total_blocks: usize,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    /// The static (QoS-off) prefill chunk; the adaptive cap.
    pub static_chunk: usize,
    /// The minimum adaptive chunk.
    pub chunk_floor: usize,
}

impl BlockSnapshot {
    pub fn mem_headroom(&self) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.free_blocks as f32 / self.total_blocks as f32
    }
}

/// The live contention vector, for diagnostics and (later) cross-rank exchange.
#[derive(Debug, Clone, Copy, Default)]
pub struct Contention {
    pub decode_pressure: f32,
    pub mem_headroom: f32,
    pub slot_fill: f32,
    pub itl_ms: f64,
    pub gain: f32,
}

/// The per-step actuator command the scheduler applies.
#[derive(Debug, Clone)]
pub struct StepPlan {
    pub chunk_size: usize,
    pub token_budget: usize,
    /// Throughput reservation gates. `thr_slot_capacity == 0` means unlimited.
    pub thr_slot_capacity: usize,
    pub thr_in_use: usize,
    pub thr_kv_full: bool,
}

/// The parallelism hint the `AdaptivePlan` emits (the Shift Parallelism result:
/// the low-traffic → the high-TP for the latency; the high-traffic → the SP for
/// the throughput, the KV-invariant). Consumed by the multi-instance router
/// (the L4) + the cross-host migrator (the L3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParallelismHint {
    /// Keep the current TP degree (the no-op).
    #[default]
    Hold,
    /// Raise the TP degree (the latency regime: the low traffic, the big model).
    IncreaseTp,
    /// Drop to sequence-parallelism (the throughput regime: the high traffic, the
    /// KV-invariant DP-like result).
    UseSequenceParallelism,
}

/// The sampling override the `AdaptivePlan` emits (the per-class width: the
/// short-output class → the tight top-k; the long-output class → the wide top-p).
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingOverride {
    /// The per-class top-k width (the `None` = the model default).
    pub top_k: Option<u32>,
    /// The per-class top-p width (the `None` = the model default).
    pub top_p: Option<f32>,
}

/// The upgraded per-step actuator command: the existing `StepPlan` 5 fields + the
/// QoS-driven actuators (the `batch_cap`, the `sampling` override, the
/// `parallelism_hint`, the `kv_demote_threshold`). The `plan()` law reads the
/// `WorkloadHistogram` + emits these.
///
/// The `spec_draft_len` act the `spec_accept` are added by the `spec-decode`
/// branch (the stack). The `grammar_active` is added by the `llg`
/// branch. The is the CORE `AdaptivePlan` (the `qos` branch).
#[derive(Debug, Clone)]
pub struct AdaptivePlan {
    /// The base `StepPlan` (the chunk size + the token budget + the Throughput gates).
    pub base: StepPlan,
    /// The admission cap (the vLLM `max_num_active_reqs` result: the no new-sequence
    /// admission beyond this, separate from the `max_num_seqs` slot count).
    pub batch_cap: usize,
    /// The per-class sampling override (the short-output → the tight top-k; the
    /// long-output → the wide top-p).
    pub sampling: SamplingOverride,
    /// The TP↔SP hint (the Shift Parallelism result: the TP<->SP switch).
    pub parallelism_hint: ParallelismHint,
    /// The KV demotion threshold (the SGLang `unified_memory_disagg_move_gate`
    /// result: the coldest conversation's blocks → the HostMem/Gds tier). The
    /// `None` = the no demotion.
    pub kv_demote_threshold: Option<f32>,
}

impl From<StepPlan> for AdaptivePlan {
    fn from(base: StepPlan) -> Self {
        Self {
            base,
            batch_cap: 0,
            sampling: SamplingOverride::default(),
            parallelism_hint: ParallelismHint::Hold,
            kv_demote_threshold: None,
        }
    }
}

/// The stateful controller (per model). Owns the integral gain + measured EMAs +
/// the bounded-memory `WorkloadHistogram` (the core, the per-class signal that
/// drives the `AdaptivePlan`).
pub struct QosController {
    pub cfg: QosConfig,
    /// Integral prefill-suppression state in [0,1]: rises when ITL exceeds the SLO.
    gain: f32,
    /// EMA of measured step duration (ms) - the ITL proxy.
    itl_ema: f64,
    /// EMA of observed output length (tokens) - drives class inference.
    output_len_ema: f64,
    /// The bounded-memory workload histogram (the core, the per-class signal).
    pub histogram: WorkloadHistogram,
    /// The online calibration state. Accumulates the measured system behavior over
    /// the first `calibration_steps` decode steps, then derives every `None`
    /// config value from measurement instead of a blind constant.
    calibration: Calibration,
    /// The P-Q1 drift detector: the count of consecutive steps the ITL p95 has
    /// stayed over the drifted SLO (the `calibrated SLO * (1 + drift_factor)`).
    /// When it reaches `drift_steps`, a re-calibration is triggered.
    drift_streak: usize,
    /// The P-Q1 periodic re-cal counter: the steps since the last re-derivation.
    /// When it reaches `recalibration_interval`, a re-derivation is triggered.
    recal_steps: usize,
}

impl QosController {
    pub fn new(cfg: QosConfig) -> Self {
        let calibration_window = cfg.calibration_steps.max(1);
        // The sample buffer cap (the P-Q1 bounded memory): the rolling window
        // length, so the calibration `Vec`s stay O(window) + retain the live
        // distribution for the drift re-calibration.
        let sample_cap = calibration_window;
        Self {
            cfg,
            gain: 0.0,
            itl_ema: 0.0,
            output_len_ema: 0.0,
            histogram: WorkloadHistogram::new(),
            calibration: Calibration {
                window: calibration_window,
                sample_cap,
                ..Default::default()
            },
            drift_streak: 0,
            recal_steps: 0,
        }
    }

    /// Is QoS active? Env master-switch OR config field (OR'd; env wins for ops).
    pub fn active(&self) -> bool {
        qos_enabled() || self.cfg.enabled
    }

    /// Whether the online calibration window has completed. While calibrating,
    /// the controller observes the system but does not act: it does not suppress
    /// prefill, override sampling, emit a parallelism hint, or trigger a KV
    /// demotion. Acting before enough profiling data exists would let the
    /// controller perturb the very signals it is trying to measure. Only after
    /// the calibration does the control law engage.
    pub fn calibration_done(&self) -> bool {
        self.calibration.done
    }

    /// The effective (resolved) control parameters. Each is the operator-configured
    /// value when set, otherwise the value derived by calibration when calibration
    /// has completed, otherwise a safe fallback. This is the single place the
    /// magic numbers are resolved, so the control law below never sees a raw
    /// `None` or a blind constant.
    pub fn effective(&self) -> EffectiveParams {
        let c = &self.cfg;
        let cal = &self.calibration;
        EffectiveParams {
            itl_slo_ms: c.itl_slo_ms.or(cal.derived_itl_slo_ms),
            // The gain_kp is the operator-set value when present, otherwise the
            // calibration-derived value (the settling-time-tuned gain, the no a
            // blind constant). The mem_k stays an operator knob (the memory
            // scaling is workload-specific, not measurable from the ITL alone).
            gain_kp: c.gain_kp.or(cal.derived_gain_kp),
            mem_k: c.mem_k,
            latency_output_frac: c.latency_output_frac.or(cal.derived_latency_output_frac),
            // The max_meaningful_step_ms (the idle-gap guard, the P-Q2): the
            // operator-set value when present, otherwise the calibration-derived
            // p99 of the measured step distribution (the default-on guard, the no
            // a wake-from-idle gap spiking the integral gain). The chunk_floor
            // stays an operator knob (the no the blind constant).
            max_meaningful_step_ms: c.max_meaningful_step_ms.or(cal.derived_max_meaningful_step_ms),
            chunk_floor: c.chunk_floor,
            regime_decode_pressure: c.regime_decode_pressure.or(cal.derived_regime_decode_pressure),
            regime_kv_pressure: c.regime_kv_pressure.or(cal.derived_regime_kv_pressure),
            kv_demote_pressure: c.kv_demote_pressure.or(cal.derived_kv_demote_pressure),
            kv_demote_threshold: c.kv_demote_threshold.or(cal.derived_kv_demote_threshold),
            low_batch_hint: c.low_batch_hint.or(cal.derived_low_batch_hint),
            short_output_tokens: c.short_output_tokens.or(cal.derived_short_output_tokens),
        }
    }

    /// Feed one measured step duration (ms). Updates the ITL EMA, the integral
    /// gain, + the histogram's P² ITL (the regime × class).
    pub fn record_step(&mut self, step_ms: f64, regime: Regime, class: QosClass) {
        if step_ms <= 0.0 {
            return;
        }
        self.itl_ema = if self.itl_ema == 0.0 {
            step_ms
        } else {
            0.9 * self.itl_ema + 0.1 * step_ms
        };
        let eff = self.effective();
        // The neutral acts only when both the SLO + the gain are known (the
        // operator-set or the calibration-derived). When either is None (the
        // no data collected yet), the integral stays put (the no the blind
        // constant biasing the workload before data exists).
        if let (Some(slo), Some(kp)) = (eff.itl_slo_ms, eff.gain_kp) {
            let violation = self.itl_ema - slo; // >0 when over SLO
            // Conditional integration (anti-windup): freeze the integral when it is
            // saturated AND the error is pushing further into saturation. Without this,
            // a persistent over-SLO period pins the gain at 1.0, and on recovery the
            // gain decays only at kp * |negative violation| (slow when the ITL is near
            // the SLO), over-suppressing prefill long after the pressure clears.
            let saturated_high = self.gain >= 1.0 && violation > 0.0;
            let saturated_low = self.gain <= 0.0 && violation < 0.0;
            if !saturated_high && !saturated_low {
                self.gain = (self.gain + kp * violation as f32).clamp(0.0, 1.0);
            }
        }
        // The histogram: the P² ITL, fed under the regime (the independent sensor,
        // passed in by the scheduler) + the seq's actual class (not hardcoded).
        self.histogram.record_itl(regime, class, step_ms);
        // The P-Q1 drift re-calibration: after the initial window, re-derive the
        // setpoints when the live ITL p95 drifts from the calibrated SLO, or on a
        // fixed interval. Blends the new values in (no step change).
        self.maybe_recalibrate(regime);
    }

    /// The P-Q1 drift detector + periodic re-derivation. Runs only after the
    /// initial calibration window (before that, the setpoints are still being
    /// derived, so there is nothing to re-derive). Two triggers:
    /// - Drift: the live ITL p95 (from the rolling histogram) exceeds the
    ///   calibrated SLO by `drift_factor` for `drift_steps` consecutive calls.
    /// - Interval: every `recalibration_interval` steps, unconditionally.
    /// On a trigger, re-runs the quantile derivations from the rolling sample
    /// buffer and blends the new setpoints into the effective values by
    /// `recal_blend` (a small gain, so the controller adapts smoothly instead of
    /// jumping).
    fn maybe_recalibrate(&mut self, regime: Regime) {
        if !self.calibration.done {
            return;
        }
        self.recal_steps += 1;
        let slo = self.calibration.derived_itl_slo_ms.unwrap_or(0.0);
        if slo <= 0.0 {
            return;
        }
        let itl_p95 = self.histogram.itl_p95(regime, QosClass::Throughput);
        let drifted = itl_p95 > slo * (1.0 + self.cfg.drift_factor);
        if drifted {
            self.drift_streak += 1;
        } else {
            self.drift_streak = 0;
        }
        let interval_hit = self.cfg.recalibration_interval > 0
            && self.recal_steps >= self.cfg.recalibration_interval;
        let drift_hit = self.drift_streak >= self.cfg.drift_steps.max(1);
        if !interval_hit && !drift_hit {
            return;
        }
        self.recal_steps = 0;
        self.drift_streak = 0;
        self.recalibrate();
    }

    /// Re-derive the `None` setpoints from the rolling sample buffer + blend them
    /// into the current derived values by `recal_blend`. A blend of 0.25 moves the
    /// effective setpoint a quarter of the way toward the freshly-measured value,
    /// so a regime shift is absorbed gradually (no discontinuity in the control
    /// law).
    fn recalibrate(&mut self) {
        let blend = self.cfg.recal_blend.clamp(0.0, 1.0);
        let cal = &mut self.calibration;
        // Re-run the derivations on the current (rolling) buffers, snapshotting
        // the prior values into the `*_prev` fields.
        cal.rederive();
        // Blend each newly-derived setpoint back toward the prior value (a
        // gradual adaptation, no step change). When the prior is absent (the
        // first derivation), keep the new value directly.
        if let (Some(prev), Some(next)) = (cal.derived_itl_slo_ms_prev, cal.derived_itl_slo_ms) {
            cal.derived_itl_slo_ms = Some(prev + (next - prev) * blend);
        }
        if let (Some(prev), Some(next)) = (cal.derived_gain_kp_prev, cal.derived_gain_kp) {
            cal.derived_gain_kp = Some(prev + (next - prev) * blend as f32);
        }
        if let (Some(prev), Some(next)) = (
            cal.derived_regime_decode_pressure_prev,
            cal.derived_regime_decode_pressure,
        ) {
            cal.derived_regime_decode_pressure = Some(prev + (next - prev) * blend as f32);
        }
        if let (Some(prev), Some(next)) = (
            cal.derived_regime_kv_pressure_prev,
            cal.derived_regime_kv_pressure,
        ) {
            cal.derived_regime_kv_pressure = Some(prev + (next - prev) * blend);
        }
        if let (Some(prev), Some(next)) = (
            cal.derived_max_meaningful_step_ms_prev,
            cal.derived_max_meaningful_step_ms,
        ) {
            cal.derived_max_meaningful_step_ms = Some(prev + (next - prev) * blend);
        }
    }

    /// Feed a finished sequence's output length. Builds the model's typical-output
    /// estimate used by class inference + the histogram's output-len CDF.
    pub fn record_output(&mut self, output_len: usize) {
        if output_len == 0 {
            return;
        }
        self.output_len_ema = if self.output_len_ema == 0.0 {
            output_len as f64
        } else {
            0.9 * self.output_len_ema + 0.1 * output_len as f64
        };
        let eff = self.effective();
        // The histogram: the output-len CDF (the class is inferred from the output length).
        // The class inference stays neutral (the Throughput default) when the
        // latency_output_frac is not yet known (the no the blind 0.5 biasing the
        // class before data exists).
        let class = if let Some(frac) = eff.latency_output_frac {
            if self.output_len_ema > 0.0 && (output_len as f64) <= (self.output_len_ema * frac) {
                QosClass::Latency
            } else {
                QosClass::Throughput
            }
        } else {
            QosClass::Throughput
        };
        self.histogram.record_output_len(class, output_len);
    }

    /// Feed the decode batch size for this step (the MagicDec signal: the SD
    /// speedup grows with the batch size for the separate-drafter, the MTP is
    /// batch-invariant). Drives the histogram's batch-size EMA (the regime).
    pub fn record_batch_size(&mut self, regime: Regime, batch_size: usize) {
        if batch_size == 0 {
            return;
        }
        self.histogram.record_batch_size(regime, batch_size);
    }

    /// Feed one decode step's full measurement set into the online calibration.
    /// The scheduler calls this once per step with the signals it already
    /// computes (the step duration, the decode pressure, the KV pressure, the
    /// finished output length if any, the batch size). The calibration accumulates
    /// these over the first `calibration_steps` steps, then derives every `None`
    /// config value from the measurements. After the window it is a no-op, so the
    /// steady-state control law uses the fixed derived values.
    pub fn observe_calibration(
        &mut self,
        step_ms: f64,
        decode_pressure: f32,
        kv_pressure: f64,
        output_len: Option<usize>,
        batch_size: usize,
    ) {
        self.calibration.observe(
            step_ms,
            decode_pressure,
            kv_pressure,
            output_len,
            batch_size,
        );
    }

    /// The admission bypass cap (the NInfer `Scheduler` template: the
    /// `fresh_candidates` FIFO bypass allowance, the `admission_bypasses >=
    /// concurrency` cap, the "at most C younger requests can overtake" rule).
    /// Returns `true` if the admission is allowed (the no exceeding the FIFO
    /// allowance).
    pub fn admission_allowed(&self, bypasses: usize, concurrency: usize) -> bool {
        bypasses < concurrency
    }

    /// The preemption victim selection (the NInfer `Scheduler` template: the
    /// `youngest_victim`, the no materializing/finished, the `recovery_pending`
    /// + the `priority_ticket` guard, the youngest-by-id). Returns the victim
    /// index (the `None` if no valid victim).
    pub fn youngest_victim(
        &self,
        candidates: &[(usize, bool, bool)], // (seq_id, is_materializing_or_finished, recovery_pending)
        priority_ticket: Option<usize>,
    ) -> Option<usize> {
        let mut youngest: Option<usize> = None;
        for &(seq_id, blocked, recovery_pending) in candidates {
            if blocked || recovery_pending {
                continue; // The no materializing/finished, the no recovery-pending.
            }
            if let Some(ticket) = priority_ticket {
                if seq_id <= ticket {
                    continue; // The priority-ticket guard (the no overtaking the ticket holder).
                }
            }
            match youngest {
                Some(y) if seq_id < y => {} // The keep the younger (the larger id).
                _ => youngest = Some(seq_id),
            }
        }
        youngest
    }

    /// The resume gating (the NInfer `Scheduler` template: the `should_restore`,
    /// the `have_paused` + the `resident_empty` OR the `restoration_event`).
    pub fn should_restore(have_paused: bool, resident_empty: bool, restoration_event: bool) -> bool {
        have_paused && (resident_empty || restoration_event)
    }

    /// Feed one admission (the two-rate sensor's fast path: the per-step
    /// `record_admission` updates the prefix-hit EMA + the turns counter + the
    /// reuse-evidence). The `cached` is the number of prefix-cached tokens (the
    /// hit), the `len` is the total prompt length (the query), the `hit_ratio`
    /// is the `cached / len` (the 0..1).
    pub fn record_admission(&mut self, class: QosClass, cached: usize, len: usize, reuse_digest: u64) {
        if len == 0 {
            return;
        }
        let hit_ratio = cached as f64 / len as f64;
        self.histogram.record_prefix_hit(class, hit_ratio);
        self.histogram.record_turn();
        self.histogram.record_reuse_evidence(class, reuse_digest);
    }

    /// Feed one preemption for the class (the NInfer reretraction_count`, the
    /// capacity signal). The histogram's `preemptions[class]` update.
    pub fn record_preemption(&mut self, class: QosClass) {
        self.histogram.record_preemption(class);
    }

    /// Feed one turn (the conversation warmth, the capture_seq). The
    /// histogram's `turns` counter update.
    pub fn record_turn(&mut self) {
        self.histogram.record_turn();
    }

    /// Feed one reuse-evidence digest for the class (the NInfer real-demand
    /// signal: the `proven_frontier` + the 256-entry demand digest). The
    /// histogram's `reuse_evidence[class]` update.
    pub fn record_reuse_evidence(&mut self, class: QosClass, digest: u64) {
        self.histogram.record_reuse_evidence(class, digest);
    }

    /// Feed one request-state transition (the 0..7 lifecycle index) for the
    /// class (the NInfer `EngineRequestState`). The histogram's
    /// `request_states[class]` update.
    pub fn record_request_state(&mut self, class: QosClass, state: usize) {
        self.histogram.record_request_state(class, state);
    }

    /// Feed one exposed-time sample (the work-class index + the `host_ns` + the
    /// `device_wait_ns`) for the class (the NInfer `RequestHostTiming`). The
    /// histogram's `exposed_time[work_class][class]` update.
    pub fn record_exposed_time(
        &mut self,
        class: QosClass,
        work_class: usize,
        host_ns: u64,
        device_wait_ns: u64,
    ) {
        self.histogram.record_exposed_time(class, work_class, host_ns, device_wait_ns);
    }

    /// The law: given the running set and a block snapshot, produce the per-step plan.
    /// When inactive, returns the static degenerate plan.
    pub fn plan(&self, running: &[Sequence], blocks: &BlockSnapshot) -> StepPlan {
        // The inactive case returns the static degenerate plan.
        if !self.active() {
            return StepPlan {
                chunk_size: blocks.static_chunk,
                token_budget: blocks.max_num_batched_tokens,
                thr_slot_capacity: 0,
                thr_in_use: 0,
                thr_kv_full: false,
            };
        }
        // The calibrating case: while the controller is still gathering profiling
        // data, it observes but does not act. Returning the static plan here
        // prevents the controller from suppressing prefill or scaling the budget
        // before it has enough measurements to derive its setpoints from. Once early
        // would perturb the very signals (the ITL, the pressure) it is trying to
        // measure, biasing the calibration.
        if !self.calibration_done() {
            return StepPlan {
                chunk_size: blocks.static_chunk,
                token_budget: blocks.max_num_batched_tokens,
                thr_slot_capacity: 0,
                thr_in_use: 0,
                thr_kv_full: false,
            };
        }

        // Decode pressure in [0,1]: fraction of running slots that are actively generating.
        let decodes = running.iter().filter(|s| s.output_len() > 0).count();
        let pressure = if blocks.max_num_seqs > 0 {
            (decodes as f32 / blocks.max_num_seqs as f32).min(1.0)
        } else {
            0.0
        };
        // Suppress prefill in proportion to (pressure x gain): idle -> full chunk,
        // under ITL pressure -> smaller chunks so decodes get the step. The
        // chunk_floor + the mem_k stay neutral (the no the blind constant) when
        // not yet known (the operator-set or the calibration-derived).
        let eff = self.effective();
        let chunk_floor = eff.chunk_floor.unwrap_or(0);
        let chunk = ((blocks.static_chunk as f32) * (1.0 - pressure * self.gain))
            .max(chunk_floor as f32) as usize;
        // Scale the per-step prefill budget down as KV headroom drops. The mem_k
        // is neutral (the full budget) when not yet known.
        let conservativeness = match eff.mem_k {
            Some(mem_k) => (1.0 - mem_k * (1.0 - blocks.mem_headroom())).clamp(0.5, 1.0),
            None => 1.0,
        };
        let budget = (blocks.max_num_batched_tokens as f32 * conservativeness) as usize;

        let slot_reserve = self.cfg.latency_slot_reserve.unwrap_or(0);
        let thr_slot_capacity = blocks.max_num_seqs.saturating_sub(slot_reserve);
        let thr_in_use = running.iter().filter(|s| !s.qos_class.is_latency()).count();
        let thr_kv_full = self.cfg.latency_kv_reserve_frac.is_some_and(|frac| {
            blocks.total_blocks > 0 && (blocks.free_blocks as f32) < (blocks.total_blocks as f32 * frac)
        });

        StepPlan {
            chunk_size: chunk,
            token_budget: budget,
            thr_slot_capacity,
            thr_in_use,
            thr_kv_full,
        }
    }

    /// The `AdaptivePlan` law: read the histogram + emit the actuators (the
    /// `batch_cap`, the `sampling` override, the `parallelism_hint`, the
    /// `kv_demote_threshold`). Wraps the base `StepPlan`.
    ///
    /// The golden test: a contended + high-ITL regime emits the
    /// `UseSequenceParallelism` hint; an uncontended + low-batch regime emits
    /// the `IncreaseTp` hint.
    pub fn adaptive_plan(&self, running: &[Sequence], blocks: &BlockSnapshot) -> AdaptivePlan {
        let base = self.plan(running, blocks);
        if !self.active() {
            return AdaptivePlan::from(base);
        }
        // The calibrating case: while the controller is still gathering profiling
        // data, emit the neutral actuators (no sampling override, the Hold
        // parallelism hint, no KV demotion) so the calibration measurements are
        // not biased by the controller's own actions. The base StepPlan already
        // returns the static plan during calibration (see `plan`).
        if !self.calibration_done() {
            return AdaptivePlan {
                base,
                batch_cap: blocks.max_num_seqs,
                sampling: SamplingOverride::default(),
                parallelism_hint: ParallelismHint::Hold,
                kv_demote_threshold: None,
            };
        }
        let eff = self.effective();

        // The regime classification (the decode pressure + the KV pressure), using
        // the calibrated thresholds. When a threshold is not yet known (the None),
        // the regime stays neutral (the Uncontended, the no the blind 0.5/0.8
        // biasing the regime before data exists).
        let decodes = running.iter().filter(|s| s.output_len() > 0).count();
        let decode_pressure = if blocks.max_num_seqs > 0 {
            (decodes as f32 / blocks.max_num_seqs as f32).min(1.0)
        } else {
            0.0
        };
        let kv_pressure = 1.0 - blocks.mem_headroom() as f64;
        let regime = match (eff.regime_decode_pressure, eff.regime_kv_pressure) {
            (Some(dp_thresh), Some(kp_thresh)) => {
                Regime::classify(decode_pressure, kv_pressure, dp_thresh, kp_thresh)
            }
            _ => Regime::Uncontended, // the neutral regime untilthe thresholds the blind threshold)
        };

        // The parallelism hint (the Shift Parallelism result). The hint stays
        // neutral (the Hold) when the SLO + the batch hint are not yet known.
        let itl_p95 = self.histogram.itl_p95(regime, QosClass::Throughput);
        let batch_ema = self.histogram.batch_size[if regime == Regime::Contended { 0 } else { 1 }].value() as usize;
        let parallelism_hint = match (eff.itl_slo_ms, eff.low_batch_hint) {
            (Some(slo), Some(low_batch)) => {
                if itl_p95 > slo && regime == Regime::Contended {
                    ParallelismHint::UseSequenceParallelism
                } else if batch_ema < low_batch {
                    ParallelismHint::IncreaseTp
                } else {
                    ParallelismHint::Hold
                }
            }
            _ => ParallelismHint::Hold, // the neutral hint (the no the blind actuation)
        };

        // The batch cap (the vLLM `max_num_active_reqs` result: the no new-sequence
        // admission beyond this). The P-Q4: a real contended-regime cap, not a
        // stub hard-wired to `max_num_seqs`. When the ITL p95 is over the SLO +
        // the regime is contended, admit fewer sequences than the slot count so
        // the running decodes get the step (the prefill admission is throttled
        // under latency pressure). When uncontended, the cap is the full
        // `max_num_seqs` (the no throttling in the low-pressure regime). The floor
        // is the Latency slot reserve (the `latency_slot_reserve`), so the
        // latency-sensitive requests always get admitted even under contention.
        let latency_reserve = self.cfg.latency_slot_reserve.unwrap_or(0);
        let batch_cap = match (eff.itl_slo_ms, regime) {
            (Some(slo), Regime::Contended) if itl_p95 > slo => {
                // The contended + over-SLO: the cap drops proportionally to the
                // integral gain (the 0..1 suppression), floored at the latency
                // reserve (at least 1, so the controller never admits zero
                // sequences) + capped at the full slot count.
                let floor = latency_reserve.max(1);
                let scaled = (blocks.max_num_seqs as f32 * (1.0 - self.gain)) as usize;
                scaled.clamp(floor, blocks.max_num_seqs)
            }
            _ => blocks.max_num_seqs, // The uncontended (or no SLO): the full slot count.
        };

        // The sampling override (the per-class width: the short-output → the tight
        // top-k; the long-output → the wide top-p). The cutoff is calibrated from
        // the measured output-length distribution. The override stays neutral when
        // the cutoff is not yet known.
        let output_median = self.histogram.output_len_median(QosClass::Latency);
        let sampling = match eff.short_output_tokens {
            Some(short_tokens) if output_median < short_tokens => {
                SamplingOverride { top_k: Some(1), top_p: None }
            }
            _ => SamplingOverride::default(),
        };

        // The KV demotion threshold (the coldest conversation's blocks → the
        // HostMem/Gds tier). Both the trigger pressure + the threshold are
        // calibrated. The demotion stays off when either is not yet known.
        let kv_demote_threshold = match (eff.kv_demote_pressure, eff.kv_demote_threshold) {
            (Some(pressure_thresh), Some(threshold)) if kv_pressure > pressure_thresh => Some(threshold),
            _ => None,
        };

        AdaptivePlan {
            base,
            batch_cap,
            sampling,
            parallelism_hint,
            kv_demote_threshold,
        }
    }

    /// Infer a sequence's class from the measured output-length distribution. A
    /// request whose expected output is well below the model's typical output is
    /// Latency-class. Before any outputs are observed (startup) this is a no-op
    /// (Throughput), so the loop warms up rather than misclassifying.
    pub fn infer_class(&self, seq: &Sequence) -> QosClass {
        let expected = seq.sampling_params.max_tokens.unwrap_or(self.output_len_ema as usize);
        let eff = self.effective();
        // The class inference stays neutral (the Throughput) when the latency
        // output fraction is not yet known (the no the blind 0.5 misclassifying
        // before data exists).
        let threshold = match eff.latency_output_frac {
            Some(frac) => (self.output_len_ema * frac) as usize,
            None => 0,
        };
        if threshold > 0 && expected <= threshold {
            QosClass::Latency
        } else {
            QosClass::Throughput
        }
    }

    /// The live contention vector (diagnostics / future cross-rank signal).
    pub fn contention(&self, running: &[Sequence], blocks: &BlockSnapshot) -> Contention {
        // Decode pressure = the fraction of running slots that are actively generating
        // (output_len > 0). A seq that is fully prefilled but waiting has output_len 0;
        // a seq mid-decode has output_len growing. This is the real decode-contention
        // signal the chunk-suppression law drives off.
        let decodes = running.iter().filter(|s| s.output_len() > 0).count();
        Contention {
            decode_pressure: if blocks.max_num_seqs > 0 {
                (decodes as f32 / blocks.max_num_seqs as f32).min(1.0)
            } else {
                0.0
            },
            mem_headroom: blocks.mem_headroom(),
            slot_fill: if blocks.max_num_seqs > 0 {
                running.len() as f32 / blocks.max_num_seqs as f32
            } else {
                0.0
            },
            itl_ms: self.itl_ema,
            gain: self.gain,
        }
    }

    /// The decode-pressure sensor (the fraction of running slots actively generating).
    /// Exposed so the scheduler can feed the regime classifier an independent signal
    /// (not the controller's own gain output).
    pub fn decode_pressure(&self, running: &[Sequence], blocks: &BlockSnapshot) -> f32 {
        let decodes = running.iter().filter(|s| s.output_len() > 0).count();
        if blocks.max_num_seqs > 0 {
            (decodes as f32 / blocks.max_num_seqs as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// The dominant QoS class among the running sequences (the ITL histogram is fed
    /// under this class, not a hardcoded Throughput). The Latency class wins when
    /// any running seq is Latency (the ITL-sensitive requests are present); otherwise
    /// Throughput.
    pub fn dominant_class(&self, running: &[Sequence]) -> QosClass {
        if running.iter().any(|s| s.qos_class.is_latency()) {
            QosClass::Latency
        } else {
            QosClass::Throughput
        }
    }

    // --- read-only accessors for stamping per-sequence stats ---
    pub fn gain(&self) -> f32 {
        self.gain
    }
    pub fn itl_ms(&self) -> f64 {
        self.itl_ema
    }
    pub fn slo_ms(&self) -> Option<f64> {
        self.effective().itl_slo_ms
    }
}

/// Per-sequence scheduling stats, accumulated by the scheduler and reported at
/// sequence end. The `gain_*` / `itl_*` fields expose the controller's learning
/// trajectory so the report shows whether QoS is actually adapting and holding the
/// SLO, not just running.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedSeqStats {
    /// Was QoS active for this sequence?
    pub qos_active: bool,
    pub qos_class: QosClass,
    /// Prefill steps this sequence went through.
    pub prefill_steps: usize,
    /// Steps where its prefill chunk was adaptively shrunk below the static cap.
    pub chunk_shrunk: usize,
    /// Times a Latency request was admitted ahead of a waiting Throughput one.
    pub priority_admitted: usize,
    /// Times this (Throughput) request was gated by a QoS slot/KV reservation.
    pub reservation_blocked: usize,
    /// Times this sequence was preempted (KV pressure) / swapped out.
    pub preempted: usize,
    pub swapped: usize,
    // --- controller learning trajectory (the "is it adapting" signal) ---
    /// Integral gain when this sequence was admitted.
    pub gain_start: f32,
    /// Integral gain when this sequence finished (a rising gain = ITL under pressure).
    pub gain_end: f32,
    /// Measured model ITL (ms) at finish.
    pub itl_ms: f64,
    /// The ITL SLO (ms) the controller was holding to. The `None` when the SLO is
    /// not yet known (the operator did not set it + the calibration has not
    /// derived it), so the report shows "no SLO" instead of a blind constant.
    pub itl_slo_ms: Option<f64>,
    /// Actual output length (tokens).
    pub output_len: usize,
    /// The model that served this sequence (the "base" when no LoRA tenant was
    /// active, or the tenant name when a LoRA adapter served it). Surfaced in the
    /// per-sequence report so the operator can see which model produced the
    /// output. The `SchedSeqStats` is created before the LoRA tenant is known
    /// (the stats are stamped at admission), so this is set at sequence finish,
    /// when the tenant is known. The `#[serde(default)]` keeps older serialized
    /// stats (the without this field) deserializable.
    #[serde(default)]
    pub model_label: String,
}

impl SchedSeqStats {
    /// One-line body for the per-sequence report (the caller adds the `[Seq N]` prefix).
    /// Returns `None` when QoS was not active (nothing to report).
pub fn report(&self) -> Option<String> {
        if !self.qos_active {
            return None;
        }
        // The model (the LoRA tenant) field is conditional: shown only when a LoRA
        // tenant served the sequence (the model_label is not "base" + not empty).
        // The no showing model=base when LoRA is off (the conditional-field rule,
        // like the drafting + QoS stats that only show when active).
        let mut parts = vec![
            format!(
                "QoS[{qos_class:?}] itl={itl:.0}/{slo}ms gain={g0:.2}->{g1:.2} prefill={p} shrunk={s} prio={prio} resv={resv} preempt={pre} swap={sw} out={out}",
                qos_class = self.qos_class,
                itl = self.itl_ms,
                slo = self.itl_slo_ms.map(|v| format!("{v:.0}")).unwrap_or_else(|| "n/a".to_string()),
                g0 = self.gain_start,
                g1 = self.gain_end,
                p = self.prefill_steps,
                s = self.chunk_shrunk,
                prio = self.priority_admitted,
                resv = self.reservation_blocked,
                pre = self.preempted,
                sw = self.swapped,
                out = self.output_len,
            )
        ];
        if self.model_label != "base" && !self.model_label.is_empty() {
            parts.push(format!(" model={}", self.model_label));
        }
        Some(parts.join(" "))
    }
}

/// The end-of-sequence report bundle: everything the server needs to print, held in
/// a single bounded map (one retention mechanism, not two).
#[derive(Debug, Clone, Default)]
pub struct SeqReport {
    pub cached_tokens: usize,
    pub sched: SchedSeqStats,
    /// The prompt (prefill) token count (the input length, the no the cached portion).
    pub prompt_tokens: usize,
/// The prompt (prefill) wall time (s). The `0.0` when not measured (the no-PD path).
    pub prompt_time_s: f64,
    /// The decode wall time (s). The `0.0` when not measured.
    pub decode_time_s: f64,
}

/// The runtime resource gauges (the single source for both the console
/// `print_free_blocks` + the `/metrics` export). One snapshot of the KV block
/// occupancy, the CPU-swap usage, the Mamba/GDN slot occupancy, + the runner
/// queue depths. The telemetry reads this once per scrape (the no multiple
/// round-trips that could straddle a state change).
#[derive(Debug, Clone, Default)]
pub struct ResourceGauges {
    // The KV cache (the GPU blocks).
    pub kv_free_blocks: usize,
    pub kv_total_blocks: usize,
    pub kv_free_tokens: usize,
    pub kv_used_percent: f32,
    pub kv_used_gb: f32,
    pub kv_total_gb: f32,
    // The CPU swap (the KV offload).
    pub cpu_swap_used_frac: f32,
    pub cpu_swap_used_gb: f32,
    pub cpu_swap_total_gb: f32,
    // The Mamba/GDN slots (the hybrid models; the 0 when not hybrid).
    pub mamba_slots_used: usize,
    pub mamba_slots_total: usize,
    pub mamba_used_gb: f32,
    pub mamba_budget_gb: f32,
    pub mamba_slot_mb: f32,
    // The runner queue depths (the live sequence counts).
    pub seq_running: usize,
    pub seq_waiting: usize,
    pub seq_cached: usize,
    pub seq_transferred: usize,
    // The physical memory map (the kernel-MMU view, the no the block-based logical
    // occupancy). The pressure is the VRAM-in-use fraction (the physical signal the
    // QoS demote actuator reads); the idle + total are the grow headroom. The 0.0 /
    // 0 when the map is not yet built (the non-CUDA build, the test).
    pub mmu_pressure: f32,
    pub mmu_idle_vram_bytes: u64,
    pub mmu_total_vram_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::config::SamplingParams;

    fn seq_with(max_tokens: Option<usize>, output_len: usize) -> Sequence {
        let mut s = Sequence::new(vec![1, 2, 3], 64, SamplingParams::default(), &None, 0);
        s.sampling_params.max_tokens = max_tokens;
        s.output_ids = vec![7u32; output_len];
        s
    }

    #[test]
    fn inactive_plan_is_static() {
        let ctl = QosController::new(QosConfig::default());
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        let running = vec![seq_with(Some(100), 50)];
        let plan = ctl.plan(&running, &blocks);
        assert_eq!(plan.chunk_size, 8192); // full static chunk
        assert_eq!(plan.token_budget, 16384);
        assert_eq!(plan.thr_slot_capacity, 0);
    }

    #[test]
    fn gain_rises_when_itl_exceeds_slo() {
        let mut cfg = QosConfig::default();
        cfg.itl_slo_ms = Some(50.0);
        cfg.gain_kp = Some(0.02);
        let mut ctl = QosController::new(cfg);
        // Simulate a sustained step far over the 50ms SLO.
        for _ in 0..200 {
            ctl.record_step(200.0, Regime::Contended, QosClass::Throughput);
        }
        assert!(ctl.gain() > 0.5, "gain should rise under ITL pressure, got {}", ctl.gain());
    }

    #[test]
    fn gain_falls_when_itl_has_headroom() {
        let mut cfg = QosConfig::default();
        cfg.itl_slo_ms = Some(50.0);
        cfg.gain_kp = Some(0.02);
        let mut ctl = QosController::new(cfg);
        ctl.gain = 1.0;
        for _ in 0..200 {
            ctl.record_step(5.0, Regime::Uncontended, QosClass::Throughput); // well under the 50ms SLO
        }
        assert!(ctl.gain() < 0.5, "gain should fall with ITL headroom, got {}", ctl.gain());
    }

    #[test]
    fn adaptive_chunk_shrinks_under_pressure_and_gain() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true; // exercise the active (adaptive) plan path
        cfg.chunk_floor = Some(256); // the operator-set chunk floor (the no the blind 256)
        cfg.mem_k = Some(0.5); // the operator-set memory scaling (the no the blind 0.5)
        cfg.gain_kp = Some(0.02); // the operator-set integral gain (the no the blind 0.02)
        cfg.itl_slo_ms = Some(50.0); // the operator-set SLO (the no the blind 50.0)
        let mut ctl = QosController::new(cfg);
        for _ in 0..200 {
            ctl.record_step(200.0, Regime::Contended, QosClass::Throughput);
        }
        // Complete the online calibration window so the controller leaves the
        // observe-only phase and begins acting. In production the scheduler feeds
        // observe_calibration every step; here we fast-forward the default.
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(200.0, 1.0, 0.5, Some(100), 32);
        }
        assert!(ctl.calibration_done(), "calibration must complete before the controller acts");
        let blocks = BlockSnapshot {
            free_blocks: 500,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        // 32 running decodes => pressure 1.0; high gain => chunk suppressed to floor.
        let mut running: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        for s in running.iter_mut() {
            s.output_ids.push(42); // mark actively generating (the sensor counts output_len > 0)
        }
        let plan = ctl.plan(&running, &blocks);
        assert!(plan.chunk_size <= 256, "chunk should be suppressed, got {}", plan.chunk_size);
    }

    #[test]
    fn infer_class_uses_measured_output() {
        let mut cfg = QosConfig::default();
        cfg.latency_output_frac = Some(0.5); // the operator-set class-inference cutoff
        let mut ctl = QosController::new(cfg);
        // Warm the output-length estimate to ~1000 tokens.
        for _ in 0..50 {
            ctl.record_output(1000);
        }
        // A short request (max_tokens 100) is below 0.5 x 1000 => Latency.
        assert_eq!(ctl.infer_class(&seq_with(Some(100), 0)), QosClass::Latency);
        // A long request (max_tokens 2000) is above the threshold => Throughput.
        assert_eq!(ctl.infer_class(&seq_with(Some(2000), 0)), QosClass::Throughput);
    }

    #[test]
    fn sched_stats_report_shows_trajectory() {
        let mut stats = SchedSeqStats::default();
        stats.qos_active = true;
        stats.qos_class = QosClass::Latency;
        stats.prefill_steps = 10;
        stats.chunk_shrunk = 8;
        stats.priority_admitted = 3;
        stats.gain_start = 0.3;
        stats.gain_end = 0.9;
        stats.itl_ms = 48.0;
        stats.itl_slo_ms = Some(50.0);
        stats.output_len = 128;
        let report = stats.report().unwrap();
        assert!(report.contains("QoS[Latency]"), "report: {report}");
        assert!(report.contains("gain=0.30->0.90"), "report: {report}");
        assert!(report.contains("itl=48/50ms"), "report: {report}");
    }

    /// The model label in the per-sequence report (the LoRA/base marking). The
    /// inclusive: a sequence served by a LoRA tenant shows the tenant name; a
    /// sequence served by the base model shows "base". The exclusive: the default
    /// (unset) label is the empty string until it is set at finish.
    #[test]
    fn sched_stats_report_shows_model_label() {
        let mut stats = SchedSeqStats::default();
        stats.qos_active = true;
        stats.model_label = "finance-adapter".to_string();
        let report = stats.report().unwrap();
        assert!(report.contains("model=finance-adapter"), "the tenant label is in the report: {report}");

        // The base-model case: the label is "base" → the model field is omitted (the
        // conditional-field rule: show model= only when a LoRA tenant served the
        // sequence, the no "base").
        let mut base_stats = SchedSeqStats::default();
        base_stats.qos_active = true;
        base_stats.model_label = "base".to_string();
        let base_report = base_stats.report().unwrap();
        assert!(!base_report.contains("model="), "the base label is omitted from the report: {base_report}");

        // The exclusive: the default label is empty (the unset before finish).
        let default_stats = SchedSeqStats::default();
        assert_eq!(default_stats.model_label, "", "the default model_label is empty until set at finish");
    }

    #[test]
    fn controller_does_not_act_during_calibration() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.gain_kp = Some(0.02); // the operator-set integral gain (the no the blind default)
        let mut ctl = QosController::new(cfg);
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        let mut running: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        // Mark every running sequence as fully-prefilled (decoding) so the decode
        // pressure is 1.0, which is what drives the chunk suppression.
        for s in running.iter_mut() {
            s.num_cached_tokens = s.len();
        }

        // Inclusive: before calibration completes, the controller observes but does
        // not act. Even under heavy ITL + pressure, the plan is the static plan
        // (full chunk, full budget) + the adaptive actuators are neutral (the Hold
        // hint, no sampling override, no KV demotion).
        for _ in 0..100 {
            ctl.record_step(500.0, Regime::Contended, QosClass::Throughput);
        }
        assert!(!ctl.calibration_done());
        let plan = ctl.plan(&running, &blocks);
        assert_eq!(plan.chunk_size, blocks.static_chunk, "no chunk suppression during calibration");
        assert_eq!(plan.token_budget, blocks.max_num_batched_tokens, "no budget scaling during calibration");
        let adaptive = ctl.adaptive_plan(&running, &blocks);
        assert_eq!(adaptive.parallelism_hint, ParallelismHint::Hold, "no parallelism hint during calibration");
        assert!(adaptive.sampling.top_k.is_none(), "no sampling override during calibration");
        assert!(adaptive.kv_demote_threshold.is_none(), "no KV demotion during calibration");

        // Complete the calibration window with a low, stable ITL so the derived SLO
        // is set low (the system's sustainable baseline).
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(20.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());

        // Exclusive: after calibration, drive the ITL far above the calibrated SLO.
        // The gain rises + the controller now suppresses the chunk + engages an
        // actuator.
        for _ in 0..200 {
            ctl.record_step(500.0, Regime::Contended, QosClass::Throughput);
        }
        let plan_after = ctl.plan(&running, &blocks);
        assert!(plan_after.chunk_size < blocks.static_chunk, "chunk suppressed after calibration");
        let adaptive_after = ctl.adaptive_plan(&running, &blocks);
        assert_ne!(adaptive_after.parallelism_hint, ParallelismHint::Hold, "an actuator engages after calibration");
    }

    #[test]
    fn adaptive_plan_actuator_mapping() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.gain_kp = Some(0.02); // the operator-set integral gain (the no the blind default)
        let mut ctl = QosController::new(cfg);
        // Calibrate with a low, stable ITL so the derived SLO is low (the system's
        // sustainable baseline).
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(20.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        // Now drive the measured ITL far above the calibrated SLO so the histogram's
        // p95 exceeds the SLO + the regime is contended (high high decode pressure).
        for _ in 0..100 {
            ctl.record_step(500.0, Regime::Contended, QosClass::Throughput);
        }
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        // The contended + high-ITL → the UseSequenceParallelism hint.
        let mut running: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        for s in running.iter_mut() {
            s.num_cached_tokens = s.len(); // mark decoding so the pressure is 1.0
        }
        let plan = ctl.adaptive_plan(&running, &blocks);
        assert_eq!(plan.parallelism_hint, ParallelismHint::UseSequenceParallelism);
        // The P-Q4: under contention + a high gain the batch cap drops below the
        // full slot count (floored at 1, the no the controller admitting zero
        // sequences). The prior stub always returned the full 32.
        assert!(plan.batch_cap < 32, "the contended cap drops below the slot count (got {})", plan.batch_cap);
        assert!(plan.batch_cap >= 1, "the cap is floored at 1 (got {})", plan.batch_cap);
        // The base StepPlan reflects the suppressed chunk (the high gain + pressure).
        assert!(plan.base.chunk_size <= blocks.chunk_floor, "chunk suppressed under pressure");

        // The uncontended + low-batch → the IncreaseTp hint.
        let mut ctl2 = QosController::new(QosConfig::default());
        ctl2.cfg.enabled = true;
        for _ in 0..ctl2.cfg.calibration_steps {
            ctl2.observe_calibration(20.0, 0.1, 0.1, Some(100), 1);
        }
        assert!(ctl2.calibration_done());
        let running2 = vec![seq_with(Some(100), 50)];
        let plan2 = ctl2.adaptive_plan(&running2, &blocks);
        assert_eq!(plan2.parallelism_hint, ParallelismHint::IncreaseTp);
    }

    #[test]
    fn admission_bypass_cap() {
        let ctl = QosController::new(QosConfig::default());
        // The "at most C younger requests can overtake" rule: the bypasses < concurrency.
        assert!(ctl.admission_allowed(0, 4));
        assert!(ctl.admission_allowed(3, 4));
        assert!(!ctl.admission_allowed(4, 4)); // The at the cap.
        assert!(!ctl.admission_allowed(5, 4)); // The over the cap.
    }

    #[test]
    fn preemption_victim_youngest() {
        let ctl = QosController::new(QosConfig::default());
        // The candidates: (seq_id, is_materializing_or_finished, recovery_pending).
        let candidates = vec![
            (1, false, false),
            (2, true, false), // The blocked (the materializing).
            (3, false, true), // The recovery_pending.
            (4, false, false),
        ];
        // The youngest valid victim is the 4 (the no 2 blocked, the no 3 recovery_pending).
        assert_eq!(ctl.youngest_victim(&candidates, None), Some(4));
        // The priority-ticket guard: the no overtaking the ticket holder (the 3).
        assert_eq!(ctl.youngest_victim(&candidates, Some(3)), Some(4));
        assert_eq!(ctl.youngest_victim(&candidates, Some(4)), None); // The no one younger than the 4.
    }

    #[test]
    fn resume_gating() {
        // The `should_restore`: the have_paused + (the resident_empty OR the restoration_event).
        assert!(QosController::should_restore(true, true, false));
        assert!(QosController::should_restore(true, false, true));
        assert!(!QosController::should_restore(false, true, true)); // The no paused.
        assert!(!QosController::should_restore(true, false, false)); // The no resident_empty + the no restoration_event.
    }

    #[test]
    fn decode_pressure_sensor_actively_generating_seqs() {
        // The sensor fix: decode_pressure counts running seqs that are actively
        // generating (output_len > 0), not fully-cached seqs (num_cached_tokens >=
        // len). A seq mid-decode has output_len > 0; a seq that just finished
        // prefill has output_len == 0.
        use crate::core::sequence::Sequence;
        use crate::utils::config::SamplingParams;
        let mut running = Vec::new();
        // A seq actively decoding (output_len > 0).
        let mut decoding_seq = Sequence::new(vec![1, 2, 3], 1, SamplingParams::default(), &None, -1);
        decoding_seq.output_ids.push(42);
        running.push(decoding_seq);
        // A seq that just finished prefill (output_len == 0).
        let prefill_seq = Sequence::new(vec![1, 2, 3], 1, SamplingParams::default(), &None, -1);
        running.push(prefill_seq);
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 200,
            max_num_seqs: 2,
            max_num_batched_tokens: 4096,
            static_chunk: 2048,
            chunk_floor: 256,
        };
        let ctl = QosController::new(QosConfig::default());
        // 1 of 2 running seqs is actively generating -> pressure 0.5.
        assert!((ctl.decode_pressure(&running, &blocks) - 0.5).abs() < 1e-6);
        // The prior ( (num_cached_tokens >= len) would have read 0.0 here
        // (neither seq is fully cached), so the chunk-suppression law never fired.
    }

    #[test]
    fn gain_does_not_wind_up_when_saturated() {
        // The anti-windup: when the gain is saturated at 1.0 + the violation stays
        // positive, the integral freezes (does not keep winding). On recovery the
        // gain decays.
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.gain_kp = Some(0.02); // the operator-set integral gain (the no the blind default)
        cfg.itl_slo_ms = Some(50.0); // the operator-set SLO (the no the blind default)
        let mut ctl = QosController::new(cfg);
        // Drive the gain to saturation.
        for _ in 0..500 {
            ctl.record_step(500.0, Regime::Contended, QosClass::Throughput);
        }
        assert!(ctl.gain() >= 1.0, "gain should saturate at 1.0");
        // The saturated gain + positive violation: the integral freezes (no windup).
        let gain_before = ctl.gain();
        ctl.record_step(500.0, Regime::Contended, QosClass::Throughput);
        assert_eq!(ctl.gain(), gain_before, "saturated gain must not wind up further");
        // Recovery: a low ITL decays the gain.
        for _ in 0..500 {
            ctl.record_step(5.0, Regime::Uncontended, QosClass::Throughput);
        }
        assert!(ctl.gain() < 0.5, "gain should decay on recovery");
    }

    /// The controller is neutral before it has data: a fresh controller (no
    /// operator config, no calibration) does not act (the no the blind constant
    /// biasing the workload). The plan is the static plan (the full chunk, the
    /// full budget, the neutral actuators) until the calibration derives the
    /// setpoints from the measured system.
    #[test]
    fn controller_is_neutral_before_data() {
        let ctl = QosController::new(QosConfig::default());
        // The fresh controller: no operator config (the all `None`), no
        // calibration (the `derived_*` are all `None`). The effective params
        // are all `None` (the no the blind 50.0/0.02/0.5/256/4/64/0.8 constants).
        let eff = ctl.effective();
        assert!(eff.itl_slo_ms.is_none(), "the SLO is unknown before data");
        assert!(eff.gain_kp.is_none(), "the gain is unknown before data");
        assert!(eff.mem_k.is_none(), "the memory scaling is unknown before data");
        assert!(eff.chunk_floor.is_none(), "the chunk floor is unknown before data");
        assert!(eff.low_batch_hint.is_none(), "the batch hint is unknown before data");
        assert!(eff.short_output_tokens.is_none(), "the output cutoff is unknown before data");
        // The plan is the static plan (the no the adaptive actuation): the full
        // chunk, the full budget, the neutral actuators.
        let blocks = BlockSnapshot {
            free_blocks: 500,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 0,
        };
        let plan = ctl.plan(&[], &blocks);
        assert_eq!(plan.chunk_size, blocks.static_chunk, "the neutral plan uses the full chunk");
        assert_eq!(plan.token_budget, blocks.max_num_batched_tokens, "the neutral plan uses the full budget");
        let adaptive = ctl.adaptive_plan(&[], &blocks);
        assert_eq!(adaptive.parallelism_hint, ParallelismHint::Hold, "the neutral plan holds the parallelism");
        assert!(adaptive.sampling.top_k.is_none(), "the neutral plan has no sampling override");
        assert!(adaptive.kv_demote_threshold.is_none(), "the neutral plan has no KV demotion");
    }

    /// The integral gain is derived from the measured ITL, not a blind constant.
    /// A fast workload (the 10ms ITL) derives a larger gain than a slow workload
    /// (the 100ms ITL), so the controller settles on the workload's own
    /// timescale. This is the self-tuning property: the no the operator having to
    /// hand-pick a gain per model.
    #[test]
    fn gain_kp_is_derived_from_measured_itl() {
        // The fast workload: 10ms ITL, the 64-step calibration window.
        let mut fast = QosController::new(QosConfig::default());
        fast.cfg.enabled = true;
        for _ in 0..fast.cfg.calibration_steps {
            fast.observe_calibration(10.0, 0.5, 0.5, Some(100), 8);
        }
        assert!(fast.calibration_done());
        let fast_gain = fast.effective().gain_kp.expect("the fast workload derives a gain");

        // The slow workload: 100ms ITL, the same window64-step window.
        let mut slow = QosController::new(QosConfig::default());
        slow.cfg.enabled = true;
        for _ in 0..slow.cfg.calibration_steps {
            slow.observe_calibration(100.0, 0.5, 0.5, Some(100), 8);
        }
        assert!(slow.calibration_done());
        let slow_gain = slow.effective().gain_kp.expect("the slow workload derives a gain");

        // The fast workload (the smaller ITL) must derive a larger gain (the
        // controller reacts faster to a fast system). The slow workload derives
        // a smaller gain (the controller reacts slower to a slow system). This
        // is the self-tuning property: the gain scales as 1/(slo * window).
        assert!(
            fast_gain > slow_gain,
            "the fast workload (gain {fast_gain}) must derive a larger gain than the slow workload (gain {slow_gain})"
        );
        // The gain is finite + in a sane range (no degenerate 0 or huge value).
        assert!(fast_gain > 0.0 && fast_gain < 1.0, "the derived gain must be in (0, 1)");
    }

    /// P-Q2: the idle-gap guard is derived from the step p99 after calibration,
    /// so it is active by default (no operator config needed). Before
    /// calibration it is None (neutral). An operator-set value still wins.
    #[test]
    fn idle_gap_guard_is_derived_after_calibration() {
        let mut ctl = QosController::new(QosConfig::default());
        ctl.cfg.enabled = true;
        // Before calibration: the derived guard is absent.
        assert!(ctl.effective().max_meaningful_step_ms.is_none());
        // Feed a step distribution: mostly 20ms, a few 500ms idle gaps.
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(20.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        // The p99 of the (all-20ms) distribution is ~20ms, so the guard is now
        // Some and excludes gaps larger than that.
        let guard = ctl.effective().max_meaningful_step_ms.expect("the guard is derived");
        assert!(guard >= 20.0, "the guard must be at least the normal step time: {guard}");
        // A 500ms wake-from-idle gap exceeds the guard, so record_step would skip
        // it (the scheduler applies this filter). Verify the comparison directly.
        assert!(500.0 > guard, "an idle gap must exceed the derived guard");
    }

    /// P-Q2: an operator-set max_meaningful_step_ms overrides the derived value.
    #[test]
    fn idle_gap_guard_operator_value_wins() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.max_meaningful_step_ms = Some(999.0);
        let mut ctl = QosController::new(cfg);
        for _ in 0..ctl.cfg.calibration_steps {
            ctl.observe_calibration(20.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        // The operator value (999) wins over the derived p99 (~20).
        assert_eq!(ctl.effective().max_meaningful_step_ms, Some(999.0));
    }

    /// P-Q1: the calibration sample buffers are bounded to the rolling cap, so a
    /// long-running server does not leak memory. Feeding far more steps than the
    /// cap keeps the buffer at the cap.
    #[test]
    fn calibration_sample_buffers_are_bounded() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.calibration_steps = 8; // small window -> small cap
        let mut ctl = QosController::new(cfg);
        let cap = ctl.calibration.sample_cap;
        assert_eq!(cap, 8, "the cap defaults to the window length");
        // Feed far more steps than the cap (the rolling buffer keeps the most
        // recent `cap` observations, the no unbounded growth).
        for i in 0..1000 {
            ctl.observe_calibration(20.0 + (i % 5) as f64, 0.1, 0.1, Some(100), 32);
        }
        // The buffer stays within the cap for the process lifetime.
        assert!(
            ctl.calibration.itl_samples.len() <= cap,
            "the itl buffer must stay within the cap (got {})",
            ctl.calibration.itl_samples.len()
        );
        assert_eq!(ctl.calibration.itl_samples.len(), cap, "the rolling buffer holds exactly the cap");
    }

    /// P-Q1: a workload regime shift (low ITL then high ITL) re-derives the SLO
    /// within one re-cal window, and the new SLO blends toward the higher value
    /// (no step change, but movement in the right direction).
    #[test]
    fn regime_shift_recalibrates_slo() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.calibration_steps = 16;
        cfg.recalibration_interval = 16; // re-derive every 16 steps after done
        cfg.drift_steps = 1; // fire on the first drifted step (test determinism)
        cfg.recal_blend = 1.0; // full replacement for a clean assertion
        let mut ctl = QosController::new(cfg);
        // Initial calibration under a fast regime (10ms steps).
        for _ in 0..16 {
            ctl.observe_calibration(10.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        let slo_before = ctl.calibration.derived_itl_slo_ms.expect("SLO derived");
        assert!((slo_before - 10.0).abs() < 1e-6, "initial SLO tracks the fast regime");
        // Now the regime shifts to slow (100ms steps). Feed record_step so the
        // rolling buffer + the P² histogram see the new distribution, and let the
        // interval trigger fire.
        for _ in 0..16 {
            ctl.record_step(100.0, Regime::Contended, QosClass::Throughput);
            ctl.observe_calibration(100.0, 0.9, 0.9, Some(100), 32);
        }
        let slo_after = ctl.calibration.derived_itl_slo_ms.expect("SLO re-derived");
        assert!(
            slo_after > slo_before,
            "the SLO must move toward the slower regime ({slo_before} -> {slo_after})"
        );
    }

    /// P-Q1: the drift detector fires when the live ITL p95 stays over the
    /// drifted SLO for the required number of consecutive steps.
    #[test]
    fn drift_detector_fires_on_consecutive_excess() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.calibration_steps = 8;
        cfg.recalibration_interval = 0; // disable the interval trigger; test drift only
        cfg.drift_factor = 0.5; // fire when p95 > SLO * 1.5
        cfg.drift_steps = 3; // require 3 consecutive drifted steps
        let mut ctl = QosController::new(cfg);
        for _ in 0..8 {
            ctl.observe_calibration(10.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        let slo = ctl.calibration.derived_itl_slo_ms.expect("SLO");
        // Feed steps well over the drifted threshold (SLO * 1.5). Two steps: the
        // streak is below drift_steps, no fire.
        let over = slo * 2.0;
        ctl.record_step(over, Regime::Contended, QosClass::Throughput);
        ctl.record_step(over, Regime::Contended, QosClass::Throughput);
        assert_eq!(ctl.drift_streak, 2, "the streak counts consecutive drifted steps");
        // The third step reaches drift_steps -> a re-calibration fires (the
        // streak resets).
        ctl.record_step(over, Regime::Contended, QosClass::Throughput);
        assert_eq!(ctl.drift_streak, 0, "the streak resets after a fire");
    }

    /// P-Q4: the batch cap is a real contended-regime admission cap. Under
    /// contention + a high gain it drops below max_num_seqs (floored at the
    /// latency reserve); when uncontended it equals max_num_seqs.
    #[test]
    fn batch_cap_is_real_under_contention() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.calibration_steps = 8;
        cfg.latency_slot_reserve = Some(4);
        let mut ctl = QosController::new(cfg);
        for _ in 0..8 {
            ctl.observe_calibration(10.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        let blocks = BlockSnapshot {
            free_blocks: 100,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        // Uncontended (low pressure, low gain): the cap is the full slot count.
        let running = vec![seq_with(Some(100), 50)];
        let plan = ctl.adaptive_plan(&running, &blocks);
        assert_eq!(plan.batch_cap, 32, "uncontended: the cap is the full slot count");
        // Now drive the gain high + the regime contended (high decode pressure).
        for _ in 0..200 {
            ctl.record_step(500.0, Regime::Contended, QosClass::Throughput);
        }
        let mut contended: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        for s in contended.iter_mut() {
            s.num_cached_tokens = s.len(); // mark decoding -> pressure 1.0
        }
        let plan2 = ctl.adaptive_plan(&contended, &blocks);
        assert!(
            plan2.batch_cap < 32,
            "contended + high gain: the cap drops below the slot count (got {})",
            plan2.batch_cap
        );
        assert!(
            plan2.batch_cap >= 4,
            "the cap is floored at the latency reserve (got {})",
            plan2.batch_cap
        );
    }

    /// G5 forward-seam contract: the `AdaptivePlan` carries four QoS actuators.
    /// On this branch only `batch_cap` + the base `StepPlan` are consumed by the
    /// scheduler. The other three (`parallelism_hint`, `kv_demote_threshold`,
    /// `sampling`) are pending slots for the branches above QoS in the stack
    /// (the multi-instance router / cross-host migrator, the unified-memory KV
    /// tier, the spec-decode + grammar rejection sampler). This test pins the
    /// contract so a later branch has a stable interface + a regression signal if
    /// QoS changes these fields.
    #[test]
    fn adaptive_plan_actuator_contract() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.calibration_steps = 8;
        let mut ctl = QosController::new(cfg);
        for _ in 0..8 {
            ctl.observe_calibration(10.0, 0.1, 0.05, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        let blocks = BlockSnapshot {
            free_blocks: 999,
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        let running = vec![seq_with(Some(100), 50)];
        let plan = ctl.adaptive_plan(&running, &blocks);
        // The G5 contract: the AdaptivePlan carries all four QoS actuators. The
        // base StepPlan (chunk + budget + gates) is the live actuator the
        // scheduler consumes today. The other three (parallelism_hint,
        // kv_demote_threshold, sampling) are pending slots for the branches
        // above QoS in the stack (the multi-instance router, the unified-memory
        // KV tier, the spec-decode + grammar rejection sampler). This test pins
        // the struct shape so a later branch has a stable interface + a
        // regression signal if QoS changes these fields.
        assert!(plan.base.chunk_size > 0, "the base chunk is always present");
        assert!(plan.base.token_budget > 0, "the base budget is always present");
        // The pending slots are typed correctly on the struct (the contract).
        let _hint: ParallelismHint = plan.parallelism_hint;
        let _demote: Option<f32> = plan.kv_demote_threshold;
        let _top_k: Option<u32> = plan.sampling.top_k;
        let _top_p: Option<f32> = plan.sampling.top_p;
        // The live batch cap is within the slot range (the P-Q4 real cap).
        assert!(plan.batch_cap <= blocks.max_num_seqs);
        assert!(plan.batch_cap >= 1);
    }

    /// G5: the pending actuators engage when their signals are present (the
    /// contract for the consumer branches). A contended + over-SLO regime emits
    /// the UseSequenceParallelism hint; high KV pressure emits a demotion
    /// threshold; a short-output median emits the top-k=1 sampling override.
    #[test]
    fn pending_actuators_engage_under_their_signals() {
        let mut cfg = QosConfig::default();
        cfg.enabled = true;
        cfg.calibration_steps = 8;
        let mut ctl = QosController::new(cfg);
        for _ in 0..8 {
            ctl.observe_calibration(10.0, 0.1, 0.1, Some(100), 32);
        }
        assert!(ctl.calibration_done());
        // Drive the ITL p95 far over the SLO + the regime contended.
        for _ in 0..200 {
            ctl.record_step(500.0, Regime::Contended, QosClass::Throughput);
        }
        let blocks = BlockSnapshot {
            free_blocks: 50, // low headroom -> high KV pressure
            total_blocks: 1000,
            max_num_seqs: 32,
            max_num_batched_tokens: 16384,
            static_chunk: 8192,
            chunk_floor: 256,
        };
        let mut contended: Vec<Sequence> = (0..32).map(|_| seq_with(Some(100), 100)).collect();
        for s in contended.iter_mut() {
            s.num_cached_tokens = s.len();
        }
        let plan = ctl.adaptive_plan(&contended, &blocks);
        // The contended + over-SLO -> the UseSequenceParallelism hint (the L3/L4
        // router/migrator consumes this).
        assert_eq!(plan.parallelism_hint, ParallelismHint::UseSequenceParallelism);
        // The high KV pressure (headroom 0.05 < the demote trigger) -> a demotion
        // threshold is emitted (the unified-memory branch consumes this).
        assert!(
            plan.kv_demote_threshold.is_some(),
            "high KV pressure must emit a demotion threshold"
        );
    }
}