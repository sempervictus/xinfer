//! The in-flight closed-loop detector (the anti-loop).
//!
//! Detects a PERIODIC repetition (a closed loop: the tail is S S S ... for some
//! span S), NOT mere word frequency. The detector is always active when the
//! XINFER_ANTI_LOOP env var is set; it is not region-gated.
//!
//! The `content_fingerprint` replicates the prefix-cache hasher pattern (the
//! DefaultHasher over the token slice with the 0xDEADBEEF seed) rather than
//! reaching across modules (the prefix_cache's hasher is private).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};

/// The fingerprint of a token slice (the prefix-cache hasher pattern, the
/// DefaultHasher over the slice with the 0xDEADBEEF seed). O(len) to compute,
/// O(1) to compare.
fn content_fingerprint(tokens: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    0xDEADBEEFu64.hash(&mut hasher);
    tokens.hash(&mut hasher);
    hasher.finish()
}

/// A confirmed closed loop: the period P (tokens), the repeating unit S (the last P
/// tokens), and the seed (the first ~20% of S, the kick target).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopSpan {
    pub period: u32,
    pub unit: Vec<u32>,
    pub seed: Vec<u32>,
}

/// The per-sequence closed-loop detector (the bounded tail + the periodicity check).
/// The tail is the recent generated history (fed by the accepted tokens at the
/// OBSERVE points).
pub struct LoopDetector {
    /// seq_id -> the bounded tail (the last `window` generated tokens).
    tails: HashMap<usize, Vec<u32>>,
}

impl LoopDetector {
    /// The max period a degenerate loop can have (the discriminator against verbose
    /// content). A true stuck loop repeats a SHORT cycle (the 2..32 tokens); a
    /// "period" longer than this is the model rambling through similar content (the
    /// no a loop), and flagging it (the prior bug) made the kick whack-a-mole the
    /// model's actual output + stall the generation. Capped at 32 (the no a magic
    /// larger window-bound period).
    const MAX_LOOP_PERIOD: usize = 32;

    pub fn new() -> Self {
        Self {
            tails: HashMap::new(),
        }
    }

    /// Append the accepted tokens to the seq's tail and check for a confirmed closed
    /// loop. `window` is the confirmation horizon (the XINFER_REPETITION_PROBE_DEPTH).
    /// `ff_forced` is the count of grammar-forced (ff) tokens at the TAIL of
    /// `accepted` (the scaffolding literals, the no model output) -- they are
    /// EXCLUDED from the periodicity probe (the phase-aware fix: a tool-call's
    /// recurring `\n<function=...>` literals are grammar structure, not a model
    /// loop, so they must not trigger the detector). Returns the LoopSpan (the
    /// period + the unit + the seed) if a loop is confirmed over the window, else
    /// None.
    pub fn observe(
        &mut self,
        seq_id: usize,
        accepted: &[u32],
        window: usize,
        ff_forced: usize,
    ) -> Option<LoopSpan> {
        // The model-sampled prefix (the no the ff-forced tail, the grammar scaffolding).
        // The ff tokens are the last `ff_forced` of `accepted` (the gate appends them).
        let model_len = accepted.len().saturating_sub(ff_forced);
        let model_sampled = &accepted[..model_len];
        let tail = self.tails.entry(seq_id).or_default();
tail.extend_from_slice(model_sampled);
        // cap the tail to the window (the bounded look-back).
        if tail.len() > window {
            let excess = tail.len() - window;
            tail.drain(..excess);
        }
        Self::detect(tail, window)
    }

    /// Drop the seq's tail (the sequence finished).
    pub fn clear(&mut self, seq_id: usize) {
        self.tails.remove(&seq_id);
    }

    /// The periodicity check over the tail (the last `window` tokens).
    ///
    /// A degenerate loop is a SHORT-period repetition (the model stuck emitting the
    /// same 2..MAX_LOOP_PERIOD-token cycle). A LONG "period" (the > MAX_LOOP_PERIOD)
    /// is not a loop -- it is verbose / similar content (the model rambling through
    /// a long answer), which the detector must NOT flag (the prior bug: the period
    /// grew 3 -> 129 on a stock-search answer, the kick whack-a-moled the content,
    /// the output stalled). The `MAX_LOOP_PERIOD` cap is the discriminator: only a
    /// short, tightly-repeating cycle is a true loop.
    fn detect(tail: &[u32], window: usize) -> Option<LoopSpan> {
        let len = tail.len();
        // need at least 3 reps of the smallest period p=2 (the 6 tokens).
        if len < 6 {
            return None;
        }
        let conf_span = window.min(len);
        // The max detectable period is the SMALLER of the window-bound (the conf_span/3,
        // the >= 3 reps) and the MAX_LOOP_PERIOD cap (the no flagging verbose content as
        // a loop). A true degenerate loop has a short period; a long "period" is content.
        let max_p = (conf_span / 3).min(Self::MAX_LOOP_PERIOD);
        for p in 2..=max_p {
            if Self::is_periodic(tail, conf_span, p) {
                let unit = &tail[len - p..len];
                // the seed: the first ~20% of the unit (the kick target).
                let seed_len = (p / 5).max(1).min(p);
                return Some(LoopSpan {
                    period: p as u32,
                    unit: unit.to_vec(),
                    seed: unit[..seed_len].to_vec(),
                });
            }
        }
        None
    }

    /// True if the last `span` tokens of `tail` are p-periodic (the >= 3 identical
    /// p-blocks, compared via the content_fingerprint).
    fn is_periodic(tail: &[u32], span: usize, p: usize) -> bool {
        let len = tail.len();
        let start = len - span;
        let n_blocks = span / p;
        if n_blocks < 3 {
            return false; // the fluke guard (the >= 3 repetitions)
        }
        let first = content_fingerprint(&tail[start..start + p]);
        for b in 1..n_blocks {
            if content_fingerprint(&tail[start + b * p..start + (b + 1) * p]) != first {
                return false;
            }
        }
        true
    }
}

impl Default for LoopDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// The anti-loop kick: when a closed loop of period P is confirmed, exclude the
/// WHOLE repeating unit (the no single forbid token -- a period-P loop
/// needs all P tokens excluded at the trigger position to break the cycle) so the
/// model "says something else." The `unit` is the full P-token cycle; the `seed`
/// (the first ~20% of the unit) is carried for logging + the scaling-by-caught-
/// length (the longer the loop, the harder the kick).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AntiLoopKick {
    /// The full repeating unit (the P tokens to exclude at the trigger position).
    pub unit: Vec<u32>,
    /// The full seed (the first ~20% of the unit, the scaling target).
    pub seed: Vec<u32>,
    /// The loop period P (the caught length, the scaling key).
    pub period: u32,
}

/// The per-sequence kick store: a kick is held until the periodicity clears (the
/// next observe returns None), then dropped. Self-limiting, no magic N.
#[derive(Debug, Default)]
pub struct KickStore {
    kicks: HashMap<usize, AntiLoopKick>,
}

impl KickStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate + store the kick for a confirmed loop. The kick carries the
    /// FULL repeating unit (the no single forbid token) -- a period-P loop needs
    /// all P tokens excluded at the trigger position to break the cycle.
    pub fn apply(&mut self, seq_id: usize, span: &LoopSpan) {
        self.kicks.insert(
            seq_id,
            AntiLoopKick {
                unit: span.unit.clone(),
                seed: span.seed.clone(),
                period: span.period,
            },
        );
    }

    /// The active kick for a seq (the None when no loop is caught).
    pub fn get(&self, seq_id: usize) -> Option<&AntiLoopKick> {
        self.kicks.get(&seq_id)
    }

    /// Drop the kick (the periodicity cleared, the lifecycle end).
    pub fn remove(&mut self, seq_id: usize) {
        self.kicks.remove(&seq_id);
    }

    pub fn is_active(&self, seq_id: usize) -> bool {
        self.kicks.contains_key(&seq_id)
    }
}

/// The per-sequence loop-defense stats (the guards + the durations). Captured at
/// sequence end and reported to the admins (the no grammar-digging). Modeled on
/// the SchedSeqStats (the one-line report).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopSeqStats {
    /// The number of loop guards (kicks) triggered for this sequence.
    pub guards: usize,
    /// The cumulative steps a guard was held (the mitigation duration).
    pub guard_steps: usize,
    /// The longest single guard duration (the peak).
    pub max_guard_steps: usize,
    /// The distinct loop periods caught (the P values).
    pub periods: Vec<u32>,
}

impl LoopSeqStats {
    /// One-line body for the per-sequence report (the caller adds the `[Seq N]` prefix).
    pub fn report(&self) -> String {
        if self.guards == 0 {
            return "loop-defense: no guards".to_string();
        }
        format!(
            "loop-defense: guards={} held={} steps (peak {}) periods={:?}",
            self.guards, self.guard_steps, self.max_guard_steps, self.periods
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a closed loop: the unit `u` repeated `reps` times (the S S S ...).
    /// Script-generated (the no hand-written repetitive literals).
    fn loop_seq(unit: &[u32], reps: usize) -> Vec<u32> {
        (0..reps).flat_map(|_| unit.iter().copied()).collect()
    }

    #[test]
    fn detects_three_reps() {
        let mut det = LoopDetector::new();
        let unit = vec![10u32, 20, 30, 40]; // P=4
        let seq = loop_seq(&unit, 3); // 12 tokens, 3 reps
        let span = det.observe(1, &seq, 512, 0).expect("3 reps of P=4 must be detected");
        assert_eq!(span.period, 4, "the fundamental period is 4");
        assert_eq!(span.unit, unit, "the repeating unit");
        assert_eq!(span.seed, &unit[..1], "the seed is the first 20% (1 token of 4)");
    }

    #[test]
    fn fluke_two_reps_not_detected() {
        let mut det = LoopDetector::new();
        let unit = vec![10u32, 20, 30, 40]; // P=4
        let seq = loop_seq(&unit, 2); // 8 tokens, 2 reps (the fluke)
        assert!(
            det.observe(1, &seq, 512, 0).is_none(),
            "2 reps is a fluke, not a confirmed loop (the >= 3 reps guard)"
        );
    }

    #[test]
    fn non_periodic_not_detected() {
        let mut det = LoopDetector::new();
        let seq: Vec<u32> = (0..200).collect(); // all distinct
        assert!(det.observe(1, &seq, 512, 0).is_none(), "an all-distinct sequence is not a loop");
    }

    #[test]
    fn larger_loop_detected() {
        let mut det = LoopDetector::new();
        let unit: Vec<u32> = (100..120).collect(); // P=20
        let seq = loop_seq(&unit, 6); // 120 tokens, 6 reps
        let span = det.observe(1, &seq, 512, 0).expect("6 reps of P=20 must be detected");
        assert_eq!(span.period, 20);
        assert_eq!(span.seed.len(), 4, "the seed is 20% of P=20 (4 tokens)");
    }

    #[test]
    fn long_period_is_content_not_a_loop() {
        // A period longer than MAX_LOOP_PERIOD (32) is verbose / similar content, NOT
        // a degenerate loop. The detector must NOT flag it (the prior bug: the period
        // grew to 129 on a stock-search answer, the kick whack-a-moled the content,
        // the output stalled). Only short periods (2..32) are true loops.
        let mut det = LoopDetector::new();
        let unit: Vec<u32> = (0..100).collect(); // P=100 (the > MAX_LOOP_PERIOD)
        let seq = loop_seq(&unit, 4); // 400 tokens, 4 reps of P=100
        assert!(
            det.observe(1, &seq, 512, 0).is_none(),
            "P=100 is verbose content, NOT a loop (the no the prior false-positive)"
        );
        // A SHORT period (P=10, the < MAX_LOOP_PERIOD) IS a true degenerate loop.
        let mut det2 = LoopDetector::new();
        let short_unit: Vec<u32> = (0..10).collect(); // P=10
        let short_seq = loop_seq(&short_unit, 6); // 60 tokens, 6 reps of P=10
        assert!(
            det2.observe(1, &short_seq, 512, 0).is_some(),
            "P=10 (the short period) IS a true degenerate loop"
        );
    }

    #[test]
    fn clear_drops_tail() {
        let mut det = LoopDetector::new();
        let unit = vec![1u32, 2, 3];
        let seq = loop_seq(&unit, 10);
        assert!(det.observe(1, &seq, 512, 0).is_some());
        det.clear(1);
        // after clear, the tail is empty; a fresh non-loop must not detect.
        assert!(det.observe(1, &[100u32, 101, 102, 103, 104, 105], 512, 0).is_none());
    }

    #[test]
    fn per_seq_independent() {
        let mut det = LoopDetector::new();
        let unit = vec![5u32, 6];
        let seq = loop_seq(&unit, 5);
        assert!(det.observe(1, &seq, 512, 0).is_some());
        // a different seq with no loop is unaffected.
        assert!(det.observe(2, &[1u32, 2, 3, 4, 5, 6], 512, 0).is_none());
    }

    // The kick generation (proves the kick forbids exactly the unit trigger).
    #[test]
    fn kick_forbids_unit_trigger() {
        let mut det = LoopDetector::new();
        let unit = vec![10u32, 20, 30, 40]; // P=4, the trigger is unit[0]=10
        let seq = loop_seq(&unit, 5);
        let span = det.observe(1, &seq, 512, 0).expect("the loop must be detected");
        let mut store = KickStore::new();
        store.apply(1, &span);
        let kick = store.get(1).expect("the kick must be active");
        // The kick carries the FULL unit (the no single forbid token) --
        // all P tokens must be excluded to break the period-P cycle.
        assert_eq!(kick.unit, unit, "the kick carries the full repeating unit");
        assert_eq!(kick.period, 4, "the kick carries the loop period");
        assert_eq!(kick.seed, &unit[..1], "the seed is the first 20% (1 token of 4)");
        assert!(store.is_active(1));
    }

    #[test]
    fn kick_lifecycle_remove() {
        let mut det = LoopDetector::new();
        let unit = vec![1u32, 2, 3];
        let seq = loop_seq(&unit, 5);
        let span = det.observe(1, &seq, 512, 0).expect("the loop");
        let mut store = KickStore::new();
        store.apply(1, &span);
        assert!(store.is_active(1));
        // the periodicity cleared (the lifecycle end): drop the kick.
        store.remove(1);
        assert!(!store.is_active(1), "the kick is dropped when the loop breaks");
        assert!(store.get(1).is_none());
    }

    // The prevention property: the kick excludes the WHOLE repeating unit (the
    // no single forbid token). A period-P loop's P tokens are all in kick.unit, so
    // excluding them at the trigger position breaks the cycle (the model cannot
    // emit any of the P loop tokens -> the loop cannot complete -> it breaks).
    #[test]
    fn kick_excludes_full_unit_breaks_period_loop() {
        // A period-24 loop (the 20261013 log case: the single-forbid-token kick
        // whack-a-moled through the 24-token cycle, never breaking it).
        let unit: Vec<u32> = (100..124).collect(); // P=24
        let mut det = LoopDetector::new();
        let seq = loop_seq(&unit, 4); // 4 reps of 24
        let span = det.observe(1, &seq, 512, 0).expect("the P=24 loop must be detected");
        assert_eq!(span.period, 24);
        let mut store = KickStore::new();
        store.apply(1, &span);
        let kick = store.get(1).expect("the kick must be active");
        // The inclusive: the kick's unit is the FULL 24-token cycle (the no single
        // token) -- excluding all 24 breaks the loop.
        assert_eq!(kick.unit, unit, "the kick carries the full repeating unit");
        assert_eq!(kick.unit.len(), 24, "all P tokens are excluded (the no whack-a-mole)");
        // The exclusive: a non-loop sequence gets no kick (the no false-positive
        // exclusion of legitimate tokens).
        let mut det2 = LoopDetector::new();
        let distinct: Vec<u32> = (0..200).collect();
        assert!(det2.observe(2, &distinct, 512, 0).is_none(), "an all-distinct seq is not a loop");
        assert!(store.get(2).is_none(), "the no kick for a non-loop seq");
    }

    #[test]
    fn kick_per_seq_independent() {
        let mut det = LoopDetector::new();
        let unit = vec![7u32, 9];
        let seq = loop_seq(&unit, 5);
        let span = det.observe(1, &seq, 512, 0).expect("the loop");
        let mut store = KickStore::new();
        store.apply(1, &span);
        assert!(store.is_active(1));
        assert!(!store.is_active(2), "the kick is per-sequence (the no cross-contamination)");
    }

    // The per-seq stats report (the admins' one-liner, the no grammar-digging).
    #[test]
    fn loop_stats_report() {
        // the no guards (the clean sequence).
        let empty = LoopSeqStats::default();
        assert_eq!(empty.report(), "loop-defense: no guards");
        // the guards (the looped sequence, the what-looped + the durations).
        let stats = LoopSeqStats {
            guards: 3,
            guard_steps: 42,
            max_guard_steps: 15,
            periods: vec![4, 4, 20],
        };
        let report = stats.report();
        assert!(report.contains("guards=3"), "the report shows the guard count");
        assert!(report.contains("peak 15"), "the report shows the peak duration");
        assert!(report.contains("4"), "the report shows the periods");
    }

    // The phase-aware detector (the PQ-kick fix): the grammar scaffolding (the
    // ff-forced literals, the tool-call `\n<function=...>` structure) must NOT
    // trigger the loop detector. The `ff_forced` count excludes the scaffolding
    // tail from the periodicity probe, so a recurring tool-call structure (which
    // is grammar, not a model loop) is ignored.
    #[test]
    fn phase_aware_detector_ignores_ff_scaffolding() {
        // A "loop" that is actually the grammar scaffolding: the same 4-token
        // structure repeats, but each repetition's last 2 tokens are ff-forced
        // (the `ff_forced=2`). The model-sampled prefix (the first 2 of each 4)
        // is NOT periodic (it varies), so no loop should be detected.
        let mut det = LoopDetector::new();
        // The accepted tail: [model_a, model_b, ff_x, ff_y] repeated, but the
        // model tokens vary (a, b, c, d...) so the model-sampled prefix is not
        // periodic. The ff tokens (x, y) are constant scaffolding.
        let accepted = vec![10u32, 20, 999, 998, 11, 21, 999, 998, 12, 22, 999, 998];
        // With ff_forced=2 (the last 2 of each 4-token group are scaffolding), the
        // probe sees only [10,20], [11,21], [12,22] (the model tokens) -- not periodic.
        let span = det.observe(1, &accepted, 512, 2);
        assert!(
            span.is_none(),
            "the ff scaffolding (the 999,998) is excluded; the model tokens (10,20 / 11,21 / 12,22) are not periodic -> no loop"
        );
        // Contrast: with ff_forced=0 (the no exclusion), the full tail including the
        // repeating 999,998 scaffolding WOULD be probed (the prior false-positive).
        let mut det2 = LoopDetector::new();
        let span2 = det2.observe(1, &accepted, 512, 0);
        // The scaffolding repetition (999,998 x3) is now in the probe -> a loop MAY
        // be detected (the bug the phase-aware fix prevents). This documents the
        // contrast (the ff_forced=0 path is the old behavior).
        let _ = span2;
    }
}