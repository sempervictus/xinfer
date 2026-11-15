use crate::utils::env::soft_mask_disabled;
use crate::utils::guidance::{GuidanceState, ParserFactory};
use candle_core::{Result, Tensor};
use llguidance::api::TopLevelGrammar;
use parking_lot::RwLock;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::Arc;
use toktrie::SimpleVob;

#[derive(Clone, Copy)]
pub struct GuidedDecodingRequest<'a> {
    pub seq_id: usize,
    pub grammar: Option<&'a TopLevelGrammar>,
    pub reasoning_end_ids: &'a [u32],
}

#[derive(Clone)]
pub struct GuidedDecodingStep {
    guided_seq_ids: Option<HashSet<usize>>,
}

impl GuidedDecodingStep {
    fn none() -> Self {
        Self {
            guided_seq_ids: None,
        }
    }

    fn new(guided_seq_ids: HashSet<usize>) -> Self {
        Self {
            guided_seq_ids: Some(guided_seq_ids),
        }
    }
}

/// Soft masking configuration for guided decoding.
/// Instead of hard masking to -inf, disallowed logits are shifted by a large value.
#[derive(Clone, Debug)]
pub struct SoftMaskConfig {
    pub mask_shift: f32,
    pub min_logit: f32,
    pub enabled: bool,
}

impl Default for SoftMaskConfig {
    fn default() -> Self {
        Self {
            mask_shift: -1000.0,
            min_logit: -1e9,
            enabled: !soft_mask_disabled(),
        }
    }
}

pub struct GuidedDecoding {
    factory: Option<Arc<ParserFactory>>,
    states: RwLock<HashMap<usize, GuidanceState>>,
    failed: RwLock<HashSet<usize>>,
    mismatch: RwLock<HashSet<usize>>,
    soft_mask: SoftMaskConfig,
}

impl GuidedDecoding {
    pub fn new(factory: Option<Arc<ParserFactory>>) -> Self {
        Self {
            factory,
            states: RwLock::new(HashMap::new()),
            failed: RwLock::new(HashSet::new()),
            mismatch: RwLock::new(HashSet::new()),
            soft_mask: SoftMaskConfig::default(),
        }
    }

    pub fn apply(
        &self,
        logits: &Tensor,
        requests: &[GuidedDecodingRequest<'_>],
    ) -> Result<(Tensor, GuidedDecodingStep)> {
        if requests.iter().all(|request| request.grammar.is_none()) {
            return Ok((logits.clone(), GuidedDecodingStep::none()));
        }

        let Some(factory) = &self.factory else {
            return Ok((logits.clone(), GuidedDecodingStep::none()));
        };

        let mut states = self.states.write();
        let mut failed = self.failed.write();
        let mut mismatch = self.mismatch.write();
        let mut modified = false;
        let batch_size = logits.dim(0)?;
        let vocab_size = logits.dim(1)?;

        let mut masks: Vec<(usize, usize, SimpleVob)> = Vec::new();
        let mut failed_seq_ids = Vec::new();
        let mut guided_seq_ids = HashSet::new();

        for request in requests {
            if request.grammar.is_none() {
                let _ = states.remove(&request.seq_id);
                let _ = failed.remove(&request.seq_id);
                let _ = mismatch.remove(&request.seq_id);
            }
        }

        for (batch_index, request) in requests.iter().enumerate() {
            let Some(grammar) = request.grammar else {
                continue;
            };

            let seq_id = request.seq_id;
            if failed.contains(&seq_id) {
                continue;
            }

            let state = match states.entry(seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => match GuidanceState::new_from_grammar_with_reasoning(
                    factory.clone(),
                    grammar,
                    request.reasoning_end_ids.to_vec(),
                ) {
                    Ok(state) => entry.insert(state),
                    Err(err) => {
                        failed.insert(seq_id);
                        crate::log_warn!(
                            "[Seq {}] Failed to create guidance state: {}. Disabling constraints for this sequence.",
                            seq_id,
                            err
                        );
                        continue;
                    }
                },
            };

            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    let mask_len = mask.len();
                    if mask_len == 0 {
                        if failed.insert(seq_id) {
                            crate::log_warn!(
                                "[Seq {}] Guidance mask length is 0. Disabling constraints for this sequence.",
                                seq_id
                            );
                        }
                        failed_seq_ids.push(seq_id);
                        continue;
                    }

                    if mask_len != vocab_size && mismatch.insert(seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Guidance mask size {} does not match vocab size {}. Clamping mask application.",
                            seq_id,
                            mask_len,
                            vocab_size
                        );
                    }

                    if !mask_allows_all(&mask, vocab_size) {
                        modified = true;
                    }
                    masks.push((batch_index, seq_id, mask));
                    guided_seq_ids.insert(seq_id);
                }
                Err(err) => {
                    if failed.insert(seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to compute guidance mask: {}. Disabling constraints for this sequence.",
                            seq_id,
                            err
                        );
                    }
                    failed_seq_ids.push(seq_id);
                }
            }
        }

        for seq_id in &failed_seq_ids {
            let _ = states.remove(seq_id);
        }

        let step = GuidedDecodingStep::new(guided_seq_ids);
        if !modified {
            return Ok((logits.clone(), step));
        }

        let mut allow_mask = vec![1u8; batch_size * vocab_size];
        for (seq_idx, _, mask) in masks {
            if mask_allows_all(&mask, vocab_size) {
                continue;
            }
            let start = seq_idx * vocab_size;
            write_allow_row(
                &mut allow_mask[start..start + vocab_size],
                &mask,
                vocab_size,
            );
        }

        let allow_mask = Tensor::from_vec(allow_mask, logits.shape().clone(), logits.device())?;
        let masked_logits = if self.soft_mask.enabled {
            let disallowed = logits
                .affine(1.0, self.soft_mask.mask_shift as f64)?
                .clamp(self.soft_mask.min_logit, f32::INFINITY)?;
            allow_mask.where_cond(&logits, &disallowed)?
        } else {
            let disallowed =
                Tensor::full(f32::NEG_INFINITY, logits.shape().clone(), logits.device())?;
            allow_mask.where_cond(&logits, &disallowed)?
        };

        Ok((masked_logits, step))
    }

    pub fn apply_fast_forward(&self, seq_ids: &[usize], tokens: &mut [u32]) {
        if self.factory.is_none() {
            return;
        }

        let mut states = self.states.write();
        for (i, seq_id) in seq_ids.iter().enumerate() {
            if let Some(state) = states.get_mut(seq_id) {
                let ff_tokens = state.compute_ff_tokens();
                if !ff_tokens.is_empty() && ff_tokens[0] != tokens[i] {
                    tokens[i] = ff_tokens[0];
                }
            }
        }
    }

    /// The full grammar-forced (ff) run at the seq's current state (empty if none).
    pub fn ff_tokens(&self, seq_id: usize) -> Vec<u32> {
        let mut states = self.states.write();
        states.get_mut(&seq_id).map(|s| s.compute_ff_tokens()).unwrap_or_default()
    }

    /// Build the per-verify-row grammar allow-masks for a speculative verify step
    /// (the lossless rejection sampler's `p_target` must be the *masked* target
    /// distribution, not the raw full-vocab softmax). Delegates to the seq's
    /// `GuidanceState::project_verify_masks` (the pure `deep_clone` projection,
    /// the committed matcher is never advanced). Returns `None` when the seq has
    /// no active guidance state (the unguided sequence, the no mask needed). The
    /// inner `Result` is the projection's outcome (the `Err` is a grammar failure
    /// mid-verify, which the caller must NOT fall back to the unmasked logits).
    pub fn project_verify_masks(
        &self,
        seq_id: usize,
        drafts: &[u32],
        verify_len: usize,
    ) -> Option<anyhow::Result<Vec<SimpleVob>>> {
        let states = self.states.read();
        states.get(&seq_id).map(|s| s.project_masks(drafts, verify_len))
    }

    /// Build the draft-side `MaskStream` for a seq (the `XINFER_SPEC_DRAFT_MASK`
    /// gate, the speculative/mask_stream.rs). Returns `None` when the seq has no
    /// active guidance state (the unguided sequence, the no draft masking). The
    /// stream wraps a `deep_clone_state` shadow (the committed matcher is never
    /// advanced, the PQ-2 / G3 invariant).
pub fn mask_stream_for(&self, seq_id: usize) -> Option<crate::speculative::mask_stream::MaskStream> {
        let states = self.states.read();
        states.get(&seq_id).map(|s| crate::speculative::mask_stream::MaskStream::new(s.deep_clone_state()))
    }

    /// Commit the full grammar-forced (ff) sequence for a seq, one token at a time.
    /// The ff tokens are deterministic (forced by the grammar), so they are appended
    /// directly without model sampling. Returns the number of tokens committed.
    pub fn commit_ff_sequence(&self, seq_id: usize) -> usize {
        let ff = self.ff_tokens(seq_id);
        if ff.is_empty() {
            return 0;
        }
        let step = GuidedDecodingStep::new(std::collections::HashSet::from([seq_id]));
        for &token in &ff {
            let _ = self.commit(&[seq_id], &[token], step.clone());
        }
        ff.len()
    }

    pub fn commit(&self, seq_ids: &[usize], tokens: &[u32], step: GuidedDecodingStep) {
        let Some(guided_seq_ids) = step.guided_seq_ids else {
            return;
        };

        let mut states = self.states.write();
        let mut failed = self.failed.write();
        for (seq_idx, seq_id) in seq_ids.iter().enumerate() {
            if !guided_seq_ids.contains(seq_id) || failed.contains(seq_id) {
                continue;
            }

            if let Some(state) = states.get_mut(seq_id) {
                if state.is_finished() {
                    continue;
                }

                let token = tokens[seq_idx];
                if let Err(err) = state.commit_token(token) {
                    if failed.insert(*seq_id) {
                        crate::log_warn!(
                            "[Seq {}] Failed to commit guided token {}: {}. Disabling constraints for this sequence.",
                            seq_id,
                            token,
                            err
                        );
                    }
                    let _ = states.remove(seq_id);
                }
            }
        }
    }

    pub fn finish(&self, seq_id: usize) {
        let mut states = self.states.write();
        let _ = states.remove(&seq_id);
        let mut failed = self.failed.write();
        let _ = failed.remove(&seq_id);
        let mut mismatch = self.mismatch.write();
        let _ = mismatch.remove(&seq_id);
    }
}

pub fn mask_allows_all(mask: &SimpleVob, vocab_size: usize) -> bool {
    if mask.len() < vocab_size {
        return false;
    }

    let words = mask.as_slice();
    let full_words = vocab_size / 32;
    if words.len() < full_words {
        return false;
    }
    if words[..full_words].iter().any(|word| *word != u32::MAX) {
        return false;
    }

    (full_words * 32..vocab_size).all(|tok| mask.is_allowed(tok as u32))
}

fn write_allow_row(row: &mut [u8], mask: &SimpleVob, vocab_size: usize) {
    row.fill(0);
    let apply_len = std::cmp::min(vocab_size, mask.len());
    mask.iter_set_entries(|idx| {
        if idx < apply_len {
            row[idx] = 1;
        }
    });
}

/// Apply a single verify-row grammar mask to a logits row (the -inf the illegal
/// tokens, the lossless-rejection `p_target` construction). A token is disallowed
/// iff it is within the mask's range (`tok < mask.len()`) AND its bit is clear.
/// Tokens beyond the mask's range are outside the grammar's constraint and stay
/// allowed. `SimpleVob::is_allowed` is unbounded (the svob `get` reads
/// `data[idx/32]` directly), so it must never be called with `tok >= mask.len()`.
/// This is the shared helper the MTP/DFlash verify path + the unit test use, so
/// the masking semantics are defined in exactly one place.
pub fn apply_vob_to_logits_row(row: &mut [f32], mask: &SimpleVob) {
    if mask_allows_all(mask, row.len()) {
        return; // the all-allowed row (the no grammar restriction, the no -inf)
    }
    let mask_len = mask.len();
    for (tok, cell) in row.iter_mut().enumerate() {
        if tok < mask_len && !mask.is_allowed(tok as u32) {
            *cell = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_vob_to_logits_row, mask_allows_all, write_allow_row};
    use toktrie::SimpleVob;

    #[test]
    fn test_mask_allows_all_respects_vocab_size() {
        let short = SimpleVob::alloc_ones(3);
        assert!(!mask_allows_all(&short, 4));

        let exact = SimpleVob::alloc_ones(4);
        assert!(mask_allows_all(&exact, 4));

        let mut partial = SimpleVob::alloc_ones(64);
        partial.disallow_token(63);
        assert!(!mask_allows_all(&partial, 64));
    }

    #[test]
    fn test_write_allow_row_clamps_to_vocab() {
        let mut mask = SimpleVob::alloc(6);
        mask.allow_token(1);
        mask.allow_token(5);
        let mut row = vec![1u8; 4];

        write_allow_row(&mut row, &mask, 4);

        assert_eq!(row, vec![0, 1, 0, 0]);
    }

    // The verify-side masking helper (the shared MTP/DFlash `p_target` construction).
    // The critical case: a SHORT mask (the grammar constrains only [0, mask_len)) must
    // -inf the in-range illegal tokens but LEAVE the out-of-range tokens allowed (the
    // `is_allowed` is unbounded, so it must never be called with tok >= mask_len).
    #[test]
    fn apply_vob_short_mask_leaves_out_of_range_allowed() {
        // A mask of length 4 allowing only tokens {1, 2} (0 and 3 disallowed).
        let mut mask = SimpleVob::alloc(4);
        mask.allow_token(1);
        mask.allow_token(2);
        // A logits row of length 8 (the vocab is wider than the mask range).
        let mut row = [10.0f32; 8];
        apply_vob_to_logits_row(&mut row, &mask);
        // In-range: token 0 + 3 disallowed -> -inf; tokens 1 + 2 allowed -> unchanged.
        assert_eq!(row[0], f32::NEG_INFINITY, "in-range disallowed token 0 -> -inf");
        assert_eq!(row[1], 10.0, "in-range allowed token 1 stays");
        assert_eq!(row[2], 10.0, "in-range allowed token 2 stays");
        assert_eq!(row[3], f32::NEG_INFINITY, "in-range disallowed token 3 -> -inf");
        // Out-of-range (tok >= mask_len=4): the mask does not constrain them -> allowed.
        assert_eq!(row[4], 10.0, "out-of-range token 4 stays allowed (the mask is short)");
        assert_eq!(row[7], 10.0, "out-of-range token 7 stays allowed");
    }

    #[test]
    fn apply_vob_all_allowed_row_is_noop() {
        // A mask that allows everything in range -> the row is untouched.
        let mask = SimpleVob::alloc_ones(8);
        let mut row = [5.0f32; 8];
        apply_vob_to_logits_row(&mut row, &mask);
        assert!(row.iter().all(|&x| x == 5.0), "all-allowed mask leaves the row unchanged");
    }

    #[test]
    fn apply_vob_singleton_forces_one_token() {
        // The forcing case: a singleton mask {3} over a length-8 row -> only token 3
        // survives (the rest -inf). This is the grammar-forced-literal position.
        let mut mask = SimpleVob::alloc(8);
        mask.allow_token(3);
        let mut row = [1.0f32; 8];
        apply_vob_to_logits_row(&mut row, &mask);
        assert_eq!(row[3], 1.0, "the forced token 3 keeps its logit");
        for (i, &x) in row.iter().enumerate() {
            if i != 3 {
                assert_eq!(x, f32::NEG_INFINITY, "token {i} (not the forced 3) is -inf");
            }
        }
    }
}
