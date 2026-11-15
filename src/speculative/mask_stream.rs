//! The draft-side grammar MaskStream (the `XINFER_SPEC_DRAFT_MASK` gate).
//!
//! A single cloned `GuidanceState` FSM walked incrementally by the *real* drafted
//! tokens. Each step yields `(mask, ff_run)`:
//! - `mask` = the grammar allow-set at the current position (the `SimpleVob`), read
//!   on a sub-clone so the walking FSM is not perturbed by the observer effect.
//! - `ff_run` = the grammar-forced (infallible) continuation, extracted so the
//!   drafter does NOT spend a forward pass proposing a literal the grammar forces.
//!
//! The drafter consumes `mask` to constrain its per-position proposal (the
//! `p_draft` shares support with the verify's masked `p_target`, the high-
//! acceptance lossless verify) and appends `ff_run` for free. `advance` commits
//! the real tokens onto the walking FSM, so the next step's mask is induced by
//! the *actual* trajectory (the positionally-correct mechanic, the no optimistic-
//! hypothetical flaw). The committed matcher is never touched (the walking FSM is
//! a `deep_clone_state`, the PQ-2 invariant).
//!
//! One trajectory, two consumers: the drafter (here) + the verify (the recorded
//! `masks` are the same stream the verify-side `project_verify_masks` would
//! re-derive, so they agree by construction).

use crate::utils::guidance::GuidanceState;
use candle_core::{Result, Tensor};
use toktrie::SimpleVob;

/// One step of the MaskStream: the position's grammar allow-set + the infallible
/// ff continuation (the literals to emit for free, the no draft forward).
pub struct MaskStep {
    /// The grammar allow-set at this position (the `SimpleVob`; the illegal tokens
    /// are -inf'd in the drafter's logits before the argmax). All-allow when
    /// `degraded` (the no constraint, the verify side enforces the grammar).
    pub mask: SimpleVob,
    /// The grammar-forced continuation (the infallible literals). When non-empty,
    /// the drafter appends these for free (no forward) + the verify's `p_target`
    /// at this position is the singleton (the residual draw is deterministic on
    /// the ff literal, the no illegal emission).
    pub ff_run: Vec<u32>,
    /// Set when the shadow FSM died at this position (the `compute_mask` failure,
    /// the parser-too-complex or the token-byte bridge desync). The drafter stops
    /// after this position (the committed prefix is kept, the failed position and
    /// beyond are discarded, the next step rebuilds a fresh shadow from the
    /// committed state). The mask is all-allow here (the no constraint, the verify
    /// side enforces the grammar, the no injection).
    pub degraded: bool,
}

/// The walking cloned-FSM MaskStream for one sequence's speculative step.
pub struct MaskStream {
    /// The shadow FSM (the `deep_clone_state`, the committed matcher is untouched).
    state: GuidanceState,
    /// The recorded per-position masks (the two-consumer trajectory: the verify
    /// reuses these instead of re-projecting, the no divergence).
    pub masks: Vec<SimpleVob>,
    /// The recorded per-position ff runs (the infallible literals, the verify's
    /// singleton rows).
    pub ff_runs: Vec<Vec<u32>>,
    /// Set once the shadow FSM dies (the `compute_mask` failure). After this, the
    /// stream is done: `next_step` returns a degraded all-allow step, `advance` is
    /// a no-op. The committed state is untouched (the PQ-2 invariant); the next
    /// speculative step builds a fresh shadow from the committed state.
    degraded: bool,
}

/// The result of a draft-step mask query: the allow-row + the degradation flag.
/// The `allow` is the U8 `[vocab]` row (1 = legal, 0 = illegal), all-allow when
/// degraded (the no constraint, the verify side enforces the grammar, the no
/// injection). The `degraded` flag tells the drafter to stop after this position
/// (the committed prefix is kept, the failed position + beyond are discarded, the
/// next step rebuilds a fresh shadow from the committed state).
pub struct DraftRow {
    /// The allow-row (the U8 `[vocab]`, 1 = legal / 0 = illegal). All-allow when
    /// degraded (the no constraint, the verify side enforces the grammar).
    pub allow: Tensor,
    /// Set when the shadow FSM died at this position (the drafter must stop after
    /// this position, the committed prefix is kept).
    pub degraded: bool,
}

impl MaskStream {
    /// Build the stream from a `deep_clone` of the seq's committed `GuidanceState`
    /// (the shadow, the PQ-2 invariant: the committed state is never advanced).
    pub fn new(state: GuidanceState) -> Self {
        Self {
            state,
            masks: Vec::new(),
            ff_runs: Vec::new(),
            degraded: false,
        }
    }

    /// Whether the shadow FSM has died (the drafter must stop after this step).
    pub fn is_degraded(&self) -> bool {
        self.degraded
    }

    /// The next step: read the current position's mask + ff run (on a sub-clone,
    /// the no observer effect on the walking FSM), record them, and return them.
    /// The caller then either (a) runs a draft forward masked by `mask` + calls
    /// `advance(drafted)`, or (b) if `ff_run` is non-empty, appends the ff tokens
    /// for free + calls `advance` (the no forward spent on a literal).
    ///
    /// On a mask computation failure (the grammar firewall, the parser-too-complex
    /// or the token-byte bridge desync), the shadow FSM is terminally dead (the
    /// llguidance matcher enters its error state, the no recovery). This method
    /// does NOT propagate the error (the no the crash the draft). It degrades: it
    /// marks the stream failed, returns an all-allow mask (the no constraint, the
    /// verify side enforces the grammar, the no injection) + an empty ff run, and
    /// sets `degraded` so the caller stops drafting after this position. The
    /// committed state is untouched (the PQ-2 invariant); the next speculative step
    /// rebuilds a fresh shadow from the committed state.
    pub fn next_step(&mut self, vocab: usize) -> MaskStep {
        // The already-degraded case: the shadow died at a prior position. Return an
        // all-allow degraded step (the no the re-probe the dead shadow, the no the
        // crash). The drafter stops after this step (the is_degraded flag). The mask
        // is all-allow at the vocab width (the no the caller's all-allow row, the
        // consistent sizing).
        if self.degraded {
            return MaskStep {
                mask: SimpleVob::alloc_ones(vocab),
                ff_run: Vec::new(),
                degraded: true,
            };
        }
        // The sub-clone reads the mask + the ff run without the observer effect
        // (the `compute_mask_or_eos`'s `ff_tokens_cache.take` + the `force_bytes`
        // advance) perturbing the walking FSM. The `compute_ff_tokens` on the
        // sub-clone is the advancing read; it is discarded with the sub-clone.
        let mut probe = self.state.deep_clone_state();
        let mask = match probe.compute_mask_or_eos() {
            Ok(mask) => mask,
            // The shadow FSM died (the parser-too-complex, the token-byte bridge
            // desync). Do NOT propagate the error (the no the crash the draft).
            // Degrade: mark the stream failed, return an all-allow mask at the vocab
            // width (the no constraint, the verify side enforces the grammar, the no
            // injection).
            Err(err) => {
                crate::log_warn!(
                    "[llg] MaskStream shadow died ({}); degrading the draft to all-allow + truncating",
                    err
                );
                self.degraded = true;
                SimpleVob::alloc_ones(vocab)
            }
        };
        let ff_run = if self.degraded {
            Vec::new() // the dead shadow has no ff (the constraint is lost)
        } else {
            probe.compute_ff_tokens()
        };
        self.masks.push(mask.clone());
        self.ff_runs.push(ff_run.clone());
        MaskStep {
            mask,
            ff_run,
            degraded: self.degraded,
        }
    }

    /// Commit the drafted token + the remaining ff literals onto the walking FSM, so
    /// the next `next_step`'s mask is induced by the actual trajectory. The
    /// `drafted` is the model's argmax (the single token it proposed this step). The
    /// remaining ff is the internally-recorded `ff_run` for this step, MINUS the
    /// drafted prefix (at a forcing position the model's argmax == the first ff
    /// literal, so it is not re-committed; at a free position the `ff_run` is empty).
    /// This avoids the double-commit of the forced literal. When the stream is
    /// degraded (the shadow died at a prior position), the advance is a no-op (the no
    /// the commit to a dead shadow, which would error). The caller stops drafting
    /// after a degraded step (the is_degraded flag).
    pub fn advance(&mut self, drafted: u32) -> Result<()> {
        if self.degraded {
            return Ok(());
        }
        self.state.commit_token(drafted).map_err(candle_core::Error::wrap)?;
        // The remaining ff literals (the no re-commit of the drafted token, which at
        // a forcing position IS the first ff literal).
        if let Some(ff) = self.ff_runs.last() {
            let remaining = if !ff.is_empty() && ff[0] == drafted {
                &ff[1..] // the drafted token was the first ff literal (the forcing case)
            } else {
                ff.as_slice() // the free case (the drafted token is not in the ff run)
            };
            for &t in remaining {
                self.state.commit_token(t).map_err(candle_core::Error::wrap)?;
            }
        }
        Ok(())
    }

/// The next step's grammar allow-row as a DEVICE U8 tensor (the `[vocab]`,
    /// 1 = legal / 0 = illegal) + the degradation flag. For the MTP draft loop's
    /// on-device `where_cond` (the DFlash idiom, the no CPU logits round-trip).
    /// The condition must be an integer dtype (the u8/u32/i64, the candle
    /// where_cond contract); the F32 allow-row was the prior bug (the "where
    /// conditions should be u8/u32/i64" crash on the MTP draft path).
    ///
    /// The per-step contract (the positionally-correct walk):
    /// 1. `let row = stream.next_allow_row(vocab, device)?;`
    /// 2. the model masks this step's draft logits with `row.allow` (the
    ///    `where_cond`) + argmaxes -> the drafted token `d`.
    /// 3. `stream.advance(d)?` -- commit the drafted token + the remaining ff
    ///    literals onto the walking FSM, so the NEXT row is induced by the actual
    ///    trajectory. The ff is tracked internally (the no caller-side ff plumbing).
    /// 4. If `row.degraded` is set, the shadow FSM died at this position: stop
    ///    drafting (the committed prefix is kept, the failed position + beyond are
    ///    discarded, the next step rebuilds a fresh shadow from the committed
    ///    state). The `allow` row is all-allow (the no constraint, the verify side
    ///    enforces the grammar, the no injection).
    ///
    /// When the current position is a forcing singleton, `allow` is that singleton
    /// + the model's argmax is forced to the literal (the infallible ff, the no
    /// draft forward spent on it).
    pub fn next_allow_row(
        &mut self,
        vocab: usize,
        device: &candle_core::Device,
    ) -> Result<DraftRow> {
        let step = self.next_step(vocab);
        // The SimpleVob -> the U8 [vocab] allow-row (1 = legal, 0 = illegal). The
        // U8 is the where_cond condition dtype (the no the F32, which candle
        // rejects as a condition). Tokens beyond the mask's range are unconstrained
        // (the 1, the allowed). When degraded, the mask is empty (the no the
        // constraint) -> the all-allow row (the no the OOB where_cond).
        let mut row = vec![1u8; vocab];
        if !step.degraded {
            let mask_len = step.mask.len().min(vocab);
            for tok in 0..mask_len {
                if !step.mask.is_allowed(tok as u32) {
                    row[tok] = 0;
                }
            }
        }
        let allow = Tensor::from_vec(row, (vocab,), device)?;
        Ok(DraftRow {
            allow,
            degraded: step.degraded,
        })
    }

    /// Whether the walking FSM has reached the grammar end (the no more drafting).
    pub fn is_finished(&self) -> bool {
        self.state.is_finished()
    }

    /// The recorded mask stream (the verify-side consumer: the same trajectory the
    /// drafter walked, the no re-projection divergence).
    pub fn recorded_masks(&self) -> &[SimpleVob] {
        &self.masks
    }

    /// The DFlash lattice allow-matrix (the `[n, vocab]` F32, 1.0 = legal / 0.0 =
    /// illegal). Pre-walks the MaskStream with the greedy trajectory (the argmax of
    /// each row's mask, the best-effort sequential induction) so slot `j`'s allow
    /// row reflects the grammar state after slots `0..j-1`. The DFlash2 lattice
    /// (`dflash_select_candidates_masked`) then walks a coherent draft path *within*
    /// these per-slot constraints. The verify-side mask (the Phase 1
    /// `project_verify_masks`) is the correctness backstop; this is the draft-side
    /// performance optimization (the `p_draft` shares support with the masked
    /// `p_target`).
    pub fn greedy_allow_matrix(
        &mut self,
        n: usize,
        vocab: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let step = self.next_step(vocab);
            let mut row = vec![1.0f32; vocab];
            if step.degraded {
                // The shadow died (the compute_mask failure). This position + all
                // remaining positions are unmasked (the all-allow row, the verify
                // side enforces the grammar, the no injection). The no the break-short
                // (the matrix must stay [n, vocab] for the lattice walk, the no an
                // OOB allow[j]). The committed prefix is kept; the next step rebuilds
                // a fresh shadow from the committed state.
                rows.push(row);
                continue;
            }
            let mask_len = step.mask.len().min(vocab);
            for tok in 0..mask_len {
                if !step.mask.is_allowed(tok as u32) {
                    row[tok] = 0.0;
                }
            }
            // The greedy trajectory: the argmax of the allow-row (the highest-index
            // allowed token, the deterministic stand-in for the draft the lattice
            // would pick). Advance the walking FSM by it so the next row is induced
            // by the actual (greedy) trajectory.
            let greedy_tok = row.iter().rposition(|&x| x > 0.0).map(|i| i as u32);
            if let Some(g) = greedy_tok {
                self.state.commit_token(g).map_err(candle_core::Error::wrap)?;
            }
            rows.push(row);
        }
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        Tensor::from_vec(flat, (n, vocab), device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::guidance::GuidanceState;
    use llguidance::api::TopLevelGrammar;
    use llguidance::ParserFactory;
    use toktrie::ApproximateTokEnv;

    /// Build a `GuidanceState` for the forcing grammar `start: <[97]> <[98]> <[99]>`
    /// (the tokens 'a'=97, 'b'=98, 'c'=99), committed up to the post-anchor state
    /// (the 'a' committed, the 'b' forced). The single-byte env (the no real
    /// tokenizer needed).
    fn forcing_state() -> GuidanceState {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm = TopLevelGrammar::from_lark("start: <[97]> <[98]> <[99]>\n".to_string());
        let mut state = GuidanceState::new_from_grammar_with_reasoning(
            std::sync::Arc::new(factory),
            &grm,
            Vec::new(),
        )
        .unwrap();
        state.commit_token(97).unwrap(); // the anchor 'a' committed; now 'b' (98) is forced
        state
    }

    // 1e.6: the MaskStream's per-step mask is the positionally-correct singleton,
    // and the ff run extracts the forced literal (the no draft forward spent on it).
    #[test]
    fn mask_stream_walks_forcing_grammar() {
        let mut stream = MaskStream::new(forcing_state());
        // Step 1: the current state (post-'a') forces 'b' (98) then 'c' (99) -- both
        // are grammar literals (the no free choice). The mask is the singleton {98}
        // (the immediate next token); the ff run is the full forced continuation
        // [98, 99] (the infallible literals, the no draft forward spent on them).
        let step1 = stream.next_step(256);
        assert!(step1.mask.is_allowed(98), "step 1 mask must allow the forced 'b' (98)");
        assert!(!step1.mask.is_allowed(97), "step 1 mask must NOT allow 'a' (97) again");
        assert_eq!(step1.ff_run, vec![98, 99], "the ff run is the full forced continuation [98, 99]");
        // Advance by the drafted token (98, the forced one). The ff_run [98, 99] has
        // 98 as its first element == the drafted token, so `advance` commits 98 once
        // (the drafted) + the remaining ff [99] (the no double-commit of 98).
        stream.advance(98).unwrap();
        // Step 2: now the state is post-[98, 99] (the 'c' committed via the ff). The
        // grammar is complete (the start rule satisfied) -> the mask is the EOS set.
        let step2 = stream.next_step(256);
        assert!(stream.is_finished() || !step2.mask.is_allowed(99), "after the full ff, 'c' (99) is no longer the immediate next (it was already committed)");
    }

    // 1e.6b: the greedy_allow_matrix (the DFlash lattice input) yields the
    // sequentially-induced per-slot masks (slot j = the state after slots 0..j-1).
    #[test]
    fn greedy_allow_matrix_is_sequentially_induced() {
        let mut stream = MaskStream::new(forcing_state());
        let dev = candle_core::Device::Cpu;
        // The vocab for the single-byte env is 256 (the byte range).
        let mat = stream.greedy_allow_matrix(2, 256, &dev).unwrap();
        let (rows, cols) = mat.dims2().unwrap();
        assert_eq!(rows, 2, "two slots (the n=2)");
        assert_eq!(cols, 256, "the full byte vocab");
        // Slot 0: the post-anchor state forces 'b' (98) -> allow[98]=1, the rest 0.
        let s0: Vec<f32> = mat.narrow(0, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(s0[98], 1.0, "slot 0 allows the forced 'b' (98)");
        assert_eq!(s0[97], 0.0, "slot 0 does NOT allow 'a' (97)");
        // Slot 1: after the greedy 'b' (98), the state forces 'c' (99) -> allow[99]=1.
        let s1: Vec<f32> = mat.narrow(0, 1, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(s1[99], 1.0, "slot 1 allows the forced 'c' (99) (the sequentially-induced mask)");
        assert_eq!(s1[98], 0.0, "slot 1 does NOT allow 'b' (98) again");
    }

    // 1e.9: the gate-off / unguided path is a true no-op. When the seq has no
    // guidance state, `mask_stream_for` returns `None` (the unguided seq), and the
    // model loop's `mask` param is `None` -> the prior unmasked draft behavior (the
    // no per-step mask kernel, the no CPU Earley walk). This is the production
    // back-compat guarantee: `XINFER_SPEC_DRAFT_MASK` off = zero overhead.
    #[test]
    fn unguided_seq_has_no_mask_stream() {
        use crate::utils::guided_decoding::GuidedDecoding;
        // A `GuidedDecoding` with no factory (the no guidance) yields no state, so
        // `mask_stream_for` is `None` (the gate-off / unguided no-op).
        let gd = GuidedDecoding::new(None);
        assert!(
            gd.mask_stream_for(42).is_none(),
            "an unguided seq (the no state) has no MaskStream (the no draft masking)"
        );
    }

    // The draft-side (greedy_allow_matrix) + verify-side (project_verify_masks)
    // masks agree on the shared trajectory: both are induced by the same committed
    // state + the same draft tokens, so the `p_draft` (masked at draft time) and the
    // `p_target` (masked at verify time) share support (the high-acceptance lossless
    // verify, the no illegal emission).
    #[test]
    fn draft_and_verify_masks_agree_on_shared_trajectory() {
        use crate::utils::guidance::GuidanceState;
        // Two independent streams from the same committed state (the draft-side + the
        // verify-side), walked with the same greedy trajectory.
        let committed = forcing_state();
        let mut draft_stream = MaskStream::new(committed.deep_clone_state());
        let verify_masks = committed.project_masks(&[98, 99], 3).unwrap();
        // The draft stream's slot-0 mask (the post-anchor singleton {98}) must equal
        // the verify's row-0 mask (the same post-anchor state).
        let step0 = draft_stream.next_step(256);
        assert_eq!(
            step0.mask.is_allowed(98),
            verify_masks[0].is_allowed(98),
            "the draft-side + verify-side slot-0 masks agree (the shared trajectory)"
        );
        assert_eq!(
            step0.mask.is_allowed(97),
            verify_masks[0].is_allowed(97),
            "the draft-side + verify-side slot-0 masks agree on the disallowed token"
        );
    }
}