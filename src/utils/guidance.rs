// src/utils/guidance.rs
// This module contains non-grammar guidance utilities:
// - GuidanceTokens: token ID collections
// - ParserFactory: llguidance parser factory
// - GuidanceState: matcher state for guided decoding

use crate::utils::config::TokenizerConfig;
use crate::utils::special_tokens::SpecialTokens;
use anyhow::Result;
use llguidance::{api::TopLevelGrammar, Matcher, ParserFactory as LlgParserFactory};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use toktrie::SimpleVob;
use toktrie_hf_tokenizers::ByteTokenizer;

// Re-export from guidance_grammar for grammar-related types
// Only export the two entrypoints: generate_grammar_from_request and build_grammar_from_request
pub use crate::utils::guidance_grammar::{
    build_grammar_from_request, generate_grammar_from_request,
};

#[derive(Clone, Debug, Default)]
pub struct GuidanceTokens {
    pub bos_token_ids: Vec<u32>,
    pub eos_token_ids: Vec<u32>,
    pub reasoning_start_ids: Vec<u32>,
    pub reasoning_end_ids: Vec<u32>,
    pub tool_call_start_ids: Vec<u32>,
    pub tool_call_end_ids: Vec<u32>,
    pub add_bos_token: bool,
}

pub fn extract_guidance_tokens(
    tokenizer: &Tokenizer,
    eos_token_ids: Vec<u32>,
    bos_token_ids: Vec<u32>,
    tokenizer_config: &TokenizerConfig,
) -> GuidanceTokens {
    let special_tokens = SpecialTokens::new(tokenizer);

    // Verify EOS token IDs are in added vocabulary if more than one provided
    let added_tokens: HashMap<u32, String> = tokenizer
        .get_added_tokens_decoder()
        .iter()
        .map(|(id, token)| (*id, token.content.clone()))
        .collect();

    let validated_eos: Vec<u32> = if eos_token_ids.len() > 1 {
        eos_token_ids
            .into_iter()
            .filter(|id| added_tokens.contains_key(id))
            .collect()
    } else {
        eos_token_ids
    };

    let validated_bos: Vec<u32> = {
        let retained: Vec<u32> = bos_token_ids
            .into_iter()
            .filter(|id| !validated_eos.contains(id))
            .collect();
        if retained.is_empty() {
            special_tokens.bos_token_ids()
        } else {
            retained
        }
    };

    // Determine if BOS token should be added based on tokenizer config
    // add_bos_token == Some(true) means the tokenizer adds BOS automatically
    let add_bos_token = tokenizer_config.add_bos_token == Some(true);

    GuidanceTokens {
        bos_token_ids: validated_bos,
        eos_token_ids: validated_eos,
        reasoning_start_ids: special_tokens.reasoning_start_ids(),
        reasoning_end_ids: special_tokens.reasoning_end_ids(),
        tool_call_start_ids: special_tokens.tool_call_start_ids(),
        tool_call_end_ids: special_tokens.tool_call_end_ids(),
        add_bos_token,
    }
}

pub type ParserFactory = LlgParserFactory;

pub fn build_llg_factory(
    tokenizer: Tokenizer,
    vocab_size: Option<usize>,
) -> Result<Arc<ParserFactory>> {
    let tokenizer_vocab = tokenizer.get_vocab_size(true);
    let target_vocab = vocab_size.map(|v| {
        if v < tokenizer_vocab {
            crate::log_warn!(
                "Requested vocab size {} is smaller than tokenizer vocab size {}. Using tokenizer size.",
                v,
                tokenizer_vocab
            );
            tokenizer_vocab
        } else {
            v
        }
    });
    let env = ByteTokenizer::from_tokenizer(tokenizer)?.into_tok_env(target_vocab)?;
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub struct GuidanceState {
    matcher: Matcher,
    /// Track generated tokens for logging and reasoning-mode transition.
    llm_tokens: Vec<u32>,
    /// vLLM/SGLang two-phase reasoning support:
    /// Token IDs that mark the end of reasoning (e.g. </think>).
    /// When non-empty, grammar constraints are deferred until after
    /// a reasoning-end token is seen. This keeps reasoning free-form
    /// and only constrains the structured output that follows.
    reasoning_end_ids: Vec<u32>,
    /// Whether reasoning has ended (the </think> token was observed).
    /// Once true, grammar masks are applied normally.
    reasoning_ended: bool,
}

impl GuidanceState {
    pub fn new_from_grammar_with_reasoning(
        factory: Arc<ParserFactory>,
        grammar: &TopLevelGrammar,
        reasoning_end_ids: Vec<u32>,
    ) -> Result<Self> {
        use crate::utils::guidance_grammar::get_lark_from_top_level_grammar;

        if tracing::enabled!(tracing::Level::DEBUG) {
            let lark = get_lark_from_top_level_grammar(grammar);
            tracing::debug!(
                "[llg] Initializing guidance parser from grammar: {} bytes, {} lines",
                lark.len(),
                lark.lines().count()
            );
            tracing::trace!("[llg] Guidance parser grammar:\n{}\n", lark);
        }

        let mut grammar = grammar.clone();
        if let Some(max_tokens) = grammar.max_tokens {
            let bos_len = 1;
            let eos_len = 1;
            grammar.max_tokens = Some(max_tokens + bos_len + eos_len);
        };
        let parser = factory.create_parser(grammar)?;
        let matcher = Matcher::new(Ok(parser));

        let reasoning_ended = reasoning_end_ids.is_empty();

        if !reasoning_ended {
            crate::log_info!(
                "[llg] Two-phase reasoning: grammar deferred until after reasoning end tokens {:?}",
                reasoning_end_ids
            );
        }

        Ok(Self {
            matcher,
            llm_tokens: Vec::new(),
            reasoning_end_ids,
            reasoning_ended,
        })
    }

    /// An independent copy of this state (the `deep_clone` matcher, the PQ-2
    /// proven-invariant scratch FSM). The `llm_tokens` + the reasoning flags are
    /// copied so the clone is a faithful shadow of the committed state at this
    /// moment. Used by the draft-side MaskStream (the speculative/mask_stream.rs)
    /// to walk the grammar trajectory WITHOUT advancing the committed state.
    pub fn deep_clone_state(&self) -> Self {
        Self {
            matcher: self.matcher.deep_clone(),
            llm_tokens: self.llm_tokens.clone(),
            reasoning_end_ids: self.reasoning_end_ids.clone(),
            reasoning_ended: self.reasoning_ended,
        }
    }

    /// The current matcher (the MaskStream reads the mask + the ff run on a
    /// sub-clone of this, the no observer effect on the walking state).
    pub fn matcher(&self) -> &Matcher {
        &self.matcher
    }

    /// Commit token and track for speculative decoding recovery.
    /// During reasoning, tokens are tracked but NOT fed to the grammar.
    /// When the reasoning-end token is seen, we transition to grammar mode.
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        self.llm_tokens.push(token);

        if !self.reasoning_ended {
            if self.reasoning_end_ids.contains(&token) {
                self.reasoning_ended = true;
                crate::log_warn!(
                    "[llg] Reasoning ended (token {}), grammar constraints now active (after {} reasoning tokens)",
                    token,
                    self.llm_tokens.len()
                );
            }
            return Ok(());
        }

        if !self.matcher.is_stopped() {
            self.matcher.consume_token(token)?;
        }
        Ok(())
    }

    /// Check if guidance is finished
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// Compute mask or return EOS token set if stopped.
    /// During reasoning, returns an all-ones mask (allow everything).
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        if !self.reasoning_ended {
            return self
                .matcher
                .compute_mask_or_eos()
                .map(|mut mask| {
                    mask.set_all(true);
                    mask
                })
                .map_err(Into::into);
        }
        self.matcher.compute_mask_or_eos().map_err(Into::into)
    }

    /// Fast-forward tokens without consuming them (for speculative decoding).
    /// During reasoning, no fast-forward is possible.
    pub fn compute_ff_tokens(&mut self) -> Vec<u32> {
        if !self.reasoning_ended {
            return Vec::new();
        }
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        self.matcher.compute_ff_tokens()
    }

    /// Build the per-verify-row grammar masks for a speculative verify step.
    ///
    /// The verify forward produces `all_logits` of shape `[verify_len, vocab]` where
    /// row `i` is the target distribution at the position reached after committing
    /// `[anchor, draft_0..i-1]`. For the lossless rejection sampler to be
    /// grammar-correct, each row's `p_target` must be the *masked* distribution (the
    /// illegal tokens zeroed), not the raw full-vocab softmax. This method returns
    /// the per-row allow-masks (the `SimpleVob` sets) so the caller can zero the
    /// illegal logits before the softmax.
    ///
    /// The construction is OPTIMISTIC: row `i`'s mask is the grammar state reached by
    /// projecting the committed matcher through `[draft_0..i-1]`. If a prior draft is
    /// grammar-invalid, the row is unreachable and is returned all-allowed (it will be
    /// discarded by the rejection sampler's accept/reject, so the optimistic mask is
    /// harmless). The committed matcher is NEVER advanced: every projection runs on a
    /// `deep_clone` (the PQ-2 invariant, the scratch is discarded).
    ///
    /// `drafts` is the per-row draft token (the `draft_i` that follows row `i`); the
    /// mask for row `i` is computed from the state after committing `drafts[0..i]`.
    /// Row 0 is the committed state directly (no projection).
    pub fn project_masks(
        &self,
        drafts: &[u32],
        verify_len: usize,
    ) -> anyhow::Result<Vec<SimpleVob>> {
        let all_allowed = || {
            let mut m = SimpleVob::new();
            m.set_all(true);
            m
        };
        let mut rows: Vec<SimpleVob> = Vec::with_capacity(verify_len);
        // Row 0: the committed state's mask (the post-anchor distribution). Read on a
        // deep_clone so this method is PURE (it never advances the committed matcher,
        // the G3 invariant). The mask-build step (`apply`) is the single intended
        // committed read; this projection is all scratch.
        rows.push(self.matcher.deep_clone().compute_mask_or_eos().unwrap_or_else(|_| all_allowed()));
        for i in 1..verify_len {
            // Project the committed state through drafts[0..i] on a deep_clone.
            let mut scratch = self.matcher.deep_clone();
            let mut reachable = true;
            for &d in &drafts[..i.min(drafts.len())] {
                if scratch.validate_tokens(&[d]).unwrap_or(0) == 0 {
                    reachable = false; // a prior draft was grammar-invalid -> row unreachable
                    break;
                }
                let _ = scratch.consume_token(d);
            }
            if !reachable {
                rows.push(all_allowed()); // the optimistic all-allowed (discarded on reject)
            } else {
                rows.push(scratch.compute_mask_or_eos().unwrap_or_else(|_| all_allowed()));
            }
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llguidance::api::TopLevelGrammar;
    use llguidance::Matcher;
    use toktrie::ApproximateTokEnv;

    #[test]
    fn test_extract_guidance_tokens() {
        // This test verifies that extract_guidance_tokens compiles
        // It doesn't actually run since we don't have a tokenizer here
        let tokens = GuidanceTokens::default();
        assert!(tokens.bos_token_ids.is_empty());
    }

    // PQ-2: prove the `deep_clone` projection semantics that `build_verify_vob`
    // depends on. The single-byte env maps each byte to a token id, so we can
    // build a grammar with forced literals and observe the intermediate masks.
    //
    // Grammar: start: <[97]> <[98]> <[99]> eos   (the tokens 'a','b','c' = 97,98,99)
    // After committing 'a', the parser forces 'b' (singleton mask). After 'b', it
    // forces 'c'. This gives clean intermediate states to project through.
    fn pq2_factory_and_matcher() -> (llguidance::ParserFactory, Matcher) {
        let env = ApproximateTokEnv::single_byte_env();
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let grm = TopLevelGrammar::from_lark(
            "start: <[97]> <[98]> <[99]>\n".to_string(),
        );
        let parser = factory.create_parser(grm).unwrap();
        (factory, Matcher::new(Ok(parser)))
    }

    // PQ-2 ASSERT A: projecting a deep_clone through one draft token yields the
    // post-draft mask (the intermediate state), not the pre-draft one.
    #[test]
    fn pq2_deep_clone_projection_yields_intermediate_mask() {
        let (_factory, matcher) = pq2_factory_and_matcher();
        let mut m = matcher;
        // Commit 'a' (97): the parser now forces 'b' (98).
        m.consume_token(97).unwrap();
        let mask_after_a = m.compute_mask_or_eos().unwrap();
        // The mask after 'a' is the singleton {98} (the grammar forces 'b').
        assert!(mask_after_a.is_allowed(98), "after committing 'a', the mask must allow 'b' (98)");
        assert!(!mask_after_a.is_allowed(97), "the mask must NOT allow 'a' (97) again (no self-loop)");

        // Project a deep_clone through the NEXT draft ('b'=98): the clone's mask
        // must be the singleton {99} (the grammar forces 'c' after 'b').
        let mut scratch = m.deep_clone();
        scratch.consume_token(98).unwrap();
        let mask_after_b = scratch.compute_mask_or_eos().unwrap();
        assert!(mask_after_b.is_allowed(99), "projecting through 'b', the mask must allow 'c' (99)");
        assert!(!mask_after_b.is_allowed(98), "projecting through 'b', the mask must NOT allow 'b' (98) again");

        // The ORIGINAL matcher (m) is still at the post-'a' state (its mask is
        // unchanged by the scratch work). Re-reading it gives the {98} singleton.
        let mask_original_reread = m.compute_mask_or_eos().unwrap();
        assert!(mask_original_reread.is_allowed(98), "the original matcher must be unperturbed by the deep_clone projection (still at post-'a')");
        assert!(!mask_original_reread.is_allowed(99), "the original must NOT have advanced to post-'b'");
    }

    // PQ-2 ASSERT B: the committed matcher is provably independent of scratch
    // projections (the G3 invariant that makes build_verify_vob safe).
    #[test]
    fn pq2_scratch_projection_does_not_perturb_committed_matcher() {
        let (_factory, matcher) = pq2_factory_and_matcher();
        let mut m = matcher;
        m.consume_token(97).unwrap(); // commit 'a'

        // Three independent scratch projections (through 'b', through 'b'+'c'),
        // each on its own deep_clone. The committed m must stay at post-'a'.
        for _ in 0..3 {
            let mut s1 = m.deep_clone();
            s1.consume_token(98).unwrap();
            let _ = s1.compute_mask_or_eos().unwrap();
            let mut s2 = m.deep_clone();
            s2.consume_token(98).unwrap();
            s2.consume_token(99).unwrap();
            let _ = s2.compute_mask_or_eos().unwrap();
        }

        // The committed matcher's state is unchanged: it still validates 'b' (98)
        // as the next token (the post-'a' state), and 'c' (99) as the one after.
        assert_eq!(m.validate_tokens(&[98, 99]).unwrap(), 2, "committed matcher must still be at post-'a' (both 'b' and 'c' valid ahead)");
        // And it has NOT consumed them (the next consume is still 'b').
        m.consume_token(98).unwrap();
        assert!(m.compute_mask_or_eos().unwrap().is_allowed(99), "after consuming 'b' on the committed matcher, the mask is the 'c' singleton");
    }

    // PQ-realworld (the verify-VOB positional correctness): drive the ACTUAL
    // `project_masks` (the method the MTP/DFlash verify path calls) on a forcing
    // grammar, and assert each verify row's mask is the positionally-correct
    // singleton. This is the real-world decode-step stepping: the committed state
    // is post-anchor, the drafts project forward, and row i's mask is the grammar
    // state after committing drafts[0..i].
    #[test]
    fn project_masks_positionally_correct_for_forcing_grammar() {
        use llguidance::ParserFactory;
        let env = ApproximateTokEnv::single_byte_env();
        let factory = Arc::new(ParserFactory::new_simple(&env).unwrap());
        // The forcing grammar: 'a' (97) then 'b' (98) then 'c' (99). After the
        // anchor 'a' is committed, the grammar forces 'b', then 'c'.
        let grm = TopLevelGrammar::from_lark("start: <[97]> <[98]> <[99]>\n".to_string());
        let mut state = GuidanceState::new_from_grammar_with_reasoning(
            factory.clone(),
            &grm,
            Vec::new(), // no reasoning (the single-phase, the grammar active from the start)
        )
        .unwrap();

        // Commit the anchor 'a' (97) -- this is what the prior decode step did.
        // The committed matcher is now at the post-'a' state (forces 'b').
        state.commit_token(97).unwrap();

        // The verify drafts: the MTP head proposed 'b' (98) then 'c' (99).
        // verify_len = 3 (the anchor row + 2 draft rows).
        let drafts = [98u32, 99];
        let rows = state.project_masks(&drafts, 3).unwrap();
        assert_eq!(rows.len(), 3, "one mask per verify row");

        // Row 0: the post-anchor state forces 'b' (98). Singleton {98}.
        assert!(rows[0].is_allowed(98), "row 0 must allow the forced 'b' (98)");
        assert!(!rows[0].is_allowed(97), "row 0 must NOT allow 'a' (97) again");
        assert!(!rows[0].is_allowed(99), "row 0 must NOT allow 'c' (99) yet (it's forced after 'b')");

        // Row 1: projecting through draft 'b' (98) reaches the post-'b' state,
        // which forces 'c' (99). Singleton {99}.
        assert!(rows[1].is_allowed(99), "row 1 (post-'b') must allow the forced 'c' (99)");
        assert!(!rows[1].is_allowed(98), "row 1 must NOT allow 'b' (98) again");

        // Row 2: projecting through 'b'+'c' reaches the post-'c' state (the EOS
        // region). The mask allows the EOS / the next structural token, NOT 'c' again.
        assert!(!rows[2].is_allowed(99), "row 2 (post-'c') must NOT allow 'c' (99) again");

        // The committed matcher is UNCHANGED by the projection (the G3 invariant):
        // it is still at the post-'a' state, so it still validates 'b' then 'c'.
        assert_eq!(state.matcher.validate_tokens(&[98, 99]).unwrap(), 2, "committed matcher must be at post-'a' (unperturbed by project_masks)");
    }

    // PQ-realworld (the stuck repro, the anti-loop collision): prove that a hard
    // -inf kick on the currently-forced singleton empties the candidate set (the
    // wedge), while a soft kick does not. This is the CPU-side proof of the exact
    // mechanism behind the user's "stuck with no output" log, without a GPU.
    #[test]
    fn kick_collision_hard_vs_soft_on_forcing_singleton() {
        use llguidance::ParserFactory;
        let env = ApproximateTokEnv::single_byte_env();
        let factory = Arc::new(ParserFactory::new_simple(&env).unwrap());
        let grm = TopLevelGrammar::from_lark("start: <[97]> <[98]> <[99]>\n".to_string());
        let mut state = GuidanceState::new_from_grammar_with_reasoning(factory, &grm, Vec::new()).unwrap();
        state.commit_token(97).unwrap(); // post-anchor: forces 'b' (98)

        // The mask for the current position is the singleton {98}.
        let mask = state.matcher.compute_mask_or_eos().unwrap();
        assert!(mask.is_allowed(98) && !mask.is_allowed(97) && !mask.is_allowed(99), "the forcing position is the singleton (only token 98 allowed)");

        // Simulate the sampler's candidate set over vocab {97,98,99} with the mask.
        // A token is a candidate iff (mask allows it) AND (logit is finite).
        let vocab = [97u32, 98, 99];
        let candidate = |kick: f32| -> Vec<u32> {
            vocab
                .iter()
                .filter(|&&t| mask.is_allowed(t))
                .filter(|&&t| {
                    // The kick penalizes token 98 (the loop unit contains it).
                    let logit = if t == 98 { kick } else { 0.0 };
                    logit.is_finite() // the -inf kick removes it from candidacy
                })
                .copied()
                .collect()
        };

        // HARD kick (-inf): the singleton {98} is removed -> empty candidate set ->
        // the Stage B degenerate guard fires -> token 0 (the wedge).
        assert_eq!(candidate(f32::NEG_INFINITY), Vec::<u32>::new(), "hard -inf kick empties the forcing singleton (the wedge)");

        // SOFT kick (-50.0): the singleton {98} survives (finite) -> the sampler
        // emits 98 (the forced token) -> the model advances (no wedge).
        assert_eq!(candidate(-50.0), vec![98], "soft kick leaves the forcing singleton sampleable (the model advances)");
    }
}
