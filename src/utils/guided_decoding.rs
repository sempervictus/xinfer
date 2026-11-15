use crate::core::runner::reasoning_budget_exhausted;
use crate::utils::env::max_reasoning_tokens;
use crate::utils::guidance::{GuidanceState, OutputPhase, ParserFactory};
use candle_core::{Device, Error, Result, Tensor};
use llguidance::api::TopLevelGrammar;
use parking_lot::RwLock;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::Arc;
use toktrie::SimpleVob;

#[derive(Clone, Copy)]
pub struct GuidedDecodingRequest<'a> {
    pub seq_id: usize,
    pub grammar: Option<&'a TopLevelGrammar>,
    /// The marker token-id sets that drive the `OutputPhase` machine on the
    /// `GuidanceState` (the reasoning start/end, the tool-call start/end).
    pub reasoning_start_ids: &'a [u32],
    pub reasoning_end_ids: &'a [u32],
    pub tool_call_start_ids: &'a [u32],
    pub tool_call_end_ids: &'a [u32],
    /// The newline token id(s) (the `"\n"` encoding), used to build the
    /// reasoning terminator run (`newline + EOR + newline`) the gate injects
    /// when the reasoning budget is exhausted.
    pub newline_tokens: &'a [u32],
    /// The per-request reasoning token budget (the `SamplingParams.reasoning_budget`).
    /// When `Some`, it overrides the `XINFER_MAX_REASONING_TOKENS` env for this
    /// sequence's reasoning-budget exhaustion check. The `None` means use the env.
    pub reasoning_budget: Option<usize>,
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

    pub fn new(guided_seq_ids: HashSet<usize>) -> Self {
        Self {
            guided_seq_ids: Some(guided_seq_ids),
        }
    }
}

pub struct GuidedDecoding {
    factory: Option<Arc<ParserFactory>>,
    states: RwLock<HashMap<usize, GuidanceState>>,
    failed: RwLock<HashSet<usize>>,
    /// Gate the per-rank `[llg]` grammar-mode logs to the rank-0 (the no 4x
    /// repetition on a multi-GPU TP run). The NCCL all-reduce makes all ranks
    /// agree, so the per-rank logs are identical.
    log_rank0: bool,
    /// The per-seq reasoning token count (the engine-stamped, the no the optional
    /// GuidanceState). The runner populates this from the DecodeSequence at the
    /// start of each step, so the reasoning budget is live on every sampling path
    /// (the MTP, DFlash, plain), not just the guided-sampling branches
    /// that create a GuidanceState.
    reasoning_counts: RwLock<HashMap<usize, usize>>,
    /// The per-seq reasoning-terminator ingredients (the EOR token + the newline
    /// tokens), stored so `reasoning_terminator` can synthesize the `newline + EOR
    /// + newline` run grammar-free (the no requiring a GuidanceState). Populated
    /// alongside `reasoning_counts` in `set_reasoning_count`.
    reasoning_terminator_parts: RwLock<HashMap<usize, (u32, Vec<u32>)>>,
    /// The per-seq consecutive mask-failure streak (the firewall re-decode bound). When a
    /// position's mask cannot be computed (the `ParserTooComplex` / the empty allow-set /
    /// the state-create failure), the row is set all-zero (the no token, the gate rejects,
    /// the seq re-decodes the position next step). This streak counts consecutive
    /// mask-failures for a seq WITHOUT committed progress; when it hits
    /// `MAX_MASK_FAIL_STREAK` the seq is terminated gracefully (the no an infinite
    /// re-decode stall on a terminally-stuck FSM, the no pool starvation in a multi-user
    /// run). Reset on any committed progress (the gate's `commit_run` > 0).
    mask_fail_streaks: RwLock<HashMap<usize, usize>>,
}

impl GuidedDecoding {
    pub fn new(factory: Option<Arc<ParserFactory>>, log_rank0: bool) -> Self {
        Self {
            factory,
            states: RwLock::new(HashMap::new()),
            failed: RwLock::new(HashSet::new()),
            log_rank0,
            reasoning_counts: RwLock::new(HashMap::new()),
            reasoning_terminator_parts: RwLock::new(HashMap::new()),
            mask_fail_streaks: RwLock::new(HashMap::new()),
        }
    }

    /// The max consecutive mask-failures before a seq is terminated gracefully (the
    /// re-decode bound, the no infinite stall on a terminally-stuck FSM). A mask
    /// failure sets the row all-zero (the no token, the gate rejects, the seq re-decodes
    /// the position next step). If the SAME position fails this many consecutive steps
    /// (the FSM is stuck, the re-decode makes no progress), the seq is terminated (the
    /// slot is freed, the client gets a completion) instead of stalling the pool.
    pub const MAX_MASK_FAIL_STREAK: usize = 32;

    /// The per-seq mask-failure streak (the re-decode bound observability).
    pub fn mask_fail_streak(&self, seq_id: usize) -> usize {
        self.mask_fail_streaks.read().get(&seq_id).copied().unwrap_or(0)
    }

    /// Record a mask failure for a seq (the firewall violation log + the streak bump).
    /// Called by the mask builders when a row's mask cannot be computed (the all-zero row,
    /// the no token, the gate rejects, the seq re-decodes). The violation is logged (the
    /// rank-0-gated WARN/ERROR) so the operator sees the firewall holding, not a silent
    /// stall.
    fn mask_fail(&self, seq_id: usize, reason: &str, empty_set: bool) {
        let mut streaks = self.mask_fail_streaks.write();
        let streak = streaks.entry(seq_id).or_insert(0);
        *streak += 1;
        if self.log_rank0 {
            if empty_set {
                crate::log_error!(
                    "[llg] seq {seq_id}: grammar allow-set is EMPTY (the committed prefix left the grammar, the firewall-integrity signal); the row is all-zero (the no token, the seq re-decodes from the last legal boundary). Streak: {streak}"
                );
            } else {
                crate::log_warn!(
                    "[llg] seq {seq_id}: grammar mask failed ({reason}); the row is all-zero (the no token, the seq re-decodes). Streak: {streak}"
                );
            }
        }
    }

    /// Clear the mask-failure streak (the committed progress, the seq is advancing).
    /// Called by the gate on a successful `commit_run` (the no a mask failure).
    pub fn clear_mask_fail_streak(&self, seq_id: usize) {
        let mut streaks = self.mask_fail_streaks.write();
        streaks.remove(&seq_id);
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
        let mut modified = false;
        let batch_size = logits.dim(0)?;
        let vocab_size = logits.dim(1)?;

        let mut masks: Vec<(usize, usize, SimpleVob)> = Vec::new();
        let mut guided_seq_ids = HashSet::new();
        // The rows whose mask FAILED (the firewall violation, the all-disallowed row, the
        // no token, the gate rejects, the seq re-decodes the position next step).
        let mut failed_rows: HashSet<usize> = HashSet::new();

        for request in requests {
            if request.grammar.is_none() {
                let _ = states.remove(&request.seq_id);
                let _ = failed.remove(&request.seq_id);
            }
        }

        for (batch_index, request) in requests.iter().enumerate() {
            let Some(grammar) = request.grammar else {
                // The unguided row: no grammar, no mask (the correct no-constraint).
                continue;
            };

            let seq_id = request.seq_id;
            // The grammar is active for this row. A mask that cannot be computed exactly
            // is a firewall VIOLATION: the row is marked all-DISALLOWED (the no token --
            // the sampler emits the degenerate sentinel, the gate REJECTS it, the seq
            // HOLDS its position, and the NEXT decode step re-attempts the position).
            // This is PREVENTION (the illegal token never reaches the Sequence), not a
            // crash + not a fail-open all-allowed (which would admit injected tokens).
            // The violation is logged + the per-seq mask-fail streak is advanced (the
            // re-decode bound, the no infinite stall on a terminally-stuck FSM).
            if failed.contains(&seq_id) {
                self.mask_fail(seq_id, "grammar in the failed set but mask was requested", false);
                failed_rows.insert(batch_index);
                continue;
            }

            let state = match states.entry(seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => match GuidanceState::new_from_grammar_with_markers(
                    factory.clone(),
                    grammar,
                    request.reasoning_end_ids.to_vec(),
                    request.tool_call_start_ids.to_vec(),
                    request.tool_call_end_ids.to_vec(),
                    request.reasoning_budget,
                    self.log_rank0,
                ) {
                    Ok(state) => entry.insert(state),
                    Err(err) => {
                        self.mask_fail(seq_id, &format!("failed to create the guidance state ({err})"), false);
                        failed_rows.insert(batch_index);
                        continue;
                    }
                },
            };

            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    if mask.len() == 0 {
                        self.mask_fail(seq_id, "grammar mask has length 0 (no token alphabet)", false);
                        failed_rows.insert(batch_index);
                        continue;
                    }
                    if mask.num_set() == 0 {
                        self.mask_fail(seq_id, "empty grammar allow-set (dead-end)", true);
                        failed_rows.insert(batch_index);
                        continue;
                    }
                    if !mask_allows_all(&mask, vocab_size) {
                        modified = true;
                    }
                    masks.push((batch_index, seq_id, mask));
                    guided_seq_ids.insert(seq_id);
                }
                Err(err) => {
                    self.mask_fail(seq_id, &format!("failed to compute the grammar mask ({err})"), false);
                    failed_rows.insert(batch_index);
                }
            }
        }

        let step = GuidedDecodingStep::new(guided_seq_ids);
        // A failed row (the firewall violation, the all-disallowed) forces the mask path
        // even if no successful mask modified the logits (the no the early return at
        // `!modified` leaking an unmasked -- fail-open -- row for a failed seq).
        if !modified && failed_rows.is_empty() {
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
        // The failed rows (the firewall violations): ALL-DISALLOWED (the no token -- the
        // sampler emits the degenerate sentinel, the gate rejects it, the seq holds its
        // position + re-decodes next step). The no all-allowed (the fail-open that admits
        // injected tokens), the no crash.
        for &row in failed_rows.iter() {
            allow_mask[row * vocab_size..(row + 1) * vocab_size].fill(0);
        }

        let allow_mask = Tensor::from_vec(allow_mask, logits.shape().clone(), logits.device())?;
        let disallowed =
            Tensor::full(f32::NEG_INFINITY, logits.shape().clone(), logits.device())?;
        let masked_logits: Tensor = allow_mask.where_cond(&logits, &disallowed)?;

        Ok((masked_logits, step))
    }

    /// Build the per-seq VOB bitset (the `[b, v/32] U32` tensor, the 8× less data than the
    /// F32 allow-mask). The VOB is the compact allowed-token set (the `SimpleVob`), passed
    /// directly to the `sample_with_vob` kernel (the no bitwise-AND sampler, the no F32
    /// mask round-trip). The `None` when no grammar is active (the no VOB needed).
    pub fn build_vob_words(
        &self,
        requests: &[GuidedDecodingRequest<'_>],
        vocab_size: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        if requests.iter().all(|request| request.grammar.is_none()) {
            return Ok(None);
        }
        let Some(factory) = &self.factory else {
            return Ok(None);
        };
        let factory = factory.clone();

        let mut states = self.states.write();
        let failed = self.failed.read();
        let mut vob_rows: Vec<Vec<u32>> = Vec::with_capacity(requests.len());
        let words_per_vob = (vocab_size + 31) / 32;

        for request in requests {
            let Some(_grammar) = request.grammar else {
                // The unguided row: no grammar, so all tokens are allowed (the correct
                // no-mask). This is NOT a fail-open — there is no constraint to enforce.
                vob_rows.push(vec![u32::MAX; words_per_vob]);
                continue;
            };
            let seq_id = request.seq_id;
            // The grammar is active for this row. A mask that cannot be computed exactly is a
            // firewall VIOLATION: the row is set ALL-ZERO (the no token allowed -- the
            // sampler emits the degenerate sentinel, the gate REJECTS it, the seq HOLDS
            // its position, and the NEXT decode step re-attempts the position). This is
            // PREVENTION (the illegal token never reaches the Sequence), not a crash +
            // not a fail-open all-allowed (which would admit injected tokens). The
            // violation is logged + the per-seq mask-fail streak is advanced (the
            // re-decode bound, the no infinite stall on a terminally-stuck FSM).
            if failed.contains(&seq_id) {
                self.mask_fail(seq_id, "grammar in the failed set but a mask was requested", false);
                vob_rows.push(vec![0u32; words_per_vob]); // the all-zero row (the no token)
                continue;
            }
            let state = match states.entry(seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    // First mask request for this sequence: create the guidance state
                    // (the lazy creation, matching build_allow_mask). A missing state
                    // here is not a firewall failure — it is the normal first-use case.
                    match GuidanceState::new_from_grammar_with_markers(
                        factory.clone(),
                        request.grammar.expect("grammar checked non-none above"),
                        request.reasoning_end_ids.to_vec(),
                        request.tool_call_start_ids.to_vec(),
                        request.tool_call_end_ids.to_vec(),
                        request.reasoning_budget,
                        self.log_rank0,
                    ) {
                        Ok(state) => entry.insert(state),
                        Err(err) => {
                            self.mask_fail(seq_id, &format!("failed to create the guidance state ({err})"), false);
                            vob_rows.push(vec![0u32; words_per_vob]); // the all-zero row
                            continue;
                        }
                    }
                }
            };
            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    // An empty allow-set for an active grammar is a dead-end (the committed
                    // prefix left the grammar, the firewall-integrity signal): the all-zero
                    // row (the no token, the gate rejects, the seq re-decodes from the last
                    // legal boundary) + the logged violation. The no crash, the no fail-open.
                    if mask.num_set() == 0 {
                        self.mask_fail(seq_id, "empty grammar allow-set (dead-end)", true);
                        vob_rows.push(vec![0u32; words_per_vob]); // the all-zero row
                        continue;
                    }
                    // The convert the SimpleVob to the [words_per_vob] U32 words (the VOB bitset).
                    let mut words = vec![0u32; words_per_vob];
                    let vob_words: Vec<u32> = mask.into();
                    for (word_idx, word) in vob_words.iter().enumerate() {
                        if word_idx < words_per_vob {
                            words[word_idx] = *word;
                        }
                    }
                    vob_rows.push(words);
                }
                Err(e) => {
                    // The `ParserTooComplex` / the stopped-state mask failure (the recoverable
                    // parser-capacity limit, the no committed-stream breach): the all-zero row
                    // (the no token, the gate rejects, the seq re-decodes) + the logged violation.
                    self.mask_fail(seq_id, &format!("failed to compute the grammar mask ({e})"), false);
                    vob_rows.push(vec![0u32; words_per_vob]); // the all-zero row
                }
            }
        }

        if vob_rows.is_empty() {
            return Ok(None);
        }
        let flat: Vec<u32> = vob_rows.into_iter().flatten().collect();
        let vob = Tensor::from_vec(flat, (requests.len(), words_per_vob), device)?;
        Ok(Some(vob))
    }

    /// The full grammar-forced (ff) run at the seq's current state (empty if none).
    /// This is an ADVANCING read (llguidance 1.9.1 `compute_ff_tokens` runs
    /// `force_bytes`). It is only called from the gate (`commit_run` path, the
    /// `spec_ff` speculative continuation) where the advance is the intended
    /// single FSM step. The sample site must NOT call this, or the parser
    /// advances twice per step (the `compute_mask` already advanced once to
    /// build the singleton VOB).
    pub fn ff_tokens(&self, seq_id: usize) -> Vec<u32> {
        let mut states = self.states.write();
        states.get_mut(&seq_id).map(|s| s.ff_run()).unwrap_or_default()
    }

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

    pub fn commit(&self, seq_ids: &[usize], tokens: &[u32], step: GuidedDecodingStep) {
        let Some(guided_seq_ids) = step.guided_seq_ids else {
            return;
        };

        let mut states = self.states.write();
        let failed = self.failed.read();
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
                    // The P2 fix: instead of disabling constraints (the failed.insert + the
                    // states.remove), resync to the last committed token boundary (the
                    // constraints stay active; the next step restarts from the last accepted token).
                    state.resync();
                    crate::log_warn!(
                        "[Seq {}] Failed to commit guided token {}: {}. Resyncing to last committed boundary.",
                        seq_id,
                        token,
                        err
                    );
                }
            }
        }
    }

    /// Commit a run of tokens to the sequence's grammar FSM (the single-writer
    /// gate). Returns the number of tokens committed (the FSM-passing prefix).
    /// Delegates to the per-sequence `GuidanceState::commit_run`.
    pub fn commit_run(&self, seq_id: usize, run: &[u32]) -> usize {
        let mut states = self.states.write();
        let Some(state) = states.get_mut(&seq_id) else {
            return run.len(); // no guidance state (the unguided sequence), pass through
        };
        let committed = state.commit_run(run);
        // Committed progress (the position validly filled) resets the mask-fail streak
        // (the re-decode bound, the no a terminally-stuck seq being terminated after a
        // step that actually made progress). A `committed == 0` (the whole run rejected,
        // the illegal-first-token / the mask-failed re-decode) does NOT reset it.
        if committed > 0 {
            self.clear_mask_fail_streak(seq_id);
        }
        committed
    }

    /// Commit the grammar-forced ff run in ONE matcher advance (the no the double-advance
    /// desync). Wraps `GuidanceState::commit_ff_run` (the `consume_ff_tokens` idiom).
    /// Returns the committed ff run (the token IDs, empty when the seq has no guidance
    /// state or the ff run is empty). The caller uses the returned IDs for the KV
    /// forward (the `run_ff_forward`) -- the commit + the KV write use the SAME ff run,
    /// the no a separate advancing `ff_tokens` read (the double-advance desync).
    pub fn commit_ff_run(&self, seq_id: usize) -> Vec<u32> {
        let mut states = self.states.write();
        let Some(state) = states.get_mut(&seq_id) else {
            return Vec::new(); // no guidance state (the unguided sequence), the no ff
        };
        let committed = state.commit_ff_run();
        if committed.is_empty() {
            return Vec::new();
        }
        self.clear_mask_fail_streak(seq_id);
        committed
    }

    /// The pre-ff `llm_tokens` boundary for a seq (the rollback target if the ff KV
    /// forward fails). The `llm_tokens.len()` BEFORE the ff commit; the gate captures
    /// this, runs the ff commit + the KV forward, and on KV failure rolls back to it.
    pub fn llm_tokens_len(&self, seq_id: usize) -> usize {
        let states = self.states.read();
        states.get(&seq_id).map(|s| s.llm_tokens_len()).unwrap_or(0)
    }

    /// Roll the seq's matcher + bookkeeping back to an explicit `llm_tokens` boundary
    /// (the ff KV-failure rollback, the no the committed-but-un-backed desync).
    pub fn resync_to(&self, seq_id: usize, boundary: usize) {
        let mut states = self.states.write();
        if let Some(state) = states.get_mut(&seq_id) {
            state.resync_to(boundary);
        }
    }

    /// Resync the seq's FSM to the last committed boundary (the P2 fix). Called by
    /// the gate after a partial `commit_run` (the legal suffix) so the matcher
    /// rolls back to the legal boundary. No-op when the seq has no guidance state.
    pub fn resync(&self, seq_id: usize) {
        let mut states = self.states.write();
        if let Some(state) = states.get_mut(&seq_id) {
            state.resync();
        }
    }

    pub fn finish(&self, seq_id: usize) {
        let mut states = self.states.write();
        let _ = states.remove(&seq_id);
        let mut failed = self.failed.write();
        let _ = failed.remove(&seq_id);
    }

    /// True if `seq_id` has an active (non-failed) grammar FSM state.
    pub fn is_guided(&self, seq_id: usize) -> bool {
        let states = self.states.read();
        let failed = self.failed.read();
        states.contains_key(&seq_id) && !failed.contains(&seq_id)
    }

    /// The count of reasoning tokens admitted for this sequence (the two-phase
    /// mode). The `0` when the sequence has no guidance state, or reasoning has
    /// ended, or the sequence is in the single-phase (full-envelope) mode. Used
    /// by the reasoning-budget promotion (the sampling bias that steers the
    /// model to the end-of-reasoning token as the count approaches the budget).
    ///
    /// Reads the engine-stamped count (the `reasoning_counts` map, populated by
    /// the runner from the `DecodeSequence`) first, falling back to the
    /// `GuidanceState` count (the prior behavior, for the guided-sampling paths
    /// that still maintain a state).
    pub fn reasoning_token_count(&self, seq_id: usize) -> usize {
        if let Some(count) = self.reasoning_counts.read().get(&seq_id) {
            return *count;
        }
        let states = self.states.read();
        states.get(&seq_id).map(|s| s.reasoning_token_count()).unwrap_or(0)
    }

    /// The per-phase token counts (the `[reasoning, text, tool_call]` tally) for a seq,
    /// read from its `GuidanceState` (the no the engine-stamped `reasoning_counts` map,
    /// which carries only the reasoning count). The `[0; 3]` when the seq has no
    /// guidance state (the unguided, the no per-phase breakdown). Used by the engine's
    /// Decoded summary + the /metrics export (the per-region token breakdown).
    pub fn phase_token_counts(&self, seq_id: usize) -> [usize; 3] {
        let states = self.states.read();
        states.get(&seq_id).map(|s| s.phase_token_counts()).unwrap_or([0; 3])
    }

    /// Populate the per-seq reasoning count from the engine-stamped value (the
    /// `DecodeSequence.reasoning_token_count`) + the terminator ingredients (the
    /// EOR token + the newline tokens, from the `SamplingParams`). Called by the
    /// runner at the start of each step so the budget check + the grammar-free
    /// terminator synthesis are live on every sampling path.
    pub fn set_reasoning_count(&self, seq_id: usize, count: usize, eor: u32, newline: Vec<u32>) {
        self.reasoning_counts.write().insert(seq_id, count);
        if eor != 0 {
            self.reasoning_terminator_parts
                .write()
                .insert(seq_id, (eor, newline));
        }
    }

    /// The reasoning terminator run for this sequence (`newline + EOR + newline`),
    /// or an empty run when no reasoning-end token is known. Synthesized grammar-free
    /// from the stored terminator ingredients (the `set_reasoning_count`), falling
    /// back to the `GuidanceState` when one exists. The gate commits this whole run
    /// atomically when the reasoning budget is exhausted, so no sampled token lands
    /// between its parts.
    pub fn reasoning_terminator(&self, seq_id: usize) -> Vec<u32> {
        // The grammar-free synthesis from the stored ingredients (the EOR + the
        // newline tokens, set by the runner from the SamplingParams).
        if let Some((eor, newline)) = self.reasoning_terminator_parts.read().get(&seq_id).cloned() {
            let mut run = newline.clone();
            run.push(eor);
            run.extend(newline);
            return run;
        }
        // The fallback: the GuidanceState's terminator (the grammar path, the
        // newline + EOR + newline built from the state's marker sets).
        let states = self.states.read();
        states.get(&seq_id).map(|s| s.reasoning_terminator()).unwrap_or_default()
    }

    /// The current conversation output region (the `Reasoning` / `Text` / `ToolCall`)
    /// for this sequence. The `Text` when the sequence has no guidance state (the
    /// unguided, the default region). Used by the gate to decide whether the
    /// reasoning-terminator injection applies (only in the `Reasoning` region).
    pub fn phase(&self, seq_id: usize) -> OutputPhase {
        let states = self.states.read();
        states.get(&seq_id).map(|s| s.phase()).unwrap_or(OutputPhase::Text)
    }

    /// Whether the reasoning budget is exhausted for this sequence (the count has
    /// reached or passed the budget). The budget is the `XINFER_MAX_REASONING_TOKENS`
    /// env (the per-request `reasoning_budget` is read by the runner directly).
    /// Returns `false` when no budget is set (the env unset) or the sequence has
    /// no guidance state. Used by the MTP/DFlash decode to suppress drafts when
    /// the budget is exhausted (the anchor is forced to the end-of-reasoning
    /// token, no further reasoning drafts).
    pub fn reasoning_budget_exhausted(&self, seq_id: usize) -> bool {
        let count = self.reasoning_token_count(seq_id);
        // The per-request budget (the `SamplingParams.reasoning_budget`, stored on
        // the `GuidanceState`) wins; the `XINFER_MAX_REASONING_TOKENS` env is the
        // default. The budget at all -> never exhausted.
        let budget = {
            let states = self.states.read();
            let env_budget = max_reasoning_tokens();
            states.get(&seq_id)
                .and_then(|s| s.reasoning_budget().or(env_budget))
                .or(env_budget)
        };
        let Some(budget) = budget else {
            return false;
        };
        reasoning_budget_exhausted(count, budget)
    }

    /// Non-mutating: grammar-legal prefix length of `tokens` from the seq's current state.
    pub fn validate_tokens(&self, seq_id: usize, tokens: &[u32]) -> Result<usize> {
        if tokens.is_empty() {
            return Ok(0);
        }
        let mut states = self.states.write();
        match states.get_mut(&seq_id) {
            Some(state) => state
                .validate_tokens(tokens)
                .map_err(|e| Error::Msg(e.to_string())),
            None => Ok(tokens.len()),
        }
    }

    /// Commit a single token to the seq's FSM (advances state; tracks reasoning).
    pub fn commit_token(&self, seq_id: usize, token: u32) {
        let mut states = self.states.write();
        if let Some(state) = states.get_mut(&seq_id) {
            if state.is_finished() {
                return;
            }
            if let Err(err) = state.commit_token(token) {
                // The P2 fix: instead of disabling constraints (the failed.insert + the
                // states.remove), resync to the last committed token boundary (the constraints
                // stay active; the next step restarts from the last accepted token).
                state.resync();
                crate::log_warn!(
                    "[Seq {}] Failed to commit guided token {}: {}. Resyncing to last committed boundary.",
                    seq_id,
                    token,
                    err
                );
            }
        }
    }

    /// Apply the seq's current grammar VOB to a single logit row; returns the masked row.
    /// No-op (returns the row) if the seq is not guided.
    pub fn mask_row(&self, seq_id: usize, row: &Tensor) -> Result<Tensor> {
        let mut states = self.states.write();
        let state = match states.get_mut(&seq_id) {
            Some(s) => s,
            None => return Ok(row.clone()),
        };
        let mask = match state.compute_mask_or_eos() {
            Ok(m) => m,
            Err(e) => return Err(Error::Msg(e.to_string())),
        };
        drop(states);
        apply_vob_to_row(row, &mask)
    }

    /// Apply the seq's current grammar VOB to every row of `logits` [n, vocab] (batched).
    pub fn mask_rows(&self, seq_id: usize, logits: &Tensor) -> Result<Tensor> {
        let mut states = self.states.write();
        let state = match states.get_mut(&seq_id) {
            Some(s) => s,
            None => return Ok(logits.clone()),
        };
        let mask = match state.compute_mask_or_eos() {
            Ok(m) => m,
            Err(e) => return Err(Error::Msg(e.to_string())),
        };
        drop(states);
        let vocab_size = logits.dims().last().copied().unwrap_or(0) as usize;
        if mask_allows_all(&mask, vocab_size) {
            return Ok(logits.clone());
        }
        let n = logits.dim(0)?;
        let mut allow = vec![0u8; vocab_size];
        write_allow_row(&mut allow, &mask, vocab_size);
        let allow = Tensor::from_vec(allow, (vocab_size,), logits.device())?;
        let allow_2d = allow.expand((n, vocab_size))?;
        let disallowed = Tensor::full(f32::NEG_INFINITY, logits.shape().clone(), logits.device())?;
        Ok(allow_2d.where_cond(logits, &disallowed)?)
    }

    /// Build the per-row grammar allow-mask `[requests.len(), vocab]` (u8, 1 = legal,
    /// 0 = illegal) for the CUDA-sampler offload path. Returns `None` when no row needs
    /// gating (no grammar, or every VOB allows the whole vocab).
    pub fn build_allow_mask(
        &self,
        requests: &[GuidedDecodingRequest<'_>],
        vocab_size: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        if self.factory.is_none() || vocab_size == 0 {
            return Ok(None);
        }
        let factory = self.factory.clone().expect("factory checked non-none above");
        let batch_size = requests.len();
        let mut states = self.states.write();
        let mut failed = self.failed.write();
        let mut any_gate = false;
        let mut allow = vec![1u8; batch_size * vocab_size];
        for (row, request) in requests.iter().enumerate() {
            let Some(grammar) = request.grammar else {
                // The unguided row: no grammar, all tokens allowed (the correct no-mask).
                let _ = states.remove(&request.seq_id);
                let _ = failed.remove(&request.seq_id);
                continue;
            };
            // The grammar is active for this row. A mask that cannot be computed exactly is a
            // firewall VIOLATION: the row is ALL-DISALLOWED (the no token -- the sampler
            // emits the degenerate sentinel, the gate REJECTS it, the seq HOLDS its
            // position, and the NEXT decode step re-attempts the position). This is
            // PREVENTION (the illegal token never reaches the Sequence), not a crash +
            // not a fail-open all-allowed (which would admit injected tokens). The
            // violation is logged + the per-seq mask-fail streak is advanced (the
            // re-decode bound, the no infinite stall on a terminally-stuck FSM).
            let seq_id = request.seq_id;
            if failed.contains(&seq_id) {
                self.mask_fail(seq_id, "grammar in the failed set but mask was requested", false);
                allow[row * vocab_size..(row + 1) * vocab_size].fill(0);
                any_gate = true;
                continue;
            }
            let state = match states.entry(seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => match GuidanceState::new_from_grammar_with_markers(
                    factory.clone(),
                    grammar,
                    request.reasoning_end_ids.to_vec(),
                    request.tool_call_start_ids.to_vec(),
                    request.tool_call_end_ids.to_vec(),
                    request.reasoning_budget,
                    self.log_rank0,
                ) {
                    Ok(state) => entry.insert(state),
                    Err(err) => {
                        self.mask_fail(seq_id, &format!("failed to create the guidance state ({err})"), false);
                        allow[row * vocab_size..(row + 1) * vocab_size].fill(0);
                        any_gate = true;
                        continue;
                    }
                },
            };
            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    if mask.len() == 0 {
                        self.mask_fail(seq_id, "grammar mask has length 0 (no token alphabet)", false);
                        allow[row * vocab_size..(row + 1) * vocab_size].fill(0);
                        any_gate = true;
                        continue;
                    }
                    if mask.num_set() == 0 {
                        self.mask_fail(seq_id, "empty grammar allow-set (dead-end)", true);
                        allow[row * vocab_size..(row + 1) * vocab_size].fill(0);
                        any_gate = true;
                        continue;
                    }
                    if !mask_allows_all(&mask, vocab_size) {
                        any_gate = true;
                        write_allow_row(
                            &mut allow[row * vocab_size..row * vocab_size + vocab_size],
                            &mask,
                            vocab_size,
                        );
                    }
                }
                Err(err) => {
                    self.mask_fail(seq_id, &format!("failed to compute the grammar mask ({err})"), false);
                    allow[row * vocab_size..(row + 1) * vocab_size].fill(0);
                    any_gate = true;
                }
            }
        }
        if !any_gate {
            return Ok(None);
        }
        Ok(Some(Tensor::from_vec(allow, (batch_size, vocab_size), device)?))
    }

    /// Build the raw VOB bitset words for the full batch. Returns
    /// `[batch_size * vocab_size/32]` u32 words (bit i set = token allowed).
    /// 8x less data than the F32 mask tensor. Returns None when no row
    /// needs gating.
    pub fn build_vob_words_flat(
        &self,
        requests: &[GuidedDecodingRequest<'_>],
        vocab_size: usize,
    ) -> Option<Vec<u32>> {
        if self.factory.is_none() || vocab_size == 0 {
            return None;
        }
        let batch_size = requests.len();
        let num_words = (vocab_size + 31) / 32;
        // Initialize to all-ones (allow everything by default); gated rows are reset
        // to zero below before their allowed bits are OR-ed in. Allow-all rows stay
        // all-ones, so a mixed batch never masks out its free rows.
        let mut words = vec![u32::MAX; batch_size * num_words];
        let mut any_gate = false;

        let mut states = self.states.write();
        let mut failed = self.failed.write();

        for (row, request) in requests.iter().enumerate() {
            let Some(grammar) = request.grammar else {
                // The unguided row: no grammar, all tokens allowed (the correct no-mask).
                // The row stays at the all-ones init (line 1004).
                let _ = states.remove(&request.seq_id);
                let _ = failed.remove(&request.seq_id);
                continue;
            };
            // The grammar is active for this row. A mask that cannot be computed exactly is a
            // firewall VIOLATION: the row is set ALL-ZERO (the no token allowed -- the
            // sampler emits the degenerate sentinel, the gate REJECTS it, the seq HOLDS
            // its position, and the NEXT decode step re-attempts the position). This is
            // PREVENTION (the illegal token never reaches the Sequence), not a crash +
            // not a fail-open all-allowed (which would admit injected tokens). The
            // violation is logged + the per-seq mask-fail streak is advanced (the
            // re-decode bound, the no infinite stall on a terminally-stuck FSM).
            let seq_id = request.seq_id;
            let zero_row = |words: &mut [u32], row: usize, num_words: usize| {
                for w in 0..num_words {
                    words[row * num_words + w] = 0;
                }
            };
            if failed.contains(&seq_id) {
                self.mask_fail(seq_id, "grammar in the failed set but a VOB mask was requested", false);
                zero_row(&mut words, row, num_words);
                any_gate = true;
                continue;
            }
            let state = match states.entry(seq_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let factory = self.factory.clone().unwrap();
                    match GuidanceState::new_from_grammar_with_markers(
                        factory,
                        grammar,
                        request.reasoning_end_ids.to_vec(),
                        request.tool_call_start_ids.to_vec(),
                        request.tool_call_end_ids.to_vec(),
                        request.reasoning_budget,
                        self.log_rank0,
                    ) {
                        Ok(state) => entry.insert(state),
                        Err(err) => {
                            self.mask_fail(seq_id, &format!("failed to create the guidance state ({err})"), false);
                            zero_row(&mut words, row, num_words);
                            any_gate = true;
                            continue;
                        }
                    }
                }
            };
            match state.compute_mask_or_eos() {
                Ok(mask) => {
                    if mask.len() == 0 {
                        self.mask_fail(seq_id, "grammar mask has length 0 (no token alphabet)", false);
                        zero_row(&mut words, row, num_words);
                        any_gate = true;
                        continue;
                    }
                    if mask.num_set() == 0 {
                        self.mask_fail(seq_id, "empty grammar allow-set (dead-end)", true);
                        zero_row(&mut words, row, num_words);
                        any_gate = true;
                        continue;
                    }
                    if !mask_allows_all(&mask, vocab_size) {
                        any_gate = true;
                        let row_base = row * num_words;
                        // Reset this gated row to zero (the default all-ones would allow
                        // everything); the allowed bits are OR-ed in below.
                        for w in 0..num_words {
                            words[row_base + w] = 0;
                        }
                        let apply_len = std::cmp::min(vocab_size, mask.len());
                        mask.iter_set_entries(|idx| {
                            if idx < apply_len {
                                words[row_base + idx / 32] |= 1u32 << (idx % 32);
                            }
                        });
                    }
                }
                Err(err) => {
                    self.mask_fail(seq_id, &format!("failed to compute the grammar mask ({err})"), false);
                    zero_row(&mut words, row, num_words);
                    any_gate = true;
                }
            }
        }

        if !any_gate {
            return None;
        }
        Some(words)
    }
}

/// Apply a grammar VOB to a single logit row (disallowed -> -inf). No-op if the mask allows all.
fn apply_vob_to_row(row: &Tensor, mask: &SimpleVob) -> Result<Tensor> {
    let vocab_size = row.dims().last().copied().unwrap_or(0) as usize;
    if mask_allows_all(mask, vocab_size) {
        return Ok(row.clone());
    }
    let mut allow = vec![0u8; vocab_size];
    write_allow_row(&mut allow, mask, vocab_size);
    let allow = Tensor::from_vec(allow, row.shape().clone(), row.device())?;
    let disallowed = Tensor::full(f32::NEG_INFINITY, row.shape().clone(), row.device())?;
    Ok(allow.where_cond(row, &disallowed)?)
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
use super::{apply_vob_to_logits_row, mask_allows_all, write_allow_row, GuidedDecoding, GuidedDecodingRequest};
    use toktrie::SimpleVob;
    use candle_core::Tensor;

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

    /// The C guard predicate: a mask that covers the vocab but allows NO token
    /// (num_set == 0) is a dead-end that must be resynced + treated as
    /// all-allowed, never as an empty allow-set. Distinguish it from the
    /// zero-LENGTH mask (a different, already-handled case) + from a normal
    /// gated mask (num_set > 0).
    #[test]
    fn test_dead_end_mask_detection() {
        // A dead-end: sized to the vocab, but no token allowed.
        let dead_end = SimpleVob::alloc(64); // all bits clear
        assert_eq!(dead_end.len(), 64, "the dead-end mask covers the vocab");
        assert_eq!(dead_end.num_set(), 0, "the dead-end mask allows no token");
        assert!(dead_end.len() != 0, "it is NOT the zero-length case");
        // The guard fires exactly on (len > 0 && num_set == 0).
        assert!(dead_end.len() > 0 && dead_end.num_set() == 0, "the dead-end predicate holds");

        // A normal gated mask: allows some tokens (the guard must NOT fire).
        let mut gated = SimpleVob::alloc(64);
        gated.allow_token(7);
        gated.allow_token(42);
        assert_eq!(gated.num_set(), 2, "the gated mask allows 2 tokens");
        assert!(
            !(gated.len() > 0 && gated.num_set() == 0),
            "the guard does NOT fire for a gated mask (it has set entries)"
        );

        // A zero-length mask: the distinct, already-handled case (the guard is
        // moot because len == 0 is checked first).
        let empty = SimpleVob::new();
        assert_eq!(empty.len(), 0, "the zero-length mask has len 0");
    }

/// The §2.3 VOB data-width assertion: the VOB bitset is 8× less data than the
    /// F32 allow-mask. For a vocab of 4096, the F32 mask is 4096 × 4 bytes = 16384
    /// bytes; the VOB bitset is 4096/32 = 128 words × 4 bytes = 512 bytes. The F32
    /// mask is 16384/512 = 32× the VOB total. Per token: the F32 mask is 4 bytes/token,
    /// the VOB is 4 bits/token (0.5 bytes/token), so the VOB is 8× less data per token.
    #[test]
    fn test_vob_is_8x_less_data_per_token() {
        let vocab_size = 4096;
        // The F32 mask total: the vocab_size × 4 bytes.
        let f32_mask_bytes = vocab_size * 4;
        // The VOB bitset total: the (vocab_size + 31) / 32 words × 4 bytes per word.
        let vob_words = (vocab_size + 31) / 32;
        let vob_bytes = vob_words * 4;
        // The F32 mask is 32× the VOB total (the 4 bytes/token vs the 0.125 bytes/token).
        assert_eq!(f32_mask_bytes, vob_bytes * 32, "the F32 mask is 32× the VOB bitset total");
        // Per token: the F32 mask is 4 bytes/token (32 bits), the VOB is 4 bits/token.
        // The VOB is 8× less data per token (32 bits / 4 bits = 8).
        let f32_per_token_bits = 32;
        let vob_per_token_bits = 4;
        assert_eq!(f32_per_token_bits, vob_per_token_bits * 8, "the VOB is 8× less data per token");
    }

    /// PROOF of the sequential gating contract: the token stream is serialized
    /// through the FSM in acquisition order. The base is committed, the ff-run is
    /// read from the post-base settled state, and the ff-run is committed — so the
    /// ordered queue [base, ff…] is appended in the the order the FSM accepted
    /// it. A future ff-after-drafted path cannot desync because the ff-read is a
    /// pure observation of the settled state that already contains the base.
    #[test]
    fn test_sequential_gating_ordered_queue() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;
        use candle_core::{Device, DType, Tensor};

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"start: "abc""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        let gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let seq_id = 0;
        let requests = vec![GuidedDecodingRequest {
            seq_id,
            grammar: Some(&grm),
            reasoning_start_ids: &[],
            reasoning_end_ids: &[],
            tool_call_start_ids: &[],
            tool_call_end_ids: &[],
            newline_tokens: &[],
            reasoning_budget: None,
        }];
        let vocab = env.tok_trie().vocab_size();
        let logits = Tensor::zeros(
            (1, vocab),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let _ = gd.apply(&logits, &requests).unwrap();

        // the base token ('a' = 97) is committed first
        let base = vec![97u32];
        let n_base = gd.commit_run(seq_id, &base);
        assert_eq!(n_base, 1, "the base is committed");

        // the ff-run is read from the post-base state
        let ff = gd.ff_tokens(seq_id);
        assert_eq!(ff, vec![98, 99], "the ff-run is 'b','c' (the post-base state)");

        // the ff-run is committed
        let n_ff = gd.commit_run(seq_id, &ff);
        assert_eq!(n_ff, 2, "the ff-run is committed");

        // the ordered queue [base, ff…] is exactly what the FSM accepted, in order
        let accepted: Vec<u32> = base.iter().chain(ff.iter()).copied().collect();
        assert_eq!(accepted, vec![97, 98, 99], "the ordered queue is in acquisition order");
    }

    /// Benchmark: measure the legacy validate_tokens time (the O(grammar) cost)
    /// for a long sequence.
    #[test]
    fn bench_validate_tokens_legacy_baseline() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::toktrie::ApproximateTokEnv;
        use crate::utils::guidance::ParserFactory;
        use candle_core::{Device, DType, Tensor};

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
let grm_str = r#"start: "a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "s" "t" "u" "v" "w" "x" "y" "z""#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());

        let gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let seq_id = 0;
        let requests = vec![GuidedDecodingRequest {
            seq_id,
            grammar: Some(&grm),
            reasoning_start_ids: &[],
            reasoning_end_ids: &[],
            tool_call_start_ids: &[],
            tool_call_end_ids: &[],
            newline_tokens: &[],
            reasoning_budget: None,
        }];
        let vocab = env.tok_trie().vocab_size();
        let logits = Tensor::zeros((1, vocab), DType::F32, &Device::Cpu).unwrap();
        let _ = gd.apply(&logits, &requests).unwrap();

        // Walk 20 tokens (the 'a' through 't').
        let tokens: Vec<u32> = (97..117).collect(); // 'a' to 'u'
        let mut committed = 0;
        for &tok in &tokens {
            let n = gd.commit_run(seq_id, &[tok]);
            committed += n;
        }
        assert_eq!(committed, tokens.len(), "all tokens committed");

        // Measure the validate_tokens time (the O(grammar) speculative walk).
        // Validate the remaining 6 tokens (the 'v' to 'z', the grammar tail).
        let draft: Vec<u32> = (118..123).collect(); // 'v' to 'z' (5 tokens, the grammar tail)
        let t0 = std::time::Instant::now();
        let n_valid = gd.validate_tokens(seq_id, &draft).unwrap_or(0);
        let elapsed_us = t0.elapsed().as_micros();
        eprintln!(
            "Legacy validate_tokens: {} tokens in {} us ({} us/token), valid={}",
            draft.len(), elapsed_us, (elapsed_us / draft.len() as u128), n_valid
        );
        assert!(n_valid <= draft.len(), "the valid prefix cannot exceed the draft length");
    }

    /// PROOF of the seal (the RC2 fix): the single `commit_run` (the gate) truncates
    /// a sampled run to the FSM-legal prefix, so an illegal token never reaches the
    /// Sequence. The inclusive: a fully-legal run commits in full (the prefix == the
    /// run). The exclusive: a run containing an illegal token commits only the legal
    /// prefix (the count < the run length), and the resync leaves the FSM at the legal
    /// boundary (the next step restarts from there, the constraints stay active).
    #[test]
    fn gate_commit_seals_the_sequence_to_the_legal_prefix() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;
        use candle_core::{Device, DType, Tensor};

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // The grammar: 'a' then 'b' then 'c' (the tokens 97, 98, 99).
        let grm_str = r#"start: <[97]> <[98]> <[99]>"#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        let gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let seq_id = 0usize;
        let requests = vec![GuidedDecodingRequest {
            seq_id,
            grammar: Some(&grm),
            reasoning_start_ids: &[],
            reasoning_end_ids: &[],
            tool_call_start_ids: &[],
            tool_call_end_ids: &[],
            newline_tokens: &[],
            reasoning_budget: None,
        }];
        // Initialize the guidance state (the apply creates it).
        let vocab = env.tok_trie().vocab_size();
        let logits = Tensor::zeros(
            (1, vocab),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let _ = gd.apply(&logits, &requests).unwrap();

        // The inclusive: a fully-legal run ('a','b','c') commits in full.
        let n_full = gd.commit_run(seq_id, &[97, 98, 99]);
        assert_eq!(n_full, 3, "a fully-legal run commits all 3 (the seal passes)");

        // The exclusive: a fresh state, a run with an illegal token ('a' then 'c' —
        // the grammar expects 'b' after 'a'). The commit_run must stop at the legal
        // prefix ('a' only, the count 1), NOT commit the illegal 'c'.
        let seq_id2 = 1usize;
        let requests2 = vec![GuidedDecodingRequest {
            seq_id: seq_id2,
            grammar: Some(&grm),
            reasoning_start_ids: &[],
            reasoning_end_ids: &[],
            tool_call_start_ids: &[],
            tool_call_end_ids: &[],
            newline_tokens: &[],
            reasoning_budget: None,
        }];
        let _ = gd.apply(&logits, &requests2).unwrap();
        let n_partial = gd.commit_run(seq_id2, &[97, 99]); // 'a' legal, 'c' illegal (expects 'b')
        assert_eq!(n_partial, 1, "the seal truncates to the legal prefix (the 'a' only)");
        // The 'c' was NOT committed (the llm_tokens hold only the 'a').
        // The resync rolls the matcher back to the committed boundary (the 'a'),
        // so the constraints stay active (not a disable-on-error).
        gd.resync(seq_id2);
    }

    /// Regression for the production crash (complete-failure.log): a fresh sequence
    /// with an active grammar hitting build_vob_words before any guidance state
    /// exists. The VOB builder must CREATE the state on first use (the lazy
    /// creation, matching build_allow_mask / build_vob_words_flat) rather than
    /// panic on the missing state. Before the fix, build_vob_words called
    /// states.get_mut and panicked "[R5] no guidance state for an active grammar"
    /// on the very first mask request for a new sequence.
    #[test]
    fn build_vob_words_creates_state_on_first_use() {
        use llguidance::{api::TopLevelGrammar, ParserFactory};
        use toktrie::ApproximateTokEnv;
        use candle_core::{Device, DType, Tensor};

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // A reasoning-style grammar (the shape from the production crash log): a
        // forced block then free text then EOS.
        let grm_str = r#"start: "a" "b" "c""#;
        let mut grm = TopLevelGrammar::from_lark(grm_str.to_string());
        grm.max_tokens = None;

        // A FRESH GuidedDecoding (no prior apply / commit, so no state exists yet).
        let gd = GuidedDecoding::new(Some(std::sync::Arc::new(factory)), true);
        let seq_id = 0usize;
        let requests = vec![GuidedDecodingRequest {
            seq_id,
            grammar: Some(&grm),
            reasoning_start_ids: &[],
            reasoning_end_ids: &[],
            tool_call_start_ids: &[],
            tool_call_end_ids: &[],
            newline_tokens: &[],
            reasoning_budget: None,
        }];
        let vocab = env.tok_trie().vocab_size();
        let logits = Tensor::zeros((1, vocab), DType::F32, &Device::Cpu).unwrap();

        // The crash scenario: build_vob_words on a fresh sequence with an active
        // grammar. Before the fix this panicked (no state); after the fix it
        // creates the state and returns a valid VOB.
        let vob = gd.build_vob_words(&requests, vocab, &Device::Cpu).expect("build_vob_words");
        let vob = vob.expect("a grammar-active request produces a VOB");
        // The VOB is the right shape (ceil(vocab/32) words for the 1 row).
        let words_per_vob = (vocab + 31) / 32;
        assert_eq!(vob.dims(), &[1, words_per_vob], "the VOB is [1 row, ceil(vocab/32) words]");
        // The first mask (at the start of "a" "b" "c") allows exactly 'a' (97).
        let flat: Vec<u32> = vob.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let allowed: Vec<u32> = (0..vocab as u32)
            .filter(|&t| flat[t as usize / 32] & (1u32 << (t as usize % 32)) != 0)
            .collect();
        assert_eq!(allowed, vec![97], "at the start of 'a' 'b' 'c' only 'a' (97) is allowed");
    }

    // The firewall re-decode bound (the no infinite stall on a terminally-stuck FSM):
    // a mask failure bumps the per-seq streak; committed progress clears it. The engine
    // applies a MODEL-DEPENDENT bound (1 for hybrid-mamba, the no the mamba-state
    // desync on re-decode; 32 for paged-attention, the idempotent KV). This test proves
    // the streak mechanics (the no a real mask failure, which needs a stuck Earley state).
    #[test]
    fn mask_fail_streak_bound_contract() {
        let gd = GuidedDecoding::new(None, true);
        // The no streak yet (the untracked seq).
        assert_eq!(gd.mask_fail_streak(99), 0);
        // The paged-attention bound constant (the re-decode limit, the no a magic number
        // scattered). The hybrid-mamba bound is 1 (the engine's model-dependent decision,
        // the no a re-decode loop that corrupts the in-place recurrent state).
        assert_eq!(GuidedDecoding::MAX_MASK_FAIL_STREAK, 32, "the paged-attention re-decode bound is 32 consecutive mask failures");
    }
}
