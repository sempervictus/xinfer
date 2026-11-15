// src/utils/guidance.rs
// This module contains non-grammar guidance utilities:
// - GuidanceTokens: token ID collections
// - ParserFactory: llguidance parser factory
// - GuidanceState: matcher state for guided decoding

use crate::utils::config::TokenizerConfig;
use crate::utils::env::{llg_full_enabled, max_reasoning_tokens};
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
    /// The token id(s) for a newline (`"\n"`), used to build the reasoning
    /// terminator run (`newline + EOR + newline`) that the gate injects when
    /// the reasoning budget is exhausted.
    pub newline_tokens: Vec<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

impl GuidanceTokens {
    /// Compress a sorted list of token IDs into ranges.
    /// E.g., [1, 2, 3, 5, 7, 8] -> [(1, 3), (5, 5), (7, 8)]
    fn compress_to_ranges(ids: &[u32]) -> Vec<(u32, u32)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let mut ranges = Vec::new();
        let mut start = ids[0];
        let mut prev = ids[0];
        for &id in &ids[1..] {
            if id == prev + 1 {
                prev = id;
            } else {
                ranges.push((start, prev));
                start = id;
                prev = id;
            }
        }
        ranges.push((start, prev));
        ranges
    }

    /// Generate a token-range expression for free text generation.
    /// Returns the RHS expression usable directly after `text: ` in Lark grammar.
    /// Uses llguidance's negated token range syntax to allow all tokens EXCEPT the
    /// excluded set (the ~20 control tokens), a 12,000x smaller mask than the
    /// allowed set. Empty exclusions fall back to the full regex.
    pub fn token_range_expression(excluded_ids: Vec<u32>) -> String {
        if excluded_ids.is_empty() {
            return r#"/(?s:.*)/"#.to_string();
        }
        let mut sorted_ids: Vec<u32> = excluded_ids;
        sorted_ids.sort();
        sorted_ids.dedup();
        let ranges = Self::compress_to_ranges(&sorted_ids);
        format!(
            "(<[^{}]>)+",
            ranges
                .iter()
                .map(|(start, end)| {
                    if start == end {
                        start.to_string()
                    } else {
                        format!("{}-{}", start, end)
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        )
        .trim()
        .to_string()
    }

    // Disallow all control tokens, used in the middle of grammars
    pub fn text_grammar_mask(&self) -> String {
        let mut ids = Vec::new();
        ids.extend_from_slice(&self.bos_token_ids);
        ids.extend_from_slice(&self.eos_token_ids);
        ids.extend_from_slice(&self.reasoning_start_ids);
        ids.extend_from_slice(&self.reasoning_end_ids);
        ids.extend_from_slice(&self.tool_call_start_ids);
        ids.extend_from_slice(&self.tool_call_end_ids);
        Self::token_range_expression(ids)
    }

    // Construct reasoning mask rule relative to how model must generate (BOS+ or prepended)
    pub fn reasoning_grammar_mask(&self) -> String {
        let mut range_exp = self.text_grammar_mask();
        if self.add_bos_token {
            range_exp = format!(
                r#"({}) {}"#,
                self.reasoning_start_ids
                    .iter()
                    .map(|&n| format!("<[{}]>", n))
                    .collect::<Vec<String>>()
                    .join(" | "),
                &range_exp
            );
        }
        format!(
            r#"{} ({})"#,
            &range_exp,
            self.reasoning_end_ids
                .iter()
                .map(|&n| format!("<[{}]>", n))
                .collect::<Vec<String>>()
                .join(" | ")
        )
    }

    /// Get the excluded token IDs for the free-text rule (the BOS + the reasoning + the
    /// tool-call IDs, NOT the EOS). Used by the ToolCallGrammar to build the text rule
    /// via token_range_expression (the token-number masking).
    pub fn get_text_excluded_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        ids.extend_from_slice(&self.bos_token_ids);
        ids.extend_from_slice(&self.reasoning_start_ids);
        ids.extend_from_slice(&self.reasoning_end_ids);
        ids.extend_from_slice(&self.tool_call_start_ids);
        ids.extend_from_slice(&self.tool_call_end_ids);
        ids
    }
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
    let add_eos_token = tokenizer_config.add_eos_token == Some(true);

    // The newline token id(s): how the tokenizer encodes `"\n"`. Used to build
    // the reasoning terminator run (`newline + EOR + newline`). A BPE tokenizer
    // usually encodes `"\n"` as a single token; if it splits into several, all
    // are captured so the injection reproduces the exact byte sequence.
    let newline_tokens: Vec<u32> = tokenizer
        .encode("\n", false)
        .map(|enc| enc.get_ids().to_vec())
        .unwrap_or_default();

    GuidanceTokens {
        bos_token_ids: validated_bos,
        eos_token_ids: validated_eos,
        reasoning_start_ids: special_tokens.reasoning_start_ids(),
        reasoning_end_ids: special_tokens.reasoning_end_ids(),
        tool_call_start_ids: special_tokens.tool_call_start_ids(),
        tool_call_end_ids: special_tokens.tool_call_end_ids(),
        newline_tokens,
        add_bos_token,
        add_eos_token,
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

/// The conversation output region, tracked by the `GuidanceState` as tokens are
/// committed through the gate. Mirrors the server parser's reasoning/tool phase
/// tracking (the `StreamToolParser::in_reasoning`, server/parser.rs) but on token
/// ids instead of decoded text.
///
/// Transitions (driven by the marker token ids on the `GuidanceTokens`):
/// - `Reasoning -> Text` on a `reasoning_end_ids` token (the EOR).
/// - `Text -> ToolCall` on a `tool_call_start_ids` token (the BOT).
/// - `ToolCall -> Text` on a `tool_call_end_ids` token (the EOT).
///
/// The initial phase is set by the constructor: a reasoning grammar (the
/// `is_reasoning_grammar`) starts in `Reasoning`; otherwise it starts in `Text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputPhase {
    Reasoning,
    Text,
    ToolCall,
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
    /// Full-envelope mode: grammar constrains from BOS to EOS including reasoning.
    full_envelope: bool,
    /// The index into `llm_tokens` of the last token successfully committed to the matcher
    /// (the reject-and-resync boundary, the P2 fix). Tokens at or after this index are uncommitted.
    last_committed: usize,
    /// The count of reasoning tokens admitted so far (the `Reasoning` phase only).
    /// Frozen once the phase leaves `Reasoning` (the EOR seen), so the budget
    /// measures reasoning length, not total generation length.
    reasoning_token_count: usize,
    /// Token IDs that mark the start of a tool call (the BOT).
    tool_call_start_ids: Vec<u32>,
    /// Token IDs that mark the end of a tool call (the EOT).
    tool_call_end_ids: Vec<u32>,
    /// The per-request reasoning token budget (the `SamplingParams.reasoning_budget`).
    /// When `Some`, it overrides the `XINFER_MAX_REASONING_TOKENS` env for this
    /// sequence's budget-exhaustion check. The `None` means use the env.
    reasoning_budget: Option<usize>,
    /// The current conversation output region (the `Reasoning` / `Text` /
    /// `ToolCall`). Updated on every committed token via the marker ids above.
    phase: OutputPhase,
    /// The rank-0 log gate (the no the 4x phase-transition log duplication on a
    /// multi-GPU TP run, where each rank has its own GuidanceState + the NCCL
    /// all-reduce makes them agree, so the rank-0 log is the representative one).
    /// Set from the constructor's `log_rank0` param (the prior code used it only for
    /// the construction logs, the no the transition logs).
    log_rank0: bool,
    /// The per-phase token counts (the `OutputPhase` discriminant index: 0=Reasoning,
    /// 1=Text, 2=ToolCall). Incremented on every committed token (the `advance_phase`),
    /// so the seq-level Decoded summary can show how many tokens were decoded in each
    /// region. The `reasoning_token_count` (the budget counter) is a separate, frozen
    /// count; this is the live per-region tally.
    phase_token_counts: [usize; 3],
}

impl GuidanceState {
    /// An independent copy of this state (the `deep_clone` matcher, the PQ-2
    /// proven-invariant scratch FSM). The `llm_tokens` + the phase/budget flags are
    /// copied so the clone is a faithful shadow of the committed state at this
    /// moment. Used by the draft-side MaskStream (the speculative/mask_stream.rs)
    /// to walk the grammar trajectory WITHOUT advancing the committed state.
    pub fn deep_clone_state(&self) -> Self {
        Self {
            matcher: self.matcher.deep_clone(),
            llm_tokens: self.llm_tokens.clone(),
            reasoning_end_ids: self.reasoning_end_ids.clone(),
            full_envelope: self.full_envelope,
            last_committed: self.last_committed,
            reasoning_token_count: self.reasoning_token_count,
            tool_call_start_ids: self.tool_call_start_ids.clone(),
            tool_call_end_ids: self.tool_call_end_ids.clone(),
            reasoning_budget: self.reasoning_budget,
            phase: self.phase,
            log_rank0: self.log_rank0,
            phase_token_counts: self.phase_token_counts,
        }
    }

    pub fn new_from_grammar_with_markers(
        factory: Arc<ParserFactory>,
        grammar: &TopLevelGrammar,
        reasoning_end_ids: Vec<u32>,
        tool_call_start_ids: Vec<u32>,
        tool_call_end_ids: Vec<u32>,
        reasoning_budget: Option<usize>,
        log_rank0: bool,
    ) -> Result<Self> {
        use crate::utils::guidance_grammar::{get_lark_from_top_level_grammar, is_reasoning_grammar};

        if tracing::enabled!(tracing::Level::DEBUG) {
            let lark = get_lark_from_top_level_grammar(grammar);
            tracing::debug!(
                "[llg] Initializing guidance parser from grammar: {} bytes, {} lines",
                lark.len(),
                lark.lines().count()
            );
            tracing::trace!("[llg] Guidance parser grammar:\n{}\n", lark);
        }

        // In full-envelope mode OR llg_full_enabled, reasoning is constrained from the start
        let llg_full_enabled = llg_full_enabled();
        let reasoning_ended = if llg_full_enabled {
            true  // Single-phase: grammar always applies
        } else {
            reasoning_end_ids.is_empty()
        };

        // The initial output phase: a reasoning grammar (the `is_reasoning_grammar`)
        // starts in `Reasoning` (until the EOR is seen); otherwise it starts in
        // `Text` (until a BOT opens a tool call). This holds in BOTH the single-phase
        // (full-envelope) + the two-phase modes — the phase machine is independent of
        // `reasoning_ended`.
        let phase = if is_reasoning_grammar(grammar) {
            OutputPhase::Reasoning
        } else {
            OutputPhase::Text
        };
        if log_rank0 {
            crate::log_info!(
                "[llg] initial output phase: {:?} (reasoning grammar: {}, full-envelope: {})",
                phase,
                is_reasoning_grammar(grammar),
                llg_full_enabled
            );
        }

        // The parser is created fresh per GuidanceState (the no cache, the no
        // deep_clone). The deep_clone corrupted the parser's internal Earley state
        // (the item-set + the stack), so the full-envelope (the
        // XINFER_LLG_FULL) grammar got stuck in the reasoning_block phase and
        // could never reach the eos rule. The fresh parser (the
        // factory.create_parser) is correct; the performance cost of
        // re-instantiating the FSM per sequence is accepted for correctness.
        let mut parser = if !reasoning_ended {
            if log_rank0 {
                crate::log_info!(
                    "[llg] Two-phase reasoning: grammar constraint deferred until after reasoning end tokens {:?}",
                    reasoning_end_ids
                );
            }
            factory.create_parser(grammar.clone())?
        } else {
            if log_rank0 {
                crate::log_info!(
                    "[llg] Full-envelope/single-phase mode: grammar constrains all generation"
                );
            }
            // Max tokens is capped by the scheduler anyway so allow grammar space to generate reasoning
            if let Some(max_tokens) = grammar.max_tokens {
                let mut grammar = grammar.clone();
                grammar.max_tokens = Some(max_tokens * 2);
                factory.create_parser(grammar)?
            } else {
                factory.create_parser(grammar.clone())?
            }
        };
        parser.start_without_prompt();
        // The GuidanceState tracks the grammar via the Matcher (the Earley parser +
        // lexer), which is the single source of truth for the allowed-token mask.
        let matcher = Matcher::new(Ok(parser));

        Ok(Self {
            matcher,
            llm_tokens: Vec::new(),
            reasoning_end_ids,
            tool_call_start_ids,
            tool_call_end_ids,
            reasoning_budget,
            phase,
            full_envelope: llg_full_enabled,
            last_committed: 0,
            reasoning_token_count: 0,
            log_rank0,
            phase_token_counts: [0; 3],
        })
    }

    /// Commit token and track for speculative decoding recovery.
    /// During reasoning, tokens are tracked but NOT fed to the grammar.
    /// When the reasoning-end token is seen, we transition to grammar mode.
    pub fn commit_token(&mut self, token: u32) -> Result<()> {
        self.llm_tokens.push(token);
        // Advance the output-phase machine on the committed token FIRST, then count.
        // The count is "reasoning content tokens" -- it increments only while the
        // phase is STILL Reasoning AFTER the token is processed. The EOR token (the
        // one that flips Reasoning -> Text) is the phase transition, not reasoning
        // content, so it is NOT counted (the no the off-by-one where budget=64 allowed
        // 64 content tokens + the EOR = 65 counted).
        self.advance_phase(token);
        if self.phase == OutputPhase::Reasoning {
            self.reasoning_token_count += 1;
        }

        // Two-phase mode: while still in the `Reasoning` phase (and not full-envelope),
        // tokens are tracked but NOT fed to the grammar. The phase machine (the
        // `advance_phase` above) already moved to `Text` on the EOR, so no
        // separate `reasoning_ended` flag is needed.
        if self.grammar_deferred() {
            if self.reasoning_end_ids.contains(&token) {
                crate::log_warn!(
                    "[llg] Reasoning ended (token {}), grammar constraints now active (after {} reasoning tokens)",
                    token,
                    self.reasoning_token_count
                );
            }
            return Ok(());
        }

        if !self.matcher.is_stopped() {
            // The Matcher's Earley parser is the single grammar-state authority;
            // consuming a token advances it directly.
            self.matcher.consume_token(token)?;
            self.last_committed = self.llm_tokens.len(); // the committed boundary advances
        }
        // The desync detector: if admitted tokens != FSM-consumed tokens while still
        // in the `Reasoning` phase, the gate let through a token the FSM rejected.
        self.warn_reasoning_desync();
        Ok(())
    }

    /// The reject-and-resync (the P2 fix): on a FSM failure (the token->byte bridge desync), roll
    /// the matcher + the llm_tokens back to the last committed token boundary (the no disable-
    /// constraints). The constraints stay active; the next decode step restarts from the last
    /// accepted token.
    pub fn resync(&mut self) {
        let rollback_count = self.llm_tokens.len().saturating_sub(self.last_committed);
        if rollback_count > 0 {
            let _ = self.matcher.rollback(rollback_count);
            self.llm_tokens.truncate(self.last_committed);
        }
    }

    /// Roll the matcher + the bookkeeping back to an explicit `llm_tokens` boundary
    /// (the no the `resync` to `last_committed`). Used by the gate's ff rollback: when
    /// the KV forward for the ff span fails, the matcher was already advanced by
    /// `commit_ff_run` (the `consume_ff_tokens`), so it is rolled back to the pre-ff
    /// boundary (captured before the ff commit) to hold the seq's position.
    pub fn resync_to(&mut self, boundary: usize) {
        let rollback_count = self.llm_tokens.len().saturating_sub(boundary);
        if rollback_count > 0 {
            let _ = self.matcher.rollback(rollback_count);
            self.llm_tokens.truncate(boundary);
            self.last_committed = boundary;
        }
    }

    /// The current `llm_tokens` length (the pre-ff boundary, the rollback target for
    /// the gate's ff KV-failure rollback).
    pub fn llm_tokens_len(&self) -> usize {
        self.llm_tokens.len()
    }

    /// Check if guidance is finished
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }

    /// The count of reasoning tokens admitted so far. Runs in both phases: the
    /// two-phase mode counts tokens until the reasoning-end flip, the single-phase
    /// (full-envelope) mode counts every generated token (the whole generation is
    /// the reasoning region). Used by the reasoning-budget sampling bias (the
    /// `apply_reasoning_bias` in the runner) to steer the end-of-reasoning token.
    pub fn reasoning_token_count(&self) -> usize {
        self.reasoning_token_count
    }

    /// The per-phase token counts (the `[reasoning, text, tool_call]` tally, the
    /// Decoded summary's per-region breakdown). The `reasoning` count is the live
    /// per-region tally (distinct from the frozen `reasoning_token_count` budget
    /// counter, which stops growing once the phase leaves Reasoning).
    pub fn phase_token_counts(&self) -> [usize; 3] {
        self.phase_token_counts
    }

    /// The per-request reasoning budget (the `SamplingParams.reasoning_budget`),
    /// or `None` when the sequence should use the `XINFER_MAX_REASONING_TOKENS` env.
    pub fn reasoning_budget(&self) -> Option<usize> {
        self.reasoning_budget
    }

    /// The effective reasoning budget (the single definition, the no a duplicated
    /// expression): the per-request `reasoning_budget` wins, the
    /// `XINFER_MAX_REASONING_TOKENS` env is the fallback. Used by BOTH the
    /// `compute_mask_or_eos` hard-forcing (the enforcement) and the phase-transition
    /// log (the display), so the two can never diverge.
    pub fn effective_reasoning_budget(&self) -> Option<usize> {
        self.reasoning_budget().or(max_reasoning_tokens())
    }

    /// The reasoning terminator run: the end-of-reasoning token (the EOR, the
    /// `reasoning_end_ids[0]`). This is the single token that closes the
    /// reasoning region in every grammar form (the `Reasoning -> Text` phase
    /// transition is driven by the EOR, the `advance_phase`). Returns an empty
    /// run when there is no reasoning-end token (nothing to terminate). The gate
    /// commits this run atomically (one `commit_run`), so no sampled token lands
    /// between its parts.
    ///
    /// The newlines around the EOR (the `"\n<EOR>\n"` chat-template shape) are
    /// the model's own output (the reasoning text ends with a newline naturally),
    /// not a forced token — so the terminator is just the EOR. Forcing the
    /// newlines would risk the FSM rejecting them (the `commit_run` stops at the
    /// first rejection, which would block the EOR itself).
    pub fn reasoning_terminator(&self) -> Vec<u32> {
        let Some(&eor) = self.reasoning_end_ids.first() else {
            return Vec::new();
        };
        vec![eor]
    }

    /// Compute mask or return EOS token set if stopped.
    /// In full-envelope mode, always apply the grammar mask (no all-ones during reasoning).
    ///
    /// The reasoning-budget HARD BOUND (the firewall, the no soft bias): when the
    /// reasoning count has reached the budget, the mask is intersected with the EOR
    /// singleton (the `reasoning_end_ids`) so the model CANNOT emit another reasoning
    /// token -- the region is forced to close. This is a prevention (the illegal
    /// over-budget tokens are masked out), not a steering (the prior +10 bias let the
    /// model reason past the budget). Applies in both the full-envelope + the two-phase
    /// modes (the budget is a hard cap regardless of the grammar deferral).
    pub fn compute_mask_or_eos(&mut self) -> Result<SimpleVob> {
        let mask = if self.full_envelope {
            self.matcher.compute_mask_or_eos()?
        } else if self.grammar_deferred() {
            // Two-phase mode: allow everything during reasoning (the grammar is deferred).
            let mut m = self.matcher.compute_mask_or_eos()?;
            m.set_all(true);
            m
        } else {
            self.matcher.compute_mask_or_eos()?
        };

        // The budget hard-bound: force the EOR when the reasoning count has reached the
        // budget. The EOR must be grammar-legal at this position (the mask already
        // reflects the grammar); we intersect with the EOR singleton so only the EOR
        // (or the EOS, if the grammar is otherwise complete) survives. If the EOR is not
        // in the current mask (the grammar is past the reasoning block already), the
        // intersection is a no-op (the budget is moot, the region already closed).
        // The effective budget: the per-request budget (the `SamplingParams.reasoning_budget`)
        // wins; the `XINFER_MAX_REASONING_TOKENS` env is the fallback (the no the env
        // being ignored when only the env is set). The hard EOR-forcing fires at the
        // effective budget.
        let budget = self.effective_reasoning_budget();
        if let Some(budget) = budget {
            if self.reasoning_token_count >= budget && !self.reasoning_end_ids.is_empty() {
                let eor = self.reasoning_end_ids[0];
                if mask.is_allowed(eor) {
                    let mut forced = mask.clone();
                    // The budget hard-bound: only the EOR survives (the region is forced
                    // to close). The grammar mask already encodes the EOS handling (the
                    // `compute_mask_or_eos` returns the EOS set when the grammar is
                    // complete), so we do NOT add a separate EOS here -- intersecting with
                    // the EOR singleton is sufficient + correct.
                    forced.set_all(false);
                    forced.allow_token(eor);
                    return Ok(forced);
                }
            }
        }
        Ok(mask)
    }

    /// Fast-forward tokens without consuming them (for speculative decoding).
    /// During reasoning, no fast-forward is possible.
    pub fn compute_ff_tokens(&mut self) -> Vec<u32> {
        if self.grammar_deferred() {
            return Vec::new();
        }
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        self.matcher.compute_ff_tokens()
    }

    /// The ff run at the current state (the grammar-forced continuation). origin/main
    /// llguidance has no non-advancing ff read; the ff is computed by the advancing
    /// compute_ff_tokens, and the sealed gate commits the base + ff as one run via
    /// try_consume_tokens, so the advancing read is correct here.
    pub fn ff_run(&mut self) -> Vec<u32> {
        if self.grammar_deferred() {
            return Vec::new();
        }
        if self.matcher.is_stopped() {
            return Vec::new();
        }
        self.matcher.compute_ff_tokens()
    }

    /// Advance the output-phase machine on a committed token (the marker-driven
    /// transitions). Called from `commit_token`/`commit_run` for each accepted
    /// token, so the phase tracks the committed stream (not the sampled-but-
    /// rejected tokens).
    fn advance_phase(&mut self, token: u32) {
        // The per-phase token tally (the Decoded summary's per-region counts): the
        // token is committed INTO the current phase (before any transition), so count
        // it against the current phase's index (Reasoning=0, Text=1, ToolCall=2).
        let phase_idx = match self.phase {
            OutputPhase::Reasoning => 0,
            OutputPhase::Text => 1,
            OutputPhase::ToolCall => 2,
        };
        self.phase_token_counts[phase_idx] += 1;
        // The phase-transition logs are rank-0-gated (the no the 4x duplication on a
        // multi-GPU TP run, where each rank has its own GuidanceState + the NCCL
        // all-reduce makes them agree, so the rank-0 log is the representative one).
        // The transition itself (the `phase` flip) is NOT gated -- it must happen on
        // every rank (the phase machine is per-rank, the no a rank-0-only state).
        match self.phase {
            OutputPhase::Reasoning => {
                if self.reasoning_end_ids.contains(&token) {
                    self.phase = OutputPhase::Text;
                    if self.log_rank0 {
                        crate::log_info!(
                            "[llg] reasoning phase ended (token {}), reasoning length {} (budget {:?})",
                            token,
                            self.reasoning_token_count,
                            self.effective_reasoning_budget()
                        );
                    }
                }
            }
            OutputPhase::Text => {
                if self.tool_call_start_ids.contains(&token) {
                    self.phase = OutputPhase::ToolCall;
                    if self.log_rank0 {
                        crate::log_info!("[llg] tool-call phase started (token {})", token);
                    }
                }
            }
            OutputPhase::ToolCall => {
                if self.tool_call_end_ids.contains(&token) {
                    self.phase = OutputPhase::Text;
                    if self.log_rank0 {
                        crate::log_info!("[llg] tool-call phase ended (token {})", token);
                    }
                }
            }
        }
    }

    /// Whether the grammar constraint is currently deferred: the two-phase mode
    /// (not full-envelope) AND the phase machine is still in `Reasoning`. In this
    /// region the mask is allow-all, fast-forward is skipped, and tokens are not
    /// fed to the matcher. In full-envelope mode the grammar is always active, so
    /// this is false. This is the single policy predicate; deferred-region
    /// check keys off, so the `!full_envelope && phase == Reasoning` test is
    /// written here here instead of at each call site.
    fn grammar_deferred(&self) -> bool {
        !self.full_envelope && self.phase == OutputPhase::Reasoning
    }

    /// The current conversation output region (the `Reasoning` / `Text` / `ToolCall`).
    pub fn phase(&self) -> OutputPhase {
        self.phase
    }

    /// The desync detector (the WARN you asked for): if the number of tokens
    /// admitted (`llm_tokens`) differs from the number the FSM actually consumed
    /// (`last_committed`) while still in the `Reasoning` phase (before the EOR),
    /// the gate is letting through tokens the FSM rejected — log it.
    fn warn_reasoning_desync(&self) {
        if self.phase == OutputPhase::Reasoning && self.llm_tokens.len() != self.last_committed {
            crate::log_warn!(
                "[llg] reasoning desync — {} tokens admitted but only {} consumed by the FSM (phase={:?})",
                self.llm_tokens.len(),
                self.last_committed,
                self.phase
            );
        }
    }

    /// Commit a run of tokens to the grammar FSM. Returns the number of tokens
    /// successfully committed (the FSM-legal prefix). The tokens after the
    /// first rejection are not committed (the FSM stays at the legal boundary);
    /// the caller resyncs if it needs to recover. During reasoning (two-phase
    /// mode), tokens are tracked but not fed to the grammar, so the full run
    /// is always accepted.
pub fn commit_run(&mut self, run: &[u32]) -> usize {
        if self.grammar_deferred() {
            // Two-phase: reasoning tokens are unconstrained, accept all.
            for &token in run {
                self.llm_tokens.push(token);
                self.advance_phase(token);
                if self.phase == OutputPhase::Reasoning {
                    self.reasoning_token_count += 1;
                }
            }
            return run.len();
        }
        // Grammar active: commit each token to the matcher, counting at the first
        // rejection (the legal prefix is committed, the FSM is not). The
        // `last_committed` advances with each accepted token (the resync boundary),
        // so a partial commit_run + the resync rolls back to the legal prefix, not
        // the start. The `reasoning_token_count` increments only while in the
        // `Reasoning` phase (the budget measures reasoning length, not total
        // generation length).
        let mut committed = 0;
        for &token in run {
            if self.matcher.is_stopped() {
                break;
            }
            if self.matcher.consume_token(token).is_err() {
                break; // FSM rejected this token (the legal prefix is committed)
            }
            self.llm_tokens.push(token);
            self.advance_phase(token);
            if self.phase == OutputPhase::Reasoning {
                self.reasoning_token_count += 1;
            }
            self.last_committed = self.llm_tokens.len(); // the committed boundary advances
            committed += 1;
        }
        // The desync detector: if admitted tokens != FSM-consumed tokens while still
        // in the `Reasoning` phase, the gate let through a token the FSM rejected.
        self.warn_reasoning_desync();
        committed
    }

    /// Commit the grammar-forced ff run in ONE matcher advance (the no the double-advance
    /// desync). The prior gate path called `ff_tokens` (the `compute_ff_tokens`, advance #1)
    /// then `commit_run` (the `consume_token` per token, advance #2) -- two advances for one
    /// ff run, which desynced the parser from the committed stream (the tool-call header
    /// re-emitted into the value region). This uses `consume_ff_tokens` (the canonical
    /// compute-then-consume idiom, ONE advance) + updates the GuidanceState bookkeeping
    /// (the `llm_tokens`, the `reasoning_token_count`, the `phase`, the `last_committed`)
    /// so the state stays consistent with the matcher. Returns the committed ff count.
    pub fn commit_ff_run(&mut self) -> Vec<u32> {
        if self.grammar_deferred() || self.matcher.is_stopped() {
            return Vec::new(); // the no ff in the deferred (reasoning) phase or a stopped matcher
        }
        let ff = self.matcher.consume_ff_tokens(); // the single advance (the compute + the consume)
        if ff.is_empty() {
            return Vec::new();
        }
        // The bookkeeping: mirror `commit_run` for the ff tokens (the reasoning count +
        // the phase + the committed boundary). The ff tokens are grammar-forced (the
// infallible), so they are always committed in full (the no a partial prefix).
        for &token in &ff {
            self.llm_tokens.push(token);
            self.advance_phase(token);
            if self.phase == OutputPhase::Reasoning {
                self.reasoning_token_count += 1;
            }
            self.last_committed = self.llm_tokens.len();
        }
        ff
    }

    /// Non-mutating: how many of `tokens` are grammar-legal from the current
    /// state. Returns the count without advancing the matcher.
    pub fn validate_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        if self.grammar_deferred() {
            return Ok(tokens.len());
        }
        if self.matcher.is_stopped() {
            return Ok(0);
        }
        self.matcher.validate_tokens(tokens)
    }

    /// Build the per-verify-row grammar masks for a speculative verify step (the
    /// lossless rejection sampler's `p_target` must be the *masked* target
    /// distribution, not the raw full-vocab softmax). Returns the per-row
    /// allow-masks (the `SimpleVob` sets) so the caller can zero the illegal logits
    /// before the softmax.
    ///
    /// The construction is OPTIMISTIC: row `i`'s mask is the grammar state reached
    /// by projecting the committed matcher through `[draft_0..i-1]`. If a prior
    /// draft is grammar-invalid, the row is unreachable and is returned all-allowed
    /// (it will be discarded by the rejection sampler's accept/reject, so the
    /// optimistic mask is harmless). The committed matcher is NEVER advanced: every
    /// projection runs on a `deep_clone` (the PQ-2 invariant, the scratch is
    /// discarded). Row 0 is the committed state directly (no projection).
    ///
    /// `drafts` is the per-row draft token (the `draft_i` that follows row `i`); the
    /// mask for row `i` is computed from the state after committing `drafts[0..i]`.
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
        // the G3 invariant). The deferred (reasoning) phase yields all-allowed (the
        // two-phase mode, the grammar not yet active).
        let row0 = if self.grammar_deferred() {
            all_allowed()
        } else {
            self.matcher.deep_clone().compute_mask_or_eos().unwrap_or_else(|_| all_allowed())
        };
        rows.push(row0);
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

    #[test]
    fn test_extract_guidance_tokens() {
        // This test verifies that extract_guidance_tokens compiles
        // It doesn't actually run since we don't have a tokenizer here
        let tokens = GuidanceTokens::default();
        assert!(tokens.bos_token_ids.is_empty());
    }

    /// Build a simple explicit tool grammar for testing (the no real tokenizer needed).
    fn explicit_tool_grammar() -> (super::ParserFactory, TopLevelGrammar) {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // Simple grammar: token 97, then token 98, then token 99.
        let grm_str = r#"
start: <[97]> <[98]> <[99]>
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        (factory, grm)
    }

    /// The reject-and-resync: a desynced parser, after resync, rolls back to the last
    /// committed token boundary (the no disable-constraints). The next commit from the resynced
    /// state succeeds (the grammar-consistent token).
    #[test]
    fn resync_rolls_back_to_last_committed() {
        let (factory, grm) = explicit_tool_grammar();
        let mut state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(factory),
            &grm,
            vec![],
            vec![],
            vec![],
            None,
            true,
        )
        .unwrap();
        // The grammar: token 97, then 98, then 99 (the explicit_tool_grammar).
        state.commit_token(97).unwrap(); // the success -> the last_committed = 1
        // The out-of-order token 99 (the grammar expects 98) -> the commit fails.
        assert!(state.commit_token(99).is_err(), "the FSM rejects the out-of-order token 99");
        // The resync rolls back to the last committed boundary (the token 97).
        state.resync();
        // After the resync, the grammar-consistent token 98 succeeds.
        state.commit_token(98).unwrap();
        state.commit_token(99).unwrap();
        assert!(state.is_finished(), "the grammar is complete (the 97, 98, 99)");
    }

    /// Build a sequence grammar (the string-literal form). The
    /// single-byte env maps 'a'->97, 'b'->98, 'c'->99.
    fn sequence_grammar() -> (super::ParserFactory, TopLevelGrammar) {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        let grm_str = r#"
start: "a" "b" "c"
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        (factory, grm)
    }

    /// Build a raw CPU Matcher (the Earley authority) for the sequence
    /// grammar, independent of any GuidanceState claimant.
    fn sequence_matcher() -> llguidance::Matcher {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;
        let (_factory, grm) = sequence_grammar();
        let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
        let parser = factory.create_parser(grm).unwrap();
        llguidance::Matcher::new(Ok(parser))
    }

    /// PROOF (the compute_ff_tokens == the Matcher forced-run): the GuidanceState's
    /// fast-forward tokens must equal the CPU Matcher's compute-run at every point in
    /// the walk. Inclusive: every forced token the CPU reports is reported by the
    /// state. Exclusive: the state reports no token the CPU does not force. Also
    /// proves the two guard clauses (the reasoning-not-ended + the matcher-stopped
    /// both yield an empty ff-run).
    #[test]
    fn compute_ff_tokens_matches_matcher_forced_run() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        let mut state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            vec![],
            vec![],
            None,
            true,
        )
        .unwrap();
        let mut matcher = sequence_matcher();

        // The guard clause: a stopped matcher yields an empty ff-run.
        // (The fresh matcher is not stopped, so consume the full sequence to stop it,
        // then check the empty ff.)
        for &tok in &[97u32, 98, 99] {
            state.commit_token(tok).unwrap();
            matcher.consume_token(tok).unwrap();
        }
        assert!(matcher.is_stopped(), "the matcher is stopped after the full sequence");
        assert!(state.compute_ff_tokens().is_empty(), "a stopped state yields no ff tokens");

        // The positive case: a fresh state at the start of the sequence forces the
        // entire remaining run (the 'a','b','c' are all forced by the sequence grammar).
        let mut fresh_state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            vec![],
            vec![],
            None,
            true,
        )
        .unwrap();
        let mut fresh_matcher = sequence_matcher();
        let state_ff = fresh_state.compute_ff_tokens();
        let matcher_ff = fresh_matcher.compute_ff_tokens();
        assert_eq!(
            state_ff, matcher_ff,
            "the state ff-run must equal the matcher ff-run (the {state_ff:?} vs the {matcher_ff:?})"
        );
        // The sequence grammar forces all 3 tokens from the start.
        assert_eq!(state_ff, vec![97, 98, 99], "the sequence grammar forces its full run from the start");
    }

    /// PROOF (the commit_run == the Matcher prefix): the inclusive + exclusive commit
    /// property. commit_run must commit exactly the grammar-legal prefix of the run
    /// (the Matcher's try_consume_tokens count) + return the prefix length. Inclusive:
    /// every legal token in the prefix is committed. Exclusive: no illegal token is
    /// committed (the run stops at the first rejection, the count reflects the prefix).
    #[test]
    fn commit_run_commits_exactly_the_legal_prefix() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();

        // Case 1 (the inclusive): a fully-legal run commits in full.
        let mut ok_state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            vec![],
            vec![],
            None,
            true,
        )
        .unwrap();
        assert_eq!(ok_state.commit_run(&[97, 98, 99]), 3, "a fully-legal run commits all 3");

        // Case 2 (the exclusive): an illegal token in the middle stops the commit.
        // The run [97, 99, 98] is illegal at index 1 (99 != 98), so commit_run must
        // commit only the legal prefix [97] + return the prefix length (1).
        let mut bad_state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            vec![],
            vec![],
            None,
            true,
        )
        .unwrap();
        let result = bad_state.commit_run(&[97, 99, 98]);
        // The commit_run returns the legal-prefix count (1) when the FSM rejects a token.
        assert_eq!(result, 1, "a run containing an illegal token commits only the legal prefix");
        // The legal prefix [97] was committed before the rejection (the llm_tokens reflects it).
        assert_eq!(bad_state.llm_tokens, vec![97], "only the legal prefix is retained");
    }

    /// PROOF (the validate_tokens == the Matcher legal count, the non-mutating):
    /// validate_tokens must report the number of grammar-legal tokens in the input
    /// WITHOUT advancing the state (the no mutation). Inclusive: it counts every legal
    /// prefix token. Exclusive: it stops at the first illegal token. The state must be
    /// unchanged after the call (the no-mutation property).
    #[test]
    fn validate_tokens_is_non_mutating_and_counts_the_legal_prefix() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        let mut state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![],
            vec![],
            vec![],
            None,
            true,
        )
        .unwrap();

        // The non-mutation: validate_tokens must not advance the state. Call it twice;
        // the result is identical (the state did not move).
        let first = state.validate_tokens(&[97, 98, 99]).unwrap();
        let second = state.validate_tokens(&[97, 98, 99]).unwrap();
        assert_eq!(first, second, "validate_tokens is non-mutating (two calls agree)");
        // The inclusive: the full legal sequence validates in full.
        assert_eq!(first, 3, "the full legal sequence validates to 3");

        // The exclusive: an illegal token bounds the legal prefix.
        let partial = state.validate_tokens(&[97, 99, 98]).unwrap();
        assert_eq!(partial, 1, "the illegal [97,99,98] is legal only for its first token (97)");

        // The state is still at the start (no mutation from any of the above).
        assert_eq!(state.llm_tokens, Vec::<u32>::new(), "validate_tokens did not mutate the state");
    }

    /// The reasoning terminator run: `newline + EOR + newline`. The inclusive: a
    /// state with a reasoning-end token + a newline token produces the 3-token
    /// run. The exclusive: a state with no reasoning-end token produces an empty
    /// run (nothing to terminate).
    #[test]
    fn reasoning_terminator_builds_newline_eor_newline() {
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let (_factory, grm) = sequence_grammar();
        let env = ApproximateTokEnv::single_byte_env();
        // A state with a reasoning-end token (99).
        let state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![99], // reasoning_end_ids
            vec![], // tool_call_start_ids
            vec![], // tool_call_end_ids
            None, // reasoning_budget
            true,
        )
        .unwrap();
        // The inclusive: the terminator is just the EOR (the [99]). The newlines
        // around the EOR are the model's own output, not a forced token.
        assert_eq!(state.reasoning_terminator(), vec![99], "the terminator is the EOR");

        // The exclusive: a state with no reasoning-end token produces an empty run.
        let state_no_eor = GuidanceState::new_from_grammar_with_markers(
            Arc::new(ParserFactory::new_simple(&env).unwrap()),
            &grm,
            vec![], // no reasoning_end_ids
            vec![],
            vec![],
            None, // reasoning_budget
            true,
        )
        .unwrap();
        assert!(state_no_eor.reasoning_terminator().is_empty(), "no EOR means no terminator run");
    }

    /// The ff-read alignment contract: after committing a token, the forced-run read
    /// (compute_ff_tokens) reflects the post-commit state, not the pre-commit one.
    /// For a byte-forcing literal "abc", after committing 'a' the remaining forced
    /// run is "bc" (98, 99) — the just-committed 'a' (97) must NOT reappear in the
    /// ff read (that would be a reflected injection). This pins the commit-then-read
    /// ordering that keeps the ff override aligned with the committed prefix.
    #[test]
    fn commit_then_ff_read_is_aligned() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // A byte-forcing literal: "abc" forces the bytes 'a','b','c' (97,98,99).
        let grm_str = r#"
start: "abc"
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        let mut state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(factory),
            &grm,
            vec![],
            vec![],
            vec![],
            None,
            true,
        )
        .unwrap();

        // Commit the first token (97 = 'a').
        state.commit_token(97).unwrap();

        // The ff read after the commit reflects the post-commit state: the remaining
        // forced run is "bc" (98, 99). The just-committed 'a' (97) must NOT reappear
        // (a reflected injection would re-emit 97).
        let ff_after = state.ff_run();
        assert_eq!(
            ff_after,
            vec![98, 99],
            "after committing 'a' the remaining forced run is 'b','c' (98,99); a stale read would re-include the committed 'a' (97)"
        );
        assert!(
            !ff_after.contains(&97),
            "the post-commit ff-read must NOT re-emit the just-committed token 97 (reflected injection); got {ff_after:?}"
        );

        // Commit the rest; the grammar completes.
        state.commit_token(98).unwrap();
        state.commit_token(99).unwrap();
        assert!(state.is_finished(), "the full forced run is committed");
    }

    // The reasoning-budget HARD BOUND (the firewall, the no soft bias): a reasoning
    // grammar with an open-ended reasoning block + a budget. At `count >= budget`, the
    // mask must be the EOR singleton (the model CANNOT emit another reasoning token --
    // the region is forced to close). This is the fix for "reasoning lengths WAY above
    // the budget" (the prior +10 bias let the model reason past the budget; the phase
    // machine never left Reasoning because the EOR was only permitted, not forced).
    #[test]
    fn reasoning_budget_forces_eor_at_limit() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // A reasoning grammar: the reasoning block is up to FIVE optional 'a' (97) tokens,
        // terminated by the EOR (100='d'). Because the 97s are optional, the EOR is
        // reachable after ANY number of them (0..5) -- so at the budget (3) the EOR is
        // grammar-legal + the budget forcing can cut the reasoning short. Then a text
        // token (101='e') + EOS (102='f').
        let grm_str = r#"
start: reasoning_block text eos
reasoning_block: <[97]>? <[97]>? <[97]>? <[97]>? <[97]>? <[100]>
text: <[101]>
eos: <[102]>
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        let mut state = GuidanceState::new_from_grammar_with_markers(
            Arc::new(factory),
            &grm,
            vec![100], // the EOR (the reasoning_end_ids)
            vec![],   // the no tool-call start
            vec![],   // the no tool-call end
            Some(3),  // the budget = 3 reasoning tokens
            true,
        )
        .unwrap();

        // The phase must start at Reasoning (the reasoning grammar).
        assert_eq!(state.phase(), OutputPhase::Reasoning, "a reasoning grammar starts in the Reasoning phase");

        // Commit 3 reasoning tokens (the budget). Each is a 97 ('a').
        for _ in 0..3 {
            state.commit_token(97).unwrap();
        }
        assert_eq!(state.reasoning_token_count(), 3, "the count reaches the budget");

        // At count == budget, the mask must be the EOR singleton (the 100) -- the budget
        // FORCES the region to close (the 97 is disallowed, even though the grammar would
        // permit two more optional 97s). This is the hard bound (the no the soft +10 bias
        // that let the model reason past the budget).
        let mask = state.compute_mask_or_eos().unwrap();
        assert!(mask.is_allowed(100), "at the budget the EOR (100) is allowed (the region closes)");
        assert!(!mask.is_allowed(97), "at the budget another reasoning token (97) is FORBIDDEN (the hard bound)");

        // Emit the EOR -> the phase leaves Reasoning (the count stops growing). The EOR is the
        // phase transition, NOT reasoning content, so it is NOT counted (the off-by-one
        // fix): the count stays at 3 (the three 97s), not 4. A subsequent Text token must
        // NOT grow it (the region is closed).
        state.commit_token(100).unwrap();
        assert_eq!(state.phase(), OutputPhase::Text, "after the EOR the phase is Text (the reasoning region is closed)");
        assert_eq!(state.reasoning_token_count(), 3, "the EOR is NOT counted (the off-by-one fix): the count is the 3 content 97s, not 4");
        // A Text-region token must not grow the reasoning count (the region is closed).
        state.commit_token(101).unwrap(); // the text token
        assert_eq!(state.reasoning_token_count(), 3, "the reasoning count is FROZEN after the EOR (the no runaway past the region)");
    }

    // The ff single-advance invariant (the tool-call double-emission fix): `commit_ff_run`
    // must advance the matcher EXACTLY ONCE (the `consume_ff_tokens` idiom, the compute +
    // the consume as one operation). The prior bug called `ff_tokens` (the advancing
    // `compute_ff_tokens`) then `commit_run` (the `consume_token` per token) -- TWO
    // advances for one ff run, which desynced the parser from the committed stream (the
    // tool-call header re-emitted into the value region). This test proves the ff run is
    // committed exactly once: after `commit_ff_run`, the matcher is at the post-ff state
    // (the grammar complete for the forcing grammar), NOT two-ff-runs ahead.
    #[test]
    fn ff_commit_is_single_advance() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // The forcing grammar: 'a' (97) then 'b' (98) then 'c' (99). After committing 'a',
        // the ff run is [98, 99] (the forced continuation).
        let grm_str = r#"
start: <[97]> <[98]> <[99]>
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        let mut state = GuidanceState::new_from_grammar_with_markers(
            std::sync::Arc::new(factory),
            &grm,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            true,
        )
        .unwrap();

        // Commit the base token 'a' (97). The parser is now at the post-'a' state (the
        // 'b' + the 'c' are forced).
        state.commit_token(97).unwrap();

        // The ff commit (the single advance): commits [98, 99] in one `consume_ff_tokens`.
        let ff = state.commit_ff_run();
        assert_eq!(ff, vec![98, 99], "the ff run is the forced continuation [98, 99]");

        // After the single ff advance, the grammar is COMPLETE (the 'a','b','c' all
        // committed). The matcher must be stopped (the no a double-advance that would
        // have the parser two ff-runs ahead, the desync).
        assert!(
            state.is_finished(),
            "after the single ff advance the grammar is complete (the no the double-advance desync)"
        );

        // The committed stream is exactly [97, 98, 99] (the no a re-emitted header).
        assert_eq!(state.llm_tokens, vec![97, 98, 99], "the committed stream is a,b,c exactly (the no re-emission)");
    }

    // The per-phase token tally (the Decoded summary's per-region breakdown): a grammar
    // with a reasoning region (the 97s, terminated by the EOR 100), a text region (the
    // 101), + a tool-call region (the 102..104). The `phase_token_counts` must tally
    // each committed token into its region (the no the prior all-in-reasoning miscount).
    #[test]
    fn phase_token_counts_tally_per_region() {
        use llguidance::api::TopLevelGrammar;
        use llguidance::ParserFactory;
        use toktrie::ApproximateTokEnv;

        let env = ApproximateTokEnv::single_byte_env();
        let factory = ParserFactory::new_simple(&env).unwrap();
        // The reasoning block (the optional 97s + the EOR 100), the text (the 101), the
        // tool-call (the 102 start, the 103 content, the 104 end), the EOS (the 105).
        let grm_str = r#"
start: reasoning_block text tool_call eos
reasoning_block: <[97]>? <[97]>? <[100]>
text: <[101]>
tool_call: <[102]> <[103]> <[104]>
eos: <[105]>
"#;
        let grm = TopLevelGrammar::from_lark(grm_str.to_string());
        let mut state = GuidanceState::new_from_grammar_with_markers(
            std::sync::Arc::new(factory),
            &grm,
            vec![100], // the EOR (the reasoning_end_ids)
            vec![102], // the tool-call start (the 102)
            vec![104], // the tool-call end (the 104)
            None,
            true,
        )
        .unwrap();

        // Commit two reasoning tokens (the 97s) + the EOR (the 100). The per-phase TALLY
        // counts every token by the region it was generated in (the EOR was generated
        // in the reasoning region, so it tallies into reasoning). This is distinct from
        // the budget counter (the `reasoning_token_count`, which excludes the EOR -- the
        // off-by-one fix). So after the EOR: tally[reasoning] = 3 (the two 97s + the EOR).
        state.commit_token(97).unwrap();
        state.commit_token(97).unwrap();
        state.commit_token(100).unwrap(); // the EOR (the Reasoning -> Text)
        assert_eq!(state.phase_token_counts(), [3, 0, 0], "the tally counts the EOR into reasoning (the region it was generated in)");

        // Commit the text token (the 101). Tallies into text.
        state.commit_token(101).unwrap();
        assert_eq!(state.phase_token_counts(), [3, 1, 0], "one text token");

        // Commit the tool-call (the 102 start -> the ToolCall phase, the 103 + the 104).
        // The 102 was generated in the Text region (it triggers Text -> ToolCall), so it
        // tallies into text. The 103 + the 104 are generated in the ToolCall region.
        state.commit_token(102).unwrap(); // the tool-call start (the Text -> ToolCall)
        state.commit_token(103).unwrap(); // the tool-call content
        state.commit_token(104).unwrap(); // the tool-call end (the ToolCall -> Text)
        assert_eq!(state.phase_token_counts(), [3, 2, 2], "the 102 tallies into text (generated there), the 103 + the 104 into tool-call");

        // Commit the EOS (the 105). Generated in the Text region (after the tool-call end).
        state.commit_token(105).unwrap();
        assert_eq!(state.phase_token_counts(), [3, 3, 2], "the EOS tallies into text (the phase is Text after the tool-call end)");
    }
}
