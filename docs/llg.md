# LLG (the grammar-FSM state management)

The llg work makes grammar-constrained decoding correct and fast by enforcing the
grammar at a single commit point and steering the sampler with the exact grammar
mask. It builds on the spec-decode work (the dynamic draft length) + the qos work
(the adaptive plan).

## The mask source

The grammar mask is produced by the **CPU (Earley + lexer) parser** in llguidance.
At each decode step the allowed-token set is the exact set of tokens the grammar
permits at the current position, computed by the parser's `compute_mask_or_eos`
(the Earley item-set bias unioned with the lexer's token-range / int-range /
regex contributions + the EOS). This mask is total: it is exact for every
supported grammar class, including the self-embedding (nested / repeated) and the
parametric (numeric, bounded-string) grammars, because it is the parser's own
item-set — not a bounded projection of it.

The mask is applied to the logits either as a compact VOB bitset (the
`XINFER_VOB_SAMPLING` path, 8x less data than the F32 allow-mask) or as an F32
allow-mask folded into the fused sampler. Both carry the identical allowed set;
only the encoding differs.

## Feature contract (the invariants every llg change preserves)

1. **The single FSM writer** — the gate (`commit_run`) is the only place the
   grammar FSM advances; the sequence append is the only place a token enters a
   Sequence. No mid-step commits, no shadow walks that desync the FSM from the
   sequence position.
2. **The exact mask** — the allowed-token set is the parser's own item-set mask
   (total over all supported grammar classes). A grammar-active row that cannot
   produce an exact mask crashes (fail-closed); it is never widened to all-allowed.
3. **The sealed commit** — a produced token run is committed through the gate,
   which keeps only the FSM-legal prefix and rolls the FSM back to that boundary
   on a partial commit, so a rejected suffix never lingers in the parser state.

## Feature sets

1. **The sealed single-writer gate** (`commit_run`, `guidance.rs`) — commits a
   produced token run to the FSM, returns the legal-prefix count, and rolls the
   FSM back to the committed boundary when the run is only partially legal. The
   fast-forward (ff) run is committed through the same gate as one ordered write,
   so the base token + the forced continuation are committed atomically (no
   double-commit, no base duplication).
2. **The anti-loop detector** (`loop_detect.rs`) — detects a closed repetition
   loop (a periodic tail, confirmed over a probe-forward horizon, not mere word
   frequency). The mitigation is a sampling-constraint kick (the looping token is
   driven to -inf, a break seed is forced), applied at the gate so it covers
   every token-production path. Gated on `XINFER_ANTI_LOOP` (default off, zero
   overhead).
3. **The full-envelope grammar** (`guidance_grammar.rs`) — the reasoning levels,
   the tool-calling, the EOS, the user-defined grammar file (the YAML). The
   `llg_full` mode (`XINFER_LLG_FULL`) always builds the full-coverage grammar (no
   tools required).
4. **The reasoning budget** — a hard cap on the reasoning-region length
   (`XINFER_MAX_REASONING_TOKENS` or the per-request `reasoning_budget`). The
   count is owned by the Sequence (stamped by the engine, read by the runner on
   every sampling path). When exhausted, the gate injects the reasoning-terminator
   run as one atomic commit, so the reasoning region ends at the budget
   regardless of the model's logits. The check is count-based, so it fires with
   or without a grammar.
5. **The grammar cache** — caches the instantiated parser (the expensive part),
   not the grammar string (the cheap part).

## Enablement (default state)

| Knob | Default | Effect |
|---|---|---|
| `XINFER_LLG_FULL` | off | always build the full-coverage grammar (no tools required). |
| `XINFER_ANTI_LOOP` | off | the in-flight closed-loop detector + the sampling kick. |
| `XINFER_DEBUG_LLG` | off | the verbose llg debug logging (the grammar resyncs, the anti-loop kicks). |
| `XINFER_MAX_REASONING_TOKENS` | unset | the reasoning token budget (drives the sampling bias that steers the end-of-reasoning token). |
| `--reasoning-grammars <file>` | - | the user-defined reasoning-grammar YAML (the levels map to inner Lark expressions). |

The grammar FSM is active whenever a grammar is requested (the tool outputs, the
tool calling). The anti-loop + the full-envelope + the reasoning budget are the
opt-in hardening.

## The mechanics

### The sealed gate (the single commit point)

```
  produced token run [t1 .. tk]   (the sampled + the ff-rewritten value)
          |
          v
  commit_run(run)  ->  consume the FSM-legal prefix, roll back the rejected suffix
          |
          +-- returns the legal-prefix length n (n <= k)
          v
  the sequence append (the single place a token enters a Sequence) takes run[..n]
```

The gate is the single point where the FSM advances. A token the FSM rejects is
truncated out before it reaches the Sequence, so the emitted stream is always a
grammar-legal prefix. When guidance is inactive for a sequence the gate is a
pass-through (the legal prefix == the whole run), preserving the unguided
behavior.

### The fast-forward (ff) override

The ff run is the grammar-forced continuation (a literal byte-run, a singleton
transition). It is read from the post-anchor parser state and committed through
the gate in generation order: the anchor (the model-sampled token), then the ff
literals (the grammar-forced), then the drafts (the model-proposed speculative
tokens). The FSM is advanced in that order (the anchor advance, the ff advance,
the drafts advance), so the committed state matches the generation trajectory.
The sequence queue is assembled in the same order (the anchor, the ff, the
accepted drafts), so the tokens enter the sequence in generation order. The ff
is committed exactly once (in the gate, the single-advance `consume_ff_tokens`
idiom), never at the sample site. When the ff KV write fails, the FSM rolls back
to the pre-ff boundary (the `resync_to`), so the ff is not committed (the no the
committed-but-unbacked desync) + the sequence re-decodes the position next step.

### The anti-loop detector

```
  recent token window  -->  periodicity check (is the tail S S S ... for some span S?)
        |
        +-- no  -->  (no kick, zero overhead)
        |
        +-- yes -->  probe-forward (does the period hold for PROBE_DEPTH?)
                      |
                      +-- confirmed  -->  sampling kick (the looping token -> -inf, the break seed forced)
                      +-- fluke       -->  (no kick, the false-positive guard)
```

### The reasoning-budget sampling bias

The reasoning length is bounded by a sampling intervention, not a grammar cap.
The grammar's `reasoning_block` rule stays unbounded; the steering is a logit bias
on the end-of-reasoning token that ramps up as the per-sequence reasoning count
approaches the budget:

```
  reasoning_token_count (per seq)
        |
        v
  reasoning_bias_value(count, budget)
        |
        +-- count < budget - budget/4          -->  bias = 0.0        (no steering, the budget is far)
        +-- budget - budget/4 <= count < budget -->  quadratic ramp 0 -> 10.0
        +-- count >= budget                   -->  bias = 10.0      (force the end token)
        |
        v
  apply_reasoning_bias: adds the bias to the end-of-reasoning token's logit, before the sampler runs.
```

The budget is one control path: the per-request `reasoning_budget` wins; the
`XINFER_MAX_REASONING_TOKENS` env is the default when the request sets none. The
bias is a sampling intervention (not the grammar mask), so it works with or
without a grammar, in both the two-phase + the full-envelope modes.

Operator-visible logs (once per sequence, no per-step spam):
- `[reasoning-bias] seq N: ramp active (count=..., budget=..., bias=...), steering end-of-reasoning token T` — the first step the bias becomes non-zero.
- `[reasoning-bias] seq N: budget hit (count=..., budget=...), forcing end-of-reasoning token T` — the step the bias reaches the max.

### The data-movement cardinalities

- **The parser state:** lives inside the llguidance `Matcher` (the Earley item-set
  + the lexer DFA state). The xinfer side holds no parallel copy. Tiny per step
  (no per-token allocation on the mask path).
- **The anti-loop kick:** the per-seq forbid token (a u32) + the seed span (a small
  `Vec<u32>`).
- **The reasoning count:** a single usize per sequence, stamped by the engine onto
  the Sequence (the always-present owner, read by the runner on every sampling
  path).

### The sync/async mechanics

- The parser advance is synchronous (in-line with the token commit, no thread).
- The anti-loop detector runs inline with the decode step (no background thread).
- The mask is computed on the CPU (the parser) and offloaded to the fused sampler
  as a VOB bitset or an F32 allow-mask.

### The data lifecycle

- The parser state lives for the sequence's grammar-constrained span (created at
  the sequence start, destroyed at the sequence end / the grammar completion).
- The anti-loop kick is held until the periodicity clears (the `kick_store`), then
  released.

### The locking

- The parser state is per-sequence inside the `Matcher` (no shared lock; the
  sequence owns its parser).
- The anti-loop `kick_store` is a `RwLock` (read by the sampler, written by the
  detector).

## Known gaps (the production-grade todos)

- **The anti-loop detector is off by default** (`XINFER_ANTI_LOOP` default off,
  zero overhead). It is not enabled in the standard deployment path, so the
  sampling-constraint kick is dormant unless the operator opts in.
- **The `/v1/grammar` mapping is unit-tested** (the
  `grammar_type_maps_to_structured_outputs` test, the pure
  `map_grammar_type_to_structured_outputs` function). A live end-to-end test (the
  constrained generation through the shared `chat_completion` path) is the
  remaining verification.
- **The ff KV reserve is operator-tunable, not automatic.** The ff-span pre-
  alloc (the `pre_allocate_ff_blocks`) uses the `XINFER_FF_RESERVE` env (the
  default 32 positions). A user grammar whose forced literal prefix exceeds the
  reserve degrades to a recoverable step error (the block allocation falls short,
  the engine re-schedules). The on-demand extension (the ff forward allocates the
  blocks it needs, the no the static reserve) is the pending work; the reserve is
  the interim bound.
- **The MTP draft-path grammar mask (the `XINFER_SPEC_DRAFT_MASK` gate) requires
  the U8 allow-row.** The `next_allow_row` returns a U8 tensor (the where_cond
  condition dtype); the prior F32 allow-row crashed the MTP draft path (the
  "where conditions should be u8/u32/i64" error). The fix is on the spec-decode
  branch (the `mask_stream` + the `qwen3_5_mtp` call site).

## Citations / references

- The Earley + lexer grammar mask: the llguidance parser (the total mask over all
  supported grammar classes), <https://github.com/guidance-ai/llguidance>.
- The anti-loop / repetition detection: the repetition-aware decoding line of work
  (e.g. "Repetition-Aware Decoding for Large Language Models", the self-repetition
  suppression literature).
- The full-envelope grammar (the reasoning + the tool-calling + the EOS in one
  grammar): the llguidance full-envelope construction,
  <https://github.com/guidance-ai/llguidance>.