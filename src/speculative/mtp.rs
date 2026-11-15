use crate::core::runner::{Model, ModelRunner, Seqs};
use crate::models::layers::linear::set_linear_is_prefill;
use crate::models::qwen3_5_mtp::Qwen3_5MtpHead;
use crate::runner::SpecStepOut;
use crate::speculative::metadata::SpecSeqInfo;
use crate::speculative::spec_stats::{spec_seq_next_k, spec_stats_update};
use crate::speculative::verify::{mtp_stats_summary, mtp_stats_update, verify_draft_greedy, verify_draft_rejection};
use crate::utils::env::{spec_draft_mask, spec_rejection_sampling};
use crate::utils::guidance::OutputPhase;
use crate::utils::guided_decoding::apply_vob_to_logits_row;
use crate::utils::logits_processor::{LogitsProcessor, Sampling};
use candle_core::{Result, Tensor};
use std::sync::Arc;

impl ModelRunner {
    /// MTP Step 1: single-token decode to get anchor token + hidden state.
    /// Tries CUDA graph replay first (the graph's internal buffer for the
    /// post-norm hidden state is accessible via take_last_hidden_for_mtp),
    /// falling back to eager forward_with_hidden.
    #[allow(unused)]
    fn mtp_decode_step1(&self, seqs: Seqs, _seq_info: &SpecSeqInfo) -> Result<(u32, Tensor)> {
        let (input_ids, positions, mut input_metadata) = match &seqs {
            Seqs::SeqRefs(seqs_ref) => self.prepare_decode(*seqs_ref)?,
            Seqs::DecodeVec(decode_seqs) => self.prepare_decode(decode_seqs.iter())?,
        };

        let _decode_guard = set_linear_is_prefill(false);

        // Try CUDA graph replay for the decode forward. The model's forward()
        // stores hidden states in last_hidden_for_mtp during both capture and
        // replay (the cached tensor shares GPU storage with the graph output,
        // so it's updated in-place on replay).
        #[cfg(all(feature = "cuda", feature = "graph"))]
        {
            let input_batch = input_ids.dim(0)?;
            let require_exact_graph = input_metadata.mamba_slot_mapping.is_some();
            let can_replay = if require_exact_graph {
                self.decode_capturer.is_exact_captured(input_batch)
            } else {
                self.decode_capturer.is_captured(input_batch)
            };
            if can_replay {
                let logits = match self.model() {
                    Model::Qwen3_5(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.decode_capturer
                            .replay(&input_ids, &positions, &input_metadata)?
                    }
                    Model::Qwen3_5MoE(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.decode_capturer
                            .replay(&input_ids, &positions, &input_metadata)?
                    }
                    Model::Qwen3VL(model) => {
                        if let Some(_guard) = model.lock_mamba_cache_for_graph() {
                            self.decode_capturer
                                .replay(&input_ids, &positions, &input_metadata)?
                        } else {
                            self.decode_capturer
                                .replay(&input_ids, &positions, &input_metadata)?
                        }
                    }
                    _ => self
                        .decode_capturer
                        .replay(&input_ids, &positions, &input_metadata)?,
                };

                let hidden_states = match self.model() {
                    Model::Qwen3_5(model) => model.take_last_hidden_for_mtp(),
                    Model::Qwen3_5MoE(model) => model.take_last_hidden_for_mtp(),
                    Model::Qwen3VL(model) => model.take_last_hidden_for_mtp(),
                    _ => None,
                };

                if let Some(hidden_states) = hidden_states {
                    let anchor_token = self.sample(&logits, seqs, false)?[0];
                    let seq_hidden = if hidden_states.dims().len() == 2 && hidden_states.dim(0)? > 1
                    {
                        hidden_states.get(hidden_states.dim(0)? - 1)?
                    } else if hidden_states.dims().len() == 2 {
                        hidden_states.get(0)?
                    } else {
                        hidden_states
                    };
                    return Ok((anchor_token, seq_hidden));
                }
            }
        }

        // Fallback: eager forward_with_hidden (no graph available or hidden state extraction failed)
        #[cfg(feature = "flashinfer")]
        if let Some(fm) = input_metadata.flashinfer_metadata.as_mut() {
            if input_metadata.is_mla {
                if fm.mla_decode_plan_info.is_none() {
                    if let Some(params) = self.flashinfer_kv_params() {
                        fm.mla_decode_plan_info = Some(attention_rs::mla::mla_decode_plan(
                            self.device(),
                            params.kv_dtype,
                            &fm.indptr_host,
                            input_ids.dim(0)?,
                            params.num_qo_heads,
                            params.page_size,
                            fm.use_cuda_graph,
                        )?);
                    }
                }
            } else if fm.decode_plan_info.is_none() {
                if let Some(params) = self.flashinfer_kv_params() {
                    fm.decode_plan_info = Some(attention_rs::flashinfer::decode_plan(
                        self.device(),
                        params.kv_dtype,
                        params.out_dtype,
                        &fm.indptr_host,
                        fm.last_len_host.as_deref(),
                        fm.kv_len_arr_host.as_deref(),
                        input_ids.dim(0)?,
                        params.num_qo_heads,
                        params.num_kv_heads,
                        params.head_dim,
                        params.page_size,
                        fm.use_cuda_graph,
                    )?);
                }
            }
        }

        let kv_cache = self.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let (logits, hidden_states) = match self.model() {
            Model::Qwen3_5(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
            )?,
            Model::Qwen3_5MoE(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
            )?,
            Model::Qwen3VL(model) => model.forward_with_hidden(
                &input_ids,
                &positions,
                kv_pairs,
                &input_metadata,
                false,
            )?,
            _ => {
                drop(kv_cache);
                candle_core::bail!("MTP Step 1 requires Qwen3.5 model");
            }
        };
        drop(kv_cache);

        let anchor_token = self.sample(&logits, seqs, false)?[0];

        let seq_hidden = if hidden_states.dims().len() == 2 && hidden_states.dim(0)? > 1 {
            hidden_states.get(hidden_states.dim(0)? - 1)?
        } else if hidden_states.dims().len() == 2 {
            hidden_states.get(0)?
        } else {
            hidden_states.clone()
        };

        Ok((anchor_token, seq_hidden))
    }

    /// Run MTP speculative decode for a batch of sequences.
    /// Returns Vec<Vec<u32>> where each inner vec contains all accepted tokens for that sequence
    /// (anchor + accepted drafts + bonus token).
    ///
    /// Optimized flow:
    ///   1. Run main model decode via CUDA graph replay (when available) + extract hidden state
    ///   2. Sample anchor token from logits
    ///   3. MTP head drafts K tokens autoregressively (no KV cache)
    ///   4. Verify: run main model on [anchor, draft_0, ..., draft_{K-1}] using native flash
    ///   5. On partial rejection: roll back GDN state to the accepted token boundary
    ///   6. Greedy-accept matching prefix; take bonus token at first mismatch
pub fn run_mtp_decode(&self, seqs: Seqs) -> Result<Vec<SpecStepOut>> {
        // The engine-stamped reasoning count (the no the optional GuidanceState,
        // which is not populated on the MTP path).
        if let Seqs::DecodeVec(decode_seqs) = &seqs {
            for ds in decode_seqs.iter() {
                let eor = ds.sampling_params.guidance_reasoning_end_ids.first().copied().unwrap_or(0);
                let newline = ds.sampling_params.guidance_newline_tokens.clone();
                self.guided_decoding().set_reasoning_count(ds.id, ds.reasoning_token_count, eor, newline);
            }
        }
        let mtp_head = match &self.mtp_head {
            Some(h) => h.clone(),
            None => {
                // The no MTP head (the plain decode fallback): the no speculative K (the
                // next_k = 0, the no draft). The tokens are the plain-decode outputs.
                let output = self.run(seqs, false)?;
                return Ok(output
                    .into_iter()
                    .map(|t| SpecStepOut {
                        tokens: vec![t],
                        next_k: 0,
                    })
                    .collect());
            }
        };

        let (batch_size, seq_infos) = match &seqs {
            Seqs::SeqRefs(s) => {
                let infos: Vec<SpecSeqInfo> = s
                    .iter()
                    .map(|seq| SpecSeqInfo {
                        id: seq.id,
                        len: seq.len(),
                        block_table: seq.block_table.clone(),
                        k: seq.spec_k,
                    })
                    .collect();
                (s.len(), infos)
            }
            Seqs::DecodeVec(d) => {
                let infos: Vec<SpecSeqInfo> = d
                    .iter()
                    .map(|ds| SpecSeqInfo {
                        id: ds.id,
                        len: ds.len,
                        block_table: ds.block_tables.clone(),
                        k: ds.spec_k,
                    })
                    .collect();
                (d.len(), infos)
            }
        };

        if batch_size != 1 {
            return self.run_mtp_decode_batch(seqs, &seq_infos, mtp_head);
        }

        let seq_info = &seq_infos[0];
        // The per-seq dynamic K: draft `seq_info.k` tokens, clamped to the graph
        // ceiling (mtp_num_speculative). The QoS knee can shrink the draft below
        // the ceiling, but it can never grow it past the captured graph size.
        // The k=0 case (the SD not enabled for this seq) falls back to the plain
        // decode (the no verify).
        let mut num_draft = seq_info.k.min(self.mtp_num_speculative);
        // The reasoning-budget suppression: when the sequence is still in the
        // `Reasoning` region + the budget is exhausted (the count >= the
        // XINFER_MAX_REASONING_TOKENS), suppress the drafts (the num_draft = 0)
        // so the step produces only the forced anchor (the end-of-reasoning
        // token, the gate injection). The reasoning region terminates at the
        // budget. The `phase == Reasoning` guard scopes this to the reasoning
        // region only — once the EOR is injected the phase moves to `Text` and
        // drafts resume for the answer region (not suppressing throughput
        // for the whole generation).
        if num_draft > 0
            && self.guided_decoding().phase(seq_info.id) == OutputPhase::Reasoning
            && self.guided_decoding().reasoning_budget_exhausted(seq_info.id)
        {
            crate::log_info!(
                "[reasoning-bias] seq {}: budget exhausted, suppressing MTP drafts (num_draft {} -> 0) to force the end-of-reasoning token",
                seq_info.id, num_draft
            );
            num_draft = 0;
        }
        if num_draft == 0 {
            // The SD not enabled for this seq (the k=0): the plain decode fallback,
            // the no speculative K (the next_k = 0).
            let output = self.run(seqs, false)?;
            return Ok(output
                .into_iter()
                .map(|t| SpecStepOut {
                    tokens: vec![t],
                    next_k: 0,
                })
                .collect());
        }

        // Step 1: Main model decode for logits + hidden state.
        let (anchor_token, seq_hidden) = self.mtp_decode_step1(seqs, seq_info)?;

        // Step 2: Draft K tokens using MTP head (GPU-resident, no per-step CPU sync)
        let embed_weight = match self.model() {
            Model::Qwen3_5(m) => m.embed_weight().clone(),
            Model::Qwen3_5MoE(m) => m.embed_weight().clone(),
            Model::Qwen3VL(m) => m
                .embed_weight()
                .expect("Qwen3VL MTP requires Qwen3.5 text backbone")
                .clone(),
            _ => unreachable!(),
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(m) => m.forward_lm_head(hidden),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(hidden),
                Model::Qwen3VL(m) => m.forward_lm_head(hidden),
                _ => unreachable!(),
            }
        };

        let base_position = seq_info.len.saturating_sub(1);
        let anchor_token_tensor = Tensor::from_vec(vec![anchor_token], (1,), self.device())?;
        // The draft-side grammar mask (the `XINFER_SPEC_DRAFT_MASK` gate). When active +
        // the seq has a grammar, build the MaskStream (the single cloned-FSM
        // trajectory) + a closure that masks each step's draft logits (the illegal
        // tokens -inf'd, the `p_draft` shares support with the verify's masked
        // `p_target`). When the gate is off OR the seq is unguided, `None` (the prior
        // unmasked behavior, full back-compat).
        let mut mask_stream = if spec_draft_mask() {
            self.guided_decoding()
                .mask_stream_for(seq_info.id)
        } else {
            None
        };
        // The draft-side grammar mask (the `XINFER_SPEC_DRAFT_MASK` gate). When a
        // stream exists (the gate on + the seq guided), hand the model loop the
        // `&mut MaskStream` (it drives the per-step allow-row + the ff advance, the
        // positionally-correct walk). When `None` (the gate off, the unguided seq),
        // the prior unmasked behavior (full back-compat).
        let (draft_tokens, _last_hidden, draft_logits) = mtp_head.draft_tokens_gpu(
            &seq_hidden,
            &anchor_token_tensor,
            num_draft,
            &embed_weight,
            lm_head_fn,
            base_position,
            mask_stream.as_mut(),
        )?;

        if draft_tokens.is_empty() {
            // The no draft produced (the empty-draft bail): the anchor only, the no
            // speculative K (the next_k = 0, the no verify).
            return Ok(vec![SpecStepOut {
                tokens: vec![anchor_token],
                next_k: 0,
            }]);
        }

        // Step 3: Verify draft tokens via prefill-style forward on [anchor, draft_0..K-1].
        let mut verify_tokens = vec![anchor_token];
        verify_tokens.extend_from_slice(&draft_tokens);
        let verify_len = verify_tokens.len();

        let block_size = self.block_size();
        let slot_mappings =
            self.compute_slot_mappings(seq_info, verify_len, block_size, "verify")?;

        let verify_input_ids = Tensor::from_vec(verify_tokens, (verify_len,), self.device())?;
        let verify_positions_tensor = Tensor::from_vec(
            (0..verify_len)
                .map(|i| (seq_info.len + i) as i64)
                .collect::<Vec<_>>(),
            (verify_len,),
            self.device(),
        )?;

        let verify_metadata =
            self.build_verify_metadata(seq_info, &slot_mappings[..verify_len], verify_len)?;

        let _prefill_guard = set_linear_is_prefill(true);

        #[cfg(all(feature = "cuda", feature = "graph"))]
        let use_mtp_graph = self
            .mtp_capturer
            .as_ref()
            .map_or(false, |c| c.is_mtp_captured(verify_len));
        #[cfg(not(all(feature = "cuda", feature = "graph")))]
        let use_mtp_graph = false;

        let all_logits_result = if use_mtp_graph {
            #[cfg(all(feature = "cuda", feature = "graph"))]
            {
                // Hold mamba cache across verify-graph replay (same as decode graphs).
                match self.model() {
                    Model::Qwen3_5(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.mtp_capturer.as_ref().unwrap().replay_mtp(
                            &verify_input_ids,
                            &verify_positions_tensor,
                            &verify_metadata,
                        )
                    }
                    Model::Qwen3_5MoE(model) => {
                        let _guard = model.lock_mamba_cache_for_graph();
                        self.mtp_capturer.as_ref().unwrap().replay_mtp(
                            &verify_input_ids,
                            &verify_positions_tensor,
                            &verify_metadata,
                        )
                    }
                    Model::Qwen3VL(model) => {
                        if let Some(_guard) = model.lock_mamba_cache_for_graph() {
                            self.mtp_capturer.as_ref().unwrap().replay_mtp(
                                &verify_input_ids,
                                &verify_positions_tensor,
                                &verify_metadata,
                            )
                        } else {
                            self.mtp_capturer.as_ref().unwrap().replay_mtp(
                                &verify_input_ids,
                                &verify_positions_tensor,
                                &verify_metadata,
                            )
                        }
                    }
                    _ => self.mtp_capturer.as_ref().unwrap().replay_mtp(
                        &verify_input_ids,
                        &verify_positions_tensor,
                        &verify_metadata,
                    ),
                }
            }
            #[cfg(not(all(feature = "cuda", feature = "graph")))]
            {
                unreachable!()
            }
        } else {
            let kv_cache = self.get_kv_cache();
            let kv_pairs = kv_cache.as_pairs();
            let res = match self.model() {
                Model::Qwen3_5(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    kv_pairs,
                    &verify_metadata,
                    false,
                ),
                Model::Qwen3_5MoE(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    kv_pairs,
                    &verify_metadata,
                    false,
                ),
                Model::Qwen3VL(model) => model.forward(
                    &verify_input_ids,
                    &verify_positions_tensor,
                    kv_pairs,
                    &verify_metadata,
                    None,
                ),
                _ => unreachable!(),
            };
            drop(kv_cache);
            res
        };
let all_logits = all_logits_result?;

        // The MTP drafter proposes from its own logits; grammar-illegal drafts are
        // rejected by the verify step (the target-logits comparison in
        // verify_draft_greedy / verify_draft_rejection), not by a draft-position mask.
        let masked_all_logits = all_logits.clone();
        // The verify: the greedy argmax accept (the default) OR the lossless
        // rejection sampling (the XINFER_SPEC_REJECTION_SAMPLING, the target's
        // sampled distribution honored, the no biased output at temperature > 0).
        // The rejection path uses the target logits; grammar-illegal drafts are
        // rejected by the min(1, p_target/p_draft) accept test + the residual draw.
        let verify_result = if spec_rejection_sampling() {
            // The lossless rejection sampling. The `p_target` is the target's
            // post-sampling distribution per verify position (the sampler's
            // transform over the MASKED logits, the exact distribution the normal
            // decode would sample from); the `p_draft` is the drafter's per-position
            // softmax.
            let sampling = self
                .cached_sampling()
                .map(|c| c.sampling)
                .unwrap_or(Sampling::ArgMax);
            let processor = LogitsProcessor::from_sampling(0, sampling);
            let vflat: Vec<f32> = masked_all_logits
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let vocab = vflat.len() / verify_len.max(1);
            // The grammar mask (the lossless-rejection correctness): the `p_target`
            // must be the target's distribution *conditioned on the grammar*, not
            // the raw full-vocab softmax. Build the per-verify-row allow-masks via
            // the optimistic `deep_clone` projection (the committed matcher is never
            // advanced), then zero the illegal logits (the -inf before the softmax)
            // so `sampling_probs_row` yields the masked distribution. When the seq
            // has no active grammar (the unguided), the projection is `None` and the
            // raw logits stand (the prior behavior, the no mask).
            let masked_vflat: Vec<f32> = match self
                .guided_decoding()
                .project_verify_masks(seq_info.id, &draft_tokens, verify_len)
            {
                Some(Ok(rows)) => {
                    let mut out = vflat.clone();
                    for (row, mask) in rows.iter().enumerate() {
                        if row >= verify_len {
                            break;
                        }
                        let r = &mut out[row * vocab..(row + 1) * vocab];
                        // The shared masking helper (the -inf the illegal tokens, the
                        // out-of-range tokens stay allowed, the unbounded `is_allowed`
                        // guard). Defined once in guided_decoding so the MTP + DFlash
                        // verify paths + the unit test all use the same semantics.
                        apply_vob_to_logits_row(r, mask);
                    }
                    out
                }
                Some(Err(e)) => {
                    // The projection failed (a grammar error mid-verify). Fail closed:
                    // falling back to the unmasked `vflat` would admit grammar-illegal
                    // tokens (the exact bug this masking exists to prevent). Instead,
                    // propagate the error so the verify is aborted + the seq resyncs on
                    // the next step (the `resync` rolls the committed matcher back to the
                    // last legal boundary). The `anyhow::Error` is converted to the
                    // `candle_core::Error` this function returns.
                    return Err(candle_core::Error::wrap(e));
                }
                None => vflat, // the unguided seq (the no grammar, the raw logits)
            };
            let target_probs_flat: Vec<f32> = (0..verify_len)
                .flat_map(|row| {
                    let r = &masked_vflat[row * vocab..(row + 1) * vocab];
                    processor.sampling_probs_row(r).unwrap_or_else(|| r.to_vec())
                })
                .collect();
            let target_probs_tensor =
                Tensor::from_vec(target_probs_flat, (verify_len, vocab), self.device())?;
            // The draft probs (the softmax of the drafter's per-position logits).
            let draft_stacked = Tensor::stack(&draft_logits, 0)?;
            let dflat: Vec<f32> = draft_stacked
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let drows = dflat.len() / vocab.max(1);
            let mut draft_probs_flat = vec![0.0f32; dflat.len()];
            for r in 0..drows {
                let row = &dflat[r * vocab..(r + 1) * vocab];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for (j, &e) in exps.iter().enumerate() {
                    draft_probs_flat[r * vocab + j] = e * inv;
                }
            }
            let draft_probs_tensor =
                Tensor::from_vec(draft_probs_flat, (drows, vocab), self.device())?;
            // The S2 per-seq rejection seed: key the RNG by the sequence id (the
            // golden-ratio mix, matching the GPU Philox per-row convention) so the
            // rejection draws are reproducible per-request + batch-independent,
            // not a fixed seed=0 shared across every sequence.
            let seed = (seq_info.id as u64).wrapping_mul(0x9E3779B97F4A7C15);
            verify_draft_rejection(&target_probs_tensor, &draft_probs_tensor, &draft_tokens, seed)?
        } else {
            verify_draft_greedy(&masked_all_logits, &draft_tokens)?
        };

        if verify_result.num_accepted < verify_result.num_proposed {
            let commit_len = 1 + verify_result.num_accepted;
            // Full-attention KV cache does not need explicit rollback: the next cycle's
            // verify will overwrite rejected positions via append_kv_cache before the
            // attention kernel reads them, and FlashInfer uses kCausal masking.
            // GDN/Mamba state, however, is mutated in-place and must be rolled back.
            let restored = self.rollback_mamba(seq_info.id, commit_len)?;
            if !restored {
                candle_core::bail!(
                    "MTP failed to roll back mamba-state snapshot for seq {} to {} verified token(s)",
                    seq_info.id,
                    commit_len
                );
            }
        }

        let mut result_tokens = Vec::with_capacity(2 + verify_result.num_accepted);
        result_tokens.push(anchor_token);
        result_tokens.extend_from_slice(&verify_result.accepted_tokens);
        result_tokens.push(verify_result.continuation_token);

        // The windowed MTP summary is logged once (the rank-0, the authoritative NCCL
        // all-reduce makes all ranks agree on the verify outcome, so the per-rank logs are
        // identical). The no 4x repetition on a 4-GPU TP run.
        if self.is_first_rank() && mtp_stats_update(verify_result.num_proposed, verify_result.num_accepted) {
            crate::log_info!("{}", mtp_stats_summary());
        }
        spec_stats_update("MTP", seq_info.id, &verify_result);

        // The per-seq next-step K (the runner-computed, the no the scheduler guessing
        // from an empty cross-process global). The rolling curve is populated in THIS
        // process (the runner), so the knee is computed here + reported back in the
        // SpecStepOut (the no the main-process SPEC_SEQ_STATS, which is empty).
        let ceiling = self.mtp_num_speculative;
        let break_even = self.spec_break_even();
        let next_k = spec_seq_next_k(seq_info.id, ceiling, break_even);
        Ok(vec![SpecStepOut {
            tokens: result_tokens,
            next_k,
        }])
    }

    fn run_mtp_decode_batch(
        &self,
        seqs: Seqs,
        seq_infos: &[SpecSeqInfo],
        mtp_head: Arc<Qwen3_5MtpHead>,
    ) -> Result<Vec<SpecStepOut>> {
        let embed_weight = match self.model() {
            Model::Qwen3_5(m) => m.embed_weight().clone(),
            Model::Qwen3_5MoE(m) => m.embed_weight().clone(),
            Model::Qwen3VL(m) => m
                .embed_weight()
                .expect("Qwen3VL MTP requires Qwen3.5 text backbone")
                .clone(),
            _ => unreachable!(),
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(m) => m.forward_lm_head(hidden),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(hidden),
                Model::Qwen3VL(m) => m.forward_lm_head(hidden),
                _ => unreachable!(),
            }
        };

        let mut anchors = Vec::with_capacity(seq_infos.len());
        let mut draft_tokens = Vec::with_capacity(seq_infos.len());
        for (index, seq_info) in seq_infos.iter().enumerate() {
            let (anchor, seq_hidden) = match &seqs {
                Seqs::SeqRefs(sequences) => {
                    self.mtp_decode_step1(Seqs::SeqRefs(&sequences[index..index + 1]), seq_info)?
                }
                Seqs::DecodeVec(sequences) => {
                    let single_sequence = vec![sequences[index].clone()];
                    self.mtp_decode_step1(Seqs::DecodeVec(&single_sequence), seq_info)?
                }
            };
            let anchor_tensor = Tensor::from_vec(vec![anchor], (1,), self.device())?;
            let base_position = seq_info.len.saturating_sub(1);
            // The per-seq K: draft `seq_info.k` tokens (the no fixed ceiling). The
            // verify below uses the batch-max K (the uniform q_len for the one
            // attention pass), but each seq drafts its own dynamic length.
            // The reasoning-budget suppression: when the sequence is still in the
            // `Reasoning` region + the budget is exhausted for this seq (the count >=
            // the XINFER_MAX_REASONING_TOKENS), draft 0 tokens (the anchor only, the
            // forced end-of-reasoning token). The `phase == Reasoning` guard scopes
            // this to the reasoning region only.
            let k = if seq_info.k > 0
                && self.guided_decoding().phase(seq_info.id) == OutputPhase::Reasoning
                && self.guided_decoding().reasoning_budget_exhausted(seq_info.id)
            {
                crate::log_info!(
                    "[reasoning-bias] seq {}: budget exhausted, suppressing MTP drafts (k {} -> 0) to force the end-of-reasoning token",
                    seq_info.id, seq_info.k
                );
                0
            } else {
                seq_info.k
            };
            let (draft, _last_hidden, _draft_logits) = mtp_head.draft_tokens_gpu(
                &seq_hidden,
                &anchor_tensor,
                k,
                &embed_weight,
                &lm_head_fn,
                base_position,
                None,
            )?;
            anchors.push(anchor);
            draft_tokens.push(draft);
        }

        if draft_tokens.iter().any(|draft| draft.is_empty()) {
            // The no draft produced for some seq (the empty-draft bail): the anchor only,
            // the no speculative K (the next_k = 0, the no verify).
            return Ok(anchors
                .into_iter()
                .map(|anchor| SpecStepOut {
                    tokens: vec![anchor],
                    next_k: 0,
                })
                .collect());
        }

        // The batch verify length: the MAX per-seq K across the batch (the uniform
        // q_len for the one attention pass). A seq with a smaller K drafts fewer
        // tokens; its verify positions beyond its draft are the anchor-only (the no
        // over-verify past the seq's own K). The batch-max K is clamped to the
        // graph ceiling (mtp_num_speculative) so the verify never exceeds the
        // captured graph size.
        let verify_len = seq_infos.iter().map(|s| s.k).max().unwrap_or(0).min(self.mtp_num_speculative) + 1;
        let mut verify_tokens = Vec::with_capacity(seq_infos.len() * verify_len);
        let mut slot_mappings = Vec::with_capacity(seq_infos.len());
        for (seq_info, (anchor, draft)) in seq_infos.iter().zip(anchors.iter().zip(&draft_tokens)) {
            // Each seq contributes exactly `verify_len` verify positions: the anchor +
            // its `draft` tokens + padding to `verify_len`. The padding uses the anchor
            // token (a no-op verify position; the verify only accepts up to the seq's
            // own draft length, the padding positions are never accepted).
            verify_tokens.push(*anchor);
            verify_tokens.extend_from_slice(draft);
            while verify_tokens.len() % verify_len != 0 {
                verify_tokens.push(*anchor);
            }
            slot_mappings.push(self.compute_slot_mappings(
                seq_info,
                verify_len,
                self.block_size(),
                "MTP batch verify",
            )?);
        }
        let q_lens = vec![verify_len; seq_infos.len()];
        let verify_metadata =
            self.build_verify_metadata_batch(seq_infos, &slot_mappings, &q_lens)?;
        let verify_input_ids = Tensor::from_vec(
            verify_tokens,
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;
        let verify_positions = Tensor::from_vec(
            seq_infos
                .iter()
                .flat_map(|seq| seq.len..seq.len + verify_len)
                .map(|position| position as i64)
                .collect::<Vec<_>>(),
            (seq_infos.len() * verify_len,),
            self.device(),
        )?;

        let _prefill_guard = set_linear_is_prefill(true);
        let kv_cache = self.get_kv_cache();
        let kv_pairs = kv_cache.as_pairs();
        let all_logits = match self.model() {
            Model::Qwen3_5(model) => model.forward(
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
            ),
            Model::Qwen3_5MoE(model) => model.forward(
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                false,
            ),
            Model::Qwen3VL(model) => model.forward(
                &verify_input_ids,
                &verify_positions,
                kv_pairs,
                &verify_metadata,
                None,
            ),
            _ => unreachable!(),
        }?;
        drop(kv_cache);

        let mut outputs = Vec::with_capacity(seq_infos.len());
        for (index, (seq_info, draft)) in seq_infos.iter().zip(&draft_tokens).enumerate() {
            let offset = index * verify_len;
            let logits = all_logits.narrow(0, offset, verify_len)?;
            // The MTP drafter proposes from its own logits; grammar-illegal drafts
            // are rejected by the verify step, not by a draft-position mask.
            let masked_logits = logits.clone();
            let verify_result = verify_draft_greedy(&masked_logits, draft)?;
            if verify_result.num_accepted < verify_result.num_proposed {
                let keep_tokens = 1 + verify_result.num_accepted;
                if !self.rollback_mamba_at(seq_info.id, keep_tokens, offset)? {
                    candle_core::bail!(
                        "MTP failed to roll back mamba-state for batch sequence {}",
                        seq_info.id
                    );
                }
            }
            let mut result = Vec::with_capacity(verify_result.num_accepted + 2);
            result.push(anchors[index]);
            result.extend_from_slice(&verify_result.accepted_tokens);
            result.push(verify_result.continuation_token);
            mtp_stats_update(verify_result.num_proposed, verify_result.num_accepted);
            spec_stats_update("MTP", seq_info.id, &verify_result);
            // The per-seq next-step K (the runner-computed, the no the scheduler
            // guessing from an empty cross-process global).
            let ceiling = self.mtp_num_speculative;
            let break_even = self.spec_break_even();
            let next_k = spec_seq_next_k(seq_info.id, ceiling, break_even);
            outputs.push(SpecStepOut {
                tokens: result,
                next_k,
            });
        }
        Ok(outputs)
    }

    /// The drafter-chaining seam (the MTP → the DFlash): run the MTP draft for a
    /// single sequence + return the draft's + the MTP's final hidden state
    /// (the anchor the DFlash drafter consumes). The `final_hidden` is the
    /// MTP head's post-draft hidden state (the `draft_tokens_gpu` return), which
    /// the DFlash `draft_tokens` accepts as its `target_hidden`. This makes the
    /// linear-then-parallel drafter chain possible (the MTP's stable-acceptance
    /// draft anchors the DFlash's batch-varying draft).
    pub fn mtp_draft_hidden(
        &self,
        seqs: Seqs,
        mtp_head: &Arc<Qwen3_5MtpHead>,
    ) -> Result<(Vec<u32>, Tensor)> {
        let seq_infos: Vec<SpecSeqInfo> = match &seqs {
            Seqs::SeqRefs(seqs_ref) => seqs_ref
                .iter()
                .map(|seq| SpecSeqInfo {
                    id: seq.id,
                    len: seq.len(),
                    block_table: seq.block_table.clone(),
                    k: seq.spec_k,
                })
                .collect(),
            Seqs::DecodeVec(decode_seqs) => decode_seqs
                .iter()
                .map(|ds| SpecSeqInfo {
                    id: ds.id,
                    len: ds.len,
                    block_table: ds.block_tables.clone(),
                    k: ds.spec_k,
                })
                .collect(),
        };
        let seq_info = &seq_infos[0];
        // The per-seq K, clamped to the graph ceiling (the no the draft exceeds
        // the captured graph size).
        let num_draft = seq_info.k.min(self.mtp_num_speculative);

        // The MTP Step 1 (the anchor token + the hidden state).
        let (anchor_token, seq_hidden) = self.mtp_decode_step1(seqs, seq_info)?;

        let embed_weight = match self.model() {
            Model::Qwen3_5(m) => m.embed_weight().clone(),
            Model::Qwen3_5MoE(m) => m.embed_weight().clone(),
            Model::Qwen3VL(m) => m
                .embed_weight()
                .expect("Qwen3VL MTP requires Qwen3.5 text backbone")
                .clone(),
            _ => unreachable!(),
        };
        let lm_head_fn = |hidden: &Tensor| -> Result<Tensor> {
            match self.model() {
                Model::Qwen3_5(m) => m.forward_lm_head(hidden),
                Model::Qwen3_5MoE(m) => m.forward_lm_head(hidden),
                Model::Qwen3VL(m) => m.forward_lm_head(hidden),
                _ => unreachable!(),
            }
        };

        let base_position = seq_info.len.saturating_sub(1);
        let anchor_token_tensor = Tensor::from_vec(vec![anchor_token], (1,), self.device())?;
        // The MTP draft (the K tokens) + the final hidden state (the chaining anchor).
        let (draft_tokens, final_hidden, _draft_logits) = mtp_head.draft_tokens_gpu(
            &seq_hidden,
            &anchor_token_tensor,
            num_draft,
            &embed_weight,
            lm_head_fn,
            base_position,
            None,
        )?;

        Ok((draft_tokens, final_hidden))
    }
}
