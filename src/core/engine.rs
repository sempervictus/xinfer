//src/core/engine.rs
use super::runner::{ModelRunner, RunnerType, Seqs};
use super::scheduler::{Scheduler, KVCACHE_SWAP_THRESHOLD};
use super::sequence::Sequence;
use super::qos::SeqReport;
use crate::core::scheduler::PD_PREFILL_STATUS_CHECK_COOLING_PERIOD;
use crate::core::sequence::{DecodeSequence, SequenceStatus};
use crate::core::lora::{DEFAULT_LORA_MAX_BYTES, DEFAULT_LORA_MAX_TENANTS};
use crate::core::mem_budget::{apply_kv_cache_cap, GpuReserve, MemoryMapSet, PoolBudget};
use crate::telemetry::Telemetry;
use crate::core::GenerationOutput;
use crate::models::layers::distributed::Comm;
#[cfg(feature = "nccl")]
use crate::models::layers::distributed::Id;
use crate::models::layers::VarBuilderX;
use crate::runner::{
    receive_local, send_and_expect_ack, send_local, MessageType, NcclId, RunnerInitRequest,
    SpecSeqStatsData, SpecStepOut,
};
use crate::speculative::{uses_builtin_mtp, uses_dflash};
use crate::speculative::spec_stats::{init_drafter_profile, preferred_drafter, spec_seq_stats_data};
use crate::server::logger::ChatCompletionLogger;
use crate::server::parser::{detect_prefilled_reasoning_end_marker, ToolConfig};
use crate::server::{EmbeddingStrategy, UsageResponse};
use crate::tools::Tool;
use crate::transfer::PdRole;
use crate::transfer::Transfer;
use crate::utils::chat_template::Message;
use crate::utils::config::{EngineConfig, EosTokenId, ModelType, SamplingParams};
use crate::utils::guidance::{build_llg_factory, extract_guidance_tokens, GuidanceTokens};
use crate::utils::guided_decoding::GuidedDecoding;
use crate::utils::guidance_grammar::{get_lark_from_top_level_grammar, is_reasoning_grammar};
use crate::utils::heartbeat::heartbeat_worker;
use crate::utils::image::{get_image_config, ImageData, ImageProcessConfig};
use crate::utils::loop_detect::{KickStore, LoopDetector, LoopSeqStats};
use crate::utils::multi_node::{recv_tcp, send_tcp};
use crate::utils::kvcache_allocator::{hybrid_mamba_graph_capture_max_batch, KVCacheAllocator};
use crate::utils::env::{
    anti_loop_enabled, moe_offload_auto, moe_offload_profile, qos_enabled,
    repetition_probe_depth, spec_ff, spec_parallel_draft,
};
use crate::utils::is_qwen3_hybrid_arch_name;
use crate::utils::progress::{progress_worker, ProgressReporter};
use crate::utils::progress::{spawn_progress_thread, ProgressLike};
use crate::utils::{chat_template::ChatTemplate, prepare_engine_config};
use crate::utils::{get_runner_path, init_config_tokenizer, spawn_runner};
use crate::{log_info, log_warn};
use candle_core::{DType, Result};
use colored::Colorize;
use futures::future::join_all;
use interprocess::local_socket::traits::Listener;
use interprocess::local_socket::{GenericNamespaced, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use interprocess::TryClone;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
pub static GLOBAL_RT: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build global Tokio runtime")
});

const REQUEST_ADMISSION_DECODE_BUDGET_TOKENS: usize = 4096;

#[derive(Debug, Clone)]
pub enum StreamItem {
    Token(String, u32), //streaming: (text, token_id)
    TokenID(u32),       //completion
    Completion((usize, usize, usize, Vec<u32>, Option<String>)), //completion
    Done((usize, usize, usize, usize, Option<String>)), //streaming end
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RequestType {
    Stream,
    Completion,
}

/// Output of `LLMEngine::preprocess`. Carries the per-request CPU work
/// (chat-template render + tokenize) the caller has already done so the
/// engine write lock can be acquired briefly, for the scheduler-add step
/// only.
#[derive(Debug, Clone)]
pub struct PreprocessedRequest {
    pub params: SamplingParams,
    pub prompt: String,
    pub image_idx: i32,
    pub token_ids: Vec<u32>,
}

#[allow(dead_code)]
pub struct LLMEngine {
    pub runners: Arc<RwLock<RunnerType>>,
    pub scheduler: Scheduler,
    pub tokenizer: Tokenizer,
    pub econfig: EngineConfig,
    default_chat_template: String,
    template: ChatTemplate,
    prompt_replay_candidates: Vec<Vec<u32>>,
    stream_decoders: HashMap<usize, super::DecodeStreamType>,
    stream_senders: HashMap<usize, Sender<StreamItem>>,
    request_types: HashMap<usize, RequestType>,
    decode_start_times: HashMap<usize, usize>,
    decode_length: HashMap<usize, usize>,
    seq_prefilled_reasoning_end: HashMap<usize, String>,
    seq_prompt_replays: HashMap<usize, Vec<u32>>,
    last_check_throughput_time: usize,
    active_requests: HashSet<usize>,
    cancelled_sequences: Vec<usize>,
    stop_flag: Arc<AtomicBool>,
    has_vision: bool,
    model_name: String,
    pub model_type: ModelType,
    pub tool_config: ToolConfig,
    pub img_cfg: Option<ImageProcessConfig>,
pub guidance_tokens: GuidanceTokens,
/// The live physical memory map. One device map per rank plus one host map,
    /// owned by value on the engine (no internal lock, no Arc). Mutated under the
    /// engine write lock (the boot carve, the grow, the reclaim, the probe
    /// refresh). The scheduler and the QoS controller read it for the physical
    /// pressure signal, which unlike the block occupancy is aware of the idle
    /// VRAM.
    mem_map: MemoryMapSet,
    seq_spec_stats: std::sync::Mutex<HashMap<usize, SpecSeqStatsData>>,
    /// The cached CUDA graph pool usage (the current + peak bytes, the last
    /// refresh time). The graph_pool_usage is a CUDA driver call that can block
    /// (the GPU busy), so it's cached + refreshed periodically (the no every
    /// scrape). The scrape reads the cache (the immediate, the no the 25s stall).
    #[cfg(all(feature = "cuda", feature = "graph"))]
    graph_pool_cache: std::sync::Mutex<(u64, u64, std::time::Instant)>,
    // The llg anti-loop detector (the per-seq closed-loop tail) + the kick store
    // (the active kicks) + the per-seq loop stats (the report carrier) + the
    // transient guard-held counter. Zero-overhead when XINFER_ANTI_LOOP is off.
    loop_detector: LoopDetector,
    kick_store: KickStore,
    seq_loop_stats: std::collections::HashMap<usize, LoopSeqStats>,
    guard_step_counts: std::collections::HashMap<usize, usize>,
}

impl LLMEngine {
    #[allow(unused_mut)]
    pub fn new(econfig: &EngineConfig, dtype: DType) -> Result<Arc<RwLock<Self>>> {
        let (model_pathes, is_gguf, mut config, config_tokenizer, tokenizer, mut generation_cfg) =
            init_config_tokenizer(econfig)?;
        let llg_factory = match build_llg_factory(tokenizer.clone(), config.vocab_size) {
            Ok(f) => Some(f),
            Err(e) => {
                crate::log_warn!("Failed to build llguidance factory: {}", e);
                None
            }
        };

        let stop_flag = Arc::new(AtomicBool::new(false));
        let model_loaded = Arc::new(AtomicBool::new(false));
        let (mut econfig, use_runner) =
            prepare_engine_config(econfig, &config, &config_tokenizer, &mut generation_cfg);
        config.kvcache_dtype = econfig.kvcache_dtype;
        config.is_f16_mode = dtype == DType::F16;

        // In case config file missing bos and eos configuration
        config.apply_generation_cfg(generation_cfg.as_ref());
        if config.eos_token_id.is_none() {
            if let Some(eos) = &config_tokenizer.eos_token {
                if let Some(token) = tokenizer.get_vocab(true).get(eos).copied() {
                    config.eos_token_id = Some(EosTokenId::Single(token));
                };
            }
        }
        if config.bos_token_id.is_none() {
            if let Some(bos) = &config_tokenizer.bos_token {
                if let Some(token) = tokenizer.get_vocab(true).get(bos).copied() {
                    config.bos_token_id = Some(token as usize);
                };
            }
        }
        let guidance_tokens = extract_guidance_tokens(
            &tokenizer,
            config
                .eos_token_id
                .as_ref()
                .map(EosTokenId::to_vec)
                .unwrap_or_default(),
            config
                .bos_token_id
                .map_or(Vec::new(), |bos| vec![bos as u32]),
            &config_tokenizer,
        );
        assert!(
            config.architectures.is_some() && config.architectures.as_ref().unwrap().len() == 1,
            "Only one architecture is supported at the moment!"
        );
        let arch = config.architectures.as_ref().unwrap()[0].clone();
        if is_qwen3_hybrid_arch_name(arch.as_str()) {
            if let Some(p_cfg) = &econfig.pd_config {
                let role = match p_cfg.role {
                    PdRole::Server => "pd-server",
                    PdRole::Client => "pd-client",
                };
                candle_core::bail!(
                    "{} is enabled with {} (hybrid Mamba model), but PD separation is not supported because Mamba states are not transferred across PD nodes.",
                    role,
                    arch
                );
            }
        }
        let (model_type, default_chat_template, is_rope_i) =
            crate::utils::get_arch_rope(&tokenizer, arch.clone())?;
        log_info!("Use ROPE interleaved {is_rope_i}");

        let is_pd_server = if let Some(p_cfg) = &econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        };

        log_info!("{:?}", econfig);

        log_info!("{:?}\n", config_tokenizer);

        log_info!("{:?}\n", config);

        let device_ids = if let Some(ids) = &econfig.device_ids {
            log_info!("Using device IDs: {:?}", ids);
            ids.clone()
        } else {
            log_info!("Using default device ID: [0]");
            vec![0]
        };
        #[cfg(feature = "nvtx")]
        assert!(
            device_ids.len() == 1,
            "nvtx profiler only workable on single rank!"
        );

        #[cfg(feature = "nvtx")]
        let use_runner = false;

        // The per-rank physical memory view (the live free, the total VRAM), captured
        // from the runner NumBlocksReport (the use_runner branch) and used to
        // reconcile the memory map (the C7). Empty on the Thread runner (the in-
        // process, the rank 0 probe is direct).
        let mut per_rank_live_free: Vec<u64> = Vec::new();
        let mut per_rank_total_vram: Vec<u64> = Vec::new();

        let runners = if !use_runner {
            let device = crate::utils::new_device(device_ids[0])?;
            log_info!("Loading model...");
            let reporter: Arc<RwLock<Box<dyn ProgressLike>>> =
                Arc::new(RwLock::new(Box::new(ProgressReporter::new(0))));
            let handle = progress_worker(1, config.num_hidden_layers, &reporter);
            let mut model_runner = {
                let _guard = candle_core::InferenceMode::enter();
                let vb = VarBuilderX::new(&model_pathes, is_gguf, dtype, &device)?;
                let transfer = if let Some(p_cfg) = &econfig.pd_config {
                    Some(Arc::new(Transfer::new(
                        p_cfg.clone(),
                        0,
                        model_loaded.clone(),
                        stop_flag.clone(),
                    )?))
                } else {
                    None
                };

                let runner = ModelRunner::new(
                    model_type.clone(),
                    &vb,
                    #[cfg(not(feature = "nccl"))]
                    Rc::new(Comm::default()),
                    #[cfg(feature = "nccl")]
                    Rc::new(
                        Comm::from_rank(
                            device.as_cuda_device().unwrap().cuda_device(),
                            0,
                            1,
                            Id::new().unwrap(),
                        )
                        .unwrap(),
                    ),
                    &mut econfig,
                    &mut config,
                    dtype,
                    is_rope_i,
                    device.clone(),
                    reporter,
                    transfer,
                    llg_factory.clone(),
                    None,
                )?;
                // Async CUDA weight H2D keeps mmap live until this sync.
                #[cfg(feature = "cuda")]
                device.synchronize()?;
                drop(vb);
                runner
            };

            if !is_pd_server {
                #[cfg(all(feature = "cuda", feature = "graph"))]
                if econfig.disable_cuda_graph {
                    log_info!("CUDA graph capture disabled by --disable-cuda-graph");
                } else if crate::utils::is_no_cuda_graph_supprt(arch.clone()) {
                    log_info!("{arch} does not supprt CUDA graph");
                } else if moe_offload_profile() {
                    crate::log_info!(
                        "CUDA graph disabled: XINFER_MOE_OFFLOAD_PROFILE syncs every MoE call"
                    );
                } else if crate::utils::qwen4_has_ple(&config) {
                    log_info!(
                        "CUDA graph disabled: Qwen4 PLE n-gram embedding gathers on host memory"
                    );
                } else {
                    match model_runner.warmup_capture() {
                        Ok(_) => {
                            log_info!("Cuda graph captured for performance enhancement!")
                        }
                        Err(e) => crate::log_error!("Unable to capture cuda graph {:?}!", e),
                    }
                }
            }

            model_loaded.store(true, Ordering::SeqCst);
            let _ = handle.join();
            RunnerType::Thread(model_runner)
        } else {
            log_info!("Loading model with runner(s)...");

            let is_multi_node = econfig.num_nodes > 1;
            let node_rank = econfig.node_rank;
            let num_nodes = econfig.num_nodes;
            let local_num_gpus = device_ids.len();
            let global_rank_offset = node_rank * local_num_gpus;

            #[cfg(feature = "nccl")]
            let nccl_id = if is_multi_node {
                let master_addr = econfig
                    .master_addr
                    .as_ref()
                    .expect("master_addr required for multi-node");
                crate::utils::multi_node::coordinate_nccl_id(
                    num_nodes,
                    node_rank,
                    master_addr,
                    econfig.master_port,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Multi-node NCCL ID exchange failed: {}", e))
                })?
            } else {
                Id::new().unwrap()
            };

            let runner_path = get_runner_path()?;

            let uuid_str = uuid::Uuid::new_v4();
            let uuid_str = uuid_str.to_string();
            let unique_id = if let Some(l_uuid) = uuid_str.split('-').last() {
                l_uuid
            } else {
                ""
            };

            #[cfg(feature = "python")]
            pyo3::Python::with_gil(|py| {
                for (rank, _) in device_ids.iter().enumerate() {
                    let sock_name = format!("{}@xinfer-runner-{}", unique_id, rank);
                    spawn_runner(py, &runner_path.display().to_string(), &sock_name, unique_id)
                        .expect("Failed to spawn runner. \n\r*****Tips: xinfer binary not found in this package, use 'build.sh' script to build.");
                }
            });

            #[cfg(not(feature = "python"))]
            for (rank, _) in device_ids.iter().enumerate() {
                let sock_name = format!("{}@xinfer-runner-{}", unique_id, rank);
                spawn_runner(&runner_path.display().to_string(), &sock_name, unique_id)
                    .expect("Failed to spawn runner subprocess. \n\r*****Tips: ensure xinfer is built with './build.sh --install --features cuda,nccl'.");
            }

            let progress_sock_name = format!("{}@xinfer-progress", unique_id);
            let progress_handle =
                spawn_progress_thread(local_num_gpus, config.num_hidden_layers, progress_sock_name);
            heartbeat_worker(Some(device_ids.len()), false, stop_flag.clone(), unique_id);
            use rayon::iter::IndexedParallelIterator;
            use rayon::iter::IntoParallelRefIterator;
            use rayon::iter::ParallelIterator;
            let engine_config = std::sync::OnceLock::<EngineConfig>::new();
            // Collect each runner's reported final num_blocks (the post-MMU-clamp value) +
            // the physical memory view (the live free, the total VRAM), keyed by the
            // local rank. The master's BlockManager uses the min block count across
            // ranks (the no the block-table OOB) + the memory map reconciles the per-
            // rank live free (the rank 0 is probed directly, the other ranks arrive
            // here). The key is the local rank (the device_ids index), not the
            // completion order (the par_iter is non-deterministic, so a Vec would
            // misalign the per-rank reconciliation).
            let reported_blocks: Arc<Mutex<HashMap<usize, (usize, u64, u64)>>> =
                Arc::new(Mutex::new(HashMap::new()));
            let reported_blocks_loop = reported_blocks.clone();
            let runner_streams: Result<Vec<LocalStream>> = device_ids
                .par_iter()
                .enumerate()
                .map(|(local_rank, dev_id)| {
                    let global_rank = global_rank_offset + local_rank;
                    let model_type = model_type.clone();
                    let sock_name = format!("{}@xinfer-runner-{}", unique_id, local_rank);
                    let listener = ListenerOptions::new()
                        .name(
                            sock_name
                                .clone()
                                .to_ns_name::<GenericNamespaced>()
                                .expect("Failed to to_ns_name"),
                        )
                        .create_sync()
                        .expect("Failed to create listener");

                    crate::log_info!(
                        "listener starting accepting runner {} (global rank {})",
                        local_rank,
                        global_rank
                    );

                    // Accept one connection
                    let mut stream = listener.accept()?;
                    crate::log_info!(
                        "Accepted runner {} (global rank {})",
                        local_rank,
                        global_rank
                    );

                    // Wait for "ready"
                    let mut reader = BufReader::new(&mut stream);
                    let mut message = String::new();
                    reader.read_line(&mut message)?;
                    if message.trim() != "ready" {
                        return Err(candle_core::Error::Msg(format!(
                            "Runner {} did not send ready",
                            local_rank
                        )));
                    }

                    // Build init message with global rank for NCCL
                    let is_hybrid_mamba = matches!(
                        model_type,
                        ModelType::Qwen3_5
                            | ModelType::Qwen3_5MoE
                            | ModelType::Qwen4
                            | ModelType::Qwen3VL
                    );
                    let mut econfig_rank = econfig.clone();
                    econfig_rank.device_ids = Some(vec![*dev_id]);
                    let init_msg = MessageType::Init(RunnerInitRequest {
                        rank: global_rank,
                        dev_id: *dev_id,
                        num_shards: econfig.num_shards.unwrap_or(1),
                        model_type,
                        config: config.clone(),
                        econfig: econfig_rank,
                        model_pathes: model_pathes.clone(),
                        is_gguf,
                        dtype: dtype.into(),
                        is_rope_i,
                        #[cfg(feature = "nccl")]
                        nccl_id: NcclId(nccl_id.clone()),
                    });

                    send_and_expect_ack(&mut stream, &init_msg, "initialize", global_rank)?;

                    // Always validate/plan allocation after model loading for accurate memory readings
                    let ecfg = engine_config.get_or_init(|| {
                        let mut econfig = econfig.clone();
                        let allocator = KVCacheAllocator::new(&econfig, &config, dtype);
                        // Propagate resolved dtype (e.g. TurboQuant → Auto on MLA) before plan/IPC.
                        econfig.kvcache_dtype = allocator.resolved_kvcache_dtype();
                        let device_ids = econfig.device_ids.clone().unwrap_or(vec![0]);
                        match allocator.plan(&device_ids, &mut econfig) {
                             Ok(_) => {
                                crate::log_info!("KVCache allocation successfully planned!");
                            }
                            Err(e) => {
                                crate::log_error!("Failed to allocate KVCache: {:?}", e);
                                panic!("Failed to allocate KVCache: {:?}", e);
                            }
                        }
                        // Apply the SAME MMU KV-cache cap the runner applies
                        // (runner.rs), so the master's `num_blocks` (the
                        // scheduler's `BlockManager`) agrees with the runner's
                        // `config.num_blocks` (the flashinfer validation bound).
                        // Without this, the master allocates block IDs up to its
                        // (larger) `num_blocks`, the runner validates against its
                        // (smaller, MMU-capped) `num_blocks`, + a block ID in
                        // [runner_num_blocks, master_num_blocks) is OOB (the
                        // mmu-fail.log flashinfer block-index errorob).
                        let device_ids2 = econfig.device_ids.clone().unwrap_or(vec![0]);
                        let allocator2 = KVCacheAllocator::new(&econfig, &config, dtype);
                        let total_free = allocator2
                            .get_available_memory(&device_ids2)
                            .unwrap_or(0);
// The measured GPU reserve (the single source of truth, shared
                        // with the runner via the allocator). The graph slice is the
                        // model-aware `cuda_graph_bytes` (zero for non-graph builds,
                        // no VRAM idled); the workspace is the model-aware
                        // `compute_workspace_budget` (scales with the actual prefill
                        // chunk + model width, not a percentage of F).
                        let graph_capture_batch = if is_hybrid_mamba {
                            hybrid_mamba_graph_capture_max_batch(econfig.max_num_parallel_reqs)
                        } else {
                            econfig.max_num_parallel_reqs.max(1)
                        };
                        let max_model_len = econfig.max_model_len.unwrap_or(32768);
                        let reserve = GpuReserve::compute(&allocator2, graph_capture_batch, max_model_len);
                        let mut budget = PoolBudget::new(total_free, &reserve);
                        let mut pool_allocs: Vec<(&str, u64)> = Vec::new();
                        pool_allocs.push(("graphs", reserve.graph_bytes));
                        pool_allocs.push(("workspace", reserve.workspace_bytes));
                        if moe_offload_auto() {
                            let moe_granted = budget.cap(budget.remainder());
                            pool_allocs.push(("moe_experts", moe_granted));
                        }
                        if !econfig.lora.adapters.is_empty() {
                            let lora_granted = budget.cap(econfig.lora.max_bytes.unwrap_or(DEFAULT_LORA_MAX_BYTES) as u64);
                            pool_allocs.push(("lora_bank", lora_granted));
                        }
                        if is_hybrid_mamba {
                            let gdn_granted = budget.cap(econfig.mamba_memory_bytes as u64);
                            pool_allocs.push(("gdn_slots", gdn_granted));
                        }
                        apply_kv_cache_cap(&budget, &mut econfig);
                        pool_allocs.push(("kv_cache", budget.remainder()));
                        // The boot-time memory budget (the static /metrics export).
                        Telemetry::global().set_mem_budget(&pool_allocs, total_free, budget.remainder());
                        econfig
                    });

                    send_and_expect_ack(
                        &mut stream,
                        &MessageType::UsableMemoryLeft(ecfg.clone()),
                        "init kvcache",
                        global_rank,
                    )?;

                    // Collect the runner's final (post-MMU-clamp) num_blocks so the
                    // master's BlockManager (the scheduler) uses the same block count
                    // the runner actually allocated (avoiding the block-table out-of-bounds
                    // error from a master/runner num_blocks divergence). The runner reports it via
                    // NumBlocksReport (the runner/runner.rs, sent after the MMU clamp).
let reported = match receive_local(&mut stream, false)? {
                        MessageType::NumBlocksReport {
                            num_blocks,
                            live_free,
                            total_vram,
                        } => Some((num_blocks, live_free, total_vram)),
                        _ => None, // the Thread runner (the in-process) does not report; the
                                   // master's own clamp (above) is used.
                    };
                    if let Some(report) = reported {
                        reported_blocks_loop.lock().unwrap().insert(local_rank, report);
                    }

                    crate::log_info!(
                        "Runner {} (global rank {}) started!",
                        local_rank,
                        global_rank
                    );
                    Ok(stream)
                })
                .collect();

            if let Ok(Some(handle)) = progress_handle.join() {
                let _ = handle.join();
            }
            if let Some(cfg) = engine_config.get() {
                econfig = cfg.clone();
                config.kvcache_dtype = econfig.kvcache_dtype;
            }
            // Clamp the master's num_blocks to the MIN across all runners' reported
            // (post-MMU-clamp) values, so the master's BlockManager (the scheduler)
            // never allocates a block ID the runners' (smaller) pools don't have
            // (avoiding the flashinfer block-table out-of-bounds error from a
            // master/runner divergence).
            let reported = reported_blocks.lock().unwrap();
            // The min block count across the runner reports (the master must not
            // allocate a block ID the runners' smaller pools don't have, the no the
            // flashinfer block-table out-of-bounds). The report is a tuple (the
            // num_blocks, the live free, the total VRAM); the min is by the num_blocks
            // (the first element).
            if let Some(&(min_blocks, _, _)) = reported.values().min_by_key(|(b, _, _)| *b) {
                if econfig.num_blocks > min_blocks {
                    crate::log_info!(
                        "Clamping master num_blocks {} -> {} (the min across runner reports)",
                        econfig.num_blocks,
                        min_blocks
                    );
                    econfig.num_blocks = min_blocks;
                }
            }
            // The per-rank physical view (the live free, the total VRAM) is captured
            // for the memory map reconciliation: the rank 0 is probed directly, the
            // other ranks' live free arrives here. The reports are keyed by the local
            // rank (the device_ids index), so the reconciliation is rank-aligned (the
            // no the completion-order Vec, which misaligned the per-rank maps). The
            // min live free across ranks is the deployment's true idle headroom (the
            // tightest rank bounds a grow).
            let rank_count = reported.len().max(1);
            per_rank_live_free = (0..rank_count)
                .map(|rank| reported.get(&rank).map(|r| r.1).unwrap_or(0))
                .collect();
            per_rank_total_vram = (0..rank_count)
                .map(|rank| reported.get(&rank).map(|r| r.2).unwrap_or(0))
                .collect();
            drop(reported);

            let local_streams = runner_streams?;

            // Multi-node forward coordination setup
            if is_multi_node && node_rank == 0 {
                let mn_config = crate::utils::multi_node::MultiNodeConfig {
                    num_nodes,
                    node_rank,
                    master_addr: econfig.master_addr.clone().unwrap_or_default(),
                    master_port: econfig.master_port,
                    local_num_gpus,
                };
                let worker_tcp_streams = crate::utils::multi_node::master_accept_worker_streams(
                    &mn_config,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Multi-node forward coord failed: {}", e))
                })?;
                RunnerType::MultiNodeMaster {
                    local_streams,
                    remote_streams: worker_tcp_streams,
                }
            } else {
                RunnerType::Process(local_streams)
            }
        };

        if econfig.max_model_len.is_none() {
            println!(
                "\n{} is not given, default to {}, Max usable kvcache tokens {}.\n",
                format!("Warn: max_model_len").yellow().bold(),
                format!("{}", 32768).red().bold(),
                format!("{}", 32768 * econfig.max_num_seqs).red().bold(),
            );
            econfig.max_model_len = Some(32768);
        }
        let runners = Arc::new(RwLock::new(runners));

        let mut scheduler = Scheduler::new(runners.clone(), &econfig, &config);

        // Preserve model-specific tool token detection for non-guided paths.
        let mut tool_config = ToolConfig::for_model_type(&model_type);
        tool_config.validate_with_tokenizer(&tokenizer, &model_type);
        let tool_call_start_ids = tool_config.tool_call_start_ids(&tokenizer);
        let tool_call_end_ids = tool_config.tool_call_end_ids(&tokenizer);

        if !tool_call_start_ids.is_empty() {
            scheduler.set_tool_call_start_tokens(tool_call_start_ids.clone());
            log_info!(
                "Tool call start token IDs set to: {:?}",
                tool_call_start_ids
            );
        } else {
            log_info!("Tool call start token IDs not set (no reliable start token)");
        }

        if !tool_call_end_ids.is_empty() {
            scheduler.set_tool_call_end_tokens(tool_call_end_ids.clone());
            log_info!("Tool call end token IDs set to: {:?}", tool_call_end_ids);
        } else {
            log_info!("Tool call end token IDs not set (no reliable end token)");
        }

        // Set tokenizer for JSON tool call detection (for models like Qwen3 that output raw JSON)
        scheduler.set_tokenizer(Arc::new(tokenizer.clone()));

        log_warn!(
            "KV-cache pool {} tokens ({} blocks x block_size {}). Per-step scheduling budget {} tokens (max {} parallel reqs, max {} reqs). Additional CPU KV Cache blocks {}.",
            econfig.max_kv_cache_tokens,
            econfig.num_blocks,
            econfig.block_size,
            econfig.max_num_batched_tokens,
            econfig.max_num_parallel_reqs,
            econfig.max_num_seqs,
            (econfig.num_blocks as f32 * econfig.cpu_mem_fold.unwrap_or(0.5f32)) as usize
        );

        // The official DeepSeek-V4 `chat` encoding starts generation after
        // `</think>`; thinking mode is opt-in for this model. Other reasoning
        // models retain the existing enabled-by-default behavior.
        let default_enable_thinking =
            !econfig.disable_reasoning && !matches!(model_type, ModelType::DeepSeekV4);
        let mut template = ChatTemplate::new(
            None,
            config_tokenizer.chat_template.clone(),
            config_tokenizer.bos_token.clone(),
            config_tokenizer.eos_token.clone(),
            None,
            true,
            default_enable_thinking,
        );
        let escaped_special_tokens = ChatTemplate::collect_escape_tokens(
            &tokenizer,
            &[&tool_config.start_token_str, &tool_config.end_token_str],
        );
        template.set_escape_tokens(escaped_special_tokens);

        let img_cfg = get_image_config(model_type.clone(), &config)?;
        if let Some(cfg) = img_cfg.as_ref() {
            template.set_preserve_tokens(cfg.prompt_marker_tokens());
        }
        let prompt_replay_candidates = Self::build_prompt_replay_candidates(
            &tokenizer,
            &template,
            &Vec::new(),
            &guidance_tokens,
        );
        let model_name = if let Some(archs) = &config.architectures {
            archs[0].to_string()
        } else {
            "default".to_string()
        };

        log_warn!("Model loaded.\n");

        // The Prometheus telemetry init (the QoS-gated, the no-op when QoS is off).
        // The set_build_info + the set_kv_cache_config publish the static info
        // metrics (the model id, the shard count, the KV config) at startup.
        if qos_enabled() {
            let telemetry = Telemetry::global();
            telemetry.set_build_info(&econfig);
            telemetry.set_kv_cache_config(econfig.num_blocks, econfig.max_kv_cache_tokens);
        }

        // The LoRA startup log (the operator visibility): the loaded
        // adapters + the residency budget. The no-op when no adapter is loaded
        // (the base model only, the zero overhead).
        if !econfig.lora.adapters.is_empty() {
            crate::log_info!(
                "LoRA: {} adapter(s) loaded ({:?}), budget {} tenants / {} bytes, policy {:?}, pinned {:?}",
                econfig.lora.adapters.len(),
                econfig.lora.adapters.iter().map(|(n, _)| n).collect::<Vec<_>>(),
                econfig.lora.max_tenants.unwrap_or(DEFAULT_LORA_MAX_TENANTS),
                econfig.lora.max_bytes.unwrap_or(DEFAULT_LORA_MAX_BYTES),
                econfig.lora.eviction_policy,
                econfig.lora.pinned
            );
        }

// Build the live physical memory map from the finalized econfig (the tier
        // sizes) plus a fresh total-VRAM probe (the driver ground truth, not the
        // boot snapshot). The map is owned by value on the engine (no internal
        // lock); it is mutated under the engine write lock (the grow, the reclaim,
        // the probe refresh) and read by the scheduler and the QoS controller for
        // the physical pressure signal, which the block occupancy is blind to.
        let device_count = econfig.num_shards.unwrap_or(1).max(1);
        let map_allocator = KVCacheAllocator::new(&econfig, &config, dtype);
        let is_hybrid_mamba = matches!(
            model_type,
            ModelType::Qwen3_5
                | ModelType::Qwen3_5MoE
                | ModelType::Qwen4
                | ModelType::Qwen3VL
        );
        let graph_capture_batch = if is_hybrid_mamba {
            hybrid_mamba_graph_capture_max_batch(econfig.max_num_parallel_reqs)
        } else {
            econfig.max_num_parallel_reqs.max(1)
        };
        let max_model_len = econfig.max_model_len.unwrap_or(32768);
        let reserve = GpuReserve::compute(&map_allocator, graph_capture_batch, max_model_len);
        let kv_bytes = econfig.kvcache_memory_bytes as u64;
        let gdn_bytes = econfig.mamba_memory_bytes as u64;
        // The GDN prefix tiers: the device LRU (the mamba_cache_capacity times the
        // prefix multiplier, the hot turn snapshots) plus the CPU spill (four times
        // the device capacity, the cold tier). The slot bytes are the mamba_slot_
        // bytes (the per-snapshot recurrent state). Registering them in the map
        // makes the GDN footprint part of the physical budget, so an untracked
        // GDN allocation cannot silently consume the idle VRAM the KV pool needs.
        // Zero when the model is not hybrid (no GDN tiers).
        let gdn_device_prefix_bytes = if is_hybrid_mamba {
            let device_prefix_capacity = econfig
                .mamba_cache_capacity
                .unwrap_or(0)
                .saturating_mul(2); // the effective_mamba_prefix_capacity (the prefix-cache-on multiplier)
            (device_prefix_capacity as u64).saturating_mul(econfig.mamba_slot_bytes as u64)
        } else {
            0
        };
        let gdn_cpu_spill_bytes = (gdn_device_prefix_bytes as u128)
            .saturating_mul(4) // the CPU_PREFIX_CACHE_CAPACITY_MULTIPLIER (the mamba_cache.rs)
            as u64;
        let cpu_swap_bytes = (econfig.num_blocks as u64)
            .saturating_mul(econfig.cpu_mem_fold.unwrap_or(0.5) as u64)
            .saturating_mul(
                map_allocator
                    .per_block_bytes()
                    .max(1) as u64,
            );
        // The live driver probe (the rank 0 cuMemGetInfo, the ground truth for the
        // idle VRAM). The closure is self-contained: it creates the CudaDevice per
        // call (the driver query is cheap), so it captures no allocator state. The
        // None on the non-CUDA build (the computed-free fallback, not a stale
        // driver value). This makes the map's live_free the physical signal rather
        // than the boot-time snapshot.
        #[cfg(feature = "cuda")]
        let probe: Option<Box<dyn Fn() -> Option<(u64, u64)> + Send + Sync>> = {
            let probe_dev_id = econfig.device_ids.clone().unwrap_or(vec![0])[0];
            Some(Box::new(move || {
                use candle_core::backend::BackendDevice;
                use candle_core::cuda_backend::cudarc::driver::sys;
                use candle_core::cuda_backend::CudaDevice;
                let _ = CudaDevice::new(probe_dev_id).ok()?;
                let mut free: usize = 0;
                let mut total: usize = 0;
                unsafe {
                    sys::lib()
                        .cuMemGetInfo_v2(&mut free as *mut usize, &mut total as *mut usize)
                        .result()
                        .ok()?;
                }
                Some((free as u64, total as u64))
            }))
        };
        #[cfg(not(feature = "cuda"))]
        let probe: Option<Box<dyn Fn() -> Option<(u64, u64)> + Send + Sync>> = None;
        // The host probe (the sysinfo, the host RAM total + available). The no the
        // device probe (the cuMemGetInfo), this fills the host map's total (the
        // physical host RAM, the no the regions sum). The None when sysinfo is
        // unavailable (the computed-free fallback).
        let host_probe: Option<Box<dyn Fn() -> Option<(u64, u64)> + Send + Sync>> = {
            Some(Box::new(|| {
                let system = sysinfo::System::new();
                let total = system.total_memory();
                let available = system.available_memory();
                Some((available, total))
            }))
        };
        let mut mem_map = MemoryMapSet::from_engine_config(
            device_count,
            0, // the total is unknown until the first probe (the physical VRAM, the no the boot free)
            0, // the weights are untracked; they show up as drift and keep the grow conservative
            reserve.graph_bytes,
            reserve.workspace_bytes,
            kv_bytes,
            gdn_bytes,
            gdn_device_prefix_bytes,
            // The host swap + the GDN spill are per-rank allocations on the shared
            // host DRAM (each rank's runner allocates its own CPU pool). The host
            // map shows the deployment total (the device_count times the per-rank).
            (cpu_swap_bytes as u128).saturating_mul(device_count as u128) as u64,
            (gdn_cpu_spill_bytes as u128).saturating_mul(device_count as u128) as u64,
            0, // the PLE table is host-resident, not a device region
            probe,
            host_probe,
        );
        // The first probe (the rank 0, the direct driver query) fills the total
        // (the physical VRAM) + the live_free (the current free). The no-probe
        // builds (the non-CUDA) leave the total at 0 (the computed free is the
        // fallback, the pressure is 1.0, the conservative).
        mem_map.refresh_all();
        crate::log_info!("{}", mem_map.report());
        // The per-rank reconciliation (the C7): the rank 0 map is probed directly
        // (the probe closure), the other ranks' live free + total arrive via the
        // NumBlocksReport (the per_rank_live_free, the per_rank_total_vram). Set
        // each rank's cached live_free so the map's aggregate pressure + idle reflect
        // the true per-rank physical state (the no the rank 0 probe only).
        for (rank, map) in mem_map.devices.iter_mut().enumerate() {
            if rank < per_rank_live_free.len() {
                map.live_free = per_rank_live_free[rank];
            }
            if rank < per_rank_total_vram.len() && per_rank_total_vram[rank] > 0 {
                map.total = per_rank_total_vram[rank];
            }
        }

        let engine = Arc::new(RwLock::new(Self {
            runners,
            scheduler,
            tokenizer,
            econfig,
            default_chat_template,
            template,
            prompt_replay_candidates,
            stream_decoders: HashMap::new(),
            stream_senders: HashMap::new(),
            request_types: HashMap::new(),
            decode_start_times: HashMap::new(),
            decode_length: HashMap::new(),
            seq_prefilled_reasoning_end: HashMap::new(),
            seq_prompt_replays: HashMap::new(),
            last_check_throughput_time: 0,
            active_requests: HashSet::new(),
            cancelled_sequences: Vec::new(),
            stop_flag: stop_flag.clone(),
            has_vision: config.is_multi_model.unwrap_or(false),
            model_type,
            tool_config,
            img_cfg,
            model_name,
            guidance_tokens,
            mem_map,
            seq_spec_stats: std::sync::Mutex::new(HashMap::new()),
            #[cfg(all(feature = "cuda", feature = "graph"))]
            graph_pool_cache: std::sync::Mutex::new((0, 0, std::time::Instant::now())),
            loop_detector: LoopDetector::new(),
            kick_store: KickStore::new(),
            seq_loop_stats: std::collections::HashMap::new(),
            guard_step_counts: std::collections::HashMap::new(),
        }));

        Self::start_engine(engine.clone());
        Ok(engine)
    }

    fn add_request_(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: &RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize)> {
        let tokens = self
            .tokenizer
            .encode_fast(prompt, true)
            .expect("encode failed!");
        let token_ids: Vec<u32> = tokens.get_ids().to_vec();
        self.add_request_pretokenized_(params, prompt, token_ids, request_type, images, image_idx)
    }

    /// Same body as `add_request_` but accepts a pre-computed `token_ids`
    /// so the caller can run tokenize outside the engine write lock. Hot-path
    /// serialization of concurrent server handlers gets cut by ~
    /// chat-template + tokenize wall × concurrency in chat-template-heavy
    /// workloads.
    fn add_request_pretokenized_(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        token_ids: Vec<u32>,
        request_type: &RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize)> {
        let length = token_ids.len();
        let raw_replay_token_ids = self.match_prompt_replay_candidate(&token_ids);
        if let Some(max_model_len) = self.econfig.max_model_len {
            if length > max_model_len - 1 {
                candle_core::bail!(
                    "Inputs token length {} exceed max_model_len {}",
                    length,
                    max_model_len
                );
            }
        }
        let mut params = params.clone();

        let mut max_tokens = if let Some(max_t) = params.max_tokens {
            max_t
        } else {
            let default_max = self.econfig.max_tokens.unwrap_or(16384);
            params.max_tokens = Some(default_max);
            default_max
        };
        let requested_max_tokens = max_tokens;

        let max_model_len = self.econfig.max_model_len.unwrap_or(max_tokens);
        // The prompt length is accounted against the context bound. When the
        // operator did not set max_model_len (the fallback is the decode budget),
        // a prompt longer than the budget leaves no room to decode. Use
        // saturating_sub so the comparison never underflows, and reject the
        // request when the prompt consumes the whole context (no room left for
        // generated tokens).
        if length >= max_model_len {
            candle_core::bail!(
                "Inputs token length {} leaves no room for generated tokens within max_model_len {}",
                length,
                max_model_len
            );
        }
        if length.saturating_add(max_tokens) > max_model_len {
            max_tokens = max_model_len - length;
            params.max_tokens = Some(max_tokens);
            log_warn!(
                "Adjusted max_tokens to {} to fit within max_model_len {} (with prompt length {}, requested max_tokens {})",
                max_tokens,
                max_model_len,
                length,
                requested_max_tokens
            );
        }
        if max_tokens == 0 {
            candle_core::bail!(
                "Inputs token length {} leaves no room for generated tokens within max_model_len {}",
                length,
                max_model_len
            );
        }

        if let Some(gen_cfg) = &self.econfig.generation_cfg {
            let temperature = params.temperature.or(gen_cfg.temperature);
            let top_k = params.top_k.or(gen_cfg.top_k);
            let top_p = params.top_p.or(gen_cfg.top_p);
            let frequency_penalty = params.frequency_penalty.or(gen_cfg.frequency_penalty);
            let presence_penalty = params.presence_penalty.or(gen_cfg.presence_penalty);
            params.temperature = temperature;
            params.top_k = top_k;
            params.top_p = top_p;
            params.frequency_penalty = frequency_penalty;
            params.presence_penalty = presence_penalty;
        }

        if let Some(stop_sequences) = &params.stop_sequences {
            let mut stop_token_ids = Vec::new();
            let mut resolved_stop_sequences = Vec::new();
            for sequence in stop_sequences {
                if sequence.is_empty() {
                    continue;
                }
                match self.tokenizer.encode(sequence.as_str(), false) {
                    Ok(encoding) => {
                        let ids = encoding.get_ids();
                        if !ids.is_empty() {
                            stop_token_ids.push(ids.to_vec());
                            resolved_stop_sequences.push(sequence.clone());
                        }
                    }
                    Err(err) => {
                        crate::log_warn!(
                            "Failed to encode stop sequence '{}': {:?}",
                            sequence,
                            err
                        );
                    }
                }
            }
            if !stop_token_ids.is_empty() {
                params.stop_token_ids = Some(stop_token_ids);
                params.stop_sequences = Some(resolved_stop_sequences);
            }
        }
        let seq = Sequence::new(
            token_ids,
            self.econfig.block_size,
            params,
            images,
            image_idx,
        );

        let prompt_required_blocks = self.scheduler.block_manager.required_blocks(&seq);
        // The live demand is the first prefill chunk, not the whole prompt. The
        // chunked prefill (the scheduler re-queues the sequence per chunk)
        // allocates the prompt incrementally, so admission only needs room for one
        // chunk plus the decode budget. The prefix-netted prompt blocks are capped
        // at the chunk size: a cold huge-context prompt is admitted on its first
        // chunk, the rest is prefilled incrementally. The cached case (the iterative
        // agentic loop) already has a small prompt block count (the uncached tail),
        // so the cap is a no-op there.
        let chunk_blocks = self
            .econfig
            .effective_prefill_chunk_size()
            .div_ceil(self.econfig.block_size.max(1));
        let live_prompt_blocks = prompt_required_blocks.min(chunk_blocks.max(1));
        let requested_decode_blocks = max_tokens.div_ceil(self.econfig.block_size);
        let minimum_decode_budget_tokens = max_tokens.min(REQUEST_ADMISSION_DECODE_BUDGET_TOKENS);
        let minimum_decode_blocks = minimum_decode_budget_tokens.div_ceil(self.econfig.block_size);
        let mut target_required_blocks =
            live_prompt_blocks.saturating_add(requested_decode_blocks);
        let mut minimum_required_blocks =
            live_prompt_blocks.saturating_add(minimum_decode_blocks);
        let mut available_blocks = self.scheduler.block_manager.get_num_free_blocks();

        while target_required_blocks > available_blocks {
            let prev_target = target_required_blocks;
            let target_required_tokens = target_required_blocks * self.econfig.block_size;
            // Free blocks in order: evict the unrelated prefix leaves (the reference-count
            // zero only, not the request's own match), then demote the coldest
            // running suffix to host (the prefix stays on the GPU as the join key),
            // then grow the KV tier from the idle VRAM. The prior eviction-only loop
            // could not free the reference-held active blocks, which stalled the
            // admission.
            let evicted = self
                .scheduler
                .evict_prefix_cache_until_free(target_required_blocks);
            if evicted > 0 {
                crate::log_warn!(
                    "Evicted {} prefix cache block(s) to reserve {} KV tokens for request admission (prompt + {} requested decode tokens).",
                    evicted,
                    target_required_tokens,
                    max_tokens
                );
            }
            if target_required_blocks > self.scheduler.block_manager.get_num_free_blocks() {
                // Demote the coldest running suffix to host (the prefix stays on the GPU as
                // the join key, the GDN boundary is captured). The grow lever (the
                // map grow plus the block pool reallocation) requires the runner to
                // actually allocate the extra blocks, so it is not wired here; the
                // demote is the relief that works with the existing primitives.
                let demoted = self.scheduler.demote_for_admission(target_required_blocks);
                if demoted > 0 {
                    crate::log_warn!(
                        "Demoted {} KV block(s) to host (the coldest running suffix) for request admission.",
                        demoted
                    );
                }
            }
            if evicted == 0 && !self.try_release_cache(target_required_tokens) {
                break;
            }
            // Recompute the demand using the live (chunk-capped) prompt blocks, not the
            // whole prompt. Evicting the prefix cache shrinks the match, so the
            // demand must be re-derived from the current match, or the loop would
            // keep the stale (larger) figure and never converge.
            let prompt_required_blocks = self.scheduler.block_manager.required_blocks(&seq);
            let chunk_blocks = self
                .econfig
                .effective_prefill_chunk_size()
                .div_ceil(self.econfig.block_size.max(1));
            let live_prompt_blocks = prompt_required_blocks.min(chunk_blocks.max(1));
            target_required_blocks = live_prompt_blocks.saturating_add(requested_decode_blocks);
            minimum_required_blocks = live_prompt_blocks.saturating_add(minimum_decode_blocks);
            available_blocks = self.scheduler.block_manager.get_num_free_blocks();

            if evicted > 0 && target_required_blocks >= prev_target {
                break;
            }
        }

        if minimum_required_blocks > available_blocks {
            let available_tokens = available_blocks * self.econfig.block_size;
            let required_tokens = minimum_required_blocks * self.econfig.block_size;
            // Reject with the exact shortfall and the physical context, not a bare "try
            // later". The operator needs to know how many tokens are free, how many
            // are needed, how much VRAM is idle, and what to change (grow the pool,
            // raise the context headroom, or trim the conversation).
            let idle_gb = self.mem_map.idle_device() as f64 / (1024.0 * 1024.0 * 1024.0);
            let pressure = self.mem_map.device_pressure();
            candle_core::bail!(
                "KV pool exhausted: {} tokens free, request needs {} tokens (prompt plus {} decode budget). Idle device VRAM {:.2} GB (pressure {:.0}%). To admit: grow the KV pool, raise the max_model_len headroom, or trim the conversation (the {} uncached prompt tokens exceed the pool).",
                available_tokens,
                required_tokens,
                minimum_decode_budget_tokens,
                idle_gb,
                pressure * 100.0,
                length
            );
        }
        if target_required_blocks > available_blocks {
            crate::log_warn!(
                "Request admitted with {} KV tokens available, below requested reservation {} tokens but enough for prompt plus {} decode budget tokens.",
                available_blocks * self.econfig.block_size,
                target_required_blocks * self.econfig.block_size,
                minimum_decode_budget_tokens
            );
        }
        let seq_id = self.scheduler.add(seq);
        if let Some(end_marker) = detect_prefilled_reasoning_end_marker(prompt) {
            self.seq_prefilled_reasoning_end.insert(seq_id, end_marker);
        }
        if let Some(replay_ids) = raw_replay_token_ids {
            self.seq_prompt_replays.insert(seq_id, replay_ids);
        }

        if *request_type == RequestType::Stream {
            let tokenizer = self.tokenizer.clone();
            let leaked: &'static _ = Box::leak(Box::new(tokenizer));
            let decoder = leaked.decode_stream(false);
            let wrapped = super::StreamWithTokenizer {
                _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
                stream: decoder,
            };
            let boxed_decoder: Box<dyn super::DecodeStreamTrait + Send + Sync> = Box::new(wrapped);
            self.stream_decoders.insert(seq_id, boxed_decoder);
        }
        self.active_requests.insert(seq_id);
        Ok((seq_id, length))
    }

    pub fn add_request(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        request_type: RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) =
            self.add_request_(params, prompt, &request_type, images, image_idx)?;
        let (tx, rx) = channel(1024);
        self.stream_senders.insert(seq_id, tx);
        self.request_types.insert(seq_id, request_type.clone());
        if self.econfig.server_mode.unwrap_or(true) && request_type != RequestType::Completion {
            log_info!(
                "[{:?}] New request [Seq_id {}, {} tokens] received! (session_id: {:?})\n",
                request_type,
                seq_id,
                prompt_length,
                params.session_id,
            );
        }
        Ok((seq_id, prompt_length, rx))
    }

    /// Pretokenized variant of `add_request`. Caller has already done
    /// tokenize (typically via `preprocess`) outside the engine write
    /// lock, so the only work that needs `&mut self` is the scheduler-add.
    pub fn add_request_pretokenized(
        &mut self,
        params: &SamplingParams,
        prompt: &str,
        token_ids: Vec<u32>,
        request_type: RequestType,
        images: &Option<ImageData>,
        image_idx: i32,
    ) -> Result<(usize, usize, Receiver<StreamItem>)> {
        let (seq_id, prompt_length) = self.add_request_pretokenized_(
            params,
            prompt,
            token_ids,
            &request_type,
            images,
            image_idx,
        )?;
        let (tx, rx) = channel(1024);
        self.stream_senders.insert(seq_id, tx);
        self.request_types.insert(seq_id, request_type.clone());
        if self.econfig.server_mode.unwrap_or(true) && request_type != RequestType::Completion {
            log_info!(
                "[{:?}] New request [Seq_id {}, {} tokens] received! (session_id: {:?})\n",
                request_type,
                seq_id,
                prompt_length,
                params.session_id,
            );
        }
        Ok((seq_id, prompt_length, rx))
    }

    /// Apply chat template + tokenize each `(params, messages)` pair under
    /// shared access (`&self`) so N concurrent server handlers can run this
    /// in parallel under their own `engine.read()` guards. Returns a list
    /// of `PreprocessedRequest`s that feed `generate_sync`
    /// or `generate_stream`, which acquire `engine.write()`
    /// only briefly for the scheduler-add step.
    pub fn preprocess(
        &self,
        params: &[SamplingParams],
        message_list: &[Vec<Message>],
        tools: &[Tool],
        log: bool,
    ) -> Result<Vec<PreprocessedRequest>> {
        if params.len() != message_list.len() {
            candle_core::bail!("size of sampling parameters is not match with size of prompts!");
        }
        let tools_vec: Vec<Tool> = tools.to_vec();
        let mut out = Vec::with_capacity(params.len());
        for (param, messages) in params.iter().zip(message_list.iter()) {
            let (prompt, image_idx) = self.apply_chat_template(param, messages, &tools_vec, log);
            let tokens = self
                .tokenizer
                .encode_fast(prompt.as_str(), true)
                .expect("encode failed!");
            let token_ids: Vec<u32> = tokens.get_ids().to_vec();
            out.push(PreprocessedRequest {
                params: param.clone(),
                prompt,
                image_idx,
                token_ids,
            });
        }
        Ok(out)
    }

    pub fn get_num_cached_tokens(&self) -> usize {
        self.scheduler.get_num_cached_tokens()
    }

    /// See [`Scheduler::get_num_cached_tokens_for_seq`].
    pub fn get_num_cached_tokens_for_seq(&self, seq_id: usize) -> Option<usize> {
        self.scheduler.get_num_cached_tokens_for_seq(seq_id)
    }

    fn trim_prompt_replay_prefix(
        replay_ids: &[u32],
        guidance_tokens: &GuidanceTokens,
    ) -> Option<Vec<u32>> {
        let start_idx = replay_ids
            .iter()
            .position(|token_id| guidance_tokens.reasoning_start_ids.contains(token_id))?;
        Some(replay_ids[start_idx..].to_vec())
    }

    fn build_prompt_replay_candidates(
        tokenizer: &Tokenizer,
        template: &ChatTemplate,
        tools: &Vec<Tool>,
        guidance_tokens: &GuidanceTokens,
    ) -> Vec<Vec<u32>> {
        let synthetic_messages = vec![Message {
            role: "user".to_string(),
            content: "__XINFER_REPLAY_PROBE__".to_string(),
            num_images: 0,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        let mut candidates = Vec::new();

        for enable_thinking in [true, false] {
            let mut replay_template = template.clone();
            replay_template.set_enable_thinking(enable_thinking);
            replay_template.set_messages(&synthetic_messages);
            let rendered = match replay_template.apply_chat_template(tools, false) {
                Ok(rendered) => rendered,
                Err(_) => continue,
            };
            let Some(replay_suffix) =
                replay_template.generation_prompt_replay_suffix(tools, &rendered)
            else {
                continue;
            };
            let Ok(encoding) = tokenizer.encode_fast(replay_suffix.as_str(), true) else {
                continue;
            };
            let ids = encoding.get_ids().to_vec();
            if let Some(replay_ids) = Self::trim_prompt_replay_prefix(&ids, guidance_tokens) {
                // crate::log_info!(
                //     "Missing suffix detected {} -> {:?}",
                //     replay_suffix,
                //     replay_ids
                // );
                candidates.push(replay_ids);
            }
        }

        candidates.sort_by_key(|ids| std::cmp::Reverse(ids.len()));
        candidates.dedup();
        candidates
    }

    fn match_prompt_replay_candidate(&self, prompt_token_ids: &[u32]) -> Option<Vec<u32>> {
        self.prompt_replay_candidates
            .iter()
            .find(|candidate| prompt_token_ids.ends_with(candidate.as_slice()))
            .cloned()
    }

pub fn get_available_kv_tokens(&self) -> usize {
        self.scheduler.get_available_kv_tokens()
    }

fn take_prompt_replay_token_ids(&mut self, seq_id: usize) -> Option<Vec<u32>> {
        self.seq_prompt_replays.remove(&seq_id)
    }

pub fn notify_runner_finished(&mut self, id: usize) -> Result<()> {
        // The author's singular finish path: ONE fire-and-forget `FinishDecode` per
        // runner (the no a second, response-bearing message on the finish socket). The
        // runner's `FinishDecode` handler is idempotent (the `spec_seq_report` drop +
        // the `runner.finished` release, both no-ops on a repeat), so a seq notified
        // from two callers (the engine `finish_step` + the server stream-end handler)
        // is safe -- the second `FinishDecode` is absorbed without a response (the no a
        // request/ack desync).
        //
        // The per-seq spec stats + the per-phase token tally are NOT pulled here (the
        // prior `GetSpecSeqStats` finish-time pull was the second, response-bearing
        // message that desynced the framing). They are cached in-process by
        // `finish_step` from the per-step `GateCommitResponse` (the stats ride the
        // message that already crosses the socket every decode step), so the master
        // reads its own cache for the end-of-seq report.
        match &mut *self.runners.write() {
            RunnerType::Thread(model_runner) => {
                // The in-process Thread runner shares this process's SPEC_SEQ_STATS
                // global, so a direct read is correct (the no a socket pull).
                let spec_stats = spec_seq_stats_data(id);
                if !spec_stats.mechanism.is_empty() {
                    self.seq_spec_stats.lock().unwrap().insert(id, spec_stats);
                }
                return Ok(model_runner.finished(id));
            }
            RunnerType::Process(ref mut runner_streams) => {
                // The spec stats + phase counts are already cached in-process by
                // `finish_step` (the per-step `GateCommitResponse`). The finish path
                // sends ONLY the fire-and-forget `FinishDecode` (the no a second,
                // response-bearing pull -- that second message is what desynced the
                // request/ack framing).
                for stream in runner_streams {
                    send_local(
                        &mut vec![stream.try_clone()?],
                        &MessageType::FinishDecode(id),
                        false,
                    )?;
                }
                return Ok(());
            }
            RunnerType::MultiNodeMaster {
                ref mut local_streams,
                ref mut remote_streams,
            } => {
                // The spec stats + phase counts are already cached in-process by
                // `finish_step` (the per-step `GateCommitResponse`). The finish path
                // sends ONLY the fire-and-forget `FinishDecode` (the no a second,
                // response-bearing pull -- that second message is what desynced the
                // request/ack framing).
                let msg = MessageType::FinishDecode(id);
                for stream in local_streams.iter_mut() {
                    send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                }
                let serialized = rmp_serde::to_vec(&msg).expect("MsgPack serialization failed");
                for tcp_stream in remote_streams.iter_mut() {
                    send_tcp(tcp_stream, &serialized)?;
                }
                return Ok(());
            }
        }
    }


    /// The end-of-sequence scheduling report (cached tokens + QoS stats) for the
    /// server's per-sequence display. Delegates to the scheduler (same process).
    pub fn get_seq_report(&self, seq_id: usize) -> Option<SeqReport> {
        self.scheduler.get_seq_report(seq_id)
    }

// The llg anti-loop methods (the get_seq_loop_stats + the observe_loop). The
    // fetch_spec_seq_stats is dropped (the §7.3 capture-before-drop: the per-seq spec
    // stats now come from the SchedD-owned map via GetSpecStats, the no runner fetch).
    /// The cached per-seq loop-defense stats (for the server's end-of-sequence report).
    pub fn get_seq_loop_stats(&self, seq_id: usize) -> Option<LoopSeqStats> {
        self.seq_loop_stats.get(&seq_id).cloned()
    }

    /// The cached per-seq spec stats (for the server's end-of-sequence report).
    pub fn get_seq_spec_stats(&self, seq_id: usize) -> Option<SpecSeqStatsData> {
        self.seq_spec_stats.lock().unwrap().get(&seq_id).cloned()
    }

    /// The llg sealed-gate: commit each produced run to the grammar FSM + keep only
    /// the FSM-passing prefix, just before the sequence append. The `ff` selects the
    /// spec-FF ordered-queue writer. For the IPC/multi-node transports the gate runs
    /// on the remote runner (which owns the FSM), so this is a `GateCommit` RPC; for
    /// the in-process Thread runner it is a direct `ModelRunner::gate_commit` call.
    fn gate_commit(
        runners: &Arc<RwLock<RunnerType>>,
        seq_ids: &[usize],
        runs: &[Vec<u32>],
        ff: bool,
        seq_infos: &[crate::speculative::metadata::SpecSeqInfo],
    ) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<SpecSeqStatsData>, Vec<[usize; 3]>) {
        match &mut *runners.write() {
            RunnerType::Thread(model_runner) => {
                // The in-process Thread runner: the spec stats + phase counts are read
                // directly (the no a socket round-trip).
                let (gated, reasoning_deltas, ff_deltas, ff_drafted, mask_fail_streaks) =
                    model_runner.gate_commit(seq_ids, runs, ff, seq_infos);
                let spec_stats: Vec<SpecSeqStatsData> =
                    seq_ids.iter().map(|&sid| spec_seq_stats_data(sid)).collect();
                let phase_counts: Vec<[usize; 3]> =
                    seq_ids.iter().map(|&sid| model_runner.phase_token_counts(sid)).collect();
                (gated, reasoning_deltas, ff_deltas, ff_drafted, mask_fail_streaks, spec_stats, phase_counts)
            }
            RunnerType::Process(streams) => {
                use crate::runner::MessageType;
                let request = MessageType::GateCommit((seq_ids.to_vec(), runs.to_vec(), ff, seq_infos.to_vec()));
                let cloned: Vec<_> = streams.iter_mut().map(|s| s.try_clone().expect("clone failed")).collect();
                let all: Result<
                    Vec<(
                        Vec<Vec<u32>>,
                        Vec<usize>,
                        Vec<usize>,
                        Vec<usize>,
                        Vec<usize>,
                        Vec<SpecSeqStatsData>,
                        Vec<[usize; 3]>,
                    )>,
                > = cloned
                    .into_par_iter()
                    .map(|mut stream| {
                        send_local(&mut vec![stream.try_clone()?], &request, false)?;
                        match receive_local(&mut stream, false)? {
                            MessageType::GateCommitResponse(
                                gated,
                                deltas,
                                ff_deltas,
                                ff_drafted,
                                mask_fail_streaks,
                                spec_stats,
                                phase_counts,
                            ) => Ok((
                                gated,
                                deltas,
                                ff_deltas,
                                ff_drafted,
                                mask_fail_streaks,
                                spec_stats,
                                phase_counts,
                            )),
                            other => candle_core::bail!("Unexpected GateCommit response: {:?}", other),
                        }
                    })
                    .collect();
                match all {
                    Ok(res) => res.into_iter().next().unwrap_or_else(|| {
                        (
                            runs.to_vec(),
                            vec![0usize; seq_ids.len()],
                            vec![0usize; seq_ids.len()],
                            vec![0usize; seq_ids.len()],
                            vec![0usize; seq_ids.len()],
                            Vec::new(),
                            Vec::new(),
                        )
                    }),
                    Err(_) => (
                        runs.to_vec(),
                        vec![0usize; seq_ids.len()],
                        vec![0usize; seq_ids.len()],
                        vec![0usize; seq_ids.len()],
                        vec![0usize; seq_ids.len()],
                        Vec::new(),
                        Vec::new(),
                    ), // the gate unavailable: pass through
                }
            }
            RunnerType::MultiNodeMaster { local_streams, remote_streams } => {
                use crate::runner::MessageType;
                let request = MessageType::GateCommit((seq_ids.to_vec(), runs.to_vec(), ff, seq_infos.to_vec()));
                let serialized = rmp_serde::to_vec(&request).expect("MsgPack serialization failed");
                let send_ok = remote_streams.iter_mut().all(|stream| {
                    send_tcp(stream, &serialized).is_ok()
                });
                if !send_ok {
                    return (
                        runs.to_vec(),
                        vec![0usize; seq_ids.len()],
                        vec![0usize; seq_ids.len()],
                        vec![0usize; seq_ids.len()],
                        vec![0usize; seq_ids.len()],
                        Vec::new(),
                        Vec::new(),
                    ); // the gate unavailable: pass through
                }
                let result = Self::gate_commit_local(local_streams, &request);
                for stream in remote_streams.iter_mut() {
                    let _ = recv_tcp(stream);
                }
                result
            }
        }
    }

    fn gate_commit_local(
        runner_streams: &mut Vec<LocalStream>,
        request: &MessageType,
    ) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<SpecSeqStatsData>, Vec<[usize; 3]>) {
        use crate::runner::MessageType;
        let cloned: Vec<LocalStream> = runner_streams.iter_mut().map(|s| s.try_clone().expect("clone failed")).collect();
        let all: Result<
            Vec<(
                Vec<Vec<u32>>,
                Vec<usize>,
                Vec<usize>,
                Vec<usize>,
                Vec<usize>,
                Vec<SpecSeqStatsData>,
                Vec<[usize; 3]>,
            )>,
        > = cloned
            .into_par_iter()
            .map(|mut stream| {
                send_local(&mut vec![stream.try_clone()?], request, false)?;
                match receive_local(&mut stream, false)? {
                    MessageType::GateCommitResponse(
                        gated,
                        deltas,
                        ff_deltas,
                        ff_drafted,
                        mask_fail_streaks,
                        spec_stats,
                        phase_counts,
                    ) => Ok((
                        gated,
                        deltas,
                        ff_deltas,
                        ff_drafted,
                        mask_fail_streaks,
                        spec_stats,
                        phase_counts,
                    )),
                    other => candle_core::bail!("Unexpected GateCommit response: {:?}", other),
                }
            })
            .collect();
        match all {
            Ok(res) => res.into_iter().next().unwrap_or_else(|| {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
            }),
            Err(_) => (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
        }
    }

    /// The anti-loop OBSERVE: check the accepted tokens for a closed loop.
    /// Zero-overhead when XINFER_ANTI_LOOP is off (the default).
    fn observe_loop(&mut self, seq_id: usize, accepted: &[u32], ff_forced: usize) {
        use crate::utils::loop_detect::LoopSeqStats;
        let probe_depth = repetition_probe_depth();
        let was_active = self.kick_store.is_active(seq_id);
        // The guard duration tracking: while a kick is held, count the steps.
        if was_active {
            *self.guard_step_counts.entry(seq_id).or_insert(0) += 1;
            if let Some(stats) = self.seq_loop_stats.get_mut(&seq_id) {
                stats.guard_steps += 1;
            }
        }
        match self.loop_detector.observe(seq_id, accepted, probe_depth, ff_forced) {
            Some(span) => {
                self.kick_store.apply(seq_id, &span);
                let kick = self.kick_store.get(seq_id).expect("just applied");
                let stats = self.seq_loop_stats.entry(seq_id).or_insert_with(LoopSeqStats::default);
                stats.guards += 1;
                if !stats.periods.contains(&span.period) {
                    stats.periods.push(span.period);
                }
                crate::log_warn!(
                    "[llg] [Seq {}] loop detected: period={} unit={:?} -> mitigation applied (exclude {} tokens)",
                    seq_id, span.period, kick.unit, kick.unit.len()
                );
                // Bridge the kick to the runner (the sampling constraint). The FULL unit
                // (the no single forbid token) -- the whole period-P cycle is excluded
                // at the trigger position to break the loop.
                self.send_loop_kick(seq_id, kick.unit.clone(), kick.seed.clone(), span.period);
            }
            None => {
                if was_active {
                    self.kick_store.remove(seq_id);
                    crate::log_info!("[llg] [Seq {}] loop cleared; kick removed", seq_id);
                    self.clear_loop_kick(seq_id);
                }
            }
        }
    }

    /// Bridge the anti-loop kick to the runner's sampler (the `SetLoopKick` IPC, the
    /// no the actor `FwdMessage::SetLoopKick`). Fire-and-forget.
    fn send_loop_kick(&self, seq_id: usize, unit: Vec<u32>, seed: Vec<u32>, period: u32) {
        use crate::runner::MessageType;
        let request = MessageType::SetLoopKick((seq_id, unit, seed, period));
        let mut result: std::result::Result<(), std::io::Error> = Ok(());
        match &mut *self.runners.write() {
            RunnerType::Thread(_) => {} // the in-process: the runner's kick_store is applied directly via gate_commit
            RunnerType::Process(streams) => {
                for stream in streams.iter_mut() {
                    if result.is_err() { break; }
                    result = send_local(&mut vec![stream.try_clone().expect("clone failed")], &request, false);
                }
            }
            RunnerType::MultiNodeMaster { local_streams, remote_streams } => {
                let serialized = rmp_serde::to_vec(&request).expect("MsgPack serialization failed");
                for stream in remote_streams.iter_mut() {
                    if result.is_err() { break; }
                    result = send_tcp(stream, &serialized);
                }
                for stream in local_streams.iter_mut() {
                    if result.is_err() { break; }
                    result = send_local(&mut vec![stream.try_clone().expect("clone failed")], &request, false);
                }
            }
        }
        if let Err(e) = result {
            crate::log_error!("send_loop_kick failed: {e}");
        }
    }

    /// Clear the anti-loop kick for a sequence (the `ClearLoopKick` IPC, the no the
    /// actor `FwdMessage::ClearLoopKick`). Fire-and-forget.
    fn clear_loop_kick(&self, seq_id: usize) {
        use crate::runner::MessageType;
        let request = MessageType::ClearLoopKick(seq_id);
        let mut result: std::result::Result<(), std::io::Error> = Ok(());
        match &mut *self.runners.write() {
            RunnerType::Thread(_) => {}
            RunnerType::Process(streams) => {
                for stream in streams.iter_mut() {
                    if result.is_err() { break; }
                    result = send_local(&mut vec![stream.try_clone().expect("clone failed")], &request, false);
                }
            }
            RunnerType::MultiNodeMaster { local_streams, remote_streams } => {
                let serialized = rmp_serde::to_vec(&request).expect("MsgPack serialization failed");
                for stream in remote_streams.iter_mut() {
                    if result.is_err() { break; }
                    result = send_tcp(stream, &serialized);
                }
                for stream in local_streams.iter_mut() {
                    if result.is_err() { break; }
                    result = send_local(&mut vec![stream.try_clone().expect("clone failed")], &request, false);
                }
            }
        }
        if let Err(e) = result {
            crate::log_error!("clear_loop_kick failed: {e}");
        }
    }

    /// Release hybrid GDN/Mamba active slots queued by scheduler-side abort paths.
    pub fn drain_pending_runner_releases(&mut self) {
        let pending = self.scheduler.take_pending_runner_releases();
        for seq_id in pending {
            // Route scheduler-side aborts through the normal cancellation path
            // so the request bookkeeping and response channel are cleaned up as
            // well as the runner slot. Avoid a second notification when the
            // request was already cancelled by another path.
            if !self.cancelled_sequences.contains(&seq_id) {
                self.cancelled_sequences.push(seq_id);
            }
        }
        if !self.cancelled_sequences.is_empty() {
            self.check_canceled(None);
        }
    }

    /// Phase 1: Schedule sequences and prepare data for the forward pass.
    /// Returns (scheduled_ids, is_prefill, cloned_sequences) or None if no work.
    pub fn prepare_step(&mut self) -> Result<Option<(Vec<usize>, bool, Vec<Sequence>)>> {
        // The QoS KV demote actuator reads the physical memory pressure (the map's
        // device_pressure, the VRAM-in-use fraction), not the block occupancy (which
        // is blind to the idle VRAM). Set it on the scheduler before the schedule, so
        // the calibrated kv_demote_threshold triggers on the true physical signal.
        self.scheduler.set_mmu_view(
            self.mem_map.device_pressure(),
            self.mem_map.idle_device(),
            self.mem_map.devices.first().map(|m| m.total).unwrap_or(0),
        );
        let (scheduled_ids, is_prefill) = match self.scheduler.schedule() {
            Ok((ids, prefill)) => (ids, prefill),
            Err(e) => {
                crate::log_error!("[Scheduler] Schedule error: {:?}", e);
                (vec![], true)
            }
        };
        if scheduled_ids.is_empty() {
            return Ok(None);
        }

        if is_prefill {
            let downgraded = self
                .scheduler
                .fallback_missing_mamba_prefix_snapshots(&scheduled_ids)?;
            if downgraded > 0 {
                crate::log_warn!(
                    "Prefill fallback: {} sequence(s) downgraded to full prefill due to missing mamba snapshots.",
                    downgraded
                );
            }
        }

        // The single-writer step view (the no write-after-snapshot): the Scheduler does
        // ALL the writes (the speculative pre-alloc) + the freeze (the Sequence snapshot) in ONE
        // atomic call. No caller can interleave a write between the snapshot and the forward. The
        // gate is the speculative decode ENABLED + the decode step (NOT the single-sequence check).
        let owned_seqs = self.scheduler.prepare_step_view(&scheduled_ids, is_prefill);
        Ok(Some((scheduled_ids, is_prefill, owned_seqs)))
    }

    /// Phase 2: Run the forward pass. Does NOT require the engine lock - only the runner lock.
    pub fn run_forward(
        runners: &Arc<RwLock<RunnerType>>,
        owned_seqs: &[Sequence],
        is_prefill: bool,
    ) -> Result<Vec<u32>> {
        match &mut *runners.write() {
            RunnerType::Thread(model_runner) => {
                let seq_refs: Vec<&Sequence> = owned_seqs.iter().collect();
                model_runner.run(Seqs::SeqRefs(&seq_refs), is_prefill)
            }
            RunnerType::Process(ref mut runner_streams) => {
                Self::run_forward_on_local_streams(runner_streams, owned_seqs, is_prefill)
            }
            RunnerType::MultiNodeMaster {
                ref mut local_streams,
                ref mut remote_streams,
            } => {
                let request = if is_prefill {
                    MessageType::RunPrefill((owned_seqs.to_vec(), true))
                } else {
                    let sequences = owned_seqs
                        .iter()
                        .map(|s| DecodeSequence::new(s))
                        .collect::<Vec<_>>();
                    MessageType::RunDecode((sequences, false))
                };
                let serialized =
                    rmp_serde::to_vec(&request).expect("MsgPack serialization failed");

                // Send to remote worker nodes via TCP (fire-and-forget the responses;
                // NCCL all-reduce synchronizes the actual computation)
                for tcp_stream in remote_streams.iter_mut() {
                    if let Err(e) = send_tcp(tcp_stream, &serialized) {
                        candle_core::bail!("Failed to send forward to remote worker: {}", e);
                    }
                }

                // Run on local runners (which also participate in the NCCL collective)
                let result =
                    Self::run_forward_on_local_streams(local_streams, owned_seqs, is_prefill);

                // Collect acks from remote workers
                for tcp_stream in remote_streams.iter_mut() {
                    match recv_tcp(tcp_stream) {
                        Ok(data) => {
                            let resp: MessageType = rmp_serde::from_slice(&data)
                                .expect("Failed to deserialize remote worker response");
                            match resp {
                                MessageType::RunResponse(ref ids) if ids.is_empty() => {
                                    candle_core::bail!("Remote worker returned empty response");
                                }
                                MessageType::RunResponse(_) => {} // OK
                                other => {
                                    candle_core::bail!("Unexpected remote response: {:?}", other);
                                }
                            }
                        }
                        Err(e) => {
                            candle_core::bail!("Failed to receive from remote worker: {}", e);
                        }
                    }
                }

                result
            }
        }
    }

    fn run_forward_on_local_streams(
        runner_streams: &mut Vec<LocalStream>,
        owned_seqs: &[Sequence],
        is_prefill: bool,
    ) -> Result<Vec<u32>> {
        let request = if is_prefill {
            MessageType::RunPrefill((owned_seqs.to_vec(), true))
        } else {
            let sequences = owned_seqs
                .iter()
                .map(|s| DecodeSequence::new(s))
                .collect::<Vec<_>>();
            MessageType::RunDecode((sequences, false))
        };

        let cloned_streams: Vec<LocalStream> = runner_streams
            .iter_mut()
            .map(|s| s.try_clone().expect("clone failed"))
            .collect();

        let all_outputs: Result<Vec<Vec<u32>>> = cloned_streams
            .into_par_iter()
            .map(|mut stream| {
                let msg = request.clone();
                send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                let response = receive_local(&mut stream, false)?;

                match response {
                    MessageType::RunResponse(output_ids) => {
                        if output_ids.len() == 0 {
                            candle_core::bail!("Runner step error, no response!")
                        } else {
                            Ok(output_ids)
                        }
                    }
                    other => {
                        candle_core::bail!("Unexpected response type: {:?}", other)
                    }
                }
            })
            .collect();

        let all_outputs = all_outputs.map_err(candle_core::Error::wrap)?;
        if let Some(output_ids) = all_outputs.first() {
            Ok(output_ids.clone())
        } else {
            candle_core::bail!("No output ids received from model runners");
        }
    }

    fn run_mtp_on_local_streams(
        runner_streams: &mut Vec<LocalStream>,
        request: &MessageType,
    ) -> Result<Vec<SpecStepOut>> {
        // MTP executes tensor-parallel attention/MoE kernels and therefore every
        // local runner must enter the same NCCL collectives. Send the request to
        // all ranks before waiting for any response; waiting on rank 0 first would
        // deadlock while the other ranks are still idle.
        let cloned_streams: Vec<LocalStream> = runner_streams
            .iter_mut()
            .map(|s| s.try_clone().expect("clone failed"))
            .collect();

        let all_outputs: Result<Vec<Vec<SpecStepOut>>> = cloned_streams
            .into_par_iter()
            .map(|mut stream| {
                send_local(&mut vec![stream.try_clone()?], request, false)?;
                match receive_local(&mut stream, false)? {
                    MessageType::RunResponseMTP(multi_tokens) if !multi_tokens.is_empty() => {
                        Ok(multi_tokens)
                    }
                    MessageType::RunResponseMTP(_) => {
                        candle_core::bail!("MTP runner returned empty response")
                    }
                    other => {
                        candle_core::bail!("Unexpected MTP response type: {:?}", other)
                    }
                }
            })
            .collect();

        let all_outputs = all_outputs.map_err(candle_core::Error::wrap)?;
        all_outputs
            .into_iter()
            .next()
            .ok_or_else(|| candle_core::Error::Msg("No response from local MTP runners".into()))
    }

    /// Run MTP speculative decode forward pass.
    /// Returns Vec<SpecStepOut> where each carries the accepted tokens + the next-step K.
    fn run_forward_mtp(
        runners: &Arc<RwLock<RunnerType>>,
        owned_seqs: &[Sequence],
    ) -> Result<Vec<SpecStepOut>> {
        match &mut *runners.write() {
            RunnerType::Thread(model_runner) => {
                let seq_refs: Vec<&Sequence> = owned_seqs.iter().collect();
                model_runner.run_mtp_decode(Seqs::SeqRefs(&seq_refs))
            }
            RunnerType::Process(ref mut runner_streams) => {
                use crate::runner::MessageType;

                let sequences = owned_seqs
                    .iter()
                    .map(|s| DecodeSequence::new(s))
                    .collect::<Vec<_>>();
                let request = MessageType::RunDecodeMTP(sequences);
                Self::run_mtp_on_local_streams(runner_streams, &request)
            }
            RunnerType::MultiNodeMaster {
                ref mut local_streams,
                ref mut remote_streams,
            } => {
                use crate::runner::MessageType;

                let sequences = owned_seqs
                    .iter()
                    .map(|s| DecodeSequence::new(s))
                    .collect::<Vec<_>>();
                let request = MessageType::RunDecodeMTP(sequences);
                let serialized =
                    rmp_serde::to_vec(&request).expect("MsgPack serialization failed");

                for tcp_stream in remote_streams.iter_mut() {
                    if let Err(e) = send_tcp(tcp_stream, &serialized) {
                        candle_core::bail!("Failed to send MTP forward to remote worker: {}", e);
                    }
                }

                let result = Self::run_mtp_on_local_streams(local_streams, &request);

                for tcp_stream in remote_streams.iter_mut() {
                    let _ = recv_tcp(tcp_stream);
                }

                result
            }
        }
    }

    fn run_dflash_on_local_streams(
        runner_streams: &mut Vec<LocalStream>,
        request: &MessageType,
    ) -> Result<Vec<Vec<u32>>> {
        let cloned_streams: Vec<LocalStream> = runner_streams
            .iter_mut()
            .map(|stream| stream.try_clone().expect("clone failed"))
            .collect();
        let all_outputs: Result<Vec<Vec<Vec<u32>>>> = cloned_streams
            .into_par_iter()
            .map(|mut stream| {
                send_local(&mut vec![stream.try_clone()?], request, false)?;
                match receive_local(&mut stream, false)? {
                    MessageType::RunResponseDFlash(tokens) if !tokens.is_empty() => Ok(tokens),
                    MessageType::RunResponseDFlash(_) => {
                        candle_core::bail!("DFlash runner returned empty response")
                    }
                    other => candle_core::bail!("Unexpected DFlash response type: {:?}", other),
                }
            })
            .collect();
        all_outputs
            .map_err(candle_core::Error::wrap)?
            .into_iter()
            .next()
            .ok_or_else(|| candle_core::Error::Msg("No response from local DFlash runners".into()))
    }

    fn run_forward_dflash(
        runners: &Arc<RwLock<RunnerType>>,
        owned_seqs: &[Sequence],
    ) -> Result<Vec<Vec<u32>>> {
        match &mut *runners.write() {
            RunnerType::Thread(model_runner) => {
                let seq_refs: Vec<&Sequence> = owned_seqs.iter().collect();
                model_runner.run_dflash_decode(Seqs::SeqRefs(&seq_refs))
            }
            RunnerType::Process(runner_streams) => {
                let sequences = owned_seqs
                    .iter()
                    .map(|sequence| DecodeSequence::new(sequence))
                    .collect::<Vec<_>>();
                let request = MessageType::RunDecodeDFlash(sequences);
                Self::run_dflash_on_local_streams(runner_streams, &request)
            }
            RunnerType::MultiNodeMaster {
                local_streams,
                remote_streams,
            } => {
                let sequences = owned_seqs
                    .iter()
                    .map(|sequence| DecodeSequence::new(sequence))
                    .collect::<Vec<_>>();
                let request = MessageType::RunDecodeDFlash(sequences);
                let serialized =
                    rmp_serde::to_vec(&request).expect("MsgPack serialization failed");
                for stream in remote_streams.iter_mut() {
                    send_tcp(stream, &serialized)?;
                }
                let result = Self::run_dflash_on_local_streams(local_streams, &request);
                for stream in remote_streams.iter_mut() {
                    let _ = recv_tcp(stream);
                }
                result
            }
        }
    }

    /// Phase 3: Postprocess forward pass results, deliver tokens, and do maintenance.
    /// Handles both single-token (normal decode/prefill) and multi-token (MTP) outputs.
    pub fn finish_step(
        &mut self,
        scheduled_ids: Vec<usize>,
        is_prefill: bool,
        multi_output_ids: Vec<Vec<u32>>,
    ) -> Result<usize> {
        if scheduled_ids.len() != multi_output_ids.len() {
            candle_core::bail!(
                "Forward output count mismatch: {} scheduled sequence(s), {} output row(s)",
                scheduled_ids.len(),
                multi_output_ids.len()
            );
        }

        // The llg sealed-gate: commit each produced run to the grammar FSM + keep
        // only the FSM-passing prefix, just before the sequence append. The `ff`
        // selects the spec-FF ordered-queue writer. When guidance is inactive the
        // gate is a pass-through (the `gated` == the raw `multi_output_ids`), so
        // this is a no-op for unguided sequences. The anti-loop OBSERVE (the
        // XINFER_ANTI_LOOP, default off) checks the accepted tokens for a closed loop.
        let seq_ids: Vec<usize> = scheduled_ids
            .iter()
            .map(|&idx| self.scheduler.get_running(idx).map(|s| s.id).unwrap_or(idx))
            .collect();
        let use_spec_ff = spec_ff() && !is_prefill;
        // The per-seq KV metadata (the `len` + the `block_table`) for the gate's
        // KV-backed ff forward (the `run_ff_forward`). Built from the scheduler's
        // running seqs (the `scheduled_ids` are the running indices).
        let seq_infos: Vec<crate::speculative::metadata::SpecSeqInfo> = scheduled_ids
            .iter()
            .filter_map(|&idx| {
                let seq = self.scheduler.get_running(idx)?;
                Some(crate::speculative::metadata::SpecSeqInfo {
                    id: seq.id,
                    len: seq.len(),
                    block_table: seq.block_table.clone(),
                    k: 0, // the ff span (the no the MTP draft K, the ff reserve is separate)
                })
            })
            .collect();
        let (gated, reasoning_deltas, ff_deltas, ff_drafted, mask_fail_streaks, spec_stats, phase_counts) =
            Self::gate_commit(&self.runners, &seq_ids, &multi_output_ids, use_spec_ff, &seq_infos);
        // The ff-length feedback: record the per-seq drafted ff run length (the no the
        // accepted count, the no the static reserve) so the NEXT step's pre-alloc
        // reserves exactly the ff span (the no the under-alloc that bailed the ff
        // forward, the no the systematic ff-drafted-but-0-accepted stat).
        self.scheduler.record_ff_len_feedback(&seq_ids, &ff_drafted);
        // The per-seq spec stats + phase counts ride the per-step `GateCommitResponse`
        // (the no a separate finish-time `GetSpecSeqStats` socket pull, which desynced
        // the finish path). Cache them in-process each step; the end-of-seq report reads
        // the master's own cache (the author's master-side-stats pattern).
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            if let Some(data) = spec_stats.get(i) {
                if !data.mechanism.is_empty() {
                    self.seq_spec_stats.lock().unwrap().insert(seq_id, data.clone());
                }
            }
            if let Some(counts) = phase_counts.get(i) {
                if let Some(seq) = self.scheduler.get_by_id_mut(seq_id) {
                    seq.sched_stats.reasoning_tokens = counts[0];
                    seq.sched_stats.text_tokens = counts[1];
                    seq.sched_stats.tool_call_tokens = counts[2];
                }
            }
        }
        // The reasoning counter: increment the engine-owned Sequence's
        // reasoning_token_count by the per-seq delta (the tokens committed while in
        // the Reasoning phase this step). The runner reads this from the Sequence
        // clone (the no the optional GuidanceState, which is not populated on the
        // MTP/spec-decode path).
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            if let Some(delta) = reasoning_deltas.get(i) {
                if *delta > 0 {
                    if let Some(seq) = self.scheduler.get_by_id_mut(seq_id) {
                        seq.reasoning_token_count += *delta;
                    }
                }
            }
            // The ff-token counter: increment the engine-owned Sequence's
            // sched_stats.ff_continuations by the per-seq delta (the grammar-forced
            // tokens auto-generated this step, the spec-ff path). Flows to the
            // /metrics export + the per-seq report at sequence finish.
            if let Some(delta) = ff_deltas.get(i) {
                if *delta > 0 {
                    if let Some(seq) = self.scheduler.get_by_id_mut(seq_id) {
                        seq.sched_stats.ff_continuations += *delta;
                    }
                }
            }
        }
        // The ff-length feedback: record the per-seq ff run length (the no the static
        // reserve) so the NEXT step's pre-alloc allocates exactly the ff span (the no
        // over- or under-allocation). The 0 ff run (the no the free-text position)
        // records 0 (the no the ff span for that seq next step). Called once with the
        // full arrays (the no per-seq, the no the loop).
        self.scheduler.record_ff_len_feedback(&seq_ids, ff_deltas.as_slice());
        // The anti-loop OBSERVE: check the accepted tokens for a closed loop.
        // Zero-overhead when XINFER_ANTI_LOOP is off (the default).
        if anti_loop_enabled() {
            for (i, &seq_id) in seq_ids.iter().enumerate() {
                if i < gated.len() {
                    // The phase-aware detector: exclude the ff-forced tokens (the
                    // grammar scaffolding, the tail of `gated[i]`) from the
                    // periodicity probe. Only the model-sampled prefix counts as a
                    // potential loop (the no false-positive on the tool-call literals).
                    let ff_forced = ff_deltas.get(i).copied().unwrap_or(0);
                    self.observe_loop(seq_id, &gated[i], ff_forced);
                }
            }
        }
        // The mask-fail re-decode bound (the no infinite stall on a terminally-stuck
        // FSM): a seq whose mask has failed `bound` consecutive steps (the no committed
        // progress, the position keeps re-decoding into the same broken state) is
        // terminated gracefully (the `cancel`, the client gets a completion, the
        // KV/mamba slot is freed). The bound is MODEL-DEPENDENT:
        // - Hybrid-mamba (the in-place recurrent state): bound = 1. A re-decode
        //   advances the mamba state by the rejected sentinel (the token 0), which is
        //   NOT in the committed sequence -- the mamba state desyncs on the FIRST
        //   re-decode (the no the idempotent KV overwrite of the paged-attention
        //   model). So a hybrid-mamba seq terminates on the first mask failure (the no
        //   a re-decode loop that corrupts the recurrent state).
        // - Paged-attention (the idempotent KV): bound = 32. A re-decode overwrites
        //   the same KV slot (the idempotent), so the seq can safely re-decode up to
        //   32 times before the terminally-stuck FSM is terminated.
        // This prevents one pathological grammar from starving the multi-user pool
        // (the self-DoS vector). The streak rides the `GateCommitResponse` IPC (the no
        // a Thread-only check, production is all IPC on CUDA).
        let is_hybrid_mamba = matches!(
            self.model_type,
            ModelType::Qwen3_5 | ModelType::Qwen3_5MoE | ModelType::Qwen4 | ModelType::Qwen3VL
        );
        let mask_fail_bound = if is_hybrid_mamba { 1 } else { GuidedDecoding::MAX_MASK_FAIL_STREAK };
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let streak = mask_fail_streaks.get(i).copied().unwrap_or(0);
            if streak >= mask_fail_bound {
                crate::log_error!(
                    "[llg] seq {seq_id}: grammar mask failed {} consecutive steps (the re-decode bound {} for model {:?}); terminating the sequence gracefully (the no livelock, the no pool starvation, the no the mamba-state desync)",
                    streak, mask_fail_bound, self.model_type
                );
                self.cancel(seq_id);
            }
        }
        // The SEAL: the append consumes the `gated` (the FSM-legal prefix from the
        // single `gate_commit`), never the raw `multi_output_ids`. When guidance is
        // inactive the `gated` == the raw (the fail-open), so this is a no-op for
        // unguided sequences.
        let multi_output_ids = gated;

        // `scheduled_ids` are indexes into scheduler.running, while
        // `multi_output_ids` is dense and ordered by position in scheduled_ids.
        // Keep that association explicit; a running index is not an output index.
        let (indices, output_by_running_index) = if is_prefill {
            let (indices, finished_indices) =
                self.scheduler.filter_prefill_finished(&scheduled_ids);
            if indices.is_empty() {
                self.check_canceled(None);
                return Ok(0);
            } else {
                if indices.len() != finished_indices.len() {
                    candle_core::bail!(
                        "Prefill result mapping mismatch: {} output position(s), {} running sequence(s)",
                        indices.len(),
                        finished_indices.len()
                    );
                }
                let wrapped: Vec<Vec<u32>> = indices
                    .iter()
                    .map(|&output_idx| {
                        multi_output_ids.get(output_idx).cloned().ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "Prefill output index {} out of range for {} output row(s)",
                                output_idx,
                                multi_output_ids.len()
                            ))
                        })
                    })
                    .collect::<Result<_>>()?;
                self.scheduler.postprocess(&finished_indices, &wrapped);
                let output_map = finished_indices
                    .iter()
                    .copied()
                    .zip(wrapped)
                    .collect::<HashMap<_, _>>();
                (finished_indices, output_map)
            }
        } else {
            self.scheduler
                .postprocess(&scheduled_ids, &multi_output_ids);
            let output_map = scheduled_ids
                .iter()
                .copied()
                .zip(multi_output_ids)
                .collect::<HashMap<_, _>>();
            (scheduled_ids, output_map)
        };

        let mut prefill_batch: Vec<(usize, usize, f32, bool)> = Vec::new();
        for &idx in &indices {
            let step_tokens = output_by_running_index.get(&idx).ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "Missing forward output for running sequence index {}",
                    idx
                ))
            })?;
            let sq = self.scheduler.get_running(idx);
            if let Some(s) = sq {
                let seq_id = s.id;
                if s.is_finished() {
                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            let prompt_start_time = s.created_time();

                            let decode_finish_time = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .expect("Time went backwards")
                                .as_millis()
                                as usize;
                            let decode_start_time = if let Some(decode_start_time) =
                                self.decode_start_times.get(&seq_id)
                            {
                                *decode_start_time
                            } else {
                                decode_finish_time
                            };

                            if s.is_tool_call_end {
                                if *request_type == RequestType::Stream {
                                    if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                        if let Some(tok) = decoder.step(s.last_token)? {
                                            let _ = sender
                                                .try_send(StreamItem::Token(tok, s.last_token));
                                        }
                                    }
                                } else {
                                    let _ = sender.try_send(StreamItem::TokenID(s.last_token));
                                }
                            }

                            if *request_type == RequestType::Stream {
                                let num_appended = s.output_ids.len()
                                    - self.decode_length.get(&seq_id).copied().unwrap_or(0);
                                let tokens_to_stream =
                                    &step_tokens[..num_appended.min(step_tokens.len())];
                                for &tok in tokens_to_stream {
                                    if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                        if let Some(text) = decoder.step(tok)? {
                                            let _ = sender.try_send(StreamItem::Token(text, tok));
                                        }
                                    }
                                }
                                let _ = sender.try_send(StreamItem::Done((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    s.output_ids.len(),
                                    s.stop_sequence.clone(),
                                )));
                            } else {
                                let _ = sender.try_send(StreamItem::Completion((
                                    prompt_start_time,
                                    decode_start_time,
                                    decode_finish_time,
                                    s.output_ids.clone(),
                                    s.stop_sequence.clone(),
                                )));
                            }
                        }
                    }

                    self.stream_senders.remove(&seq_id);
                    self.stream_decoders.remove(&seq_id);
                    if let Some(_) = self.active_requests.get(&seq_id) {
                        self.active_requests.remove(&seq_id);
                    }
                    self.decode_start_times.remove(&seq_id);
                    self.decode_length.remove(&seq_id);
                    self.seq_prefilled_reasoning_end.remove(&seq_id);
                    self.seq_prompt_replays.remove(&seq_id);
                    let _ = self.notify_runner_finished(seq_id);
                    if self.econfig.server_mode.unwrap_or(true) {
                        self.scheduler.print_free_blocks();
                    }
                } else {
                    if !self.decode_start_times.contains_key(&seq_id) {
                        let cur_time = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis() as usize;
                        self.decode_start_times.insert(seq_id, cur_time);
                        self.decode_length.insert(seq_id, 1);

                        let time_costs = cur_time - s.created_time();
                        if time_costs / 100 > 0 && s.len() > 0 {
                            prefill_batch.push((
                                seq_id,
                                s.len(),
                                time_costs as f32 / 1000f32,
                                s.num_cached_tokens > 0,
                            ));
                        }
                    }

                    if let Some(length) = self.decode_length.get_mut(&seq_id) {
                        *length = s.output_len();
                    }

                    let mut token_ids = if step_tokens.len() > 1 {
                        step_tokens.to_vec()
                    } else if self.is_pd_mode() && s.pd_first_token.is_some() && s.output_len() == 2
                    {
                        vec![s.pd_first_token.unwrap_or(s.last_token), s.last_token]
                    } else {
                        vec![s.last_token]
                    };
                    if let Some(mut replay_ids) = self.take_prompt_replay_token_ids(seq_id) {
                        replay_ids.extend(token_ids);
                        token_ids = replay_ids;
                    }

                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            if *request_type == RequestType::Stream {
                                if let Some(decoder) = self.stream_decoders.get_mut(&seq_id) {
                                    for token_id in token_ids {
                                        if let Some(tok) = decoder.step(token_id)? {
                                            let result =
                                                sender.try_send(StreamItem::Token(tok, token_id));
                                            if result.is_err() {
                                                crate::log_error!(
                                                    "Error when sending token to client [seq_id {}]",
                                                    seq_id
                                                );
                                                self.cancelled_sequences.push(seq_id);
                                            }
                                        }
                                    }
                                }
                            } else {
                                for token_id in token_ids {
                                    if let Err(_) = sender.try_send(StreamItem::TokenID(token_id)) {
                                        crate::log_error!(
                                            "Error when sending token to client [seq_id {}]",
                                            seq_id
                                        );
                                        self.cancelled_sequences.push(seq_id);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !prefill_batch.is_empty() {
            let seq_ids: Vec<usize> = prefill_batch.iter().map(|(id, _, _, _)| *id).collect();
            let total_tokens: usize = prefill_batch.iter().map(|(_, t, _, _)| *t).sum();
            let max_time: f32 = prefill_batch
                .iter()
                .map(|(_, _, t, _)| *t)
                .fold(0.0f32, f32::max);
            if max_time > 0.0 {
                crate::log_info!(
                    "Prefilling {} seq(s) {:?}: {} total tokens in {:.2}s ({:.2} tokens/s)",
                    prefill_batch.len(),
                    seq_ids,
                    total_tokens,
                    max_time,
                    total_tokens as f32 / max_time,
                );
            }
        }
        self.scheduler.clear_finished();

        if indices.is_empty()
            && self.scheduler.kv_cache_usage_percent() > KVCACHE_SWAP_THRESHOLD + 0.01f32
        {
            if !self.try_release_cache(0) {
                if let Some(oldest_seq_id) = self.active_requests.clone().iter().min() {
                    crate::log_error!(
                        "Unable to schedule task(s), drop the oldest active request (seq_id: {:?})",
                        oldest_seq_id
                    );
                    self.cancelled_sequences.push(*oldest_seq_id);
                    self.check_canceled(Some(
                        "Unable to schedule task(s), this request has been dropped!".to_string(),
                    ));
                }
                if self.scheduler.kv_cache_usage_percent() > 0.99f32 {
                    self.free_resources();
                }
            }
        } else {
            self.check_canceled(None);
        }
        if self.econfig.server_mode.unwrap_or(true) {
            self.may_print_decoding_throughput(&indices);
        }
        Ok(indices.len())
    }

    pub fn try_release_cache(&mut self, tokens_required: usize) -> bool {
        if tokens_required == 0 {
            return self.scheduler.evict_prefix_cache_blocks(1) > 0;
        }

        let required_blocks = tokens_required.div_ceil(self.econfig.block_size);
        self.scheduler
            .evict_prefix_cache_until_free(required_blocks)
            > 0
    }

    pub fn check_canceled(&mut self, reason: Option<String>) {
        if !self.cancelled_sequences.is_empty() {
            for i in 0..self.cancelled_sequences.len() {
                let seq_id = self.cancelled_sequences[i];
                self.scheduler.cancel(seq_id);
                // Ensure model-side per-sequence state (e.g., Qwen3.5 Mamba cache slot) is released.
                let _ = self.notify_runner_finished(seq_id);
                if let Some(_) = self.active_requests.get(&seq_id) {
                    self.active_requests.remove(&seq_id);
                }
                self.stream_decoders.remove(&seq_id);
                self.decode_start_times.remove(&seq_id);
                self.seq_prefilled_reasoning_end.remove(&seq_id);
                self.seq_prompt_replays.remove(&seq_id);
                if let Some(r) = &reason {
                    if let Some(sender) = self.stream_senders.get_mut(&seq_id) {
                        if let Some(request_type) = self.request_types.get(&seq_id) {
                            if *request_type == RequestType::Stream {
                                let _ = sender.try_send(StreamItem::Error(r.clone()));
                            }
                        }
                    }
                }
                self.stream_senders.remove(&seq_id);
            }
            self.scheduler.clear_finished();
            self.cancelled_sequences.clear();
        }
        self.drain_pending_runner_releases();
    }

    pub fn may_print_decoding_throughput(&mut self, active_indices: &Vec<usize>) {
        if active_indices.is_empty() || self.active_requests.is_empty() {
            return;
        }
        let cur_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as usize;
        if cur_time - self.last_check_throughput_time < 10000 {
            return;
        }
        self.last_check_throughput_time = cur_time;

        let mut total_decoded_length = 0;
        let mut total_decoded_time_costs = 0;

        let mut seq_ids = Vec::new();
        for &idx in active_indices {
            if let Some(seq) = self.scheduler.get_running(idx) {
                if let Some(length) = self.decode_length.get(&seq.id) {
                    total_decoded_length += length;
                }
                if let Some(start_time) = self.decode_start_times.get(&seq.id) {
                    total_decoded_time_costs += cur_time - start_time;
                }
                seq_ids.push(seq.id);
            }
        }

        if total_decoded_length > 0 && total_decoded_time_costs / 1000 > 0 {
            let avg_throughput = total_decoded_length / (total_decoded_time_costs / 1000);
            crate::log_info!(
                "Decoding: {} active request(s) [Seq: {:?}], avg. {} tokens/s per request (total: {} tokens/s)",
                active_indices.len(),
                seq_ids,
                avg_throughput,
                avg_throughput * active_indices.len()
            )
        }

        if total_decoded_length % 100 > 50 {
            self.scheduler.print_free_blocks();
        }
    }

    pub fn cancel(&mut self, seq_id: usize) {
        self.cancelled_sequences.push(seq_id);
    }

    pub fn cancel_all_with_reason(&mut self, reason: Option<String>) -> bool {
        for seq_id in &self.active_requests {
            self.cancelled_sequences.push(*seq_id);
        }
        let has_requests_to_cancel = self.cancelled_sequences.len() > 0;
        self.check_canceled(reason);
        has_requests_to_cancel
    }

    pub fn apply_chat_template(
        &self,
        params: &SamplingParams,
        messages: &Vec<Message>,
        tools: &Vec<Tool>,
        log: bool,
    ) -> (String, i32) {
        // let mut collected_images = Vec::new();
        let mut prompt_template = self.template.clone();
        let default_enable_thinking = self.template.enable_thinking();
        if let Some(grammar) = &params.grammar {
            if is_reasoning_grammar(&grammar) {
                prompt_template.set_enable_thinking(true);
            } else {
                // Grammar without reasoning rules: thinking is handled at the mask
                // level (GuidanceState defers grammar until after </think>), so use
                // the default thinking setting — same as the no-grammar path.
                prompt_template
                    .set_enable_thinking(params.thinking.unwrap_or(default_enable_thinking));
            }
        } else {
            prompt_template.set_enable_thinking(params.thinking.unwrap_or(default_enable_thinking));
        };
        prompt_template.set_reasoning_effort(
            params
                .reasoning_effort
                .as_ref()
                .and_then(|effort| effort.chat_template_value()),
        );
        prompt_template.set_messages(messages);
        let image_idx: i32 = 0;
        let prompt_processed = prompt_template
            .apply_chat_template(tools, log)
            .map_err(candle_core::Error::wrap);
        let prompt = if prompt_processed.is_ok() {
            prompt_processed.unwrap()
        } else {
            if log {
                crate::log_error!(
                    "Applying Chat Template failed: {:?}, use default template!",
                    prompt_processed
                );
            }

            let mut prompt = "".to_string();
            for message in messages {
                if message.role == "user" {
                    let escaped = prompt_template.escape_text(&message.content);
                    prompt += &self.default_chat_template.replace("{}", &escaped);
                    prompt += "\n";
                }
            }
            prompt
        };
        if log {
            log_info!(
                "Prompt after applying Chat Template: {}",
                prompt.replace("\n", "")
            );
        }
        // Generation alignment and open/close parity enforcement
        if let Some(grammar) = &params.grammar {
            let lark = get_lark_from_top_level_grammar(grammar);
            if lark.contains("start: bos ") {
                // BOS-based trimming: trim the last BOS token from the prompt tail.
                // Only trim if the prompt actually ends with the BOS string to avoid
                // splitting in the middle of a multi-turn conversation.
                if let Ok(bos_string) = self
                    .tokenizer
                    .decode(&self.guidance_tokens.bos_token_ids, false)
                {
                    if prompt.trim_end().ends_with(&bos_string) {
                        if let Some((prefix, _)) = prompt.rsplit_once(&bos_string) {
                            return (prefix.to_string(), image_idx);
                        }
                    }
                }
                for bos_token in self.guidance_tokens.bos_token_ids.iter() {
                    if let Ok(bos_string) = self.tokenizer.decode(&[*bos_token], false) {
                        if prompt.trim_end().ends_with(&bos_string) {
                            if let Some((prefix, _)) = prompt.rsplit_once(&bos_string) {
                                return (prefix.to_string(), image_idx);
                            }
                        }
                    }
                }
            }
        }
(prompt, image_idx)
    }

    pub fn is_idle(&self) -> bool {
        self.active_requests.is_empty()
    }

    pub fn is_pd_mode(&self) -> bool {
        self.econfig.pd_config.is_some()
    }

    pub fn is_pd_server(&self) -> bool {
        if let Some(p_cfg) = &self.econfig.pd_config {
            matches!(p_cfg.role, PdRole::Server)
        } else {
            false
        }
    }

    /// Registers preprocessed completion requests. Callers should run
    /// `preprocess(...)` under `engine.read()` first, then call this under
    /// `engine.write()` so the write lock is held only for request admission.
    pub fn generate_sync(
        &mut self,
        preprocessed: Vec<PreprocessedRequest>,
        images: Option<ImageData>,
        logger: &Option<Arc<ChatCompletionLogger>>,
    ) -> Result<Vec<(usize, usize, mpsc::Receiver<StreamItem>)>> {
        let mut receivers = Vec::with_capacity(preprocessed.len());
        for pp in preprocessed {
            if let Some(ref l) = logger {
                l.log_prompt(&pp.prompt);
            }
            let (seq_id, prompt_length, rx) = self.add_request_pretokenized(
                &pp.params,
                &pp.prompt,
                pp.token_ids,
                RequestType::Completion,
                &images,
                pp.image_idx,
            )?;
            receivers.push((seq_id, prompt_length, rx));
        }
        Ok(receivers)
    }

    pub async fn collect_sync_results(
        receivers: Vec<(usize, usize, mpsc::Receiver<StreamItem>)>,
        tokenizer: Arc<Tokenizer>,
        logger: Option<Arc<ChatCompletionLogger>>,
    ) -> Result<Vec<GenerationOutput>> {
        let decoded_tokens = Arc::new(AtomicUsize::new(0));
        let decode_start_time = Arc::new(AtomicUsize::new(0));

        // Create futures for each receiver (do NOT spawn detached tasks)
        let tasks = receivers
            .into_iter()
            .map(|(seq_id, prompt_length, mut rx)| {
                let decoded_tokens = decoded_tokens.clone();
                let decode_start_time_clone = decode_start_time.clone();
                let tokenizer = Arc::clone(&tokenizer);
                let logger = logger.clone();
                async move {
                    let mut output: Option<GenerationOutput> = None;
                    let mut collected_token_ids: Vec<u32> = Vec::new();

                    // Initialize decoder for incremental logging if needed
                    let mut decoder = if logger.is_some() {
                        let tokenizer_clone = tokenizer.as_ref().clone();
                        let leaked: &'static _ = Box::leak(Box::new(tokenizer_clone));
                        let decoder = leaked.decode_stream(true);
                        let wrapped = super::StreamWithTokenizer {
                            _tokenizer: unsafe { Box::from_raw(leaked as *const _ as *mut _) },
                            stream: decoder,
                        };
                        Some(Box::new(wrapped) as Box<dyn super::DecodeStreamTrait + Send + Sync>)
                    } else {
                        None
                    };

                    while let Some(msg) = rx.recv().await {
                        match msg {
                            StreamItem::Completion((
                                prompt_start,
                                decode_start,
                                decode_finish,
                                decoded_ids,
                                stop_sequence,
                            )) => {
                                let decoded_len = decoded_ids.len();
                                let decode_output = tokenizer
                                    .decode(&decoded_ids, true)
                                    .expect("unable to decode!");

                                output = Some(GenerationOutput {
                                    seq_id,
                                    prompt_length,
                                    prompt_start_time: prompt_start,
                                    decode_start_time: decode_start,
                                    decode_finish_time: decode_finish,
                                    decoded_length: decoded_len,
                                    decode_output,
                                    stop_sequence,
                                });
                                break;
                            }
                            StreamItem::Token(_, _) => {
                                decoded_tokens.fetch_add(1, Ordering::Relaxed);

                                decode_start_time_clone
                                    .compare_exchange(
                                        0,
                                        SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis()
                                            as usize,
                                        Ordering::SeqCst,
                                        Ordering::SeqCst,
                                    )
                                    .ok();
                            }
                            StreamItem::TokenID(id) => {
                                decoded_tokens.fetch_add(1, Ordering::Relaxed);
                                collected_token_ids.push(id);

                                if let Some(d) = decoder.as_mut() {
                                    if let Ok(Some(text)) = d.step(id) {
                                        if let Some(l) = &logger {
                                            l.log_stream_token(&text);
                                        }
                                    }
                                }

                                decode_start_time_clone
                                    .compare_exchange(
                                        0,
                                        SystemTime::now()
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis()
                                            as usize,
                                        Ordering::SeqCst,
                                        Ordering::SeqCst,
                                    )
                                    .ok();
                            }
                            StreamItem::Done(_) => {
                                crate::log_info!("Sequence [seq_id {}] finished!", seq_id);
                                break;
                            }
                            StreamItem::Error(e) => {
                                crate::log_error!("Error in Seq {}: {}", seq_id, e);
                                break;
                            }
                        }
                    }

                    output
                }
            });

        // Wait for all decoding tasks
        let outputs = join_all(tasks).await;

        // Collect successful outputs
        let results: Vec<_> = outputs.into_iter().filter_map(|r| r).collect();

        Ok(results)
    }

    pub fn free_resources(&mut self) {
        crate::log_error!("***Release all resources for future usage!");
        self.scheduler.clear_finished();
        let released_ids = self.scheduler.release_waitings();
        for seq_id in released_ids {
            let _ = self.notify_runner_finished(seq_id);
        }
        self.seq_prefilled_reasoning_end.clear();
        self.seq_prompt_replays.clear();
    }

    /// Returns the reasoning end marker when the prompt for this sequence
    /// already ended with an opening think marker.
    pub fn get_prefilled_reasoning_end_marker(&self, seq_id: usize) -> Option<String> {
        self.seq_prefilled_reasoning_end.get(&seq_id).cloned()
    }

    /// Registers a preprocessed streaming request. Callers should run
    /// `preprocess(...)` under `engine.read()` first, then call this under
    /// `engine.write()` so the write lock is held only for request admission.
    pub fn generate_stream(
        &mut self,
        preprocessed: PreprocessedRequest,
        images: Option<ImageData>,
        logger: &Option<Arc<ChatCompletionLogger>>,
    ) -> Result<(usize, usize, Option<String>, mpsc::Receiver<StreamItem>)> {
        if let Some(ref l) = logger {
            l.log_prompt(&preprocessed.prompt);
        }
        match self.add_request_pretokenized(
            &preprocessed.params,
            &preprocessed.prompt,
            preprocessed.token_ids,
            RequestType::Stream,
            &images,
            preprocessed.image_idx,
        ) {
            Ok((seq_id, prompt_length, rx)) => {
                let prefilled_reasoning_end = self.get_prefilled_reasoning_end_marker(seq_id);
                Ok((seq_id, prompt_length, prefilled_reasoning_end, rx))
            }
            Err(e) => {
                candle_core::bail!("{:?}", e)
            }
        }
    }

    pub fn get_usage_stats(&self, session_id: Option<String>) -> Result<UsageResponse> {
        match session_id {
            Some(sid) => {
                let available_kvcache_tokens = self.scheduler.get_available_kv_tokens();
                let max_model_len = self
                    .econfig
                    .max_model_len
                    .ok_or_else(|| candle_core::Error::msg("max_model_len not set!"))?;
                let (seq_id, status) =
                    self.scheduler.find_seq_by_session_id(&sid).ok_or_else(|| {
                        candle_core::Error::msg(format!("Seq with session_id {} not found", sid))
                    })?;
                let token_used = self.scheduler.get_seq_token_usage(seq_id)?;

                let total_kv_cache_tokens = self.scheduler.get_total_kv_tokens();
                let (swap_used, total_swap_memory) = self.scheduler.get_cpu_swap_usage();

                let session_status = match status {
                    SequenceStatus::FinishSwapped | SequenceStatus::Swapped => {
                        "Swapped".to_string()
                    }
                    _ => status.to_string(),
                };

                Ok(UsageResponse {
                    token_used,
                    max_model_len,
                    used_kvcache_tokens: total_kv_cache_tokens - available_kvcache_tokens,
                    total_kv_cache_tokens,
                    swap_used,
                    total_swap_memory,
                    session_status,
                })
            }
            _ => {
                candle_core::bail!("No session_id provided")
            }
        }
    }

/// Publish the live QoS + scheduler resource gauges to the Prometheus
    /// registry (the /metrics endpoint). The monolith reads the QosController's
    /// live gauges (the gain, the ITL, the SLO, the WorkloadHistogram estimators)
    /// + the Scheduler's resource gauges (the KV usage, the CPU swap) directly the
    /// global Telemetry registry, (the no the actor branch's fire-and-forget
    /// `SchedMessage::PublishTelemetry` RPC, which was the F-QoS-3 stale-gauges
    /// defect). Sync: the monolith scheduler methods are sync.
    pub fn publish_telemetry(&self) {
        let telemetry = Telemetry::global();
        telemetry.publish_live_gauges(&self.scheduler.policy());
        telemetry.publish_scheduler_gauges(&self.scheduler);
        // The LoRA registry (the per-runner source of truth, the no the process-wide
        // global OnceLock which is empty in the server process when the runner is
        // separate). The `None` when the runner is not Thread-mode (the Process LoRA
        // registry reachable) or Lo LoRA is configured.
        let lora_registry = match &*self.runners.read() {
            RunnerType::Thread(model_runner) => Some(model_runner.lora_registry.clone()),
            _ => None,
        };
        telemetry.publish_lora_gauges(lora_registry);
        telemetry.publish_spec_gauges();
        // The spec-k live feedback (the inputs + the outputs of the dynamic K
        // law, so the operator can see how the K is being driven).
        let is_mtp = uses_builtin_mtp(&self.econfig);
        telemetry.publish_spec_feedback(&self.scheduler.policy().spec_feedback(is_mtp));
        // The runtime memory + scheduling gauges (the CUDA graph pool, the per-step
        // budget, the CPU KV blocks, the Mamba active/prefix split). The values
        // come from the engine config (the static) + the scheduler's
        // resource_gauges (the runtime Mamba split) + the runner's graph capturer
        // (the CUDA graph pool usage, the no the hardcoded 0 placeholder).
        let g = self.scheduler.resource_gauges();
        // The CUDA graph pool usage (the current + peak bytes, the 0 when the
        // runner is not Thread-mode or the pool is unavailable). The graph
        // capturer is only present when the cuda + graph features are on.
        #[cfg(all(feature = "cuda", feature = "graph"))]
        let (graph_current_bytes, graph_peak_bytes) = {
            // The cached graph pool usage (the no the blocking CUDA driver call on
            // every scrape). Refresh the cache every 5s (the graph pool usage is
            // slow-changing, the 5s refresh is fresh enough for the /metrics).
            let mut cache = self.graph_pool_cache.lock().unwrap_or_else(|e| e.into_inner());
            if cache.2.elapsed() > std::time::Duration::from_secs(5) {
                if let RunnerType::Thread(model_runner) = &*self.runners.read() {
                    let (cur, peak) = model_runner.decode_capturer.graph_pool_usage();
                    *cache = (cur, peak, std::time::Instant::now());
                }
            }
            (cache.0, cache.1)
        };
        #[cfg(not(all(feature = "cuda", feature = "graph")))]
        let (graph_current_bytes, graph_peak_bytes) = (0, 0);
        telemetry.publish_runtime_gauges(
            (graph_current_bytes / 1_000_000) as u64, // the MB
            (graph_peak_bytes / 1_000_000) as u64, // the MB
            self.econfig.max_num_batched_tokens,
            self.econfig.max_num_parallel_reqs,
            // The CPU KV blocks (the GPU KV blocks × the cpu_mem_fold fraction, the 0.5
            // default). The float math (the no the integer truncation to 0).
            ((self.econfig.num_blocks as f64) * (self.econfig.cpu_mem_fold.unwrap_or(0.5) as f64)) as usize,
            (g.mamba_used_gb as f64 * 1e9) as u64,
            0, // the Mamba prefix bytes (the not separately tracked, the 0 placeholder)
            (g.mamba_slot_mb as f64 * 1e6) as u64,
        );
    }

    pub fn embed(
        &mut self,
        inputs: &[String],
        strategy: EmbeddingStrategy,
    ) -> Result<(Vec<Vec<f32>>, usize)> {
        let mut outputs = Vec::new();
        let mut prompt_tokens = 0;

        for input in inputs {
            let tokens = self
                .tokenizer
                .encode_fast(input.as_str(), true)
                .map_err(candle_core::Error::wrap)?;
            let token_ids: Vec<u32> = tokens.get_ids().to_vec();
            if token_ids.is_empty() {
                candle_core::bail!("Embedding input cannot be empty");
            }

            if let Some(max_model_len) = self.econfig.max_model_len {
                if token_ids.len() > max_model_len - 1 {
                    candle_core::bail!(
                        "Embedding input length {} exceeds max_model_len {}",
                        token_ids.len(),
                        max_model_len
                    );
                }
            }

            let mut seq = Sequence::new(
                token_ids.clone(),
                self.econfig.block_size,
                SamplingParams::default(),
                &None,
                -1,
            );
            let required_blocks = self.scheduler.block_manager.required_blocks(&seq);
            let available_blocks = self.scheduler.block_manager.get_num_free_blocks();
            if required_blocks > available_blocks {
                let available_tokens = available_blocks * self.econfig.block_size;
                let required_tokens = required_blocks * self.econfig.block_size;
                candle_core::bail!(
                    "Remaining {} kvcache tokens, but embedding requires {} new tokens, please retry later",
                    available_tokens,
                    required_tokens
                );
            }
            self.scheduler.block_manager.allocate(&mut seq)?;

            let mut chunked_mean: Option<Vec<f32>> = None;
            let mut last_vec: Option<Vec<f32>> = None;
            let mut processed_tokens = 0usize;
            let chunk_size = self.econfig.effective_prefill_chunk_size();

            while seq.num_cached_tokens < seq.len() {
                let remaining = seq.len() - seq.num_cached_tokens;
                let chunk_tokens = std::cmp::min(chunk_size, remaining);
                let embedding_result = match &mut *self.runners.write() {
                    RunnerType::Thread(model_runner) => model_runner.embed(&[&seq], &strategy),
                    RunnerType::Process(ref mut runner_streams) => {
                        let request = MessageType::RunEmbed((vec![seq.clone()], strategy.clone()));
                        let cloned_streams: Vec<LocalStream> = runner_streams
                            .iter_mut()
                            .map(|s| s.try_clone().expect("clone failed"))
                            .collect();

                        let all_outputs: Result<Vec<Vec<Vec<f32>>>> = cloned_streams
                            .into_par_iter()
                            .map(|mut stream| {
                                let msg = request.clone();
                                send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                                let response = receive_local(&mut stream, false)?;

                                match response {
                                    MessageType::RunResponseEmbed(output_embed) => {
                                        if output_embed.len() == 0 {
                                            candle_core::bail!("Runner step error, no response!")
                                        } else {
                                            Ok(output_embed)
                                        }
                                    }
                                    other => {
                                        candle_core::bail!("Unexpected response type: {:?}", other)
                                    }
                                }
                            })
                            .collect();
                        let all_outputs = all_outputs.map_err(candle_core::Error::wrap)?;
                        if let Some(output_embed) = all_outputs.first() {
                            Ok(output_embed.clone())
                        } else {
                            candle_core::bail!(
                                "No output embedding states received from model runners"
                            );
                        }
                    }
                    RunnerType::MultiNodeMaster {
                        local_streams: ref mut runner_streams,
                        ref mut remote_streams,
                    } => {
                        let request = MessageType::RunEmbed((vec![seq.clone()], strategy.clone()));
                        let serialized =
                            rmp_serde::to_vec(&request).expect("MsgPack serialization failed");

                        for tcp_stream in remote_streams.iter_mut() {
                            send_tcp(tcp_stream, &serialized)?;
                        }

                        let cloned_streams: Vec<LocalStream> = runner_streams
                            .iter_mut()
                            .map(|s| s.try_clone().expect("clone failed"))
                            .collect();

                        let all_outputs: Result<Vec<Vec<Vec<f32>>>> = cloned_streams
                            .into_par_iter()
                            .map(|mut stream| {
                                let msg = request.clone();
                                send_local(&mut vec![stream.try_clone()?], &msg, false)?;
                                let response = receive_local(&mut stream, false)?;

                                match response {
                                    MessageType::RunResponseEmbed(output_embed) => {
                                        if output_embed.is_empty() {
                                            candle_core::bail!("Runner step error, no response!")
                                        } else {
                                            Ok(output_embed)
                                        }
                                    }
                                    other => {
                                        candle_core::bail!("Unexpected response type: {:?}", other)
                                    }
                                }
                            })
                            .collect();
                        let all_outputs = all_outputs.map_err(candle_core::Error::wrap);

                        for tcp_stream in remote_streams.iter_mut() {
                            let data = recv_tcp(tcp_stream)?;
                            let response: MessageType = rmp_serde::from_slice(&data)
                                .expect("Failed to deserialize remote worker response");
                            match response {
                                MessageType::RunResponseEmbed(output_embed) => {
                                    if output_embed.is_empty() {
                                        candle_core::bail!(
                                            "Remote worker returned empty embedding response"
                                        );
                                    }
                                }
                                MessageType::Error(err) => {
                                    candle_core::bail!("Remote worker embedding failed: {}", err);
                                }
                                other => {
                                    candle_core::bail!(
                                        "Unexpected remote embedding response: {:?}",
                                        other
                                    );
                                }
                            }
                        }

                        if let Some(output_embed) = all_outputs?.first() {
                            Ok(output_embed.clone())
                        } else {
                            candle_core::bail!(
                                "No output embedding states received from model runners"
                            );
                        }
                    }
                };
                let embedding_vecs = match embedding_result {
                    Ok(v) => v,
                    Err(e) => {
                        self.scheduler.block_manager.deallocate(&seq);
                        return Err(e);
                    }
                };
                if let Some(vec) = embedding_vecs.into_iter().next() {
                    match strategy {
                        EmbeddingStrategy::Mean => {
                            if chunked_mean.is_none() {
                                chunked_mean = Some(vec![0.0; vec.len()]);
                            }
                            if let Some(sum) = chunked_mean.as_mut() {
                                for (dst, val) in sum.iter_mut().zip(vec.iter()) {
                                    *dst += val * chunk_tokens as f32;
                                }
                            }
                        }
                        EmbeddingStrategy::Last => {
                            last_vec = Some(vec);
                        }
                    }
                }

                seq.num_cached_tokens += chunk_tokens;
                processed_tokens += chunk_tokens;
                if seq.len() > chunk_size {
                    if chunk_tokens < chunk_size {
                        crate::log_info!(
                            "Embedding chunk prefilled finished {}/{} (Seq {})",
                            seq.num_cached_tokens,
                            seq.len(),
                            seq.id
                        );
                    } else {
                        crate::log_info!(
                            "Embedding chunk prefilled {}/{} (Seq {})",
                            seq.num_cached_tokens,
                            seq.len(),
                            seq.id
                        );
                    }
                }
            }

            self.scheduler.block_manager.deallocate(&seq);
            prompt_tokens += seq.len();

            let final_vec = match strategy {
                EmbeddingStrategy::Mean => {
                    let mut sum = chunked_mean.unwrap_or_default();
                    if processed_tokens > 0 {
                        for v in sum.iter_mut() {
                            *v /= processed_tokens as f32;
                        }
                    }
                    sum
                }
                EmbeddingStrategy::Last => last_vec.unwrap_or_default(),
            };
            outputs.push(final_vec);
        }

        Ok((outputs, prompt_tokens))
    }

pub fn start_engine(engine: Arc<RwLock<Self>>) {
        GLOBAL_RT.spawn(async move {
            let engine = engine.clone();
            let (is_pd_server, runners, mtp_enabled, dflash_enabled) = {
                let guard = engine.read();
                let has_mtp = uses_builtin_mtp(&guard.econfig);
                let has_dflash = uses_dflash(&guard.econfig);
                (
                    guard.is_pd_mode() && guard.is_pd_server(),
                    guard.runners.clone(),
                    has_mtp,
                    has_dflash,
                )
            };
            // Initialize the global drafter profile when any drafter is active (the
            // MTP or the DFlash), so the dynamic drafter choice (the
            // `DrafterProfile::select`) has a live acceptance signal to read.
            init_drafter_profile(mtp_enabled || dflash_enabled);
            loop {
                let idle = {
                    let guard = engine.read();
                    guard.is_idle() && !is_pd_server
                };

                if idle {
                    tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
                    continue;
                }

                let mut task_processed = 0;

                // Phase 1: Schedule (engine lock held briefly)
                let prep = {
                    let mut guard = engine.write();
                    match guard.prepare_step() {
                        Ok(p) => p,
                        Err(e) => {
                            crate::log_error!("[Engine Loop] Prepare error: {:?}", e);
                            if !guard.cancel_all_with_reason(Some(e.to_string())) {
                                std::process::exit(1);
                            }
                            None
                        }
                    }
                };
                // Engine lock released -- server can accept new requests during forward pass

                if let Some((scheduled_ids, is_prefill, owned_seqs)) = prep {
                    // The dynamic-K drathe c1bf8ac37 feature): the MTP path returns
                    // per-seq `SpecStepOut` (the tokens + the runner-computed next-K).
                    // The next-K feedback is recorded so the NEXT step's pre-alloc uses
                    // the runner's real signal (the no the scheduler guessing from an
                    // empty cross-process global). The DFlash + plain paths return
                    // plain token vecs (the no next-K feedback).
                    let mut mtp_next_k: Vec<usize> = Vec::new();
                    // The dynamic drafter choice: when the drafter profile has acceptance data
                    // (the spec-decode dynamic-drafter feature), `preferred_drafter` picks the
                    // drafter kind for the current workload (the 0 = the MTP, the 1 = the DFlash,
                    // the 2 = the chained). It OVERRIDES the static enable flags so the right
                    // drafter serves the right workload. When the profile is empty (the no data
                    // yet) or disabled, the static flags stand (the prior behavior). The kind 2
                    // (the chained) has no dedicated path on this branch (the chain seam is the
                    // llg work), so it falls back to the DFlash (the higher-acceptance drafter).
let preferred = preferred_drafter();
                    let prefer_dflash = matches!(preferred, Some(1) | Some(2));
                    let prefer_mtp = matches!(preferred, Some(0));
                    // The MTP + DFlash paths both batch natively (the run_mtp_decode_batch /
                    // the run_dflash_decode_batch).
                    // The shared speculative-draft slot cap (the MTP + DFlash parity):
                    // both drafters batch up to spec_parallel_draft() sequences (the
                    // default 0 = single-seq only, the origin/main behavior); sequences
                    // beyond the cap fall back to plain decode. The MTP + DFlash verify
                    // buffers are sized to this cap (runner.rs), so the verify batch
                    // always fits. Parallel drafting is a per-instance concern: each
                    // engine has its own owned_seqs, so two instances can both draft
                    // concurrently.
                    // The cap: XINFER_SPEC_PARALLEL_DRAFT when set, else unbounded
                    // (usize::MAX) — no static fallback to max_num_seqs.
                    let spec_slots = spec_parallel_draft()
                        .unwrap_or(usize::MAX);
                    let dflash_ok = owned_seqs.len() <= spec_slots;
                    let use_dflash = dflash_enabled && !is_prefill && dflash_ok
                        && (!prefer_mtp || preferred.is_none());
                    // The MTP dispatch cap (the F-LLG-1 fix): the GDN MTP verify buffers
                    // were sized at model build to the PLACEHOLDER max_num_seqs
                    // (config.rs), NOT the final accommodating value the MMU raises it to.
                    // Capping the MTP dispatch batch to that placeholder keeps the verify
                    // batch (num_seqs × (K+1)) within the buffer (max_num_parallel_reqs × (spec_tokens+1)),
                    // making the `narrow` OOB structurally impossible. The DFlash path is
                    // capped separately by spec_parallel_draft() (the dflash_ok above).
// The cap keys off `max_num_parallel_reqs` (the MMU-finalized parallelism,
                    // the same quantity the GDN M sized the verify buffers to,
                    // runner.rs), NOT `max_num_seqs` (the CLI placeholder), so the
                    // dispatch cap and the buffer always agree.
                    let mtp_batch_cap = engine.read().econfig.max_num_parallel_reqs.max(1);
                    let mtp_ok = owned_seqs.len() <= spec_slots
                        && owned_seqs.len() <= mtp_batch_cap;
                    let use_mtp = mtp_enabled && !use_dflash && !is_prefill && mtp_ok
                        && (!prefer_dflash || preferred.is_none());

                    let forward_result: Result<Vec<Vec<u32>>> = if use_mtp {
                        match Self::run_forward_mtp(&runners, &owned_seqs) {
                            Ok(step_outs) => {
                                mtp_next_k = step_outs.iter().map(|o| o.next_k).collect();
                                Ok(step_outs.into_iter().map(|o| o.tokens).collect())
                            }
                            Err(e) => Err(e),
                        }
                    } else if use_dflash {
                        Self::run_forward_dflash(&runners, &owned_seqs)
                    } else {
                        Self::run_forward(&runners, &owned_seqs, is_prefill)
                            .map(|ids| ids.into_iter().map(|t| vec![t]).collect())
                    };

                    match forward_result {
                        Ok(multi_output_ids) => {
                            let mut guard = engine.write();
                            // The dynamic-K feedback: record the runner-reported per-seq next-K
                            // so the NEXT step's pre-alloc uses the runner's real signal.
                            if !mtp_next_k.is_empty() {
                                guard.scheduler.record_spec_k_feedback(&scheduled_ids, &mtp_next_k);
                            }
                            match guard.finish_step(scheduled_ids, is_prefill, multi_output_ids) {
                                Ok(n) => task_processed = n,
                                Err(e) => {
                                    crate::log_error!("[Engine Loop] Finish error: {:?}", e);
                                    guard.cancel_all_with_reason(Some(e.to_string()));
                                }
                            }
                        }
                        Err(e) => {
                            crate::log_error!("[Engine Loop] Forward error: {:?}", e);
                            let mut guard = engine.write();
                            guard.cancel_all_with_reason(Some(e.to_string()));
                        }
                    }
                } else {
                    let mut guard = engine.write();
                    guard.check_canceled(None);
                }

                if task_processed == 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(if is_pd_server {
                        PD_PREFILL_STATUS_CHECK_COOLING_PERIOD as u64
                    } else {
                        1
                    }))
                    .await;
                }
                tokio::task::yield_now().await;
            }
        });
    }

    pub fn get_model_info(&self) -> (bool, String, Option<usize>) {
        (
            self.has_vision,
            self.model_name.clone(),
            self.econfig.max_model_len,
        )
    }

    /// Get a clone of the chat template for external use (e.g., tokenization without generation)
    pub fn get_chat_template(&self) -> ChatTemplate {
        self.template.clone()
    }

    /// Get a clone of the default chat template for grammar generation
    pub fn get_default_chat_template(&self) -> String {
        self.default_chat_template.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::LLMEngine;
    use crate::utils::guidance::GuidanceTokens;

    // The ff fast-forward is now KV-backed (the gate's `run_ff_forward` writes the ff
    // span's KV before committing the ff run, the no the un-backed desync that wedged
    // tool-call generation). The ff literals are appended to the Sequence in one step
    // (the batched ff, the "for free" CPU-side extraction) + the KV is allocated (the
    // scheduler's `pre_allocate_ff_blocks`) + freed (the next step's `may_append`
    // reconciliation, the unused reserve). The ff is gated on `spec_ff() && !is_prefill`
    // (the env + the decode phase), inlined in `finish_step` (the no a separate seam).
    #[test]
    fn ff_prealloc_reserves_the_ff_span() {
        // The `pre_allocate_ff_blocks` reserves the per-seq learned ff length past the
        // base (the seq.len + 1 + ff_len), so the ff forward (the `run_ff_forward`) has
        // KV room. The ff_len is the previous step's actual ff run (the runner feedback,
        // not a static reserve); the floor is the FF_RESERVE_DEFAULT (the first step
        // with no history). This test verifies the reserve math with a given ff_len
        // (not a full scheduler instance, which is heavy).
        let ff_len: usize = 12; // the learned ff run length (not the static 32)
        let seq_len: usize = 100;
        let block_size: usize = 16;
        // The needed length: the base (seq_len) + the ff span (1 + ff_len).
        let needed_len = seq_len + 1 + ff_len;
        let needed_blocks = needed_len.div_ceil(block_size);
        // The prior block count (the seq at seq_len, the no the ff span yet).
        let prior_blocks = seq_len.div_ceil(block_size);
        assert!(
            needed_blocks > prior_blocks,
            "the ff pre-alloc reserves more blocks than the prior (the ff span needs room): needed {needed_blocks} > prior {prior_blocks}"
        );
        // The reserve is the learned ff length (the no a static cap): the needed length
        // is exactly seq_len + 1 + ff_len.
        assert_eq!(needed_len, seq_len + 1 + ff_len, "the ff pre-alloc covers the learned ff span (the seq_len + 1 + ff_len)");
    }

    #[test]
    fn trim_prompt_replay_prefix_accepts_single_reasoning_token() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42, 99],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            newline_tokens: Vec::new(),
            add_bos_token: false,
            add_eos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[99], &guidance_tokens),
            Some(vec![99])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_accepts_multi_token_suffix_when_first_token_is_reasoning() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            newline_tokens: Vec::new(),
            add_bos_token: false,
            add_eos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[42, 7], &guidance_tokens),
            Some(vec![42, 7])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_trims_leading_non_reasoning_tokens() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            newline_tokens: Vec::new(),
            add_bos_token: false,
            add_eos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[7, 42, 8], &guidance_tokens),
            Some(vec![42, 8])
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_rejects_suffix_without_reasoning_token() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            newline_tokens: Vec::new(),
            add_bos_token: false,
            add_eos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[7, 8], &guidance_tokens),
            None
        );
    }

    #[test]
    fn trim_prompt_replay_prefix_rejects_empty_suffix() {
        let guidance_tokens = GuidanceTokens {
            bos_token_ids: Vec::new(),
            eos_token_ids: Vec::new(),
            reasoning_start_ids: vec![42],
            reasoning_end_ids: vec![100],
            tool_call_start_ids: Vec::new(),
            tool_call_end_ids: Vec::new(),
            newline_tokens: Vec::new(),
            add_bos_token: false,
            add_eos_token: false,
        };

        assert_eq!(
            LLMEngine::trim_prompt_replay_prefix(&[], &guidance_tokens),
            None
        );
    }

    }
