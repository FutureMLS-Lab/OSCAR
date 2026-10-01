# OSCAR delta vs upstream f652135d52 (2026-04-11): porting checklist

## OSCAR-related (port)

| status | +/- lines | markers | file |
|---|---:|---:|---|
| M | 1765 | 186 | `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| M | 1559 | 208 | `python/sglang/srt/layers/attention/triton_backend.py` |
| A | 1422 | 102 | `python/sglang/srt/layers/attention/triton_ops/mla_packed_decode.py` |
| A | 1383 | 11 | `python/sglang/srt/models/kimi_k3.py` |
| A | 1241 | 2 | `python/sglang/srt/utils/cuda_vmm_utils.py` |
| A | 1153 | 4 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/decode/flash_with_topk_idx.py` |
| A | 1070 | 202 | `python/sglang/srt/mem_cache/mla_int2_kv_pool.py` |
| A | 1069 | 72 | `python/sglang/QuantKernel/gpu_flush_int2.py` |
| A | 1032 | 177 | `python/sglang/srt/mem_cache/unified_kv_pool.py` |
| A | 1002 | 151 | `python/sglang/srt/mem_cache/mla_packed_kv_pool.py` |
| A | 965 | 90 | `python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py` |
| A | 937 | 12 | `python/sglang/srt/models/kimi_k3_vl.py` |
| M | 811 | 195 | `python/sglang/srt/mem_cache/memory_pool.py` |
| A | 788 | 3 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/prefill/flash_with_topk_idx.py` |
| A | 718 | 8 | `python/sglang/srt/models/minimax_m3.py` |
| A | 653 | 66 | `python/sglang/QuantKernel/fused_hadamard_int2_kv.py` |
| M | 628 | 260 | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |
| A | 613 | 6 | `python/sglang/srt/layers/attn_residual.py` |
| M | 593 | 1 | `python/sglang/srt/layers/attention/fla/kda.py` |
| A | 581 | 58 | `python/sglang/srt/mem_cache/kv_quant_kernels.py` |
| A | 562 | 68 | `python/sglang/QuantKernel/mla_latent_int2.py` |
| A | 550 | 62 | `python/sglang/srt/layers/attention/minimax_sparse_backend.py` |
| M | 534 | 130 | `python/sglang/srt/mem_cache/common.py` |
| M | 515 | 93 | `python/sglang/srt/model_executor/pool_configurator.py` |
| A | 506 | 63 | `python/sglang/srt/layers/attention/quantized_kv_prefill.py` |
| A | 487 | 3 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/prefill/topk_sparse.py` |
| A | 468 | 6 | `python/sglang/srt/layers/k3_sp_collective.py` |
| A | 460 | 3 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/decode/topk_sparse.py` |
| A | 451 | 51 | `python/sglang/srt/mem_cache/mixed_kv_audit.py` |
| A | 410 | 95 | `python/sglang/srt/mem_cache/unified_kv_allocator.py` |
| A | 403 | 18 | `python/sglang/srt/models/gemma4_unified.py` |
| A | 382 | 9 | `python/sglang/srt/layers/k3_ar_fusion.py` |
| M | 354 | 109 | `python/sglang/srt/environ.py` |
| M | 348 | 15 | `python/sglang/srt/layers/attention/fla/fused_recurrent.py` |
| A | 348 | 7 | `python/sglang/srt/multimodal/processors/gemma4_unified_image_processing.py` |
| M | 344 | 45 | `python/sglang/srt/layers/attention/flashattention_backend.py` |
| A | 318 | 4 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/common/utils.py` |
| M | 317 | 60 | `python/sglang/srt/models/utils.py` |
| M | 308 | 11 | `python/sglang/srt/multimodal/mm_utils.py` |
| D | 306 | 8 | `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md` |
| M | 300 | 29 | `python/sglang/srt/mem_cache/radix_cache.py` |
| A | 298 | 9 | `python/sglang/srt/multimodal/kimi_k3_image_processing.py` |
| A | 216 | 10 | `python/sglang/srt/configs/gemma4_unified.py` |
| M | 208 | 40 | `python/sglang/srt/server_args.py` |
| M | 200 | 13 | `python/sglang/multimodal_gen/runtime/layers/quantization/modelopt_quant.py` |
| M | 194 | 17 | `python/sglang/multimodal_gen/runtime/utils/quantization_utils.py` |
| M | 193 | 65 | `python/sglang/srt/layers/attention/nsa_backend.py` |
| M | 177 | 28 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| A | 164 | 5 | `python/sglang/srt/multimodal/processors/gemma4_unified_processing.py` |
| M | 159 | 55 | `python/sglang/srt/models/qwen3_5.py` |
| M | 158 | 21 | `python/sglang/srt/configs/model_config.py` |
| A | 134 | 8 | `python/sglang/srt/layers/moe/route_quant_handoff.py` |
| M | 126 | 4 | `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| A | 124 | 5 | `python/sglang/srt/configs/kimi_k3.py` |
| A | 111 | 50 | `python/sglang/srt/mem_cache/mixed_kv_prefix_mixin.py` |
| A | 96 | 7 | `python/sglang/srt/layers/attention/nsa/packed_staging.py` |
| M | 95 | 17 | `python/sglang/srt/managers/scheduler.py` |
| A | 94 | 3 | `python/sglang/srt/layers/k3_gemm_ar.py` |
| M | 89 | 7 | `python/sglang/srt/managers/scheduler_runtime_checker_mixin.py` |
| M | 81 | 5 | `python/sglang/srt/layers/attention/vision.py` |
| M | 75 | 17 | `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| A | 72 | 5 | `python/sglang/srt/layers/attention/minimax_sparse_staging.py` |
| A | 67 | 3 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/common/index.py` |
| A | 66 | 5 | `python/sglang/srt/multimodal/processors/gemma4_unified.py` |
| A | 56 | 1 | `python/sglang/srt/layers/zero_copy_context.py` |
| M | 43 | 1 | `python/sglang/srt/layers/quantization/unquant.py` |
| M | 27 | 9 | `python/sglang/srt/managers/schedule_batch.py` |
| M | 22 | 2 | `python/sglang/srt/model_executor/forward_batch_info.py` |
| M | 20 | 4 | `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py` |
| M | 19 | 4 | `python/sglang/srt/mem_cache/base_prefix_cache.py` |
| M | 18 | 4 | `python/sglang/srt/layers/attention/attention_registry.py` |
| A | 17 | 11 | `python/sglang/QuantKernel/__init__.py` |
| A | 15 | 7 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/__init__.py` |
| M | 15 | 3 | `python/sglang/srt/mem_cache/chunk_cache.py` |
| M | 15 | 1 | `python/sglang/srt/model_executor/model_runner.py` |
| M | 15 | 6 | `python/sglang/srt/models/glm4_moe.py` |
| M | 14 | 2 | `python/sglang/srt/model_loader/weight_utils.py` |
| M | 13 | 6 | `python/sglang/srt/models/qwen3.py` |
| M | 12 | 6 | `python/sglang/srt/models/qwen3_moe.py` |
| M | 10 | 1 | `python/sglang/srt/models/kimi_linear.py` |
| M | 6 | 1 | `python/sglang/srt/configs/__init__.py` |
| A | 0 | 2 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/common/__init__.py` |
| A | 0 | 2 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/decode/__init__.py` |
| A | 0 | 2 | `python/sglang/srt/layers/attention/minimax_sparse_kernels/prefill/__init__.py` |

## No OSCAR marker (Together-fork / other; port only if a model needs it)

| status | +/- lines | file |
|---|---:|---|
| M | 703 | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| M | 618 | `python/sglang/srt/layers/attention/fla/chunk_intra.py` |
| D | 498 | `python/sglang/multimodal_gen/tools/compare_diffusion_trajectory_similarity.py` |
| D | 494 | `python/sglang/multimodal_gen/tools/convert_modelopt_fp8_checkpoint.py` |
| D | 464 | `python/sglang/srt/layers/attention/fla/solve_tril.py` |
| A | 412 | `python/sglang/srt/multimodal/transport/memory_pool.py` |
| A | 361 | `python/sglang/srt/multimodal/transport/cuda_ipc.py` |
| A | 217 | `python/sglang/srt/layers/attention/vision_rope.py` |
| M | 216 | `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| M | 180 | `python/sglang/multimodal_gen/test/server/accuracy_utils.py` |
| M | 175 | `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py` |
| M | 170 | `python/sglang/srt/layers/attention/fla/l2norm.py` |
| M | 167 | `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` |
| A | 165 | `python/sglang/srt/multimodal/encoder_preprocessing.py` |
| A | 161 | `python/sglang/srt/runtime_context.py` |
| M | 160 | `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` |
| D | 147 | `python/sglang/srt/layers/attention/fla/chunk_scaled_dot_kkt.py` |
| A | 144 | `python/sglang/srt/multimodal/processors/kimi_common.py` |
| D | 142 | `python/sglang/jit_kernel/tests/diffusion/test_diffusion_modelopt_fp8_scaled_mm.py` |
| M | 112 | `python/sglang/multimodal_gen/runtime/loader/transformer_load_utils.py` |
| M | 107 | `python/sglang/multimodal_gen/test/run_suite.py` |
| M | 87 | `python/sglang/srt/layers/moe/topk.py` |
| M | 76 | `python/sglang/multimodal_gen/test/server/accuracy_hooks.py` |
| M | 76 | `python/sglang/multimodal_gen/test/server/component_accuracy.py` |
| A | 75 | `python/sglang/srt/layers/moe/triton_kernels_compat.py` |
| M | 72 | `python/sglang/srt/layers/quantization/mxfp4.py` |
| M | 71 | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| M | 68 | `python/sglang/srt/managers/schedule_policy.py` |
| M | 66 | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| M | 65 | `python/sglang/srt/layers/attention/linear/kda_backend.py` |
| M | 63 | `python/sglang/srt/layers/attention/fla/cumsum.py` |
| M | 59 | `python/sglang/srt/utils/hf_transformers_utils.py` |
| A | 57 | `python/sglang/srt/kernels_compat.py` |
| M | 47 | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| M | 44 | `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` |
| M | 44 | `python/sglang/srt/models/kimi_vl_moonvit.py` |
| M | 44 | `python/sglang/srt/models/minimax_m2.py` |
| M | 38 | `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model/SKILL.md` |
| M | 33 | `python/sglang/srt/layers/activation.py` |
| M | 29 | `python/sglang/srt/observability/scheduler_metrics_mixin.py` |
| M | 28 | `python/sglang/srt/layers/dp_attention.py` |
| M | 27 | `python/sglang/srt/layers/attention/fla/chunk.py` |
| M | 23 | `python/sglang/srt/eplb/expert_distribution.py` |
| A | 20 | `python/sglang/srt/multimodal/transport/__init__.py` |
| M | 19 | `python/sglang/srt/configs/kimi_linear.py` |
| M | 18 | `python/sglang/srt/layers/vocab_parallel_embedding.py` |
| M | 18 | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` |
| M | 17 | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` |
| M | 16 | `python/sglang/srt/model_executor/forward_batch_deepseek_mha_mixin.py` |
| M | 14 | `python/sglang/srt/layers/attention/fla/chunk_fwd.py` |
| M | 14 | `python/sglang/srt/layers/attention/fla/chunk_o.py` |
| M | 14 | `python/sglang/srt/layers/attention/fla/wy_fast.py` |
| A | 14 | `python/sglang/srt/layers/dcp/planner.py` |
| M | 13 | `python/sglang/multimodal_gen/test/server/accuracy_config.py` |
| M | 12 | `python/sglang/srt/configs/qwen3_asr.py` |
| M | 12 | `python/sglang/srt/layers/attention/linear/gdn_backend.py` |
| M | 11 | `python/sglang/srt/layers/communicator.py` |
| M | 11 | `python/sglang/srt/layers/quantization/utils.py` |
| A | 11 | `python/sglang/srt/model_executor/runner.py` |
| M | 11 | `python/sglang/srt/utils/common.py` |
| M | 10 | `python/sglang/srt/layers/linear.py` |
| M | 10 | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` |
| A | 10 | `python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/context.py` |
| A | 9 | `python/sglang/srt/layers/__init__.py` |
| M | 9 | `python/sglang/srt/models/deepseek_janus_pro.py` |
| M | 8 | `python/sglang/srt/layers/moe/utils.py` |
| M | 7 | `python/sglang/srt/layers/attention/fla/chunk_intra_token_parallel.py` |
| M | 6 | `python/sglang/srt/layers/attention/fla/utils.py` |
| M | 6 | `python/sglang/srt/layers/moe/moe_runner/base.py` |
| M | 6 | `python/sglang/srt/layers/moe/moe_runner/triton.py` |
| M | 6 | `python/sglang/srt/managers/cache_controller.py` |
| M | 6 | `python/sglang/srt/mem_cache/hiradix_cache.py` |
| M | 4 | `python/sglang/multimodal_gen/runtime/layers/quantization/__init__.py` |
| M | 4 | `python/sglang/srt/models/deepseek_v2.py` |
| M | 4 | `python/sglang/srt/models/glm4_moe_lite.py` |
| M | 4 | `python/sglang/srt/models/mimo_v2_flash.py` |
| M | 3 | `python/sglang/multimodal_gen/test/server/test_server_utils.py` |
| M | 3 | `python/sglang/srt/distributed/parallel_state.py` |
| M | 3 | `python/sglang/srt/layers/moe/__init__.py` |
| M | 3 | `python/sglang/srt/layers/moe/moe_runner/triton_kernels.py` |
| M | 3 | `python/sglang/srt/mem_cache/cache_init_params.py` |
| M | 3 | `python/sglang/srt/mem_cache/swa_memory_pool.py` |
| M | 2 | `python/sglang/srt/mem_cache/hicache_storage.py` |
| M | 2 | `python/sglang/srt/models/bailing_moe.py` |
| M | 2 | `python/sglang/srt/models/deepseek.py` |
| M | 2 | `python/sglang/srt/models/exaone_moe.py` |
| M | 2 | `python/sglang/srt/models/interns1pro.py` |
| M | 2 | `python/sglang/srt/models/lfm2_moe.py` |
| M | 2 | `python/sglang/srt/models/llada2.py` |
| M | 2 | `python/sglang/srt/models/llama.py` |
| M | 2 | `python/sglang/srt/models/nemotron_h.py` |
| M | 2 | `python/sglang/srt/models/qwen2_moe.py` |
| M | 2 | `python/sglang/srt/models/sdar_moe.py` |
| M | 1 | `python/sglang/srt/managers/io_struct.py` |
| M | 1 | `python/sglang/srt/managers/utils.py` |
| A | 0 | `python/sglang/srt/layers/attention/fla/__init__.py` |
| A | 0 | `python/sglang/srt/layers/dcp/__init__.py` |
| A | 0 | `python/sglang/srt/model_executor/runner_backend_utils/__init__.py` |
| A | 0 | `python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py` |
| A | 0 | `python/sglang/srt/multimodal/__init__.py` |
