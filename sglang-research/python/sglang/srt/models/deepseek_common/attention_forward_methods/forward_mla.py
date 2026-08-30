from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_cpu,
    _is_cublas_ge_129,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _use_aiter,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA


def _packed_latent_pool(forward_batch, layer_id):
    """The packed-INT2 latent pool, or ``None`` for every other pool.

    Duck-typed on the two methods that define the contract so this file keeps
    no import dependency on the OSCAR pools, and so nothing that merely looks
    like an MLA pool can accidentally qualify.
    """
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is None or not hasattr(pool, "packed_read_operands"):
        return None
    if not hasattr(pool, "rotate_latent"):
        return None
    return pool

if _is_cuda:
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

    from sglang.srt.utils.custom_op import register_custom_op

    # TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
    # Wrap bmm_fp8 as a custom op so torch.compile does not trace into
    # torch.cuda.current_blas_handle() (which returns a non-Tensor).
    @register_custom_op(mutates_args=["out"])
    def _bmm_fp8_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)
        return out


if _use_aiter:
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla


class DeepseekMLAForwardMixin:

    def init_mla_forward(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )

    def forward_absorb_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        # Kimi-K3 gates the attention output per head (g_proj, sigmoid). The
        # expanded-MHA path applies it (forward_mha: attn_output * output_gate
        # before o_proj); this path never did, so serving K3 as a real MLA
        # latent produced fluent-shaped garbage -- an ungated output, not a
        # quantization artefact. It reproduced with kv_cache_dtype=bfloat16 and
        # no packed pool at all, which is what isolated it to the forward path.
        #
        # That is why use_expanded_mha_cache was pinned True: not caution, a
        # missing gate. Computed here because only prepare has hidden_states,
        # and always assigned -- including None -- so a stale gate from an
        # earlier layer can never be applied to this one.
        _has_gate = getattr(self, "g_proj", None) is not None
        if not getattr(self, "_mla_gate_logged", False):
            # Log once per layer that the gate is present and being applied.
            # The gate fix was asserted to be live on the strength of the code
            # reading correctly; nothing ever confirmed it from a running
            # server. Five hypotheses about this path have now been wrong, and
            # the two that were RIGHT -- the missing gate, and the kernel being
            # innocent -- were both settled by an instrument rather than an
            # argument.
            import logging as _lg

            self._mla_gate_logged = True
            _lg.getLogger(__name__).info(
                "[MLA-GATE] layer=%s g_proj=%s qk_head_dim=%s scaling=%.6f "
                "kv_lora_rank=%s v_head_dim=%s",
                getattr(self, "layer_id", "?"), _has_gate,
                getattr(self, "qk_head_dim", None), getattr(self, "scaling", float("nan")),
                getattr(self, "kv_lora_rank", None), getattr(self, "v_head_dim", None),
            )
        self._mla_output_gate = (
            self.g_proj(hidden_states)[0].sigmoid() if _has_gate else None
        )

        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        q = self.q_a_layernorm(q)
                        k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_nsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                if not self.skip_topk or prev_topk_indices is None:
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    topk_indices = prev_topk_indices
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    if not self.skip_topk or prev_topk_indices is None:
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = prev_topk_indices
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if (_use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn) or (
                    get_is_capture_mode() and self.w_kc.dtype == torch.float8_e4m3fnuz
                ):
                    # fp8 Triton kernel: always on gfx950,
                    # cudagraph-only on gfx942 (hides launch overhead)
                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        elif self.w_kc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                q_nope_out = torch.bmm(
                    q_nope.to(torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(torch.bfloat16) * self.w_scale,
                )
            else:
                # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                    q_nope.transpose(0, 1),
                    (
                        torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                q_nope_out = bmm_fp8(
                    q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
                )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        skip_rope_for_nsa_tilelang_fused = self._skip_rope_for_nsa_tilelang_fused()
        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not skip_rope_for_nsa_tilelang_fused)
            and (not _use_aiter or not _is_gfx95_supported or self.use_nsa)
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    def forward_absorb_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):
        save_kv_cache = True

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            if self._skip_rope_for_nsa_tilelang_fused() and self.rotary_emb is not None:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )
                q_cat, _, k_pe_fused, _ = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )
                q_nope_fused = q_cat[..., : self.kv_lora_rank]
                q_pe_fused = q_cat[..., self.kv_lora_rank :]
                save_kv_cache = False
                if llama_4_scaling is not None:
                    q_nope_fused *= llama_4_scaling
                attn_output = self.attn_mqa(
                    q_nope_fused,
                    None,
                    None,
                    forward_batch,
                    q_rope=q_pe_fused,
                    k_rope=k_pe_fused,
                    save_kv_cache=save_kv_cache,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
            else:
                extra_args = {}
                if self._fuse_rope_for_trtllm_mla(forward_batch):
                    extra_args = {
                        "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                        "is_neox": self.rotary_emb.is_neox_style,
                        "llama_4_scaling": llama_4_scaling,
                    }
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
        else:
            _packed = None
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Packed-INT2 latent: the cache holds the *rotated* latent, because
            # that is the frame the quantizer is calibrated in and un-rotating
            # every row a kernel touches would cost more than the read. Cancel
            # it at the two ends instead, where it is a 512x512 matmul on the
            # query and on the attention output rather than on the cache:
            #
            #     q . c^T = (q R) . (c R)^T          out = (sum p (c R)) R^T
            #
            # The fresh keys of this forward go to the kernel rotated too, so
            # they sit in the same frame as the cached rows they are
            # concatenated with; the pool rotates its own copy on the write.
            # Only the plain-concat branch above builds the q/k this needs; the
            # aiter fused path writes the cache itself and has no `k` to rotate.
            _packed = (
                None if _use_aiter_gfx95
                else _packed_latent_pool(forward_batch, self.attn_mqa.layer_id)
            )
            if _packed is not None:
                _lid = self.attn_mqa.layer_id
                _packed.set_kv_buffer(self.attn_mqa, forward_batch.out_cache_loc, k, k_nope)
                save_kv_cache = False
                k_nope = _packed.rotate_latent(_lid, k_nope)
                q = torch.cat(
                    [_packed.rotate_latent(_lid, q_nope_out), q_pe], dim=-1
                )
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
            if _packed is not None:
                attn_output = _packed.unrotate_output(
                    self.attn_mqa.layer_id,
                    attn_output.view(-1, self.num_local_heads, self.kv_lora_rank),
                )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
                )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        # Same gate, same place as the expanded-MHA path: on the
        # [tokens, num_local_heads * v_head_dim] tensor, before o_proj.
        _gate = getattr(self, "_mla_output_gate", None)
        if _gate is not None:
            attn_bmm_output = attn_bmm_output * _gate
        _trace_attn_out(self, "absorb", attn_bmm_output)
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        # Return topk_indices for the next layer when enabling index cache
        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices

    def _fuse_rope_for_trtllm_mla(
        self: DeepseekV2AttentionMLA, forward_batch: ForwardBatch
    ) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        if self.current_attention_backend == "nsa":
            return (
                get_global_server_args().nsa_decode_backend == "trtllm"
                or get_global_server_args().nsa_prefill_backend == "trtllm"
            ) and forward_batch.attn_backend.kv_cache_dtype == torch.float8_e4m3fn

        return (
            self.current_attention_backend == "trtllm_mla"
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
        )

    def _skip_rope_for_nsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
        """
        Check if we should skip rope and use fused rope+cache path for TileLang NSA on gfx95.
        """
        server_args = get_global_server_args()
        return (
            _use_aiter_gfx95
            and self.current_attention_backend == "nsa"
            and (
                server_args.nsa_decode_backend == "tilelang"
                or server_args.nsa_prefill_backend == "tilelang"
            )
        )


def _trace_attn_out(mod, tag: str, t) -> None:
    """Log the pre-o_proj tensor's norm once per layer, in BOTH MLA paths.

    K3's absorbed path yields fluent but off-task text while the expanded-MHA
    path scores 61.7, and the BF16 control proves the fault is in the forward
    rather than in quantization. Adding g_proj fixed the loudest symptom and not
    the defect, and scaling was measured correct, so what is left is unknown.

    Both paths reduce to the same [tokens, num_local_heads * v_head_dim] tensor
    immediately before o_proj, which makes them directly comparable. Running the
    two arms on the same prompt and diffing this per layer says whether they
    diverge at the FIRST full-attention layer -- a math difference -- or drift
    apart later, which would be accumulation. Those have different causes, and
    seven rounds of reasoning without this measurement produced six wrong
    answers.
    """
    import os

    if os.environ.get("SGLANG_OSCAR_TRACE_ATTN_OUT", "0") in ("0", "", "false"):
        return
    # Skip CUDA-graph capture. The tensors flowing through capture are dummy
    # warmup values, so the first call -- which is what a once-per-layer trace
    # records -- captured all zeros for the MHA arm and meaningless values for
    # the absorbed one. Same mistake the c_kv dump made: a "first call" trace
    # lands in capture, not in real decode.
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if get_is_capture_mode():
            return
    except Exception:  # noqa: BLE001
        pass
    if getattr(mod, "_attn_out_traced", False):
        return
    mod._attn_out_traced = True
    import logging as _lg

    try:
        f = t.float()
        _lg.getLogger(__name__).info(
            "[ATTN-OUT] path=%s layer=%s shape=%s norm=%.6e absmax=%.6e mean=%.6e",
            tag, getattr(mod, "layer_id", "?"), tuple(t.shape),
            f.norm().item(), f.abs().max().item(), f.mean().item(),
        )
    except Exception:  # noqa: BLE001
        pass

