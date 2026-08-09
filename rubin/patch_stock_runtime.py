#!/usr/bin/env python3
"""Patch the pinned Kimi-K3 SGLang image for latent K3V3 quality simulation."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_memory_pool(text: str) -> str:
    text = replace_once(
        text,
        "import abc\nimport copy\n",
        "import abc\nimport concurrent.futures\nimport copy\nimport shutil\n",
        label="memory_pool async calibration imports",
    )
    init_anchor = """        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        self._create_buffers()
"""
    init_replacement = """        self.kv_cache_dim = (
            override_kv_cache_dim
            if self.dsa_kv_cache_store_fp8
            else (kv_lora_rank + qk_rope_head_dim)
        )

        # Kimi-K3 quality-only simulation: quantize the 512-d MLA latent before
        # it enters the cache. The RoPE suffix remains in the model dtype.
        self._kimi_latent_layer_ids = [
            int(value)
            for value in os.environ.get(
                "SGLANG_KIMI_LATENT_LAYER_IDS",
                ",".join(str(i) for i in range(layer_num)),
            ).split(",")
            if value
        ]
        if len(self._kimi_latent_layer_ids) != layer_num:
            raise ValueError(
                "SGLANG_KIMI_LATENT_LAYER_IDS must contain exactly "
                f"{layer_num} ids, got {self._kimi_latent_layer_ids}"
            )
        self._kimi_latent_dump_dir = os.environ.get(
            "SGLANG_KIMI_LATENT_CALIBRATION_DIR"
        )
        self._kimi_latent_dump_local_dir = os.environ.get(
            "SGLANG_KIMI_LATENT_CALIBRATION_LOCAL_DIR"
        )
        self._kimi_latent_dump_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if self._kimi_latent_dump_dir
            and self._kimi_latent_dump_local_dir
            else None
        )
        self._kimi_latent_dump_limit = int(
            os.environ.get("SGLANG_KIMI_LATENT_CALIBRATION_TOKENS", "8192")
        )
        self._kimi_latent_dump_min_chunk = int(
            os.environ.get("SGLANG_KIMI_LATENT_CALIBRATION_MIN_CHUNK", "256")
        )
        self._kimi_latent_dump_saved = {}
        self._kimi_latent_dump_chunks = {}
        self._kimi_latent_codebooks = None
        self._kimi_latent_rotations = None
        self._kimi_latent_oscar = False
        self._kimi_latent_control_file = os.environ.get(
            "SGLANG_KIMI_LATENT_CONTROL_FILE"
        )
        self._kimi_latent_codebook_dir = os.environ.get(
            "SGLANG_KIMI_LATENT_CODEBOOK_DIR"
        )
        self._kimi_latent_control_mode = None
        self._kimi_latent_clip_ratio = float(
            os.environ.get("SGLANG_KIMI_LATENT_OSCAR_CLIP_RATIO", "0.96")
        )
        codebook_path = os.environ.get("SGLANG_KIMI_LATENT_K3_CODEBOOK")
        if codebook_path:
            state = torch.load(codebook_path, map_location="cpu", weights_only=True)
            if state.get("format") != "kimi_mla_latent_lutb_int3_e4m3":
                raise ValueError(
                    f"Unsupported Kimi latent codebook format in {codebook_path}"
                )
            if state.get("method") != "lloyd_max":
                raise ValueError("Kimi latent K3V3 requires offline Lloyd-Max")
            self._kimi_latent_oscar = bool(state.get("oscar", False))
            requested_oscar = (
                os.environ.get("SGLANG_KIMI_LATENT_USE_OSCAR", "0") == "1"
            )
            if requested_oscar != self._kimi_latent_oscar:
                raise ValueError(
                    f"Codebook oscar={self._kimi_latent_oscar} does not match "
                    f"SGLANG_KIMI_LATENT_USE_OSCAR={int(requested_oscar)}"
                )
            if self._kimi_latent_oscar and abs(
                float(state.get("clip_ratio", self._kimi_latent_clip_ratio))
                - self._kimi_latent_clip_ratio
            ) > 1e-7:
                raise ValueError("Runtime and fitted latent OSCAR clip ratios differ")
            codebooks = []
            rotations = []
            layers = state.get("layers", {})
            for global_layer_id in self._kimi_latent_layer_ids:
                entry = layers.get(global_layer_id, layers.get(str(global_layer_id)))
                if entry is None:
                    raise ValueError(
                        f"Codebook {codebook_path} lacks layer {global_layer_id}"
                    )
                codebook = entry["codebook"].float()
                if codebook.shape != (8,) or not torch.all(
                    codebook[1:] > codebook[:-1]
                ):
                    raise ValueError(
                        f"Invalid latent codebook for layer {global_layer_id}"
                    )
                rounded = codebook.to(torch.float8_e4m3fn).float()
                if not torch.equal(codebook, rounded):
                    raise ValueError("Latent codebook entries must be exact E4M3")
                codebooks.append(codebook)
                if self._kimi_latent_oscar:
                    rotation = entry["rotation"].float()
                    if rotation.shape != (kv_lora_rank, kv_lora_rank):
                        raise ValueError(
                            f"Invalid latent rotation for layer {global_layer_id}: "
                            f"{tuple(rotation.shape)}"
                        )
                    rotations.append(rotation)
            self._kimi_latent_codebooks = torch.stack(codebooks).to(
                device=self.device, dtype=torch.float32
            )
            if rotations:
                self._kimi_latent_rotations = torch.stack(rotations).to(
                    device=self.device, dtype=self.dtype
                )
            logger.info(
                "Enabled Kimi MLA latent K3V3: codebook=%s oscar=%s layers=%s",
                codebook_path,
                self._kimi_latent_oscar,
                self._kimi_latent_layer_ids,
            )

        self._create_buffers()
"""
    text = replace_once(
        text,
        init_anchor,
        init_replacement,
        label="memory_pool latent initialization",
    )

    methods_anchor = """    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
"""
    methods_replacement = """    def _refresh_kimi_latent_control(self, local_layer_id: int) -> None:
        if (
            not self._kimi_latent_control_file
            or local_layer_id != self.start_layer
        ):
            return
        try:
            with open(self._kimi_latent_control_file) as handle:
                mode = handle.read().strip()
        except FileNotFoundError:
            mode = "calibration"
        if mode == self._kimi_latent_control_mode:
            return
        if mode == "calibration":
            self._kimi_latent_codebooks = None
            self._kimi_latent_rotations = None
            self._kimi_latent_oscar = False
            self._kimi_latent_control_mode = mode
            logger.info("Kimi latent control switched to calibration/native mode")
            return
        names = {
            "offline_lm_qp": ("latent_lloyd_max.pt", False),
            "oscar_offline_lm_qp": ("latent_oscar_lloyd_max.pt", True),
        }
        if mode not in names:
            raise ValueError(f"Unsupported Kimi latent control mode: {mode!r}")
        if not self._kimi_latent_codebook_dir:
            raise ValueError("SGLANG_KIMI_LATENT_CODEBOOK_DIR is required")
        filename, expected_oscar = names[mode]
        codebook_path = os.path.join(self._kimi_latent_codebook_dir, filename)
        state = torch.load(codebook_path, map_location="cpu", weights_only=True)
        if (
            state.get("format") != "kimi_mla_latent_lutb_int3_e4m3"
            or state.get("method") != "lloyd_max"
            or bool(state.get("oscar", False)) != expected_oscar
        ):
            raise ValueError(f"Incompatible Kimi latent codebook {codebook_path}")
        layers = state.get("layers", {})
        codebooks = []
        rotations = []
        for global_layer_id in self._kimi_latent_layer_ids:
            entry = layers.get(global_layer_id, layers.get(str(global_layer_id)))
            if entry is None:
                raise ValueError(
                    f"Codebook {codebook_path} lacks layer {global_layer_id}"
                )
            codebooks.append(entry["codebook"].float())
            if expected_oscar:
                rotations.append(entry["rotation"].float())
        self._kimi_latent_codebooks = torch.stack(codebooks).to(
            device=self.device, dtype=torch.float32
        )
        self._kimi_latent_rotations = (
            torch.stack(rotations).to(device=self.device, dtype=self.dtype)
            if rotations
            else None
        )
        self._kimi_latent_oscar = expected_oscar
        self._kimi_latent_control_mode = mode
        logger.info(
            "Kimi latent control switched to %s using %s", mode, codebook_path
        )

    def _kimi_global_latent_layer_id(self, local_layer_id: int) -> int:
        local = local_layer_id - self.start_layer
        if not 0 <= local < len(self._kimi_latent_layer_ids):
            raise IndexError(f"Invalid dense MLA layer id {local_layer_id}")
        return self._kimi_latent_layer_ids[local]

    @staticmethod
    def _copy_kimi_latent_dump(source: str, destination: str) -> None:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(source, destination)

    def _maybe_dump_kimi_latent(
        self, local_layer_id: int, cache_k_nope: torch.Tensor
    ) -> None:
        if not self._kimi_latent_dump_dir:
            return
        if cache_k_nope.shape[0] < self._kimi_latent_dump_min_chunk:
            return
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return
        global_layer_id = self._kimi_global_latent_layer_id(local_layer_id)
        saved = int(self._kimi_latent_dump_saved.get(global_layer_id, 0))
        remaining = self._kimi_latent_dump_limit - saved
        if remaining <= 0:
            return
        count = min(int(cache_k_nope.shape[0]), remaining)
        chunk = int(self._kimi_latent_dump_chunks.get(global_layer_id, 0))
        directory = os.path.join(
            self._kimi_latent_dump_dir,
            f"layer_{global_layer_id}",
            "latent",
        )
        if self._kimi_latent_dump_executor is None:
            os.makedirs(directory, exist_ok=True)
        write_directory = (
            os.path.join(
                self._kimi_latent_dump_local_dir,
                f"layer_{global_layer_id}",
                "latent",
            )
            if self._kimi_latent_dump_local_dir
            else directory
        )
        os.makedirs(write_directory, exist_ok=True)
        local_path = os.path.join(write_directory, f"{chunk}.pt")
        final_path = os.path.join(directory, f"{chunk}.pt")
        torch.save(
            cache_k_nope[:count].detach().float().cpu(),
            local_path,
        )
        if self._kimi_latent_dump_executor is not None:
            self._kimi_latent_dump_executor.submit(
                self._copy_kimi_latent_dump, local_path, final_path
            )
        self._kimi_latent_dump_saved[global_layer_id] = saved + count
        self._kimi_latent_dump_chunks[global_layer_id] = chunk + 1
        logger.info(
            "Dumped Kimi MLA latent layer=%d chunk=%d tokens=%d total=%d/%d",
            global_layer_id,
            chunk,
            count,
            saved + count,
            self._kimi_latent_dump_limit,
        )

    def _fake_quantize_kimi_latent(
        self, local_layer_id: int, cache_k_nope: torch.Tensor
    ) -> torch.Tensor:
        self._refresh_kimi_latent_control(local_layer_id)
        if self._kimi_latent_codebooks is None:
            return cache_k_nope
        local = local_layer_id - self.start_layer
        original_dtype = cache_k_nope.dtype
        original_shape = cache_k_nope.shape
        values = cache_k_nope.reshape(-1, self.kv_lora_rank)
        if self._kimi_latent_rotations is not None:
            rotation = self._kimi_latent_rotations[local]
            values = values.to(rotation.dtype) @ rotation
            clip_index = min(
                int(self._kimi_latent_clip_ratio * self.kv_lora_rank),
                self.kv_lora_rank - 1,
            )
            threshold = torch.kthvalue(
                values.float().abs(), clip_index + 1, dim=-1
            ).values
            values = values.float().clamp(
                min=-threshold.unsqueeze(-1),
                max=threshold.unsqueeze(-1),
            )
        codebook = self._kimi_latent_codebooks[local]
        boundaries = (codebook[:-1] + codebook[1:]) * 0.5
        indices = torch.bucketize(values.float(), boundaries)
        values = codebook[indices]
        if self._kimi_latent_rotations is not None:
            rotation = self._kimi_latent_rotations[local]
            values = values.to(rotation.dtype) @ rotation.T
        return values.reshape(original_shape).to(original_dtype).contiguous()

    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
"""
    text = replace_once(
        text,
        methods_anchor,
        methods_replacement,
        label="memory_pool latent methods",
    )

    write_anchor = """        layer_id = layer.layer_id
        self._write_mla_kv_buffer(
            self.kv_buffer[layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
        )
"""
    write_replacement = """        layer_id = layer.layer_id
        self._maybe_dump_kimi_latent(layer_id, cache_k_nope)
        original_cache_k_nope = cache_k_nope
        cache_k_nope = self._fake_quantize_kimi_latent(layer_id, cache_k_nope)
        if cache_k_nope.data_ptr() != original_cache_k_nope.data_ptr():
            original_cache_k_nope.copy_(cache_k_nope)
            cache_k_nope = original_cache_k_nope
        self._write_mla_kv_buffer(
            self.kv_buffer[layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
        )
"""
    text = replace_once(
        text,
        write_anchor,
        write_replacement,
        label="memory_pool latent write hook",
    )

    hybrid_anchor = """    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
"""
    hybrid_replacement = """    def get_v_head_dim(self):
        return self.full_kv_pool.get_value_buffer(0).shape[-1]

    def prepare_kimi_latent_kv(
        self,
        layer: RadixAttention,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        if not self.use_mla:
            return cache_k, cache_v
        with self._transfer_id_context(layer):
            return self.full_kv_pool.prepare_kimi_latent_kv(
                layer.layer_id, cache_k, cache_v
            )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
"""
    text = replace_once(
        text,
        hybrid_anchor,
        hybrid_replacement,
        label="hybrid latent prepare delegation",
    )

    pool_prepare_anchor = """    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
"""
    pool_prepare_replacement = """    def prepare_kimi_latent_kv(
        self,
        local_layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        if cache_k.shape[-1] != self.kv_cache_dim:
            # Expanded-MHA prefill passes per-head K (192-d for Kimi K3).
            # set_mla_kv_buffer already transformed the normalized latent.
            return cache_k, cache_v
        cache_k_nope = cache_k[..., : self.kv_lora_rank]
        cache_k_rope = cache_k[..., self.kv_lora_rank :]
        self._maybe_dump_kimi_latent(local_layer_id, cache_k_nope)
        cache_k_nope = self._fake_quantize_kimi_latent(
            local_layer_id, cache_k_nope
        )
        cache_k = torch.cat((cache_k_nope, cache_k_rope), dim=-1)
        if cache_v is not None and cache_v.shape[-1] == self.kv_lora_rank:
            cache_v = cache_k_nope.reshape_as(cache_v)
        return cache_k.contiguous(), (
            cache_v.contiguous() if cache_v is not None else None
        )

    def _write_mla_kv_buffer(
        self,
        dst_buffer: torch.Tensor,
"""
    return replace_once(
        text,
        pool_prepare_anchor,
        pool_prepare_replacement,
        label="MLA latent current-token preparation",
    )


def patch_triton_backend(text: str) -> str:
    text = replace_once(
        text,
        "from dataclasses import dataclass\n",
        "import os\nfrom dataclasses import dataclass\n",
        label="triton backend os import",
    )
    helper_anchor = """_MLA_DECODE_MIN_BLOCK_KV = 32


def _mla_decode_kv_splits_cap(
"""
    helper_replacement = """_MLA_DECODE_MIN_BLOCK_KV = 32


def _kimi_latent_qp_control_active() -> bool:
    control_file = os.environ.get("SGLANG_KIMI_LATENT_CONTROL_FILE")
    if not control_file:
        return True
    try:
        with open(control_file) as handle:
            return handle.read().strip() in {
                "offline_lm_qp",
                "oscar_offline_lm_qp",
            }
    except FileNotFoundError:
        return False


def _kimi_latent_scaled_e4m3_qdq(values: torch.Tensor) -> torch.Tensor:
    if (
        os.environ.get("SGLANG_KIMI_LATENT_Q_QDQ", "0") != "1"
        or not _kimi_latent_qp_control_active()
    ):
        return values
    block_size = 32
    if values.shape[-1] % block_size:
        raise ValueError(
            f"Kimi MLA Q dimension {values.shape[-1]} is not block-32 aligned"
        )
    shaped = values.float().reshape(*values.shape[:-1], -1, block_size)
    max_value = shaped.abs().amax(dim=-1, keepdim=True)
    min_scale = torch.tensor(
        2.0**-127, dtype=torch.float32, device=values.device
    )
    scale = torch.pow(
        2.0,
        torch.ceil(torch.log2((max_value / 448.0).clamp_min(min_scale))),
    )
    restored = (
        (shaped / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .float()
        * scale
    )
    return restored.reshape_as(values).to(values.dtype)


def _kimi_latent_p_qdq_enabled() -> bool:
    return (
        os.environ.get("SGLANG_KIMI_LATENT_P_QDQ", "0") == "1"
        and _kimi_latent_qp_control_active()
    )


def _mla_decode_kv_splits_cap(
"""
    text = replace_once(
        text,
        helper_anchor,
        helper_replacement,
        label="triton backend Q QDQ helper",
    )

    extend_anchor = """    ):
        # TODO: reuse the buffer across layers
        attn_out = getattr(forward_batch, "_attn_output", None)
"""
    extend_replacement = """    ):
        q = _kimi_latent_scaled_e4m3_qdq(q)
        latent_prepare = getattr(
            self.token_to_kv_pool, "prepare_kimi_latent_kv", None
        )
        if self.use_mla and latent_prepare is not None and k is not None:
            k, v = latent_prepare(layer, k, v)
        # TODO: reuse the buffer across layers
        attn_out = getattr(forward_batch, "_attn_output", None)
"""
    text = replace_once(
        text,
        extend_anchor,
        extend_replacement,
        label="triton backend extend Q QDQ",
    )

    extend_call_anchor = """            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )
        return o

    def _forward_extend_dcp(
"""
    extend_call_replacement = """            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            p_qdq=_kimi_latent_p_qdq_enabled(),
        )
        return o

    def _forward_extend_dcp(
"""
    text = replace_once(
        text,
        extend_call_anchor,
        extend_call_replacement,
        label="triton backend extend P QDQ",
    )

    decode_anchor = """        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
"""
    decode_replacement = """        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        q = _kimi_latent_scaled_e4m3_qdq(q)
        latent_prepare = getattr(
            self.token_to_kv_pool, "prepare_kimi_latent_kv", None
        )
        if self.use_mla and latent_prepare is not None and k is not None:
            k, v = latent_prepare(layer, k, v)

        # TODO: reuse the buffer across layers
"""
    text = replace_once(
        text,
        decode_anchor,
        decode_replacement,
        label="triton backend decode Q QDQ",
    )

    decode_call_anchor = """            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )
        return o


class TritonMultiStepDraftBackend:
"""
    decode_call_replacement = """            page_size=self.page_size,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            p_qdq=_kimi_latent_p_qdq_enabled(),
        )
        return o


class TritonMultiStepDraftBackend:
"""
    return replace_once(
        text,
        decode_call_anchor,
        decode_call_replacement,
        label="triton backend decode P QDQ",
    )


def triton_qdq_helpers() -> str:
    return """
@triton.jit
def _round_up_to_ue8m0(values):
    min_value: tl.constexpr = 1.1754943508222875e-38
    bounded = tl.maximum(values.to(tl.float32), min_value)
    return tl.exp2(tl.ceil(tl.log2(bounded))).to(tl.float32)


@triton.jit
def _scaled_e4m3_qdq(values, ROWS: tl.constexpr, COLS: tl.constexpr):
    shaped = tl.reshape(values, (ROWS, COLS // 32, 32))
    scale = _round_up_to_ue8m0(
        tl.max(tl.abs(shaped), axis=2) / 448.0
    )
    normalized = tl.minimum(
        tl.maximum(shaped / scale[:, :, None], -448.0),
        448.0,
    )
    quantized = normalized.to(tl.float8e4nv)
    restored = quantized.to(tl.float32) * scale[:, :, None]
    return tl.reshape(restored, (ROWS, COLS))


"""


def patch_extend_attention(text: str) -> str:
    text = replace_once(
        text,
        "@triton.jit\ndef tanh(x):\n",
        triton_qdq_helpers() + "@triton.jit\ndef tanh(x):\n",
        label="extend P QDQ helpers",
    )
    text = replace_once(
        text,
        """    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 1,
""",
        """    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    P_QDQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 1,
""",
        label="extend P QDQ constexpr",
    )
    for label, old, new in (
        (
            "extend prefix P QDQ",
            """            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if PAGE_SIZE == 1:
""",
            """            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)
            if P_QDQ:
                p = _scaled_e4m3_qdq(p, BLOCK_M, BLOCK_N)

            if PAGE_SIZE == 1:
""",
        ),
        (
            "extend current P QDQ",
            """            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
""",
            """            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)
            if P_QDQ:
                p = _scaled_e4m3_qdq(p, BLOCK_M, BLOCK_N)

            offs_v = (
""",
        ),
    ):
        text = replace_once(text, old, new, label=label)
    text = replace_once(
        text,
        """    skip_prefix=False,
    skip_extend=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
):
""",
        """    skip_prefix=False,
    skip_extend=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    p_qdq=False,
):
""",
        label="extend public P QDQ argument",
    )
    return replace_once(
        text,
        """        HAS_SINK=HAS_SINK,
        STORE_TRANSPOSE=_is_hip,
""",
        """        HAS_SINK=HAS_SINK,
        P_QDQ=p_qdq,
        STORE_TRANSPOSE=_is_hip,
""",
        label="extend kernel P QDQ launch",
    )


def patch_decode_attention(text: str) -> str:
    text = replace_once(
        text,
        "@triton.jit\ndef tanh(x):\n",
        triton_qdq_helpers() + "@triton.jit\ndef tanh(x):\n",
        label="decode P QDQ helpers",
    )
    text = replace_once(
        text,
        """    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
""",
        """    HAS_MLA: tl.constexpr = False,
    P_QDQ: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
""",
        label="decode P QDQ constexpr",
    )
    text = replace_once(
        text,
        """            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
""",
        """            p = tl.exp(qk - n_e_max[:, None])
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            if P_QDQ:
                p = _scaled_e4m3_qdq(p, BLOCK_H, BLOCK_N)
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

""",
        label="decode grouped P QDQ operation",
    )
    signature_anchor = """    has_mla=False,
    use_pdl=False,
    page_size: int = 1,
"""
    if text.count(signature_anchor) != 3:
        raise RuntimeError(
            "decode P QDQ signatures: expected three grouped/dispatch anchors, "
            f"found {text.count(signature_anchor)}"
        )
    text = text.replace(
        signature_anchor,
        """    has_mla=False,
    p_qdq=False,
    use_pdl=False,
    page_size: int = 1,
""",
    )
    text = replace_once(
        text,
        """        HAS_MLA=has_mla,
        USE_PDL=use_pdl,
""",
        """        HAS_MLA=has_mla,
        P_QDQ=p_qdq,
        USE_PDL=use_pdl,
""",
        label="decode grouped kernel P QDQ launch",
    )
    text = replace_once(
        text,
        """        has_mla=has_mla,
        use_pdl=use_pdl,
        page_size=page_size,
""",
        """        has_mla=has_mla,
        p_qdq=p_qdq,
        use_pdl=use_pdl,
        page_size=page_size,
""",
        label="decode grouped internal P QDQ forwarding",
    )
    return replace_once(
        text,
        """            has_mla=has_mla,
            use_pdl=use_pdl,
            page_size=page_size,
""",
        """            has_mla=has_mla,
            p_qdq=p_qdq,
            use_pdl=use_pdl,
            page_size=page_size,
""",
        label="decode dispatch P QDQ forwarding",
    )


def patch_kimi_k3_config(text: str) -> str:
    text = replace_once(
        text,
        "from transformers.configuration_utils import PretrainedConfig\n",
        "import os\n\nfrom transformers.configuration_utils import PretrainedConfig\n",
        label="Kimi config os import",
    )
    anchor = """        else:
            self.text_config = text_config

        if vision_config is None:
"""
    replacement = """        else:
            self.text_config = text_config

        if (
            os.environ.get("SGLANG_KIMI_TWO_LAYER_SMOKE", "0") == "1"
            and self.text_config.linear_attn_config is not None
        ):
            self.text_config.num_hidden_layers = 2
            linear = dict(self.text_config.linear_attn_config)
            linear["kda_layers"] = [1]
            linear["full_attn_layers"] = [2]
            self.text_config.linear_attn_config = linear
            self.text_config.attn_res_block_size = 2

        if vision_config is None:
"""
    return replace_once(
        text,
        anchor,
        replacement,
        label="Kimi two-layer smoke config",
    )


def patch_load_model_utils(text: str) -> str:
    return replace_once(
        text,
        "UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing\n",
        """UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(
    os.environ.get("SGLANG_MODEL_LOAD_BARRIER_TIMEOUT_S", "480")
)  # allow large multi-node checkpoints to tolerate uneven storage throughput
""",
        label="configurable model-load barrier timeout",
    )


PATCHERS = {
    "sglang/srt/mem_cache/memory_pool.py": patch_memory_pool,
    "sglang/srt/layers/attention/triton_backend.py": patch_triton_backend,
    "sglang/kernels/ops/attention/extend_attention.py": patch_extend_attention,
    "sglang/kernels/ops/attention/decode_attention.py": patch_decode_attention,
    "sglang/srt/configs/kimi_k3.py": patch_kimi_k3_config,
    "sglang/srt/model_executor/model_runner_components/load_model_utils.py": (
        patch_load_model_utils
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "python_root",
        type=Path,
        help="Directory containing the stock sglang Python package",
    )
    args = parser.parse_args()
    for relative, patcher in PATCHERS.items():
        path = args.python_root / relative
        source = path.read_text()
        patched = patcher(source)
        path.write_text(patched)
        print(f"patched {path}")


if __name__ == "__main__":
    main()
