"""Online OSCAR rotation calibration for the unified mixed INT2 KV pool.

The collector sees every prefill's Q/K/V through ``RadixAttention`` while
the pool runs with identity rotations, keeps the exact one-pass second
moments (``qqt`` per KV head on the device, K/V rows in pinned host memory),
and turns them into per-layer ``R = E H P_br`` rotations that the pool
installs in place."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.oscar_rotation_paths import (
    get_oscar_checkpoint_pair,
    get_oscar_pair_artifact_paths,
)

logger = logging.getLogger(__name__)

_GRAM_CHUNK_TOKENS = 512
_COVARIANCE_CHUNK_TOKENS = 2048
_ORTHOGONALITY_TOLERANCE = 5e-3
# Every _RHO_QUERY_STRIDE-th calibration token keeps its query rows, so the
# query heads' attention overlap (the C2.7 head-resolved value objective) can
# be measured against the stored keys in save_moments; 32 leaves ~1k samples
# at the 30k budget, enough to settle a grp x grp matrix to three digits.
_RHO_QUERY_STRIDE = 32

# The pool that owns a pending calibrator registers it here; the attention
# layer reads it on every forward, so there is exactly one per process and the
# lookup must stay a plain module attribute.
_active_calibrator: Optional["OscarOnlineCalibrator"] = None


def set_active_oscar_calibrator(calibrator: Optional["OscarOnlineCalibrator"]) -> None:
    global _active_calibrator
    _active_calibrator = calibrator


def get_active_oscar_calibrator() -> Optional["OscarOnlineCalibrator"]:
    return _active_calibrator


def build_hadamard(n: int, *, device=None, dtype=torch.float64) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            (torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0
        ) / math.sqrt(2)
    return h


def bit_reversal_perm(d: int, *, device=None) -> torch.Tensor:
    if d < 1 or d & (d - 1):
        raise ValueError(f"Bit-reversal size must be a power of two, got {d}")
    bits = int(math.log2(d))
    return torch.tensor(
        [int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(d)],
        device=device,
        dtype=torch.long,
    )


def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    d = eigenvalues.shape[-1]
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(d, device=eigenvalues.device)
    perm = torch.empty(d, device=eigenvalues.device, dtype=torch.long)
    perm[br] = sorted_idx
    return torch.eye(d, device=eigenvalues.device, dtype=eigenvalues.dtype)[:, perm]


def compose_r_h_pbr(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    return eigenvectors @ hadamard @ make_br_perm_matrix(eigenvalues)


def prompt_ids_sha256(input_ids: list[list[int]]) -> str:
    payload = json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class OscarCalibrationResult(msgspec.Struct, kw_only=True):
    k_rotations: torch.Tensor
    v_rotations: torch.Tensor
    k_state: dict
    v_state: dict
    generation_id: str


_active_latent_dump: Optional["OscarLatentDump"] = None


def get_active_oscar_latent_dump() -> Optional["OscarLatentDump"]:
    """The MLA latent dump for ``rotation/tools/fit_mla_joint_latent.py``,
    created on first use from SGLANG_OSCAR_MLA_LATENT_DUMP_DIR; None when the
    knob is unset. Each TP rank writes its own heads."""
    global _active_latent_dump
    if _active_latent_dump is not None:
        return _active_latent_dump
    directory = envs.SGLANG_OSCAR_MLA_LATENT_DUMP_DIR.get()
    if not directory:
        return None
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _active_latent_dump = OscarLatentDump(
        Path(directory),
        token_budget=envs.SGLANG_OSCAR_MLA_LATENT_DUMP_TOKENS.get(),
        rank=rank,
    )
    return _active_latent_dump


class OscarLatentDump:
    """Collect raw MLA prefill rows for the joint latent fitter: every latent
    ``c_kv`` and ``k_pe`` row up to ``token_budget`` per layer, plus the
    absorbed queries ``q_nope_out`` / ``q_pe`` of every ``query_stride``-th
    token (so the fitter can measure the heads' attention overlap against the
    same keys). Rows are kept on the host and written once per layer when the
    budget is met, as ``layer_<id>_rank<r>.pt``; the rows are in the model's
    own frame, before the pool's rotation."""

    def __init__(self, directory: Path, *, token_budget: int, rank: int, query_stride: int = _RHO_QUERY_STRIDE):
        self.directory = directory
        self.token_budget = int(token_budget)
        self.rank = int(rank)
        self.query_stride = int(query_stride)
        self._rows: dict[int, list] = {}
        self._counts: dict[int, int] = {}
        self._written: set[int] = set()

    def observe(
        self,
        *,
        layer_id: int,
        c_kv: torch.Tensor,
        k_pe: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if layer_id in self._written:
            return
        saved = self._counts.get(layer_id, 0)
        remaining = self.token_budget - saved
        if remaining <= 0:
            return
        n = min(int(c_kv.shape[0]), remaining)
        rows = torch.arange(saved, saved + n)
        pick = (rows % self.query_stride) == 0
        self._rows.setdefault(layer_id, []).append(
            {
                "c_kv": c_kv[:n].reshape(n, -1).detach().to("cpu", dtype=torch.bfloat16),
                "k_pe": k_pe[:n].reshape(n, -1).detach().to("cpu", dtype=torch.bfloat16),
                "positions": positions[:n].detach().to("cpu", dtype=torch.int32),
                "q_rows": rows[pick],
                "q_nope": q_nope[:n][pick.to(q_nope.device)].detach().to("cpu", dtype=torch.bfloat16),
                "q_pe": q_pe[:n][pick.to(q_pe.device)].detach().to("cpu", dtype=torch.bfloat16),
            }
        )
        self._counts[layer_id] = saved + n
        if self._counts[layer_id] >= self.token_budget:
            self._write(layer_id)

    def _write(self, layer_id: int) -> None:
        parts = self._rows.pop(layer_id)
        payload = {key: torch.cat([p[key] for p in parts], dim=0) for key in ("c_kv", "k_pe", "positions", "q_rows", "q_nope", "q_pe")}
        payload["tokens"] = self._counts[layer_id]
        payload["rank"] = self.rank
        payload["query_stride"] = self.query_stride
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"layer_{layer_id}_rank{self.rank}.pt"
        tmp = str(path) + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self._written.add(layer_id)
        logger.info("OSCAR latent dump: layer %d, %d rows, %d query samples -> %s", layer_id, payload["tokens"], int(payload["q_rows"].numel()), path)


class OscarOnlineCalibrator:
    """Collect exact one-pass qqt/sst sufficient statistics on the GPU.

    ``layer_ids`` are the global ids the attention layers report; results
    are produced in that order. ``tp_group`` is the attention TP group
    coordinator (None when TP=1 or under test)."""

    def __init__(
        self,
        *,
        layer_ids: Sequence[int],
        local_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        device,
        total_q_heads: int,
        total_kv_heads: int,
        tp_size: int,
        tp_rank: int,
        tp_group=None,
        model_path: str,
        model_revision: str | None = None,
    ):
        self.device = torch.device(device)
        self.max_token_budget = envs.SGLANG_OSCAR_CALIBRATION_TOKENS.get()
        self.token_budget = 0
        self.local_layers = [int(layer_id) for layer_id in layer_ids]
        self.local_kv_heads = int(local_kv_heads)
        self.total_q_heads = int(total_q_heads)
        self.total_kv_heads = int(total_kv_heads)
        self.head_dim = int(head_dim)
        self.v_head_dim = int(v_head_dim)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.tp_group = tp_group
        # Models with fewer KV heads than TP ranks replicate each head over
        # tp_size / total_kv_heads consecutive ranks (rank r holds head r // rep).
        self.kv_replication = max(1, self.tp_size // max(1, self.total_kv_heads))
        self.model_path = model_path
        self.model_revision = model_revision
        self.state = "idle"
        self.prompt_sha256 = ""
        self._validate_geometry()

        self._counts: dict[int, int] = {}
        self._gqa_ratios: dict[int, int] = {}
        self._q_grams: dict[int, torch.Tensor] = {}
        self._k_values: dict[int, torch.Tensor] = {}
        self._v_values: dict[int, torch.Tensor] = {}
        self._q_samples: dict[int, torch.Tensor] = {}
        self._positions: dict[int, torch.Tensor] = {}
        self._scaling: dict[int, float] = {}

    def _validate_geometry(self) -> None:
        if self.max_token_budget <= 0:
            raise ValueError("OSCAR calibration token budget must be positive")
        if not self.local_layers:
            raise ValueError("OSCAR calibration needs at least one attention layer")
        if self.head_dim != self.v_head_dim:
            raise ValueError(
                "Online OSCAR calibration currently requires equal K/V head dimensions"
            )
        if self.head_dim & (self.head_dim - 1):
            raise ValueError(
                f"Online OSCAR calibration requires power-of-two head_dim, got {self.head_dim}"
            )
        if self.tp_size > self.total_kv_heads:
            if self.tp_size % self.total_kv_heads:
                raise ValueError(
                    f"TP ({self.tp_size}) must be a multiple of the global KV heads ({self.total_kv_heads}) when they replicate"
                )
        elif self.total_kv_heads % self.tp_size:
            raise ValueError(
                f"Global KV heads ({self.total_kv_heads}) must divide TP ({self.tp_size})"
            )
        if self.local_kv_heads * self.tp_size != self.total_kv_heads * self.kv_replication:
            raise ValueError(
                "Local/global KV-head geometry does not describe TP shards (with replication)"
            )

    # -- Collection ---------------------------------------------------------

    def start(self, *, prompt_sha256: str, token_budget: int | None = None) -> None:
        if self.state != "idle":
            raise RuntimeError(f"Cannot start OSCAR calibrator from state={self.state}")
        if token_budget is None:
            token_budget = self.max_token_budget
        if token_budget <= 0 or token_budget > self.max_token_budget:
            raise ValueError(
                "OSCAR calibration selected token count must be in "
                f"[1, {self.max_token_budget}], got {token_budget}"
            )
        self.token_budget = token_budget
        self.prompt_sha256 = prompt_sha256
        self._counts = {layer_id: 0 for layer_id in self.local_layers}
        self._gqa_ratios = {}
        self._q_grams = {
            layer_id: torch.zeros(
                (self.local_kv_heads, self.head_dim, self.head_dim),
                dtype=torch.float64,
                device=self.device,
            )
            for layer_id in self.local_layers
        }
        self._k_values = {}
        self._v_values = {}
        self._q_samples = {}
        self._positions = {}
        self._scaling = {}
        self.state = "collecting"

    @property
    def complete(self) -> bool:
        return self.state == "collecting" and all(
            count == self.token_budget for count in self._counts.values()
        )

    def min_captured_tokens(self) -> int:
        return min(self._counts.values(), default=0)

    def observe(
        self,
        *,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
    ) -> None:
        if self.state != "collecting" or layer_id not in self._counts:
            return
        saved = self._counts[layer_id]
        remaining = self.token_budget - saved
        if remaining <= 0:
            return
        num_tokens = min(int(q.shape[0]), int(k.shape[0]), int(v.shape[0]), remaining)
        if num_tokens <= 0:
            return
        q = q[:num_tokens].reshape(num_tokens, -1, self.head_dim)
        k = k[:num_tokens].reshape(num_tokens, self.local_kv_heads, self.head_dim)
        v = v[:num_tokens].reshape(num_tokens, self.local_kv_heads, self.v_head_dim)
        gqa_ratio = self._gqa_ratio_for(layer_id, local_q_heads=q.shape[1])

        for chunk_start in range(0, num_tokens, _GRAM_CHUNK_TOKENS):
            chunk_stop = min(chunk_start + _GRAM_CHUNK_TOKENS, num_tokens)
            chunk_tokens = chunk_stop - chunk_start
            q_grouped = (
                q[chunk_start:chunk_stop]
                .reshape(chunk_tokens, self.local_kv_heads, gqa_ratio, self.head_dim)
                .permute(1, 0, 2, 3)
                .reshape(self.local_kv_heads, chunk_tokens * gqa_ratio, self.head_dim)
                .to(torch.float64)
            )
            self._q_grams[layer_id].add_(torch.bmm(q_grouped.transpose(1, 2), q_grouped))

        if layer_id not in self._k_values:
            self._allocate_host_rows(
                layer_id, k_dtype=k.dtype, v_dtype=v.dtype, q_heads=int(q.shape[1])
            )
        self._k_values[layer_id][saved : saved + num_tokens].copy_(k, non_blocking=True)
        self._v_values[layer_id][saved : saved + num_tokens].copy_(v, non_blocking=True)
        if positions is not None:
            self._positions[layer_id][saved : saved + num_tokens].copy_(
                positions[:num_tokens].to(torch.int32), non_blocking=True
            )
            rows = torch.arange(saved, saved + num_tokens, device=q.device)
            pick = (rows % _RHO_QUERY_STRIDE) == 0
            if bool(pick.any()):
                self._q_samples[layer_id][(rows[pick] // _RHO_QUERY_STRIDE).cpu()] = (
                    q[pick].detach().to("cpu", dtype=torch.float32)
                )
        if scaling is not None:
            self._scaling[layer_id] = float(scaling)
        self._counts[layer_id] = saved + num_tokens

    def _gqa_ratio_for(self, layer_id: int, *, local_q_heads: int) -> int:
        if local_q_heads * self.tp_size != self.total_q_heads:
            raise ValueError(
                "Local/global Q-head geometry does not describe disjoint TP shards"
            )
        if local_q_heads % self.local_kv_heads:
            raise ValueError(
                f"Local Q heads ({local_q_heads}) must be divisible by local "
                f"KV heads ({self.local_kv_heads})"
            )
        gqa_ratio = local_q_heads // self.local_kv_heads
        previous_ratio = self._gqa_ratios.setdefault(layer_id, gqa_ratio)
        if previous_ratio != gqa_ratio:
            raise ValueError(
                f"Layer {layer_id} GQA ratio changed from {previous_ratio} to {gqa_ratio}"
            )
        return gqa_ratio

    def _allocate_host_rows(self, layer_id: int, *, k_dtype, v_dtype, q_heads: int) -> None:
        pin_memory = self.device.type == "cuda"
        self._positions[layer_id] = torch.zeros(
            (self.token_budget,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self._q_samples[layer_id] = torch.zeros(
            ((self.token_budget + _RHO_QUERY_STRIDE - 1) // _RHO_QUERY_STRIDE, q_heads, self.head_dim),
            dtype=torch.float32,
            device="cpu",
        )
        self._k_values[layer_id] = torch.empty(
            (self.token_budget, self.local_kv_heads, self.head_dim),
            dtype=k_dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._v_values[layer_id] = torch.empty(
            (self.token_budget, self.local_kv_heads, self.v_head_dim),
            dtype=v_dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

    # -- Reduction ----------------------------------------------------------

    def local_covariance_sums(self) -> torch.Tensor:
        """``[2, L, hd, hd]`` float64: per-layer sums over local KV heads of
        the Q covariance (K objective) and the attention-energy-weighted V
        covariance (V objective)."""
        if not self.complete:
            incomplete = {
                layer_id: count
                for layer_id, count in self._counts.items()
                if count != self.token_budget
            }
            raise RuntimeError(
                "OSCAR calibration token budget was not reached for every layer: "
                f"{incomplete} (target={self.token_budget})"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        k_sums = []
        v_sums = []
        for layer_id in self.local_layers:
            q_cov = self._q_grams[layer_id] / (self.token_budget * self._gqa_ratios[layer_id])
            k_sums.append(q_cov.sum(dim=0))
            v_sums.append(self._energy_weighted_v_covariance(layer_id, q_cov).sum(dim=0))
        return torch.stack((torch.stack(k_sums), torch.stack(v_sums)))

    def _energy_weighted_v_covariance(self, layer_id: int, q_cov: torch.Tensor) -> torch.Tensor:
        denominator = torch.zeros(self.local_kv_heads, dtype=torch.float64, device=self.device)
        numerator = torch.zeros(
            (self.local_kv_heads, self.v_head_dim, self.v_head_dim),
            dtype=torch.float64,
            device=self.device,
        )
        for start in range(0, self.token_budget, _COVARIANCE_CHUNK_TOKENS):
            stop = min(start + _COVARIANCE_CHUNK_TOKENS, self.token_budget)
            k_chunk = self._k_values[layer_id][start:stop].to(
                device=self.device, dtype=torch.float64, non_blocking=True
            )
            v_chunk = self._v_values[layer_id][start:stop].to(
                device=self.device, dtype=torch.float64, non_blocking=True
            )
            energy = torch.einsum("thd,hde,the->th", k_chunk, q_cov, k_chunk)
            denominator.add_(energy.sum(dim=0))
            numerator.add_(torch.einsum("th,thd,the->hde", energy, v_chunk, v_chunk))
        return numerator / denominator.clamp_min(1e-12)[:, None, None]

    def _co_attention(self, layer_id: int) -> Optional[torch.Tensor]:
        """``[H, grp, grp]``: mean over the sampled calibration queries of
        ``sum_t a[j, t] a[j', t]`` for the query heads reading each KV head,
        the attention overlap that weights the cross-head terms of the C2.7
        head-resolved post-W_O value objective. None when the attention layer
        did not hand over positions and scaling, or no sampled query had its
        whole sequence inside the captured rows."""
        if layer_id not in self._q_samples or layer_id not in self._scaling:
            return None
        gqa = self._gqa_ratios[layer_id]
        positions = self._positions[layer_id]
        samples = self._q_samples[layer_id]
        rho = torch.zeros((self.local_kv_heads, gqa, gqa), dtype=torch.float64, device=self.device)
        n = 0
        for s in range(samples.shape[0]):
            r = s * _RHO_QUERY_STRIDE
            if r >= self.token_budget:
                break
            start = r - int(positions[r])
            if start < 0 or int(positions[start]) != 0:
                continue
            keys = self._k_values[layer_id][start : r + 1].to(self.device, dtype=torch.float32)
            q_s = samples[s].to(self.device).reshape(self.local_kv_heads, gqa, self.head_dim)
            logits = torch.einsum("hjd,lhd->hjl", q_s, keys) * self._scaling[layer_id]
            a = torch.softmax(logits, dim=-1)
            rho.add_(torch.einsum("hjl,hkl->hjk", a, a).to(torch.float64))
            n += 1
        return None if n == 0 else rho / n

    def save_moments(self, directory: Path) -> str:
        """Write this rank's per-(layer, local KV head) sufficient statistics
        so the OSCAR-2 transform family (per-head, centered, whitened / flat /
        fixed-rate key metrics, output-aware values) can be fitted offline by
        ``rotation/tools/fit_oscar2_variants.py`` without re-running the model:

        * ``M_q``  ``[H, hd, hd]``   mean of ``q^T q`` over the tokens and the
                                      query heads that read each KV head
        * ``k_sum`` ``[H, hd]``       sum of ``k`` over tokens
        * ``M_k``  ``[H, hd, hd]``   sum of ``k^T k`` over tokens
        * ``S_v``  ``[H, vd, vd]``   attention-energy-weighted value covariance
                                      (the V objective of ``sst``)
        * ``rho``  ``[H, grp, grp]`` attention overlap of the query heads reading
                                      each KV head (None when not observed)
        * ``count``                   tokens behind every sum

        Each TP rank writes its own heads (``oscar_moments_rank<r>.pt``); the
        fitter merges them. Float64 throughout, like the rotations.
        """
        if not self.complete:
            raise RuntimeError("OSCAR calibration moments requested before the token budget was met")
        if self.tp_rank % self.kv_replication:
            logger.info("OSCAR calibration moments: rank %d replicates rank %d's KV heads, not written", self.tp_rank, self.tp_rank - self.tp_rank % self.kv_replication)
            return ""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        layers: dict = {}
        for layer_id in self.local_layers:
            q_cov = self._q_grams[layer_id] / (self.token_budget * self._gqa_ratios[layer_id])
            k_sum = torch.zeros((self.local_kv_heads, self.head_dim), dtype=torch.float64, device=self.device)
            m_k = torch.zeros(
                (self.local_kv_heads, self.head_dim, self.head_dim), dtype=torch.float64, device=self.device
            )
            for start in range(0, self.token_budget, _COVARIANCE_CHUNK_TOKENS):
                stop = min(start + _COVARIANCE_CHUNK_TOKENS, self.token_budget)
                k_chunk = self._k_values[layer_id][start:stop].to(
                    device=self.device, dtype=torch.float64, non_blocking=True
                )
                k_sum.add_(k_chunk.sum(dim=0))
                m_k.add_(torch.einsum("thd,the->hde", k_chunk, k_chunk))
            s_v = self._energy_weighted_v_covariance(layer_id, q_cov)
            rho = self._co_attention(layer_id)
            layers[layer_id] = {
                "M_q": q_cov.detach().cpu(),
                "k_sum": k_sum.cpu(),
                "M_k": m_k.cpu(),
                "S_v": s_v.detach().cpu(),
                "rho": None if rho is None else rho.cpu(),
                "q_scaling": self._scaling.get(layer_id),
                "count": self.token_budget,
            }
        payload = {
            "format_version": 2,
            "model_path": self.model_path,
            "model_revision": self.model_revision,
            "prompt_sha256": self.prompt_sha256,
            "tokens": self.token_budget,
            # Primary replica ranks only, renumbered so the fitter's merge sees
            # one file per distinct KV-head shard.
            "tp_rank": self.tp_rank // self.kv_replication,
            "tp_size": self.tp_size // self.kv_replication,
            "local_kv_heads": self.local_kv_heads,
            "global_kv_heads": self.total_kv_heads,
            "global_q_heads": self.total_q_heads,
            "head_dim": self.head_dim,
            "v_head_dim": self.v_head_dim,
            "layers": layers,
        }
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"oscar_moments_rank{self.tp_rank // self.kv_replication}.pt"
        tmp = str(path) + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        logger.info(
            "OSCAR calibration moments for %d layers x %d KV heads written to %s",
            len(layers),
            self.local_kv_heads,
            path,
        )
        return str(path)

    def allocate_result_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_layers = len(self.local_layers)
        k_rotations = torch.empty(
            (num_layers, self.head_dim, self.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        return k_rotations, torch.empty_like(k_rotations)

    def finalize(
        self,
        *,
        covariance_sums: torch.Tensor,
        buffers: tuple[torch.Tensor, torch.Tensor],
    ) -> OscarCalibrationResult:
        """All-reduce the moments over TP, decompose on rank 0, and fill the
        rotation buffers there; other ranks receive them in ``broadcast``."""
        if self.state != "collecting":
            raise RuntimeError(f"Cannot finalize OSCAR calibrator from state={self.state}")
        if envs.SGLANG_OSCAR_CALIBRATION_SAVE_MOMENTS.get():
            k_path, _ = get_oscar_checkpoint_pair()
            self.save_moments(Path(os.path.dirname(k_path)))
        if self.tp_group is not None and self.tp_group.world_size > 1:
            torch.distributed.all_reduce(covariance_sums, group=self.tp_group.device_group)
        # Every head was summed once per replica rank.
        covariances = covariance_sums / (self.total_kv_heads * self.kv_replication)
        covariances = (covariances + covariances.transpose(-1, -2)) / 2

        k_rotations, v_rotations = buffers
        k_state: dict = {}
        v_state: dict = {}
        generation_id = ""
        if self.tp_rank == 0:
            num_layers = len(self.local_layers)
            flat = covariances.reshape(2 * num_layers, self.head_dim, self.head_dim)
            eigenvalues, eigenvectors = torch.linalg.eigh(flat)
            hadamard = build_hadamard(self.head_dim, device=self.device, dtype=torch.float64)
            rotations = torch.stack(
                [
                    compose_r_h_pbr(eigenvectors[i], eigenvalues[i], hadamard)
                    for i in range(2 * num_layers)
                ]
            )
            k_rotations.copy_(rotations[:num_layers].float().contiguous())
            v_rotations.copy_(rotations[num_layers:].float().contiguous())
            generation_id = uuid.uuid4().hex
            k_state = self._checkpoint_state(
                "qqt_r_h_pbr", k_rotations, eigenvalues[:num_layers].float(), generation_id
            )
            v_state = self._checkpoint_state(
                "sst_r_h_pbr", v_rotations, eigenvalues[num_layers:].float(), generation_id
            )
        self.state = "computed"
        return OscarCalibrationResult(
            k_rotations=k_rotations,
            v_rotations=v_rotations,
            k_state=k_state,
            v_state=v_state,
            generation_id=generation_id,
        )

    def broadcast_result(self, result: OscarCalibrationResult) -> None:
        if self.tp_group is not None and self.tp_group.world_size > 1:
            source_rank = self.tp_group.ranks[0]
            for tensor in (result.k_rotations, result.v_rotations):
                torch.distributed.broadcast(
                    tensor, src=source_rank, group=self.tp_group.device_group
                )
        self.state = "finalized"

    def _checkpoint_state(
        self,
        objective: str,
        rotations: torch.Tensor,
        eigenvalues: torch.Tensor,
        generation_id: str,
    ) -> dict:
        return {
            "format_version": 1,
            "objective": objective,
            "source_grouping": "layer",
            "calibration": {
                "generation_id": generation_id,
                "model_path": self.model_path,
                "model_revision": self.model_revision,
                "prompt_sha256": self.prompt_sha256,
                "tokens": self.token_budget,
                "max_tokens": self.max_token_budget,
                "global_q_heads": self.total_q_heads,
                "global_kv_heads": self.total_kv_heads,
                "tp_size": self.tp_size,
                "post_rope": True,
                "accumulator_dtype": "float64",
                "created_at_unix": time.time(),
            },
            "layers": {
                layer_id: {
                    "layer_id": layer_id,
                    "rotation": rotations[i].detach().cpu(),
                    "eigenvalues": eigenvalues[i].detach().cpu(),
                }
                for i, layer_id in enumerate(self.local_layers)
            },
        }

    # -- Publication --------------------------------------------------------

    def publish(self, result: OscarCalibrationResult) -> None:
        """Write both checkpoints under the pair lock: temp files, validate,
        pending manifest, rename both, complete manifest."""
        if self.tp_rank != 0:
            return
        configured_k, configured_v = get_oscar_checkpoint_pair()
        k_path = Path(configured_k)
        v_path = Path(configured_v)
        if k_path.parent != v_path.parent:
            raise ValueError(
                "Online OSCAR calibration currently requires K/V checkpoints "
                "to share one destination directory"
            )
        directory = k_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = get_oscar_pair_artifact_paths()
        lock_path = Path(artifacts["lock"])
        pending_path = Path(artifacts["pending"])
        manifest_path = Path(artifacts["complete"])
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            tmp_paths = []
            try:
                tmp_k = self._write_checkpoint_temp(result.k_state, directory, "k")
                tmp_paths.append(tmp_k)
                tmp_v = self._write_checkpoint_temp(result.v_state, directory, "v")
                tmp_paths.append(tmp_v)
                self._validate_checkpoint(tmp_k, result.generation_id)
                self._validate_checkpoint(tmp_v, result.generation_id)
                self._write_manifest(
                    pending_path,
                    {
                        "generation_id": result.generation_id,
                        "k_path": k_path.name,
                        "v_path": v_path.name,
                    },
                )
                os.replace(tmp_k, k_path)
                tmp_paths.remove(tmp_k)
                os.replace(tmp_v, v_path)
                tmp_paths.remove(tmp_v)
                self._write_manifest(
                    manifest_path,
                    {
                        "generation_id": result.generation_id,
                        "k_path": k_path.name,
                        "v_path": v_path.name,
                        "prompt_sha256": self.prompt_sha256,
                        "tokens": self.token_budget,
                    },
                )
                pending_path.unlink(missing_ok=True)
            finally:
                for path in tmp_paths:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _write_checkpoint_temp(state: dict, directory: Path, label: str) -> str:
        fd, path = tempfile.mkstemp(prefix=f".oscar_{label}_", suffix=".pt.tmp", dir=directory)
        os.close(fd)
        torch.save(state, path)
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())
        return path

    def _validate_checkpoint(self, path: str, generation_id: str) -> None:
        state = torch.load(path, map_location="cpu")
        if state.get("calibration", {}).get("generation_id") != generation_id:
            raise ValueError(f"Invalid OSCAR checkpoint generation id in {path}")
        layers = state.get("layers", {})
        if set(map(int, layers.keys())) != set(self.local_layers):
            raise ValueError(f"Invalid OSCAR layer set in {path}")
        eye = torch.eye(self.head_dim, dtype=torch.float32)
        for layer_id in self.local_layers:
            entry = layers.get(layer_id, layers.get(str(layer_id)))
            rotation = entry["rotation"]
            if rotation.shape != (self.head_dim, self.head_dim):
                raise ValueError(f"Layer {layer_id} rotation has invalid shape {rotation.shape}")
            if not bool(torch.isfinite(rotation).all()):
                raise ValueError(f"Layer {layer_id} rotation is non-finite")
            check = rotation.float()
            orthogonality_error = (check @ check.T - eye).abs().max().item()
            if orthogonality_error > _ORTHOGONALITY_TOLERANCE:
                raise ValueError(
                    f"Layer {layer_id} rotation is not orthogonal "
                    f"(max error={orthogonality_error:.3e})"
                )

    @staticmethod
    def _write_manifest(path: Path, manifest: dict) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix=".oscar_manifest_", suffix=".json.tmp", dir=path.parent
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(manifest, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

    def release(self) -> None:
        self._q_grams.clear()
        self._k_values.clear()
        self._v_values.clear()
        self.state = "released"
