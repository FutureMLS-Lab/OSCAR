"""Online OSCAR rotation calibration for full-attention mixed INT2 KV pools."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.mem_cache.memory_pool import (
    get_oscar_checkpoint_pair,
    get_oscar_pair_artifact_paths,
)


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


@dataclass
class OscarCalibrationResult:
    k_rotations: torch.Tensor
    v_rotations: torch.Tensor
    k_state: dict
    v_state: dict
    generation_id: str


class OscarOnlineCalibrator:
    """Collect exact one-pass qqt/sst sufficient statistics on the GPU."""

    def __init__(
        self,
        *,
        pool,
        total_q_heads: int,
        total_kv_heads: int,
        model_path: str,
        model_revision: str | None = None,
    ):
        self.pool = pool
        self.device = torch.device(pool.device)
        self.max_token_budget = envs.SGLANG_OSCAR_CALIBRATION_TOKENS.get()
        self.token_budget = 0
        self.local_layers = list(
            range(pool.start_layer, pool.start_layer + pool.layer_num)
        )
        self.local_kv_heads = int(pool.head_num)
        self.total_q_heads = int(total_q_heads)
        self.total_kv_heads = int(total_kv_heads)
        self.head_dim = int(pool.head_dim)
        self.v_head_dim = int(pool.v_head_dim)
        self.model_path = model_path
        self.model_revision = model_revision
        self.state = "idle"
        self.prompt_sha256 = ""

        tp_size = get_attention_tp_size()
        if self.max_token_budget <= 0:
            raise ValueError("OSCAR calibration token budget must be positive")
        if self.head_dim != self.v_head_dim:
            raise ValueError(
                "Online OSCAR calibration currently requires equal K/V head dimensions"
            )
        if self.head_dim & (self.head_dim - 1):
            raise ValueError(
                f"Online OSCAR calibration requires power-of-two head_dim, got {self.head_dim}"
            )
        if tp_size > self.total_kv_heads:
            raise ValueError(
                "Online OSCAR calibration does not support replicated KV heads "
                f"(TP={tp_size}, global KV heads={self.total_kv_heads})"
            )
        if self.total_kv_heads % tp_size:
            raise ValueError(
                f"Global KV heads ({self.total_kv_heads}) must divide TP ({tp_size})"
            )
        if self.local_kv_heads * tp_size != self.total_kv_heads:
            raise ValueError(
                "Local/global KV-head geometry does not describe disjoint TP shards"
            )

        self._counts: dict[int, int] = {}
        self._gqa_ratios: dict[int, int] = {}
        self._q_grams: dict[int, torch.Tensor] = {}
        self._k_values: dict[int, torch.Tensor] = {}
        self._v_values: dict[int, torch.Tensor] = {}

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
                (
                    self.local_kv_heads,
                    self.head_dim,
                    self.head_dim,
                ),
                dtype=torch.float64,
                device=self.device,
            )
            for layer_id in self.local_layers
        }
        self._k_values = {}
        self._v_values = {}
        self.state = "collecting"

    @property
    def complete(self) -> bool:
        return self.state == "collecting" and all(
            count == self.token_budget for count in self._counts.values()
        )

    def observe(
        self,
        *,
        layer_id: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if self.state != "collecting" or layer_id not in self._counts:
            return
        saved = self._counts[layer_id]
        remaining = self.token_budget - saved
        if remaining <= 0:
            return
        if q is None or k is None or v is None:
            raise ValueError("OSCAR calibration requires original Q, K, and V tensors")

        num_tokens = min(int(q.shape[0]), int(k.shape[0]), int(v.shape[0]), remaining)
        if num_tokens <= 0:
            return
        q = q[:num_tokens].reshape(num_tokens, -1, self.head_dim)
        k = k[:num_tokens].reshape(num_tokens, self.local_kv_heads, self.head_dim)
        v = v[:num_tokens].reshape(num_tokens, self.local_kv_heads, self.v_head_dim)
        local_q_heads = q.shape[1]
        if local_q_heads * get_attention_tp_size() != self.total_q_heads:
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

        gram_chunk_tokens = 512
        for chunk_start in range(0, num_tokens, gram_chunk_tokens):
            chunk_stop = min(chunk_start + gram_chunk_tokens, num_tokens)
            chunk_tokens = chunk_stop - chunk_start
            q_grouped = (
                q[chunk_start:chunk_stop]
                .reshape(
                    chunk_tokens,
                    self.local_kv_heads,
                    gqa_ratio,
                    self.head_dim,
                )
                .permute(1, 0, 2, 3)
                .reshape(
                    self.local_kv_heads,
                    chunk_tokens * gqa_ratio,
                    self.head_dim,
                )
                .to(torch.float64)
            )
            self._q_grams[layer_id].add_(
                torch.bmm(q_grouped.transpose(1, 2), q_grouped)
            )

        if layer_id not in self._k_values:
            pin_memory = self.device.type == "cuda"
            self._k_values[layer_id] = torch.empty(
                (
                    self.token_budget,
                    self.local_kv_heads,
                    self.head_dim,
                ),
                dtype=k.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
            self._v_values[layer_id] = torch.empty(
                (
                    self.token_budget,
                    self.local_kv_heads,
                    self.v_head_dim,
                ),
                dtype=v.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
        self._k_values[layer_id][saved : saved + num_tokens].copy_(k, non_blocking=True)
        self._v_values[layer_id][saved : saved + num_tokens].copy_(v, non_blocking=True)
        self._counts[layer_id] = saved + num_tokens

    def _local_covariance_sums(self) -> torch.Tensor:
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

        k_sums = []
        v_sums = []
        chunk_size = 2048
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for layer_id in self.local_layers:
            gqa_ratio = self._gqa_ratios[layer_id]
            q_cov = self._q_grams[layer_id] / (self.token_budget * gqa_ratio)
            k_sums.append(q_cov.sum(dim=0))

            denominator = torch.zeros(
                self.local_kv_heads, dtype=torch.float64, device=self.device
            )
            numerator = torch.zeros(
                (
                    self.local_kv_heads,
                    self.v_head_dim,
                    self.v_head_dim,
                ),
                dtype=torch.float64,
                device=self.device,
            )
            for start in range(0, self.token_budget, chunk_size):
                stop = min(start + chunk_size, self.token_budget)
                k_chunk = self._k_values[layer_id][start:stop].to(
                    device=self.device,
                    dtype=torch.float64,
                    non_blocking=True,
                )
                v_chunk = self._v_values[layer_id][start:stop].to(
                    device=self.device,
                    dtype=torch.float64,
                    non_blocking=True,
                )
                energy = torch.einsum("thd,hde,the->th", k_chunk, q_cov, k_chunk)
                denominator.add_(energy.sum(dim=0))
                numerator.add_(
                    torch.einsum("th,thd,the->hde", energy, v_chunk, v_chunk)
                )
            per_head_v = numerator / denominator.clamp_min(1e-12)[:, None, None]
            v_sums.append(per_head_v.sum(dim=0))

        return torch.stack((torch.stack(k_sums), torch.stack(v_sums)))

    def allocate_result(self) -> OscarCalibrationResult:
        num_layers = len(self.local_layers)
        k_rotations = torch.empty(
            (num_layers, self.head_dim, self.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        return OscarCalibrationResult(
            k_rotations=k_rotations,
            v_rotations=torch.empty_like(k_rotations),
            k_state={},
            v_state={},
            generation_id="",
        )

    def finalize(
        self,
        covariance_sums: torch.Tensor | None = None,
        result: OscarCalibrationResult | None = None,
    ) -> OscarCalibrationResult:
        if self.state != "collecting":
            raise RuntimeError(
                f"Cannot finalize OSCAR calibrator from state={self.state}"
            )
        if covariance_sums is None:
            covariance_sums = self._local_covariance_sums()
        if result is None:
            result = self.allocate_result()
        tp_group = get_attention_tp_group()
        if tp_group.world_size > 1:
            torch.distributed.all_reduce(covariance_sums, group=tp_group.device_group)
        covariances = covariance_sums / self.total_kv_heads
        covariances = (covariances + covariances.transpose(-1, -2)) / 2

        num_layers = len(self.local_layers)
        k_values = torch.empty(
            (num_layers, self.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        v_values = torch.empty_like(k_values)
        if get_attention_tp_rank() == 0:
            flat_covariances = covariances.reshape(
                2 * num_layers, self.head_dim, self.head_dim
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(flat_covariances)
            hadamard = build_hadamard(
                self.head_dim, device=self.device, dtype=torch.float64
            )
            rotations = torch.stack(
                [
                    compose_r_h_pbr(eigenvectors[i], eigenvalues[i], hadamard)
                    for i in range(2 * num_layers)
                ]
            )
            result.k_rotations.copy_(rotations[:num_layers].float().contiguous())
            result.v_rotations.copy_(rotations[num_layers:].float().contiguous())
            k_values.copy_(eigenvalues[:num_layers].float().contiguous())
            v_values.copy_(eigenvalues[num_layers:].float().contiguous())
            result.generation_id = uuid.uuid4().hex
            result.k_state = self._checkpoint_state(
                "qqt_r_h_pbr",
                result.k_rotations,
                k_values,
                result.generation_id,
            )
            result.v_state = self._checkpoint_state(
                "sst_r_h_pbr",
                result.v_rotations,
                v_values,
                result.generation_id,
            )
        self.state = "computed"
        return result

    def broadcast_result(self, result: OscarCalibrationResult) -> None:
        tp_group = get_attention_tp_group()
        if tp_group.world_size > 1:
            source_rank = tp_group.ranks[0]
            for tensor in (result.k_rotations, result.v_rotations):
                torch.distributed.broadcast(
                    tensor, src=source_rank, group=tp_group.device_group
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
                "tp_size": get_attention_tp_size(),
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

    def publish(self, result: OscarCalibrationResult) -> None:
        if get_attention_tp_rank() != 0:
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
        fd, path = tempfile.mkstemp(
            prefix=f".oscar_{label}_", suffix=".pt.tmp", dir=directory
        )
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
        for layer_id in self.local_layers:
            entry = layers.get(layer_id, layers.get(str(layer_id)))
            rotation = entry["rotation"]
            if rotation.shape != (self.head_dim, self.head_dim):
                raise ValueError(
                    f"Layer {layer_id} rotation has invalid shape {rotation.shape}"
                )
            if not bool(torch.isfinite(rotation).all()):
                raise ValueError(f"Layer {layer_id} rotation is non-finite")
            check = rotation.float()
            eye = torch.eye(self.head_dim, dtype=check.dtype)
            orthogonality_error = (check @ check.T - eye).abs().max().item()
            if orthogonality_error > 5e-3:
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
        if getattr(self.pool, "_oscar_calibrator", None) is self:
            self.pool._oscar_calibrator = None


def prompt_ids_sha256(input_ids: list[list[int]]) -> str:
    payload = json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
