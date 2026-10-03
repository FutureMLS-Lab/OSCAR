"""Loading and validation of product-quantization codebook files for the
unified INT2 pool's PQ / RVQ quantizers.

Accepted file schemas (all tensors fp32 or fp16):

* per-layer: ``{"codebooks_per_layer": [L, n_sub, n_centroids, sub_dim],
  "layer_ids": [L] (optional, default 0..L-1), "n_sub", "sub_dim",
  "n_centroids"}``
* residual (RVQ): ``{"codebooks_stage1": [L, n_sub, c1, sub_dim],
  "codebooks_stage2": [L, n_sub, c2, sub_dim], "layer_ids", ...}``
* shared (legacy): ``{"codebooks": list of n_sub [n_centroids, sub_dim]}``,
  one codebook for every layer."""

from __future__ import annotations

from typing import Optional, Sequence

import msgspec
import torch

from sglang.QuantKernel.oscar_pq_kv import pq_code_width, pq_codebook_norms


class PQCodebookHeader(msgspec.Struct, frozen=True, kw_only=True):
    kind: str  # "per_layer" | "residual" | "shared"
    n_sub: int
    n_centroids: int
    sub_dim: int
    layer_ids: Optional[list[int]] = None
    stage2_centroids: int = 0

    @property
    def stage1_code_width(self) -> int:
        # Stage-1 codes are always one byte per sub-vector: the attention and
        # prefix kernels read them as bytes (lookup tables index by byte).
        return self.n_sub

    @property
    def stage2_code_width(self) -> int:
        """Bytes per row of the residual codes; a 16-centroid stage packs two
        codes per byte."""
        if self.kind != "residual":
            return 0
        return pq_code_width(self.n_sub, self.stage2_centroids)

    @property
    def code_bytes(self) -> int:
        return self.stage1_code_width + self.stage2_code_width


class PQCodebookSet(msgspec.Struct, frozen=True, kw_only=True):
    """Device-resident codebooks for the layers one pool holds, in the pool's
    local layer order."""

    header: PQCodebookHeader
    codebooks: list  # per local layer: fp16 [n_sub, n_centroids, sub_dim]
    norms: list  # per local layer: fp32 [n_sub, n_centroids]
    stage2_codebooks: Optional[list] = None
    stage2_norms: Optional[list] = None
    source: str = ""

    @property
    def residual(self) -> bool:
        return self.stage2_codebooks is not None

    @property
    def n_sub(self) -> int:
        return self.header.n_sub

    @property
    def stage1_code_width(self) -> int:
        return self.header.stage1_code_width

    @property
    def stage2_code_width(self) -> int:
        return self.header.stage2_code_width

    @property
    def code_bytes(self) -> int:
        return self.header.code_bytes

    def describe(self, head_dim: int) -> str:
        bits = 8.0 * self.code_bytes / head_dim
        text = f"n_sub={self.n_sub} centroids={self.header.n_centroids} {bits:.2f} bits/value"
        if self.residual:
            text += f" (residual stage, {self.header.stage2_centroids} centroids)"
        return text


def _as_int_layer_ids(raw, n_layers: int, label: str) -> list[int]:
    if len(raw) != n_layers:
        raise ValueError(f"{label} has {n_layers} layer codebooks but {len(raw)} layer_ids")
    layer_ids = []
    for value in raw:
        try:
            layer_id = int(value)
            exact = float(value) == layer_id
        except (TypeError, ValueError):
            exact = False
        if not exact:
            raise ValueError(f"{label} layer_id {value!r} is not an integer")
        layer_ids.append(layer_id)
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"{label} layer_ids must be unique: {layer_ids}")
    return layer_ids


def _check_centroid_count(n_centroids: int, label: str) -> None:
    if n_centroids > 256:
        raise ValueError(f"{label} has {n_centroids} centroids; uint8 codes support at most 256")
    if n_centroids & (n_centroids - 1):
        raise ValueError(f"{label} centroid count must be a power of two, got {n_centroids}")


def read_pq_codebook_header(data: dict, *, expected_head_dim: int, label: str) -> PQCodebookHeader:
    """Validate a loaded codebook file against the row geometry."""
    stage2_centroids = 0
    layer_ids = None
    if "codebooks_stage1" in data:
        kind = "residual"
        stage1 = data["codebooks_stage1"]
        stage2 = data.get("codebooks_stage2")
        if stage1.ndim != 4 or stage2 is None or stage2.ndim != 4:
            raise ValueError(f"{label} RVQ codebooks must be rank-4 stage1/stage2 tensors")
        if (
            stage1.shape[0] != stage2.shape[0]
            or stage1.shape[1] != stage2.shape[1]
            or stage1.shape[3] != stage2.shape[3]
        ):
            raise ValueError(
                f"{label} RVQ stage shapes are incompatible: "
                f"{tuple(stage1.shape)} vs {tuple(stage2.shape)}"
            )
        n_sub, n_centroids, sub_dim = (int(x) for x in stage1.shape[1:])
        stage2_centroids = int(stage2.shape[2])
        _check_centroid_count(stage2_centroids, f"{label} RVQ stage2")
        layer_ids = _as_int_layer_ids(
            data.get("layer_ids", list(range(int(stage1.shape[0])))), int(stage1.shape[0]), label
        )
    elif "codebooks_per_layer" in data:
        kind = "per_layer"
        books = data["codebooks_per_layer"]
        if books.ndim != 4:
            raise ValueError(
                f"{label} codebooks_per_layer must be rank 4, got shape={tuple(books.shape)}"
            )
        n_sub, n_centroids, sub_dim = (int(x) for x in books.shape[1:])
        layer_ids = _as_int_layer_ids(
            data.get("layer_ids", list(range(int(books.shape[0])))), int(books.shape[0]), label
        )
    elif "codebooks" in data:
        kind = "shared"
        books = data["codebooks"]
        if not books:
            raise ValueError(f"{label} shared codebook list is empty")
        stacked = torch.stack(list(books))
        if stacked.ndim != 3:
            raise ValueError(
                f"{label} shared codebooks must stack to rank 3, got shape={tuple(stacked.shape)}"
            )
        n_sub, n_centroids, sub_dim = (int(x) for x in stacked.shape)
    else:
        raise ValueError(f"{label} file contains no supported codebook tensor")

    declared = {"n_sub": n_sub, "sub_dim": sub_dim}
    if "n_centroids" in data:
        declared["n_centroids"] = n_centroids
    for key, actual in declared.items():
        if key in data and int(data[key]) != actual:
            raise ValueError(
                f"{label} metadata {key}={data[key]} does not match tensor shape ({actual})"
            )
    _check_centroid_count(n_centroids, label)
    if n_sub * sub_dim != expected_head_dim:
        raise ValueError(
            f"{label} codebook reconstructs {n_sub}*{sub_dim}={n_sub * sub_dim} dims, "
            f"expected {expected_head_dim}"
        )
    return PQCodebookHeader(
        kind=kind,
        n_sub=n_sub,
        n_centroids=n_centroids,
        sub_dim=sub_dim,
        layer_ids=layer_ids,
        stage2_centroids=stage2_centroids,
    )


def _load_file(path: str) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"PQ codebook file {path} must hold a dict, got {type(data).__name__}")
    return data


def pq_code_bytes_per_row(path: str, *, head_dim: int, label: str) -> int:
    """Bytes one K or V row costs under the codebook at ``path``; used to
    price the pool before it exists."""
    return read_pq_codebook_header(_load_file(path), expected_head_dim=head_dim, label=label).code_bytes


def _select_layers(
    stacked: torch.Tensor, header_layer_ids: list[int], layer_ids: Sequence[int], label: str
) -> list[torch.Tensor]:
    index = {lid: i for i, lid in enumerate(header_layer_ids)}
    missing = [lid for lid in layer_ids if lid not in index]
    if missing:
        raise ValueError(f"{label} has no codebook for layers {missing}")
    return [stacked[index[lid]] for lid in layer_ids]


def load_pq_codebook_set(
    path: str, *, head_dim: int, layer_ids: Sequence[int], device, label: str
) -> PQCodebookSet:
    """Load the codebooks for ``layer_ids`` (global ids, pool order) onto ``device``."""
    data = _load_file(path)
    header = read_pq_codebook_header(data, expected_head_dim=head_dim, label=label)
    dev = torch.device(device)

    def _place(book: torch.Tensor) -> torch.Tensor:
        return book.to(torch.float16).to(dev).contiguous()

    stage2_codebooks = None
    stage2_norms = None
    if header.kind == "shared":
        shared = _place(torch.stack(list(data["codebooks"])))
        codebooks = [shared] * len(layer_ids)
        norms = [pq_codebook_norms(shared)] * len(layer_ids)
    else:
        key = "codebooks_stage1" if header.kind == "residual" else "codebooks_per_layer"
        codebooks = [
            _place(b) for b in _select_layers(data[key], header.layer_ids, layer_ids, label)
        ]
        norms = [pq_codebook_norms(b) for b in codebooks]
        if header.kind == "residual":
            stage2_codebooks = [
                _place(b)
                for b in _select_layers(
                    data["codebooks_stage2"], header.layer_ids, layer_ids, f"{label} stage2"
                )
            ]
            stage2_norms = [pq_codebook_norms(b) for b in stage2_codebooks]
    return PQCodebookSet(
        header=header,
        codebooks=codebooks,
        norms=norms,
        stage2_codebooks=stage2_codebooks,
        stage2_norms=stage2_norms,
        source=path,
    )
