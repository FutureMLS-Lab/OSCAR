#!/usr/bin/env python3
"""Compute OSCAR K/V rotation matrices from dumped Q/K/V tensors.

Expected dump layout:

  <dump_path>/layer_<id>/q/<chunk_id>.pt
  <dump_path>/layer_<id>/k/<chunk_id>.pt
  <dump_path>/layer_<id>/v/<chunk_id>.pt

The output checkpoint schema matches ``load_oscar_rotations`` in
``sglang.srt.mem_cache.memory_pool``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def build_hadamard(n: int) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    if n == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    h = build_hadamard(n // 2)
    return torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / math.sqrt(2)


def bit_reversal_perm(d: int) -> torch.Tensor:
    if d < 1 or d & (d - 1):
        raise ValueError(f"Bit-reversal size must be a power of two, got {d}")
    bits = int(math.log2(d))
    return torch.tensor([int(bin(i)[2:].zfill(bits)[::-1], 2) for i in range(d)])


def make_br_perm_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    d = len(eigenvalues)
    sorted_idx = torch.argsort(eigenvalues, descending=True)
    br = bit_reversal_perm(d)
    perm = torch.zeros(d, dtype=torch.long)
    for i in range(d):
        perm[br[i]] = sorted_idx[i]
    return torch.eye(d, dtype=torch.float64)[:, perm]


def load_tensor(layer_dir: Path, name: str, chunk_id) -> torch.Tensor:
    """Load a single chunk (chunk_id is int) or concat all chunks (chunk_id == "all").

    "all" mode skips chunk 0 (a 6-token warmup batch produced by the prefill
    schedule that would dominate hessian estimation with degenerate samples).
    """
    sub_dir = layer_dir / name
    if isinstance(chunk_id, str) and chunk_id == "all":
        chunk_paths = sorted(
            sub_dir.glob("*.pt"),
            key=lambda p: int(p.stem),
        )
        chunk_paths = [p for p in chunk_paths if int(p.stem) != 0]
        if not chunk_paths:
            raise FileNotFoundError(f"No chunk files in {sub_dir}")
        tensors = [
            torch.load(str(p), map_location="cpu").float().double()
            for p in chunk_paths
        ]
        return torch.cat(tensors, dim=0)
    path = sub_dir / f"{chunk_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing dumped tensor: {path}")
    return torch.load(str(path), map_location="cpu").float().double()


def eigdecomp_from_flat(x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_flat.double()
    cov = x.T @ x / x.shape[0]
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvecs, eigvals


def compute_ktk(layer_dir: Path, chunk_id: int, head_dim: int):
    k = load_tensor(layer_dir, "k", chunk_id)
    return eigdecomp_from_flat(k.reshape(-1, head_dim))


def compute_vtv(layer_dir: Path, chunk_id: int, head_dim: int):
    v = load_tensor(layer_dir, "v", chunk_id)
    return eigdecomp_from_flat(v.reshape(-1, head_dim))


def compute_qqt(layer_dir: Path, chunk_id: int, head_dim: int):
    q = load_tensor(layer_dir, "q", chunk_id)
    k = load_tensor(layer_dir, "k", chunk_id)
    n_heads = q.shape[1] if q.ndim >= 3 else q.shape[0]
    kv_heads = k.shape[1] if k.ndim >= 3 else k.shape[0]
    gqa_ratio = n_heads // kv_heads
    q_flat = q.reshape(-1, n_heads, head_dim)

    cov = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    for h in range(kv_heads):
        qg = q_flat[:, h * gqa_ratio : (h + 1) * gqa_ratio, :].reshape(-1, head_dim)
        cov += qg.T @ qg / qg.shape[0]
    cov /= kv_heads
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvecs, eigvals


def compute_sst(layer_dir: Path, chunk_id: int, head_dim: int):
    q = load_tensor(layer_dir, "q", chunk_id)
    k = load_tensor(layer_dir, "k", chunk_id)
    v = load_tensor(layer_dir, "v", chunk_id)
    n_heads = q.shape[1] if q.ndim >= 3 else q.shape[0]
    kv_heads = k.shape[1] if k.ndim >= 3 else k.shape[0]
    gqa_ratio = n_heads // kv_heads
    q_flat = q.reshape(-1, n_heads, head_dim)
    k_flat = k.reshape(-1, kv_heads, head_dim)
    v_flat = v.reshape(-1, kv_heads, head_dim)
    n_tokens = q_flat.shape[0]

    cov = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    for h in range(kv_heads):
        qg = q_flat[:, h * gqa_ratio : (h + 1) * gqa_ratio, :].reshape(-1, head_dim)
        kh = k_flat[:, h, :]
        vh = v_flat[:, h, :]
        qtq = qg.T @ qg / qg.shape[0]
        weights = (kh @ qtq * kh).sum(1)
        weights = weights / weights.sum().clamp(min=1e-12) * n_tokens
        vw = vh * weights.unsqueeze(1).sqrt()
        cov += vw.T @ vw / n_tokens
    cov /= kv_heads
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(cov)
    return eigvecs, eigvals


def simulate_int2_asym(x: torch.Tensor) -> torch.Tensor:
    max_quant_val = 3
    x_fp32 = x.float()
    val_min = x_fp32.amin(dim=-1, keepdim=True)
    val_max = x_fp32.amax(dim=-1, keepdim=True)
    scale = (val_max - val_min).clamp(min=1e-8) / max_quant_val
    zero = -val_min / scale
    q = (x_fp32 / scale + zero + 0.5).to(torch.int32).clamp(0, max_quant_val)
    return ((q.float() - zero) * scale).to(x.dtype)


def build_pi_permutation(
    hessian_eigvals: torch.Tensor, residual_eigvals: torch.Tensor
) -> torch.Tensor:
    d = len(hessian_eigvals)
    h_order = torch.argsort(hessian_eigvals, descending=True)
    e_order = torch.argsort(residual_eigvals, descending=False)
    pi = torch.zeros(d, d, dtype=torch.float64)
    for r in range(d):
        pi[e_order[r], h_order[r]] = 1.0
    return pi


def compute_uresidual_layer(
    layer_dir: Path,
    chunk_id: int,
    head_dim: int,
    ref_k_rot: torch.Tensor,
    ref_v_rot: torch.Tensor,
):
    q = load_tensor(layer_dir, "q", chunk_id)
    k = load_tensor(layer_dir, "k", chunk_id)
    v = load_tensor(layer_dir, "v", chunk_id)
    n_heads = q.shape[1] if q.ndim >= 3 else q.shape[0]
    kv_heads = k.shape[1] if k.ndim >= 3 else k.shape[0]
    gqa_ratio = n_heads // kv_heads
    q_flat = q.reshape(-1, n_heads, head_dim)
    k_flat = k.reshape(-1, kv_heads, head_dim)
    v_flat = v.reshape(-1, kv_heads, head_dim)
    n_tokens = q_flat.shape[0]

    c_q = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    c_v = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    for h in range(kv_heads):
        qg = q_flat[:, h * gqa_ratio : (h + 1) * gqa_ratio, :].reshape(-1, head_dim)
        kh = k_flat[:, h, :]
        vh = v_flat[:, h, :]
        qtq = qg.T @ qg / qg.shape[0]
        c_q += qtq
        weights = (kh @ qtq * kh).sum(1)
        weights = weights / weights.sum().clamp(min=1e-12) * n_tokens
        c_v += (vh * weights.unsqueeze(1).sqrt()).T @ (
            vh * weights.unsqueeze(1).sqrt()
        ) / n_tokens
    c_q = (c_q / kv_heads + (c_q / kv_heads).T) / 2
    c_v = (c_v / kv_heads + (c_v / kv_heads).T) / 2
    eigvals_q, u_q = torch.linalg.eigh(c_q)
    eigvals_h, u_h = torch.linalg.eigh(c_v)

    e_k = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    e_v = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    for h in range(kv_heads):
        k_rot = k_flat[:, h, :] @ ref_k_rot.double()
        v_rot = v_flat[:, h, :] @ ref_v_rot.double()
        dk = (simulate_int2_asym(k_rot) - k_rot).double()
        dv = (simulate_int2_asym(v_rot) - v_rot).double()
        e_k += dk.T @ dk / dk.shape[0]
        e_v += dv.T @ dv / dv.shape[0]
    e_k = (e_k / kv_heads + (e_k / kv_heads).T) / 2
    e_v = (e_v / kv_heads + (e_v / kv_heads).T) / 2
    eigvals_e_k, u_e_k = torch.linalg.eigh(e_k)
    eigvals_e_v, u_e_v = torch.linalg.eigh(e_v)

    r_k = u_e_k @ build_pi_permutation(eigvals_q, eigvals_e_k) @ u_q.T
    r_v = u_e_v @ build_pi_permutation(eigvals_h, eigvals_e_v) @ u_h.T
    return r_k, eigvals_q, r_v, eigvals_h


HESSIAN_FNS = {
    "ktk": compute_ktk,
    "vtv": compute_vtv,
    "qqt": compute_qqt,
    "sst": compute_sst,
}

METHOD_TARGETS = {
    "ktk": [("k", "ktk")],
    "vtv": [("v", "vtv")],
    "qqt": [("k", "qqt")],
    "sst": [("v", "sst")],
    "ktk_vtv": [("k", "ktk"), ("v", "vtv")],
    "qqt_sst": [("k", "qqt"), ("v", "sst")],
}


def compose_rotation(
    rotation: torch.Tensor,
    eigvals: torch.Tensor,
    hadamard: torch.Tensor,
    composition: str,
) -> torch.Tensor:
    """Combine the data-derived eigvec rotation R (a.k.a. Uk), the fixed
    Hadamard H, and the eigenvalue-sorted bit-reversal permutation P (Pbr).

    Naming convention: list factors left-to-right in matrix-multiplication
    order. e.g. ``h_r_pbr`` => H @ R @ P.
    """
    pbr = make_br_perm_matrix(eigvals)
    if composition == "plain":
        return rotation                       # R          (Uk only)
    if composition == "pbr":
        return pbr                            # P          (Pbr only)
    if composition == "br":
        return rotation @ pbr                 # R · P
    if composition == "br_h128":
        return rotation @ pbr @ hadamard      # R · P · H
    if composition == "r_h":
        return rotation @ hadamard            # R · H
    if composition == "h_pbr":
        return hadamard @ pbr                 # H · P
    if composition == "h_r_pbr":
        return hadamard @ rotation @ pbr      # H · R · P
    if composition == "h_pbr_r":
        return hadamard @ pbr @ rotation      # H · P · R
    if composition == "r_h_pbr":
        return rotation @ hadamard @ pbr      # R · H · P (default, validated)
    raise ValueError(f"Unknown composition: {composition}")


def layer_dirs(dump_path: Path) -> list[Path]:
    dirs = [p for p in dump_path.iterdir() if p.is_dir() and p.name.startswith("layer_")]
    return sorted(dirs, key=lambda p: int(p.name.split("_", 1)[1]))


def empty_result(objective: str) -> dict:
    return {
        "format_version": 1,
        "objective": objective,
        "source_grouping": "layer",
        "layers": {},
    }


def add_layer(result: dict, layer_id: int, rotation: torch.Tensor, eigvals: torch.Tensor):
    # Save as fp32 to match what the runtime kernels consume.
    result["layers"][layer_id] = {
        "layer_id": layer_id,
        "rotation": rotation.float().contiguous(),
        "eigenvalues": eigvals.float().contiguous(),
    }


def get_rotation_layer(state: dict, layer_id: int) -> torch.Tensor:
    layers = state["layers"]
    entry = layers.get(layer_id, layers.get(str(layer_id)))
    if entry is None:
        raise KeyError(f"Missing layer {layer_id} in reference rotation checkpoint")
    return entry["rotation"]


def write_hadamard_rotation(
    output_dir: Path,
    head_dim: int,
    num_layers: int,
    layer_ids: list[int] | None = None,
) -> None:
    """Pure fixed Hadamard rotation (data-free; no dump required).

    When ``layer_ids`` is provided, the rotation file is keyed by those global
    layer ids (in the order given) instead of ``range(num_layers)``. This is
    required for hybrid models where the full-attention layers are sparse
    (e.g. Qwen3.5: layers 3, 7, 11, ...). ``len(layer_ids)`` must equal
    ``num_layers``.
    """
    h = build_hadamard(head_dim).float()
    err = (h @ h.T - torch.eye(head_dim)).abs().max().item()
    print(f"Hadamard orthogonality error: {err:.2e}")
    eigvals = torch.ones(head_dim, dtype=torch.float32)
    if layer_ids is not None:
        if len(layer_ids) != num_layers:
            raise ValueError(
                f"write_hadamard_rotation: --layer-ids has {len(layer_ids)} "
                f"entries but num_layers={num_layers}"
            )
        keys = list(layer_ids)
    else:
        keys = list(range(num_layers))
    for target in ("k", "v"):
        result = empty_result("hadamard")
        for layer_id in keys:
            add_layer(result, layer_id, h, eigvals)
        path = output_dir / f"{target}_rotation_hadamard.pt"
        torch.save(result, str(path))
        print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=None,
        help="Calibration dump dir. Required for all methods except 'hadamard'.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--chunk-id",
        default="all",
        help='Dump chunk id to use, or "all" to concat every chunk (skipping the '
        "6-token warmup chunk 0). Default: all.",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "ktk", "vtv", "qqt", "sst", "ktk_vtv", "qqt_sst",
            "uresidual", "hadamard",
        ],
        help="'hadamard' = fixed Hadamard matrix per layer (no calibration).",
    )
    parser.add_argument(
        "--composition",
        default="plain",
        choices=[
            "plain",     # R
            "pbr",       # P
            "br",        # R · P
            "br_h128",   # R · P · H
            "r_h",       # R · H
            "h_pbr",     # H · P
            "h_r_pbr",   # H · R · P
            "h_pbr_r",   # H · P · R
            "r_h_pbr",   # R · H · P (validated best)
        ],
        help="Ignored when --method hadamard.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of layers to emit. Required when --method hadamard and no "
        "--dump-path is given; otherwise inferred from the dump.",
    )
    parser.add_argument(
        "--layer-ids",
        type=str,
        default=None,
        help="Comma-separated list of global layer ids to use as keys in the "
        "rotation file (e.g. '3,7,11,15,19,23,27,31' for Qwen3.5-4B). When "
        "omitted, hadamard mode emits keys range(num_layers). Required for "
        "hybrid models where full-attention layers are sparse so the in-server "
        "OSCAR pool can look up rotations by global id.",
    )
    parser.add_argument("--ref-k-rotation", type=Path, default=None)
    parser.add_argument("--ref-v-rotation", type=Path, default=None)
    args = parser.parse_args()

    layer_ids = None
    if args.layer_ids is not None:
        layer_ids = [int(x) for x in args.layer_ids.split(",") if x.strip()]
        if not layer_ids:
            raise ValueError("--layer-ids was provided but parsed to an empty list")

    if args.method == "hadamard":
        output_dir = args.output_dir or (args.dump_path if args.dump_path else None)
        if output_dir is None:
            raise ValueError("--output-dir is required when --method hadamard")
        output_dir.mkdir(parents=True, exist_ok=True)
        if layer_ids is not None:
            num_layers = len(layer_ids)
            if args.num_layers is not None and args.num_layers != num_layers:
                raise ValueError(
                    f"--num-layers={args.num_layers} disagrees with "
                    f"--layer-ids (len={num_layers})"
                )
        elif args.num_layers is not None:
            num_layers = args.num_layers
        elif args.dump_path is not None:
            num_layers = len(layer_dirs(args.dump_path))
        else:
            raise ValueError(
                "hadamard method needs --layer-ids, --num-layers, or --dump-path "
                "to know how many layers to emit"
            )
        print(
            f"Method=hadamard head_dim={args.head_dim} num_layers={num_layers} "
            f"layer_ids={layer_ids}"
        )
        write_hadamard_rotation(output_dir, args.head_dim, num_layers, layer_ids)
        return

    if args.dump_path is None:
        raise ValueError(f"--dump-path is required for --method {args.method}")
    output_dir = args.output_dir or args.dump_path
    output_dir.mkdir(parents=True, exist_ok=True)
    hadamard = build_hadamard(args.head_dim)
    dirs = layer_dirs(args.dump_path)
    print(f"Found {len(dirs)} layers in {args.dump_path}")
    print(f"Method={args.method} composition={args.composition} chunk={args.chunk_id}")

    if args.method == "uresidual":
        if not args.ref_k_rotation or not args.ref_v_rotation:
            raise ValueError("uresidual requires --ref-k-rotation and --ref-v-rotation")
        ref_k = torch.load(str(args.ref_k_rotation), map_location="cpu")
        ref_v = torch.load(str(args.ref_v_rotation), map_location="cpu")
        k_result = empty_result("uresidual")
        v_result = empty_result("uresidual")
        for layer_dir in dirs:
            layer_id = int(layer_dir.name.split("_", 1)[1])
            r_k, ev_k, r_v, ev_v = compute_uresidual_layer(
                layer_dir,
                args.chunk_id,
                args.head_dim,
                get_rotation_layer(ref_k, layer_id),
                get_rotation_layer(ref_v, layer_id),
            )
            k_err = (r_k @ r_k.T - torch.eye(args.head_dim, dtype=torch.float64)).abs().max().item()
            v_err = (r_v @ r_v.T - torch.eye(args.head_dim, dtype=torch.float64)).abs().max().item()
            print(f"  Layer {layer_id:>2}: K={k_err:.1e}, V={v_err:.1e}")
            add_layer(k_result, layer_id, r_k, ev_k)
            add_layer(v_result, layer_id, r_v, ev_v)
        torch.save(k_result, str(output_dir / "k_rotation_uresidual.pt"))
        torch.save(v_result, str(output_dir / "v_rotation_uresidual.pt"))
        return

    results = {
        (target, hessian): empty_result(f"{hessian}_{args.composition}")
        for target, hessian in METHOD_TARGETS[args.method]
    }
    for layer_dir in dirs:
        layer_id = int(layer_dir.name.split("_", 1)[1])
        errors = []
        for target, hessian in METHOD_TARGETS[args.method]:
            rotation, eigvals = HESSIAN_FNS[hessian](layer_dir, args.chunk_id, args.head_dim)
            loaded_rotation = compose_rotation(
                rotation, eigvals, hadamard, args.composition
            )
            err = (
                loaded_rotation @ loaded_rotation.T
                - torch.eye(args.head_dim, dtype=torch.float64)
            ).abs().max().item()
            errors.append(f"{target.upper()}({hessian})={err:.1e}")
            add_layer(results[(target, hessian)], layer_id, loaded_rotation, eigvals)
        print(f"  Layer {layer_id:>2}: {', '.join(errors)}")

    for (target, hessian), result in results.items():
        path = output_dir / f"{target}_rotation_{hessian}_{args.composition}.pt"
        torch.save(result, str(path))
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
