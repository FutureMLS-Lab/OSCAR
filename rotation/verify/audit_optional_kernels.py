#!/usr/bin/env python3
"""Every import of an upstream-only module must be unreachable or guarded.

Upstream ships a kernel package this fork does not vendor. Model code imports
from it at ~30 sites, each with a slow path beside it, but written as bare
imports because upstream knows the package is there. In this fork an
unguarded one raises ModuleNotFoundError from inside the first real forward
pass -- after the weights load, tens of minutes into a multi-GPU run -- and
reads like a model defect rather than a missing optional dependency. Two such
crashes cost two full load cycles before this check existed.

A site passes if it is:
  * inside try/except ImportError,
  * preceded by kernels_module_available() in the same function,
  * a function body whose only callers sit behind a dispatch gate that itself
    performs the availability check (the gate is named and verified here, so
    deleting it fails this check rather than silently re-arming the crash),
  * reachable only with an env flag upstream defaults to False, or
  * dead code with no caller in this fork.

Run on CPU. Exits non-zero on an unguarded site.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

GUARD_CALL = "kernels_module_available"

# Modules upstream has and this fork does not. sglang.kernels is the whole
# kernel package; mega_moe is a single upstream MoE module reached only
# through the megamoe a2a backend. Both fail the same way -- an import inside
# a forward pass -- so both belong to the same audit.
OPTIONAL_PREFIXES = (
    "sglang.kernels",
    "sglang.srt.layers.moe.mega_moe",
)

# body -> the dispatch gate that decides whether it runs. The gate's own
# definition must contain an availability check (or a helper that is itself
# guarded), which this script verifies rather than assumes.
DISPATCH_GATED = {
    ("srt/layers/attn_residual.py", "_aggregate_fast"): "_use_fast",
    ("srt/layers/attn_residual.py", "_aggregate_hip"): "_use_hip_fused",
    ("srt/models/kimi_k3.py", "_forward_mega_experts"): "_use_mega_moe",
    ("srt/models/kimi_k3.py", "_ep_front"): "_ep_front_eligible",
    ("srt/models/kimi_k3.py", "_ep_front_overlap"): "_ep_front_eligible",
    ("srt/models/kimi_k3.py", "forward_qkvbfg_fused"): "do_fuse_qkvbfg",
}
# Helpers that are themselves guarded, so a gate may delegate to one of them.
GUARDED_HELPERS = {GUARD_CALL, "_k3_fused_gemm_available", "_routing_contract_ok"}

FLAG_GATED_FILES = {
    "srt/layers/k3_ar_fusion.py": "SGLANG_K3_AR_FUSION",
    "srt/layers/k3_sp_collective.py": "SGLANG_K3_SP_COLLECTIVE",
    "srt/layers/k3_gemm_ar.py": "SGLANG_K3_GEMM_AR",
}
DEAD = {"srt/layers/attention/vision_rope.py"}


def _parents(tree):
    p = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            p[c] = n
    return p


def _try_guarded(node, ancestors):
    for a in ancestors:
        if isinstance(a, ast.Try) and any(
            node is x or node in ast.walk(x) for x in a.body
        ):
            for h in a.handlers:
                if isinstance(h.type, ast.Name):
                    names = {h.type.id}
                elif isinstance(h.type, ast.Tuple):
                    names = {e.id for e in h.type.elts if isinstance(e, ast.Name)}
                else:
                    names = {"BARE"}
                if names & {
                    "ImportError",
                    "ModuleNotFoundError",
                    "Exception",
                    "BaseException",
                    "BARE",
                }:
                    return True
    return False


def _gate_is_guarded(src: str, tree: ast.AST, gate: str) -> bool:
    """The gate must decide using an availability check or a guarded helper."""
    for node in ast.walk(tree):
        seg = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == gate:
            seg = ast.get_source_segment(src, node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgt = ast.get_source_segment(src, node) or ""
            if f"{gate} =" in tgt or f"{gate}=" in tgt:
                seg = tgt
        if seg and any(h in seg for h in GUARDED_HELPERS):
            return True
    return False


def audit(root: pathlib.Path):
    ok, bare = [], []
    for f in sorted(root.rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        if not any(p in src for p in OPTIONAL_PREFIXES):
            continue
        rel = str(f.relative_to(root))
        tree = ast.parse(src)
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            if not (mod or "").startswith(OPTIONAL_PREFIXES):
                continue
            anc, cur = [], node
            while cur in parents:
                cur = parents[cur]
                anc.append(cur)
            fn = next(
                (a for a in anc if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef))),
                None,
            )
            name = fn.name if fn else "<module>"
            where = f"{rel}:{node.lineno} in {name}()"

            if _try_guarded(node, anc):
                ok.append((where, "try/except"))
                continue
            if fn is not None and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == GUARD_CALL
                for c in ast.walk(fn)
            ):
                ok.append((where, "availability gate"))
                continue
            gate = DISPATCH_GATED.get((rel, name))
            if gate is not None:
                if _gate_is_guarded(src, tree, gate):
                    ok.append((where, f"dispatch gate {gate}()"))
                else:
                    bare.append(
                        f"{where} -- its gate {gate} no longer performs an "
                        f"availability check"
                    )
                continue
            if rel in FLAG_GATED_FILES:
                ok.append((where, f"needs {FLAG_GATED_FILES[rel]}=1"))
                continue
            if rel in DEAD:
                ok.append((where, "no caller in this fork"))
                continue
            bare.append(f"{where} -- unguarded on a live path")
    return ok, bare


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(pathlib.Path(__file__).resolve().parents[2] / "sglang-research/python/sglang"),
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    ok, bare = audit(pathlib.Path(a.root))
    if a.verbose:
        for w, why in ok:
            print(f"  ok   {w}  [{why}]")
    if bare:
        print(f"FAIL: {len(bare)} unguarded optional-module import site(s):")
        for b in bare:
            print(f"  {b}")
        print(
            "\nEach needs an availability gate; the fallback beside it already "
            "produces the same result."
        )
        return 1
    print(f"optional-kernel imports: {len(ok)} site(s), all guarded or unreachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
