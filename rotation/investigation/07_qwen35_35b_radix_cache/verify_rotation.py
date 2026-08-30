#!/usr/bin/env python3
"""Is this rotation a genuine calibrated orthogonal matrix, or is it Hadamard?

A missing or unreadable rotation file does not make the server fail; the pool
falls back to a Hadamard transform and serves at Hadamard quality, which for
INT2 K costs roughly a factor of two in relative error (project threshold:
2-bit needs K rel-err <= 0.1; Hadamard sits near 0.2). The failure is silent and
looks like "OSCAR just is not very good on this model", so the file has to be
checked, not assumed.

Two things are asked of each matrix:

* **Orthogonality**: ``max|M M^T - I|``. Anything above ~1e-3 is not a rotation
  and the dequantized K will not come back to where it started.
* **Not Hadamard**: a (normalized) Hadamard matrix has every entry at the same
  magnitude ``1/sqrt(n)``, so the spread of ``|entry|`` is ~0. A calibrated
  rotation concentrates energy and has a wide spread. Reported as
  ``std(|M|)/mean(|M|)``; the Qwen3.5-4B sibling measured 0.28.

Usage: verify_rotation.py ROT_DIR
"""
import sys
from pathlib import Path

import torch


def describe(name: str, M: torch.Tensor) -> bool:
    M = M.to(torch.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        print(f"  {name}: shape {tuple(M.shape)} is not a square matrix -- SKIP")
        return False
    n = M.shape[0]
    I = torch.eye(n, dtype=M.dtype)
    orth = float((M @ M.T - I).abs().max())
    absM = M.abs()
    spread = float(absM.std() / absM.mean())
    hadamard_mag = 1.0 / (n ** 0.5)
    frac_at_had = float(((absM - hadamard_mag).abs() < 1e-6).to(torch.float64).mean())
    ok_orth = orth < 1e-3
    ok_cal = spread > 0.05
    print(f"  {name}: n={n} max|MM^T-I|={orth:.3e} "
          f"spread(|M|)={spread:.4f} frac_entries_at_1/sqrt(n)={frac_at_had:.4f} "
          f"-> {'orthogonal' if ok_orth else 'NOT ORTHOGONAL'}, "
          f"{'calibrated' if ok_cal else 'HADAMARD-LIKE'}")
    return ok_orth and ok_cal


def walk(prefix, obj, state, depth=0):
    """Find every 2-D tensor in a nested checkpoint and check it.

    These files are ``{format_version, objective, source_grouping, layers:
    {layer_id: tensor}}``, so a checker that only looks one level deep sees a
    dict of metadata, checks nothing, and reports success. It must recurse, and
    it must count how many matrices it actually looked at -- a verdict drawn
    from zero matrices is not a verdict.
    """
    if depth > 4:
        return
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 2 and obj.shape[0] == obj.shape[1]:
            state["n"] += 1
            state["ok"] &= describe(prefix, obj)
        elif obj.ndim == 3:
            # Per-head stack: check each head separately. Never head-slice a
            # bare 3-D rotation for use, but for *checking* each [n,n] slice
            # must be orthogonal on its own.
            for i in range(obj.shape[0]):
                state["n"] += 1
                state["ok"] &= describe(f"{prefix}[head{i}]", obj[i])
        else:
            print(f"  {prefix}: shape {tuple(obj.shape)} -- not a matrix")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (torch.Tensor, dict, list)):
                walk(f"{prefix}[{k}]", v, state, depth + 1)
            else:
                print(f"  {prefix}[{k}] = {v!r}")
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(f"{prefix}[{i}]", v, state, depth + 1)


def main():
    d = Path(sys.argv[1])
    files = sorted(d.glob("*.pt"))
    if not files:
        print(f"NO ROTATION FILES in {d}")
        sys.exit(1)
    state = {"ok": True, "n": 0}
    for f in files:
        print(f"{f.name}  ({f.stat().st_size} bytes)")
        obj = torch.load(f, map_location="cpu", weights_only=False)
        walk("", obj, state)
    print(f"matrices checked: {state['n']}")
    if state["n"] == 0:
        print("VERDICT: NOTHING CHECKED -- the file holds no square matrix "
              "where this script looked; do not trust any number from it")
        sys.exit(3)
    print("VERDICT:", "genuine calibrated orthogonal rotations"
          if state["ok"] else "SUSPECT -- do not trust these numbers")
    sys.exit(0 if state["ok"] else 2)


if __name__ == "__main__":
    main()
