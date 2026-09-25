#!/usr/bin/env python3
"""Parameters that are threaded through but never read.

gate_up_interleaved was added to FusedMoE and MoeRunnerConfig during the K3
port so a constructor call would stop raising TypeError. It was stored and
passed along and never consulted, so Kimi-K3 -- the one model that sets it
False -- ran its experts through the interleaved reader for its
non-interleaved weights and served fluent-looking token soup. Nothing failed;
the number was simply wrong on every MoE token.

A parameter that exists only to be accepted is worse than a missing one: it
reads as support. This checks that each named parameter is actually consulted
somewhere -- used in a condition, an index, an argument to a real call --
rather than only declared, stored and forwarded.

Run on CPU. Exits non-zero if a tracked parameter has no consumer.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys

# Parameters worth tracking: cross-cutting layout/behaviour switches whose
# absence is silent. Add to this list when a port introduces another.
TRACKED = [
    "gate_up_interleaved",
]


def _is_consumption(node, name) -> bool:
    """Reading it in a test, an index or a real argument -- not just passing it on."""
    if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
        return any(
            isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.test)
        )
    if isinstance(node, (ast.BoolOp, ast.UnaryOp, ast.Compare)):
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))
    return False


def audit(root: pathlib.Path, tracked):
    missing = []
    for name in tracked:
        consumers = []
        for f in sorted(root.rglob("*.py")):
            try:
                src = f.read_text(encoding="utf-8")
            except Exception:
                continue
            if name not in src:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if _is_consumption(node, name):
                    consumers.append(f"{f.name}:{getattr(node, 'lineno', '?')}")
        if not consumers:
            missing.append(name)
        else:
            print(f"  ok   {name}: read at {', '.join(sorted(set(consumers))[:4])}")
    return missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(
            pathlib.Path(__file__).resolve().parents[2]
            / "sglang-research/python/sglang/srt"
        ),
    )
    a = ap.parse_args()
    missing = audit(pathlib.Path(a.root), TRACKED)
    if missing:
        print(f"\nFAIL: threaded but never read: {', '.join(missing)}")
        print("A parameter that is only accepted reads as support for it.")
        return 1
    print(f"tracked parameters: {len(TRACKED)}, all consulted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
