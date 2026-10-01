#!/usr/bin/env python3
"""Names read but never bound, in the code paths a model actually runs.

Removing Kimi-K3's expanded-MHA layout left `if use_expanded_cache:` behind in
forward_normal_prepare. That branch is the MHA prefill path -- the triton
backend selects it for an extend with no prefix hit -- so every absorbed-path
run passed and the NameError only surfaced after a 25-minute 96-shard load on
sixteen GPUs, reading as a model failure.

Python binds nothing at import time inside a function body, so a dangling name
in a rarely-taken branch is invisible to `import sglang` and to every smoke
test that does not happen to take that branch.

The rule is deliberately narrow: report a name only when the WHOLE file never
binds it -- not at module level, not in any function, not in any class, not as
a parameter, import, loop target or comprehension variable. A full scope
analysis was tried first and produced 359 findings on this tree, nearly all of
them closures reading an enclosing parameter. A checker that has to be sifted
is one nobody runs, so this trades recall for a clean signal: it cannot see a
name that is bound on some other branch, but everything it does print is real.

No GPU, no imports of the code under test -- pure AST, so it runs anywhere.
Exits non-zero if any file in scope reads an unbound name.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import pathlib
import sys

_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__"}

# Paths worth scanning: the model and attention code a serving run executes.
# Widening this to the whole fork buries a real finding under vendored code.
DEFAULT_SCOPE = [
    "sglang-research/python/sglang/srt/models",
    "sglang-research/python/sglang/srt/layers/attention",
    "sglang-research/python/sglang/srt/mem_cache",
]



def _bound_by(node, sink):
    """Every name this statement/expression binds in the current scope."""
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            sink.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                sink.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sink.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            sink.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            sink.update(n.names)


def _params(fn):
    a = fn.args
    out = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
    if a.vararg:
        out.add(a.vararg.arg)
    if a.kwarg:
        out.add(a.kwarg.arg)
    return out


def _file_bindings(tree):
    """Every name bound anywhere in the file, at any nesting depth."""
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound |= _params(node)
            bound.add(node.name)
        elif isinstance(node, ast.Lambda):
            bound |= _params(node)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        # match/case captures: `case C(kw=x)`, `case [*rest]`, `case {**rest}`.
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    return bound


def audit(root: pathlib.Path, scope):
    findings = []
    files = []
    for rel in scope:
        base = root / rel
        files.extend(sorted(base.rglob("*.py")) if base.is_dir() else [base])

    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError as exc:
            findings.append((path.relative_to(root), exc.lineno or 0, "<parse>", str(exc)))
            continue

        # A star-import can supply anything; do not guess in those files.
        if any(
            isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
            for n in tree.body
        ):
            continue

        bound = _file_bindings(tree)
        # Attribute bases and annotations count as reads like any other.
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
                continue
            if node.id in bound or node.id in _BUILTINS:
                continue
            findings.append((path.relative_to(root), node.lineno, "", node.id))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=pathlib.Path)
    ap.add_argument("--path", action="append", default=None)
    args = ap.parse_args()

    findings = audit(args.root.resolve(), args.path or DEFAULT_SCOPE)
    if not findings:
        print("audit_undefined_names: no unbound reads in scope")
        return 0
    for path, line, _fn, name in findings:
        print(f"UNBOUND {path}:{line}: {name}")
    print(f"\n{len(findings)} unbound read(s). Each is a NameError waiting on a branch.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
