"""Three-way equivalence test for the prod-KL (kl-probe truncated_kl) ports.

Compares, on float64 randomized cases plus edge cases:
1. the genuine kl_probe.metrics.truncated_kl (AST-extracted so pyarrow is
   not imported),
2. the torch port `prod_truncated_kl` in w3a8_lut_sim.py,
3. the pure-python port in wikitext2_kl_tail_prodkl.py,
all evaluated in the exact-candidate-logprob regime used by our harnesses
(candidate list covers every reference id, so kl-probe's floor clamp is
inactive). Also checks the clamp path stays non-negative and that identical
distributions give exactly zero.
"""

import ast
import importlib.util
import math
import pathlib
import random
import sys

import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_truncated_kl(metrics_path):
    tree = ast.parse(pathlib.Path(metrics_path).read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "truncated_kl",
            "_require_finite_logprobs",
        }:
            yield node


def main():
    base = pathlib.Path(sys.argv[1])
    namespace = {"math": math, "__name__": "kl_probe_extract"}
    module_ast = ast.Module(
        body=list(extract_truncated_kl(base / "metrics.py")),
        type_ignores=[],
    )
    exec(compile(module_ast, "metrics_extract", "exec"), namespace)
    kl_probe_fn = namespace["truncated_kl"]

    w3 = load_module(base / "w3a8_lut_sim.py", "w3")
    client = load_module(base / "wikitext2_kl_tail_prodkl.py", "client")

    rng = random.Random(20260819)
    max_kl_diff = 0.0
    max_tail_diff = 0.0
    cases = 0
    for trial in range(1000):
        k = rng.choice([5, 20, 50])
        vocab = rng.choice([200, 1000])
        teacher_logits = [rng.gauss(0.0, 3.0) for _ in range(vocab)]
        student_logits = [
            value + rng.gauss(0.0, rng.choice([0.0, 0.3, 1.5]))
            for value in teacher_logits
        ]

        def log_softmax(logits):
            top = max(logits)
            total = math.log(sum(math.exp(v - top) for v in logits)) + top
            return [v - total for v in logits]

        teacher_logp_full = log_softmax(teacher_logits)
        student_logp_full = log_softmax(student_logits)
        ids = sorted(
            range(vocab), key=lambda i: teacher_logp_full[i], reverse=True
        )[:k]
        teacher_logp = [teacher_logp_full[i] for i in ids]
        student_logp = [student_logp_full[i] for i in ids]

        ref_kl, ref_tail = kl_probe_fn(ids, teacher_logp, ids, student_logp)
        torch_kl, torch_tail = w3.prod_truncated_kl(
            torch.tensor(teacher_logp, dtype=torch.float64),
            torch.tensor(student_logp, dtype=torch.float64),
        )
        client_kl, client_tail = client.prod_truncated_kl(
            teacher_logp, student_logp
        )
        max_kl_diff = max(
            max_kl_diff,
            abs(ref_kl - torch_kl),
            abs(ref_kl - client_kl),
        )
        max_tail_diff = max(
            max_tail_diff,
            abs(ref_tail - torch_tail),
            abs(ref_tail - client_tail),
        )
        cases += 1

    assert max_kl_diff < 1e-9, max_kl_diff
    assert max_tail_diff < 1e-9, max_tail_diff

    # identical distributions -> exactly zero KL in all three ports
    logp = [math.log(p) for p in (0.5, 0.3, 0.1, 0.05, 0.02)]
    for value in (
        kl_probe_fn([1, 2, 3, 4, 5], logp, [1, 2, 3, 4, 5], logp)[0],
        w3.prod_truncated_kl(
            torch.tensor(logp, dtype=torch.float64),
            torch.tensor(logp, dtype=torch.float64),
        )[0],
        client.prod_truncated_kl(logp, logp)[0],
    ):
        assert value == 0.0, value

    # kl-probe clamp path (inactive in our harnesses) stays non-negative
    clamped, _ = kl_probe_fn(
        [1, 2, 3], [math.log(0.5), math.log(0.3), math.log(0.1)],
        [1, 2], [math.log(0.6), math.log(0.2)],
    )
    assert clamped >= 0.0

    print(
        f"PRODKL_EQUIVALENCE_OK cases={cases} "
        f"max_kl_diff={max_kl_diff:.3e} max_tail_diff={max_tail_diff:.3e}"
    )


if __name__ == "__main__":
    main()
