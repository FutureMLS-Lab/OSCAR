"""MATH-500 eval with deterministic grading (no LLM equality-checker).

simple_evals' MathEval needs an LLM equality-checker that must reply exactly
"yes"; a verbose thinking model (Qwen3.5) never does, so it scores 0. Instead
we extract the final answer from \\boxed{} (which these models emit reliably)
and grade with normalized-string + sympy equivalence — deterministic, no grader.
"""
import re

import pandas
from simple_evals import common
from simple_evals.types import Eval, EvalResult, SamplerBase, SingleEvalResult

QUERY_TEMPLATE = """
Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.

{Question}
""".strip()


def last_boxed(s: str):
    """Return the content of the last \\boxed{...}, brace-balanced."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def extract_answer(text: str):
    if not text:
        return None
    b = last_boxed(text)
    if b is not None:
        return b.strip()
    m = re.findall(r"(?is)Answer\s*:\s*(.+?)(?:\n|$)", text)
    if m:
        return m[-1].strip()
    return None


def _normalize(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\\(?:text|mbox|mathrm|textbf)\s*\{([^{}]*)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""),
                 ("\\;", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\cdot", "*"), ("\\times", "*"), ("\\%", ""), ("%", ""),
                 ("\\$", ""), ("$", ""), ("^{\\circ}", ""), ("^\\circ", ""), (" ", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac(\d)(\d)", r"((\1)/(\2))", s)
    s = s.replace("\\", "").replace("{", "").replace("}", "").rstrip(".")
    return s


def math_equal(pred, gold) -> bool:
    if pred is None:
        return False
    p, g = _normalize(pred), _normalize(gold)
    if p == g:
        return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application)
        tr = standard_transformations + (implicit_multiplication_application,)
        if sympy.simplify(parse_expr(p, transformations=tr)
                          - parse_expr(g, transformations=tr)) == 0:
            return True
    except Exception:
        pass
    try:
        if abs(float(p) - float(g)) < 1e-6:
            return True
    except Exception:
        pass
    return False


class MATH500Eval(Eval):
    def __init__(self, num_examples: int | None = None, n_repeats: int = 1):
        df = pandas.read_csv(
            "https://openaipublic.blob.core.windows.net/simple-evals/math_500_test.csv")
        rows = [r.to_dict() for _, r in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rows = rows[:num_examples]
        self.examples = rows * n_repeats

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            msgs = [sampler._pack_message(
                content=QUERY_TEMPLATE.format(Question=row["Question"]), role="user")]
            resp = sampler(msgs)
            extracted = extract_answer(resp.response_text)
            # The CSV "Answer" is a full worked solution ending in \boxed{...};
            # extract the boxed gold before comparing.
            gold = extract_answer(str(row["Answer"])) or str(row["Answer"])
            score = 1.0 if math_equal(extracted, gold) else 0.0
            return SingleEvalResult(
                html="", score=score,
                convo=resp.actual_queried_message_list + [
                    {"role": "assistant", "content": resp.response_text}],
                metrics={"chars": len(resp.response_text or "")},
            )

        return common.aggregate_results(common.map_with_progress(fn, self.examples))
