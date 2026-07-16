"""Minimal AIME-2025 eval for the simple_evals framework.

AIME answers are integers 0-999, so grading is exact integer match — no LLM
equality checker needed. Dataset: ``math-ai/aime25`` (30 problems, cached under
/shared/huggingface). Mirrors the simple_evals Eval interface so
``run_simple_eval.py`` can treat it like GPQA/HumanEval.
"""
import re

from simple_evals import common
from simple_evals.types import Eval, EvalResult, SamplerBase, SingleEvalResult

QUERY_TEMPLATE = """
Solve the following AIME problem step by step. The final answer is an integer between 0 and 999.
The last line of your response must be of the form Answer: $ANSWER (without quotes) where $ANSWER is that integer.

{problem}

Remember to put your final integer answer on its own line after "Answer:".
""".strip()


def _last_int(s):
    ms = re.findall(r"-?\d+", s or "")
    return ms[-1] if ms else None


def extract_aime_answer(text: str):
    """Prefer \\boxed{}, then an 'Answer:' line, then the last integer."""
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        v = _last_int(boxed[-1])
        if v is not None:
            return v
    m = re.search(r"(?is)Answer\s*:\s*(.+?)(?:\n|$)", text)
    if m:
        v = _last_int(m.group(1))
        if v is not None:
            return v
    return _last_int(text)


class AIMEEval(Eval):
    def __init__(self, num_examples: int | None = None, n_repeats: int = 1,
                 dataset: str = "math-ai/aime25"):
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        from datasets import load_dataset
        ds = load_dataset(dataset)
        split = "test" if "test" in ds else list(ds.keys())[0]
        rows = [{"problem": r["problem"], "answer": str(r["answer"]).strip()}
                for r in ds[split]]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rows = rows[:num_examples]
        self.examples = rows * n_repeats

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [sampler._pack_message(
                content=QUERY_TEMPLATE.format(problem=row["problem"]), role="user")]
            resp = sampler(prompt_messages)
            extracted = extract_aime_answer(resp.response_text)
            correct = False
            if extracted is not None:
                try:
                    correct = int(extracted) == int(row["answer"])
                except ValueError:
                    correct = False
            score = 1.0 if correct else 0.0
            return SingleEvalResult(
                html="", score=score,
                convo=resp.actual_queried_message_list + [
                    {"role": "assistant", "content": resp.response_text}],
                metrics={"chars": len(resp.response_text or "")},
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
