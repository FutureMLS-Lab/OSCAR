"""HumanEval with code extraction that survives a thinking model.

simple_evals' own ``find_code`` looks for a ```python fence and, finding none,
falls back to the *entire* response, then slices from the first ``":\\n    "`` it
sees. Qwen3-30B-A3B puts its reasoning in ``<think>...</think>`` and then emits
bare, unfenced code, so that first ``":\\n    "`` lands inside the prose and the
"completion" starts mid-sentence. Nearly every sample becomes a syntax error and
BF16 scores 0.146 instead of ~0.9 -- a harness artifact, not a model or
quantization effect.

Same container as ``aime_eval`` / ``math500_eval``, which exist for the same
reason: a verbose thinking model breaking an upstream grader.

HumanEval executes ``prompt + completion``, and the prompt already ends with the
signature and docstring, so the extracted completion must be the indented body
only.
"""

import re

from simple_evals import common
from simple_evals.common import HTML_JINJA
from simple_evals.humaneval_eval import HumanEval as _UpstreamHumanEval
from simple_evals.humaneval_eval import evaluate_functional_correctness
from simple_evals.types import EvalResult, SamplerBase, SingleEvalResult

from human_eval.evaluation import estimate_pass_at_k

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_DEF = re.compile(r"^[ \t]*(?:async\s+)?def\s+\w+\s*\(", re.MULTILINE)
_CODE_START = re.compile(r"^[ \t]*(?:from |import |@|def |class |async def )", re.MULTILINE)

INSTRUCTION = (
    "Read the following function signature and docstring, and fully implement "
    "the function described. Your response should only contain the code for "
    "this function.\n"
)


def find_code(completion: str) -> str:
    """Extract the runnable body of one model response."""
    text = completion or ""

    # 1. reasoning is never the answer
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        # unterminated think block (hit the token budget): nothing usable after
        text = text.split("<think>", 1)[0]

    # 2. a fenced block wins; take the LAST, since earlier ones are drafts
    blocks = _FENCE.findall(text)
    code = blocks[-1] if blocks else text

    # 3. drop leading prose
    m = _CODE_START.search(code)
    if m:
        code = code[m.start():]

    # 4. strip a restated signature so what remains is the body. Only when a
    #    def is actually present -- upstream's unconditional
    #    ``[find(":\n    ") + 2:]`` silently drops a character when find returns
    #    -1, which is how the no-fence case produced garbage.
    if _DEF.search(code):
        marker = code.find(":\n")
        if marker != -1:
            code = code[marker + 2:]
            if not code.startswith((" ", "\t")):
                code = "    " + code

    if not code.strip():
        return ""
    # the executor concatenates prompt + completion; keep it indented
    if not code.startswith((" ", "\t", "\n")):
        code = "    " + code
    return code


class HumanEval(_UpstreamHumanEval):
    """Upstream dataset and sandboxed execution, corrected extraction."""

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(sample):
            prompt_messages = [
                sampler._pack_message(
                    role="user", content=INSTRUCTION + sample["prompt"]
                )
            ]
            completions = [
                find_code(sampler(prompt_messages).response_text)
                for _ in range(self._num_samples_per_task)
            ]
            results = evaluate_functional_correctness(sample, completions)
            total = len(results)
            correct = sum(results)
            score = correct / total
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=completions[0], role="assistant"),
                score=score,
                correct_answer=[1] * len(results),
                extracted_answer=results,
            )
            convo = prompt_messages + [
                dict(content=c, role="assistant") for c in completions
            ]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    f"pass@{k}": estimate_pass_at_k([total], [correct], k)
                    for k in self._ks_passes
                    if total >= k
                },
            )

        results = common.map_with_progress(
            fn, self.examples, num_threads=getattr(self, "_num_threads", 3)
        )
        return common.aggregate_results(results)
