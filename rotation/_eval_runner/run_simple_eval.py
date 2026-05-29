#!/usr/bin/env python3
"""Thin wrapper that drives OpenAI's `simple-evals` (vendored at
`third_party/simple_evals/`) against an OpenAI-compatible sglang server.

Currently only ``--task gpqa`` is wired up. AIME and LiveCodeBench are not
part of simple-evals and need separate clients (deferred).

Usage:
  python run_simple_eval.py \
    --task gpqa \
    --model Qwen/Qwen3-8B \
    --base-url http://127.0.0.1:31060/v1 \
    --max-tokens 32768 \
    --temperature 1.0 --top-p 0.95 --top-k 40 \
    --n-repeats 1 \
    --output-dir <dir>
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SE_DIR = REPO / "third_party" / "simple_evals"
assert SE_DIR.is_dir(), (
    f"missing {SE_DIR}; clone https://github.com/openai/simple-evals.git "
    "into third_party/ and rename the directory to simple_evals (underscore)"
)
sys.path.insert(0, str(SE_DIR.parent))


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["gpqa"],
                   help="simple-evals task; only gpqa wired up for now")
    p.add_argument("--model", required=True, help="HF model id served by sglang")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--num-examples", type=int, default=None,
                   help="Restrict to N examples (default: all)")
    p.add_argument("--variant", default="diamond", help="GPQA variant: diamond | main")
    p.add_argument("--num-threads", type=int, default=32,
                   help="Client-side concurrency cap. simple-evals defaults to "
                        "os.cpu_count() which on big pods spikes the server "
                        "above its CUDA-graph batch capture limit and turns "
                        "CUDA graph off (eager → 2–3× slower).")
    p.add_argument("--system-message", default="You are a helpful assistant.")
    p.add_argument("--output-dir", required=True)
    return p


class SglangChatSampler:
    """Pass top_p and top_k to the sglang OpenAI-compat endpoint
    (simple_evals' own ChatCompletionSampler only sends temperature)."""

    image_format = "url"

    def __init__(self, model, base_url, api_key, system_message,
                 temperature, top_p, top_k, max_tokens):
        import httpx
        from openai import OpenAI
        # Long read timeout: thinking models generate 20-40k tokens; at 10 tok/s
        # that takes 2000-4000 s. Default 600 s fires before any response arrives.
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(connect=10.0, read=7200.0, write=30.0, pool=10.0),
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def __call__(self, message_list):
        from simple_evals.types import SamplerResponse
        import openai
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body={"top_k": self.top_k},
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": resp.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request:", e, flush=True)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                backoff = 2 ** trial
                print(f"  sampler retry {trial} in {backoff}s: {e}", flush=True)
                time.sleep(backoff)
                trial += 1
                if trial > 8:
                    raise


def main():
    args = _build_argparser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sampler = SglangChatSampler(
        model=args.model, base_url=args.base_url, api_key=args.api_key,
        system_message=args.system_message, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, max_tokens=args.max_tokens,
    )

    # Cap simple-evals' map_with_progress concurrency. GPQAEval.__call__
    # calls common.map_with_progress(fn, examples) without a num_threads arg,
    # which would otherwise default to os.cpu_count(). Monkey-patch the default
    # so the server stays at <= cuda-graph-max-bs concurrent requests.
    from simple_evals import common as _se_common
    _orig_map = _se_common.map_with_progress
    def _patched_map(f, xs, num_threads=None, pbar=True):
        if num_threads is None:
            num_threads = args.num_threads
        return _orig_map(f, xs, num_threads=num_threads, pbar=pbar)
    _se_common.map_with_progress = _patched_map

    # Monkey-patch ANSWER_PATTERN_MULTICHOICE back to the permissive `\s*`
    # (matches newlines), instead of openai's newer `[ \t]*` which fails on
    # "Answer:\n<letter>" outputs that thinking models do produce.
    import re
    _RELAXED = r"(?i)Answer\s*:\s*([A-D])"
    _se_common.ANSWER_PATTERN_MULTICHOICE = _RELAXED
    # gpqa_eval.py captured the symbol at import time, so patch the eval
    # module too if already imported (defensive — actually imported below).
    try:
        import simple_evals.gpqa_eval as _gpqa
        _gpqa.ANSWER_PATTERN_MULTICHOICE = _RELAXED
    except ImportError:
        pass

    # I/O dump: capture every (prompt, response) pair to io_log.jsonl so the
    # framework-vs-server contribution to noise can be checked offline.
    _io_log_path = out / "io_log.jsonl"
    _io_log_f = open(_io_log_path, "w")
    _orig_call = SglangChatSampler.__call__
    import threading
    _io_lock = threading.Lock()
    def _logging_call(self, message_list):
        resp = _orig_call(self, message_list)
        try:
            with _io_lock:
                import json as _json
                _io_log_f.write(_json.dumps({
                    "messages": message_list,
                    "response": resp.response_text,
                    "model": self.model,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "max_tokens": self.max_tokens,
                }) + "\n")
                _io_log_f.flush()
        except Exception:
            pass
        return resp
    SglangChatSampler.__call__ = _logging_call

    if args.task == "gpqa":
        from simple_evals.gpqa_eval import GPQAEval
        evaluator = GPQAEval(
            n_repeats=args.n_repeats, variant=args.variant,
            num_examples=args.num_examples,
        )
    else:
        raise ValueError(f"task {args.task} not wired up yet")

    print(f"=== running {args.task} eval ===", flush=True)
    print(f"  model={args.model}  base_url={args.base_url}")
    print(f"  n_repeats={args.n_repeats}  num_examples={args.num_examples}")
    print(f"  temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print(f"  max_tokens={args.max_tokens}", flush=True)
    t0 = time.time()
    result = evaluator(sampler)
    elapsed = time.time() - t0

    # simple_evals.EvalResult: top-line `score` lives on the dataclass attribute,
    # NOT inside `metrics`. Merge them so downstream sees a single dict.
    metrics = dict(result.metrics or {})
    if getattr(result, "score", None) is not None:
        metrics["score"] = float(result.score)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Pretty score table; downstream consumers grep
    # `^|\\s+<task>/score\\s+\\|` to pull the final number from eval.log.
    lines = [
        f"Evaluation results for {args.task} on {args.model}",
        "=" * 100,
        "+" + "-" * 20 + "+" + "-" * 24 + "+",
        "|       Metric         |         Value          |",
        "+" + "-" * 20 + "+" + "-" * 24 + "+",
    ]
    for k in sorted(metrics.keys()):
        v = metrics[k]
        try:
            v_str = f"{float(v):.6f}"
        except (TypeError, ValueError):
            v_str = str(v)
        lines.append(f"|   {args.task}/{k:<14s} | {v_str:>22s} |")
    lines.append("+" + "-" * 20 + "+" + "-" * 24 + "+")
    lines.append(f"(elapsed: {elapsed:.1f}s)")
    log_text = "\n".join(lines) + "\n"
    (out / "eval.log").write_text(log_text)
    print(log_text)


if __name__ == "__main__":
    main()
