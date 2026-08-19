"""Client for vLLM's OpenAI-compatible /v1/completions endpoint.

Uses vLLM extensions verified against the local clone
(vllm/entrypoints/openai/completion/protocol.py):
- `prompt` accepts a list of token ids (identical tokenization on both servers).
- `return_tokens_as_token_ids` makes logprob tokens come back as "token_id:N"
  strings, so output ids are recovered without re-tokenizing.
- `prompt_logprobs: k` + `echo: true` + `max_tokens: 0` returns per-prompt-token
  top-k logprobs (the prefill-scoring equivalent of sglang's input_top_logprobs).
  Response entries are per-position dicts {token_id: {logprob, rank, ...}},
  first entry None; the actual token is always included, even beyond top-k.
"""

from __future__ import annotations

import asyncio

import httpx

from .engine import GenerateResult, PositionScore


def _parse_token_id(token: str) -> int:
    """Tokens arrive as 'token_id:123' with return_tokens_as_token_ids."""
    if not token.startswith("token_id:"):
        raise RuntimeError(f"expected 'token_id:N' token from vllm, got {token!r}")
    return int(token[len("token_id:") :])


def _parse_prompt_logprob_entry(
    entry: dict, expected_token_id: int, top_k: int, pos: int
) -> PositionScore:
    """One prompt_logprobs entry -> PositionScore.

    Entry maps token_id (JSON string key) -> {logprob, rank, decoded_token}.
    The actual token is always present; it may be an extra beyond-top-k entry,
    so the top-k list is rebuilt by rank order.
    """
    items = [(int(tid), v) for tid, v in entry.items()]
    actual = next((v for tid, v in items if tid == expected_token_id), None)
    if actual is None:
        raise RuntimeError(
            f"logprob alignment error at generated position {pos}: token "
            f"{expected_token_id} missing from vllm prompt_logprobs entry"
        )
    ranked = sorted(items, key=lambda kv: kv[1]["rank"])[:top_k]
    if len(ranked) < top_k:
        raise RuntimeError(
            f"vllm returned {len(ranked)} top logprobs at generated position "
            f"{pos}, requested {top_k} — check the server's --max-logprobs"
        )
    return PositionScore(
        token_id=expected_token_id,
        logprob=float(actual["logprob"]),
        top_ids=[tid for tid, _ in ranked],
        top_logprobs=[float(v["logprob"]) for _, v in ranked],
    )


class VLLMClient:
    def __init__(self, base_url: str, concurrency: int = 32, timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._model_name: str | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def model_info(self) -> dict:
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json()["data"]
        if not data:
            raise RuntimeError(f"no models served at {self.base_url}")
        return {"model_path": data[0]["id"]}

    async def _model(self) -> str:
        if self._model_name is None:
            self._model_name = (await self.model_info())["model_path"]
        return self._model_name

    async def _completions(self, payload: dict, retries: int = 3) -> dict:
        async with self._sem:
            for attempt in range(retries + 1):
                try:
                    resp = await self._client.post("/v1/completions", json=payload)
                    if 400 <= resp.status_code < 500:
                        # client errors are not retryable; surface the body
                        raise RuntimeError(
                            f"/v1/completions returned {resp.status_code}: {resp.text[:500]}"
                        )
                    resp.raise_for_status()
                    return resp.json()
                except (httpx.TransportError, httpx.HTTPStatusError):
                    if attempt == retries:
                        raise
                    await asyncio.sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")

    async def generate(
        self,
        input_ids: list[int],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        sampling_seed: int | None = None,
    ) -> GenerateResult:
        payload = {
            "model": await self._model(),
            "prompt": input_ids,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": 0,  # chosen-token logprobs only
            "return_tokens_as_token_ids": True,
        }
        if sampling_seed is not None:
            payload["seed"] = sampling_seed
        ret = await self._completions(payload)
        choice = ret["choices"][0]
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens") or []
        return GenerateResult(
            output_ids=[_parse_token_id(t) for t in tokens],
            decode_logprobs=[float(x) for x in (lp.get("token_logprobs") or [])],
            text=choice.get("text", ""),
            finish_reason=choice.get("finish_reason") or "unknown",
        )

    async def score(self, input_ids: list[int], prompt_len: int, top_k: int) -> list[PositionScore]:
        """Prefill-score a full (prompt + generation) sequence; one PositionScore
        per generated token, same contract as SGLangClient.score."""
        gen_len = len(input_ids) - prompt_len
        if gen_len <= 0:
            return []
        ret = await self._completions(
            {
                "model": await self._model(),
                "prompt": input_ids,
                "max_tokens": 0,
                "echo": True,
                "prompt_logprobs": top_k,
                "temperature": 0.0,
            }
        )
        entries = ret["choices"][0].get("prompt_logprobs")
        if entries is None:
            raise RuntimeError("vllm response has no prompt_logprobs field")
        if len(entries) != len(input_ids):
            raise RuntimeError(
                f"vllm returned {len(entries)} prompt_logprobs entries for "
                f"{len(input_ids)} input tokens"
            )
        return [
            _parse_prompt_logprob_entry(
                entry, expected_token_id=input_ids[prompt_len + pos], top_k=top_k, pos=pos
            )
            for pos, entry in enumerate(entries[-gen_len:])
        ]
