"""Async client for SGLang's native /generate API.

Uses the native endpoint (not the OpenAI-compatible one) because it accepts
`input_ids` directly (identical tokenization on both servers is a hard
requirement for token-level comparison) and exposes prefill logprobs via
`logprob_start_len` / `top_logprobs_num` without the OpenAI top_logprobs cap.

SGLang returns logprob entries as (logprob, token_id[, text]) tuples.
"""

from __future__ import annotations

import asyncio

import httpx

from .engine import GenerateResult, PositionScore


class SGLangClient:
    def __init__(self, base_url: str, concurrency: int = 32, timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def model_info(self) -> dict:
        resp = await self._client.get("/get_model_info")
        resp.raise_for_status()
        info = resp.json()
        return {"model_path": info.get("model_path", ""), **info}

    async def _generate(self, payload: dict, retries: int = 3) -> dict:
        async with self._sem:
            for attempt in range(retries + 1):
                try:
                    resp = await self._client.post("/generate", json=payload)
                    if 400 <= resp.status_code < 500:
                        # client errors are not retryable; surface the body
                        raise RuntimeError(
                            f"/generate returned {resp.status_code}: {resp.text[:500]}"
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
        """Sample a completion; return_logprob gives us the output token ids and
        the decode-time logprob of each sampled token (kept as a diagnostic
        against the prefill-scored logprobs)."""
        sampling_params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }
        if sampling_seed is not None:
            sampling_params["sampling_seed"] = sampling_seed
        ret = await self._generate(
            {
                "input_ids": input_ids,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
        )
        meta = ret["meta_info"]
        out = meta.get("output_token_logprobs") or []
        return GenerateResult(
            output_ids=[int(e[1]) for e in out],
            decode_logprobs=[float(e[0]) for e in out],
            text=ret.get("text", ""),
            finish_reason=(meta.get("finish_reason") or {}).get("type", "unknown"),
        )

    async def score(self, input_ids: list[int], prompt_len: int, top_k: int) -> list[PositionScore]:
        """Prefill-score a full (prompt + generation) sequence.

        Returns one PositionScore per generated token (positions prompt_len..end),
        i.e. the model's next-token distribution given the identical preceding
        context, plus the exact logprob of the token that was actually there.
        """
        gen_len = len(input_ids) - prompt_len
        if gen_len <= 0:
            return []
        ret = await self._generate(
            {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
                "return_logprob": True,
                # Ask for a little earlier than needed and slice from the tail:
                # robust to sglang's off-by-one conventions at the start of the
                # logprob window (the first entry can be None-padded).
                "logprob_start_len": max(prompt_len - 1, 0),
                "top_logprobs_num": top_k,
            }
        )
        meta = ret["meta_info"]
        token_lps = meta["input_token_logprobs"][-gen_len:]
        top_lps = meta["input_top_logprobs"][-gen_len:]
        if len(token_lps) != gen_len or len(top_lps) != gen_len:
            raise RuntimeError(
                f"server returned {len(token_lps)} token / {len(top_lps)} top logprob "
                f"entries, expected {gen_len}"
            )
        scores: list[PositionScore] = []
        for pos, (tok_entry, top_entry) in enumerate(zip(token_lps, top_lps)):
            token_id = int(tok_entry[1])
            expected = input_ids[prompt_len + pos]
            if token_id != expected:
                raise RuntimeError(
                    f"logprob alignment error at generated position {pos}: "
                    f"server token {token_id} != sequence token {expected}"
                )
            top_entry = top_entry or []
            if len(top_entry) < top_k:
                raise RuntimeError(
                    f"server returned {len(top_entry)} top logprobs at generated "
                    f"position {pos}, requested {top_k} — refusing to silently "
                    "compare at a smaller k"
                )
            scores.append(
                PositionScore(
                    token_id=token_id,
                    logprob=float(tok_entry[0]),
                    top_ids=[int(e[1]) for e in top_entry],
                    top_logprobs=[float(e[0]) for e in top_entry],
                )
            )
        return scores
