"""Engine-agnostic client interface and shared types.

Both engines are driven the same way: generate a completion from token ids,
and prefill-score a full (prompt + generation) sequence for per-position
top-k logprobs. The concrete clients (sglang_client, vllm_client) implement
this interface; make_client() picks one from the spec's detected engine.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import ModelSpec, Provider


@dataclass
class PositionScore:
    token_id: int
    logprob: float  # logprob of the actual token at this position
    top_ids: list[int]
    top_logprobs: list[float]


@dataclass
class GenerateResult:
    output_ids: list[int]
    decode_logprobs: list[float]  # decode-time logprob of each sampled token
    text: str
    finish_reason: str


class EngineClient(Protocol):
    async def generate(
        self,
        input_ids: list[int],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        sampling_seed: int | None = None,
    ) -> GenerateResult: ...

    async def score(
        self, input_ids: list[int], prompt_len: int, top_k: int
    ) -> list[PositionScore]: ...

    async def model_info(self) -> dict:
        """Normalized: at least {"model_path": <served checkpoint>}."""
        ...

    async def close(self) -> None: ...


def make_client(spec: ModelSpec, base_url: str, concurrency: int = 32) -> EngineClient:
    if spec.engine is Provider.VLLM:
        from .vllm_client import VLLMClient

        return VLLMClient(base_url, concurrency=concurrency)
    # SGLANG and TGL both speak the sglang native API
    from .sglang_client import SGLangClient

    return SGLangClient(base_url, concurrency=concurrency)


async def wait_healthy(base_url: str, timeout_s: float = 1800.0, interval_s: float = 10.0) -> None:
    """Poll /health until the server is up (model loads can take many minutes).
    Both sglang and vllm expose GET /health -> 200. A 404 is fatal: whatever
    answered is serving routes but has no /health, i.e. some other service is
    squatting on the configured port."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=10.0) as client:
        while True:
            try:
                resp = await client.get("/health")
                if resp.status_code == 200:
                    return
                if resp.status_code == 404:
                    raise RuntimeError(
                        f"server at {base_url} answered 404 to /health — another "
                        "service appears to be listening on this port"
                    )
            except httpx.TransportError:
                pass
            if loop.time() > deadline:
                raise TimeoutError(f"server at {base_url} not healthy after {timeout_s:.0f}s")
            await asyncio.sleep(interval_s)
