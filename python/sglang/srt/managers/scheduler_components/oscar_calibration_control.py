"""Scheduler-side control of startup OSCAR calibration.

Both actions run the same shape on every TP rank: each step executes
locally, its outcome is MIN-reduced over the TP CPU group, and the first
step that fails anywhere stops all ranks with the same answer, so no rank
enters a collective the others skipped."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    OscarCalibrationReqInput,
    OscarCalibrationReqOutput,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.oscar_calibration import OscarOnlineCalibrator
from sglang.srt.mem_cache.unified_kv_pool import UnifiedInt2HPKVPool

logger = logging.getLogger(__name__)


class SchedulerOscarCalibrationControl:
    def __init__(
        self,
        *,
        get_kv_pool: Callable[[], Any],
        flush_cache: Callable[[], bool],
        is_fully_idle: Callable[[], bool],
        tp_cpu_group,
        tp_size: int,
        attn_tp_rank: int,
        device: str,
    ) -> None:
        self._get_kv_pool = get_kv_pool
        self._flush_cache = flush_cache
        self._is_fully_idle = is_fully_idle
        self._tp_cpu_group = tp_cpu_group
        self._tp_size = int(tp_size)
        self._attn_tp_rank = int(attn_tp_rank)
        self._device_module = torch.get_device_module(device)

    def handle(self, recv_req: OscarCalibrationReqInput) -> OscarCalibrationReqOutput:
        if recv_req.action not in ("start", "finalize"):
            return OscarCalibrationReqOutput(
                success=False,
                message=f"Unknown OSCAR calibration action: {recv_req.action}",
            )
        pool = self._get_kv_pool()
        if isinstance(pool, HybridLinearKVPool):
            # A hybrid linear-attention model wraps the calibrating unified pool;
            # the collector, the publish and the detach all live on the inner one.
            pool = pool.full_kv_pool
        calibrator = (
            pool.oscar_calibrator if isinstance(pool, UnifiedInt2HPKVPool) else None
        )
        locally_ready = calibrator is not None and self._is_fully_idle()
        if not self._agree(locally_ready):
            return OscarCalibrationReqOutput(
                success=False,
                message=(
                    "Every TP rank must have a pending OSCAR calibrator and an "
                    "idle scheduler before calibration control"
                ),
            )
        if recv_req.action == "start":
            return self._start(recv_req, calibrator)
        return self._finalize(pool, calibrator)

    # -- Consensus helpers --------------------------------------------------

    def _agree(self, local_ok: bool) -> bool:
        status = torch.tensor([int(local_ok)], dtype=torch.int32)
        if self._tp_size > 1:
            torch.distributed.all_reduce(
                status, op=torch.distributed.ReduceOp.MIN, group=self._tp_cpu_group
            )
        return bool(status.item())

    def _min_over_ranks(self, value: int) -> int:
        tensor = torch.tensor([value], dtype=torch.int64)
        if self._tp_size > 1:
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MIN, group=self._tp_cpu_group
            )
        return int(tensor.item())

    def _step(self, fn: Callable[[], Any], failure: str) -> tuple[Any, Optional[str]]:
        """Run ``fn`` locally and agree on success; returns ``(value, None)``
        or ``(None, message)`` where the message is this rank's error or
        ``failure`` when only another rank failed."""
        local_error = ""
        value = None
        try:
            value = fn()
            ok = True
        except Exception as exc:
            ok = False
            local_error = str(exc)
            logger.exception("OSCAR calibration step failed: %s", failure)
        if not self._agree(ok):
            return None, (local_error or failure)
        return value, None

    def _flush_or_fail(self) -> bool:
        if not self._flush_cache():
            raise RuntimeError("scheduler refused to flush the KV cache")
        return True

    # -- Actions ------------------------------------------------------------

    def _start(
        self, recv_req: OscarCalibrationReqInput, calibrator: OscarOnlineCalibrator
    ) -> OscarCalibrationReqOutput:
        if not recv_req.prompt_sha256:
            return OscarCalibrationReqOutput(
                success=False, message="OSCAR calibration start requires a prompt SHA-256"
            )
        if recv_req.token_budget <= 0:
            return OscarCalibrationReqOutput(
                success=False,
                message="OSCAR calibration start requires a positive token count",
            )
        _, error = self._step(
            self._flush_or_fail, "A TP rank failed to flush cache before OSCAR calibration"
        )
        if error is None:
            _, error = self._step(
                lambda: calibrator.start(
                    prompt_sha256=recv_req.prompt_sha256,
                    token_budget=recv_req.token_budget,
                ),
                "A TP rank failed to arm the OSCAR collector",
            )
        if error is not None:
            return OscarCalibrationReqOutput(success=False, message=error)
        return OscarCalibrationReqOutput(
            success=True, message="OSCAR calibration collector armed"
        )

    def _finalize(
        self, pool: UnifiedInt2HPKVPool, calibrator: OscarOnlineCalibrator
    ) -> OscarCalibrationReqOutput:
        captured = self._min_over_ranks(calibrator.min_captured_tokens())
        if not self._agree(calibrator.complete):
            return OscarCalibrationReqOutput(
                success=False,
                message=(
                    "OSCAR calibration did not reach the token budget on every "
                    f"rank/layer (minimum captured={captured})"
                ),
                captured_tokens=captured,
            )

        def _fail(message: str) -> OscarCalibrationReqOutput:
            return OscarCalibrationReqOutput(
                success=False, message=message, captured_tokens=captured
            )

        covariance_sums, error = self._step(
            calibrator.local_covariance_sums,
            "A TP rank failed to prepare OSCAR covariance sums",
        )
        if error is not None:
            return _fail(error)
        buffers, error = self._step(
            calibrator.allocate_result_buffers,
            "A TP rank failed to allocate OSCAR result buffers",
        )
        if error is not None:
            return _fail(error)
        result, error = self._step(
            lambda: calibrator.finalize(covariance_sums=covariance_sums, buffers=buffers),
            "A TP rank failed OSCAR eigendecomposition",
        )
        if error is not None:
            return _fail(error)
        _, error = self._step(
            lambda: calibrator.broadcast_result(result),
            "A TP rank failed to broadcast OSCAR rotations",
        )
        if error is not None:
            return _fail(error)
        _, error = self._step(
            lambda: calibrator.publish(result),
            "Another TP rank failed to publish OSCAR checkpoints",
        )
        if error is not None:
            return _fail(error)
        _, error = self._step(
            self._synchronize_and_flush,
            "A TP rank failed to flush the identity-basis KV cache",
        )
        if error is not None:
            return _fail(error)
        _, error = self._step(
            lambda: self._install(pool, result.k_rotations, result.v_rotations),
            "A TP rank failed to install OSCAR rotations",
        )
        if error is not None:
            return _fail(error)
        if self._tp_size > 1:
            torch.distributed.barrier(group=self._tp_cpu_group)
        pool.detach_oscar_calibrator()
        calibrator.release()
        self._device_module.empty_cache()
        envs.SGLANG_OSCAR_CALIBRATION_ACTIVE.set(False)
        return OscarCalibrationReqOutput(
            success=True,
            message=f"Installed OSCAR rotation generation {result.generation_id}",
            captured_tokens=captured,
        )

    def _synchronize_and_flush(self) -> bool:
        self._device_module.synchronize()
        return self._flush_or_fail()

    def _install(self, pool: UnifiedInt2HPKVPool, k_rotations, v_rotations) -> bool:
        pool.update_oscar_rotations_(k_rotations, v_rotations)
        self._device_module.synchronize()
        return True
