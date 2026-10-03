"""Destinations of the OSCAR K/V rotation pair and the startup-calibration
decision taken from them.

A launch calibrates when both destinations are known (configured, or
defaulted under ``HF_HOME``) and either file is missing or a previous
publication was interrupted. The decision is snapshotted into
``SGLANG_OSCAR_CALIBRATION_ACTIVE`` before the scheduler processes spawn, so
every process agrees without re-checking the filesystem."""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

K_ROTATION_FILENAME = "k_rotation_qqt_r_h_pbr.pt"
V_ROTATION_FILENAME = "v_rotation_sst_r_h_pbr.pt"
_DEFAULT_PROMPT_IDENTITY = "openai-simple-evals/gpqa_diamond-seed0-v1"


def _hf_home() -> str:
    return os.path.realpath(
        os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    )


def ensure_oscar_rotation_paths(
    *, model_path: str, revision: str | None
) -> tuple[str, str]:
    """Return the configured K/V destinations, defaulting both to a cache
    directory keyed by model, revision, prompt source and token budget."""
    configured_k = envs.SGLANG_OSCAR_K_ROTATION_PATH.get()
    configured_v = envs.SGLANG_OSCAR_V_ROTATION_PATH.get()
    if configured_k and configured_v:
        return configured_k, configured_v
    if configured_k or configured_v:
        raise ValueError(
            "OSCAR requires both SGLANG_OSCAR_K_ROTATION_PATH and "
            "SGLANG_OSCAR_V_ROTATION_PATH when either is configured"
        )

    model_identity = (
        os.path.realpath(model_path) if os.path.exists(model_path) else model_path
    )
    prompt_path = envs.SGLANG_OSCAR_CALIBRATION_PROMPTS_PATH.get()
    prompt_identity = (
        os.path.realpath(os.path.abspath(prompt_path))
        if prompt_path
        else _DEFAULT_PROMPT_IDENTITY
    )
    digest = hashlib.sha256(
        (
            f"{model_identity}\0{revision or 'main'}\0{prompt_identity}\0"
            f"{envs.SGLANG_OSCAR_CALIBRATION_TOKENS.get()}"
        ).encode()
    ).hexdigest()[:12]
    model_name = os.path.basename(model_path.rstrip("/")) or "model"
    safe_name = "".join(
        char if char.isalnum() or char in "._-" else "-" for char in model_name
    )[:64]
    output_dir = os.path.join(_hf_home(), "oscar-rotations", f"{safe_name}-{digest}")
    k_path = os.path.join(output_dir, K_ROTATION_FILENAME)
    v_path = os.path.join(output_dir, V_ROTATION_FILENAME)
    envs.SGLANG_OSCAR_K_ROTATION_PATH.set(k_path)
    envs.SGLANG_OSCAR_V_ROTATION_PATH.set(v_path)
    logger.info("Using default OSCAR rotation cache paths: K=%s V=%s", k_path, v_path)
    return k_path, v_path


def get_oscar_checkpoint_pair() -> tuple[str, str]:
    k_path = envs.SGLANG_OSCAR_K_ROTATION_PATH.get()
    v_path = envs.SGLANG_OSCAR_V_ROTATION_PATH.get()
    if not k_path or not v_path:
        raise ValueError(
            "OSCAR requires both SGLANG_OSCAR_K_ROTATION_PATH and "
            "SGLANG_OSCAR_V_ROTATION_PATH"
        )
    k_path = os.path.realpath(os.path.abspath(k_path))
    v_path = os.path.realpath(os.path.abspath(v_path))
    if k_path == v_path or (
        os.path.isfile(k_path)
        and os.path.isfile(v_path)
        and os.path.samefile(k_path, v_path)
    ):
        raise ValueError("OSCAR K/V checkpoint destinations must be distinct files")
    return k_path, v_path


def get_oscar_pair_artifact_paths() -> dict[str, str]:
    """Lock file and the pending/complete manifests that make a pair
    publication atomic across processes."""
    k_path, v_path = get_oscar_checkpoint_pair()
    parent = os.path.dirname(k_path)
    if parent != os.path.dirname(v_path):
        raise ValueError(
            "Startup OSCAR calibration requires K/V checkpoint destinations "
            "in the same directory"
        )
    pair_id = hashlib.sha256(f"K={k_path}\0V={v_path}".encode()).hexdigest()[:16]
    prefix = os.path.join(parent, f".oscar_{pair_id}")
    lock_dir = os.path.realpath(
        os.path.abspath(envs.SGLANG_OSCAR_CALIBRATION_LOCK_DIR.get())
    )
    return {
        "lock": os.path.join(lock_dir, f"oscar_{pair_id}.lock"),
        "pending": f"{prefix}.pending.json",
        "complete": f"{prefix}.complete.json",
    }


def determine_oscar_calibration_required() -> bool:
    """Decide from the filesystem whether this launch must fit the pair."""
    configured_k = envs.SGLANG_OSCAR_K_ROTATION_PATH.get()
    configured_v = envs.SGLANG_OSCAR_V_ROTATION_PATH.get()
    if not configured_k and not configured_v:
        return False
    if not configured_k or not configured_v:
        raise ValueError(
            "OSCAR requires both K/V rotation destination paths when either is set"
        )
    k_path, v_path = get_oscar_checkpoint_pair()
    both_exist = os.path.isfile(k_path) and os.path.isfile(v_path)
    # A complete pair in two directories is a hand-made layout; publication
    # into it is not supported, so it is loaded as-is.
    if both_exist and os.path.dirname(k_path) != os.path.dirname(v_path):
        return False
    artifacts = get_oscar_pair_artifact_paths()
    return not both_exist or os.path.isfile(artifacts["pending"])


def oscar_calibration_required() -> bool:
    """The decision inherited by every process of this launch."""
    active = envs.SGLANG_OSCAR_CALIBRATION_ACTIVE
    if active.is_set():
        return active.get()
    return determine_oscar_calibration_required()


def oscar_pending_marker_for(path: str) -> str | None:
    """Pending manifest guarding ``path`` when it is one of the configured
    pair's files and the pair shares a directory; None otherwise."""
    configured_k = envs.SGLANG_OSCAR_K_ROTATION_PATH.get()
    configured_v = envs.SGLANG_OSCAR_V_ROTATION_PATH.get()
    if not (configured_k and configured_v):
        return None
    supplied = os.path.realpath(os.path.abspath(path))
    raw_pair = {
        os.path.realpath(os.path.abspath(configured_k)),
        os.path.realpath(os.path.abspath(configured_v)),
    }
    if supplied not in raw_pair:
        return None
    canonical_k, canonical_v = get_oscar_checkpoint_pair()
    if os.path.dirname(canonical_k) != os.path.dirname(canonical_v):
        return None
    return get_oscar_pair_artifact_paths()["pending"]


class OscarPairReadLock:
    """Shared lock held while a non-calibrating launch loads the pair, so a
    concurrent calibrating process cannot replace K between the K and V
    loads. ``acquire`` is a no-op when nothing needs protecting."""

    def __init__(self) -> None:
        self._handle = None

    def acquire(self) -> None:
        if oscar_calibration_required():
            return
        if not (
            envs.SGLANG_OSCAR_K_ROTATION_PATH.get()
            and envs.SGLANG_OSCAR_V_ROTATION_PATH.get()
        ):
            return
        try:
            lock_path = get_oscar_pair_artifact_paths()["lock"]
        except ValueError:
            # Two-directory legacy pairs are never published online.
            return
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        self._handle = open(lock_path, "a+")  # noqa: SIM115
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_SH)

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
