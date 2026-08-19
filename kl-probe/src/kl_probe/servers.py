"""Docker orchestration for model servers.

The compose file has two service slots: `ref` and `cand`. A run has one
reference and a list of candidates that run **sequentially** through the
single reusable `cand` slot — we write the candidate's image/launch/env into the
`.env` file, bring up the `cand` service, score, tear it down, then reconfigure
for the next candidate. So functions take an optional `cand_name`; None selects
the first candidate (single-candidate convenience)."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

from .config import ModelSpec, Provider, RunCfg, ServerRuntime

COMPOSE_DIR = Path(__file__).resolve().parent.parent.parent / "docker"

# Values land inside a `sh -c` command line in the compose file; refuse anything
# that could be re-interpreted by the shell rather than trying to quote it.
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._/:@,= -]*$")
_SAFE_DEVICE_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
# Patch basenames are interpolated into the compose command's static loop, so
# they must be shell-inert (no spaces, no metacharacters).
_SAFE_PATCH_NAME = re.compile(r"^[A-Za-z0-9._-]+\.py$")

# In-container directory where loader patches are bind-mounted (read-only).
PATCHES_MOUNT = "/kl-probe-patches"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


def _project_name(cfg: RunCfg) -> str:
    """Compose project name, derived from run_name so a `servers up` and a later
    `servers down`/`run` for the same config target the same containers."""
    return f"probe-{_slug(cfg.run_name)}"[:63]


def _selected_cand(cfg: RunCfg, cand_name: str | None) -> tuple[ModelSpec, ServerRuntime]:
    """(spec, runtime) for a candidate by name; None = the first candidate."""
    if cand_name is None:
        return cfg.candidate, cfg.runtime.cand
    for spec in cfg.candidates:
        if spec.name == cand_name:
            return spec, cfg.runtime.candidates[cand_name]
    raise ValueError(f"unknown candidate name: {cand_name!r}")


def _launch_cmd(cfg: RunCfg, spec: ModelSpec) -> str:
    """Full in-container launch command for a spec, per its detected engine."""
    if spec.engine is Provider.VLLM:
        # --max-logprobs caps prompt_logprobs server-side; size it to the resolved
        # scoring.top_logprobs_num so score() never gets a truncated top-k. cli
        # resolves it to an int before launch; if it somehow arrives unresolved
        # ("auto"), fall back to 40 (the M3 default), which is also the floor.
        tln = cfg.scoring.top_logprobs_num
        max_logprobs = max(40, tln) if isinstance(tln, int) else 40
        return (
            f"vllm serve {spec.model_path} --host 0.0.0.0 --port 30000 "
            f"--tensor-parallel-size {spec.tp} --max-logprobs {max_logprobs} "
            f"{spec.extra_args}"
        ).strip()
    # SGLANG and TGL share the sglang launcher
    return (
        f"python3 -m sglang.launch_server --model-path {spec.model_path} "
        f"--host 0.0.0.0 --port 30000 --tp-size {spec.tp} {spec.extra_args}"
    ).strip()


def _device_ids(runtime: ServerRuntime) -> list[str]:
    for gpu in runtime.gpus:
        if not _SAFE_DEVICE_ID.match(gpu):
            raise ValueError(f"unsafe GPU device id: {gpu!r}")
    return runtime.gpus


def _patch_files(spec: ModelSpec) -> list[tuple[str, str]]:
    """Resolve a spec's loader patches to (host_abs_path, basename) pairs."""
    pairs = []
    for p in spec.patches:
        host = Path(p).expanduser().resolve()
        if not host.is_file():
            raise ValueError(f"loader patch not found: {p}")
        if not _SAFE_PATCH_NAME.match(host.name):
            raise ValueError(f"unsafe loader patch filename: {host.name!r}")
        pairs.append((str(host), host.name))
    names = [name for _, name in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"loader patch basenames must be unique: {names}")
    return pairs


def _compose_override(cfg: RunCfg, cand_name: str | None = None) -> dict:
    cand_spec, cand_runtime = _selected_cand(cfg, cand_name)
    services = {}
    sides = (
        ("ref", cfg.reference, cfg.runtime.ref),
        ("cand", cand_spec, cand_runtime),
    )
    for service, spec, runtime in sides:
        entry = {
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "device_ids": _device_ids(runtime),
                                "capabilities": ["gpu"],
                            }
                        ]
                    }
                }
            }
        }
        if spec.env:
            for key, val in spec.env.items():
                if not _SAFE_VALUE.match(val):
                    raise ValueError(f"unsafe character in {service} env {key}={val!r}")
            # merged by key with the base file's environment list
            entry["environment"] = dict(spec.env)
        if spec.patches:
            # merged by mount target with the base file's volumes
            entry["volumes"] = [
                f"{host}:{PATCHES_MOUNT}/{name}:ro" for host, name in _patch_files(spec)
            ]
        services[service] = entry
    return {"services": services}


def _env_for(cfg: RunCfg, cand_name: str | None = None) -> dict[str, str]:
    cand_spec, cand_runtime = _selected_cand(cfg, cand_name)

    def side_env(prefix: str, spec: ModelSpec, runtime: ServerRuntime) -> dict[str, str]:
        return {
            f"{prefix}_IMAGE": spec.image,
            f"{prefix}_PORT": str(runtime.port),
            f"{prefix}_LAUNCH_CMD": _launch_cmd(cfg, spec),
            # comma-joined basenames consumed by the compose command's static
            # patch loop; empty when the side has no loader patches
            f"{prefix}_PATCHES": ",".join(name for _, name in _patch_files(spec)),
        }

    env = {
        "HF_HOME_HOST": cfg.hf_home,
        # Shared across projects/models on purpose: kernel caches are keyed by
        # engine version + shapes, and concurrent writers produce identical
        # content-addressed files, so collisions are benign.
        "ENGINE_CACHE_HOST": str(COMPOSE_DIR / ".engine-cache"),
        **side_env("REF", cfg.reference, cfg.runtime.ref),
        **side_env("CAND", cand_spec, cand_runtime),
    }
    for key, val in env.items():
        if not _SAFE_VALUE.match(val):
            raise ValueError(f"unsafe character in {key}={val!r}")
    return env


def write_env_file(cfg: RunCfg, cand_name: str | None = None) -> Path:
    env_path = COMPOSE_DIR / f".env.{_project_name(cfg)}"
    # HF_TOKEN deliberately not written to disk; it is passed through the
    # process environment of the `docker compose` call instead.
    env_path.write_text("".join(f"{k}={v}\n" for k, v in _env_for(cfg, cand_name).items()))
    return env_path


def write_compose_override_file(cfg: RunCfg, cand_name: str | None = None) -> Path:
    override_path = COMPOSE_DIR / f".devices.{_project_name(cfg)}.yml"
    override_path.write_text(yaml.safe_dump(_compose_override(cfg, cand_name), sort_keys=False))
    return override_path


def _profiles(cfg: RunCfg, only: str | None = None) -> list[str]:
    """only: restrict to 'ref' or 'cand' — lets a big pair share one GPU set by
    running the servers sequentially instead of side by side."""
    profiles = []
    if cfg.runtime.ref.managed and only in (None, "ref"):
        profiles += ["--profile", "ref"]
    # candidates all flow through the single 'cand' slot
    if only in (None, "cand"):
        profiles += ["--profile", "cand"]
    return profiles


def compose(
    cfg: RunCfg,
    *args: str,
    check: bool = True,
    only: str | None = None,
    cand_name: str | None = None,
) -> subprocess.CompletedProcess:
    env_path = write_env_file(cfg, cand_name)
    override_path = write_compose_override_file(cfg, cand_name)
    (COMPOSE_DIR / ".engine-cache").mkdir(exist_ok=True)
    cmd = [
        "sudo",
        "--preserve-env=HF_TOKEN",
        "docker",
        "compose",
        "--project-name",
        _project_name(cfg),
        "--env-file",
        str(env_path),
        "-f",
        str(COMPOSE_DIR / "docker-compose.yml"),
        "-f",
        str(override_path),
        *_profiles(cfg, only),
        *args,
    ]
    env = {**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")}
    return subprocess.run(cmd, check=check, env=env)


def _prewarm_weights(cfg: RunCfg, spec: ModelSpec) -> None:
    """Pull the checkpoint's safetensors into the page cache with parallel
    sequential reads before the server mmaps them. /data is WekaFS: without
    this, sglang's mmap loader stalls on synchronous network page faults
    (hours cold vs ~30s warm — see model-evals/minimax-m2.7 README)."""
    if spec.model_path.startswith("/"):
        repo_dir = Path(spec.model_path)
    else:
        repo_dir = (
            Path(cfg.hf_home)
            / "hub"
            / f"models--{spec.model_path.replace('/', '--')}"
            / "snapshots"
        )
    files = list(repo_dir.rglob("*.safetensors")) if repo_dir.is_dir() else []
    if not files:
        print(f"prewarm: no local snapshot for {spec.model_path}; skipping (first download)")
        return
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"prewarm: reading {len(files)} safetensors ({total_gb:.0f} GB) for {spec.model_path} …")
    subprocess.run(
        ["xargs", "-0", "-r", "-P", "16", "-n", "1", "cat"],
        input=b"\0".join(str(f).encode() for f in files),
        stdout=subprocess.DEVNULL,
        check=True,
    )


def up(cfg: RunCfg, only: str | None = None, cand_name: str | None = None) -> list[str]:
    """Start the managed servers detached. Returns the base URLs to poll.
    only='ref'/'cand' starts just that server (sequential staging); cand_name
    picks which candidate flows through the 'cand' slot."""
    cand_spec, cand_runtime = _selected_cand(cfg, cand_name)
    wanted = {
        "ref": (cfg.reference, cfg.runtime.ref),
        "cand": (cand_spec, cand_runtime),
    }
    if only is not None:
        wanted = {only: wanted[only]}
    managed = [(spec, runtime) for spec, runtime in wanted.values() if runtime.managed]
    for spec, _runtime in managed:
        _prewarm_weights(cfg, spec)
    if managed:
        compose(cfg, "up", "-d", only=only, cand_name=cand_name)
    return [runtime.url for _spec, runtime in managed]


def down(cfg: RunCfg, only: str | None = None, cand_name: str | None = None) -> None:
    compose(cfg, "down", only=only, cand_name=cand_name)


def logs(cfg: RunCfg, follow: bool = False, cand_name: str | None = None) -> None:
    compose(cfg, "logs", *(["-f"] if follow else []), check=False, cand_name=cand_name)
