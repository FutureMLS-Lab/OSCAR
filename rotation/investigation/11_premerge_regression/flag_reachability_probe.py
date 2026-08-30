#!/usr/bin/env python3
"""Is ``SGLANG_OSCAR_MLA_KV_REAL_KERNEL`` reachable at all?

``mla_int2_kv_pool._real_kernel_enabled`` does::

    from sglang.srt import environ as envs
    return bool(envs.SGLANG_OSCAR_MLA_KV_REAL_KERNEL.get())

``environ`` is a *module*; the ``EnvBool`` descriptors are attributes of the
``Envs`` class, exposed as the module-level singleton ``environ.envs``. The
module itself has no such attribute and no module-level ``__getattr__``, so the
lookup raises ``AttributeError`` -- which the surrounding
``except Exception: return False`` swallows. This probe decides that
empirically instead of by reading, and checks the same question for
``_quant_requested``.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    out: dict = {}
    from sglang.srt import environ as environ_mod
    from sglang.srt.environ import envs as envs_singleton

    out["environ_module_file"] = environ_mod.__file__
    out["module_has_attr"] = hasattr(
        environ_mod, "SGLANG_OSCAR_MLA_KV_REAL_KERNEL")
    out["singleton_has_attr"] = hasattr(
        envs_singleton, "SGLANG_OSCAR_MLA_KV_REAL_KERNEL")
    out["module_has_getattr"] = hasattr(environ_mod, "__getattr__")
    try:
        environ_mod.SGLANG_OSCAR_MLA_KV_REAL_KERNEL
        out["module_lookup"] = "ok"
    except Exception as e:                                    # noqa: BLE001
        out["module_lookup"] = f"{type(e).__name__}: {e}"

    from sglang.srt.mem_cache import mla_int2_kv_pool as pool
    out["pool_file"] = pool.__file__

    for val in (None, "0", "1", "true"):
        if val is None:
            os.environ.pop("SGLANG_OSCAR_MLA_KV_REAL_KERNEL", None)
        else:
            os.environ["SGLANG_OSCAR_MLA_KV_REAL_KERNEL"] = val
        out[f"_real_kernel_enabled(env={val!r})"] = pool._real_kernel_enabled()
        out[f"envs_singleton.get(env={val!r})"] = \
            envs_singleton.SGLANG_OSCAR_MLA_KV_REAL_KERNEL.get()
    os.environ.pop("SGLANG_OSCAR_MLA_KV_REAL_KERNEL", None)

    os.environ["SGLANG_OSCAR_MLA_KV_ROTATION_PATH"] = "/nonempty"
    out["_quant_requested(rot_path set)"] = pool._quant_requested()
    os.environ.pop("SGLANG_OSCAR_MLA_KV_ROTATION_PATH", None)

    out["VERDICT"] = (
        "FLAG_IS_INERT: _real_kernel_enabled() is False for every value of the "
        "env var, so the packed kernel branch is unreachable"
        if not any(v for k, v in out.items()
                   if k.startswith("_real_kernel_enabled"))
        else "flag reachable"
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
