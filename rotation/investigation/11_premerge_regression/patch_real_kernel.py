#!/usr/bin/env python3
"""Make ``SGLANG_OSCAR_MLA_KV_REAL_KERNEL=1`` actually reach the packed kernel.

Applied ONLY to the arm-local copy of the tree, never to the merge candidate.
Two edits, both asserted so a silent no-op is impossible:

1. ``from sglang.srt import environ as envs`` -> ``from sglang.srt.environ
   import envs``. As merged, ``_real_kernel_enabled()`` looks the ``EnvBool``
   up on the *module* rather than on the ``Envs`` singleton, raises
   ``AttributeError``, and its own ``except Exception: return False`` swallows
   it -- so the flag is inert for every value. Without this edit an arm with
   the flag set to 1 measures the torch fake-quant, i.e. exactly the flag=0
   arm, and "flag=1 changed nothing" would be vacuous.
2. a one-shot ``REAL_PACKED_KERNEL_ACTIVE`` log line inside the packed branch,
   so the server log proves which path ran instead of the manifest asserting it.
"""
from __future__ import annotations

import pathlib
import sys

OLD_IMPORT = (
    "        from sglang.srt import environ as envs\n"
    "        return bool(envs.SGLANG_OSCAR_MLA_KV_REAL_KERNEL.get())"
)
NEW_IMPORT = (
    "        from sglang.srt.environ import envs\n"
    "        return bool(envs.SGLANG_OSCAR_MLA_KV_REAL_KERNEL.get())"
)
ANCHOR = (
    "            _codes, _params, c_kv_q = quantize_dequantize_reuse(\n"
    "                c_kv, self._group_size, self._lloyd_max\n"
    "            )\n"
    "            c_kv_q = c_kv_q.clone()"
)
MARKER = ANCHOR + (
    "\n            global _REAL_KERNEL_LOGGED\n"
    "            if not _REAL_KERNEL_LOGGED:\n"
    "                _REAL_KERNEL_LOGGED = True\n"
    "                logger.warning(\n"
    "                    '[MLAInt2] REAL_PACKED_KERNEL_ACTIVE layer=%s shape=%s '\n"
    "                    'group=%d lloyd_max=%s',\n"
    "                    layer_id, tuple(c_kv.shape), self._group_size,\n"
    "                    self._lloyd_max,\n"
    "                )"
)
DEF = "def _real_kernel_enabled() -> bool:"
FLAG_DECL = "_REAL_KERNEL_LOGGED = False\n\n\n" + DEF


def main() -> int:
    p = pathlib.Path(sys.argv[1])
    s = p.read_text()
    for needle, name in ((OLD_IMPORT, "module-import idiom"),
                         (ANCHOR, "packed-kernel call site"),
                         (DEF, "_real_kernel_enabled def")):
        n = s.count(needle)
        if n != 1:
            print(f"FATAL: expected exactly 1 {name}, found {n}")
            return 1
    s = s.replace(OLD_IMPORT, NEW_IMPORT)
    s = s.replace(ANCHOR, MARKER)
    s = s.replace(DEF, FLAG_DECL, 1)
    p.write_text(s)
    ok = ("from sglang.srt.environ import envs" in s
          and "REAL_PACKED_KERNEL_ACTIVE" in s
          and "_REAL_KERNEL_LOGGED = False" in s)
    print("PATCH_APPLIED" if ok else "PATCH_VERIFY_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
