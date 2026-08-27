#!/usr/bin/env python3
"""Does the KDA config view actually intercept, and does it forward the rest?

A shim is only as good as its interception. Two ways this fails silently:

  * the override does not take effect and every KDA layer is built at the MLA
    value width (256 instead of 128) -- shapes stay self-consistent inside the
    module, so nothing raises;
  * the forwarding is wrong and some unrelated field comes back as None, which
    surfaces much later as a confusing failure in an unrelated place.

So test the view directly, against a real Glm5NextTextConfig built from the
actual checkpoint. This runs on CPU and does not construct the attention module
itself -- that needs the model's kernels and a GPU; the question here is purely
whether the config the module WOULD receive is the right one.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, "/work/oscar/sglang-research/python")


def main() -> int:
    cands = sorted(glob.glob("/hf/hub/models--zai-org--GLM-5.3-Flash/snapshots/*/"))
    if not cands:
        print("no GLM-5.3-Flash snapshot under /hf")
        return 2
    raw = json.load(open(os.path.join(cands[0], "config.json")))["text_config"]

    from sglang.srt.configs.glm5_next import Glm5NextTextConfig
    from sglang.srt.models.glm5_next_kda import _KdaConfigView

    cfg = Glm5NextTextConfig(**raw)
    view = _KdaConfigView(cfg)

    ok = True

    # 1. The override. This is the whole reason the file exists.
    print(f"  config.v_head_dim      = {cfg.v_head_dim}   (MLA width)")
    print(f"  config.linear_head_dim = {cfg.linear_head_dim}   (KDA width)")
    print(f"  view.v_head_dim        = {view.v_head_dim}")
    good = view.v_head_dim == cfg.linear_head_dim == 128 and cfg.v_head_dim == 256
    ok &= good
    print(f"  -> override {'OK' if good else 'FAILED'}"
          + ("" if good else "  <-- KDA layers would be built at the MLA width"))

    # 2. The dict the module actually indexes.
    lac = view.linear_attn_config
    want = {"head_dim": 128, "num_heads": 64, "short_conv_kernel_size": 4}
    print(f"\n  view.linear_attn_config = {lac}")
    for k, v in want.items():
        good = lac.get(k) == v
        ok &= good
        print(f"    {k:<24} {lac.get(k)}  expect {v}  {'ok' if good else 'WRONG'}")

    # 3. Forwarding. Fields the module does not read today but might, plus the
    #    ones the rest of the model needs off the same object.
    print()
    for name in ("hidden_size", "num_hidden_layers", "kv_lora_rank",
                 "qk_rope_head_dim", "rms_norm_eps", "hc_mult",
                 "index_head_dim", "layer_types"):
        a, b = getattr(cfg, name, "<absent>"), getattr(view, name, "<absent>")
        if isinstance(a, list):
            a, b = f"list[{len(a)}]", f"list[{len(b)}]"
        good = a == b
        ok &= good
        print(f"  forward {name:<20} {b}  {'ok' if good else f'WRONG (real={a})'}")

    # 4. A negative control: what the module would have built WITHOUT the view.
    #    A test that cannot distinguish shim-present from shim-absent proves
    #    nothing about the shim.
    print(f"\n  [control] without the view, head_v_dim would be "
          f"{cfg.v_head_dim} -- {cfg.v_head_dim // cfg.linear_head_dim}x the "
          f"real KDA width, and v_proj [8192, 4096] = 64 x 128 would not fit it")

    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
