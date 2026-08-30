#!/usr/bin/env python3
"""Print the config the server is ACTUALLY serving with.

The flags we passed and the config that took effect have already diverged once
on this model: the head/worker scripts were fixed and the code was fixed, and
the server still came up with disable_radix_cache=True.  Only the live endpoint
settles it, so every run prints this before it is allowed to score anything.
"""

import json
import sys
import urllib.request

KEYS = [
    # cache / paging -- the thing that has silently disagreed before
    "disable_radix_cache",
    "mamba_scheduler_strategy",
    "page_size",
    # cuda graph: max_bs alone is not enough, the captured LIST is what decides
    # whether a decode batch gets padded into a bigger graph
    "disable_cuda_graph",
    "cuda_graph_max_bs",
    "cuda_graph_bs",
    "max_running_requests",
    # INT2 / OSCAR
    "kv_cache_dtype",
    "kv_cache_quant_group_size",
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "mem_fraction_static",
    "context_length",
    # topology
    "tp_size",
    "pp_size",
    "nnodes",
]


def main(port):
    url = "http://127.0.0.1:%s/get_server_info" % port
    with urllib.request.urlopen(url, timeout=120) as r:
        info = json.load(r)
    # /server_info nests server_args on some builds and flattens it on others.
    sa = info.get("server_args") if isinstance(info.get("server_args"), dict) else info
    for k in KEYS:
        v = sa.get(k, info.get(k, "<absent>"))
        print("[cfg] %-28s %s" % (k, v))

    # sink 64 / recent 256 is a project-wide invariant; confirm it reached the
    # pool rather than trusting that the env var was exported.
    for k in sorted(sa.keys()):
        if "mixed_kv" in k or "oscar" in k.lower():
            print("[cfg] %-28s %s" % (k, sa[k]))

    ok = True
    if sa.get("page_size") != 8:
        print("[cfg] FAIL page_size != 8 (INT2 mixed-KV needs page_size == N_Q == 8)")
        ok = False
    if str(sa.get("kv_cache_dtype")) != "int2":
        print("[cfg] FAIL kv_cache_dtype is not int2")
        ok = False
    if sa.get("kv_cache_quant_group_size") not in (None, "<absent>"):
        print("[cfg] WARN kv_cache_quant_group_size is set; must be UNSET on K3 "
              "(K head dim 192 = 3x64)")
    if sa.get("disable_cuda_graph"):
        print("[cfg] WARN cuda graph is DISABLED")
    print("[cfg] verdict", "OK" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "31255"))
