#!/usr/bin/env python3
"""Which (kv_group_num, decode batch) pairs the head-tiling defect hits.

The grouped INT2 decode wrapper picks its head tile from the batch size:

    kv_group_num <= 8 : BLOCK_H = 4  if batch >= 16 else 8
    kv_group_num >  8 : BLOCK_H = 16 if batch >= 16 else 8

and the kernel's head mapping is only valid when a head block lies wholly
inside one KV group, i.e. ``BLOCK_H >= kv_group_num or kv_group_num %
BLOCK_H == 0``. Everything else silently attends to the wrong KV head.

Note the two regimes point opposite ways: kv_group_num 5-7 breaks at *large*
batch (MiniMax-M2.7), kv_group_num 9-15 breaks at *small* batch (GLM-4.7).
Scanning a HF cache for the models actually in use is the point of ``--scan``.

Usage:
    python3 affected_geometries.py                       # the rule, as a table
    python3 affected_geometries.py --scan /shared/huggingface/hub
"""
import argparse
import glob
import json
import os


def block_h(batch: int, kv_group_num: int) -> int:
    if kv_group_num <= 8:
        return 4 if batch >= 16 else 8
    return 16 if batch >= 16 else 8


def is_broken(batch: int, kv_group_num: int) -> bool:
    bh = block_h(batch, kv_group_num)
    return bh < kv_group_num and kv_group_num % bh != 0


def verdict(kv_group_num: int) -> str:
    small = is_broken(8, kv_group_num)
    large = is_broken(16, kv_group_num)
    if small and large:
        return "BROKEN at every batch"
    if large:
        return "BROKEN at batch >= 16"
    if small:
        return "BROKEN at batch < 16"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", metavar="HF_HUB_DIR")
    ap.add_argument("--max-group", type=int, default=64)
    args = ap.parse_args()

    if not args.scan:
        print(f"{'kv_group_num':>13}  {'BLOCK_H bs<16':>13}  {'BLOCK_H bs>=16':>14}  verdict")
        for g in range(1, args.max_group + 1):
            print(f"{g:>13}  {block_h(8, g):>13}  {block_h(16, g):>14}  {verdict(g)}")
        return

    rows = {}
    pattern = os.path.join(args.scan, "models--*", "snapshots", "*", "config.json")
    for cfg in glob.glob(pattern):
        name = cfg.split("models--")[1].split(os.sep)[0].replace("--", "/")
        try:
            d = json.load(open(cfg))
        except Exception:
            continue
        t = d.get("text_config") or d.get("llm_config") or d
        q, kv = t.get("num_attention_heads"), t.get("num_key_value_heads")
        if not q or not kv:
            continue
        rows[name] = (q, kv, q // kv)

    print(f"{'model':<50} {'q':>4} {'kv':>4} {'group':>6}  verdict")
    for name in sorted(rows):
        q, kv, g = rows[name]
        print(f"{name:<50} {q:>4} {kv:>4} {g:>6}  {verdict(g)}")


if __name__ == "__main__":
    main()
