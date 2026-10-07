#!/usr/bin/env python3
"""KL divergence between two serving arms at long context, on real long documents.

For each window length L in ``--lengths``, windows of L tokens are cut from
PG-19 test books (the first L tokens of each book that is long enough, books
in dataset order) until ``--scored-tokens`` tokens at positions >= L/2 are
covered. Every window goes through the real serving path with input logprobs
requested from position L/2 - 1, which makes a server launched with
``SGLANG_OSCAR_LOGITS_DUMP_DIR`` write the full-vocabulary logits of every
``--stride``-th predicted position >= L/2 (one file per prefill chunk).

The run without ``--ref-dir`` is the reference (BF16): its dumps are kept
under ``--out-dir/L<L>/w<i>/``. A run with ``--ref-dir`` scores itself
against that reference, position by position: KL(P_ref || Q_arm), top-1
agreement and the NLL of the true next token under both, then deletes its own
dumps. Positions and the tokens fed at each row must match exactly between
the two runs, or the run fails.

  python kl_longctx.py --port 31021 --model Qwen/Qwen3-8B --dump-dir RUN/dump \\
      --out-dir /scratch/oscar2/qwen3-8b/kl/bf16 [--ref-dir /scratch/oscar2/qwen3-8b/kl/bf16]
"""
import argparse
import glob
import json
import math
import os
import shutil
import time

import requests
import torch


def load_books(model: str, dataset: str, cache_dir: str):
    from transformers import AutoTokenizer

    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, model.replace("/", "__") + ".pt")
    if os.path.exists(cache):
        books = torch.load(cache)
        print(f"[kl] {dataset}: {len(books)} books from cache {cache}", flush=True)
        return books
    from datasets import load_dataset

    ds = load_dataset(dataset, split="test")
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    books = []
    t0 = time.time()
    for text in ds["text"]:
        books.append(tok(text, return_tensors=None, add_special_tokens=False)["input_ids"])
    torch.save(books, cache)
    print(f"[kl] {dataset}: {len(books)} books tokenized in {time.time() - t0:.0f}s "
          f"(longest {max(len(b) for b in books)} tokens) -> {cache}", flush=True)
    return books


def make_windows(books, L: int, scored_budget: int):
    """One window per long-enough book in dataset order; a second pass takes the
    next L tokens of the same books if the first pass does not fill the budget."""
    need = math.ceil(scored_budget / (L // 2))
    windows, src = [], []
    for pass_no in (0, 1):
        for bi, b in enumerate(books):
            if len(b) >= (pass_no + 1) * L:
                windows.append(b[pass_no * L : (pass_no + 1) * L])
                src.append((bi, pass_no))
                if len(windows) == need:
                    return windows, src
    raise SystemExit(f"only {len(windows)} windows of {L} tokens available, need {need}")


def post_window(base_url: str, input_ids, logprob_start_len: int, retries: int = 3):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{base_url}/generate", json=payload, timeout=7200)
            r.raise_for_status()
            lps = r.json()["meta_info"]["input_token_logprobs"]
            if len(lps) < len(input_ids) - 1:
                raise RuntimeError(f"short logprobs: {len(lps)} for a window of {len(input_ids)}")
            return
        except Exception as e:  # noqa: BLE001 - retried, then raised
            if attempt == retries - 1:
                raise
            print(f"[kl] retry {attempt + 1} after error: {e}", flush=True)
            time.sleep(5)


def collect_dump(dump_dir: str, before: set, dest: str, ids, score_from: int, stride: int):
    """Move the chunk files the server wrote for one window into `dest` and
    check that they cover exactly the expected positions with the right tokens."""
    time.sleep(1.0)
    files = sorted(f for f in glob.glob(os.path.join(dump_dir, "chunk_*.pt")) if f not in before)
    if not files:
        raise RuntimeError("the server wrote no logits dump for this window (SGLANG_OSCAR_LOGITS_DUMP_DIR unset?)")
    os.makedirs(dest, exist_ok=True)
    pos, fed = [], []
    for f in files:
        d = torch.load(f)
        keep = d["pred_pos"] < len(ids)  # the last row predicts beyond the window
        pos.append(d["pred_pos"][keep])
        fed.append(d["input_ids"][keep])
        shutil.move(f, os.path.join(dest, os.path.basename(f)))
    pos = torch.cat(pos)
    fed = torch.cat(fed)
    expected = torch.tensor([p for p in range(score_from, len(ids)) if p % stride == 0], dtype=torch.int64)
    order = torch.argsort(pos)
    pos, fed = pos[order], fed[order]
    if not torch.equal(pos, expected):
        raise RuntimeError(f"dump positions mismatch: got {pos.numel()} rows {pos[:4].tolist()}..., expected {expected.numel()} rows {expected[:4].tolist()}...")
    want = torch.tensor([ids[p - 1] for p in pos.tolist()], dtype=torch.int64)
    if not torch.equal(fed, want):
        bad = (fed != want).nonzero().flatten()[:4].tolist()
        raise RuntimeError(f"dump token mismatch at rows {bad}: fed {fed[bad].tolist()} vs window {want[bad].tolist()}")
    return pos.numel()


def load_rows(wdir: str, n_ids: int):
    pos, fed, logits = [], [], []
    for f in sorted(glob.glob(os.path.join(wdir, "chunk_*.pt"))):
        d = torch.load(f)
        keep = d["pred_pos"] < n_ids
        pos.append(d["pred_pos"][keep]); fed.append(d["input_ids"][keep]); logits.append(d["logits"][keep])
    pos = torch.cat(pos); order = torch.argsort(pos)
    return pos[order], torch.cat(fed)[order], torch.cat(logits)[order]


def score_window(ref_dir: str, arm_dir: str, ids, device: str):
    p_pos, p_fed, p_log = load_rows(ref_dir, len(ids))
    q_pos, q_fed, q_log = load_rows(arm_dir, len(ids))
    if not torch.equal(p_pos, q_pos) or not torch.equal(p_fed, q_fed):
        raise RuntimeError(f"reference/arm dumps disagree on positions or tokens ({ref_dir} vs {arm_dir})")
    if p_log.shape != q_log.shape:
        raise RuntimeError(f"vocab mismatch {tuple(p_log.shape)} vs {tuple(q_log.shape)}")
    nxt = torch.tensor([ids[p] for p in p_pos.tolist()], dtype=torch.int64)
    kl, top1, nll_p, nll_q = [], [], [], []
    for i in range(0, p_log.shape[0], 256):
        lp = torch.log_softmax(p_log[i : i + 256].to(device).float(), dim=-1)
        lq = torch.log_softmax(q_log[i : i + 256].to(device).float(), dim=-1)
        kl.append((lp.exp() * (lp - lq)).sum(-1).cpu())
        top1.append((lp.argmax(-1) == lq.argmax(-1)).float().cpu())
        n = nxt[i : i + 256].to(device)
        nll_p.append(-lp.gather(1, n[:, None]).squeeze(1).cpu())
        nll_q.append(-lq.gather(1, n[:, None]).squeeze(1).cpu())
    return torch.cat(kl), torch.cat(top1), torch.cat(nll_p), torch.cat(nll_q)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="emozilla/pg19-test")
    ap.add_argument("--lengths", default="8192,16384,32768,65536,131072")
    ap.add_argument("--scored-tokens", type=int, default=131072, help="scored positions per length before striding")
    ap.add_argument("--stride", type=int, default=32, help="must equal the server's SGLANG_OSCAR_LOGITS_DUMP_STRIDE")
    ap.add_argument("--dump-dir", required=True, help="the server's SGLANG_OSCAR_LOGITS_DUMP_DIR")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref-dir", default=None, help="reference run's --out-dir; omitted = this run is the reference")
    ap.add_argument("--label", default="")
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--chat-wrap", default=None, metavar="INSTRUCTION")
    ap.add_argument("--keep-dumps", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    os.makedirs(a.out_dir, exist_ok=True)
    books = load_books(a.model, a.dataset, os.path.join(os.path.dirname(a.out_dir.rstrip("/")), "tokcache"))
    prefix = []
    if a.bos or a.chat_wrap:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
        if a.chat_wrap:
            pre = tok.apply_chat_template([{"role": "user", "content": a.chat_wrap}], add_generation_prompt=True, tokenize=True)
            prefix = list(pre["input_ids"]) if hasattr(pre, "keys") else list(pre)
        elif tok.bos_token_id is not None:
            prefix = [tok.bos_token_id]
        print(f"[kl] prefix of {len(prefix)} tokens before every window", flush=True)
    results = {}
    for L in [int(x) for x in a.lengths.split(",")]:
        windows, src = make_windows(books, L, a.scored_tokens)
        score_from = L // 2 + len(prefix)
        ldir = os.path.join(a.out_dir, f"L{L}")
        t0 = time.time()
        stats = {"kl": [], "top1": [], "nll_ref": [], "nll_arm": []}
        n_pos = 0
        for wi, w in enumerate(windows):
            ids = prefix + (w[:-len(prefix)] if prefix else w)   # keep the request at exactly L tokens
            wdir = os.path.join(ldir, f"w{wi}")
            if a.ref_dir is None and glob.glob(os.path.join(wdir, "chunk_*.pt")):
                n_pos += load_rows(wdir, len(ids))[0].numel()
                continue
            before = set(glob.glob(os.path.join(a.dump_dir, "chunk_*.pt")))
            post_window(base, ids, score_from - 1)
            n_pos += collect_dump(a.dump_dir, before, wdir, ids, score_from, a.stride)
            if a.ref_dir is not None:
                kl, top1, nll_p, nll_q = score_window(os.path.join(a.ref_dir, f"L{L}", f"w{wi}"), wdir, ids, a.device)
                stats["kl"].append(kl); stats["top1"].append(top1); stats["nll_ref"].append(nll_p); stats["nll_arm"].append(nll_q)
                if not a.keep_dumps:
                    shutil.rmtree(wdir)
        rec = {"L": L, "score_from": L // 2, "stride": a.stride, "windows": len(windows), "sources": src,
               "n_positions": n_pos, "sec": time.time() - t0, "label": a.label, "dataset": a.dataset, "prefix_tokens": len(prefix)}
        if a.ref_dir is not None:
            kl = torch.cat(stats["kl"]); top1 = torch.cat(stats["top1"]); nll_p = torch.cat(stats["nll_ref"]); nll_q = torch.cat(stats["nll_arm"])
            if not torch.isfinite(kl).all():
                raise RuntimeError("non-finite KL")
            rec.update(kl_mean=kl.mean().item(), kl_median=kl.median().item(), kl_p90=kl.quantile(0.9).item(), kl_max=kl.max().item(),
                       top1_agree=top1.mean().item(), nll_ref=nll_p.mean().item(), nll_arm=nll_q.mean().item(),
                       ppl_ref=math.exp(nll_p.mean().item()), ppl_arm=math.exp(nll_q.mean().item()))
            print(f"KL [{a.label}] L={L} kl={rec['kl_mean']:.5f} median={rec['kl_median']:.5f} p90={rec['kl_p90']:.4f} "
                  f"top1={rec['top1_agree']:.4f} ppl_ref={rec['ppl_ref']:.4f} ppl_arm={rec['ppl_arm']:.4f} n={n_pos} ({rec['sec']:.0f}s)", flush=True)
        else:
            print(f"REF [{a.label}] L={L} windows={len(windows)} n={n_pos} ({rec['sec']:.0f}s)", flush=True)
        results[str(L)] = rec
        json.dump(results, open(os.path.join(a.out_dir, "kl.json"), "w"), indent=1)
    open(os.path.join(a.out_dir, "DONE"), "w").write(json.dumps({"lengths": a.lengths, "ref": a.ref_dir}))
    print("KL_DONE " + json.dumps({k: {kk: v[kk] for kk in ("kl_mean", "top1_agree", "ppl_arm") if kk in v} for k, v in results.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
