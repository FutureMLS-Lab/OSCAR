#!/bin/bash
# Post-merge smoke: does each KV geometry path still serve coherently?
# ARM=uniform    Qwen3-8B  INT2 shared rotation, uniform quantizer   (regression path)
# ARM=lloydmax   Qwen3-8B  INT2 + SGLANG_LLOYD_MAX=1                 (flush lloyd_max arg restored in merge)
# ARM=gemma      Gemma-4-12B-it INT2 two-geometry + per-geometry clip + recent512
set -uo pipefail
ARM=${ARM:?set ARM}
W=/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/oscar
OUT=$W/rotation/investigation/06_postmerge_smoke/$ARM; mkdir -p "$OUT"
PORT=$((31700 + RANDOM % 200))
git config --global --add safe.directory "$W" 2>/dev/null || true
rm -rf /dev/shm/nccl* /dev/shm/torch_* /dev/shm/sglang* 2>/dev/null || true
# gemma4 needs transformers >= 5.5 (Gemma4TextConfig); the oscar env has 5.3, so
# that arm uses the dedicated g4 venv (transformers 5.9) the gemma work built.
G4V=/home/charlie/CoQuant/.RUD/gemma4-12b/oscar-g4-venv
if [ "$ARM" = gemma ] && [ -d "$G4V" ]; then
  source $G4V/bin/activate
  echo "[smoke] gemma: using g4 venv transformers=$(python -c 'import transformers;print(transformers.__version__)')"
else
  source /home/charlie/miniconda3/etc/profile.d/conda.sh; conda activate oscar
fi
export PYTHONPATH=$W/sglang-research/python:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_smoke_$ARM; mkdir -p $TRITON_CACHE_DIR
export HF_HOME=/shared/huggingface
DEVS=$(ls /var/run/nvidia-container-devices 2>/dev/null | paste -sd,)
if [ -n "$DEVS" ]; then export CUDA_VISIBLE_DEVICES="$DEVS"; else unset CUDA_VISIBLE_DEVICES; fi
nvidia-smi -L 2>&1 | head -2
# the smoke must test the MERGED code: fast-forward this worktree to the pushed merge
( cd $W && git fetch -q oscar zhongzhu/hybrid-model && git merge --ff-only FETCH_HEAD 2>&1 | tail -1 )
echo "[smoke] ARM=$ARM  HEAD=$(cd $W && git log --oneline -1)"
# the gemma merge must be an ancestor (later commits on the branch are fine)
( cd $W && git merge-base --is-ancestor 01b57a00d4 HEAD ) || { echo "[smoke] ABORT: gemma merge is not in this worktree"; exit 1; }

COMMON=(SGLANG_ENABLE_MIXED_KV_WINDOWS=1 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        SGLANG_MIXED_KV_HP_MAX_SPLITS=8 SGLANG_MIXED_KV_HP_DTYPE=bfloat16
        SGLANG_MIXED_KV_SCALE_DTYPE=float32 SGLANG_OSCAR_ABSORB_V_ROTATION=0
        SGLANG_OSCAR_K_CLIP_RATIO=0.96 SGLANG_OSCAR_V_CLIP_RATIO=0.92)
case $ARM in
  uniform|lloydmax)
    ZOO=$W/rotation/OSCAR-RotationZoo/Qwen3-8B/seq20000_prompt83_group128
    MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
    ENVV=("${COMMON[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
          SGLANG_OSCAR_K_ROTATION_PATH=$ZOO/k_rotation_qqt_r_h_pbr.pt
          SGLANG_OSCAR_V_ROTATION_PATH=$ZOO/v_rotation_sst_r_h_pbr.pt)
    [ "$ARM" = lloydmax ] && ENVV+=(SGLANG_LLOYD_MAX=1) || ENVV+=(SGLANG_LLOYD_MAX=0)
    ARGS=(--tensor-parallel-size 1 --kv-cache-quant-group-size 128) ;;
  q332b)
    ZOO=$W/rotation/OSCAR-RotationZoo/Qwen3-32B/seq16000_prompt69_group128
    MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/ | head -1)
    ENVV=("${COMMON[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
          SGLANG_LLOYD_MAX=0
          SGLANG_OSCAR_K_ROTATION_PATH=$ZOO/k_rotation_qqt_r_h_pbr.pt
          SGLANG_OSCAR_V_ROTATION_PATH=$ZOO/v_rotation_sst_r_h_pbr.pt)
    ARGS=(--tensor-parallel-size 2 --kv-cache-quant-group-size 128) ;;
  q354b)
    ROT=$W/rotation/qwen3.5-4B/rotations/calibrated
    MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
    ENVV=("${COMMON[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
          SGLANG_LLOYD_MAX=0
          SGLANG_OSCAR_K_ROTATION_PATH=$ROT/k_rotation.pt
          SGLANG_OSCAR_V_ROTATION_PATH=$ROT/v_rotation.pt)
    ARGS=(--tensor-parallel-size 1 --kv-cache-quant-group-size 256) ;;
  q3535b)
    ROT=$W/rotation/qwen3.5-35B-A3B/rotations/calibrated
    MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/*/ | head -1)
    ENVV=("${COMMON[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
          SGLANG_LLOYD_MAX=1
          SGLANG_OSCAR_K_ROTATION_PATH=$ROT/k_rotation.pt
          SGLANG_OSCAR_V_ROTATION_PATH=$ROT/v_rotation.pt)
    ARGS=(--tensor-parallel-size 4 --kv-cache-quant-group-size 256) ;;
  shared30b)
    CAL=$W/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt117_group128/rotations
    MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)
    ENVV=("${COMMON[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
          SGLANG_LLOYD_MAX=0
          SGLANG_OSCAR_K_ROTATION_PATH=$CAL/k_rotation_qqt_r_h_pbr.pt
          SGLANG_OSCAR_V_ROTATION_PATH=$CAL/v_rotation_sst_r_h_pbr.pt)
    ARGS=(--tensor-parallel-size 2 --kv-cache-quant-group-size 128) ;;
  perhead30b)
    CAL=$W/rotation/qwen3-30b-a3b/GPQA/seq30000_prompt117_group128/rotations
    MODEL=$(ls -d /shared/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)
    ENVV=("${COMMON[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=256
          SGLANG_LLOYD_MAX=0
          SGLANG_OSCAR_K_ROTATION_PATH=$CAL/k_perhead.pt
          SGLANG_OSCAR_V_ROTATION_PATH=$CAL/v_perhead.pt)
    ARGS=(--tensor-parallel-size 2 --kv-cache-quant-group-size 128) ;;
  gemma)
    GROT=/home/charlie/CoQuant/.RUD/gemma4-12b/work/CoQuant/rotation/gemma-4-12B-it/GPQA/latest/rotations
    MODEL=$(ls -d /shared/huggingface/hub/models--google--gemma-4-12B-it/snapshots/*/ | head -1)
    ENVV=("${COMMON[@]}" SGLANG_MIXED_KV_PREFIX_TOKENS=64 SGLANG_MIXED_KV_RECENT_TOKENS=512
          SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS=8192 SGLANG_LLOYD_MAX=0
          SGLANG_OSCAR_K_CLIP_RATIO_SLIDING=0.94 SGLANG_OSCAR_V_CLIP_RATIO_SLIDING=0.88
          SGLANG_OSCAR_K_ROTATION_PATH=$GROT/k_rotation_qqt_r_h_pbr.pt
          SGLANG_OSCAR_V_ROTATION_PATH=$GROT/v_rotation_sst_r_h_pbr.pt)
    # INT2 KV is compact enough that the pool sizes itself to ~850k tokens and
    # then the two-geometry arenas allocate on top of the committed budget and
    # OOM. Cap the pool instead of shaving mem-fraction.
    ARGS=(--tensor-parallel-size 1 --kv-cache-quant-group-size 128
          --context-length 8192 --max-total-tokens 65536
          --disable-piecewise-cuda-graph) ;;
esac

if [ -z "${MODEL:-}" ] || [ ! -d "$MODEL" ]; then
  echo "[smoke] $ARM ABORT: model path did not resolve; hub contents:"
  ls /shared/huggingface/hub/ 2>/dev/null | head -20
  exit 1
fi
( cd $W && PYTHONNOUSERSITE=1 python3 tests/run_cpu_checks.py 2>&1 | tail -4 \
    | sed "s/^/[smoke] $ARM cpucheck: /" )

PREFILL_BE=fa3; GRAPH=(--cuda-graph-max-bs 16)
MEMFRAC=0.75
if [ "$ARM" = gemma ]; then PREFILL_BE=triton; GRAPH=(--disable-cuda-graph); MEMFRAC=0.60; fi
env "${ENVV[@]}" python -m sglang.launch_server --model-path "$MODEL" --served-model-name m \
  "${ARGS[@]}" --kv-cache-dtype int2 \
  --prefill-attention-backend $PREFILL_BE --decode-attention-backend triton \
  --mem-fraction-static $MEMFRAC --max-running-requests 8 "${GRAPH[@]}" \
  --disable-radix-cache --trust-remote-code --host 127.0.0.1 --port $PORT \
  > "$OUT/server.log" 2>&1 &
SP=$!
for i in $(seq 1 180); do curl -s localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SP 2>/dev/null || { echo "[smoke] $ARM SERVER DIED"; tail -25 "$OUT/server.log"; exit 1; }; sleep 5; done
curl -s localhost:$PORT/health >/dev/null 2>&1 || { echo "[smoke] $ARM TIMEOUT"; tail -20 "$OUT/server.log"; exit 1; }
echo "[smoke] $ARM server up"
grep -o "per-head: [0-9]* kv heads" "$OUT/server.log" | head -1
if [ "$ARM" = perhead30b ] && ! grep -q "per-head: 4 kv heads" "$OUT/server.log"; then
  echo "[smoke] perhead30b ABORT: loader did not report the per-head path"; fi

python3 - "$PORT" "$OUT" "$ARM" <<'PY'
import collections, json, re, sys, urllib.request
port, out, arm = sys.argv[1], sys.argv[2], sys.argv[3]
filler = " ".join("note %d says %d." % (i, i % 7) for i in range(600))   # ~2.5k tok -> real quant tier
prompts = [("short", "What is 3+4? Reply briefly.", "7"),
           ("long",  filler + " Now: compute the sum of the first 40 positive integers, then subtract 20. Put the final answer in \\boxed{}.", "800")]


def judge(t, prompt="", expect=None):
    """Detect the failure modes INT2 KV actually produces: digit/symbol soup and
    repetition loops. A letter-ratio threshold cannot do this -- markdown-heavy
    but coherent replies ('**Analyze the Request:**') score as low as garbage.

    Two traps this has already fallen into, hence the care:
      * scoring the request-failure string itself as fluent prose;
      * measuring repetition over letters only, so a model faithfully
        enumerating a repetitive prompt ('note 7 says 0, note 8 says 1') reads
        as 'note says note says ...' -- a fake loop. Numbers are kept as
        tokens, and n-grams the prompt already contains are not counted.
    """
    if t.startswith("<FAILED") or not t.strip():
        return False, {"reason": "no usable response"}
    # A terse correct answer is a pass, and no prose metric applies to it:
    # Gemma-4-12B-it is not a thinking model and answers "What is 3+4? Reply
    # briefly." with "7", which has zero words to measure.
    if expect and expect in t and len(t.strip()) <= 40:
        return True, {"terse_correct": True, "text_len": len(t)}
    if len(t) < 60:
        return False, {"reason": "response too short to judge"}
    body = t.replace("<think>", " ").replace("</think>", " ")
    words = re.findall(r"[A-Za-z][A-Za-z']*", body)
    nw = len(words)
    mean_len = (sum(map(len, words)) / nw) if nw else 0.0
    alnum = [c for c in body if c.isalnum()]
    digit_frac = (sum(c.isdigit() for c in alnum) / len(alnum)) if alnum else 1.0

    tok = lambda s: [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']*|\d+", s)]
    seq = tok(body)
    grams = lambda xs: [tuple(xs[i:i + 5]) for i in range(max(0, len(xs) - 4))]
    from_prompt = set(grams(tok(prompt)))
    mine = [g for g in grams(seq) if g not in from_prompt]
    c = collections.Counter(mine)
    top_gram = (c.most_common(1)[0][1] / max(1, len(mine))) if mine else 0.0

    m = {"words": nw, "mean_word_len": round(mean_len, 2),
         "digit_frac": round(digit_frac, 3), "top_5gram_frac": round(top_gram, 3)}
    checks = {"enough_words": nw >= 25,            # real prose, not a stub
              "wordlike": 2.2 <= mean_len <= 9.0,  # not char soup, not glued tokens
              "not_digit_soup": digit_frac < 0.55,
              "no_repeat_loop": top_gram < 0.06}
    m["failed_checks"] = [k for k, v in checks.items() if not v]
    return all(checks.values()), m


res = {}
for tag, content, expect in prompts:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": content}],
                       "max_tokens": 400, "temperature": 0.6, "top_p": 0.95, "top_k": 20}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        t = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"] or ""
    except Exception as e:
        t = "<FAILED %s>" % e
    ok, m = judge(t, content, expect)
    m["has_expected_answer"] = expect in t
    res[tag] = dict(coherent=ok, len=len(t), text=t, **m)
    print("[smoke] %-10s %-5s coherent=%s len=%d words=%s wl=%s dig=%s rep=%s %s | %s" % (
        arm, tag, ok, len(t), m.get("words"), m.get("mean_word_len"), m.get("digit_frac"),
        m.get("top_5gram_frac"), ("BAD:" + ",".join(m.get("failed_checks", []))) if not ok else "",
        t[:100].replace("\n", " ")), flush=True)
json.dump(res, open("%s/smoke.json" % out, "w"), indent=1)
ok = all(r["coherent"] for r in res.values())
print("[smoke] %s VERDICT=%s (coherent prose, no digit soup, no repetition loop, both prompts)" % (
    arm, "PASS" if ok else "FAIL"))
PY
if grep -q '"coherent": false' "$OUT/smoke.json" 2>/dev/null; then
  echo "[smoke] $ARM server-side traceback:"
  grep -nE "Traceback|Error:|error:|^[A-Za-z.]*Error" "$OUT/server.log" | tail -8
  tail -25 "$OUT/server.log"
fi
kill -KILL $SP 2>/dev/null || true; pkill -KILL -P $SP 2>/dev/null || true
echo "[smoke] $ARM done"
