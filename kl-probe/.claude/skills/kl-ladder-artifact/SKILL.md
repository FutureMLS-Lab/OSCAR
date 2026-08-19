---
name: kl-ladder-artifact
description: Publish (or update) the per-bucket KL ladder artifact for a kl-probe run — the shareable page ranking every candidate against the reference, with paired-bootstrap findings. Use when asked to visualize, share, report, or write up the results of a run under runs/<name>/<stamp>/, or to update an existing ladder artifact with new arms.
---

# KL ladder artifact

Turns a finished `runs/<name>/<stamp>/` into the standard ladder page: one card per position
bucket, every candidate ranked inside it, heat-shaded on the ranking metric, plus a findings
block carrying the conclusions that no single table shows.

`metrics.json` has per-bucket percentiles but **no run-level aggregate and no CIs**, so the
numbers this page reports have to be computed. That is what `scripts/build_ladder.py analyze`
is for, and it must run before any prose is written.

## Procedure

**1. Analyze first — never write findings from `metrics.json` alone.**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run python .claude/skills/kl-ladder-artifact/scripts/build_ladder.py analyze \
  --run runs/<name>/<stamp> --pairs armA:armB armC:armD
```

Prints token-weighted mean KL per arm and **paired** bootstrap CIs over `pid`.

- Compare arms with the *paired* differences. The per-arm marginal CIs are ~3× wider and
  overlap for every arm; ranking on them is wrong and reads as "underpowered" when the run
  is not.
- Apply the **noise floor** (default 0.00042, measured on a K=11 pair differing in one layer
  of eleven). A paired CI can exclude zero on a gap that a same-checkpoint replicate could
  not distinguish from rerun noise. Anything below the floor is a TIE, however tight the CI.
- Also worth pulling for the copy: per-bucket splits (bucket 0 separates almost nothing at
  64 prompts), and `exp(-mean(logprob))` from `scores_<arm>.parquet` as a KL-independent
  check — though at ~0.0005 effect sizes ppl has disagreed with KL, so don't use it to break
  ties.

**2. Write the copy** — an intro paragraph, 1–4 findings cards, a footer. This is the part
that cannot be generated: the cards are the argument the page is making.

`cards.json`:

```json
[{"tone":"key",  "lbl":"K-curve, same token set", "big":"0.0301 → 0.0255 → 0.0250",
  "sub":"K=0 → best K=14 → K=21. <code>midsplit2</code> lands within <strong>0.00045</strong> of the 21-layer anchor."},
 {"tone":"good", "lbl":"Splitting beat contiguity", "big":"midsplit2 · 0.02546", "sub":"…"},
 {"tone":"flat", "lbl":"Lattice phase", "big":"+0.00027", "sub":"…null…"},
 {"tone":"warn", "lbl":"Noise floor — read this first", "big":"0.00042", "sub":"…"}]
```

`tone`: `key` headline · `good` confirmed win · `flat` null result · `warn` caveat the reader
must apply. `sub` accepts inline HTML. **Always include a `warn` card for the noise floor** —
without it the ladder reads as if every rank ordering is a result.

Footer must carry the caveats that apply to every row, at minimum: the FP4 candidates run on a
different engine build with `fp8_e4m3` KV vs the reference's bf16 (inflates all absolute KLs,
cancels arm-vs-arm); bucket 0 dominates and separates little; the last bucket may hold a
handful of tokens from one sequence; and which artifacts were reused from which prior run.

**3. Build and publish.**

```bash
uv run python .claude/skills/kl-ladder-artifact/scripts/build_ladder.py build \
  --run runs/<name>/<stamp> --out /tmp/.../ladder.html \
  --title "MiniMax-M3 MXFP8 Placement (K=14) — Per-Bucket Ladder" \
  --intro intro.html --findings cards.json [--extra extra.html] --footer footer.html
```

Then the **Artifact** tool with that `file_path`, a `favicon`, and a one-line `description`.
Write the file to the scratchpad, not the repo.

- **Updating an existing ladder**: republish the *same file path* in the same conversation to
  keep the URL. From a different conversation, pass the artifact's `url`. Keep the favicon
  stable — a changed icon reads as a different page.
- **A new experiment gets a new page**, not an update: the arms, token set and conclusions all
  differ.

## What the script handles

- Token-weighted **Overall** row across all buckets (KL max = max over buckets; KL p99 = the
  token-weighted mean of per-bucket p99s, *not* a true global percentile — say so in the footer).
- Layer sets pulled from `scripts/make_skip_variant.py` `ARMS`, so the page cannot drift from
  the checkpoints. Arms absent from that table (anchors like `nvfp4-alt3x6`) need `--meta`:
  `{"nvfp4-alt3x6": {"layers": "3–5,12–14,…", "k": 21, "origin": "hf", "full": "togethercomputer/…"}}`.
- Set `"new": 1` in `--meta` for arms first scored in this run — they get a badge, which is how
  a reader tells them from reused rows.
- The line under each model name defaults to the MiniMax-M3 phrasing `K=… · MXFP8 layers …`,
  built from `ARMS`. For a study whose arms differ along some other axis, set `"sub"` per arm
  in `--meta` instead — e.g. `"sub": "8-bit block [128,128] · routed experts only"` when the
  variable is *quantized surface* rather than which layers are held at higher precision.
- `--extra` injects a `<section>` above the bucket tables for anything the tables can't show
  (e.g. a fitted depth-value bar chart). The template already carries the CSS: `.pcard`,
  `.prow`, `.pband`, `.ptrack`, `.pbar.hi` / `.pbar.lo`, `.pval`, and `--uni` on `.pchart`
  places the dashed reference line.

## Page conventions (already in the template — don't re-derive)

Ranked by **KL p99** by default with a toggle to KL mean; the ranking column is heat-shaded
green→red on a log scale. Chips highlight one arm across every bucket. Light/dark via
`prefers-color-scheme` plus `data-theme` overrides. Indigo accent, monospace numerals,
`tabular-nums`. Tables scroll inside their own container so the page never scrolls sideways.

The template is `assets/ladder.html.tmpl` — a body fragment with no `<!doctype>`/`<html>`/
`<body>`, since the Artifact tool supplies those. Placeholders: `{{TITLE}} {{H1}} {{INTRO}}
{{FINDINGS}} {{EXTRA}} {{FOOTER}} {{DATA}}`. Edit the template to change the design for every
future ladder; edit the copy to change one page.
