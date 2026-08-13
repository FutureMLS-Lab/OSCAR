# Kimi-K3 OSCAR tooling

These scripts are cluster-agnostic. Paths, model revisions, and output
directories must be supplied by the caller.

1. Launch the BF16-KV calibration server with:

   ```bash
   --oscar-qkv-dump-path <dump-dir> \
   --oscar-qkv-dump-tokens 8192
   ```

2. Send deterministic calibration tokens:

   ```bash
   python run_calibration.py \
     --base-url http://127.0.0.1:30000 \
     --blocks-path <wikitext-blocks.pt> \
     --tokenizer-path <model-or-tokenizer>
   ```

3. Generate and validate rotations:

   ```bash
   DUMP_PATH=<dump-dir> \
   OUTPUT_DIR=<rotation-dir> \
   MODEL_CONFIG=<config.json> \
   ./compute_rotation.sh
   ```

`wikitext2_ppl.py` and `wikitext2_kl.py` evaluate an already-running server.
They never launch Kubernetes jobs or assume a shared filesystem.
