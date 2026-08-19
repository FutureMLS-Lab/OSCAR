"""Loader fix: serve excluded-from-NVFP4 MoE experts as serialized FP8 (MXFP8
or block-FP8), not bf16.

Needed for the "skipedge" / "skip5" FP4 checkpoints, where a few MoE layers
(e.g. edge layers 3 and 59 in Minimax-M3-0603-FP4-sepscale-skipedge-mxfp8) are
kept at higher precision while the rest are NVFP4. Those layers are listed in
`exclude_modules`, and the stock `ModelOptFp4Config` path returns `None` for an
excluded FusedMoE -> the engine builds bf16 experts and loads the FP8 expert
bytes into them unscaled -> garbage generations at any temperature.

Fix (two parts):

1. In `ModelOptFp4Config.get_quant_method`, for an excluded FusedMoE, probe the
   checkpoint for that layer's `experts.0.w1.weight_scale_inv` tensor and, if
   present, serve the layer with an `Fp8MoEMethod` configured from the probe:
     - uint8 scale  -> MXFP8 (UE8M0, weight_block_size=[1, 32])
     - float32 scale -> block-FP8 (block size = weight.shape // scale.shape)
   This mirrors the image's existing `_infer_excluded_fp8_linear_config`
   machinery for excluded *Linear* layers. Excluded MoE layers with no
   serialized-FP8 expert tensors (e.g. MTP layers, the original reason for
   returning None) keep the stock unquantized fallback, so uniform-NVFP4
   (skip3) and uniform-MXFP8 (0530/0602) models are unaffected.

2. Those layers are run on the **triton MoE runner**, not flashinfer_trtllm:
   flashinfer 0.6.12 (this image) removed gemm1_alpha/gemm1_clamp_limit from
   `trtllm_fp8_block_scale_moe` (0.6.11.post1 still had them), and M3's
   swigluoai activation (alpha=1.702, limit=7.0) needs them -> TypeError at the
   first forward pass. The triton fused_experts path supports fp8 w8a8 with
   block_shape [1, 32] and gemm1_alpha/limit (GPU-verified vs a dequantized
   bf16 reference: cosine 0.9993). UE8M0 uint8 scales are converted losslessly
   to float32 (2^(u8-127)) after load; weights stay FP8.

Supersedes patches/minimax-m3-fp4-excluded-moe-fp8.py (per-tensor
`ModelOptFp8MoEMethod` mismatches block-scaled experts; still garbage).

Target image: tgl:de-minimax-m3-dea4bd032-b200-flashinfer0612-base
Run inside the container:  python3 minimax-m3-fp4-excluded-moe-mxfp8.py
"""

S = "/sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py"
c = open(S).read()

MARKER_V2 = "_ExcludedFp8TritonMoEMethod"
MARKER_V1 = "_get_excluded_fp8_moe_method"

# --- 1. per-layer config cache, next to the existing linear-config cache ---
OLD_INIT = "        self.group_size = group_size\n        self._excluded_fp8_linear_config = None\n"
NEW_INIT = (
    "        self.group_size = group_size\n"
    "        self._excluded_fp8_linear_config = None\n"
    "        self._excluded_fp8_moe_configs = {}\n"
)

# --- 2. probe + config-inference + method helpers, inserted right after
# _get_excluded_fp8_linear_method (mirrors its structure) ---
OLD_HELPERS = (
    "        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod\n"
    "\n"
    "        return Fp8LinearMethod(config)\n"
    "\n"
    "    @staticmethod\n"
    "    def _should_use_mxfp8_for_excluded_linear(prefix: str) -> bool:\n"
)
NEW_HELPERS = (
    "        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod\n"
    "\n"
    "        return Fp8LinearMethod(config)\n"
    "\n"
    "    @staticmethod\n"
    "    def _load_excluded_fp8_moe_probe_tensors(prefix: str):\n"
    '        """Load (weight, scale) of expert-0 w1 for an excluded FusedMoE layer.\n'
    "\n"
    "        ``prefix`` is the sglang module prefix (e.g.\n"
    "        ``model.layers.3.mlp.experts``) while checkpoint keys keep the HF\n"
    "        naming (``language_model.model.layers.3.block_sparse_moe.experts.0.\n"
    "        w1.weight_scale_inv``), so match on layer index + expert suffix.\n"
    "        Returns ``None`` when the layer has no serialized-FP8 expert\n"
    "        tensors, in which case the stock unquantized fallback is correct\n"
    "        (e.g. MTP layers with no quantization scales).\n"
    '        """\n'
    "        import json\n"
    "        from pathlib import Path\n"
    "\n"
    "        from safetensors import safe_open\n"
    "\n"
    "        from sglang.srt.server_args import get_global_server_args\n"
    "\n"
    '        layer_match = re.search(r"\\.layers\\.(\\d+)\\.", prefix + ".")\n'
    "        if layer_match is None:\n"
    "            return None\n"
    "        scale_re = re.compile(\n"
    '            r"\\.layers\\." + layer_match.group(1)\n'
    '            + r"\\.[a-z_.]*experts\\.0\\.w1\\.weight_scale_inv$"\n'
    "        )\n"
    "\n"
    "        model_path = Path(get_global_server_args().model_path)\n"
    '        index_file = model_path / "model.safetensors.index.json"\n'
    "        if not index_file.exists():\n"
    "            return None\n"
    "        with open(index_file) as f:\n"
    '            weight_map = json.load(f)["weight_map"]\n'
    "        for key, shard in weight_map.items():\n"
    "            if not scale_re.search(key):\n"
    "                continue\n"
    '            weight_key = key[: -len("_scale_inv")]\n'
    "            weight_shard = weight_map.get(weight_key)\n"
    "            if weight_shard is None:\n"
    "                return None\n"
    "            with safe_open(\n"
    '                str(model_path / shard), framework="pt", device="cpu"\n'
    "            ) as f:\n"
    "                scale = f.get_tensor(key)\n"
    "            with safe_open(\n"
    '                str(model_path / weight_shard), framework="pt", device="cpu"\n'
    "            ) as f:\n"
    "                weight = f.get_tensor(weight_key)\n"
    "            return weight, scale\n"
    "        return None\n"
    "\n"
    "    def _infer_excluded_fp8_moe_config(self, prefix: str):\n"
    '        """Infer the FP8 format of an excluded MoE layer\'s expert weights.\n'
    "\n"
    "        Mirrors ``_infer_excluded_fp8_linear_config`` but probes the\n"
    "        layer's own routed-expert tensors: skipedge-style ckpts keep edge\n"
    "        MoE layers at MXFP8 (uint8 UE8M0 scales) or block-FP8 (float32\n"
    "        scales) while the remaining layers are NVFP4.\n"
    '        """\n'
    '        layer_match = re.search(r"\\.layers\\.(\\d+)\\.", prefix + ".")\n'
    "        layer_key = layer_match.group(1) if layer_match else prefix\n"
    "        if layer_key in self._excluded_fp8_moe_configs:\n"
    "            return self._excluded_fp8_moe_configs[layer_key]\n"
    "\n"
    "        probe = self._load_excluded_fp8_moe_probe_tensors(prefix)\n"
    "        if probe is None:\n"
    "            self._excluded_fp8_moe_configs[layer_key] = None\n"
    "            return None\n"
    "        weight, scale = probe\n"
    "\n"
    "        if scale.dtype == torch.uint8:\n"
    "            use_mxfp8 = True\n"
    "            weight_block_size = [1, 32]\n"
    "        else:\n"
    "            use_mxfp8 = False\n"
    "            weight_block_size = [\n"
    "                weight.shape[0] // scale.shape[0],\n"
    "                weight.shape[1] // scale.shape[1],\n"
    "            ]\n"
    "\n"
    "        from sglang.srt.layers.quantization.fp8 import Fp8Config\n"
    "\n"
    "        config = Fp8Config(\n"
    "            is_checkpoint_fp8_serialized=True,\n"
    '            activation_scheme="dynamic",\n'
    "            ignored_layers=[],\n"
    "            weight_block_size=weight_block_size,\n"
    "            packed_modules_mapping=self.packed_modules_mapping,\n"
    "            use_mxfp8=use_mxfp8,\n"
    "        )\n"
    "        logger.info(\n"
    '            "Excluded MoE layer %s: serving experts as %s (block %s) on the"\n'
    '            " triton MoE runner.",\n'
    "            prefix,\n"
    '            "MXFP8" if use_mxfp8 else "block-FP8",\n'
    "            weight_block_size,\n"
    "        )\n"
    "        self._excluded_fp8_moe_configs[layer_key] = config\n"
    "        return config\n"
    "\n"
    "    def _get_excluded_fp8_moe_method(self, prefix: str):\n"
    "        config = self._infer_excluded_fp8_moe_config(prefix)\n"
    "        if config is None:\n"
    "            return None\n"
    "\n"
    "        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod\n"
    "\n"
    "        class _ExcludedFp8TritonMoEMethod(Fp8MoEMethod):\n"
    '            """Serve an excluded-from-NVFP4 FP8 MoE layer via the triton runner.\n'
    "\n"
    "            flashinfer 0.6.12 dropped gemm1_alpha/gemm1_clamp_limit\n"
    "            (swigluoai) from trtllm_fp8_block_scale_moe, so the\n"
    "            flashinfer_trtllm runner cannot execute these layers. The\n"
    "            triton fused_experts path supports fp8 w8a8 with block_shape\n"
    "            [1, 32] (and [128, 128]) plus gemm1_alpha/limit. UE8M0 uint8\n"
    "            scales are converted losslessly to float32 (2^(u8-127)) after\n"
    "            load; expert weights stay FP8.\n"
    '            """\n'
    "\n"
    "            def create_weights(self, layer, *args, **kwargs):\n"
    "                # FusedMoE.weight_loader swaps w1<->w3 into the w31 layout\n"
    "                # when layer.use_flashinfer_trtllm_moe is set and the\n"
    '                # method is an Fp8MoEMethod ("Flashinfer assumes w31\n'
    '                # format"). This layer runs on the triton runner, which\n'
    "                # consumes the canonical [w1; w3] (gate-first) layout, so\n"
    "                # force canonical loading for this layer only.\n"
    "                layer.use_flashinfer_trtllm_moe = False\n"
    "                super().create_weights(layer, *args, **kwargs)\n"
    "\n"
    "            def create_moe_runner(self, layer, moe_runner_config):\n"
    "                self.moe_runner_config = moe_runner_config\n"
    "                self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)\n"
    "\n"
    "            def apply(self, layer, dispatch_output):\n"
    "                import os\n"
    "\n"
    '                if os.environ.get("SGLANG_EXCLUDED_MOE_ZERO") == "1":\n'
    "                    # debug: contribute nothing from these routed experts\n"
    "                    from sglang.srt.layers.moe.token_dispatcher import (\n"
    "                        StandardCombineInput,\n"
    "                    )\n"
    "\n"
    "                    return StandardCombineInput(\n"
    "                        hidden_states=torch.zeros_like(\n"
    "                            dispatch_output.hidden_states\n"
    "                        )\n"
    "                    )\n"
    "                topk_output = dispatch_output.topk_output\n"
    '                if hasattr(topk_output, "to_standard"):\n'
    "                    # The global flashinfer_trtllm backend makes TopK emit\n"
    "                    # the bypassed format (routing fused into the trtllm\n"
    "                    # kernel); the triton runner needs explicit topk ids\n"
    "                    # and weights, so materialize them via select_experts.\n"
    "                    topk_config = topk_output.topk_config\n"
    "                    if (\n"
    "                        get_moe_runner_backend().is_flashinfer_trtllm()\n"
    '                        and getattr(topk_config, "routed_scaling_factor", None)\n'
    "                        not in (None, 1.0)\n"
    "                        and getattr(\n"
    '                            topk_config, "apply_routed_scaling_factor_on_output", False\n'
    "                        )\n"
    "                    ):\n"
    "                        # The model compensates for the bypassed trtllm path\n"
    "                        # by post-multiplying the routed output with\n"
    "                        # routed_scaling_factor (MiniMaxM3MoE.forward_normal\n"
    "                        # `need_post_alpha`, keyed on the GLOBAL backend, so\n"
    "                        # it fires for this layer too). select_experts would\n"
    "                        # apply it again in topk_weights -> double-scaling.\n"
    "                        # Neutralize it in the conversion.\n"
    "                        import dataclasses\n"
    "\n"
    "                        topk_output = topk_output._replace(\n"
    "                            topk_config=dataclasses.replace(\n"
    "                                topk_config, routed_scaling_factor=1.0\n"
    "                            )\n"
    "                        )\n"
    "                    dispatch_output = dispatch_output._replace(\n"
    "                        topk_output=topk_output.to_standard(\n"
    '                            layer_id=getattr(layer, "layer_id", None)\n'
    "                        )\n"
    "                    )\n"
    "                return super().apply(layer, dispatch_output)\n"
    "\n"
    "            def process_weights_after_loading(self, layer):\n"
    "                if self.use_mxfp8:\n"
    '                    for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):\n'
    "                        param = getattr(layer, name)\n"
    "                        param.data = (\n"
    "                            (param.data.to(torch.int32) << 23)\n"
    "                            .view(torch.float32)\n"
    "                            .contiguous()\n"
    "                        )\n"
    "                        param.format_ue8m0 = False\n"
    "                    self.use_mxfp8 = False\n"
    "                    self.quant_config.use_mxfp8 = False\n"
    "                super().process_weights_after_loading(layer)\n"
    "\n"
    "        return _ExcludedFp8TritonMoEMethod(config)\n"
    "\n"
    "    @staticmethod\n"
    "    def _should_use_mxfp8_for_excluded_linear(prefix: str) -> bool:\n"
)

# --- 3. route excluded FusedMoE through the probe before the stock path ---
OLD_GQM = (
    "    def get_quant_method(self, layer: torch.nn.Module, prefix: str):\n"
    "        from sglang.srt.layers.linear import LinearBase\n"
    "\n"
    "        if isinstance(layer, LinearBase) and self._should_use_mxfp8_for_excluded_linear(\n"
)
NEW_GQM = (
    "    def get_quant_method(self, layer: torch.nn.Module, prefix: str):\n"
    "        from sglang.srt.layers.linear import LinearBase\n"
    "        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE\n"
    "\n"
    "        if isinstance(layer, FusedMoE) and self.is_layer_excluded(prefix):\n"
    "            # Excluded-from-NVFP4 MoE experts may be serialized FP8 on disk\n"
    "            # (skipedge/skip5 ckpts keep edge layers at MXFP8/block-FP8).\n"
    "            # The stock path returns None -> bf16 experts load the FP8\n"
    "            # bytes unscaled -> garbage. Serve them as FP8 instead.\n"
    "            method = self._get_excluded_fp8_moe_method(prefix)\n"
    "            if method is not None:\n"
    "                return method\n"
    "            # No FP8 expert tensors in the ckpt (e.g. MTP layers): fall\n"
    "            # through to the stock unquantized fallback.\n"
    "\n"
    "        if isinstance(layer, LinearBase) and self._should_use_mxfp8_for_excluded_linear(\n"
)

if MARKER_V2 in c:
    print("[patch] already applied (v2)")
    raise SystemExit(0)
if MARKER_V1 in c:
    raise SystemExit("[patch] v1 of this patch is applied; start from a fresh image/container")

for label, old in (("init-cache", OLD_INIT), ("helpers", OLD_HELPERS), ("gqm", OLD_GQM)):
    if c.count(old) != 1:
        raise SystemExit(
            f"[patch] PATTERN '{label}' matched {c.count(old)}x, expected 1 "
            "(modelopt_quant.py changed?)"
        )

c = c.replace(OLD_INIT, NEW_INIT, 1)
c = c.replace(OLD_HELPERS, NEW_HELPERS, 1)
c = c.replace(OLD_GQM, NEW_GQM, 1)

compile(c, S, "exec")  # syntax sanity check before writing
open(S, "w").write(c)
print("[patch] excluded-MoE-MXFP8 fix applied (v2: triton runner for excluded MoE)")
