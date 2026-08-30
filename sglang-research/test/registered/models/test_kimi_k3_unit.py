import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.configs.kimi_linear import as_kimi_linear_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    OscarQKVCapture,
)


class TestKimiK3ConfigAndWeights(unittest.TestCase):
    def test_nested_text_config_uses_kda_head_dim(self):
        values = {
            "model_type": "kimi_linear",
            "hidden_size": 256,
            "num_attention_heads": 2,
            "num_hidden_layers": 4,
            "linear_attn_config": {
                "head_dim": 128,
                "kda_layers": [1, 3],
                "full_attn_layers": [2, 4],
            },
        }
        remote_config = SimpleNamespace(
            model_type="kimi_linear", to_dict=lambda: values
        )

        config = as_kimi_linear_config(remote_config)

        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.linear_layer_ids, [0, 2])
        self.assertEqual(config.full_attention_layer_ids, [1, 3])

    def test_checkpoint_name_mapping_is_explicit(self):
        adapt = KimiK3ForConditionalGeneration.adapt_checkpoint_name

        self.assertEqual(
            adapt("language_model.model.layers.0.mlp.experts.0.w1.weight_packed"),
            "model.layers.0.mlp.experts.0.w1.weight",
        )
        self.assertIsNone(adapt("vision_tower.blocks.0.weight"))
        self.assertIsNone(adapt("multi_modal_projector.linear.weight"))

        with self.assertRaisesRegex(ValueError, "Unexpected"):
            adapt("unclassified.weight")
        with self.assertRaisesRegex(ValueError, "MTP"):
            adapt("language_model.model.layers.93.self_attn.weight")

    def test_mxfp4_ignore_patterns_are_regex_aware(self):
        ignored = [
            r"re:.*self_attn.*",
            r"re:.*self_attention_res_proj.*",
            r"re:.*mlp\.gate$",
        ]
        self.assertTrue(is_layer_skipped("model.layers.0.self_attn.qkv_proj", ignored))
        self.assertTrue(
            is_layer_skipped("model.layers.0.self_attention_res_proj", ignored)
        )
        self.assertTrue(is_layer_skipped("model.layers.0.mlp.gate", ignored))
        self.assertFalse(
            is_layer_skipped("model.layers.0.mlp.experts.w13_weight", ignored)
        )

    def test_k3_quantization_config_adds_bf16_projections(self):
        config = object.__new__(ModelConfig)
        config.hf_config = SimpleNamespace(
            architectures=["KimiK3ForConditionalGeneration"],
            quantization_config=None,
            compression_config=None,
        )
        config.hf_text_config = SimpleNamespace(
            quantization_config={
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
                "ignore": [r"re:.*self_attn.*"],
            }
        )

        parsed = ModelConfig._parse_quant_hf_config(config)

        self.assertIn(r"re:.*mlp\.gate$", parsed["ignore"])
        self.assertIn(r"re:.*self_attention_res_proj.*", parsed["ignore"])
        self.assertIn(r"re:.*mlp_res_proj.*", parsed["ignore"])
        self.assertIn(r"re:.*output_attn_res_proj.*", parsed["ignore"])
        self.assertIn(r"re:.*routed_expert_down_proj.*", parsed["ignore"])
        self.assertIn(r"re:.*routed_expert_up_proj.*", parsed["ignore"])

    def test_calibration_capture_is_explicit_and_bounded(self):
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True)
        )
        with TemporaryDirectory() as directory:
            capture = OscarQKVCapture(directory, token_limit=3, layer_id=7)
            tensors = [torch.arange(8).reshape(2, 1, 4) + i for i in range(3)]

            with (
                patch(
                    "sglang.srt.models.kimi_k3.get_attention_tp_size", return_value=1
                ),
                patch(
                    "sglang.srt.models.kimi_k3.get_attention_tp_rank", return_value=0
                ),
                patch(
                    "sglang.srt.models.kimi_k3.get_attention_dp_rank", return_value=0
                ),
            ):
                capture(*tensors, forward_batch)
                capture(*tensors, forward_batch)

            self.assertEqual(capture.saved_tokens, 3)
            for name in ("q", "k", "v"):
                chunks = sorted((Path(directory) / "layer_7" / name).glob("*.pt"))
                self.assertEqual([path.name for path in chunks], ["0.pt", "1.pt"])
                self.assertEqual(sum(torch.load(path).shape[0] for path in chunks), 3)


if __name__ == "__main__":
    unittest.main()
