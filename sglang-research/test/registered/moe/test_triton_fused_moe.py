import unittest

import torch
from tqdm import tqdm

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_kernels import TritonKernelsQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-large")


class TestFusedMOE(CustomTestCase):
    NUM_EXPERTS = [8, 64]
    TOP_KS = [2, 4]

    @staticmethod
    def create_random_cuda_tensor(shape, dtype, mean=0, std=0.01):
        """Create a random CUDA tensor

        Args:
            shape: Tensor shape
            dtype: Data type
            mean: Mean value
            std: Standard deviation

        Returns:
            torch.Tensor: Randomly initialized CUDA tensor
        """
        return torch.empty(shape, dtype=dtype, device="cuda").normal_(mean, std)

    def get_tolerance(self, dtype):
        """Get tolerance values for different data types

        Args:
            dtype: Data type

        Returns:
            tuple: (relative tolerance, absolute tolerance)
        """
        if dtype == torch.float32:
            return 1e-5, 1e-5
        elif dtype in [torch.float16, torch.bfloat16]:
            return 1e-5, 1e-5
        else:
            return 1e-2, 1e-2  # Default values for other types

    def torch_naive_moe(
        self,
        a,
        w1,
        w2,
        score,
        topk,
        return_per_expert: bool = False,
    ):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)

        if w1.dtype == torch.float8_e4m3fn:
            w1_compute = w1.to(a.dtype)
            w2_compute = w2.to(a.dtype)
        else:
            w1_compute = w1
            w2_compute = w2

        for i in range(w1_compute.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = SiluAndMul()(
                    a[mask] @ w1_compute[i].transpose(0, 1)
                ) @ w2_compute[i].transpose(0, 1)

        weighted = out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(
            out.dtype
        )

        if return_per_expert:
            return weighted

        return weighted.sum(dim=1)

    def _test_case(self, m, n, k, e, topk, dtype):
        rtol, atol = self.get_tolerance(dtype)

        a = self.create_random_cuda_tensor((m, k), dtype)
        w1 = self.create_random_cuda_tensor((e, 2 * n, k), dtype)
        w2 = self.create_random_cuda_tensor((e, k, n), dtype)
        w1_tri = w1.clone()
        w2_tri = w2.clone()
        w1_tri = w1_tri.transpose(-2, -1).contiguous()
        w2_tri = w2_tri.transpose(-2, -1).contiguous()
        score = self.create_random_cuda_tensor((m, e), dtype)

        topk_op = TopK(
            top_k=topk,
            renormalize=False,
            use_grouped_topk=False,
        )
        topk_op.topk_config.output_format = TopKOutputFormat.TRITON_KERNEL
        triton_topk_output = topk_op.forward_cuda(
            hidden_states=a,
            router_logits=score,
        )

        quant_info = TritonKernelsQuantInfo(w13_weight=w1_tri, w2_weight=w2_tri)

        dispatch_output = StandardDispatchOutput(
            hidden_states=a, hidden_states_scale=None, topk_output=triton_topk_output
        )

        torch_per_expert = self.torch_naive_moe(
            a, w1, w2, score, topk, return_per_expert=True
        )
        torch_combined = torch_per_expert.sum(dim=1)

        def run_runner(config):
            runner = MoeRunner(MoeRunnerBackend.TRITON_KERNELS, config)
            result = runner.run(dispatch_output, quant_info)
            return result.hidden_states

        # Combined output (no_combine=False)
        non_fused_config = MoeRunnerConfig(inplace=False)
        non_fused_output = run_runner(non_fused_config)
        torch.testing.assert_close(
            non_fused_output, torch_combined, rtol=rtol, atol=atol
        )

        # Per-expert output (no_combine=True)
        non_fused_no_combine_config = MoeRunnerConfig(
            inplace=False, no_combine=True, top_k=topk
        )
        non_fused_no_combine_output = run_runner(non_fused_no_combine_config)
        torch.testing.assert_close(
            non_fused_no_combine_output, torch_per_expert, rtol=rtol, atol=atol
        )

    def test_various_configurations(self):
        m_values = [1, 32, 64, 256]
        n_values = [128, 1024]
        k_values = [128, 512, 1024]
        dtypes = [torch.bfloat16]

        # Calculate total number of tests
        total_tests = (
            len(m_values)
            * len(n_values)
            * len(k_values)
            * len(self.NUM_EXPERTS)
            * len(self.TOP_KS)
            * len(dtypes)
        )

        # Create progress bar
        with tqdm(total=total_tests, desc="Running MoE tests") as pbar:
            for m in m_values:
                for n in n_values:
                    for k in k_values:
                        for e in self.NUM_EXPERTS:
                            for topk in self.TOP_KS:
                                for dtype in dtypes:
                                    with self.subTest(
                                        m=m,
                                        n=n,
                                        k=k,
                                        e=e,
                                        topk=topk,
                                        dtype=dtype,
                                    ):
                                        self._test_case(
                                            m,
                                            n,
                                            k,
                                            e,
                                            topk,
                                            dtype,
                                        )
                                        torch.cuda.empty_cache()
                                    pbar.update(1)

    def test_situ_with_checkpoint_parameters(self):
        m, n, k, e, topk = 4, 128, 128, 8, 2
        dtype = torch.bfloat16
        a = self.create_random_cuda_tensor((m, k), dtype, std=0.1)
        gate_up = self.create_random_cuda_tensor((e, 2 * n, k), dtype, std=0.1)
        w2 = self.create_random_cuda_tensor((e, k, n), dtype, std=0.1)
        score = self.create_random_cuda_tensor((m, e), dtype)

        # triton_kernels applies its fused activation to adjacent output pairs.
        # Kimi's separate w1/w3 checkpoint rows are interleaved before swizzle.
        gate_up_interleaved = (
            gate_up.view(e, 2, n, k)
            .transpose(1, 2)
            .reshape(e, 2 * n, k)
            .contiguous()
        )
        w1_tri = gate_up_interleaved.transpose(-2, -1).contiguous()
        w2_tri = w2.transpose(-2, -1).contiguous()

        topk_op = TopK(top_k=topk, renormalize=False, use_grouped_topk=False)
        topk_op.topk_config.output_format = TopKOutputFormat.TRITON_KERNEL
        triton_topk_output = topk_op.forward_cuda(
            hidden_states=a,
            router_logits=score,
        )
        dispatch_output = StandardDispatchOutput(
            hidden_states=a,
            hidden_states_scale=None,
            topk_output=triton_topk_output,
        )
        quant_info = TritonKernelsQuantInfo(
            w13_weight=w1_tri,
            w2_weight=w2_tri,
            w13_bias=torch.zeros(e, 2 * n, device="cuda", dtype=torch.float32),
            w2_bias=torch.zeros(e, k, device="cuda", dtype=torch.float32),
        )
        config = MoeRunnerConfig(
            inplace=False,
            activation="situ",
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )
        actual = MoeRunner(MoeRunnerBackend.TRITON_KERNELS, config).run(
            dispatch_output, quant_info
        ).hidden_states

        scores = torch.softmax(score.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, topk)
        expected = torch.zeros_like(actual)
        for token_id in range(m):
            for route_id in range(topk):
                expert_id = topk_ids[token_id, route_id]
                gate = a[token_id].float() @ gate_up[expert_id, :n].float().T
                up = a[token_id].float() @ gate_up[expert_id, n:].float().T
                activated = (
                    4.0
                    * torch.tanh(gate / 4.0)
                    * torch.sigmoid(gate)
                    * 25.0
                    * torch.tanh(up / 25.0)
                )
                expert_output = activated @ w2[expert_id].float().T
                expected[token_id] += (
                    expert_output * topk_weights[token_id, route_id]
                ).to(dtype)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
