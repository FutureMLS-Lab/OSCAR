"""Manifold-Constrained Hyper-Connections (mHC), as used by GLM-5.3-Flash.

Ported field-for-field from ``transformers.models.glm5_next``
(``Glm5NextTextHyperConnection``, transformers 5.16.1). It is written against
the reference rather than derived, because the shapes alone do not determine
the maths and a plausible-looking guess here fails in the worst way.

The shapes DO look determinative, which is the trap. ``base`` is [24] and
``fn`` is [24, 4*hidden], and 24 = (2 + H) * H at H=4 splits as 4 + 4 + 16 --
so "per-stream input weights, per-stream output weights, an H x H mixing
matrix" is the obvious reading, and it is the right *split*. What no shape
reveals is that ``comb`` is softmaxed and then projected onto the
doubly-stochastic manifold by **20 Sinkhorn-Knopp iterations**, that ``post``
is ``2 * sigmoid`` (range [0, 2], not [0, 1]), that ``pre`` carries an epsilon,
or that the whole mapping runs in float32 under an *unweighted* RMS norm. An
implementation with the right split and ordinary linear mixing would load every
tensor, run at full speed, and emit fluent nonsense -- the exact failure mode
that cost two debugging cycles on Kimi-K3's latent path.

Reference:
    pre, post, comb = split(F.linear(rmsnorm(streams.flatten(2)), fn), [H, H, H*H])
    pre  = sigmoid(pre  * scale[0] + base[:H])   + eps
    post = 2 * sigmoid(post * scale[1] + base[H:2H])
    comb = softmax(comb * scale[2] + base[2H:], -1) + eps ; Sinkhorn(iters)
    collapsed = (pre[..., None] * streams).sum(stream_axis)

and the decoder layer then writes back

    streams = post[..., None] * y[..., None, :] + comb.mT @ streams
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class UnweightedRMSNorm(nn.Module):
    """RMS norm with no learned gain.

    A separate class rather than sglang's ``RMSNorm`` with a ones-initialised
    weight: this one has no parameter at all, so the checkpoint (which ships
    none for it) loads without a spurious missing-key, and no optimiser or
    quantizer can pick it up as something to touch.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class HyperConnection(nn.Module):
    """One mHC site. A decoder layer owns two: one for attention, one for MLP."""

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.input_norm = UnweightedRMSNorm(eps=rms_norm_eps)
        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``hidden_streams`` is [..., hc_mult, hidden]; returns (post, comb, collapsed)."""
        hc = self.hc_mult
        # float32 throughout, and the flatten is over the LAST TWO axes
        # (streams, hidden) -- the reference uses flatten(start_dim=2) on a
        # [B, S, H, D] tensor. sglang hands attention a flat [tokens, H, D],
        # so the equivalent is flatten(start_dim=-2); writing 2 here would
        # flatten the wrong pair and still produce a correctly-shaped result.
        flat = self.input_norm(hidden_streams.flatten(start_dim=-2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split(
            [hc, hc, hc * hc], dim=-1
        )
        pre_b, post_b, comb_b = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)

        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(
            hc, hc
        )
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        # Sinkhorn-Knopp: the FIRST step normalises columns only, then each of
        # the remaining (iters - 1) steps does rows then columns. Not a
        # symmetric loop -- starting with rows, or running iters full steps,
        # gives a different (still plausible) matrix.
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=-2).to(
            hidden_streams.dtype
        )
        return post, comb, collapsed


def hc_writeback(
    post: torch.Tensor,
    comb: torch.Tensor,
    sublayer_out: torch.Tensor,
    residual_streams: torch.Tensor,
) -> torch.Tensor:
    """streams = post * y  +  comb^T @ streams   (reference decoder layer)."""
    dtype = residual_streams.dtype
    return post.to(dtype).unsqueeze(-1) * sublayer_out.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual_streams
    )


class HyperHead(nn.Module):
    """Final collapse of the stream axis. GLM-5.3-Flash uses an UNWEIGHTED mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=-2)
