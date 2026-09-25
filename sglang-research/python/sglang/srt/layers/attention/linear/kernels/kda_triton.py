import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu

if not is_cpu():
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.fla.kda import chunk_kda


_KDA_SEEN_SIGNATURES = set()


def _check_kda_bounds(
    where,
    *,
    q,
    ssm_states,
    cache_indices,
    cu_seqlens,
    A_log,
    v_head_dim=None,
    k=None,
    v=None,
    a=None,
    b=None,
    dt_bias=None,
):
    """Validate the indices the KDA kernels dereference.

    An out-of-range slot index reaches the GPU as an illegal memory access
    inside a Triton launch -- with CUDA graphs on, it corrupts state silently
    and the model emits word salad instead of faulting. Both are terrible ways
    to learn that an index was wrong, so check it on the host when asked.

    Enabled with SGLANG_KDA_DEBUG_BOUNDS=1; off by default (it syncs).
    """
    import os

    if os.environ.get("SGLANG_KDA_DEBUG_BOUNDS", "0") != "1":
        return
    # Every check below reads device memory from the host. That is a sync, and
    # a sync during CUDA graph capture aborts the capture. Capture also feeds
    # dummy metadata, so there is nothing here worth checking.
    if torch.cuda.is_current_stream_capturing():
        return
    n_slots = ssm_states.shape[0]
    n_heads = q.shape[-2]
    problems = []

    # Report the whole shape picture once. The first three guesses (slot
    # range, A_log width, cu_seqlens) all came back clean while the kernel
    # still faulted, which means the violated invariant is one nobody has
    # written down yet -- so print the facts instead of guessing a fourth.
    # Log each distinct shape signature once, not just the first call: the
    # first decode launch replays cleanly on one GPU at exactly the reported
    # shapes, so the launch that faults is a later one the single-shot dump
    # never showed. Batch size varies, so this stays a handful of lines, and
    # the last one printed before the crash is the launch that did it.
    sig = (
        tuple(q.shape), tuple(q.stride()),
        tuple(ssm_states.shape),
        tuple(a.shape) if a is not None else None,
        tuple(b.shape) if b is not None else None,
        int(cache_indices.numel()),
    )
    if sig not in _KDA_SEEN_SIGNATURES:
        _KDA_SEEN_SIGNATURES.add(sig)
        import sys

        def _d(t):
            return f"{tuple(t.shape)}/{tuple(t.stride())}" if t is not None else "None"

        print(
            f"[kda-bounds] {where}: q={_d(q)} k={_d(k)} v={_d(v)} a={_d(a)} "
            f"b={_d(b)} dt_bias={_d(dt_bias)} A_log={_d(A_log)} "
            f"state={_d(ssm_states)} idx_n={cache_indices.numel()} "
            f"idx_range=[{int(cache_indices.min()) if cache_indices.numel() else -1},"
            f"{int(cache_indices.max()) if cache_indices.numel() else -1}] "
            f"cu={cu_seqlens.tolist()[:8] if cu_seqlens is not None else None}",
            file=sys.stderr, flush=True,
        )

    # The state pool is [slots, H, V, K]; the kernel walks it with the head
    # and dim counts it infers from q/k/v. A mismatch there indexes past the
    # slot with a perfectly legal slot number.
    if ssm_states.dim() == 4:
        s_h, s_v, s_k = ssm_states.shape[1], ssm_states.shape[2], ssm_states.shape[3]
        if n_heads > s_h:
            problems.append(
                f"q has {n_heads} heads; the state pool holds {s_h} per slot"
            )
        if q.shape[-1] > s_k:
            problems.append(
                f"q head dim {q.shape[-1]} exceeds the state pool's K {s_k}"
            )
        if v_head_dim is not None and v_head_dim > s_v:
            problems.append(
                f"v head dim {v_head_dim} exceeds the state pool's V {s_v}"
            )
    if cache_indices.numel():
        lo = int(cache_indices.min())
        hi = int(cache_indices.max())
        if lo < 0 or hi >= n_slots:
            problems.append(
                f"cache_indices range [{lo}, {hi}] outside state pool "
                f"[0, {n_slots})"
            )
    if dt_bias is not None and v is not None:
        need = v.shape[-2] * q.shape[-1]
        if dt_bias.numel() < need:
            problems.append(
                f"dt_bias has {dt_bias.numel()} entries; the kernel walks "
                f"HV*K = {need}"
            )
    if a is not None and v is not None:
        need_a = v.shape[-2] * q.shape[-1]
        if a.shape[-1] < need_a:
            problems.append(
                f"a is {a.shape[-1]} wide; the kernel reads HV*K = {need_a} "
                f"per token"
            )
    if A_log is not None and A_log.numel() < n_heads:
        problems.append(
            f"A_log has {A_log.numel()} entries; the kernel indexes it by head "
            f"and there are {n_heads} heads"
        )
    if cu_seqlens is not None and cu_seqlens.numel():
        cs = cu_seqlens.tolist()
        if cs != sorted(cs) or cs[0] != 0:
            problems.append(f"cu_seqlens not a non-decreasing offset list: {cs[:8]}")
        # The kernel derives its sequence count as len(cu_seqlens) - 1 and
        # walks q/k/v to cu_seqlens[-1]. Under pipeline parallelism the batch
        # is split into microbatches, so metadata built for one microbatch and
        # used with another's tensors would be self-consistent on every check
        # above and still run the kernel off the end of q.
        n_tokens = q.shape[1] if q.dim() == 4 else q.shape[0]
        if cs[-1] != n_tokens:
            problems.append(
                f"cu_seqlens ends at {cs[-1]} but q carries {n_tokens} tokens "
                f"-- the kernel would read past the end"
            )
        if (len(cs) - 1) != cache_indices.numel():
            problems.append(
                f"cu_seqlens describes {len(cs) - 1} sequences but there are "
                f"{cache_indices.numel()} state slots"
            )
    if problems:
        raise RuntimeError(
            f"KDA {where}: " + "; ".join(problems)
            + f"  (q={tuple(q.shape)}, ssm_states={tuple(ssm_states.shape)}, "
            f"n_indices={cache_indices.numel()})"
        )


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        _check_kda_bounds(
            "decode",
            q=q,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            cu_seqlens=query_start_loc,
            A_log=A_log,
            v_head_dim=v.shape[-1],
            k=k,
            v=v,
            a=a,
            b=b,
            dt_bias=dt_bias,
        )
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=kwargs.get("lower_bound"),
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # A_log/dt_bias/lower_bound are forwarded so the kernel activates the
        # gate itself, the way decode already does and the way upstream serves
        # K3. Two things depend on it. The gate cumsum has to be taken in log2
        # space (chunk_kda_fwd scales it by log2(e)) because every chunk kernel
        # below reads it with exp2; a caller that pre-activates in natural-log
        # space and passes the result silently halves the decay. And `safe_gate`
        # is derived from `lower_bound is not None` -- K3 trains with
        # gate_lower_bound=-5.0, and that flag changes whether the diagonal
        # blocks arrive pre-inverted, so getting it wrong is a correctness bug,
        # not a tuning knob.
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            A_log=kwargs.get("A_log"),
            dt_bias=kwargs.get("dt_bias"),
            lower_bound=kwargs.get("lower_bound"),
            output_intermediate_states=return_intermediate_states,
        )
