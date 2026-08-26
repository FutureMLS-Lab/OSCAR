//
// flash-attn-mixed.cu
//
// Mixed flash attention kernel that reads from two KV cache tiers:
//   LP (low precision, q2_0 quantised)  – most K/V entries
//   HP (high precision, f16)             – recent + sink entries
//
// One CUDA thread block handles one query row (iq3,iq2,iq1). The T threads each
// compute the full KQ dot (strided over DK) so no cross-thread reduction of the
// score is needed; each thread owns a disjoint DV/T slice of the value
// accumulator, giving an independent online-softmax per thread that is
// bit-identical across threads. The two tiers (LP then HP) feed one shared
// online-softmax so the merge is mathematically exact.
// op_params[4]==1 marks mixed mode.
//

#include "ggml-cuda.h"
#include "common.cuh"
#include "fattn-common.cuh"

// ============================================================
//  Dequantise a single q2_0 block row into fp32
// ============================================================

static void dbg_dequantize_q2_0(const block_q2_0 * x, float * y, int64_t k) {
    constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
    for (int64_t i = 0; i < k; i += QK2_0) {
        const float d = __half2float(x->d);
        const float m = __half2float(x->m);
        const uint8_t * qs = x->qs;
        for (int j = 0; j < QK2_0; ++j) {
            const int code = (qs[j/4] >> (2*(j%4))) & 0x03;
            y[i + j] = m + d * c[code];
        }
        ++x;
    }
}

// ============================================================
//  KQ dot product helpers
// ============================================================

__device__ __forceinline__ float fp16_to_f32(ggml_fp16_t h) {
    return __half2float(__half(h));
}

__device__ __forceinline__ float kq_dot_q2_0(const char * kd, const float * q, int64_t DK) {
    constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
    float sum = 0.0f;
    const int nblocks = DK / QK2_0;
    for (int b = 0; b < nblocks; ++b) {
        const block_q2_0 * kb = ((const block_q2_0 *)kd) + b;
        const float mean = __half2float(kb->m);
        const float d = __half2float(kb->d);
        const uint8_t * qs = kb->qs;
        for (int j = 0; j < QK2_0; ++j) {
            const int code = (qs[j/4] >> (2*(j%4))) & 0x03;
            sum += (mean + d * c[code]) * q[b*QK2_0 + j];
        }
    }
    return sum;
}

__device__ __forceinline__ float kq_dot_f16(const half * kd, const float * q, int64_t DK) {
    float sum = 0.0f;
    for (int i = 0; i < DK; ++i) {
        sum += __half2float(kd[i]) * q[i];
    }
    return sum;
}

// ============================================================
//  Fused mixed-attention CUDA kernel
// ============================================================
template <int T>
void __global__ flash_attn_ext_mixed_kernel(
    const char * q, int64_t nbq1, int64_t nbq2, int64_t nbq3,
    const char * k_lp, int64_t nbk1, int64_t nbk2, int64_t nbk3,
    const char * v_lp, int64_t nbv1, int64_t nbv2, int64_t nbv3,
    const char * mask_lp_data, int64_t mask_lp_nb1, int64_t mask_lp_nb2, int64_t mask_lp_nb3, int64_t mask_lp_ne2, int64_t mask_lp_ne3,
    const char * k_hp, int64_t nbkh1, int64_t nbkh2, int64_t nbkh3,
    const char * v_hp, int64_t nbvh1, int64_t nbvh2, int64_t nbvh3,
    const char * mask_hp_data, int64_t mask_hp_nb1, int64_t mask_hp_nb2, int64_t mask_hp_nb3, int64_t mask_hp_ne2, int64_t mask_hp_ne3,
    float * dst,
    int64_t DK, int64_t DV, int64_t n_kv, int64_t n_hp, int64_t N, int32_t n_head, int64_t nseq,
    float scale, float max_bias, float logit_softcap,
    int rk2, int rk3, int rv2, int rv3, int rk2h, int rk3h, int rv2h, int rv3h)
{
    // blockIdx.x encodes (iq3,iq2,iq1) packed linearly
    const int64_t ir = blockIdx.x;
    const int iq3 = ir / (n_head * N);
    const int iq2 = (ir - iq3 * n_head * N) / N;
    const int iq1 = ir - iq3 * n_head * N - iq2 * N;

    const uint32_t h = iq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const float slope = (max_bias > 0.0f) ? (h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1)) : 1.0f;

    const float * pq = (const float *)((const char *) q + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));

    const int lane = threadIdx.x;
    const int slice = DV / T;            // each thread owns slice DV/T values of V
    const int j0 = lane * slice;

    float v_acc[128];                    // DV/T <= 128 (DV<=512, T>=4)
    for (int j = 0; j < slice; ++j) v_acc[j] = 0.0f;

    float M = -INFINITY;
    float S = 0.0f;

    const ggml_fp16_t * mp_lp = mask_lp_data ? (const ggml_fp16_t *)(mask_lp_data + iq1*mask_lp_nb1 + (iq2%mask_lp_ne2)*mask_lp_nb2 + (iq3%mask_lp_ne3)*mask_lp_nb3) : NULL;
    const ggml_fp16_t * mp_hp = mask_hp_data ? (const ggml_fp16_t *)(mask_hp_data + iq1*mask_hp_nb1 + (iq2%mask_hp_ne2)*mask_hp_nb2 + (iq3%mask_hp_ne3)*mask_hp_nb3) : NULL;

    const int ik2 = iq2 / rk2;  const int ik3 = iq3 / rk3;
    const int iv2 = iq2 / rv2;  const int iv3 = iq3 / rv3;
    const int ik2h = iq2 / rk2h; const int ik3h = iq3 / rk3h;
    const int iv2h = iq2 / rv2h; const int iv3h = iq3 / rv3h;

    // ---- LP tier (q2_0 K/V) ----
    for (int64_t ic = 0; ic < n_kv; ++ic) {
        const float mv = mp_lp ? slope * fp16_to_f32(mp_lp[ic]) : 0.0f;
        if (mv == -INFINITY) continue;

        const char * kd = (const char *) k_lp + (ic * nbk1 + ik2 * nbk2 + ik3 * nbk3);
        float s = kq_dot_q2_0(kd, pq, DK) * scale;
        if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
        s += mv;

        const float Mold = M;
        float ms = 1.0f, vs = 1.0f;
        const char * vd = (const char *) v_lp + (ic * nbv1 + iv2 * nbv2 + iv3 * nbv3);
        if (s > M) { M = s; ms = expf(Mold - M); for (int j = 0; j < slice; ++j) v_acc[j] *= ms; }
        else       { vs = expf(s - M); }
        // V dequant: q2_0 block
        const block_q2_0 * vblocks = (const block_q2_0 *) vd;
        for (int j = 0; j < slice; ++j) {
            const int dim = j0 + j;
            const block_q2_0 * vb = &vblocks[dim / QK2_0];
            const int sub = dim % QK2_0;
            const int code = (vb->qs[sub/4] >> (2*(sub%4))) & 0x03;
            constexpr float cc[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
            const float val = __half2float(vb->m) + __half2float(vb->d) * cc[code];
            v_acc[j] += vs * val;
        }
        S = S * ms + vs;
    }

    // ---- HP tier (f16 K/V) ----
    for (int64_t ic = 0; ic < n_hp; ++ic) {
        const float mv = mp_hp ? slope * fp16_to_f32(mp_hp[ic]) : 0.0f;
        if (mv == -INFINITY) continue;

        const char * kd = (const char *) k_hp + (ic * nbkh1 + ik2h * nbkh2 + ik3h * nbkh3);
        float s = kq_dot_f16((const half *)kd, pq, DK) * scale;
        if (logit_softcap != 0.0f) s = logit_softcap * tanhf(s);
        s += mv;

        const float Mold = M;
        float ms = 1.0f, vs = 1.0f;
        const char * vd = (const char *) v_hp + (ic * nbvh1 + iv2h * nbvh2 + iv3h * nbvh3);
        if (s > M) { M = s; ms = expf(Mold - M); for (int j = 0; j < slice; ++j) v_acc[j] *= ms; }
        else       { vs = expf(s - M); }
        const half * vh = (const half *) vd;
        for (int j = 0; j < slice; ++j) v_acc[j] += vs * __half2float(vh[j0 + j]);
        S = S * ms + vs;
    }

    const float S_inv = (S == 0.0f) ? 0.0f : 1.0f / S;
    float * out = (float *)((char *) dst + (iq3 * n_head * N + iq1 * N + iq2) * DV * sizeof(float));
    for (int j = 0; j < slice; ++j) out[j0 + j] = v_acc[j] * S_inv;
}

void ggml_cuda_flash_attn_ext_mixed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k_lp = dst->src[1];
    const ggml_tensor * v_lp = dst->src[2];
    const ggml_tensor * mask_lp = dst->src[3];
    const ggml_tensor * k_hp = dst->src[5];
    const ggml_tensor * v_hp = dst->src[6];
    const ggml_tensor * mask_hp = dst->src[7];

    GGML_TENSOR_LOCALS(int64_t, neq, q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, k_lp, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, k_lp, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, v_lp, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, v_lp, nb);
    GGML_TENSOR_LOCALS(int64_t, nekh, k_hp, ne);
    GGML_TENSOR_LOCALS(size_t,  nbkh, k_hp, nb);
    GGML_TENSOR_LOCALS(int64_t, nevh, v_hp, ne);
    GGML_TENSOR_LOCALS(size_t,  nbvh, v_hp, nb);

    const int64_t DK = nek0;
    const int64_t DV = nev0;
    const int64_t N  = neq1;
    const int64_t n_head = neq2;
    const int64_t nseq   = neq3;

    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) scale /= logit_softcap;

    const int64_t n_kv = nek1;
    const int64_t n_hp = nekh1;

    const int rk2 = neq2 / nek2, rk3 = neq3 / nek3;
    const int rv2 = neq2 / nev2, rv3 = neq3 / nev3;
    const int rk2h = neq2 / nekh2, rk3h = neq3 / nekh3;
    const int rv2h = neq2 / nevh2, rv3h = neq3 / nevh3;

    // pick thread count: DV must be divisible by T
    int T = (DV >= 256) ? 256 : (DV >= 128) ? 128 : (DV >= 64) ? 64 : 32;
    while (T > 1 && DV % T != 0) T /= 2;
    GGML_ASSERT(DV % T == 0 && "mixed FA: DV must divide thread count");

    const int64_t nrows = nseq * n_head * N;
    const dim3 grid(nrows);
    const dim3 block(T);

    cudaStream_t stream = ctx.stream();

    flash_attn_ext_mixed_kernel<256><<<grid, block, 0, stream>>>(
        (const char *) q->data,
        nbq1, nbq2, nbq3,
        (const char *) k_lp->data, nbk1, nbk2, nbk3,
        (const char *) v_lp->data, nbv1, nbv2, nbv3,
        mask_lp ? (const char *) mask_lp->data : nullptr, mask_lp ? mask_lp->nb[1] : 0, mask_lp ? mask_lp->nb[2] : 0, mask_lp ? mask_lp->nb[3] : 0, mask_lp ? mask_lp->ne[2] : 0, mask_lp ? mask_lp->ne[3] : 0,
        (const char *) k_hp->data, nbkh1, nbkh2, nbkh3,
        (const char *) v_hp->data, nbvh1, nbvh2, nbvh3,
        mask_hp ? (const char *) mask_hp->data : nullptr, mask_hp ? mask_hp->nb[1] : 0, mask_hp ? mask_hp->nb[2] : 0, mask_hp ? mask_hp->nb[3] : 0, mask_hp ? mask_hp->ne[2] : 0, mask_hp ? mask_hp->ne[3] : 0,
        (float *) dst->data,
        DK, DV, n_kv, n_hp, N, n_head, nseq,
        scale, max_bias, logit_softcap,
        rk2, rk3, rv2, rv3, rk2h, rk3h, rv2h, rv3h);

    GGML_ASSERT(cudaGetLastError() == cudaSuccess);
    cudaDeviceSynchronize();
}
