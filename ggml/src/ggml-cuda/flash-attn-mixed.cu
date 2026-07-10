#include "ggml-cuda.h"
#include "common.cuh"
#include "fattn-common.cuh"

// local decode helper for debug dump (mirrors ggml_cpu/quants.c::dequantize_row_q2_0)
static void dbg_dequantize_q2_0(const block_q2_0 * x, float * y, int64_t k) {
    constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
    for (int64_t i = 0; i < k; i += QK2_0) {
        const float d = __half2float(x[i/QK2_0].d);
        const float m = __half2float(x[i/QK2_0].m);
        for (int j = 0; j < QK2_0; ++j) {
            const int by = j/4, sub = j%4;
            const int code = (x[i/QK2_0].qs[by] >> (2*sub)) & 0x03;
            y[i+j] = m + d*c[code];
        }
    }
}

// OSCAR two-tier mixed-precision fused flash-attention (INT2-LP history + F16-HP
// sink/recent) ported from the CPU reference in ggml-cpu/ops.cpp
// (ggml_compute_forward_flash_attn_ext_mixed).
//
// Layout matches ggml_flash_attn_ext_mixed():
//   src[0]=Q (f16)            src[1]=K_lp (q2_0)       src[2]=V_lp (q2_0)
//   src[3]=mask_lp (f16)      src[4]=NULL
//   src[5]=K_hp (f16)         src[6]=V_hp (f16)        src[7]=mask_hp (f16)
//   op_params[0]=scale  op_params[1]=max_bias  op_params[2]=logit_softcap
//   op_params[4]==1 marks mixed mode.
//
// One CUDA thread block handles one query row (iq3,iq2,iq1). The T threads each
// compute the full KQ dot (strided over DK) so no cross-thread reduction of the
// score is needed; each thread owns a disjoint DV/T slice of the value
// accumulator, giving an independent online-softmax per thread that is
// bit-identical across threads. The two tiers (LP then HP) feed one shared
// running max/sum exactly like the CPU reference.
// Q is F32 in the mixed op (ggml asserts nbq0 == sizeof(float)); K may be q2_0 or f16.
// OSCAR q2_0 decode (NO_HADAMARD path): value = group_mean + d_block * centroid[code],
// where group_mean is stored in the FIRST block's m of each 4-block group (128-elem
// Hadamard group) and d_block is that block's own sigma.
__device__ __forceinline__ float fp16_to_f32(ggml_fp16_t h) {
    return __half2float(*(__half *)&h);
}
__device__ __forceinline__ float kq_dot_q2_0(const char * K_c, const float * q, int DK) {
    const block_q2_0 * K = (const block_q2_0 *) K_c;
    constexpr float c[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
    float sum = 0.0f;
    for (int i = 0; i < DK; i += 32) {
        const int ib = i / 32;
        const float mean = __half2float(K[ib].m);   // set-rows replicates group mean into every block's m
        const float d = __half2float(K[ib].d);      // per-block sigma
        for (int j = 0; j < 32; ++j) {
            const int  by  = j / 4;
            const int  sub = j % 4;
            const int  code = (K[ib].qs[by] >> (2 * sub)) & 0x03;
            sum += (mean + d * c[code]) * q[i + j];
        }
    }
    return sum;
}

__device__ __forceinline__ float kq_dot_f16(const char * K_c, const float * q, int DK) {
    const half * K = (const half *) K_c;
    float sum = 0.0f;
    for (int i = 0; i < DK; ++i) {
        sum += __half2float(K[i]) * q[i];
    }
    return sum;
}

template <int T>
__global__ void flash_attn_ext_mixed_kernel(
        const char  * q,
        const char  * k_lp, const char * v_lp,
        const char  * mask_lp_data, int64_t mask_lp_nb1, int64_t mask_lp_nb2, int64_t mask_lp_nb3, int64_t mask_lp_ne2, int64_t mask_lp_ne3,
        const char  * k_hp, const char * v_hp,
        const char  * mask_hp_data, int64_t mask_hp_nb1, int64_t mask_hp_nb2, int64_t mask_hp_nb3, int64_t mask_hp_ne2, int64_t mask_hp_ne3,
        float * __restrict__ dst,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk1, int64_t nbk2, int64_t nbk3,
        int64_t nbv1, int64_t nbv2, int64_t nbv3,
        int64_t nbkh1, int64_t nbkh2, int64_t nbkh3,
        int64_t nbvh1, int64_t nbvh2, int64_t nbvh3,
        int64_t n_kv, int64_t n_hp,
        int64_t DK, int64_t DV, int64_t N, int64_t n_head, int64_t nseq,
        float scale, float max_bias, float logit_softcap,
        int rk2, int rk3, int rv2, int rv3, int rk2h, int rk3h, int rv2h, int rv3h) {

    const int64_t ir = blockIdx.x;
    if (ir >= nseq * n_head * N) return;

    const int iq3 = ir / (n_head * N);
    const int iq2 = (ir - iq3 * n_head * N) / N;
    const int iq1 = (ir - iq3 * n_head * N - iq2 * N);

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
        // dequantize v_lp (q2_0) slice and MAD: value = m + d * centroid[code].
        // V row layout: 8 contiguous q2_0 blocks (12B each) cover DV=256; the KV-row
        // stride is nbv1=96. So dim j is in block (j/32) at offset (j/32)*sizeof(block_q2_0)
        // from row base vd. Matches VEC dequantize_V_q2_0 (ib=j/32, x[ib] contiguous).
        // (set-rows replicates the group mean into every block's m field.)
        for (int j = 0; j < slice; ++j) {
            const int  dim  = j0 + j;
            const block_q2_0 * vb = (const block_q2_0 *)(vd + (int64_t)(dim / 32) * sizeof(block_q2_0));
            const float mean = __half2float(vb->m);
            const float d    = __half2float(vb->d);
            const int  z  = dim % 32;
            const int  by = z / 4;
            const int  sub = z % 4;
            const int  code = (vb->qs[by] >> (2 * sub)) & 0x03;
            constexpr float cc[4] = {-0.9816f, -0.4528f, 0.4528f, 0.9816f};
            v_acc[j] += vs * (mean + d * cc[code]);
        }
        S = S * ms + vs;
    }

    // ---- HP tier (f16 K/V) ----
    for (int64_t ic = 0; ic < n_hp; ++ic) {
        const float mv = mp_hp ? slope * fp16_to_f32(mp_hp[ic]) : 0.0f;
        if (mv == -INFINITY) continue;

        const char * kd = (const char *) k_hp + (ic * nbkh1 + ik2h * nbkh2 + ik3h * nbkh3);
        float s = kq_dot_f16(kd, pq, DK) * scale;
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

    static int dbg_call = 0;
    static bool dbg_done = false;
    if (!dbg_done && dbg_call++ >= 30) {
        // only proceed once a real (populated) row exists: check k_lp[1] block0 d != 0
        float probe_d = 0;
        {
            block_q2_0 b;
            cudaMemcpy(&b, (const char *)k_lp->data + (int64_t)1*nbk1, sizeof(block_q2_0), cudaMemcpyDeviceToHost);
            probe_d = __half2float(b.d);
        }
        if (probe_d <= 0.0f) {
            // not populated yet; skip this dump, keep polling on later calls (no early return)
        } else {
        dbg_done = true;
        fprintf(stderr, "[MIXED-DBG] q=%d k_lp=%d v_lp=%d k_hp=%d v_hp=%d DK=%ld DV=%ld n_kv=%ld n_hp=%ld N=%ld nh=%ld nseq=%ld nbq0=%zu nbk0=%zu nbk1=%zu nbk2=%zu nbk3=%zu nbv1=%zu nbv2=%zu nbv3=%zu nek2=%ld nev2=%ld rk2=%d rv2=%d\n",
                (int)q->type, (int)k_lp->type, (int)v_lp->type, (int)k_hp->type, (int)v_hp->type,
                (long)DK, (long)DV, (long)n_kv, (long)n_hp, (long)N, (long)n_head, (long)nseq,
                q->nb[0], k_lp->nb[0], k_lp->nb[1], k_lp->nb[2], k_lp->nb[3],
                v_lp->nb[1], v_lp->nb[2], v_lp->nb[3],
                (long)nek2, (long)nev2, rk2, rv2);
        // Dump KV rows ic=1 and ic=n_kv-1 (raw bytes + decoded) + Q for head 0, iq1=1.
        {
            const int ics[2] = {1, (int)n_kv - 1};
            for (int w=0; w<2; ++w) {
                const int ic = ics[w];
                const size_t rowb = (size_t)DK / 32 * sizeof(block_q2_0);
                const char * krow = (const char *)k_lp->data + (int64_t)ic*nbk1 + 0*nbk2 + 0*nbk3;
                const char * vrow = (const char *)v_lp->data + (int64_t)ic*nbv1 + 0*nbv2 + 0*nbv3;
                std::vector<uint8_t> kb(rowb), vb(rowb);
                cudaMemcpy(kb.data(), krow, rowb, cudaMemcpyDeviceToHost);
                cudaMemcpy(vb.data(), vrow, rowb, cudaMemcpyDeviceToHost);
                std::vector<float> kf(DK), vf(DV);
                dbg_dequantize_q2_0((const block_q2_0*)kb.data(), kf.data(), DK);
                dbg_dequantize_q2_0((const block_q2_0*)vb.data(), vf.data(), DV);
                fprintf(stderr, "[MIXED-DBG] ic=%d k raw[0..11]:", ic);
                for (int i=0;i<12;i++) fprintf(stderr, " %02x", kb[i]);
                {
                    const block_q2_0 * b0 = (const block_q2_0*)kb.data();
                    fprintf(stderr, " | d=%.4f m=%.4f", __half2float(b0->d), __half2float(b0->m));
                }
                fprintf(stderr, " | k decoded[0..7]:");
                for (int i=0;i<8;i++) fprintf(stderr, " %.4f", kf[i]);
                fprintf(stderr, "\n[MIXED-DBG] ic=%d v raw[0..11]:", ic);
                for (int i=0;i<12;i++) fprintf(stderr, " %02x", vb[i]);
                {
                    const block_q2_0 * b0 = (const block_q2_0*)vb.data();
                    fprintf(stderr, " | d=%.4f m=%.4f", __half2float(b0->d), __half2float(b0->m));
                }
                fprintf(stderr, " | v decoded[0..7]:");
                for (int i=0;i<8;i++) fprintf(stderr, " %.4f", vf[i]);
                fprintf(stderr, "\n");
            }
            const float * pq = (const float *)((const char *)q->data + (int64_t)1*nbq1 + 0*nbq2 + 0*nbq3);
            fprintf(stderr, "[MIXED-DBG] Q[1] first16:");
            for (int i=0;i<16;i++) fprintf(stderr, " %.4f", pq[i]);
            fprintf(stderr, "\n"); fflush(stderr);
            // Recompute full LP attention on CPU (KQ dot via q2_0 decode, masked) for head 0, iq1=1
            if (mask_lp != nullptr) {
                const int head = 0;
                std::vector<float> allK(n_kv*DK), allV(n_kv*DV);
                for (int64_t ic=0; ic<n_kv; ++ic) {
                    const char * krow = (const char *)k_lp->data + ic*nbk1 + (int64_t)head*nbk2;
                    const char * vrow = (const char *)v_lp->data + ic*nbv1 + (int64_t)head*nbv2;
                    std::vector<uint8_t> kb((size_t)DK/32*sizeof(block_q2_0)), vb((size_t)DV/32*sizeof(block_q2_0));
                    cudaMemcpy(kb.data(), krow, kb.size(), cudaMemcpyDeviceToHost);
                    cudaMemcpy(vb.data(), vrow, vb.size(), cudaMemcpyDeviceToHost);
                    dbg_dequantize_q2_0((const block_q2_0*)kb.data(), allK.data()+ic*DK, DK);
                    dbg_dequantize_q2_0((const block_q2_0*)vb.data(), allV.data()+ic*DV, DV);
                }
                const ggml_fp16_t * mp = (const ggml_fp16_t *)((const char *)mask_lp->data + (int64_t)1*mask_lp->nb[1] + (int64_t)(head%(int64_t)mask_lp->ne[2])*mask_lp->nb[2]);
                std::vector<float> dots(n_kv);
                float mx = -1e30f;
                for (int64_t ic=0; ic<n_kv; ++ic) {
                    float d=0; for (int i=0;i<DK;i++) d += pq[i]*allK[ic*DK+i];
                    dots[ic] = d*scale + __half2float(mp[ic]);
                    if (dots[ic] > mx) mx = dots[ic];
                }
                float sum=0; for (int64_t ic=0; ic<n_kv; ++ic) sum += expf(dots[ic]-mx);
                std::vector<float> out(DV, 0);
                for (int64_t ic=0; ic<n_kv; ++ic) { float w = expf(dots[ic]-mx)/sum; for (int i=0;i<DV;i++) out[i]+=w*allV[ic*DV+i]; }
                fprintf(stderr, "[MIXED-DBG] CPU-LP out[0..7]:");
                for (int i=0;i<8;i++) fprintf(stderr, " %.4f", out[i]);
                fprintf(stderr, "\n"); fflush(stderr);
                const float * ko = (const float *)((const char *)dst->data + ((int64_t)0*n_head*N + (int64_t)1*N + head)*DV);
                fprintf(stderr, "[MIXED-DBG] KERNEL dst(iq1=1,h=0)[0..7]:");
                for (int i=0;i<8;i++) fprintf(stderr, " %.4f", ko[i]);
                fprintf(stderr, "\n"); fflush(stderr);
            } else {
                fprintf(stderr, "[MIXED-DBG] mask_lp is NULL\n"); fflush(stderr);
            }
        }
        }
    }

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
        (const char *) k_lp->data, (const char *) v_lp->data,
        mask_lp ? (const char *) mask_lp->data : nullptr, mask_lp ? mask_lp->nb[1] : 0, mask_lp ? mask_lp->nb[2] : 0, mask_lp ? mask_lp->nb[3] : 0, mask_lp ? mask_lp->ne[2] : 0, mask_lp ? mask_lp->ne[3] : 0,
        (const char *) k_hp->data, (const char *) v_hp->data,
        mask_hp ? (const char *) mask_hp->data : nullptr, mask_hp ? mask_hp->nb[1] : 0, mask_hp ? mask_hp->nb[2] : 0, mask_hp ? mask_hp->nb[3] : 0, mask_hp ? mask_hp->ne[2] : 0, mask_hp ? mask_hp->ne[3] : 0,
        (float *) dst->data,
        nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, nbkh1, nbkh2, nbkh3, nbvh1, nbvh2, nbvh3,
        n_kv, n_hp, DK, DV, N, n_head, nseq, scale, max_bias, logit_softcap,
        rk2, rk3, rv2, rv3, rk2h, rk3h, rv2h, rv3h);

    GGML_ASSERT(cudaGetLastError() == cudaSuccess);
}
