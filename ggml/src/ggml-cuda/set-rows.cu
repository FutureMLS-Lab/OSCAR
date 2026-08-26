#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include <cstdlib>

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


// OSCAR q2_0 (INT2) SET_ROWS write kernel. Quantizes an f32 K/V row into q2_0 using
// the calibrated OSCAR encode: per-128-group mean, optional outlier clip (clip_ratio
// percentile), per-32-block sigma, and 2-bit Lloyd-Max packing (vs < -0.6745s -> 0,
// < 0 -> 1, < 0.6745s -> 2, else 3). The group mean is replicated into every block's
// m field so the per-block dequant (m + d*centroid) is exact. Requires ne00 % 128 == 0
// and the LLAMA_KV_NO_HADAMARD path (no in-quant OWHT). Faithful port of Metal's
// kernel_set_rows_q2_0.
template <typename idx_t>
static __global__ void set_rows_cuda_q2_0(
        const char * src0, const char * src1, char * dst,
        const int32_t ne01, const int32_t ne11, const int32_t ne12,
        const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb10, const uint64_t nb11, const uint64_t nb12,
        const uint64_t nb1,  const uint64_t nb2,  const uint64_t nb3,
        const int32_t nk0, const float clip_ratio) {

    const int32_t i03 = blockIdx.z;
    const int32_t i02 = blockIdx.y;
    const int32_t i01 = blockIdx.x;
    if (i01 >= ne01) return;

    const int32_t i12 = i03 % ne12;
    const int32_t i11 = i02 % ne11;
    const int32_t i10 = i01;

    const idx_t i1 = *((const idx_t *) (src1 + i10*nb10 + i11*nb11 + i12*nb12));

    block_q2_0 * dst_row = (block_q2_0 *) (dst + (uint64_t)i1*nb1 + i02*nb2 + i03*nb3);
    const float * src_row = (const float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    const int ngrp = nk0 / 4;  // 128-wide groups per row (4 blocks of 32)

    __shared__ float sh[128];
    __shared__ float sh_sigma[4];
    __shared__ float sh_mean;
    __shared__ float sh_thr;

    const unsigned t = threadIdx.x;  // 0..31

    for (int g = 0; g < ngrp; ++g) {
        const float * gsrc = src_row + g * 128;

        for (int k = 0; k < 4; ++k) {
            sh[t * 4 + k] = gsrc[t * 4 + k];
        }
        __syncthreads();

        // group mean
        if (t == 0) {
            float s = 0.0f;
            for (int j = 0; j < 128; ++j) s += sh[j];
            sh_mean = s / 128.0f;
        }
        __syncthreads();

        const float mean = sh_mean;
        for (int k = 0; k < 4; ++k) sh[t * 4 + k] -= mean;
        __syncthreads();

        // OSCAR outlier clip: threshold = clip_ratio percentile over the 128 group,
        // found by exact rank counting (matches CPU qsort + index selection).
        if (clip_ratio > 0.0f && clip_ratio < 1.0f) {
            if (t == 0) sh_thr = 0.0f;
            __syncthreads();

            int idx = (int)(clip_ratio * 128.0f);
            if (idx >= 128) idx = 127;
            for (int k = 0; k < 4; ++k) {
                const float a = fabsf(sh[t * 4 + k]);
                int lo = 0, le = 0;
                for (int j = 0; j < 128; ++j) {
                    const float aj = fabsf(sh[j]);
                    lo += (aj <  a) ? 1 : 0;
                    le += (aj <= a) ? 1 : 0;
                }
                if (lo <= idx && idx < le) sh_thr = a;
            }
            __syncthreads();

            const float thr = sh_thr;
            for (int k = 0; k < 4; ++k) {
                float v = sh[t * 4 + k];
                if (v >  thr) v =  thr;
                if (v < -thr) v = -thr;
                sh[t * 4 + k] = v;
            }
            __syncthreads();
        }

        // per-32-block sigma (RMS)
        if (t < 4) {
            float ss = 0.0f;
            for (int j = 0; j < 32; ++j) ss += sh[t * 32 + j] * sh[t * 32 + j];
            sh_sigma[t] = sqrtf(ss / 32.0f);
        }
        __syncthreads();

        // quantize: thread t writes one packed byte (block b = t/8, byte t%8)
        const int   b         = t / 8;
        const float sigma     = sh_sigma[b];
        const float inv_sigma = (sigma > 1e-8f) ? (1.0f / sigma) : 0.0f;

        uint8_t packed = 0;
        for (int k = 0; k < 4; ++k) {
            const float vs = sh[t * 4 + k] * inv_sigma;
            uint8_t code;
            if      (vs < -0.6745f) code = 0;
            else if (vs <  0.0f)    code = 1;
            else if (vs <  0.6745f)  code = 2;
            else                     code = 3;
            packed |= code << (2 * k);
        }

        block_q2_0 & blk = dst_row[g * 4 + b];
        blk.qs[t % 8] = packed;
        if (t % 8 == 0) {
            blk.d = __float2half(sigma);
            blk.m = __float2half(mean);
        }
        __syncthreads();
    }
}

    void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

        if (dst->type == GGML_TYPE_Q2_0) {
            // OSCAR INT2 KV write: dedicated kernel (per-128-group mean + clip + per-block
            // Lloyd-Max). clip_ratio comes from LLAMA_KV_CLIP_RATIO (0 disables).
            float clip_ratio = 0.0f;
            if (const char * e = getenv("LLAMA_KV_CLIP_RATIO")) {
                clip_ratio = (float) atof(e);
            }

            GGML_TENSOR_BINARY_OP_LOCALS(src0, src1, dst);

            const int32_t nk0 = (int32_t)(ne00 / ggml_blck_size(GGML_TYPE_Q2_0));
            cudaStream_t stream = ctx.stream();

            const dim3 grid_size(ne01, ne02, ne03);
            const dim3 block_size(32, 1, 1);

            const char * src0_d = (const char *) src0->data;
            const char * src1_d = (const char *) src1->data;
            char       * dst_d  = (char *)       dst->data;

            if (src1->type == GGML_TYPE_I64) {
                set_rows_cuda_q2_0<int64_t><<<grid_size, block_size, 0, stream>>>(
                    src0_d, src1_d, dst_d,
                    ne01, ne11, ne12,
                    nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3,
                    nk0, clip_ratio);
            } else {
                set_rows_cuda_q2_0<int32_t><<<grid_size, block_size, 0, stream>>>(
                    src0_d, src1_d, dst_d,
                    ne01, ne11, ne12,
                    nb01, nb02, nb03, nb10, nb11, nb12, nb1, nb2, nb3,
                    nk0, clip_ratio);
            }
            GGML_ASSERT(cudaGetLastError() == cudaSuccess);
            return;
        }

        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    }
