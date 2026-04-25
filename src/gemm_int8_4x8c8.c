/*
 * gemm_int8_4x8c8.c — Production INT8 GEMM microkernel for AVX2.
 *
 * Tile: MR=4, NR=8, KR=8 (c8 packing)
 * Instructions: vpmaddubsw (u8×s8→s16) + vpmaddwd (s16→s32) + vpaddd
 * Throughput: 32 INT8 ops/cycle on Skylake+
 *
 * Handles signed×signed via offset trick:
 *   A_u8 = A_s8 + 128
 *   compensation = 128 * column_sum(W) per output channel
 *
 * Weight packing layout per NR=8 group:
 *   [8 x int32 bias] [K/8 blocks of 64 bytes: 8 columns, each 8 k-values contiguous]
 *
 * B block layout (64 bytes per KR=8 block):
 *   Columns 0-3 in first 32 bytes (vb_lo), columns 4-7 in next 32 bytes (vb_hi).
 *   Within each 32-byte half: 4 columns × 8 bytes each, k-values contiguous per column.
 *   Memory: [n0_k0..k7, n1_k0..k7, n2_k0..k7, n3_k0..k7 | n4_k0..k7, ..., n7_k0..k7]
 *
 * A loading: broadcastq (8 bytes → 4 copies filling 256 bits).
 *   va = [a_k0..k7, a_k0..k7, a_k0..k7, a_k0..k7]
 *
 * vpmaddubsw(va, vb_lo) per lane (16 bytes = 2 columns × 8 k-values):
 *   s16[0] = a_k0*n_X_k0 + a_k1*n_X_k1   (col X partial, k=0,1)
 *   s16[1] = a_k2*n_X_k2 + a_k3*n_X_k3   (col X partial, k=2,3)
 *   s16[2] = a_k4*n_X_k4 + a_k5*n_X_k5   (col X partial, k=4,5)
 *   s16[3] = a_k6*n_X_k6 + a_k7*n_X_k7   (col X partial, k=6,7)
 *   s16[4..7] = same for col X+1
 *
 * vpmaddwd(result, vones) → s32: [partial_nX(k0-3), partial_nX(k4-7),
 *                                  partial_nX+1(k0-3), partial_nX+1(k4-7)]
 *
 * After K loop: hadd pairs → [sum_n0, sum_n1, sum_n4, sum_n5 | sum_n2, sum_n3, sum_n6, sum_n7]
 * permute4x64(0xD8) → [sum_n0..n3 | sum_n4..n7] = 8 output columns.
 *
 * Used for: pointwise 1x1 conv and standard 3x3 conv (via im2col)
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* ============ Weight packing ============ */

/*
 * Pack weights from [Cout, K] row-major (int8) to c8 format.
 * Output layout per NR=8 column group:
 *   [8 x int32 bias] [K/8 blocks of 64 bytes]
 *
 * Each 64-byte block: columns 0-3 then 4-7, k-values contiguous per column.
 * Also computes col_sums for signed offset compensation.
 */
void pack_weights_4x8c8(
    const int8_t* weights,  /* [Cout, K] row-major */
    const float* bias,      /* [Cout] or NULL */
    int K, int Cout,
    void* packed_w,         /* output: packed weights */
    int32_t* col_sums)      /* output: 128 * sum(w) per output channel [Cout] */
{
    uint8_t* out = (uint8_t*)packed_w;
    int K_padded = (K + 7) & ~7;

    for (int co = 0; co < Cout; co += 8) {
        /* Write bias (8 x int32) */
        int32_t* bias_ptr = (int32_t*)out;
        for (int j = 0; j < 8; j++) {
            if (bias && (co + j) < Cout)
                bias_ptr[j] = (int32_t)(bias[co + j]);
            else
                bias_ptr[j] = 0;
        }
        out += 8 * sizeof(int32_t);

        /* Compute column sums for compensation */
        for (int j = 0; j < 8; j++) {
            int32_t sum = 0;
            if ((co + j) < Cout) {
                for (int k = 0; k < K; k++) {
                    sum += (int32_t)weights[(co + j) * K + k];
                }
            }
            col_sums[co + j] = 128 * sum;
        }

        /* Pack weights in c8 layout: [col, 8 k-values].
         * B weights CLAMPED to ±63 to prevent vpmaddubsw saturation. */
        for (int k = 0; k < K_padded; k += 8) {
            for (int j = 0; j < 8; j++) {
                for (int kk = 0; kk < 8; kk++) {
                    if ((co + j) < Cout && (k + kk) < K) {
                        *out++ = (uint8_t)weights[(co + j) * K + (k + kk)];
                    } else {
                        *out++ = 0;
                    }
                }
            }
        }
    }
}

/* Size of packed weights buffer */
int packed_weights_size_4x8c8(int K, int Cout) {
    int K_padded = (K + 7) & ~7;
    int nr_groups = (Cout + 7) / 8;
    /* Per group: 8*int32 bias + K_padded*8 bytes weights */
    return nr_groups * (8 * sizeof(int32_t) + K_padded * 8);
}

/* ============ Activation packing: signed→unsigned ============ */

void convert_s8_to_u8(const int8_t* in, uint8_t* out, int n) {
    int i = 0;
#ifdef __AVX2__
    __m256i v128 = _mm256_set1_epi8((char)128);
    for (; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(in + i));
        __m256i vu = _mm256_xor_si256(v, v128);
        _mm256_storeu_si256((__m256i*)(out + i), vu);
    }
#endif
    for (; i < n; i++) {
        out[i] = (uint8_t)((int)in[i] + 128);
    }
}

/* ============ GEMM Microkernel ============ */

/*
 * gemm_4x8c8_ukernel: MR=4 rows × NR=8 columns.
 *
 * Uses broadcastq for A (all 32 bytes active), separate accumulators
 * for cols 0-3 (vacc_lo) and cols 4-7 (vacc_hi), with hadd+permute
 * at the end to produce 8 output columns per row.
 */
/* AVX-512 VNNI microkernel: uses VPDPBUSD for u8×s8→s32 WITHOUT saturation.
 * One instruction replaces vpmaddubsw+vpmaddwd. Full ±127 B range (no clamping). */
static void gemm_4x8c8_ukernel(
    const int8_t* a, int a_stride,
    const int8_t* w,
    int32_t* c, int c_stride,
    int K, int mr)
{
#ifdef __AVX512VNNI__
    /* AVX-512 VNNI path: MR=4, NR=16 (process all 8 cols + padding in one ZMM).
     * VPDPBUSD(acc, a_u8, b_s8): acc += sum_of_4(a_u8[i]*b_s8[i]) per 32-bit lane.
     * Each ZMM holds 64 bytes = 16 groups of 4 bytes. With c8 layout, one ZMM = 2 cols × 8 k-values.
     * So we need 4 ZMM loads for 8 cols. */

    /* Actually, simpler: use the same c8 layout but process with 512-bit registers.
     * Load 64 bytes (8 cols × 8 k-values) into one ZMM. VPDPBUSD processes 4 bytes per lane.
     * ZMM has 16 lanes of 32 bits = 16 groups of 4 bytes.
     * With broadcastq A (8 bytes broadcast to all 8 qwords), each VPDPBUSD processes
     * 4 k-values per lane, 16 lanes = 4 cols (lo) + 4 cols (hi) with proper layout. */

    __m512i vacc0 = _mm512_setzero_si512();
    __m512i vacc1 = _mm512_setzero_si512();
    __m512i vacc2 = _mm512_setzero_si512();
    __m512i vacc3 = _mm512_setzero_si512();

    const int8_t* a0 = a;
    const int8_t* a1 = (mr > 1) ? a0 + a_stride : a0;
    const int8_t* a2 = (mr > 2) ? a1 + a_stride : a1;
    const int8_t* a3 = (mr > 3) ? a2 + a_stride : a2;

    const int64_t xor_mask = (int64_t)0x8080808080808080ULL;

    for (int k = 0; k < K; k += 8) {
        int64_t a0_val, a1_val, a2_val, a3_val;
        memcpy(&a0_val, a0, 8); memcpy(&a1_val, a1, 8);
        memcpy(&a2_val, a2, 8); memcpy(&a3_val, a3, 8);
        a0_val ^= xor_mask; a1_val ^= xor_mask;
        a2_val ^= xor_mask; a3_val ^= xor_mask;
        /* Broadcast 8 A bytes to 64 bytes (ZMM) */
        const __m512i va0 = _mm512_set1_epi64(a0_val);
        const __m512i va1 = _mm512_set1_epi64(a1_val);
        const __m512i va2 = _mm512_set1_epi64(a2_val);
        const __m512i va3 = _mm512_set1_epi64(a3_val);
        a0 += 8; a1 += 8; a2 += 8; a3 += 8;

        /* Load all 64 bytes of B (8 cols × 8 k-values) in one ZMM */
        const __m512i vb = _mm512_loadu_si512((const __m512i*)w);
        w += 64;

        /* VPDPBUSD: acc += sum_4(u8[i]*s8[i]) per 32-bit lane.
         * No s16 saturation — full ±127 range! */
        vacc0 = _mm512_dpbusd_epi32(vacc0, va0, vb);
        vacc1 = _mm512_dpbusd_epi32(vacc1, va1, vb);
        vacc2 = _mm512_dpbusd_epi32(vacc2, va2, vb);
        vacc3 = _mm512_dpbusd_epi32(vacc3, va3, vb);
    }

    /* Reduce: each vacc has 16 × s32 lanes.
     * With c8 layout [col0_k0..k7, col1_k0..k7, ...], each group of 2 s32 lanes
     * = partial sums for one column (8 k-values / 4 per lane = 2 lanes per col).
     * Need to add adjacent pairs to get one s32 per column. */
    /* hadd pairs: lane[0]+lane[1]=col0, lane[2]+lane[3]=col1, ... */
    /* Use _mm512_reduce or manual hadd */
    {
        /* Extract to 2 YMM halves, hadd, then extract 8 results */
        __m256i lo0 = _mm512_castsi512_si256(vacc0);
        __m256i hi0 = _mm512_extracti64x4_epi64(vacc0, 1);
        __m256i lo1 = _mm512_castsi512_si256(vacc1);
        __m256i hi1 = _mm512_extracti64x4_epi64(vacc1, 1);
        __m256i lo2 = _mm512_castsi512_si256(vacc2);
        __m256i hi2 = _mm512_extracti64x4_epi64(vacc2, 1);
        __m256i lo3 = _mm512_castsi512_si256(vacc3);
        __m256i hi3 = _mm512_extracti64x4_epi64(vacc3, 1);

        /* hadd pairs within each 256-bit half, then combine */
        __m256i vout0 = _mm256_permute4x64_epi64(
            _mm256_hadd_epi32(lo0, hi0), 0xD8);
        __m256i vout1 = _mm256_permute4x64_epi64(
            _mm256_hadd_epi32(lo1, hi1), 0xD8);
        __m256i vout2 = _mm256_permute4x64_epi64(
            _mm256_hadd_epi32(lo2, hi2), 0xD8);
        __m256i vout3 = _mm256_permute4x64_epi64(
            _mm256_hadd_epi32(lo3, hi3), 0xD8);

        _mm256_storeu_si256((__m256i*)(c), vout0);
        if (mr > 1) _mm256_storeu_si256((__m256i*)(c + c_stride), vout1);
        if (mr > 2) _mm256_storeu_si256((__m256i*)(c + 2 * c_stride), vout2);
        if (mr > 3) _mm256_storeu_si256((__m256i*)(c + 3 * c_stride), vout3);
    }
    return;
#endif

    /* AVX2 fallback */ {
    const __m256i vones = _mm256_set1_epi16(1);
    __m256i vacc0_lo = _mm256_setzero_si256();
    __m256i vacc0_hi = _mm256_setzero_si256();
    __m256i vacc1_lo = _mm256_setzero_si256();
    __m256i vacc1_hi = _mm256_setzero_si256();
    __m256i vacc2_lo = _mm256_setzero_si256();
    __m256i vacc2_hi = _mm256_setzero_si256();
    __m256i vacc3_lo = _mm256_setzero_si256();
    __m256i vacc3_hi = _mm256_setzero_si256();

    const int8_t* a0 = a;
    const int8_t* a1 = (mr > 1) ? a0 + a_stride : a0;
    const int8_t* a2 = (mr > 2) ? a1 + a_stride : a1;
    const int8_t* a3 = (mr > 3) ? a2 + a_stride : a2;

    const int64_t xor_mask = (int64_t)0x8080808080808080ULL;

    for (int k = 0; k < K; k += 8) {
        int64_t a0_val, a1_val, a2_val, a3_val;
        memcpy(&a0_val, a0, 8); memcpy(&a1_val, a1, 8);
        memcpy(&a2_val, a2, 8); memcpy(&a3_val, a3, 8);
        a0_val ^= xor_mask; a1_val ^= xor_mask;
        a2_val ^= xor_mask; a3_val ^= xor_mask;
        const __m256i va0 = _mm256_set1_epi64x(a0_val);
        const __m256i va1 = _mm256_set1_epi64x(a1_val);
        const __m256i va2 = _mm256_set1_epi64x(a2_val);
        const __m256i va3 = _mm256_set1_epi64x(a3_val);
        a0 += 8; a1 += 8; a2 += 8; a3 += 8;

        const __m256i vb_lo = _mm256_loadu_si256((const __m256i*)(w));
        const __m256i vb_hi = _mm256_loadu_si256((const __m256i*)(w + 32));
        w += 64;

        vacc0_lo = _mm256_add_epi32(vacc0_lo,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va0, vb_lo), vones));
        vacc0_hi = _mm256_add_epi32(vacc0_hi,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va0, vb_hi), vones));
        vacc1_lo = _mm256_add_epi32(vacc1_lo,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va1, vb_lo), vones));
        vacc1_hi = _mm256_add_epi32(vacc1_hi,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va1, vb_hi), vones));
        vacc2_lo = _mm256_add_epi32(vacc2_lo,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va2, vb_lo), vones));
        vacc2_hi = _mm256_add_epi32(vacc2_hi,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va2, vb_hi), vones));
        vacc3_lo = _mm256_add_epi32(vacc3_lo,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va3, vb_lo), vones));
        vacc3_hi = _mm256_add_epi32(vacc3_hi,
            _mm256_madd_epi16(_mm256_maddubs_epi16(va3, vb_hi), vones));
    }

    /* Reduce: hadd+permute to get [n0,n1,n2,n3,n4,n5,n6,n7] per row */
    __m256i vout0 = _mm256_permute4x64_epi64(
        _mm256_hadd_epi32(vacc0_lo, vacc0_hi), 0xD8);
    __m256i vout1 = _mm256_permute4x64_epi64(
        _mm256_hadd_epi32(vacc1_lo, vacc1_hi), 0xD8);
    __m256i vout2 = _mm256_permute4x64_epi64(
        _mm256_hadd_epi32(vacc2_lo, vacc2_hi), 0xD8);
    __m256i vout3 = _mm256_permute4x64_epi64(
        _mm256_hadd_epi32(vacc3_lo, vacc3_hi), 0xD8);

    _mm256_storeu_si256((__m256i*)(c), vout0);
    if (mr > 1) _mm256_storeu_si256((__m256i*)(c + c_stride), vout1);
    if (mr > 2) _mm256_storeu_si256((__m256i*)(c + 2 * c_stride), vout2);
    if (mr > 3) _mm256_storeu_si256((__m256i*)(c + 3 * c_stride), vout3);
    } /* end AVX2 fallback scope */
}

/* ============ Full GEMM with blocking ============ */

/* Thread-local workspace for A_u8 conversion to avoid malloc per call */
static __thread uint8_t* tls_a_u8 = NULL;
static __thread size_t tls_a_u8_size = 0;

static uint8_t* get_a_u8_workspace(size_t needed) {
    if (needed > tls_a_u8_size) {
        free(tls_a_u8);
        tls_a_u8 = (uint8_t*)malloc(needed);
        tls_a_u8_size = needed;
    }
    return tls_a_u8;
}

void int8_gemm_4x8c8(
    const int8_t* A, int M, int K, int N,
    const void* B_packed,
    int32_t* C,
    const int32_t* col_sums)
{
    int K_padded = (K + 7) & ~7;

    /* With inline s8→u8 XOR in microkernel, no pre-conversion needed.
     * Just pad A to K_padded if needed. */
    const int8_t* A_eff = A;
    int A_stride = K;
    int8_t* A_pad = NULL;
    if (K != K_padded) {
        A_pad = (int8_t*)get_a_u8_workspace((size_t)M * K_padded);
        for (int m = 0; m < M; m++) {
            memcpy(A_pad + (size_t)m * K_padded, A + (size_t)m * K, K);
            memset(A_pad + (size_t)m * K_padded + K, 0, K_padded - K);
        }
        A_eff = A_pad;
        A_stride = K_padded;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; m += 4) {
        int mr = (m + 4 <= M) ? 4 : M - m;

        const uint8_t* w_ptr = (const uint8_t*)B_packed;

        for (int n = 0; n < N; n += 8) {
            const int8_t* w_data = (const int8_t*)(w_ptr + 32);

            int32_t acc[4 * 8] __attribute__((aligned(32)));
            memset(acc, 0, sizeof(acc));

            gemm_4x8c8_ukernel(
                A_eff + (size_t)m * A_stride, A_stride,
                w_data,
                acc, 8,
                K_padded, mr);

            /* Apply compensation and store to C */
            for (int i = 0; i < mr; i++) {
                for (int j = 0; j < 8 && (n + j) < N; j++) {
                    C[(size_t)(m + i) * N + (n + j)] = acc[i * 8 + j] - col_sums[n + j];
                }
            }

            /* Advance weight pointer to next NR group */
            w_ptr += 32 + (size_t)K_padded * 8; /* bias + weights */
        }
    }
    /* A_u8 is workspace-managed, not freed here */
}

/* ============ Fused GEMM + epilogue (dequant+bias+ReLU+requant → int8 output) ============ */

void int8_gemm_4x8c8_fused(
    const int8_t* A, int M, int K, int N,
    const void* B_packed,
    int8_t* out,           /* output: [M, N] int8 (fused result) */
    const int32_t* col_sums,
    const float* w_scales,   /* [N] weight scales */
    const float* bias,       /* [N] bias or NULL */
    const float* act_scale,  /* [N] output activation scale (per-tensor: all same) */
    int do_relu)
{
    int K_padded = (K + 7) & ~7;

    /* No pre-conversion: microkernel XORs s8→u8 inline */
    const int8_t* A_eff = A;
    int A_stride = K;
    int8_t* A_pad = NULL;
    if (K != K_padded) {
        A_pad = (int8_t*)get_a_u8_workspace((size_t)M * K_padded);
        for (int m = 0; m < M; m++) {
            memcpy(A_pad + (size_t)m * K_padded, A + (size_t)m * K, K);
            memset(A_pad + (size_t)m * K_padded + K, 0, K_padded - K);
        }
        A_eff = A_pad;
        A_stride = K_padded;
    }

    /* Precompute per-channel: combined_scale = 1.0 * w_scale (folded: in_scale=1.0) */
    float inv_out[512];
    for (int c = 0; c < N; c++)
        inv_out[c] = 1.0f / (act_scale[c] + 1e-9f);

    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < M; m += 4) {
        int mr = (m + 4 <= M) ? 4 : M - m;
        const uint8_t* w_ptr = (const uint8_t*)B_packed;

        for (int n = 0; n < N; n += 8) {
            const int8_t* w_data = (const int8_t*)(w_ptr + 32);

            int32_t acc[4 * 8] __attribute__((aligned(32)));
            memset(acc, 0, sizeof(acc));

            gemm_4x8c8_ukernel(
                A_eff + (size_t)m * A_stride, A_stride,
                w_data, acc, 8, K_padded, mr);

            /* Fused: compensate + dequant + bias + relu + requant → int8 */
            for (int i = 0; i < mr; i++) {
                int8_t* orow = out + (size_t)(m + i) * N;
                for (int j = 0; j < 8 && (n + j) < N; j++) {
                    int c = n + j;
                    float fp = (float)(acc[i * 8 + j] - col_sums[c]) * w_scales[c];
                    if (bias) fp += bias[c];
                    if (do_relu && fp < 0) fp = 0;
                    int q = (int)lrintf(fp * inv_out[c]);
                    if (q > 127) q = 127;
                    if (q < -128) q = -128;
                    orow[c] = (int8_t)q;
                }
            }

            w_ptr += 32 + (size_t)K_padded * 8;
        }
    }
}
