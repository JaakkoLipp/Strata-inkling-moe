// src/inkling/dot_kernels.cpp - dot products the vendored ggml-cpu computes with its scalar fallback, here with SIMD.
//
// Q2_0 (64 2-bit weights + one f16 scale per block): Unsloth's Inkling GGUFs keep many down-projection experts in it
// (23 of 48 expert tensors in UD-IQ1_S). The vendored ggml (third_party/strata-ggml, pinned before llama.cpp added
// SIMD Q2_0 kernels) runs it one weight at a time, which made the prompt several times slower than it should be.
//
// Like llama.cpp's later AVX2 / VNNI kernels, these keep the integer sums of each 32-weight half exact and accumulate
// the scaled halves in float vector lanes, summed once at the end. That is a different float order from ggml's scalar
// loop, so the results agree to rounding (tests/store_test.cpp holds them to a double-precision reference).
#include "dot_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define INK_X86 1
#endif

namespace strata::inkling {

namespace {

struct BlockQ2_0 {
    uint16_t d;
    uint8_t qs[16];
};
struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ2_0) == 18, "block_q2_0 layout");
static_assert(sizeof(BlockQ8_0) == 34, "block_q8_0 layout");

#if defined(INK_X86) && (defined(__GNUC__) || defined(__clang__))
#define INK_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))
#define INK_VNNI_TARGET __attribute__((target("avx2,fma,f16c,avxvnni")))
#define INK_HAVE_AVX2_KERNEL 1
#define INK_HAVE_VNNI_KERNEL 1
#elif defined(INK_X86) && defined(__AVX2__)
#define INK_AVX2_TARGET
#define INK_HAVE_AVX2_KERNEL 1
#endif

#if defined(INK_HAVE_AVX2_KERNEL)
// the 2-bit codes of one 32-weight half (8 bytes) as 32 unsigned bytes, in the order of ydeint() below:
// the code of weight 4b+s is (qs[b] >> 2s) & 3; dword s of each 128-bit lane holds the four b of that lane
INK_AVX2_TARGET static inline __m256i q2_codes(const uint8_t* qs) {
    uint32_t a, b;
    std::memcpy(&a, qs, 4);
    std::memcpy(&b, qs + 4, 4);
    __m256i q = _mm256_setr_epi32((int)a, (int)a, (int)a, (int)a, (int)b, (int)b, (int)b, (int)b);
    q = _mm256_srlv_epi32(q, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
    return _mm256_and_si256(q, _mm256_set1_epi8(3));
}

// the 32 activations of a Q8_0 block in the same order
INK_AVX2_TARGET static inline __m256i ydeint(const int8_t* y) {
    const __m256i perm = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                          0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    return _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(y)), perm);
}

INK_AVX2_TARGET static inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// codes mean q - 1: sum((q - 1) y) = sum(q y) - sum(y), both from one maddubs each, exact in int16 / int32
INK_AVX2_TARGET void dot_q2_0_q8_0_avx2(int n, float* s, size_t, const void* vx, size_t, const void* vy, size_t, int) {
    const auto* x = static_cast<const BlockQ2_0*>(vx);
    const auto* y = static_cast<const BlockQ8_0*>(vy);
    const int nb = n / 64;
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d0 = _cvtsh_ss(x[i].d);
        for (int k = 0; k < 2; ++k) {
            const BlockQ8_0& yb = y[i * 2 + k];
            const __m256i q = q2_codes(&x[i].qs[k * 8]);
            const __m256i yv = ydeint(yb.qs);
            const __m256i p16 = _mm256_sub_epi16(_mm256_maddubs_epi16(q, yv), _mm256_maddubs_epi16(ones8, yv));
            const __m256 p = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones16));
            const __m256 d = _mm256_set1_ps(d0 * _cvtsh_ss(yb.d));
            if (k == 0) acc0 = _mm256_fmadd_ps(p, d, acc0);
            else acc1 = _mm256_fmadd_ps(p, d, acc1);
        }
    }
    *s = hsum_ps(_mm256_add_ps(acc0, acc1));
}

bool cpu_has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
    return true;   // MSVC: compiled only when the whole build targets AVX2
#endif
}
#endif

#if defined(INK_HAVE_VNNI_KERNEL)
// AVX-VNNI (Intel Alder Lake and newer, AMD Zen 5): u8 x s8 straight into int32
INK_VNNI_TARGET void dot_q2_0_q8_0_vnni(int n, float* s, size_t, const void* vx, size_t, const void* vy, size_t, int) {
    const auto* x = static_cast<const BlockQ2_0*>(vx);
    const auto* y = static_cast<const BlockQ8_0*>(vy);
    const int nb = n / 64;
    const __m256i ones8 = _mm256_set1_epi8(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d0 = _cvtsh_ss(x[i].d);
        for (int k = 0; k < 2; ++k) {
            const BlockQ8_0& yb = y[i * 2 + k];
            const __m256i q = q2_codes(&x[i].qs[k * 8]);
            const __m256i yv = ydeint(yb.qs);
            const __m256i qy = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), q, yv);
            const __m256i sy = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ones8, yv);
            const __m256 p = _mm256_cvtepi32_ps(_mm256_sub_epi32(qy, sy));
            const __m256 d = _mm256_set1_ps(d0 * _cvtsh_ss(yb.d));
            if (k == 0) acc0 = _mm256_fmadd_ps(p, d, acc0);
            else acc1 = _mm256_fmadd_ps(p, d, acc1);
        }
    }
    *s = hsum_ps(_mm256_add_ps(acc0, acc1));
}

bool cpu_has_avxvnni() { return cpu_has_avx2() && __builtin_cpu_supports("avxvnni"); }
#endif

}  // namespace

const char* q2_0_kernel_name() {
#if defined(INK_HAVE_VNNI_KERNEL)
    if (cpu_has_avxvnni()) return "avx-vnni";
#endif
#if defined(INK_HAVE_AVX2_KERNEL)
    if (cpu_has_avx2()) return "avx2";
#endif
    return "ggml scalar";
}

ggml_vec_dot_t vec_dot_fn(int wtype) {
    if (wtype == GGML_TYPE_Q2_0) {
#if defined(INK_HAVE_VNNI_KERNEL)
        static const bool vnni = cpu_has_avxvnni();
        if (vnni) return dot_q2_0_q8_0_vnni;
#endif
#if defined(INK_HAVE_AVX2_KERNEL)
        static const bool avx2 = cpu_has_avx2();
        if (avx2) return dot_q2_0_q8_0_avx2;
#endif
    }
    const auto* tr = ggml_get_type_traits_cpu((ggml_type)wtype);
    return tr ? tr->vec_dot : nullptr;
}

ggml_vec_dot_t q2_0_kernel(int which) {
#if defined(INK_HAVE_VNNI_KERNEL)
    if (which == 2) return cpu_has_avxvnni() ? dot_q2_0_q8_0_vnni : nullptr;
#endif
#if defined(INK_HAVE_AVX2_KERNEL)
    if (which == 1) return cpu_has_avx2() ? dot_q2_0_q8_0_avx2 : nullptr;
#endif
    if (which == 0) return ggml_get_type_traits_cpu(GGML_TYPE_Q2_0)->vec_dot;
    return nullptr;
}

}  // namespace strata::inkling
