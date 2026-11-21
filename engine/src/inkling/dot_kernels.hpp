// src/inkling/dot_kernels.hpp - the dot product the engine uses for a weight type: ggml-cpu's, or this engine's SIMD
// kernel where the vendored ggml has only a scalar one (Q2_0 on x86: AVX-VNNI or AVX2). Agrees with ggml's to rounding.
#pragma once

#include "ggml-cpu.h"

namespace strata::inkling {

ggml_vec_dot_t vec_dot_fn(int wtype);
/// The Q2_0 kernel the engine picks on this CPU ("avx-vnni", "avx2" or "ggml scalar"), for the log.
const char* q2_0_kernel_name();
/// A specific Q2_0 kernel for the tests: 0 ggml's, 1 AVX2, 2 AVX-VNNI; nullptr where this CPU or build has none.
ggml_vec_dot_t q2_0_kernel(int which);

}  // namespace strata::inkling
