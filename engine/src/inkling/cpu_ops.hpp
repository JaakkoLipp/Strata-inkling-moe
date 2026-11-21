// src/inkling/cpu_ops.hpp - the CPU arithmetic of the Inkling engine: a thread pool, matrix x activations through
// ggml-cpu's per-type dot products (every quant the GGUFs use: IQ1_S..IQ4_XS, Q2_0, Q*_K, Q8_0, NVFP4, MXFP4, BF16,
// F16, F32), and the small vector operations of the model.
//
// The activations are quantized once per matrix type to the type its dot product wants (Q8_K for the K and IQ quants,
// Q8_0 for the 32/64-blocks, BF16/F16/F32 for the float types), exactly as ggml-cpu's mul_mat does, so the results
// match llama.cpp's CPU backend to the rounding of the summation order.
#pragma once

#include "model.hpp"

#include "ggml-cpu.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::inkling {

/// A fixed pool that runs one parallel loop at a time; the calling thread works too. Workers spin briefly between
/// loops (a decode step runs hundreds of small loops) and then sleep.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return n_; }
    /// Calls fn(begin, end) over [0, n) in pieces of `grain`, on all threads; returns when all are done.
    void run(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn);

private:
    void worker();
    void drain();
    int n_ = 1;
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    std::atomic<uint64_t> epoch_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    std::atomic<int> sleepers_{0};
    const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
    int64_t total_ = 0, grain_ = 1;
    std::atomic<int64_t> next_{0};
};

/// Activations of `ntok` tokens, `n` values each, in the form a dot product of type `vtype` reads.
struct QAct {
    int vtype = 0;
    int64_t n = 0;
    int ntok = 0;
    size_t row_bytes = 0;
    std::vector<uint8_t> buf;
    const void* row(int t) const { return buf.data() + (size_t)t * row_bytes; }
};

/// The dot-product input type for weights of ggml type `wtype`.
int vec_dot_type(int wtype);
/// Quantizes x [ntok][n] (row stride `ldx` floats) for weights of type `wtype`.
void quantize_acts(const float* x, int ntok, int64_t n, int64_t ldx, int wtype, QAct& q, ThreadPool* pool = nullptr);

/// Dot products of rows [r0, r1) of W with `na` activation rows, in 16 x 16 tiles so that both the weight rows and
/// the activations of a tile stay in L1 (ggml-cpu's mul_mat blocks the same way). act(a) gives activation row a,
/// out(a, r, value) takes each result.
template <class Act, class Out>
inline void tiled_dots(ggml_vec_dot_t vd, int n, const Mat& W, int64_t r0, int64_t r1, int na, Act act, Out out) {
    constexpr int64_t TR = 16;
    constexpr int TA = 16;
    for (int64_t rb = r0; rb < r1; rb += TR) {
        const int64_t re = rb + TR < r1 ? rb + TR : r1;
        for (int ab = 0; ab < na; ab += TA) {
            const int ae = ab + TA < na ? ab + TA : na;
            for (int64_t r = rb; r < re; ++r) {
                const uint8_t* w = W.row(r);
                for (int a = ab; a < ae; ++a) {
                    float s;
                    vd(n, &s, 0, w, 0, act(a), 0, 1);
                    out(a, r, s);
                }
            }
        }
    }
}

/// y[t][r] = W.row(r) . x[t] for rows [r0, r1); y has row stride `ldy`. No threading: callers split rows.
void dot_rows(const Mat& W, const QAct& xq, int64_t r0, int64_t r1, float* y, int64_t ldy);
/// The same for every row, split over the pool.
void matmul(ThreadPool& pool, const Mat& W, const QAct& xq, float* y, int64_t ldy);
/// Quantizes x and multiplies: y [ntok][W.rows].
void matmul_f(ThreadPool& pool, const Mat& W, const float* x, int ntok, int64_t ldx, float* y, int64_t ldy, QAct& scratch);

// the model's small operations, per row of n values
void rms_norm(const float* x, float* y, int64_t n, float eps, const float* gamma = nullptr);
/// Scales x so that its L2 norm is at most `threshold` (llama.cpp's inkling2 clip: norm + 1e-6, factor clamped to 1).
void l2_clip(float* x, int64_t n, float threshold);
float dot_f32(const float* a, const float* b, int64_t n);

}  // namespace strata::inkling
