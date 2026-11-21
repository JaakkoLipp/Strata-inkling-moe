// src/inkling/cpu_ops.cpp - see cpu_ops.hpp.
#include "cpu_ops.hpp"

#include "dot_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define INK_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define INK_PAUSE() __asm__ __volatile__("yield")
#else
#define INK_PAUSE() do {} while (0)
#endif

namespace strata::inkling {

// ---------------------------------------------------------------- thread pool

ThreadPool::ThreadPool(int n_threads) : n_(std::max(1, n_threads)) {
    for (int i = 1; i < n_; ++i) threads_.emplace_back([this] { worker(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void ThreadPool::drain() {
    const int64_t g = grain_, total = total_;
    for (;;) {
        int64_t i = next_.fetch_add(g);
        if (i >= total) break;
        (*fn_)(i, std::min(total, i + g));
    }
}

void ThreadPool::worker() {
    // 0, the epoch at construction: a run() that starts before this thread is first scheduled must still be seen
    uint64_t seen = 0;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        int spins = 0;
        while (epoch_.load() == seen && !stop_.load()) {
            INK_PAUSE();
            if (++spins == 256) {
                spins = 0;
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(300)) {
                    std::unique_lock<std::mutex> lk(m_);
                    sleepers_.fetch_add(1);
                    cv_.wait(lk, [&] { return epoch_.load() != seen || stop_.load(); });
                    sleepers_.fetch_sub(1);
                    break;
                }
            }
        }
        if (stop_.load()) return;
        seen = epoch_.load();
        drain();
        if (pending_.fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> lk(m_);
            done_cv_.notify_one();
        }
    }
}

void ThreadPool::run(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    grain = std::max<int64_t>(1, grain);
    if (n_ == 1 || n <= grain) {
        fn(0, n);
        return;
    }
    fn_ = &fn;
    total_ = n;
    grain_ = grain;
    next_.store(0);
    pending_.store(n_ - 1);
    epoch_.fetch_add(1);
    if (sleepers_.load() > 0) {
        std::lock_guard<std::mutex> lk(m_);
        cv_.notify_all();
    }
    drain();
    int spins = 0;
    while (pending_.load() != 0) {
        INK_PAUSE();
        if (++spins > 20000) {
            std::unique_lock<std::mutex> lk(m_);
            done_cv_.wait(lk, [&] { return pending_.load() == 0; });
            break;
        }
    }
}

// ---------------------------------------------------------------- matrices

int vec_dot_type(int wtype) {
    const auto* tr = ggml_get_type_traits_cpu((ggml_type)wtype);
    if (!tr || !tr->vec_dot) throw std::runtime_error(std::string("no CPU dot product for ") + ggml_type_name((ggml_type)wtype));
    return tr->vec_dot_type;
}

void quantize_acts(const float* x, int ntok, int64_t n, int64_t ldx, int wtype, QAct& q, ThreadPool* pool) {
    const int vt = vec_dot_type(wtype);
    q.vtype = vt;
    q.n = n;
    q.ntok = ntok;
    q.row_bytes = (size_t)ggml_row_size((ggml_type)vt, n);
    if (q.buf.size() < q.row_bytes * (size_t)ntok) q.buf.resize(q.row_bytes * (size_t)ntok);
    auto one = [&](int t) {
        uint8_t* dst = q.buf.data() + (size_t)t * q.row_bytes;
        const float* src = x + (size_t)t * ldx;
        if (vt == GGML_TYPE_F32) {
            std::memcpy(dst, src, sizeof(float) * n);
        } else {
            const auto* tr = ggml_get_type_traits_cpu((ggml_type)vt);
            if (!tr->from_float) throw std::runtime_error(std::string("cannot quantize activations to ") + ggml_type_name((ggml_type)vt));
            tr->from_float(src, dst, n);
        }
    };
    if (pool && ntok >= 8) {
        pool->run(ntok, std::max<int64_t>(1, ntok / (4 * pool->size())), [&](int64_t a, int64_t b) {
            for (int64_t t = a; t < b; ++t) one((int)t);
        });
    } else {
        for (int t = 0; t < ntok; ++t) one(t);
    }
}

void dot_rows(const Mat& W, const QAct& xq, int64_t r0, int64_t r1, float* y, int64_t ldy) {
    tiled_dots(vec_dot_fn(W.type), (int)W.cols, W, r0, r1, xq.ntok, [&](int t) { return xq.row(t); },
               [&](int t, int64_t r, float s) { y[(size_t)t * ldy + r] = s; });
}

void matmul(ThreadPool& pool, const Mat& W, const QAct& xq, float* y, int64_t ldy) {
    if (xq.n != W.cols) throw std::runtime_error("matmul: activation length does not match the matrix");
    const int64_t chunks = std::max<int64_t>(1, 4 * pool.size());
    const int64_t grain = std::max<int64_t>(16, ((W.rows + chunks - 1) / chunks + 15) / 16 * 16);
    pool.run(W.rows, grain, [&](int64_t a, int64_t b) { dot_rows(W, xq, a, b, y, ldy); });
}

void matmul_f(ThreadPool& pool, const Mat& W, const float* x, int ntok, int64_t ldx, float* y, int64_t ldy, QAct& scratch) {
    quantize_acts(x, ntok, W.cols, ldx, W.type, scratch, &pool);
    matmul(pool, W, scratch, y, ldy);
}

// ---------------------------------------------------------------- vectors

void rms_norm(const float* x, float* y, int64_t n, float eps, const float* gamma) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
    if (gamma)
        for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale * gamma[i];
    else
        for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

void l2_clip(float* x, int64_t n, float threshold) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float norm = std::sqrt((float)ss) + 1e-6f;
    float f = threshold / norm;
    if (f > 1.0f) f = 1.0f;
    if (f < 0.0f) f = 0.0f;
    for (int64_t i = 0; i < n; ++i) x[i] *= f;
}

float dot_f32(const float* a, const float* b, int64_t n) {
    // ggml-cpu's SIMD f32 dot (what its mul_mat uses for f32 operands)
    static const ggml_vec_dot_t vd = ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot;
    float s;
    vd((int)n, &s, 0, a, 0, b, 0, 1);
    return s;
}

}  // namespace strata::inkling
