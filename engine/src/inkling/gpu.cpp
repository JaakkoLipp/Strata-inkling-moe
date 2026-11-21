// src/inkling/gpu.cpp - see gpu.hpp.
#include "gpu.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#if INKLING_BACKENDS
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#endif

namespace strata::inkling {

#if INKLING_BACKENDS

bool gpu_built() { return true; }

struct Gpu::Impl {
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t dev = nullptr;
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::string name;
    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
    }
};

std::unique_ptr<Gpu> Gpu::open(const std::string& which, int cpu_threads, std::string& why) {
    if (which == "off") {
        why = "the GPU is off (--gpu off)";
        return nullptr;
    }
    std::unique_ptr<Gpu> g(new Gpu());
    g->p_ = std::make_unique<Impl>();
    Impl& p = *g->p_;
    if (which == "cpu") {
        p.backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(p.backend, cpu_threads);
        p.name = "ggml CPU backend (standing in for a GPU)";
    } else {
        std::vector<ggml_backend_dev_t> gpus;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) gpus.push_back(d);
        }
        if (gpus.empty()) {
            why = "no GPU found (this build: CPU only, or no CUDA / ROCm device)";
            return nullptr;
        }
        size_t idx = 0;
        if (which != "auto") {
            idx = (size_t)std::atoi(which.c_str());
            if (idx >= gpus.size()) {
                why = "there is no GPU " + which + " (" + std::to_string(gpus.size()) + " found)";
                return nullptr;
            }
        }
        p.dev = gpus[idx];
        p.backend = ggml_backend_dev_init(p.dev, nullptr);
        if (!p.backend) {
            why = std::string("the GPU ") + ggml_backend_dev_description(p.dev) + " did not start";
            return nullptr;
        }
        p.name = ggml_backend_dev_description(p.dev);
    }
    p.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(p.backend));
    return g;
}

Gpu::~Gpu() = default;

std::string Gpu::name() const { return p_->name; }

uint64_t Gpu::free_bytes() const {
    if (!p_->dev) return 0;
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(p_->dev, &f, &t);
    return f;
}

bool Gpu::upload(Model& m, std::string& why) {
    Impl& p = *p_;
    std::vector<Mat*> mats = m.gpu_mats();
    if (mats.empty()) {
        why = "the model has no matrices for the GPU";
        return false;
    }
    ggml_init_params ip{ggml_tensor_overhead() * (mats.size() + 4), nullptr, true};
    p.wctx = ggml_init(ip);
    std::vector<ggml_tensor*> ts;
    uint64_t need = 0;
    for (Mat* W : mats) {
        ggml_tensor* t = ggml_new_tensor_2d(p.wctx, (ggml_type)W->type, W->cols, W->rows);
        ts.push_back(t);
        need += ggml_nbytes(t);
    }
    if (p.dev) {
        const uint64_t free = free_bytes();
        // the graphs' activations and the backend's own buffers need room beside the weights
        if (free && need + (512ull << 20) > free) {
            why = "the dense weights need " + std::to_string((need >> 20) + 512) + " MiB of VRAM, " +
                  std::to_string(free >> 20) + " MiB are free";
            return false;
        }
    }
    p.wbuf = ggml_backend_alloc_ctx_tensors(p.wctx, p.backend);
    if (!p.wbuf) {
        why = "could not allocate " + std::to_string(need >> 20) + " MiB on " + p.name;
        return false;
    }
    ggml_backend_buffer_set_usage(p.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ModelFile f(m.path(), false);
    std::vector<uint8_t> stage(64u << 20);
    for (size_t i = 0; i < mats.size(); ++i) {
        Mat* W = mats[i];
        for (uint64_t at = 0; at < W->nbytes;) {
            const size_t n = (size_t)std::min<uint64_t>(stage.size(), W->nbytes - at);
            f.read(stage.data(), n, W->file_offset + at);
            ggml_backend_tensor_set(ts[i], stage.data(), at, n);
            at += n;
        }
        W->gpu = ts[i];
    }
    weight_bytes_ = need;
    return true;
}

void Gpu::run(const Job* jobs, int n) {
    Impl& p = *p_;
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(2 * n + 4) + ggml_graph_overhead_custom(64, false), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    std::vector<std::pair<const float*, ggml_tensor*>> ins;
    std::vector<ggml_tensor*> outs(n);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 64, false);
    for (int i = 0; i < n; ++i) {
        const Job& j = jobs[i];
        ggml_tensor* x = nullptr;
        for (auto& [ptr, t] : ins)
            if (ptr == j.x && t->ne[0] == j.W->cols && t->ne[1] == j.ntok) x = t;
        if (!x) {
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, j.W->cols, j.ntok);
            ggml_set_input(x);
            ins.push_back({j.x, x});
        }
        outs[i] = ggml_mul_mat(ctx, static_cast<ggml_tensor*>(j.W->gpu), x);
        ggml_set_output(outs[i]);
        ggml_build_forward_expand(gf, outs[i]);
    }
    if (!ggml_gallocr_alloc_graph(p.galloc, gf)) {
        ggml_free(ctx);
        throw std::runtime_error("out of memory on " + p.name + " for a graph");
    }
    for (auto& [ptr, t] : ins) ggml_backend_tensor_set(t, ptr, 0, ggml_nbytes(t));
    if (ggml_backend_graph_compute(p.backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("a graph failed on " + p.name);
    }
    for (int i = 0; i < n; ++i) {
        const Job& j = jobs[i];
        const int64_t rows = outs[i]->ne[0];
        if (j.rows_out == rows) {
            ggml_backend_tensor_get(outs[i], j.y, 0, ggml_nbytes(outs[i]));
        } else {
            for (int t = 0; t < j.ntok; ++t)
                ggml_backend_tensor_get(outs[i], j.y + (size_t)t * j.rows_out, (size_t)t * rows * sizeof(float),
                                        (size_t)j.rows_out * sizeof(float));
        }
    }
    ggml_free(ctx);
}

#else  // the root engine project: no backend registry

bool gpu_built() { return false; }

struct Gpu::Impl {};

std::unique_ptr<Gpu> Gpu::open(const std::string& which, int, std::string& why) {
    why = which == "off" ? "the GPU is off (--gpu off)"
                         : "this build has no GPU backends (build src/inkling's own project with -DINKLING_GPU=cuda or hip)";
    return nullptr;
}
Gpu::~Gpu() = default;
std::string Gpu::name() const { return ""; }
uint64_t Gpu::free_bytes() const { return 0; }
bool Gpu::upload(Model&, std::string& why) {
    why = "no GPU backends in this build";
    return false;
}
void Gpu::run(const Job*, int) { throw std::runtime_error("no GPU backends in this build"); }

#endif

}  // namespace strata::inkling
