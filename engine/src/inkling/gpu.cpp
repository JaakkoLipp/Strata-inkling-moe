// src/inkling/gpu.cpp - see gpu.hpp.
#include "gpu.hpp"

#include "expert_store.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

namespace {

/// A graph built once and replayed: its own context and buffer (inputs, intermediates, outputs).
struct Cached {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph* gf = nullptr;
    std::vector<std::pair<const float*, ggml_tensor*>> ins;   // dense: host input -> tensor
    std::vector<ggml_tensor*> outs;
    // experts: the inputs and the output
    ggml_tensor *x = nullptr, *ids = nullptr, *dn = nullptr;
    void free() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr;
        ctx = nullptr;
    }
};

/// One expert type (gate_up type and shape, down type and shape): a pool of slots in VRAM.
struct Pool {
    int t_gu = 0, t_dn = 0;
    int64_t E = 0, R1 = 0, F = 0;          // gate_up [E -> R1], down [F -> E]
    uint64_t b_gu = 0, b_dn = 0;           // bytes of one expert's matrices
    ggml_tensor *gu = nullptr, *dn = nullptr;
    ggml_tensor *sgu = nullptr, *sdn = nullptr;   // [1][slots] the experts' output scales
    std::unique_ptr<SlotCache> cache;
    int dummy = 0;                         // the first of K reserved slots: rank k of a token the CPU computes points
                                           // at dummy + k (the batched kernel needs a token's K ids distinct)
    int filled = 0;                        // slots holding an expert (they never empty again)
};

/// Pinned host memory for the copies of one call (the device copies from it by DMA, without the driver's bounce through
/// pageable memory, which on ROCm made every small copy synchronous: ~150 us a call).
struct Staging {
    ggml_backend_buffer_t buf = nullptr;
    uint8_t* base = nullptr;
    size_t size = 0, used = 0;
    void init(ggml_backend_dev_t dev, size_t bytes) {
        if (!dev) return;
        ggml_backend_buffer_type_t bt = ggml_backend_dev_host_buffer_type(dev);
        if (!bt) return;
        buf = ggml_backend_buft_alloc_buffer(bt, bytes);
        if (!buf) return;
        base = static_cast<uint8_t*>(ggml_backend_buffer_get_base(buf));
        size = bytes;
    }
    void reset() { used = 0; }
    /// n bytes of the staging area, or nullptr when it is full (the copy then goes from the caller's memory)
    uint8_t* take(size_t n) {
        const size_t at = (used + 255) & ~(size_t)255;
        if (!base || at + n > size) return nullptr;
        used = at + n;
        return base + at;
    }
    void free() {
        if (buf) ggml_backend_buffer_free(buf);
        buf = nullptr;
        base = nullptr;
    }
};

struct MoeParams {
    ggml_backend_t backend;
    int K;
    float beta, lb;
};

}  // namespace

struct Gpu::Impl {
    ggml_backend_t backend = nullptr;
    ggml_backend_dev_t dev = nullptr;
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::string name;
    std::map<std::vector<uintptr_t>, Cached> dense;   // key: the jobs (W, x, ntok, y, rows_out)

    // the expert tier
    ggml_context* ectx = nullptr;
    ggml_backend_buffer_t ebuf = nullptr;
    std::vector<Pool> pools;
    std::vector<int> layer_pool;                       // physical layer -> pool, -1: none
    const Model* model = nullptr;
    int n_expert = 0, K = 0;
    float beta = 0, lb = 0;
    uint64_t pool_bytes = 0;
    std::vector<uint8_t> uses;                         // per (layer, expert): uses seen, saturating
    std::map<std::pair<int, int>, Cached> moe;         // key: (pool, T); a layer only chooses the pool
    Cached moe_tmp;                                    // a graph of an odd T, freed after use
    Cached* running = nullptr;
    int run_layer = -1, run_T = 0;
    std::vector<int> run_slot, pinned;
    std::vector<int32_t> run_ids;
    std::vector<float> run_out;
    GpuExpertStats est;
    Staging st_dense, st_moe;
    float* run_dst = nullptr;   // moe_end: where the GPU's per-rank outputs are (staging or run_out)

    ~Impl() {
        st_dense.free();
        st_moe.free();
        for (auto& [k, c] : dense) c.free();
        for (auto& [k, c] : moe) c.free();
        moe_tmp.free();
        if (ebuf) ggml_backend_buffer_free(ebuf);
        if (ectx) ggml_free(ectx);
        if (galloc) ggml_gallocr_free(galloc);
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (backend) ggml_backend_free(backend);
    }

    MoeParams params() const { return MoeParams{backend, K, beta, lb}; }

    bool alloc(Cached& c) {
        c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
        return c.buf != nullptr;
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
    p.st_dense.init(p.dev, 64u << 20);
    p.st_moe.init(p.dev, 64u << 20);
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

uint64_t Gpu::total_bytes() const {
    if (!p_->dev) return 0;
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(p_->dev, &f, &t);
    return t;
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
    std::vector<uintptr_t> key;
    key.reserve((size_t)n * 5);
    uint64_t out_bytes = 0;
    for (int i = 0; i < n; ++i) {
        const Job& j = jobs[i];
        key.insert(key.end(), {(uintptr_t)j.W, (uintptr_t)j.x, (uintptr_t)j.ntok, (uintptr_t)j.y, (uintptr_t)j.rows_out});
        out_bytes += (uint64_t)j.W->rows * j.ntok * sizeof(float);
    }
    // decoding's shapes are kept (the same ~100 sets every token); a prompt chunk's graph is built for the call in
    // the shared allocator (its building is small next to 256 tokens of work, and keeping them would take GiBs)
    bool keep = out_bytes <= (32ull << 20) && p.dense.size() < 4096;
    for (int i = 0; i < n; ++i) keep = keep && jobs[i].ntok == 1;
    Cached tmp;
    Cached* c = nullptr;
    if (keep) {
        auto it = p.dense.find(key);
        if (it != p.dense.end()) c = &it->second;
    }
    if (!c) {
        Cached nc;
        ggml_init_params ip{ggml_tensor_overhead() * (size_t)(2 * n + 4) + ggml_graph_overhead_custom(64, false), nullptr,
                            true};
        nc.ctx = ggml_init(ip);
        nc.gf = ggml_new_graph_custom(nc.ctx, 64, false);
        nc.outs.resize(n);
        for (int i = 0; i < n; ++i) {
            const Job& j = jobs[i];
            ggml_tensor* x = nullptr;
            for (auto& [ptr, t] : nc.ins)
                if (ptr == j.x && t->ne[0] == j.W->cols && t->ne[1] == j.ntok) x = t;
            if (!x) {
                x = ggml_new_tensor_2d(nc.ctx, GGML_TYPE_F32, j.W->cols, j.ntok);
                ggml_set_input(x);
                nc.ins.push_back({j.x, x});
            }
            nc.outs[i] = ggml_mul_mat(nc.ctx, static_cast<ggml_tensor*>(j.W->gpu), x);
            ggml_set_output(nc.outs[i]);
            ggml_build_forward_expand(nc.gf, nc.outs[i]);
        }
        const bool ok = keep ? p.alloc(nc) : ggml_gallocr_alloc_graph(p.galloc, nc.gf);
        if (!ok) {
            nc.free();
            throw std::runtime_error("out of memory on " + p.name + " for a graph");
        }
        if (keep) c = &p.dense.emplace(key, nc).first->second;
        else {
            tmp = nc;
            c = &tmp;
        }
    }
    // every copy is queued on the compute stream from pinned memory, and one wait covers them all
    p.st_dense.reset();
    for (auto& [ptr, t] : c->ins) {
        const size_t nb = ggml_nbytes(t);
        uint8_t* st = p.st_dense.take(nb);
        if (st) std::memcpy(st, ptr, nb);
        ggml_backend_tensor_set_async(p.backend, t, st ? (const void*)st : (const void*)ptr, 0, nb);
    }
    if (ggml_backend_graph_compute_async(p.backend, c->gf) != GGML_STATUS_SUCCESS) {
        tmp.free();
        throw std::runtime_error("a graph failed on " + p.name);
    }
    std::vector<uint8_t*> got(n, nullptr);
    for (int i = 0; i < n; ++i) {
        const Job& j = jobs[i];
        ggml_tensor* o = c->outs[i];
        got[i] = p.st_dense.take(ggml_nbytes(o));
        if (got[i]) {
            ggml_backend_tensor_get_async(p.backend, o, got[i], 0, ggml_nbytes(o));
            continue;
        }
        const int64_t rows = o->ne[0];
        if (j.rows_out == rows) {
            ggml_backend_tensor_get_async(p.backend, o, j.y, 0, ggml_nbytes(o));
        } else {
            for (int t = 0; t < j.ntok; ++t)
                ggml_backend_tensor_get_async(p.backend, o, j.y + (size_t)t * j.rows_out, (size_t)t * rows * sizeof(float),
                                              (size_t)j.rows_out * sizeof(float));
        }
    }
    ggml_backend_synchronize(p.backend);
    for (int i = 0; i < n; ++i) {
        if (!got[i]) continue;
        const Job& j = jobs[i];
        const int64_t rows = c->outs[i]->ne[0];
        const float* src = reinterpret_cast<const float*>(got[i]);
        if (j.rows_out == rows) std::memcpy(j.y, src, (size_t)rows * j.ntok * sizeof(float));
        else
            for (int t = 0; t < j.ntok; ++t)
                std::memcpy(j.y + (size_t)t * j.rows_out, src + (size_t)t * rows, (size_t)j.rows_out * sizeof(float));
    }
    tmp.free();
}

// ------------------------------------------------------------------------------------------------ the expert tier

namespace {

/// The expert graph of one layer for T tokens: gu = W_gu[ids] . x, scaled; SiTU(gate, up); dn = W_dn[ids] . act, scaled.
bool build_moe(const MoeParams& p, Cached& c, const Pool& P, int T, std::string* why) {
    const int K = p.K;
    ggml_init_params ip{ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true};
    c.ctx = ggml_init(ip);
    ggml_context* ctx = c.ctx;
    c.x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, P.E, 1, T);
    c.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, T);
    for (ggml_tensor* t : {c.x, c.ids}) ggml_set_input(t);
    // the experts' scales, gathered on the GPU by slot: [1, K, T]
    ggml_tensor* flat = ggml_reshape_1d(ctx, c.ids, (int64_t)K * T);
    ggml_tensor* sgu = ggml_reshape_3d(ctx, ggml_get_rows(ctx, P.sgu, flat), 1, K, T);
    ggml_tensor* sdn = ggml_reshape_3d(ctx, ggml_get_rows(ctx, P.sdn, flat), 1, K, T);
    ggml_tensor* gu = ggml_mul_mat_id(ctx, P.gu, c.x, c.ids);                 // [R1, K, T]
    gu = ggml_mul(ctx, gu, sgu);
    const int64_t F = P.F;
    ggml_tensor* g = ggml_cont(ctx, ggml_view_3d(ctx, gu, F, K, T, gu->nb[1], gu->nb[2], 0));
    ggml_tensor* u = ggml_cont(ctx, ggml_view_3d(ctx, gu, F, K, T, gu->nb[1], gu->nb[2], (size_t)F * sizeof(float)));
    // situ(gate, up) = beta tanh(gate / beta) sigmoid(gate) * lb tanh(up / lb)
    ggml_tensor* act = ggml_mul(ctx, ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, g, 1.0f / p.beta)), p.beta),
                                ggml_sigmoid(ctx, g));
    if (p.lb > 0.0f) u = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, u, 1.0f / p.lb)), p.lb);
    act = ggml_mul(ctx, act, u);                                               // [F, K, T]
    ggml_tensor* dn = ggml_mul_mat_id(ctx, P.dn, act, c.ids);                 // [E, K, T]
    dn = ggml_mul(ctx, dn, sdn);
    ggml_set_output(dn);
    c.dn = dn;
    c.gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(c.gf, dn);
    if (why) {
        // every node must run on this backend (an op it lacks would fail at compute time)
        for (int i = 0; i < ggml_graph_n_nodes(c.gf); ++i) {
            ggml_tensor* nd = ggml_graph_node(c.gf, i);
            if (!ggml_backend_supports_op(p.backend, nd)) {
                *why = std::string("the GPU backend cannot run ") + ggml_op_desc(nd) + " for " + ggml_type_name((ggml_type)P.t_gu) +
                       " / " + ggml_type_name((ggml_type)P.t_dn) + " experts";
                c.free();
                return false;
            }
        }
    }
    c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, p.backend);
    if (!c.buf) {
        if (why) *why = "out of VRAM for the expert graph";
        c.free();
        return false;
    }
    return true;
}

}  // namespace

bool Gpu::init_experts(const Model& m, uint64_t bytes, std::string& why) {
    Impl& p = *p_;
    const Hparams& h = m.hp();
    p.model = &m;
    p.n_expert = h.n_expert;
    p.K = h.n_expert_used;
    p.beta = h.situ_beta;
    p.lb = h.situ_linear_beta;
    p.layer_pool.assign(h.n_layer, -1);
    // the expert types, and how many expert-bytes of the model each holds
    std::vector<double> share;
    for (int l = 0; l < h.n_layer; ++l) {
        const Layer& L = m.layers()[l];
        int pi = -1;
        for (size_t i = 0; i < p.pools.size(); ++i)
            if (p.pools[i].t_gu == L.gate_up.type && p.pools[i].t_dn == L.down.type && p.pools[i].E == L.gate_up.cols &&
                p.pools[i].R1 == L.gate_up.rows && p.pools[i].F == L.down.cols)
                pi = (int)i;
        if (pi < 0) {
            Pool P;
            P.t_gu = L.gate_up.type;
            P.t_dn = L.down.type;
            P.E = L.gate_up.cols;
            P.R1 = L.gate_up.rows;
            P.F = L.down.cols;
            P.b_gu = L.gate_up.bytes;
            P.b_dn = L.down.bytes;
            p.pools.push_back(std::move(P));
            share.push_back(0);
            pi = (int)p.pools.size() - 1;
        }
        p.layer_pool[l] = pi;
        share[pi] += (double)h.n_expert * (double)(L.gate_up.bytes + L.down.bytes);
    }
    double total = 0;
    for (double s : share) total += s;
    // the slots each pool gets: its share of the budget, at most every expert of its layers
    std::vector<int> n_slots(p.pools.size());
    uint64_t need = 0;
    for (size_t i = 0; i < p.pools.size(); ++i) {
        const Pool& P = p.pools[i];
        const uint64_t per = P.b_gu + P.b_dn;
        const uint64_t all = (uint64_t)std::llround(share[i] / (double)per);
        n_slots[i] = (int)std::min<uint64_t>(all, (uint64_t)((double)bytes * share[i] / total) / per);
        if (n_slots[i] < p.K + 2) n_slots[i] = 0;
        need += (uint64_t)n_slots[i] * per;
    }
    if (need == 0) {
        why = "the VRAM budget holds no expert";
        p.pools.clear();
        return false;
    }
    ggml_init_params ip{ggml_tensor_overhead() * (4 * p.pools.size() + 2), nullptr, true};
    p.ectx = ggml_init(ip);
    for (size_t i = 0; i < p.pools.size(); ++i) {
        Pool& P = p.pools[i];
        if (!n_slots[i]) continue;
        P.gu = ggml_new_tensor_3d(p.ectx, (ggml_type)P.t_gu, P.E, P.R1, n_slots[i]);
        P.dn = ggml_new_tensor_3d(p.ectx, (ggml_type)P.t_dn, P.F, P.E, n_slots[i]);
        P.sgu = ggml_new_tensor_2d(p.ectx, GGML_TYPE_F32, 1, n_slots[i]);
        P.sdn = ggml_new_tensor_2d(p.ectx, GGML_TYPE_F32, 1, n_slots[i]);
        if (ggml_nbytes(P.gu) != P.b_gu * n_slots[i] || ggml_nbytes(P.dn) != P.b_dn * n_slots[i]) {
            why = "an expert's size in the file is not its ggml size";
            return false;
        }
    }
    p.ebuf = ggml_backend_alloc_ctx_tensors(p.ectx, p.backend);
    if (!p.ebuf) {
        why = "could not allocate " + std::to_string(need >> 20) + " MiB of VRAM for experts";
        return false;
    }
    ggml_backend_buffer_set_usage(p.ebuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(p.ebuf, 0);
    for (size_t i = 0; i < p.pools.size(); ++i) {
        Pool& P = p.pools[i];
        if (!n_slots[i]) continue;
        P.cache = std::make_unique<SlotCache>(n_slots[i], h.n_layer * h.n_expert);
        // K slots stay empty: the ranks the CPU computes point at them (their outputs are not used)
        P.dummy = n_slots[i] - p.K;
        for (int k = 0; k < p.K; ++k) {
            P.cache->slot(P.dummy + k).pins = 1 << 20;
            P.cache->slot(P.dummy + k).state = SlotCache::State::Ready;
        }
        // can this backend run the graph for this type?
        for (int T : {1, 2}) {
            Cached probe;
            if (!build_moe(p.params(), probe, P, T, &why)) return false;
            probe.free();
        }
    }
    for (int l = 0; l < h.n_layer; ++l)
        if (p.layer_pool[l] >= 0 && !p.pools[p.layer_pool[l]].cache) p.layer_pool[l] = -1;
    p.uses.assign((size_t)h.n_layer * h.n_expert, 0);
    p.pool_bytes = need;
    return true;
}

bool Gpu::has_experts() const { return p_->ebuf != nullptr; }
uint64_t Gpu::expert_bytes() const { return p_->pool_bytes; }

int Gpu::expert_slots() const {
    int n = 0;
    for (const Pool& P : p_->pools)
        if (P.cache) n += P.cache->n_slots() - p_->K;
    return n;
}

int Gpu::lookup(int layer, int e) {
    Impl& p = *p_;
    const int pi = p.layer_pool.empty() ? -1 : p.layer_pool[layer];
    if (pi < 0) return -1;
    SlotCache& C = *p.pools[pi].cache;
    const int s = C.find(layer * p.n_expert + e);
    if (s < 0 || C.slot(s).state != SlotCache::State::Ready) return -1;
    C.touch(s);
    return s;
}

bool Gpu::want(int layer, int e) {
    Impl& p = *p_;
    const int pi = p.layer_pool.empty() ? -1 : p.layer_pool[layer];
    if (pi < 0) return false;
    uint8_t& u = p.uses[(size_t)layer * p.n_expert + e];
    if (u < 255) ++u;
    // while the pool has empty slots every used expert goes up; once it is full, one used twice
    const Pool& P = p.pools[pi];
    return P.filled < P.cache->n_slots() - p.K || u >= 2;
}

bool Gpu::admit(int layer, int e, const ExpertView& v) {
    Impl& p = *p_;
    const int pi = p.layer_pool.empty() ? -1 : p.layer_pool[layer];
    if (pi < 0) return false;
    Pool& P = p.pools[pi];
    SlotCache& C = *P.cache;
    const int key = layer * p.n_expert + e;
    if (C.find(key) >= 0) return true;
    const int s = C.claim(key, false);
    if (s < 0) return false;
    ggml_backend_tensor_set(P.gu, v.gate_up.data, (size_t)s * P.b_gu, P.b_gu);
    ggml_backend_tensor_set(P.dn, v.down.data, (size_t)s * P.b_dn, P.b_dn);
    const Layer& L = p.model->layers()[layer];
    const float sg = L.gate_up.scale.empty() ? 1.0f : L.gate_up.scale[e], sd = L.down.scale.empty() ? 1.0f : L.down.scale[e];
    ggml_backend_tensor_set(P.sgu, &sg, (size_t)s * sizeof(float), sizeof(float));
    ggml_backend_tensor_set(P.sdn, &sd, (size_t)s * sizeof(float), sizeof(float));
    C.slot(s).state = SlotCache::State::Ready;
    if (P.filled < C.n_slots() - p.K) ++P.filled;
    ++p.est.uploads;
    p.est.upload_bytes += P.b_gu + P.b_dn;
    return true;
}

void Gpu::moe_begin(int layer, const float* x, int T, const int* slot) {
    Impl& p = *p_;
    const int pi = p.layer_pool[layer];
    Pool& P = p.pools[pi];
    const int K = p.K;
    Cached* c = nullptr;
    const bool keep = T == 1 || p.moe.size() < 2 * p.pools.size() + 2;
    if (keep) {
        auto it = p.moe.find({pi, T});
        if (it == p.moe.end()) {
            Cached nc;
            if (!build_moe(p.params(), nc, P, T, nullptr)) throw std::runtime_error("out of VRAM for the expert graph on " + p.name);
            it = p.moe.emplace(std::make_pair(pi, T), nc).first;
        }
        c = &it->second;
    } else {
        p.moe_tmp.free();
        if (!build_moe(p.params(), p.moe_tmp, P, T, nullptr)) throw std::runtime_error("out of VRAM for the expert graph on " + p.name);
        c = &p.moe_tmp;
    }
    std::vector<int32_t> ids((size_t)T * K);
    p.run_slot.assign(slot, slot + (size_t)T * K);
    p.pinned.clear();
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = slot[i] >= 0 ? slot[i] : P.dummy + (int)(i % K);
        if (slot[i] >= 0) {
            ++P.cache->slot(slot[i]).pins;   // admit() must not overwrite it while the graph reads it
            p.pinned.push_back(slot[i]);
        }
    }
    // queued on the compute stream: no wait until moe_end (ids lives until then, in run_ids)
    p.run_ids.swap(ids);
    p.st_moe.reset();
    const size_t nx = ggml_nbytes(c->x), ni = ggml_nbytes(c->ids), no = ggml_nbytes(c->dn);
    uint8_t* sx = p.st_moe.take(nx);
    uint8_t* si = p.st_moe.take(ni);
    uint8_t* so = p.st_moe.take(no);
    if (sx) std::memcpy(sx, x, nx);
    if (si) std::memcpy(si, p.run_ids.data(), ni);
    ggml_backend_tensor_set_async(p.backend, c->x, sx ? (const void*)sx : (const void*)x, 0, nx);
    ggml_backend_tensor_set_async(p.backend, c->ids, si ? (const void*)si : (const void*)p.run_ids.data(), 0, ni);
    if (ggml_backend_graph_compute_async(p.backend, c->gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("the expert graph failed on " + p.name);
    if (so) {
        p.run_dst = reinterpret_cast<float*>(so);
    } else {
        p.run_out.resize(ggml_nelements(c->dn));
        p.run_dst = p.run_out.data();
    }
    ggml_backend_tensor_get_async(p.backend, c->dn, p.run_dst, 0, no);
    p.running = c;
    p.run_layer = layer;
    p.run_T = T;
}

void Gpu::moe_end(float* out) {
    Impl& p = *p_;
    if (!p.running) return;
    ggml_backend_synchronize(p.backend);
    Cached& c = *p.running;
    const int K = p.K, T = p.run_T;
    const int64_t E = c.dn->ne[0];
    for (size_t i = 0; i < (size_t)T * K; ++i) {
        ++p.est.lookups;
        if (p.run_slot[i] < 0) continue;
        ++p.est.hits;
        std::memcpy(out + i * E, p.run_dst + i * E, sizeof(float) * E);
    }
    SlotCache& C = *p.pools[p.layer_pool[p.run_layer]].cache;
    for (int s : p.pinned) --C.slot(s).pins;
    p.pinned.clear();
    p.running = nullptr;
}

GpuExpertStats Gpu::expert_stats() const { return p_->est; }
void Gpu::reset_expert_stats() { p_->est = GpuExpertStats{}; }

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
uint64_t Gpu::total_bytes() const { return 0; }
bool Gpu::upload(Model&, std::string& why) {
    why = "no GPU backends in this build";
    return false;
}
void Gpu::run(const Job*, int) { throw std::runtime_error("no GPU backends in this build"); }
bool Gpu::init_experts(const Model&, uint64_t, std::string& why) {
    why = "no GPU backends in this build";
    return false;
}
bool Gpu::has_experts() const { return false; }
uint64_t Gpu::expert_bytes() const { return 0; }
int Gpu::expert_slots() const { return 0; }
int Gpu::lookup(int, int) { return -1; }
bool Gpu::want(int, int) { return false; }
bool Gpu::admit(int, int, const ExpertView&) { return false; }
void Gpu::moe_begin(int, const float*, int, const int*) {
    throw std::runtime_error("no GPU backends in this build");
}
void Gpu::moe_end(float*) {}
GpuExpertStats Gpu::expert_stats() const { return {}; }
void Gpu::reset_expert_stats() {}

#endif

}  // namespace strata::inkling
