// src/inkling/engine.cpp - see engine.hpp. Each step names the line of llama.cpp's inkling2.cpp it follows.
#include "engine.hpp"

#include "dot_kernels.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace strata::inkling {

using clk = std::chrono::steady_clock;
static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// STRATA_INKLING_DEBUG=1: the sum and first values of the tensors llama.cpp's eval-callback names, to find where two
// implementations part (tools/inkling/parity.md)
static const bool g_debug = std::getenv("STRATA_INKLING_DEBUG") != nullptr;
static void dbg(const char* name, int il, const float* x, size_t n) {
    if (!g_debug) return;
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += x[i];
    char nm[64];
    if (il >= 0) std::snprintf(nm, sizeof nm, "%s-%d", name, il);
    else std::snprintf(nm, sizeof nm, "%s", name);
    std::fprintf(stderr, "DBG %-24s sum=%.6f first=%.4f %.4f %.4f\n", nm, s, x[0], n > 1 ? x[1] : 0.0f, n > 2 ? x[2] : 0.0f);
}

static int default_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 4;
}

// the scratch of one chunk; every per-token array is [T][dim]
struct Engine::Buf {
    int T = 0;
    std::vector<float> cur, inpSA, in0, in1, in2, mixout, q, k, v, r, sinks, kdyn, vdyn, att, gate, o, ffn_inp, x, y;
    std::vector<float> prelude, msgs, tmp, g2, h, ga, fo;
    std::vector<float> logits, gu, act, dnk, wts;
    std::vector<int> ids;
    QAct qa, qb;
};

Engine::Engine(const Model& model, ExpertStore& store, const EngineOptions& opt) : m_(model), store_(store), opt_(opt) {
    const Hparams& h = m_.hp();
    if (opt_.threads <= 0) opt_.threads = default_threads();
    if (opt_.chunk <= 0) opt_.chunk = 256;
    if (opt_.n_ctx <= 0) opt_.n_ctx = 32768;
    pool_ = std::make_unique<ThreadPool>(opt_.threads);

    const int n_slots = m_.n_slots();
    st_.k.resize(n_slots);
    st_.v.resize(n_slots);
    st_.cap.assign(n_slots, 0);
    st_.conv.resize(n_slots);
    const int kvd = h.kv_dim();
    const size_t conv_floats = (size_t)(h.n_conv - 1) * (2 * kvd + 2 * h.n_embd);
    for (const Application& a : m_.schedule()) {
        const bool swa = m_.layers()[a.phys].swa;
        // a sliding-window slot keeps a ring that holds the window and one chunk written ahead of its queries
        const int cap = swa ? std::min(opt_.n_ctx, h.n_swa + opt_.chunk) : opt_.n_ctx;
        st_.cap[a.slot] = cap;
        // new[] leaves the pages untouched: the OS commits RAM only for the positions that get written
        st_.k[a.slot].reset(new uint16_t[(size_t)cap * kvd]);
        st_.v[a.slot].reset(new uint16_t[(size_t)cap * kvd]);
        st_.conv[a.slot].assign(conv_floats, 0.0f);
    }

    b_ = std::make_unique<Buf>();
    const int T = opt_.chunk, E = h.n_embd;
    auto sz = [&](std::vector<float>& v, size_t n) { v.assign(n, 0.0f); };
    sz(b_->cur, (size_t)T * E); sz(b_->inpSA, (size_t)T * E); sz(b_->in0, (size_t)T * E); sz(b_->in1, (size_t)T * E);
    sz(b_->in2, (size_t)T * E); sz(b_->mixout, (size_t)4 * T * E); sz(b_->q, (size_t)T * h.q_dim());
    sz(b_->k, (size_t)T * kvd); sz(b_->v, (size_t)T * kvd); sz(b_->r, (size_t)T * h.n_rel * h.n_head);
    sz(b_->sinks, (size_t)T * h.n_head); sz(b_->kdyn, (size_t)T * h.n_conv * h.n_head_kv);
    sz(b_->vdyn, (size_t)T * h.n_conv * h.n_head_kv); sz(b_->att, (size_t)T * h.q_dim()); sz(b_->gate, (size_t)T * h.q_dim());
    sz(b_->o, (size_t)T * E); sz(b_->ffn_inp, (size_t)T * E); sz(b_->x, (size_t)T * E); sz(b_->y, (size_t)T * E);
    sz(b_->prelude, (size_t)T * E); sz(b_->msgs, (size_t)4 * T * E); sz(b_->tmp, (size_t)T * E);
    sz(b_->g2, (size_t)2 * T * E); sz(b_->h, (size_t)T * std::max(1, h.gate_rank)); sz(b_->ga, (size_t)T * E);
    sz(b_->fo, (size_t)T * E); sz(b_->logits, (size_t)T * h.n_expert);
    sz(b_->wts, (size_t)T * h.n_expert_used); b_->ids.assign((size_t)T * h.n_expert_used, 0);
    sz(b_->dnk, (size_t)T * h.n_expert_used * E);
}

Engine::~Engine() = default;

void Engine::reset() {
    st_.n_past = 0;
    for (auto& c : st_.conv) std::fill(c.begin(), c.end(), 0.0f);
}

uint64_t Engine::kv_bytes_full() const {
    uint64_t s = 0;
    for (int c : st_.cap) s += (uint64_t)c * m_.hp().kv_dim() * 2 * sizeof(uint16_t);
    return s;
}

uint64_t Engine::snapshot_bytes() const {
    const Hparams& h = m_.hp();
    uint64_t s = 0;
    for (const Application& a : m_.schedule()) {
        s += st_.conv[a.slot].size() * sizeof(float);
        if (m_.layers()[a.phys].swa) s += (uint64_t)st_.cap[a.slot] * h.kv_dim() * 2 * sizeof(uint16_t);
    }
    return s;
}

void Engine::snapshot(Snapshot& out) const {
    const size_t kvd = (size_t)m_.hp().kv_dim();
    out.n_past = st_.n_past;
    out.conv = st_.conv;
    out.ring_k.resize(st_.k.size());
    out.ring_v.resize(st_.v.size());
    for (const Application& a : m_.schedule()) {
        const int s = a.slot;
        if (!m_.layers()[a.phys].swa) {
            out.ring_k[s].clear();
            out.ring_v[s].clear();
            continue;
        }
        const size_t n = (size_t)std::min(st_.cap[s], st_.n_past) * kvd;
        out.ring_k[s].assign(st_.k[s].get(), st_.k[s].get() + n);
        out.ring_v[s].assign(st_.v[s].get(), st_.v[s].get() + n);
    }
}

bool Engine::rollback(const Snapshot& snap) {
    if (!snap.valid() || snap.n_past > st_.n_past || snap.conv.size() != st_.conv.size()) return false;
    st_.n_past = snap.n_past;
    st_.conv = snap.conv;
    for (const Application& a : m_.schedule()) {
        const int s = a.slot;
        if (!m_.layers()[a.phys].swa) continue;   // positions >= n_past are written again before they are read
        std::memcpy(st_.k[s].get(), snap.ring_k[s].data(), snap.ring_k[s].size() * sizeof(uint16_t));
        std::memcpy(st_.v[s].get(), snap.ring_v[s].data(), snap.ring_v[s].size() * sizeof(uint16_t));
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ building blocks

// mix(): attention over the depth. states are in llama.cpp's order (the blocks last to first, then cur); every
// state is RMS-normalized (eps 1e-6, no gain), scored against each query at 1/sqrt(n_embd), and the normalized
// states are averaged with the softmax weights. out is [nq][T][E].
void Engine::mix(const std::vector<const float*>& states, int T, const float* query, int nq, float* out) {
    const int E = m_.hp().n_embd;
    const int S = (int)states.size();
    const float scale = 1.0f / std::sqrt((float)E);
    pool_->run(T, 1, [&](int64_t t0, int64_t t1) {
        thread_local std::vector<float> ns, sc;
        ns.resize((size_t)S * E);
        sc.resize(S);
        for (int64_t t = t0; t < t1; ++t) {
            for (int s = 0; s < S; ++s) rms_norm(states[s] + (size_t)t * E, &ns[(size_t)s * E], E, 1e-6f);
            for (int qi = 0; qi < nq; ++qi) {
                const float* qv = query + (size_t)qi * E;
                float mx = -INFINITY;
                for (int s = 0; s < S; ++s) {
                    sc[s] = dot_f32(&ns[(size_t)s * E], qv, E) * scale;
                    mx = std::max(mx, sc[s]);
                }
                double sum = 0;
                for (int s = 0; s < S; ++s) { sc[s] = std::exp(sc[s] - mx); sum += sc[s]; }
                const float inv = (float)(1.0 / sum);
                float* o = out + ((size_t)qi * T + t) * E;
                for (int e = 0; e < E; ++e) o[e] = 0.0f;
                for (int s = 0; s < S; ++s) {
                    const float p = sc[s] * inv;
                    const float* n = &ns[(size_t)s * E];
                    for (int e = 0; e < E; ++e) o[e] += n[e] * p;
                }
            }
        }
    });
}

// control(): (1 - forget) * cur + gain * message, forget and gain = tanh(W . down . [rms(cur), rms(message)])
void Engine::control(const LoopLayer& LL, int k, const float* cur, const float* msg, int T, float* out) {
    const int E = m_.hp().n_embd, R = m_.hp().gate_rank;
    Buf& b = *b_;
    for (int t = 0; t < T; ++t) {
        rms_norm(cur + (size_t)t * E, &b.g2[(size_t)t * 2 * E], E, 1e-6f);
        rms_norm(msg + (size_t)t * E, &b.g2[(size_t)t * 2 * E + E], E, 1e-6f);
    }
    matmul_f(*pool_, LL.down[k], b.g2.data(), T, 2 * E, b.h.data(), R, b.qa);
    matmul_f(*pool_, LL.gain[k], b.h.data(), T, R, b.ga.data(), E, b.qa);
    matmul_f(*pool_, LL.forget[k], b.h.data(), T, R, b.fo.data(), E, b.qa);
    for (size_t i = 0; i < (size_t)T * E; ++i) {
        const float c = cur[i], gain = std::tanh(b.ga[i]), forget = std::tanh(b.fo[i]);
        out[i] = (c - forget * c) + gain * msg[i];
    }
}

// convolve(): causal short convolution over the sequence with a residual (inkling2.cpp convolve). past holds the
// last n_conv - 1 inputs, oldest first, and is advanced. dyn ([T][n_heads][n_conv]) adds per-head input-dependent
// taps. cur is replaced by (cur + conv) / sqrt((dyn ? 2 : 1) * n_conv + 1).
void Engine::convolve(float* cur, int T, int C, const std::vector<float>& w, const float* dyn, int n_heads, float* past) {
    const int nc = m_.hp().n_conv, np = nc - 1;
    const float scale = 1.0f / std::sqrt((float)((dyn ? 2 : 1) * nc + 1));
    const int per_head = dyn ? C / n_heads : 1;
    auto X = [&](int idx) -> const float* { return idx < np ? past + (size_t)idx * C : cur + (size_t)(idx - np) * C; };
    std::vector<float>& out = b_->tmp;
    if (out.size() < (size_t)T * C) out.resize((size_t)T * C);
    pool_->run(C, 256, [&](int64_t c0, int64_t c1) {
        for (int t = 0; t < T; ++t) {
            float* o = &out[(size_t)t * C];
            for (int64_t c = c0; c < c1; ++c) {
                float s = 0.0f;
                for (int k = 0; k < nc; ++k) s += X(t + k)[c] * w[(size_t)c * nc + k];
                o[c] = s;
            }
            if (dyn) {
                const float* d = dyn + (size_t)t * n_heads * nc;
                for (int lag = 0; lag < nc; ++lag) {
                    const float* xin = X(t + nc - 1 - lag);
                    for (int64_t c = c0; c < c1; ++c) o[c] += xin[c] * d[(c / per_head) * nc + lag];
                }
            }
        }
    });
    // the new history: the last n_conv - 1 inputs of [past, cur]
    std::vector<float> hist((size_t)np * C);
    for (int k = 0; k < np; ++k) std::memcpy(&hist[(size_t)k * C], X(T + k), sizeof(float) * C);
    std::memcpy(past, hist.data(), sizeof(float) * hist.size());
    for (size_t i = 0; i < (size_t)T * C; ++i) cur[i] = (cur[i] + out[i]) * scale;
}

// ------------------------------------------------------------------------------------------------ attention

void Engine::attention(const Layer& L, int slot, int T, int p0) {
    const Hparams& h = m_.hp();
    Buf& b = *b_;
    const int E = h.n_embd, qd = h.q_dim(), kvd = h.kv_dim(), hd = h.head_dim, nh = h.n_head, nkv = h.n_head_kv;
    const int nrel = h.n_rel, nc = h.n_conv;
    const bool swa = L.swa;
    float* conv = st_.conv[slot].data();
    float* past_k = conv;
    float* past_v = conv + (size_t)(nc - 1) * kvd;
    float* past_a = conv + (size_t)2 * (nc - 1) * kvd;

    // projections (inkling2.cpp: Qcur, Kcur, Vcur, Rcur, sinks, the output gate)
    matmul_f(*pool_, L.wq, b.in0.data(), T, E, b.q.data(), qd, b.qa);
    matmul_f(*pool_, L.wk, b.in1.data(), T, E, b.k.data(), kvd, b.qa);
    matmul_f(*pool_, L.wv, b.in2.data(), T, E, b.v.data(), kvd, b.qb);
    if (!swa) {
        matmul_f(*pool_, L.k_conv_proj, b.in1.data(), T, E, b.kdyn.data(), nc * nkv, b.qa);
        matmul_f(*pool_, L.v_conv_proj, b.in2.data(), T, E, b.vdyn.data(), nc * nkv, b.qb);
    }
    convolve(b.k.data(), T, kvd, L.k_conv, swa ? nullptr : b.kdyn.data(), nkv, past_k);
    convolve(b.v.data(), T, kvd, L.v_conv, swa ? nullptr : b.vdyn.data(), nkv, past_v);
    matmul_f(*pool_, L.wrel, b.in0.data(), T, E, b.r.data(), (int64_t)nrel * nh, b.qa);
    matmul_f(*pool_, L.sink_proj, b.in0.data(), T, E, b.sinks.data(), nh, b.qa);
    matmul_f(*pool_, L.wgate, b.in0.data(), T, E, b.gate.data(), qd, b.qa);

    // clips, the value norm and the log-scaled positions of the full-attention layers
    for (int t = 0; t < T; ++t) {
        float tau = 1.0f;
        if (!swa) {
            float pf = (float)(p0 + t) + 1.0f;
            pf = pf * (1.0f / h.log_floor);
            if (pf < 1.0f) pf = 1.0f;
            tau = std::log(pf) * h.log_alpha + 1.0f;
        }
        for (int hh = 0; hh < nh; ++hh) {
            float* qv = &b.q[(size_t)t * qd + (size_t)hh * hd];
            for (int i = 0; i < hd; ++i) qv[i] *= L.q_clip[(size_t)hh * hd + i];
            l2_clip(qv, hd, h.qk_clip);
            float* rv = &b.r[(size_t)t * nrel * nh + (size_t)hh * nrel];
            for (int i = 0; i < nrel; ++i) rv[i] *= L.rel_clip[(size_t)hh * nrel + i];
            l2_clip(rv, nrel, h.rel_clip);
            if (!swa) {
                for (int i = 0; i < hd; ++i) qv[i] *= tau;
                for (int i = 0; i < nrel; ++i) rv[i] *= tau;
            }
        }
        for (int g = 0; g < nkv; ++g) {
            l2_clip(&b.k[(size_t)t * kvd + (size_t)g * hd], hd, h.qk_clip);
            float* vv = &b.v[(size_t)t * kvd + (size_t)g * hd];
            rms_norm(vv, vv, hd, h.eps);
        }
    }

    dbg("Qcur", slot, b.q.data(), (size_t)T * qd);
    dbg("Kcur", slot, b.k.data(), (size_t)T * kvd);
    dbg("Vcur", slot, b.v.data(), (size_t)T * kvd);
    dbg("Rcur", slot, b.r.data(), (size_t)T * nrel * nh);
    // the cache: f16, the full layers by position, the sliding-window layers in a ring
    const int cap = st_.cap[slot];
    uint16_t* K = st_.k[slot].get();
    uint16_t* V = st_.v[slot].get();
    for (int t = 0; t < T; ++t) {
        const int pos = p0 + t;
        const size_t idx = (size_t)(swa ? pos % cap : pos);
        ggml_cpu_fp32_to_fp16(&b.k[(size_t)t * kvd], K + idx * kvd, kvd);
        ggml_cpu_fp32_to_fp16(&b.v[(size_t)t * kvd], V + idx * kvd, kvd);
    }

    // scores: q.k / sqrt(head_dim) + rel_bias(distance) - sink, softmax with one extra zero logit, times V
    const int extent = swa ? h.n_swa : h.rel_extent;
    const int G = nh / nkv;
    const float scale = 1.0f / std::sqrt((float)hd);
    // a prompt's queries go in blocks of 8 per KV head: the block's keys and values are widened to f32 once, not once per
    // query (each query's arithmetic is the same either way, so the result does not depend on the chunking)
    const int QB = T > 1 ? 8 : 1;
    const int nblk = (T + QB - 1) / QB;
    pool_->run((int64_t)nblk * nkv, 1, [&](int64_t a, int64_t bb) {
        thread_local std::vector<float> sc, rb, krow, vrow, kf, vf;
        for (int64_t job = a; job < bb; ++job) {
          const int blk = (int)(job / nkv), g = (int)(job % nkv);
          const int t0 = blk * QB, t1 = std::min(T, t0 + QB);
          const int ulo = swa ? std::max(0, p0 + t0 - h.n_swa + 1) : 0, uhi = p0 + t1 - 1, un = uhi - ulo + 1;
          const bool widened = t1 - t0 > 1 && un <= 8192;
          krow.resize(hd);
          vrow.resize(hd);
          if (widened) {
              kf.resize((size_t)un * hd);
              vf.resize((size_t)un * hd);
              for (int j = ulo; j <= uhi; ++j) {
                  const size_t row = (size_t)(swa ? j % cap : j) * kvd + (size_t)g * hd;
                  ggml_cpu_fp16_to_fp32(K + row, &kf[(size_t)(j - ulo) * hd], hd);
                  ggml_cpu_fp16_to_fp32(V + row, &vf[(size_t)(j - ulo) * hd], hd);
              }
          }
          auto key_at = [&](int j) -> const float* {
              if (widened) return &kf[(size_t)(j - ulo) * hd];
              ggml_cpu_fp16_to_fp32(K + (size_t)(swa ? j % cap : j) * kvd + (size_t)g * hd, krow.data(), hd);
              return krow.data();
          };
          auto value_at = [&](int j) -> const float* {
              if (widened) return &vf[(size_t)(j - ulo) * hd];
              ggml_cpu_fp16_to_fp32(V + (size_t)(swa ? j % cap : j) * kvd + (size_t)g * hd, vrow.data(), hd);
              return vrow.data();
          };
          for (int t = t0; t < t1; ++t) {
            const int p = p0 + t;
            const int jlo = swa ? std::max(0, p - h.n_swa + 1) : 0;
            const int n = p - jlo + 1;
            sc.resize((size_t)G * n);
            rb.resize((size_t)G * extent);
            for (int gi = 0; gi < G; ++gi) {
                const int hh = g * G + gi;
                const float* rv = &b.r[(size_t)t * nrel * nh + (size_t)hh * nrel];
                const int nd = std::min(extent, n);
                for (int d = 0; d < nd; ++d) {
                    const float* bias = &L.rel_bias[(size_t)d * nrel];
                    float s = 0.0f;
                    for (int i = 0; i < nrel; ++i) s += bias[i] * rv[i];
                    rb[(size_t)gi * extent + d] = s;
                }
            }
            // q stays f32 and each cached key is widened once for the G query heads that share it (llama.cpp's tiled
            // prompt path does the same; its one-token path rounds q to f16 instead)
            for (int jj = 0; jj < n; ++jj) {
                const int j = jlo + jj;
                const float* kr = key_at(j);
                const int d = p - j;
                for (int gi = 0; gi < G; ++gi) {
                    const float s = dot_f32(kr, &b.q[(size_t)t * qd + (size_t)(g * G + gi) * hd], hd);
                    // llama.cpp folds the bias into the mask, which is added after the scale (build_attn, rel_pos)
                    sc[(size_t)gi * n + jj] = s * scale + (d < extent ? rb[(size_t)gi * extent + d] : 0.0f);
                }
            }
            for (int gi = 0; gi < G; ++gi) {
                const int hh = g * G + gi;
                const float sink = b.sinks[(size_t)t * nh + hh];
                float* s = &sc[(size_t)gi * n];
                float mx = 0.0f;   // the zero sink logit takes part in the max
                static const bool f16_mask = std::getenv("STRATA_INKLING_F16_MASK") != nullptr;
                if (f16_mask) {
                    // llama.cpp's flash path: the bias - sink mask in f16 (to see how much of a difference that is)
                    for (int jj = 0; jj < n; ++jj) {
                        const int d = p - (jlo + jj);
                        const float bias = d < extent ? rb[(size_t)gi * extent + d] : 0.0f;
                        const float qk = s[jj] - bias;
                        s[jj] = qk + ggml_fp16_to_fp32(ggml_fp32_to_fp16(bias - sink));
                        mx = std::max(mx, s[jj]);
                    }
                } else {
                    for (int jj = 0; jj < n; ++jj) {
                        s[jj] -= sink;
                        mx = std::max(mx, s[jj]);
                    }
                }
                double sum = std::exp(0.0f - mx);
                for (int jj = 0; jj < n; ++jj) { s[jj] = std::exp(s[jj] - mx); sum += s[jj]; }
                const float inv = (float)(1.0 / sum);
                for (int jj = 0; jj < n; ++jj) s[jj] *= inv;
            }
            float* outs[16];
            for (int gi = 0; gi < G; ++gi) {
                outs[gi] = &b.att[(size_t)t * qd + (size_t)(g * G + gi) * hd];
                std::fill(outs[gi], outs[gi] + hd, 0.0f);
            }
            for (int jj = 0; jj < n; ++jj) {
                const float* vr = value_at(jlo + jj);
                for (int gi = 0; gi < G; ++gi) {
                    const float pw = sc[(size_t)gi * n + jj];
                    float* o = outs[gi];
                    for (int i = 0; i < hd; ++i) o[i] += pw * vr[i];
                }
            }
          }
        }
    });

    // gate, output projection, the post-attention convolution
    dbg("kqv_out", slot, b.att.data(), (size_t)T * qd);
    for (size_t i = 0; i < (size_t)T * qd; ++i) b.att[i] *= sigmoidf(b.gate[i]);
    matmul_f(*pool_, L.wo, b.att.data(), T, qd, b.o.data(), E, b.qa);
    dbg("wo_out", slot, b.o.data(), (size_t)T * E);
    convolve(b.o.data(), T, E, L.attn_post_conv, nullptr, 0, past_a);
    dbg("attn_out", slot, b.o.data(), (size_t)T * E);
}

// ------------------------------------------------------------------------------------------------ experts

void Engine::prefetch_guess(int layer, int T) {
    // the experts this layer will most likely pick, from its input before attention: read them while attention runs
    const Hparams& h = m_.hp();
    const Layer& L = m_.layers()[layer];
    Buf& b = *b_;
    const int E = h.n_embd, NE = h.n_expert, K = h.n_expert_used;
    std::vector<float> xg((size_t)E), lg((size_t)NE);
    std::vector<int> want;
    for (int t = 0; t < T; ++t) {
        rms_norm(&b.cur[(size_t)t * E], xg.data(), E, h.eps, L.ffn_norm.data());
        QAct q;
        quantize_acts(xg.data(), 1, E, E, L.router.type, q);
        dot_rows(L.router, q, 0, NE, lg.data(), NE);
        std::vector<int> idx(NE);
        std::iota(idx.begin(), idx.end(), 0);
        std::vector<float> sel(NE);
        for (int e = 0; e < NE; ++e) sel[e] = sigmoidf(lg[e]) + (L.exp_probs_b.empty() ? 0.0f : L.exp_probs_b[e]);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int c) { return sel[a] > sel[c]; });
        want.insert(want.end(), idx.begin(), idx.begin() + K);
    }
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    store_.prefetch(layer, want.data(), (int)want.size());
}

// MoE stage times (STRATA_INKLING_PROFILE=1 prints them at exit)
static double g_moe_t[6];
static const bool g_prof = std::getenv("STRATA_INKLING_PROFILE") != nullptr;
struct ProfPrint {
    ~ProfPrint() {
        if (g_prof)
            std::fprintf(stderr, "PROFILE moe: router %.2f s, gate_up %.2f s, situ %.2f s, quant %.2f s, down %.2f s, sum %.2f s\n",
                         g_moe_t[0], g_moe_t[1], g_moe_t[2], g_moe_t[3], g_moe_t[4], g_moe_t[5]);
    }
} g_prof_print;

void Engine::moe(int layer, int T) {
    const Hparams& h = m_.hp();
    auto tp = clk::now();
    auto lap = [&](int i) {
        if (!g_prof) return;
        auto n = clk::now();
        g_moe_t[i] += secs(n - tp);
        tp = n;
    };
    const Layer& L = m_.layers()[layer];
    Buf& b = *b_;
    const int E = h.n_embd, NE = h.n_expert, K = h.n_expert_used, F = h.n_ff_exp;

    // router: sigmoid probabilities, top-k on probability + bias, weights normalized (+1e-20) and scaled
    matmul_f(*pool_, L.router, b.x.data(), T, E, b.logits.data(), NE, b.qa);
    std::vector<int> idx(NE);
    std::vector<float> prob(NE), sel(NE);
    for (int t = 0; t < T; ++t) {
        for (int e = 0; e < NE; ++e) {
            prob[e] = sigmoidf(b.logits[(size_t)t * NE + e]);
            sel[e] = prob[e] + (L.exp_probs_b.empty() ? 0.0f : L.exp_probs_b[e]);
        }
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int c) {
            return sel[a] > sel[c] || (sel[a] == sel[c] && a < c);
        });
        if (g_debug) {
            static const int dt = std::getenv("STRATA_INKLING_DEBUG_TOKEN") ? std::atoi(std::getenv("STRATA_INKLING_DEBUG_TOKEN")) : 0;
            if (t == dt) {
                std::partial_sort(idx.begin() + K, idx.begin() + K + 1, idx.end(), [&](int a, int c) { return sel[a] > sel[c]; });
                std::fprintf(stderr, "DBG margin-%d: 8th %d %.7f 9th %d %.7f\n", layer, idx[K - 1], sel[idx[K - 1]], idx[K], sel[idx[K]]);
            }
        }
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            b.ids[(size_t)t * K + k] = idx[k];
            b.wts[(size_t)t * K + k] = prob[idx[k]];
            sum += prob[idx[k]];
        }
        sum += 1e-20f;
        for (int k = 0; k < K; ++k) b.wts[(size_t)t * K + k] = b.wts[(size_t)t * K + k] / sum * h.expert_scale;
    }

    dbg("ffn_moe_logits", layer, b.logits.data(), (size_t)T * NE);
    dbg("ffn_moe_weights_scaled", layer, b.wts.data(), (size_t)T * K);
    if (g_debug) {
        // STRATA_INKLING_DEBUG_TOKEN=t: the experts of token t of the chunk (default 0)
        static const int dt = std::getenv("STRATA_INKLING_DEBUG_TOKEN") ? std::atoi(std::getenv("STRATA_INKLING_DEBUG_TOKEN")) : 0;
        if (dt < T) {
            std::fprintf(stderr, "DBG topk-%d:", layer);
            for (int k = 0; k < K; ++k) std::fprintf(stderr, " %d", b.ids[(size_t)dt * K + k]);
            std::fprintf(stderr, "\n");
        }
    }
    // the experts this chunk needs, each with the (token, rank) pairs routed to it
    std::vector<std::vector<int>> by_expert(NE);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) by_expert[b.ids[(size_t)t * K + k]].push_back(t * K + k);
    std::vector<int> needed;
    for (int e = 0; e < NE; ++e)
        if (!by_expert[e].empty()) needed.push_back(e);

    quantize_acts(b.x.data(), T, E, E, L.gate_up.type, b.qa, pool_.get());
    lap(0);
    // a prompt's experts come in waves; with a small cache, half the pinnable slots hold the wave being computed and
    // the other half the next one, read meanwhile
    int wave_max = std::max(1, store_.max_pinned());
    if (store_.mode() == StoreMode::Stream && (int)needed.size() > wave_max) wave_max = std::max(1, wave_max / 2);
    if (T > 1) store_.prefetch(layer, needed.data(), std::min((int)needed.size(), wave_max));
    std::vector<ExpertView> views;
    const int R1 = 2 * F;
    for (size_t w0 = 0; w0 < needed.size(); w0 += (size_t)wave_max) {
        const int nw = (int)std::min<size_t>((size_t)wave_max, needed.size() - w0);
        const int* wids = &needed[w0];
        views.resize(nw);
        store_.acquire(layer, wids, nw, views.data());
        if (w0 + nw < needed.size())
            store_.prefetch(layer, wids + nw, (int)std::min<size_t>((size_t)wave_max, needed.size() - w0 - nw));
        // assignments of this wave, grouped by expert
        std::vector<int> off(nw + 1, 0), asg;
        for (int u = 0; u < nw; ++u) {
            off[u + 1] = off[u] + (int)by_expert[wids[u]].size();
            asg.insert(asg.end(), by_expert[wids[u]].begin(), by_expert[wids[u]].end());
        }
        const int na = off[nw];
        if (b.gu.size() < (size_t)na * R1) b.gu.resize((size_t)na * R1);
        if (b.act.size() < (size_t)na * F) b.act.resize((size_t)na * F);

        // gate_up for every (expert, row), all of the expert's tokens at once
        {
            const ggml_vec_dot_t vd = vec_dot_fn(L.gate_up.type);
            // small pieces, claimed as threads come free: experts differ a lot in how many tokens they carry
            const int64_t total = (int64_t)nw * R1;
            const int64_t grain = T > 1 ? 64 : std::max<int64_t>(16, (total / (8 * pool_->size()) + 15) / 16 * 16);
            pool_->run(total, grain, [&](int64_t a, int64_t c) {
                // the piece may span experts: one tiled run per expert it covers
                for (int64_t i = a; i < c;) {
                    const int u = (int)(i / R1);
                    const int64_t r0 = i % R1, r1 = std::min<int64_t>(R1, r0 + (c - i));
                    const int a0 = off[u];
                    tiled_dots(vd, (int)E, views[u].gate_up, r0, r1, off[u + 1] - a0,
                               [&](int k) { return b.qa.row(asg[a0 + k] / K); },
                               [&](int k, int64_t r, float s) { b.gu[(size_t)(a0 + k) * R1 + r] = s; });
                    i += r1 - r0;
                }
            });
        }
        lap(1);
        // the per-expert scale, then situ(gate, up) = beta tanh(gate/beta) sigmoid(gate) * lb tanh(up/lb)
        {
            const float beta = h.situ_beta, ib = 1.0f / beta, lb = h.situ_linear_beta, il = lb > 0 ? 1.0f / lb : 0.0f;
            pool_->run(na, 1, [&](int64_t a, int64_t c) {
                for (int64_t ai = a; ai < c; ++ai) {
                    int u = 0;
                    while (off[u + 1] <= ai) ++u;
                    const float s = L.gate_up.scale.empty() ? 1.0f : L.gate_up.scale[wids[u]];
                    const float* g = &b.gu[(size_t)ai * R1];
                    float* o = &b.act[(size_t)ai * F];
                    for (int i = 0; i < F; ++i) {
                        const float gv = g[i] * s, uv = g[F + i] * s;
                        float act = std::tanh(gv * ib) * beta;
                        act = act * sigmoidf(gv);
                        const float up = lb > 0.0f ? std::tanh(uv * il) * lb : uv;
                        o[i] = act * up;
                    }
                }
            });
        }
        if (g_debug && needed.size() <= (size_t)wave_max) {
            double sg = 0;
            for (int ai = 0; ai < na; ++ai) {
                int u = 0;
                while (off[u + 1] <= ai) ++u;
                const float sc = L.gate_up.scale.empty() ? 1.0f : L.gate_up.scale[wids[u]];
                for (int r = 0; r < R1; ++r) sg += b.gu[(size_t)ai * R1 + r] * sc;
            }
            std::fprintf(stderr, "DBG %-24s sum=%.6f\n", ("ffn_moe_gate_up_scaled-" + std::to_string(layer)).c_str(), sg);
            dbg("ffn_moe_situ", layer, b.act.data(), (size_t)na * F);
        }
        lap(2);
        quantize_acts(b.act.data(), na, F, F, L.down.type, b.qb, pool_.get());
        lap(3);
        // down for every (expert, row), scaled, into the token's rank slot
        {
            const ggml_vec_dot_t vd = vec_dot_fn(L.down.type);
            const int64_t total = (int64_t)nw * E;
            const int64_t grain = T > 1 ? 64 : std::max<int64_t>(16, (total / (8 * pool_->size()) + 15) / 16 * 16);
            pool_->run(total, grain, [&](int64_t a, int64_t c) {
                for (int64_t i = a; i < c;) {
                    const int u = (int)(i / E);
                    const int64_t r0 = i % E, r1 = std::min<int64_t>(E, r0 + (c - i));
                    const int a0 = off[u];
                    const float s = L.down.scale.empty() ? 1.0f : L.down.scale[wids[u]];
                    tiled_dots(vd, (int)F, views[u].down, r0, r1, off[u + 1] - a0,
                               [&](int k) { return b.qb.row(a0 + k); },
                               [&](int k, int64_t r, float d) { b.dnk[(size_t)asg[a0 + k] * E + r] = d * s; });
                    i += r1 - r0;
                }
            });
        }
        store_.release(layer, wids, nw);
        lap(4);
    }

    dbg("ffn_moe_down_scaled", layer, b.dnk.data(), (size_t)T * K * E);
    // weighted sum in rank order (llama.cpp adds the experts' views one after the other)
    pool_->run(T, 1, [&](int64_t a, int64_t c) {
        for (int64_t t = a; t < c; ++t) {
            float* y = &b.y[(size_t)t * E];
            for (int k = 0; k < K; ++k) {
                const float w = b.wts[(size_t)t * K + k];
                const float* d = &b.dnk[((size_t)t * K + k) * E];
                if (k == 0) for (int e = 0; e < E; ++e) y[e] = d[e] * w;
                else for (int e = 0; e < E; ++e) y[e] += d[e] * w;
            }
        }
    });
    lap(5);
}

// ------------------------------------------------------------------------------------------------ one layer

static void states_of(const std::vector<std::vector<float>>& blocks, const float* cur, std::vector<const float*>& out) {
    out.clear();
    for (size_t i = blocks.size(); i-- > 0;) out.push_back(blocks[i].data());
    out.push_back(cur);
}

void Engine::apply(const Application& app, int T, std::vector<std::vector<float>>& blocks, const float* msgs) {
    const Hparams& h = m_.hp();
    const Layer& L = m_.layers()[app.phys];
    Buf& b = *b_;
    const int E = h.n_embd;
    const size_t TE = (size_t)T * E;
    const LoopLayer* LL = msgs ? &m_.loop_layers()[app.phys - h.core_start] : nullptr;
    float* conv = st_.conv[app.slot].data();
    float* past_f = conv + (size_t)2 * (h.n_conv - 1) * h.kv_dim() + (size_t)(h.n_conv - 1) * E;

    auto t0 = clk::now();
    if (opt_.lookahead && T == 1 && store_.mode() == StoreMode::Stream) prefetch_guess(app.phys, T);
    std::memcpy(b.inpSA.data(), b.cur.data(), TE * sizeof(float));

    // attention inputs: the SWA layers mix one, the full-attention layers one each for Q, K and V
    std::vector<const float*> states;
    if (L.n_attn_res_query > 0) {
        states_of(blocks, b.cur.data(), states);
        mix(states, T, L.attn_res_query.data(), L.n_attn_res_query, b.mixout.data());
        const bool one = L.n_attn_res_query == 1;
        std::memcpy(b.in0.data(), b.mixout.data(), TE * sizeof(float));
        std::memcpy(b.in1.data(), b.mixout.data() + (one ? 0 : TE), TE * sizeof(float));
        std::memcpy(b.in2.data(), b.mixout.data() + (one ? 0 : 2 * TE), TE * sizeof(float));
    } else {
        std::memcpy(b.in0.data(), b.cur.data(), TE * sizeof(float));
        std::memcpy(b.in1.data(), b.cur.data(), TE * sizeof(float));
        std::memcpy(b.in2.data(), b.cur.data(), TE * sizeof(float));
    }
    float* ins[3] = {b.in0.data(), b.in1.data(), b.in2.data()};
    if (msgs) {
        for (int j = 0; j < 3; ++j) {
            control(*LL, j, ins[j], msgs + (size_t)j * TE, T, b.ffn_inp.data());
            std::memcpy(ins[j], b.ffn_inp.data(), TE * sizeof(float));
        }
    }
    if (L.swa && msgs) {
        for (int j = 0; j < 3; ++j)
            for (int t = 0; t < T; ++t) rms_norm(ins[j] + (size_t)t * E, ins[j] + (size_t)t * E, E, h.eps, L.attn_norm.data());
    } else if (L.swa) {
        for (int t = 0; t < T; ++t) rms_norm(ins[0] + (size_t)t * E, ins[0] + (size_t)t * E, E, h.eps, L.attn_norm.data());
        std::memcpy(ins[1], ins[0], TE * sizeof(float));
        std::memcpy(ins[2], ins[0], TE * sizeof(float));
    } else {
        const std::vector<float>* g[3] = {&L.q_inp_norm, &L.k_inp_norm, &L.v_inp_norm};
        for (int j = 0; j < 3; ++j)
            for (int t = 0; t < T; ++t) rms_norm(ins[j] + (size_t)t * E, ins[j] + (size_t)t * E, E, h.eps, g[j]->data());
    }
    auto t1 = clk::now();
    attention(L, app.slot, T, st_.n_past);
    auto t2 = clk::now();

    // the first layer of each block starts a new residual stream; the old one joins the mixed blocks
    if (app.phys % h.res_block == 0) {
        blocks.emplace_back(b.inpSA.begin(), b.inpSA.begin() + TE);
        std::memcpy(b.cur.data(), b.o.data(), TE * sizeof(float));
    } else {
        for (size_t i = 0; i < TE; ++i) b.cur[i] = b.inpSA[i] + b.o[i];
    }
    std::memcpy(b.ffn_inp.data(), b.cur.data(), TE * sizeof(float));
    dbg("ffn_inp", app.slot, b.ffn_inp.data(), TE);

    // MoE input: mixed over the depth, the cross-pass message, the norm
    states_of(blocks, b.cur.data(), states);
    mix(states, T, L.ffn_res_query.data(), 1, b.x.data());
    if (msgs) {
        control(*LL, 3, b.x.data(), msgs + (size_t)3 * TE, T, b.tmp.data());
        std::memcpy(b.x.data(), b.tmp.data(), TE * sizeof(float));
    }
    for (int t = 0; t < T; ++t) rms_norm(&b.x[(size_t)t * E], &b.x[(size_t)t * E], E, h.eps, L.ffn_norm.data());
    dbg("ffn_norm_in", app.slot, b.x.data(), TE);
    auto t3 = clk::now();
    moe(app.phys, T);
    auto t4 = clk::now();
    dbg("ffn_moe_out", app.slot, b.y.data(), TE);
    convolve(b.y.data(), T, E, L.ffn_post_conv, nullptr, 0, past_f);
    dbg("ffn_out", app.slot, b.y.data(), TE);
    for (size_t i = 0; i < TE; ++i) b.cur[i] = b.ffn_inp[i] + b.y[i];
    dbg("l_out", app.slot, b.cur.data(), TE);
    auto t5 = clk::now();
    stats_.dense_s += secs(t1 - t0) + secs(t3 - t2) + secs(t5 - t4);
    stats_.attn_s += secs(t2 - t1);
    stats_.moe_s += secs(t4 - t3);
}

// ------------------------------------------------------------------------------------------------ the chunk

void Engine::head(int T, const std::vector<std::vector<float>>& blocks, std::vector<float>* last, std::vector<float>* all,
                  int all_row0) {
    const Hparams& h = m_.hp();
    Buf& b = *b_;
    const int E = h.n_embd, NV = h.n_vocab;
    auto t0 = clk::now();
    const int first = all ? 0 : T - 1;
    const int n = T - first;
    std::vector<const float*> states;
    states_of(blocks, b.cur.data(), states);
    mix(states, T, m_.output_res_query().data(), 1, b.x.data());
    for (int t = first; t < T; ++t) rms_norm(&b.x[(size_t)t * E], &b.x[(size_t)t * E], E, h.eps, m_.output_norm().data());
    // the head's rows past the vocabulary are padding: only the first n_vocab are computed
    Mat W = m_.output();
    W.rows = NV;
    std::vector<float> lg((size_t)n * NV);
    matmul_f(*pool_, W, &b.x[(size_t)first * E], n, E, lg.data(), NV, b.qa);
    if (h.softcap != 0.0f) {
        const float cap = h.softcap, ic = 1.0f / cap;
        for (float& v : lg) v = std::tanh(v * ic) * cap;
    }
    if (all) std::memcpy(all->data() + (size_t)all_row0 * NV, lg.data(), lg.size() * sizeof(float));
    if (last) last->assign(lg.end() - NV, lg.end());
    stats_.head_s += secs(clk::now() - t0);
}

void Engine::run_chunk(const int* tokens, int T, std::vector<float>* last, std::vector<float>* all, int all_row0) {
    const Hparams& h = m_.hp();
    Buf& b = *b_;
    const int E = h.n_embd;
    const size_t TE = (size_t)T * E;
    b.T = T;
    for (int t = 0; t < T; ++t) {
        float* c = &b.cur[(size_t)t * E];
        m_.embed_row(tokens[t], c);
    }
    dbg("embd", -1, b.cur.data(), TE);
    for (int t = 0; t < T; ++t) {
        float* c = &b.cur[(size_t)t * E];
        rms_norm(c, c, E, h.eps, m_.tok_norm().data());
    }
    dbg("node_12", -1, b.cur.data(), TE);
    std::vector<std::vector<float>> blocks, prelude_blocks, bank;
    std::vector<float> bank_out;
    int last_pass = -1;
    for (const Application& app : m_.schedule()) {
        const bool core = h.looped() && app.phys >= h.core_start && app.phys < h.core_end;
        if (core && app.phys == h.core_start) {
            if (app.pass == 0) {
                // every pass starts from the prelude output
                std::memcpy(b.prelude.data(), b.cur.data(), TE * sizeof(float));
                prelude_blocks = blocks;
            } else {
                // the bank of the previous pass: its blocks after the first core block, and its output
                bank.assign(blocks.begin() + (ptrdiff_t)prelude_blocks.size() + 1, blocks.end());
                bank_out.assign(b.cur.begin(), b.cur.begin() + TE);
                std::memcpy(b.cur.data(), b.prelude.data(), TE * sizeof(float));
                blocks = prelude_blocks;
            }
            last_pass = app.pass;
        }
        const float* msgs = nullptr;
        if (core && app.pass > 0) {
            std::vector<const float*> states;
            states_of(bank, bank_out.data(), states);
            mix(states, T, m_.loop_layers()[app.phys - h.core_start].msg_query.data(), 4, b.msgs.data());
            msgs = b.msgs.data();
        }
        apply(app, T, blocks, msgs);
    }
    (void)last_pass;
    if (last || all) head(T, blocks, last, all, all_row0);
    st_.n_past += T;
    stats_.tokens += (uint64_t)T;
}

void Engine::eval(const int* tokens, int n, std::vector<float>* last, std::vector<float>* all) {
    if (n <= 0) return;
    if (st_.n_past + n > opt_.n_ctx)
        throw std::runtime_error("the context is full (" + std::to_string(st_.n_past + n) + " > " + std::to_string(opt_.n_ctx) + " tokens)");
    for (int i = 0; i < n; ++i)
        if (tokens[i] < 0 || tokens[i] >= m_.hp().n_vocab) throw std::runtime_error("token id " + std::to_string(tokens[i]) + " is out of range");
    if (all) all->resize((size_t)n * m_.hp().n_vocab);
    for (int i = 0; i < n; i += opt_.chunk) {
        const int T = std::min(opt_.chunk, n - i);
        const bool is_last = i + T == n;
        run_chunk(tokens + i, T, is_last ? last : nullptr, all, i);
    }
}

}  // namespace strata::inkling
