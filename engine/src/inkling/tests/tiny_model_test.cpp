// src/inkling/tests/tiny_model_test.cpp - the Inkling engine on a tiny random model it writes itself (no download, no GPU).
//
// The model has the real architecture at toy sizes: a prelude, a 3-pass loop core and a coda, sliding-window and
// full-attention layers, relative-position bias, short convolutions, depth mixing, the cross-pass controllers and a
// sigmoid-routed MoE with per-expert scales. It is written twice, in the root ("inkling2") and the Mainline_Support
// ("inkling2_loop") layout. The engine computes every token's dot products the same way whatever the chunking or
// where the experts are, so these must hold bit for bit:
//   * a prompt in chunks of 5 = the same prompt one token at a time = the prompt in one chunk
//     (this crosses the sliding window several times, so the ring buffer and the conv states are exercised);
//   * the root file = the mainline file (the loop tensors' name and order mapping);
//   * experts streamed through a 4-slot cache with direct reads = experts in RAM = experts memory-mapped;
//   * rollback to a snapshot, then other tokens = a fresh run of the same tokens.
#include "engine.hpp"
#include "expert_store.hpp"
#include "model.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace strata::inkling;

namespace {

int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------- a minimal GGUF writer

struct KV {
    std::string key;
    uint32_t type;
    std::vector<uint8_t> bytes;
};

struct TensorOut {
    std::string name;
    std::vector<int64_t> ne;
    int type;
    std::vector<uint8_t> data;
};

class Writer {
public:
    void u32(const std::string& k, uint32_t v) { put(k, 4, &v, 4); }
    void f32(const std::string& k, float v) { put(k, 6, &v, 4); }
    void str(const std::string& k, const std::string& v) {
        KV kv{k, 8, {}};
        uint64_t n = v.size();
        app(kv.bytes, &n, 8);
        app(kv.bytes, v.data(), v.size());
        kvs.push_back(kv);
    }
    void bools(const std::string& k, const std::vector<bool>& v) {
        KV kv{k, 9, {}};
        uint32_t et = 7;
        uint64_t n = v.size();
        app(kv.bytes, &et, 4);
        app(kv.bytes, &n, 8);
        for (bool b : v) kv.bytes.push_back(b ? 1 : 0);
        kvs.push_back(kv);
    }
    void ints(const std::string& k, const std::vector<int32_t>& v) {
        KV kv{k, 9, {}};
        uint32_t et = 5;
        uint64_t n = v.size();
        app(kv.bytes, &et, 4);
        app(kv.bytes, &n, 8);
        app(kv.bytes, v.data(), v.size() * 4);
        kvs.push_back(kv);
    }
    void tensor(const std::string& name, std::vector<int64_t> ne, int type, const std::vector<float>& vals) {
        TensorOut t{name, ne, type, {}};
        const int64_t row = ne[0];
        int64_t n = 1;
        for (int64_t d : ne) n *= d;
        if ((int64_t)vals.size() != n) { std::fprintf(stderr, "bad tensor %s\n", name.c_str()); std::exit(2); }
        if (type == GGML_TYPE_F32) {
            t.data.resize(n * 4);
            std::memcpy(t.data.data(), vals.data(), n * 4);
        } else {
            t.data.resize(ggml_row_size((ggml_type)type, row) * (n / row));
            ggml_quantize_chunk((ggml_type)type, vals.data(), t.data.data(), 0, n / row, row, nullptr);
        }
        ts.push_back(std::move(t));
    }
    void write(const std::string& path) {
        std::vector<uint8_t> out;
        app(out, "GGUF", 4);
        uint32_t ver = 3;
        uint64_t nt = ts.size(), nk = kvs.size();
        app(out, &ver, 4);
        app(out, &nt, 8);
        app(out, &nk, 8);
        for (const KV& kv : kvs) {
            uint64_t n = kv.key.size();
            app(out, &n, 8);
            app(out, kv.key.data(), n);
            app(out, &kv.type, 4);
            app(out, kv.bytes.data(), kv.bytes.size());
        }
        uint64_t off = 0;
        std::vector<uint64_t> offs;
        for (const TensorOut& t : ts) {
            uint64_t n = t.name.size();
            app(out, &n, 8);
            app(out, t.name.data(), n);
            uint32_t nd = (uint32_t)t.ne.size();
            app(out, &nd, 4);
            for (int64_t d : t.ne) { uint64_t u = (uint64_t)d; app(out, &u, 8); }
            uint32_t ty = (uint32_t)t.type;
            app(out, &ty, 4);
            app(out, &off, 8);
            offs.push_back(off);
            off += (t.data.size() + 31) / 32 * 32;
        }
        while (out.size() % 32) out.push_back(0);
        for (const TensorOut& t : ts) {
            app(out, t.data.data(), t.data.size());
            while (out.size() % 32) out.push_back(0);
        }
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); std::exit(2); }
        std::fwrite(out.data(), 1, out.size(), f);
        std::fclose(f);
    }

private:
    static void app(std::vector<uint8_t>& v, const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        v.insert(v.end(), b, b + n);
    }
    void put(const std::string& k, uint32_t type, const void* p, size_t n) {
        KV kv{k, type, {}};
        app(kv.bytes, p, n);
        kvs.push_back(kv);
    }
    std::vector<KV> kvs;
    std::vector<TensorOut> ts;
};

// ---------------------------------------------------------------- the tiny model

constexpr int E = 64, NH = 4, NKV = 2, HD = 16, FF = 32, NE = 8, K = 2, NL = 12, NREL = 4, SWA = 8, EXT = 8, NC = 4;
constexpr int BLOCK = 3, CORE0 = 3, CORE1 = 9, PASSES = 3, RANK = 8, VOCAB = 50, VPAD = 64;

bool is_full(int l) { return l % 3 == 2; }

std::map<std::string, std::vector<float>> make_weights(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::map<std::string, std::vector<float>> w;
    auto r = [&](size_t n, float sd, float mean = 0.0f) {
        std::vector<float> v(n);
        for (float& x : v) x = mean + sd * nd(rng);
        return v;
    };
    auto L = [](int i, const char* n) { return "blk." + std::to_string(i) + "." + n; };
    w["token_embd.weight"] = r((size_t)E * VPAD, 1.0f);
    w["token_embd_norm.weight"] = r(E, 0.1f, 1.0f);
    w["output.weight"] = r((size_t)E * VPAD, 0.3f);
    w["output_norm.weight"] = r(E, 0.1f, 1.0f);
    w["output_res_query.weight"] = r(E, 0.5f);
    for (int i = 0; i < NL; ++i) {
        const bool full = is_full(i);
        w[L(i, "attn_q.weight")] = r((size_t)E * NH * HD, 0.3f);
        w[L(i, "attn_k.weight")] = r((size_t)E * NKV * HD, 0.3f);
        w[L(i, "attn_v.weight")] = r((size_t)E * NKV * HD, 0.3f);
        w[L(i, "attn_output.weight")] = r((size_t)NH * HD * E, 0.2f);
        w[L(i, "attn_gate.weight")] = r((size_t)E * NH * HD, 0.3f);
        w[L(i, "attn_q_clip.weight")] = r((size_t)HD * NH, 0.2f, 1.0f);
        w[L(i, "attn_sink_proj.weight")] = r((size_t)E * NH, 0.2f);
        w[L(i, "attn_rel.weight")] = r((size_t)E * NREL * NH, 0.3f);
        w[L(i, "attn_rel_bias.weight")] = r((size_t)NREL * (full ? EXT : SWA), 0.5f);
        w[L(i, "attn_rel_clip.weight")] = r((size_t)NREL * NH, 0.2f, 1.0f);
        w[L(i, "attn_k_conv.weight")] = r((size_t)NC * NKV * HD, 0.3f);
        w[L(i, "attn_v_conv.weight")] = r((size_t)NC * NKV * HD, 0.3f);
        w[L(i, "attn_post_conv.weight")] = r((size_t)NC * E, 0.3f);
        w[L(i, "ffn_post_conv.weight")] = r((size_t)NC * E, 0.3f);
        if (full) {
            w[L(i, "attn_q_inp_norm.weight")] = r(E, 0.1f, 1.0f);
            w[L(i, "attn_k_inp_norm.weight")] = r(E, 0.1f, 1.0f);
            w[L(i, "attn_v_inp_norm.weight")] = r(E, 0.1f, 1.0f);
            w[L(i, "attn_k_conv_proj.weight")] = r((size_t)E * NC * NKV, 0.3f);
            w[L(i, "attn_v_conv_proj.weight")] = r((size_t)E * NC * NKV, 0.3f);
        } else {
            w[L(i, "attn_norm.weight")] = r(E, 0.1f, 1.0f);
        }
        if (i > 0) w[L(i, "attn_res_query.weight")] = r((size_t)E * (full ? 3 : 1), 0.5f);
        w[L(i, "ffn_res_query.weight")] = r(E, 0.5f);
        w[L(i, "ffn_norm.weight")] = r(E, 0.1f, 1.0f);
        w[L(i, "ffn_gate_inp.weight")] = r((size_t)E * NE, 0.5f);
        w[L(i, "exp_probs_b.bias")] = r(NE, 0.1f);
        w[L(i, "ffn_gate_up_exps.weight")] = r((size_t)E * 2 * FF * NE, 0.4f);
        w[L(i, "ffn_down_exps.weight")] = r((size_t)FF * E * NE, 0.4f);
        w[L(i, "ffn_gate_up_exps.scale")] = r(NE, 0.1f, 1.0f);
        w[L(i, "ffn_down_exps.scale")] = r(NE, 0.1f, 1.0f);
    }
    for (int i = CORE0; i < CORE1; ++i) {
        w[L(i, "msg_query")] = r((size_t)4 * E, 0.5f);   // [4][E]: q, k, v, ffn (written per layout below)
        for (const char* c : {"q", "k", "v", "ffn"}) {
            w[L(i, (std::string("ctl_") + c + "_down").c_str())] = r((size_t)2 * E * RANK, 0.2f);
            w[L(i, (std::string("ctl_") + c + "_gain").c_str())] = r((size_t)RANK * E, 0.2f);
            w[L(i, (std::string("ctl_") + c + "_forget").c_str())] = r((size_t)RANK * E, 0.2f);
        }
    }
    return w;
}

void write_model(const std::string& path, std::map<std::string, std::vector<float>>& w, bool root) {
    Writer g;
    const std::string P = root ? "inkling2." : "inkling2_loop.";
    const int apps = NL + (PASSES - 1) * (CORE1 - CORE0);
    g.str("general.architecture", root ? "inkling2" : "inkling2_loop");
    g.u32(P + "block_count", root ? apps : NL);
    g.u32(P + "context_length", 4096);
    g.u32(P + "embedding_length", E);
    g.u32(P + "feed_forward_length", FF);
    g.ints(P + "attention.head_count", std::vector<int32_t>(root ? apps : NL, NH));
    g.ints(P + "attention.head_count_kv", std::vector<int32_t>(root ? apps : NL, NKV));
    g.f32(P + "attention.layer_norm_rms_epsilon", 1e-6f);
    g.u32(P + "expert_count", NE);
    g.u32(P + "expert_used_count", K);
    g.u32(P + "attention.key_length", HD);
    g.u32(P + "attention.value_length", HD);
    if (root) {
        g.u32(P + "num_loops", PASSES);
        g.u32(P + "loop.core_start", CORE0);
        g.u32(P + "loop.core_end", CORE1);
    } else {
        g.u32(P + "loop.count", PASSES);
        g.u32(P + "loop.start", CORE0);
        g.u32(P + "loop.end", CORE1);
    }
    g.u32(P + "loop.gate_rank", RANK);
    g.u32(P + "vocab_size", VPAD);
    g.u32(P + "vocab_size_unpadded", VOCAB);
    g.u32(P + "expert_feed_forward_length", FF);
    g.u32(P + "attention.sliding_window", SWA);
    std::vector<bool> pat;
    for (int i = 0; i < (root ? apps : NL); ++i) pat.push_back(!is_full(i % NL));
    g.bools(P + "attention.sliding_window_pattern", pat);
    g.u32(P + "shortconv.l_cache", NC);
    g.f32(P + "expert_weights_scale", 2.0f);
    g.f32(P + "activation.situ_beta", 4.0f);
    g.f32(P + "activation.situ_linear_beta", 25.0f);
    g.u32(P + "attn_res.block_size", BLOCK);
    g.f32(P + "final_logit_softcapping", 30.0f);
    g.u32(P + "attention.relative_length", NREL);
    g.u32(P + "attention.relative_extent", EXT);
    g.f32(P + "attention.qk_clip", 2.5f);      // small, so that the clips act
    g.f32(P + "attention.relative_clip", 1.5f);
    g.u32(P + "attention.log_scale_floor", 4);  // small, so that the log scale acts on a short prompt
    g.f32(P + "attention.log_scale_alpha", 0.1f);
    g.u32("tokenizer.ggml.eos_token_id", 7);

    auto L = [](int i, const char* n) { return "blk." + std::to_string(i) + "." + n; };
    auto T = [&](const std::string& name, std::vector<int64_t> ne, int type) { g.tensor(name, ne, type, w.at(name)); };
    T("token_embd.weight", {E, VPAD}, GGML_TYPE_Q8_0);
    T("token_embd_norm.weight", {E}, GGML_TYPE_F32);
    T("output.weight", {E, VPAD}, GGML_TYPE_Q8_0);
    T("output_norm.weight", {E}, GGML_TYPE_F32);
    T("output_res_query.weight", {E}, GGML_TYPE_F32);
    for (int i = 0; i < NL; ++i) {
        const bool full = is_full(i);
        T(L(i, "attn_q.weight"), {E, NH * HD}, GGML_TYPE_Q8_0);
        T(L(i, "attn_k.weight"), {E, NKV * HD}, GGML_TYPE_Q8_0);
        T(L(i, "attn_v.weight"), {E, NKV * HD}, GGML_TYPE_F32);
        T(L(i, "attn_output.weight"), {NH * HD, E}, GGML_TYPE_Q8_0);
        T(L(i, "attn_gate.weight"), {E, NH * HD}, GGML_TYPE_F32);
        T(L(i, "attn_q_clip.weight"), {HD, NH}, GGML_TYPE_F32);
        T(L(i, "attn_sink_proj.weight"), {E, NH}, GGML_TYPE_F32);
        T(L(i, "attn_rel.weight"), {E, NREL * NH}, GGML_TYPE_F32);
        T(L(i, "attn_rel_bias.weight"), {NREL, full ? EXT : SWA}, GGML_TYPE_F32);
        T(L(i, "attn_rel_clip.weight"), {NREL, NH}, GGML_TYPE_F32);
        T(L(i, "attn_k_conv.weight"), {NC, NKV * HD}, GGML_TYPE_F32);
        T(L(i, "attn_v_conv.weight"), {NC, NKV * HD}, GGML_TYPE_F32);
        T(L(i, "attn_post_conv.weight"), {NC, E}, GGML_TYPE_F32);
        T(L(i, "ffn_post_conv.weight"), {NC, E}, GGML_TYPE_F32);
        if (full) {
            T(L(i, "attn_q_inp_norm.weight"), {E}, GGML_TYPE_F32);
            T(L(i, "attn_k_inp_norm.weight"), {E}, GGML_TYPE_F32);
            T(L(i, "attn_v_inp_norm.weight"), {E}, GGML_TYPE_F32);
            T(L(i, "attn_k_conv_proj.weight"), {E, NC * NKV}, GGML_TYPE_F32);
            T(L(i, "attn_v_conv_proj.weight"), {E, NC * NKV}, GGML_TYPE_F32);
        } else {
            T(L(i, "attn_norm.weight"), {E}, GGML_TYPE_F32);
        }
        if (i > 0) T(L(i, "attn_res_query.weight"), {E, full ? 3 : 1}, GGML_TYPE_F32);
        T(L(i, "ffn_res_query.weight"), {E}, GGML_TYPE_F32);
        T(L(i, "ffn_norm.weight"), {E}, GGML_TYPE_F32);
        T(L(i, "ffn_gate_inp.weight"), {E, NE}, GGML_TYPE_F32);
        T(L(i, "exp_probs_b.bias"), {NE}, GGML_TYPE_F32);
        T(L(i, "ffn_gate_up_exps.weight"), {E, 2 * FF, NE}, GGML_TYPE_Q8_0);
        T(L(i, "ffn_down_exps.weight"), {FF, E, NE}, GGML_TYPE_Q8_0);
        T(L(i, "ffn_gate_up_exps.scale"), {NE}, GGML_TYPE_F32);
        T(L(i, "ffn_down_exps.scale"), {NE}, GGML_TYPE_F32);
    }
    const char* ctl[4] = {"q", "k", "v", "ffn"};
    const char* root_ctl[4] = {"loop_attn", "loop_k", "loop_v", "loop_mlp"};
    for (int i = CORE0; i < CORE1; ++i) {
        for (int c = 0; c < 4; ++c) {
            const std::string src = std::string("ctl_") + ctl[c];
            const std::string dst = root ? std::string(root_ctl[c]) : std::string("loop_") + ctl[c] + "_ctrl";
            g.tensor(L(i, (dst + "_down.weight").c_str()), {2 * E, RANK}, GGML_TYPE_F32, w.at(L(i, (src + "_down").c_str())));
            g.tensor(L(i, (dst + "_gain.weight").c_str()), {RANK, E}, GGML_TYPE_F32, w.at(L(i, (src + "_gain").c_str())));
            g.tensor(L(i, (dst + "_forget.weight").c_str()), {RANK, E}, GGML_TYPE_F32, w.at(L(i, (src + "_forget").c_str())));
        }
        if (!root) g.tensor(L(i, "loop_msg_query.weight"), {E, 4}, GGML_TYPE_F32, w.at(L(i, "msg_query")));
    }
    if (root) {
        // conversion/inkling2.py's inverse: layer c's (q, k, v, ffn) = (res[2c], kv[2c], kv[2c+1], res[2c+1])
        const int nc = CORE1 - CORE0;
        std::vector<float> res((size_t)2 * nc * E), kv((size_t)2 * nc * E);
        for (int c = 0; c < nc; ++c) {
            const std::vector<float>& m = w.at(L(CORE0 + c, "msg_query"));
            std::memcpy(&res[(size_t)(2 * c) * E], &m[0 * E], E * 4);
            std::memcpy(&kv[(size_t)(2 * c) * E], &m[1 * E], E * 4);
            std::memcpy(&kv[(size_t)(2 * c + 1) * E], &m[2 * E], E * 4);
            std::memcpy(&res[(size_t)(2 * c + 1) * E], &m[3 * E], E * 4);
        }
        g.tensor("loop_res_query.weight", {E, 2 * nc}, GGML_TYPE_F32, res);
        g.tensor("loop_kv_res_query.weight", {E, 2 * nc}, GGML_TYPE_F32, kv);
    }
    g.write(path);
}

// ---------------------------------------------------------------- runs

std::vector<float> run(const std::string& path, StoreMode mode, int chunk, const std::vector<int>& ids, bool one_by_one) {
    Model m(path, true);
    StoreOptions so;
    so.mode = mode;
    so.cache_bytes = 4 * (m.expert_bytes_max() + 8 * ModelFile::kAlign);   // 4 slots: the prompt's experts come in waves
    so.io_threads = 2;
    auto store = make_store(m, so);
    EngineOptions eo;
    eo.n_ctx = 256;
    eo.threads = 3;
    eo.chunk = chunk;
    Engine eng(m, *store, eo);
    std::vector<float> all;
    if (one_by_one) {
        std::vector<float> last;
        for (size_t i = 0; i < ids.size(); ++i) {
            eng.eval(&ids[i], 1, &last);
            all.insert(all.end(), last.begin(), last.end());
        }
    } else {
        eng.eval(ids.data(), (int)ids.size(), nullptr, &all);
    }
    return all;
}

bool same(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "%s: sizes differ (%zu, %zu)\n", what, a.size(), b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (!(a[i] == b[i]) || !std::isfinite(a[i])) {
            std::fprintf(stderr, "%s: first difference at %zu (%g vs %g)\n", what, i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = argc > 1 ? argv[1] : (tmp ? tmp : ".");
    const std::string mainline = dir + "/inkling_tiny_mainline.gguf", root = dir + "/inkling_tiny_root.gguf";
    std::mt19937 rng(20261121);
    auto w = make_weights(rng);
    write_model(mainline, w, false);
    write_model(root, w, true);

    std::vector<int> ids;
    for (int i = 0; i < 40; ++i) ids.push_back((int)((i * 37 + 11) % VOCAB));

    {
        Model m(mainline, true);
        CHECK(m.hp().n_layer == NL);
        CHECK((int)m.schedule().size() == NL + (PASSES - 1) * (CORE1 - CORE0));
        CHECK(m.n_slots() == (int)m.schedule().size());
        CHECK(m.layout() == "inkling2_loop");
        Model r(root, true);
        CHECK(r.layout() == "inkling2");
        CHECK(r.hp().n_layer == NL && r.hp().n_passes == PASSES && r.hp().core_start == CORE0);
    }

    const auto base = run(mainline, StoreMode::Resident, 5, ids, false);
    CHECK(base.size() == ids.size() * VOCAB);
    bool finite = true;
    for (float x : base) finite &= std::isfinite(x);
    CHECK(finite);
    CHECK(same(base, run(mainline, StoreMode::Resident, 5, ids, true), "chunk 5 vs one token at a time"));
    CHECK(same(base, run(mainline, StoreMode::Resident, 40, ids, false), "chunk 5 vs one chunk"));
    CHECK(same(base, run(root, StoreMode::Resident, 5, ids, false), "mainline vs root layout"));
    CHECK(same(base, run(mainline, StoreMode::Stream, 5, ids, false), "resident vs streamed experts"));
    CHECK(same(base, run(mainline, StoreMode::Mapped, 5, ids, false), "resident vs mapped experts"));
    CHECK(same(base, run(root, StoreMode::Stream, 3, ids, true), "streamed root, one token at a time"));

    // a snapshot after 20 tokens, 10 others, back, then the real continuation = a fresh run
    {
        Model m(mainline, true);
        StoreOptions so;
        so.mode = StoreMode::Resident;
        auto store = make_store(m, so);
        EngineOptions eo;
        eo.n_ctx = 256;
        eo.threads = 2;
        eo.chunk = 5;
        Engine eng(m, *store, eo);
        std::vector<float> last;
        eng.eval(ids.data(), 20, &last);
        Snapshot snap;
        eng.snapshot(snap);
        std::vector<int> other(10, 3);
        eng.eval(other.data(), 10, &last);
        CHECK(eng.rollback(snap));
        CHECK(eng.n_past() == 20);
        std::vector<float> rest;
        eng.eval(ids.data() + 20, 20, nullptr, &rest);
        std::vector<float> tail(base.begin() + 20 * VOCAB, base.end());
        CHECK(same(tail, rest, "rollback then continue vs fresh"));
        Snapshot later;
        eng.snapshot(later);
        eng.reset();
        CHECK(!eng.rollback(later));   // a snapshot ahead of the state cannot be used
    }

    std::remove(mainline.c_str());
    std::remove(root.c_str());
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("inkling tiny model test: ok\n");
    return 0;
}
