// src/inkling/model.cpp - see model.hpp.
#include "model.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::inkling {
namespace {

std::string blk(int i, const char* name) { return "blk." + std::to_string(i) + "." + name; }

std::string dims_str(const std::vector<int64_t>& ne) {
    std::string s = "[";
    for (size_t i = 0; i < ne.size(); ++i) s += (i ? ", " : "") + std::to_string(ne[i]);
    return s + "]";
}

}  // namespace

Model::Model(const std::string& path, bool embed_from_file) : path_(path) {
    ggml_cpu_init();
    gguf_ = std::make_unique<GgufFile>(path);
    file_ = std::make_unique<ModelFile>(path, false);
    load_hparams();
    build_schedule();
    load_weights(embed_from_file);
}

Model::~Model() {
    if (arena_) aligned_free(arena_);
}

void Model::load_hparams() {
    const GgufFile& g = *gguf_;
    layout_ = g.get_string("general.architecture", "");
    std::string P;
    std::string k_passes, k_start, k_end, k_rank;
    if (layout_ == "inkling2_loop") {
        P = "inkling2_loop.";
        k_passes = P + "loop.count"; k_start = P + "loop.start"; k_end = P + "loop.end"; k_rank = P + "loop.gate_rank";
    } else if (layout_ == "inkling2") {
        P = "inkling2.";
        k_passes = P + "num_loops"; k_start = P + "loop.core_start"; k_end = P + "loop.core_end"; k_rank = P + "loop.gate_rank";
    } else {
        throw std::runtime_error("this is not an Inkling 2 model (general.architecture = '" + layout_ + "')");
    }
    auto first = [&](const std::string& key) -> int64_t {
        const GgufValue* v = g.get(key);
        if (!v) throw std::runtime_error("the model has no " + key);
        if (v->kind == GgufValue::Kind::IntArray) {
            if (v->ia.empty()) throw std::runtime_error(key + " is empty");
            for (int64_t x : v->ia)
                if (x != v->ia[0]) throw std::runtime_error(key + " differs between layers; this engine needs one value");
            return v->ia[0];
        }
        return g.get_int(key, 0);
    };
    auto need_int = [&](const std::string& key) -> int64_t {
        if (!g.has(key)) throw std::runtime_error("the model has no " + key);
        return g.get_int(key, 0);
    };
    auto need_float = [&](const std::string& key) -> float {
        if (!g.has(key)) throw std::runtime_error("the model has no " + key);
        return (float)g.get_float(key, 0);
    };

    Hparams& h = hp_;
    // the physical layers are the blk.N tensors in the file (the root layout's block_count counts applications)
    int n_phys = 0;
    for (const GgufTensor& t : g.tensors()) {
        if (t.name.rfind("blk.", 0) != 0) continue;
        n_phys = std::max(n_phys, std::atoi(t.name.c_str() + 4) + 1);
    }
    h.n_layer = n_phys;
    h.n_embd = (int)need_int(P + "embedding_length");
    h.n_head = (int)first(P + "attention.head_count");
    h.n_head_kv = (int)first(P + "attention.head_count_kv");
    h.head_dim = (int)need_int(P + "attention.key_length");
    if (g.get_int(P + "attention.value_length", h.head_dim) != h.head_dim)
        throw std::runtime_error("key and value head sizes differ");
    if (g.get_int(P + "attention.key_length_swa", h.head_dim) != h.head_dim)
        throw std::runtime_error("the sliding-window layers have another head size; not supported");
    h.n_ff_exp = (int)first(P + "expert_feed_forward_length");
    h.n_expert = (int)need_int(P + "expert_count");
    h.n_expert_used = (int)first(P + "expert_used_count");
    h.n_vocab_padded = (int)need_int(P + "vocab_size");
    h.n_vocab = (int)need_int(P + "vocab_size_unpadded");
    h.n_rel = (int)need_int(P + "attention.relative_length");
    h.rel_extent = (int)need_int(P + "attention.relative_extent");
    h.n_swa = (int)need_int(P + "attention.sliding_window");
    h.n_conv = (int)need_int(P + "shortconv.l_cache");
    h.res_block = (int)need_int(P + "attn_res.block_size");
    h.eps = need_float(P + "attention.layer_norm_rms_epsilon");
    h.qk_clip = need_float(P + "attention.qk_clip");
    h.rel_clip = need_float(P + "attention.relative_clip");
    h.expert_scale = (float)g.get_float(P + "expert_weights_scale", 1.0);
    h.situ_beta = need_float(P + "activation.situ_beta");
    h.situ_linear_beta = (float)g.get_float(P + "activation.situ_linear_beta", 0.0);
    h.softcap = (float)g.get_float(P + "final_logit_softcapping", 0.0);
    h.log_floor = (float)need_int(P + "attention.log_scale_floor");
    h.log_alpha = need_float(P + "attention.log_scale_alpha");
    h.n_ctx_train = (int)g.get_int(P + "context_length", 0);
    h.eos = (int)g.get_int("tokenizer.ggml.eos_token_id", -1);
    h.pad = (int)g.get_int("tokenizer.ggml.padding_token_id", -1);

    const GgufValue* pat = g.get(P + "attention.sliding_window_pattern");
    if (!pat || pat->kind != GgufValue::Kind::BoolArray || (int)pat->ba.size() < n_phys)
        throw std::runtime_error("the model has no usable sliding_window_pattern");
    h.swa.assign(pat->ba.begin(), pat->ba.begin() + n_phys);

    if (g.has(k_passes)) {
        h.n_passes = (int)g.get_int(k_passes, 1);
        h.core_start = (int)need_int(k_start);
        h.core_end = (int)need_int(k_end);
        h.gate_rank = (int)need_int(k_rank);
        if (g.get_bool(P + "loop.varying", false))
            std::fprintf(stderr, "inkling: note: the model was trained for a varying pass count; running %d passes\n", h.n_passes);
    }
    if (h.n_passes < 1) h.n_passes = 1;

    // the checks llama.cpp makes (inkling2.cpp load_arch_hparams), and the ones this engine relies on
    if (n_phys <= 0 || h.n_embd <= 0 || h.n_head <= 0 || h.n_head_kv <= 0 || h.n_head % h.n_head_kv || h.head_dim <= 0 ||
        h.n_expert <= 0 || h.n_expert_used <= 0 || h.n_expert_used > h.n_expert || h.n_ff_exp <= 0 || h.n_rel <= 0 ||
        h.rel_extent <= 0 || h.n_swa <= 0 || h.res_block <= 0 || h.n_conv < 2 || h.n_conv > 64 || h.n_vocab <= 0 ||
        h.n_vocab > h.n_vocab_padded || h.log_floor <= 0 || h.log_alpha < 0)
        throw std::runtime_error("invalid Inkling dimensions in the model header");
    if (h.looped() && (h.core_start <= 0 || h.core_start >= h.core_end || h.core_end > n_phys ||
                       h.core_start % h.res_block || h.core_end % h.res_block || h.gate_rank <= 0))
        throw std::runtime_error("invalid loop core [" + std::to_string(h.core_start) + ", " + std::to_string(h.core_end) + ")");
    const int64_t bc = g.get_int(P + "block_count", n_phys);
    const int64_t apps = n_phys + (int64_t)(h.n_passes - 1) * (h.looped() ? h.n_core() : 0);
    if (bc != n_phys && bc != apps)
        throw std::runtime_error("block_count " + std::to_string(bc) + " fits neither the " + std::to_string(n_phys) +
                                 " layers in the file nor their " + std::to_string(apps) + " applications");
}

void Model::build_schedule() {
    const Hparams& h = hp_;
    schedule_.clear();
    const int start = h.looped() ? h.core_start : h.n_layer;
    const int end = h.looped() ? h.core_end : h.n_layer;
    for (int i = 0; i < start; ++i) schedule_.push_back({i, 0, i});
    for (int pass = 0; pass < (h.looped() ? h.n_passes : 0); ++pass)
        for (int ip = start; ip < end; ++ip)
            schedule_.push_back({ip, pass, pass == 0 ? ip : h.n_layer + (pass - 1) * (end - start) + (ip - start)});
    for (int i = end; i < h.n_layer; ++i) schedule_.push_back({i, 0, i});
    n_slots_ = h.n_layer + (h.looped() ? (h.n_passes - 1) * (end - start) : 0);
}

void Model::load_weights(bool embed_from_file) {
    const GgufFile& g = *gguf_;
    const Hparams& h = hp_;
    const int E = h.n_embd;

    auto need = [&](const std::string& name) -> const GgufTensor& {
        const GgufTensor* t = g.tensor(name);
        if (!t) throw std::runtime_error("the model has no tensor " + name);
        return *t;
    };
    // GGUF writers drop trailing dimensions of 1 ([2304, 1] is stored as [2304]): compare without them
    auto trim = [](std::vector<int64_t> ne) {
        while (ne.size() > 1 && ne.back() == 1) ne.pop_back();
        return ne;
    };
    auto check = [&](const GgufTensor& t, std::vector<int64_t> ne) {
        if (trim(t.ne) != trim(ne)) throw std::runtime_error(t.name + " is " + dims_str(t.ne) + ", expected " + dims_str(ne));
    };

    // pass 1: the matrices kept in RAM, to size the arena once
    std::vector<std::pair<const GgufTensor*, Mat*>> mats;
    auto mat = [&](const std::string& name, int64_t cols, int64_t rows, Mat& m) {
        const GgufTensor& t = need(name);
        check(t, {cols, rows});
        m.type = t.type;
        m.cols = cols;
        m.rows = rows;
        m.row_bytes = (size_t)ggml_bytes(t.type, cols);
        mats.push_back({&t, &m});
    };
    auto vecf = [&](const std::string& name, std::vector<int64_t> ne, std::vector<float>& out, bool optional = false) {
        const GgufTensor* t = g.tensor(name);
        if (!t) {
            if (optional) { out.clear(); return; }
            throw std::runtime_error("the model has no tensor " + name);
        }
        check(*t, ne);
        std::vector<uint8_t> raw(t->nbytes);
        file_->read(raw.data(), raw.size(), t->offset);
        out.resize((size_t)t->nelements());
        if (t->type == GGML_TYPE_F32) {
            std::memcpy(out.data(), raw.data(), raw.size());
        } else {
            const auto* tr = ggml_get_type_traits((ggml_type)t->type);
            if (!tr->to_float) throw std::runtime_error(name + " has a type that cannot be read as floats");
            tr->to_float(raw.data(), out.data(), t->nelements());
        }
    };

    layers_.assign(h.n_layer, Layer{});
    const int64_t nconv = h.n_conv, kvd = h.kv_dim(), qd = h.q_dim();
    for (int i = 0; i < h.n_layer; ++i) {
        Layer& L = layers_[i];
        L.swa = h.swa[i];
        const int64_t extent = L.swa ? h.n_swa : h.rel_extent;
        mat(blk(i, "attn_q.weight"), E, qd, L.wq);
        mat(blk(i, "attn_k.weight"), E, kvd, L.wk);
        mat(blk(i, "attn_v.weight"), E, kvd, L.wv);
        mat(blk(i, "attn_output.weight"), qd, E, L.wo);
        mat(blk(i, "attn_gate.weight"), E, qd, L.wgate);
        mat(blk(i, "attn_rel.weight"), E, (int64_t)h.n_rel * h.n_head, L.wrel);
        mat(blk(i, "attn_sink_proj.weight"), E, h.n_head, L.sink_proj);
        vecf(blk(i, "attn_q_clip.weight"), {h.head_dim, h.n_head}, L.q_clip);
        vecf(blk(i, "attn_rel_bias.weight"), {h.n_rel, extent}, L.rel_bias);
        vecf(blk(i, "attn_rel_clip.weight"), {h.n_rel, h.n_head}, L.rel_clip);
        vecf(blk(i, "attn_k_conv.weight"), {nconv, kvd}, L.k_conv);
        vecf(blk(i, "attn_v_conv.weight"), {nconv, kvd}, L.v_conv);
        vecf(blk(i, "attn_post_conv.weight"), {nconv, E}, L.attn_post_conv);
        vecf(blk(i, "ffn_post_conv.weight"), {nconv, E}, L.ffn_post_conv);
        if (L.swa) {
            vecf(blk(i, "attn_norm.weight"), {E}, L.attn_norm);
        } else {
            vecf(blk(i, "attn_q_inp_norm.weight"), {E}, L.q_inp_norm);
            vecf(blk(i, "attn_k_inp_norm.weight"), {E}, L.k_inp_norm);
            vecf(blk(i, "attn_v_inp_norm.weight"), {E}, L.v_inp_norm);
            mat(blk(i, "attn_k_conv_proj.weight"), E, nconv * h.n_head_kv, L.k_conv_proj);
            mat(blk(i, "attn_v_conv_proj.weight"), E, nconv * h.n_head_kv, L.v_conv_proj);
        }
        if (i > 0) {
            L.n_attn_res_query = L.swa ? 1 : 3;
            vecf(blk(i, "attn_res_query.weight"), {E, L.n_attn_res_query}, L.attn_res_query);
        }
        vecf(blk(i, "ffn_res_query.weight"), {E}, L.ffn_res_query);
        vecf(blk(i, "ffn_norm.weight"), {E}, L.ffn_norm);
        mat(blk(i, "ffn_gate_inp.weight"), E, h.n_expert, L.router);
        vecf(blk(i, "exp_probs_b.bias"), {h.n_expert}, L.exp_probs_b, true);

        auto experts = [&](const char* name, int64_t cols, int64_t rows, ExpertTensor& et) {
            const GgufTensor& t = need(blk(i, name));
            check(t, {cols, rows, h.n_expert});
            et.type = t.type;
            et.cols = cols;
            et.rows = rows;
            et.row_bytes = (size_t)ggml_bytes(t.type, cols);
            et.bytes = (uint64_t)et.row_bytes * (uint64_t)rows;
            et.offset = t.offset;
            std::string sname = blk(i, name);
            sname.replace(sname.size() - 6, 6, "scale");   // ".weight" -> ".scale"
            vecf(sname, {h.n_expert}, et.scale, true);
        };
        experts("ffn_gate_up_exps.weight", E, 2 * (int64_t)h.n_ff_exp, L.gate_up);
        experts("ffn_down_exps.weight", h.n_ff_exp, E, L.down);
    }

    // the cross-pass controllers of the loop core
    loop_.clear();
    if (h.looped()) {
        const bool root = layout_ == "inkling2";
        const char* mainline_ctrl[4] = {"loop_q_ctrl", "loop_k_ctrl", "loop_v_ctrl", "loop_ffn_ctrl"};
        const char* root_ctrl[4] = {"loop_attn", "loop_k", "loop_v", "loop_mlp"};
        std::vector<float> res_q, kv_q;
        if (root) {
            vecf("loop_res_query.weight", {E, 2 * (int64_t)h.n_core()}, res_q);
            vecf("loop_kv_res_query.weight", {E, 2 * (int64_t)h.n_core()}, kv_q);
        }
        loop_.resize(h.n_core());
        for (int c = 0; c < h.n_core(); ++c) {
            const int il = h.core_start + c;
            LoopLayer& LL = loop_[c];
            for (int k = 0; k < 4; ++k) {
                std::string base = root ? root_ctrl[k] : mainline_ctrl[k];
                mat(blk(il, (base + "_down.weight").c_str()), 2 * (int64_t)E, h.gate_rank, LL.down[k]);
                mat(blk(il, (base + "_gain.weight").c_str()), h.gate_rank, E, LL.gain[k]);
                mat(blk(il, (base + "_forget.weight").c_str()), h.gate_rank, E, LL.forget[k]);
            }
            if (root) {
                // the converter's order (conversion/inkling2.py): q[2c], kv[2c], kv[2c+1], q[2c+1] = the q, k, v, ffn messages
                LL.msg_query.resize(4 * (size_t)E);
                const float* rows[4] = {&res_q[(size_t)(2 * c) * E], &kv_q[(size_t)(2 * c) * E],
                                        &kv_q[(size_t)(2 * c + 1) * E], &res_q[(size_t)(2 * c + 1) * E]};
                for (int k = 0; k < 4; ++k) std::memcpy(&LL.msg_query[(size_t)k * E], rows[k], sizeof(float) * E);
            } else {
                vecf(blk(il, "loop_msg_query.weight"), {E, 4}, LL.msg_query);
            }
        }
    }

    vecf("token_embd_norm.weight", {E}, tok_norm_);
    vecf("output_norm.weight", {E}, output_norm_);
    vecf("output_res_query.weight", {E}, output_res_query_);
    mat("output.weight", E, h.n_vocab_padded, output_);
    {
        const GgufTensor& t = need("token_embd.weight");
        check(t, {E, h.n_vocab_padded});
        tok_embd_.type = t.type;
        tok_embd_.cols = E;
        tok_embd_.rows = h.n_vocab_padded;
        tok_embd_.row_bytes = (size_t)ggml_bytes(t.type, E);
        tok_embd_offset_ = t.offset;
        embed_from_file_ = embed_from_file;
        if (!embed_from_file) mats.push_back({&t, &tok_embd_});
    }

    // pass 2: one arena, every matrix read straight into it
    uint64_t total = 0;
    for (auto& [t, m] : mats) total += (t->nbytes + 63) / 64 * 64;
    arena_bytes_ = total;
    arena_ = static_cast<uint8_t*>(aligned_alloc_bytes((size_t)total, 64));
    arena_used_ = 0;
    for (auto& [t, m] : mats) {
        uint8_t* dst = arena_ + arena_used_;
        file_->read(dst, (size_t)t->nbytes, t->offset);
        m->data = dst;
        arena_used_ += (t->nbytes + 63) / 64 * 64;
    }
    dense_bytes_ = arena_bytes_;
    for (const Layer& L : layers_) {
        dense_bytes_ += sizeof(float) * (L.q_clip.size() + L.rel_clip.size() + L.rel_bias.size() + L.k_conv.size() +
                                         L.v_conv.size() + L.attn_post_conv.size() + L.ffn_post_conv.size() + L.attn_res_query.size());
    }
}

void Model::embed_row(int token, float* out) const {
    if (token < 0 || token >= hp_.n_vocab) throw std::runtime_error("token id " + std::to_string(token) + " is out of range");
    const auto* tr = ggml_get_type_traits((ggml_type)tok_embd_.type);
    if (embed_from_file_) {
        std::vector<uint8_t> row(tok_embd_.row_bytes);
        file_->read(row.data(), row.size(), tok_embd_offset_ + (uint64_t)token * tok_embd_.row_bytes);
        if (tok_embd_.type == GGML_TYPE_F32) std::memcpy(out, row.data(), row.size());
        else tr->to_float(row.data(), out, tok_embd_.cols);
    } else {
        if (tok_embd_.type == GGML_TYPE_F32) std::memcpy(out, tok_embd_.row(token), tok_embd_.row_bytes);
        else tr->to_float(tok_embd_.row(token), out, tok_embd_.cols);
    }
}

uint64_t Model::expert_bytes_total() const {
    uint64_t s = 0;
    for (const Layer& L : layers_) s += (L.gate_up.bytes + L.down.bytes) * (uint64_t)hp_.n_expert;
    return s;
}

uint64_t Model::expert_bytes_max() const {
    uint64_t m = 0;
    for (const Layer& L : layers_) m = std::max<uint64_t>(m, L.gate_up.bytes + L.down.bytes);
    return m;
}

const std::vector<std::string>* Model::vocab() const {
    const GgufValue* v = gguf_->get("tokenizer.ggml.tokens");
    return v && v->kind == GgufValue::Kind::StringArray ? &v->sa : nullptr;
}

}  // namespace strata::inkling
