// src/inkling/model.hpp - Inkling 2 (thinkingmachines/Inkling-2-Mini) as the engine sees it: the hyperparameters, the
// dense weights (kept in RAM), where every expert sits in the file (read on demand by the expert store), and the order
// in which the layers run.
//
// Two GGUF layouts carry the same tensors byte for byte (unsloth/Inkling-2-Mini-GGUF):
//   * the repository's root files: architecture "inkling2", block_count 48 (the layer applications), the loop
//     controllers named loop_{attn,k,v,mlp}_{down,gain,forget} and two global cross-pass query tables;
//   * Mainline_Support/: architecture "inkling2_loop" (llama.cpp's layout), block_count 24, the controllers named
//     loop_{q,k,v,ffn}_ctrl_{down,gain,forget} and one loop_msg_query [n_embd, 4] per core layer.
// Both load into the same structures here.
//
// The forward pass is llama.cpp's (src/models/inkling2.cpp, inkling2-loop.cpp at ggml-org/llama.cpp#32694 + #32704),
// which is the reference the parity tools compare against.
#pragma once

#include "file_io.hpp"
#include "gguf.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::inkling {

/// A matrix in ggml layout: `rows` rows of `cols` elements of `type`, each row `row_bytes` long.
struct Mat {
    int type = 0;
    int64_t cols = 0, rows = 0;
    size_t row_bytes = 0;
    const uint8_t* data = nullptr;
    const uint8_t* row(int64_t r) const { return data + (size_t)r * row_bytes; }
    bool valid() const { return data != nullptr; }
};

/// Where one expert tensor of a layer is in the file: expert e's matrix starts at offset + e * bytes.
struct ExpertTensor {
    int type = 0;
    int64_t cols = 0, rows = 0;       ///< of one expert
    size_t row_bytes = 0;
    uint64_t bytes = 0;               ///< of one expert
    uint64_t offset = 0;              ///< of expert 0
    std::vector<float> scale;         ///< per-expert output scale (NVFP4 global scale), empty = 1
};

struct Hparams {
    int n_embd = 0, n_layer = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_ff_exp = 0, n_expert = 0, n_expert_used = 0;
    int n_vocab = 0, n_vocab_padded = 0;
    int n_rel = 0, rel_extent = 0, n_swa = 0, n_conv = 0, res_block = 0;
    float eps = 1e-6f, qk_clip = 0, rel_clip = 0, expert_scale = 1, situ_beta = 0, situ_linear_beta = 0;
    float softcap = 0, log_floor = 1, log_alpha = 0;
    int core_start = 0, core_end = 0, n_passes = 1, gate_rank = 0;
    std::vector<bool> swa;            ///< per physical layer
    int eos = -1, pad = -1;
    int n_ctx_train = 0;
    int kv_dim() const { return n_head_kv * head_dim; }
    int q_dim() const { return n_head * head_dim; }
    bool looped() const { return core_end > core_start && n_passes > 1; }
    int n_core() const { return core_end - core_start; }
};

/// One run of a physical layer: which layer, in which pass, and which cache slot holds its attention and conv state.
struct Application {
    int phys = 0;
    int pass = 0;      ///< 0 outside the loop core
    int slot = 0;      ///< KV + conv state index, unique per application
};

/// The cross-pass controller of one core layer: four controllers (q, k, v, ffn), each down [2n_embd -> rank],
/// gain and forget [rank -> n_embd], and their four message queries.
struct LoopLayer {
    std::array<Mat, 4> down, gain, forget;
    std::vector<float> msg_query;     ///< [4][n_embd]
};

struct Layer {
    bool swa = true;
    Mat wq, wk, wv, wo, wgate, wrel, sink_proj, k_conv_proj, v_conv_proj;
    std::vector<float> q_clip, rel_clip, rel_bias;             // [n_head][head_dim], [n_head][n_rel], [extent][n_rel]
    std::vector<float> k_conv, v_conv, attn_post_conv, ffn_post_conv;   // [channels][n_conv]
    std::vector<float> attn_norm, q_inp_norm, k_inp_norm, v_inp_norm;
    std::vector<float> attn_res_query;                         // [nq][n_embd], nq 0 (layer 0), 1 (SWA) or 3
    int n_attn_res_query = 0;
    std::vector<float> ffn_res_query, ffn_norm;
    Mat router;                                                // [n_embd -> n_expert]
    std::vector<float> exp_probs_b;                            // [n_expert], empty = none
    ExpertTensor gate_up, down;
};

class Model {
public:
    /// Reads the header and the dense weights of `path`. `embed_from_file` leaves the token embedding table on disk
    /// (one row is read per token): ~250 MB less RAM for a few microseconds per prompt token.
    Model(const std::string& path, bool embed_from_file);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const Hparams& hp() const { return hp_; }
    const std::vector<Layer>& layers() const { return layers_; }
    const std::vector<LoopLayer>& loop_layers() const { return loop_; }
    const std::vector<Application>& schedule() const { return schedule_; }
    int n_slots() const { return n_slots_; }

    const Mat& output() const { return output_; }
    const std::vector<float>& output_norm() const { return output_norm_; }
    const std::vector<float>& output_res_query() const { return output_res_query_; }
    const std::vector<float>& tok_norm() const { return tok_norm_; }
    /// Dequantizes the embedding row of `token` into `out` (n_embd floats).
    void embed_row(int token, float* out) const;

    const GgufFile& gguf() const { return *gguf_; }
    const std::string& layout() const { return layout_; }   ///< "inkling2" (root) or "inkling2_loop" (mainline)
    const std::string& path() const { return path_; }
    uint64_t dense_bytes() const { return dense_bytes_; }   ///< RAM the dense weights take
    uint64_t expert_bytes_total() const;                    ///< all experts of all layers, as in the file
    uint64_t expert_bytes_max() const;                      ///< the largest single expert (gate_up + down)
    /// The tokenizer vocabulary as stored in the GGUF (for the dump tools); empty if absent.
    const std::vector<std::string>* vocab() const;

private:
    void load_hparams();
    void build_schedule();
    void load_weights(bool embed_from_file);

    std::string path_, layout_;
    std::unique_ptr<GgufFile> gguf_;
    std::unique_ptr<ModelFile> file_;
    Hparams hp_;
    std::vector<Layer> layers_;
    std::vector<LoopLayer> loop_;
    std::vector<Application> schedule_;
    int n_slots_ = 0;
    Mat output_, tok_embd_;
    bool embed_from_file_ = false;
    uint64_t tok_embd_offset_ = 0;
    std::vector<float> output_norm_, output_res_query_, tok_norm_;
    uint8_t* arena_ = nullptr;
    uint64_t arena_bytes_ = 0, arena_used_ = 0, dense_bytes_ = 0;
};

}  // namespace strata::inkling
