// src/inkling/engine.hpp - the Inkling 2 forward pass on the CPU, with the experts behind an ExpertStore.
//
// One sequence. Tokens go through in chunks (the prompt) or one at a time (decoding); each of the 48 layer applications
// has its own attention cache and short-convolution state ("slots", as in llama.cpp's inkling2_loop). The 8 full-
// attention slots keep every position; the 40 sliding-window slots keep a ring of the last window + chunk positions.
//
// The arithmetic follows llama.cpp's graph (src/models/inkling2*.cpp at ggml-org/llama.cpp#32704) step by step,
// including where it rounds to f16: the attention cache is f16, the queries are rounded to f16 before q.k and the
// attention weights before they multiply V, as ggml-cpu's mul_mat does for an f16 operand.
#pragma once

#include "cpu_ops.hpp"
#include "expert_store.hpp"
#include "model.hpp"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

namespace strata::inkling {

class Gpu;

struct EngineOptions {
    Gpu* gpu = nullptr;       ///< the dense matrices with a `gpu` copy run there (gpu.hpp)
    int n_ctx = 32768;
    int threads = 0;          ///< 0: the physical cores
    int chunk = 256;          ///< prompt tokens per step
    /// decode: guess each layer's experts from its input and read them during attention. Off by default: on the
    /// machines measured (docs/INKLING.md) the guess is right about half the time, so it read 37-60% more from the SSD
    /// for the same or lower speed.
    bool lookahead = false;
};

struct EngineStats {
    uint64_t tokens = 0;
    double attn_s = 0, moe_s = 0, dense_s = 0, head_s = 0;
};

/// The state of the sequence: the attention caches and the convolution states of every slot.
struct SeqState {
    int n_past = 0;
    std::vector<std::unique_ptr<uint16_t[]>> k, v;   ///< per slot, [cap][kv_dim] f16
    std::vector<int> cap;                             ///< per slot, positions held (ring for sliding-window slots)
    std::vector<std::vector<float>> conv;             ///< per slot: K, V, attention-out and FFN-out histories
};

/// What a rollback to an earlier position needs: the parts that cannot be cut back (the convolution states and the
/// sliding-window rings, ~160 MiB). The full-attention caches are cut back by position, so they are not copied.
struct Snapshot {
    int n_past = -1;
    std::vector<std::vector<float>> conv;
    std::vector<std::vector<uint16_t>> ring_k, ring_v;   ///< per slot, empty for full-attention slots
    bool valid() const { return n_past >= 0; }
};

class Engine {
public:
    Engine(const Model& model, ExpertStore& store, const EngineOptions& opt);
    ~Engine();

    const Model& model() const { return m_; }
    int n_ctx() const { return opt_.n_ctx; }
    int n_past() const { return st_.n_past; }
    int threads() const { return pool_->size(); }
    /// Forgets the sequence.
    void reset();

    /// Runs `n` tokens at positions n_past..n_past+n-1. `last` (if not null) gets the logits of the last token,
    /// `all` (if not null) those of every token, [n][n_vocab]. Throws when the context is full.
    void eval(const int* tokens, int n, std::vector<float>* last, std::vector<float>* all = nullptr);

    /// Remembers the current position; rollback() returns to it as long as the sequence only grew since.
    void snapshot(Snapshot& out) const;
    /// Back to a snapshot taken at an earlier (or the same) position of this sequence. False when it cannot be.
    bool rollback(const Snapshot& s);
    /// Bytes a snapshot takes.
    uint64_t snapshot_bytes() const;
    /// RAM of the attention caches if the whole context is used.
    uint64_t kv_bytes_full() const;

    EngineStats stats() const { return stats_; }
    void reset_stats() { stats_ = EngineStats{}; }
    ThreadPool& pool() { return *pool_; }
    Gpu* gpu() const { return opt_.gpu; }   ///< the GPU of the dense weights and the expert tier, or null

private:
    struct Buf;
    /// One matrix product: y [T][ldy] = W . x [T][ldx]
    struct MM {
        const Mat* W;
        const float* x;
        int64_t ldx;
        float* y;
        int64_t ldy;
    };
    /// Runs the products whose weights are on the GPU as one graph there, the others on the CPU.
    void mm(std::initializer_list<MM> jobs, int T);
    void run_chunk(const int* tokens, int T, std::vector<float>* last, std::vector<float>* all, int all_row0);
    void apply(const Application& app, int T, std::vector<std::vector<float>>& blocks, const float* msgs);
    void attention(const Layer& L, int slot, int T, int p0);
    void moe(int layer, int T);
    void mix(const std::vector<const float*>& states, int T, const float* query, int nq, float* out);
    void control(const LoopLayer& LL, int k, const float* cur, const float* msg, int T, float* out);
    void convolve(float* cur, int T, int channels, const std::vector<float>& w, const float* dyn, int n_heads,
                  float* past);
    void head(int T, const std::vector<std::vector<float>>& blocks, std::vector<float>* last, std::vector<float>* all,
              int all_row0);
    void prefetch_guess(int layer, int T);

    const Model& m_;
    ExpertStore& store_;
    EngineOptions opt_;
    std::unique_ptr<ThreadPool> pool_;
    SeqState st_;
    std::unique_ptr<Buf> b_;
    EngineStats stats_;
};

}  // namespace strata::inkling
