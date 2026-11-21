// src/inkling/gpu.hpp - the dense weights and the hot experts on a GPU (NVIDIA through ggml's CUDA backend, AMD through
// its HIP backend: the same code for both).
//
// The dense matrices (attention projections, the output head: 1.1-1.7 GiB) go to VRAM once and their matrix products
// run there; they are then not kept in RAM, which leaves that RAM to the expert cache. Graphs are built once per shape
// and replayed, so a product costs one upload, one launch and one download.
//
// The rest of the VRAM is an expert tier in front of the RAM cache (init_experts): one slot pool per expert type, with
// the same CLOCK eviction the RAM cache uses. An expert is copied up from RAM once it has been used (twice when the pool
// is full), and from then on the layer's GPU-resident experts run there (moe_begin: gate_up, SiTU, down through ggml's
// mul_mat_id) while the CPU computes the others; moe_end hands back the per-rank outputs, which the engine sums in rank
// order exactly as for the CPU's.
//
// Built with ggml's backend registry (the standalone project, src/inkling/CMakeLists.txt); the root engine project has
// no registry and builds the stub (gpu_built() false).
#pragma once

#include "model.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strata::inkling {

bool gpu_built();

struct ExpertView;

struct GpuExpertStats {
    uint64_t hits = 0;        ///< (token, rank) lookups served by the GPU tier
    uint64_t lookups = 0;     ///< all (token, rank) lookups the tier saw
    uint64_t uploads = 0;     ///< experts copied up
    uint64_t upload_bytes = 0;
};

class Gpu {
public:
    /// `which`: "auto" (the first GPU, none if there is none), "off", "cpu" (ggml's CPU backend standing in for a GPU:
    /// for tests), or a GPU index. nullptr (with `why`) when there is none to use.
    static std::unique_ptr<Gpu> open(const std::string& which, int cpu_threads, std::string& why);
    ~Gpu();
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;

    std::string name() const;
    uint64_t free_bytes() const;     ///< free VRAM now (0 for the CPU stand-in)
    uint64_t total_bytes() const;    ///< all of the device's VRAM (0 for the CPU stand-in)
    uint64_t weight_bytes() const { return weight_bytes_; }
    /// Uploads the model's GPU matrices (Model::gpu_mats) from the file. False with `why` when they do not fit.
    bool upload(Model& m, std::string& why);

    /// y[t][r] = W.row(r) . x[t] for r < rows_out (rows_out <= W.rows: the first rows), x with row stride W.cols.
    struct Job {
        const Mat* W;
        const float* x;
        int ntok;
        float* y;
        int64_t rows_out;
    };
    /// Runs the jobs in one graph (inputs shared where the same x is given). The graph of a set of jobs is kept and
    /// replayed when the same set comes again (decoding repeats the same ~100 sets every token).
    void run(const Job* jobs, int n);

    // ---------------------------------------------------------------------------------------------- the expert tier
    /// Sets up the expert pools in `bytes` of VRAM, split over the expert types by their share of the model. False with
    /// `why` when the backend cannot run the expert graph for a type, or the budget holds no expert.
    bool init_experts(const Model& m, uint64_t bytes, std::string& why);
    bool has_experts() const;
    uint64_t expert_bytes() const;   ///< VRAM the pools take
    int expert_slots() const;        ///< experts the pools hold
    /// The pool slot of expert e of physical layer `layer`, or -1 (a hit counts as a use for the eviction).
    int lookup(int layer, int e);
    /// Counts a use of an expert the CPU computes; true when it should be copied up now (admit()).
    bool want(int layer, int e);
    /// Copies an expert from RAM into a free or evicted slot (not one pinned by a running moe_begin). False if none.
    bool admit(int layer, int e, const ExpertView& v);
    /// Starts the layer's GPU experts. x: [T][n_embd] (read until moe_end); slot: [T][K] the pool slot of each
    /// (token, rank) or -1 (the CPU computes it). Returns at once; the GPU works while the CPU does.
    void moe_begin(int layer, const float* x, int T, const int* slot);
    /// Waits for moe_begin's graph and writes out[(t * K + k) * n_embd ..] for the ranks with a slot.
    void moe_end(float* out);
    GpuExpertStats expert_stats() const;
    void reset_expert_stats();

private:
    Gpu() = default;
    struct Impl;
    std::unique_ptr<Impl> p_;
    uint64_t weight_bytes_ = 0;
};

}  // namespace strata::inkling
