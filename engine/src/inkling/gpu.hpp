// src/inkling/gpu.hpp - the dense weights on a GPU (NVIDIA through ggml's CUDA backend, AMD through its HIP backend).
//
// The dense matrices (attention projections, the output head: 1.1-1.7 GiB) go to VRAM once and their matrix products
// run there; they are then not kept in RAM, which leaves that RAM to the expert cache. The experts stay where the CPU
// path keeps them (RAM, streamed from the SSD) and the CPU computes them, as Strata's main engine does for the experts
// that are not in VRAM. Built with ggml's backend registry (the standalone project, src/inkling/CMakeLists.txt); the
// root engine project has no registry and builds the stub (gpu_built() false).
#pragma once

#include "model.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strata::inkling {

bool gpu_built();

class Gpu {
public:
    /// `which`: "auto" (the first GPU, none if there is none), "off", "cpu" (ggml's CPU backend standing in for a GPU:
    /// for tests), or a GPU index. nullptr (with `why`) when there is none to use.
    static std::unique_ptr<Gpu> open(const std::string& which, int cpu_threads, std::string& why);
    ~Gpu();
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;

    std::string name() const;
    uint64_t free_bytes() const;
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
    /// Runs the jobs in one graph (inputs shared where the same x is given).
    void run(const Job* jobs, int n);

private:
    Gpu() = default;
    struct Impl;
    std::unique_ptr<Impl> p_;
    uint64_t weight_bytes_ = 0;
};

}  // namespace strata::inkling
