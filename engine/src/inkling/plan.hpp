// src/inkling/plan.hpp - how much of a RAM budget the experts get, and whether they all fit.
#pragma once

#include <cstdint>
#include <string>

namespace strata::inkling {

struct MemoryPlan {
    uint64_t budget = 0;          ///< what the engine may use in all
    uint64_t dense = 0;           ///< the dense weights
    uint64_t kv = 0;              ///< the attention caches at the full context
    uint64_t scratch = 0;         ///< chunk buffers and the rest
    uint64_t experts_all = 0;     ///< every expert of the model
    uint64_t expert_cache = 0;    ///< what is left for experts
    bool all_fit = false;         ///< every expert fits in what is left
    std::string why;              ///< one line for the log
};

/// `budget` 0 = what the OS reports as available minus a margin for the rest of the system.
/// `cache_override` > 0 sets the expert cache directly.
MemoryPlan plan_memory(uint64_t budget, uint64_t dense, uint64_t kv, uint64_t scratch, uint64_t experts_all,
                       uint64_t cache_override);

}  // namespace strata::inkling
