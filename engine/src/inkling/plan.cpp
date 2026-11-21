// src/inkling/plan.cpp - see plan.hpp.
#include "plan.hpp"

#include "file_io.hpp"

#include <algorithm>
#include <cstdio>

namespace strata::inkling {

static double gib(uint64_t b) { return b / 1073741824.0; }

MemoryPlan plan_memory(uint64_t budget, uint64_t dense, uint64_t kv, uint64_t scratch, uint64_t experts_all,
                       uint64_t cache_override) {
    MemoryPlan p;
    p.dense = dense;
    p.kv = kv;
    p.scratch = scratch;
    p.experts_all = experts_all;
    char line[320];
    if (budget == 0) {
        // what the OS calls available, less a margin for everything else on the computer: 1 GiB or 10%
        const uint64_t avail = available_ram_bytes();
        const uint64_t margin = std::max<uint64_t>(1ull << 30, avail / 10);
        budget = avail > margin ? avail - margin : 0;
        std::snprintf(line, sizeof line, "budget %.2f GiB (%.2f GiB available, %.2f GiB left to the system)", gib(budget),
                      gib(avail), gib(margin));
    } else {
        std::snprintf(line, sizeof line, "budget %.2f GiB (given)", gib(budget));
    }
    p.budget = budget;
    const uint64_t fixed = dense + kv + scratch;
    if (cache_override > 0) {
        p.expert_cache = cache_override;
    } else {
        p.expert_cache = budget > fixed ? budget - fixed : 0;
    }
    p.all_fit = p.expert_cache >= experts_all;
    char rest[320];
    std::snprintf(rest, sizeof rest, "; dense %.2f + attention cache %.2f + buffers %.2f GiB; experts %.2f of %.2f GiB%s",
                  gib(dense), gib(kv), gib(scratch), gib(std::min(p.expert_cache, experts_all)), gib(experts_all),
                  p.all_fit ? " (all in RAM)" : " (the rest read from the SSD)");
    p.why = std::string(line) + rest;
    return p;
}

}  // namespace strata::inkling
