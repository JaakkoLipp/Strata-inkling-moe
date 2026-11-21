// src/inkling/serve.hpp - the engine side of the line protocol the Strata server (terminal/serve) speaks with its
// engines, for one conversation at a time:
//
//   engine: READY <context> stop
//   server: GEN <max_new> [key=value ...] <id,id,...>     (GEN 0 prefill=1 <ids>: read the prompt, make nothing)
//   engine: RESUME <n>            the first n prompt tokens were already in the state
//           PP <done> <total> <ms> <tok/s>                 after every prompt chunk
//           T <id>  [LP <logprob> id:logprob ...]          every token, the end token included
//           DONE <generated> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel|nonfinite> 0 0 <reused>
//                <hits> <lookups> <ram_blobs> <file_blobs> <file_mb> <prompt_read> 0
//   server: STOP (ends the request at the next chunk or token), QUIT
//
// Sampling keys: temperature, top_p, top_k, min_p, seed, penalty_repeat / penalty_freq / penalty_present /
// penalty_last_n, logprobs, allowed. Unknown keys are ignored.
#pragma once

#include "engine.hpp"
#include "expert_store.hpp"
#include "plan.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace strata::inkling {

struct Sampling {
    float temperature = 0.0f;   ///< 0 = greedy
    float top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;              ///< 0 = the whole vocabulary
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int penalty_last_n = 64;
    uint64_t seed = 0;
    int logprobs = -1;          ///< -1 = none, 0 = the chosen token's, k = and the top k
    std::vector<int> allowed;   ///< empty = all
    bool prefill_only = false;
};

/// Parses "key=value" into s; false for a malformed value of a known key (unknown keys are accepted and ignored).
bool parse_sampling_key(const std::string& kv, Sampling& s, std::string& err);
/// The next token from logits (modified in place by the penalties and the filters).
int sample_token(std::vector<float>& logits, const Sampling& s, const std::vector<int>& recent, std::mt19937_64& rng);
/// log-softmax of the chosen token and the top k, as "LP <lp> id:lp ...".
std::string logprob_line(const std::vector<float>& logits, int chosen, int k);

int run_serve(Engine& eng, ExpertStore& store, const MemoryPlan& plan);

}  // namespace strata::inkling
