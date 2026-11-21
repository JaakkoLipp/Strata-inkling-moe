// src/inkling/serve.cpp - see serve.hpp.
#include "serve.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace strata::inkling {

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

bool parse_sampling_key(const std::string& kv, Sampling& s, std::string& err) {
    const size_t eq = kv.find('=');
    if (eq == std::string::npos) return true;
    const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    char* end = nullptr;
    auto num = [&](double& out) {
        out = std::strtod(v.c_str(), &end);
        if (end == v.c_str() || *end || !std::isfinite(out)) {
            err = "GEN: bad value for " + k;
            return false;
        }
        return true;
    };
    double d = 0;
    if (k == "temperature") { if (!num(d)) return false; s.temperature = (float)std::max(0.0, d); }
    else if (k == "top_p") { if (!num(d)) return false; s.top_p = (float)std::clamp(d, 0.0, 1.0); }
    else if (k == "min_p") { if (!num(d)) return false; s.min_p = (float)std::clamp(d, 0.0, 1.0); }
    else if (k == "top_k") { if (!num(d)) return false; s.top_k = std::max(0, (int)d); }
    else if (k == "penalty_repeat") { if (!num(d)) return false; s.penalty_repeat = (float)d; }
    else if (k == "penalty_freq") { if (!num(d)) return false; s.penalty_freq = (float)d; }
    else if (k == "penalty_present") { if (!num(d)) return false; s.penalty_present = (float)d; }
    else if (k == "penalty_last_n") { if (!num(d)) return false; s.penalty_last_n = std::max(0, (int)d); }
    else if (k == "seed") { if (!num(d)) return false; s.seed = (uint64_t)d; }
    else if (k == "logprobs") { if (!num(d)) return false; s.logprobs = std::max(0, (int)d); }
    else if (k == "prefill") { s.prefill_only = v == "1"; }
    else if (k == "allowed") {
        std::stringstream ss(v);
        std::string t;
        while (std::getline(ss, t, ','))
            if (!t.empty()) s.allowed.push_back(std::atoi(t.c_str()));
    }
    return true;
}

int sample_token(std::vector<float>& lg, const Sampling& s, const std::vector<int>& recent, std::mt19937_64& rng) {
    const int n = (int)lg.size();
    if (!s.allowed.empty()) {
        std::vector<float> keep(n, -INFINITY);
        for (int id : s.allowed)
            if (id >= 0 && id < n) keep[id] = lg[id];
        lg.swap(keep);
    }
    // penalties over the last penalty_last_n tokens (llama.cpp's: divide a positive logit, multiply a negative one)
    if ((s.penalty_repeat != 1.0f || s.penalty_freq != 0.0f || s.penalty_present != 0.0f) && s.penalty_last_n > 0) {
        std::unordered_map<int, int> count;
        const size_t from = recent.size() > (size_t)s.penalty_last_n ? recent.size() - (size_t)s.penalty_last_n : 0;
        for (size_t i = from; i < recent.size(); ++i) count[recent[i]]++;
        for (auto& [id, c] : count) {
            if (id < 0 || id >= n) continue;
            float& x = lg[id];
            if (s.penalty_repeat != 1.0f) x = x > 0 ? x / s.penalty_repeat : x * s.penalty_repeat;
            x -= (float)c * s.penalty_freq + (c > 0 ? s.penalty_present : 0.0f);
        }
    }
    if (s.temperature <= 0.0f) return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    // candidates: the top_k (all when 0), most likely first
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    int kk = s.top_k > 0 ? std::min(s.top_k, n) : n;
    const bool filtered = s.top_p < 1.0f || s.min_p > 0.0f;
    if (kk < n) {
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(kk);
    } else if (filtered) {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    }
    float mx = -INFINITY;
    for (int i : idx) mx = std::max(mx, lg[i]);
    std::vector<double> p(idx.size());
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((double)(lg[idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    if (filtered) {
        double cum = 0;
        const double p0 = p[0] / sum;
        for (size_t i = 0; i < idx.size(); ++i) {
            const double pi = p[i] / sum;
            if (s.min_p > 0.0f && pi < s.min_p * p0) { keep = i; break; }
            cum += pi;
            if (cum >= s.top_p) { keep = i + 1; break; }
        }
        keep = std::max<size_t>(1, keep);
    }
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i)
        if ((r -= p[i]) <= 0) return idx[i];
    return idx[keep - 1];
}

std::string logprob_line(const std::vector<float>& lg, int chosen, int k) {
    float mx = *std::max_element(lg.begin(), lg.end());
    double sum = 0;
    for (float x : lg) sum += std::exp((double)(x - mx));
    const double lse = mx + std::log(sum);
    char b[64];
    std::snprintf(b, sizeof b, "LP %.6f", (double)lg[chosen] - lse);
    std::string out = b;
    if (k > 0) {
        std::vector<int> idx(lg.size());
        std::iota(idx.begin(), idx.end(), 0);
        k = std::min<int>(k, (int)idx.size());
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int c) { return lg[a] > lg[c]; });
        for (int i = 0; i < k; ++i) {
            std::snprintf(b, sizeof b, " %d:%.6f", idx[i], (double)lg[idx[i]] - lse);
            out += b;
        }
    }
    return out;
}

namespace {

struct Session {
    std::vector<int> live;          ///< the tokens the engine state holds
    std::vector<float> logits;      ///< the logits after the last of them (for a request that adds nothing)
    Snapshot snap;                  ///< at the end of the last prompt
    std::vector<int> snap_tokens;
};

bool is_prefix(const std::vector<int>& a, const std::vector<int>& b) {
    return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}

}  // namespace

int run_serve(Engine& eng, ExpertStore& store, const MemoryPlan& plan) {
    const Hparams& h = eng.model().hp();
    std::vector<int> eos_ids = {h.eos};
    // <|endoftext|> ends a raw completion too
    if (const auto* vocab = eng.model().vocab())
        for (int i = 0; i < (int)vocab->size(); ++i)
            if ((*vocab)[i] == "<|endoftext|>" && i != h.eos) eos_ids.push_back(i);
    auto is_eos = [&](int t) { return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end(); };

    std::atomic<bool> stop{false}, quit{false};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;

    std::printf("INFO engine=inkling experts=%s expert_cache_mib=%llu budget_mib=%llu\n", store_mode_name(store.mode()),
                (unsigned long long)(store.ram_bytes() >> 20), (unsigned long long)(plan.budget >> 20));
    std::printf("READY %d stop\n", eng.n_ctx());
    std::fflush(stdout);

    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
            if (l == "QUIT") break;
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });

    Session ss;
    auto handle_gen = [&](const std::string& line) {
        std::vector<std::string> f;
        {
            std::stringstream is(line);
            std::string w;
            while (is >> w) f.push_back(w);
        }
        if (f.size() < 3) { std::printf("ERR GEN needs max_new and the prompt ids\n"); return; }
        const long long max_new = std::atoll(f[1].c_str());
        Sampling smp;
        std::string err;
        for (size_t i = 2; i + 1 < f.size(); ++i)
            if (!parse_sampling_key(f[i], smp, err)) { std::printf("ERR %s\n", err.c_str()); return; }
        std::vector<int> ids;
        {
            std::stringstream is(f.back());
            std::string t;
            while (std::getline(is, t, ','))
                if (!t.empty()) ids.push_back(std::atoi(t.c_str()));
        }
        if (ids.empty()) { std::printf("ERR GEN without prompt ids\n"); return; }
        for (int t : ids)
            if (t < 0 || t >= h.n_vocab) { std::printf("ERR token id %d is out of range\n", t); return; }
        if ((long long)ids.size() > (long long)eng.n_ctx()) {
            std::printf("ERR the prompt (%zu tokens) is longer than the context (%d)\n", ids.size(), eng.n_ctx());
            return;
        }
        stop.store(false);

        // what of the prompt the state already holds: all it holds, the last prompt's end, or nothing
        size_t reuse = 0;
        if (is_prefix(ss.live, ids) && !ss.live.empty() && (ss.live.size() < ids.size() || !ss.logits.empty())) {
            reuse = ss.live.size();
        } else if (ss.snap.valid() && is_prefix(ss.snap_tokens, ids) && ss.snap_tokens.size() < ids.size() &&
                   (size_t)eng.n_past() >= ss.snap_tokens.size() && eng.rollback(ss.snap)) {
            reuse = ss.snap_tokens.size();
            ss.live = ss.snap_tokens;
            ss.logits.clear();
        } else {
            eng.reset();
            ss.live.clear();
            ss.logits.clear();
        }
        if (reuse > 0) std::printf("RESUME %zu\n", reuse);
        std::fflush(stdout);

        const StoreStats s0 = store.stats();
        auto t0 = clk::now();
        const size_t fresh = ids.size() - reuse;
        size_t done = 0;
        bool cancelled = false;
        const int chunk = 256;
        try {
            while (done < fresh) {
                const int n = (int)std::min<size_t>((size_t)chunk, fresh - done);
                const bool last = done + n == fresh;
                eng.eval(ids.data() + reuse + done, n, last ? &ss.logits : nullptr);
                ss.live.insert(ss.live.end(), ids.begin() + (long)(reuse + done), ids.begin() + (long)(reuse + done + n));
                done += n;
                const double ms = ms_since(t0);
                std::printf("PP %zu %zu %.0f %.1f\n", done, fresh, ms, ms > 0 ? 1000.0 * done / ms : 0.0);
                std::fflush(stdout);
                if (stop.load() && done < fresh) { cancelled = true; break; }
            }
        } catch (const std::exception& e) {
            std::printf("ERR %s\n", e.what());
            eng.reset();
            ss = Session{};
            return;
        }
        const double prompt_ms = ms_since(t0);
        if (!cancelled) {
            eng.snapshot(ss.snap);
            ss.snap_tokens = ss.live;
        } else {
            ss.logits.clear();   // the state holds a prefix of the prompt; a later request can go on from it
        }
        if (cancelled || smp.prefill_only || max_new <= 0) {
            std::printf("DONE 0 %zu %.1f 0.0 %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n", ids.size(), prompt_ms,
                        cancelled ? "cancel" : "length", reuse, done);
            std::fflush(stdout);
            return;
        }

        std::mt19937_64 rng(smp.seed ? smp.seed : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::vector<int> recent(ids.end() - std::min<long>(1, (long)ids.size()), ids.end());   // the reply, and the token before it
        const StoreStats s1 = store.stats();
        auto d0 = clk::now();
        long long produced = 0;
        const char* finish = "length";
        std::vector<float> lg;
        try {
            for (;;) {
                lg = ss.logits;
                bool finite = true;
                for (float x : lg)
                    if (!std::isfinite(x)) { finite = false; break; }
                if (!finite) { finish = "nonfinite"; break; }
                const int t = sample_token(lg, smp, recent, rng);
                std::printf("T %d\n", t);
                if (smp.logprobs >= 0) std::printf("%s\n", logprob_line(ss.logits, t, smp.logprobs).c_str());
                std::fflush(stdout);
                ++produced;
                recent.push_back(t);
                if (is_eos(t)) { finish = "stop"; break; }
                if (produced >= max_new) break;
                if (stop.load()) { finish = "cancel"; break; }
                if (eng.n_past() + 1 > eng.n_ctx()) break;
                eng.eval(&t, 1, &ss.logits);
                ss.live.push_back(t);
            }
        } catch (const std::exception& e) {
            std::printf("ERR %s\n", e.what());
            eng.reset();
            ss = Session{};
            return;
        }
        const double decode_ms = ms_since(d0);
        const StoreStats s2 = store.stats();
        const uint64_t hits = s2.hits - s1.hits, lookups = hits + (s2.misses - s1.misses);
        const double file_mb = (double)(s2.bytes_read - s0.bytes_read) / 1e6;
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu %llu %llu %llu %llu %.1f %zu 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, reuse, (unsigned long long)hits, (unsigned long long)lookups,
                    (unsigned long long)hits, (unsigned long long)(lookups - hits), file_mb, fresh);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "inkling serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s); %lld generated in "
                     "%.0f ms (%.2f tok/s, %s); experts %.1f%% from RAM, %.0f MB from the SSD\n",
                     ids.size(), reuse, fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * fresh / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * produced / decode_ms : 0.0, finish,
                     lookups ? 100.0 * hits / lookups : 100.0, file_mb);
    };

    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return !lines.empty() || eof; });
            if (lines.empty()) break;
            line = lines.front();
            lines.pop_front();
        }
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) == 0) handle_gen(line);
        else if (line.rfind("GENI ", 0) == 0) std::printf("ERR pictures are not supported by the Inkling engine yet\n");
        else if (line.rfind("VRAM", 0) == 0) std::printf("ERR the Inkling engine runs on the CPU: there is no VRAM cache\n");
        else if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) std::printf("ERR SAVE and RESTORE are not supported by the Inkling engine\n");
        else if (line == "STATS") {
            StoreStats s = store.stats();
            std::printf("STATS hits=%llu misses=%llu read_mb=%.1f stall_s=%.2f prefetched=%llu prefetch_used=%llu\n",
                        (unsigned long long)s.hits, (unsigned long long)s.misses, s.bytes_read / 1e6, s.stall_s,
                        (unsigned long long)s.prefetched, (unsigned long long)s.prefetch_used);
        } else if (!line.empty()) std::printf("ERR unknown command: %s\n", line.substr(0, 40).c_str());
        std::fflush(stdout);
    }
    quit.store(true);
    if (reader.joinable()) reader.detach();
    return 0;
}

}  // namespace strata::inkling
