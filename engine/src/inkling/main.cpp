// src/inkling/main.cpp - strata-inkling: the Inkling 2 engine (thinkingmachines/Inkling-2-Mini GGUFs).
//
//   strata-inkling --model FILE [options] --serve                 the line protocol the Strata server speaks
//   strata-inkling --model FILE --info                            what the model is and how memory would be split
//   strata-inkling --model FILE --dump-logits OUT --ids-file F    logits of every position (parity tools)
//   strata-inkling --model FILE --bench [--prompt-len N --gen N]  prompt and decode speed, expert cache numbers
//
// Memory: the dense weights (1.1-1.7 GiB) are always in RAM. --experts auto (the default) keeps every expert in RAM
// when the budget holds them and otherwise streams them from the SSD through a cache of the rest of the budget.
#include "engine.hpp"
#include "expert_store.hpp"
#include "file_io.hpp"
#include "gpu.hpp"
#include "model.hpp"
#include "plan.hpp"
#include "serve.hpp"

#include "ggml.h"

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::inkling;
using clk = std::chrono::steady_clock;

namespace {

struct Args {
    std::string model, mode = "info", dump_prefix, ids_file;
    std::string experts = "auto", gpu = "auto";
    int ctx = 32768, threads = 0, chunk = 256, io_threads = 8, last = 0, prompt_len = 128, gen = 32;
    double ram_budget = 0, expert_cache = 0, vram_budget = -1;
    bool direct = true, lookahead = false, embed_ram = false;
};

void usage() {
    std::fprintf(stderr,
                 "usage: strata-inkling --model FILE [--serve | --info | --dump-logits PREFIX --ids-file F | --bench]\n"
                 "  --gpu auto|off|N    the dense weights and the hot experts on a GPU (auto: the first one, if this build\n"
                 "                      has GPU backends)\n"
                 "  --vram-budget GIB   VRAM the engine may use in all (default: free VRAM less 1 GiB); what the dense\n"
                 "                      weights leave holds the most-used experts. 0: the dense weights only\n"
                 "  --ctx N             context length (default 32768)\n"
                 "  --threads N         compute threads (default: the physical cores)\n"
                 "  --chunk N           prompt tokens per step (default 256)\n"
                 "  --experts MODE      auto (default), resident, mmap or stream\n"
                 "  --ram-budget GIB    RAM the engine may use in all (default: available RAM less a margin)\n"
                 "  --expert-cache GIB  stream: the expert cache size (overrides the budget split)\n"
                 "  --io-threads N      stream: parallel SSD reads (default 8)\n"
                 "  --no-direct-io      stream: read through the OS page cache\n"
                 "  --lookahead         stream: read each layer's likely experts during its attention (off: it read\n"
                 "                      more than it saved on the machines measured)\n"
                 "  --embed-ram         keep the token embedding table in RAM (default: read rows from the file)\n"
                 "  --last N            dump-logits: only the last N positions\n"
                 "  --prompt-len N --gen N   bench: prompt tokens and generated tokens\n");
}

Args parse(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) throw std::runtime_error(std::string(argv[i]) + " needs a value");
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--model" || s == "-m") a.model = need(i);
        else if (s == "--serve") a.mode = "serve";
        else if (s == "--info") a.mode = "info";
        else if (s == "--bench") a.mode = "bench";
        else if (s == "--dump-logits") { a.mode = "dump"; a.dump_prefix = need(i); }
        else if (s == "--ids-file") a.ids_file = need(i);
        else if (s == "--last") a.last = std::atoi(need(i));
        else if (s == "--ctx" || s == "--context" || s == "--max-context") a.ctx = std::atoi(need(i));
        else if (s == "--threads" || s == "-t") a.threads = std::atoi(need(i));
        else if (s == "--chunk" || s == "--prefill") a.chunk = std::atoi(need(i));
        else if (s == "--experts") a.experts = need(i);
        else if (s == "--gpu") a.gpu = need(i);
        else if (s == "--ram-budget") a.ram_budget = std::atof(need(i));
        else if (s == "--vram-budget") a.vram_budget = std::atof(need(i));
        else if (s == "--expert-cache") a.expert_cache = std::atof(need(i));
        else if (s == "--io-threads") a.io_threads = std::atoi(need(i));
        else if (s == "--no-direct-io") a.direct = false;
        else if (s == "--lookahead") a.lookahead = true;
        else if (s == "--no-lookahead") a.lookahead = false;
        else if (s == "--embed-ram") a.embed_ram = true;
        else if (s == "--prompt-len") a.prompt_len = std::atoi(need(i));
        else if (s == "--gen") a.gen = std::atoi(need(i));
        else if (s == "--help" || s == "-h") { usage(); std::exit(0); }
        else if (s.rfind("--", 0) == 0) {
            // an option the Strata server writes for its other engines (drafting, batching, ...): noted and skipped,
            // with its value when one follows, so a shared run config does not stop this engine from starting
            std::string v = i + 1 < argc && argv[i + 1][0] != '-' ? argv[++i] : "";
            std::fprintf(stderr, "inkling: note: ignoring %s%s%s (not an option of this engine)\n", s.c_str(),
                         v.empty() ? "" : " ", v.c_str());
        } else {
            throw std::runtime_error("unexpected argument " + s + " (--help lists the options)");
        }
    }
    if (a.model.empty()) throw std::runtime_error("--model is required");
    return a;
}

std::vector<int> read_ids(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<int> ids;
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".ids") == 0) {   // int32 binary (dump_logits)
        int32_t v;
        while (f.read(reinterpret_cast<char*>(&v), 4)) ids.push_back(v);
        return ids;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    for (char& c : s)
        if (c == ',') c = ' ';
    std::stringstream is(s);
    int v;
    while (is >> v) ids.push_back(v);
    return ids;
}

double gib(uint64_t b) { return b / 1073741824.0; }

}  // namespace

int main(int argc, char** argv) {
    try {
        Args a = parse(argc, argv);
        auto t_load = clk::now();
        // the GPU first: with one, the dense matrices go straight from the file to VRAM and never take RAM
        std::string why;
        std::unique_ptr<Gpu> gpu = a.mode == "info" ? nullptr : Gpu::open(a.gpu, a.threads > 0 ? a.threads : 4, why);
        if (!gpu && a.gpu != "auto" && a.gpu != "off" && a.mode != "info") throw std::runtime_error(why);
        Model model(a.model, !a.embed_ram, gpu != nullptr);
        if (gpu) {
            if (gpu->upload(model, why)) {
                std::fprintf(stderr, "inkling: GPU: %s, dense weights in VRAM (%.2f GiB)\n", gpu->name().c_str(),
                             gpu->weight_bytes() / 1073741824.0);
                // the experts get what the budget leaves after the dense weights and room for the graphs (0.5 GiB)
                const uint64_t G = 1ull << 30, reserve = G / 2;
                uint64_t room = 0;
                if (a.vram_budget >= 0) {
                    const uint64_t budget = (uint64_t)(a.vram_budget * (double)G);
                    room = budget > gpu->weight_bytes() + reserve ? budget - gpu->weight_bytes() - reserve : 0;
                    if (gpu->total_bytes()) room = std::min(room, gpu->free_bytes() > G / 2 ? gpu->free_bytes() - G / 2 : 0);
                } else if (gpu->total_bytes()) {
                    room = gpu->free_bytes() > G + reserve ? gpu->free_bytes() - G - reserve : 0;
                }
                if (room > 0) {
                    std::string ewhy;
                    if (gpu->init_experts(model, room, ewhy))
                        std::fprintf(stderr, "inkling: GPU: %d experts fit in %.2f GiB of VRAM (the most-used go there)\n",
                                     gpu->expert_slots(), gpu->expert_bytes() / (double)G);
                    else
                        std::fprintf(stderr, "inkling: GPU: no experts in VRAM (%s)\n", ewhy.c_str());
                }
            } else {
                std::fprintf(stderr, "inkling: GPU: %s; the dense weights stay in RAM\n", why.c_str());
                model.load_gpu_mats_to_ram();
                gpu.reset();
            }
        } else if (a.mode != "info") {
            std::fprintf(stderr, "inkling: GPU: none (%s); CPU only\n", why.c_str());
        }
        const Hparams& h = model.hp();
        std::fprintf(stderr, "inkling: %s (%s layout): %d layers, %d applications, %d experts (top %d), vocab %d\n",
                     a.model.c_str(), model.layout().c_str(), h.n_layer, (int)model.schedule().size(), h.n_expert,
                     h.n_expert_used, h.n_vocab);

        // the attention caches at the full context: 8 full slots keep every position, the others a ring
        uint64_t kv = 0;
        for (const Application& app : model.schedule()) {
            const bool swa = model.layers()[app.phys].swa;
            const uint64_t cap = swa ? std::min<uint64_t>(a.ctx, (uint64_t)h.n_swa + a.chunk) : (uint64_t)a.ctx;
            kv += cap * h.kv_dim() * 2 * 2;
        }
        const uint64_t scratch = (uint64_t)a.chunk * h.n_embd * 4 * 64 + (uint64_t)a.chunk * h.n_expert_used * h.n_ff_exp * 4 * 4 +
                                 (64ull << 20);
        MemoryPlan plan = plan_memory((uint64_t)(a.ram_budget * 1073741824.0), model.dense_bytes(), kv, scratch,
                                      model.expert_bytes_total(), (uint64_t)(a.expert_cache * 1073741824.0));
        std::fprintf(stderr, "inkling: memory: %s\n", plan.why.c_str());

        StoreOptions so;
        so.io_threads = a.io_threads;
        so.direct_io = a.direct;
        so.cache_bytes = plan.expert_cache;
        if (a.experts == "auto") so.mode = plan.all_fit ? StoreMode::Resident : StoreMode::Stream;
        else so.mode = parse_store_mode(a.experts);
        if (so.mode == StoreMode::Stream && so.cache_bytes == 0)
            throw std::runtime_error("the RAM budget leaves nothing for the expert cache; raise --ram-budget or set --expert-cache");

        if (a.mode == "info") {
            std::printf("layout %s\nlayers %d\napplications %d\ndense_gib %.3f\nexperts_gib %.3f\nexpert_mib %.2f\n"
                        "kv_gib_at_ctx %.3f\nbudget_gib %.3f\nexpert_cache_gib %.3f\nall_fit %d\n",
                        model.layout().c_str(), h.n_layer, (int)model.schedule().size(), gib(model.dense_bytes()),
                        gib(model.expert_bytes_total()), model.expert_bytes_max() / 1048576.0, gib(kv), gib(plan.budget),
                        gib(plan.expert_cache), plan.all_fit ? 1 : 0);
            return 0;
        }

        auto store = make_store(model, so);
        std::fprintf(stderr, "inkling: experts: %s\n", store->describe().c_str());
        EngineOptions eo;
        eo.gpu = gpu.get();
        eo.n_ctx = a.ctx;
        eo.threads = a.threads;
        eo.chunk = a.chunk;
        eo.lookahead = a.lookahead;
        Engine eng(model, *store, eo);
        std::fprintf(stderr, "inkling: ready in %.1f s, %d threads, context %d\n",
                     std::chrono::duration<double>(clk::now() - t_load).count(), eng.threads(), eng.n_ctx());

        if (a.mode == "serve") return run_serve(eng, *store, plan);

        if (a.mode == "dump") {
            std::vector<int> ids = read_ids(a.ids_file);
            if (ids.empty()) throw std::runtime_error("no token ids in " + a.ids_file);
            std::vector<float> all;
            auto t0 = clk::now();
            eng.eval(ids.data(), (int)ids.size(), nullptr, &all);
            const double dt = std::chrono::duration<double>(clk::now() - t0).count();
            const int nv = h.n_vocab, n = (int)ids.size();
            const int first = a.last > 0 && a.last < n ? n - a.last : 0;
            FILE* f = std::fopen((a.dump_prefix + ".logits").c_str(), "wb");
            if (!f) throw std::runtime_error("cannot write " + a.dump_prefix + ".logits");
            std::fwrite(all.data() + (size_t)first * nv, sizeof(float), (size_t)(n - first) * nv, f);
            std::fclose(f);
            std::fprintf(stderr, "inkling: %d tokens in %.2f s (%.1f tok/s), wrote %d positions\n", n, dt, n / dt, n - first);
            return 0;
        }

        if (a.mode == "bench") {
            std::vector<int> ids;
            if (!a.ids_file.empty()) ids = read_ids(a.ids_file);
            std::mt19937 rng(1234);
            while ((int)ids.size() < a.prompt_len) ids.push_back((int)(rng() % 150000) + 1000);
            ids.resize(a.prompt_len);
            std::vector<float> logits;
            store->reset_stats();
            auto t0 = clk::now();
            eng.eval(ids.data(), (int)ids.size(), &logits);
            const double tp = std::chrono::duration<double>(clk::now() - t0).count();
            StoreStats sp = store->stats();
            EngineStats ep = eng.stats();
            std::printf("prompt time: attention %.2f s, experts %.2f s, dense %.2f s, head %.2f s\n", ep.attn_s, ep.moe_s,
                        ep.dense_s, ep.head_s);
            store->reset_stats();
            eng.reset_stats();
            if (gpu) gpu->reset_expert_stats();
            auto t1 = clk::now();
            std::vector<int> out;
            for (int i = 0; i < a.gen; ++i) {
                int best = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
                out.push_back(best);
                eng.eval(&best, 1, &logits);
            }
            const double tg = std::chrono::duration<double>(clk::now() - t1).count();
            StoreStats sg = store->stats();
            EngineStats es = eng.stats();
            std::printf("prompt %d tokens: %.2f s, %.2f tok/s\n", a.prompt_len, tp, a.prompt_len / tp);
            std::printf("decode %d tokens: %.2f s, %.2f tok/s\n", a.gen, tg, a.gen / tg);
            auto rate = [](const StoreStats& s) { return s.hits + s.misses ? 100.0 * s.hits / (s.hits + s.misses) : 100.0; };
            std::printf("experts (prompt): hits %.1f%%, read %.2f GiB, stall %.2f s\n", rate(sp), gib(sp.bytes_read), sp.stall_s);
            std::printf("experts (decode): hits %.1f%%, read %.2f GiB (%.1f MiB/token), stall %.2f s, prefetched %llu used %llu\n",
                        rate(sg), gib(sg.bytes_read), sg.bytes_read / 1048576.0 / std::max(1, a.gen), sg.stall_s,
                        (unsigned long long)sg.prefetched, (unsigned long long)sg.prefetch_used);
            std::printf("ssd (decode): %.2f GiB in %.2f s of I/O-thread time\n", gib(sg.bytes_read), sg.read_s);
            if (gpu && gpu->has_experts()) {
                const GpuExpertStats gs = gpu->expert_stats();
                std::printf("vram experts (decode): %.1f%% of lookups on the GPU, %llu copied up (%.2f GiB)\n",
                            gs.lookups ? 100.0 * gs.hits / gs.lookups : 0.0, (unsigned long long)gs.uploads,
                            gib(gs.upload_bytes));
            }
            std::printf("decode time: attention %.2f s, experts %.2f s, dense %.2f s, head %.2f s\n", es.attn_s, es.moe_s,
                        es.dense_s, es.head_s);
#if !defined(_WIN32)
            {
                struct rusage ru {};
                getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
                const double peak = ru.ru_maxrss / 1073741824.0;   // bytes on macOS
#else
                const double peak = ru.ru_maxrss / 1048576.0;      // KiB on Linux
#endif
                std::printf("peak RAM (resident): %.2f GiB\n", peak);
            }
#endif
            std::printf("tokens:");
            for (int t : out) std::printf(" %d", t);
            std::printf("\n");
            return 0;
        }
        throw std::runtime_error("unknown mode " + a.mode);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inkling: error: %s\n", e.what());
        std::printf("ERR %s\n", e.what());
        std::fflush(stdout);
        return 1;
    }
}
