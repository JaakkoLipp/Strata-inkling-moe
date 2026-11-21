// src/inkling/expert_store.cpp - see expert_store.hpp.
#include "expert_store.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::inkling {

using clk = std::chrono::steady_clock;
static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }

StoreMode parse_store_mode(const std::string& s) {
    if (s == "resident" || s == "ram") return StoreMode::Resident;
    if (s == "mmap" || s == "mapped") return StoreMode::Mapped;
    if (s == "stream" || s == "ssd") return StoreMode::Stream;
    throw std::runtime_error("unknown expert mode '" + s + "' (resident, mmap or stream)");
}

const char* store_mode_name(StoreMode m) {
    switch (m) {
        case StoreMode::Resident: return "resident";
        case StoreMode::Mapped: return "mmap";
        case StoreMode::Stream: return "stream";
    }
    return "?";
}

static Mat expert_mat(const ExpertTensor& t, const uint8_t* data) {
    Mat m;
    m.type = t.type;
    m.cols = t.cols;
    m.rows = t.rows;
    m.row_bytes = t.row_bytes;
    m.data = data;
    return m;
}

// ------------------------------------------------------------------------------------------------ SlotCache

SlotCache::SlotCache(int n_slots, int n_keys) : slots_(n_slots), key_to_slot_(n_keys, -1) {}

int SlotCache::claim(int key, bool for_prefetch) {
    const int n = (int)slots_.size();
    if (n == 0) return -1;
    auto take = [&](int i) {
        Slot& s = slots_[i];
        if (s.key >= 0) {
            key_to_slot_[s.key] = -1;
            if (s.prefetch) ++wasted_;
        }
        s.key = key;
        s.uses = 0;   // a new expert earns its second chance by being used again (one-off experts leave first)
        s.prefetch = for_prefetch;
        s.state = State::Loading;
        key_to_slot_[key] = i;
        return i;
    };
    if (for_prefetch) {
        // a guess never pushes out an expert that is in use: only empty slots and ones whose count ran down
        for (int step = 0; step < n; ++step) {
            int i = hand_;
            hand_ = (hand_ + 1) % n;
            const Slot& s = slots_[i];
            if (s.state == State::Loading || s.pins > 0) continue;
            if (s.state == State::Empty || s.uses == 0) return take(i);
        }
        return -1;
    }
    for (int step = 0; step < 4 * n; ++step) {
        int i = hand_;
        hand_ = (hand_ + 1) % n;
        Slot& s = slots_[i];
        if (s.state == State::Loading || s.pins > 0) continue;
        if (s.state == State::Empty) return take(i);
        if (s.uses > 0) {
            --s.uses;
            continue;
        }
        return take(i);
    }
    return -1;
}

namespace {

// ------------------------------------------------------------------------------------------------ Resident

class ResidentStore final : public ExpertStore {
public:
    explicit ResidentStore(const Model& m) : m_(m) {
        const Hparams& h = m.hp();
        uint64_t total = 0;
        for (const Layer& L : m.layers()) total += (L.gate_up.bytes + L.down.bytes) * (uint64_t)h.n_expert;
        bytes_ = total;
        arena_ = static_cast<uint8_t*>(aligned_alloc_bytes((size_t)total, 64));
        ModelFile f(m.path(), false);
        uint64_t at = 0;
        auto t0 = clk::now();
        for (const Layer& L : m.layers()) {
            for (const ExpertTensor* t : {&L.gate_up, &L.down}) {
                const uint64_t n = t->bytes * (uint64_t)h.n_expert;
                base_.push_back(arena_ + at);
                f.read(arena_ + at, (size_t)n, t->offset);
                at += n;
            }
        }
        std::fprintf(stderr, "inkling: read all experts into RAM (%.2f GiB) in %.1f s\n", total / 1073741824.0, secs(clk::now() - t0));
    }
    ~ResidentStore() override { aligned_free(arena_); }
    StoreMode mode() const override { return StoreMode::Resident; }
    void acquire(int layer, const int* ids, int n, ExpertView* out) override {
        const Layer& L = m_.layers()[layer];
        for (int i = 0; i < n; ++i) {
            out[i].gate_up = expert_mat(L.gate_up, base_[2 * layer] + (uint64_t)ids[i] * L.gate_up.bytes);
            out[i].down = expert_mat(L.down, base_[2 * layer + 1] + (uint64_t)ids[i] * L.down.bytes);
        }
        hits_ += (uint64_t)n;
    }
    void release(int, const int*, int) override {}
    int max_pinned() const override { return m_.hp().n_expert; }
    uint64_t ram_bytes() const override { return bytes_; }
    StoreStats stats() const override { StoreStats s; s.hits = hits_; return s; }
    void reset_stats() override { hits_ = 0; }
    std::string describe() const override {
        char b[128];
        std::snprintf(b, sizeof b, "resident: all experts in RAM (%.2f GiB)", bytes_ / 1073741824.0);
        return b;
    }

private:
    const Model& m_;
    uint8_t* arena_ = nullptr;
    uint64_t bytes_ = 0;
    std::vector<const uint8_t*> base_;
    uint64_t hits_ = 0;
};

// ------------------------------------------------------------------------------------------------ Mapped

class MappedStore final : public ExpertStore {
public:
    explicit MappedStore(const Model& m) : m_(m) {
        size_ = m.gguf().file_size();
#if defined(_WIN32)
        int n = MultiByteToWideChar(CP_UTF8, 0, m.path().c_str(), -1, nullptr, 0);
        std::wstring w(n > 0 ? n - 1 : 0, L'\0');
        if (n > 1) MultiByteToWideChar(CP_UTF8, 0, m.path().c_str(), -1, w.data(), n);
        file_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + m.path());
        map_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_) throw std::runtime_error("cannot map " + m.path());
        base_ = static_cast<const uint8_t*>(MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0));
        if (!base_) throw std::runtime_error("cannot map " + m.path());
#else
        fd_ = ::open(m.path().c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("cannot open " + m.path());
        void* p = ::mmap(nullptr, (size_t)size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map " + m.path());
        base_ = static_cast<const uint8_t*>(p);
#endif
    }
    ~MappedStore() override {
#if defined(_WIN32)
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle(map_);
        if (file_ && file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (base_) ::munmap(const_cast<uint8_t*>(base_), (size_t)size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }
    StoreMode mode() const override { return StoreMode::Mapped; }
    void acquire(int layer, const int* ids, int n, ExpertView* out) override {
        const Layer& L = m_.layers()[layer];
        for (int i = 0; i < n; ++i) {
            out[i].gate_up = expert_mat(L.gate_up, base_ + L.gate_up.offset + (uint64_t)ids[i] * L.gate_up.bytes);
            out[i].down = expert_mat(L.down, base_ + L.down.offset + (uint64_t)ids[i] * L.down.bytes);
        }
        hits_ += (uint64_t)n;
    }
    void release(int, const int*, int) override {}
    void prefetch(int layer, const int* ids, int n) override {
#if !defined(_WIN32)
        const Layer& L = m_.layers()[layer];
        const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
        for (int i = 0; i < n; ++i) {
            for (const ExpertTensor* t : {&L.gate_up, &L.down}) {
                uintptr_t a = (uintptr_t)(base_ + t->offset + (uint64_t)ids[i] * t->bytes);
                uintptr_t a0 = a / page * page;
                ::madvise((void*)a0, (size_t)(a + t->bytes - a0), MADV_WILLNEED);
            }
        }
#else
        (void)layer; (void)ids; (void)n;
#endif
    }
    int max_pinned() const override { return m_.hp().n_expert; }
    uint64_t ram_bytes() const override { return 0; }
    StoreStats stats() const override { StoreStats s; s.hits = hits_; return s; }
    void reset_stats() override { hits_ = 0; }
    std::string describe() const override { return "mmap: experts mapped from the file, the OS page cache keeps them"; }

private:
    const Model& m_;
    uint64_t size_ = 0;
    const uint8_t* base_ = nullptr;
    uint64_t hits_ = 0;
#if defined(_WIN32)
    HANDLE file_ = nullptr, map_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ------------------------------------------------------------------------------------------------ Stream

class StreamStore final : public ExpertStore {
public:
    // Experts differ in size between layers (UD-Q4_K_XL: 4.6 MiB NVFP4 next to 8.1 MiB Q8_0). One slot size for all
    // would leave a third of the budget empty, so every expert size has its own pool of slots, and the budget is split
    // in proportion to what each size class holds in the file (every class keeps the same share of its experts).
    StreamStore(const Model& m, const StoreOptions& opt) : m_(m), file_(m.path(), opt.direct_io) {
        const Hparams& h = m.hp();
        const size_t A = ModelFile::kAlign;
        auto cap = [&](uint64_t bytes) { return (size_t)((bytes + 2 * A + A - 1) / A * A); };
        layer_pool_.assign(h.n_layer, -1);
        for (int l = 0; l < h.n_layer; ++l) {
            const Layer& L = m.layers()[l];
            const size_t cg = cap(L.gate_up.bytes), cd = cap(L.down.bytes);
            int p = -1;
            for (size_t i = 0; i < pools_.size(); ++i)
                if (pools_[i].cap_gu == cg && pools_[i].cap_down == cd) p = (int)i;
            if (p < 0) {
                pools_.emplace_back();
                p = (int)pools_.size() - 1;
                pools_[p].cap_gu = cg;
                pools_[p].cap_down = cd;
                pools_[p].slot_bytes = cg + cd;
            }
            pools_[p].layers++;
            layer_pool_[l] = p;
        }
        const int min_slots = 2 * h.n_expert_used;
        uint64_t total = 0, need_min = 0;
        for (const Pool& P : pools_) {
            total += (uint64_t)P.layers * h.n_expert * P.slot_bytes;
            need_min += (uint64_t)min_slots * P.slot_bytes;
        }
        if (opt.cache_bytes < need_min)
            throw std::runtime_error("the expert cache needs at least " + std::to_string((need_min + (1 << 20) - 1) >> 20) +
                                     " MiB for this model; raise --expert-cache or --ram-budget");
        const double share = std::min(1.0, (double)opt.cache_bytes / (double)total);
        for (Pool& P : pools_) {
            const int all = P.layers * h.n_expert;
            int n = (int)((double)all * share);
            n = std::max(min_slots, std::min(all, n));
            P.arena = static_cast<uint8_t*>(aligned_alloc_bytes((size_t)n * P.slot_bytes, A));
            P.cache = std::make_unique<SlotCache>(n, h.n_layer * h.n_expert);
            P.gu_off.assign(n, 0);
            P.down_off.assign(n, 0);
        }
        for (int i = 0; i < std::max(1, opt.io_threads); ++i) io_.emplace_back([this] { io_loop(); });
    }
    ~StreamStore() override {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        io_cv_.notify_all();
        for (auto& t : io_) t.join();
        for (Pool& P : pools_) aligned_free(P.arena);
    }

    StoreMode mode() const override { return StoreMode::Stream; }
    int max_pinned() const override {
        int m = INT32_MAX;
        for (const Pool& P : pools_) m = std::min(m, P.cache->n_slots() - m_.hp().n_expert_used);
        return std::max(1, m);
    }
    uint64_t ram_bytes() const override {
        uint64_t s = 0;
        for (const Pool& P : pools_) s += (uint64_t)P.cache->n_slots() * P.slot_bytes;
        return s;
    }

    void acquire(int layer, const int* ids, int n, ExpertView* out) override {
        const int E = m_.hp().n_expert;
        const int pi = layer_pool_[layer];
        Pool& P = pools_[pi];
        SlotCache& C = *P.cache;
        std::vector<int> slots(n);
        std::unique_lock<std::mutex> lk(mu_);
        bool queued = false;
        for (int i = 0; i < n; ++i) {
            const int key = layer * E + ids[i];
            int s = C.find(key);
            if (s >= 0) {
                auto& sl = C.slot(s);
                if (sl.prefetch) {
                    sl.prefetch = false;
                    ++stats_.prefetch_used;
                }
                ++stats_.hits;
                C.touch(s);
                if (sl.state == SlotCache::State::Loading) promote(pi, s);
            } else {
                // no slot: wait for a read in flight to finish (it can then be evicted); all pinned is a real dead end
                while ((s = C.claim(key, false)) < 0) {
                    if (!error_.empty()) throw std::runtime_error(error_);
                    bool loading = false;
                    for (int j = 0; j < C.n_slots() && !loading; ++j) loading = C.slot(j).state == SlotCache::State::Loading;
                    if (!loading) throw std::runtime_error("the expert cache has no free slot (all pinned); raise --expert-cache");
                    ready_cv_.wait(lk);
                }
                ++stats_.misses;
                demand_.push_back({pi, s});
                queued = true;
            }
            C.slot(s).pins++;
            slots[i] = s;
        }
        if (queued) io_cv_.notify_all();
        auto t0 = clk::now();
        bool waited = false;
        for (int i = 0; i < n; ++i) {
            while (C.slot(slots[i]).state != SlotCache::State::Ready) {
                if (!error_.empty()) throw std::runtime_error(error_);
                waited = true;
                ready_cv_.wait(lk);
            }
        }
        if (waited) stats_.stall_s += secs(clk::now() - t0);
        const Layer& L = m_.layers()[layer];
        for (int i = 0; i < n; ++i) {
            const uint8_t* base = P.arena + (size_t)slots[i] * P.slot_bytes;
            out[i].gate_up = expert_mat(L.gate_up, base + P.gu_off[slots[i]]);
            out[i].down = expert_mat(L.down, base + P.cap_gu + P.down_off[slots[i]]);
        }
    }

    void release(int layer, const int* ids, int n) override {
        const int E = m_.hp().n_expert;
        SlotCache& C = *pools_[layer_pool_[layer]].cache;
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < n; ++i) {
            int s = C.find(layer * E + ids[i]);
            if (s >= 0 && C.slot(s).pins > 0) C.slot(s).pins--;
        }
    }

    void prefetch(int layer, const int* ids, int n) override {
        const int E = m_.hp().n_expert;
        const int pi = layer_pool_[layer];
        SlotCache& C = *pools_[pi].cache;
        std::lock_guard<std::mutex> lk(mu_);
        bool queued = false;
        for (int i = 0; i < n; ++i) {
            const int key = layer * E + ids[i];
            if (C.find(key) >= 0) continue;
            int s = C.claim(key, true);
            if (s < 0) break;
            ++stats_.prefetched;
            spec_.push_back({pi, s});
            queued = true;
        }
        if (queued) io_cv_.notify_all();
    }

    StoreStats stats() const override {
        std::lock_guard<std::mutex> lk(mu_);
        StoreStats s = stats_;
        s.prefetch_wasted = 0;
        for (const Pool& P : pools_) s.prefetch_wasted += (uint64_t)P.cache->evicted_unused_prefetch();
        return s;
    }
    void reset_stats() override {
        std::lock_guard<std::mutex> lk(mu_);
        stats_ = StoreStats{};
    }
    std::string describe() const override {
        int slots = 0, all = 0;
        for (const Pool& P : pools_) {
            slots += P.cache->n_slots();
            all += P.layers * m_.hp().n_expert;
        }
        char b[256];
        std::snprintf(b, sizeof b, "stream: %d of %d experts fit in %.2f GiB of RAM (%zu slot sizes), %zu I/O threads, %s reads",
                      slots, all, ram_bytes() / 1073741824.0, pools_.size(), io_.size(), file_.direct() ? "direct" : "buffered");
        return b;
    }

private:
    struct Pool {
        int layers = 0;
        size_t cap_gu = 0, cap_down = 0, slot_bytes = 0;
        uint8_t* arena = nullptr;
        std::unique_ptr<SlotCache> cache;
        std::vector<size_t> gu_off, down_off;
    };
    struct Job {
        int pool, slot;
        bool operator==(const Job& o) const { return pool == o.pool && slot == o.slot; }
    };

    // a demand read overtakes the guesses queued before it
    void promote(int pi, int s) {
        auto it = std::find(spec_.begin(), spec_.end(), Job{pi, s});
        if (it != spec_.end()) {
            spec_.erase(it);
            demand_.push_front({pi, s});
            io_cv_.notify_all();
        }
    }

    void io_loop() {
        const int E = m_.hp().n_expert;
        for (;;) {
            Job j;
            int key;
            {
                std::unique_lock<std::mutex> lk(mu_);
                io_cv_.wait(lk, [&] { return stop_ || !demand_.empty() || !spec_.empty(); });
                if (stop_) return;
                if (!demand_.empty()) { j = demand_.front(); demand_.pop_front(); }
                else { j = spec_.front(); spec_.pop_front(); }
                key = pools_[j.pool].cache->slot(j.slot).key;
            }
            Pool& P = pools_[j.pool];
            const int layer = key / E, e = key % E;
            const Layer& L = m_.layers()[layer];
            uint8_t* base = P.arena + (size_t)j.slot * P.slot_bytes;
            auto t0 = clk::now();
            std::string err;
            size_t og = 0, od = 0;
            try {
                og = file_.read_aligned(base, (size_t)L.gate_up.bytes, L.gate_up.offset + (uint64_t)e * L.gate_up.bytes);
                od = file_.read_aligned(base + P.cap_gu, (size_t)L.down.bytes, L.down.offset + (uint64_t)e * L.down.bytes);
            } catch (const std::exception& ex) {
                err = ex.what();
            }
            const double dt = secs(clk::now() - t0);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!err.empty()) {
                    error_ = err;
                } else {
                    P.gu_off[j.slot] = og;
                    P.down_off[j.slot] = od;
                    P.cache->slot(j.slot).state = SlotCache::State::Ready;
                    stats_.bytes_read += L.gate_up.bytes + L.down.bytes;
                    stats_.read_s += dt;
                }
            }
            ready_cv_.notify_all();
        }
    }

    const Model& m_;
    ModelFile file_;
    std::vector<Pool> pools_;
    std::vector<int> layer_pool_;
    mutable std::mutex mu_;
    std::condition_variable io_cv_, ready_cv_;
    std::deque<Job> demand_, spec_;
    std::vector<std::thread> io_;
    bool stop_ = false;
    std::string error_;
    StoreStats stats_;
};

}  // namespace

std::unique_ptr<ExpertStore> make_store(const Model& model, const StoreOptions& opt) {
    switch (opt.mode) {
        case StoreMode::Resident: return std::make_unique<ResidentStore>(model);
        case StoreMode::Mapped: return std::make_unique<MappedStore>(model);
        case StoreMode::Stream: return std::make_unique<StreamStore>(model, opt);
    }
    throw std::runtime_error("unknown expert store mode");
}

}  // namespace strata::inkling
