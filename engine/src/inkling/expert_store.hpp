// src/inkling/expert_store.hpp - where the experts of an Inkling model live while it runs.
//
// The experts are 85-90% of every Inkling-2-Mini GGUF (6 GiB of 7.2 in UD-IQ1_S, 14.7 of 16.4 in UD-Q4_K_XL) and a
// token reads only 8 of the 128 in each of its 48 layer applications. So on a computer with little RAM the dense
// weights (1.1-1.7 GiB) stay in RAM and the experts are a cache in front of the SSD:
//
//   * `Resident`: every expert read into RAM at start. For RAM that holds the whole model.
//   * `Mapped`:   the file is memory-mapped and the OS page cache decides what stays. Simple; it competes with the
//                 rest of the system for RAM and reads 4-128 KiB at a time.
//   * `Stream`:   a fixed RAM budget of expert slots. A missing expert is read whole (gate_up + down, 2-5 MiB) with
//                 direct I/O by a pool of I/O threads, so the SSD sees large reads at a high queue depth and nothing
//                 lands in the page cache. Slots in use are pinned; the rest are evicted by a CLOCK with use counts
//                 (frequently used experts survive a burst of one-off ones).
//
// The loop core helps the cache: layers 6-17 run three times per token with the same weights, so the experts of the
// second and third pass are often still in RAM from the first.
#pragma once

#include "file_io.hpp"
#include "model.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::inkling {

/// One expert in RAM: its gate_up matrix (2 n_ff rows) and its down matrix (n_embd rows).
struct ExpertView {
    Mat gate_up, down;
};

struct StoreStats {
    uint64_t hits = 0, misses = 0, prefetched = 0, prefetch_used = 0, prefetch_wasted = 0;
    uint64_t bytes_read = 0;
    double read_s = 0;        ///< summed time the I/O threads spent reading
    double stall_s = 0;       ///< time compute waited for experts
};

enum class StoreMode { Resident, Mapped, Stream };
StoreMode parse_store_mode(const std::string& s);
const char* store_mode_name(StoreMode m);

class ExpertStore {
public:
    virtual ~ExpertStore() = default;
    virtual StoreMode mode() const = 0;
    /// Makes experts ids[0..n) of physical layer `layer` available and pins them; out[i] is valid until release().
    /// Blocks until all are in RAM.
    virtual void acquire(int layer, const int* ids, int n, ExpertView* out) = 0;
    virtual void release(int layer, const int* ids, int n) = 0;
    /// acquire() in two halves, so that a layer can compute the experts it has while the others are read: begin() pins
    /// the experts and starts the reads without waiting; wait() waits for some of them and gives their views. Every
    /// begun expert is released once.
    virtual void begin(int layer, const int* ids, int n) { (void)layer; (void)ids; (void)n; }
    virtual void wait(int layer, const int* ids, int n, ExpertView* out) { acquire(layer, ids, n, out); }
    /// Whether an expert is in RAM now (for ordering work; it can change once the engine claims slots).
    virtual bool ready(int layer, int id) const { (void)layer; (void)id; return true; }
    /// A hint: these experts will probably be needed soon. Never blocks; may do nothing.
    virtual void prefetch(int layer, const int* ids, int n) { (void)layer; (void)ids; (void)n; }
    /// How many experts can be pinned at the same time (the MoE runs a prompt's experts in waves of at most this).
    virtual int max_pinned() const = 0;
    virtual uint64_t ram_bytes() const = 0;
    virtual StoreStats stats() const = 0;
    virtual void reset_stats() {}
    virtual std::string describe() const = 0;
};

struct StoreOptions {
    StoreMode mode = StoreMode::Stream;
    uint64_t cache_bytes = 0;     ///< Stream: the RAM budget for expert slots
    int io_threads = 4;           ///< Stream: parallel reads
    bool direct_io = true;        ///< Stream: bypass the page cache where the OS allows
};

std::unique_ptr<ExpertStore> make_store(const Model& model, const StoreOptions& opt);

// ------------------------------------------------------------------------------------------------------------------
// The slot cache of the Stream store, apart from the I/O so it can be tested without a model.

class SlotCache {
public:
    enum class State : uint8_t { Empty, Loading, Ready };
    struct Slot {
        int key = -1;
        State state = State::Empty;
        int pins = 0;
        uint8_t uses = 0;          ///< CLOCK counter, saturating
        bool prefetch = false;     ///< loaded on a hint and not used yet
    };

    SlotCache(int n_slots, int n_keys);
    int n_slots() const { return (int)slots_.size(); }
    /// The slot holding `key`, or -1.
    int find(int key) const { return key_to_slot_[key]; }
    /// A slot for `key` that is not pinned and not loading, evicting its old key (the CLOCK victim); -1 if none.
    int claim(int key, bool for_prefetch);
    Slot& slot(int i) { return slots_[i]; }
    const Slot& slot(int i) const { return slots_[i]; }
    void touch(int i) {
        Slot& s = slots_[i];
        if (s.uses < 3) ++s.uses;
    }
    int evicted_unused_prefetch() const { return wasted_; }

private:
    std::vector<Slot> slots_;
    std::vector<int> key_to_slot_;
    int hand_ = 0;
    int wasted_ = 0;
};

}  // namespace strata::inkling
