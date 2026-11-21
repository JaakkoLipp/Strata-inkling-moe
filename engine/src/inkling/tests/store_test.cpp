// src/inkling/tests/store_test.cpp - the expert slot cache (CLOCK with use counts, pins, guesses), the RAM budget
// split, and the aligned direct reads of a scratch file. No model, no GPU.
#include "dot_kernels.hpp"
#include "expert_store.hpp"
#include "file_io.hpp"
#include "plan.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <random>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::inkling;

static int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

static void ready(SlotCache& c, int s) { c.slot(s).state = SlotCache::State::Ready; }

static void test_claim_fills_empty_slots_first() {
    SlotCache c(4, 100);
    int a = c.claim(10, false), b = c.claim(11, false), d = c.claim(12, false), e = c.claim(13, false);
    CHECK(a >= 0 && b >= 0 && d >= 0 && e >= 0);
    CHECK(a != b && b != d && d != e && a != e);
    CHECK(c.find(10) == a && c.find(13) == e);
    // every slot is loading: nothing can be evicted
    CHECK(c.claim(14, false) == -1);
}

static void test_pinned_slots_survive() {
    SlotCache c(2, 100);
    int a = c.claim(1, false), b = c.claim(2, false);
    ready(c, a);
    ready(c, b);
    c.slot(a).pins = 1;
    int n = c.claim(3, false);
    CHECK(n == b);                 // the unpinned one goes
    CHECK(c.find(2) == -1);
    CHECK(c.find(1) == a);
    CHECK(c.find(3) == b);
}

static void test_frequent_expert_outlives_one_offs() {
    // key 1 is used again and again; keys 100.. are each used once: CLOCK with counts keeps key 1
    SlotCache c(4, 1000);
    int hot = c.claim(1, false);
    ready(c, hot);
    for (int i = 0; i < 3; ++i) c.touch(hot);
    for (int k = 100; k < 140; ++k) {
        int s = c.claim(k, false);
        CHECK(s >= 0);
        ready(c, s);
        if (k % 2 == 0) c.touch(hot);   // the hot expert keeps being used
    }
    CHECK(c.find(1) == hot);
}

static void test_guess_never_evicts_used_experts() {
    SlotCache c(2, 100);
    int a = c.claim(1, false), b = c.claim(2, false);
    ready(c, a);
    ready(c, b);
    c.touch(a);
    c.touch(b);
    // both have a use count: a guess finds no slot and changes nothing
    CHECK(c.claim(50, true) == -1);
    CHECK(c.find(1) == a && c.find(2) == b);
    CHECK(c.slot(a).uses > 0 && c.slot(b).uses > 0);
}

static void test_unused_guess_is_counted_when_evicted() {
    SlotCache c(1, 100);
    int s = c.claim(7, true);
    CHECK(s == 0);
    ready(c, s);
    CHECK(c.slot(s).prefetch);
    int t = c.claim(8, false);
    CHECK(t == 0);
    CHECK(c.evicted_unused_prefetch() == 1);
    CHECK(c.find(7) == -1 && c.find(8) == 0);
}

static void test_plan_split() {
    const uint64_t G = 1ull << 30;
    MemoryPlan p = plan_memory(8 * G, 2 * G, 1 * G, G / 2, 15 * G, 0);
    CHECK(p.budget == 8 * G);
    CHECK(p.expert_cache == 8 * G - 2 * G - 1 * G - G / 2);
    CHECK(!p.all_fit);
    MemoryPlan q = plan_memory(32 * G, 2 * G, 1 * G, G / 2, 15 * G, 0);
    CHECK(q.all_fit);
    MemoryPlan r = plan_memory(8 * G, 2 * G, 1 * G, G / 2, 15 * G, 3 * G);
    CHECK(r.expert_cache == 3 * G);
    MemoryPlan z = plan_memory(2 * G, 2 * G, 1 * G, G / 2, 15 * G, 0);
    CHECK(z.expert_cache == 0);
}

static void test_aligned_reads(const std::string& dir) {
    const std::string path = dir + "/inkling_store_test.bin";
    const size_t n = 3 * ModelFile::kAlign + 777;
    std::vector<uint8_t> data(n);
    for (size_t i = 0; i < n; ++i) data[i] = (uint8_t)(i * 131 + 7);
    FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) return;
    std::fwrite(data.data(), 1, n, f);
    std::fclose(f);
    for (bool direct : {false, true}) {
        ModelFile mf(path, direct);
        // unaligned starts, a span that runs past the end of the file, and a read inside one block
        const uint64_t offs[] = {0, 1, 4095, 4096, 5000, n - 900};
        for (uint64_t off : offs) {
            const size_t len = (size_t)std::min<uint64_t>(900, n - off);
            uint8_t* buf = static_cast<uint8_t*>(aligned_alloc_bytes(ModelFile::span_bytes(off, len), ModelFile::kAlign));
            const size_t at = mf.read_aligned(buf, len, off);
            CHECK(std::memcmp(buf + at, data.data() + off, len) == 0);
            aligned_free(buf);
        }
    }
    std::remove(path.c_str());
}

// the engine's Q2_0 kernels (SIMD where the vendored ggml has a scalar one) against a double-precision sum: within
// float rounding of the sum of |terms|, as ggml's scalar loop is
static void test_q2_0_dot() {
    ggml_cpu_init();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte(0, 255), q8(-127, 127);
    std::uniform_real_distribution<float> sc(0.001f, 2.0f);
    for (int which = 0; which <= 2; ++which) {
        ggml_vec_dot_t vd = q2_0_kernel(which);
        if (!vd) continue;
        for (int n : {64, 1152, 2304}) {
            const int nb = n / 64;
            std::vector<uint8_t> x((size_t)nb * 18), y((size_t)nb * 2 * 34);
            for (int trial = 0; trial < 50; ++trial) {
                for (int i = 0; i < nb; ++i) {
                    const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng) * (trial % 2 ? -1.0f : 1.0f));
                    std::memcpy(&x[(size_t)i * 18], &d, 2);
                    for (int j = 0; j < 16; ++j) x[(size_t)i * 18 + 2 + j] = (uint8_t)byte(rng);
                }
                for (int i = 0; i < 2 * nb; ++i) {
                    const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng));
                    std::memcpy(&y[(size_t)i * 34], &d, 2);
                    for (int j = 0; j < 32; ++j) y[(size_t)i * 34 + 2 + j] = (uint8_t)(int8_t)q8(rng);
                }
                double ref = 0, mag = 0;
                for (int i = 0; i < nb; ++i) {
                    ggml_fp16_t h;
                    std::memcpy(&h, &x[(size_t)i * 18], 2);
                    const double d0 = ggml_fp16_to_fp32(h);
                    for (int j = 0; j < 64; ++j) {
                        const int q = ((x[(size_t)i * 18 + 2 + j / 4] >> ((j % 4) * 2)) & 3) - 1;
                        const int yb = i * 2 + j / 32;
                        std::memcpy(&h, &y[(size_t)yb * 34], 2);
                        const double d1 = ggml_fp16_to_fp32(h);
                        const double t = d0 * d1 * q * (int8_t)y[(size_t)yb * 34 + 2 + j % 32];
                        ref += t;
                        mag += std::fabs(t);
                    }
                }
                float got = 0;
                vd(n, &got, 0, x.data(), 0, y.data(), 0, 1);
                CHECK(std::fabs(got - ref) <= 1e-6 * mag + 1e-6);
            }
        }
    }
    std::printf("q2_0 dot: the engine uses %s\n", q2_0_kernel_name());
}

int main(int argc, char** argv) {
    test_q2_0_dot();
    test_claim_fills_empty_slots_first();
    test_pinned_slots_survive();
    test_frequent_expert_outlives_one_offs();
    test_guess_never_evicts_used_experts();
    test_unused_guess_is_counted_when_evicted();
    test_plan_split();
    const char* tmp = std::getenv("TMPDIR");
    test_aligned_reads(argc > 1 ? argv[1] : (tmp ? tmp : "."));
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("inkling store test: ok\n");
    return 0;
}
