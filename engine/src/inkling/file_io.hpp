// src/inkling/file_io.hpp - positional reads of a model file, from any thread, optionally past the OS page cache.
//
// Why direct I/O: on a computer with little RAM the expert store's own cache is the only cache that should hold
// experts. A buffered read leaves a second copy in the page cache, which the kernel then has to reclaim from the
// same small RAM. With direct I/O (O_DIRECT on Linux, F_NOCACHE on macOS, FILE_FLAG_NO_BUFFERING on Windows) the
// bytes go from the SSD into the store's buffer and nowhere else. Direct reads need the file offset, the length and
// the buffer aligned to the device's block; `read_aligned` reads the aligned span around a request and says where
// the requested bytes start inside it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::inkling {

class ModelFile {
public:
    /// Opens `path` for reading. With `direct`, tries the platform's uncached mode and falls back to buffered reads
    /// (with a hint to drop the pages afterwards) where the file system refuses it; `direct()` says which one is on.
    ModelFile(const std::string& path, bool direct);
    ~ModelFile();
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    bool direct() const { return direct_; }
    /// The alignment a direct read needs (offset, length and buffer). 1 for buffered reads.
    size_t alignment() const { return direct_ ? kAlign : 1; }
    static constexpr size_t kAlign = 4096;

    /// Reads exactly `n` bytes at `offset` into `dst` with buffered I/O (any alignment). Throws on a short read.
    void read(void* dst, size_t n, uint64_t offset) const;

    /// Reads the bytes [offset, offset + n) into `buf` (which must be `kAlign`-aligned and hold `span_bytes(offset, n)`)
    /// and returns the position of `offset` inside `buf`. Direct when the file is open for it.
    size_t read_aligned(uint8_t* buf, size_t n, uint64_t offset) const;

    /// The buffer size `read_aligned` needs for a request of `n` bytes at `offset`, at most `n + 2 * kAlign`.
    static size_t span_bytes(uint64_t offset, size_t n) {
        uint64_t a0 = offset / kAlign * kAlign;
        uint64_t a1 = (offset + n + kAlign - 1) / kAlign * kAlign;
        return (size_t)(a1 - a0);
    }

    /// Tells the OS that the buffered pages of [offset, offset + n) are not needed again (no-op where unsupported).
    void drop_cache(uint64_t offset, uint64_t n) const;

private:
    void read_raw(int which, void* dst, size_t n, uint64_t offset) const;   // which: 0 buffered, 1 direct
    std::string path_;
    bool direct_ = false;
#if defined(_WIN32)
    void* h_ = nullptr;        // buffered handle
    void* hd_ = nullptr;       // FILE_FLAG_NO_BUFFERING handle
#else
    int fd_ = -1;              // buffered
    int fdd_ = -1;             // direct (Linux O_DIRECT); on macOS the same file with F_NOCACHE
#endif
};

/// Memory with the alignment direct reads need. Freed with `aligned_free`.
void* aligned_alloc_bytes(size_t n, size_t align);
void aligned_free(void* p);

/// What the OS reports as available RAM (bytes), or 0 when unknown. Counts reclaimable page cache as available.
uint64_t available_ram_bytes();
/// Total physical RAM (bytes), or 0 when unknown.
uint64_t total_ram_bytes();

}  // namespace strata::inkling
