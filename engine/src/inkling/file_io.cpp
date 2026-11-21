// src/inkling/file_io.cpp - see file_io.hpp.
#include "file_io.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
#endif

namespace strata::inkling {

#if defined(_WIN32)

static std::wstring widen(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    return w;
}

ModelFile::ModelFile(const std::string& path, bool direct) : path_(path) {
    std::wstring w = widen(path);
    h_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h_ == INVALID_HANDLE_VALUE) { h_ = nullptr; throw std::runtime_error("cannot open " + path); }
    if (direct) {
        hd_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
        if (hd_ == INVALID_HANDLE_VALUE) hd_ = nullptr;
        direct_ = hd_ != nullptr;
    }
}

ModelFile::~ModelFile() {
    if (h_) CloseHandle(h_);
    if (hd_) CloseHandle(hd_);
}

void ModelFile::read_raw(int which, void* dst, size_t n, uint64_t offset) const {
    HANDLE h = which ? (HANDLE)hd_ : (HANDLE)h_;
    auto* p = static_cast<uint8_t*>(dst);
    while (n) {
        OVERLAPPED ov{};
        ov.Offset = (DWORD)(offset & 0xffffffffu);
        ov.OffsetHigh = (DWORD)(offset >> 32);
        DWORD want = (DWORD)std::min<size_t>(n, 1u << 30), got = 0;
        if (!ReadFile(h, p, want, &got, &ov) || got == 0)
            throw std::runtime_error("read failed in " + path_ + " at " + std::to_string(offset));
        p += got; n -= got; offset += got;
    }
}

void ModelFile::drop_cache(uint64_t, uint64_t) const {}

void* aligned_alloc_bytes(size_t n, size_t align) {
    void* p = _aligned_malloc(n ? n : align, align);
    if (!p) throw std::bad_alloc();
    return p;
}
void aligned_free(void* p) { _aligned_free(p); }

uint64_t available_ram_bytes() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys : 0;
}
uint64_t total_ram_bytes() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? m.ullTotalPhys : 0;
}

#else  // POSIX

ModelFile::ModelFile(const std::string& path, bool direct) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    if (!direct) return;
#if defined(__linux__)
    fdd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fdd_ >= 0) {
        // some file systems accept O_DIRECT at open and refuse the read (EINVAL): probe one block
        void* probe = aligned_alloc_bytes(kAlign, kAlign);
        ssize_t r = ::pread(fdd_, probe, kAlign, 0);
        aligned_free(probe);
        if (r < 0) { ::close(fdd_); fdd_ = -1; }
    }
#elif defined(__APPLE__)
    fdd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fdd_ >= 0 && fcntl(fdd_, F_NOCACHE, 1) != 0) { ::close(fdd_); fdd_ = -1; }
#endif
    direct_ = fdd_ >= 0;
}

ModelFile::~ModelFile() {
    if (fd_ >= 0) ::close(fd_);
    if (fdd_ >= 0) ::close(fdd_);
}

void ModelFile::read_raw(int which, void* dst, size_t n, uint64_t offset) const {
    int fd = which ? fdd_ : fd_;
    auto* p = static_cast<uint8_t*>(dst);
    while (n) {
        ssize_t r = ::pread(fd, p, n, (off_t)offset);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0)
            throw std::runtime_error("read failed in " + path_ + " at " + std::to_string(offset) + ": " +
                                     (r < 0 ? std::strerror(errno) : "end of file"));
        p += r; n -= (size_t)r; offset += (uint64_t)r;
    }
}

void ModelFile::drop_cache(uint64_t offset, uint64_t n) const {
#if defined(__linux__)
    posix_fadvise(fd_, (off_t)offset, (off_t)n, POSIX_FADV_DONTNEED);
#else
    (void)offset; (void)n;
#endif
}

void* aligned_alloc_bytes(size_t n, size_t align) {
    void* p = nullptr;
    if (posix_memalign(&p, align, n ? n : align) != 0) throw std::bad_alloc();
    return p;
}
void aligned_free(void* p) { std::free(p); }

#if defined(__linux__)
static uint64_t meminfo_kb(const char* key) {
    std::ifstream f("/proc/meminfo");
    std::string k;
    uint64_t v = 0;
    std::string unit;
    while (f >> k >> v >> unit)
        if (k == key) return v;
    return 0;
}
// a cgroup v2 limit (a container): what is left under memory.max, counting its own clean page cache as free
static uint64_t cgroup_room() {
    std::ifstream fm("/sys/fs/cgroup/memory.max"), fc("/sys/fs/cgroup/memory.current"), fs("/sys/fs/cgroup/memory.stat");
    std::string max;
    uint64_t cur = 0;
    if (!(fm >> max) || max == "max" || !(fc >> cur)) return 0;
    uint64_t lim = std::strtoull(max.c_str(), nullptr, 10);
    uint64_t file = 0;
    std::string k;
    uint64_t v;
    while (fs >> k >> v)
        if (k == "inactive_file") file = v;
    uint64_t used = cur > file ? cur - file : 0;
    return lim > used ? lim - used : 1;
}
#endif

uint64_t available_ram_bytes() {
#if defined(__linux__)
    uint64_t avail = meminfo_kb("MemAvailable:") * 1024;
    uint64_t cg = cgroup_room();
    if (cg && (avail == 0 || cg < avail)) avail = cg;
    return avail;
#elif defined(__APPLE__)
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) return 0;
    uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    return (uint64_t)(vm.free_count + vm.inactive_count + vm.purgeable_count) * page;
#else
    return 0;
#endif
}

uint64_t total_ram_bytes() {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof mem;
    return sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0 ? mem : 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? (uint64_t)pages * (uint64_t)page : 0;
#endif
}

#endif

void ModelFile::read(void* dst, size_t n, uint64_t offset) const { read_raw(0, dst, n, offset); }

size_t ModelFile::read_aligned(uint8_t* buf, size_t n, uint64_t offset) const {
    if (!direct_) {
        read_raw(0, buf, n, offset);
        drop_cache(offset, n);
        return 0;
    }
    uint64_t a0 = offset / kAlign * kAlign;
    size_t span = span_bytes(offset, n);
    const size_t need = (size_t)(offset - a0) + n;   // the aligned span may run past the end of the file
    size_t got = 0;
    while (got < need) {
        size_t want = span - got;
#if defined(_WIN32)
        OVERLAPPED ov{};
        uint64_t at = a0 + got;
        ov.Offset = (DWORD)(at & 0xffffffffu);
        ov.OffsetHigh = (DWORD)(at >> 32);
        DWORD r = 0;
        if (!ReadFile((HANDLE)hd_, buf + got, (DWORD)want, &r, &ov) || r == 0)
            throw std::runtime_error("direct read failed in " + path_ + " at " + std::to_string(at));
#else
        ssize_t r = ::pread(fdd_, buf + got, want, (off_t)(a0 + got));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0)
            throw std::runtime_error("direct read failed in " + path_ + " at " + std::to_string(a0 + got) + ": " +
                                     (r < 0 ? std::strerror(errno) : "end of file"));
#endif
        got += (size_t)r;
    }
    return (size_t)(offset - a0);
}

}  // namespace strata::inkling
