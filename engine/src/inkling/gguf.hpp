// src/inkling/gguf.hpp - a small GGUF reader for the Inkling engine: the metadata, the tensor table and where each
// tensor's bytes are in the file. It reads only the header; tensor bytes are read later by whoever needs them (the
// dense weights once at load, the experts on demand by the expert store).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace strata::inkling {

/// One metadata value. Arrays keep their elements in the widest type of their kind.
struct GgufValue {
    enum class Kind { Int, Float, Bool, String, IntArray, FloatArray, BoolArray, StringArray };
    Kind kind = Kind::Int;
    int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    std::vector<int64_t> ia;
    std::vector<double> fa;
    std::vector<bool> ba;
    std::vector<std::string> sa;
};

struct GgufTensor {
    std::string name;
    int type = 0;                 ///< ggml_type
    std::vector<int64_t> ne;      ///< dims, ne[0] is the row length
    uint64_t offset = 0;          ///< absolute file offset of the first byte
    uint64_t nbytes = 0;          ///< whole tensor
    int64_t nelements() const {
        int64_t n = 1;
        for (int64_t d : ne) n *= d;
        return n;
    }
};

class GgufFile {
public:
    /// Reads the header of `path`. Throws std::runtime_error with the reason on a bad or truncated file.
    explicit GgufFile(const std::string& path);

    const std::string& path() const { return path_; }
    uint64_t file_size() const { return file_size_; }
    uint64_t data_offset() const { return data_offset_; }

    bool has(const std::string& key) const { return kv_.count(key) != 0; }
    const GgufValue* get(const std::string& key) const;
    int64_t get_int(const std::string& key, int64_t def) const;
    double get_float(const std::string& key, double def) const;
    bool get_bool(const std::string& key, bool def) const;
    std::string get_string(const std::string& key, const std::string& def) const;

    const GgufTensor* tensor(const std::string& name) const;
    const std::vector<GgufTensor>& tensors() const { return tensors_; }
    const std::map<std::string, GgufValue>& kv() const { return kv_; }

private:
    std::string path_;
    uint64_t file_size_ = 0;
    uint64_t data_offset_ = 0;
    std::map<std::string, GgufValue> kv_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, size_t> by_name_;
};

/// Bytes of `n` elements of ggml type `type` (n a multiple of the block size). Throws for an unknown type.
uint64_t ggml_bytes(int type, int64_t n);

}  // namespace strata::inkling
