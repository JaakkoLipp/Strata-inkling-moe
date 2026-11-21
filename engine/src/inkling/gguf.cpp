// src/inkling/gguf.cpp - see gguf.hpp.
#include "gguf.hpp"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::inkling {
namespace {

// a buffered reader over stdio: the header of a GGUF holds the whole vocabulary (hundreds of thousands of strings)
class Reader {
public:
    explicit Reader(const std::string& path) {
        f_ = std::fopen(path.c_str(), "rb");
        if (!f_) throw std::runtime_error("cannot open " + path);
        buf_.resize(1 << 20);
    }
    ~Reader() { if (f_) std::fclose(f_); }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    void read(void* dst, size_t n) {
        auto* out = static_cast<uint8_t*>(dst);
        while (n) {
            if (pos_ == len_) {
                len_ = std::fread(buf_.data(), 1, buf_.size(), f_);
                pos_ = 0;
                if (len_ == 0) throw std::runtime_error("the GGUF header is truncated");
            }
            size_t k = std::min(n, len_ - pos_);
            std::memcpy(out, buf_.data() + pos_, k);
            pos_ += k; out += k; n -= k; consumed_ += k;
        }
    }
    template <class T> T get() { T v; read(&v, sizeof v); return v; }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (n > (1ull << 30)) throw std::runtime_error("a GGUF string is implausibly long");
        std::string s(n, '\0');
        if (n) read(s.data(), n);
        return s;
    }
    uint64_t consumed() const { return consumed_; }
    uint64_t file_size() {
        long cur = std::ftell(f_);
        std::fseek(f_, 0, SEEK_END);
#if defined(_WIN32)
        uint64_t sz = (uint64_t)_ftelli64(f_);
#else
        uint64_t sz = (uint64_t)ftello(f_);
#endif
        std::fseek(f_, cur, SEEK_SET);
        return sz;
    }

private:
    std::FILE* f_ = nullptr;
    std::vector<uint8_t> buf_;
    size_t pos_ = 0, len_ = 0;
    uint64_t consumed_ = 0;
};

enum : uint32_t { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

bool is_int(uint32_t t) { return t <= T_I32 || t == T_U64 || t == T_I64; }

int64_t read_int(Reader& r, uint32_t t) {
    switch (t) {
        case T_U8: return r.get<uint8_t>();
        case T_I8: return r.get<int8_t>();
        case T_U16: return r.get<uint16_t>();
        case T_I16: return r.get<int16_t>();
        case T_U32: return r.get<uint32_t>();
        case T_I32: return r.get<int32_t>();
        case T_U64: return (int64_t)r.get<uint64_t>();
        case T_I64: return r.get<int64_t>();
        default: throw std::runtime_error("not an integer GGUF type");
    }
}

GgufValue read_value(Reader& r, uint32_t t) {
    GgufValue v;
    if (is_int(t)) { v.kind = GgufValue::Kind::Int; v.i = read_int(r, t); }
    else if (t == T_F32) { v.kind = GgufValue::Kind::Float; v.f = r.get<float>(); }
    else if (t == T_F64) { v.kind = GgufValue::Kind::Float; v.f = r.get<double>(); }
    else if (t == T_BOOL) { v.kind = GgufValue::Kind::Bool; v.b = r.get<uint8_t>() != 0; }
    else if (t == T_STR) { v.kind = GgufValue::Kind::String; v.s = r.str(); }
    else if (t == T_ARR) {
        uint32_t et = r.get<uint32_t>();
        uint64_t n = r.get<uint64_t>();
        if (n > (1ull << 28)) throw std::runtime_error("a GGUF array is implausibly long");
        if (is_int(et)) { v.kind = GgufValue::Kind::IntArray; v.ia.reserve(n); for (uint64_t i = 0; i < n; ++i) v.ia.push_back(read_int(r, et)); }
        else if (et == T_F32) { v.kind = GgufValue::Kind::FloatArray; for (uint64_t i = 0; i < n; ++i) v.fa.push_back(r.get<float>()); }
        else if (et == T_F64) { v.kind = GgufValue::Kind::FloatArray; for (uint64_t i = 0; i < n; ++i) v.fa.push_back(r.get<double>()); }
        else if (et == T_BOOL) { v.kind = GgufValue::Kind::BoolArray; for (uint64_t i = 0; i < n; ++i) v.ba.push_back(r.get<uint8_t>() != 0); }
        else if (et == T_STR) { v.kind = GgufValue::Kind::StringArray; v.sa.reserve(n); for (uint64_t i = 0; i < n; ++i) v.sa.push_back(r.str()); }
        else throw std::runtime_error("unsupported GGUF array element type " + std::to_string(et));
    } else {
        throw std::runtime_error("unsupported GGUF value type " + std::to_string(t));
    }
    return v;
}

}  // namespace

uint64_t ggml_bytes(int type, int64_t n) {
    if (type < 0 || type >= GGML_TYPE_COUNT) throw std::runtime_error("unknown ggml type " + std::to_string(type));
    const int64_t bs = ggml_blck_size((ggml_type)type);
    if (bs <= 0 || ggml_type_size((ggml_type)type) == 0) throw std::runtime_error("ggml type " + std::to_string(type) + " is not usable");
    if (n % bs) throw std::runtime_error("a row of " + std::to_string(n) + " elements is not a whole number of " +
                                         ggml_type_name((ggml_type)type) + " blocks");
    return ggml_row_size((ggml_type)type, n);
}

GgufFile::GgufFile(const std::string& path) : path_(path) {
    Reader r(path);
    file_size_ = r.file_size();
    char magic[4];
    r.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) throw std::runtime_error(path + " is not a GGUF file");
    uint32_t version = r.get<uint32_t>();
    if (version < 2 || version > 3) throw std::runtime_error("GGUF version " + std::to_string(version) + " is not supported");
    uint64_t n_tensors = r.get<uint64_t>();
    uint64_t n_kv = r.get<uint64_t>();
    if (n_tensors > (1u << 20) || n_kv > (1u << 20)) throw std::runtime_error("implausible GGUF header counts");
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.str();
        uint32_t t = r.get<uint32_t>();
        kv_[key] = read_value(r, t);
    }
    std::vector<uint64_t> rel;
    tensors_.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = r.str();
        uint32_t nd = r.get<uint32_t>();
        if (nd == 0 || nd > 4) throw std::runtime_error("tensor " + t.name + " has " + std::to_string(nd) + " dimensions");
        for (uint32_t d = 0; d < nd; ++d) t.ne.push_back((int64_t)r.get<uint64_t>());
        t.type = (int)r.get<uint32_t>();
        rel.push_back(r.get<uint64_t>());
        t.nbytes = ggml_bytes(t.type, t.ne[0]) * (uint64_t)(t.nelements() / t.ne[0]);
        by_name_[t.name] = tensors_.size();
        tensors_.push_back(std::move(t));
    }
    const uint64_t align = (uint64_t)get_int("general.alignment", 32);
    if (align == 0 || (align & (align - 1))) throw std::runtime_error("bad general.alignment");
    data_offset_ = (r.consumed() + align - 1) / align * align;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        tensors_[i].offset = data_offset_ + rel[i];
        if (tensors_[i].offset + tensors_[i].nbytes > file_size_)
            throw std::runtime_error("tensor " + tensors_[i].name + " runs past the end of the file (an incomplete download?)");
    }
}

const GgufValue* GgufFile::get(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}

int64_t GgufFile::get_int(const std::string& key, int64_t def) const {
    const GgufValue* v = get(key);
    if (!v) return def;
    if (v->kind == GgufValue::Kind::Int) return v->i;
    if (v->kind == GgufValue::Kind::Float) return (int64_t)v->f;
    if (v->kind == GgufValue::Kind::Bool) return v->b;
    throw std::runtime_error(key + " is not a number");
}

double GgufFile::get_float(const std::string& key, double def) const {
    const GgufValue* v = get(key);
    if (!v) return def;
    if (v->kind == GgufValue::Kind::Float) return v->f;
    if (v->kind == GgufValue::Kind::Int) return (double)v->i;
    throw std::runtime_error(key + " is not a number");
}

bool GgufFile::get_bool(const std::string& key, bool def) const {
    const GgufValue* v = get(key);
    if (!v) return def;
    if (v->kind == GgufValue::Kind::Bool) return v->b;
    if (v->kind == GgufValue::Kind::Int) return v->i != 0;
    throw std::runtime_error(key + " is not a boolean");
}

std::string GgufFile::get_string(const std::string& key, const std::string& def) const {
    const GgufValue* v = get(key);
    if (!v) return def;
    if (v->kind != GgufValue::Kind::String) throw std::runtime_error(key + " is not a string");
    return v->s;
}

const GgufTensor* GgufFile::tensor(const std::string& name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

}  // namespace strata::inkling
