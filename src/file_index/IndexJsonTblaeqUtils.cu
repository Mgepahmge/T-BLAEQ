#include "IndexJsonTblaeqUtils.cuh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace tblaeq {
namespace file_index {

bool host_is_little_endian() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t*>(&x) == 1;
}

uint64_t bswap64(uint64_t x) {
    return ((x & 0x00000000000000FFULL) << 56) |
        ((x & 0x000000000000FF00ULL) << 40) |
        ((x & 0x0000000000FF0000ULL) << 24) |
        ((x & 0x00000000FF000000ULL) << 8) |
        ((x & 0x000000FF00000000ULL) >> 8) |
        ((x & 0x0000FF0000000000ULL) >> 24) |
        ((x & 0x00FF000000000000ULL) >> 40) |
        ((x & 0xFF00000000000000ULL) >> 56);
}

void append_u8(std::vector<uint8_t>& out, uint8_t v) { out.push_back(v); }

void append_u64_le(std::vector<uint8_t>& out, uint64_t v) {
    if (!host_is_little_endian()) v = bswap64(v);
    uint8_t b[8];
    std::memcpy(b, &v, 8);
    out.insert(out.end(), b, b + 8);
}

void append_double_le(std::vector<uint8_t>& out, double v) {
    static_assert(sizeof(double) == 8, "double must be 8 bytes");
    uint64_t u = 0;
    std::memcpy(&u, &v, 8);
    append_u64_le(out, u);
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string strip_utf8_bom(std::string s) {
    if (s.size() >= 3 &&
        static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB &&
        static_cast<unsigned char>(s[2]) == 0xBF) {
        return s.substr(3);
    }
    return s;
}

uint64_t parse_size_arg(const std::string& text) {
    const std::string s = to_lower_ascii(trim(text));
    static const std::regex re(R"(^(\d+)([a-z]+)?$)");
    std::smatch m;
    if (!std::regex_match(s, m, re)) throw std::runtime_error("invalid size: " + text);
    uint64_t value = std::stoull(m[1].str());
    const std::string unit = m[2].matched ? m[2].str() : "b";
    uint64_t mul = 1;
    if (unit == "b") mul = 1;
    else if (unit == "kb") mul = 1000ULL;
    else if (unit == "kib") mul = 1024ULL;
    else if (unit == "mb") mul = 1000ULL * 1000ULL;
    else if (unit == "mib") mul = 1024ULL * 1024ULL;
    else if (unit == "gb") mul = 1000ULL * 1000ULL * 1000ULL;
    else if (unit == "gib") mul = 1024ULL * 1024ULL * 1024ULL;
    else throw std::runtime_error("invalid size unit: " + text);
    if (value > std::numeric_limits<uint64_t>::max() / mul) throw std::runtime_error("size overflow: " + text);
    return value * mul;
}

uint8_t parse_seq(const std::string& seq_name) {
    static const std::regex re(R"(seq(\d+))");
    std::smatch m;
    if (!std::regex_match(seq_name, m, re)) throw std::runtime_error("invalid sequence name: " + seq_name);
    unsigned long v = std::stoul(m[1].str());
    if (v > 255UL) throw std::runtime_error("seq does not fit uint8: " + seq_name);
    return static_cast<uint8_t>(v);
}

uint8_t parse_seq_arg(const std::string& s0) {
    std::string s = trim(s0);
    if (s.rfind("seq", 0) == 0) s = s.substr(3);
    unsigned long v = std::stoul(s);
    if (v > 255UL) throw std::runtime_error("seq does not fit uint8: " + s0);
    return static_cast<uint8_t>(v);
}

std::vector<uint8_t> parse_seq_list(const std::vector<std::string>& inputs) {
    std::vector<uint8_t> seqs;
    for (const auto& raw : inputs) {
        std::string s = trim(raw);
        size_t start = 0;
        while (start <= s.size()) {
            size_t comma = s.find(',', start);
            std::string part = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            part = trim(part);
            if (!part.empty()) seqs.push_back(parse_seq_arg(part));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    std::sort(seqs.begin(), seqs.end());
    seqs.erase(std::unique(seqs.begin(), seqs.end()), seqs.end());
    return seqs;
}

double parse_timestamp_from_filename(const std::filesystem::path& p) {
    const std::string stem = p.stem().string();
    size_t pos = 0;
    double v = std::stod(stem, &pos);
    if (pos != stem.size()) throw std::runtime_error("filename stem is not a pure timestamp: " + p.string());
    return v;
}

int64_t timestamp_to_us(double timestamp) {
    const long double scaled = static_cast<long double>(timestamp) * 1000000.0L;
    if (scaled < static_cast<long double>(std::numeric_limits<int64_t>::min()) ||
        scaled > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("timestamp microseconds overflow int64");
    }
    return static_cast<int64_t>(std::llround(scaled));
}

std::array<double, 3> split_timestamp_3d(double timestamp) {
    const int64_t signed_us = timestamp_to_us(timestamp);
    if (signed_us < 0) throw std::runtime_error("negative timestamp is not supported");
    const uint64_t ts_us = static_cast<uint64_t>(signed_us);
    constexpr uint64_t mask = (1ULL << 21) - 1ULL;
    return {
        static_cast<double>(((ts_us >> 42) & mask) + 1ULL),
        static_cast<double>(((ts_us >> 21) & mask) + 1ULL),
        static_cast<double>((ts_us & mask) + 1ULL)
    };
}

size_t binary_index_record_size(size_t n, size_t file_name_bytes) {
    return 4 * 8 + 1 + 1 + n * 8 + 1 + file_name_bytes + 8 + (n + 1) * 8;
}

std::string binary_struct_format_string(size_t n, size_t file_name_bytes) {
    std::ostringstream oss;
    oss << "<4dBB" << n << "dB" << file_name_bytes << "sQ" << (n + 1) << "Q";
    return oss.str();
}

std::string fixed_file_name_to_string(const char file_name[MAX_FILE_NAME_BYTES]) {
    size_t n = 0;
    while (n < MAX_FILE_NAME_BYTES && file_name[n] != '\0') ++n;
    return std::string(file_name, n);
}

void set_fixed_file_name(IndexRecordFixed& rec, const std::string& name, size_t file_name_bytes) {
    if (file_name_bytes > MAX_FILE_NAME_BYTES) throw std::runtime_error("file_name_bytes exceeds runtime maximum");
    if (name.size() > file_name_bytes) throw std::runtime_error("file_name too long for fixed field: " + name);
    std::memset(rec.file_name, 0, sizeof(rec.file_name));
    std::memcpy(rec.file_name, name.data(), name.size());
}

std::string extension_lower(const std::filesystem::path& p) {
    return to_lower_ascii(p.extension().string());
}

} // namespace file_index
} // namespace tblaeq
