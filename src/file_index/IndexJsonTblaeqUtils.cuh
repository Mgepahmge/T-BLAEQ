#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "IndexJsonTblaeqTypes.cuh"

namespace tblaeq {
namespace file_index {

bool host_is_little_endian();
uint64_t bswap64(uint64_t x);

void append_u8(std::vector<uint8_t>& out, uint8_t v);
void append_u64_le(std::vector<uint8_t>& out, uint64_t v);
void append_double_le(std::vector<uint8_t>& out, double v);

std::string trim(const std::string& s);
std::string to_lower_ascii(std::string s);
std::string strip_utf8_bom(std::string s);
uint64_t parse_size_arg(const std::string& text);

uint8_t parse_seq(const std::string& seq_name);
uint8_t parse_seq_arg(const std::string& s0);
std::vector<uint8_t> parse_seq_list(const std::vector<std::string>& inputs);
double parse_timestamp_from_filename(const std::filesystem::path& p);

int64_t timestamp_to_us(double timestamp);
std::array<double, 3> split_timestamp_3d(double timestamp);

size_t binary_index_record_size(size_t n, size_t file_name_bytes);
std::string binary_struct_format_string(size_t n, size_t file_name_bytes);
std::string fixed_file_name_to_string(const char file_name[MAX_FILE_NAME_BYTES]);
void set_fixed_file_name(IndexRecordFixed& rec, const std::string& name, size_t file_name_bytes);

std::string extension_lower(const std::filesystem::path& p);

} // namespace file_index
} // namespace tblaeq
