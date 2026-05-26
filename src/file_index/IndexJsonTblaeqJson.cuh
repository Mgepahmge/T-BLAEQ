#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "IndexJsonTblaeqTypes.cuh"

namespace tblaeq {
namespace file_index {

Meta load_meta(const std::filesystem::path& output_dir);
IndexRecordFixed parse_index_line(const std::string& line, size_t line_no, const Meta& meta);

std::vector<IndexRecordFixed> load_index_records(const std::filesystem::path& output_dir, const Meta& meta);
std::vector<IndexRecordFixed> load_candidate_records(const std::filesystem::path& output_dir,
                                                      const Meta& meta,
                                                      const std::vector<size_t>& candidates);

} // namespace file_index
} // namespace tblaeq
