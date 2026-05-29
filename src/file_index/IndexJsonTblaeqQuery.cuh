#pragma once

#include "IndexJsonTblaeqTypes.cuh"

class QueryHandler;

namespace tblaeq {
namespace file_index {

struct QueryPayloadMatch {
    size_t rank = 0;
    size_t item_no = 0;
    uint8_t seq = 0;
    uint8_t dtype = 0;
    uint8_t slot = 0;
    double timestamp = 0.0;
    int64_t diff_us = 0;
    uint64_t payload_offset = 0;
    uint64_t payload_size = 0;
    std::string data_file;
};

struct QueryServiceResult {
    size_t candidate_count = 0;
    size_t record_count = 0;
    size_t slot_count = 0;
    std::vector<QueryPayloadMatch> matches;
};

int run_query(QueryArgs args);
QueryServiceResult run_query_service(const QueryArgs& args, QueryHandler& handler);

} // namespace file_index
} // namespace tblaeq
