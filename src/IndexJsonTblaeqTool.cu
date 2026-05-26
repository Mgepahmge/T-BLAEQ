#include "src/file_index/IndexJsonTblaeqBuild.cuh"
#include "src/file_index/IndexJsonTblaeqQuery.cuh"
#include "src/file_index/IndexJsonTblaeqUtils.cuh"

#include "src/CLI11.hpp"
#include "src/core/IndexBuilder.cuh"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace tblaeq::file_index;

    BuildArgs build_args;
    QueryArgs query_args;
    build_args.height = IndexBuilder::kDefaultHeight;

    CLI::App app{"Build JSONL index + T-BLAEQ index, then query both."};
    auto* build = app.add_subcommand("build", "Build JSONL index + T-BLAEQ index");
    auto* query = app.add_subcommand("query", "Query both indexes (T-BLAEQ coarse + JSONL refine)");
    app.require_subcommand(1);

    build->add_option("root", build_args.root, "input dataset root directory")->required()->check(
        CLI::ExistingDirectory);
    build->add_option("csv", build_args.csv_path, "coordinate CSV path")->required()->check(CLI::ExistingFile);
    build->add_option("index_root", build_args.index_root, "index root directory")->required();
    build->add_option("--index-dir", build_args.index_dir, "index file directory name")->default_val(kDefaultIndexDir);
    build->add_option("--tblaeq-dir", build_args.tblaeq_dir, "T-BLAEQ index directory name")->default_val(
        kDefaultTblaeqDir);
    build->add_option("--index-file", build_args.index_file, "main JSONL index file name")->default_val("index.jsonl");
    build->add_option("--max-data-file-size", build_args.max_data_file_size_text, "max data file size, e.g. 1GiB")->
           default_val("1GiB");
    build->add_flag("--verify", build_args.verify, "verify binary index copies after build");
    build->add_flag("--force-cpu", build_args.force_cpu, "force CPU index building");
    build->add_option("--height", build_args.height, "hierarchy height (levels), must be >= 2")->default_val(
        IndexBuilder::kDefaultHeight);
    build->add_option("--ratios", build_args.ratios_text, "comma-separated coarsening ratios (height-1 entries)")
         ->default_val(build_args.ratios_text);

    std::string seq_filter;
    query->add_option("index_root", query_args.index_root, "index root directory")->required()->check(
        CLI::ExistingDirectory);
    query->add_option("--index-dir", query_args.index_dir, "index file directory name")->default_val(kDefaultIndexDir);
    query->add_option("--tblaeq-dir", query_args.tblaeq_dir, "T-BLAEQ index directory name")->default_val(
        kDefaultTblaeqDir);
    query->add_option("--mode", query_args.mode, "knn or range")->default_val("knn")->check(CLI::IsMember({
        "knn", "range"
    }));
    auto* opt_timestamp = query->add_option("--timestamp", query_args.timestamp, "query timestamp (knn mode)");
    auto* opt_start = query->add_option("--start", query_args.start, "range start timestamp");
    auto* opt_end = query->add_option("--end", query_args.end, "range end timestamp");
    auto* opt_seq = query->add_option("--seq", seq_filter, "sequence filter, accepts 1 or seq0001");
    query->add_option("--knn-k", query_args.knn_k, "K for knn mode")->default_val(10);
    query->add_option("--range-limit", query_args.range_limit, "max returned matches in range mode (0 means all)")->
           default_val(0);
    query->add_flag("--include-padding", query_args.include_padding, "include zero-size padded slots");
    auto* opt_dump = query->add_option("--dump-dir", query_args.dump_dir,
                                       "optional directory to dump matched payload bytes");

    CLI11_PARSE(app, argc, argv);

    try {
        if (*build) {
            build_args.max_data_file_size = parse_size_arg(build_args.max_data_file_size_text);
            return run_build(build_args);
        }
        if (*query) {
            query_args.has_timestamp = opt_timestamp->count() > 0;
            query_args.has_start = opt_start->count() > 0;
            query_args.has_end = opt_end->count() > 0;
            query_args.has_seq = opt_seq->count() > 0;
            query_args.has_dump_dir = opt_dump->count() > 0;
            if (query_args.has_seq) query_args.seq = parse_seq_arg(seq_filter);
            return run_query(query_args);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
