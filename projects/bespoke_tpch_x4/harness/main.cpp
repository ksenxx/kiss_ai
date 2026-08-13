// Standalone harness for the bespoke TPC-H engine.
//
// Usage: ./db <parquet_dir_with_trailing_slash>
//
// Reads query request lines ("<query_id> KEY=VAL ...") from stdin until an
// empty line or EOF, executes them, writes result<N>.csv files into the
// current working directory, and prints per-query timings to stdout in the
// same format as the paper's fasttest host:
//   "<N> | Execution ms: <elapsed>"
// It also prints "Ingest ms: <elapsed>" for the build phase on stderr.

#include <chrono>
#include <iostream>
#include <string>

#include "builder_impl.hpp"
#include "loader_impl.hpp"
#include "query_impl.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <parquet_dir/>\n";
        return 1;
    }
    std::string parquet_path = argv[1];
    if (!parquet_path.empty() && parquet_path.back() != '/') {
        parquet_path.push_back('/');
    }

    ParquetTables* tables = load(parquet_path);

    const auto t0 = std::chrono::steady_clock::now();
    Database* db = build(tables);
    const auto t1 = std::chrono::steady_clock::now();
    std::cerr << "Ingest ms: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << "\n";

    query(db);
    return 0;
}
