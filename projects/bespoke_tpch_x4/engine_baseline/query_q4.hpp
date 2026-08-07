#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q4ResultRow {
    std::string orderpriority;
    int64_t order_count = 0;
};

std::vector<Q4ResultRow> run_q4(const Database& db, const Q4Args& args);
void write_q4_csv(const std::string& filename, const std::vector<Q4ResultRow>& rows);