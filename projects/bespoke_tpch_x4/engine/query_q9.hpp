#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q9ResultRow {
    std::string nation;
    int32_t o_year = 0;
    int64_t sum_profit_raw = 0;
};

std::vector<Q9ResultRow> run_q9(const Database& db, const Q9Args& args);
void write_q9_csv(const std::string& filename, const std::vector<Q9ResultRow>& rows);