#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q17ResultRow {
    int64_t sum_extendedprice_raw = 0;
};

std::vector<Q17ResultRow> run_q17(const Database& db, const Q17Args& args);
void write_q17_csv(const std::string& filename, const std::vector<Q17ResultRow>& rows);