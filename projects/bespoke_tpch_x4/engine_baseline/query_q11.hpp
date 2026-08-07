#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q11ResultRow {
    int32_t ps_partkey = 0;
    int64_t value_raw = 0;
};

std::vector<Q11ResultRow> run_q11(const Database& db, const Q11Args& args);
void write_q11_csv(const std::string& filename, const std::vector<Q11ResultRow>& rows);