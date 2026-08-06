#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q19ResultRow {
    int64_t revenue_raw = 0;
};

std::vector<Q19ResultRow> run_q19(const Database& db, const Q19Args& args);
void write_q19_csv(const std::string& filename, const std::vector<Q19ResultRow>& rows);