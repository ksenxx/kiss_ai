#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q6ResultRow {
    int64_t revenue_raw = 0;
};

std::vector<Q6ResultRow> run_q6(const Database& db, const Q6Args& args);
void write_q6_csv(const std::string& filename, const std::vector<Q6ResultRow>& rows);