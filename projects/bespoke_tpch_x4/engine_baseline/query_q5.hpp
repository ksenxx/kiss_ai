#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q5ResultRow {
    std::string n_name;
    int64_t revenue_raw = 0;
};

std::vector<Q5ResultRow> run_q5(const Database& db, const Q5Args& args);
void write_q5_csv(const std::string& filename, const std::vector<Q5ResultRow>& rows);