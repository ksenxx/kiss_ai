#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q12ResultRow {
    std::string shipmode;
    int64_t high_line_count = 0;
    int64_t low_line_count = 0;
};

std::vector<Q12ResultRow> run_q12(const Database& db, const Q12Args& args);
void write_q12_csv(const std::string& filename, const std::vector<Q12ResultRow>& rows);