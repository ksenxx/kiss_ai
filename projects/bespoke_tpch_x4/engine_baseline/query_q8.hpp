#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <vector>

struct Q8ResultRow {
    int32_t o_year = 0;
    double mkt_share = 0.0;
};

std::vector<Q8ResultRow> run_q8(const Database& db, const Q8Args& args);
void write_q8_csv(const std::string& filename, const std::vector<Q8ResultRow>& rows);