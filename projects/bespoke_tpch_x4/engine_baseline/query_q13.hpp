#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <vector>

struct Q13ResultRow {
    int64_t c_count = 0;
    int64_t custdist = 0;
};

std::vector<Q13ResultRow> run_q13(const Database& db, const Q13Args& args);
void write_q13_csv(const std::string& filename, const std::vector<Q13ResultRow>& rows);