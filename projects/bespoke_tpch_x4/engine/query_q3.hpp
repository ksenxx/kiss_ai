#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q3ResultRow {
    int32_t orderkey = 0;
    int64_t revenue_raw = 0;
    int16_t orderdate_offset = 0;
    int32_t shippriority = 0;
};

std::vector<Q3ResultRow> run_q3(const Database& db, const Q3Args& args);
void write_q3_csv(const Database& db,
                  const std::string& filename,
                  const std::vector<Q3ResultRow>& rows);