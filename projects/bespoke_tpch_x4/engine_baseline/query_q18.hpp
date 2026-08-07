#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q18ResultRow {
    std::string c_name;
    int32_t c_custkey = 0;
    int32_t o_orderkey = 0;
    int16_t o_orderdate_offset = 0;
    int32_t o_totalprice_raw = 0;
    int64_t sum_qty_raw = 0;
};

std::vector<Q18ResultRow> run_q18(const Database& db, const Q18Args& args);
void write_q18_csv(const Database& db,
                   const std::string& filename,
                   const std::vector<Q18ResultRow>& rows);