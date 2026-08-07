#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q16ResultRow {
    std::string p_brand;
    std::string p_type;
    int32_t p_size = 0;
    int64_t supplier_cnt = 0;
};

std::vector<Q16ResultRow> run_q16(const Database& db, const Q16Args& args);
void write_q16_csv(const std::string& filename, const std::vector<Q16ResultRow>& rows);