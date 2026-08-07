#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q15ResultRow {
    int32_t s_suppkey = 0;
    std::string s_name;
    std::string s_address;
    std::string s_phone;
    int64_t total_revenue_raw = 0;
};

std::vector<Q15ResultRow> run_q15(const Database& db, const Q15Args& args);
void write_q15_csv(const std::string& filename, const std::vector<Q15ResultRow>& rows);