#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

struct Q10ResultRow {
    int32_t custkey = 0;
    std::string_view name;
    int64_t revenue_raw = 0;
    int32_t acctbal_raw = 0;
    std::string_view nation_name;
    std::string_view address;
    std::string_view phone;
    std::string_view comment;
};

std::vector<Q10ResultRow> run_q10(const Database& db, const Q10Args& args);
void write_q10_csv(const std::string& filename, const std::vector<Q10ResultRow>& rows);