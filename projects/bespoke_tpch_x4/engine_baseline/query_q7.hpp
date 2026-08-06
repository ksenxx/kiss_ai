#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q7ResultRow {
    std::string supp_nation;
    std::string cust_nation;
    int32_t l_year = 0;
    int64_t revenue_raw = 0;
};

std::vector<Q7ResultRow> run_q7(const Database& db, const Q7Args& args);
void write_q7_csv(const std::string& filename, const std::vector<Q7ResultRow>& rows);