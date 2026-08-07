#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q20ResultRow {
    std::string s_name;
    std::string s_address;
};

std::vector<Q20ResultRow> run_q20(const Database& db, const Q20Args& args);
void write_q20_csv(const std::string& filename, const std::vector<Q20ResultRow>& rows);