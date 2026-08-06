#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q21ResultRow {
    std::string s_name;
    int64_t numwait = 0;
};

std::vector<Q21ResultRow> run_q21(const Database& db, const Q21Args& args);
void write_q21_csv(const std::string& filename, const std::vector<Q21ResultRow>& rows);