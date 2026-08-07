#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct Q22ResultRow {
    std::string cntrycode;
    int64_t numcust = 0;
    int64_t totacctbal_raw = 0;
};

std::vector<Q22ResultRow> run_q22(const Database& db, const Q22Args& args);
void write_q22_csv(const std::string& filename, const std::vector<Q22ResultRow>& rows);