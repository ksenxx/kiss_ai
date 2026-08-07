#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <vector>

struct Q14ResultRow {
    double promo_revenue = 0.0;
};

std::vector<Q14ResultRow> run_q14(const Database& db, const Q14Args& args);
void write_q14_csv(const std::string& filename, const std::vector<Q14ResultRow>& rows);