#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <string>
#include <vector>

struct Q1ResultRow {
    std::string returnflag;
    std::string linestatus;
    double sum_qty = 0.0;
    double sum_base_price = 0.0;
    double sum_disc_price = 0.0;
    double sum_charge = 0.0;
    double avg_qty = 0.0;
    double avg_price = 0.0;
    double avg_disc = 0.0;
    int64_t count_order = 0;
};

std::vector<Q1ResultRow> run_q1(const Database& db, const Q1Args& args);
void write_q1_csv(const std::string& filename, const std::vector<Q1ResultRow>& rows);