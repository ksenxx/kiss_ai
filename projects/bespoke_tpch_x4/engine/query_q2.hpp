#pragma once

#include "args_parser.hpp"
#include "builder_impl.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct Q2ResultRow {
    int32_t s_acctbal_raw = 0;
    std::string_view s_name;
    std::string_view n_name;
    int32_t p_partkey = 0;
    std::string_view p_mfgr;
    std::string_view s_address;
    std::string_view s_phone;
    std::string_view s_comment;
};

std::vector<Q2ResultRow> run_q2(const Database& db, const Q2Args& args);
void write_q2_csv(const std::string& filename, const std::vector<Q2ResultRow>& rows);