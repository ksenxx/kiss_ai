#include "query_q14.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q14 {
constexpr int32_t kPriceScale = 100;
constexpr int32_t kDiscountScale = 100;

int64_t days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
}

bool is_leap(int year) {
    if (year % 400 == 0) {
        return true;
    }
    if (year % 100 == 0) {
        return false;
    }
    return year % 4 == 0;
}

int days_in_month(int year, int month) {
    static const int kDays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && is_leap(year)) {
        return 29;
    }
    return kDays[month - 1];
}

int16_t parse_date_offset(const std::string& value, int32_t base_days) {
    const int year = std::stoi(value.substr(0, 4));
    const int month = std::stoi(value.substr(5, 2));
    const int day = std::stoi(value.substr(8, 2));
    const int32_t days = static_cast<int32_t>(days_from_civil(year, month, day));
    return static_cast<int16_t>(days - base_days);
}

int16_t parse_date_offset_add_months(const std::string& value,
                                     int32_t base_days,
                                     int months) {
    int year = std::stoi(value.substr(0, 4));
    int month = std::stoi(value.substr(5, 2));
    int day = std::stoi(value.substr(8, 2));
    int month_index = month - 1 + months;
    year += month_index / 12;
    month_index %= 12;
    if (month_index < 0) {
        month_index += 12;
        year -= 1;
    }
    month = month_index + 1;
    const int dim = days_in_month(year, month);
    if (day > dim) {
        day = dim;
    }
    const int32_t days = static_cast<int32_t>(days_from_civil(year, month, day));
    return static_cast<int16_t>(days - base_days);
}

std::string_view get_string_view(const StringColumn& column, size_t row) {
    const uint32_t start = column.offsets[row];
    const uint32_t end = column.offsets[row + 1];
    return std::string_view(column.data.data() + start, end - start);
}

int32_t max_value(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return -1;
    }
    int32_t max_val = values.front();
    for (int32_t value : values) {
        if (value > max_val) {
            max_val = value;
        }
    }
    return max_val;
}

bool starts_with_promo(std::string_view value) {
    constexpr std::string_view kPrefix = "PROMO";
    return value.size() >= kPrefix.size() &&
           value.compare(0, kPrefix.size(), kPrefix) == 0;
}
}  // namespace q14

#ifdef TRACE
namespace q14_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t part_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t part_rows_scanned = 0;
    uint64_t part_rows_emitted = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
    uint64_t join_build_rows_in = 0;
    uint64_t join_probe_rows_in = 0;
    uint64_t join_rows_emitted = 0;
    uint64_t agg_rows_in = 0;
    uint64_t groups_created = 0;
    uint64_t agg_rows_emitted = 0;
    uint64_t query_output_rows = 0;
};

inline TraceData& data() {
    static TraceData trace;
    return trace;
}

inline void reset() { data() = TraceData{}; }

inline void record_timing(const char* name, uint64_t ns) {
    auto& trace = data();
    if (std::strcmp(name, "q14_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q14_part_scan") == 0) {
        trace.part_scan_ns += ns;
    } else if (std::strcmp(name, "q14_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q14_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q14_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q14_part_scan " << trace.part_scan_ns << "\n";
    std::cout << "PROFILE q14_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q14_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "COUNT q14_part_scan_rows_scanned " << trace.part_rows_scanned << "\n";
    std::cout << "COUNT q14_part_scan_rows_emitted " << trace.part_rows_emitted << "\n";
    std::cout << "COUNT q14_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q14_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q14_part_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q14_part_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q14_part_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q14_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q14_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q14_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q14_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q14_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q14_trace::record_timing)
#define TRACE_SET(field, value) (q14_trace::data().field = (value))
#define TRACE_ADD(field, value) (q14_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q14ResultRow> run_q14(const Database& db, const Q14Args& args) {
#ifdef TRACE
    q14_trace::reset();
#endif
    std::vector<Q14ResultRow> results;
    {
        PROFILE_SCOPE("q14_total");
    const auto& lineitem = db.lineitem;
    const int16_t* __restrict shipdate = lineitem.shipdate.data();
    const int32_t* __restrict partkey = lineitem.partkey.data();
    const int32_t* __restrict discounted_price = lineitem.discounted_price.data();
        const auto& part = db.part;

        const int16_t start_offset = q14::parse_date_offset(args.DATE, db.base_date_days);
        const int16_t end_offset =
            q14::parse_date_offset_add_months(args.DATE, db.base_date_days, 1);

        const int32_t max_partkey = q14::max_value(part.partkey);
        if (max_partkey < 0) {
#ifdef TRACE
            TRACE_SET(query_output_rows, 1);
            q14_trace::emit();
#endif
            return {Q14ResultRow{}};
        }

        std::vector<uint8_t> promo_type(part.type.dictionary.size(), 0);
        for (size_t idx = 0; idx < part.type.dictionary.size(); ++idx) {
            if (q14::starts_with_promo(part.type.dictionary[idx])) {
                promo_type[idx] = 1;
            }
        }
        const auto* part_type_codes = part.type.codes.data();

        std::vector<int8_t> partkey_promo(static_cast<size_t>(max_partkey) + 1, -1);
        uint64_t part_emitted = 0;
        {
            PROFILE_SCOPE("q14_part_scan");
            TRACE_SET(part_rows_scanned, part.row_count);
            for (uint32_t i = 0; i < part.row_count; ++i) {
                const int32_t key = part.partkey[i];
                if (key >= 0) {
                    partkey_promo[static_cast<size_t>(key)] =
                        static_cast<int8_t>(promo_type[part_type_codes[i]]);
                    part_emitted += 1;
                }
            }
        }
        TRACE_SET(part_rows_emitted, part_emitted);

        int64_t promo_sum = 0;
        int64_t total_sum = 0;
        uint64_t lineitems_emitted = 0;
        uint64_t lineitems_for_join = 0;
        uint64_t lineitems_scanned = 0;

        {
            PROFILE_SCOPE("q14_lineitem_scan");
            for (const auto& shard : lineitem.shards) {
                if (shard.max_shipdate < start_offset || shard.min_shipdate >= end_offset) {
                    continue;
                }
                const bool shard_in_range =
                    (start_offset <= shard.min_shipdate && shard.max_shipdate < end_offset);
                if (shard.contiguous) {
                    const uint32_t start = shard.start;
                    const uint32_t end = shard.end;
                    if (shard_in_range) {
                        lineitems_scanned += static_cast<uint64_t>(end - start);
                        lineitems_for_join += static_cast<uint64_t>(end - start);
                        for (uint32_t li_idx = start; li_idx < end; ++li_idx) {
                            const int32_t key = partkey[li_idx];
                            if (key < 0 || static_cast<size_t>(key) >= partkey_promo.size()) {
                                continue;
                            }
                            const int8_t promo_flag =
                                partkey_promo[static_cast<size_t>(key)];
                            if (promo_flag < 0) {
                                continue;
                            }
                            const int32_t disc_price = discounted_price[li_idx];
                            total_sum += disc_price;
                            if (promo_flag) {
                                promo_sum += disc_price;
                            }
                            lineitems_emitted += 1;
                        }
                    } else {
                        for (uint32_t li_idx = start; li_idx < end; ++li_idx) {
                            lineitems_scanned += 1;
                            const int16_t ship = shipdate[li_idx];
                            if (ship < start_offset || ship >= end_offset) {
                                continue;
                            }
                            lineitems_for_join += 1;
                            const int32_t key = partkey[li_idx];
                            if (key < 0 || static_cast<size_t>(key) >= partkey_promo.size()) {
                                continue;
                            }
                            const int8_t promo_flag =
                                partkey_promo[static_cast<size_t>(key)];
                            if (promo_flag < 0) {
                                continue;
                            }
                            const int32_t disc_price = discounted_price[li_idx];
                            total_sum += disc_price;
                            if (promo_flag) {
                                promo_sum += disc_price;
                            }
                            lineitems_emitted += 1;
                        }
                    }
                } else {
                    if (shard_in_range) {
                        lineitems_scanned +=
                            static_cast<uint64_t>(shard.row_indices.size());
                        lineitems_for_join +=
                            static_cast<uint64_t>(shard.row_indices.size());
                        for (uint32_t li_idx : shard.row_indices) {
                            const int32_t key = partkey[li_idx];
                            if (key < 0 || static_cast<size_t>(key) >= partkey_promo.size()) {
                                continue;
                            }
                            const int8_t promo_flag =
                                partkey_promo[static_cast<size_t>(key)];
                            if (promo_flag < 0) {
                                continue;
                            }
                            const int32_t disc_price = discounted_price[li_idx];
                            total_sum += disc_price;
                            if (promo_flag) {
                                promo_sum += disc_price;
                            }
                            lineitems_emitted += 1;
                        }
                    } else {
                        for (uint32_t li_idx : shard.row_indices) {
                            lineitems_scanned += 1;
                            const int16_t ship = shipdate[li_idx];
                            if (ship < start_offset || ship >= end_offset) {
                                continue;
                            }
                            lineitems_for_join += 1;
                            const int32_t key = partkey[li_idx];
                            if (key < 0 || static_cast<size_t>(key) >= partkey_promo.size()) {
                                continue;
                            }
                            const int8_t promo_flag =
                                partkey_promo[static_cast<size_t>(key)];
                            if (promo_flag < 0) {
                                continue;
                            }
                            const int32_t disc_price = discounted_price[li_idx];
                            total_sum += disc_price;
                            if (promo_flag) {
                                promo_sum += disc_price;
                            }
                            lineitems_emitted += 1;
                        }
                    }
                }
            }
        }
        TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
        TRACE_SET(lineitem_rows_emitted, lineitems_emitted);
        TRACE_SET(join_build_rows_in, part_emitted);
        TRACE_SET(join_probe_rows_in, lineitems_for_join);
        TRACE_SET(join_rows_emitted, lineitems_emitted);
        TRACE_SET(agg_rows_in, lineitems_emitted);

        {
            PROFILE_SCOPE("q14_agg_finalize");
            Q14ResultRow row;
            if (total_sum > 0) {
                row.promo_revenue = 100.0 * static_cast<double>(promo_sum) /
                                    static_cast<double>(total_sum);
            } else {
                row.promo_revenue = 0.0;
            }
            results.push_back(row);
            TRACE_SET(groups_created, 1);
            TRACE_SET(agg_rows_emitted, 1);
        }
        TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q14_trace::emit();
#endif
    return results;
}

void write_q14_csv(const std::string& filename, const std::vector<Q14ResultRow>& rows) {
    std::ofstream out(filename);
    out << "promo_revenue\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        out << row.promo_revenue << "\n";
    }
}