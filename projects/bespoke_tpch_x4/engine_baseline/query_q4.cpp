#include "query_q4.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "trace_utils.hpp"

namespace {
int64_t q4_days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
}

bool q4_is_leap(int year) {
    if (year % 400 == 0) {
        return true;
    }
    if (year % 100 == 0) {
        return false;
    }
    return year % 4 == 0;
}

int q4_days_in_month(int year, int month) {
    static const int kDays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month == 2 && q4_is_leap(year)) {
        return 29;
    }
    return kDays[month - 1];
}

int16_t q4_parse_date_offset(const std::string& value, int32_t base_days) {
    const int year = std::stoi(value.substr(0, 4));
    const int month = std::stoi(value.substr(5, 2));
    const int day = std::stoi(value.substr(8, 2));
    const int32_t days = static_cast<int32_t>(q4_days_from_civil(year, month, day));
    return static_cast<int16_t>(days - base_days);
}

int16_t q4_parse_date_offset_add_months(const std::string& value,
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
    const int dim = q4_days_in_month(year, month);
    if (day > dim) {
        day = dim;
    }
    const int32_t days = static_cast<int32_t>(q4_days_from_civil(year, month, day));
    return static_cast<int16_t>(days - base_days);
}

inline bool q4_is_late(uint32_t packed) {
    return static_cast<uint16_t>(packed) < static_cast<uint16_t>(packed >> 16);
}

std::string escape_csv_field_q4(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            out.push_back('\\');
            out.push_back('"');
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('"');
    return out;
}
}  // namespace

#ifdef TRACE
namespace q4_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t orders_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t orders_rows_scanned = 0;
    uint64_t orders_rows_emitted = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
    uint64_t join_build_rows_in = 0;
    uint64_t join_probe_rows_in = 0;
    uint64_t join_rows_emitted = 0;
    uint64_t agg_rows_in = 0;
    uint64_t groups_created = 0;
    uint64_t agg_rows_emitted = 0;
    uint64_t sort_rows_in = 0;
    uint64_t sort_rows_out = 0;
    uint64_t query_output_rows = 0;
};

inline TraceData& data() {
    static TraceData trace;
    return trace;
}

inline void reset() { data() = TraceData{}; }

inline void record_timing(const char* name, uint64_t ns) {
    auto& trace = data();
    if (std::strcmp(name, "q4_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q4_orders_scan") == 0) {
        trace.orders_scan_ns += ns;
    } else if (std::strcmp(name, "q4_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q4_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q4_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q4_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q4_orders_scan " << trace.orders_scan_ns << "\n";
    std::cout << "PROFILE q4_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q4_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q4_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q4_orders_scan_rows_scanned " << trace.orders_rows_scanned << "\n";
    std::cout << "COUNT q4_orders_scan_rows_emitted " << trace.orders_rows_emitted << "\n";
    std::cout << "COUNT q4_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned << "\n";
    std::cout << "COUNT q4_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted << "\n";
    std::cout << "COUNT q4_order_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q4_order_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q4_order_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q4_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q4_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q4_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q4_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q4_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q4_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q4_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q4_trace::record_timing)
#define TRACE_SET(field, value) (q4_trace::data().field = (value))
#define TRACE_ADD(field, value) (q4_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q4ResultRow> run_q4(const Database& db, const Q4Args& args) {
#ifdef TRACE
    q4_trace::reset();
#endif
    std::vector<Q4ResultRow> results;
    {
        PROFILE_SCOPE("q4_total");
    const int16_t start_offset = q4_parse_date_offset(args.DATE, db.base_date_days);
    const int16_t end_offset =
        q4_parse_date_offset_add_months(args.DATE, db.base_date_days, 3);

    const auto& orders = db.orders;
    const auto& lineitem = db.lineitem;

    std::vector<int64_t> counts(orders.orderpriority.dictionary.size(), 0);
    int64_t* __restrict counts_data = counts.data();
    const uint32_t counts_size = static_cast<uint32_t>(counts.size());

    const int16_t* orderdates = orders.orderdate.data();
    const size_t orders_row_count = orders.row_count;
    const int16_t* date_start =
        std::lower_bound(orderdates, orderdates + orders_row_count, start_offset);
    const int16_t* date_end =
        std::lower_bound(orderdates, orderdates + orders_row_count, end_offset);
    const uint32_t start_idx = static_cast<uint32_t>(date_start - orderdates);
    const uint32_t end_idx = static_cast<uint32_t>(date_end - orderdates);

    const int32_t* __restrict orderkeys = orders.orderkey.data();
    const uint16_t* __restrict priority_codes = orders.orderpriority.codes.data();
    const uint32_t* __restrict commit_receipt = lineitem.commit_receipt.data();
    const bool orderkey_sorted = lineitem.orderkey_sorted;
    const auto* __restrict order_ranges = orders.lineitem_ranges.data();

#ifdef TRACE
    uint64_t orders_emitted = 0;
    uint64_t lineitems_scanned = 0;
    uint64_t lineitems_emitted = 0;
    uint64_t orders_with_late = 0;
#endif
    {
        PROFILE_SCOPE("q4_orders_scan");
        TRACE_SET(orders_rows_scanned, static_cast<uint64_t>(end_idx - start_idx));
        if (orderkey_sorted) {
            const auto* __restrict range_ptr = order_ranges + start_idx;
            const auto* __restrict range_end = order_ranges + end_idx;
            const uint16_t* __restrict priority_ptr = priority_codes + start_idx;
            for (; range_ptr != range_end; ++range_ptr, ++priority_ptr) {
#ifdef TRACE
                orders_emitted += 1;
#endif
                const auto range = *range_ptr;
                if (range.end == 0) {
                    continue;
                }

                bool has_late_lineitem = false;
#ifdef TRACE
                const uint64_t lineitem_start_ns = trace_utils::get_time_ns();
#endif
                const uint32_t* commit_ptr = commit_receipt + range.start;
                const uint32_t* commit_end = commit_receipt + range.end;
                for (; commit_ptr < commit_end; ++commit_ptr) {
#ifdef TRACE
                    lineitems_scanned += 1;
#endif
                    if (q4_is_late(*commit_ptr)) {
#ifdef TRACE
                        lineitems_emitted += 1;
#endif
                        has_late_lineitem = true;
                        break;
                    }
                }
#ifdef TRACE
                q4_trace::record_timing("q4_lineitem_scan",
                                        trace_utils::get_time_ns() - lineitem_start_ns);
#endif
                if (!has_late_lineitem) {
                    continue;
                }
#ifdef TRACE
                orders_with_late += 1;
#endif

                const uint32_t code = *priority_ptr;
                counts_data[code] += 1;
            }
        } else {
            const int32_t* __restrict lineitem_orderkey = lineitem.orderkey.data();
            for (uint32_t o_idx = start_idx; o_idx < end_idx; ++o_idx) {
#ifdef TRACE
                orders_emitted += 1;
#endif
                const auto range = order_ranges[o_idx];
                if (range.end == 0) {
                    continue;
                }

                bool has_late_lineitem = false;
#ifdef TRACE
                const uint64_t lineitem_start_ns = trace_utils::get_time_ns();
#endif
                const int32_t orderkey = orderkeys[o_idx];
                const uint32_t start = range.start;
                const uint32_t end = range.end;
                const uint32_t* commit_receipt_ptr = commit_receipt + start;
                const int32_t* orderkey_ptr = lineitem_orderkey + start;
                for (uint32_t li_idx = start; li_idx < end;
                     ++li_idx, ++commit_receipt_ptr, ++orderkey_ptr) {
#ifdef TRACE
                    lineitems_scanned += 1;
#endif
                    if (*orderkey_ptr != orderkey) {
                        continue;
                    }
                    if (q4_is_late(*commit_receipt_ptr)) {
#ifdef TRACE
                        lineitems_emitted += 1;
#endif
                        has_late_lineitem = true;
                        break;
                    }
                }
#ifdef TRACE
                q4_trace::record_timing("q4_lineitem_scan",
                                        trace_utils::get_time_ns() - lineitem_start_ns);
#endif
                if (!has_late_lineitem) {
                    continue;
                }
#ifdef TRACE
                orders_with_late += 1;
#endif

                const uint32_t code = priority_codes[o_idx];
                if (code < counts_size) {
                    counts_data[code] += 1;
                }
            }
        }
    }
    TRACE_SET(orders_rows_emitted, orders_emitted);
    TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
    TRACE_SET(lineitem_rows_emitted, lineitems_emitted);
    TRACE_SET(join_build_rows_in, orders_emitted);
    TRACE_SET(join_probe_rows_in, lineitems_scanned);
    TRACE_SET(join_rows_emitted, orders_with_late);
    TRACE_SET(agg_rows_in, orders_with_late);

    {
        PROFILE_SCOPE("q4_agg_finalize");
        results.reserve(counts.size());
        uint64_t groups_created = 0;
        for (uint32_t code = 0; code < counts.size(); ++code) {
            if (counts[code] == 0) {
                continue;
            }
            Q4ResultRow row;
            row.orderpriority = orders.orderpriority.dictionary[code];
            row.order_count = counts[code];
            results.push_back(std::move(row));
            groups_created += 1;
        }
        TRACE_SET(groups_created, groups_created);
        TRACE_SET(agg_rows_emitted, groups_created);
    }

    {
        PROFILE_SCOPE("q4_sort");
        TRACE_SET(sort_rows_in, results.size());
        std::sort(results.begin(), results.end(),
                  [](const Q4ResultRow& a, const Q4ResultRow& b) {
                      return a.orderpriority < b.orderpriority;
                  });
        TRACE_SET(sort_rows_out, results.size());
    }
    TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q4_trace::emit();
#endif

    return results;
}

void write_q4_csv(const std::string& filename, const std::vector<Q4ResultRow>& rows) {
    std::ofstream out(filename);
    out << "o_orderpriority,order_count\n";
    for (const auto& row : rows) {
        out << escape_csv_field_q4(row.orderpriority) << ',' << row.order_count << "\n";
    }
}