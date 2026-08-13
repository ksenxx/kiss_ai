#include "query_q12.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "trace_utils.hpp"

namespace q12 {
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

std::string escape_csv_field(const std::string& value) {
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
}  // namespace q12

#ifdef TRACE
namespace q12_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
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
    if (std::strcmp(name, "q12_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q12_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q12_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q12_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q12_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q12_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q12_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q12_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q12_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q12_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q12_lineitem_orders_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q12_lineitem_orders_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q12_lineitem_orders_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q12_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q12_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q12_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q12_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q12_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q12_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q12_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q12_trace::record_timing)
#define TRACE_SET(field, value) (q12_trace::data().field = (value))
#define TRACE_ADD(field, value) (q12_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q12ResultRow> run_q12(const Database& db, const Q12Args& args) {
#ifdef TRACE
    q12_trace::reset();
#endif
    std::vector<Q12ResultRow> results;
    {
        PROFILE_SCOPE("q12_total");
    const auto& lineitem = db.lineitem;
    const auto& orders = db.orders;

    const int16_t start_offset = q12::parse_date_offset(args.DATE, db.base_date_days);
    const int16_t end_offset =
        q12::parse_date_offset_add_months(args.DATE, db.base_date_days, 12);
    const int32_t start_offset_i = static_cast<int32_t>(start_offset);
    const int32_t end_offset_i = static_cast<int32_t>(end_offset);
    const uint32_t date_span = static_cast<uint32_t>(end_offset - start_offset);
    constexpr int32_t kReceiptLagMaxDays = 30;
    const int32_t shipdate_min = start_offset_i - kReceiptLagMaxDays;

    int32_t allowed_code1 = -1;
    int32_t allowed_code2 = -1;
    for (uint32_t code = 0; code < lineitem.shipmode.dictionary.size(); ++code) {
        const auto& value = lineitem.shipmode.dictionary[code];
        if (value == args.SHIPMODE1) {
            allowed_code1 = static_cast<int32_t>(code);
        } else if (value == args.SHIPMODE2) {
            allowed_code2 = static_cast<int32_t>(code);
        }
    }
    if (allowed_code1 < 0 && allowed_code2 < 0) {
#ifdef TRACE
        q12_trace::emit();
#endif
        return {};
    }

    int32_t high_code1 = -1;
    int32_t high_code2 = -1;
    for (uint32_t code = 0; code < orders.orderpriority.dictionary.size(); ++code) {
        const auto& value = orders.orderpriority.dictionary[code];
        if (value == "1-URGENT") {
            high_code1 = static_cast<int32_t>(code);
        } else if (value == "2-HIGH") {
            high_code2 = static_cast<int32_t>(code);
        }
    }

    std::vector<int64_t> high_counts(lineitem.shipmode.dictionary.size(), 0);
    std::vector<int64_t> low_counts(lineitem.shipmode.dictionary.size(), 0);

    uint64_t lineitems_emitted = 0;
    uint64_t lineitems_for_join = 0;
    {
        PROFILE_SCOPE("q12_lineitem_scan");
        const uint16_t* __restrict shipmode_codes = lineitem.shipmode.codes.data();
        const uint32_t* __restrict commit_receipt = lineitem.commit_receipt.data();
        const int16_t* __restrict shipdates = lineitem.shipdate.data();
        const int32_t* __restrict orderkeys = lineitem.orderkey.data();
        const int32_t* __restrict orderkey_to_row = orders.orderkey_to_row.data();
        const size_t orderkey_to_row_size = orders.orderkey_to_row.size();
        const uint16_t* __restrict priority_codes = orders.orderpriority.codes.data();
        int64_t* __restrict high_counts_ptr = high_counts.data();
        int64_t* __restrict low_counts_ptr = low_counts.data();
        uint64_t lineitems_scanned = 0;

        if (lineitem.shards.empty()) {
            lineitems_scanned = lineitem.row_count;
            for (uint32_t li_idx = 0; li_idx < lineitem.row_count; ++li_idx) {
                const int32_t shipmode_code =
                    static_cast<int32_t>(shipmode_codes[li_idx]);
                if (shipmode_code != allowed_code1 && shipmode_code != allowed_code2) {
                    continue;
                }
                const uint32_t packed = commit_receipt[li_idx];
                const int32_t commitdate = static_cast<int16_t>(packed & 0xFFFF);
                const int32_t receiptdate = static_cast<int16_t>(packed >> 16);
                if (commitdate >= receiptdate) {
                    continue;
                }
                if (static_cast<int32_t>(shipdates[li_idx]) >= commitdate) {
                    continue;
                }
                if (static_cast<uint32_t>(receiptdate - start_offset_i) >= date_span) {
                    continue;
                }
                lineitems_for_join += 1;
                const int32_t orderkey = orderkeys[li_idx];
                if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
                    continue;
                }
                const int32_t order_row = orderkey_to_row[orderkey];
                if (order_row < 0) {
                    continue;
                }
                const int32_t priority_code =
                    static_cast<int32_t>(priority_codes[order_row]);
                if (priority_code == high_code1 || priority_code == high_code2) {
                    high_counts_ptr[static_cast<uint32_t>(shipmode_code)] += 1;
                } else {
                    low_counts_ptr[static_cast<uint32_t>(shipmode_code)] += 1;
                }
                lineitems_emitted += 1;
            }
        } else {
            for (const auto& shard : lineitem.shards) {
                if (shard.max_shipdate < shipdate_min ||
                    shard.min_shipdate >= end_offset_i) {
                    continue;
                }
                const uint32_t* __restrict row_indices = shard.row_indices.data();
                const size_t shard_rows = shard.row_indices.size();
                lineitems_scanned += shard_rows;
                for (size_t pos = 0; pos < shard_rows; ++pos) {
                    const uint32_t li_idx = row_indices[pos];
                    const int32_t shipmode_code =
                        static_cast<int32_t>(shipmode_codes[li_idx]);
                    if (shipmode_code != allowed_code1 && shipmode_code != allowed_code2) {
                        continue;
                    }
                    const uint32_t packed = commit_receipt[li_idx];
                    const int32_t commitdate = static_cast<int16_t>(packed & 0xFFFF);
                    const int32_t receiptdate = static_cast<int16_t>(packed >> 16);
                    if (commitdate >= receiptdate) {
                        continue;
                    }
                    if (static_cast<int32_t>(shipdates[li_idx]) >= commitdate) {
                        continue;
                    }
                    if (static_cast<uint32_t>(receiptdate - start_offset_i) >=
                        date_span) {
                        continue;
                    }
                    lineitems_for_join += 1;
                    const int32_t orderkey = orderkeys[li_idx];
                    if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
                        continue;
                    }
                    const int32_t order_row = orderkey_to_row[orderkey];
                    if (order_row < 0) {
                        continue;
                    }
                    const int32_t priority_code =
                        static_cast<int32_t>(priority_codes[order_row]);
                    if (priority_code == high_code1 || priority_code == high_code2) {
                        high_counts_ptr[static_cast<uint32_t>(shipmode_code)] += 1;
                    } else {
                        low_counts_ptr[static_cast<uint32_t>(shipmode_code)] += 1;
                    }
                    lineitems_emitted += 1;
                }
            }
        }
        TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
    }
    TRACE_SET(lineitem_rows_emitted, lineitems_emitted);
    TRACE_SET(join_build_rows_in, orders.row_count);
    TRACE_SET(join_probe_rows_in, lineitems_for_join);
    TRACE_SET(join_rows_emitted, lineitems_emitted);
    TRACE_SET(agg_rows_in, lineitems_emitted);

    {
        PROFILE_SCOPE("q12_agg_finalize");
        results.reserve(2);
        for (uint32_t code = 0; code < lineitem.shipmode.dictionary.size(); ++code) {
            if (static_cast<int32_t>(code) != allowed_code1 &&
                static_cast<int32_t>(code) != allowed_code2) {
                continue;
            }
            const int64_t high = high_counts[code];
            const int64_t low = low_counts[code];
            if (high == 0 && low == 0) {
                continue;
            }
            Q12ResultRow row;
            row.shipmode = lineitem.shipmode.dictionary[code];
            row.high_line_count = high;
            row.low_line_count = low;
            results.push_back(std::move(row));
        }
        TRACE_SET(groups_created, results.size());
        TRACE_SET(agg_rows_emitted, results.size());
    }

    {
        PROFILE_SCOPE("q12_sort");
        TRACE_SET(sort_rows_in, results.size());
        std::sort(results.begin(), results.end(),
                  [](const Q12ResultRow& a, const Q12ResultRow& b) {
                      return a.shipmode < b.shipmode;
                  });
        TRACE_SET(sort_rows_out, results.size());
    }
    TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q12_trace::emit();
#endif
    return results;
}

void write_q12_csv(const std::string& filename, const std::vector<Q12ResultRow>& rows) {
    std::ofstream out(filename);
    out << "l_shipmode,high_line_count,low_line_count\n";
    for (const auto& row : rows) {
        out << q12::escape_csv_field(row.shipmode) << ',' << row.high_line_count << ','
            << row.low_line_count << "\n";
    }
}