#include "query_q10.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q10 {
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

std::string escape_csv_field(std::string_view value) {
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

void append_csv_field(std::string& out, std::string_view value) {
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
}

void append_fixed_2(std::string& out, int64_t value) {
    if (value < 0) {
        out.push_back('-');
        value = -value;
    }
    const int64_t integer = value / kPriceScale;
    const int64_t frac = value % kPriceScale;
    char buf[32];
    auto res = std::to_chars(std::begin(buf), std::end(buf), integer);
    out.append(buf, res.ptr);
    out.push_back('.');
    out.push_back(static_cast<char>('0' + (frac / 10)));
    out.push_back(static_cast<char>('0' + (frac % 10)));
}

template <typename T>
int32_t max_value(const std::vector<T>& values) {
    if (values.empty()) {
        return 0;
    }
    return static_cast<int32_t>(*std::max_element(values.begin(), values.end()));
}

void radix_sort_rows_by_orderkey(const std::vector<int32_t>& orderkey,
                                 std::vector<uint32_t>& rows) {
    if (rows.size() < 2) {
        return;
    }
    constexpr uint32_t kRadix = 1u << 16;
    constexpr uint32_t kMask = kRadix - 1u;
    std::vector<uint32_t> tmp(rows.size());
    std::vector<uint32_t> counts(kRadix);

    auto pass = [&](uint32_t shift,
                    const std::vector<uint32_t>& input,
                    std::vector<uint32_t>& output) {
        std::fill(counts.begin(), counts.end(), 0);
        for (const uint32_t idx : input) {
            const uint32_t key = static_cast<uint32_t>(orderkey[idx]);
            counts[(key >> shift) & kMask] += 1;
        }
        uint32_t sum = 0;
        for (uint32_t& count : counts) {
            const uint32_t current = count;
            count = sum;
            sum += current;
        }
        for (const uint32_t idx : input) {
            const uint32_t key = static_cast<uint32_t>(orderkey[idx]);
            output[counts[(key >> shift) & kMask]++] = idx;
        }
    };

    pass(0, rows, tmp);
    pass(16, tmp, rows);
}
}  // namespace q10

#ifdef TRACE
namespace q10_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t customer_scan_ns = 0;
    uint64_t orders_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t nation_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t customer_rows_scanned = 0;
    uint64_t customer_rows_emitted = 0;
    uint64_t orders_rows_scanned = 0;
    uint64_t orders_rows_emitted = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
    uint64_t nation_rows_scanned = 0;
    uint64_t nation_rows_emitted = 0;
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
    if (std::strcmp(name, "q10_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q10_customer_scan") == 0) {
        trace.customer_scan_ns += ns;
    } else if (std::strcmp(name, "q10_orders_scan") == 0) {
        trace.orders_scan_ns += ns;
    } else if (std::strcmp(name, "q10_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q10_nation_scan") == 0) {
        trace.nation_scan_ns += ns;
    } else if (std::strcmp(name, "q10_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q10_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q10_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q10_customer_scan " << trace.customer_scan_ns << "\n";
    std::cout << "PROFILE q10_orders_scan " << trace.orders_scan_ns << "\n";
    std::cout << "PROFILE q10_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q10_nation_scan " << trace.nation_scan_ns << "\n";
    std::cout << "PROFILE q10_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q10_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q10_customer_scan_rows_scanned " << trace.customer_rows_scanned
              << "\n";
    std::cout << "COUNT q10_customer_scan_rows_emitted " << trace.customer_rows_emitted
              << "\n";
    std::cout << "COUNT q10_orders_scan_rows_scanned " << trace.orders_rows_scanned
              << "\n";
    std::cout << "COUNT q10_orders_scan_rows_emitted " << trace.orders_rows_emitted
              << "\n";
    std::cout << "COUNT q10_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q10_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q10_nation_scan_rows_scanned " << trace.nation_rows_scanned
              << "\n";
    std::cout << "COUNT q10_nation_scan_rows_emitted " << trace.nation_rows_emitted
              << "\n";
    std::cout << "COUNT q10_order_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q10_order_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q10_order_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q10_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q10_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q10_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q10_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q10_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q10_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q10_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q10_trace::record_timing)
#define TRACE_SET(field, value) (q10_trace::data().field = (value))
#define TRACE_ADD(field, value) (q10_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

struct Q10AggEntry {
    int32_t custkey = 0;
    int32_t row_idx = 0;
    int32_t nationkey = 0;
    int64_t revenue = 0;
};

struct OrderRevenue {
    int32_t custkey = 0;
    int64_t revenue = 0;
};

void radix_sort_order_revenues(std::vector<OrderRevenue>& entries) {
    if (entries.size() < 2) {
        return;
    }
    constexpr uint32_t kRadix = 1u << 16;
    constexpr uint32_t kMask = kRadix - 1u;
    std::vector<OrderRevenue> tmp(entries.size());
    std::vector<uint32_t> counts(kRadix);

    auto pass = [&](uint32_t shift,
                    const std::vector<OrderRevenue>& input,
                    std::vector<OrderRevenue>& output) {
        std::fill(counts.begin(), counts.end(), 0);
        for (const auto& entry : input) {
            const uint32_t key = static_cast<uint32_t>(entry.custkey);
            counts[(key >> shift) & kMask] += 1;
        }
        uint32_t sum = 0;
        for (uint32_t& count : counts) {
            const uint32_t current = count;
            count = sum;
            sum += current;
        }
        for (const auto& entry : input) {
            const uint32_t key = static_cast<uint32_t>(entry.custkey);
            output[counts[(key >> shift) & kMask]++] = entry;
        }
    };

    pass(0, entries, tmp);
    pass(16, tmp, entries);
}

void radix_sort_groups(std::vector<Q10AggEntry>& groups) {
    if (groups.size() < 2) {
        return;
    }
    constexpr uint32_t kRadix = 1u << 16;
    constexpr uint32_t kMask = kRadix - 1u;
    std::vector<Q10AggEntry> tmp(groups.size());
    std::vector<uint32_t> counts(kRadix);

    auto pass_custkey = [&](uint32_t shift,
                            const std::vector<Q10AggEntry>& input,
                            std::vector<Q10AggEntry>& output) {
        std::fill(counts.begin(), counts.end(), 0);
        for (const auto& entry : input) {
            const uint32_t key = static_cast<uint32_t>(entry.custkey);
            counts[(key >> shift) & kMask] += 1;
        }
        uint32_t sum = 0;
        for (uint32_t& count : counts) {
            const uint32_t current = count;
            count = sum;
            sum += current;
        }
        for (const auto& entry : input) {
            const uint32_t key = static_cast<uint32_t>(entry.custkey);
            output[counts[(key >> shift) & kMask]++] = entry;
        }
    };

    auto pass_revenue = [&](uint32_t shift,
                            const std::vector<Q10AggEntry>& input,
                            std::vector<Q10AggEntry>& output) {
        std::fill(counts.begin(), counts.end(), 0);
        for (const auto& entry : input) {
            const uint64_t key = ~static_cast<uint64_t>(entry.revenue);
            counts[(key >> shift) & kMask] += 1;
        }
        uint32_t sum = 0;
        for (uint32_t& count : counts) {
            const uint32_t current = count;
            count = sum;
            sum += current;
        }
        for (const auto& entry : input) {
            const uint64_t key = ~static_cast<uint64_t>(entry.revenue);
            output[counts[(key >> shift) & kMask]++] = entry;
        }
    };

    pass_custkey(0, groups, tmp);
    pass_custkey(16, tmp, groups);
    pass_revenue(0, groups, tmp);
    pass_revenue(16, tmp, groups);
    pass_revenue(32, groups, tmp);
    pass_revenue(48, tmp, groups);
}

std::vector<Q10ResultRow> run_q10(const Database& db, const Q10Args& args) {
#ifdef TRACE
    q10_trace::reset();
#endif
    std::vector<Q10ResultRow> results;
    {
        PROFILE_SCOPE("q10_total");
    const auto& customer = db.customer;
    const auto& orders = db.orders;
    const auto& lineitem = db.lineitem;

    const int16_t start_offset = q10::parse_date_offset(args.DATE, db.base_date_days);
    const int16_t end_offset =
        q10::parse_date_offset_add_months(args.DATE, db.base_date_days, 3);

    int32_t returnflag_code = -1;
    for (uint32_t code = 0; code < lineitem.returnflag.dictionary.size(); ++code) {
        if (lineitem.returnflag.dictionary[code] == "R") {
            returnflag_code = static_cast<int32_t>(code);
            break;
        }
    }
    if (returnflag_code < 0) {
#ifdef TRACE
        q10_trace::emit();
#endif
        return {};
    }

    const int32_t max_custkey = q10::max_value(customer.custkey);
    std::vector<int32_t> custkey_to_row(static_cast<size_t>(max_custkey) + 1, -1);
    uint64_t customers_emitted = 0;
    {
        PROFILE_SCOPE("q10_customer_scan");
        TRACE_SET(customer_rows_scanned, customer.row_count);
        for (uint32_t i = 0; i < customer.row_count; ++i) {
            const int32_t key = customer.custkey[i];
            if (key >= 0) {
                custkey_to_row[static_cast<size_t>(key)] = static_cast<int32_t>(i);
                customers_emitted += 1;
            }
        }
    }
    TRACE_SET(customer_rows_emitted, customers_emitted);

    std::vector<OrderRevenue> order_revenues;

    {
        PROFILE_SCOPE("q10_orders_scan");
        uint32_t order_start = 0;
        uint32_t order_end = static_cast<uint32_t>(orders.row_count);
        if (!orders.orderdate.empty()) {
            const auto begin_it =
                std::lower_bound(orders.orderdate.begin(),
                                 orders.orderdate.end(),
                                 start_offset);
            const auto end_it =
                std::lower_bound(orders.orderdate.begin(),
                                 orders.orderdate.end(),
                                 end_offset);
            order_start = static_cast<uint32_t>(begin_it - orders.orderdate.begin());
            order_end = static_cast<uint32_t>(end_it - orders.orderdate.begin());
        }
        const bool lineitem_sorted = lineitem.orderkey_sorted;
        order_revenues.reserve(order_end - order_start);
        std::vector<uint32_t> order_rows;
        order_rows.reserve(order_end - order_start);
        for (uint32_t o_idx = order_start; o_idx < order_end; ++o_idx) {
            order_rows.push_back(o_idx);
        }
        if (lineitem_sorted) {
            q10::radix_sort_rows_by_orderkey(orders.orderkey, order_rows);
        }
        TRACE_SET(orders_rows_scanned,
                  static_cast<uint64_t>(order_rows.size()));
        const int32_t* __restrict lineitem_orderkey = lineitem.orderkey.data();
        const uint8_t* __restrict group_codes = lineitem.returnflag_linestatus.data();
        const int32_t* __restrict discounted_price = lineitem.discounted_price.data();
        const int32_t* __restrict orders_custkey = orders.custkey.data();
        const int32_t* __restrict orders_orderkey = orders.orderkey.data();
        const auto* __restrict order_ranges = orders.lineitem_ranges.data();
        const size_t revenue_size = static_cast<size_t>(max_custkey) + 1;
        const uint32_t linestatus_count =
            static_cast<uint32_t>(lineitem.linestatus.dictionary.size());
        const uint32_t returnflag_base =
            static_cast<uint32_t>(returnflag_code) * linestatus_count;
        const uint32_t returnflag_span = linestatus_count;
        for (uint32_t pos = 0; pos < order_rows.size(); ++pos) {
            const uint32_t o_idx = order_rows[pos];
            const int32_t custkey = orders_custkey[o_idx];
            if (custkey < 0 || static_cast<size_t>(custkey) >= revenue_size) {
                continue;
            }
            TRACE_ADD(orders_rows_emitted, 1);
            const int32_t orderkey = orders_orderkey[o_idx];
            const auto range = order_ranges[o_idx];
            if (range.end == 0) {
                continue;
            }

            const uint32_t start = range.start;
            const uint32_t end = range.end;
            TRACE_ADD(lineitem_rows_scanned, static_cast<uint64_t>(end - start));
#ifdef TRACE
            const uint64_t lineitem_start_ns = trace_utils::get_time_ns();
#endif
            int64_t order_revenue = 0;
            if (lineitem_sorted) {
                for (uint32_t li_idx = start; li_idx < end; ++li_idx) {
                    const uint32_t group_code = group_codes[li_idx];
                    if ((group_code - returnflag_base) >= returnflag_span) {
                        continue;
                    }
                    const int64_t revenue =
                        static_cast<int64_t>(discounted_price[li_idx]);
                    order_revenue += revenue;
                    TRACE_ADD(lineitem_rows_emitted, 1);
                }
            } else {
                for (uint32_t li_idx = start; li_idx < end; ++li_idx) {
                    if (lineitem_orderkey[li_idx] != orderkey) {
                        continue;
                    }
                    const uint32_t group_code = group_codes[li_idx];
                    if ((group_code - returnflag_base) >= returnflag_span) {
                        continue;
                    }
                    const int64_t revenue =
                        static_cast<int64_t>(discounted_price[li_idx]);
                    order_revenue += revenue;
                    TRACE_ADD(lineitem_rows_emitted, 1);
                }
            }
            if (order_revenue != 0) {
                order_revenues.push_back(OrderRevenue{custkey, order_revenue});
            }
#ifdef TRACE
            q10_trace::record_timing("q10_lineitem_scan",
                                     trace_utils::get_time_ns() - lineitem_start_ns);
#endif
        }
    }
    TRACE_SET(join_build_rows_in, q10_trace::data().orders_rows_emitted);
    TRACE_SET(join_probe_rows_in, q10_trace::data().lineitem_rows_scanned);
    TRACE_SET(join_rows_emitted, q10_trace::data().lineitem_rows_emitted);
    TRACE_SET(agg_rows_in, q10_trace::data().lineitem_rows_emitted);

    int32_t max_nationkey = 0;
    for (const auto& row : db.nation.rows) {
        max_nationkey = std::max(max_nationkey, row.nationkey);
    }
    std::vector<std::string_view> nation_name(static_cast<size_t>(max_nationkey) + 1);
    {
        PROFILE_SCOPE("q10_nation_scan");
        TRACE_SET(nation_rows_scanned, db.nation.rows.size());
        uint64_t nation_emitted = 0;
        for (const auto& row : db.nation.rows) {
            if (row.nationkey >= 0) {
                nation_name[static_cast<size_t>(row.nationkey)] = row.name;
                nation_emitted += 1;
            }
        }
        TRACE_SET(nation_rows_emitted, nation_emitted);
    }

    radix_sort_order_revenues(order_revenues);

    std::vector<Q10AggEntry> groups;
    {
        PROFILE_SCOPE("q10_agg_finalize");
        groups.reserve(order_revenues.size());
        const int32_t* __restrict customer_nationkey = customer.nationkey.data();
        size_t idx = 0;
        while (idx < order_revenues.size()) {
            const int32_t custkey = order_revenues[idx].custkey;
            int64_t revenue = 0;
            do {
                revenue += order_revenues[idx].revenue;
                idx += 1;
            } while (idx < order_revenues.size() &&
                     order_revenues[idx].custkey == custkey);
            const int32_t row_idx = custkey_to_row[static_cast<size_t>(custkey)];
            if (row_idx < 0) {
                continue;
            }
            const int32_t nationkey = customer_nationkey[static_cast<size_t>(row_idx)];
            if (nationkey < 0 ||
                static_cast<size_t>(nationkey) >= nation_name.size()) {
                continue;
            }
            Q10AggEntry entry;
            entry.custkey = custkey;
            entry.row_idx = row_idx;
            entry.nationkey = nationkey;
            entry.revenue = revenue;
            groups.push_back(entry);
        }
        TRACE_SET(groups_created, groups.size());
        TRACE_SET(agg_rows_emitted, groups.size());
    }

    {
        PROFILE_SCOPE("q10_sort");
        TRACE_SET(sort_rows_in, groups.size());
        radix_sort_groups(groups);
        TRACE_SET(sort_rows_out, groups.size());
    }
    results.reserve(groups.size());
    for (const auto& entry : groups) {
        Q10ResultRow row;
        row.custkey = entry.custkey;
        row.name = q10::get_string_view(customer.name,
                                        static_cast<size_t>(entry.row_idx));
        row.revenue_raw = entry.revenue;
        row.acctbal_raw = customer.acctbal[static_cast<size_t>(entry.row_idx)];
        row.nation_name = nation_name[static_cast<size_t>(entry.nationkey)];
        row.address = q10::get_string_view(
            customer.address, static_cast<size_t>(entry.row_idx));
        row.phone = q10::get_string_view(
            customer.phone, static_cast<size_t>(entry.row_idx));
        row.comment = q10::get_string_view(
            customer.comment, static_cast<size_t>(entry.row_idx));
        results.push_back(std::move(row));
    }
    TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q10_trace::emit();
#endif
    return results;
}

void write_q10_csv(const std::string& filename, const std::vector<Q10ResultRow>& rows) {
    std::ofstream out(filename, std::ios::binary);
    std::string buffer;
    buffer.reserve(1 << 20);
    buffer.append(
        "c_custkey,c_name,revenue,c_acctbal,n_name,c_address,c_phone,c_comment\n");
    for (const auto& row : rows) {
        char int_buf[32];
        auto int_res = std::to_chars(std::begin(int_buf), std::end(int_buf),
                                     row.custkey);
        buffer.append(int_buf, int_res.ptr);
        buffer.push_back(',');
        q10::append_csv_field(buffer, row.name);
        buffer.push_back(',');
        q10::append_fixed_2(buffer, row.revenue_raw);
        buffer.push_back(',');
        q10::append_fixed_2(buffer, row.acctbal_raw);
        buffer.push_back(',');
        q10::append_csv_field(buffer, row.nation_name);
        buffer.push_back(',');
        q10::append_csv_field(buffer, row.address);
        buffer.push_back(',');
        q10::append_csv_field(buffer, row.phone);
        buffer.push_back(',');
        q10::append_csv_field(buffer, row.comment);
        buffer.push_back('\n');
        if (buffer.size() >= (1 << 20)) {
            out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            buffer.clear();
        }
    }
    if (!buffer.empty()) {
        out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    }
}