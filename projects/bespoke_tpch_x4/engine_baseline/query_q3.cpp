#include "query_q3.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "trace_utils.hpp"

namespace q3 {
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

void civil_from_days(int z, int& y, unsigned& m, unsigned& d) {
    z += 719468;
    const int era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<int>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
}

int16_t parse_date_offset(const std::string& value, int32_t base_days) {
    const int year = std::stoi(value.substr(0, 4));
    const int month = std::stoi(value.substr(5, 2));
    const int day = std::stoi(value.substr(8, 2));
    const int32_t days = static_cast<int32_t>(days_from_civil(year, month, day));
    return static_cast<int16_t>(days - base_days);
}

std::string format_date(int32_t base_days, int16_t offset) {
    const int32_t days = base_days + offset;
    int year = 0;
    unsigned month = 0;
    unsigned day = 0;
    civil_from_days(days, year, month, day);
    std::ostringstream oss;
    oss << std::setw(4) << std::setfill('0') << year << '-'
        << std::setw(2) << std::setfill('0') << month << '-'
        << std::setw(2) << std::setfill('0') << day;
    return oss.str();
}

int32_t max_value(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return 0;
    }
    return *std::max_element(values.begin(), values.end());
}
}  // namespace q3

#ifdef TRACE
namespace q3_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t customer_scan_ns = 0;
    uint64_t orders_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t customer_rows_scanned = 0;
    uint64_t customer_rows_emitted = 0;
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
    if (std::strcmp(name, "q3_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q3_customer_scan") == 0) {
        trace.customer_scan_ns += ns;
    } else if (std::strcmp(name, "q3_orders_scan") == 0) {
        trace.orders_scan_ns += ns;
    } else if (std::strcmp(name, "q3_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q3_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q3_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q3_customer_scan " << trace.customer_scan_ns << "\n";
    std::cout << "PROFILE q3_orders_scan " << trace.orders_scan_ns << "\n";
    std::cout << "PROFILE q3_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q3_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q3_customer_scan_rows_scanned " << trace.customer_rows_scanned << "\n";
    std::cout << "COUNT q3_customer_scan_rows_emitted " << trace.customer_rows_emitted << "\n";
    std::cout << "COUNT q3_orders_scan_rows_scanned " << trace.orders_rows_scanned << "\n";
    std::cout << "COUNT q3_orders_scan_rows_emitted " << trace.orders_rows_emitted << "\n";
    std::cout << "COUNT q3_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned << "\n";
    std::cout << "COUNT q3_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted << "\n";
    std::cout << "COUNT q3_order_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q3_order_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q3_order_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q3_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q3_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q3_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q3_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q3_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q3_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q3_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q3_trace::record_timing)
#define TRACE_SET(field, value) (q3_trace::data().field = (value))
#define TRACE_ADD(field, value) (q3_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q3ResultRow> run_q3(const Database& db, const Q3Args& args) {
    std::vector<Q3ResultRow> results;
#ifdef TRACE
    q3_trace::reset();
#endif
    {
        PROFILE_SCOPE("q3_total");
        const int16_t date_offset = q3::parse_date_offset(args.DATE, db.base_date_days);
        const auto& customer = db.customer;
        const auto& orders = db.orders;
        const auto& lineitem = db.lineitem;
        const int32_t* __restrict discounted_price = lineitem.discounted_price.data();
        const int16_t* __restrict shipdate = lineitem.shipdate.data();
        const int32_t* __restrict lineitem_orderkey = lineitem.orderkey.data();
        const int16_t* __restrict orderdate = orders.orderdate.data();
        const int32_t* __restrict orders_custkey = orders.custkey.data();
        const int32_t* __restrict orders_orderkey = orders.orderkey.data();
        const int32_t* __restrict orders_shippriority = orders.shippriority.data();

        int32_t segment_code = -1;
        const auto& segment_dict = customer.mktsegment.dictionary;
        for (size_t i = 0; i < segment_dict.size(); ++i) {
            if (segment_dict[i] == args.SEGMENT) {
                segment_code = static_cast<int32_t>(i);
                break;
            }
        }

        const int32_t max_custkey = q3::max_value(customer.custkey);
        std::vector<uint8_t> cust_in_segment(static_cast<size_t>(max_custkey) + 1, 0);
        const uint16_t* __restrict mktsegment_codes = customer.mktsegment.codes.data();
        const int32_t* __restrict customer_custkey = customer.custkey.data();
        const uint8_t* __restrict cust_in_segment_ptr = cust_in_segment.data();
        uint64_t cust_segment_count = 0;
        {
            PROFILE_SCOPE("q3_customer_scan");
            TRACE_SET(customer_rows_scanned, customer.row_count);
            if (segment_code >= 0) {
                for (uint32_t i = 0; i < customer.row_count; ++i) {
                    const uint32_t code = mktsegment_codes[i];
                    if (static_cast<int32_t>(code) != segment_code) {
                        continue;
                    }
                    const int32_t key = customer_custkey[i];
                    if (key >= 0) {
                        cust_in_segment[static_cast<size_t>(key)] = 1;
                        cust_segment_count += 1;
                    }
                }
            }
            TRACE_SET(customer_rows_emitted, cust_segment_count);
        }

        const uint32_t orders_row_count = static_cast<uint32_t>(orders.row_count);
        const uint32_t order_limit = static_cast<uint32_t>(
            std::lower_bound(orderdate, orderdate + orders_row_count, date_offset) -
            orderdate);
        results.reserve(order_limit / 4);
        uint64_t orders_emitted = 0;
        uint64_t orders_scanned = 0;
        uint64_t lineitems_scanned = 0;
        uint64_t lineitems_emitted = 0;
        uint64_t orders_with_lineitems = 0;

        {
            PROFILE_SCOPE("q3_orders_scan");
            auto handle_sorted_order = [&](uint32_t o_idx) {
                const auto range = orders.lineitem_ranges[o_idx];
                if (range.end == 0) {
                    return;
                }

                int64_t revenue = 0;
                bool has_lineitem = false;
                const uint32_t start = range.start;
                const uint32_t end = range.end;
                lineitems_scanned += static_cast<uint64_t>(end - start);
                #ifdef TRACE
                const uint64_t scan_start = trace_utils::get_time_ns();
                #endif
                const int16_t* __restrict ship_ptr = shipdate + start;
                const int32_t* __restrict price_ptr = discounted_price + start;
                const uint32_t len = end - start;
                for (uint32_t offset = 0; offset < len; ++offset) {
                    if (ship_ptr[offset] <= date_offset) {
                        continue;
                    }
                    revenue += price_ptr[offset];
                    has_lineitem = true;
                    lineitems_emitted += 1;
                }
                #ifdef TRACE
                q3_trace::record_timing("q3_lineitem_scan",
                                        trace_utils::get_time_ns() - scan_start);
                #endif

                if (!has_lineitem) {
                    return;
                }

                Q3ResultRow row;
                row.orderkey = orders_orderkey[o_idx];
                row.revenue_raw = revenue;
                row.orderdate_offset = orderdate[o_idx];
                row.shippriority = orders_shippriority[o_idx];
                results.push_back(std::move(row));
                orders_with_lineitems += 1;
            };

            auto handle_unsorted_order = [&](uint32_t o_idx) {
                const int32_t orderkey = orders_orderkey[o_idx];
                const auto range = orders.lineitem_ranges[o_idx];
                if (range.end == 0) {
                    return;
                }

                int64_t revenue = 0;
                bool has_lineitem = false;
                const uint32_t start = range.start;
                const uint32_t end = range.end;
                lineitems_scanned += static_cast<uint64_t>(end - start);
                #ifdef TRACE
                const uint64_t scan_start = trace_utils::get_time_ns();
                #endif
                for (uint32_t li_idx = start; li_idx < end; ++li_idx) {
                    if (lineitem_orderkey[li_idx] != orderkey) {
                        continue;
                    }
                    if (shipdate[li_idx] <= date_offset) {
                        continue;
                    }
                    revenue += discounted_price[li_idx];
                    has_lineitem = true;
                    lineitems_emitted += 1;
                }
                #ifdef TRACE
                q3_trace::record_timing("q3_lineitem_scan",
                                        trace_utils::get_time_ns() - scan_start);
                #endif

                if (!has_lineitem) {
                    return;
                }

                Q3ResultRow row;
                row.orderkey = orderkey;
                row.revenue_raw = revenue;
                row.orderdate_offset = orderdate[o_idx];
                row.shippriority = orders_shippriority[o_idx];
                results.push_back(std::move(row));
                orders_with_lineitems += 1;
            };

            if (lineitem.orderkey_sorted) {
                for (uint32_t o_idx = 0; o_idx < order_limit; ++o_idx) {
                    orders_scanned += 1;
                    const int32_t custkey = orders_custkey[o_idx];
                    if (custkey < 0 || static_cast<size_t>(custkey) >= cust_in_segment.size()) {
                        continue;
                    }
                    if (!cust_in_segment_ptr[static_cast<size_t>(custkey)]) {
                        continue;
                    }
                    orders_emitted += 1;
                    handle_sorted_order(o_idx);
                }
            } else {
                for (uint32_t o_idx = 0; o_idx < order_limit; ++o_idx) {
                    orders_scanned += 1;
                    const int32_t custkey = orders_custkey[o_idx];
                    if (custkey < 0 || static_cast<size_t>(custkey) >= cust_in_segment.size()) {
                        continue;
                    }
                    if (!cust_in_segment_ptr[static_cast<size_t>(custkey)]) {
                        continue;
                    }
                    orders_emitted += 1;
                    handle_unsorted_order(o_idx);
                }
            }
            TRACE_SET(orders_rows_scanned, orders_scanned);
            TRACE_SET(orders_rows_emitted, orders_emitted);
        }

        TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
        TRACE_SET(lineitem_rows_emitted, lineitems_emitted);
        TRACE_SET(join_build_rows_in, orders_emitted);
        TRACE_SET(join_probe_rows_in, lineitems_scanned);
        TRACE_SET(join_rows_emitted, lineitems_emitted);
        TRACE_SET(agg_rows_in, lineitems_emitted);
        TRACE_SET(groups_created, orders_with_lineitems);
        TRACE_SET(agg_rows_emitted, orders_with_lineitems);

        TRACE_SET(sort_rows_in, results.size());
        {
            PROFILE_SCOPE("q3_sort");
            std::sort(results.begin(), results.end(),
                      [](const Q3ResultRow& a, const Q3ResultRow& b) {
                          if (a.revenue_raw != b.revenue_raw) {
                              return a.revenue_raw > b.revenue_raw;
                          }
                          return a.orderdate_offset < b.orderdate_offset;
                      });
        }
        TRACE_SET(sort_rows_out, results.size());
    }

    TRACE_SET(query_output_rows, results.size());
#ifdef TRACE
    q3_trace::emit();
#endif
    return results;
}

void write_q3_csv(const Database& db,
                  const std::string& filename,
                  const std::vector<Q3ResultRow>& rows) {
    std::ofstream out(filename);
    out << "l_orderkey,revenue,o_orderdate,o_shippriority\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double revenue = static_cast<double>(row.revenue_raw) / q3::kPriceScale;
        out << row.orderkey << ',' << revenue << ','
            << q3::format_date(db.base_date_days, row.orderdate_offset) << ','
            << row.shippriority << "\n";
    }
}