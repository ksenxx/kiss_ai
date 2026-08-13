#include "query_q5.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q5 {
constexpr int32_t kPriceScale = 100;
constexpr int32_t kDiscountScale = 100;
constexpr int64_t kRevenueScale = static_cast<int64_t>(kPriceScale);

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

int16_t parse_date_offset_add_years(const std::string& value,
                                    int32_t base_days,
                                    int years) {
    int year = std::stoi(value.substr(0, 4));
    int month = std::stoi(value.substr(5, 2));
    int day = std::stoi(value.substr(8, 2));
    year += years;
    const int dim = days_in_month(year, month);
    if (day > dim) {
        day = dim;
    }
    const int32_t days = static_cast<int32_t>(days_from_civil(year, month, day));
    return static_cast<int16_t>(days - base_days);
}

std::string escape_csv_field_q5(const std::string& value) {
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

}  // namespace q5

#ifdef TRACE
namespace q5_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t nation_scan_ns = 0;
    uint64_t customer_scan_ns = 0;
    uint64_t supplier_scan_ns = 0;
    uint64_t orders_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t nation_rows_scanned = 0;
    uint64_t nation_rows_emitted = 0;
    uint64_t customer_rows_scanned = 0;
    uint64_t customer_rows_emitted = 0;
    uint64_t supplier_rows_scanned = 0;
    uint64_t supplier_rows_emitted = 0;
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
    if (std::strcmp(name, "q5_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q5_nation_scan") == 0) {
        trace.nation_scan_ns += ns;
    } else if (std::strcmp(name, "q5_customer_scan") == 0) {
        trace.customer_scan_ns += ns;
    } else if (std::strcmp(name, "q5_supplier_scan") == 0) {
        trace.supplier_scan_ns += ns;
    } else if (std::strcmp(name, "q5_orders_scan") == 0) {
        trace.orders_scan_ns += ns;
    } else if (std::strcmp(name, "q5_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q5_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q5_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q5_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q5_nation_scan " << trace.nation_scan_ns << "\n";
    std::cout << "PROFILE q5_customer_scan " << trace.customer_scan_ns << "\n";
    std::cout << "PROFILE q5_supplier_scan " << trace.supplier_scan_ns << "\n";
    std::cout << "PROFILE q5_orders_scan " << trace.orders_scan_ns << "\n";
    std::cout << "PROFILE q5_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q5_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q5_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q5_nation_scan_rows_scanned " << trace.nation_rows_scanned << "\n";
    std::cout << "COUNT q5_nation_scan_rows_emitted " << trace.nation_rows_emitted << "\n";
    std::cout << "COUNT q5_customer_scan_rows_scanned " << trace.customer_rows_scanned
              << "\n";
    std::cout << "COUNT q5_customer_scan_rows_emitted " << trace.customer_rows_emitted
              << "\n";
    std::cout << "COUNT q5_supplier_scan_rows_scanned " << trace.supplier_rows_scanned
              << "\n";
    std::cout << "COUNT q5_supplier_scan_rows_emitted " << trace.supplier_rows_emitted
              << "\n";
    std::cout << "COUNT q5_orders_scan_rows_scanned " << trace.orders_rows_scanned << "\n";
    std::cout << "COUNT q5_orders_scan_rows_emitted " << trace.orders_rows_emitted << "\n";
    std::cout << "COUNT q5_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q5_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q5_order_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q5_order_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q5_order_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q5_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q5_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q5_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q5_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q5_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q5_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q5_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q5_trace::record_timing)
#define TRACE_SET(field, value) (q5_trace::data().field = (value))
#define TRACE_ADD(field, value) (q5_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q5ResultRow> run_q5(const Database& db, const Q5Args& args) {
#ifdef TRACE
    q5_trace::reset();
#endif
    std::vector<Q5ResultRow> results;
    {
        PROFILE_SCOPE("q5_total");
    const auto region_it = db.region.name_to_key.find(args.REGION);
    if (region_it == db.region.name_to_key.end()) {
        return {};
    }
    const int32_t region_key = region_it->second;
    const int16_t start_offset = q5::parse_date_offset(args.DATE, db.base_date_days);
    const int16_t end_offset = q5::parse_date_offset_add_years(args.DATE,
                                                               db.base_date_days,
                                                               1);

    const auto& nation = db.nation;
    const auto& customer = db.customer;
    const auto& supplier = db.supplier;
    const auto& orders = db.orders;
    const auto& lineitem = db.lineitem;

    int32_t max_nationkey = 0;
    for (const auto& row : nation.rows) {
        max_nationkey = std::max(max_nationkey, row.nationkey);
    }

    std::vector<uint8_t> nation_in_region(static_cast<size_t>(max_nationkey) + 1, 0);
    std::vector<std::string_view> nation_name(static_cast<size_t>(max_nationkey) + 1);
    {
        PROFILE_SCOPE("q5_nation_scan");
        TRACE_SET(nation_rows_scanned, nation.rows.size());
        uint64_t nation_emitted = 0;
        for (const auto& row : nation.rows) {
            if (row.nationkey < 0) {
                continue;
            }
            const size_t idx = static_cast<size_t>(row.nationkey);
            nation_name[idx] = row.name;
            if (row.regionkey == region_key) {
                nation_in_region[idx] = 1;
                nation_emitted += 1;
            }
        }
        TRACE_SET(nation_rows_emitted, nation_emitted);
    }

    const int16_t* __restrict cust_nationkey_data = nullptr;
    const int16_t* __restrict supp_nationkey_data = nullptr;
    std::vector<int16_t> cust_nationkey_fallback;
    std::vector<int16_t> supp_nationkey_fallback;
    {
        PROFILE_SCOPE("q5_customer_scan");
        if (!customer.nationkey_by_custkey.empty()) {
            cust_nationkey_data = customer.nationkey_by_custkey.data();
            TRACE_SET(customer_rows_scanned, 0);
            TRACE_SET(customer_rows_emitted, 0);
        } else {
            TRACE_SET(customer_rows_scanned, customer.row_count);
            uint64_t customer_emitted = 0;
            cust_nationkey_fallback.assign(customer.row_count + 1, -1);
            const int32_t* __restrict custkey_ptr = customer.custkey.data();
            const int32_t* __restrict nationkey_ptr = customer.nationkey.data();
            for (uint32_t i = 0; i < customer.row_count; ++i) {
                const int32_t key = custkey_ptr[i];
                const int32_t nation_key = nationkey_ptr[i];
                cust_nationkey_fallback[static_cast<size_t>(key)] =
                    nation_in_region[static_cast<size_t>(nation_key)] ?
                        static_cast<int16_t>(nation_key) :
                        static_cast<int16_t>(-1);
                customer_emitted += 1;
            }
            TRACE_SET(customer_rows_emitted, customer_emitted);
            cust_nationkey_data = cust_nationkey_fallback.data();
        }
    }

    {
        PROFILE_SCOPE("q5_supplier_scan");
        if (!supplier.nationkey_by_suppkey.empty()) {
            supp_nationkey_data = supplier.nationkey_by_suppkey.data();
            TRACE_SET(supplier_rows_scanned, 0);
            TRACE_SET(supplier_rows_emitted, 0);
        } else {
            TRACE_SET(supplier_rows_scanned, supplier.row_count);
            uint64_t supplier_emitted = 0;
            supp_nationkey_fallback.assign(supplier.row_count + 1, -1);
            const int32_t* __restrict suppkey_ptr = supplier.suppkey.data();
            const int32_t* __restrict nationkey_ptr = supplier.nationkey.data();
            for (uint32_t i = 0; i < supplier.row_count; ++i) {
                const int32_t key = suppkey_ptr[i];
                supp_nationkey_fallback[static_cast<size_t>(key)] =
                    static_cast<int16_t>(nationkey_ptr[i]);
                supplier_emitted += 1;
            }
            TRACE_SET(supplier_rows_emitted, supplier_emitted);
            supp_nationkey_data = supp_nationkey_fallback.data();
        }
    }

    std::vector<int64_t> revenue_by_nation(static_cast<size_t>(max_nationkey) + 1, 0);
    int64_t* __restrict revenue_data = revenue_by_nation.data();

    const bool orderkey_sorted = lineitem.orderkey_sorted;
    const int16_t* orderdates = orders.orderdate.data();
    const size_t orders_row_count = orders.row_count;
    const int16_t* date_start =
        std::lower_bound(orderdates, orderdates + orders_row_count, start_offset);
    const int16_t* date_end =
        std::lower_bound(orderdates, orderdates + orders_row_count, end_offset);
    const uint32_t start_idx = static_cast<uint32_t>(date_start - orderdates);
    const uint32_t end_idx = static_cast<uint32_t>(date_end - orderdates);

    const int32_t* __restrict orderkeys = orders.orderkey.data();
    const int32_t* __restrict custkeys = orders.custkey.data();
    const int16_t* __restrict order_cust_nation = orders.cust_nationkey.data();
    const bool use_order_cust_nation = !orders.cust_nationkey.empty();
    const auto* __restrict order_ranges = orders.lineitem_ranges.data();
    const int32_t* __restrict lineitem_suppkey = lineitem.suppkey.data();
    const int16_t* __restrict lineitem_supp_nation = lineitem.supp_nationkey.data();
    const bool use_lineitem_supp_nation = !lineitem.supp_nationkey.empty();
    const int32_t* __restrict lineitem_discounted_price =
        lineitem.discounted_price.data();

    uint64_t orders_emitted = 0;
    uint64_t lineitems_scanned = 0;
    uint64_t lineitems_emitted = 0;
    {
        PROFILE_SCOPE("q5_orders_scan");
        TRACE_SET(orders_rows_scanned, static_cast<uint64_t>(end_idx - start_idx));
        if (orderkey_sorted) {
            for (uint32_t o_idx = start_idx; o_idx < end_idx; ++o_idx) {
                int16_t cnation = -1;
                if (use_order_cust_nation) {
                    cnation = order_cust_nation[o_idx];
                } else {
                    const int32_t custkey = custkeys[o_idx];
                    cnation = cust_nationkey_data[static_cast<size_t>(custkey)];
                }
                if (cnation < 0 ||
                    !nation_in_region[static_cast<size_t>(cnation)]) {
                    continue;
                }
#ifdef TRACE
                orders_emitted += 1;
#endif

                const auto range = order_ranges[o_idx];
                if (range.end == 0) {
                    continue;
                }

                if (o_idx + 1 < end_idx) {
                    const auto next_range = order_ranges[o_idx + 1];
                    if (next_range.end != 0) {
                        if (use_lineitem_supp_nation) {
                            __builtin_prefetch(lineitem_supp_nation + next_range.start, 0,
                                               1);
                        } else {
                            __builtin_prefetch(lineitem_suppkey + next_range.start, 0, 1);
                        }
                        __builtin_prefetch(lineitem_discounted_price + next_range.start, 0,
                                           1);
                    }
                }

                const uint32_t start = range.start;
                const uint32_t end = range.end;
#ifdef TRACE
                const uint64_t lineitem_start_ns = trace_utils::get_time_ns();
#endif
                int64_t* revenue_bucket =
                    &revenue_data[static_cast<size_t>(cnation)];
                const int32_t* price_ptr = lineitem_discounted_price + start;
                int64_t revenue_local = 0;
                if (use_lineitem_supp_nation) {
                    const int16_t* nation_ptr = lineitem_supp_nation + start;
                    const int16_t* nation_end = lineitem_supp_nation + end;
                    for (; nation_ptr != nation_end; ++nation_ptr, ++price_ptr) {
#ifdef TRACE
                        lineitems_scanned += 1;
#endif
                        if (*nation_ptr != cnation) {
                            continue;
                        }
                        revenue_local += static_cast<int64_t>(*price_ptr);
#ifdef TRACE
                        lineitems_emitted += 1;
#endif
                    }
                } else {
                    const int32_t* suppkey_ptr = lineitem_suppkey + start;
                    const int32_t* suppkey_end = lineitem_suppkey + end;
                    for (; suppkey_ptr != suppkey_end;
                         ++suppkey_ptr, ++price_ptr) {
#ifdef TRACE
                        lineitems_scanned += 1;
#endif
                        const int32_t suppkey = *suppkey_ptr;
                        const int16_t snation = supp_nationkey_data[suppkey];
                        if (snation != cnation) {
                            continue;
                        }
                        revenue_local += static_cast<int64_t>(*price_ptr);
#ifdef TRACE
                        lineitems_emitted += 1;
#endif
                    }
                }
                *revenue_bucket += revenue_local;
#ifdef TRACE
                q5_trace::record_timing("q5_lineitem_scan",
                                        trace_utils::get_time_ns() - lineitem_start_ns);
#endif
            }
        } else {
            const int32_t* __restrict lineitem_orderkey = lineitem.orderkey.data();
            for (uint32_t o_idx = start_idx; o_idx < end_idx; ++o_idx) {
                int16_t cnation = -1;
                if (use_order_cust_nation) {
                    cnation = order_cust_nation[o_idx];
                } else {
                    const int32_t custkey = custkeys[o_idx];
                    cnation = cust_nationkey_data[static_cast<size_t>(custkey)];
                }
                if (cnation < 0 ||
                    !nation_in_region[static_cast<size_t>(cnation)]) {
                    continue;
                }
#ifdef TRACE
                orders_emitted += 1;
#endif

                const int32_t orderkey = orderkeys[o_idx];
                const auto range = order_ranges[o_idx];
                if (range.end == 0) {
                    continue;
                }

                if (o_idx + 1 < end_idx) {
                    const auto next_range = order_ranges[o_idx + 1];
                    if (next_range.end != 0) {
                        if (use_lineitem_supp_nation) {
                            __builtin_prefetch(lineitem_supp_nation + next_range.start, 0,
                                               1);
                        } else {
                            __builtin_prefetch(lineitem_suppkey + next_range.start, 0, 1);
                        }
                        __builtin_prefetch(lineitem_discounted_price + next_range.start, 0,
                                           1);
                    }
                }

                const uint32_t start = range.start;
                const uint32_t end = range.end;
#ifdef TRACE
                const uint64_t lineitem_start_ns = trace_utils::get_time_ns();
#endif
                int64_t* revenue_bucket = &revenue_data[static_cast<size_t>(cnation)];
                const int32_t* price_ptr = lineitem_discounted_price + start;
                const int32_t* orderkey_ptr = lineitem_orderkey + start;
                int64_t revenue_local = 0;
                if (use_lineitem_supp_nation) {
                    const int16_t* nation_ptr = lineitem_supp_nation + start;
                    const int16_t* nation_end = lineitem_supp_nation + end;
                    for (; nation_ptr != nation_end;
                         ++nation_ptr, ++price_ptr, ++orderkey_ptr) {
#ifdef TRACE
                        lineitems_scanned += 1;
#endif
                        if (*orderkey_ptr != orderkey) {
                            continue;
                        }
                        if (*nation_ptr != cnation) {
                            continue;
                        }
                        revenue_local += static_cast<int64_t>(*price_ptr);
#ifdef TRACE
                        lineitems_emitted += 1;
#endif
                    }
                } else {
                    const int32_t* suppkey_ptr = lineitem_suppkey + start;
                    const int32_t* suppkey_end = lineitem_suppkey + end;
                    for (; suppkey_ptr != suppkey_end;
                         ++suppkey_ptr, ++price_ptr, ++orderkey_ptr) {
#ifdef TRACE
                        lineitems_scanned += 1;
#endif
                        if (*orderkey_ptr != orderkey) {
                            continue;
                        }
                        const int32_t suppkey = *suppkey_ptr;
                        const int16_t snation = supp_nationkey_data[suppkey];
                        if (snation != cnation) {
                            continue;
                        }
                        revenue_local += static_cast<int64_t>(*price_ptr);
#ifdef TRACE
                        lineitems_emitted += 1;
#endif
                    }
                }
                *revenue_bucket += revenue_local;
#ifdef TRACE
                q5_trace::record_timing("q5_lineitem_scan",
                                        trace_utils::get_time_ns() - lineitem_start_ns);
#endif
            }
        }
    }
    TRACE_SET(orders_rows_emitted, orders_emitted);
    TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
    TRACE_SET(lineitem_rows_emitted, lineitems_emitted);
    TRACE_SET(join_build_rows_in, orders_emitted);
    TRACE_SET(join_probe_rows_in, lineitems_scanned);
    TRACE_SET(join_rows_emitted, lineitems_emitted);
    TRACE_SET(agg_rows_in, lineitems_emitted);

    {
        PROFILE_SCOPE("q5_agg_finalize");
        results.reserve(nation.rows.size());
        uint64_t groups_created = 0;
        for (size_t idx = 0; idx < revenue_by_nation.size(); ++idx) {
            if (!nation_in_region[idx]) {
                continue;
            }
            const int64_t revenue = revenue_by_nation[idx];
            if (revenue == 0) {
                continue;
            }
            Q5ResultRow row;
            row.n_name = std::string(nation_name[idx]);
            row.revenue_raw = revenue;
            results.push_back(std::move(row));
            groups_created += 1;
        }
        TRACE_SET(groups_created, groups_created);
        TRACE_SET(agg_rows_emitted, groups_created);
    }

    {
        PROFILE_SCOPE("q5_sort");
        TRACE_SET(sort_rows_in, results.size());
        std::sort(results.begin(), results.end(),
                  [](const Q5ResultRow& a, const Q5ResultRow& b) {
                      if (a.revenue_raw != b.revenue_raw) {
                          return a.revenue_raw > b.revenue_raw;
                      }
                      return a.n_name < b.n_name;
                  });
        TRACE_SET(sort_rows_out, results.size());
    }
    TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q5_trace::emit();
#endif

    return results;
}

void write_q5_csv(const std::string& filename, const std::vector<Q5ResultRow>& rows) {
    std::ofstream out(filename);
    out << "n_name,revenue\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double revenue =
            static_cast<double>(row.revenue_raw) / static_cast<double>(q5::kRevenueScale);
        out << q5::escape_csv_field_q5(row.n_name) << ',' << revenue << "\n";
    }
}