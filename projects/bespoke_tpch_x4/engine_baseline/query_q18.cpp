#include "query_q18.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q18 {
constexpr int32_t kPriceScale = 100;

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

int32_t parse_scaled_value(const std::string& value, int32_t scale) {
    const double numeric = std::stod(value);
    return static_cast<int32_t>(std::llround(numeric * static_cast<double>(scale)));
}

std::string_view get_string_view(const StringColumn& column, size_t row) {
    const uint32_t start = column.offsets[row];
    const uint32_t end = column.offsets[row + 1];
    return std::string_view(column.data.data() + start, end - start);
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

template <typename T>
int32_t max_value(const std::vector<T>& values) {
    if (values.empty()) {
        return -1;
    }
    return static_cast<int32_t>(*std::max_element(values.begin(), values.end()));
}
}  // namespace q18

#ifdef TRACE
namespace q18_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t customer_scan_ns = 0;
    uint64_t orders_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
    uint64_t customer_rows_scanned = 0;
    uint64_t customer_rows_emitted = 0;
    uint64_t orders_rows_scanned = 0;
    uint64_t orders_rows_emitted = 0;
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
    if (std::strcmp(name, "q18_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q18_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q18_customer_scan") == 0) {
        trace.customer_scan_ns += ns;
    } else if (std::strcmp(name, "q18_orders_scan") == 0) {
        trace.orders_scan_ns += ns;
    } else if (std::strcmp(name, "q18_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q18_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q18_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q18_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q18_customer_scan " << trace.customer_scan_ns << "\n";
    std::cout << "PROFILE q18_orders_scan " << trace.orders_scan_ns << "\n";
    std::cout << "PROFILE q18_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q18_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q18_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q18_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q18_customer_scan_rows_scanned " << trace.customer_rows_scanned
              << "\n";
    std::cout << "COUNT q18_customer_scan_rows_emitted " << trace.customer_rows_emitted
              << "\n";
    std::cout << "COUNT q18_orders_scan_rows_scanned " << trace.orders_rows_scanned << "\n";
    std::cout << "COUNT q18_orders_scan_rows_emitted " << trace.orders_rows_emitted << "\n";
    std::cout << "COUNT q18_customer_orders_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q18_customer_orders_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q18_customer_orders_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q18_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q18_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q18_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q18_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q18_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q18_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q18_trace

#define PROFILE_SCOPE(name) \
    ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q18_trace::record_timing)
#define TRACE_SET(field, value) (q18_trace::data().field = (value))
#define TRACE_ADD(field, value) (q18_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q18ResultRow> run_q18(const Database& db, const Q18Args& args) {
    const auto& customer = db.customer;
    const auto& orders = db.orders;
    const auto& lineitem = db.lineitem;

#ifdef TRACE
    q18_trace::reset();
#endif
    std::vector<Q18ResultRow> results;
    {
        PROFILE_SCOPE("q18_total");
        if (args.QUANTITY == "<<NULL>>") {
#ifdef TRACE
            TRACE_SET(query_output_rows, 0);
            q18_trace::emit();
#endif
            return {};
        }

        const int32_t quantity_threshold =
            q18::parse_scaled_value(args.QUANTITY, q18::kPriceScale);

        const bool dense_orderkeys =
            !orders.orderkey.empty() && orders.orderkey.front() == 1 &&
            orders.orderkey.back() == static_cast<int32_t>(orders.row_count);
        const bool dense_custkeys =
            !customer.custkey.empty() && customer.custkey.front() == 1 &&
            customer.custkey.back() == static_cast<int32_t>(customer.row_count);

        if (dense_orderkeys) {
            if (orders.row_count == 0 || customer.row_count == 0) {
#ifdef TRACE
                TRACE_SET(query_output_rows, 0);
                q18_trace::emit();
#endif
                return {};
            }

            const size_t sum_size = orders.row_count + 1;
            std::vector<int32_t> sum_qty(sum_size, 0);
            std::vector<uint8_t> qualified(sum_size, 0);
            std::vector<int32_t> qualifying_orderkeys;
            qualifying_orderkeys.reserve(1024);
            {
                PROFILE_SCOPE("q18_lineitem_scan");
                const int32_t* __restrict orderkeys = lineitem.orderkey.data();
                const int16_t* __restrict prices_quantity = lineitem.quantity.data();
                int32_t* __restrict sum_qty_data = sum_qty.data();
                uint8_t* __restrict qualified_data = qualified.data();
                const size_t row_count = lineitem.row_count;
#ifdef TRACE
                uint64_t group_count = 0;
#endif
                for (size_t li_idx = 0; li_idx < row_count; ++li_idx) {
                    const size_t idx = static_cast<size_t>(orderkeys[li_idx]);
                    const int32_t prev = sum_qty_data[idx];
#ifdef TRACE
                    if (prev == 0) {
                        ++group_count;
                    }
#endif
                    const int32_t updated = prev + prices_quantity[li_idx];
                    sum_qty_data[idx] = updated;
                    if (updated > quantity_threshold && !qualified_data[idx]) {
                        qualified_data[idx] = 1;
                        qualifying_orderkeys.push_back(orderkeys[li_idx]);
                    }
                }
#ifdef TRACE
                TRACE_ADD(groups_created, group_count);
                TRACE_ADD(agg_rows_emitted, group_count);
#endif
            }
            TRACE_ADD(lineitem_rows_scanned, lineitem.row_count);
            TRACE_ADD(lineitem_rows_emitted, lineitem.row_count);
            TRACE_ADD(agg_rows_in, lineitem.row_count);

            std::vector<int32_t> custkey_to_row;
            if (!dense_custkeys) {
                const int32_t max_custkey = q18::max_value(customer.custkey);
                if (max_custkey < 0) {
#ifdef TRACE
                    TRACE_SET(query_output_rows, 0);
                    q18_trace::emit();
#endif
                    return {};
                }
                custkey_to_row.assign(static_cast<size_t>(max_custkey) + 1, -1);
                {
                    PROFILE_SCOPE("q18_customer_scan");
                    const int32_t* __restrict custkeys = customer.custkey.data();
                    int32_t* __restrict custkey_to_row_data = custkey_to_row.data();
                    const size_t row_count = customer.row_count;
                    for (size_t row = 0; row < row_count; ++row) {
                        custkey_to_row_data[static_cast<size_t>(custkeys[row])] =
                            static_cast<int32_t>(row);
                    }
                }
                TRACE_ADD(customer_rows_scanned, customer.row_count);
                TRACE_ADD(customer_rows_emitted, customer.row_count);
                TRACE_ADD(join_build_rows_in, customer.row_count);
            }

            results.reserve(1024);
            uint64_t orders_emitted = 0;
            uint64_t orders_probe = 0;
            const int32_t* __restrict custkeys = orders.custkey.data();
            const int16_t* __restrict orderdates = orders.orderdate.data();
            const int32_t* __restrict totalprices = orders.totalprice.data();
            for (const int32_t orderkey : qualifying_orderkeys) {
                const int64_t order_row_signed =
                    static_cast<int64_t>(orderkey) - 1;
                if (order_row_signed < 0 ||
                    static_cast<size_t>(order_row_signed) >= orders.row_count) {
                    continue;
                }
                const size_t order_row = static_cast<size_t>(order_row_signed);
                const int32_t custkey = custkeys[order_row];
                int32_t cust_row = -1;
                if (dense_custkeys) {
                    const int64_t cust_row_signed =
                        static_cast<int64_t>(custkey) - 1;
                    if (cust_row_signed < 0 ||
                        static_cast<size_t>(cust_row_signed) >= customer.row_count) {
                        continue;
                    }
                    cust_row = static_cast<int32_t>(cust_row_signed);
                } else {
                    cust_row = custkey_to_row[static_cast<size_t>(custkey)];
                    if (cust_row < 0) {
                        continue;
                    }
                }
                ++orders_probe;
                results.emplace_back();
                auto& row = results.back();
                row.c_name = std::string(
                    q18::get_string_view(customer.name, static_cast<size_t>(cust_row)));
                row.c_custkey = custkey;
                row.o_orderkey = orderkey;
                row.o_orderdate_offset = orderdates[order_row];
                row.o_totalprice_raw = totalprices[order_row];
                row.sum_qty_raw =
                    static_cast<int64_t>(sum_qty[static_cast<size_t>(orderkey)]);
                ++orders_emitted;
            }
            TRACE_ADD(orders_rows_scanned, qualifying_orderkeys.size());
            TRACE_ADD(orders_rows_emitted, orders_emitted);
            TRACE_ADD(join_probe_rows_in, orders_probe);
            TRACE_ADD(join_rows_emitted, orders_emitted);
        } else {
            const int32_t max_orderkey =
                std::max(q18::max_value(orders.orderkey),
                         q18::max_value(lineitem.orderkey));
            if (max_orderkey < 0) {
#ifdef TRACE
                TRACE_SET(query_output_rows, 0);
                q18_trace::emit();
#endif
                return {};
            }

            std::vector<int32_t> sum_qty(static_cast<size_t>(max_orderkey) + 1, 0);
            {
                PROFILE_SCOPE("q18_lineitem_scan");
                const int32_t* __restrict orderkeys = lineitem.orderkey.data();
                const int16_t* __restrict prices_quantity = lineitem.quantity.data();
                int32_t* __restrict sum_qty_data = sum_qty.data();
                const size_t row_count = lineitem.row_count;
                for (size_t li_idx = 0; li_idx < row_count; ++li_idx) {
                    const size_t idx = static_cast<size_t>(orderkeys[li_idx]);
                    sum_qty_data[idx] += prices_quantity[li_idx];
                }
            }
            TRACE_ADD(lineitem_rows_scanned, lineitem.row_count);
            TRACE_ADD(lineitem_rows_emitted, lineitem.row_count);
            TRACE_ADD(agg_rows_in, lineitem.row_count);

#ifdef TRACE
            {
                PROFILE_SCOPE("q18_agg_finalize");
                uint64_t group_count = 0;
                for (const auto value : sum_qty) {
                    if (value > 0) {
                        ++group_count;
                    }
                }
                TRACE_ADD(groups_created, group_count);
                TRACE_ADD(agg_rows_emitted, group_count);
            }
#endif

            const int32_t max_custkey = q18::max_value(customer.custkey);
            if (max_custkey < 0) {
#ifdef TRACE
                TRACE_SET(query_output_rows, 0);
                q18_trace::emit();
#endif
                return {};
            }

            std::vector<int32_t> custkey_to_row(
                static_cast<size_t>(max_custkey) + 1, -1);
            {
                PROFILE_SCOPE("q18_customer_scan");
                const int32_t* __restrict custkeys = customer.custkey.data();
                int32_t* __restrict custkey_to_row_data = custkey_to_row.data();
                const size_t row_count = customer.row_count;
                for (size_t row = 0; row < row_count; ++row) {
                    custkey_to_row_data[static_cast<size_t>(custkeys[row])] =
                        static_cast<int32_t>(row);
                }
            }
            TRACE_ADD(customer_rows_scanned, customer.row_count);
            TRACE_ADD(customer_rows_emitted, customer.row_count);
            TRACE_ADD(join_build_rows_in, customer.row_count);

            results.reserve(1024);
            uint64_t orders_emitted = 0;
            uint64_t orders_probe = 0;
            {
                PROFILE_SCOPE("q18_orders_scan");
                const int32_t* __restrict orderkeys = orders.orderkey.data();
                const int32_t* __restrict custkeys = orders.custkey.data();
                const int16_t* __restrict orderdates = orders.orderdate.data();
                const int32_t* __restrict totalprices = orders.totalprice.data();
                const size_t row_count = orders.row_count;
                for (size_t o_idx = 0; o_idx < row_count; ++o_idx) {
                    const int32_t orderkey = orderkeys[o_idx];
                    const int32_t total_qty =
                        sum_qty[static_cast<size_t>(orderkey)];
                    if (total_qty <= quantity_threshold) {
                        continue;
                    }
                    const int32_t custkey = custkeys[o_idx];
                    const int32_t cust_row =
                        custkey_to_row[static_cast<size_t>(custkey)];
                    if (cust_row < 0) {
                        continue;
                    }
                    ++orders_probe;
                    results.emplace_back();
                    auto& row = results.back();
                    row.c_name = std::string(
                        q18::get_string_view(customer.name, static_cast<size_t>(cust_row)));
                    row.c_custkey = custkey;
                    row.o_orderkey = orderkey;
                    row.o_orderdate_offset = orderdates[o_idx];
                    row.o_totalprice_raw = totalprices[o_idx];
                    row.sum_qty_raw = static_cast<int64_t>(total_qty);
                    ++orders_emitted;
                }
            }
            TRACE_ADD(orders_rows_scanned, orders.row_count);
            TRACE_ADD(orders_rows_emitted, orders_emitted);
            TRACE_ADD(join_probe_rows_in, orders_probe);
            TRACE_ADD(join_rows_emitted, orders_emitted);
        }

        {
            PROFILE_SCOPE("q18_sort");
            std::sort(results.begin(), results.end(),
                      [](const Q18ResultRow& a, const Q18ResultRow& b) {
                          if (a.o_totalprice_raw != b.o_totalprice_raw) {
                              return a.o_totalprice_raw > b.o_totalprice_raw;
                          }
                          if (a.o_orderdate_offset != b.o_orderdate_offset) {
                              return a.o_orderdate_offset < b.o_orderdate_offset;
                          }
                          return a.o_orderkey < b.o_orderkey;
                      });
        }
        TRACE_ADD(sort_rows_in, results.size());
        TRACE_ADD(sort_rows_out, results.size());
        TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q18_trace::emit();
#endif
    return results;
}

void write_q18_csv(const Database& db,
                   const std::string& filename,
                   const std::vector<Q18ResultRow>& rows) {
    std::ofstream out(filename);
    out << "c_name,c_custkey,o_orderkey,o_orderdate,o_totalprice,sum(l_quantity)\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double totalprice =
            static_cast<double>(row.o_totalprice_raw) / q18::kPriceScale;
        const double sum_qty =
            static_cast<double>(row.sum_qty_raw) / q18::kPriceScale;
        out << q18::escape_csv_field(row.c_name) << ','
            << row.c_custkey << ','
            << row.o_orderkey << ','
            << q18::format_date(db.base_date_days, row.o_orderdate_offset) << ','
            << totalprice << ','
            << sum_qty << "\n";
    }
}