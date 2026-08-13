#include "query_q7.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "trace_utils.hpp"

namespace q7 {
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

int32_t year_from_offset(int32_t base_days, int16_t offset) {
    const int32_t days = base_days + offset;
    int year = 0;
    unsigned month = 0;
    unsigned day = 0;
    civil_from_days(days, year, month, day);
    return year;
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

int32_t max_value(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return 0;
    }
    return *std::max_element(values.begin(), values.end());
}

}  // namespace q7

#ifdef TRACE
namespace q7_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t orderkey_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t orderkey_rows_scanned = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
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
    if (std::strcmp(name, "q7_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q7_orderkey_scan") == 0) {
        trace.orderkey_scan_ns += ns;
    } else if (std::strcmp(name, "q7_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q7_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q7_orderkey_scan " << trace.orderkey_scan_ns << "\n";
    std::cout << "PROFILE q7_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "COUNT q7_orderkey_scan_rows_scanned " << trace.orderkey_rows_scanned
              << "\n";
    std::cout << "COUNT q7_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q7_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q7_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q7_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q7_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q7_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q7_trace

#define TRACE_SET(field, value) (q7_trace::data().field = (value))
#define TRACE_ADD(field, value) (q7_trace::data().field += (value))
#else
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q7ResultRow> run_q7(const Database& db, const Q7Args& args) {
#ifdef TRACE
    q7_trace::reset();
#endif
#ifdef TRACE
    const uint64_t total_start = trace_utils::get_time_ns();
#endif
    const auto nation1_it = db.nation.name_to_key.find(args.NATION1);
    const auto nation2_it = db.nation.name_to_key.find(args.NATION2);
    if (nation1_it == db.nation.name_to_key.end() ||
        nation2_it == db.nation.name_to_key.end()) {
        return {};
    }
    const int32_t nation1 = nation1_it->second;
    const int32_t nation2 = nation2_it->second;

    const int16_t start_offset =
        q7::parse_date_offset("1995-01-01", db.base_date_days);
    const int16_t end_offset =
        q7::parse_date_offset("1996-12-31", db.base_date_days);
    const int16_t year_1996_offset =
        q7::parse_date_offset("1996-01-01", db.base_date_days);

    const auto& orders = db.orders;
    const auto& supplier = db.supplier;
    const auto& lineitem = db.lineitem;

    const bool same_nation = nation1 == nation2;

    const int32_t max_suppkey = q7::max_value(supplier.suppkey);
    std::vector<int8_t> supp_tag(static_cast<size_t>(max_suppkey) + 1, -1);
    for (uint32_t i = 0; i < supplier.row_count; ++i) {
        const int32_t key = supplier.suppkey[i];
        if (key < 0) {
            continue;
        }
        const int32_t nation = supplier.nationkey[i];
        if (nation == nation1) {
            supp_tag[static_cast<size_t>(key)] = 0;
        } else if (!same_nation && nation == nation2) {
            supp_tag[static_cast<size_t>(key)] = 1;
        }
    }

    std::array<std::array<std::array<int64_t, 2>, 2>, 2> revenue{};

    uint64_t lineitems_scanned = 0;
    uint64_t lineitems_emitted = 0;
    uint64_t orderkeys_scanned = 0;

    const bool use_order_ranges =
        lineitem.orderkey_sorted &&
        orders.lineitem_ranges.size() == orders.row_count;

    if (use_order_ranges) {
        struct ScanRange {
            uint32_t start = 0;
            uint32_t end = 0;
            int8_t cust_idx = -1;
        };
        std::vector<ScanRange> ranges;
        {
#ifdef TRACE
            trace_utils::ScopedTimer order_timer("q7_orderkey_scan",
                                                 &q7_trace::record_timing);
#endif
            const auto* __restrict order_cust_nation = orders.cust_nationkey.data();
            const auto* __restrict lineitem_ranges = orders.lineitem_ranges.data();
            ranges.reserve(orders.row_count / 10);
            for (uint32_t row = 0; row < orders.row_count; ++row) {
                const int16_t cust_nation = order_cust_nation[row];
                int8_t cust_idx = -1;
                if (cust_nation == nation1) {
                    cust_idx = 0;
                } else if (!same_nation && cust_nation == nation2) {
                    cust_idx = 1;
                } else {
                    continue;
                }
                const auto range = lineitem_ranges[row];
                if (range.end <= range.start) {
                    continue;
                }
                ranges.push_back(ScanRange{range.start, range.end, cust_idx});
            }

            if (!ranges.empty()) {
                constexpr uint32_t kRadixBits = 16;
                constexpr uint32_t kRadix = 1u << kRadixBits;
                std::vector<uint32_t> offsets(kRadix + 1, 0);
                std::vector<ScanRange> temp(ranges.size());
                for (const auto& range : ranges) {
                    const uint32_t bucket = range.start >> kRadixBits;
                    offsets[bucket + 1] += 1;
                }
                for (uint32_t i = 1; i < offsets.size(); ++i) {
                    offsets[i] += offsets[i - 1];
                }
                for (const auto& range : ranges) {
                    const uint32_t bucket = range.start >> kRadixBits;
                    const uint32_t pos = offsets[bucket]++;
                    temp[pos] = range;
                }
                ranges.swap(temp);
            }
        }

        orderkeys_scanned = ranges.size();
        const int16_t* __restrict shipdate = lineitem.shipdate.data();
        const int32_t* __restrict suppkey = lineitem.suppkey.data();
        const int32_t* __restrict discounted_price = lineitem.discounted_price.data();
        const auto* __restrict lineitem_ranges = orders.lineitem_ranges.data();
        const size_t supp_tag_size = supp_tag.size();
        const int32_t ship_span = static_cast<int32_t>(end_offset - start_offset);
        {
#ifdef TRACE
            trace_utils::ScopedTimer scan_timer("q7_lineitem_scan",
                                                &q7_trace::record_timing);
#endif
            for (const auto& range : ranges) {
                const int8_t cust_idx = range.cust_idx;
#ifdef TRACE
                lineitems_scanned += static_cast<uint64_t>(range.end - range.start);
#endif
                const int16_t* ship_ptr = shipdate + range.start;
                const int16_t* ship_end = shipdate + range.end;
                const int32_t* supp_ptr = suppkey + range.start;
                const int32_t* price_ptr = discounted_price + range.start;
                for (; ship_ptr != ship_end; ++ship_ptr, ++supp_ptr, ++price_ptr) {
                    const int32_t ship = static_cast<int32_t>(*ship_ptr);
                    if (static_cast<uint32_t>(ship - start_offset) >
                        static_cast<uint32_t>(ship_span)) {
                        continue;
                    }
                    const int32_t skey = *supp_ptr;
                    const int8_t supp_idx = supp_tag[static_cast<size_t>(skey)];
                    if (supp_idx < 0) {
                        continue;
                    }
                    if (same_nation) {
                        if (supp_idx != 0 || cust_idx != 0) {
                            continue;
                        }
                    } else if (supp_idx == cust_idx) {
                        continue;
                    }
                    const int32_t year_idx = ship >= year_1996_offset;
                    revenue[static_cast<size_t>(supp_idx)]
                           [static_cast<size_t>(cust_idx)]
                           [static_cast<size_t>(year_idx)] +=
                        static_cast<int64_t>(*price_ptr);
#ifdef TRACE
                    ++lineitems_emitted;
#endif
                }
            }
        }
    } else {
        const int16_t* __restrict shipdate = lineitem.shipdate.data();
        const int32_t* __restrict suppkey = lineitem.suppkey.data();
        const int32_t* __restrict orderkey = lineitem.orderkey.data();
        const int32_t* __restrict discounted_price = lineitem.discounted_price.data();
        const auto* __restrict orderkey_to_row = orders.orderkey_to_row.data();
        const auto* __restrict order_cust_nation = orders.cust_nationkey.data();
        const size_t orderkey_size = orders.orderkey_to_row.size();
        const size_t orders_size = orders.cust_nationkey.size();
        const size_t supp_tag_size = supp_tag.size();
        const int32_t ship_span = static_cast<int32_t>(end_offset - start_offset);
#ifdef TRACE
        trace_utils::ScopedTimer scan_timer("q7_lineitem_scan",
                                            &q7_trace::record_timing);
#endif
        for (uint32_t li_idx = 0; li_idx < lineitem.row_count; ++li_idx) {
#ifdef TRACE
            ++lineitems_scanned;
#endif
            const int32_t ship = static_cast<int32_t>(shipdate[li_idx]);
            if (static_cast<uint32_t>(ship - start_offset) >
                static_cast<uint32_t>(ship_span)) {
                continue;
            }
            const int32_t skey = suppkey[li_idx];
            const int8_t supp_idx = supp_tag[static_cast<size_t>(skey)];
            if (supp_idx < 0) {
                continue;
            }
            const int32_t okey = orderkey[li_idx];
            if (okey < 0 || static_cast<size_t>(okey) >= orderkey_size) {
                continue;
            }
            const int32_t row = orderkey_to_row[static_cast<size_t>(okey)];
            if (row < 0 || static_cast<size_t>(row) >= orders_size) {
                continue;
            }
            const int16_t cust_nation = order_cust_nation[row];
            int8_t cust_idx = -1;
            if (cust_nation == nation1) {
                cust_idx = 0;
            } else if (!same_nation && cust_nation == nation2) {
                cust_idx = 1;
            } else {
                continue;
            }
            if (same_nation) {
                if (supp_idx != 0 || cust_idx != 0) {
                    continue;
                }
            } else if (supp_idx == cust_idx) {
                continue;
            }
            const int32_t year_idx = ship >= year_1996_offset;
            revenue[static_cast<size_t>(supp_idx)][static_cast<size_t>(cust_idx)]
                   [static_cast<size_t>(year_idx)] +=
                static_cast<int64_t>(discounted_price[li_idx]);
#ifdef TRACE
            ++lineitems_emitted;
#endif
        }
    }

    std::vector<Q7ResultRow> results;
    const std::array<int32_t, 2> nations = {nation1, nation2};
    const int32_t nation_count = same_nation ? 1 : 2;
    for (int32_t supp_idx = 0; supp_idx < nation_count; ++supp_idx) {
        for (int32_t cust_idx = 0; cust_idx < nation_count; ++cust_idx) {
            if (same_nation) {
                if (supp_idx != 0 || cust_idx != 0) {
                    continue;
                }
            } else if (supp_idx == cust_idx) {
                continue;
            }
            const int32_t supp_key = nations[static_cast<size_t>(supp_idx)];
            const int32_t cust_key = nations[static_cast<size_t>(cust_idx)];
            const auto supp_row = db.nation.nationkey_to_row.find(supp_key);
            const auto cust_row = db.nation.nationkey_to_row.find(cust_key);
            if (supp_row == db.nation.nationkey_to_row.end() ||
                cust_row == db.nation.nationkey_to_row.end()) {
                continue;
            }
            for (int32_t year_idx = 0; year_idx < 2; ++year_idx) {
                if (revenue[static_cast<size_t>(supp_idx)]
                           [static_cast<size_t>(cust_idx)]
                           [static_cast<size_t>(year_idx)] == 0) {
                    continue;
                }
                Q7ResultRow row;
                row.supp_nation = db.nation.rows[supp_row->second].name;
                row.cust_nation = db.nation.rows[cust_row->second].name;
                row.l_year = year_idx == 0 ? 1995 : 1996;
                row.revenue_raw = revenue[static_cast<size_t>(supp_idx)]
                                          [static_cast<size_t>(cust_idx)]
                                          [static_cast<size_t>(year_idx)];
                results.push_back(std::move(row));
            }
        }
    }

    TRACE_SET(agg_rows_in, lineitems_emitted);
    TRACE_SET(groups_created, results.size());
    TRACE_SET(agg_rows_emitted, results.size());
    TRACE_SET(query_output_rows, results.size());
    TRACE_SET(orderkey_rows_scanned, orderkeys_scanned);
    TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
    TRACE_SET(lineitem_rows_emitted, lineitems_emitted);

    std::sort(results.begin(), results.end(),
              [](const Q7ResultRow& a, const Q7ResultRow& b) {
                  if (a.supp_nation != b.supp_nation) {
                      return a.supp_nation < b.supp_nation;
                  }
                  if (a.cust_nation != b.cust_nation) {
                      return a.cust_nation < b.cust_nation;
                  }
                  return a.l_year < b.l_year;
              });

#ifdef TRACE
    q7_trace::record_timing("q7_total",
                            trace_utils::get_time_ns() - total_start);
    q7_trace::emit();
#endif
    return results;
}

void write_q7_csv(const std::string& filename, const std::vector<Q7ResultRow>& rows) {
    std::ofstream out(filename);
    out << "supp_nation,cust_nation,l_year,revenue\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double revenue = static_cast<double>(row.revenue_raw) / q7::kPriceScale;
        out << q7::escape_csv_field(row.supp_nation) << ','
            << q7::escape_csv_field(row.cust_nation) << ',' << row.l_year << ','
            << revenue << "\n";
    }
}