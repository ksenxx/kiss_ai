#include "query_q9.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q9 {
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

int32_t year_from_offset(int32_t base_days, int16_t offset) {
    const int32_t days = base_days + offset;
    int year = 0;
    unsigned month = 0;
    unsigned day = 0;
    civil_from_days(days, year, month, day);
    return year;
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
        return 0;
    }
    return static_cast<int32_t>(*std::max_element(values.begin(), values.end()));
}

}  // namespace q9

#ifdef TRACE
namespace q9_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t partsupp_build_ns = 0;
    uint64_t orders_build_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
};

inline TraceData& data() {
    static TraceData trace;
    return trace;
}

inline void reset() { data() = TraceData{}; }

inline void record_timing(const char* name, uint64_t ns) {
    auto& trace = data();
    if (std::strcmp(name, "q9_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q9_partsupp_build") == 0) {
        trace.partsupp_build_ns += ns;
    } else if (std::strcmp(name, "q9_orders_build") == 0) {
        trace.orders_build_ns += ns;
    } else if (std::strcmp(name, "q9_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q9_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q9_partsupp_build " << trace.partsupp_build_ns << "\n";
    std::cout << "PROFILE q9_orders_build " << trace.orders_build_ns << "\n";
    std::cout << "PROFILE q9_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "COUNT q9_lineitem_scan_rows_scanned "
              << trace.lineitem_rows_scanned << "\n";
    std::cout << "COUNT q9_lineitem_scan_rows_emitted "
              << trace.lineitem_rows_emitted << "\n";
}
}  // namespace q9_trace
#define TRACE_TIMER(name) \
    trace_utils::ScopedTimer timer_##name(#name, q9_trace::record_timing)
#define TRACE_ADD(field, value) (q9_trace::data().field += (value))
#else
#define TRACE_TIMER(name)
#define TRACE_ADD(field, value)
#endif

std::vector<Q9ResultRow> run_q9(const Database& db, const Q9Args& args) {
#ifdef TRACE
    const auto total_start = trace_utils::get_time_ns();
    q9_trace::reset();
#endif
    const auto& part = db.part;
    const auto& partsupp = db.partsupp;
    const auto& supplier = db.supplier;
    const auto& lineitem = db.lineitem;
    const auto& orders = db.orders;

    const std::string_view color = args.COLOR;

    const int32_t max_partkey =
        std::max(q9::max_value(part.partkey), q9::max_value(partsupp.partkey));
    std::vector<uint8_t> part_ok_by_key(static_cast<size_t>(max_partkey) + 1, 0);
    for (uint32_t i = 0; i < part.row_count; ++i) {
        const int32_t partkey = part.partkey[i];
        if (partkey < 0 || static_cast<size_t>(partkey) >= part_ok_by_key.size()) {
            continue;
        }
        const std::string_view name = q9::get_string_view(part.name, i);
        if (name.find(color) != std::string_view::npos) {
            part_ok_by_key[static_cast<size_t>(partkey)] = 1;
        }
    }

    int32_t max_nationkey = 0;
    for (const auto& row : db.nation.rows) {
        max_nationkey = std::max(max_nationkey, row.nationkey);
    }
    std::vector<std::string_view> nation_name(static_cast<size_t>(max_nationkey) + 1);
    for (const auto& row : db.nation.rows) {
        if (row.nationkey >= 0) {
            nation_name[static_cast<size_t>(row.nationkey)] = row.name;
        }
    }

    std::vector<uint32_t> part_counts(static_cast<size_t>(max_partkey) + 1, 0);
    uint32_t max_supp_per_part = 0;
    {
        TRACE_TIMER(q9_partsupp_build);
        for (uint32_t i = 0; i < partsupp.row_count; ++i) {
            const int32_t partkey = partsupp.partkey[i];
            if (partkey < 0 ||
                static_cast<size_t>(partkey) >= part_counts.size()) {
                continue;
            }
            const uint32_t count = ++part_counts[static_cast<size_t>(partkey)];
            max_supp_per_part = std::max(max_supp_per_part, count);
        }
    }

    const size_t part_slots = static_cast<size_t>(max_partkey) + 1;
    const size_t slots_per_part =
        std::max<size_t>(1, static_cast<size_t>(max_supp_per_part));
    const uint64_t kInvalidSupp = (static_cast<uint64_t>(0xFFFFFFFFu) << 32);
    std::vector<uint64_t> part_suppcost(part_slots * slots_per_part, kInvalidSupp);
    std::vector<uint32_t> part_write_pos(part_counts.size(), 0);
    {
        TRACE_TIMER(q9_partsupp_build);
        for (uint32_t i = 0; i < partsupp.row_count; ++i) {
            const int32_t partkey = partsupp.partkey[i];
            if (partkey < 0 ||
                static_cast<size_t>(partkey) >= part_counts.size()) {
                continue;
            }
            const uint32_t slot = part_write_pos[static_cast<size_t>(partkey)]++;
            const size_t base =
                static_cast<size_t>(partkey) * slots_per_part + slot;
            part_suppcost[base] =
                (static_cast<uint64_t>(static_cast<uint32_t>(partsupp.suppkey[i])) << 32) |
                static_cast<uint32_t>(partsupp.supplycost[i]);
        }
    }


    const int32_t max_orderkey = q9::max_value(orders.orderkey);
    std::vector<uint8_t> order_year_by_key(
        static_cast<size_t>(max_orderkey) + 1,
        std::numeric_limits<uint8_t>::max());
    int32_t min_year = std::numeric_limits<int32_t>::max();
    int32_t max_year = std::numeric_limits<int32_t>::min();
    if (orders.row_count > 0) {
        TRACE_TIMER(q9_orders_build);
        int16_t min_orderdate = std::numeric_limits<int16_t>::max();
        int16_t max_orderdate = std::numeric_limits<int16_t>::min();
        for (uint32_t i = 0; i < orders.row_count; ++i) {
            const int16_t orderdate = orders.orderdate[i];
            min_orderdate = std::min(min_orderdate, orderdate);
            max_orderdate = std::max(max_orderdate, orderdate);
        }
        const size_t date_span =
            static_cast<size_t>(max_orderdate - min_orderdate) + 1;
        std::vector<int16_t> year_by_orderdate(date_span, 0);
        for (int32_t offset = min_orderdate; offset <= max_orderdate; ++offset) {
            const int32_t year = q9::year_from_offset(db.base_date_days,
                                                      static_cast<int16_t>(offset));
            year_by_orderdate[static_cast<size_t>(offset - min_orderdate)] =
                static_cast<int16_t>(year);
        }

        min_year = year_by_orderdate.front();
        max_year = year_by_orderdate.back();

        for (uint32_t i = 0; i < orders.row_count; ++i) {
            const int32_t orderkey = orders.orderkey[i];
            if (orderkey < 0 ||
                static_cast<size_t>(orderkey) >= order_year_by_key.size()) {
                continue;
            }
            const int16_t orderdate = orders.orderdate[i];
            const int16_t year_value =
                year_by_orderdate[static_cast<size_t>(orderdate - min_orderdate)];
            order_year_by_key[static_cast<size_t>(orderkey)] =
                static_cast<uint8_t>(year_value - min_year);
        }
    }
    if (min_year > max_year) {
        return {};
    }

    const int32_t year_span = max_year - min_year + 1;
    const size_t group_span = static_cast<size_t>(max_nationkey + 1) *
                              static_cast<size_t>(year_span);
    std::vector<int64_t> totals(group_span, 0);
    std::vector<uint8_t> totals_used(group_span, 0);

    {
        TRACE_TIMER(q9_lineitem_scan);
        const uint8_t* __restrict part_ok_data = part_ok_by_key.data();
        const uint64_t* __restrict part_suppcost_data = part_suppcost.data();
        const uint32_t* __restrict part_counts_data = part_counts.data();
        const uint8_t* __restrict order_year_data = order_year_by_key.data();
        int64_t* __restrict totals_data = totals.data();
        uint8_t* __restrict totals_used_data = totals_used.data();
        const uint32_t lineitem_count = lineitem.row_count;
        const size_t slots_per_part_local = slots_per_part;
        const int32_t min_year_local = min_year;
        const int32_t year_span_local = year_span;
        const bool fixed_four = (slots_per_part_local == 4);

        const int32_t* __restrict lineitem_partkey = lineitem.partkey.data();
        const int32_t* __restrict lineitem_suppkey = lineitem.suppkey.data();
        const int32_t* __restrict lineitem_orderkey = lineitem.orderkey.data();
        const int32_t* __restrict lineitem_discounted_price =
            lineitem.discounted_price.data();
        const int16_t* __restrict lineitem_quantity = lineitem.quantity.data();
        const int16_t* __restrict lineitem_supp_nationkey =
            lineitem.supp_nationkey.data();

        if (fixed_four) {
            for (uint32_t li_idx = 0; li_idx < lineitem_count; ++li_idx) {
                TRACE_ADD(lineitem_rows_scanned, 1);
                const int32_t partkey = *lineitem_partkey++;
                if (__builtin_expect(
                        part_ok_data[static_cast<size_t>(partkey)] == 0, 1)) {
                    ++lineitem_suppkey;
                    ++lineitem_orderkey;
                    ++lineitem_quantity;
                    ++lineitem_discounted_price;
                    ++lineitem_supp_nationkey;
                    continue;
                }

                const int32_t suppkey = *lineitem_suppkey++;
                const int32_t orderkey = *lineitem_orderkey++;
                const int32_t quantity = *lineitem_quantity++;
                const int32_t revenue = *lineitem_discounted_price++;
                const int16_t nationkey_value = *lineitem_supp_nationkey++;

                const size_t base = static_cast<size_t>(partkey) * 4;
                int32_t supplycost = 0;
                bool matched = false;
                const uint64_t e0 = part_suppcost_data[base];
                if (static_cast<int32_t>(e0 >> 32) == suppkey) {
                    supplycost = static_cast<int32_t>(e0);
                    matched = true;
                } else {
                    const uint64_t e1 = part_suppcost_data[base + 1];
                    if (static_cast<int32_t>(e1 >> 32) == suppkey) {
                        supplycost = static_cast<int32_t>(e1);
                        matched = true;
                    } else {
                        const uint64_t e2 = part_suppcost_data[base + 2];
                        if (static_cast<int32_t>(e2 >> 32) == suppkey) {
                            supplycost = static_cast<int32_t>(e2);
                            matched = true;
                        } else {
                            const uint64_t e3 = part_suppcost_data[base + 3];
                            if (static_cast<int32_t>(e3 >> 32) == suppkey) {
                                supplycost = static_cast<int32_t>(e3);
                                matched = true;
                            }
                        }
                    }
                }

                if (!matched) {
                    continue;
                }

                const int32_t nationkey = static_cast<int32_t>(nationkey_value);
                if (nationkey < 0) {
                    continue;
                }

                const uint8_t year_offset =
                    order_year_data[static_cast<size_t>(orderkey)];

                const int64_t supply_cost =
                    (static_cast<int64_t>(supplycost) *
                     static_cast<int64_t>(quantity)) /
                    q9::kPriceScale;

                const size_t group_index =
                    static_cast<size_t>(nationkey) *
                        static_cast<size_t>(year_span_local) +
                    static_cast<size_t>(year_offset);
                totals_data[group_index] += revenue - supply_cost;
                totals_used_data[group_index] = 1;
                TRACE_ADD(lineitem_rows_emitted, 1);
            }
        } else {
            for (uint32_t li_idx = 0; li_idx < lineitem_count; ++li_idx) {
                TRACE_ADD(lineitem_rows_scanned, 1);
                const int32_t partkey = *lineitem_partkey++;
                if (__builtin_expect(
                        part_ok_data[static_cast<size_t>(partkey)] == 0, 1)) {
                    ++lineitem_suppkey;
                    ++lineitem_orderkey;
                    ++lineitem_quantity;
                    ++lineitem_discounted_price;
                    ++lineitem_supp_nationkey;
                    continue;
                }

                const int32_t suppkey = *lineitem_suppkey++;
                const int32_t orderkey = *lineitem_orderkey++;
                const int32_t quantity = *lineitem_quantity++;
                const int32_t revenue = *lineitem_discounted_price++;
                const int16_t nationkey_value = *lineitem_supp_nationkey++;

                int32_t supplycost = 0;
                bool matched = false;
                const size_t base =
                    static_cast<size_t>(partkey) * slots_per_part_local;
                const uint32_t count =
                    part_counts_data[static_cast<size_t>(partkey)];
                for (uint32_t idx = 0; idx < count; ++idx) {
                    const size_t pos = base + idx;
                    const uint64_t entry = part_suppcost_data[pos];
                    if (static_cast<int32_t>(entry >> 32) == suppkey) {
                        supplycost = static_cast<int32_t>(entry);
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    continue;
                }

                const int32_t nationkey = static_cast<int32_t>(nationkey_value);
                if (nationkey < 0) {
                    continue;
                }

                const uint8_t year_offset =
                    order_year_data[static_cast<size_t>(orderkey)];

                const int64_t supply_cost =
                    (static_cast<int64_t>(supplycost) *
                     static_cast<int64_t>(quantity)) /
                    q9::kPriceScale;

                const size_t group_index =
                    static_cast<size_t>(nationkey) *
                        static_cast<size_t>(year_span_local) +
                    static_cast<size_t>(year_offset);
                totals_data[group_index] += revenue - supply_cost;
                totals_used_data[group_index] = 1;
                TRACE_ADD(lineitem_rows_emitted, 1);
            }
        }
    }

    std::vector<Q9ResultRow> results;
    results.reserve(group_span);
    for (int32_t nationkey = 0; nationkey <= max_nationkey; ++nationkey) {
        if (static_cast<size_t>(nationkey) >= nation_name.size() ||
            nation_name[static_cast<size_t>(nationkey)].empty()) {
            continue;
        }
        for (int32_t year_offset = 0; year_offset < year_span; ++year_offset) {
            const size_t idx = static_cast<size_t>(nationkey) *
                                   static_cast<size_t>(year_span) +
                               static_cast<size_t>(year_offset);
            if (!totals_used[idx]) {
                continue;
            }
            Q9ResultRow row;
            row.nation = std::string(nation_name[static_cast<size_t>(nationkey)]);
            row.o_year = min_year + year_offset;
            row.sum_profit_raw = totals[idx];
            results.push_back(std::move(row));
        }
    }

    std::sort(results.begin(), results.end(),
              [](const Q9ResultRow& a, const Q9ResultRow& b) {
                  if (a.nation != b.nation) {
                      return a.nation < b.nation;
                  }
                  return a.o_year > b.o_year;
              });

#ifdef TRACE
    q9_trace::data().total_ns += trace_utils::get_time_ns() - total_start;
    q9_trace::emit();
#endif
    return results;
}

void write_q9_csv(const std::string& filename, const std::vector<Q9ResultRow>& rows) {
    std::ofstream out(filename);
    out << "nation,o_year,sum_profit\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double profit =
            static_cast<double>(row.sum_profit_raw) / q9::kPriceScale;
        out << q9::escape_csv_field(row.nation) << ',' << row.o_year << ','
            << profit << "\n";
    }
}