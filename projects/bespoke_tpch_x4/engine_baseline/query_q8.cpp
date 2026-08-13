#include "query_q8.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q8 {
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

std::string_view get_string_view(const StringColumn& column, size_t row) {
    const uint32_t start = column.offsets[row];
    const uint32_t end = column.offsets[row + 1];
    return std::string_view(column.data.data() + start, end - start);
}

template <typename T>
int32_t max_value(const std::vector<T>& values) {
    if (values.empty()) {
        return 0;
    }
    return static_cast<int32_t>(*std::max_element(values.begin(), values.end()));
}
}  // namespace q8

#ifdef TRACE
namespace q8_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t orders_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t orders_rows_scanned = 0;
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
    if (std::strcmp(name, "q8_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q8_orders_scan") == 0) {
        trace.orders_scan_ns += ns;
    } else if (std::strcmp(name, "q8_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q8_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q8_orders_scan " << trace.orders_scan_ns << "\n";
    std::cout << "PROFILE q8_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "COUNT q8_orders_scan_rows_scanned " << trace.orders_rows_scanned
              << "\n";
    std::cout << "COUNT q8_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q8_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
}
}  // namespace q8_trace
#define TRACE_TIMER(name) \
    trace_utils::ScopedTimer timer_##name(#name, q8_trace::record_timing)
#define TRACE_ADD(field, value) (q8_trace::data().field += (value))
#else
#define TRACE_TIMER(name)
#define TRACE_ADD(field, value)
#endif

std::vector<Q8ResultRow> run_q8(const Database& db, const Q8Args& args) {
#ifdef TRACE
    const auto total_start = trace_utils::get_time_ns();
    q8_trace::reset();
#endif
    const auto region_it = db.region.name_to_key.find(args.REGION);
    if (region_it == db.region.name_to_key.end()) {
        return {};
    }
    const auto nation_it = db.nation.name_to_key.find(args.NATION);
    if (nation_it == db.nation.name_to_key.end()) {
        return {};
    }
    const int32_t region_key = region_it->second;
    const int32_t target_nation = nation_it->second;

    const int16_t start_offset = q8::parse_date_offset("1995-01-01", db.base_date_days);
    const int16_t end_offset = q8::parse_date_offset("1996-12-31", db.base_date_days);
    const int16_t split_offset = q8::parse_date_offset("1996-01-01", db.base_date_days);
    const int32_t start_year = q8::year_from_offset(db.base_date_days, start_offset);
    const int32_t end_year = q8::year_from_offset(db.base_date_days, end_offset);
    const int32_t year_span = end_year - start_year + 1;

    const auto& nation = db.nation;
    const auto& customer = db.customer;
    const auto& supplier = db.supplier;
    const auto& part = db.part;
    const auto& orders = db.orders;
    const auto& lineitem = db.lineitem;

    int32_t max_nationkey = 0;
    for (const auto& row : nation.rows) {
        max_nationkey = std::max(max_nationkey, row.nationkey);
    }
    std::vector<uint8_t> nation_in_region(static_cast<size_t>(max_nationkey) + 1, 0);
    for (const auto& row : nation.rows) {
        if (row.regionkey == region_key) {
            nation_in_region[static_cast<size_t>(row.nationkey)] = 1;
        }
    }

    const int32_t max_custkey = q8::max_value(customer.custkey);
    std::vector<uint8_t> cust_in_region(static_cast<size_t>(max_custkey) + 1, 0);
    for (uint32_t i = 0; i < customer.row_count; ++i) {
        const int32_t key = customer.custkey[i];
        const int32_t nkey = customer.nationkey[i];
        if (nation_in_region[static_cast<size_t>(nkey)]) {
            cust_in_region[static_cast<size_t>(key)] = 1;
        }
    }

    const int32_t max_suppkey = q8::max_value(supplier.suppkey);
    std::vector<uint8_t> supp_is_target(static_cast<size_t>(max_suppkey) + 1, 0);
    for (uint32_t i = 0; i < supplier.row_count; ++i) {
        const int32_t key = supplier.suppkey[i];
        if (supplier.nationkey[i] == target_nation) {
            supp_is_target[static_cast<size_t>(key)] = 1;
        }
    }

    const int32_t max_partkey = q8::max_value(part.partkey);
    std::vector<uint8_t> part_ok_by_key(static_cast<size_t>(max_partkey) + 1, 0);
    std::vector<uint8_t> type_matches(part.type.dictionary.size(), 0);
    for (size_t idx = 0; idx < part.type.dictionary.size(); ++idx) {
        if (part.type.dictionary[idx] == args.TYPE) {
            type_matches[idx] = 1;
        }
    }
    const auto* type_codes = part.type.codes.data();
    for (uint32_t i = 0; i < part.row_count; ++i) {
        const int32_t key = part.partkey[i];
        if (type_matches[type_codes[i]]) {
            part_ok_by_key[static_cast<size_t>(key)] = 1;
        }
    }

    std::array<int64_t, 2> total_by_year{};
    std::array<int64_t, 2> nation_by_year{};

#ifdef TRACE
    uint64_t orders_scanned = 0;
    uint64_t lineitems_scanned = 0;
    uint64_t lineitems_emitted = 0;
#endif

    const bool use_order_ranges =
        lineitem.orderkey_sorted && orders.lineitem_ranges.size() == orders.row_count;

    if (use_order_ranges) {
        const auto start_it =
            std::lower_bound(orders.orderdate.begin(), orders.orderdate.end(), start_offset);
        const auto end_it =
            std::upper_bound(orders.orderdate.begin(), orders.orderdate.end(), end_offset);
        const uint32_t start_idx =
            static_cast<uint32_t>(start_it - orders.orderdate.begin());
        const uint32_t end_idx =
            static_cast<uint32_t>(end_it - orders.orderdate.begin());

#ifdef TRACE
        uint64_t lineitem_scan_ns = 0;
        const auto orders_scan_start = trace_utils::get_time_ns();
#endif
        const auto* __restrict part_ok = part_ok_by_key.data();
        const auto* __restrict supp_target = supp_is_target.data();
        const auto* __restrict cust_region = cust_in_region.data();
        const auto* __restrict order_custkey = orders.custkey.data();
        const auto* __restrict order_date = orders.orderdate.data();
        const auto* __restrict order_ranges = orders.lineitem_ranges.data();
        const auto* __restrict li_partkey = lineitem.partkey.data();
        const auto* __restrict li_suppkey = lineitem.suppkey.data();
        const int32_t* __restrict li_extendedprice = lineitem.extendedprice.data();
        const uint8_t* __restrict li_discount = lineitem.discount.data();
        constexpr int32_t discount_scale = q8::kDiscountScale;
        for (uint32_t i = start_idx; i < end_idx; ++i) {
#ifdef TRACE
            ++orders_scanned;
#endif
            const int32_t custkey = order_custkey[i];
            if (!cust_region[static_cast<size_t>(custkey)]) {
                continue;
            }
            const auto range = order_ranges[i];
            if (range.end <= range.start) {
                continue;
            }
            const size_t year_index =
                static_cast<size_t>(order_date[i] >= split_offset);
#ifdef TRACE
            const auto lineitem_start = trace_utils::get_time_ns();
#endif
            const uint32_t li_start = range.start;
            const uint32_t li_end = range.end;
            const int32_t* part_ptr = li_partkey + li_start;
            const int32_t* supp_ptr = li_suppkey + li_start;
            const int32_t* ext_ptr = li_extendedprice + li_start;
            const uint8_t* disc_ptr = li_discount + li_start;
            const int32_t* part_end = li_partkey + li_end;
            for (; part_ptr != part_end;
                 ++part_ptr, ++supp_ptr, ++ext_ptr, ++disc_ptr) {
#ifdef TRACE
                ++lineitems_scanned;
#endif
                const int32_t partkey = *part_ptr;
                if (!part_ok[static_cast<size_t>(partkey)]) {
                    continue;
                }
                const int32_t suppkey = *supp_ptr;
                const int64_t volume =
                    static_cast<int64_t>(*ext_ptr) *
                    (discount_scale - static_cast<int32_t>(*disc_ptr));
#ifdef TRACE
                ++lineitems_emitted;
#endif
                total_by_year[year_index] += volume;
                nation_by_year[year_index] +=
                    volume * static_cast<int64_t>(supp_target[suppkey]);
            }
#ifdef TRACE
            lineitem_scan_ns += trace_utils::get_time_ns() - lineitem_start;
#endif
        }
#ifdef TRACE
        q8_trace::data().lineitem_scan_ns += lineitem_scan_ns;
        q8_trace::data().orders_scan_ns +=
            trace_utils::get_time_ns() - orders_scan_start - lineitem_scan_ns;
#endif
    } else {
        const int32_t max_orderkey = q8::max_value(orders.orderkey);
        std::vector<uint8_t> order_ok_by_key(static_cast<size_t>(max_orderkey) + 1, 0);
        {
            TRACE_TIMER(q8_orders_scan);
            const auto* __restrict order_key = orders.orderkey.data();
            const auto* __restrict order_date = orders.orderdate.data();
            const auto* __restrict order_custkey = orders.custkey.data();
            const auto* __restrict cust_region = cust_in_region.data();
            for (uint32_t i = 0; i < orders.row_count; ++i) {
#ifdef TRACE
                ++orders_scanned;
#endif
                const int32_t orderkey = order_key[i];
                const int16_t orderdate = order_date[i];
                if (orderdate < start_offset || orderdate > end_offset) {
                    continue;
                }
                const int32_t custkey = order_custkey[i];
                if (!cust_region[static_cast<size_t>(custkey)]) {
                    continue;
                }
                order_ok_by_key[static_cast<size_t>(orderkey)] = 1;
            }
        }

        {
            TRACE_TIMER(q8_lineitem_scan);
            const auto* __restrict part_ok = part_ok_by_key.data();
            const auto* __restrict supp_target = supp_is_target.data();
            const auto* __restrict li_partkey = lineitem.partkey.data();
            const auto* __restrict li_suppkey = lineitem.suppkey.data();
            const auto* __restrict li_orderkey = lineitem.orderkey.data();
            const int32_t* __restrict li_extendedprice = lineitem.extendedprice.data();
            const uint8_t* __restrict li_discount = lineitem.discount.data();
            constexpr int32_t discount_scale = q8::kDiscountScale;
            for (uint32_t li_idx = 0; li_idx < lineitem.row_count; ++li_idx) {
#ifdef TRACE
                ++lineitems_scanned;
#endif
                const int32_t partkey = li_partkey[li_idx];
                if (!part_ok[static_cast<size_t>(partkey)]) {
                    continue;
                }

                const int32_t orderkey = li_orderkey[li_idx];
                if (!order_ok_by_key[static_cast<size_t>(orderkey)]) {
                    continue;
                }
                const int32_t year_index =
                    orders.orderdate[static_cast<size_t>(orders.orderkey_to_row[orderkey])] >=
                            split_offset
                        ? 1
                        : 0;

                const int32_t suppkey = li_suppkey[li_idx];
                const int64_t volume =
                    static_cast<int64_t>(li_extendedprice[li_idx]) *
                    (discount_scale - static_cast<int32_t>(li_discount[li_idx]));
#ifdef TRACE
                ++lineitems_emitted;
#endif
                total_by_year[static_cast<size_t>(year_index)] += volume;
                nation_by_year[static_cast<size_t>(year_index)] +=
                    volume * static_cast<int64_t>(supp_target[suppkey]);
            }
        }
    }

    TRACE_ADD(orders_rows_scanned, orders_scanned);
    TRACE_ADD(lineitem_rows_scanned, lineitems_scanned);
    TRACE_ADD(lineitem_rows_emitted, lineitems_emitted);

    std::vector<Q8ResultRow> results;
    results.reserve(2);
    for (int32_t offset = 0; offset < year_span; ++offset) {
        const int64_t total = total_by_year[static_cast<size_t>(offset)];
        if (total == 0) {
            continue;
        }
        Q8ResultRow row;
        row.o_year = start_year + offset;
        row.mkt_share = static_cast<double>(nation_by_year[static_cast<size_t>(offset)]) /
                        static_cast<double>(total);
        results.push_back(row);
    }

#ifdef TRACE
    q8_trace::record_timing("q8_total", trace_utils::get_time_ns() - total_start);
    q8_trace::emit();
#endif
    return results;
}

void write_q8_csv(const std::string& filename, const std::vector<Q8ResultRow>& rows) {
    std::ofstream out(filename);
    out << "o_year,mkt_share\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        out << row.o_year << "," << row.mkt_share << "\n";
    }
}