#include "query_q1.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "trace_utils.hpp"

namespace {
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

struct Q1Agg {
    int64_t sum_qty = 0;
    int64_t sum_base_price = 0;
    int64_t sum_disc_price = 0;
    __int128 sum_charge_num = 0;
    int64_t sum_disc = 0;
    int64_t count_order = 0;
};

int32_t parse_delta_days(const std::string& value) {
    return static_cast<int32_t>(std::stoi(value));
}
}  // namespace

#ifdef TRACE
namespace q1_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t rows_scanned = 0;
    uint64_t rows_emitted = 0;
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
    if (std::strcmp(name, "q1_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q1_scan") == 0) {
        trace.scan_ns += ns;
    } else if (std::strcmp(name, "q1_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q1_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q1_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q1_scan " << trace.scan_ns << "\n";
    std::cout << "PROFILE q1_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q1_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q1_lineitem_scan_rows_scanned " << trace.rows_scanned << "\n";
    std::cout << "COUNT q1_lineitem_scan_rows_emitted " << trace.rows_emitted << "\n";
    std::cout << "COUNT q1_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q1_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q1_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q1_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q1_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q1_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q1_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q1_trace::record_timing)
#define TRACE_SET(field, value) (q1_trace::data().field = (value))
#define TRACE_ADD(field, value) (q1_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q1ResultRow> run_q1(const Database& db, const Q1Args& args) {
    std::vector<Q1ResultRow> rows;
#ifdef TRACE
    q1_trace::reset();
#endif
    {
        PROFILE_SCOPE("q1_total");
        const int32_t delta_days = parse_delta_days(args.DELTA);
        const int32_t base_target =
            static_cast<int32_t>(days_from_civil(1998, 12, 1) - delta_days);
        const int32_t cutoff_offset = base_target - db.base_date_days;
        const int16_t cutoff_offset_short = static_cast<int16_t>(cutoff_offset);

        const auto& li = db.lineitem;
        const size_t returnflag_count = li.returnflag.dictionary.size();
        const size_t linestatus_count = li.linestatus.dictionary.size();
        const size_t group_capacity = returnflag_count * linestatus_count;
        std::vector<Q1Agg> aggregates(group_capacity);
        size_t group_count = 0;
        TRACE_SET(rows_scanned, li.row_count);
        {
            PROFILE_SCOPE("q1_scan");
            const auto* __restrict shipdate = li.shipdate.data();
            const auto* __restrict group_codes = li.returnflag_linestatus.data();
            const int32_t* __restrict extendedprice = li.extendedprice.data();
            const int32_t* __restrict discounted_price = li.discounted_price.data();
            const uint8_t* __restrict discount = li.discount.data();
            const uint8_t* __restrict tax = li.tax.data();
            const int16_t* __restrict quantity = li.quantity.data();
            const size_t row_count = li.row_count;
            const int16_t* __restrict ship_ptr = shipdate;
            const uint8_t* __restrict group_ptr = group_codes;
            const int32_t* __restrict price_ptr = extendedprice;
            const int32_t* __restrict disc_price_ptr = discounted_price;
            const uint8_t* __restrict disc_ptr = discount;
            const uint8_t* __restrict tax_ptr = tax;
            const int16_t* __restrict qty_ptr = quantity;
            size_t i = 0;
            {
                for (; i + 4 <= row_count; i += 4) {
                    const int16_t s0 = ship_ptr[0];
                    const int16_t s1 = ship_ptr[1];
                    const int16_t s2 = ship_ptr[2];
                    const int16_t s3 = ship_ptr[3];
                    const int16_t max01 = s0 > s1 ? s0 : s1;
                    const int16_t max23 = s2 > s3 ? s2 : s3;
                    const int16_t max_value = max01 > max23 ? max01 : max23;
                    if (__builtin_expect(max_value <= cutoff_offset_short, 1)) {
                        TRACE_ADD(rows_emitted, 4);
#define Q1_PROCESS_ROW_NO_FILTER(offset)                                           \
    do {                                                                           \
        const size_t group_index = static_cast<size_t>(group_ptr[(offset)]);       \
        auto& agg = aggregates[group_index];                                       \
        const int32_t base = price_ptr[(offset)];                                  \
        const int32_t disc_price = disc_price_ptr[(offset)];                       \
        const int32_t tax_multiplier =                                              \
            kDiscountScale + static_cast<int32_t>(tax_ptr[(offset)]);              \
        agg.sum_qty += static_cast<int64_t>(qty_ptr[(offset)]);                    \
        agg.sum_base_price += static_cast<int64_t>(base);                          \
        agg.sum_disc_price += static_cast<int64_t>(disc_price);                    \
        agg.sum_charge_num += static_cast<__int128>(                               \
            static_cast<int64_t>(disc_price) * tax_multiplier);                    \
        agg.sum_disc += static_cast<int64_t>(disc_ptr[(offset)]);                  \
        agg.count_order += 1;                                                      \
    } while (false)
                        Q1_PROCESS_ROW_NO_FILTER(0);
                        Q1_PROCESS_ROW_NO_FILTER(1);
                        Q1_PROCESS_ROW_NO_FILTER(2);
                        Q1_PROCESS_ROW_NO_FILTER(3);
#undef Q1_PROCESS_ROW_NO_FILTER
                    } else {
#define Q1_PROCESS_ROW_FILTER(offset)                                              \
    do {                                                                           \
        if (__builtin_expect(ship_ptr[(offset)] > cutoff_offset_short, 0)) {       \
            break;                                                                 \
        }                                                                          \
        TRACE_ADD(rows_emitted, 1);                                                \
        const size_t group_index = static_cast<size_t>(group_ptr[(offset)]);       \
        auto& agg = aggregates[group_index];                                       \
        const int32_t base = price_ptr[(offset)];                                  \
        const int32_t disc_price = disc_price_ptr[(offset)];                       \
        const int32_t tax_multiplier =                                              \
            kDiscountScale + static_cast<int32_t>(tax_ptr[(offset)]);              \
        agg.sum_qty += static_cast<int64_t>(qty_ptr[(offset)]);                    \
        agg.sum_base_price += static_cast<int64_t>(base);                          \
        agg.sum_disc_price += static_cast<int64_t>(disc_price);                    \
        agg.sum_charge_num += static_cast<__int128>(                               \
            static_cast<int64_t>(disc_price) * tax_multiplier);                    \
        agg.sum_disc += static_cast<int64_t>(disc_ptr[(offset)]);                  \
        agg.count_order += 1;                                                      \
    } while (false)
                        Q1_PROCESS_ROW_FILTER(0);
                        Q1_PROCESS_ROW_FILTER(1);
                        Q1_PROCESS_ROW_FILTER(2);
                        Q1_PROCESS_ROW_FILTER(3);
#undef Q1_PROCESS_ROW_FILTER
                    }
                    ship_ptr += 4;
                    group_ptr += 4;
                    price_ptr += 4;
                    disc_price_ptr += 4;
                    disc_ptr += 4;
                    tax_ptr += 4;
                    qty_ptr += 4;
                }
                for (; i < row_count; ++i, ++ship_ptr, ++group_ptr, ++price_ptr,
                     ++disc_price_ptr, ++disc_ptr, ++tax_ptr, ++qty_ptr) {
                    if (__builtin_expect(*ship_ptr > cutoff_offset_short, 0)) {
                        continue;
                    }
                    TRACE_ADD(rows_emitted, 1);
                    const size_t group_index = static_cast<size_t>(*group_ptr);
                    auto& agg = aggregates[group_index];
                    const int32_t base = *price_ptr;
                    const int32_t disc_price = *disc_price_ptr;
                    const int32_t tax_multiplier =
                        kDiscountScale + static_cast<int32_t>(*tax_ptr);
                    agg.sum_qty += static_cast<int64_t>(*qty_ptr);
                    agg.sum_base_price += static_cast<int64_t>(base);
                    agg.sum_disc_price += static_cast<int64_t>(disc_price);
                    agg.sum_charge_num += static_cast<__int128>(
                        static_cast<int64_t>(disc_price) * tax_multiplier);
                    agg.sum_disc += static_cast<int64_t>(*disc_ptr);
                    agg.count_order += 1;
                }
            }
        }

        TRACE_SET(agg_rows_in, q1_trace::data().rows_emitted);
        {
            PROFILE_SCOPE("q1_agg_finalize");
            rows.reserve(group_capacity);
            for (size_t idx = 0; idx < group_capacity; ++idx) {
                const uint32_t returnflag_code = static_cast<uint32_t>(idx / linestatus_count);
                const uint32_t linestatus_code = static_cast<uint32_t>(idx % linestatus_count);
                const auto& agg = aggregates[idx];
                if (agg.count_order == 0) {
                    continue;
                }
                const double count = static_cast<double>(agg.count_order);
                Q1ResultRow row;
                row.returnflag = li.returnflag.dictionary[returnflag_code];
                row.linestatus = li.linestatus.dictionary[linestatus_code];
                row.sum_qty = static_cast<double>(agg.sum_qty) / kPriceScale;
                row.sum_base_price = static_cast<double>(agg.sum_base_price) / kPriceScale;
                row.sum_disc_price = static_cast<double>(agg.sum_disc_price) / kPriceScale;
                row.sum_charge = static_cast<double>(agg.sum_charge_num) /
                                 (static_cast<double>(kPriceScale) * kDiscountScale);
                row.avg_qty = row.sum_qty / count;
                row.avg_price = row.sum_base_price / count;
                row.avg_disc = static_cast<double>(agg.sum_disc) / kDiscountScale / count;
                row.count_order = agg.count_order;
                rows.push_back(std::move(row));
            }
            group_count = rows.size();
        }

        TRACE_SET(groups_created, group_count);
        TRACE_SET(agg_rows_emitted, group_count);

        TRACE_SET(sort_rows_in, rows.size());
        {
            PROFILE_SCOPE("q1_sort");
            std::sort(rows.begin(), rows.end(),
                      [](const Q1ResultRow& a, const Q1ResultRow& b) {
                          if (a.returnflag == b.returnflag) {
                              return a.linestatus < b.linestatus;
                          }
                          return a.returnflag < b.returnflag;
                      });
        }
        TRACE_SET(sort_rows_out, rows.size());
    }

    TRACE_SET(query_output_rows, rows.size());
#ifdef TRACE
    q1_trace::emit();
#endif
    return rows;
}

void write_q1_csv(const std::string& filename, const std::vector<Q1ResultRow>& rows) {
    std::ofstream out(filename);
    out << "l_returnflag,l_linestatus,sum_qty,sum_base_price,sum_disc_price,sum_charge,avg_qty,"
           "avg_price,avg_disc,count_order\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        out << row.returnflag << ',' << row.linestatus << ',' << row.sum_qty << ','
            << row.sum_base_price << ',' << row.sum_disc_price << ',' << row.sum_charge << ','
            << row.avg_qty << ',' << row.avg_price << ',';
        out << std::setprecision(4) << row.avg_disc << std::setprecision(2) << ','
            << row.count_order << "\n";
    }
}