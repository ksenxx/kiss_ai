#include "query_q6.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "trace_utils.hpp"

namespace q6 {
constexpr int32_t kPriceScale = 100;
constexpr int32_t kDiscountScale = 100;

struct DateParts {
    int year = 0;
    int month = 0;
    int day = 0;
};

DateParts parse_date_parts(const std::string& value) {
    DateParts parts;
    parts.year = std::stoi(value.substr(0, 4));
    parts.month = std::stoi(value.substr(5, 2));
    parts.day = std::stoi(value.substr(8, 2));
    return parts;
}

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

DateParts add_years(const DateParts& base, int years) {
    DateParts out = base;
    out.year += years;
    const int dim = days_in_month(out.year, out.month);
    if (out.day > dim) {
        out.day = dim;
    }
    return out;
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

int32_t parse_scaled_value(const std::string& value, int32_t scale) {
    const double numeric = std::stod(value);
    return static_cast<int32_t>(std::llround(numeric * static_cast<double>(scale)));
}
}  // namespace q6

#ifdef TRACE
namespace q6_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
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
    if (std::strcmp(name, "q6_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q6_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q6_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q6_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q6_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q6_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "COUNT q6_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q6_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q6_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q6_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q6_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q6_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q6_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q6_trace::record_timing)
#define TRACE_SET(field, value) (q6_trace::data().field = (value))
#define TRACE_ADD(field, value) (q6_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q6ResultRow> run_q6(const Database& db, const Q6Args& args) {
#ifdef TRACE
    q6_trace::reset();
#endif
    std::vector<Q6ResultRow> rows;
    {
        PROFILE_SCOPE("q6_total");
    const int16_t start_offset = q6::parse_date_offset(args.DATE, db.base_date_days);
    const int16_t end_offset =
        q6::parse_date_offset_add_years(args.DATE, db.base_date_days, 1);
    const auto start_parts = q6::parse_date_parts(args.DATE);
    const auto end_parts = q6::add_years(start_parts, 1);
    int32_t start_month_index = start_parts.year * 12 + (start_parts.month - 1);
    int32_t end_month_index = end_parts.year * 12 + (end_parts.month - 1);
    if (end_parts.day == 1) {
        end_month_index -= 1;
    }
    if (end_month_index < start_month_index) {
        end_month_index = start_month_index;
    }

    const int32_t discount_raw = q6::parse_scaled_value(args.DISCOUNT, q6::kDiscountScale);
    const int32_t discount_min = discount_raw - 1;
    const int32_t discount_max = discount_raw + 1;
    const int32_t discount_span = discount_max - discount_min;
    const int32_t quantity_limit = q6::parse_scaled_value(args.QUANTITY, q6::kPriceScale);
    const int32_t date_span = static_cast<int32_t>(end_offset - start_offset);
    const uint32_t date_span_u = static_cast<uint32_t>(date_span);
    const uint32_t discount_span_u = static_cast<uint32_t>(discount_span);

    const auto& lineitem = db.lineitem;
    const size_t row_count = lineitem.row_count;
    int64_t revenue_raw = 0;
    uint64_t lineitems_emitted = 0;
    uint64_t lineitems_scanned = 0;

    {
        PROFILE_SCOPE("q6_lineitem_scan");
        const int16_t* __restrict shipdate = lineitem.shipdate.data();
        const int32_t* __restrict extendedprice = lineitem.extendedprice.data();
        const uint8_t* __restrict discount = lineitem.discount.data();
        const int16_t* __restrict quantity = lineitem.quantity.data();
        if (lineitem.shards.empty()) {
            lineitems_scanned = row_count;
            const int16_t* __restrict ship_ptr = shipdate;
            const uint8_t* __restrict disc_ptr = discount;
            const int16_t* __restrict qty_ptr = quantity;
            const int32_t* __restrict price_ptr = extendedprice;
            for (size_t i = 0; i < row_count;
                 ++i, ++ship_ptr, ++disc_ptr, ++qty_ptr, ++price_ptr) {
                const int32_t ship = static_cast<int32_t>(*ship_ptr);
                if (__builtin_expect(static_cast<uint32_t>(ship - start_offset) >=
                                         date_span_u,
                                     1)) {
                    continue;
                }
                const int32_t disc = static_cast<int32_t>(*disc_ptr);
                if (__builtin_expect(static_cast<uint32_t>(disc - discount_min) >
                                         discount_span_u,
                                     1)) {
                    continue;
                }
                if (__builtin_expect(static_cast<int32_t>(*qty_ptr) >= quantity_limit, 1)) {
                    continue;
                }
                revenue_raw += static_cast<int64_t>(*price_ptr) * static_cast<int64_t>(disc);
                lineitems_emitted += 1;
            }
        } else {
            for (const auto& shard : lineitem.shards) {
                const int32_t shard_index = shard.year * 12 + (shard.month - 1);
                if (shard_index < start_month_index || shard_index > end_month_index) {
                    continue;
                }
                if (shard.max_shipdate < start_offset || shard.min_shipdate >= end_offset) {
                    continue;
                }
                if (shard.max_discount < discount_min || shard.min_discount > discount_max) {
                    continue;
                }
                if (static_cast<int32_t>(shard.min_quantity) >= quantity_limit) {
                    continue;
                }
                const bool full_date =
                    shard.min_shipdate >= start_offset && shard.max_shipdate < end_offset;
                const bool full_quantity = shard.max_quantity < quantity_limit;
                const int32_t disc = shard.min_discount;
                if (shard.contiguous) {
                    const size_t start = shard.start;
                    const size_t end = shard.end;
                    const size_t shard_rows = end - start;
                    lineitems_scanned += shard_rows;
                    const int16_t* __restrict ship_ptr = shipdate + start;
                    const int32_t* __restrict price_ptr = extendedprice + start;
                    if (full_date && full_quantity) {
                        for (size_t i = 0; i < shard_rows; ++i, ++price_ptr) {
                            revenue_raw +=
                                static_cast<int64_t>(*price_ptr) * static_cast<int64_t>(disc);
                        }
                        lineitems_emitted += shard_rows;
                    } else if (full_date) {
                        const int16_t* __restrict qty_ptr = quantity + start;
                        for (size_t i = 0; i < shard_rows;
                             ++i, ++qty_ptr, ++price_ptr) {
                            if (__builtin_expect(static_cast<int32_t>(*qty_ptr) >=
                                                     quantity_limit,
                                                 1)) {
                                continue;
                            }
                            revenue_raw +=
                                static_cast<int64_t>(*price_ptr) * static_cast<int64_t>(disc);
                            lineitems_emitted += 1;
                        }
                    } else if (full_quantity) {
                        for (size_t i = 0; i < shard_rows;
                             ++i, ++ship_ptr, ++price_ptr) {
                            const int32_t ship = static_cast<int32_t>(*ship_ptr);
                            if (__builtin_expect(static_cast<uint32_t>(ship - start_offset) >=
                                                     date_span_u,
                                                 1)) {
                                continue;
                            }
                            revenue_raw +=
                                static_cast<int64_t>(*price_ptr) * static_cast<int64_t>(disc);
                            lineitems_emitted += 1;
                        }
                    } else {
                        const int16_t* __restrict qty_ptr = quantity + start;
                        for (size_t i = 0; i < shard_rows;
                             ++i, ++ship_ptr, ++qty_ptr, ++price_ptr) {
                            const int32_t ship = static_cast<int32_t>(*ship_ptr);
                            if (__builtin_expect(static_cast<uint32_t>(ship - start_offset) >=
                                                     date_span_u,
                                                 1)) {
                                continue;
                            }
                            if (__builtin_expect(static_cast<int32_t>(*qty_ptr) >=
                                                     quantity_limit,
                                                 1)) {
                                continue;
                            }
                            revenue_raw +=
                                static_cast<int64_t>(*price_ptr) * static_cast<int64_t>(disc);
                            lineitems_emitted += 1;
                        }
                    }
                } else {
                    constexpr size_t kPrefetchDistance = 16;
                    const uint32_t* __restrict row_indices = shard.row_indices.data();
                    const size_t shard_rows = shard.row_indices.size();
                    lineitems_scanned += shard_rows;
                    if (full_date && full_quantity) {
                        for (size_t pos = 0; pos < shard_rows; ++pos) {
                            if (pos + kPrefetchDistance < shard_rows) {
                                const uint32_t next = row_indices[pos + kPrefetchDistance];
                                __builtin_prefetch(extendedprice + next);
                            }
                            const uint32_t i = row_indices[pos];
                            revenue_raw += static_cast<int64_t>(extendedprice[i]) *
                                           static_cast<int64_t>(disc);
                        }
                        lineitems_emitted += shard_rows;
                    } else if (full_date) {
                        for (size_t pos = 0; pos < shard_rows; ++pos) {
                            if (pos + kPrefetchDistance < shard_rows) {
                                const uint32_t next = row_indices[pos + kPrefetchDistance];
                                __builtin_prefetch(quantity + next);
                                __builtin_prefetch(extendedprice + next);
                            }
                            const uint32_t i = row_indices[pos];
                            if (__builtin_expect(static_cast<int32_t>(quantity[i]) >=
                                                     quantity_limit,
                                                 1)) {
                                continue;
                            }
                            revenue_raw += static_cast<int64_t>(extendedprice[i]) *
                                           static_cast<int64_t>(disc);
                            lineitems_emitted += 1;
                        }
                    } else if (full_quantity) {
                        for (size_t pos = 0; pos < shard_rows; ++pos) {
                            if (pos + kPrefetchDistance < shard_rows) {
                                const uint32_t next = row_indices[pos + kPrefetchDistance];
                                __builtin_prefetch(shipdate + next);
                                __builtin_prefetch(extendedprice + next);
                            }
                            const uint32_t i = row_indices[pos];
                            const int32_t ship = static_cast<int32_t>(shipdate[i]);
                            if (__builtin_expect(static_cast<uint32_t>(ship - start_offset) >=
                                                     date_span_u,
                                                 1)) {
                                continue;
                            }
                            revenue_raw += static_cast<int64_t>(extendedprice[i]) *
                                           static_cast<int64_t>(disc);
                            lineitems_emitted += 1;
                        }
                    } else {
                        for (size_t pos = 0; pos < shard_rows; ++pos) {
                            if (pos + kPrefetchDistance < shard_rows) {
                                const uint32_t next = row_indices[pos + kPrefetchDistance];
                                __builtin_prefetch(shipdate + next);
                                __builtin_prefetch(quantity + next);
                                __builtin_prefetch(extendedprice + next);
                            }
                            const uint32_t i = row_indices[pos];
                            const int32_t ship = static_cast<int32_t>(shipdate[i]);
                            if (__builtin_expect(static_cast<uint32_t>(ship - start_offset) >=
                                                     date_span_u,
                                                 1)) {
                                continue;
                            }
                            if (__builtin_expect(static_cast<int32_t>(quantity[i]) >=
                                                     quantity_limit,
                                                 1)) {
                                continue;
                            }
                            revenue_raw += static_cast<int64_t>(extendedprice[i]) *
                                           static_cast<int64_t>(disc);
                            lineitems_emitted += 1;
                        }
                    }
                }
            }
        }
    }
    TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
    TRACE_SET(lineitem_rows_emitted, lineitems_emitted);
    TRACE_SET(agg_rows_in, lineitems_emitted);

    {
        PROFILE_SCOPE("q6_agg_finalize");
        rows.push_back(Q6ResultRow{revenue_raw});
        TRACE_SET(groups_created, 1);
        TRACE_SET(agg_rows_emitted, 1);
    }
    TRACE_SET(query_output_rows, rows.size());
    }
#ifdef TRACE
    q6_trace::emit();
#endif
    return rows;
}

void write_q6_csv(const std::string& filename, const std::vector<Q6ResultRow>& rows) {
    std::ofstream out(filename);
    out << "revenue\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double revenue =
            static_cast<double>(row.revenue_raw) / (q6::kPriceScale * q6::kDiscountScale);
        out << revenue << "\n";
    }
}