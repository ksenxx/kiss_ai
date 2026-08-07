#include "query_q15.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q15 {
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

DateParts add_months(const DateParts& base, int months) {
    DateParts out = base;
    int month_index = out.month - 1 + months;
    out.year += month_index / 12;
    month_index %= 12;
    if (month_index < 0) {
        month_index += 12;
        out.year -= 1;
    }
    out.month = month_index + 1;
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

}  // namespace q15

#ifdef TRACE
namespace q15_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t supplier_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
    uint64_t supplier_rows_scanned = 0;
    uint64_t supplier_rows_emitted = 0;
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
    if (std::strcmp(name, "q15_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q15_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q15_supplier_scan") == 0) {
        trace.supplier_scan_ns += ns;
    } else if (std::strcmp(name, "q15_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q15_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q15_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q15_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q15_supplier_scan " << trace.supplier_scan_ns << "\n";
    std::cout << "PROFILE q15_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q15_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q15_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q15_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q15_supplier_scan_rows_scanned " << trace.supplier_rows_scanned
              << "\n";
    std::cout << "COUNT q15_supplier_scan_rows_emitted " << trace.supplier_rows_emitted
              << "\n";
    std::cout << "COUNT q15_supplier_revenue_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q15_supplier_revenue_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q15_supplier_revenue_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q15_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q15_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q15_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q15_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q15_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q15_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q15_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q15_trace::record_timing)
#define TRACE_SET(field, value) (q15_trace::data().field = (value))
#define TRACE_ADD(field, value) (q15_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q15ResultRow> run_q15(const Database& db, const Q15Args& args) {
#ifdef TRACE
    q15_trace::reset();
#endif
    std::vector<Q15ResultRow> results;
    {
        PROFILE_SCOPE("q15_total");
        const auto& lineitem = db.lineitem;
        const auto& supplier = db.supplier;

        const int16_t start_offset = q15::parse_date_offset(args.DATE, db.base_date_days);
        const int16_t end_offset =
            q15::parse_date_offset_add_months(args.DATE, db.base_date_days, 3);
        const auto start_parts = q15::parse_date_parts(args.DATE);
        const auto end_parts = q15::add_months(start_parts, 3);
        int32_t start_month_index = start_parts.year * 12 + (start_parts.month - 1);
        int32_t end_month_index = end_parts.year * 12 + (end_parts.month - 1);
        if (end_parts.day == 1) {
            end_month_index -= 1;
        }
        if (end_month_index < start_month_index) {
            end_month_index = start_month_index;
        }
        const int32_t date_span = static_cast<int32_t>(end_offset - start_offset);
        const int32_t start_offset_i = static_cast<int32_t>(start_offset);
        const uint32_t date_span_u = static_cast<uint32_t>(date_span);

        int32_t max_suppkey_value = -1;
        for (uint32_t row = 0; row < supplier.row_count; ++row) {
            const int32_t key = supplier.suppkey[row];
            if (key > max_suppkey_value) {
                max_suppkey_value = key;
            }
        }
        const size_t initial_capacity =
            std::max<size_t>(static_cast<size_t>(max_suppkey_value + 1), 1);
        std::vector<int64_t> revenue_by_suppkey(initial_capacity, 0);
        int64_t max_revenue = 0;
        uint64_t suppliers_with_revenue = 0;

        uint64_t lineitems_emitted = 0;
        uint64_t lineitems_scanned = 0;
        {
            PROFILE_SCOPE("q15_lineitem_scan");
            const int16_t* __restrict shipdate = lineitem.shipdate.data();
            const int32_t* __restrict suppkey = lineitem.suppkey.data();
            const int32_t* __restrict discounted_price =
                lineitem.discounted_price.data();
            int64_t* __restrict revenue_ptr = revenue_by_suppkey.data();
            constexpr size_t kPrefetchDistance = 8;
            if (!lineitem.shards.empty()) {
            }
            if (lineitem.shards.empty()) {
                lineitems_scanned = lineitem.row_count;
                const size_t count = lineitem.row_count;
                const int16_t* __restrict ship_ptr = shipdate;
                const int32_t* __restrict supp_ptr = suppkey;
                const int32_t* __restrict price_ptr = discounted_price;
                size_t pos = 0;
                for (; pos + 3 < count; pos += 4) {
                    const int32_t ship0 = static_cast<int32_t>(ship_ptr[0]);
                    const int32_t ship1 = static_cast<int32_t>(ship_ptr[1]);
                    const int32_t ship2 = static_cast<int32_t>(ship_ptr[2]);
                    const int32_t ship3 = static_cast<int32_t>(ship_ptr[3]);
                    if (static_cast<uint32_t>(ship0 - start_offset_i) < date_span_u) {
                        revenue_ptr[static_cast<uint32_t>(supp_ptr[0])] +=
                            static_cast<int64_t>(price_ptr[0]);
                        lineitems_emitted += 1;
                    }
                    if (static_cast<uint32_t>(ship1 - start_offset_i) < date_span_u) {
                        revenue_ptr[static_cast<uint32_t>(supp_ptr[1])] +=
                            static_cast<int64_t>(price_ptr[1]);
                        lineitems_emitted += 1;
                    }
                    if (static_cast<uint32_t>(ship2 - start_offset_i) < date_span_u) {
                        revenue_ptr[static_cast<uint32_t>(supp_ptr[2])] +=
                            static_cast<int64_t>(price_ptr[2]);
                        lineitems_emitted += 1;
                    }
                    if (static_cast<uint32_t>(ship3 - start_offset_i) < date_span_u) {
                        revenue_ptr[static_cast<uint32_t>(supp_ptr[3])] +=
                            static_cast<int64_t>(price_ptr[3]);
                        lineitems_emitted += 1;
                    }
                    ship_ptr += 4;
                    supp_ptr += 4;
                    price_ptr += 4;
                }
                for (; pos < count; ++pos, ++ship_ptr, ++supp_ptr, ++price_ptr) {
                    const int32_t shipdate_value = static_cast<int32_t>(*ship_ptr);
                    if (static_cast<uint32_t>(shipdate_value - start_offset_i) >=
                        date_span_u) {
                        continue;
                    }
                    revenue_ptr[static_cast<uint32_t>(*supp_ptr)] +=
                        static_cast<int64_t>(*price_ptr);
                    lineitems_emitted += 1;
                }
            } else {
                for (const auto& shard : lineitem.shards) {
                    const int32_t shard_index = shard.year * 12 + (shard.month - 1);
                    if (shard_index < start_month_index || shard_index > end_month_index) {
                        continue;
                    }
                    const bool full_range =
                        shard.min_shipdate >= start_offset &&
                        shard.max_shipdate < end_offset;
                    if (shard.contiguous) {
                        const uint32_t start = shard.start;
                        const uint32_t end = shard.end;
                        const size_t count = static_cast<size_t>(end - start);
                        lineitems_scanned += static_cast<uint64_t>(count);
                        if (count == 0) {
                            continue;
                        }
                        const int32_t* __restrict supp_ptr = suppkey + start;
                        const int32_t* __restrict price_ptr =
                            discounted_price + start;
                        if (full_range) {
                            size_t pos = 0;
                            for (; pos + 3 < count; pos += 4) {
                                revenue_ptr[static_cast<uint32_t>(supp_ptr[0])] +=
                                    static_cast<int64_t>(price_ptr[0]);
                                revenue_ptr[static_cast<uint32_t>(supp_ptr[1])] +=
                                    static_cast<int64_t>(price_ptr[1]);
                                revenue_ptr[static_cast<uint32_t>(supp_ptr[2])] +=
                                    static_cast<int64_t>(price_ptr[2]);
                                revenue_ptr[static_cast<uint32_t>(supp_ptr[3])] +=
                                    static_cast<int64_t>(price_ptr[3]);
                                supp_ptr += 4;
                                price_ptr += 4;
                            }
                            for (; pos < count; ++pos, ++supp_ptr, ++price_ptr) {
                                revenue_ptr[static_cast<uint32_t>(*supp_ptr)] +=
                                    static_cast<int64_t>(*price_ptr);
                            }
                            lineitems_emitted += static_cast<uint64_t>(count);
                        } else {
                            const int16_t* __restrict ship_ptr = shipdate + start;
                            size_t pos = 0;
                            for (; pos + 3 < count; pos += 4) {
                                const int32_t ship0 = static_cast<int32_t>(ship_ptr[0]);
                                const int32_t ship1 = static_cast<int32_t>(ship_ptr[1]);
                                const int32_t ship2 = static_cast<int32_t>(ship_ptr[2]);
                                const int32_t ship3 = static_cast<int32_t>(ship_ptr[3]);
                                if (static_cast<uint32_t>(ship0 - start_offset_i) <
                                    date_span_u) {
                                    revenue_ptr[static_cast<uint32_t>(supp_ptr[0])] +=
                                        static_cast<int64_t>(price_ptr[0]);
                                    lineitems_emitted += 1;
                                }
                                if (static_cast<uint32_t>(ship1 - start_offset_i) <
                                    date_span_u) {
                                    revenue_ptr[static_cast<uint32_t>(supp_ptr[1])] +=
                                        static_cast<int64_t>(price_ptr[1]);
                                    lineitems_emitted += 1;
                                }
                                if (static_cast<uint32_t>(ship2 - start_offset_i) <
                                    date_span_u) {
                                    revenue_ptr[static_cast<uint32_t>(supp_ptr[2])] +=
                                        static_cast<int64_t>(price_ptr[2]);
                                    lineitems_emitted += 1;
                                }
                                if (static_cast<uint32_t>(ship3 - start_offset_i) <
                                    date_span_u) {
                                    revenue_ptr[static_cast<uint32_t>(supp_ptr[3])] +=
                                        static_cast<int64_t>(price_ptr[3]);
                                    lineitems_emitted += 1;
                                }
                                ship_ptr += 4;
                                supp_ptr += 4;
                                price_ptr += 4;
                            }
                            for (; pos < count;
                                 ++pos, ++ship_ptr, ++supp_ptr, ++price_ptr) {
                                const int32_t shipdate_value =
                                    static_cast<int32_t>(*ship_ptr);
                                if (static_cast<uint32_t>(shipdate_value -
                                                          start_offset_i) >=
                                    date_span_u) {
                                    continue;
                                }
                                revenue_ptr[static_cast<uint32_t>(*supp_ptr)] +=
                                    static_cast<int64_t>(*price_ptr);
                                lineitems_emitted += 1;
                            }
                        }
                    } else {
                        const uint32_t* __restrict row_indices = shard.row_indices.data();
                        const size_t shard_rows = shard.row_indices.size();
                        lineitems_scanned += shard_rows;
                        if (shard_rows == 0) {
                            continue;
                        }
                        if (full_range) {
                            for (size_t pos = 0; pos < shard_rows; ++pos) {
                                if (pos + kPrefetchDistance < shard_rows) {
                                    const uint32_t pre_idx =
                                        row_indices[pos + kPrefetchDistance];
                                    __builtin_prefetch(suppkey + pre_idx, 0, 1);
                                    __builtin_prefetch(discounted_price + pre_idx,
                                                       0, 1);
                                }
                                const uint32_t li_idx = row_indices[pos];
                                revenue_ptr[static_cast<uint32_t>(suppkey[li_idx])] +=
                                    static_cast<int64_t>(discounted_price[li_idx]);
                            }
                            lineitems_emitted += shard_rows;
                        } else {
                            for (size_t pos = 0; pos < shard_rows; ++pos) {
                                if (pos + kPrefetchDistance < shard_rows) {
                                    const uint32_t pre_idx =
                                        row_indices[pos + kPrefetchDistance];
                                    __builtin_prefetch(shipdate + pre_idx, 0, 1);
                                    __builtin_prefetch(suppkey + pre_idx, 0, 1);
                                    __builtin_prefetch(discounted_price + pre_idx,
                                                       0, 1);
                                }
                                const uint32_t li_idx = row_indices[pos];
                                const int32_t shipdate_value =
                                    static_cast<int32_t>(shipdate[li_idx]);
                                if (static_cast<uint32_t>(shipdate_value -
                                                          start_offset_i) >=
                                    date_span_u) {
                                    continue;
                                }
                                revenue_ptr[static_cast<uint32_t>(suppkey[li_idx])] +=
                                    static_cast<int64_t>(discounted_price[li_idx]);
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

        if (lineitems_emitted == 0) {
#ifdef TRACE
            q15_trace::emit();
#endif
            return {};
        }

        for (const int64_t revenue_value : revenue_by_suppkey) {
            if (revenue_value <= 0) {
                continue;
            }
            suppliers_with_revenue += 1;
            if (revenue_value > max_revenue) {
                max_revenue = revenue_value;
            }
        }

        if (max_revenue == 0) {
#ifdef TRACE
            q15_trace::emit();
#endif
            return {};
        }

        std::vector<int32_t> suppkey_to_row(revenue_by_suppkey.size(), -1);
        uint64_t supplier_emitted = 0;
        {
            PROFILE_SCOPE("q15_supplier_scan");
            TRACE_SET(supplier_rows_scanned, supplier.row_count);
            for (uint32_t row = 0; row < supplier.row_count; ++row) {
                const int32_t key = supplier.suppkey[row];
                if (key >= 0 && static_cast<size_t>(key) < suppkey_to_row.size()) {
                    suppkey_to_row[static_cast<size_t>(key)] = static_cast<int32_t>(row);
                    supplier_emitted += 1;
                }
            }
        }
        TRACE_SET(supplier_rows_emitted, supplier_emitted);
        TRACE_SET(join_build_rows_in, supplier_emitted);

        {
            PROFILE_SCOPE("q15_agg_finalize");
            for (size_t idx = 0; idx < revenue_by_suppkey.size(); ++idx) {
                const int64_t revenue_value = revenue_by_suppkey[idx];
                if (revenue_value != max_revenue) {
                    continue;
                }
                const int32_t row = suppkey_to_row[idx];
                if (row < 0) {
                    continue;
                }
                Q15ResultRow out;
                out.s_suppkey = static_cast<int32_t>(idx);
                out.s_name = std::string(q15::get_string_view(supplier.name, row));
                out.s_address = std::string(q15::get_string_view(supplier.address, row));
                out.s_phone = std::string(q15::get_string_view(supplier.phone, row));
                out.total_revenue_raw = revenue_value;
                results.push_back(std::move(out));
            }
            TRACE_SET(groups_created, suppliers_with_revenue);
            TRACE_SET(agg_rows_emitted, suppliers_with_revenue);
        }
        TRACE_SET(join_probe_rows_in, suppliers_with_revenue);
        TRACE_SET(join_rows_emitted, results.size());

        {
            PROFILE_SCOPE("q15_sort");
            TRACE_SET(sort_rows_in, results.size());
            std::sort(results.begin(), results.end(),
                      [](const Q15ResultRow& a, const Q15ResultRow& b) {
                          return a.s_suppkey < b.s_suppkey;
                      });
            TRACE_SET(sort_rows_out, results.size());
        }
        TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q15_trace::emit();
#endif
    return results;
}

void write_q15_csv(const std::string& filename, const std::vector<Q15ResultRow>& rows) {
    std::ofstream out(filename);
    out << "s_suppkey,s_name,s_address,s_phone,total_revenue\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double revenue = static_cast<double>(row.total_revenue_raw) / q15::kPriceScale;
        out << row.s_suppkey << ','
            << q15::escape_csv_field(row.s_name) << ','
            << q15::escape_csv_field(row.s_address) << ','
            << q15::escape_csv_field(row.s_phone) << ','
            << revenue << "\n";
    }
}