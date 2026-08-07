#include "query_q20.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string_view>
#include <cmath>
#include <vector>

#include "trace_utils.hpp"

namespace q20 {
constexpr int32_t kPriceScale = 100;
constexpr double kYearSelectivity = 1.0 / 7.0;

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

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

uint64_t pack_key(int32_t partkey, int32_t suppkey) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(partkey)) << 32) |
           static_cast<uint32_t>(suppkey);
}

inline uint64_t mix_hash(uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

class FlatHashMap {
public:
    FlatHashMap() = default;

    void reserve(size_t expected) {
        const size_t min_capacity = expected < 8 ? 8 : expected * 2;
        size_t capacity = 1;
        while (capacity < min_capacity) {
            capacity <<= 1;
        }
        rehash(capacity);
    }

    void add(uint64_t key, int64_t value) {
        if (capacity_ == 0) {
            rehash(8);
        }
        if ((size_ + 1) * 10 > capacity_ * 7) {
            rehash(capacity_ * 2);
        }
        size_t idx = mix_hash(key) & mask_;
        while (occupied_[idx]) {
            if (keys_[idx] == key) {
                values_[idx] += value;
                return;
            }
            idx = (idx + 1) & mask_;
        }
        occupied_[idx] = 1;
        keys_[idx] = key;
        values_[idx] = value;
        ++size_;
    }

    const int64_t* find(uint64_t key) const {
        if (capacity_ == 0) {
            return nullptr;
        }
        size_t idx = mix_hash(key) & mask_;
        while (occupied_[idx]) {
            if (keys_[idx] == key) {
                return &values_[idx];
            }
            idx = (idx + 1) & mask_;
        }
        return nullptr;
    }

    size_t size() const { return size_; }

private:
    void rehash(size_t new_capacity) {
        std::vector<uint64_t> old_keys;
        std::vector<int64_t> old_values;
        std::vector<uint8_t> old_occupied;
        if (capacity_ > 0) {
            old_keys.swap(keys_);
            old_values.swap(values_);
            old_occupied.swap(occupied_);
        }
        keys_.assign(new_capacity, 0);
        values_.assign(new_capacity, 0);
        occupied_.assign(new_capacity, 0);
        capacity_ = new_capacity;
        mask_ = capacity_ - 1;
        size_ = 0;
        if (!old_keys.empty()) {
            for (size_t idx = 0; idx < old_keys.size(); ++idx) {
                if (!old_occupied[idx]) {
                    continue;
                }
                add(old_keys[idx], old_values[idx]);
            }
        }
    }

    std::vector<uint64_t> keys_;
    std::vector<int64_t> values_;
    std::vector<uint8_t> occupied_;
    size_t capacity_ = 0;
    size_t mask_ = 0;
    size_t size_ = 0;
};
}  // namespace q20

#ifdef TRACE
namespace q20_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t part_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t partsupp_scan_ns = 0;
    uint64_t supplier_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t part_rows_scanned = 0;
    uint64_t part_rows_emitted = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
    uint64_t partsupp_rows_scanned = 0;
    uint64_t partsupp_rows_emitted = 0;
    uint64_t supplier_rows_scanned = 0;
    uint64_t supplier_rows_emitted = 0;
    uint64_t join_build_rows_in = 0;
    uint64_t join_probe_rows_in = 0;
    uint64_t join_rows_emitted = 0;
    uint64_t supplier_join_build_rows_in = 0;
    uint64_t supplier_join_probe_rows_in = 0;
    uint64_t supplier_join_rows_emitted = 0;
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
    if (std::strcmp(name, "q20_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q20_part_scan") == 0) {
        trace.part_scan_ns += ns;
    } else if (std::strcmp(name, "q20_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q20_partsupp_scan") == 0) {
        trace.partsupp_scan_ns += ns;
    } else if (std::strcmp(name, "q20_supplier_scan") == 0) {
        trace.supplier_scan_ns += ns;
    } else if (std::strcmp(name, "q20_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q20_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q20_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q20_part_scan " << trace.part_scan_ns << "\n";
    std::cout << "PROFILE q20_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q20_partsupp_scan " << trace.partsupp_scan_ns << "\n";
    std::cout << "PROFILE q20_supplier_scan " << trace.supplier_scan_ns << "\n";
    std::cout << "PROFILE q20_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q20_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q20_part_scan_rows_scanned " << trace.part_rows_scanned << "\n";
    std::cout << "COUNT q20_part_scan_rows_emitted " << trace.part_rows_emitted << "\n";
    std::cout << "COUNT q20_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q20_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q20_partsupp_scan_rows_scanned " << trace.partsupp_rows_scanned
              << "\n";
    std::cout << "COUNT q20_partsupp_scan_rows_emitted " << trace.partsupp_rows_emitted
              << "\n";
    std::cout << "COUNT q20_supplier_scan_rows_scanned " << trace.supplier_rows_scanned
              << "\n";
    std::cout << "COUNT q20_supplier_scan_rows_emitted " << trace.supplier_rows_emitted
              << "\n";
    std::cout << "COUNT q20_lineitem_partsupp_join_build_rows_in "
              << trace.join_build_rows_in << "\n";
    std::cout << "COUNT q20_lineitem_partsupp_join_probe_rows_in "
              << trace.join_probe_rows_in << "\n";
    std::cout << "COUNT q20_lineitem_partsupp_join_rows_emitted "
              << trace.join_rows_emitted << "\n";
    std::cout << "COUNT q20_supplier_join_build_rows_in "
              << trace.supplier_join_build_rows_in << "\n";
    std::cout << "COUNT q20_supplier_join_probe_rows_in "
              << trace.supplier_join_probe_rows_in << "\n";
    std::cout << "COUNT q20_supplier_join_rows_emitted "
              << trace.supplier_join_rows_emitted << "\n";
    std::cout << "COUNT q20_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q20_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q20_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q20_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q20_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q20_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q20_trace

#define PROFILE_SCOPE(name) \
    ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q20_trace::record_timing)
#define TRACE_SET(field, value) (q20_trace::data().field = (value))
#define TRACE_ADD(field, value) (q20_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q20ResultRow> run_q20(const Database& db, const Q20Args& args) {
#ifdef TRACE
    q20_trace::reset();
#endif
    std::vector<Q20ResultRow> results;
    {
        PROFILE_SCOPE("q20_total");
        if (args.COLOR == "<<NULL>>" || args.DATE == "<<NULL>>" ||
            args.NATION == "<<NULL>>") {
            TRACE_SET(query_output_rows, 0);
        } else {
            const auto& part = db.part;
            const auto& lineitem = db.lineitem;
            const auto& partsupp = db.partsupp;
            const auto& supplier = db.supplier;
            const auto& nation = db.nation;

            const auto nation_it = nation.name_to_key.find(args.NATION);
            if (nation_it == nation.name_to_key.end()) {
                TRACE_SET(query_output_rows, 0);
            } else {
                const int32_t target_nation = nation_it->second;
                const int16_t target_nation_key = static_cast<int16_t>(target_nation);

                const int16_t start_offset =
                    q20::parse_date_offset(args.DATE, db.base_date_days);
                const int16_t end_offset =
                    q20::parse_date_offset_add_years(args.DATE, db.base_date_days, 1);
                const auto start_parts = q20::parse_date_parts(args.DATE);
                const auto end_parts = q20::add_years(start_parts, 1);
                int32_t start_month_index =
                    start_parts.year * 12 + (start_parts.month - 1);
                int32_t end_month_index =
                    end_parts.year * 12 + (end_parts.month - 1);
                if (end_parts.day == 1) {
                    end_month_index -= 1;
                }
                if (end_month_index < start_month_index) {
                    end_month_index = start_month_index;
                }

                const int32_t max_partkey = q20::max_value(part.partkey);
                if (max_partkey < 0) {
                    TRACE_SET(query_output_rows, 0);
                } else {
                    const size_t partkey_word_count =
                        static_cast<size_t>(max_partkey) / 64 + 1;
                    std::vector<uint64_t> partkey_color_bits(partkey_word_count, 0);
                    size_t partkey_color_count = 0;
                    TRACE_SET(part_rows_scanned, part.row_count);
                    {
                        PROFILE_SCOPE("q20_part_scan");
                        for (uint32_t row = 0; row < part.row_count; ++row) {
                            const int32_t partkey = part.partkey[row];
                            if (partkey < 0 || partkey > max_partkey) {
                                continue;
                            }
                            const std::string_view name =
                                q20::get_string_view(part.name, row);
                            if (q20::starts_with(name, args.COLOR)) {
                                const uint32_t key = static_cast<uint32_t>(partkey);
                                partkey_color_bits[key >> 6] |=
                                    (1ULL << (key & 63U));
                                ++partkey_color_count;
                                TRACE_ADD(part_rows_emitted, 1);
                            }
                        }
                    }

                    const double part_selectivity =
                        part.row_count == 0
                            ? 0.0
                            : static_cast<double>(partkey_color_count) /
                                  static_cast<double>(part.row_count);
                    const double expected_groups =
                        static_cast<double>(lineitem.row_count) * q20::kYearSelectivity *
                        part_selectivity;
                    q20::FlatHashMap sum_qty;
                    sum_qty.reserve(
                        static_cast<size_t>(std::max(16.0, expected_groups)) + 1);
                    {
                        PROFILE_SCOPE("q20_lineitem_scan");
                        const int16_t* __restrict shipdate = lineitem.shipdate.data();
                        const int32_t* __restrict partkey_ptr = lineitem.partkey.data();
                        const int32_t* __restrict suppkey_ptr = lineitem.suppkey.data();
                        const int16_t* __restrict supp_nation_ptr =
                            lineitem.supp_nationkey.data();
                        const int16_t* __restrict quantity_ptr =
                            lineitem.quantity.data();
                        const uint64_t* __restrict partkey_color_ptr =
                            partkey_color_bits.data();
                        const int32_t date_span =
                            static_cast<int32_t>(end_offset - start_offset);
                        uint64_t lineitems_scanned = 0;
                        if (lineitem.shards.empty()) {
                            lineitems_scanned = lineitem.row_count;
                            for (uint32_t row = 0; row < lineitem.row_count; ++row) {
                                const int32_t ship =
                                    static_cast<int32_t>(shipdate[row]);
                                if (static_cast<uint32_t>(ship - start_offset) >=
                                    static_cast<uint32_t>(date_span)) {
                                    continue;
                                }
                                if (supp_nation_ptr[row] != target_nation_key) {
                                    continue;
                                }
                                const uint32_t partkey =
                                    static_cast<uint32_t>(partkey_ptr[row]);
                                if (!(partkey_color_ptr[partkey >> 6] &
                                      (1ULL << (partkey & 63U)))) {
                                    continue;
                                }
                                const uint32_t suppkey =
                                    static_cast<uint32_t>(suppkey_ptr[row]);
                                TRACE_ADD(lineitem_rows_emitted, 1);
                                const uint64_t key = q20::pack_key(
                                    static_cast<int32_t>(partkey),
                                    static_cast<int32_t>(suppkey));
                                sum_qty.add(key, quantity_ptr[row]);
                            }
                        } else {
                            for (const auto& shard : lineitem.shards) {
                                if (shard.supp_nationkey != target_nation_key) {
                                    continue;
                                }
                                const int32_t shard_index =
                                    shard.year * 12 + (shard.month - 1);
                                if (shard_index < start_month_index ||
                                    shard_index > end_month_index) {
                                    continue;
                                }
                                const bool full_range =
                                    shard.min_shipdate >= start_offset &&
                                    shard.max_shipdate < end_offset;
                                if (shard.contiguous) {
                                    const uint32_t start = shard.start;
                                    const uint32_t end = shard.end;
                                    lineitems_scanned +=
                                        static_cast<uint64_t>(end - start);
                                    if (full_range) {
                                        for (uint32_t row = start; row < end; ++row) {
                                            const uint32_t partkey =
                                                static_cast<uint32_t>(partkey_ptr[row]);
                                            if (!(partkey_color_ptr[partkey >> 6] &
                                                  (1ULL << (partkey & 63U)))) {
                                                continue;
                                            }
                                            const uint32_t suppkey =
                                                static_cast<uint32_t>(suppkey_ptr[row]);
                                            TRACE_ADD(lineitem_rows_emitted, 1);
                                            const uint64_t key =
                                                q20::pack_key(static_cast<int32_t>(partkey),
                                                              static_cast<int32_t>(suppkey));
                                            sum_qty.add(key, quantity_ptr[row]);
                                        }
                                    } else {
                                        for (uint32_t row = start; row < end; ++row) {
                                            const int32_t ship =
                                                static_cast<int32_t>(shipdate[row]);
                                            if (static_cast<uint32_t>(ship -
                                                                      start_offset) >=
                                                static_cast<uint32_t>(date_span)) {
                                                continue;
                                            }
                                            const uint32_t partkey =
                                                static_cast<uint32_t>(partkey_ptr[row]);
                                            if (!(partkey_color_ptr[partkey >> 6] &
                                                  (1ULL << (partkey & 63U)))) {
                                                continue;
                                            }
                                            const uint32_t suppkey =
                                                static_cast<uint32_t>(suppkey_ptr[row]);
                                            TRACE_ADD(lineitem_rows_emitted, 1);
                                            const uint64_t key =
                                                q20::pack_key(static_cast<int32_t>(partkey),
                                                              static_cast<int32_t>(suppkey));
                                            sum_qty.add(key, quantity_ptr[row]);
                                        }
                                    }
                                } else {
                                    const uint32_t* __restrict row_indices =
                                        shard.row_indices.data();
                                    const size_t shard_rows = shard.row_indices.size();
                                    lineitems_scanned += shard_rows;
                                    if (full_range) {
                                        for (size_t pos = 0; pos < shard_rows; ++pos) {
                                            const uint32_t row = row_indices[pos];
                                            const uint32_t partkey =
                                                static_cast<uint32_t>(partkey_ptr[row]);
                                            if (!(partkey_color_ptr[partkey >> 6] &
                                                  (1ULL << (partkey & 63U)))) {
                                                continue;
                                            }
                                            const uint32_t suppkey =
                                                static_cast<uint32_t>(suppkey_ptr[row]);
                                            TRACE_ADD(lineitem_rows_emitted, 1);
                                            const uint64_t key =
                                                q20::pack_key(static_cast<int32_t>(partkey),
                                                              static_cast<int32_t>(suppkey));
                                            sum_qty.add(key, quantity_ptr[row]);
                                        }
                                    } else {
                                        for (size_t pos = 0; pos < shard_rows; ++pos) {
                                            const uint32_t row = row_indices[pos];
                                            const int32_t ship =
                                                static_cast<int32_t>(shipdate[row]);
                                            if (static_cast<uint32_t>(ship -
                                                                      start_offset) >=
                                                static_cast<uint32_t>(date_span)) {
                                                continue;
                                            }
                                            const uint32_t partkey =
                                                static_cast<uint32_t>(partkey_ptr[row]);
                                            if (!(partkey_color_ptr[partkey >> 6] &
                                                  (1ULL << (partkey & 63U)))) {
                                                continue;
                                            }
                                            const uint32_t suppkey =
                                                static_cast<uint32_t>(suppkey_ptr[row]);
                                            TRACE_ADD(lineitem_rows_emitted, 1);
                                            const uint64_t key =
                                                q20::pack_key(static_cast<int32_t>(partkey),
                                                              static_cast<int32_t>(suppkey));
                                            sum_qty.add(key, quantity_ptr[row]);
                                        }
                                    }
                                }
                            }
                        }
                        TRACE_SET(lineitem_rows_scanned, lineitems_scanned);
                    }
                    TRACE_SET(agg_rows_in, q20_trace::data().lineitem_rows_emitted);

                    const int32_t max_suppkey =
                        q20::max_value(supplier.suppkey);
                    std::vector<uint8_t> qualified_suppkeys;
                    size_t qualified_count = 0;
                    if (max_suppkey >= 0) {
                        qualified_suppkeys.assign(
                            static_cast<size_t>(max_suppkey) + 1, 0);
                    }
                    TRACE_SET(partsupp_rows_scanned, partsupp.row_count);
                    TRACE_SET(join_build_rows_in, sum_qty.size());
                    {
                        PROFILE_SCOPE("q20_partsupp_scan");
                        const uint64_t* __restrict partkey_color_ptr =
                            partkey_color_bits.data();
                        const int16_t* __restrict supp_nation_map =
                            supplier.nationkey_by_suppkey.data();
                        const size_t supp_nation_map_size =
                            supplier.nationkey_by_suppkey.size();
                        for (uint32_t row = 0; row < partsupp.row_count; ++row) {
                            const uint32_t partkey =
                                static_cast<uint32_t>(partsupp.partkey[row]);
                            if (!(partkey_color_ptr[partkey >> 6] &
                                  (1ULL << (partkey & 63U)))) {
                                continue;
                            }
                            const uint32_t suppkey =
                                static_cast<uint32_t>(partsupp.suppkey[row]);
                            if (suppkey >= supp_nation_map_size ||
                                supp_nation_map[suppkey] != target_nation_key) {
                                continue;
                            }
                            const uint64_t key = q20::pack_key(
                                static_cast<int32_t>(partkey),
                                static_cast<int32_t>(suppkey));
                            const int64_t* sum_ptr = sum_qty.find(key);
                            if (sum_ptr == nullptr) {
                                continue;
                            }
                            TRACE_ADD(join_probe_rows_in, 1);
                            const int64_t sum_qty_scaled = *sum_ptr;
                            const int64_t avail_scaled =
                                static_cast<int64_t>(partsupp.availqty[row]) * 2 *
                                q20::kPriceScale;
                            if (avail_scaled > sum_qty_scaled) {
                                if (suppkey <= static_cast<uint32_t>(max_suppkey) &&
                                    !qualified_suppkeys[static_cast<size_t>(suppkey)]) {
                                    qualified_suppkeys[static_cast<size_t>(suppkey)] = 1;
                                    ++qualified_count;
                                    TRACE_ADD(partsupp_rows_emitted, 1);
                                }
                            }
                        }
                    }
                    TRACE_SET(join_rows_emitted, qualified_count);

                    {
                        PROFILE_SCOPE("q20_agg_finalize");
                        TRACE_SET(groups_created, sum_qty.size());
                        TRACE_SET(agg_rows_emitted, sum_qty.size());
                    }

                    results.reserve(qualified_count);
                    TRACE_SET(supplier_rows_scanned, supplier.row_count);
                    TRACE_SET(supplier_join_build_rows_in, qualified_count);
                    TRACE_SET(supplier_join_probe_rows_in, supplier.row_count);
                    {
                        PROFILE_SCOPE("q20_supplier_scan");
                        for (uint32_t row = 0; row < supplier.row_count; ++row) {
                            const int32_t suppkey = supplier.suppkey[row];
                            if (suppkey < 0 || suppkey > max_suppkey ||
                                qualified_suppkeys.empty() ||
                                !qualified_suppkeys[static_cast<size_t>(suppkey)]) {
                                continue;
                            }
                            if (supplier.nationkey[row] != target_nation) {
                                continue;
                            }
                            TRACE_ADD(supplier_rows_emitted, 1);
                            TRACE_ADD(supplier_join_rows_emitted, 1);
                            Q20ResultRow result;
                            result.s_name =
                                std::string(q20::get_string_view(supplier.name, row));
                            result.s_address =
                                std::string(q20::get_string_view(supplier.address,
                                                                 row));
                            results.push_back(std::move(result));
                        }
                    }

                    TRACE_SET(sort_rows_in, results.size());
                    {
                        PROFILE_SCOPE("q20_sort");
                        std::sort(results.begin(), results.end(),
                                  [](const Q20ResultRow& a, const Q20ResultRow& b) {
                                      return a.s_name < b.s_name;
                                  });
                    }
                    TRACE_SET(sort_rows_out, results.size());
                    TRACE_SET(query_output_rows, results.size());
                }
            }
        }
    }

#ifdef TRACE
    q20_trace::emit();
#endif
    return results;
}

void write_q20_csv(const std::string& filename, const std::vector<Q20ResultRow>& rows) {
    std::ofstream out(filename);
    out << "s_name,s_address\n";
    for (const auto& row : rows) {
        out << q20::escape_csv_field(row.s_name) << ','
            << q20::escape_csv_field(row.s_address) << "\n";
    }
}