#include "query_q16.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "trace_utils.hpp"

#if defined(__GNUC__)
#define Q16_LIKELY(x) __builtin_expect(!!(x), 1)
#define Q16_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define Q16_LIKELY(x) (x)
#define Q16_UNLIKELY(x) (x)
#endif

namespace q16 {
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

int32_t max_value(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return -1;
    }
    return *std::max_element(values.begin(), values.end());
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

bool has_customer_complaints(std::string_view comment) {
    constexpr std::string_view kCustomer = "Customer";
    constexpr std::string_view kComplaints = "Complaints";
    const size_t customer_pos = comment.find(kCustomer);
    if (customer_pos == std::string_view::npos) {
        return false;
    }
    const size_t complaints_pos = comment.find(kComplaints, customer_pos + kCustomer.size());
    return complaints_pos != std::string_view::npos;
}
}  // namespace q16

#ifdef TRACE
namespace q16_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t part_scan_ns = 0;
    uint64_t supplier_scan_ns = 0;
    uint64_t partsupp_scan_ns = 0;
    uint64_t entries_sort_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t results_sort_ns = 0;
    uint64_t part_rows_scanned = 0;
    uint64_t part_rows_emitted = 0;
    uint64_t supplier_rows_scanned = 0;
    uint64_t supplier_rows_emitted = 0;
    uint64_t partsupp_rows_scanned = 0;
    uint64_t partsupp_rows_emitted = 0;
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
    if (std::strcmp(name, "q16_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q16_part_scan") == 0) {
        trace.part_scan_ns += ns;
    } else if (std::strcmp(name, "q16_supplier_scan") == 0) {
        trace.supplier_scan_ns += ns;
    } else if (std::strcmp(name, "q16_partsupp_scan") == 0) {
        trace.partsupp_scan_ns += ns;
    } else if (std::strcmp(name, "q16_entries_sort") == 0) {
        trace.entries_sort_ns += ns;
    } else if (std::strcmp(name, "q16_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q16_results_sort") == 0) {
        trace.results_sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q16_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q16_part_scan " << trace.part_scan_ns << "\n";
    std::cout << "PROFILE q16_supplier_scan " << trace.supplier_scan_ns << "\n";
    std::cout << "PROFILE q16_partsupp_scan " << trace.partsupp_scan_ns << "\n";
    std::cout << "PROFILE q16_entries_sort " << trace.entries_sort_ns << "\n";
    std::cout << "PROFILE q16_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q16_results_sort " << trace.results_sort_ns << "\n";
    std::cout << "COUNT q16_part_scan_rows_scanned " << trace.part_rows_scanned << "\n";
    std::cout << "COUNT q16_part_scan_rows_emitted " << trace.part_rows_emitted << "\n";
    std::cout << "COUNT q16_supplier_scan_rows_scanned " << trace.supplier_rows_scanned
              << "\n";
    std::cout << "COUNT q16_supplier_scan_rows_emitted " << trace.supplier_rows_emitted
              << "\n";
    std::cout << "COUNT q16_partsupp_scan_rows_scanned " << trace.partsupp_rows_scanned
              << "\n";
    std::cout << "COUNT q16_partsupp_scan_rows_emitted " << trace.partsupp_rows_emitted
              << "\n";
    std::cout << "COUNT q16_part_partsupp_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q16_part_partsupp_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q16_part_partsupp_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q16_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q16_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q16_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q16_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q16_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q16_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q16_trace

#define PROFILE_SCOPE(name) \
    ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q16_trace::record_timing)
#define TRACE_SET(field, value) (q16_trace::data().field = (value))
#define TRACE_ADD(field, value) (q16_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q16ResultRow> run_q16(const Database& db, const Q16Args& args) {
    const auto& part = db.part;
    const auto& partsupp = db.partsupp;
    const auto& supplier = db.supplier;

#ifdef TRACE
    q16_trace::reset();
#endif
    std::vector<Q16ResultRow> results;
    struct AggRow {
        uint32_t brand_code = 0;
        uint32_t type_code = 0;
        int32_t size = 0;
        int64_t supplier_cnt = 0;
    };
    std::vector<AggRow> agg_rows;
    {
        PROFILE_SCOPE("q16_total");
        std::vector<int32_t> sizes;
        sizes.reserve(8);
        int32_t max_size = -1;
        auto add_size = [&sizes, &max_size](const std::string& value) {
            if (value == "<<NULL>>") {
                return;
            }
            int32_t parsed = 0;
            for (char ch : value) {
                if (ch < '0' || ch > '9') {
                    return;
                }
                parsed = parsed * 10 + (ch - '0');
            }
            sizes.push_back(parsed);
            if (parsed > max_size) {
                max_size = parsed;
            }
        };
        add_size(args.SIZE1);
        add_size(args.SIZE2);
        add_size(args.SIZE3);
        add_size(args.SIZE4);
        add_size(args.SIZE5);
        add_size(args.SIZE6);
        add_size(args.SIZE7);
        add_size(args.SIZE8);

        if (sizes.empty()) {
#ifdef TRACE
            TRACE_SET(query_output_rows, 0);
            q16_trace::emit();
#endif
            return {};
        }
        constexpr uint32_t kMaxPartSize = 50;
        uint64_t size_mask = 0;
        for (int32_t size : sizes) {
            if (size >= 0 && size <= static_cast<int32_t>(kMaxPartSize)) {
                size_mask |= 1ULL << static_cast<uint32_t>(size);
            }
        }

        const int32_t max_partkey =
            part.row_count ? static_cast<int32_t>(part.row_count) : -1;
        if (max_partkey < 0) {
#ifdef TRACE
            TRACE_SET(query_output_rows, 0);
            q16_trace::emit();
#endif
            return {};
        }
        std::vector<int16_t> partkey_to_group(static_cast<size_t>(max_partkey) + 1, -1);
        std::vector<uint8_t> type_prefix_banned(part.type.dictionary.size(), 0);
        for (size_t idx = 0; idx < part.type.dictionary.size(); ++idx) {
            if (q16::starts_with(part.type.dictionary[idx], args.TYPE)) {
                type_prefix_banned[idx] = 1;
            }
        }
        struct GroupInfo {
            uint32_t brand_code = 0;
            uint32_t type_code = 0;
            int32_t size = 0;
        };
        const uint32_t brand_count = static_cast<uint32_t>(part.brand.dictionary.size());
        const uint32_t type_count = static_cast<uint32_t>(part.type.dictionary.size());
        const uint32_t group_stride = static_cast<uint32_t>(max_size) + 1;
        std::vector<int16_t> group_lut(
            static_cast<size_t>(brand_count) * type_count * group_stride, -1);
        std::vector<uint32_t> brand_type_base(
            static_cast<size_t>(brand_count) * type_count);
        for (uint32_t brand_code = 0; brand_code < brand_count; ++brand_code) {
            const uint32_t base = brand_code * type_count;
            for (uint32_t type_code = 0; type_code < type_count; ++type_code) {
                brand_type_base[base + type_code] = (base + type_code) * group_stride;
            }
        }
        std::vector<GroupInfo> groups;
        groups.reserve(65536);
        int32_t banned_brand_code = -1;
        for (size_t idx = 0; idx < part.brand.dictionary.size(); ++idx) {
            if (part.brand.dictionary[idx] == args.BRAND) {
                banned_brand_code = static_cast<int32_t>(idx);
                break;
            }
        }
        uint64_t eligible_parts = 0;
        {
            PROFILE_SCOPE("q16_part_scan");
            const uint64_t size_mask_local = size_mask;
            const auto* __restrict type_codes = part.type.codes.data();
            const auto* __restrict brand_codes = part.brand.codes.data();
            const auto* __restrict size_ptr = part.size.data();
            const auto* __restrict partkey_ptr = part.partkey.data();
            const auto* __restrict base_ptr = brand_type_base.data();
            auto* __restrict partkey_group = partkey_to_group.data();
            auto* __restrict group_lut_ptr = group_lut.data();
            const auto* __restrict type_banned = type_prefix_banned.data();
            if (banned_brand_code >= 0) {
                const uint32_t banned_brand =
                    static_cast<uint32_t>(banned_brand_code);
                for (uint32_t row = 0; row < part.row_count; ++row) {
                    const uint32_t brand_code = brand_codes[row];
                    if (Q16_UNLIKELY(brand_code == banned_brand)) {
                        continue;
                    }
                    const uint32_t size_u = static_cast<uint32_t>(size_ptr[row]);
                    if (Q16_LIKELY((size_mask_local & (1ULL << size_u)) == 0)) {
                        continue;
                    }
                    const uint32_t type_code = type_codes[row];
                    if (Q16_UNLIKELY(type_banned[type_code])) {
                        continue;
                    }
                    const size_t base_index =
                        static_cast<size_t>(brand_code) * type_count + type_code;
                    const size_t group_index =
                        static_cast<size_t>(base_ptr[base_index]) + size_u;
                    int32_t group_id = static_cast<int32_t>(group_lut_ptr[group_index]);
                    if (group_id < 0) {
                        group_id = static_cast<int32_t>(groups.size());
                        groups.push_back(GroupInfo{
                            brand_code,
                            type_code,
                            static_cast<int32_t>(size_u)});
                        group_lut_ptr[group_index] = static_cast<int16_t>(group_id);
                    }
                    partkey_group[partkey_ptr[row]] = static_cast<int16_t>(group_id);
                    ++eligible_parts;
                }
            } else {
                for (uint32_t row = 0; row < part.row_count; ++row) {
                    const uint32_t size_u = static_cast<uint32_t>(size_ptr[row]);
                    if (Q16_LIKELY((size_mask_local & (1ULL << size_u)) == 0)) {
                        continue;
                    }
                    const uint32_t type_code = type_codes[row];
                    if (Q16_UNLIKELY(type_banned[type_code])) {
                        continue;
                    }
                    const uint32_t brand_code = brand_codes[row];
                    const size_t base_index =
                        static_cast<size_t>(brand_code) * type_count + type_code;
                    const size_t group_index =
                        static_cast<size_t>(base_ptr[base_index]) + size_u;
                    int32_t group_id = static_cast<int32_t>(group_lut_ptr[group_index]);
                    if (group_id < 0) {
                        group_id = static_cast<int32_t>(groups.size());
                        groups.push_back(GroupInfo{
                            brand_code,
                            type_code,
                            static_cast<int32_t>(size_u)});
                        group_lut_ptr[group_index] = static_cast<int16_t>(group_id);
                    }
                    partkey_group[partkey_ptr[row]] = static_cast<int16_t>(group_id);
                    ++eligible_parts;
                }
            }
        }
        TRACE_ADD(part_rows_scanned, part.row_count);
        TRACE_ADD(part_rows_emitted, eligible_parts);
        TRACE_ADD(join_build_rows_in, eligible_parts);

        const int32_t max_suppkey =
            supplier.row_count ? static_cast<int32_t>(supplier.row_count) : -1;
        if (max_suppkey < 0) {
#ifdef TRACE
            TRACE_SET(query_output_rows, 0);
            q16_trace::emit();
#endif
            return {};
        }
        std::vector<uint8_t> banned_suppkey(static_cast<size_t>(max_suppkey) + 1, 0);
        uint64_t banned_count = 0;
        {
            PROFILE_SCOPE("q16_supplier_scan");
            for (uint32_t row = 0; row < supplier.row_count; ++row) {
                const int32_t suppkey = supplier.suppkey[row];
                const std::string_view comment = q16::get_string_view(supplier.comment, row);
                if (Q16_UNLIKELY(q16::has_customer_complaints(comment))) {
                    banned_suppkey[static_cast<size_t>(suppkey)] = 1;
                    ++banned_count;
                }
            }
        }
        TRACE_ADD(supplier_rows_scanned, supplier.row_count);
        TRACE_ADD(supplier_rows_emitted, banned_count);

        std::vector<uint16_t> entry_groups;
        std::vector<int32_t> entry_suppkeys;
        entry_groups.reserve(static_cast<size_t>(partsupp.row_count / 6 + 1));
        entry_suppkeys.reserve(static_cast<size_t>(partsupp.row_count / 6 + 1));
        std::vector<uint16_t> group_bucket_counts(groups.size(), 0);
        uint64_t filtered_rows = 0;

        {
            PROFILE_SCOPE("q16_partsupp_scan");
            const auto* __restrict partkey_ptr = partsupp.partkey.data();
            const auto* __restrict suppkey_ptr = partsupp.suppkey.data();
            const auto* __restrict partkey_group = partkey_to_group.data();
            const auto* __restrict banned_ptr = banned_suppkey.data();
            auto* __restrict bucket_counts = group_bucket_counts.data();
            const uint32_t part_row_count = static_cast<uint32_t>(part.row_count);
            const uint32_t partsupp_row_count = static_cast<uint32_t>(partsupp.row_count);
            const uint32_t rows_per_part =
                (part_row_count > 0 && partsupp_row_count % part_row_count == 0)
                    ? (partsupp_row_count / part_row_count)
                    : 0;
            if (rows_per_part > 0) {
                for (uint32_t part_idx = 0; part_idx < part_row_count; ++part_idx) {
                    const int32_t group_id =
                        static_cast<int32_t>(partkey_group[part_idx + 1]);
                    if (Q16_LIKELY(group_id < 0)) {
                        continue;
                    }
                    const uint32_t base = part_idx * rows_per_part;
                    for (uint32_t offset = 0; offset < rows_per_part; ++offset) {
                        const uint32_t row = base + offset;
                        const int32_t suppkey = suppkey_ptr[row];
                        if (Q16_UNLIKELY(banned_ptr[suppkey])) {
                            continue;
                        }
                        ++filtered_rows;
                        bucket_counts[group_id] += 1;
                        entry_groups.push_back(static_cast<uint16_t>(group_id));
                        entry_suppkeys.push_back(suppkey);
                    }
                }
            } else {
                for (uint32_t row = 0; row < partsupp_row_count; ++row) {
                    const int32_t group_id =
                        static_cast<int32_t>(partkey_group[partkey_ptr[row]]);
                    if (Q16_LIKELY(group_id < 0)) {
                        continue;
                    }
                    const int32_t suppkey = suppkey_ptr[row];
                    if (Q16_UNLIKELY(banned_ptr[suppkey])) {
                        continue;
                    }
                    ++filtered_rows;
                    bucket_counts[group_id] += 1;
                    entry_groups.push_back(static_cast<uint16_t>(group_id));
                    entry_suppkeys.push_back(suppkey);
                }
            }
        }
        TRACE_ADD(partsupp_rows_scanned, partsupp.row_count);
        TRACE_ADD(partsupp_rows_emitted, filtered_rows);
        TRACE_ADD(join_probe_rows_in, partsupp.row_count);
        TRACE_ADD(join_rows_emitted, filtered_rows);

        if (filtered_rows == 0) {
#ifdef TRACE
            TRACE_SET(query_output_rows, 0);
            q16_trace::emit();
#endif
            return {};
        }

        {
            PROFILE_SCOPE("q16_agg_finalize");
            std::vector<uint32_t> group_offsets(groups.size() + 1, 0);
            for (size_t idx = 0; idx < groups.size(); ++idx) {
                group_offsets[idx + 1] = group_offsets[idx] + group_bucket_counts[idx];
            }
            std::vector<uint32_t> write_offsets = group_offsets;
            const size_t entry_count = entry_suppkeys.size();
            std::vector<int32_t> grouped_suppkeys(entry_count);
            const auto* __restrict entry_group_ptr = entry_groups.data();
            const auto* __restrict entry_supp_ptr = entry_suppkeys.data();
            for (size_t idx = 0; idx < entry_count; ++idx) {
                const uint32_t group_id = entry_group_ptr[idx];
                const uint32_t pos = write_offsets[group_id]++;
                grouped_suppkeys[pos] = entry_supp_ptr[idx];
            }
            std::vector<int16_t> suppkey_seen(static_cast<size_t>(max_suppkey) + 1, -1);
            auto* seen_ptr = suppkey_seen.data();
            const auto* suppkeys_ptr = grouped_suppkeys.data();
            for (size_t idx = 0; idx < groups.size(); ++idx) {
                const uint32_t start = group_offsets[idx];
                const uint32_t end = group_offsets[idx + 1];
                if (start == end) {
                    continue;
                }
                const int16_t group_stamp = static_cast<int16_t>(idx);
                int64_t supplier_cnt = 0;
                for (uint32_t pos = start; pos < end; ++pos) {
                    const int32_t suppkey = suppkeys_ptr[pos];
                    if (seen_ptr[suppkey] != group_stamp) {
                        seen_ptr[suppkey] = group_stamp;
                        ++supplier_cnt;
                    }
                }
                const auto& group = groups[idx];
                agg_rows.push_back(AggRow{
                    group.brand_code,
                    group.type_code,
                    group.size,
                    supplier_cnt});
            }
        }
        TRACE_ADD(agg_rows_in, filtered_rows);
        TRACE_ADD(groups_created, agg_rows.size());
        TRACE_ADD(agg_rows_emitted, agg_rows.size());

        {
            PROFILE_SCOPE("q16_results_sort");
            std::vector<uint32_t> brand_rank(part.brand.dictionary.size(), 0);
            std::vector<uint32_t> type_rank(part.type.dictionary.size(), 0);
            std::vector<size_t> brand_idx(part.brand.dictionary.size());
            std::vector<size_t> type_idx(part.type.dictionary.size());
            std::iota(brand_idx.begin(), brand_idx.end(), 0);
            std::iota(type_idx.begin(), type_idx.end(), 0);
            std::sort(brand_idx.begin(), brand_idx.end(),
                      [&](size_t a, size_t b) {
                          return part.brand.dictionary[a] < part.brand.dictionary[b];
                      });
            std::sort(type_idx.begin(), type_idx.end(),
                      [&](size_t a, size_t b) {
                          return part.type.dictionary[a] < part.type.dictionary[b];
                      });
            for (size_t idx = 0; idx < brand_idx.size(); ++idx) {
                brand_rank[brand_idx[idx]] = static_cast<uint32_t>(idx);
            }
            for (size_t idx = 0; idx < type_idx.size(); ++idx) {
                type_rank[type_idx[idx]] = static_cast<uint32_t>(idx);
            }
            std::sort(agg_rows.begin(), agg_rows.end(),
                      [&](const AggRow& a, const AggRow& b) {
                          if (a.supplier_cnt != b.supplier_cnt) {
                              return a.supplier_cnt > b.supplier_cnt;
                          }
                          const uint32_t brand_a = brand_rank[a.brand_code];
                          const uint32_t brand_b = brand_rank[b.brand_code];
                          if (brand_a != brand_b) {
                              return brand_a < brand_b;
                          }
                          const uint32_t type_a = type_rank[a.type_code];
                          const uint32_t type_b = type_rank[b.type_code];
                          if (type_a != type_b) {
                              return type_a < type_b;
                          }
                          return a.size < b.size;
                      });
        }
        TRACE_ADD(sort_rows_in, agg_rows.size());
        TRACE_ADD(sort_rows_out, agg_rows.size());
        results.reserve(agg_rows.size());
        for (const auto& row : agg_rows) {
            Q16ResultRow out;
            out.p_brand = part.brand.dictionary[row.brand_code];
            out.p_type = part.type.dictionary[row.type_code];
            out.p_size = row.size;
            out.supplier_cnt = row.supplier_cnt;
            results.push_back(std::move(out));
        }
        TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q16_trace::emit();
#endif
    return results;
}

void write_q16_csv(const std::string& filename, const std::vector<Q16ResultRow>& rows) {
    std::ofstream out(filename);
    out << "p_brand,p_type,p_size,supplier_cnt\n";
    for (const auto& row : rows) {
        out << q16::escape_csv_field(row.p_brand) << ','
            << q16::escape_csv_field(row.p_type) << ','
            << row.p_size << ','
            << row.supplier_cnt << "\n";
    }
}