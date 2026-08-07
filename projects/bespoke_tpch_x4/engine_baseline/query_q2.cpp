#include "query_q2.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <unordered_map>

#include "trace_utils.hpp"

namespace {
constexpr int32_t kQ2PriceScale = 100;

std::string_view get_string_view(const StringColumn& column, size_t row) {
    const uint32_t start = column.offsets[row];
    const uint32_t end = column.offsets[row + 1];
    return std::string_view(column.data.data() + start, end - start);
}

void write_csv_field_q2(std::ostream& out, std::string_view value) {
    out.put('"');
    for (char ch : value) {
        if (ch == '"') {
            out.put('\\');
            out.put('"');
        } else {
            out.put(ch);
        }
    }
    out.put('"');
}

}  // namespace

#ifdef TRACE
namespace q2_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t part_scan_ns = 0;
    uint64_t supplier_scan_ns = 0;
    uint64_t nation_scan_ns = 0;
    uint64_t partsupp_min_scan_ns = 0;
    uint64_t partsupp_result_scan_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t part_rows_scanned = 0;
    uint64_t part_rows_emitted = 0;
    uint64_t supplier_rows_scanned = 0;
    uint64_t supplier_rows_emitted = 0;
    uint64_t nation_rows_scanned = 0;
    uint64_t nation_rows_emitted = 0;
    uint64_t partsupp_min_rows_scanned = 0;
    uint64_t partsupp_min_rows_emitted = 0;
    uint64_t partsupp_result_rows_scanned = 0;
    uint64_t partsupp_result_rows_emitted = 0;
    uint64_t join_min_build_rows_in = 0;
    uint64_t join_min_probe_rows_in = 0;
    uint64_t join_min_rows_emitted = 0;
    uint64_t join_result_build_rows_in = 0;
    uint64_t join_result_probe_rows_in = 0;
    uint64_t join_result_rows_emitted = 0;
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
    if (std::strcmp(name, "q2_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q2_part_scan") == 0) {
        trace.part_scan_ns += ns;
    } else if (std::strcmp(name, "q2_supplier_scan") == 0) {
        trace.supplier_scan_ns += ns;
    } else if (std::strcmp(name, "q2_nation_scan") == 0) {
        trace.nation_scan_ns += ns;
    } else if (std::strcmp(name, "q2_partsupp_min_scan") == 0) {
        trace.partsupp_min_scan_ns += ns;
    } else if (std::strcmp(name, "q2_partsupp_result_scan") == 0) {
        trace.partsupp_result_scan_ns += ns;
    } else if (std::strcmp(name, "q2_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q2_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q2_part_scan " << trace.part_scan_ns << "\n";
    std::cout << "PROFILE q2_supplier_scan " << trace.supplier_scan_ns << "\n";
    std::cout << "PROFILE q2_nation_scan " << trace.nation_scan_ns << "\n";
    std::cout << "PROFILE q2_partsupp_min_scan " << trace.partsupp_min_scan_ns << "\n";
    std::cout << "PROFILE q2_partsupp_result_scan " << trace.partsupp_result_scan_ns << "\n";
    std::cout << "PROFILE q2_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q2_part_scan_rows_scanned " << trace.part_rows_scanned << "\n";
    std::cout << "COUNT q2_part_scan_rows_emitted " << trace.part_rows_emitted << "\n";
    std::cout << "COUNT q2_supplier_scan_rows_scanned " << trace.supplier_rows_scanned << "\n";
    std::cout << "COUNT q2_supplier_scan_rows_emitted " << trace.supplier_rows_emitted << "\n";
    std::cout << "COUNT q2_nation_scan_rows_scanned " << trace.nation_rows_scanned << "\n";
    std::cout << "COUNT q2_nation_scan_rows_emitted " << trace.nation_rows_emitted << "\n";
    std::cout << "COUNT q2_partsupp_min_scan_rows_scanned " << trace.partsupp_min_rows_scanned
              << "\n";
    std::cout << "COUNT q2_partsupp_min_scan_rows_emitted " << trace.partsupp_min_rows_emitted
              << "\n";
    std::cout << "COUNT q2_partsupp_result_scan_rows_scanned "
              << trace.partsupp_result_rows_scanned << "\n";
    std::cout << "COUNT q2_partsupp_result_scan_rows_emitted "
              << trace.partsupp_result_rows_emitted << "\n";
    std::cout << "COUNT q2_partsupp_min_join_build_rows_in " << trace.join_min_build_rows_in
              << "\n";
    std::cout << "COUNT q2_partsupp_min_join_probe_rows_in " << trace.join_min_probe_rows_in
              << "\n";
    std::cout << "COUNT q2_partsupp_min_join_rows_emitted " << trace.join_min_rows_emitted
              << "\n";
    std::cout << "COUNT q2_partsupp_result_join_build_rows_in " << trace.join_result_build_rows_in
              << "\n";
    std::cout << "COUNT q2_partsupp_result_join_probe_rows_in " << trace.join_result_probe_rows_in
              << "\n";
    std::cout << "COUNT q2_partsupp_result_join_rows_emitted " << trace.join_result_rows_emitted
              << "\n";
    std::cout << "COUNT q2_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q2_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q2_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q2_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q2_trace::record_timing)
#define TRACE_SET(field, value) (q2_trace::data().field = (value))
#define TRACE_ADD(field, value) (q2_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q2ResultRow> run_q2(const Database& db, const Q2Args& args) {
    std::vector<Q2ResultRow> results;
#ifdef TRACE
    q2_trace::reset();
#endif
    {
        PROFILE_SCOPE("q2_total");
        const int32_t target_size = std::stoi(args.SIZE);
        const std::string_view type_suffix = args.TYPE;
        const auto region_it = db.region.name_to_key.find(args.REGION);
        if (region_it == db.region.name_to_key.end()) {
            return {};
        }
        const int32_t region_key = region_it->second;

        const auto& part = db.part;
        const auto& supplier = db.supplier;
        const auto& partsupp = db.partsupp;
        const auto& nation = db.nation;

        const uint32_t part_row_count = static_cast<uint32_t>(part.row_count);
        const uint32_t supplier_row_count = static_cast<uint32_t>(supplier.row_count);
        const uint32_t nation_row_count = static_cast<uint32_t>(nation.rows.size());

        std::vector<uint8_t> nation_in_region(nation_row_count, 0);
        std::vector<std::string_view> nation_name(nation_row_count);
        {
            PROFILE_SCOPE("q2_nation_scan");
            TRACE_SET(nation_rows_scanned, nation.rows.size());
            uint64_t nation_in_region_count = 0;
            for (const auto& row : nation.rows) {
                if (row.nationkey < 0 ||
                    static_cast<uint32_t>(row.nationkey) >= nation_row_count) {
                    continue;
                }
                const uint32_t idx = static_cast<uint32_t>(row.nationkey);
                nation_name[idx] = row.name;
                if (row.regionkey == region_key) {
                    nation_in_region[idx] = 1;
                    nation_in_region_count += 1;
                }
            }
            TRACE_SET(nation_rows_emitted, nation_in_region_count);
        }

        std::vector<uint8_t> supplier_in_region(supplier_row_count, 0);
        std::vector<std::string_view> supplier_nation_name(supplier_row_count);
        uint64_t supplier_in_region_count = 0;
        {
            PROFILE_SCOPE("q2_supplier_scan");
            TRACE_SET(supplier_rows_scanned, supplier.row_count);
            const int32_t* supplier_nationkey = supplier.nationkey.data();
            const uint8_t* nation_ok = nation_in_region.data();
            const std::string_view* nation_name_ptr = nation_name.data();
            uint8_t* supplier_ok = supplier_in_region.data();
            for (uint32_t i = 0; i < supplier.row_count; ++i) {
                const uint32_t nkey = static_cast<uint32_t>(supplier_nationkey[i]);
                if (nkey < nation_row_count) {
                    supplier_nation_name[i] = nation_name_ptr[nkey];
                    const uint8_t ok = nation_ok[nkey];
                    supplier_ok[i] = ok;
                    supplier_in_region_count += ok;
                }
            }
            TRACE_SET(supplier_rows_emitted, supplier_in_region_count);
        }

        std::vector<uint32_t> part_ok_list;
        part_ok_list.reserve(part_row_count / 8);
        uint64_t part_ok_count = 0;
        {
            PROFILE_SCOPE("q2_part_scan");
            TRACE_SET(part_rows_scanned, part.row_count);
            const size_t suffix_len = type_suffix.size();
            const char* suffix_data = type_suffix.data();
            std::vector<uint8_t> type_suffix_ok(part.type.dictionary.size(), 0);
            for (size_t idx = 0; idx < part.type.dictionary.size(); ++idx) {
                const auto& type_value = part.type.dictionary[idx];
                if (type_value.size() < suffix_len) {
                    continue;
                }
                if (std::memcmp(type_value.data() + (type_value.size() - suffix_len),
                                suffix_data, suffix_len) == 0) {
                    type_suffix_ok[idx] = 1;
                }
            }
            const auto* type_codes = part.type.codes.data();
            const int32_t* size_ptr = part.size.data();
            const uint8_t* type_ok_ptr = type_suffix_ok.data();
            uint32_t i = 0;
            const uint32_t row_count = static_cast<uint32_t>(part.row_count);
            for (; i + 4 <= row_count; i += 4) {
                const uint16_t t0 = type_codes[i];
                const uint16_t t1 = type_codes[i + 1];
                const uint16_t t2 = type_codes[i + 2];
                const uint16_t t3 = type_codes[i + 3];
                if (size_ptr[i] == target_size && type_ok_ptr[t0]) {
                    part_ok_list.push_back(i);
                    part_ok_count += 1;
                }
                if (size_ptr[i + 1] == target_size && type_ok_ptr[t1]) {
                    part_ok_list.push_back(i + 1);
                    part_ok_count += 1;
                }
                if (size_ptr[i + 2] == target_size && type_ok_ptr[t2]) {
                    part_ok_list.push_back(i + 2);
                    part_ok_count += 1;
                }
                if (size_ptr[i + 3] == target_size && type_ok_ptr[t3]) {
                    part_ok_list.push_back(i + 3);
                    part_ok_count += 1;
                }
            }
            for (; i < row_count; ++i) {
                if (size_ptr[i] == target_size && type_ok_ptr[type_codes[i]]) {
                    part_ok_list.push_back(i);
                    part_ok_count += 1;
                }
            }
            TRACE_SET(part_rows_emitted, part_ok_count);
        }

        TRACE_SET(join_min_build_rows_in, part_ok_count + supplier_in_region_count);
        const int32_t kMaxCost = std::numeric_limits<int32_t>::max();
        constexpr uint32_t kBestSupplierCapacity = 16;
        uint64_t candidate_rows = 0;
        uint64_t parts_with_min_cost = 0;
        uint64_t partsupp_rows_scanned = 0;
        uint64_t result_matches = 0;
        results.reserve(part_ok_count + 8);
        {
            PROFILE_SCOPE("q2_partsupp_min_scan");
            const int32_t* partsupp_partkey = partsupp.partkey.data();
            const int32_t* partsupp_suppkey = partsupp.suppkey.data();
            const int32_t* partsupp_cost = partsupp.supplycost.data();
            const uint8_t* supplier_ok_ptr = supplier_in_region.data();
            const int32_t* partsupp_partkey_end =
                partsupp_partkey + partsupp.row_count;
            const int32_t* search_ptr = partsupp_partkey;
            const int32_t* supplier_acctbal = supplier.acctbal.data();
            const uint16_t* part_mfgr_codes = part.mfgr.codes.data();
            const uint32_t partsupp_row_count = static_cast<uint32_t>(partsupp.row_count);
            const uint32_t stride =
                (part_row_count > 0 && partsupp_row_count % part_row_count == 0)
                    ? (partsupp_row_count / part_row_count)
                    : 0;
            std::vector<uint32_t> best_supp_overflow;
            best_supp_overflow.reserve(8);
            for (const uint32_t prow : part_ok_list) {
                const int32_t partkey = static_cast<int32_t>(prow + 1);
                const int32_t* range_start = nullptr;
                const int32_t* range_end = nullptr;
                if (stride > 0) {
                    const uint32_t base = prow * stride;
                    if (base + stride <= partsupp_row_count) {
                        const int32_t* candidate = partsupp_partkey + base;
                        const bool start_ok = (base == 0) || (candidate[-1] != partkey);
                        const bool end_ok =
                            (base + stride == partsupp_row_count) ||
                            (partsupp_partkey[base + stride] != partkey);
                        if (start_ok && end_ok && candidate[0] == partkey &&
                            candidate[static_cast<uint32_t>(stride - 1)] == partkey) {
                            range_start = candidate;
                            range_end = candidate + stride;
                        }
                    }
                }
                if (range_start == nullptr) {
                    search_ptr = std::lower_bound(search_ptr, partsupp_partkey_end, partkey);
                    if (search_ptr == partsupp_partkey_end || *search_ptr != partkey) {
                        continue;
                    }
                    range_start = search_ptr;
                    range_end = range_start;
                    while (range_end < partsupp_partkey_end && *range_end == partkey) {
                        ++range_end;
                    }
                    search_ptr = range_end;
                }
                const uint32_t start_idx =
                    static_cast<uint32_t>(range_start - partsupp_partkey);
                const uint32_t end_idx =
                    static_cast<uint32_t>(range_end - partsupp_partkey);
                partsupp_rows_scanned += (end_idx - start_idx);

                int32_t min_cost = kMaxCost;
                uint32_t best_supp[kBestSupplierCapacity];
                uint32_t best_count = 0;
                if (!best_supp_overflow.empty()) {
                    best_supp_overflow.clear();
                }
                for (uint32_t row = start_idx; row < end_idx; ++row) {
                    const uint32_t srow =
                        static_cast<uint32_t>(partsupp_suppkey[row] - 1);
                    if (srow >= supplier_row_count || !supplier_ok_ptr[srow]) {
                        continue;
                    }
                    candidate_rows += 1;
                    const int32_t cost = partsupp_cost[row];
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_count = 0;
                        best_supp_overflow.clear();
                        best_supp[best_count++] = srow;
                    } else if (cost == min_cost) {
                        if (best_count < kBestSupplierCapacity) {
                            best_supp[best_count++] = srow;
                        } else {
                            best_supp_overflow.push_back(srow);
                        }
                    }
                }

                if (min_cost == kMaxCost) {
                    continue;
                }
                parts_with_min_cost += 1;
                const std::string_view p_mfgr =
                    part.mfgr.dictionary[part_mfgr_codes[prow]];
                const int32_t p_partkey = static_cast<int32_t>(prow + 1);
                for (uint32_t i = 0; i < best_count; ++i) {
                    const uint32_t srow = best_supp[i];
                    Q2ResultRow row;
                    row.s_acctbal_raw = supplier_acctbal[srow];
                    row.s_name = get_string_view(supplier.name, srow);
                    row.n_name = supplier_nation_name[srow];
                    row.p_partkey = p_partkey;
                    row.p_mfgr = p_mfgr;
                    row.s_address = get_string_view(supplier.address, srow);
                    row.s_phone = get_string_view(supplier.phone, srow);
                    row.s_comment = get_string_view(supplier.comment, srow);
                    results.push_back(row);
                    result_matches += 1;
                }
                for (uint32_t srow : best_supp_overflow) {
                    Q2ResultRow row;
                    row.s_acctbal_raw = supplier_acctbal[srow];
                    row.s_name = get_string_view(supplier.name, srow);
                    row.n_name = supplier_nation_name[srow];
                    row.p_partkey = p_partkey;
                    row.p_mfgr = p_mfgr;
                    row.s_address = get_string_view(supplier.address, srow);
                    row.s_phone = get_string_view(supplier.phone, srow);
                    row.s_comment = get_string_view(supplier.comment, srow);
                    results.push_back(row);
                    result_matches += 1;
                }
            }
            TRACE_SET(partsupp_min_rows_scanned, partsupp_rows_scanned);
        }
        TRACE_SET(join_min_probe_rows_in, partsupp_rows_scanned);
        TRACE_SET(partsupp_min_rows_emitted, candidate_rows);
        TRACE_SET(join_min_rows_emitted, candidate_rows);

        TRACE_SET(join_result_build_rows_in,
                  part_ok_count + supplier_in_region_count + parts_with_min_cost);
        TRACE_SET(join_result_probe_rows_in, candidate_rows);
        TRACE_SET(partsupp_result_rows_scanned, candidate_rows);
        TRACE_SET(partsupp_result_rows_emitted, result_matches);
        TRACE_SET(join_result_rows_emitted, result_matches);

        TRACE_SET(sort_rows_in, results.size());
        {
            PROFILE_SCOPE("q2_sort");
            std::sort(results.begin(), results.end(),
                      [](const Q2ResultRow& a, const Q2ResultRow& b) {
                          if (a.s_acctbal_raw != b.s_acctbal_raw) {
                              return a.s_acctbal_raw > b.s_acctbal_raw;
                          }
                          if (a.n_name != b.n_name) {
                              return a.n_name < b.n_name;
                          }
                          if (a.s_name != b.s_name) {
                              return a.s_name < b.s_name;
                          }
                          return a.p_partkey < b.p_partkey;
                      });
        }
        TRACE_SET(sort_rows_out, results.size());
    }

    TRACE_SET(query_output_rows, results.size());
#ifdef TRACE
    q2_trace::emit();
#endif
    return results;
}

void write_q2_csv(const std::string& filename, const std::vector<Q2ResultRow>& rows) {
    std::ofstream out(filename);
    out << "s_acctbal,s_name,n_name,p_partkey,p_mfgr,s_address,s_phone,s_comment\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double acctbal = static_cast<double>(row.s_acctbal_raw) / kQ2PriceScale;
        out << acctbal << ',';
        write_csv_field_q2(out, row.s_name);
        out << ',';
        write_csv_field_q2(out, row.n_name);
        out << ',' << row.p_partkey << ',';
        write_csv_field_q2(out, row.p_mfgr);
        out << ',';
        write_csv_field_q2(out, row.s_address);
        out << ',';
        write_csv_field_q2(out, row.s_phone);
        out << ',';
        write_csv_field_q2(out, row.s_comment);
        out << "\n";
    }
}