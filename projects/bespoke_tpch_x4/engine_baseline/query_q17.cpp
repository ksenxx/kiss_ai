#include "query_q17.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "trace_utils.hpp"

namespace q17 {
constexpr int32_t kPriceScale = 100;
int32_t max_value(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return -1;
    }
    return *std::max_element(values.begin(), values.end());
}

int32_t find_dictionary_code(const std::vector<std::string>& dictionary,
                             const std::string& value) {
    for (size_t i = 0; i < dictionary.size(); ++i) {
        if (dictionary[i] == value) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}
}  // namespace q17

#ifdef TRACE
namespace q17_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t part_scan_ns = 0;
    uint64_t lineitem_sum_scan_ns = 0;
    uint64_t lineitem_filter_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t part_rows_scanned = 0;
    uint64_t part_rows_emitted = 0;
    uint64_t lineitem_sum_rows_scanned = 0;
    uint64_t lineitem_sum_rows_emitted = 0;
    uint64_t lineitem_filter_rows_scanned = 0;
    uint64_t lineitem_filter_rows_emitted = 0;
    uint64_t join_build_rows_in = 0;
    uint64_t join_probe_rows_in = 0;
    uint64_t join_rows_emitted = 0;
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
    if (std::strcmp(name, "q17_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q17_part_scan") == 0) {
        trace.part_scan_ns += ns;
    } else if (std::strcmp(name, "q17_lineitem_sum_scan") == 0) {
        trace.lineitem_sum_scan_ns += ns;
    } else if (std::strcmp(name, "q17_lineitem_filter_scan") == 0) {
        trace.lineitem_filter_scan_ns += ns;
    } else if (std::strcmp(name, "q17_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q17_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q17_part_scan " << trace.part_scan_ns << "\n";
    std::cout << "PROFILE q17_lineitem_sum_scan " << trace.lineitem_sum_scan_ns << "\n";
    std::cout << "PROFILE q17_lineitem_filter_scan " << trace.lineitem_filter_scan_ns
              << "\n";
    std::cout << "PROFILE q17_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "COUNT q17_part_scan_rows_scanned " << trace.part_rows_scanned << "\n";
    std::cout << "COUNT q17_part_scan_rows_emitted " << trace.part_rows_emitted << "\n";
    std::cout << "COUNT q17_lineitem_sum_scan_rows_scanned "
              << trace.lineitem_sum_rows_scanned << "\n";
    std::cout << "COUNT q17_lineitem_sum_scan_rows_emitted "
              << trace.lineitem_sum_rows_emitted << "\n";
    std::cout << "COUNT q17_lineitem_filter_scan_rows_scanned "
              << trace.lineitem_filter_rows_scanned << "\n";
    std::cout << "COUNT q17_lineitem_filter_scan_rows_emitted "
              << trace.lineitem_filter_rows_emitted << "\n";
    std::cout << "COUNT q17_part_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q17_part_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q17_part_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q17_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q17_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q17_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q17_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q17_trace

#define PROFILE_SCOPE(name) \
    ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q17_trace::record_timing)
#define TRACE_SET(field, value) (q17_trace::data().field = (value))
#define TRACE_ADD(field, value) (q17_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q17ResultRow> run_q17(const Database& db, const Q17Args& args) {
    const auto& part = db.part;
    const auto& lineitem = db.lineitem;

#ifdef TRACE
    q17_trace::reset();
#endif
    std::vector<Q17ResultRow> rows;
    {
        PROFILE_SCOPE("q17_total");
        if (args.BRAND == "<<NULL>>" || args.CONTAINER == "<<NULL>>") {
            rows.push_back(Q17ResultRow{0});
#ifdef TRACE
            TRACE_SET(query_output_rows, rows.size());
            q17_trace::emit();
#endif
            return rows;
        }

        const int32_t max_partkey = q17::max_value(part.partkey);
        if (max_partkey < 0) {
            rows.push_back(Q17ResultRow{0});
#ifdef TRACE
            TRACE_SET(query_output_rows, rows.size());
            q17_trace::emit();
#endif
            return rows;
        }

        const int32_t brand_code =
            q17::find_dictionary_code(part.brand.dictionary, args.BRAND);
        const int32_t container_code =
            q17::find_dictionary_code(part.container.dictionary, args.CONTAINER);
        if (brand_code < 0 || container_code < 0) {
            rows.push_back(Q17ResultRow{0});
#ifdef TRACE
            TRACE_SET(query_output_rows, rows.size());
            q17_trace::emit();
#endif
            return rows;
        }

        const size_t match_words = (static_cast<size_t>(max_partkey) >> 6) + 1;
        std::vector<uint64_t> part_match_bits(match_words, 0);
        uint64_t part_match_count = 0;
        {
            PROFILE_SCOPE("q17_part_scan");
            for (uint32_t row = 0; row < part.row_count; ++row) {
                const int32_t partkey = part.partkey[row];
                if (partkey < 0 || partkey > max_partkey) {
                    continue;
                }
                if (part.brand.codes[row] == brand_code &&
                    part.container.codes[row] == container_code) {
                    const uint32_t key = static_cast<uint32_t>(partkey);
                    part_match_bits[key >> 6] |= (1ULL << (key & 63u));
                    ++part_match_count;
                }
            }
        }
        TRACE_ADD(part_rows_scanned, part.row_count);
        TRACE_ADD(part_rows_emitted, part_match_count);
        TRACE_ADD(join_build_rows_in, part_match_count);

        if (part_match_count == 0) {
            rows.push_back(Q17ResultRow{0});
#ifdef TRACE
            TRACE_SET(query_output_rows, rows.size());
            q17_trace::emit();
#endif
            return rows;
        }

        std::vector<int64_t> sum_qty(static_cast<size_t>(max_partkey) + 1, 0);
        std::vector<int32_t> cnt_qty(static_cast<size_t>(max_partkey) + 1, 0);
        struct MatchRow {
            int32_t partkey = 0;
            int16_t quantity = 0;
            int32_t extendedprice = 0;
        };
        std::vector<MatchRow> matching_rows;
        matching_rows.reserve(part_match_count * 4);
        {
            PROFILE_SCOPE("q17_lineitem_sum_scan");
            const auto* partkey_data =
                reinterpret_cast<const uint32_t*>(lineitem.partkey.data());
            const auto* quantity_data = lineitem.quantity.data();
            const auto* extendedprice_data = lineitem.extendedprice.data();
            const auto* match_data = part_match_bits.data();
            auto* sum_data = sum_qty.data();
            auto* cnt_data = cnt_qty.data();
            const uint32_t row_count = static_cast<uint32_t>(lineitem.row_count);
            uint32_t row = 0;
            for (; row + 3 < row_count; row += 4) {
                const uint32_t partkey0 = partkey_data[row];
                if (match_data[partkey0 >> 6] & (1ULL << (partkey0 & 63u))) {
                    sum_data[static_cast<size_t>(partkey0)] += quantity_data[row];
                    cnt_data[static_cast<size_t>(partkey0)] += 1;
                    matching_rows.push_back(
                        MatchRow{static_cast<int32_t>(partkey0),
                                 quantity_data[row],
                                 extendedprice_data[row]});
                }
                const uint32_t partkey1 = partkey_data[row + 1];
                if (match_data[partkey1 >> 6] & (1ULL << (partkey1 & 63u))) {
                    sum_data[static_cast<size_t>(partkey1)] += quantity_data[row + 1];
                    cnt_data[static_cast<size_t>(partkey1)] += 1;
                    matching_rows.push_back(
                        MatchRow{static_cast<int32_t>(partkey1),
                                 quantity_data[row + 1],
                                 extendedprice_data[row + 1]});
                }
                const uint32_t partkey2 = partkey_data[row + 2];
                if (match_data[partkey2 >> 6] & (1ULL << (partkey2 & 63u))) {
                    sum_data[static_cast<size_t>(partkey2)] += quantity_data[row + 2];
                    cnt_data[static_cast<size_t>(partkey2)] += 1;
                    matching_rows.push_back(
                        MatchRow{static_cast<int32_t>(partkey2),
                                 quantity_data[row + 2],
                                 extendedprice_data[row + 2]});
                }
                const uint32_t partkey3 = partkey_data[row + 3];
                if (match_data[partkey3 >> 6] & (1ULL << (partkey3 & 63u))) {
                    sum_data[static_cast<size_t>(partkey3)] += quantity_data[row + 3];
                    cnt_data[static_cast<size_t>(partkey3)] += 1;
                    matching_rows.push_back(
                        MatchRow{static_cast<int32_t>(partkey3),
                                 quantity_data[row + 3],
                                 extendedprice_data[row + 3]});
                }
            }
            for (; row < row_count; ++row) {
                const uint32_t partkey = partkey_data[row];
                if (!(match_data[partkey >> 6] & (1ULL << (partkey & 63u)))) {
                    continue;
                }
                sum_data[static_cast<size_t>(partkey)] += quantity_data[row];
                cnt_data[static_cast<size_t>(partkey)] += 1;
                matching_rows.push_back(
                    MatchRow{static_cast<int32_t>(partkey),
                             quantity_data[row],
                             extendedprice_data[row]});
            }
        }
        TRACE_ADD(lineitem_sum_rows_scanned, lineitem.row_count);
        TRACE_ADD(lineitem_sum_rows_emitted, matching_rows.size());
        TRACE_ADD(agg_rows_in, matching_rows.size());

        {
            PROFILE_SCOPE("q17_agg_finalize");
            uint64_t group_count = 0;
            for (const auto cnt : cnt_qty) {
                if (cnt > 0) {
                    ++group_count;
                }
            }
            TRACE_ADD(groups_created, group_count);
            TRACE_ADD(agg_rows_emitted, group_count);
        }

        int64_t sum_extended = 0;
        uint64_t filter_emitted = 0;
        {
            PROFILE_SCOPE("q17_lineitem_filter_scan");
            for (const auto& row : matching_rows) {
                const int32_t cnt = cnt_qty[static_cast<size_t>(row.partkey)];
                if (cnt == 0) {
                    continue;
                }
                const int64_t lhs =
                    static_cast<int64_t>(row.quantity) * 5 * cnt;
                if (lhs >= sum_qty[static_cast<size_t>(row.partkey)]) {
                    continue;
                }
                sum_extended += row.extendedprice;
                ++filter_emitted;
            }
        }
        TRACE_ADD(lineitem_filter_rows_scanned, matching_rows.size());
        TRACE_ADD(lineitem_filter_rows_emitted, filter_emitted);
        TRACE_ADD(join_probe_rows_in, lineitem.row_count);
        TRACE_ADD(join_rows_emitted, filter_emitted);

        rows.push_back(Q17ResultRow{sum_extended});
        TRACE_SET(query_output_rows, rows.size());
    }
#ifdef TRACE
    q17_trace::emit();
#endif
    return rows;
}

void write_q17_csv(const std::string& filename, const std::vector<Q17ResultRow>& rows) {
    std::ofstream out(filename);
    out << "avg_yearly\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double avg_yearly =
            static_cast<double>(row.sum_extendedprice_raw) /
            (7.0 * static_cast<double>(q17::kPriceScale));
        out << avg_yearly << "\n";
    }
}