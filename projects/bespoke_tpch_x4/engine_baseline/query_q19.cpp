#include "query_q19.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q19 {
constexpr int32_t kPriceScale = 100;
constexpr int32_t kDiscountScale = 100;

int32_t parse_scaled_value(const std::string& value, int32_t scale) {
    const double numeric = std::stod(value);
    return static_cast<int32_t>(std::llround(numeric * static_cast<double>(scale)));
}

int32_t max_value(const std::vector<int32_t>& values) {
    if (values.empty()) {
        return -1;
    }
    return *std::max_element(values.begin(), values.end());
}

bool in_container_list(std::string_view container,
                       const std::string_view* options,
                       size_t count) {
    for (size_t idx = 0; idx < count; ++idx) {
        if (container == options[idx]) {
            return true;
        }
    }
    return false;
}

std::string_view trim_right_spaces(std::string_view value) {
    size_t end = value.size();
    while (end > 0 && value[end - 1] == ' ') {
        --end;
    }
    return value.substr(0, end);
}

bool equals_trimmed(std::string_view value, std::string_view target) {
    return trim_right_spaces(value) == target;
}
}  // namespace q19

#ifdef TRACE
namespace q19_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t part_scan_ns = 0;
    uint64_t lineitem_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t part_rows_scanned = 0;
    uint64_t part_rows_emitted = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
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
    if (std::strcmp(name, "q19_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q19_part_scan") == 0) {
        trace.part_scan_ns += ns;
    } else if (std::strcmp(name, "q19_lineitem_scan") == 0) {
        trace.lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q19_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q19_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q19_part_scan " << trace.part_scan_ns << "\n";
    std::cout << "PROFILE q19_lineitem_scan " << trace.lineitem_scan_ns << "\n";
    std::cout << "PROFILE q19_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "COUNT q19_part_scan_rows_scanned " << trace.part_rows_scanned << "\n";
    std::cout << "COUNT q19_part_scan_rows_emitted " << trace.part_rows_emitted << "\n";
    std::cout << "COUNT q19_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q19_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q19_part_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q19_part_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q19_part_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q19_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q19_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q19_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q19_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q19_trace

#define PROFILE_SCOPE(name) \
    ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q19_trace::record_timing)
#define TRACE_SET(field, value) (q19_trace::data().field = (value))
#define TRACE_ADD(field, value) (q19_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q19ResultRow> run_q19(const Database& db, const Q19Args& args) {
#ifdef TRACE
    q19_trace::reset();
#endif
    std::vector<Q19ResultRow> results;
    {
        PROFILE_SCOPE("q19_total");
        const bool group1_enabled =
            args.QUANTITY1 != "<<NULL>>" && args.BRAND1 != "<<NULL>>";
        const bool group2_enabled =
            args.QUANTITY2 != "<<NULL>>" && args.BRAND2 != "<<NULL>>";
        const bool group3_enabled =
            args.QUANTITY3 != "<<NULL>>" && args.BRAND3 != "<<NULL>>";

        if (!group1_enabled && !group2_enabled && !group3_enabled) {
            TRACE_SET(agg_rows_emitted, 1);
            TRACE_SET(groups_created, 1);
            results = {Q19ResultRow{}};
        } else {
            const auto& part = db.part;
            const auto& lineitem = db.lineitem;

            const int32_t max_partkey =
                std::max(q19::max_value(part.partkey),
                         q19::max_value(lineitem.partkey));
            if (max_partkey < 0) {
                TRACE_SET(agg_rows_emitted, 1);
                TRACE_SET(groups_created, 1);
                results = {Q19ResultRow{}};
            } else {
                int32_t shipmode_code1 = -1;
                int32_t shipmode_code2 = -1;
                for (size_t idx = 0; idx < lineitem.shipmode.dictionary.size();
                     ++idx) {
                    const auto trimmed =
                        q19::trim_right_spaces(lineitem.shipmode.dictionary[idx]);
                    if (trimmed == "AIR") {
                        shipmode_code1 = static_cast<int32_t>(idx);
                    } else if (trimmed == "AIR REG") {
                        shipmode_code2 = static_cast<int32_t>(idx);
                    }
                }

                int32_t shipinstruct_code = -1;
                for (size_t idx = 0; idx < lineitem.shipinstruct.dictionary.size();
                     ++idx) {
                    if (q19::equals_trimmed(lineitem.shipinstruct.dictionary[idx],
                                            "DELIVER IN PERSON")) {
                        shipinstruct_code = static_cast<int32_t>(idx);
                        break;
                    }
                }

                int32_t quantity1 = 0;
                int32_t quantity2 = 0;
                int32_t quantity3 = 0;
                if (group1_enabled) {
                    quantity1 =
                        q19::parse_scaled_value(args.QUANTITY1, q19::kPriceScale);
                }
                if (group2_enabled) {
                    quantity2 =
                        q19::parse_scaled_value(args.QUANTITY2, q19::kPriceScale);
                }
                if (group3_enabled) {
                    quantity3 =
                        q19::parse_scaled_value(args.QUANTITY3, q19::kPriceScale);
                }

                constexpr std::string_view kContainers1[] = {"SM CASE", "SM BOX",
                                                             "SM PACK", "SM PKG"};
                constexpr std::string_view kContainers2[] = {"MED BAG", "MED BOX",
                                                             "MED PKG", "MED PACK"};
                constexpr std::string_view kContainers3[] = {"LG CASE", "LG BOX",
                                                             "LG PACK", "LG PKG"};

                const int32_t quantity1_high = quantity1 + 10 * q19::kPriceScale;
                const int32_t quantity2_high = quantity2 + 10 * q19::kPriceScale;
                const int32_t quantity3_high = quantity3 + 10 * q19::kPriceScale;

                std::vector<uint8_t> partkey_group_mask(
                    static_cast<size_t>(max_partkey) + 1, 0);
                TRACE_SET(part_rows_scanned, part.row_count);
                {
                    PROFILE_SCOPE("q19_part_scan");
                    for (uint32_t row = 0; row < part.row_count; ++row) {
                        const int32_t partkey = part.partkey[row];
                        const std::string_view brand =
                            part.brand.dictionary[part.brand.codes[row]];
                        const std::string_view container =
                            part.container.dictionary[part.container.codes[row]];
                        const int32_t size = part.size[row];

                        uint8_t mask = 0;
                        if (group1_enabled && brand == args.BRAND1 &&
                            q19::in_container_list(
                                container, kContainers1,
                                sizeof(kContainers1) / sizeof(kContainers1[0])) &&
                            size >= 1 && size <= 5) {
                            mask |= 0x1;
                        }
                        if (group2_enabled && brand == args.BRAND2 &&
                            q19::in_container_list(
                                container, kContainers2,
                                sizeof(kContainers2) / sizeof(kContainers2[0])) &&
                            size >= 1 && size <= 10) {
                            mask |= 0x2;
                        }
                        if (group3_enabled && brand == args.BRAND3 &&
                            q19::in_container_list(
                                container, kContainers3,
                                sizeof(kContainers3) / sizeof(kContainers3[0])) &&
                            size >= 1 && size <= 15) {
                            mask |= 0x4;
                        }
                        partkey_group_mask[static_cast<size_t>(partkey)] = mask;
                        if (mask != 0) {
                            TRACE_ADD(part_rows_emitted, 1);
                        }
                    }
                }
                TRACE_SET(join_build_rows_in, q19_trace::data().part_rows_emitted);

                int64_t revenue_raw = 0;
                if (shipinstruct_code >= 0 &&
                    (shipmode_code1 >= 0 || shipmode_code2 >= 0)) {
                    PROFILE_SCOPE("q19_lineitem_scan");
                    const auto* __restrict shipmode_codes =
                        lineitem.shipmode.codes.data();
                    const auto* __restrict shipinstruct_codes =
                        lineitem.shipinstruct.codes.data();
                    const auto* __restrict partkey_col = lineitem.partkey.data();
                    const auto* __restrict quantity_col = lineitem.quantity.data();
                    const auto* __restrict extendedprice_col =
                        lineitem.extendedprice.data();
                    const auto* __restrict discount_col = lineitem.discount.data();
                    const auto* __restrict part_mask_col = partkey_group_mask.data();
                    const bool has_two_shipmodes =
                        shipmode_code1 >= 0 && shipmode_code2 >= 0;
                    const int32_t single_shipmode_code =
                        shipmode_code1 >= 0 ? shipmode_code1 : shipmode_code2;
                    for (uint32_t li_idx = 0; li_idx < lineitem.row_count; ++li_idx) {
                        TRACE_ADD(lineitem_rows_scanned, 1);
                        const int32_t shipmode_code =
                            static_cast<int32_t>(shipmode_codes[li_idx]);
                        if (has_two_shipmodes) {
                            if (shipmode_code != shipmode_code1 &&
                                shipmode_code != shipmode_code2) {
                                continue;
                            }
                        } else if (shipmode_code != single_shipmode_code) {
                            continue;
                        }
                        const int32_t shipinstruct_code_row =
                            static_cast<int32_t>(shipinstruct_codes[li_idx]);
                        if (shipinstruct_code_row != shipinstruct_code) {
                            continue;
                        }
                        const int32_t partkey = partkey_col[li_idx];
                        TRACE_ADD(join_probe_rows_in, 1);
                        const uint8_t mask = part_mask_col[static_cast<size_t>(partkey)];
                        if (mask == 0) {
                            continue;
                        }
                        const int32_t quantity =
                            static_cast<int32_t>(quantity_col[li_idx]);
                        switch (mask) {
                            case 0x1:
                                if (quantity < quantity1 ||
                                    quantity > quantity1_high) {
                                    continue;
                                }
                                break;
                            case 0x2:
                                if (quantity < quantity2 ||
                                    quantity > quantity2_high) {
                                    continue;
                                }
                                break;
                            case 0x4:
                                if (quantity < quantity3 ||
                                    quantity > quantity3_high) {
                                    continue;
                                }
                                break;
                            default: {
                                bool matches = false;
                                if ((mask & 0x1) && quantity >= quantity1 &&
                                    quantity <= quantity1_high) {
                                    matches = true;
                                }
                                if (!matches && (mask & 0x2) &&
                                    quantity >= quantity2 &&
                                    quantity <= quantity2_high) {
                                    matches = true;
                                }
                                if (!matches && (mask & 0x4) &&
                                    quantity >= quantity3 &&
                                    quantity <= quantity3_high) {
                                    matches = true;
                                }
                                if (!matches) {
                                    continue;
                                }
                                break;
                            }
                        }
                        TRACE_ADD(lineitem_rows_emitted, 1);
                        TRACE_ADD(join_rows_emitted, 1);
                        revenue_raw +=
                            static_cast<int64_t>(extendedprice_col[li_idx]) *
                            static_cast<int64_t>(q19::kDiscountScale -
                                                 static_cast<int32_t>(
                                                     discount_col[li_idx]));
                    }
                }

                TRACE_SET(agg_rows_in, q19_trace::data().lineitem_rows_emitted);
                {
                    PROFILE_SCOPE("q19_agg_finalize");
                    TRACE_SET(groups_created, 1);
                    TRACE_SET(agg_rows_emitted, 1);
                }

                results = {Q19ResultRow{revenue_raw}};
            }
        }

        TRACE_SET(query_output_rows, results.size());
    }

#ifdef TRACE
    q19_trace::emit();
#endif
    return results;
}

void write_q19_csv(const std::string& filename, const std::vector<Q19ResultRow>& rows) {
    std::ofstream out(filename);
    out << "revenue\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double revenue =
            static_cast<double>(row.revenue_raw) /
            (q19::kPriceScale * q19::kDiscountScale);
        out << revenue << "\n";
    }
}