#include "query_q21.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q21 {
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
}  // namespace q21

#ifdef TRACE
namespace q21_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t supplier_scan_ns = 0;
    uint64_t orders_lineitem_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t supplier_rows_scanned = 0;
    uint64_t supplier_rows_emitted = 0;
    uint64_t orders_rows_scanned = 0;
    uint64_t orders_rows_emitted = 0;
    uint64_t lineitem_rows_scanned = 0;
    uint64_t lineitem_rows_emitted = 0;
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
    if (std::strcmp(name, "q21_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q21_supplier_scan") == 0) {
        trace.supplier_scan_ns += ns;
    } else if (std::strcmp(name, "q21_orders_lineitem_scan") == 0) {
        trace.orders_lineitem_scan_ns += ns;
    } else if (std::strcmp(name, "q21_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q21_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q21_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q21_supplier_scan " << trace.supplier_scan_ns << "\n";
    std::cout << "PROFILE q21_orders_lineitem_scan " << trace.orders_lineitem_scan_ns
              << "\n";
    std::cout << "PROFILE q21_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q21_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q21_supplier_scan_rows_scanned " << trace.supplier_rows_scanned
              << "\n";
    std::cout << "COUNT q21_supplier_scan_rows_emitted " << trace.supplier_rows_emitted
              << "\n";
    std::cout << "COUNT q21_orders_scan_rows_scanned " << trace.orders_rows_scanned
              << "\n";
    std::cout << "COUNT q21_orders_scan_rows_emitted " << trace.orders_rows_emitted
              << "\n";
    std::cout << "COUNT q21_lineitem_scan_rows_scanned " << trace.lineitem_rows_scanned
              << "\n";
    std::cout << "COUNT q21_lineitem_scan_rows_emitted " << trace.lineitem_rows_emitted
              << "\n";
    std::cout << "COUNT q21_orders_lineitem_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q21_orders_lineitem_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q21_orders_lineitem_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q21_supplier_join_build_rows_in "
              << trace.supplier_join_build_rows_in << "\n";
    std::cout << "COUNT q21_supplier_join_probe_rows_in "
              << trace.supplier_join_probe_rows_in << "\n";
    std::cout << "COUNT q21_supplier_join_rows_emitted "
              << trace.supplier_join_rows_emitted << "\n";
    std::cout << "COUNT q21_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q21_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q21_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q21_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q21_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q21_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q21_trace

#define PROFILE_SCOPE(name) \
    ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q21_trace::record_timing)
#define TRACE_SET(field, value) (q21_trace::data().field = (value))
#define TRACE_ADD(field, value) (q21_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q21ResultRow> run_q21(const Database& db, const Q21Args& args) {
#ifdef TRACE
    q21_trace::reset();
#endif
    std::vector<Q21ResultRow> results;
    {
        PROFILE_SCOPE("q21_total");
        if (args.NATION == "<<NULL>>") {
            TRACE_SET(query_output_rows, 0);
        } else {
            const auto nation_it = db.nation.name_to_key.find(args.NATION);
            if (nation_it == db.nation.name_to_key.end()) {
                TRACE_SET(query_output_rows, 0);
            } else {
                const int32_t target_nation = nation_it->second;

                const auto& orders = db.orders;
                const auto& lineitem = db.lineitem;
                const auto& supplier = db.supplier;
                const auto& li_orderkey = lineitem.orderkey;
                const auto& li_suppkey = lineitem.suppkey;
                const auto& li_commit_receipt = lineitem.commit_receipt;
                const bool orderkey_sorted = lineitem.orderkey_sorted;

                uint16_t final_status_code = 0;
                bool has_final_status = false;
                for (uint16_t code = 0;
                     code < static_cast<uint16_t>(orders.orderstatus.dictionary.size());
                     ++code) {
                    if (orders.orderstatus.dictionary[code] == "F") {
                        final_status_code = code;
                        has_final_status = true;
                        break;
                    }
                }
                if (!has_final_status) {
                    TRACE_SET(query_output_rows, 0);
                } else {
                    const int32_t max_suppkey = q21::max_value(supplier.suppkey);
                    if (max_suppkey < 0) {
                        TRACE_SET(query_output_rows, 0);
                    } else {
                        std::vector<int32_t> suppkey_to_row(
                            static_cast<size_t>(max_suppkey) + 1, -1);
                        std::vector<uint8_t> is_target_supplier(
                            static_cast<size_t>(max_suppkey) + 1, 0);
                        uint32_t target_supplier_count = 0;
                        TRACE_SET(supplier_rows_scanned, supplier.row_count);
                        {
                            PROFILE_SCOPE("q21_supplier_scan");
                            for (uint32_t row = 0; row < supplier.row_count; ++row) {
                                const int32_t suppkey = supplier.suppkey[row];
                                if (suppkey >= 0 && suppkey <= max_suppkey) {
                                    suppkey_to_row[static_cast<size_t>(suppkey)] =
                                        static_cast<int32_t>(row);
                                    if (supplier.nationkey[row] == target_nation) {
                                        is_target_supplier[static_cast<size_t>(suppkey)] = 1;
                                        ++target_supplier_count;
                                    }
                                    TRACE_ADD(supplier_rows_emitted, 1);
                                }
                            }
                        }

                        std::vector<int64_t> wait_counts(
                            static_cast<size_t>(max_suppkey) + 1, 0);

                        const uint32_t max_suppkey_u =
                            static_cast<uint32_t>(max_suppkey);
                        TRACE_SET(orders_rows_scanned, orders.row_count);
                        TRACE_SET(supplier_join_build_rows_in,
                                  q21_trace::data().supplier_rows_emitted);
                        {
                            PROFILE_SCOPE("q21_orders_lineitem_scan");
                            const auto* __restrict orderstatus_codes =
                                orders.orderstatus.codes.data();
                            const auto* __restrict orderkeys = orders.orderkey.data();
                            const auto* __restrict ranges = orders.lineitem_ranges.data();
                            const auto* __restrict target_supplier =
                                is_target_supplier.data();
                            const auto* __restrict suppkeys = li_suppkey.data();
                            const auto* __restrict commit_receipts =
                                li_commit_receipt.data();
                            if (orderkey_sorted) {
                                for (uint32_t o_idx = 0; o_idx < orders.row_count;
                                     ++o_idx) {
                                    const uint32_t status_code =
                                        orderstatus_codes[o_idx];
                                    if (status_code != final_status_code) {
                                        continue;
                                    }
                                    const auto range = ranges[o_idx];
                                    if (range.end == 0) {
                                        continue;
                                    }
                                    const uint32_t line_count =
                                        range.end - range.start;
                                    if (line_count <= 1) {
                                        continue;
                                    }
                                    TRACE_ADD(orders_rows_emitted, 1);

                                    int32_t first_suppkey = -1;
                                    bool has_multiple_suppliers = false;
                                    int32_t late_suppkey = -1;
                                    int32_t late_line_count = 0;
                                    bool multiple_late = false;

                                    const int32_t* __restrict supp_ptr =
                                        suppkeys + range.start;
                                    const int32_t* __restrict supp_end =
                                        suppkeys + range.end;
                                    const uint32_t* __restrict commit_ptr =
                                        commit_receipts + range.start;
#ifdef TRACE
                                    uint32_t scanned_rows = 0;
#endif
                                    for (; supp_ptr != supp_end; ++supp_ptr, ++commit_ptr) {
#ifdef TRACE
                                        ++scanned_rows;
#endif
                                        const int32_t suppkey = *supp_ptr;
                                        if (static_cast<uint32_t>(suppkey) > max_suppkey_u) {
                                            continue;
                                        }
                                        TRACE_ADD(lineitem_rows_emitted, 1);
                                        if (first_suppkey < 0) {
                                            first_suppkey = suppkey;
                                        } else if (suppkey != first_suppkey) {
                                            has_multiple_suppliers = true;
                                        }
                                        const uint32_t packed = *commit_ptr;
                                        const uint32_t commitdate =
                                            packed & 0xFFFFu;
                                        const uint32_t receiptdate = packed >> 16;
                                        if (receiptdate > commitdate) {
                                            if (late_suppkey == -1 ||
                                                late_suppkey == suppkey) {
                                                late_suppkey = suppkey;
                                                late_line_count += 1;
                                            } else {
                                                multiple_late = true;
                                                break;
                                            }
                                        }
                                    }
#ifdef TRACE
                                    TRACE_ADD(lineitem_rows_scanned, scanned_rows);
#endif

                                    if (multiple_late || !has_multiple_suppliers ||
                                        late_suppkey < 0) {
                                        continue;
                                    }
                                    if (!target_supplier[static_cast<size_t>(late_suppkey)]) {
                                        continue;
                                    }
                                    TRACE_ADD(join_rows_emitted, 1);
                                    TRACE_ADD(supplier_join_probe_rows_in, 1);
                                    TRACE_ADD(supplier_join_rows_emitted, 1);
                                    TRACE_ADD(agg_rows_in,
                                              static_cast<uint64_t>(late_line_count));
                                    wait_counts[static_cast<size_t>(late_suppkey)] +=
                                        late_line_count;
                                }
                            } else {
                                for (uint32_t o_idx = 0; o_idx < orders.row_count;
                                     ++o_idx) {
                                    const uint32_t status_code =
                                        orderstatus_codes[o_idx];
                                    if (status_code != final_status_code) {
                                        continue;
                                    }
                                    const int32_t orderkey = orderkeys[o_idx];
                                    const auto range = ranges[o_idx];
                                    if (range.end == 0) {
                                        continue;
                                    }
                                    const uint32_t line_count =
                                        range.end - range.start;
                                    if (line_count <= 1) {
                                        continue;
                                    }
                                    TRACE_ADD(orders_rows_emitted, 1);

                                    int32_t first_suppkey = -1;
                                    bool has_multiple_suppliers = false;
                                    int32_t late_suppkey = -1;
                                    int32_t late_line_count = 0;
                                    bool multiple_late = false;

                                    const int32_t* __restrict order_ptr =
                                        li_orderkey.data() + range.start;
                                    const int32_t* __restrict supp_ptr =
                                        suppkeys + range.start;
                                    const int32_t* __restrict supp_end =
                                        suppkeys + range.end;
                                    const uint32_t* __restrict commit_ptr =
                                        commit_receipts + range.start;
#ifdef TRACE
                                    uint32_t scanned_rows = 0;
#endif
                                    for (; supp_ptr != supp_end;
                                         ++supp_ptr, ++order_ptr, ++commit_ptr) {
#ifdef TRACE
                                        ++scanned_rows;
#endif
                                        if (*order_ptr != orderkey) {
                                            continue;
                                        }
                                        const int32_t suppkey = *supp_ptr;
                                        if (static_cast<uint32_t>(suppkey) > max_suppkey_u) {
                                            continue;
                                        }
                                        TRACE_ADD(lineitem_rows_emitted, 1);
                                        if (first_suppkey < 0) {
                                            first_suppkey = suppkey;
                                        } else if (suppkey != first_suppkey) {
                                            has_multiple_suppliers = true;
                                        }
                                        const uint32_t packed = *commit_ptr;
                                        const uint32_t commitdate =
                                            packed & 0xFFFFu;
                                        const uint32_t receiptdate = packed >> 16;
                                        if (receiptdate > commitdate) {
                                            if (late_suppkey == -1 ||
                                                late_suppkey == suppkey) {
                                                late_suppkey = suppkey;
                                                late_line_count += 1;
                                            } else {
                                                multiple_late = true;
                                                break;
                                            }
                                        }
                                    }
#ifdef TRACE
                                    TRACE_ADD(lineitem_rows_scanned, scanned_rows);
#endif

                                    if (multiple_late || !has_multiple_suppliers ||
                                        late_suppkey < 0) {
                                        continue;
                                    }
                                    if (!target_supplier[static_cast<size_t>(late_suppkey)]) {
                                        continue;
                                    }
                                    TRACE_ADD(join_rows_emitted, 1);
                                    TRACE_ADD(supplier_join_probe_rows_in, 1);
                                    TRACE_ADD(supplier_join_rows_emitted, 1);
                                    TRACE_ADD(agg_rows_in,
                                              static_cast<uint64_t>(late_line_count));
                                    wait_counts[static_cast<size_t>(late_suppkey)] +=
                                        late_line_count;
                                }
                            }
                        }
                        TRACE_SET(join_build_rows_in,
                                  q21_trace::data().orders_rows_emitted);
                        TRACE_SET(join_probe_rows_in,
                                  q21_trace::data().lineitem_rows_scanned);

                        results.reserve(target_supplier_count);
                        {
                            PROFILE_SCOPE("q21_agg_finalize");
                            for (size_t suppkey = 0; suppkey < wait_counts.size();
                                 ++suppkey) {
                                const int64_t count = wait_counts[suppkey];
                                if (count == 0) {
                                    continue;
                                }
                                const int32_t row = suppkey_to_row[suppkey];
                                if (row < 0) {
                                    continue;
                                }
                                TRACE_ADD(groups_created, 1);
                                Q21ResultRow result;
                                result.s_name = std::string(q21::get_string_view(
                                    supplier.name, static_cast<size_t>(row)));
                                result.numwait = count;
                                results.push_back(std::move(result));
                            }
                            TRACE_SET(agg_rows_emitted, q21_trace::data().groups_created);
                        }

                        TRACE_SET(sort_rows_in, results.size());
                        {
                            PROFILE_SCOPE("q21_sort");
                            std::sort(results.begin(), results.end(),
                                      [](const Q21ResultRow& a,
                                         const Q21ResultRow& b) {
                                          if (a.numwait != b.numwait) {
                                              return a.numwait > b.numwait;
                                          }
                                          return a.s_name < b.s_name;
                                      });
                        }
                        TRACE_SET(sort_rows_out, results.size());
                        TRACE_SET(query_output_rows, results.size());
                    }
                }
            }
        }
    }

#ifdef TRACE
    q21_trace::emit();
#endif
    return results;
}

void write_q21_csv(const std::string& filename, const std::vector<Q21ResultRow>& rows) {
    std::ofstream out(filename);
    out << "s_name,numwait\n";
    for (const auto& row : rows) {
        out << q21::escape_csv_field(row.s_name) << ',' << row.numwait << "\n";
    }
}