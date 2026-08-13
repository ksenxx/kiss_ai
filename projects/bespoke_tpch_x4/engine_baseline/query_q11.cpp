#include "query_q11.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "trace_utils.hpp"

namespace {
constexpr int32_t kQ11PriceScale = 100;

template <typename T>
int32_t q11_max_value(const std::vector<T>& values) {
    if (values.empty()) {
        return 0;
    }
    return static_cast<int32_t>(*std::max_element(values.begin(), values.end()));
}
}  // namespace

#ifdef TRACE
namespace q11_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t supplier_scan_ns = 0;
    uint64_t partsupp_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
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
    if (std::strcmp(name, "q11_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q11_supplier_scan") == 0) {
        trace.supplier_scan_ns += ns;
    } else if (std::strcmp(name, "q11_partsupp_scan") == 0) {
        trace.partsupp_scan_ns += ns;
    } else if (std::strcmp(name, "q11_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q11_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q11_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q11_supplier_scan " << trace.supplier_scan_ns << "\n";
    std::cout << "PROFILE q11_partsupp_scan " << trace.partsupp_scan_ns << "\n";
    std::cout << "PROFILE q11_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q11_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q11_supplier_scan_rows_scanned " << trace.supplier_rows_scanned
              << "\n";
    std::cout << "COUNT q11_supplier_scan_rows_emitted " << trace.supplier_rows_emitted
              << "\n";
    std::cout << "COUNT q11_partsupp_scan_rows_scanned " << trace.partsupp_rows_scanned
              << "\n";
    std::cout << "COUNT q11_partsupp_scan_rows_emitted " << trace.partsupp_rows_emitted
              << "\n";
    std::cout << "COUNT q11_supplier_partsupp_join_build_rows_in " << trace.join_build_rows_in
              << "\n";
    std::cout << "COUNT q11_supplier_partsupp_join_probe_rows_in " << trace.join_probe_rows_in
              << "\n";
    std::cout << "COUNT q11_supplier_partsupp_join_rows_emitted " << trace.join_rows_emitted
              << "\n";
    std::cout << "COUNT q11_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q11_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q11_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q11_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q11_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q11_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q11_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q11_trace::record_timing)
#define TRACE_SET(field, value) (q11_trace::data().field = (value))
#define TRACE_ADD(field, value) (q11_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q11ResultRow> run_q11(const Database& db, const Q11Args& args) {
#ifdef TRACE
    q11_trace::reset();
#endif
    std::vector<Q11ResultRow> results;
    {
        PROFILE_SCOPE("q11_total");
    const auto nation_it = db.nation.name_to_key.find(args.NATION);
    if (nation_it == db.nation.name_to_key.end()) {
#ifdef TRACE
        q11_trace::emit();
#endif
        return {};
    }
    const int32_t nation_key = nation_it->second;
    const double fraction = std::stod(args.FRACTION);

    const auto& supplier = db.supplier;
    const auto& partsupp = db.partsupp;

    const int32_t max_suppkey = q11_max_value(supplier.suppkey);
    std::vector<uint8_t> supp_in_nation(static_cast<size_t>(max_suppkey) + 1, 0);
    uint64_t suppliers_emitted = 0;
    {
        PROFILE_SCOPE("q11_supplier_scan");
        TRACE_SET(supplier_rows_scanned, supplier.row_count);
        const int32_t* suppkey_ptr = supplier.suppkey.data();
        const int32_t* nationkey_ptr = supplier.nationkey.data();
        for (uint32_t i = 0; i < supplier.row_count; ++i) {
            if (nationkey_ptr[i] != nation_key) {
                continue;
            }
            const int32_t suppkey = suppkey_ptr[i];
            supp_in_nation[static_cast<size_t>(suppkey)] = 1;
            suppliers_emitted += 1;
        }
    }
    TRACE_SET(supplier_rows_emitted, suppliers_emitted);

    std::vector<Q11ResultRow> aggregates;
    aggregates.reserve(partsupp.row_count / 8 + 1);

    int64_t total_value = 0;
    uint64_t partsupp_emitted = 0;
    {
        PROFILE_SCOPE("q11_partsupp_scan");
        TRACE_SET(partsupp_rows_scanned, partsupp.row_count);
        const uint32_t row_count = static_cast<uint32_t>(partsupp.row_count);
        const int32_t* __restrict suppkey_ptr = partsupp.suppkey.data();
        const int32_t* __restrict partkey_ptr = partsupp.partkey.data();
        const int32_t* __restrict value_ptr = partsupp.supplycost_availqty.data();
        const uint8_t* __restrict supp_mask_ptr = supp_in_nation.data();
        if (row_count > 0) {
            int32_t current_partkey = partkey_ptr[0];
            int64_t current_sum = 0;
            for (uint32_t i = 0; i < row_count; ++i) {
                const int32_t partkey = partkey_ptr[i];
                if (partkey != current_partkey) {
                    if (current_sum != 0) {
                        aggregates.push_back(Q11ResultRow{current_partkey, current_sum});
                    }
                    current_partkey = partkey;
                    current_sum = 0;
                }
                const int32_t suppkey = suppkey_ptr[i];
                if (supp_mask_ptr[static_cast<size_t>(suppkey)]) {
                    const int32_t value = value_ptr[i];
                    total_value += value;
                    current_sum += value;
                    partsupp_emitted += 1;
                }
            }
            if (current_sum != 0) {
                aggregates.push_back(Q11ResultRow{current_partkey, current_sum});
            }
        }
    }
    TRACE_SET(partsupp_rows_emitted, partsupp_emitted);
    TRACE_SET(join_build_rows_in, suppliers_emitted);
    TRACE_SET(join_probe_rows_in, partsupp.row_count);
    TRACE_SET(join_rows_emitted, partsupp_emitted);
    TRACE_SET(agg_rows_in, partsupp_emitted);

    const double threshold = static_cast<double>(total_value) * fraction;

    {
        PROFILE_SCOPE("q11_agg_finalize");
        results.reserve(aggregates.size());
        for (const auto& agg : aggregates) {
            if (static_cast<double>(agg.value_raw) > threshold) {
                results.push_back(agg);
            }
        }
        TRACE_SET(groups_created, aggregates.size());
        TRACE_SET(agg_rows_emitted, results.size());
    }

    {
        PROFILE_SCOPE("q11_sort");
        TRACE_SET(sort_rows_in, results.size());
        std::sort(results.begin(), results.end(),
                  [](const Q11ResultRow& a, const Q11ResultRow& b) {
                      if (a.value_raw != b.value_raw) {
                          return a.value_raw > b.value_raw;
                      }
                      return a.ps_partkey < b.ps_partkey;
                  });
        TRACE_SET(sort_rows_out, results.size());
    }
    TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q11_trace::emit();
#endif
    return results;
}

void write_q11_csv(const std::string& filename, const std::vector<Q11ResultRow>& rows) {
    std::ofstream out(filename);
    out << "ps_partkey,value\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double value = static_cast<double>(row.value_raw) / kQ11PriceScale;
        out << row.ps_partkey << ',' << value << "\n";
    }
}