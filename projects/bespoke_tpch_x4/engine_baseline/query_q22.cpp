#include "query_q22.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q22 {
constexpr int32_t kPriceScale = 100;

struct CodeFilter {
    std::array<uint8_t, 256> numeric_allowed{};
    std::vector<int32_t> numeric_codes;
    std::vector<std::string> extra_codes;
    std::vector<uint16_t> extra_codes_packed;
};

inline uint16_t pack_prefix_code(char c0, char c1) {
    return static_cast<uint16_t>(static_cast<uint8_t>(c0) << 8) |
           static_cast<uint16_t>(static_cast<uint8_t>(c1));
}

inline bool parse_numeric_code(std::string_view value, int32_t& code_out) {
    if (value.size() < 2) {
        return false;
    }
    const char c0 = value[0];
    const char c1 = value[1];
    if (c0 < '0' || c0 > '9' || c1 < '0' || c1 > '9') {
        return false;
    }
    code_out = (c0 - '0') * 10 + (c1 - '0');
    return true;
}

inline void add_code(CodeFilter& filter, const std::string& value) {
    if (value == "<<NULL>>") {
        return;
    }
    if (value.size() < 2) {
        return;
    }
    int32_t code = 0;
    if (parse_numeric_code(value, code)) {
        if (!filter.numeric_allowed[code]) {
            filter.numeric_allowed[code] = 1;
            filter.numeric_codes.push_back(code);
        }
        return;
    }
    const std::string prefix = value.substr(0, 2);
    const uint16_t packed = pack_prefix_code(prefix[0], prefix[1]);
    for (const auto& entry : filter.extra_codes_packed) {
        if (entry == packed) {
            return;
        }
    }
    filter.extra_codes.push_back(prefix);
    filter.extra_codes_packed.push_back(packed);
}

inline bool phone_in_filter(const StringColumn& column, uint32_t row,
                            const CodeFilter& filter) {
    const uint32_t start = column.offsets[row];
    const uint32_t end = column.offsets[row + 1];
    if (end - start < 2) {
        return false;
    }
    const char* ptr = column.data.data() + start;
    const char c0 = ptr[0];
    const char c1 = ptr[1];
    if (c0 >= '0' && c0 <= '9' && c1 >= '0' && c1 <= '9') {
        const int32_t code = (c0 - '0') * 10 + (c1 - '0');
        return filter.numeric_allowed[code] != 0;
    }
    if (filter.extra_codes.empty()) {
        return false;
    }
    const uint16_t packed = pack_prefix_code(c0, c1);
    for (const auto& entry : filter.extra_codes_packed) {
        if (entry == packed) {
            return true;
        }
    }
    return false;
}

inline bool phone_in_filter_with_code(const StringColumn& column, uint32_t row,
                                      const CodeFilter& filter, bool& is_numeric,
                                      int32_t& numeric_code, int32_t& extra_index) {
    const uint32_t start = column.offsets[row];
    const uint32_t end = column.offsets[row + 1];
    if (end - start < 2) {
        return false;
    }
    const char* ptr = column.data.data() + start;
    const char c0 = ptr[0];
    const char c1 = ptr[1];
    if (c0 >= '0' && c0 <= '9' && c1 >= '0' && c1 <= '9') {
        numeric_code = (c0 - '0') * 10 + (c1 - '0');
        if (filter.numeric_allowed[numeric_code] == 0) {
            return false;
        }
        is_numeric = true;
        return true;
    }
    if (filter.extra_codes.empty()) {
        return false;
    }
    const uint16_t packed = pack_prefix_code(c0, c1);
    for (size_t idx = 0; idx < filter.extra_codes_packed.size(); ++idx) {
        if (filter.extra_codes_packed[idx] == packed) {
            is_numeric = false;
            extra_index = static_cast<int32_t>(idx);
            return true;
        }
    }
    return false;
}

inline bool phone_in_filter_with_prefix(const StringColumn& column,
                                        const std::vector<uint8_t>& prefix_codes,
                                        uint32_t row, const CodeFilter& filter,
                                        bool& is_numeric, int32_t& numeric_code,
                                        int32_t& extra_index) {
    const uint8_t prefix = prefix_codes[row];
    if (prefix != 255) {
        numeric_code = prefix;
        if (filter.numeric_allowed[numeric_code] == 0) {
            return false;
        }
        is_numeric = true;
        return true;
    }
    if (filter.extra_codes.empty()) {
        return false;
    }
    const uint32_t start = column.offsets[row];
    const uint32_t end = column.offsets[row + 1];
    if (end - start < 2) {
        return false;
    }
    const char* ptr = column.data.data() + start;
    const uint16_t packed = pack_prefix_code(ptr[0], ptr[1]);
    for (size_t idx = 0; idx < filter.extra_codes_packed.size(); ++idx) {
        if (filter.extra_codes_packed[idx] == packed) {
            is_numeric = false;
            extra_index = static_cast<int32_t>(idx);
            return true;
        }
    }
    return false;
}

inline std::string numeric_code_to_string(int32_t code) {
    char buffer[2] = {static_cast<char>('0' + (code / 10)),
                      static_cast<char>('0' + (code % 10))};
    return std::string(buffer, 2);
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
}  // namespace q22

#ifdef TRACE
namespace q22_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t customer_avg_scan_ns = 0;
    uint64_t customer_filter_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t customer_avg_rows_scanned = 0;
    uint64_t customer_avg_rows_emitted = 0;
    uint64_t customer_filter_rows_scanned = 0;
    uint64_t customer_filter_rows_emitted = 0;
    uint64_t orders_join_build_rows_in = 0;
    uint64_t orders_join_probe_rows_in = 0;
    uint64_t orders_join_rows_emitted = 0;
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
    if (std::strcmp(name, "q22_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q22_customer_avg_scan") == 0) {
        trace.customer_avg_scan_ns += ns;
    } else if (std::strcmp(name, "q22_customer_filter_scan") == 0) {
        trace.customer_filter_scan_ns += ns;
    } else if (std::strcmp(name, "q22_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q22_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q22_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q22_customer_avg_scan " << trace.customer_avg_scan_ns
              << "\n";
    std::cout << "PROFILE q22_customer_filter_scan " << trace.customer_filter_scan_ns
              << "\n";
    std::cout << "PROFILE q22_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q22_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q22_customer_avg_scan_rows_scanned "
              << trace.customer_avg_rows_scanned << "\n";
    std::cout << "COUNT q22_customer_avg_scan_rows_emitted "
              << trace.customer_avg_rows_emitted << "\n";
    std::cout << "COUNT q22_customer_filter_scan_rows_scanned "
              << trace.customer_filter_rows_scanned << "\n";
    std::cout << "COUNT q22_customer_filter_scan_rows_emitted "
              << trace.customer_filter_rows_emitted << "\n";
    std::cout << "COUNT q22_orders_join_build_rows_in "
              << trace.orders_join_build_rows_in << "\n";
    std::cout << "COUNT q22_orders_join_probe_rows_in "
              << trace.orders_join_probe_rows_in << "\n";
    std::cout << "COUNT q22_orders_join_rows_emitted "
              << trace.orders_join_rows_emitted << "\n";
    std::cout << "COUNT q22_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q22_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q22_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q22_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q22_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q22_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q22_trace

#define PROFILE_SCOPE(name) \
    ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q22_trace::record_timing)
#define TRACE_SET(field, value) (q22_trace::data().field = (value))
#define TRACE_ADD(field, value) (q22_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q22ResultRow> run_q22(const Database& db, const Q22Args& args) {
#ifdef TRACE
    q22_trace::reset();
#endif
    std::vector<Q22ResultRow> results;
    {
        PROFILE_SCOPE("q22_total");
        q22::CodeFilter filter;
        filter.numeric_codes.reserve(7);
        filter.extra_codes.reserve(7);
        filter.extra_codes_packed.reserve(7);
        q22::add_code(filter, args.I1);
        q22::add_code(filter, args.I2);
        q22::add_code(filter, args.I3);
        q22::add_code(filter, args.I4);
        q22::add_code(filter, args.I5);
        q22::add_code(filter, args.I6);
        q22::add_code(filter, args.I7);
        if (!filter.numeric_codes.empty() || !filter.extra_codes.empty()) {
            const auto& customer = db.customer;
            const auto& orders = db.orders;

            const bool has_extra = !filter.extra_codes.empty();
            if (!has_extra) {
                struct CandidateNumeric {
                    int32_t custkey = 0;
                    int32_t acctbal = 0;
                    uint8_t code = 0;
                };

                std::vector<CandidateNumeric> candidates;
                candidates.reserve(customer.row_count / 3 + 1);

                int64_t acctbal_sum = 0;
                int64_t acctbal_count = 0;
                uint64_t allowed_masks[2] = {0, 0};
                for (const int32_t code : filter.numeric_codes) {
                    const uint64_t bit = 1ULL << (code & 63);
                    allowed_masks[static_cast<size_t>(code >> 6)] |= bit;
                }
                TRACE_SET(customer_avg_rows_scanned, customer.row_count);
                {
                    PROFILE_SCOPE("q22_customer_avg_scan");
                    const auto* acctbal_ptr = customer.acctbal.data();
                    const auto* custkey_ptr = customer.custkey.data();
                    const auto* prefix_ptr = customer.phone_prefix_code.data();
                    const uint32_t row_count =
                        static_cast<uint32_t>(customer.row_count);
                    for (uint32_t row = 0; row < row_count; ++row) {
                        const uint8_t prefix = prefix_ptr[row];
                        const uint64_t mask =
                            allowed_masks[static_cast<size_t>(prefix >> 6)];
                        if ((mask & (1ULL << (prefix & 63))) == 0) {
                            continue;
                        }
                        const int32_t acctbal = acctbal_ptr[row];
                        if (acctbal <= 0) {
                            continue;
                        }
                        acctbal_sum += acctbal;
                        acctbal_count += 1;
                        candidates.push_back(
                            CandidateNumeric{custkey_ptr[row], acctbal, prefix});
                        TRACE_ADD(customer_avg_rows_emitted, 1);
                    }
                }

                if (acctbal_count > 0 && !candidates.empty()) {
                    struct Agg {
                        int64_t numcust = 0;
                        int64_t totacctbal_raw = 0;
                    };
                    std::array<Agg, 100> totals{};
                    const int64_t avg_floor = acctbal_sum / acctbal_count;

                    TRACE_SET(customer_filter_rows_scanned, candidates.size());
                    TRACE_SET(orders_join_build_rows_in, orders.row_count);
                    {
                        PROFILE_SCOPE("q22_customer_filter_scan");
                        const auto* offsets = orders.orders_by_customer_offsets.data();
                        for (const auto& candidate : candidates) {
                            if (candidate.acctbal <= avg_floor) {
                                continue;
                            }
                            TRACE_ADD(orders_join_probe_rows_in, 1);
                            const int32_t custkey = candidate.custkey;
                            const uint32_t* base = offsets + custkey;
                            if (base[0] != base[1]) {
                                continue;
                            }
                            TRACE_ADD(orders_join_rows_emitted, 1);
                            TRACE_ADD(customer_filter_rows_emitted, 1);
                            TRACE_ADD(agg_rows_in, 1);
                            const int32_t numeric_code = candidate.code;
                            Agg& agg = totals[numeric_code];
                            agg.numcust += 1;
                            agg.totacctbal_raw += candidate.acctbal;
                        }
                    }

                    {
                        PROFILE_SCOPE("q22_agg_finalize");
                        results.reserve(filter.numeric_codes.size());
                        for (const int32_t code : filter.numeric_codes) {
                            const Agg& agg = totals[code];
                            if (agg.numcust == 0) {
                                continue;
                            }
                            Q22ResultRow row;
                            row.cntrycode = q22::numeric_code_to_string(code);
                            row.numcust = agg.numcust;
                            row.totacctbal_raw = agg.totacctbal_raw;
                            results.push_back(std::move(row));
                        }
                    }

                    TRACE_SET(groups_created, results.size());
                    TRACE_SET(agg_rows_emitted, results.size());

                    TRACE_SET(sort_rows_in, results.size());
                    {
                        PROFILE_SCOPE("q22_sort");
                        std::sort(results.begin(), results.end(),
                                  [](const Q22ResultRow& a,
                                     const Q22ResultRow& b) {
                                      return a.cntrycode < b.cntrycode;
                                  });
                    }
                    TRACE_SET(sort_rows_out, results.size());
                }
            } else {
                struct Candidate {
                    int32_t custkey = 0;
                    int32_t acctbal = 0;
                    int16_t code = 0;
                    uint8_t is_numeric = 0;
                };

                std::vector<Candidate> candidates;
                candidates.reserve(customer.row_count / 3 + 1);

                int64_t acctbal_sum = 0;
                int64_t acctbal_count = 0;
                TRACE_SET(customer_avg_rows_scanned, customer.row_count);
                {
                    PROFILE_SCOPE("q22_customer_avg_scan");
                    const auto* acctbal_ptr = customer.acctbal.data();
                    const auto* custkey_ptr = customer.custkey.data();
                    const uint32_t row_count =
                        static_cast<uint32_t>(customer.row_count);
                    for (uint32_t row = 0; row < row_count; ++row) {
                        bool is_numeric = true;
                        int32_t numeric_code = 0;
                        int32_t extra_index = -1;
                        if (!q22::phone_in_filter_with_prefix(
                                customer.phone, customer.phone_prefix_code, row,
                                filter, is_numeric, numeric_code, extra_index)) {
                            continue;
                        }
                        const int32_t acctbal = acctbal_ptr[row];
                        if (acctbal <= 0) {
                            continue;
                        }
                        acctbal_sum += acctbal;
                        acctbal_count += 1;
                        Candidate candidate;
                        candidate.custkey = custkey_ptr[row];
                        candidate.acctbal = acctbal;
                        candidate.is_numeric = static_cast<uint8_t>(is_numeric);
                        candidate.code = static_cast<int16_t>(is_numeric ? numeric_code
                                                                          : extra_index);
                        candidates.push_back(candidate);
                        TRACE_ADD(customer_avg_rows_emitted, 1);
                    }
                }

                if (acctbal_count > 0 && !candidates.empty()) {
                    struct Agg {
                        int64_t numcust = 0;
                        int64_t totacctbal_raw = 0;
                    };
                    std::array<Agg, 100> totals{};
                    std::vector<Agg> extra_totals(filter.extra_codes.size());
                    const int64_t avg_floor = acctbal_sum / acctbal_count;

                    TRACE_SET(customer_filter_rows_scanned, candidates.size());
                    TRACE_SET(orders_join_build_rows_in, orders.row_count);
                    {
                        PROFILE_SCOPE("q22_customer_filter_scan");
                        const auto* offsets = orders.orders_by_customer_offsets.data();
                        for (const auto& candidate : candidates) {
                            if (candidate.acctbal <= avg_floor) {
                                continue;
                            }
                            TRACE_ADD(orders_join_probe_rows_in, 1);
                            const int32_t custkey = candidate.custkey;
                            const uint32_t* base = offsets + custkey;
                            if (base[0] != base[1]) {
                                continue;
                            }
                            TRACE_ADD(orders_join_rows_emitted, 1);
                            TRACE_ADD(customer_filter_rows_emitted, 1);
                            TRACE_ADD(agg_rows_in, 1);
                            if (candidate.is_numeric) {
                                const int32_t numeric_code = candidate.code;
                                Agg& agg = totals[numeric_code];
                                agg.numcust += 1;
                                agg.totacctbal_raw += candidate.acctbal;
                            } else {
                                const size_t extra_idx =
                                    static_cast<size_t>(candidate.code);
                                Agg& agg = extra_totals[extra_idx];
                                agg.numcust += 1;
                                agg.totacctbal_raw += candidate.acctbal;
                            }
                        }
                    }

                    {
                        PROFILE_SCOPE("q22_agg_finalize");
                        results.reserve(filter.numeric_codes.size() +
                                        filter.extra_codes.size());
                        for (const int32_t code : filter.numeric_codes) {
                            const Agg& agg = totals[code];
                            if (agg.numcust == 0) {
                                continue;
                            }
                            Q22ResultRow row;
                            row.cntrycode = q22::numeric_code_to_string(code);
                            row.numcust = agg.numcust;
                            row.totacctbal_raw = agg.totacctbal_raw;
                            results.push_back(std::move(row));
                        }
                        for (size_t idx = 0; idx < extra_totals.size(); ++idx) {
                            const Agg& agg = extra_totals[idx];
                            if (agg.numcust == 0) {
                                continue;
                            }
                            Q22ResultRow row;
                            row.cntrycode = filter.extra_codes[idx];
                            row.numcust = agg.numcust;
                            row.totacctbal_raw = agg.totacctbal_raw;
                            results.push_back(std::move(row));
                        }
                    }

                    TRACE_SET(groups_created, results.size());
                    TRACE_SET(agg_rows_emitted, results.size());

                    TRACE_SET(sort_rows_in, results.size());
                    {
                        PROFILE_SCOPE("q22_sort");
                        std::sort(results.begin(), results.end(),
                                  [](const Q22ResultRow& a,
                                     const Q22ResultRow& b) {
                                      return a.cntrycode < b.cntrycode;
                                  });
                    }
                    TRACE_SET(sort_rows_out, results.size());
                }
            }
        }

        TRACE_SET(query_output_rows, results.size());
    }

#ifdef TRACE
    q22_trace::emit();
#endif
    return results;
}

void write_q22_csv(const std::string& filename, const std::vector<Q22ResultRow>& rows) {
    std::ofstream out(filename);
    out << "cntrycode,numcust,totacctbal\n";
    out.setf(std::ios::fixed);
    out << std::setprecision(2);
    for (const auto& row : rows) {
        const double totacctbal =
            static_cast<double>(row.totacctbal_raw) / q22::kPriceScale;
        out << q22::escape_csv_field(row.cntrycode) << ','
            << row.numcust << ',' << totacctbal << "\n";
    }
}