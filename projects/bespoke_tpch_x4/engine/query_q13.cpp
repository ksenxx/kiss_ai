#include "query_q13.hpp"

#include "audit.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include "trace_utils.hpp"

namespace q13 {
inline uint32_t build_alpha_mask(std::string_view value) {
    uint32_t mask = 0;
    for (char ch : value) {
        if (ch >= 'a' && ch <= 'z') {
            mask |= 1u << static_cast<uint32_t>(ch - 'a');
        } else if (ch >= 'A' && ch <= 'Z') {
            mask |= 1u << static_cast<uint32_t>(ch - 'A');
        }
    }
    return mask;
}

inline uint64_t build_bigram_mask(std::string_view value) {
    if (value.size() < 2) {
        return 0;
    }
    uint64_t mask = 0;
    const unsigned char* data =
        reinterpret_cast<const unsigned char*>(value.data());
    for (size_t i = 0; i + 1 < value.size(); ++i) {
        const uint32_t hash = (static_cast<uint32_t>(data[i]) * 131u +
                               static_cast<uint32_t>(data[i + 1])) &
                              63u;
        mask |= 1ULL << hash;
    }
    return mask;
}

inline const char* find_substring(const char* start,
                                  const char* end,
                                  const char* needle,
                                  size_t needle_len) {
    if (needle_len == 0) {
        return start;
    }
    const char first = needle[0];
    const char* last = end - needle_len;
    const char* cur = start;
    while (cur <= last) {
        const auto remaining = static_cast<size_t>(last - cur + 1);
        const void* hit = std::memchr(cur, first, remaining);
        if (!hit) {
            return end;
        }
        cur = static_cast<const char*>(hit);
        if (needle_len == 1 ||
            std::memcmp(cur + 1, needle + 1, needle_len - 1) == 0) {
            return cur;
        }
        ++cur;
    }
    return end;
}

inline bool matches_pattern(const char* comment,
                            size_t comment_len,
                            const char* word1,
                            size_t word1_len,
                            const char* word2,
                            size_t word2_len) {
    const char* end = comment + comment_len;
    const char* pos1 = find_substring(comment, end, word1, word1_len);
    if (pos1 == end) {
        return false;
    }
    const char* pos2 =
        find_substring(pos1 + word1_len, end, word2, word2_len);
    return pos2 != end;
}

template <typename T>
int32_t max_value(const std::vector<T>& values) {
    if (values.empty()) {
        return 0;
    }
    return static_cast<int32_t>(*std::max_element(values.begin(), values.end()));
}
}  // namespace q13

#ifdef TRACE
namespace q13_trace {
struct TraceData {
    uint64_t total_ns = 0;
    uint64_t orders_scan_ns = 0;
    uint64_t customer_scan_ns = 0;
    uint64_t agg_finalize_ns = 0;
    uint64_t sort_ns = 0;
    uint64_t orders_rows_scanned = 0;
    uint64_t orders_rows_emitted = 0;
    uint64_t customer_rows_scanned = 0;
    uint64_t customer_rows_emitted = 0;
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
    if (std::strcmp(name, "q13_total") == 0) {
        trace.total_ns += ns;
    } else if (std::strcmp(name, "q13_orders_scan") == 0) {
        trace.orders_scan_ns += ns;
    } else if (std::strcmp(name, "q13_customer_scan") == 0) {
        trace.customer_scan_ns += ns;
    } else if (std::strcmp(name, "q13_agg_finalize") == 0) {
        trace.agg_finalize_ns += ns;
    } else if (std::strcmp(name, "q13_sort") == 0) {
        trace.sort_ns += ns;
    }
}

inline void emit() {
    const auto& trace = data();
    std::cout << "PROFILE q13_total " << trace.total_ns << "\n";
    std::cout << "PROFILE q13_orders_scan " << trace.orders_scan_ns << "\n";
    std::cout << "PROFILE q13_customer_scan " << trace.customer_scan_ns << "\n";
    std::cout << "PROFILE q13_agg_finalize " << trace.agg_finalize_ns << "\n";
    std::cout << "PROFILE q13_sort " << trace.sort_ns << "\n";
    std::cout << "COUNT q13_orders_scan_rows_scanned " << trace.orders_rows_scanned << "\n";
    std::cout << "COUNT q13_orders_scan_rows_emitted " << trace.orders_rows_emitted << "\n";
    std::cout << "COUNT q13_customer_scan_rows_scanned " << trace.customer_rows_scanned
              << "\n";
    std::cout << "COUNT q13_customer_scan_rows_emitted " << trace.customer_rows_emitted
              << "\n";
    std::cout << "COUNT q13_agg_rows_in " << trace.agg_rows_in << "\n";
    std::cout << "COUNT q13_agg_groups_created " << trace.groups_created << "\n";
    std::cout << "COUNT q13_agg_rows_emitted " << trace.agg_rows_emitted << "\n";
    std::cout << "COUNT q13_sort_rows_in " << trace.sort_rows_in << "\n";
    std::cout << "COUNT q13_sort_rows_out " << trace.sort_rows_out << "\n";
    std::cout << "COUNT q13_query_output_rows " << trace.query_output_rows << "\n";
}
}  // namespace q13_trace

#define PROFILE_SCOPE(name) ::trace_utils::ScopedTimer _timer_##__LINE__(name, &q13_trace::record_timing)
#define TRACE_SET(field, value) (q13_trace::data().field = (value))
#define TRACE_ADD(field, value) (q13_trace::data().field += (value))
#else
#define PROFILE_SCOPE(name)
#define TRACE_SET(field, value)
#define TRACE_ADD(field, value)
#endif

std::vector<Q13ResultRow> run_q13(const Database& db, const Q13Args& args) {
#ifdef TRACE
    q13_trace::reset();
#endif
    std::vector<Q13ResultRow> results;
    {
        PROFILE_SCOPE("q13_total");
        const auto& orders = db.orders;
        const auto& customer = db.customer;

        const int32_t max_custkey = q13::max_value(customer.custkey);

        // Per-customer base order counts are parameter independent (available
        // via orders_by_customer_offsets); only orders whose comment MATCHES
        // the pattern (rare) must be subtracted.
        const auto& cust_offsets = orders.orders_by_customer_offsets;
        std::vector<int32_t> matched_counts(static_cast<size_t>(max_custkey) + 1,
                                            0);
        const char* comment_data = orders.comment.data.data();
        const uint32_t* comment_offsets = orders.comment.offsets.data();
        const uint32_t* comment_masks = orders.comment.alpha_mask.data();
        const auto* comment_bigrams = orders.comment.bigram_mask.data();
        const char* word1 = args.WORD1.data();
        const char* word2 = args.WORD2.data();
        const size_t word1_len = args.WORD1.size();
        const size_t word2_len = args.WORD2.size();
        const uint32_t required_mask =
            q13::build_alpha_mask(args.WORD1) | q13::build_alpha_mask(args.WORD2);
        const uint64_t required_bigram_mask =
            q13::build_bigram_mask(args.WORD1) | q13::build_bigram_mask(args.WORD2);
        // Q13 has a single execution path: precomputed per-customer base
        // order counts (orders_by_customer_offsets) minus pattern-matching
        // orders found by a full comment scan. The alpha/bigram masks are
        // pure over-approximating prefilters; when the words contain no
        // letters/bigrams the prefilter is disabled and every comment is
        // string-searched directly.
        if (required_mask == 0 && required_bigram_mask == 0) {
            AUDIT_PATH("q13 fast nomask");
        } else {
            AUDIT_PATH("q13 fast masked");
        }
        {
            PROFILE_SCOPE("q13_orders_scan");
            const uint32_t order_count = static_cast<uint32_t>(orders.row_count);
#pragma omp parallel
            {
                std::vector<int32_t> local;
#pragma omp for schedule(dynamic, 16384) nowait
                for (uint32_t i = 0; i < order_count; ++i) {
                    bool matches = false;
                    if (required_mask == 0 ||
                        (comment_masks[i] & required_mask) == required_mask) {
                        if (required_bigram_mask == 0 ||
                            (comment_bigrams[i] & required_bigram_mask) ==
                                required_bigram_mask) {
                            const uint32_t start = comment_offsets[i];
                            const uint32_t end = comment_offsets[i + 1];
                            matches = q13::matches_pattern(
                                comment_data + start,
                                static_cast<size_t>(end - start), word1,
                                word1_len, word2, word2_len);
                        }
                    }
                    if (matches) {
                        local.push_back(orders.custkey[i]);
                    }
                }
#pragma omp critical
                for (const int32_t custkey : local) {
                    if (custkey >= 0 && custkey <= max_custkey) {
                        matched_counts[static_cast<size_t>(custkey)] += 1;
                    }
                }
            }
        }

        int32_t max_count = 0;
#pragma omp parallel for schedule(static) reduction(max : max_count)
        for (uint32_t i = 0; i < customer.row_count; ++i) {
            const int32_t custkey = customer.custkey[i];
            int32_t base = 0;
            if (custkey >= 0 &&
                static_cast<size_t>(custkey) + 1 < cust_offsets.size()) {
                base = static_cast<int32_t>(
                    cust_offsets[static_cast<size_t>(custkey) + 1] -
                    cust_offsets[static_cast<size_t>(custkey)]);
            }
            if (base > max_count) {
                max_count = base;
            }
        }

        std::vector<int64_t> dist_counts(static_cast<size_t>(max_count) + 1, 0);
        {
            PROFILE_SCOPE("q13_customer_scan");
#pragma omp parallel
            {
                std::vector<int64_t> local(dist_counts.size(), 0);
#pragma omp for schedule(static) nowait
                for (uint32_t i = 0; i < customer.row_count; ++i) {
                    const int32_t custkey = customer.custkey[i];
                    int32_t count = 0;
                    if (custkey >= 0 &&
                        static_cast<size_t>(custkey) + 1 < cust_offsets.size()) {
                        count = static_cast<int32_t>(
                            cust_offsets[static_cast<size_t>(custkey) + 1] -
                            cust_offsets[static_cast<size_t>(custkey)]);
                    }
                    if (custkey >= 0 && custkey <= max_custkey) {
                        count -= matched_counts[static_cast<size_t>(custkey)];
                    }
                    if (count >= 0 &&
                        static_cast<size_t>(count) < local.size()) {
                        local[static_cast<size_t>(count)] += 1;
                    }
                }
#pragma omp critical
                for (size_t c = 0; c < local.size(); ++c) {
                    dist_counts[c] += local[c];
                }
            }
        }

        {
            PROFILE_SCOPE("q13_agg_finalize");
            results.reserve(dist_counts.size());
            for (size_t i = 0; i < dist_counts.size(); ++i) {
                const int64_t value = dist_counts[i];
                if (value == 0) {
                    continue;
                }
                Q13ResultRow row;
                row.c_count = static_cast<int64_t>(i);
                row.custdist = value;
                results.push_back(row);
            }
            TRACE_SET(groups_created, results.size());
            TRACE_SET(agg_rows_emitted, results.size());
        }

        {
            PROFILE_SCOPE("q13_sort");
            TRACE_SET(sort_rows_in, results.size());
            std::sort(results.begin(), results.end(),
                      [](const Q13ResultRow& a, const Q13ResultRow& b) {
                          if (a.custdist != b.custdist) {
                              return a.custdist > b.custdist;
                          }
                          return a.c_count > b.c_count;
                      });
            TRACE_SET(sort_rows_out, results.size());
        }
        TRACE_SET(query_output_rows, results.size());
    }
#ifdef TRACE
    q13_trace::emit();
#endif
    return results;
}

void write_q13_csv(const std::string& filename, const std::vector<Q13ResultRow>& rows) {
    std::ofstream out(filename);
    out << "c_count,custdist\n";
    for (const auto& row : rows) {
        out << row.c_count << ',' << row.custdist << "\n";
    }
}