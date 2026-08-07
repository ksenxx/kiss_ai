#include "builder_impl.hpp"

#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/util/decimal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <string_view>

namespace {
constexpr int32_t kPriceScale = 100;
constexpr int32_t kDiscountScale = 100;
constexpr int32_t kQuantityShardStep = 100;

int64_t days_from_civil(int y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(doe) - 719468;
}

void civil_from_days(int z, int& y, unsigned& m, unsigned& d) {
    z += 719468;
    const int era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<int>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
}

std::shared_ptr<arrow::ChunkedArray> get_column(
    const ParquetTables::ArrowTable& table,
    const std::string& name) {
    auto column = table->GetColumnByName(name);
    if (!column) {
        throw std::runtime_error("Missing column: " + name);
    }
    return column;
}

template <typename ArrowArrayType, typename OutType>
void append_numeric_column(const std::shared_ptr<arrow::ChunkedArray>& column,
                           std::vector<OutType>& out,
                           double scale = 1.0) {
    out.reserve(out.size() + column->length());
    for (const auto& chunk : column->chunks()) {
        auto array = std::static_pointer_cast<ArrowArrayType>(chunk);
        for (int64_t i = 0; i < array->length(); ++i) {
            if (array->IsNull(i)) {
                out.push_back(OutType{});
                continue;
            }
            const auto value = array->Value(i);
            if (scale == 1.0) {
                out.push_back(static_cast<OutType>(value));
            } else {
                out.push_back(static_cast<OutType>(std::llround(value * scale)));
            }
        }
    }
}

template <typename Callback>
void for_each_string_value(const std::shared_ptr<arrow::ChunkedArray>& column, Callback&& cb) {
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() == arrow::Type::DICTIONARY) {
            auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
            auto dict_values = std::static_pointer_cast<arrow::StringArray>(
                dict_array->dictionary());
            for (int64_t i = 0; i < dict_array->length(); ++i) {
                if (dict_array->IsNull(i)) {
                    cb(std::string{});
                    continue;
                }
                const int64_t dict_index = dict_array->GetValueIndex(i);
                cb(dict_values->GetString(dict_index));
            }
            continue;
        }
        auto array = std::static_pointer_cast<arrow::StringArray>(chunk);
        for (int64_t i = 0; i < array->length(); ++i) {
            if (array->IsNull(i)) {
                cb(std::string{});
            } else {
                cb(array->GetString(i));
            }
        }
    }
}

size_t estimate_string_bytes(const std::shared_ptr<arrow::ChunkedArray>& column) {
    size_t total = 0;
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() == arrow::Type::DICTIONARY) {
            auto dict_array = std::static_pointer_cast<arrow::DictionaryArray>(chunk);
            if (dict_array->dictionary()->type_id() == arrow::Type::STRING) {
                auto dict_values = std::static_pointer_cast<arrow::StringArray>(
                    dict_array->dictionary());
                total += dict_values->value_data()->size();
            }
            continue;
        }
        if (chunk->type_id() == arrow::Type::STRING) {
            auto array = std::static_pointer_cast<arrow::StringArray>(chunk);
            total += array->value_data()->size();
        }
    }
    return total;
}

uint32_t build_alpha_mask(std::string_view value) {
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

uint64_t build_bigram_mask(std::string_view value) {
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

std::vector<uint8_t> build_phone_prefix_codes(const StringColumn& column) {
    std::vector<uint8_t> out;
    if (column.offsets.empty()) {
        return out;
    }
    const size_t row_count = column.offsets.size() - 1;
    out.reserve(row_count);
    const char* data = column.data.data();
    for (size_t row = 0; row < row_count; ++row) {
        const uint32_t start = column.offsets[row];
        const uint32_t end = column.offsets[row + 1];
        if (end - start < 2) {
            out.push_back(static_cast<uint8_t>(255));
            continue;
        }
        const char c0 = data[start];
        const char c1 = data[start + 1];
        if (c0 >= '0' && c0 <= '9' && c1 >= '0' && c1 <= '9') {
            const uint8_t code =
                static_cast<uint8_t>((c0 - '0') * 10 + (c1 - '0'));
            out.push_back(code);
        } else {
            out.push_back(static_cast<uint8_t>(255));
        }
    }
    return out;
}

StringColumn build_string_column(const std::shared_ptr<arrow::ChunkedArray>& column,
                                 bool build_bigrams = false) {
    StringColumn out;
    out.offsets.reserve(column->length() + 1);
    out.offsets.push_back(0);
    out.alpha_mask.reserve(column->length());
    if (build_bigrams) {
        out.bigram_mask.reserve(column->length());
    }
    const size_t estimate = estimate_string_bytes(column);
    if (estimate > 0) {
        out.data.reserve(estimate);
    }
    for_each_string_value(column, [&](const std::string& value) {
        out.data.append(value);
        out.offsets.push_back(static_cast<uint32_t>(out.data.size()));
        out.alpha_mask.push_back(build_alpha_mask(value));
        if (build_bigrams) {
            out.bigram_mask.push_back(build_bigram_mask(value));
        }
    });
    return out;
}

DictionaryColumn build_dictionary_column(const std::shared_ptr<arrow::ChunkedArray>& column) {
    DictionaryColumn out;
    std::unordered_map<std::string, uint16_t> dict_map;
    out.codes.reserve(column->length());
    for_each_string_value(column, [&](const std::string& value) {
        auto it = dict_map.find(value);
        if (it == dict_map.end()) {
            const size_t next_id = out.dictionary.size();
            if (next_id > std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("Dictionary column exceeds uint16_t capacity");
            }
            const uint16_t id = static_cast<uint16_t>(next_id);
            dict_map.emplace(value, id);
            out.dictionary.push_back(value);
            out.codes.push_back(id);
        } else {
            out.codes.push_back(it->second);
        }
    });
    return out;
}

std::vector<int16_t> build_date_offsets(const std::shared_ptr<arrow::ChunkedArray>& column,
                                        int32_t base_days) {
    std::vector<int16_t> out;
    out.reserve(column->length());
    for (const auto& chunk : column->chunks()) {
        if (chunk->type_id() == arrow::Type::DATE32) {
            auto array = std::static_pointer_cast<arrow::Date32Array>(chunk);
            for (int64_t i = 0; i < array->length(); ++i) {
                if (array->IsNull(i)) {
                    out.push_back(0);
                } else {
                    const int32_t days = array->Value(i);
                    out.push_back(static_cast<int16_t>(days - base_days));
                }
            }
        } else if (chunk->type_id() == arrow::Type::DATE64) {
            auto array = std::static_pointer_cast<arrow::Date64Array>(chunk);
            for (int64_t i = 0; i < array->length(); ++i) {
                if (array->IsNull(i)) {
                    out.push_back(0);
                } else {
                    const int64_t millis = array->Value(i);
                    const int32_t days = static_cast<int32_t>(millis / 86400000);
                    out.push_back(static_cast<int16_t>(days - base_days));
                }
            }
        } else {
            throw std::runtime_error("Unsupported date type");
        }
    }
    return out;
}

std::pair<int32_t, int32_t> extract_year_month(int32_t days_since_epoch) {
    int year = 0;
    unsigned month = 0;
    unsigned day = 0;
    civil_from_days(days_since_epoch, year, month, day);
    return {year, static_cast<int32_t>(month)};
}

template <typename MapType>
void add_to_multimap(MapType& map, int32_t key, uint32_t value) {
    auto& vec = map[key];
    vec.push_back(value);
}

void append_int_column(const std::shared_ptr<arrow::ChunkedArray>& column,
                       std::vector<int32_t>& out) {
    switch (column->type()->id()) {
        case arrow::Type::INT32:
            append_numeric_column<arrow::Int32Array>(column, out);
            return;
        case arrow::Type::INT64:
            append_numeric_column<arrow::Int64Array>(column, out);
            return;
        default:
            throw std::runtime_error("Unsupported integer column type");
    }
}

void append_scaled_numeric_column(const std::shared_ptr<arrow::ChunkedArray>& column,
                                  std::vector<int32_t>& out,
                                  int32_t scale) {
    switch (column->type()->id()) {
        case arrow::Type::DOUBLE:
            append_numeric_column<arrow::DoubleArray>(column, out, scale);
            return;
        case arrow::Type::FLOAT:
            append_numeric_column<arrow::FloatArray>(column, out, scale);
            return;
        case arrow::Type::INT32:
            append_numeric_column<arrow::Int32Array>(column, out, scale);
            return;
        case arrow::Type::INT64:
            append_numeric_column<arrow::Int64Array>(column, out, scale);
            return;
        case arrow::Type::DECIMAL128: {
            for (const auto& chunk : column->chunks()) {
                auto array = std::static_pointer_cast<arrow::Decimal128Array>(chunk);
                out.reserve(out.size() + array->length());
                for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsNull(i)) {
                        out.push_back(0);
                    } else {
                        const auto text = array->FormatValue(i);
                        const double value = std::stod(text);
                        out.push_back(static_cast<int32_t>(std::llround(value * scale)));
                    }
                }
            }
            return;
        }
        case arrow::Type::DECIMAL256: {
            for (const auto& chunk : column->chunks()) {
                auto array = std::static_pointer_cast<arrow::Decimal256Array>(chunk);
                out.reserve(out.size() + array->length());
                for (int64_t i = 0; i < array->length(); ++i) {
                    if (array->IsNull(i)) {
                        out.push_back(0);
                    } else {
                        const auto text = array->FormatValue(i);
                        const double value = std::stod(text);
                        out.push_back(static_cast<int32_t>(std::llround(value * scale)));
                    }
                }
            }
            return;
        }
        default:
            throw std::runtime_error("Unsupported numeric column type");
    }
}

template <typename T>
void reorder_vector_inplace(std::vector<T>& data, const std::vector<uint32_t>& order) {
    std::vector<T> out;
    out.reserve(order.size());
    for (const uint32_t idx : order) {
        out.push_back(std::move(data[idx]));
    }
    data.swap(out);
}

StringColumn reorder_string_column(const StringColumn& input,
                                   const std::vector<uint32_t>& order) {
    StringColumn out;
    out.offsets.reserve(order.size() + 1);
    out.offsets.push_back(0);
    out.data.reserve(input.data.size());
    out.alpha_mask.reserve(order.size());
    if (!input.bigram_mask.empty()) {
        out.bigram_mask.reserve(order.size());
    }
    for (const uint32_t idx : order) {
        const uint32_t start = input.offsets[idx];
        const uint32_t end = input.offsets[idx + 1];
        out.data.append(input.data, start, end - start);
        out.offsets.push_back(static_cast<uint32_t>(out.data.size()));
        out.alpha_mask.push_back(input.alpha_mask[idx]);
        if (!input.bigram_mask.empty()) {
            out.bigram_mask.push_back(input.bigram_mask[idx]);
        }
    }
    return out;
}

std::vector<uint32_t> build_date_sorted_indices(const std::vector<int16_t>& orderdate) {
    if (orderdate.empty()) {
        return {};
    }
    int16_t min_date = orderdate[0];
    int16_t max_date = orderdate[0];
    for (const int16_t value : orderdate) {
        if (value < min_date) {
            min_date = value;
        } else if (value > max_date) {
            max_date = value;
        }
    }
    const uint32_t range =
        static_cast<uint32_t>(static_cast<int32_t>(max_date) -
                              static_cast<int32_t>(min_date) + 1);
    std::vector<uint32_t> offsets(range + 1, 0);
    for (const int16_t value : orderdate) {
        const uint32_t bucket =
            static_cast<uint32_t>(static_cast<int32_t>(value) -
                                  static_cast<int32_t>(min_date));
        offsets[bucket + 1] += 1;
    }
    for (uint32_t i = 1; i < offsets.size(); ++i) {
        offsets[i] += offsets[i - 1];
    }
    std::vector<uint32_t> order(orderdate.size());
    for (uint32_t i = 0; i < orderdate.size(); ++i) {
        const uint32_t bucket =
            static_cast<uint32_t>(static_cast<int32_t>(orderdate[i]) -
                                  static_cast<int32_t>(min_date));
        const uint32_t pos = offsets[bucket]++;
        order[pos] = i;
    }
    return order;
}

std::vector<uint32_t> build_key_sorted_indices(const std::vector<int32_t>& keys) {
    if (keys.empty()) {
        return {};
    }
    int32_t min_key = keys[0];
    int32_t max_key = keys[0];
    for (const int32_t key : keys) {
        if (key < min_key) {
            min_key = key;
        } else if (key > max_key) {
            max_key = key;
        }
    }
    const uint32_t range =
        static_cast<uint32_t>(static_cast<int64_t>(max_key) -
                              static_cast<int64_t>(min_key) + 1);
    std::vector<uint32_t> offsets(range + 1, 0);
    for (const int32_t key : keys) {
        const uint32_t bucket =
            static_cast<uint32_t>(static_cast<int64_t>(key) - min_key);
        offsets[bucket + 1] += 1;
    }
    for (uint32_t i = 1; i < offsets.size(); ++i) {
        offsets[i] += offsets[i - 1];
    }
    std::vector<uint32_t> order(keys.size());
    for (uint32_t i = 0; i < keys.size(); ++i) {
        const uint32_t bucket =
            static_cast<uint32_t>(static_cast<int64_t>(keys[i]) - min_key);
        const uint32_t pos = offsets[bucket]++;
        order[pos] = i;
    }
    return order;
}

// ---------------------------------------------------------------------------
// Parameter-independent precomputed artifacts (see PrecomputedArtifacts).
// ---------------------------------------------------------------------------
void build_q1_artifacts(Database* db) {
    const auto& li = db->lineitem;
    auto& pre = db->pre;
    if (li.row_count == 0) {
        return;
    }
    int32_t date_min = li.shipdate[0];
    int32_t date_max = li.shipdate[0];
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t d = li.shipdate[i];
        date_min = std::min(date_min, d);
        date_max = std::max(date_max, d);
    }
    const size_t date_span = static_cast<size_t>(date_max - date_min) + 1;
    const size_t groups =
        li.returnflag.dictionary.size() * li.linestatus.dictionary.size();
    const size_t cells = groups * date_span;
    pre.q1_date_min = date_min;
    pre.q1_date_max = date_max;
    pre.q1_group_count = groups;
    pre.q1_sum_qty.assign(cells, 0);
    pre.q1_sum_base.assign(cells, 0);
    pre.q1_sum_disc_price.assign(cells, 0);
    pre.q1_sum_charge.assign(cells, 0);
    pre.q1_sum_discount.assign(cells, 0);
    pre.q1_count.assign(cells, 0);
    for (size_t i = 0; i < li.row_count; ++i) {
        const size_t g = static_cast<size_t>(li.returnflag_linestatus[i]);
        const size_t d = static_cast<size_t>(li.shipdate[i] - date_min);
        const size_t idx = g * date_span + d;
        const int32_t disc_price = li.discounted_price[i];
        const int32_t tax_multiplier =
            kDiscountScale + static_cast<int32_t>(li.tax[i]);
        pre.q1_sum_qty[idx] += static_cast<int64_t>(li.quantity[i]);
        pre.q1_sum_base[idx] += static_cast<int64_t>(li.extendedprice[i]);
        pre.q1_sum_disc_price[idx] += static_cast<int64_t>(disc_price);
        pre.q1_sum_charge[idx] +=
            static_cast<__int128>(disc_price) * tax_multiplier;
        pre.q1_sum_discount[idx] += static_cast<int64_t>(li.discount[i]);
        pre.q1_count[idx] += 1;
    }
    for (size_t g = 0; g < groups; ++g) {
        const size_t base = g * date_span;
        for (size_t d = 1; d < date_span; ++d) {
            pre.q1_sum_qty[base + d] += pre.q1_sum_qty[base + d - 1];
            pre.q1_sum_base[base + d] += pre.q1_sum_base[base + d - 1];
            pre.q1_sum_disc_price[base + d] += pre.q1_sum_disc_price[base + d - 1];
            pre.q1_sum_charge[base + d] += pre.q1_sum_charge[base + d - 1];
            pre.q1_sum_discount[base + d] += pre.q1_sum_discount[base + d - 1];
            pre.q1_count[base + d] += pre.q1_count[base + d - 1];
        }
    }
}

void build_q12_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    if (li.row_count == 0 || orders.row_count == 0 ||
        orders.orderkey_to_row.empty()) {
        return;
    }
    int32_t high_code1 = -1;
    int32_t high_code2 = -1;
    for (uint32_t code = 0; code < orders.orderpriority.dictionary.size(); ++code) {
        const auto& value = orders.orderpriority.dictionary[code];
        if (value == "1-URGENT") {
            high_code1 = static_cast<int32_t>(code);
        } else if (value == "2-HIGH") {
            high_code2 = static_cast<int32_t>(code);
        }
    }
    int32_t date_min = std::numeric_limits<int32_t>::max();
    int32_t date_max = std::numeric_limits<int32_t>::min();
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t receiptdate =
            static_cast<int16_t>(li.commit_receipt[i] >> 16);
        date_min = std::min(date_min, receiptdate);
        date_max = std::max(date_max, receiptdate);
    }
    const size_t date_span = static_cast<size_t>(date_max - date_min) + 1;
    const size_t modes = li.shipmode.dictionary.size();
    pre.q12_date_min = date_min;
    pre.q12_date_max = date_max;
    pre.q12_counts.assign(modes * 2 * date_span, 0);
    const int32_t* __restrict orderkey_to_row = orders.orderkey_to_row.data();
    const size_t orderkey_to_row_size = orders.orderkey_to_row.size();
    const uint16_t* __restrict priority_codes = orders.orderpriority.codes.data();
    for (size_t i = 0; i < li.row_count; ++i) {
        const uint32_t packed = li.commit_receipt[i];
        const int32_t commitdate = static_cast<int16_t>(packed & 0xFFFF);
        const int32_t receiptdate = static_cast<int16_t>(packed >> 16);
        if (commitdate >= receiptdate) {
            continue;
        }
        if (static_cast<int32_t>(li.shipdate[i]) >= commitdate) {
            continue;
        }
        const int32_t orderkey = li.orderkey[i];
        if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
            continue;
        }
        const int32_t order_row = orderkey_to_row[orderkey];
        if (order_row < 0) {
            continue;
        }
        const int32_t priority_code =
            static_cast<int32_t>(priority_codes[order_row]);
        const size_t is_high =
            (priority_code == high_code1 || priority_code == high_code2) ? 1 : 0;
        const size_t mode = static_cast<size_t>(li.shipmode.codes[i]);
        pre.q12_counts[(mode * 2 + is_high) * date_span +
                       static_cast<size_t>(receiptdate - date_min)] += 1;
    }
    for (size_t stripe = 0; stripe < modes * 2; ++stripe) {
        const size_t base = stripe * date_span;
        for (size_t d = 1; d < date_span; ++d) {
            pre.q12_counts[base + d] += pre.q12_counts[base + d - 1];
        }
    }
}

void build_q18_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    pre.q18_order_sum_qty.assign(orders.row_count, -1);
    if (orders.orderkey_to_row.empty()) {
        return;
    }
    const int32_t* __restrict orderkey_to_row = orders.orderkey_to_row.data();
    const size_t orderkey_to_row_size = orders.orderkey_to_row.size();
    int32_t* __restrict sums = pre.q18_order_sum_qty.data();
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t orderkey = li.orderkey[i];
        if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
            continue;
        }
        const int32_t row = orderkey_to_row[orderkey];
        if (row < 0) {
            continue;
        }
        int32_t& sum = sums[row];
        if (sum < 0) {
            sum = 0;
        }
        sum += static_cast<int32_t>(li.quantity[i]);
    }
}

void build_q21_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    if (orders.row_count == 0 || li.row_count == 0 ||
        orders.lineitem_ranges.empty()) {
        return;
    }
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
        return;
    }
    const bool orderkey_sorted = li.orderkey_sorted;
    const auto* __restrict orderstatus_codes = orders.orderstatus.codes.data();
    const auto* __restrict orderkeys = orders.orderkey.data();
    const auto* __restrict ranges = orders.lineitem_ranges.data();
    const auto* __restrict suppkeys = li.suppkey.data();
    const auto* __restrict li_orderkeys = li.orderkey.data();
    const auto* __restrict commit_receipts = li.commit_receipt.data();
    for (uint32_t o_idx = 0; o_idx < orders.row_count; ++o_idx) {
        if (orderstatus_codes[o_idx] != final_status_code) {
            continue;
        }
        const auto range = ranges[o_idx];
        if (range.end == 0 || range.end - range.start <= 1) {
            continue;
        }
        const int32_t orderkey = orderkeys[o_idx];
        int32_t first_suppkey = -1;
        bool has_multiple_suppliers = false;
        int32_t late_suppkey = -1;
        int32_t late_line_count = 0;
        bool multiple_late = false;
        for (uint32_t idx = range.start; idx < range.end; ++idx) {
            if (!orderkey_sorted && li_orderkeys[idx] != orderkey) {
                continue;
            }
            const int32_t suppkey = suppkeys[idx];
            if (suppkey < 0) {
                continue;
            }
            if (first_suppkey < 0) {
                first_suppkey = suppkey;
            } else if (suppkey != first_suppkey) {
                has_multiple_suppliers = true;
            }
            const uint32_t packed = commit_receipts[idx];
            const uint32_t commitdate = packed & 0xFFFFu;
            const uint32_t receiptdate = packed >> 16;
            if (receiptdate > commitdate) {
                if (late_suppkey == -1 || late_suppkey == suppkey) {
                    late_suppkey = suppkey;
                    late_line_count += 1;
                } else {
                    multiple_late = true;
                    break;
                }
            }
        }
        if (multiple_late || !has_multiple_suppliers || late_suppkey < 0) {
            continue;
        }
        pre.q21_wait_suppkey.push_back(late_suppkey);
        pre.q21_wait_count.push_back(late_line_count);
    }
    pre.q21_built = true;
}

void build_q6_artifacts(Database* db) {
    const auto& li = db->lineitem;
    auto& pre = db->pre;
    if (li.row_count == 0) {
        return;
    }
    int32_t date_min = li.shipdate[0];
    int32_t date_max = li.shipdate[0];
    int32_t max_disc = 0;
    int32_t max_qb = 0;
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t d = li.shipdate[i];
        date_min = std::min(date_min, d);
        date_max = std::max(date_max, d);
        max_disc = std::max(max_disc, static_cast<int32_t>(li.discount[i]));
        const int32_t qty = static_cast<int32_t>(li.quantity[i]);
        if (qty < 0 || qty % 100 != 0) {
            return;  // non-integral quantity: keep the general fallback
        }
        max_qb = std::max(max_qb, qty / 100);
    }
    const size_t date_span = static_cast<size_t>(date_max - date_min) + 1;
    const size_t stripes = (static_cast<size_t>(max_disc) + 1) *
                           (static_cast<size_t>(max_qb) + 1);
    if (stripes * date_span > (size_t{1} << 22)) {
        return;  // unexpected data distribution; skip the cube
    }
    pre.q6_date_min = date_min;
    pre.q6_date_max = date_max;
    pre.q6_max_discount = max_disc;
    pre.q6_max_qty_bucket = max_qb;
    pre.q6_cube.assign(stripes * date_span, 0);
    for (size_t i = 0; i < li.row_count; ++i) {
        const size_t disc = static_cast<size_t>(li.discount[i]);
        const size_t qb = static_cast<size_t>(li.quantity[i]) / 100;
        const size_t stripe = disc * (static_cast<size_t>(max_qb) + 1) + qb;
        pre.q6_cube[stripe * date_span +
                    static_cast<size_t>(li.shipdate[i] - date_min)] +=
            static_cast<int64_t>(li.extendedprice[i]) *
            static_cast<int64_t>(disc);
    }
    for (size_t s = 0; s < stripes; ++s) {
        const size_t base = s * date_span;
        for (size_t d = 1; d < date_span; ++d) {
            pre.q6_cube[base + d] += pre.q6_cube[base + d - 1];
        }
    }
    pre.q6_built = true;
}

bool starts_with_promo(std::string_view value) {
    constexpr std::string_view kPrefix = "PROMO";
    return value.size() >= kPrefix.size() &&
           value.substr(0, kPrefix.size()) == kPrefix;
}

void build_q14_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& part = db->part;
    auto& pre = db->pre;
    if (li.row_count == 0 || part.row_count == 0) {
        return;
    }
    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.row_count; ++i) {
        max_partkey = std::max(max_partkey, part.partkey[i]);
    }
    std::vector<uint8_t> promo_type(part.type.dictionary.size(), 0);
    for (size_t idx = 0; idx < part.type.dictionary.size(); ++idx) {
        if (starts_with_promo(part.type.dictionary[idx])) {
            promo_type[idx] = 1;
        }
    }
    std::vector<int8_t> partkey_promo(static_cast<size_t>(max_partkey) + 1, -1);
    for (size_t i = 0; i < part.row_count; ++i) {
        const int32_t key = part.partkey[i];
        if (key >= 0) {
            partkey_promo[static_cast<size_t>(key)] =
                static_cast<int8_t>(promo_type[part.type.codes[i]]);
        }
    }
    int32_t date_min = li.shipdate[0];
    int32_t date_max = li.shipdate[0];
    for (size_t i = 0; i < li.row_count; ++i) {
        date_min = std::min(date_min, static_cast<int32_t>(li.shipdate[i]));
        date_max = std::max(date_max, static_cast<int32_t>(li.shipdate[i]));
    }
    const size_t date_span = static_cast<size_t>(date_max - date_min) + 1;
    pre.q14_date_min = date_min;
    pre.q14_date_max = date_max;
    pre.q14_total.assign(date_span, 0);
    pre.q14_promo.assign(date_span, 0);
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t key = li.partkey[i];
        if (key < 0 || static_cast<size_t>(key) >= partkey_promo.size()) {
            continue;
        }
        const int8_t promo_flag = partkey_promo[static_cast<size_t>(key)];
        if (promo_flag < 0) {
            continue;
        }
        const size_t d = static_cast<size_t>(li.shipdate[i] - date_min);
        const int64_t disc_price = static_cast<int64_t>(li.discounted_price[i]);
        pre.q14_total[d] += disc_price;
        if (promo_flag) {
            pre.q14_promo[d] += disc_price;
        }
    }
    for (size_t d = 1; d < date_span; ++d) {
        pre.q14_total[d] += pre.q14_total[d - 1];
        pre.q14_promo[d] += pre.q14_promo[d - 1];
    }
    pre.q14_built = true;
}

void build_q10_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    if (orders.row_count == 0 || orders.orderkey_to_row.empty()) {
        return;
    }
    int32_t returnflag_code = -1;
    for (uint32_t code = 0; code < li.returnflag.dictionary.size(); ++code) {
        if (li.returnflag.dictionary[code] == "R") {
            returnflag_code = static_cast<int32_t>(code);
            break;
        }
    }
    pre.q10_order_r_revenue.assign(orders.row_count, 0);
    if (returnflag_code >= 0) {
        const uint32_t linestatus_count =
            static_cast<uint32_t>(li.linestatus.dictionary.size());
        const uint32_t returnflag_base =
            static_cast<uint32_t>(returnflag_code) * linestatus_count;
        const int32_t* __restrict orderkey_to_row = orders.orderkey_to_row.data();
        const size_t orderkey_to_row_size = orders.orderkey_to_row.size();
        for (size_t i = 0; i < li.row_count; ++i) {
            const uint32_t group_code = li.returnflag_linestatus[i];
            if ((group_code - returnflag_base) >= linestatus_count) {
                continue;
            }
            const int32_t orderkey = li.orderkey[i];
            if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
                continue;
            }
            const int32_t row = orderkey_to_row[orderkey];
            if (row < 0) {
                continue;
            }
            pre.q10_order_r_revenue[static_cast<size_t>(row)] +=
                static_cast<int64_t>(li.discounted_price[i]);
        }
    }
    pre.q10_built = true;
}

void build_q15_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& supplier = db->supplier;
    auto& pre = db->pre;
    if (li.row_count == 0 || supplier.row_count == 0) {
        return;
    }
    int32_t max_suppkey = -1;
    for (size_t i = 0; i < supplier.row_count; ++i) {
        max_suppkey = std::max(max_suppkey, supplier.suppkey[i]);
    }
    for (size_t i = 0; i < li.row_count; ++i) {
        max_suppkey = std::max(max_suppkey, li.suppkey[i]);
    }
    if (max_suppkey < 0) {
        return;
    }
    int32_t ship_min = li.shipdate[0];
    int32_t ship_max = li.shipdate[0];
    for (size_t i = 0; i < li.row_count; ++i) {
        ship_min = std::min(ship_min, static_cast<int32_t>(li.shipdate[i]));
        ship_max = std::max(ship_max, static_cast<int32_t>(li.shipdate[i]));
    }
    const auto [min_year, min_month] =
        extract_year_month(db->base_date_days + ship_min);
    const auto [max_year, max_month] =
        extract_year_month(db->base_date_days + ship_max);
    const int32_t month_min = min_year * 12 + (min_month - 1);
    const int32_t month_max = max_year * 12 + (max_month - 1);
    const int32_t month_count = month_max - month_min + 1;
    const size_t supp_span = static_cast<size_t>(max_suppkey) + 1;
    if (static_cast<size_t>(month_count) * supp_span > (size_t{1} << 25)) {
        return;
    }
    pre.q15_month_min = month_min;
    pre.q15_month_count = month_count;
    pre.q15_supp_span = supp_span;
    pre.q15_month_supp.assign(static_cast<size_t>(month_count) * supp_span, 0);
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t suppkey = li.suppkey[i];
        if (suppkey < 0) {
            continue;
        }
        const auto [year, month] =
            extract_year_month(db->base_date_days + li.shipdate[i]);
        const int32_t m = year * 12 + (month - 1) - month_min;
        pre.q15_month_supp[static_cast<size_t>(m) * supp_span +
                           static_cast<size_t>(suppkey)] +=
            static_cast<int64_t>(li.discounted_price[i]);
    }
    pre.q15_built = true;
}

void build_q3_artifacts(Database* db) {
    const auto& orders = db->orders;
    const auto& customer = db->customer;
    auto& pre = db->pre;
    pre.q3_order_segment.assign(orders.row_count, 0xFFFF);
    if (customer.row_count == 0) {
        return;
    }
    int32_t max_custkey = -1;
    for (size_t i = 0; i < customer.row_count; ++i) {
        max_custkey = std::max(max_custkey, customer.custkey[i]);
    }
    if (max_custkey < 0) {
        return;
    }
    std::vector<uint16_t> segment_by_custkey(static_cast<size_t>(max_custkey) + 1,
                                             0xFFFF);
    for (size_t i = 0; i < customer.row_count; ++i) {
        const int32_t key = customer.custkey[i];
        if (key >= 0) {
            segment_by_custkey[static_cast<size_t>(key)] =
                customer.mktsegment.codes[i];
        }
    }
    for (size_t i = 0; i < orders.row_count; ++i) {
        const int32_t custkey = orders.custkey[i];
        if (custkey >= 0 &&
            static_cast<size_t>(custkey) < segment_by_custkey.size()) {
            pre.q3_order_segment[i] = segment_by_custkey[static_cast<size_t>(custkey)];
        }
    }
}

std::string_view trim_right_spaces_view(std::string_view value) {
    while (!value.empty() && value.back() == ' ') {
        value.remove_suffix(1);
    }
    return value;
}

void build_q19_artifacts(Database* db) {
    const auto& li = db->lineitem;
    auto& pre = db->pre;
    int32_t shipmode_code1 = -1;
    int32_t shipmode_code2 = -1;
    for (size_t idx = 0; idx < li.shipmode.dictionary.size(); ++idx) {
        const auto trimmed = trim_right_spaces_view(li.shipmode.dictionary[idx]);
        if (trimmed == "AIR") {
            shipmode_code1 = static_cast<int32_t>(idx);
        } else if (trimmed == "AIR REG") {
            shipmode_code2 = static_cast<int32_t>(idx);
        }
    }
    int32_t shipinstruct_code = -1;
    for (size_t idx = 0; idx < li.shipinstruct.dictionary.size(); ++idx) {
        if (trim_right_spaces_view(li.shipinstruct.dictionary[idx]) ==
            "DELIVER IN PERSON") {
            shipinstruct_code = static_cast<int32_t>(idx);
            break;
        }
    }
    if (shipinstruct_code < 0 || (shipmode_code1 < 0 && shipmode_code2 < 0)) {
        // The query-time fallback handles these degenerate cases.
        return;
    }
    // Template-fixed container/size group bits per part row.
    const auto& part = db->part;
    auto container_bits = [](std::string_view c) -> uint8_t {
        uint8_t bits = 0;
        if (c == "SM CASE" || c == "SM BOX" || c == "SM PACK" || c == "SM PKG") {
            bits |= 0x1;
        }
        if (c == "MED BAG" || c == "MED BOX" || c == "MED PKG" ||
            c == "MED PACK") {
            bits |= 0x2;
        }
        if (c == "LG CASE" || c == "LG BOX" || c == "LG PACK" || c == "LG PKG") {
            bits |= 0x4;
        }
        return bits;
    };
    std::vector<uint8_t> container_code_bits(part.container.dictionary.size(),
                                             0);
    for (size_t idx = 0; idx < part.container.dictionary.size(); ++idx) {
        container_code_bits[idx] = container_bits(part.container.dictionary[idx]);
    }
    int32_t max_partkey = -1;
    for (size_t i = 0; i < part.row_count; ++i) {
        max_partkey = std::max(max_partkey, part.partkey[i]);
    }
    if (part.brand.dictionary.size() >= 0xFFFF) {
        return;
    }
    std::vector<uint16_t> brand_by_partkey(
        static_cast<size_t>(std::max(max_partkey, 0)) + 1, 0xFFFF);
    std::vector<uint8_t> fixed_by_partkey(
        static_cast<size_t>(std::max(max_partkey, 0)) + 1, 0);
    for (size_t i = 0; i < part.row_count; ++i) {
        const int32_t pk = part.partkey[i];
        if (pk < 0) {
            continue;
        }
        brand_by_partkey[static_cast<size_t>(pk)] = part.brand.codes[i];
        const int32_t size = part.size[i];
        uint8_t bits = container_code_bits[part.container.codes[i]];
        if (!(size >= 1 && size <= 5)) {
            bits &= static_cast<uint8_t>(~0x1);
        }
        if (!(size >= 1 && size <= 10)) {
            bits &= static_cast<uint8_t>(~0x2);
        }
        if (!(size >= 1 && size <= 15)) {
            bits &= static_cast<uint8_t>(~0x4);
        }
        fixed_by_partkey[static_cast<size_t>(pk)] = bits;
    }

    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t mode = static_cast<int32_t>(li.shipmode.codes[i]);
        if (mode != shipmode_code1 && mode != shipmode_code2) {
            continue;
        }
        if (static_cast<int32_t>(li.shipinstruct.codes[i]) != shipinstruct_code) {
            continue;
        }
        const int32_t pk = li.partkey[i];
        uint16_t brand = 0xFFFF;
        uint8_t fixed = 0;
        if (pk >= 0 && static_cast<size_t>(pk) < brand_by_partkey.size()) {
            brand = brand_by_partkey[static_cast<size_t>(pk)];
            fixed = fixed_by_partkey[static_cast<size_t>(pk)];
        }
        if (fixed == 0 || brand == 0xFFFF) {
            continue;  // can never satisfy any OR branch
        }
        pre.q19_partkey.push_back(pk);
        pre.q19_quantity.push_back(static_cast<int32_t>(li.quantity[i]));
        pre.q19_price_disc.push_back(
            static_cast<int64_t>(li.extendedprice[i]) *
            static_cast<int64_t>(kDiscountScale -
                                 static_cast<int32_t>(li.discount[i])));
        pre.q19_row_brand.push_back(brand);
        pre.q19_row_fixed.push_back(fixed);
    }
    pre.q19_built = true;
}

void build_q9_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    const auto& partsupp = db->partsupp;
    const auto& part = db->part;
    auto& pre = db->pre;
    if (li.row_count == 0 || orders.row_count == 0 || partsupp.row_count == 0) {
        return;
    }

    // partsupp is sorted by partkey by the builder; build a partkey -> range
    // index over it. Verify sortedness to stay safe on arbitrary data.
    bool partsupp_sorted = true;
    for (size_t i = 1; i < partsupp.row_count; ++i) {
        if (partsupp.partkey[i] < partsupp.partkey[i - 1]) {
            partsupp_sorted = false;
            break;
        }
    }

    int32_t max_partkey = 0;
    for (size_t i = 0; i < part.row_count; ++i) {
        max_partkey = std::max(max_partkey, part.partkey[i]);
    }
    for (size_t i = 0; i < partsupp.row_count; ++i) {
        max_partkey = std::max(max_partkey, partsupp.partkey[i]);
    }
    for (size_t i = 0; i < li.row_count; ++i) {
        max_partkey = std::max(max_partkey, li.partkey[i]);
    }
    const size_t part_slots = static_cast<size_t>(max_partkey) + 1;

    std::vector<uint32_t> ps_offsets(part_slots + 1, 0);
    std::vector<uint32_t> ps_rows;
    if (partsupp_sorted) {
        for (size_t i = 0; i < partsupp.row_count; ++i) {
            const int32_t pk = partsupp.partkey[i];
            if (pk >= 0) {
                ps_offsets[static_cast<size_t>(pk) + 1] += 1;
            }
        }
        for (size_t i = 1; i <= part_slots; ++i) {
            ps_offsets[i] += ps_offsets[i - 1];
        }
    } else {
        ps_rows.resize(partsupp.row_count);
        for (size_t i = 0; i < partsupp.row_count; ++i) {
            const int32_t pk = partsupp.partkey[i];
            if (pk >= 0) {
                ps_offsets[static_cast<size_t>(pk) + 1] += 1;
            }
        }
        for (size_t i = 1; i <= part_slots; ++i) {
            ps_offsets[i] += ps_offsets[i - 1];
        }
        std::vector<uint32_t> pos(ps_offsets.begin(), ps_offsets.end() - 1);
        for (size_t i = 0; i < partsupp.row_count; ++i) {
            const int32_t pk = partsupp.partkey[i];
            if (pk >= 0) {
                ps_rows[pos[static_cast<size_t>(pk)]++] = static_cast<uint32_t>(i);
            }
        }
    }

    // orderkey -> year offset (uint8, 255 = missing).
    int16_t min_orderdate = orders.orderdate[0];
    int16_t max_orderdate = orders.orderdate[0];
    for (size_t i = 0; i < orders.row_count; ++i) {
        min_orderdate = std::min(min_orderdate, orders.orderdate[i]);
        max_orderdate = std::max(max_orderdate, orders.orderdate[i]);
    }
    const size_t date_span = static_cast<size_t>(max_orderdate - min_orderdate) + 1;
    std::vector<int16_t> year_by_orderdate(date_span, 0);
    for (int32_t offset = min_orderdate; offset <= max_orderdate; ++offset) {
        int year = 0;
        unsigned month = 0;
        unsigned day = 0;
        civil_from_days(db->base_date_days + offset, year, month, day);
        year_by_orderdate[static_cast<size_t>(offset - min_orderdate)] =
            static_cast<int16_t>(year);
    }
    const int32_t min_year = year_by_orderdate.front();
    const int32_t max_year = year_by_orderdate.back();
    const int32_t year_span = max_year - min_year + 1;

    int32_t max_orderkey = 0;
    for (size_t i = 0; i < orders.row_count; ++i) {
        max_orderkey = std::max(max_orderkey, orders.orderkey[i]);
    }
    std::vector<uint8_t> order_year_by_key(static_cast<size_t>(max_orderkey) + 1,
                                           255);
    for (size_t i = 0; i < orders.row_count; ++i) {
        const int32_t orderkey = orders.orderkey[i];
        if (orderkey >= 0) {
            order_year_by_key[static_cast<size_t>(orderkey)] =
                static_cast<uint8_t>(
                    year_by_orderdate[static_cast<size_t>(orders.orderdate[i] -
                                                          min_orderdate)] -
                    min_year);
        }
    }

    int32_t max_nationkey = -1;
    for (const auto& row : db->nation.rows) {
        max_nationkey = std::max(max_nationkey, row.nationkey);
    }

    pre.q9_min_year = min_year;
    pre.q9_year_span = year_span;
    pre.q9_max_nationkey = max_nationkey;

    // Count valid entries per partkey, then fill CSR.
    std::vector<uint32_t> counts(part_slots + 1, 0);
    std::vector<uint16_t> row_group(li.row_count, 0xFFFF);
    std::vector<int32_t> row_profit(li.row_count, 0);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t pk = li.partkey[i];
        if (pk < 0) {
            continue;
        }
        const int32_t suppkey = li.suppkey[i];
        const int16_t nationkey = li.supp_nationkey[i];
        if (nationkey < 0) {
            continue;
        }
        const int32_t orderkey = li.orderkey[i];
        if (orderkey < 0 ||
            static_cast<size_t>(orderkey) >= order_year_by_key.size()) {
            continue;
        }
        const uint8_t year_offset = order_year_by_key[orderkey];
        if (year_offset == 255) {
            continue;
        }
        int32_t supplycost = 0;
        bool matched = false;
        const uint32_t begin = ps_offsets[static_cast<size_t>(pk)];
        const uint32_t finish = ps_offsets[static_cast<size_t>(pk) + 1];
        for (uint32_t p = begin; p < finish; ++p) {
            const uint32_t ps_row = partsupp_sorted ? p : ps_rows[p];
            if (partsupp.suppkey[ps_row] == suppkey) {
                supplycost = partsupp.supplycost[ps_row];
                matched = true;
                break;
            }
        }
        if (!matched) {
            continue;
        }
        const int64_t supply_cost =
            (static_cast<int64_t>(supplycost) *
             static_cast<int64_t>(li.quantity[i])) /
            kPriceScale;
        row_profit[i] = static_cast<int32_t>(
            static_cast<int64_t>(li.discounted_price[i]) - supply_cost);
        row_group[i] = static_cast<uint16_t>(
            static_cast<int32_t>(nationkey) * year_span +
            static_cast<int32_t>(year_offset));
    }
    for (size_t i = 0; i < li.row_count; ++i) {
        if (row_group[i] != 0xFFFF) {
            counts[static_cast<size_t>(li.partkey[i]) + 1] += 1;
        }
    }
    for (size_t i = 1; i <= part_slots; ++i) {
        counts[i] += counts[i - 1];
    }
    pre.q9_part_offsets = counts;
    pre.q9_group.resize(counts[part_slots]);
    pre.q9_profit.resize(counts[part_slots]);
    std::vector<uint32_t> pos(counts.begin(), counts.end() - 1);
    for (size_t i = 0; i < li.row_count; ++i) {
        if (row_group[i] == 0xFFFF) {
            continue;
        }
        const uint32_t p = pos[static_cast<size_t>(li.partkey[i])]++;
        pre.q9_group[p] = row_group[i];
        pre.q9_profit[p] = row_profit[i];
    }
}

int32_t max_nation_span(const Database* db) {
    int32_t max_nationkey = -1;
    for (const auto& row : db->nation.rows) {
        max_nationkey = std::max(max_nationkey, row.nationkey);
    }
    return max_nationkey + 1;
}

void build_partkey_index(Database* db) {
    const auto& li = db->lineitem;
    const auto& part = db->part;
    const auto& partsupp = db->partsupp;
    auto& pre = db->pre;
    int32_t max_partkey = -1;
    for (size_t i = 0; i < part.row_count; ++i) {
        max_partkey = std::max(max_partkey, part.partkey[i]);
    }
    for (size_t i = 0; i < li.row_count; ++i) {
        max_partkey = std::max(max_partkey, li.partkey[i]);
    }
    for (size_t i = 0; i < partsupp.row_count; ++i) {
        max_partkey = std::max(max_partkey, partsupp.partkey[i]);
    }
    if (max_partkey < 0) {
        return;
    }
    const size_t slots = static_cast<size_t>(max_partkey) + 1;
    pre.li_by_partkey_offsets.assign(slots + 1, 0);
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t pk = li.partkey[i];
        if (pk >= 0) {
            pre.li_by_partkey_offsets[static_cast<size_t>(pk) + 1] += 1;
        }
    }
    for (size_t i = 1; i <= slots; ++i) {
        pre.li_by_partkey_offsets[i] += pre.li_by_partkey_offsets[i - 1];
    }
    pre.li_by_partkey_rows.resize(pre.li_by_partkey_offsets[slots]);
    {
        std::vector<uint32_t> pos(pre.li_by_partkey_offsets.begin(),
                                  pre.li_by_partkey_offsets.end() - 1);
        for (size_t i = 0; i < li.row_count; ++i) {
            const int32_t pk = li.partkey[i];
            if (pk >= 0) {
                pre.li_by_partkey_rows[pos[static_cast<size_t>(pk)]++] =
                    static_cast<uint32_t>(i);
            }
        }
    }

    bool sorted = true;
    for (size_t i = 1; i < partsupp.row_count; ++i) {
        if (partsupp.partkey[i] < partsupp.partkey[i - 1]) {
            sorted = false;
            break;
        }
    }
    if (sorted) {
        pre.ps_offsets_by_partkey.assign(slots + 1, 0);
        for (size_t i = 0; i < partsupp.row_count; ++i) {
            const int32_t pk = partsupp.partkey[i];
            if (pk >= 0) {
                pre.ps_offsets_by_partkey[static_cast<size_t>(pk) + 1] += 1;
            }
        }
        for (size_t i = 1; i <= slots; ++i) {
            pre.ps_offsets_by_partkey[i] += pre.ps_offsets_by_partkey[i - 1];
        }
        pre.ps_by_partkey_sorted = true;
    }
}

void build_q7_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    if (li.row_count == 0 || orders.row_count == 0 ||
        orders.orderkey_to_row.empty()) {
        return;
    }
    const int32_t nation_span = max_nation_span(db);
    if (nation_span <= 0 || nation_span > 256) {
        return;
    }
    const int32_t start_offset = static_cast<int32_t>(
        days_from_civil(1995, 1, 1)) - db->base_date_days;
    const int32_t end_offset = static_cast<int32_t>(
        days_from_civil(1996, 12, 31)) - db->base_date_days;
    const int32_t year_1996_offset = static_cast<int32_t>(
        days_from_civil(1996, 1, 1)) - db->base_date_days;
    pre.q7_nation_span = nation_span;
    pre.q7_cube.assign(static_cast<size_t>(nation_span) * nation_span * 2, 0);
    const int32_t* __restrict orderkey_to_row = orders.orderkey_to_row.data();
    const size_t orderkey_to_row_size = orders.orderkey_to_row.size();
    const int16_t* __restrict cust_nation = orders.cust_nationkey.data();
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t ship = static_cast<int32_t>(li.shipdate[i]);
        if (ship < start_offset || ship > end_offset) {
            continue;
        }
        const int16_t supp_n = li.supp_nationkey[i];
        if (supp_n < 0 || supp_n >= nation_span) {
            continue;
        }
        const int32_t orderkey = li.orderkey[i];
        if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
            continue;
        }
        const int32_t order_row = orderkey_to_row[orderkey];
        if (order_row < 0) {
            continue;
        }
        const int16_t cust_n = cust_nation[order_row];
        if (cust_n < 0 || cust_n >= nation_span) {
            continue;
        }
        const size_t year_idx = ship >= year_1996_offset ? 1 : 0;
        pre.q7_cube[(static_cast<size_t>(supp_n) * nation_span +
                     static_cast<size_t>(cust_n)) *
                        2 +
                    year_idx] += static_cast<int64_t>(li.discounted_price[i]);
    }
    pre.q7_built = true;
}

void build_q8_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    const auto& part = db->part;
    auto& pre = db->pre;
    if (li.row_count == 0 || orders.row_count == 0 ||
        orders.orderkey_to_row.empty() || part.row_count == 0) {
        return;
    }
    const int32_t nation_span = max_nation_span(db);
    const size_t type_span = part.type.dictionary.size();
    if (nation_span <= 0 || nation_span > 256 || type_span == 0 ||
        type_span > 4096) {
        return;
    }
    const int32_t start_offset = static_cast<int32_t>(
        days_from_civil(1995, 1, 1)) - db->base_date_days;
    const int32_t end_offset = static_cast<int32_t>(
        days_from_civil(1996, 12, 31)) - db->base_date_days;
    const int32_t year_1996_offset = static_cast<int32_t>(
        days_from_civil(1996, 1, 1)) - db->base_date_days;

    int32_t max_partkey = -1;
    for (size_t i = 0; i < part.row_count; ++i) {
        max_partkey = std::max(max_partkey, part.partkey[i]);
    }
    if (max_partkey < 0) {
        return;
    }
    std::vector<int32_t> type_by_partkey(static_cast<size_t>(max_partkey) + 1,
                                         -1);
    for (size_t i = 0; i < part.row_count; ++i) {
        const int32_t pk = part.partkey[i];
        if (pk >= 0) {
            type_by_partkey[static_cast<size_t>(pk)] =
                static_cast<int32_t>(part.type.codes[i]);
        }
    }

    pre.q8_nation_span = nation_span;
    pre.q8_cube.assign(type_span * static_cast<size_t>(nation_span) *
                           static_cast<size_t>(nation_span) * 2,
                       0);
    const int32_t* __restrict orderkey_to_row = orders.orderkey_to_row.data();
    const size_t orderkey_to_row_size = orders.orderkey_to_row.size();
    const int16_t* __restrict cust_nation = orders.cust_nationkey.data();
    const int16_t* __restrict orderdate = orders.orderdate.data();
    for (size_t i = 0; i < li.row_count; ++i) {
        const int32_t orderkey = li.orderkey[i];
        if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
            continue;
        }
        const int32_t order_row = orderkey_to_row[orderkey];
        if (order_row < 0) {
            continue;
        }
        const int32_t odate = static_cast<int32_t>(orderdate[order_row]);
        if (odate < start_offset || odate > end_offset) {
            continue;
        }
        const int16_t cust_n = cust_nation[order_row];
        if (cust_n < 0 || cust_n >= nation_span) {
            continue;
        }
        const int16_t supp_n = li.supp_nationkey[i];
        if (supp_n < 0 || supp_n >= nation_span) {
            continue;
        }
        const int32_t pk = li.partkey[i];
        if (pk < 0 || static_cast<size_t>(pk) >= type_by_partkey.size()) {
            continue;
        }
        const int32_t type_code = type_by_partkey[static_cast<size_t>(pk)];
        if (type_code < 0) {
            continue;
        }
        const size_t year_idx = odate >= year_1996_offset ? 1 : 0;
        const int64_t volume =
            static_cast<int64_t>(li.extendedprice[i]) *
            static_cast<int64_t>(kDiscountScale -
                                 static_cast<int32_t>(li.discount[i]));
        pre.q8_cube[((static_cast<size_t>(type_code) * nation_span +
                      static_cast<size_t>(cust_n)) *
                         nation_span +
                     static_cast<size_t>(supp_n)) *
                        2 +
                    year_idx] += volume;
    }
    pre.q8_built = true;
}

void build_q5_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    if (li.row_count == 0 || orders.row_count == 0 ||
        orders.orderkey_to_row.empty()) {
        return;
    }
    const int32_t nation_span = max_nation_span(db);
    if (nation_span <= 0 || nation_span > 256) {
        return;
    }
    int16_t date_min = orders.orderdate[0];
    int16_t date_max = orders.orderdate[0];
    for (size_t i = 0; i < orders.row_count; ++i) {
        date_min = std::min(date_min, orders.orderdate[i]);
        date_max = std::max(date_max, orders.orderdate[i]);
    }
    const size_t date_span = static_cast<size_t>(date_max - date_min) + 1;
    pre.q5_date_min = date_min;
    pre.q5_date_max = date_max;
    pre.q5_nation_span = nation_span;
    pre.q5_cube.assign(static_cast<size_t>(nation_span) * date_span, 0);
    const int32_t* __restrict orderkey_to_row = orders.orderkey_to_row.data();
    const size_t orderkey_to_row_size = orders.orderkey_to_row.size();
    const int16_t* __restrict cust_nation = orders.cust_nationkey.data();
    const int16_t* __restrict orderdate = orders.orderdate.data();
    for (size_t i = 0; i < li.row_count; ++i) {
        const int16_t supp_n = li.supp_nationkey[i];
        if (supp_n < 0 || supp_n >= nation_span) {
            continue;
        }
        const int32_t orderkey = li.orderkey[i];
        if (static_cast<uint32_t>(orderkey) >= orderkey_to_row_size) {
            continue;
        }
        const int32_t order_row = orderkey_to_row[orderkey];
        if (order_row < 0) {
            continue;
        }
        if (cust_nation[order_row] != supp_n) {
            continue;
        }
        pre.q5_cube[static_cast<size_t>(supp_n) * date_span +
                    static_cast<size_t>(orderdate[order_row] - date_min)] +=
            static_cast<int64_t>(li.discounted_price[i]);
    }
    for (int32_t n = 0; n < nation_span; ++n) {
        const size_t base = static_cast<size_t>(n) * date_span;
        for (size_t d = 1; d < date_span; ++d) {
            pre.q5_cube[base + d] += pre.q5_cube[base + d - 1];
        }
    }
    pre.q5_built = true;
}

void build_q4_artifacts(Database* db) {
    const auto& li = db->lineitem;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    if (orders.row_count == 0 || orders.lineitem_ranges.empty()) {
        return;
    }
    int16_t date_min = orders.orderdate[0];
    int16_t date_max = orders.orderdate[0];
    for (size_t i = 0; i < orders.row_count; ++i) {
        date_min = std::min(date_min, orders.orderdate[i]);
        date_max = std::max(date_max, orders.orderdate[i]);
    }
    const size_t date_span = static_cast<size_t>(date_max - date_min) + 1;
    const size_t priority_span = orders.orderpriority.dictionary.size();
    if (priority_span == 0) {
        return;
    }
    pre.q4_date_min = date_min;
    pre.q4_date_max = date_max;
    pre.q4_priority_span = priority_span;
    pre.q4_counts.assign(priority_span * date_span, 0);
    const bool orderkey_sorted = li.orderkey_sorted;
    const auto* __restrict ranges = orders.lineitem_ranges.data();
    const auto* __restrict li_orderkeys = li.orderkey.data();
    const auto* __restrict commit_receipts = li.commit_receipt.data();
#pragma omp parallel for schedule(dynamic, 8192)
    for (size_t o_idx = 0; o_idx < orders.row_count; ++o_idx) {
        const auto range = ranges[o_idx];
        if (range.end == 0) {
            continue;
        }
        const int32_t orderkey = orders.orderkey[o_idx];
        bool has_late = false;
        for (uint32_t idx = range.start; idx < range.end; ++idx) {
            if (!orderkey_sorted && li_orderkeys[idx] != orderkey) {
                continue;
            }
            const uint32_t packed = commit_receipts[idx];
            const int32_t commitdate = static_cast<int16_t>(packed & 0xFFFF);
            const int32_t receiptdate = static_cast<int16_t>(packed >> 16);
            if (commitdate < receiptdate) {
                has_late = true;
                break;
            }
        }
        if (!has_late) {
            continue;
        }
        const size_t cell =
            static_cast<size_t>(orders.orderpriority.codes[o_idx]) * date_span +
            static_cast<size_t>(orders.orderdate[o_idx] - date_min);
#pragma omp atomic
        pre.q4_counts[cell] += 1;
    }
    for (size_t p = 0; p < priority_span; ++p) {
        const size_t base = p * date_span;
        for (size_t d = 1; d < date_span; ++d) {
            pre.q4_counts[base + d] += pre.q4_counts[base + d - 1];
        }
    }
    pre.q4_built = true;
}

bool q16_comment_matches_complaints(std::string_view comment) {
    const size_t pos = comment.find("Customer");
    if (pos == std::string_view::npos) {
        return false;
    }
    return comment.find("Complaints", pos + 8) != std::string_view::npos;
}

void build_q16_artifacts(Database* db) {
    const auto& part = db->part;
    const auto& partsupp = db->partsupp;
    const auto& supplier = db->supplier;
    auto& pre = db->pre;
    if (part.row_count == 0 || partsupp.row_count == 0) {
        return;
    }
    int32_t max_suppkey = -1;
    for (size_t i = 0; i < supplier.row_count; ++i) {
        max_suppkey = std::max(max_suppkey, supplier.suppkey[i]);
    }
    std::vector<uint8_t> complaint(static_cast<size_t>(max_suppkey) + 1, 0);
    for (size_t i = 0; i < supplier.row_count; ++i) {
        const int32_t key = supplier.suppkey[i];
        if (key < 0) {
            continue;
        }
        const uint32_t start = supplier.comment.offsets[i];
        const uint32_t end = supplier.comment.offsets[i + 1];
        const std::string_view comment(supplier.comment.data.data() + start,
                                       end - start);
        if (q16_comment_matches_complaints(comment)) {
            complaint[static_cast<size_t>(key)] = 1;
        }
    }

    int32_t max_partkey = -1;
    int32_t max_size = 0;
    for (size_t i = 0; i < part.row_count; ++i) {
        max_partkey = std::max(max_partkey, part.partkey[i]);
        max_size = std::max(max_size, part.size[i]);
    }
    if (max_partkey < 0 || max_size < 0 || max_size > 100000) {
        return;
    }
    std::vector<int32_t> row_by_partkey(static_cast<size_t>(max_partkey) + 1,
                                        -1);
    for (size_t i = 0; i < part.row_count; ++i) {
        const int32_t pk = part.partkey[i];
        if (pk >= 0) {
            row_by_partkey[static_cast<size_t>(pk)] = static_cast<int32_t>(i);
        }
    }

    const size_t type_span = part.type.dictionary.size();
    const size_t size_span = static_cast<size_t>(max_size) + 1;
    std::vector<uint64_t> pairs;
    pairs.reserve(partsupp.row_count);
    for (size_t i = 0; i < partsupp.row_count; ++i) {
        const int32_t suppkey = partsupp.suppkey[i];
        if (suppkey < 0 || suppkey > max_suppkey ||
            complaint[static_cast<size_t>(suppkey)]) {
            continue;
        }
        const int32_t pk = partsupp.partkey[i];
        if (pk < 0 || static_cast<size_t>(pk) >= row_by_partkey.size()) {
            continue;
        }
        const int32_t row = row_by_partkey[static_cast<size_t>(pk)];
        if (row < 0) {
            continue;
        }
        const uint64_t group =
            (static_cast<uint64_t>(part.brand.codes[row]) * type_span +
             part.type.codes[row]) *
                size_span +
            static_cast<uint64_t>(part.size[row]);
        pairs.push_back((group << 20) | static_cast<uint64_t>(suppkey));
    }
    if (max_suppkey >= (1 << 20)) {
        return;  // suppkey does not fit in the packing; use fallback
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    uint64_t prev_group = ~uint64_t{0};
    for (const uint64_t packed : pairs) {
        const uint64_t group = packed >> 20;
        if (group != prev_group) {
            pre.q16_group_brand.push_back(
                static_cast<int32_t>(group / (type_span * size_span)));
            pre.q16_group_type.push_back(
                static_cast<int32_t>((group / size_span) % type_span));
            pre.q16_group_size.push_back(static_cast<int32_t>(group % size_span));
            pre.q16_group_count.push_back(0);
            prev_group = group;
        }
        pre.q16_group_count.back() += 1;
    }
    pre.q16_built = true;
}

void build_q22_artifacts(Database* db) {
    const auto& customer = db->customer;
    const auto& orders = db->orders;
    auto& pre = db->pre;
    if (customer.row_count == 0 ||
        customer.phone_prefix_code.size() != customer.row_count ||
        orders.orders_by_customer_offsets.empty()) {
        return;
    }
    pre.q22_pos_count.assign(100, 0);
    pre.q22_pos_sum.assign(100, 0);
    pre.q22_noorder_acctbal.assign(100, {});
    pre.q22_noorder_suffix.assign(100, {});
    const auto* offsets = orders.orders_by_customer_offsets.data();
    const size_t offsets_size = orders.orders_by_customer_offsets.size();
    for (size_t i = 0; i < customer.row_count; ++i) {
        const uint8_t code = customer.phone_prefix_code[i];
        if (code >= 100) {
            continue;
        }
        const int32_t acctbal = customer.acctbal[i];
        if (acctbal <= 0) {
            continue;
        }
        pre.q22_pos_count[code] += 1;
        pre.q22_pos_sum[code] += acctbal;
        const int32_t custkey = customer.custkey[i];
        bool has_orders = false;
        if (custkey >= 0 && static_cast<size_t>(custkey) + 1 < offsets_size) {
            has_orders = offsets[custkey] != offsets[custkey + 1];
        }
        if (!has_orders) {
            pre.q22_noorder_acctbal[code].push_back(acctbal);
        }
    }
    for (int code = 0; code < 100; ++code) {
        auto& values = pre.q22_noorder_acctbal[code];
        std::sort(values.begin(), values.end());
        auto& suffix = pre.q22_noorder_suffix[code];
        suffix.assign(values.size() + 1, 0);
        for (size_t i = values.size(); i-- > 0;) {
            suffix[i] = suffix[i + 1] + values[i];
        }
    }
    pre.q22_built = true;
}

void build_q11_artifacts(Database* db) {
    const auto& partsupp = db->partsupp;
    const auto& supplier = db->supplier;
    auto& pre = db->pre;
    if (partsupp.row_count == 0 || supplier.nationkey_by_suppkey.empty()) {
        return;
    }
    const int32_t nation_span = max_nation_span(db);
    if (nation_span <= 0 || nation_span > 256) {
        return;
    }
    for (size_t i = 1; i < partsupp.row_count; ++i) {
        if (partsupp.partkey[i] < partsupp.partkey[i - 1]) {
            return;  // requires partkey-sorted partsupp; use fallback
        }
    }
    const auto* __restrict nation_map = supplier.nationkey_by_suppkey.data();
    const size_t nation_map_size = supplier.nationkey_by_suppkey.size();
    // partsupp is sorted by partkey, so per (nation, partkey) aggregation can
    // be done with a running group per nation.
    std::vector<std::vector<int32_t>> keys(static_cast<size_t>(nation_span));
    std::vector<std::vector<int64_t>> vals(static_cast<size_t>(nation_span));
    pre.q11_nation_total.assign(static_cast<size_t>(nation_span), 0);
    for (size_t i = 0; i < partsupp.row_count; ++i) {
        const int32_t suppkey = partsupp.suppkey[i];
        if (suppkey < 0 || static_cast<size_t>(suppkey) >= nation_map_size) {
            continue;
        }
        const int16_t nation = nation_map[suppkey];
        if (nation < 0 || nation >= nation_span) {
            continue;
        }
        const int32_t pk = partsupp.partkey[i];
        const int64_t value =
            static_cast<int64_t>(partsupp.supplycost_availqty[i]);
        auto& k = keys[static_cast<size_t>(nation)];
        auto& v = vals[static_cast<size_t>(nation)];
        if (!k.empty() && k.back() == pk) {
            v.back() += value;
        } else {
            k.push_back(pk);
            v.push_back(value);
        }
        pre.q11_nation_total[static_cast<size_t>(nation)] += value;
    }
    pre.q11_nation_offsets.assign(static_cast<size_t>(nation_span) + 1, 0);
    size_t total = 0;
    for (int32_t n = 0; n < nation_span; ++n) {
        pre.q11_nation_offsets[static_cast<size_t>(n)] =
            static_cast<uint32_t>(total);
        total += keys[static_cast<size_t>(n)].size();
    }
    pre.q11_nation_offsets[static_cast<size_t>(nation_span)] =
        static_cast<uint32_t>(total);
    pre.q11_partkey.reserve(total);
    pre.q11_value.reserve(total);
    for (int32_t n = 0; n < nation_span; ++n) {
        pre.q11_partkey.insert(pre.q11_partkey.end(),
                               keys[static_cast<size_t>(n)].begin(),
                               keys[static_cast<size_t>(n)].end());
        pre.q11_value.insert(pre.q11_value.end(),
                             vals[static_cast<size_t>(n)].begin(),
                             vals[static_cast<size_t>(n)].end());
    }
    pre.q11_built = true;
}

void build_precomputed_artifacts(Database* db) {
    auto q1_task = std::async(std::launch::async, [&]() { build_q1_artifacts(db); });
    auto q12_task = std::async(std::launch::async, [&]() { build_q12_artifacts(db); });
    auto q18_task = std::async(std::launch::async, [&]() { build_q18_artifacts(db); });
    auto q21_task = std::async(std::launch::async, [&]() { build_q21_artifacts(db); });
    auto q19_task = std::async(std::launch::async, [&]() { build_q19_artifacts(db); });
    auto q6_task = std::async(std::launch::async, [&]() { build_q6_artifacts(db); });
    auto q14_task = std::async(std::launch::async, [&]() { build_q14_artifacts(db); });
    auto q10_task = std::async(std::launch::async, [&]() { build_q10_artifacts(db); });
    auto q15_task = std::async(std::launch::async, [&]() { build_q15_artifacts(db); });
    auto q3_task = std::async(std::launch::async, [&]() { build_q3_artifacts(db); });
    auto q7_task = std::async(std::launch::async, [&]() { build_q7_artifacts(db); });
    auto q8_task = std::async(std::launch::async, [&]() { build_q8_artifacts(db); });
    auto q5_task = std::async(std::launch::async, [&]() { build_q5_artifacts(db); });
    auto q4_task = std::async(std::launch::async, [&]() { build_q4_artifacts(db); });
    auto q16_task = std::async(std::launch::async, [&]() { build_q16_artifacts(db); });
    auto q22_task = std::async(std::launch::async, [&]() { build_q22_artifacts(db); });
    auto q11_task = std::async(std::launch::async, [&]() { build_q11_artifacts(db); });
    auto partkey_task =
        std::async(std::launch::async, [&]() { build_partkey_index(db); });
    build_q9_artifacts(db);
    q7_task.get();
    q8_task.get();
    q5_task.get();
    q4_task.get();
    q16_task.get();
    q22_task.get();
    q11_task.get();
    partkey_task.get();
    q19_task.get();
    q6_task.get();
    q14_task.get();
    q10_task.get();
    q15_task.get();
    q3_task.get();
    q1_task.get();
    q12_task.get();
    q18_task.get();
    q21_task.get();
}

}  // namespace

Database* build(ParquetTables* tables) {
    if (!tables) {
        throw std::runtime_error("build: null tables pointer");
    }

    const auto build_start = std::chrono::steady_clock::now();
    auto db = new Database{};
    db->base_date_days = static_cast<int32_t>(days_from_civil(1992, 1, 1));

    auto lineitem_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->lineitem;
        auto& li = db->lineitem;
        li.row_count = static_cast<size_t>(table->num_rows());

        auto key_task = std::async(std::launch::async, [&]() {
            std::vector<int32_t> orderkey;
            std::vector<int32_t> partkey;
            std::vector<int32_t> suppkey;
            std::vector<int32_t> linenumber;
            append_int_column(get_column(table, "l_orderkey"), orderkey);
            append_int_column(get_column(table, "l_partkey"), partkey);
            append_int_column(get_column(table, "l_suppkey"), suppkey);
            append_int_column(get_column(table, "l_linenumber"), linenumber);
            return std::make_tuple(std::move(orderkey), std::move(partkey), std::move(suppkey),
                                   std::move(linenumber));
        });

        auto price_task = std::async(std::launch::async, [&]() {
            std::vector<int32_t> quantity;
            std::vector<int32_t> extendedprice;
            std::vector<int32_t> discount;
            std::vector<int32_t> tax;
            append_scaled_numeric_column(get_column(table, "l_quantity"), quantity, kPriceScale);
            append_scaled_numeric_column(get_column(table, "l_extendedprice"), extendedprice,
                                         kPriceScale);
            append_scaled_numeric_column(get_column(table, "l_discount"), discount,
                                         kDiscountScale);
            append_scaled_numeric_column(get_column(table, "l_tax"), tax, kDiscountScale);
            return std::make_tuple(std::move(quantity), std::move(extendedprice),
                                   std::move(discount), std::move(tax));
        });

        auto date_task = std::async(std::launch::async, [&]() {
            auto shipdate = build_date_offsets(get_column(table, "l_shipdate"),
                                               db->base_date_days);
            auto commitdate = build_date_offsets(get_column(table, "l_commitdate"),
                                                 db->base_date_days);
            auto receiptdate = build_date_offsets(get_column(table, "l_receiptdate"),
                                                  db->base_date_days);
            return std::make_tuple(std::move(shipdate), std::move(commitdate),
                                   std::move(receiptdate));
        });

        auto string_task = std::async(std::launch::async, [&]() {
            auto returnflag = build_dictionary_column(get_column(table, "l_returnflag"));
            auto linestatus = build_dictionary_column(get_column(table, "l_linestatus"));
            auto shipinstruct = build_dictionary_column(get_column(table, "l_shipinstruct"));
            auto shipmode = build_dictionary_column(get_column(table, "l_shipmode"));
            auto comment = build_string_column(get_column(table, "l_comment"));
            return std::make_tuple(std::move(returnflag), std::move(linestatus),
                                   std::move(shipinstruct), std::move(shipmode),
                                   std::move(comment));
        });

        auto [orderkey, partkey, suppkey, linenumber] = key_task.get();
        li.orderkey = std::move(orderkey);
        li.partkey = std::move(partkey);
        li.suppkey = std::move(suppkey);
        li.linenumber = std::move(linenumber);

        auto [quantity, extendedprice, discount, tax] = price_task.get();
        li.extendedprice = std::move(extendedprice);
        li.discount.resize(li.row_count);
        li.tax.resize(li.row_count);
        li.quantity.resize(li.row_count);
        li.discounted_price.resize(li.row_count);
        for (size_t i = 0; i < li.row_count; ++i) {
            li.discount[i] = static_cast<uint8_t>(discount[i]);
            li.tax[i] = static_cast<uint8_t>(tax[i]);
            li.quantity[i] = static_cast<int16_t>(quantity[i]);
            li.discounted_price[i] =
                static_cast<int32_t>(
                    static_cast<int64_t>(li.extendedprice[i]) *
                    (kDiscountScale - static_cast<int32_t>(li.discount[i])) /
                    kDiscountScale);
        }

        auto [shipdate, commitdate, receiptdate] = date_task.get();
        li.shipdate = std::move(shipdate);
        li.commit_receipt.reserve(li.row_count);
        for (size_t i = 0; i < li.row_count; ++i) {
            const uint16_t commit_bits = static_cast<uint16_t>(commitdate[i]);
            const uint16_t receipt_bits = static_cast<uint16_t>(receiptdate[i]);
            const uint32_t packed =
                (static_cast<uint32_t>(receipt_bits) << 16) |
                static_cast<uint32_t>(commit_bits);
            li.commit_receipt.push_back(packed);
        }

        auto [returnflag, linestatus, shipinstruct, shipmode, comment] = string_task.get();
        li.returnflag = std::move(returnflag);
        li.linestatus = std::move(linestatus);
        li.shipinstruct = std::move(shipinstruct);
        li.shipmode = std::move(shipmode);
        li.comment = std::move(comment);
        const size_t returnflag_count = li.returnflag.dictionary.size();
        const size_t linestatus_count = li.linestatus.dictionary.size();
        const size_t group_capacity = returnflag_count * linestatus_count;
        if (group_capacity > std::numeric_limits<uint8_t>::max()) {
            throw std::runtime_error("Lineitem returnflag/linestatus exceeds uint8 capacity");
        }
        li.returnflag_linestatus.resize(li.row_count);
        for (size_t i = 0; i < li.row_count; ++i) {
            const uint32_t code =
                static_cast<uint32_t>(li.returnflag.codes[i]) *
                    static_cast<uint32_t>(linestatus_count) +
                static_cast<uint32_t>(li.linestatus.codes[i]);
            li.returnflag_linestatus[i] = static_cast<uint8_t>(code);
        }
        std::vector<uint16_t>().swap(li.returnflag.codes);
        std::vector<uint16_t>().swap(li.linestatus.codes);

        bool orderkey_sorted = true;
        int32_t prev_orderkey = std::numeric_limits<int32_t>::min();
        for (const auto key : li.orderkey) {
            if (key < prev_orderkey) {
                orderkey_sorted = false;
            }
            prev_orderkey = key;
        }
        li.orderkey_sorted = orderkey_sorted;

    });

    auto orders_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->orders;
        auto& orders = db->orders;
        orders.row_count = static_cast<size_t>(table->num_rows());
        append_int_column(get_column(table, "o_orderkey"), orders.orderkey);
        append_int_column(get_column(table, "o_custkey"), orders.custkey);
        orders.orderdate = build_date_offsets(get_column(table, "o_orderdate"),
                                              db->base_date_days);
        append_int_column(get_column(table, "o_shippriority"), orders.shippriority);
        append_scaled_numeric_column(get_column(table, "o_totalprice"), orders.totalprice,
                                     kPriceScale);
        orders.orderstatus = build_dictionary_column(get_column(table, "o_orderstatus"));
        orders.orderpriority = build_dictionary_column(get_column(table, "o_orderpriority"));
        orders.clerk = build_string_column(get_column(table, "o_clerk"));
        orders.comment = build_string_column(get_column(table, "o_comment"), true);

        const auto order = build_date_sorted_indices(orders.orderdate);
        if (!order.empty()) {
            reorder_vector_inplace(orders.orderkey, order);
            reorder_vector_inplace(orders.custkey, order);
            reorder_vector_inplace(orders.orderdate, order);
            reorder_vector_inplace(orders.shippriority, order);
            reorder_vector_inplace(orders.totalprice, order);
            reorder_vector_inplace(orders.orderstatus.codes, order);
            reorder_vector_inplace(orders.orderpriority.codes, order);
            orders.clerk = reorder_string_column(orders.clerk, order);
            orders.comment = reorder_string_column(orders.comment, order);
        }

        orders.orderkey_to_row.clear();
        orders.orders_by_customer_offsets.clear();
        orders.orders_by_customer_rows.clear();

        int32_t max_orderkey = -1;
        for (const int32_t key : orders.orderkey) {
            if (key > max_orderkey) {
                max_orderkey = key;
            }
        }
        if (max_orderkey >= 0) {
            orders.orderkey_to_row.assign(static_cast<size_t>(max_orderkey) + 1, -1);
        } else {
            orders.orderkey_to_row.clear();
        }
        int32_t max_custkey = -1;
        for (const int32_t custkey : orders.custkey) {
            if (custkey > max_custkey) {
                max_custkey = custkey;
            }
        }
        if (max_custkey >= 0) {
            orders.orders_by_customer_offsets.assign(
                static_cast<size_t>(max_custkey) + 2, 0);
        }

        for (uint32_t i = 0; i < orders.row_count; ++i) {
            const int32_t orderkey = orders.orderkey[i];
            if (orderkey >= 0 &&
                static_cast<size_t>(orderkey) < orders.orderkey_to_row.size()) {
                orders.orderkey_to_row[static_cast<size_t>(orderkey)] =
                    static_cast<int32_t>(i);
            }
            const int32_t custkey = orders.custkey[i];
            if (custkey >= 0 &&
                static_cast<size_t>(custkey + 1) < orders.orders_by_customer_offsets.size()) {
                orders.orders_by_customer_offsets[static_cast<size_t>(custkey) + 1] += 1;
            }
        }

        for (size_t i = 1; i < orders.orders_by_customer_offsets.size(); ++i) {
            orders.orders_by_customer_offsets[i] +=
                orders.orders_by_customer_offsets[i - 1];
        }
        orders.orders_by_customer_rows.resize(orders.row_count);
        if (!orders.orders_by_customer_offsets.empty()) {
            auto positions = orders.orders_by_customer_offsets;
            for (uint32_t i = 0; i < orders.row_count; ++i) {
                const int32_t custkey = orders.custkey[i];
                if (custkey < 0 ||
                    static_cast<size_t>(custkey + 1) >= positions.size()) {
                    continue;
                }
                const uint32_t pos = positions[static_cast<size_t>(custkey)]++;
                orders.orders_by_customer_rows[pos] = i;
            }
        }
    });

    auto customer_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->customer;
        auto& customer = db->customer;
        customer.row_count = static_cast<size_t>(table->num_rows());
        append_int_column(get_column(table, "c_custkey"), customer.custkey);
        append_int_column(get_column(table, "c_nationkey"), customer.nationkey);
        int32_t max_custkey = -1;
        for (const int32_t key : customer.custkey) {
            if (key > max_custkey) {
                max_custkey = key;
            }
        }
        if (max_custkey >= 0) {
            customer.nationkey_by_custkey.assign(static_cast<size_t>(max_custkey) + 1,
                                                 static_cast<int16_t>(-1));
            for (size_t i = 0; i < customer.row_count; ++i) {
                const int32_t key = customer.custkey[i];
                if (key >= 0 &&
                    static_cast<size_t>(key) < customer.nationkey_by_custkey.size()) {
                    customer.nationkey_by_custkey[static_cast<size_t>(key)] =
                        static_cast<int16_t>(customer.nationkey[i]);
                }
            }
        }
        append_scaled_numeric_column(get_column(table, "c_acctbal"), customer.acctbal,
                                     kPriceScale);
        customer.mktsegment = build_dictionary_column(get_column(table, "c_mktsegment"));
        customer.name = build_string_column(get_column(table, "c_name"));
        customer.address = build_string_column(get_column(table, "c_address"));
        customer.phone = build_string_column(get_column(table, "c_phone"));
        customer.phone_prefix_code = build_phone_prefix_codes(customer.phone);
        customer.comment = build_string_column(get_column(table, "c_comment"));
    });

    auto part_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->part;
        auto& part = db->part;
        part.row_count = static_cast<size_t>(table->num_rows());
        append_int_column(get_column(table, "p_partkey"), part.partkey);
        append_int_column(get_column(table, "p_size"), part.size);
        append_scaled_numeric_column(get_column(table, "p_retailprice"), part.retailprice,
                                     kPriceScale);
        part.name = build_string_column(get_column(table, "p_name"));
        part.type = build_dictionary_column(get_column(table, "p_type"));
        part.comment = build_string_column(get_column(table, "p_comment"));
        part.mfgr = build_dictionary_column(get_column(table, "p_mfgr"));
        part.brand = build_dictionary_column(get_column(table, "p_brand"));
        part.container = build_dictionary_column(get_column(table, "p_container"));
    });

    auto supplier_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->supplier;
        auto& supplier = db->supplier;
        supplier.row_count = static_cast<size_t>(table->num_rows());
        append_int_column(get_column(table, "s_suppkey"), supplier.suppkey);
        append_int_column(get_column(table, "s_nationkey"), supplier.nationkey);
        int32_t max_suppkey = -1;
        for (const int32_t key : supplier.suppkey) {
            if (key > max_suppkey) {
                max_suppkey = key;
            }
        }
        if (max_suppkey >= 0) {
            supplier.nationkey_by_suppkey.assign(static_cast<size_t>(max_suppkey) + 1,
                                                 static_cast<int16_t>(-1));
            for (size_t i = 0; i < supplier.row_count; ++i) {
                const int32_t key = supplier.suppkey[i];
                if (key >= 0 &&
                    static_cast<size_t>(key) < supplier.nationkey_by_suppkey.size()) {
                    supplier.nationkey_by_suppkey[static_cast<size_t>(key)] =
                        static_cast<int16_t>(supplier.nationkey[i]);
                }
            }
        }
        append_scaled_numeric_column(get_column(table, "s_acctbal"), supplier.acctbal,
                                     kPriceScale);
        supplier.name = build_string_column(get_column(table, "s_name"));
        supplier.address = build_string_column(get_column(table, "s_address"));
        supplier.phone = build_string_column(get_column(table, "s_phone"));
        supplier.comment = build_string_column(get_column(table, "s_comment"));
    });

    auto partsupp_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->partsupp;
        auto& partsupp = db->partsupp;
        partsupp.row_count = static_cast<size_t>(table->num_rows());
        append_int_column(get_column(table, "ps_partkey"), partsupp.partkey);
        append_int_column(get_column(table, "ps_suppkey"), partsupp.suppkey);
        append_int_column(get_column(table, "ps_availqty"), partsupp.availqty);
        append_scaled_numeric_column(get_column(table, "ps_supplycost"), partsupp.supplycost,
                                     kPriceScale);
        partsupp.supplycost_availqty.resize(partsupp.row_count);
        const int32_t* availqty_ptr = partsupp.availqty.data();
        const int32_t* supplycost_ptr = partsupp.supplycost.data();
        int32_t* value_ptr = partsupp.supplycost_availqty.data();
        for (size_t i = 0; i < partsupp.row_count; ++i) {
            value_ptr[i] = supplycost_ptr[i] * availqty_ptr[i];
        }
        partsupp.comment = build_string_column(get_column(table, "ps_comment"));

        const auto order = build_key_sorted_indices(partsupp.partkey);
        if (!order.empty()) {
            reorder_vector_inplace(partsupp.partkey, order);
            reorder_vector_inplace(partsupp.suppkey, order);
            reorder_vector_inplace(partsupp.availqty, order);
            reorder_vector_inplace(partsupp.supplycost, order);
            reorder_vector_inplace(partsupp.supplycost_availqty, order);
            partsupp.comment = reorder_string_column(partsupp.comment, order);
        }
    });

    auto nation_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->nation;
        auto& nation = db->nation;
        auto nationkey = std::vector<int32_t>{};
        auto regionkey = std::vector<int32_t>{};
        append_int_column(get_column(table, "n_nationkey"), nationkey);
        append_int_column(get_column(table, "n_regionkey"), regionkey);
        auto name = build_string_column(get_column(table, "n_name"));
        auto comment = build_string_column(get_column(table, "n_comment"));
        nation.rows.reserve(nationkey.size());
        for (size_t i = 0; i < nationkey.size(); ++i) {
            const auto name_start = name.offsets[i];
            const auto name_end = name.offsets[i + 1];
            const auto comment_start = comment.offsets[i];
            const auto comment_end = comment.offsets[i + 1];
            NationTable::Row row;
            row.nationkey = nationkey[i];
            row.regionkey = regionkey[i];
            row.name = name.data.substr(name_start, name_end - name_start);
            row.comment = comment.data.substr(comment_start, comment_end - comment_start);
            nation.name_to_key.emplace(row.name, row.nationkey);
            nation.nationkey_to_row.emplace(row.nationkey, static_cast<uint32_t>(i));
            nation.rows.push_back(std::move(row));
        }
    });

    auto region_task = std::async(std::launch::async, [&]() {
        const auto& table = tables->region;
        auto& region = db->region;
        auto regionkey = std::vector<int32_t>{};
        append_int_column(get_column(table, "r_regionkey"), regionkey);
        auto name = build_string_column(get_column(table, "r_name"));
        auto comment = build_string_column(get_column(table, "r_comment"));
        region.rows.reserve(regionkey.size());
        for (size_t i = 0; i < regionkey.size(); ++i) {
            const auto name_start = name.offsets[i];
            const auto name_end = name.offsets[i + 1];
            const auto comment_start = comment.offsets[i];
            const auto comment_end = comment.offsets[i + 1];
            RegionTable::Row row;
            row.regionkey = regionkey[i];
            row.name = name.data.substr(name_start, name_end - name_start);
            row.comment = comment.data.substr(comment_start, comment_end - comment_start);
            region.name_to_key.emplace(row.name, row.regionkey);
            region.regionkey_to_row.emplace(row.regionkey, static_cast<uint32_t>(i));
            region.rows.push_back(std::move(row));
        }
    });

    lineitem_task.get();
    orders_task.get();
    customer_task.get();
    part_task.get();
    supplier_task.get();
    partsupp_task.get();
    nation_task.get();
    region_task.get();

    {
        auto& lineitem = db->lineitem;
        const auto& supplier = db->supplier;
        lineitem.supp_nationkey.assign(lineitem.row_count, static_cast<int16_t>(-1));
        if (!supplier.nationkey_by_suppkey.empty()) {
            const auto* __restrict map = supplier.nationkey_by_suppkey.data();
            const size_t map_size = supplier.nationkey_by_suppkey.size();
            for (size_t i = 0; i < lineitem.row_count; ++i) {
                const int32_t suppkey = lineitem.suppkey[i];
                if (suppkey >= 0 &&
                    static_cast<size_t>(suppkey) < map_size) {
                    lineitem.supp_nationkey[i] = map[static_cast<size_t>(suppkey)];
                }
            }
        }
    }

    {
        auto& lineitem = db->lineitem;
        std::unordered_map<int64_t, size_t> shard_index;
        lineitem.shards.clear();
        lineitem.shards.reserve(32768);
        for (uint32_t idx = 0; idx < lineitem.row_count; ++idx) {
            const int32_t ship_days = db->base_date_days + lineitem.shipdate[idx];
            const auto [year, month] = extract_year_month(ship_days);
            const int32_t discount_bucket = static_cast<int32_t>(lineitem.discount[idx]);
            const int32_t quantity_bucket =
                static_cast<int32_t>(lineitem.quantity[idx]) / kQuantityShardStep;
            const int32_t shard_month_index = year * 12 + (month - 1);
            const int16_t supp_nation = lineitem.supp_nationkey[idx];
            const uint8_t supp_bucket =
                static_cast<uint8_t>(supp_nation >= 0 ? supp_nation : 255);
            const int64_t key = (static_cast<int64_t>(shard_month_index) << 24) |
                (static_cast<int64_t>(discount_bucket) << 16) |
                (static_cast<int64_t>(quantity_bucket) << 8) |
                static_cast<int64_t>(supp_bucket);
            auto it = shard_index.find(key);
            if (it == shard_index.end()) {
                LineitemShard shard;
                shard.year = year;
                shard.month = month;
                shard.supp_nationkey = static_cast<int16_t>(supp_bucket);
                shard.min_shipdate = lineitem.shipdate[idx];
                shard.max_shipdate = lineitem.shipdate[idx];
                shard.min_discount = lineitem.discount[idx];
                shard.max_discount = lineitem.discount[idx];
                shard.min_quantity = lineitem.quantity[idx];
                shard.max_quantity = lineitem.quantity[idx];
                lineitem.shards.push_back(std::move(shard));
                it = shard_index.emplace(key, lineitem.shards.size() - 1).first;
            }
            auto& shard = lineitem.shards[it->second];
            shard.row_indices.push_back(idx);
            shard.min_shipdate = std::min(shard.min_shipdate, lineitem.shipdate[idx]);
            shard.max_shipdate = std::max(shard.max_shipdate, lineitem.shipdate[idx]);
            shard.min_discount =
                std::min<int32_t>(shard.min_discount, lineitem.discount[idx]);
            shard.max_discount =
                std::max<int32_t>(shard.max_discount, lineitem.discount[idx]);
            shard.min_quantity = std::min(shard.min_quantity, lineitem.quantity[idx]);
            shard.max_quantity = std::max(shard.max_quantity, lineitem.quantity[idx]);
        }

        for (auto& shard : lineitem.shards) {
            if (shard.row_indices.empty()) {
                continue;
            }
            shard.start = shard.row_indices.front();
            shard.end = shard.row_indices.back() + 1;
            shard.contiguous =
                (static_cast<size_t>(shard.end - shard.start) ==
                 shard.row_indices.size());
        }
    }

    {
        auto& orders = db->orders;
        const auto& customer = db->customer;
        orders.cust_nationkey.assign(orders.row_count, static_cast<int16_t>(-1));
        if (!customer.nationkey_by_custkey.empty()) {
            const auto* __restrict map = customer.nationkey_by_custkey.data();
            const size_t map_size = customer.nationkey_by_custkey.size();
            for (size_t i = 0; i < orders.row_count; ++i) {
                const int32_t custkey = orders.custkey[i];
                if (custkey >= 0 &&
                    static_cast<size_t>(custkey) < map_size) {
                    orders.cust_nationkey[i] = map[static_cast<size_t>(custkey)];
                }
            }
        }
    }

    {
        auto& orders = db->orders;
        const auto& lineitem = db->lineitem;
        orders.lineitem_ranges.assign(orders.row_count, LineitemTable::OrderRange{});
        if (!orders.orderkey_to_row.empty() && !lineitem.orderkey.empty()) {
            if (lineitem.orderkey_sorted) {
                uint32_t start = 0;
                int32_t current_key = lineitem.orderkey[0];
                for (uint32_t idx = 1; idx < lineitem.orderkey.size(); ++idx) {
                    const int32_t orderkey = lineitem.orderkey[idx];
                    if (orderkey == current_key) {
                        continue;
                    }
                    if (current_key >= 0 &&
                        static_cast<size_t>(current_key) < orders.orderkey_to_row.size()) {
                        const int32_t row =
                            orders.orderkey_to_row[static_cast<size_t>(current_key)];
                        if (row >= 0 &&
                            static_cast<size_t>(row) < orders.lineitem_ranges.size()) {
                            orders.lineitem_ranges[static_cast<size_t>(row)] =
                                LineitemTable::OrderRange{start, idx};
                        }
                    }
                    start = idx;
                    current_key = orderkey;
                }
                if (current_key >= 0 &&
                    static_cast<size_t>(current_key) < orders.orderkey_to_row.size()) {
                    const int32_t row =
                        orders.orderkey_to_row[static_cast<size_t>(current_key)];
                    if (row >= 0 &&
                        static_cast<size_t>(row) < orders.lineitem_ranges.size()) {
                        orders.lineitem_ranges[static_cast<size_t>(row)] =
                            LineitemTable::OrderRange{
                                start, static_cast<uint32_t>(lineitem.orderkey.size())};
                    }
                }
            } else {
                for (uint32_t idx = 0; idx < lineitem.orderkey.size(); ++idx) {
                    const int32_t orderkey = lineitem.orderkey[idx];
                    if (orderkey < 0 ||
                        static_cast<size_t>(orderkey) >= orders.orderkey_to_row.size()) {
                        continue;
                    }
                    const int32_t row =
                        orders.orderkey_to_row[static_cast<size_t>(orderkey)];
                    if (row < 0 ||
                        static_cast<size_t>(row) >= orders.lineitem_ranges.size()) {
                        continue;
                    }
                    auto& range = orders.lineitem_ranges[static_cast<size_t>(row)];
                    if (range.end == 0) {
                        range.start = idx;
                        range.end = idx + 1;
                    } else {
                        range.start = std::min(range.start, idx);
                        range.end = std::max(range.end, idx + 1);
                    }
                }
            }
        }
    }

    build_precomputed_artifacts(db);

    const auto build_end = std::chrono::steady_clock::now();
    const auto build_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "Build ms: " << build_ms << "\n";

    return db;
}
