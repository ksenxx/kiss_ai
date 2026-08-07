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

    const auto build_end = std::chrono::steady_clock::now();
    const auto build_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "Build ms: " << build_ms << "\n";

    return db;
}
