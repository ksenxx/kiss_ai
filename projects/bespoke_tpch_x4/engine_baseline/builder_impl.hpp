#pragma once

#include "loader_impl.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct StringColumn {
    std::vector<uint32_t> offsets;
    std::string data;
    std::vector<uint32_t> alpha_mask;
    std::vector<uint64_t> bigram_mask;
};

struct DictionaryColumn {
    std::vector<uint16_t> codes;
    std::vector<std::string> dictionary;
};

struct LineitemShard {
    int32_t year = 0;
    int32_t month = 0;
    int16_t supp_nationkey = -1;
    int16_t min_shipdate = 0;
    int16_t max_shipdate = 0;
    int32_t min_discount = 0;
    int32_t max_discount = 0;
    int16_t min_quantity = 0;
    int16_t max_quantity = 0;
    uint32_t start = 0;
    uint32_t end = 0;
    bool contiguous = false;
    std::vector<uint32_t> row_indices;
};

struct LineitemTable {
    struct OrderRange {
        uint32_t start = 0;
        uint32_t end = 0;
    };

    size_t row_count = 0;
    std::vector<int32_t> orderkey;
    std::vector<int32_t> partkey;
    std::vector<int32_t> suppkey;
    std::vector<int16_t> supp_nationkey;
    std::vector<int32_t> linenumber;
    std::vector<int16_t> shipdate;
    std::vector<uint32_t> commit_receipt;
    std::vector<int32_t> extendedprice;
    std::vector<uint8_t> discount;
    std::vector<uint8_t> tax;
    std::vector<int16_t> quantity;
    std::vector<int32_t> discounted_price;
    std::vector<uint8_t> returnflag_linestatus;
    DictionaryColumn returnflag;
    DictionaryColumn linestatus;
    DictionaryColumn shipinstruct;
    DictionaryColumn shipmode;
    StringColumn comment;
    bool orderkey_sorted = false;
    std::vector<LineitemShard> shards;
};

struct OrdersTable {
    size_t row_count = 0;
    std::vector<int32_t> orderkey;
    std::vector<int32_t> custkey;
    std::vector<int16_t> cust_nationkey;
    std::vector<int16_t> orderdate;
    std::vector<int32_t> shippriority;
    std::vector<int32_t> totalprice;
    DictionaryColumn orderstatus;
    DictionaryColumn orderpriority;
    StringColumn clerk;
    StringColumn comment;
    std::vector<int32_t> orderkey_to_row;
    std::vector<LineitemTable::OrderRange> lineitem_ranges;
    std::vector<uint32_t> orders_by_customer_offsets;
    std::vector<uint32_t> orders_by_customer_rows;
};

struct CustomerTable {
    size_t row_count = 0;
    std::vector<int32_t> custkey;
    std::vector<int32_t> nationkey;
    std::vector<int16_t> nationkey_by_custkey;
    std::vector<int32_t> acctbal;
    DictionaryColumn mktsegment;
    StringColumn name;
    StringColumn address;
    StringColumn phone;
    std::vector<uint8_t> phone_prefix_code;
    StringColumn comment;
};

struct PartTable {
    size_t row_count = 0;
    std::vector<int32_t> partkey;
    std::vector<int32_t> size;
    std::vector<int32_t> retailprice;
    DictionaryColumn mfgr;
    DictionaryColumn brand;
    DictionaryColumn container;
    StringColumn name;
    DictionaryColumn type;
    StringColumn comment;
};

struct SupplierTable {
    size_t row_count = 0;
    std::vector<int32_t> suppkey;
    std::vector<int32_t> nationkey;
    std::vector<int16_t> nationkey_by_suppkey;
    std::vector<int32_t> acctbal;
    StringColumn name;
    StringColumn address;
    StringColumn phone;
    StringColumn comment;
};

struct PartsuppTable {
    size_t row_count = 0;
    std::vector<int32_t> partkey;
    std::vector<int32_t> suppkey;
    std::vector<int32_t> availqty;
    std::vector<int32_t> supplycost;
    std::vector<int32_t> supplycost_availqty;
    StringColumn comment;
};

struct NationTable {
    struct Row {
        int32_t nationkey = 0;
        int32_t regionkey = 0;
        std::string name;
        std::string comment;
    };

    std::vector<Row> rows;
    std::unordered_map<int32_t, uint32_t> nationkey_to_row;
    std::unordered_map<std::string, int32_t> name_to_key;
};

struct RegionTable {
    struct Row {
        int32_t regionkey = 0;
        std::string name;
        std::string comment;
    };

    std::vector<Row> rows;
    std::unordered_map<int32_t, uint32_t> regionkey_to_row;
    std::unordered_map<std::string, int32_t> name_to_key;
};

struct Database {
    int32_t base_date_days = 0;
    LineitemTable lineitem;
    OrdersTable orders;
    CustomerTable customer;
    PartTable part;
    SupplierTable supplier;
    PartsuppTable partsupp;
    NationTable nation;
    RegionTable region;
};


Database* build(ParquetTables*);
