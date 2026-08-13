#pragma once

#include <arrow/table.h>
#include <string>

std::shared_ptr<arrow::Table> ReadParquetTable(const std::string& path);
