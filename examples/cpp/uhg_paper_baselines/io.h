// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.h"
#include <H5Cpp.h>
#include <regex>
#include <sstream>

namespace paper_baselines {

inline Vectors LoadVectors(const std::string& path, const std::string& split) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    Vectors data;
    auto dense = file.openDataSet(split); auto space = dense.getSpace();
    Require(space.getSimpleExtentNdims() == 2, "dense vectors must be a matrix");
    hsize_t shape[2]; space.getSimpleExtentDims(shape);
    Require(shape[0] > 0 && shape[0] < std::numeric_limits<Id>::max() && shape[1] > 0, "invalid data shape");
    data.dim = shape[1]; data.dense.resize(shape[0] * shape[1]);
    dense.read(data.dense.data(), H5::PredType::NATIVE_FLOAT);
    auto labels = file.openDataSet(split + "_labels");
    Require(labels.getSpace().getSimpleExtentNpoints() == static_cast<int64_t>(shape[0]), "label count mismatch");
    data.labels.resize(shape[0]); labels.read(data.labels.data(), H5::PredType::NATIVE_INT64);
    auto sparse = file.openDataSet(split + "_sparse");
    std::vector<uint8_t> blob(sparse.getSpace().getSimpleExtentNpoints());
    sparse.read(blob.data(), H5::PredType::NATIVE_UINT8);
    size_t cursor = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        Require(cursor + 4 <= blob.size(), "truncated sparse length");
        uint32_t n; std::memcpy(&n, blob.data() + cursor, 4); cursor += 4;
        Require(uint64_t(n) * 8 <= blob.size() - cursor, "truncated sparse data");
        Sparse vector; vector.entries.resize(n);
        for (uint32_t j = 0; j < n; ++j)
            std::memcpy(&vector.entries[j].id, blob.data() + cursor + size_t(j) * 4, 4);
        cursor += size_t(n) * 4;
        for (uint32_t j = 0; j < n; ++j)
            std::memcpy(&vector.entries[j].value, blob.data() + cursor + size_t(j) * 4, 4);
        cursor += size_t(n) * 4;
        data.sparse.push_back(std::move(vector));
    }
    Require(cursor == blob.size(), "extra sparse data after final vector");
    data.Validate(); return data;
}

template <class T> std::vector<T> LoadNpy(const std::string& path, const std::string& dtype,
                                         std::vector<size_t>& shape) {
    std::ifstream in(path, std::ios::binary);
    char magic[6]; in.read(magic, 6);
    Require(in.good() && std::string(magic, 6) == std::string("\x93NUMPY", 6), "invalid NPY magic: " + path);
    auto major = Read<uint8_t>(in); auto minor = Read<uint8_t>(in); (void)minor;
    Require(major == 1 || major == 2, "unsupported NPY version");
    uint32_t header_size = major == 1 ? Read<uint16_t>(in) : Read<uint32_t>(in);
    Require(header_size < 65536, "excessive NPY header");
    std::string header(header_size, '\0'); in.read(header.data(), header.size());
    Require(in.good(), "truncated NPY header");
    Require(header.find("'" + dtype + "'") != std::string::npos, "NPY dtype mismatch: " + path);
    Require(header.find("'fortran_order': False") != std::string::npos, "Fortran NPY unsupported");
    std::smatch match;
    Require(std::regex_search(header, match, std::regex("'shape': *\\(([^)]*)\\)")), "missing NPY shape");
    std::string dimensions = match[1]; std::replace(dimensions.begin(), dimensions.end(), ',', ' ');
    std::istringstream parse(dimensions); size_t dim, count = 1; shape.clear();
    while (parse >> dim) {
        Require(dim > 0 && dim < 1000000000 && count <= 1000000000 / dim, "invalid NPY shape");
        shape.push_back(dim); count *= dim;
    }
    Require(!shape.empty(), "empty NPY shape");
    std::vector<T> values(count); in.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    Require(in.good() && in.peek() == EOF, "NPY payload size mismatch"); return values;
}
}  // namespace paper_baselines
