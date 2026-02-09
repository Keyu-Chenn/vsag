// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <limits>
#include <memory>

#include "common.h"
#include "flatten_interface.h"
#include "flatten_datacell.h"
#include "sparse_vector_datacell.h"
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "quantization/quantizer.h"
#include "utils/byte_buffer.h"

namespace vsag {

/**
 * HybridVectorDataCell manages both dense and sparse vectors
 * Each hybrid vector contains one dense vector and one sparse vector
 */

// Hybrid Computer that combines dense and sparse computers
class HybridComputer : public ComputerInterface {
public:
    HybridComputer(const ComputerInterfacePtr& dense_computer,
                   const ComputerInterfacePtr& sparse_computer,
                   float dense_weight = 0.5f,
                   float sparse_weight = 0.5f);


    // Accessors for internal computers
    ComputerInterfacePtr
    GetDenseComputer() const {
        return dense_computer_;
    }

    ComputerInterfacePtr
    GetSparseComputer() const {
        return sparse_computer_;
    }

private:
    ComputerInterfacePtr dense_computer_;
    ComputerInterfacePtr sparse_computer_;
    float dense_weight_;
    float sparse_weight_;
};


class HybridVectorDataCell : public FlattenInterface {
public:
    HybridVectorDataCell() = default;

    HybridVectorDataCell(const FlattenInterfaceParamPtr& dense_param,
                        const FlattenInterfaceParamPtr& sparse_param,
                        const IndexCommonParam& common_param);

    void
    Query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          Allocator* allocator = nullptr) override {
        return this->query(result_dists, computer, idx, id_count, allocator);
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        if (query == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "Query must not be null");
        }
        return this->factory_computer(static_cast<const float*>(query));
    }

    float
    ComputePairVectors(InnerIdType id1, InnerIdType id2) override {
        float dense_dist = dense_cell_->ComputePairVectors(id1, id2);
        float sparse_dist = sparse_cell_->ComputePairVectors(id1, id2);
        return dense_weight_ * dense_dist + sparse_weight_ * sparse_dist;
    }

    void
    Train(const void* data, uint64_t count) override;

    void
    InsertVector(const void* vector, InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override;

    bool
    UpdateVector(const void* vector, InnerIdType idx = std::numeric_limits<InnerIdType>::max()) override;

    void
    BatchInsertVector(const void* vectors, InnerIdType count, InnerIdType* idx_vec = nullptr) override;

    bool
    Decode(const uint8_t* codes, DataType* data) override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                          "Decode function is not directly supported for HybridVectorDataCell");
    }

    void
    Resize(InnerIdType new_capacity) override {
        dense_cell_->Resize(new_capacity);
        sparse_cell_->Resize(new_capacity);
    }

    void
    Prefetch(InnerIdType id) override {
        dense_cell_->Prefetch(id);
        sparse_cell_->Prefetch(id);
    }

    void
    ExportModel(const FlattenInterfacePtr& other) const override;

    void
    MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) override;

    [[nodiscard]] std::string
    GetQuantizerName() override;

    [[nodiscard]] MetricType
    GetMetricType() override {
        return dense_cell_->GetMetricType();
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType id, bool& need_release) const override;

    void
    Release(const uint8_t* data) const override;

    [[nodiscard]] bool
    InMemory() const override {
        return dense_cell_->InMemory() && sparse_cell_->InMemory();
    }

    bool
    HoldMolds() const override {
        return dense_cell_->HoldMolds() || sparse_cell_->HoldMolds();
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    void
    InitIO(const IOParamPtr& io_param) override {
        dense_cell_->InitIO(io_param);
        sparse_cell_->InitIO(io_param);
    }

    // Accessors for dense and sparse components
    FlattenInterfacePtr
    GetDenseCell() {
        return dense_cell_;
    }

    FlattenInterfacePtr
    GetSparseCell() {
        return sparse_cell_;
    }

    // Set weight for combining dense and sparse distances
    void
    SetHybridWeight(float dense_weight, float sparse_weight) {
        if (dense_weight < 0.0f || sparse_weight < 0.0f) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                              "Weights must be non-negative");
        }
        dense_weight_ = dense_weight;
        sparse_weight_ = sparse_weight;
    }

    [[nodiscard]] MetricType
    GetDenseMetricType() {
        return dense_cell_->GetMetricType();
    }

    [[nodiscard]] MetricType
    GetSparseMetricType() {
        return sparse_cell_->GetMetricType();
    }

    [[nodiscard]] const uint8_t*
    GetDenseCodesById(InnerIdType id, bool& need_release) const {
        return dense_cell_->GetCodesById(id, need_release);
    }

    [[nodiscard]] const uint8_t*
    GetSparseCodesById(InnerIdType id, bool& need_release) const {
        return sparse_cell_->GetCodesById(id, need_release);
    }

private:
    void
    query(float* result_dists,
          const ComputerInterfacePtr& computer,
          const InnerIdType* idx,
          InnerIdType id_count,
          Allocator* allocator);

    ComputerInterfacePtr
    factory_computer(const float* query);

    uint64_t
    GetTotalCount() const {
        return dense_cell_->TotalCount();
    };

    uint64_t
    GetDenseVectorSize() const {
        return dense_cell_->code_size_;
    }

private:
    FlattenInterfacePtr dense_cell_{nullptr};
    FlattenInterfacePtr sparse_cell_{nullptr};

    Allocator* allocator_{nullptr};

    int64_t dense_dim_;
    // Weights for combining dense and sparse distances
    float dense_weight_{0.5f};
    float sparse_weight_{0.5f};
};

}  // namespace vsag
