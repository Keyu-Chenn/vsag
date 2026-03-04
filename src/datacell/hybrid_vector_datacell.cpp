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

#include "hybrid_vector_datacell.h"
#include <cstring>

namespace vsag {

// ============================================================================
// HybridComputer Implementation
// ============================================================================

HybridComputer::HybridComputer(const ComputerInterfacePtr& dense_computer,
                               const ComputerInterfacePtr& sparse_computer,
                               float dense_weight,
                               float sparse_weight)
    : dense_computer_(dense_computer),
      sparse_computer_(sparse_computer),
      dense_weight_(dense_weight),
      sparse_weight_(sparse_weight) {
    if (!dense_computer_ || !sparse_computer_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Dense and sparse computers must not be null");
    }
    if (dense_weight_ < 0.0f || sparse_weight_ < 0.0f) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Weights must be non-negative");
    }
}



// ============================================================================
// HybridVectorDataCell Implementation
// ============================================================================

HybridVectorDataCell::HybridVectorDataCell(const FlattenInterfaceParamPtr& dense_param,
                                          const FlattenInterfaceParamPtr& sparse_param,
                                          const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()), dense_dim_(common_param.dim_) {
    if (!dense_param || !sparse_param) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Dense and sparse parameters must not be null");
    }

    // Create dense cell
    dense_cell_ = FlattenInterface::MakeInstance(dense_param, common_param);

    // Create sparse cell
    sparse_cell_ = FlattenInterface::MakeInstance(sparse_param, common_param);

}

void
HybridVectorDataCell::query(float* result_dists,
                            const ComputerInterfacePtr& computer,
                            const InnerIdType* idx,
                            InnerIdType id_count,
                            Allocator* allocator) {
    auto hybrid_comp = std::dynamic_pointer_cast<HybridComputer>(computer);
    if (hybrid_comp == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                          "Computer must be HybridComputer for hybrid queries");
    }

    // Compute hybrid distances by querying both dense and sparse cells
    Allocator* search_alloc = allocator == nullptr ? allocator_ : allocator;
    Vector<float> dense_dists(id_count, 0.0f, search_alloc);
    Vector<float> sparse_dists(id_count, 0.0f, search_alloc);

    // Query dense cell
    dense_cell_->Query(dense_dists.data(), hybrid_comp->GetDenseComputer(),
                      idx, id_count, search_alloc);

    // Query sparse cell
    sparse_cell_->Query(sparse_dists.data(), hybrid_comp->GetSparseComputer(),
                       idx, id_count, search_alloc);

    // Combine results with weights
    for (InnerIdType i = 0; i < id_count; ++i) {
        result_dists[i] = dense_weight_ * dense_dists[i] + sparse_weight_ * sparse_dists[i];
    }
}

ComputerInterfacePtr
HybridVectorDataCell::factory_computer(const float* query) {
    // query is a byte stream: [dense_vector | sparse_vector]
    const uint8_t* query_ptr = reinterpret_cast<const uint8_t*>(query);

    // Extract dense part (fixed size)
    const float* dense_query = query;

    // Extract sparse part (starts after dense vector)
    uint64_t dense_size = GetDenseVectorSize();
    const void* sparse_query = query_ptr + dense_size;

    // Create computers for both dense and sparse parts
    auto computer_dense = dense_cell_->FactoryComputer(dense_query);
    auto computer_sparse = sparse_cell_->FactoryComputer(sparse_query);

    // Create and return HybridComputer as shared_ptr
    return std::make_shared<HybridComputer>(
        computer_dense,
        computer_sparse,
        dense_weight_,
        sparse_weight_
    );
}

void
HybridVectorDataCell::Train(const void* data, uint64_t count) {
    if (data == nullptr || count == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Training data must not be null and count must be positive");
    }

    const uint8_t* data_ptr = static_cast<const uint8_t*>(data);
    auto dense_size = this->GetDenseVectorSize() * count;
    // Train both cells
    dense_cell_->Train(data_ptr, count);
    sparse_cell_->Train(data_ptr + dense_size, count);
}

void
HybridVectorDataCell::InsertVector(const void* vector, InnerIdType idx) {
    if (vector == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Vector must not be null");
    }

    const uint8_t* vec_ptr = static_cast<const uint8_t*>(vector);
    uint64_t dense_size = GetDenseVectorSize();

    // Insert dense part
    dense_cell_->InsertVector(vec_ptr, idx);

    // Insert sparse part
    sparse_cell_->InsertVector(vec_ptr + dense_size, idx);
}

bool
HybridVectorDataCell::UpdateVector(const void* vector, InnerIdType idx) {
    if (vector == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Vector must not be null");
    }

    const uint8_t* vec_ptr = static_cast<const uint8_t*>(vector);
    uint64_t dense_size = GetDenseVectorSize();

    bool dense_updated = dense_cell_->UpdateVector(vec_ptr, idx);
    bool sparse_updated = sparse_cell_->UpdateVector(vec_ptr + dense_size, idx);

    return dense_updated && sparse_updated;
}

void
HybridVectorDataCell::BatchInsertVector(const void* vectors, InnerIdType count,
                                        InnerIdType* idx_vec) {
    if (vectors == nullptr || count == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Vectors must not be null and count must be positive");
    }

    const uint8_t* vec_ptr = static_cast<const uint8_t*>(vectors);
    uint64_t dense_size = GetDenseVectorSize();

    // vectors layout: [all dense vectors | all sparse vectors]
    // Dense part: count * dense_dim floats
    // Sparse part: count * SparseVector structures

    const float* dense_vectors = reinterpret_cast<const float*>(vec_ptr);
    const uint8_t* sparse_vectors = vec_ptr + (count * dense_size);

    // Batch insert dense vectors (more efficient as a batch)
    dense_cell_->BatchInsertVector(dense_vectors, count, idx_vec);

    // Batch insert sparse vectors
    sparse_cell_->BatchInsertVector(sparse_vectors, count, idx_vec);
}


void
HybridVectorDataCell::ExportModel(const FlattenInterfacePtr& other) const {
    auto hybrid_other = std::dynamic_pointer_cast<HybridVectorDataCell>(other);
    if (!hybrid_other) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Can only export to another HybridVectorDataCell");
    }

    dense_cell_->ExportModel(hybrid_other->dense_cell_);
    sparse_cell_->ExportModel(hybrid_other->sparse_cell_);
}

void
HybridVectorDataCell::MergeOther(const FlattenInterfacePtr& other, InnerIdType bias) {
    auto hybrid_other = std::dynamic_pointer_cast<HybridVectorDataCell>(other);
    if (!hybrid_other) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Can only merge with another HybridVectorDataCell");
    }

    dense_cell_->MergeOther(hybrid_other->dense_cell_, bias);
    sparse_cell_->MergeOther(hybrid_other->sparse_cell_, bias);
}

std::string
HybridVectorDataCell::GetQuantizerName() {
    return "HybridQuantizer(" + dense_cell_->GetQuantizerName() + "," +
           sparse_cell_->GetQuantizerName() + ")";
}

const uint8_t*
HybridVectorDataCell::GetCodesById(InnerIdType id, bool& need_release) const { // remember to deallocate
    // Allocate buffer for combined codes
    bool dense_release = false, sparse_release = false;
    const uint8_t* dense_codes = dense_cell_->GetCodesById(id, dense_release);
    const uint8_t* sparse_codes = sparse_cell_->GetCodesById(id, sparse_release);

    if (dense_codes == nullptr || sparse_codes == nullptr) {
        if (dense_release && dense_codes) dense_cell_->Release(dense_codes);
        if (sparse_release && sparse_codes) sparse_cell_->Release(sparse_codes);
        need_release = false;
        return nullptr;
    }

    // 读取 len (uint32_t)
    const uint32_t* len_ptr = reinterpret_cast<const uint32_t*>(sparse_codes);
    uint32_t sparse_len = *len_ptr;
    // 计算总大小

    auto dense_size = GetDenseVectorSize();
    uint64_t sparse_size = sizeof(uint32_t) +                    // len
                           sizeof(uint32_t) * sparse_len +       // ids
                           sizeof(float) * sparse_len;           // vals
    uint64_t total_size = dense_size + sparse_size;

    // allocate here and remember to deallocate when using this function GetCodesById()
    uint8_t* combined = static_cast<uint8_t*>(allocator_->Allocate(total_size));
    std::memcpy(combined, dense_codes, dense_size);
    std::memcpy(combined + dense_size, sparse_codes, sparse_size);

    return combined;
}

void
HybridVectorDataCell::Release(const uint8_t* data) const {
    dense_cell_->Release(data);
    sparse_cell_->Release(data);
}

bool
HybridVectorDataCell::GetCodesById(InnerIdType id, uint8_t* codes) const {
    if (codes == nullptr) {
        return false;
    }

    uint64_t dense_code_size = GetDenseVectorSize();
    bool dense_success = dense_cell_->GetCodesById(id, codes);
    bool sparse_success = sparse_cell_->GetCodesById(id, codes + dense_code_size);

    return dense_success && sparse_success;
}

void
HybridVectorDataCell::Serialize(StreamWriter& writer) {
    // Write weights
    StreamWriter::WriteObj(writer, dense_weight_);
    StreamWriter::WriteObj(writer, sparse_weight_);
    StreamWriter::WriteObj(writer, dense_dim_);
    // Serialize both cells
    dense_cell_->Serialize(writer);
    sparse_cell_->Serialize(writer);
}

void
HybridVectorDataCell::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    StreamReader::ReadObj(reader, dense_weight_);
    StreamReader::ReadObj(reader, sparse_weight_);
    StreamReader::ReadObj(reader, dense_dim_);
    dense_cell_->Deserialize(reader);
    sparse_cell_->Deserialize(reader);
}




}  // namespace vsag

