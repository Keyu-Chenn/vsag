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
                               float dense_query_norm,
                               float sparse_query_norm,
                               float dense_weight,
                               float sparse_weight)
    : dense_computer_(dense_computer),
      sparse_computer_(sparse_computer),
      dense_weight_(dense_weight),
      sparse_weight_(sparse_weight),
      dense_query_norm_(dense_query_norm),
      sparse_query_norm_(sparse_query_norm) {
    if (!dense_computer_ || !sparse_computer_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                          "Dense and sparse computers must not be null");
    }
    if (dense_weight_ < 0.0f || sparse_weight_ < 0.0f) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                           "Weights must be non-negative");
    }
}

bool
HybridComputer::TryGetSparseDistance(InnerIdType inner_id, float& distance) const {
    if (sparse_distance_table_ == nullptr or inner_id >= sparse_distance_table_size_) {
        return false;
    }
    distance = sparse_distance_table_[inner_id];
    return true;
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
    IndexCommonParam common_param_sparse = common_param;
    common_param_sparse.dim_ = 4096;
    sparse_cell_ = FlattenInterface::MakeInstance(sparse_param, common_param_sparse);

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
    const float lower_bound = hybrid_comp->GetSearchLowerBound();

    // // no prune
    // // Query dense cell
    // if (std::abs(dense_weight_) > 1e-5)
    // dense_cell_->Query(dense_dists.data(), hybrid_comp->GetDenseComputer(),
    //                   idx, id_count, search_alloc);
    //
    //
    //
    // // Query sparse cell
    // if (std::abs(dense_weight_ - 1) > 1e-5)
    // sparse_cell_->Query(sparse_dists.data(), hybrid_comp->GetSparseComputer(),
    //                    idx, id_count, search_alloc);
    //
    // for (InnerIdType i = 0; i < id_count; i++) {
    //     result_dists[i] = dense_weight_ * dense_dists[i] + sparse_weight_ * sparse_dists[i];
    //     // if (sparse_dists[i] < 0)
    //     // std::cout << "sparse_dist: " << sparse_dists[i] << "  ";
    // }



    // // prune sparse compute
    // // Query dense cell
    // if (std::abs(dense_weight_) > 1e-5)
    // dense_cell_->Query(dense_dists.data(), hybrid_comp->GetDenseComputer(),
    //                   idx, id_count, search_alloc);
    //
    // std::vector<InnerIdType> idx_sparse;
    // InnerIdType id_count_sparse = 0;
    // float threshold = 0.8;
    // for (InnerIdType i = 0; i < id_count; i ++) {
    //     if (dense_weight_ * dense_dists[i] + (1 - dense_weight_) * threshold * dense_dists[i] < lower_bound) {
    //         idx_sparse.push_back(idx[i]);
    //         id_count_sparse++;
    //     }
    // }
    //
    //
    // // Query sparse cell
    // // sparse_cell_->Query(sparse_dists.data(), hybrid_comp->GetSparseComputer(),
    // //                    idx, id_count, search_alloc);
    // if (std::abs(dense_weight_ - 1) > 1e-5)
    // sparse_cell_->Query(sparse_dists.data(), hybrid_comp->GetSparseComputer(),
    //                    idx_sparse.data(), id_count_sparse, search_alloc);
    //
    // // Combine results with weights
    // InnerIdType j = 0;
    // for (InnerIdType i = 0; i < id_count; ++i) {
    //     if (j < idx_sparse.size() && idx[i] == idx_sparse[j]) {
    //         result_dists[i] = dense_weight_ * dense_dists[i] + sparse_weight_ * sparse_dists[j];
    //         j++;
    //     }
    //     else {
    //         result_dists[i] = dense_weight_ * dense_dists[i];
    //     }
    // }
    // // for (InnerIdType i = 0; i < id_count; i++) {
    // //     result_dists[i] = dense_weight_ * dense_dists[i] + sparse_weight_ * sparse_dists[i];
    // //     // if (sparse_dists[i] < 0)
    // //     // std::cout << "sparse_dist: " << sparse_dists[i] << "  ";
    // // }




    if (id_count == 0) {
        return;
    }

    constexpr float kWeightEpsilon = 1e-5F;
    constexpr float kDistanceEpsilon = 1e-6F;

    if (std::abs(sparse_weight_) <= kWeightEpsilon) {
        if (std::abs(dense_weight_) > kWeightEpsilon) {
            dense_cell_->Query(
                dense_dists.data(), hybrid_comp->GetDenseComputer(), idx, id_count, search_alloc);
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            result_dists[i] = dense_weight_ * dense_dists[i];
        }
        return;
    }

    if (std::abs(dense_weight_) <= kWeightEpsilon or
        lower_bound == std::numeric_limits<float>::max()) {
        if (std::abs(dense_weight_) > kWeightEpsilon) {
            dense_cell_->Query(
                dense_dists.data(), hybrid_comp->GetDenseComputer(), idx, id_count, search_alloc);
        }
        Vector<InnerIdType> idx_sparse(search_alloc);
        Vector<InnerIdType> sparse_positions(search_alloc);
        idx_sparse.reserve(id_count);
        sparse_positions.reserve(id_count);
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (not hybrid_comp->TryGetSparseDistance(idx[i], sparse_dists[i])) {
                idx_sparse.push_back(idx[i]);
                sparse_positions.push_back(i);
            }
        }
        if (not idx_sparse.empty()) {
            Vector<float> selected_sparse_dists(idx_sparse.size(), 0.0F, search_alloc);
            sparse_cell_->Query(selected_sparse_dists.data(),
                                hybrid_comp->GetSparseComputer(),
                                idx_sparse.data(),
                                static_cast<InnerIdType>(idx_sparse.size()),
                                search_alloc);
            for (InnerIdType i = 0; i < idx_sparse.size(); ++i) {
                sparse_dists[sparse_positions[i]] = selected_sparse_dists[i];
            }
        }
        for (InnerIdType i = 0; i < id_count; ++i) {
            result_dists[i] = dense_weight_ * dense_dists[i] + sparse_weight_ * sparse_dists[i];
        }
        return;
    }

    const float score_threshold = 1.0F - lower_bound;
    const float prune_scale = std::max(0.0F, hybrid_comp->GetPruneScale());

    if (hybrid_comp->HasSparseDistanceTable() and std::abs(dense_weight_) > kWeightEpsilon) {
        const float query_dense_norm = hybrid_comp->GetDenseQueryNorm();
        Vector<InnerIdType> idx_dense(search_alloc);
        Vector<InnerIdType> dense_positions(search_alloc);
        idx_dense.reserve(id_count);
        dense_positions.reserve(id_count);

        for (InnerIdType i = 0; i < id_count; ++i) {
            if (not hybrid_comp->TryGetSparseDistance(idx[i], sparse_dists[i])) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "sparse distance table is missing an in-range hybrid id");
            }
            const float sparse_ip = 1.0F - sparse_dists[i];
            const float dense_ip_upper_bound =
                idx[i] < dense_norms_.size() ? query_dense_norm * dense_norms_[idx[i]]
                                             : std::numeric_limits<float>::infinity();
            const float score_upper_bound =
                dense_weight_ * prune_scale * dense_ip_upper_bound + sparse_weight_ * sparse_ip;
            if (score_upper_bound > score_threshold + kDistanceEpsilon) {
                idx_dense.push_back(idx[i]);
                dense_positions.push_back(i);
            } else {
                result_dists[i] = lower_bound + kDistanceEpsilon;
            }
        }

        if (not idx_dense.empty()) {
            Vector<float> selected_dense_dists(idx_dense.size(), 0.0F, search_alloc);
            dense_cell_->Query(selected_dense_dists.data(),
                               hybrid_comp->GetDenseComputer(),
                               idx_dense.data(),
                               static_cast<InnerIdType>(idx_dense.size()),
                               search_alloc);
            for (InnerIdType i = 0; i < idx_dense.size(); ++i) {
                const auto pos = dense_positions[i];
                result_dists[pos] = dense_weight_ * selected_dense_dists[i] +
                                    sparse_weight_ * sparse_dists[pos];
            }
        }
        return;
    }

    if (std::abs(dense_weight_) > kWeightEpsilon) {
        dense_cell_->Query(
            dense_dists.data(), hybrid_comp->GetDenseComputer(), idx, id_count, search_alloc);
    }

    const float query_sparse_norm = hybrid_comp->GetSparseQueryNorm();
    Vector<InnerIdType> idx_sparse(search_alloc);
    Vector<InnerIdType> sparse_positions(search_alloc);
    idx_sparse.reserve(id_count);
    sparse_positions.reserve(id_count);

    for (InnerIdType i = 0; i < id_count; ++i) {
        const auto id = idx[i];
        const float dense_ip = 1.0F - dense_dists[i];
        float sparse_ip_upper_bound =
            id < sparse_norms_.size() ? query_sparse_norm * sparse_norms_[id]
                                      : std::numeric_limits<float>::infinity();
        const float score_upper_bound =
            dense_weight_ * dense_ip + sparse_weight_ * prune_scale * sparse_ip_upper_bound;

        if (score_upper_bound > score_threshold + kDistanceEpsilon) {
            idx_sparse.push_back(id);
            sparse_positions.push_back(i);
        } else {
            result_dists[i] = lower_bound + kDistanceEpsilon;
        }
    }

    if (not idx_sparse.empty()) {
        Vector<float> selected_sparse_dists(idx_sparse.size(), 0.0F, search_alloc);
        Vector<InnerIdType> idx_sparse_to_compute(search_alloc);
        Vector<InnerIdType> sparse_compute_positions(search_alloc);
        idx_sparse_to_compute.reserve(idx_sparse.size());
        sparse_compute_positions.reserve(idx_sparse.size());

        for (InnerIdType i = 0; i < idx_sparse.size(); ++i) {
            if (not hybrid_comp->TryGetSparseDistance(idx_sparse[i], selected_sparse_dists[i])) {
                idx_sparse_to_compute.push_back(idx_sparse[i]);
                sparse_compute_positions.push_back(i);
            }
        }

        if (not idx_sparse_to_compute.empty()) {
            Vector<float> computed_sparse_dists(idx_sparse_to_compute.size(), 0.0F, search_alloc);
            sparse_cell_->Query(computed_sparse_dists.data(),
                                hybrid_comp->GetSparseComputer(),
                                idx_sparse_to_compute.data(),
                                static_cast<InnerIdType>(idx_sparse_to_compute.size()),
                                search_alloc);
            for (InnerIdType i = 0; i < idx_sparse_to_compute.size(); ++i) {
                selected_sparse_dists[sparse_compute_positions[i]] = computed_sparse_dists[i];
            }
        }

        for (InnerIdType i = 0; i < idx_sparse.size(); ++i) {
            const auto pos = sparse_positions[i];
            result_dists[pos] =
                dense_weight_ * dense_dists[pos] + sparse_weight_ * selected_sparse_dists[i];
        }
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
    const auto& sparse_query_vector = *static_cast<const SparseVector*>(sparse_query);
    const auto dense_query_norm = compute_dense_norm(dense_query);
    const auto sparse_query_norm = compute_sparse_norm(sparse_query_vector);

    // Create computers for both dense and sparse parts
    auto computer_dense = dense_cell_->FactoryComputer(dense_query);
    auto computer_sparse = sparse_cell_->FactoryComputer(sparse_query);

    // Create and return HybridComputer as shared_ptr
    return std::make_shared<HybridComputer>(
        computer_dense,
        computer_sparse,
        dense_query_norm,
        sparse_query_norm,
        dense_weight_,
        sparse_weight_
    );
}

float
HybridVectorDataCell::compute_sparse_norm(const SparseVector& sparse_vector) const {
    float norm_sqr = 0.0F;
    for (uint32_t i = 0; i < sparse_vector.len_; ++i) {
        norm_sqr += sparse_vector.vals_[i] * sparse_vector.vals_[i];
    }
    return std::sqrt(norm_sqr);
}

float
HybridVectorDataCell::compute_dense_norm(const float* dense_vector) const {
    float norm_sqr = 0.0F;
    for (int64_t i = 0; i < dense_dim_; ++i) {
        norm_sqr += dense_vector[i] * dense_vector[i];
    }
    return std::sqrt(norm_sqr);
}

void
HybridVectorDataCell::ensure_norm_capacity(InnerIdType capacity) {
    if (dense_norms_.size() < capacity) {
        dense_norms_.resize(capacity, 0.0F);
    }
    if (sparse_norms_.size() < capacity) {
        sparse_norms_.resize(capacity, 0.0F);
    }
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
    ensure_norm_capacity(idx + 1);
    dense_norms_[idx] = compute_dense_norm(reinterpret_cast<const float*>(vec_ptr));
    const auto& sparse_vector = *reinterpret_cast<const SparseVector*>(vec_ptr + dense_size);
    sparse_norms_[idx] = compute_sparse_norm(sparse_vector);
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
    if (idx < sparse_norms_.size()) {
        dense_norms_[idx] = compute_dense_norm(reinterpret_cast<const float*>(vec_ptr));
        const auto& sparse_vector = *reinterpret_cast<const SparseVector*>(vec_ptr + dense_size);
        sparse_norms_[idx] = compute_sparse_norm(sparse_vector);
    }

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

    const auto* sparse_array = reinterpret_cast<const SparseVector*>(sparse_vectors);
    if (idx_vec == nullptr) {
        ensure_norm_capacity(static_cast<InnerIdType>(sparse_cell_->TotalCount()));
        const auto start_id = static_cast<InnerIdType>(sparse_cell_->TotalCount() - count);
        for (InnerIdType i = 0; i < count; ++i) {
            dense_norms_[start_id + i] = compute_dense_norm(dense_vectors + i * dense_dim_);
            sparse_norms_[start_id + i] = compute_sparse_norm(sparse_array[i]);
        }
    } else {
        InnerIdType max_id = 0;
        for (InnerIdType i = 0; i < count; ++i) {
            max_id = std::max(max_id, idx_vec[i]);
        }
        ensure_norm_capacity(max_id + 1);
        for (InnerIdType i = 0; i < count; ++i) {
            dense_norms_[idx_vec[i]] = compute_dense_norm(dense_vectors + i * dense_dim_);
            sparse_norms_[idx_vec[i]] = compute_sparse_norm(sparse_array[i]);
        }
    }
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
    StreamWriter::WriteVector(writer, sparse_norms_);
    // Serialize both cells
    dense_cell_->Serialize(writer);
    sparse_cell_->Serialize(writer);
}

void
HybridVectorDataCell::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    StreamReader::ReadObj(reader, dense_weight_);
    StreamReader::ReadObj(reader, sparse_weight_);
    StreamReader::ReadObj(reader, dense_dim_);
    StreamReader::ReadVector(reader, sparse_norms_);
    dense_cell_->Deserialize(reader);
    sparse_cell_->Deserialize(reader);
}




}  // namespace vsag
