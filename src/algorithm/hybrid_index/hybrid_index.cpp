//
// Created by root on 2026/1/8.
//
#include "hybrid_index.h"

#include <atomic>
#include <mutex>
#include <optional>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_interface.h"
#include "fmt/chrono.h"
#include "impl/heap/standard_heap.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "typing.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"

namespace vsag {

HybridIndex::HybridIndex(const HybridIndexParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param), sparse_vector_(common_param.allocator_.get()), dense_vector_(common_param.allocator_.get()){
    this->alpha_ = param->alpha;
}

ParamPtr
HybridIndex::CheckAndMappingExternalParam(const JsonType& external_param,
                                          const IndexCommonParam& common_param) {

    auto inner_json = external_param;

    auto hybrid_index_parameter = std::make_shared<HybridIndexParameter>();
    hybrid_index_parameter->FromJson(inner_json);

    return hybrid_index_parameter;
}


void
HybridIndex::Train(const DatasetPtr& data) {
}

std::vector<int64_t>
HybridIndex::Build(const DatasetPtr& data) {
    this->Train(data);
    return this->Add(data);
}


std::vector<int64_t>
HybridIndex::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    int64_t dense_num = data->GetNumElements();
    auto dense_vecs = data->GetFloat32Vectors();
    auto sparse_vecs = data->GetSparseVectors();

    int64_t dim = data->GetDim();

    for (int i = 0; i < dense_num; i++) {
        for (int j = 0; j < dim; j++) {
            this->dense_vector_.push_back(dense_vecs[i * dim + j]);
        }
        this->sparse_vector_.push_back(sparse_vecs[i]);
    }

    this->total_count_ += dense_num;
    return failed_ids;
}

DatasetPtr
HybridIndex::KnnSearch(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const FilterPtr& filter) const {
    SearchRequest req;
    req.query_ = query;
    req.topk_ = k;
    req.params_str_ = parameters;
    if (filter != nullptr) {
        req.filter_ = filter;
    }
    return this->SearchWithRequest(req);
}

DatasetPtr
HybridIndex::SearchWithRequest(const SearchRequest& request) const {
    // support 1 query only
    auto heap = vsag::DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, request.topk_);

    auto dense_query = request.query_->GetFloat32Vectors();
    auto sparse_query = request.query_->GetSparseVectors();

    for (int i = 0; i < this->total_count_; i++) {
        float dis = 0;
        // dense ip distance
        float dense_dis = 0;
        for (int j = 0; j < this->dim_; j++) {
            dense_dis += dense_query[j] * dense_vector_[i * this->dim_ + j];
        }

        // sparse ip distance
        float sparse_dis = 0;
        auto cur_sparse_vector = sparse_vector_[i];
        auto num_base_len = cur_sparse_vector.len_;
        auto num_query_len = sparse_query[0].len_;
        int j = 0, k = 0;
        while (j < num_base_len && k < num_query_len) {
            if (cur_sparse_vector.ids_[j] < sparse_query[0].ids_[k]) j++;
            else if (cur_sparse_vector.ids_[j] > sparse_query[0].ids_[k]) k++;
            else {
                sparse_dis += cur_sparse_vector.vals_[j] * sparse_query[0].vals_[k];
                k++;
                j++;
            }
        }

        JsonType params = vsag::JsonType::Parse(request.params_str_);
        auto search_alpha = params["alpha"].GetFloat();
        dis = search_alpha * dense_dis + (1 - search_alpha) * sparse_dis;
        heap->Push(dis, i);
    }

    auto [results, dists, ids] = create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_); // structed binding
    for (int j = 0; j < request.topk_; j++) {
        dists[j] = heap->Top().first;
        ids[j] = heap->Top().second;
        heap->Pop();
    }
    return results;
}

float
HybridIndex::CalcDistanceById(const float* vector, int64_t id) const {
    float result = 0.0F;
    return result;
}


void
HybridIndex::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
}

void
HybridIndex::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    this->attr_filter_index_->GetAttribute(0, inner_id, attr);
}

void
HybridIndex::Deserialize(StreamReader& reader) {

}

void
HybridIndex::InitFeatures() {

}

DatasetPtr
HybridIndex::RangeSearch(const vsag::DatasetPtr& query,
                        float radius,
                        const std::string& parameters,
                        const vsag::FilterPtr& filter,
                        int64_t limited_size) const {
    auto result = vsag::Dataset::Make();
    return result;

}

void
HybridIndex::Serialize(StreamWriter& writer) const {

}

}  // namespace vsag