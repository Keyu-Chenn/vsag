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

HybridIndex::~HybridIndex() {
    for (auto i : sparse_vector_) {
        this->allocator_->Deallocate(i.ids_);
        this->allocator_->Deallocate(i.vals_);
    }
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

    int64_t dim = data->GetDim();

    dense_vector_.resize(dense_num * dim);
    sparse_vector_.resize(dense_num);
    auto dense_vecs = data->GetFloat32Vectors();
    auto sparse_vecs = data->GetSparseVectors();


    for (int i = 0; i < dense_num; i++) {
        for (int j = 0; j < dim; j++) {
            this->dense_vector_[i * dim + j] = dense_vecs[i * dim + j];
        }
        sparse_vector_[i].len_ = sparse_vecs[i].len_;
        sparse_vector_[i].ids_ = (uint32_t*)this->allocator_->Allocate(sparse_vector_[i].len_ * sizeof(uint32_t));
        std::memcpy(sparse_vector_[i].ids_, sparse_vecs[i].ids_,sparse_vector_[i].len_ * sizeof(uint32_t));
        sparse_vector_[i].vals_ = (float*)this->allocator_->Allocate(sparse_vector_[i].len_ * sizeof(float));
        std::memcpy(sparse_vector_[i].vals_, sparse_vecs[i].vals_, sparse_vector_[i].len_ * sizeof(float));

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
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout<< "train: " << std::endl;

    // support 1 query only
    auto heap = vsag::DistanceHeap::MakeInstanceBySize<false, true>(this->allocator_, request.topk_);

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
        //
        // std::cout << i << std::endl;
        // std::cout << "dense: " << std::endl;
        // for (int m = 0; m < this->dim_; m++) {
        //     std::cout << dense_vector_[i * this->dim_ + m] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "sparse" << std::endl;
        // std::cout << "ids: " << std::endl;
        // for (int m = 0; m < cur_sparse_vector.len_; m++) {
        //     std::cout << cur_sparse_vector.ids_[m] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "vals: " << std::endl;
        // for (int m = 0; m < cur_sparse_vector.len_; m++) {
        //     std::cout << cur_sparse_vector.vals_[m] << " ";
        // }
        // std::cout << std::endl;
    }


    //
    // std::cout << std::endl;
    // std::cout << "query: " << std::endl;
    // std::cout << "dense: " << std::endl;
    // for (int i = 0; i < this->dim_; i++) {
    //     std::cout << dense_query[i] << ' ';
    // }
    // std::cout << std::endl;
    // std::cout << "sparse: " << std::endl;
    // std::cout << "ids: ";
    // for (int j = 0; j < sparse_query[0].len_; j++) {
    //     std::cout << sparse_query[0].ids_[j] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "vals: ";
    // for (int j = 0; j < sparse_query[0].len_; j++) {
    //     std::cout << sparse_query[0].vals_[j] << " ";
    // }
    //
    // std::cout << std::endl;
    // std::cout << "result: " << std::endl;
    auto [results, dists, ids] = create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_); // structed binding
    for (int j = 0; j < request.topk_; j++) {
        dists[request.topk_ - j - 1] = heap->Top().first;
        ids[request.topk_ - j - 1] = heap->Top().second;
        heap->Pop();
        // std::cout<< "top" << j << ": " << std::endl;
        // std::cout<<"ids[j]:" << ids[j] << std::endl;
        // std::cout<<"dist[j]" << dists[j] << std::endl;
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
    auto footer = Footer::Parse(reader);

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    auto metadata = footer->GetMetadata();
    auto basic_info = metadata->Get("basic_info");
    if (basic_info.Contains(INDEX_PARAM)) {
        std::string index_param_string = basic_info[INDEX_PARAM].GetString();
        auto index_param = std::make_shared<HybridIndexParameter>();
        index_param->FromString(index_param_string);
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message =
                fmt::format("Hybrid index parameter not match, current: {}, new: {}",
                            this->create_param_ptr_->ToString(),
                            index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }
    dim_ = basic_info["dim"].GetInt();
    total_count_ = basic_info["total_count"].GetInt();

    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Deserialize(buffer_reader);
    }

    this->label_table_->Deserialize(buffer_reader);

    // Dense vector - 逐个读取
    uint64_t dense_size = 0;
    StreamReader::ReadObj(buffer_reader, dense_size);
    dense_vector_.resize(dense_size);
    for (uint64_t i = 0; i < dense_size; ++i) {
        StreamReader::ReadObj(buffer_reader, dense_vector_[i]);
    }

    // Sparse vector - 逐个读取
    uint64_t sparse_size = 0;
    StreamReader::ReadObj(buffer_reader, sparse_size);
    sparse_vector_.resize(sparse_size);
    for (uint64_t i = 0; i < sparse_size; ++i) {
        uint32_t len = 0;
        StreamReader::ReadObj(buffer_reader, len);
        sparse_vector_[i].len_ = len;

        if (len > 0) {
            sparse_vector_[i].ids_ = (uint32_t*)this->allocator_->Allocate(len * sizeof(uint32_t));
            sparse_vector_[i].vals_ = (float*)this->allocator_->Allocate(len * sizeof(float));

            // 逐个读取 ids 和 vals
            for (uint32_t j = 0; j < len; ++j) {
                StreamReader::ReadObj(buffer_reader, sparse_vector_[i].ids_[j]);
                StreamReader::ReadObj(buffer_reader, sparse_vector_[i].vals_[j]);
            }
        } else {
            sparse_vector_[i].ids_ = nullptr;
            sparse_vector_[i].vals_ = nullptr;
        }
    }

    // 读取 alpha
    StreamReader::ReadObj(buffer_reader, alpha_);
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



void HybridIndex::Serialize(StreamWriter& writer) const {
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }
    this->label_table_->Serialize(writer);

    // Dense vector
    uint64_t dense_size = dense_vector_.size();
    StreamWriter::WriteObj(writer, dense_size);
    for (uint64_t i = 0; i < dense_size; ++i) {
        StreamWriter::WriteObj(writer, dense_vector_[i]);  // 逐个写入
    }

    // Sparse vector
    uint64_t sparse_size = sparse_vector_.size();
    StreamWriter::WriteObj(writer, sparse_size);
    for (const auto& sv : sparse_vector_) {
        StreamWriter::WriteObj(writer, sv.len_);
        for (uint32_t i = 0; i < sv.len_; ++i) {
            StreamWriter::WriteObj(writer, sv.ids_[i]);
            StreamWriter::WriteObj(writer, sv.vals_[i]);
        }
    }

    StreamWriter::WriteObj(writer, alpha_);

    // Footer
    auto metadata = std::make_shared<Metadata>();
    JsonType basic_info;
    basic_info["dim"].SetInt(dim_);
    basic_info["total_count"].SetInt(total_count_);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    metadata->Set("basic_info", basic_info);
    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}


}  // namespace vsag