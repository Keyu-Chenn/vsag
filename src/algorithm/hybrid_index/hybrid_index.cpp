//
// Created by root on 2026/1/8.
//
#include "hybrid_index.h"
#include <H5Cpp.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_set>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_interface.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "fmt/chrono.h"
#include "impl/heap/standard_heap.h"
#include "impl/pruning_strategy.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "typing.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"

namespace vsag {

HybridIndex::HybridIndex(const HybridIndexParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      ef_construction_(param->ef_construction),
      max_degree_(param->max_degree),
      alpha_(param->alpha),
      searcher_(std::make_unique<BasicSearcher>(common_param)) {
    const char* dense_param_str =
        R"({
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            }
        }
    )";
    auto dense_param_json = JsonType::Parse(dense_param_str);
    auto dense_param = std::make_shared<FlattenDataCellParameter>();
    dense_param->FromJson(dense_param_json);

    const char* sparse_param_str =
        R"({
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "sparse"
            }
        }
    )";

    JsonType sparse_param_json = JsonType::Parse(sparse_param_str);
    auto sparse_param = std::make_shared<SparseVectorDataCellParameter>();
    sparse_param->FromJson(sparse_param_json);

    hybrid_codes_ = std::make_shared<HybridVectorDataCell>(dense_param, sparse_param, common_param);
    hybrid_codes_->SetHybridWeight(alpha_, 1-alpha_);

    JsonType json;
    json["max_degree"].SetInt(max_degree_);
    JsonType io_json;
    io_json["type"].SetString("memory_io");
    json["io_params"].SetJson(io_json);
    auto graph_params = GraphInterfaceParameter::GetGraphParameterByJson(GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, json);
    graph_ = GraphInterface::MakeInstance(graph_params, common_param);
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
HybridIndex::Train(const DatasetPtr& base) {
    // const auto* base_data_dense = base->GetFloat32Vectors();
    // const auto* base_data_sparse = base->GetSparseVectors();
    // auto num_elements = base->GetNumElements();
    //
    // // 计算 dense 部分的总大小
    // size_t dense_total_size = num_elements * dim_ * sizeof(float);
    //
    // // 计算 sparse 部分的总大小
    // size_t sparse_total_size = 0;
    // for (size_t i = 0; i < num_elements; ++i) {
    //     const auto& sv = base_data_sparse[i];
    //     sparse_total_size += sizeof(uint32_t) +                    // len
    //                         sizeof(uint32_t) * sv.len_ + // ids
    //                         sizeof(float) * sv.len_;      // vals
    // }
    //
    // auto base_data = static_cast<uint8_t*>(allocator_->Allocate(dense_total_size + sparse_total_size));
    // memcpy(base_data, base_data_dense, dense_total_size);
    // // 复制 sparse（逐个序列化）
    // uint8_t* sparse_ptr = base_data + dense_total_size;
    // for (size_t i = 0; i < num_elements; ++i) {
    //     const auto& sv = base_data_sparse[i];
    //     uint32_t len = sv.len_;
    //
    //     // len
    //     std::memcpy(sparse_ptr, &len, sizeof(uint32_t));
    //     sparse_ptr += sizeof(uint32_t);
    //
    //     if (len > 0) {
    //         // ids
    //         std::memcpy(sparse_ptr, sv.ids_, sizeof(uint32_t) * len);
    //         sparse_ptr += sizeof(uint32_t) * len;
    //
    //         // vals
    //         std::memcpy(sparse_ptr, sv.vals_, sizeof(float) * len);
    //         sparse_ptr += sizeof(float) * len;
    //     }
    // }
    //
    // this->hybrid_codes_->Train(base_data, base->GetNumElements());
    // allocator_->Deallocate(base_data);
}

std::vector<int64_t>
HybridIndex::Build(const DatasetPtr& data) {
    // this->Train(data);  train is used when qualification is needed
    graph_->Resize(data->GetNumElements());
    return this->Add(data);
}

void
HybridIndex::add_one_point(InnerIdType inner_id, const float* vector) {
    if (!allocator_) {
        std::cout << "HybridIndex::allocator_ is null!" << std::endl;
    }

    // if (graph_->total_count_ == 0) {
    //     graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
    //     entry_point_id_ = inner_id;
    // }
    // else {

        // single alpha neighbors

        // InnerSearchParam search_param;
        // search_param.ef = ef_construction_;
        // search_param.topk = max_degree_;
        // search_param.search_mode = KNN_SEARCH;
        // search_param.ep = entry_point_id_;

        // auto vl = std::make_shared<VisitedList>(graph_->max_capacity_, allocator_);
        // Statistics discard_stats;
        // auto results = searcher_->Search(graph_, hybrid_codes_, vl, vector, search_param,
        //                                 (LabelTablePtr)nullptr, discard_stats);
        // auto e_mutex = std::make_shared<EmptyMutex>();
        // mutually_connect_new_element(inner_id, results, graph_, hybrid_codes_, e_mutex, allocator_);


        // all alpha neighbors
        // 静态变量，只读取一次HDF5文件
        static std::vector<int64_t> precomputed_neighbors;
        static std::vector<int64_t> precomputed_neighbor_counts;
        static int64_t precomputed_num_points = 0;
        static int64_t precomputed_max_neighbors = 0;
        static bool neighbors_loaded = false;

        if (!neighbors_loaded) {
            try {
                // std::string h5_file = "/tbase-project/vsag/build-release/examples/cpp/601_output_neighbors_k32.h5";
                // std::string h5_file = "/tbase-project/vsag/scripts/UHG/data/fhg/webis-touche2020_fhg_alpha_0_5.h5";
                // std::string h5_file = "/tbase-project/vsag/scripts/UHG/data/fhg/nq_fhg_alpha_0_5.h5";
                // std::string h5_file = "/tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg_v2.h5";
                std::string h5_file = "/tbase-project/vsag/scripts/UHG/data/fhg/msmarco_fhg_alpha_0_5.h5";
                std::cout << "Loading precomputed neighbors from " << h5_file << std::endl;
                H5::H5File file(h5_file, H5F_ACC_RDONLY);

                // 读取邻居数据
                H5::DataSet neighbors_dataset = file.openDataSet("neighbors");
                H5::DataSpace neighbors_dataspace = neighbors_dataset.getSpace();
                hsize_t dims[2];
                neighbors_dataspace.getSimpleExtentDims(dims);

                precomputed_num_points = dims[0];
                precomputed_max_neighbors = dims[1];

                precomputed_neighbors.resize(precomputed_num_points * precomputed_max_neighbors);
                neighbors_dataset.read(precomputed_neighbors.data(), H5::PredType::NATIVE_INT64);

                // 读取每个点的实际邻居数量
                H5::DataSet counts_dataset = file.openDataSet("neighbor_counts");
                precomputed_neighbor_counts.resize(precomputed_num_points);
                counts_dataset.read(precomputed_neighbor_counts.data(), H5::PredType::NATIVE_INT64);

                neighbors_loaded = true;
                std::cout << "Loaded precomputed neighbors for " << precomputed_num_points
                          << " points" << std::endl;

            } catch (H5::Exception& e) {
                std::cerr << "Failed to load precomputed neighbors: " << e.getDetailMsg() << std::endl;
                neighbors_loaded = false;
            }
        }

        // 使用预计算的邻居
        if (neighbors_loaded && inner_id < precomputed_num_points) {
            int64_t actual_count = std::min(precomputed_neighbor_counts[inner_id],
                                            static_cast<int64_t>(max_degree_));
            Vector<InnerIdType> neighbor_vec(allocator_);
            neighbor_vec.resize(actual_count);

            int64_t valid_count = 0;
            const int64_t row_offset = inner_id * precomputed_max_neighbors;
            for (int64_t i = 0; i < actual_count; ++i) {
                int64_t neighbor_id = precomputed_neighbors[row_offset + i];
                if (neighbor_id >= 0) {
                    neighbor_vec[valid_count++] = static_cast<InnerIdType>(neighbor_id);
                }
            }
            neighbor_vec.resize(valid_count);

            // 插入邻居关系
            graph_->InsertNeighborsById(inner_id, neighbor_vec);
        }
    // }

}

std::vector<int64_t>
HybridIndex::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    int64_t vec_num = data->GetNumElements();

    int64_t dim = data->GetDim();
    auto labels = data->GetIds();


    auto dense_vecs = data->GetFloat32Vectors();
    auto sparse_vecs = data->GetSparseVectors();

    const auto dense_bytes = dim * sizeof(float);
    const auto sparse_ref_bytes = sizeof(sparse_vecs[0]);
    auto hybrid_vec = static_cast<uint8_t*>(allocator_->Allocate(dense_bytes + sparse_ref_bytes));

    // insert labels and points
    for (auto offset = 0; offset < vec_num; offset++) {
        auto inner_id = total_count_ + offset;
        label_table_->Insert(inner_id, labels[offset]);
        std::memcpy(hybrid_vec, dense_vecs + offset * dim, dense_bytes);
        std::memcpy(hybrid_vec + dense_bytes, sparse_vecs + offset, sparse_ref_bytes);
        hybrid_codes_->InsertVector(hybrid_vec, inner_id);
        add_one_point(inner_id, reinterpret_cast<const float*>(hybrid_vec));
    }

    allocator_->Deallocate(hybrid_vec);

    this->total_count_ += vec_num;
    return failed_ids;
}

DatasetPtr
HybridIndex::KnnSearch(const DatasetPtr& query,
                      int64_t k,
                      const std::string& parameters,
                      const FilterPtr& filter) const {

    //
    // auto sparsevec = query->GetSparseVectors()[0];
    // std::cout << "ids: ";
    // for (int lk = 0; lk < sparsevec.len_; ++lk) {
    //     std::cout << sparsevec.ids_[lk] << "  ";
    // }
    // std::cout << std::endl;


    auto parsed_search_param = JsonType::Parse(parameters);
    InnerSearchParam search_param;
    search_param.ef = parsed_search_param["ef_search"].GetInt();
    search_param.hybrid_candidate_set_size = parsed_search_param.Contains("hybrid_candidate_set_size")
                                                 ? parsed_search_param["hybrid_candidate_set_size"].GetInt()
                                                 : 0;
    search_param.max_hops = parsed_search_param.Contains("max_hops")
                                ? parsed_search_param["max_hops"].GetInt()
                                : 0;
    search_param.topk = k;
    search_param.search_mode = KNN_SEARCH;
    search_param.is_hybrid = true;
    // search_param.ep = entry_point_id_;
    // search_param.ep = parsed_search_param["entry_point"].GetInt();

    // 支持多入口点和单入口点，两种方式兼容
    if (parsed_search_param.Contains("entry_points") &&
        parsed_search_param["entry_points"].IsArray()) {
        // 多入口点：从 JSON 数组解析
        // 格式：{"entry_points": [1, 2, 3, ...]}
        const auto& ep_array = parsed_search_param["entry_points"].GetVector();
        std::unordered_set<InnerIdType> dedup_entry_points;
        for (int i = 0; i < ep_array.size(); ++i) {
            // JSON 中存的是 label id，需要转换为 inner id
            int64_t label_id = ep_array[i];
            if (not label_table_->CheckLabel(label_id)) {
                continue;
            }
            auto inner_id = label_table_->GetIdByLabel(label_id);
            if (dedup_entry_points.insert(inner_id).second) {
                search_param.eps.push_back(inner_id);
            }
        }
        // eps 为空时（全部 label 无效）退回默认入口点
        if (search_param.eps.empty()) {
            search_param.ep = entry_point_id_;
        }
        } else if (parsed_search_param.Contains("entry_point")) {
            // 单入口点（保持向后兼容）
            search_param.ep = parsed_search_param["entry_point"].GetInt();
        } else {
            // 没有指定入口点，使用默认
            search_param.ep = entry_point_id_;
        }


    auto search_alpha_ = parsed_search_param["alpha"].GetFloat();
    hybrid_codes_->SetHybridWeight(search_alpha_, 1-search_alpha_);
    search_param.hybrid_prune_scale = parsed_search_param.Contains("hybrid_prune_scale")
                                          ? parsed_search_param["hybrid_prune_scale"].GetFloat()
                                          : 1.0F;

    if (query->GetExtraInfos() != nullptr &&
        query->GetExtraInfoSize() >= static_cast<int64_t>(sizeof(int64_t))) {
        int64_t sparse_distance_table_size = 0;
        std::memcpy(
            &sparse_distance_table_size, query->GetExtraInfos(), sizeof(sparse_distance_table_size));
        const auto expected_extra_info_size =
            static_cast<int64_t>(sizeof(sparse_distance_table_size)) +
            static_cast<int64_t>(sizeof(float)) * sparse_distance_table_size;
        CHECK_ARGUMENT(sparse_distance_table_size >= 0,
                       "sparse distance count in query extra info should not be negative");
        CHECK_ARGUMENT(query->GetExtraInfoSize() == expected_extra_info_size,
                       "query extra info size does not match sparse distance table size");
        search_param.sparse_distance_table =
            reinterpret_cast<const float*>(query->GetExtraInfos() + sizeof(sparse_distance_table_size));
        search_param.sparse_distance_table_size = sparse_distance_table_size;
    }

    auto dense_vector = query->GetFloat32Vectors();
    auto dim = query->GetDim();
    auto sparse_vector = query->GetSparseVectors();
    auto hybrid_vector = (int8_t*)(allocator_->Allocate(dim * sizeof(float) + sizeof(sparse_vector[0])));
    std::memcpy(hybrid_vector, dense_vector, dim * sizeof(float));
    std::memcpy(hybrid_vector + dim * sizeof(float), sparse_vector, sizeof(sparse_vector[0]));

    auto vl = std::make_shared<VisitedList>(graph_->max_capacity_, allocator_);
    Statistics discard_stats;
    auto search_results = searcher_->Search(graph_, hybrid_codes_, vl, hybrid_vector, search_param,
                                    (LabelTablePtr)nullptr, discard_stats);

    allocator_->Deallocate(hybrid_vector);
    int64_t result_size = search_results->Size();
    auto results = Dataset::Make();

    results->NumElements(1)->Dim(result_size)->Owner(true, allocator_);
    auto ids = (int64_t*)allocator_->Allocate(result_size * sizeof(int64_t));
    results->Ids(ids);
    auto dists = (float*)allocator_->Allocate(result_size * sizeof(float));
    results->Distances(dists);

    for (auto j = result_size - 1; j >= 0; --j) {
        if (j < result_size) {
            dists[j] = search_results->Top().first;
            ids[j]= label_table_->GetLabelById(search_results->Top().second);
        }
        search_results->Pop();
    }
    results->Statistics(discard_stats.Dump());
    // for (int j = 0; j < result_size; ++j) {
    //     std::cout << "id: " << ids[j] << "      ";
    //     std::cout << "dist: " << dists[j] << std::endl;
    // }
    // allocator_->Deallocate(ids);
    // allocator_->Deallocate(dists);
    // std::cout << std::endl;
    return results;
}

DatasetPtr
HybridIndex::SearchWithRequest(const SearchRequest& request) const {
    // // support 1 query only
    // auto heap = vsag::DistanceHeap::MakeInstanceBySize<false, true>(this->allocator_, request.topk_);
    //
    // auto dense_query = request.query_->GetFloat32Vectors();
    // auto sparse_query = request.query_->GetSparseVectors();
    //
    // for (int i = 0; i < this->total_count_; i++) {
    //     float dis = 0;
    //     // dense ip distance
    //     float dense_dis = 0;
    //     for (int j = 0; j < this->dim_; j++) {
    //         dense_dis += dense_query[j] * dense_vector_[i * this->dim_ + j];
    //     }
    //
    //     // sparse ip distance
    //     float sparse_dis = 0;
    //     auto cur_sparse_vector = sparse_vector_[i];
    //     auto num_base_len = cur_sparse_vector.len_;
    //     auto num_query_len = sparse_query[0].len_;
    //     int j = 0, k = 0;
    //     while (j < num_base_len && k < num_query_len) {
    //         if (cur_sparse_vector.ids_[j] < sparse_query[0].ids_[k]) j++;
    //         else if (cur_sparse_vector.ids_[j] > sparse_query[0].ids_[k]) k++;
    //         else {
    //             sparse_dis += cur_sparse_vector.vals_[j] * sparse_query[0].vals_[k];
    //             k++;
    //             j++;
    //         }
    //     }
    //
    //     JsonType params = vsag::JsonType::Parse(request.params_str_);
    //     auto search_alpha = params["alpha"].GetFloat();
    //     dis = search_alpha * dense_dis + (1 - search_alpha) * sparse_dis;
    //     heap->Push(dis, i);
    // }
    //
    //
    // auto [results, dists, ids] = create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_); // structed binding
    // for (int j = 0; j < request.topk_; j++) {
    //     dists[request.topk_ - j - 1] = heap->Top().first;
    //     ids[request.topk_ - j - 1] = heap->Top().second;
    //     heap->Pop();
    // }
    // return results;
    auto data = Dataset::Make();
    return data;
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
    label_table_->Serialize(writer);
    hybrid_codes_->Serialize(writer);
    graph_->Serialize(writer);

    StreamWriter::WriteObj(writer, entry_point_id_);
    StreamWriter::WriteObj(writer, ef_construction_);
    StreamWriter::WriteObj(writer, max_degree_);
    StreamWriter::WriteObj(writer, total_count_);
    StreamWriter::WriteObj(writer, alpha_);
}

void
HybridIndex::Deserialize(StreamReader& reader) {
    std::cout << "begin label" << std::endl;
    label_table_->Deserialize(reader);
    std::cout << "end label" << std::endl;
    hybrid_codes_->Deserialize(reader);
    graph_->Deserialize(reader);

    StreamReader::ReadObj(reader, entry_point_id_);
    StreamReader::ReadObj(reader, ef_construction_);
    StreamReader::ReadObj(reader, max_degree_);
    StreamReader::ReadObj(reader, total_count_);
    StreamReader::ReadObj(reader, alpha_);

}

}  // namespace vsag
