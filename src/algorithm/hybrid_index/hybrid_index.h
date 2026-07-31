//
// Created by root on 2026/1/8.
//

#pragma once

#include "algorithm/inner_index_interface.h"
#include "algorithm/sindi/sindi.h"
#include "algorithm/sparse_index.h"
#include "datacell/graph_interface.h"
#include "datacell/hybrid_vector_datacell.h"
#include "hybrid_index_parameter.h"
#include "impl/label_table.h"
#include "impl/searcher/basic_searcher.h"
#include "index_common_param.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "utils/visited_list.h"
#include "vsag/dataset.h"
#include "vsag/filter.h"


namespace vsag {

class DenseEntryHNSWGraph;

class HybridIndex : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);
public:
    explicit HybridIndex(const HybridIndexParameterPtr& param, const IndexCommonParam& common_param);
    explicit HybridIndex(const ParamPtr& param, const IndexCommonParam& common_param)
        : HybridIndex(std::dynamic_pointer_cast<HybridIndexParameter>(param), common_param){};


    void
    add_one_point(InnerIdType inner_id, const float* vector);

    std::vector<int64_t>
    Add(const DatasetPtr& data) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    float
    CalcDistanceById(const float* vector, int64_t id) const override;

    void
    GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const override;

    [[nodiscard]] IndexType
    GetIndexType() const override {
        return IndexType::HYBRID;
    }

    std::string
    GetName() const override {
        return INDEX_HYBRID;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->total_count_;
    }

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    void
    Train(const DatasetPtr& data) override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    void
    Serialize(StreamWriter& writer) const override;

private:
    enum class SearchMethod { UHG, UHGS, UHGH };

    void
    load_precomputed_neighbors();

    void
    load_sindi_index();

    void
    load_dense_entry_hnsw_graph();

    void
    add_hybrid_graph_point(InnerIdType inner_id, const float* vector);

    void
    add_dense_entry_point(InnerIdType inner_id, const float* dense_vector);

    void
    reset_graph_visited_pool();

    [[nodiscard]] SearchMethod
    resolve_search_method(const JsonType& parsed_search_param, float alpha) const;

    [[nodiscard]] std::vector<InnerIdType>
    get_sindi_entry_points(const DatasetPtr& query,
                           const JsonType& parsed_search_param,
                           int64_t k,
                           const FilterPtr& filter) const;

    [[nodiscard]] std::vector<InnerIdType>
    get_dense_entry_points(const DatasetPtr& query,
                           const JsonType& parsed_search_param,
                           int64_t k,
                           Statistics& stats) const;

    [[nodiscard]] DatasetPtr
    graph_knn_search(const DatasetPtr& query,
                     int64_t k,
                     const JsonType& parsed_search_param,
                     float alpha,
                     const std::vector<InnerIdType>& entry_points) const;

    // Vector<float> dense_vector_;
    // Vector<SparseVector> sparse_vector_;
    HybridVectorDataCellPtr hybrid_codes_{nullptr};
    IndexCommonParam common_param_;

    GraphInterfacePtr graph_{nullptr};
    GraphInterfacePtr dense_entry_graph_{nullptr};
    std::shared_ptr<DenseEntryHNSWGraph> dense_entry_hnsw_graph_{nullptr};

    BasicSearcherPtr searcher_{nullptr};
    BasicSearcherPtr dense_entry_searcher_{nullptr};
    std::shared_ptr<VisitedListPool> graph_visited_pool_{nullptr};
    InnerIdType entry_point_id_{std::numeric_limits<InnerIdType>::max()};
    InnerIdType dense_entry_point_id_{std::numeric_limits<InnerIdType>::max()};
    uint64_t ef_construction_{20};
    int64_t max_degree_{10};
    uint64_t dense_entry_ef_construction_{20};
    int64_t dense_entry_max_degree_{10};

    uint64_t total_count_{0};
    float alpha_{0.5};
    bool enable_sindi_{true};
    bool enable_dense_entry_{true};
    uint64_t default_sindi_bk_{0};
    uint64_t default_dense_entry_bk_{0};
    uint64_t default_dense_entry_ef_search_{0};
    float auto_sparse_alpha_max_{0.5F};
    float auto_dense_alpha_min_{0.5F};
    std::string search_method_{"auto"};
    std::string graph_path_;
    std::string sindi_index_path_;
    std::string dense_entry_hnsw_graph_path_;
    SINDIParameterPtr sindi_parameter_{nullptr};
    std::shared_ptr<SINDI> sindi_index_{nullptr};
    bool sindi_loaded_from_path_{false};
    MutexArrayPtr graph_mutex_{nullptr};
    MutexArrayPtr dense_entry_graph_mutex_{nullptr};

    std::vector<int64_t> precomputed_neighbors_;
    std::vector<int64_t> precomputed_neighbor_counts_;
    int64_t precomputed_num_points_{0};
    int64_t precomputed_max_neighbors_{0};
    bool precomputed_neighbors_loaded_{false};

};
}
