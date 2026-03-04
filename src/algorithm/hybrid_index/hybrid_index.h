//
// Created by root on 2026/1/8.
//

#pragma once

#include "algorithm/inner_index_interface.h"
#include "algorithm/sparse_index.h"
#include "datacell/graph_interface.h"
#include "datacell/hybrid_vector_datacell.h"
#include "hybrid_index_parameter.h"
#include "impl/label_table.h"
#include "impl/searcher/basic_searcher.h"
#include "index_common_param.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/dataset.h"
#include "vsag/filter.h"


namespace vsag {

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


    // Vector<float> dense_vector_;
    // Vector<SparseVector> sparse_vector_;
    HybridVectorDataCellPtr hybrid_codes_{nullptr};

    GraphInterfacePtr graph_{nullptr};

    BasicSearcherPtr searcher_{nullptr};
    InnerIdType entry_point_id_{std::numeric_limits<InnerIdType>::max()};
    uint64_t ef_construction_{20};
    int64_t max_degree_{10};

    uint64_t total_count_{0};
    float alpha_{0.5};

};
}