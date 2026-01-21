//
// Created by root on 2026/1/8.
//

#pragma once

#include "algorithm/inner_index_interface.h"
#include "hybrid_index_parameter.h"
#include "impl/label_table.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/filter.h"
#include "algorithm/sparse_index.h"
#include "vsag/dataset.h"
#include "index_common_param.h"

namespace vsag {
/** In dataset.h, define HybridVector like this:
struct HybridVector {
    float* dense_vector_;
    int64_t dim_;
    SparseVector sparse_vector_;
    HybridVector() : dense_vector_{nullptr}, dim_{0}, sparse_vector_() {
    }
};
**/

class HybridIndex : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);
public:
    explicit HybridIndex(const HybridIndexParameterPtr& param, const IndexCommonParam& common_param);
    explicit HybridIndex(const ParamPtr& param, const IndexCommonParam& common_param)
        : HybridIndex(std::dynamic_pointer_cast<HybridIndexParameter>(param), common_param){};

    ~HybridIndex() override;

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
    Vector<float> dense_vector_;
    Vector<SparseVector> sparse_vector_;
    uint64_t total_count_{0};
    float alpha_;

};
}
