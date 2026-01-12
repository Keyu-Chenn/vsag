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

namespace vsag {

class SafeThreadPool;

DEFINE_POINTER2(AttrInvertedInterface, AttributeInvertedInterface);
DEFINE_POINTER(FlattenInterface);
class HybridIndex : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit HybridIndex(const HybridIndexParameterPtr& param, const IndexCommonParam& common_param);
    explicit HybridIndex(const ParamPtr& param, const IndexCommonParam& common_param)
        : HybridIndex(std::dynamic_pointer_cast<HybridIndexParameter>(param), common_param){};

    ~HybridIndex() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& base) override;

    std::vector<int64_t>
    Build(const DatasetPtr& base) override;

    float
    CalcDistanceById(const float* vector, int64_t id) const override;

    void
    Deserialize(StreamReader& reader) override;

    uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<HybridIndex>(this->create_param_ptr_, param);
    }

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

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    bool
    Remove(int64_t label) override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Train(const DatasetPtr& data) override;

    void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override;

    void
    UpdateAttribute(int64_t id,
                    const AttributeSet& new_attrs,
                    const AttributeSet& origin_attrs) override;

private:
    void
    resize(uint64_t new_size);

    void
    add_one(const float* data, InnerIdType inner_id);

private:
    FlattenInterfacePtr inner_codes_{nullptr};

    uint64_t total_count_{0};

    uint64_t resize_increase_count_bit_{DEFAULT_RESIZE_BIT};

    mutable std::shared_mutex global_mutex_;
    mutable std::shared_mutex add_mutex_;

    std::atomic<InnerIdType> max_capacity_{0};

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;

};
}
