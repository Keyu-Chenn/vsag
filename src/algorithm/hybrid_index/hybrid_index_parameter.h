//
// Created by root on 2026/1/8.
//

#pragma once

#include "algorithm/inner_index_parameter.h"
#include "algorithm/index_search_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(FlattenDataCellParam, FlattenDataCellParameter);
DEFINE_POINTER(HybridIndexParameter);

class HybridIndexParameter : public InnerIndexParameter {
public:
    explicit HybridIndexParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

public:
    FlattenInterfaceParamPtr base_codes_param{nullptr};
};

class HybridIndexSearchParameter : public IndexSearchParameter {
public:
    static HybridIndexSearchParameter
    FromJson(const std::string& json_string) {
        if (json_string.empty()) {
            return HybridIndexSearchParameter();
        }
        auto params = JsonType::Parse(json_string);
        HybridIndexSearchParameter obj;
        obj.IndexSearchParameter::FromJson(params);
        return obj;
    }
public:
    float alpha{0.5};
};
}

