//
// Created by root on 2026/1/8.
//

#pragma once

#include "algorithm/inner_index_parameter.h"
#include "algorithm/index_search_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(HybridIndexParameter);

class HybridIndexParameter : public InnerIndexParameter {
public:
    HybridIndexParameter() = default;
    void FromJson(const JsonType& json) override;
    JsonType ToJson() const override;

public:
    JsonType sparse_json;
    float alpha{0.5};
};

class HybridIndexSearchParameter : public IndexSearchParameter {
public:

};


}

