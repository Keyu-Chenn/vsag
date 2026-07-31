//
// Created by root on 2026/1/8.
//

#pragma once

#include "algorithm/inner_index_parameter.h"
#include "algorithm/index_search_parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

#include <string>

namespace vsag {
DEFINE_POINTER(HybridIndexParameter);

class HybridIndexParameter : public InnerIndexParameter {
public:
    HybridIndexParameter() = default;
    void FromJson(const JsonType& json) override;
    JsonType ToJson() const override;

public:
    JsonType sparse_json;
    JsonType sindi_json;
    float alpha{0.5};
    uint64_t ef_construction{20};
    int64_t max_degree{10};
    std::string graph_path;
    std::string sindi_index_path;
    std::string dense_entry_hnsw_graph_path;
    std::string search_method{"auto"};
    bool enable_sindi{true};
    bool enable_dense_entry{true};
    uint64_t dense_entry_ef_construction{20};
    int64_t dense_entry_max_degree{10};
    uint64_t default_sindi_bk{0};
    uint64_t default_dense_entry_bk{0};
    uint64_t default_dense_entry_ef_search{0};
    float auto_sparse_alpha_max{0.5F};
    float auto_dense_alpha_min{0.5F};

};

class HybridIndexSearchParameter : public IndexSearchParameter {
public:

};


}
