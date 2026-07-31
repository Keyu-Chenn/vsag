//
// Created by root on 2026/1/8.
//

#include "hybrid_index_parameter.h"
#include <algorithm>
#include <cctype>
#include <fmt/format.h>
#include "datacell/flatten_datacell_parameter.h"
#include "impl/logger/logger.h"
#include "inner_string_params.h"
#include "vsag/constants.h"

namespace vsag {
void
HybridIndexParameter::FromJson(const JsonType& json) {
    sparse_json = json;
    if (json.Contains("alpha")) {
        alpha = json["alpha"].GetFloat();
        CHECK_ARGUMENT((0.0F <= alpha and alpha <= 1.0F),
                       fmt::format("alpha must in [0, 1], got {}", alpha));
    }
    if (json.Contains("ef_construction")) {
        ef_construction = json["ef_construction"].GetInt();
    }
    if (json.Contains("max_degree")) {
        max_degree = json["max_degree"].GetInt();
    }
    if (json.Contains("graph_path")) {
        graph_path = json["graph_path"].GetString();
    }
    if (json.Contains("sindi_index_path")) {
        sindi_index_path = json["sindi_index_path"].GetString();
    }
    if (json.Contains("dense_entry_hnsw_graph_path")) {
        dense_entry_hnsw_graph_path = json["dense_entry_hnsw_graph_path"].GetString();
    }
    if (json.Contains("dense_entry_hnsw_index_path")) {
        dense_entry_hnsw_graph_path = json["dense_entry_hnsw_index_path"].GetString();
    }
    if (json.Contains("hnsw_index_path")) {
        dense_entry_hnsw_graph_path = json["hnsw_index_path"].GetString();
    }
    if (json.Contains("search_method")) {
        search_method = json["search_method"].GetString();
    }
    if (json.Contains("method")) {
        search_method = json["method"].GetString();
    }
    std::transform(search_method.begin(), search_method.end(), search_method.begin(), ::tolower);
    CHECK_ARGUMENT((search_method == "auto" or search_method == "uhg" or search_method == "uhgs" or
                    search_method == "uhgh"),
                   fmt::format("search_method must be one of uhg/uhgs/uhgh/auto, got {}",
                               search_method));

    if (json.Contains("enable_sindi")) {
        enable_sindi = json["enable_sindi"].GetBool();
    }
    if (json.Contains("enable_dense_entry")) {
        enable_dense_entry = json["enable_dense_entry"].GetBool();
    }
    if (json.Contains("dense_entry_ef_construction")) {
        dense_entry_ef_construction = json["dense_entry_ef_construction"].GetInt();
    } else {
        dense_entry_ef_construction = ef_construction;
    }
    if (json.Contains("dense_entry_max_degree")) {
        dense_entry_max_degree = json["dense_entry_max_degree"].GetInt();
    } else {
        dense_entry_max_degree = max_degree;
    }
    if (json.Contains("default_sindi_bk")) {
        default_sindi_bk = json["default_sindi_bk"].GetInt();
    }
    if (json.Contains("default_dense_entry_bk")) {
        default_dense_entry_bk = json["default_dense_entry_bk"].GetInt();
    }
    if (json.Contains("default_dense_entry_ef_search")) {
        default_dense_entry_ef_search = json["default_dense_entry_ef_search"].GetInt();
    }
    if (json.Contains("sindi")) {
        sindi_json = json["sindi"];
    }
    if (json.Contains("auto")) {
        auto auto_json = json["auto"];
        if (auto_json.Contains("sparse_alpha_max")) {
            auto_sparse_alpha_max = auto_json["sparse_alpha_max"].GetFloat();
        }
        if (auto_json.Contains("dense_alpha_min")) {
            auto_dense_alpha_min = auto_json["dense_alpha_min"].GetFloat();
        }
    }
    CHECK_ARGUMENT((0.0F <= auto_sparse_alpha_max and auto_sparse_alpha_max <= 1.0F),
                   fmt::format("auto.sparse_alpha_max must in [0, 1], got {}",
                               auto_sparse_alpha_max));
    CHECK_ARGUMENT((0.0F <= auto_dense_alpha_min and auto_dense_alpha_min <= 1.0F),
                   fmt::format("auto.dense_alpha_min must in [0, 1], got {}",
                               auto_dense_alpha_min));
    CHECK_ARGUMENT(max_degree > 0, fmt::format("max_degree must be positive, got {}", max_degree));
    CHECK_ARGUMENT(dense_entry_max_degree > 0,
                   fmt::format("dense_entry_max_degree must be positive, got {}",
                               dense_entry_max_degree));
}

JsonType
HybridIndexParameter::ToJson() const {
    JsonType json;
    json["alpha"].SetFloat(alpha);
    json["ef_construction"].SetInt(ef_construction);
    json["max_degree"].SetInt(max_degree);
    json["graph_path"].SetString(graph_path);
    json["sindi_index_path"].SetString(sindi_index_path);
    json["dense_entry_hnsw_graph_path"].SetString(dense_entry_hnsw_graph_path);
    json["search_method"].SetString(search_method);
    json["enable_sindi"].SetBool(enable_sindi);
    json["enable_dense_entry"].SetBool(enable_dense_entry);
    json["dense_entry_ef_construction"].SetInt(dense_entry_ef_construction);
    json["dense_entry_max_degree"].SetInt(dense_entry_max_degree);
    json["default_sindi_bk"].SetInt(default_sindi_bk);
    json["default_dense_entry_bk"].SetInt(default_dense_entry_bk);
    json["default_dense_entry_ef_search"].SetInt(default_dense_entry_ef_search);
    JsonType auto_json;
    auto_json["sparse_alpha_max"].SetFloat(auto_sparse_alpha_max);
    auto_json["dense_alpha_min"].SetFloat(auto_dense_alpha_min);
    json["auto"].SetJson(auto_json);
    json["sindi"].SetJson(sindi_json);
    return json;
}
}  // namespace vsag
