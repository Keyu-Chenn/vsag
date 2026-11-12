// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "hnsw.h"

#include <nlohmann/json.hpp>
#include <vsag/dataset.h>

namespace paper_baselines {

// Public-API-level adapter for the added HNSW baselines. It deliberately
// mirrors the existing HybridIndex benchmark boundary: one Dataset query and
// one JSON parameter string enter KnnSearch, and an owning Dataset containing
// final top-k labels/distances is returned.
class PublicHybridHNSW {
 public:
    PublicHybridHNSW(const HybridHNSW& index, const Vectors& base)
        : index_(index), base_(base), workspace_(base.size()) {
        Require(index.size() == base.size(), "public adapter/index size mismatch");
    }

    vsag::DatasetPtr KnnSearch(const vsag::DatasetPtr& query,
                               int64_t k,
                               const std::string& parameters) {
        Require(query != nullptr, "query Dataset must not be null");
        Require(query->GetNumElements() == 1, "public adapter accepts one query");
        Require(query->GetDim() == static_cast<int64_t>(base_.dim), "query dimension mismatch");
        Require(k > 0, "top-k must be positive");
        const auto* dense = query->GetFloat32Vectors();
        const auto* sparse = query->GetSparseVectors();
        Require(dense != nullptr && sparse != nullptr, "hybrid query is incomplete");

        auto parsed = nlohmann::json::parse(parameters);
        Require(parsed.is_object(), "search parameters must be a JSON object");
        const float alpha = RequiredNumber<float>(parsed, "alpha");
        HybridHNSW::SearchConfig config;
        config.ef = RequiredNumber<size_t>(parsed, "ef_search");
        config.tau_dense = OptionalNumber<float>(parsed, "tau_dense", 0.8F);
        config.tau_hybrid = OptionalNumber<float>(parsed, "tau_hybrid", 1.0F);

        Stats query_stats;
        SparseView sparse_view(sparse[0].ids_, sparse[0].vals_, sparse[0].len_);
        auto hits = index_.Search(dense, sparse_view, alpha, static_cast<size_t>(k),
                                  config, workspace_, query_stats);
        stats_ += query_stats;

        auto result = vsag::Dataset::Make();
        auto* ids = new int64_t[hits.size()];
        auto* distances = new float[hits.size()];
        for (size_t i = 0; i < hits.size(); ++i) {
            Require(hits[i].second < base_.size(), "search returned an invalid inner ID");
            Require(std::isfinite(hits[i].first), "search returned a nonfinite distance");
            ids[i] = base_.labels[hits[i].second];
            distances[i] = hits[i].first;
        }
        result->NumElements(1)
            ->Dim(static_cast<int64_t>(hits.size()))
            ->Ids(ids)
            ->Distances(distances)
            ->Owner(true);
        result->Statistics(nlohmann::json({{"dense", query_stats.dense},
                                           {"sparse", query_stats.sparse},
                                           {"dense_expansions", query_stats.dense_expansions},
                                           {"hybrid_expansions", query_stats.hybrid_expansions},
                                           {"transition", query_stats.transition}}).dump());
        return result;
    }

    Stats TakeStats() {
        Stats output = stats_;
        stats_ = {};
        return output;
    }

 private:
    template <typename T>
    static T RequiredNumber(const nlohmann::json& parsed, const char* key) {
        Require(parsed.contains(key) && parsed.at(key).is_number(),
                std::string("missing/non-numeric search parameter: ") + key);
        return parsed.at(key).get<T>();
    }

    template <typename T>
    static T OptionalNumber(const nlohmann::json& parsed, const char* key, T fallback) {
        if (!parsed.contains(key)) return fallback;
        Require(parsed.at(key).is_number(),
                std::string("non-numeric search parameter: ") + key);
        return parsed.at(key).get<T>();
    }

    const HybridHNSW& index_;
    const Vectors& base_;
    HybridHNSW::Workspace workspace_;
    Stats stats_;
};

struct PublicSparseQueries {
    std::vector<std::vector<Id>> ids;
    std::vector<std::vector<float>> values;
    std::vector<vsag::SparseVector> views;

    explicit PublicSparseQueries(const std::vector<Sparse>& source)
        : ids(source.size()), values(source.size()), views(source.size()) {
        for (size_t i = 0; i < source.size(); ++i) {
            ids[i].reserve(source[i].size());
            values[i].reserve(source[i].size());
            for (const auto& entry : source[i].entries) {
                ids[i].push_back(entry.id);
                values[i].push_back(entry.value);
            }
            views[i].len_ = static_cast<uint32_t>(ids[i].size());
            views[i].ids_ = ids[i].data();
            views[i].vals_ = values[i].data();
        }
    }
};

}  // namespace paper_baselines
