// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#include "hnsw.h"
#include "ivf.h"
#include "public_api.h"
#include <filesystem>
#include <iostream>
#include <unistd.h>

using namespace paper_baselines;

template <class F> void Throws(F function) {
    bool threw = false;
    try { function(); } catch (const std::exception&) { threw = true; }
    Require(threw, "expected validation error");
}

Vectors Fixture() {
    Vectors v; v.dim = 24;
    std::mt19937 gen(1201); std::uniform_real_distribution<float> random(-1, 1);
    for (Id i = 0; i < 320; ++i) {
        std::vector<float> row(v.dim); for (auto& x : row) x = random(gen);
        float norm = std::sqrt(Dot(row.data(), row.data(), row.size()));
        for (float x : row) v.dense.push_back(x / norm);
        Sparse sparse;
        for (Id term = 0; term < 40; ++term) if ((gen() % 3) == 0) {
            sparse.Push(term, random(gen));
        }
        v.sparse.push_back(sparse); v.labels.push_back(10000 + 7 * i);
    }
    v.Validate(); return v;
}

// Independent double-precision exhaustive oracle, including signed/nonunit IP.
std::vector<Hit> Exact(const Vectors& base, const float* qd, const Sparse& qs, float alpha, size_t k) {
    std::vector<Hit> result;
    for (Id i = 0; i < base.size(); ++i) {
        double d = 0, s = 0;
        for (size_t j = 0; j < base.dim; ++j) d += double(qd[j]) * base.at(i)[j];
        for (const auto& query_entry : qs.entries)
            for (const auto& base_entry : base.sparse[i].entries)
                if (query_entry.id == base_entry.id)
                    s += double(query_entry.value) * base_entry.value;
        result.emplace_back(1 - (alpha * d + (1 - alpha) * s), i);
    }
    std::sort(result.begin(), result.end()); result.resize(k); return result;
}

std::vector<Hit> ExactPruned(const Vectors& base, const float* qd, const Sparse& qs,
                             float alpha, size_t k, float base_keep, float query_keep) {
    auto clipped_query = qs.Keep(query_keep);
    std::vector<Hit> result;
    for (Id i = 0; i < base.size(); ++i) {
        auto clipped_base = base.sparse[i].Keep(base_keep);
        result.emplace_back(1 - (alpha * Dot(qd, base.at(i), base.dim) +
                                 (1 - alpha) * SparseDot(clipped_query, clipped_base)), i);
    }
    std::sort(result.begin(), result.end()); result.resize(k); return result;
}

void Same(const std::vector<Hit>& a, const std::vector<Hit>& b) {
    Require(a.size() == b.size(), "result count differs");
    for (size_t i = 0; i < a.size(); ++i) {
        Require(a[i].second == b[i].second,
                "ranked IDs differ at rank " + std::to_string(i) + ": " +
                std::to_string(a[i].second) + " vs " + std::to_string(b[i].second));
        Require(std::abs(a[i].first - b[i].first) < 2e-5,
                "distance differs from exact raw hybrid score at rank " + std::to_string(i) +
                ": " + std::to_string(a[i].first) + " vs " + std::to_string(b[i].first));
    }
}

void SamePublic(const vsag::DatasetPtr& actual, const std::vector<Hit>& expected,
                const Vectors& base) {
    Require(actual != nullptr && actual->GetDim() == static_cast<int64_t>(expected.size()),
            "public result shape differs");
    for (size_t i = 0; i < expected.size(); ++i) {
        Require(actual->GetIds()[i] == base.labels[expected[i].second],
                "public result label differs at rank " + std::to_string(i));
        Require(std::abs(actual->GetDistances()[i] - expected[i].first) < 2e-5,
                "public result distance differs at rank " + std::to_string(i));
    }
}

int main() {
    try {
        openblas_set_num_threads(1);
        auto base = Fixture();
        auto path = (std::filesystem::temp_directory_path() / ("uhg_paper_test_" + std::to_string(getpid()))).string();
        HybridHNSW::Config c; c.m = 12; c.ef_construction = 100;
        HybridHNSW naive(base, c); naive.Build();
        Require(naive.max_level() > 0 && naive.edges() > base.size(), "missing multilevel HNSW topology");
        HybridHNSW::Workspace work(base.size());
        HybridHNSW::SearchConfig search; search.ef = base.size();

        std::vector<Id> public_ids;
        std::vector<float> public_values;
        for (const auto& entry : base.sparse[19].entries) {
            public_ids.push_back(entry.id);
            public_values.push_back(entry.value);
        }
        vsag::SparseVector public_sparse;
        public_sparse.len_ = static_cast<uint32_t>(public_ids.size());
        public_sparse.ids_ = public_ids.data();
        public_sparse.vals_ = public_values.data();
        auto public_query = vsag::Dataset::Make();
        public_query->NumElements(1)
            ->Dim(static_cast<int64_t>(base.dim))
            ->Float32Vectors(base.at(37))
            ->SparseVectors(&public_sparse)
            ->Owner(false);
        PublicHybridHNSW public_naive(naive, base);
        SamePublic(public_naive.KnnSearch(public_query, 50,
                                          R"({"alpha":0.5,"ef_search":320,"tau_dense":0.8,"tau_hybrid":1.0})"),
                   Exact(base, base.at(37), base.sparse[19], .5, 50), base);
        Throws([&] { public_naive.KnnSearch(public_query, 50, R"({"alpha":0.5})"); });
        for (float alpha : {0.f, .3f, .5f, .7f, 1.f})
            for (size_t k : {size_t(1), size_t(50), size_t(100), size_t(200)}) {
                Stats stats;
                auto actual = naive.Search(base.at(37), base.sparse[19], alpha, k, search, work, stats);
                Same(actual, Exact(base, base.at(37), base.sparse[19], alpha, k));
            }
        naive.Save(path + ".hnsw");
        auto restored = HybridHNSW::Load(base, path + ".hnsw");
        Stats a, b;
        search.ef = 64;
        Same(naive.Search(base.at(8), base.sparse[11], .3, 50, search, work, a),
             restored.Search(base.at(8), base.sparse[11], .3, 50, search, work, b));
        Require(restored.edges() == naive.edges(), "flat L0 changed serialized edge count");
        restored.Save(path + ".roundtrip");
        auto roundtrip = HybridHNSW::Load(base, path + ".roundtrip");
        Same(restored.Search(base.at(8), base.sparse[11], .3, 50, search, work, a),
             roundtrip.Search(base.at(8), base.sparse[11], .3, 50, search, work, b));
        auto changed = base; changed.dense[0] += .001f;
        Throws([&] { HybridHNSW::Load(changed, path + ".hnsw"); });
        Throws([&] { naive.Search(base.at(0), base.sparse[0], .5, 100, search, work, a); });
        Throws([&] { naive.Search(base.at(0), base.sparse[0], 1.1, 10, search, work, a); });

        c.build_threads = 4;
        c.parallel_warmup = 32;
        HybridHNSW parallel(base, c); parallel.Build();
        Require(parallel.max_level() == naive.max_level() && parallel.edges() > base.size(),
                "parallel HNSW topology is incomplete");
        search.ef = base.size();
        Same(parallel.Search(base.at(37), base.sparse[19], .5, 200, search, work, a),
             Exact(base, base.at(37), base.sparse[19], .5, 200));
        parallel.Save(path + ".parallel");
        auto parallel_restored = HybridHNSW::Load(base, path + ".parallel");
        Require(parallel_restored.config().build_threads == 4 &&
                parallel_restored.config().parallel_warmup == 32,
                "parallel build provenance was not serialized");

        c.build_threads = 1;
        c.parallel_warmup = 10000;
        c.alpha_build = 1;
        HybridHNSW dense(base, c); dense.Build();
        c.alpha_build = .5; c.optimized = true;
        HybridHNSW optimized(base, c); optimized.Build();
        bool l0_changed = false;
        for (Id i = 0; i < base.size(); ++i) {
            Require(dense.graph()[i].size() == optimized.graph()[i].size(), "refinement changed layer assignment");
            for (size_t l = 1; l < dense.graph()[i].size(); ++l)
                Require(dense.graph()[i][l] == optimized.graph()[i][l], "refinement changed dense upper graph");
            l0_changed |= dense.graph()[i][0] != optimized.graph()[i][0];
            Require(optimized.graph()[i][0].size() <= c.m, "refined L0 degree exceeds M");
        }
        Require(l0_changed, "refinement did not change any L0 edges");
        Stats two_stage;
        search.ef = base.size(); search.tau_dense = .8; search.tau_hybrid = 1;
        Same(optimized.Search(base.at(37), base.sparse[19], .3, 200, search, work, two_stage),
             Exact(base, base.at(37), base.sparse[19], .3, 200));
        Require(two_stage.dense_expansions > 0 && two_stage.hybrid_expansions > 0 && two_stage.transition > 0,
                "optimized search omitted a phase or queue conversion");
        optimized.Save(path + ".optimized");
        auto opt_restored = HybridHNSW::Load(base, path + ".optimized");
        search.ef = 100;
        for (float tau : {0.f, .6f, 1.f}) {
            search.tau_dense = tau; search.tau_hybrid = tau;
            Same(optimized.Search(base.at(10), base.sparse[20], .7, 50, search, work, a),
                 opt_restored.Search(base.at(10), base.sparse[20], .7, 50, search, work, b));
        }
        c.build_threads = 4; c.parallel_warmup = 32;
        HybridHNSW parallel_optimized(base, c); parallel_optimized.Build();
        search.ef = base.size(); search.tau_dense = .8; search.tau_hybrid = 1;
        Same(parallel_optimized.Search(base.at(37), base.sparse[19], .3, 200,
                                       search, work, a),
             Exact(base, base.at(37), base.sparse[19], .3, 200));
        c.build_threads = 1; c.parallel_warmup = 10000;
        HybridHNSW pruned(base, c); pruned.Build();
        const auto topology_before_search_pruning = pruned.graph();
        pruned.ConfigureSearchSparsePruning(.6);
        Require(pruned.graph() == topology_before_search_pruning,
                "search sparse pruning changed the constructed graph");
        Require(std::abs(pruned.search_base_sparse_keep() - .6F) < 1e-6F &&
                std::abs(pruned.search_query_sparse_keep() - .6F) < 1e-6F,
                "search sparse retention was not configured");
        auto output = pruned.Search(base.at(2), base.sparse[7], .5, 50, search, work, a);
        Require(output.size() == 50, "pruned result count");
        Same(output, ExactPruned(base, base.at(2), base.sparse[7], .5, 50, .6F, .6F));
        pruned.Save(path + ".search_pruned");
        auto search_pruned_restored = HybridHNSW::Load(base, path + ".search_pruned");
        Require(search_pruned_restored.search_base_sparse_keep() == 1 &&
                search_pruned_restored.search_query_sparse_keep() == 1,
                "search-only sparse retention leaked into the serialized construction index");
        pruned.ConfigureSearchSparsePruning(.6, 1);
        Require(pruned.search_base_sparse_keep() == .6F &&
                pruned.search_query_sparse_keep() == 1.0F,
                "independent base/query sparse retention was not configured");
        output = pruned.Search(base.at(2), base.sparse[7], .5, 50, search, work, a);
        Same(output, ExactPruned(base, base.at(2), base.sparse[7], .5, 50, .6F, 1.0F));
        Throws([&] { naive.ConfigureSearchSparsePruning(.6); });
        c.construction_sparse_keep = .6F;
        HybridHNSW construction_pruned(base, c); construction_pruned.Build();
        construction_pruned.Save(path + ".construction_pruned");
        auto construction_pruned_restored = HybridHNSW::Load(base, path + ".construction_pruned");
        Require(std::abs(construction_pruned_restored.config().construction_sparse_keep - .6F) < 1e-6F,
                "construction sparse retention was not serialized");
        search.ef = base.size();
        construction_pruned_restored.ConfigureSearchSparsePruning(.6F, 1.0F);
        output = construction_pruned_restored.Search(base.at(2), base.sparse[7], .5, 50,
                                                     search, work, a);
        Same(output, ExactPruned(base, base.at(2), base.sparse[7], .5, 50, .6F, 1.0F));
        Sparse signed_values{{1, 2, 3, 4}, {-.9f, .1f, .8f, -.2f}};
        auto keep = signed_values.Keep(.5);
        Require(keep.entries.size() == 2 && keep.entries[0].id == 1 && keep.entries[1].id == 3,
                "small-magnitude pruning/sorted IDs");

        UnifiedIVF::Config ic; ic.clusters = 16; ic.sketch_dim = 128; ic.iterations = 8;
        UnifiedIVF ivf(base, ic); ivf.Build();
        for (float alpha : {0.f, .3f, .5f, .7f, 1.f}) {
            Stats stats;
            Same(ivf.Search(base.at(37), base.sparse[19], alpha, 200, base.size(), stats),
                 Exact(base, base.at(37), base.sparse[19], alpha, 200));
            Require(stats.scanned == base.size(), "all-list IVF did not scan every document");
        }
        Stats partial;
        ivf.Search(base.at(3), base.sparse[7], .5, 10, 30, partial);
        Require(partial.scanned >= 30 && partial.scanned < base.size(), "IVF did not honor list budget");
        ivf.Save(path + ".ivf"); auto loaded = UnifiedIVF::Load(base, path + ".ivf");
        Same(ivf.Search(base.at(8), base.sparse[9], .7, 50, 80, a),
             loaded.Search(base.at(8), base.sparse[9], .7, 50, 80, b));
        Throws([&] { UnifiedIVF::Load(changed, path + ".ivf"); });
        // Query-only terms must not silently disappear from JL.
        RademacherJL jl(128, 42); float projection[128];
        jl.Project(Sparse{{999999}, {2}}, projection);
        Require(std::abs(Dot(projection, projection, 128) - 4) < 1e-4, "JL term column norm");
        std::ofstream truncated(path + ".bad", std::ios::binary); truncated << "bad"; truncated.close();
        Throws([&] { HybridHNSW::Load(base, path + ".bad"); });
        for (auto suffix : {".hnsw", ".roundtrip", ".parallel", ".optimized",
                            ".search_pruned", ".construction_pruned", ".ivf", ".bad"})
            std::filesystem::remove(path + suffix);
        std::cout << "PASS: exact weighted-IP oracle; serial/parallel multilevel HNSW; dense upper-layer preservation; two-phase queues/tau; paper-aligned pruning without rerank; IVF full/partial scan; JL unseen terms; serialization/input validation\n";
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
    return 0;
}
