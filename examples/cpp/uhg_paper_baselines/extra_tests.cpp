// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#include "ladr.h"
#include "sinnamon.h"
#include <filesystem>
#include <iostream>
#include <set>
#include <unistd.h>

using namespace paper_baselines;

template<class Function> void MustThrow(Function function) {
    bool failed = false;
    try { function(); } catch (const std::exception&) { failed = true; }
    Require(failed, "expected input validation failure");
}

Vectors Data(size_t count = 128) {
    Vectors base; base.dim = 33;
    std::mt19937 random(421);
    std::uniform_real_distribution<float> number(-.2F, .2F);
    for (Id id = 0; id < count; ++id) {
        for (size_t j = 0; j < base.dim; ++j) base.dense.push_back(number(random));
        Sparse sparse;
        for (Id term = 0; term < 80; ++term)
            if (random() % 5 == 0) sparse.Push(term, std::abs(number(random)));
        base.sparse.push_back(sparse); base.labels.push_back(1009 + 13 * id);
    }
    base.Validate(); return base;
}

double DenseOracle(const Vectors& base, const float* query, Id id) {
    double value = 0;
    for (size_t j = 0; j < base.dim; ++j) value += double(query[j]) * base.at(id)[j];
    return value;
}
double SparseOracle(const Sparse& query, const Sparse& doc) {
    double value = 0;
    for (const auto& q : query.entries)
        for (const auto& d : doc.entries)
            if (q.id == d.id) value += double(q.value) * d.value;
    return value;
}
std::vector<Hit> Exact(const Vectors& base, const float* qd, const Sparse& qs, float alpha, size_t k) {
    std::vector<Hit> result;
    for (Id id = 0; id < base.size(); ++id)
        result.emplace_back(1 - (alpha * DenseOracle(base, qd, id) +
                                 (1 - alpha) * SparseOracle(qs, base.sparse[id])), id);
    std::sort(result.begin(), result.end()); result.resize(std::min(k, result.size())); return result;
}
void Equal(const std::vector<Hit>& actual, const std::vector<Hit>& expected) {
    Require(actual.size() == expected.size(), "result size mismatch");
    for (size_t j = 0; j < actual.size(); ++j) {
        Require(actual[j].second == expected[j].second, "ranked result ID mismatch");
        Require(std::abs(actual[j].first - expected[j].first) < 2e-6F, "result score mismatch");
    }
}

// Independent direct translation of the paper's set-based Algorithms 1/2.
// No generation counters, frontier heap or expanded-node optimization.
std::vector<Hit> ReferenceLadr(const Vectors& base, const LadrGraph& graph, const float* qd,
                              const Sparse& qs, float alpha, size_t k, const std::vector<Id>& seeds,
                              size_t depth, size_t neighbor_count, bool adaptive, bool hybrid) {
    std::set<Id> seen(seeds.begin(), seeds.end());
    while (true) {
        std::vector<std::pair<double, Id>> ranked;
        for (Id id : seen) ranked.emplace_back(-DenseOracle(base, qd, id), id);
        std::sort(ranked.begin(), ranked.end());
        std::vector<Id> expand;
        if (!adaptive) expand = seeds;
        else for (size_t j = 0; j < std::min(depth, ranked.size()); ++j) expand.push_back(ranked[j].second);
        std::set<Id> pending;
        for (Id id : expand) {
            auto [neighbors, count] = graph.Row(id, neighbor_count);
            for (size_t j = 0; j < count; ++j)
                if (!seen.count(neighbors[j])) pending.insert(neighbors[j]);
        }
        seen.insert(pending.begin(), pending.end());
        if (!adaptive || pending.empty()) break;
    }
    std::vector<Hit> result;
    for (Id id : seen) {
        const double dense = DenseOracle(base, qd, id);
        const double score = hybrid ? alpha * dense + (1 - alpha) * SparseOracle(qs, base.sparse[id]) : dense;
        result.emplace_back(1 - score, id);
    }
    std::sort(result.begin(), result.end()); result.resize(std::min(k, result.size())); return result;
}

void SketchTests() {
    WeakSinnamon sketch(4, 42);
    Id a = 1, b = 2;
    while (sketch.Bucket(a) != sketch.Bucket(b)) ++b;
    Sparse doc({a, b}, {2, 5}), query({a, b}, {3, 7});
    float d[4], q[4];
    sketch.Document(doc, d); sketch.Query(query, q);
    Require(d[sketch.Bucket(a)] == 5, "documents must take bucket maximum");
    Require(q[sketch.Bucket(a)] == 10, "queries must sum colliding coordinates");
    Require(Dot(d, q, 4) == 50, "asymmetric sketch collision result");
    Require(SparseOracle(query, doc) == 41, "collision oracle");
    sketch.Document(Sparse{}, d); Require(Dot(d, q, 4) == 0, "empty document sketch");
    MustThrow([&] { sketch.Prepare({Sparse({1}, {-1})}); });
    auto base = Data(); sketch.Prepare(base.sparse);
    for (const auto& ds : base.sparse) for (const auto& qs : base.sparse) {
        sketch.Document(ds, d); sketch.Query(qs, q);
        Require(Dot(d, q, 4) + 1e-6 >= SparseOracle(qs, ds), "Weak Sinnamon upper bound violated");
    }
    std::cout << "PASS asymmetric sketch, collisions, nonnegative domain, upper bound\n";
}

void SparseScanTests() {
    std::mt19937 random(815);
    std::uniform_real_distribution<float> value(0, 1);
    for (Id scale : {1U, 4096U, 1U << 20, std::numeric_limits<Id>::max() / 1000}) {
        for (size_t trial = 0; trial < 30; ++trial) {
            Sparse query, document;
            for (Id term = 0; term < 200; ++term) {
                if (random() % 8 == 0) query.Push(term * scale, value(random));
                if (random() % 3 == 0) document.Push(term * scale, value(random));
            }
            SinnamonSparseQuery lookup(query);
            const float actual = lookup.Dot(document), expected = SparseDot(query, document);
            Require(std::memcmp(&actual, &expected, sizeof(float)) == 0,
                    "Sinnamon sparse lookup changed float accumulation");
        }
    }
    for (const Sparse& query : {Sparse{}, Sparse({0, 5, 10}, {0, 1, 0}),
                                Sparse({std::numeric_limits<Id>::max()}, {2})}) {
        SinnamonSparseQuery lookup(query);
        for (const Sparse& document : {Sparse{}, Sparse({0, 3, 5, 10, 20}, {2, 7, 4, 3, 9}),
                                       Sparse({std::numeric_limits<Id>::max()}, {3})}) {
            const float actual = lookup.Dot(document), expected = SparseDot(query, document);
            Require(std::memcmp(&actual, &expected, sizeof(float)) == 0,
                    "Sinnamon sparse lookup edge case mismatch");
        }
    }
    std::cout << "PASS Sinnamon sparse lookup float bits, empty/zero terms, large-ID fallback\n";
}

void IvfTests(const std::string& path) {
    // Exercise multiple 128-row batches and each dense kernel's tail path.
    auto base = Data(521);
    SinnamonIVF::Config config; config.clusters = 3; config.sketch_dim = 16; config.iterations = 5;
    SinnamonIVF index(base, config); index.Build(); index.Save(path);
    auto loaded = SinnamonIVF::Load(base, path);
    Require(loaded.lists() == index.lists(), "IVF serialization changed assignments");
    for (Id qi : {1U, 11U, 29U, 71U}) for (float alpha : {0.0F, .3F, .5F, .7F, 1.0F}) {
        const auto* qd = base.at(qi); const auto& qs = base.sparse[(qi+7)%base.size()];
        Stats stats;
        auto full = index.Search(qd, qs, alpha, 20, base.size(), stats);
        Equal(full, Exact(base, qd, qs, alpha, 20));
        Require(stats.scanned == base.size(), "full scan must examine every document once");
        Stats repeated;
        Equal(full, loaded.Search(qd, qs, alpha, 20, base.size(), repeated));
        Stats limited;
        auto partial = loaded.Search(qd, qs, alpha, 20, 30, limited);
        Require(partial.size() == 20 && limited.scanned >= 30 && limited.scanned <= base.size(),
                "IVF list-boundary document budget");
        for (auto [distance, id] : partial) {
            const auto expected = 1 - (alpha * DenseOracle(base, qd, id) +
                                      (1 - alpha) * SparseOracle(qs, base.sparse[id]));
            Require(std::abs(distance - expected) < 2e-6, "IVF final score must use original vectors");
        }
    }
    base.sparse[0].Push(10000, 1);
    MustThrow([&] { auto wrong = SinnamonIVF::Load(base, path); (void)wrong; });
    std::filesystem::remove(path);
    std::cout << "PASS Sinnamon IVF exact full scan, raw rerank, budget, serialization\n";
}

void LadrTests() {
    Vectors chain; chain.dim = 1; chain.dense = {1, 2, 3, 4, 5};
    chain.labels = {10, 20, 30, 40, 50}; chain.sparse.resize(5);
    chain.sparse[0] = Sparse({1}, {100});
    auto graph = LadrGraph::FromRows({{1}, {2}, {3}, {4}, {3}});
    Ladr index(chain, graph); Ladr::Workspace work; Stats stats;
    float q = 1; Sparse qs({1}, {1});
    auto proactive = index.Search(&q, qs, .5, 2, {0}, 1, 128, false, false, work, stats);
    Require(proactive[0].second == 1 && work.scored.size() == 2, "Proactive must expand only one hop");
    auto adaptive = index.Search(&q, qs, .5, 2, {0}, 1, 128, true, false, work, stats);
    Require(adaptive[0].second == 4 && work.rounds == 4 && work.scored.size() == 5,
            "Adaptive must follow multiple improving hops to convergence");
    auto hybrid = index.Search(&q, qs, .5, 2, {0}, 1, 128, true, true, work, stats);
    Require(hybrid[0].second == 0 && work.scored.size() == 5, "hybrid rerank must not change exploration");
    Require(index.Search(&q, qs, .5, 2, {}, 1, 128, true, true, work, stats).empty(),
            "empty seeds must not be silently padded");
    auto base = Data(80);
    std::vector<std::vector<Id>> rows(base.size());
    for (Id id = 0; id < base.size(); ++id)
        for (Id j = 1; j <= 5; ++j) rows[id].push_back((id + j * 7) % base.size());
    auto random_graph = LadrGraph::FromRows(rows); Ladr random_index(base, random_graph);
    for (Id qi : {0U, 7U, 21U, 39U}) for (size_t depth : {1U, 3U, 10U, 100U})
        for (bool ad : {false, true}) for (bool hr : {false, true}) for (float alpha : {.3F, .7F}) {
            const std::vector<Id> seeds = {3, 9, 20, 3};
            Stats measured;
            Equal(random_index.Search(base.at(qi), base.sparse[qi], alpha, 20, seeds, depth, 5, ad, hr, work, measured),
                  ReferenceLadr(base, random_graph, base.at(qi), base.sparse[qi], alpha, 20, seeds, depth, 5, ad, hr));
            Require(measured.dense == work.scored.size(), "LADR must score a document once per query");
        }
    std::cout << "PASS LADR one-hop, iterative convergence, independent oracle, reset, empty/duplicate seeds\n";
}

void GraphTests(const std::string& path) {
    auto base = Data(4);
    const std::vector<Id> order = {2, 0, 3, 1};
    const size_t vector_offset = 4 + 128 * 4, label_offset = vector_offset + base.dim * 4;
    const size_t row_bytes = label_offset + 8;
    for (bool old_header : {false, true}) {
        const size_t header = old_header ? 100 : 96;
        std::vector<char> data(header + base.size() * row_bytes, 0);
        auto put = [&](size_t offset, auto value) { std::memcpy(data.data()+offset, &value, sizeof(value)); };
        put(0, size_t{0}); put(8, base.size()); put(16, base.size()); put(24, row_bytes);
        put(32, label_offset); put(40, vector_offset); put(48, int{0});
        const size_t max_m = old_header ? 60 : 56;
        put(max_m, size_t{64}); put(max_m+8, size_t{128}); put(max_m+16, size_t{64});
        for (Id internal = 0; internal < base.size(); ++internal) {
            const size_t start = header + size_t(internal) * row_bytes;
            put(start, uint16_t{1}); put(start+4, Id((internal+1)%base.size()));
            put(start+label_offset, base.labels[order[internal]]);
        }
        auto graph = LadrGraph::FromNativeHNSW(base, data);
        for (Id i = 0; i < base.size(); ++i) {
            auto [ids, count] = graph.Row(order[i], 128);
            Require(count == 1 && ids[0] == order[(i+1)%base.size()], "HNSW insertion/label mapping");
        }
        graph.Save(base, path); auto loaded = LadrGraph::Load(base, path);
        Require(graph.offsets == loaded.offsets && graph.neighbors == loaded.neighbors, "graph serialization");
        std::filesystem::remove(path);
        put(header+4, Id(base.size()));
        MustThrow([&] { LadrGraph::FromNativeHNSW(base, data); });
    }
    std::cout << "PASS native HNSW graph extraction, both headers, permuted labels, invalid edge, serialization\n";
}

int main() {
    try {
        openblas_set_num_threads(1);
        const auto prefix = (std::filesystem::temp_directory_path() /
                             ("uhg_extra_test_" + std::to_string(getpid()))).string();
        SketchTests(); SparseScanTests(); IvfTests(prefix+".sinnamon"); LadrTests(); GraphTests(prefix+".ladr");
        std::cout << "All extra baseline correctness tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
