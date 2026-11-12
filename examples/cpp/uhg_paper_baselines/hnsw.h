// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.h"

#include <memory>
#include <mutex>

namespace paper_baselines {

// Independent HNSW implementation for these paper baselines.
// Construction uses exponential levels, greedy upper layers, ef-construction,
// diversified neighbor selection, reciprocal edges, M upper/2M bottom capacity.
class HybridHNSW {
 public:
    struct Config {
        uint32_t m = 64, ef_construction = 200, ef_hybrid = 200, seed = 42, build_threads = 1;
        uint32_t parallel_warmup = 10000;
        float alpha_build = 0.5f;
        float construction_sparse_keep = 1.0f;
        bool optimized = false;
    };
    struct SearchConfig { size_t ef = 100; float tau_dense = 0.99f, tau_hybrid = 1; };
    using Layers = std::vector<std::vector<Id>>;
    struct Workspace {
        std::vector<uint32_t> visited;
        std::vector<Id> score_ids;
        std::vector<float> score_distances;
        uint32_t epoch = 0;
        explicit Workspace(size_t n) : visited(n, 0) {
            score_ids.reserve(128);
            score_distances.reserve(128);
        }
        void Reset() {
            if (++epoch == 0) { std::fill(visited.begin(), visited.end(), 0); epoch = 1; }
        }
        bool Mark(Id id) { if (visited[id] == epoch) return false; visited[id] = epoch; return true; }
    };
    explicit HybridHNSW(const Vectors& base, Config config)
        : base_(base), config_(config), graph_(base.size()) {
        Require(config.m >= 2 && config.m <= 4096, "M must be in [2,4096]");
        Require(config.ef_construction >= config.m && config.ef_hybrid > 0, "invalid construction budget");
        Require(!config.optimized || config.ef_hybrid >= config.m,
                "optimized ef_hybrid must be at least M");
        Require(config.alpha_build >= 0 && config.alpha_build <= 1, "invalid construction alpha");
        Require(config.construction_sparse_keep > 0 && config.construction_sparse_keep <= 1,
                "invalid construction sparse retention");
        Require(config.optimized || config.construction_sparse_keep == 1,
                "Naive HNSW construction uses complete sparse vectors");
        Require(config.build_threads >= 1 && config.build_threads <= 1024, "build threads must be in [1,1024]");
        Require(config.parallel_warmup >= 1, "parallel warmup must be positive");
    }
    const Config& config() const { return config_; }
    size_t size() const { return base_.size(); }
    const std::vector<Layers>& graph() const { return graph_; }
    int max_level() const { return max_level_; }
    float search_base_sparse_keep() const { return search_base_sparse_keep_; }
    float search_query_sparse_keep() const { return search_query_sparse_keep_; }
    uint64_t edges() const {
        uint64_t n = 0;
        for (Id id = 0; id < graph_.size(); ++id) {
            if (!search_l0_.empty())
                n += search_l0_[size_t(id) * search_l0_stride_];
            else
                n += graph_[id][0].size();
            for (size_t layer = 1; layer < graph_[id].size(); ++layer) n += graph_[id][layer].size();
        }
        return n;
    }
    void Build() {
        Require(!built_, "index already built");
        std::mt19937 random(config_.seed);
        std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), 1);
        std::vector<int> levels(base_.size());
        for (Id id = 0; id < base_.size(); ++id) {
            levels[id] = std::min(32, static_cast<int>(-std::log(uniform(random)) / std::log(config_.m)));
            graph_[id].resize(levels[id] + 1);
        }
        entry_ = 0;
        max_level_ = levels[0];
        float alpha = config_.optimized ? 1 : config_.alpha_build;
        auto node_locks = std::make_unique<std::mutex[]>(base_.size());
        std::mutex entry_lock;
        size_t parallel_begin = config_.build_threads == 1 ? base_.size()
            : std::min<size_t>(base_.size(), config_.parallel_warmup);
        Workspace serial_work(base_.size());
        for (Id id = 1; id < parallel_begin; ++id)
            Insert(id, levels[id], alpha, serial_work, node_locks.get(), entry_lock);
#pragma omp parallel num_threads(config_.build_threads)
        {
            Workspace work(base_.size());
#pragma omp for schedule(dynamic, 1)
            for (int64_t raw_id = static_cast<int64_t>(parallel_begin);
                 raw_id < static_cast<int64_t>(base_.size()); ++raw_id)
                Insert(static_cast<Id>(raw_id), levels[raw_id], alpha, work, node_locks.get(), entry_lock);
        }
        if (config_.optimized) {
            if (config_.construction_sparse_keep < 1) {
                construction_pruned_.reserve(base_.size());
                for (const auto& sparse : base_.sparse)
                    construction_pruned_.push_back(sparse.Keep(config_.construction_sparse_keep));
            }
            Refine();
            std::vector<Sparse>().swap(construction_pruned_);
        }
        built_ = true;
        Validate();
    }
    void ConfigureSearchSparsePruning(float base_sparse_keep, float query_sparse_keep) {
        Require(built_, "configure search pruning before build/load");
        Require(base_sparse_keep > 0 && base_sparse_keep <= 1,
                "invalid base search sparse retention");
        Require(query_sparse_keep > 0 && query_sparse_keep <= 1,
                "invalid query search sparse retention");
        Require(config_.optimized || (base_sparse_keep == 1 && query_sparse_keep == 1),
                "Naive HNSW search uses complete sparse vectors");
        std::vector<Sparse> pruned;
        if (base_sparse_keep < 1) {
            pruned.reserve(base_.size());
            for (const auto& sparse : base_.sparse)
                pruned.push_back(sparse.Keep(base_sparse_keep));
        }
        search_pruned_.swap(pruned);
        search_base_sparse_keep_ = base_sparse_keep;
        search_query_sparse_keep_ = query_sparse_keep;
    }
    void ConfigureSearchSparsePruning(float sparse_keep) {
        ConfigureSearchSparsePruning(sparse_keep, sparse_keep);
    }
    std::vector<Hit> Search(const float* qd, const Sparse& qs, float alpha, size_t k,
                            SearchConfig config, Workspace& work, Stats& stats) const {
        return Search(qd, SparseView(qs), alpha, k, config, work, stats);
    }
    std::vector<Hit> Search(const float* qd, SparseView qs, float alpha, size_t k,
                            SearchConfig config, Workspace& work, Stats& stats) const {
        Require(built_, "search before build/load");
        Require(alpha >= 0 && alpha <= 1 && std::isfinite(alpha), "alpha outside [0,1]");
        Require(k > 0 && k <= base_.size(), "invalid k");
        Require(config.ef >= k, "ef must be >= k");
        Require(config.tau_dense >= 0 && config.tau_dense <= 1 &&
                config.tau_hybrid >= 0 && config.tau_hybrid <= 1, "tau outside [0,1]");
        size_t ef = std::min(config.ef, base_.size());
        Sparse clipped;
        SparseView search_sparse = qs;
        if (search_query_sparse_keep_ < 1) {
            clipped = qs.Keep(search_query_sparse_keep_);
            search_sparse = SparseView(clipped);
        }
        AlignedScorer scorer(base_, qd, search_sparse, alpha, stats,
                             search_pruned_.empty() ? nullptr : &search_pruned_);
        Id ep = entry_;
        for (int l = max_level_; l > 0; --l)
            ep = GreedyAligned(ep, l, scorer,
                               config_.optimized ? AlignedScorer::Metric::Dense
                                                 : AlignedScorer::Metric::Hybrid,
                               work);
        work.Reset();
        Nearest candidates;
        Worst results;
        auto first_metric = config_.optimized ? AlignedScorer::Metric::Dense
                                              : AlignedScorer::Metric::Hybrid;
        float initial = scorer.Score(ep, first_metric);
        candidates.emplace(initial, ep); results.emplace(initial, ep); work.Mark(ep);
        if (config_.optimized) {
            TraverseAligned(0, ef, scorer, AlignedScorer::Metric::Dense, config.tau_dense,
                            candidates, results, work, &stats.dense_expansions);
            // The paper leaves visited/queue reuse unspecified. Keep the UNION
            // of both queues, re-score every unique retained node exactly once,
            // and re-expand it under the new metric with a fresh visited epoch.
            // No second independent dense search or discarded candidate queue.
            std::vector<Id> retained;
            while (!candidates.empty()) { retained.push_back(candidates.top().second); candidates.pop(); }
            while (!results.empty()) { retained.push_back(results.top().second); results.pop(); }
            std::sort(retained.begin(), retained.end());
            retained.erase(std::unique(retained.begin(), retained.end()), retained.end());
            work.Reset();
            work.score_distances.resize(retained.size());
            scorer.ScoreBatch(retained.data(), retained.size(), work.score_distances.data(),
                              AlignedScorer::Metric::Hybrid);
            for (size_t i = 0; i < retained.size(); ++i) {
                Hit hit(work.score_distances[i], retained[i]);
                candidates.push(hit); Offer(results, hit, ef); work.Mark(retained[i]);
            }
            stats.transition += retained.size();
        }
        TraverseAligned(0, ef, scorer, AlignedScorer::Metric::Hybrid,
                        config_.optimized ? config.tau_hybrid : 1,
                        candidates, results, work, &stats.hybrid_expansions);
        // Algorithm 2 returns the top-k queue under the traversal metric.
        // Sparse pruning is therefore part of that metric; there is no
        // undocumented complete-vector re-ranking pass after graph search.
        return Sorted(results, k);
    }
    void Save(const std::string& path) const {
        Require(built_, "save before build/load");
        std::ofstream out(path, std::ios::binary); Header(out, kMagic, base_);
        Write(out, config_.m); Write(out, config_.ef_construction); Write(out, config_.ef_hybrid);
        Write(out, config_.seed); Write(out, config_.build_threads); Write(out, config_.parallel_warmup);
        // Reuse the existing on-disk provenance field. Legacy complete-vector
        // indexes store 1.0 and remain byte-compatible.
        Write(out, config_.alpha_build); Write(out, config_.construction_sparse_keep);
        Write<uint8_t>(out, config_.optimized); Write(out, entry_); Write<int32_t>(out, max_level_);
        for (Id id = 0; id < graph_.size(); ++id) {
            const auto& node = graph_[id];
            Write<uint32_t>(out, node.size());
            for (size_t layer = 0; layer < node.size(); ++layer) {
                if (layer == 0 && node[layer].empty() && !search_l0_.empty()) {
                    const auto* row = search_l0_.data() + size_t(id) * search_l0_stride_;
                    Write<uint64_t>(out, row[0]);
                    out.write(reinterpret_cast<const char*>(row + 1), row[0] * sizeof(Id));
                    Require(out.good(), "index vector write failed");
                } else {
                    WriteVector(out, node[layer]);
                }
            }
        }
    }
    static HybridHNSW Load(const Vectors& base, const std::string& path) {
        std::ifstream in(path, std::ios::binary); CheckHeader(in, kMagic, base);
        Config c;
        c.m = Read<uint32_t>(in); c.ef_construction = Read<uint32_t>(in); c.ef_hybrid = Read<uint32_t>(in);
        c.seed = Read<uint32_t>(in); c.build_threads = Read<uint32_t>(in);
        c.parallel_warmup = Read<uint32_t>(in);
        c.alpha_build = Read<float>(in);
        c.construction_sparse_keep = Read<float>(in);
        auto optimized = Read<uint8_t>(in); Require(optimized <= 1, "invalid index type"); c.optimized = optimized;
        HybridHNSW graph(base, c); graph.entry_ = Read<Id>(in); graph.max_level_ = Read<int32_t>(in);
        const size_t l0_capacity = (c.optimized ? 1 : 2) * size_t(c.m);
        graph.search_l0_stride_ = l0_capacity + 1;
        Require(base.size() <= std::numeric_limits<size_t>::max() / graph.search_l0_stride_,
                "flat L0 graph is too large");
        graph.search_l0_.assign(base.size() * graph.search_l0_stride_, 0);
        for (Id id = 0; id < graph.graph_.size(); ++id) {
            auto& node = graph.graph_[id];
            auto layers = Read<uint32_t>(in); Require(layers >= 1 && layers <= 33, "invalid layer count");
            node.resize(layers);
            auto count = Read<uint64_t>(in);
            Require(count <= l0_capacity, "invalid L0 vector length");
            auto* row = graph.search_l0_.data() + size_t(id) * graph.search_l0_stride_;
            row[0] = static_cast<uint32_t>(count);
            in.read(reinterpret_cast<char*>(row + 1), count * sizeof(Id));
            Require(in.good(), "truncated L0 vector");
            for (size_t layer = 1; layer < layers; ++layer)
                node[layer] = ReadVector<Id>(in, c.m);
        }
        Require(in.peek() == EOF, "unexpected trailing index data");
        graph.built_ = true;
        graph.Validate();
        return graph;
    }

 private:
    static constexpr uint64_t kMagic = 0x334e485042474855ULL;
    const Vectors& base_;
    Config config_;
    float search_base_sparse_keep_ = 1;
    float search_query_sparse_keep_ = 1;
    std::vector<Sparse> construction_pruned_;
    std::vector<Sparse> search_pruned_;
    std::vector<Layers> graph_;
    // VSAG GraphDataCell row: [uint32 count][fixed maximum-degree ID slots].
    // HNSW's sparse upper levels stay in their native layer representation;
    // L0 dominates graph traversal and memory traffic.
    std::vector<uint32_t> search_l0_;
    size_t search_l0_stride_ = 0;
    Id entry_ = 0;
    int max_level_ = -1;
    bool built_ = false;

    void PrepareSearchLayout(bool release_nested) {
        if (!search_l0_.empty()) return;
        const size_t capacity = (config_.optimized ? 1 : 2) * size_t(config_.m);
        search_l0_stride_ = capacity + 1;
        Require(base_.size() <= std::numeric_limits<size_t>::max() / search_l0_stride_,
                "flat L0 graph is too large");
        search_l0_.assign(base_.size() * search_l0_stride_, 0);
        for (Id id = 0; id < base_.size(); ++id) {
            const auto& neighbors = graph_[id][0];
            Require(neighbors.size() <= capacity, "L0 degree exceeds flat row capacity");
            auto* row = search_l0_.data() + size_t(id) * search_l0_stride_;
            row[0] = static_cast<uint32_t>(neighbors.size());
            std::copy(neighbors.begin(), neighbors.end(), row + 1);
        }
        if (release_nested) {
            for (auto& layers : graph_) std::vector<Id>().swap(layers[0]);
        }
    }

    std::pair<const Id*, size_t> SearchNeighbors(Id id, int layer) const {
        if (layer == 0) {
            auto* self = const_cast<HybridHNSW*>(this);
            self->PrepareSearchLayout(false);
            const auto* row = search_l0_.data() + size_t(id) * search_l0_stride_;
            return {reinterpret_cast<const Id*>(row + 1), row[0]};
        }
        const auto& neighbors = graph_[id][layer];
        return {neighbors.data(), neighbors.size()};
    }

    float PairDistance(Id a, Id b, float alpha) const {
        if (alpha == 1) return 1 - Dot(base_.at(a), base_.at(b), base_.dim);
        const auto& sparse = construction_pruned_.empty() ? base_.sparse : construction_pruned_;
        return 1 - (alpha * Dot(base_.at(a), base_.at(b), base_.dim) +
                    (1 - alpha) * SparseDot(sparse[a], sparse[b]));
    }
    template <class F>
    void ForEachNeighbor(Id id, int layer, std::mutex* locks, F visit) const {
        if (locks == nullptr) {
            for (Id neighbor : graph_[id][layer]) visit(neighbor);
            return;
        }
        std::vector<Id> neighbors;
        {
            std::lock_guard<std::mutex> guard(locks[id]);
            neighbors = graph_[id][layer];
        }
        for (Id neighbor : neighbors) visit(neighbor);
    }
    template <class D> Id Greedy(Id ep, int layer, D distance) const {
        return Greedy(ep, layer, distance, nullptr);
    }
    template <class D> Id Greedy(Id ep, int layer, D distance, std::mutex* locks) const {
        Hit best(distance(ep), ep);
        bool changed = true;
        while (changed) {
            changed = false;
            ForEachNeighbor(best.second, layer, locks, [&](Id id) {
                Hit hit(distance(id), id);
                if (hit < best) { best = hit; changed = true; }
            });
        }
        return best.second;
    }
    Id GreedyAligned(Id ep, int layer, const AlignedScorer& scorer,
                     AlignedScorer::Metric metric, Workspace& work) const {
        Hit best(scorer.Score(ep, metric), ep);
        bool changed = true;
        while (changed) {
            changed = false;
            const auto [neighbors, neighbor_count] = SearchNeighbors(best.second, layer);
            work.score_distances.resize(neighbor_count);
            scorer.ScoreBatch(neighbors, neighbor_count, work.score_distances.data(), metric);
            for (size_t i = 0; i < neighbor_count; ++i) {
                Hit hit(work.score_distances[i], neighbors[i]);
                if (hit < best) { best = hit; changed = true; }
            }
        }
        return best.second;
    }
    template <class D> void Traverse(int layer, size_t ef, D distance, float tau,
                                     Nearest& candidates, Worst& results, Workspace& work,
                                     uint64_t* expansions = nullptr, std::mutex* locks = nullptr) const {
        while (!candidates.empty()) {
            Hit current = candidates.top();
            // Keep the unexpanded frontier for the dense-to-hybrid transition.
            if (results.size() >= ef && current > results.top()) break;
            candidates.pop();
            if (expansions) ++*expansions;
            size_t updates = 0;
            ForEachNeighbor(current.second, layer, locks, [&](Id id) {
                if (!work.Mark(id)) return;
                Hit hit(distance(id), id);
                if (results.size() < ef || hit < results.top()) {
                    candidates.push(hit); Offer(results, hit, ef); ++updates;
                }
            });
            // Define one iteration as expanding one node. Do not early-stop
            // before the result queue fills; tau=1 is standard convergence.
            if (tau < 1 && results.size() >= ef && updates < ef * (1.0 - tau)) break;
        }
    }
    void TraverseAligned(int layer, size_t ef, const AlignedScorer& scorer,
                         AlignedScorer::Metric metric, float tau,
                         Nearest& candidates, Worst& results, Workspace& work,
                         uint64_t* expansions) const {
        while (!candidates.empty()) {
            Hit current = candidates.top();
            if (results.size() >= ef && current > results.top()) break;
            candidates.pop();
            if (expansions) ++*expansions;
            if (!candidates.empty()) {
                const auto next = candidates.top().second;
                if (layer == 0) {
                    const auto* row = search_l0_.data() + size_t(next) * search_l0_stride_;
                    vsag::Prefetch(row);
                } else if (graph_[next].size() > static_cast<size_t>(layer) &&
                           !graph_[next][layer].empty()) {
                    vsag::Prefetch(graph_[next][layer].data());
                }
            }

            work.score_ids.clear();
            const auto [neighbors, neighbor_count] = SearchNeighbors(current.second, layer);
            work.score_ids.reserve(std::max(work.score_ids.capacity(), neighbor_count));
            for (size_t i = 0; i < neighbor_count; ++i)
                if (work.Mark(neighbors[i])) work.score_ids.push_back(neighbors[i]);
            work.score_distances.resize(work.score_ids.size());
            scorer.ScoreBatch(work.score_ids.data(), work.score_ids.size(),
                              work.score_distances.data(), metric);

            size_t updates = 0;
            for (size_t i = 0; i < work.score_ids.size(); ++i) {
                Hit hit(work.score_distances[i], work.score_ids[i]);
                if (results.size() < ef || hit < results.top()) {
                    candidates.push(hit); Offer(results, hit, ef); ++updates;
                }
            }
            if (tau < 1 && results.size() >= ef && updates < ef * (1.0 - tau)) break;
        }
    }
    template <class D> std::vector<Hit> LayerSearch(Id ep, int layer, size_t ef, D distance,
                                                   Workspace& work, std::mutex* locks = nullptr) const {
        work.Reset(); work.Mark(ep);
        Hit first(distance(ep), ep);
        Nearest candidates; candidates.push(first); Worst results; results.push(first);
        Traverse(layer, ef, distance, 1, candidates, results, work, nullptr, locks);
        return Sorted(results, ef);
    }
    std::vector<Id> Select(Id center, std::vector<Hit> candidates, size_t m, float alpha) const {
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                       [&](Hit h) { return h.second == center; }), candidates.end());
        std::sort(candidates.begin(), candidates.end());
        std::vector<Id> selected;
        for (auto [distance, id] : candidates) {
            bool accept = true;
            if (candidates.size() >= m) {
                for (Id previous : selected)
                    if (PairDistance(id, previous, alpha) < distance) { accept = false; break; }
            }
            if (accept) selected.push_back(id);
            if (selected.size() == m) break;
        }
        return selected;
    }
    void Insert(Id id, int level, float alpha, Workspace& work, std::mutex* node_locks,
                std::mutex& entry_lock) {
        std::unique_lock<std::mutex> entry_guard(entry_lock);
        Id ep = entry_;
        int observed_max_level = max_level_;
        // A node that raises the maximum level keeps this lock until it is
        // fully connected. Ordinary insertions release it immediately.
        if (level <= observed_max_level) entry_guard.unlock();
        auto distance = [&](Id other) { return PairDistance(id, other, alpha); };
        for (int l = observed_max_level; l > level; --l)
            ep = Greedy(ep, l, distance, node_locks);

        std::vector<std::vector<Id>> forward(level + 1);
        for (int l = std::min(level, observed_max_level); l >= 0; --l) {
            auto candidates = LayerSearch(ep, l, config_.ef_construction, distance, work, node_locks);
            forward[l] = Select(id, candidates, config_.m, alpha);
            if (!candidates.empty()) ep = candidates.front().second;
        }
        {
            std::lock_guard<std::mutex> guard(node_locks[id]);
            graph_[id] = forward;
        }
        // Publishing reciprocal edges only after all forward layers are ready
        // prevents another insertion from observing a half-initialized node.
        for (int l = std::min(level, observed_max_level); l >= 0; --l) {
            for (Id other : forward[l]) {
                std::lock_guard<std::mutex> guard(node_locks[other]);
                auto& reverse = graph_[other][l];
                if (std::find(reverse.begin(), reverse.end(), id) == reverse.end()) reverse.push_back(id);
                size_t capacity = l == 0 ? 2 * config_.m : config_.m;
                if (reverse.size() > capacity) {
                    std::vector<Hit> choices;
                    for (Id v : reverse) choices.emplace_back(PairDistance(other, v, alpha), v);
                    reverse = Select(other, std::move(choices), capacity, alpha);
                }
            }
        }
        if (entry_guard.owns_lock()) {
            entry_ = id;
            max_level_ = level;
        }
    }
    void Refine() {
        std::vector<std::vector<Id>> refined(base_.size());
#pragma omp parallel num_threads(config_.build_threads)
        {
            Workspace work(base_.size());
#pragma omp for schedule(dynamic, 1)
            for (int64_t raw_id = 0; raw_id < static_cast<int64_t>(base_.size()); ++raw_id) {
            Id id = static_cast<Id>(raw_id);
            // Algorithm 1: all self-searches read the ORIGINAL dense graph.
            // +1 reserves space for the self hit, which is excluded from edges.
            auto candidates = LayerSearch(id, 0, config_.ef_hybrid + 1,
                [&](Id other) { return PairDistance(id, other, config_.alpha_build); }, work);
            refined[id] = Select(id, candidates, config_.m, config_.alpha_build);
            }
        }
        // Batch directed L0 replacement; no undocumented reciprocal-edge pass.
        for (Id id = 0; id < base_.size(); ++id) graph_[id][0] = std::move(refined[id]);
    }
    void Validate() const {
        Require(entry_ < base_.size() && max_level_ >= 0 && max_level_ <= 32, "invalid entry/level");
        Require(graph_[entry_].size() == size_t(max_level_ + 1), "entry level mismatch");
        std::vector<Id> sorted;
        for (Id id = 0; id < graph_.size(); ++id) {
            Require(!graph_[id].empty() && graph_[id].size() <= size_t(max_level_ + 1), "invalid node levels");
            for (size_t l = 0; l < graph_[id].size(); ++l) {
                if (l == 0 && !search_l0_.empty()) {
                    const auto* row = search_l0_.data() + size_t(id) * search_l0_stride_;
                    sorted.assign(row + 1, row + 1 + row[0]);
                } else {
                    sorted = graph_[id][l];
                }
                std::sort(sorted.begin(), sorted.end());
                const size_t capacity = l == 0 && !config_.optimized ? 2 * config_.m : config_.m;
                Require(sorted.size() <= capacity, "degree exceeds capacity");
                Require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "duplicate graph edge");
                for (Id other : sorted)
                    Require(other < graph_.size() && other != id && graph_[other].size() > l, "invalid graph edge");
            }
        }
    }
};
}  // namespace paper_baselines
