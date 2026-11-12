// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.h"
#include <cblas.h>

namespace paper_baselines {

// Bruch et al., arXiv:2309.09013, Sec. 4.2.2, Eqs. (6)-(8).
// For nonnegative vectors, use all sketch coordinates for the upper bound
// (the effective-dimension convention in Sec. 4.3). Documents take maxima;
// queries add colliding coordinates. This asymmetry is essential.
class WeakSinnamon {
 public:
    WeakSinnamon(size_t dim, uint64_t seed) : dim_(dim), seed_(seed) {
        Require(dim > 0, "Sinnamon sketch dimension must be positive");
    }
    size_t Bucket(Id term) const {
        uint64_t z = static_cast<uint64_t>(term) + seed_ + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return (z ^ (z >> 31)) % dim_;
    }
    void Prepare(const std::vector<Sparse>& vectors) const {
        for (const auto& v : vectors)
            for (const auto& entry : v.entries)
                Require(std::isfinite(entry.value) && entry.value >= 0,
                        "nonnegative Sinnamon variant requires nonnegative sparse vectors");
    }
    void Document(const Sparse& vector, float* output) const {
        std::fill_n(output, dim_, 0.0F);
        for (const auto& entry : vector.entries) {
            auto& value = output[Bucket(entry.id)];
            value = std::max(value, entry.value);
        }
    }
    void Query(const Sparse& vector, float* output) const {
        std::fill_n(output, dim_, 0.0F);
        for (const auto& entry : vector.entries) output[Bucket(entry.id)] += entry.value;
    }
 private:
    size_t dim_;
    uint64_t seed_;
};

// Nonnegative query weights indexed by term remove the repeated two-pointer
// merge when an IVF scan scores many documents. -1 marks an absent term, so
// matching zero weights still follow the original accumulation order. Cap the
// scratch allocation and retain the merge path for very large term IDs.
class SinnamonSparseQuery {
 public:
    explicit SinnamonSparseQuery(const Sparse& query, bool enabled = true) : query_(query) {
        constexpr size_t kMaxTerms = 1 << 20;  // At most 4 MiB per active query.
        if (!enabled || query.empty() || query.entries.back().id >= kMaxTerms) return;
        weights_.assign(size_t(query.entries.back().id) + 1, -1.0F);
        for (const auto& entry : query.entries) weights_[entry.id] = entry.value;
    }
    float Dot(const Sparse& document) const {
        if (weights_.empty()) return SparseDot(query_, document);
        float sum = 0;
        for (const auto& entry : document.entries) {
            if (entry.id >= weights_.size()) break;
            const float weight = weights_[entry.id];
            if (weight >= 0) sum += weight * entry.value;
        }
        return sum;
    }
 private:
    const Sparse& query_;
    std::vector<float> weights_;
};

// Hybrid IVF, Algorithms 5/6 of the same paper; the sketch dimension and
// cluster count follow zj.pdf Sec. 5.1.3. Cluster in the sketched space, then
// evaluate complete original dense+sparse vectors in the selected lists.
class SinnamonIVF {
 public:
    struct Config {
        uint32_t clusters = 1000, sketch_dim = 1024, iterations = 20, seed = 42;
        size_t assignment_batch_size = 1024;  // Execution-only; not serialized.
    };
    explicit SinnamonIVF(const Vectors& base, Config config)
        : base_(base), config_(config), projector_(config.sketch_dim, config.seed),
          dim_(base.dim + config.sketch_dim) {
        Require(config.clusters > 0 && config.clusters <= base.size(), "invalid cluster count");
        Require(config.sketch_dim > 0 && config.sketch_dim <= 65536 && config.iterations > 0,
                "invalid sketch dimension / training iterations");
        Require(config.assignment_batch_size > 0, "invalid assignment batch size");
        projector_.Prepare(base.sparse);
    }
    const Config& config() const { return config_; }
    const std::vector<std::vector<Id>>& lists() const { return lists_; }
    // Validate the nonnegative input domain outside query timing. Query
    // hashing itself is performed by Search and remains inside the timer.
    void PrepareQueries(const std::vector<Sparse>& queries) { projector_.Prepare(queries); }
    void Build() {
        Require(!built_, "index already built");
        size_t n = base_.size(), p = config_.clusters;
        auto project = [&](Id id, float* row) {
            std::copy(base_.at(id), base_.at(id) + base_.dim, row);
            projector_.Document(base_.sparse[id], row + base_.dim);
            Normalize(row, dim_);
        };
        // The projected representation is invariant across Lloyd iterations.
        // Cache it once: on the largest paper corpus this is about 72 GB
        // (67.5 GiB),
        // which fits the experiment machine and avoids repeating every sparse
        // hash projection twenty times. It is construction scratch space and is
        // released before serialization/search.
        Require(n <= std::numeric_limits<size_t>::max() / dim_, "projected matrix is too large");
        std::vector<float> projected(n * dim_);
        for (Id id = 0; id < n; ++id) project(id, projected.data() + size_t(id) * dim_);
        // Spherical KMeans: unit input directions; normalized mean directions.
        // Seeded distinct random initial centers, then full Lloyd iterations.
        std::vector<Id> permutation(n); std::iota(permutation.begin(), permutation.end(), 0);
        std::mt19937 random(config_.seed); std::shuffle(permutation.begin(), permutation.end(), random);
        centroids_.resize(p * dim_);
        for (size_t c = 0; c < p; ++c)
            std::copy_n(projected.data() + size_t(permutation[c]) * dim_, dim_,
                        centroids_.data() + c * dim_);
        size_t batch_size = std::min(n, config_.assignment_batch_size);
        std::vector<float> scores(batch_size * p), sums(p * dim_);
        std::vector<Id> assignment(n, std::numeric_limits<Id>::max());
        std::vector<uint32_t> counts(p);
        auto assign = [&](bool accumulate) {
            if (accumulate) {
                std::fill(sums.begin(), sums.end(), 0);
                std::fill(counts.begin(), counts.end(), 0);
            }
            size_t changed = 0;
            for (size_t begin = 0; begin < n; begin += batch_size) {
                size_t rows = std::min(batch_size, n - begin);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, p, dim_, 1,
                            projected.data() + begin * dim_, dim_, centroids_.data(), dim_, 0,
                            scores.data(), p);
                for (size_t i = 0; i < rows; ++i) {
                    Id best = std::max_element(scores.data() + i * p, scores.data() + (i+1) * p) -
                              (scores.data() + i * p);
                    changed += assignment[begin + i] != best; assignment[begin + i] = best;
                    if (accumulate) {
                        ++counts[best];
                        float* dest = sums.data() + size_t(best) * dim_;
                        const float* src = projected.data() + (begin + i) * dim_;
                        // Preserve full Lloyd's document-order float accumulation.
#pragma omp simd
                        for (size_t j = 0; j < dim_; ++j) dest[j] += src[j];
                    }
                }
            }
            return changed;
        };
        for (size_t iteration = 0; iteration < config_.iterations; ++iteration) {
            size_t changed = assign(true);
            bool reseeded = false;
            for (size_t c = 0; c < p; ++c) {
                if (!counts[c]) {
                    std::copy_n(projected.data() + size_t(permutation[(iteration*p+c)%n]) * dim_,
                                dim_, sums.data() + c * dim_);
                    reseeded = true;
                }
                Normalize(sums.data() + c * dim_, dim_);
            }
            centroids_.swap(sums);
            if (!changed && !reseeded) break;
        }
        assign(false);  // Lists must match the FINAL centroid directions.
        lists_.assign(p, {});
        for (Id i = 0; i < n; ++i) lists_[assignment[i]].push_back(i);
        built_ = true; Validate();
    }
    std::vector<Hit> Search(const float* qd, const Sparse& qs, float alpha, size_t k,
                            size_t examine, Stats& stats) {
        Require(built_, "search before build/load");
        Require(k > 0 && k <= base_.size() && examine >= k, "invalid k/examine budget");
        Require(alpha >= 0 && alpha <= 1 && std::isfinite(alpha), "alpha outside [0,1]");
        std::vector<float> query(dim_);
        projector_.Query(qs, query.data() + base_.dim);
        for (size_t j = 0; j < base_.dim; ++j) query[j] = alpha * qd[j];
        for (size_t j = base_.dim; j < dim_; ++j) query[j] *= 1 - alpha;
        std::vector<Hit> order;
        order.reserve(config_.clusters);
        Id c = 0;
        for (; c + 3 < config_.clusters; c += 4) {
            float score0 = 0, score1 = 0, score2 = 0, score3 = 0;
            vsag::FP32ComputeIPBatch4(query.data(), dim_,
                                      centroids_.data() + size_t(c) * dim_,
                                      centroids_.data() + size_t(c + 1) * dim_,
                                      centroids_.data() + size_t(c + 2) * dim_,
                                      centroids_.data() + size_t(c + 3) * dim_,
                                      score0, score1, score2, score3);
            order.emplace_back(-score0, c);
            order.emplace_back(-score1, c + 1);
            order.emplace_back(-score2, c + 2);
            order.emplace_back(-score3, c + 3);
        }
        for (; c < config_.clusters; ++c)
            order.emplace_back(-Dot(query.data(), centroids_.data() + size_t(c) * dim_, dim_), c);
        std::sort(order.begin(), order.end());
        Worst result;
        size_t scanned = 0;
        AlignedScorer scorer(base_, qd, qs, alpha, stats);
        SinnamonSparseQuery sparse_query(qs, alpha != 1);
        std::vector<float> distances;
        for (auto [distance, cluster] : order) {
            (void)distance; ++stats.lists;
            const auto& list = lists_[cluster];
            // Keep prefetched sparse rows close to their use. A multiple of
            // four preserves the dense batch/scalar kernel chosen for each ID.
            constexpr size_t kScoreBatch = 128;
            for (size_t begin = 0; begin < list.size(); begin += kScoreBatch) {
                const size_t count = std::min(kScoreBatch, list.size() - begin);
                distances.resize(count);
                scorer.ScoreBatchWithSparse(list.data() + begin, count, distances.data(),
                                             AlignedScorer::Metric::Hybrid, [&](Id id) {
                    return sparse_query.Dot(base_.sparse[id]);
                });
                for (size_t i = 0; i < count; ++i)
                    Offer(result, Hit(distances[i], list[begin + i]), k);
            }
            scanned += list.size();
            // Algorithm 6 uses a document budget, stopping at a list boundary.
            if (scanned >= examine) break;
        }
        stats.scanned += scanned;
        return Sorted(result, k);
    }
    void Save(const std::string& path) const {
        Require(built_, "save before build/load");
        std::ofstream out(path, std::ios::binary); Header(out, kMagic, base_);
        Write(out, config_.clusters); Write(out, config_.sketch_dim);
        Write(out, config_.iterations); Write(out, config_.seed); WriteVector(out, centroids_);
        for (const auto& list : lists_) WriteVector(out, list);
    }
    static SinnamonIVF Load(const Vectors& base, const std::string& path) {
        std::ifstream in(path, std::ios::binary); CheckHeader(in, kMagic, base);
        Config c; c.clusters = Read<uint32_t>(in); c.sketch_dim = Read<uint32_t>(in);
        c.iterations = Read<uint32_t>(in); c.seed = Read<uint32_t>(in);
        SinnamonIVF index(base, c);
        index.centroids_ = ReadVector<float>(in, size_t(c.clusters) * index.dim_);
        index.lists_.resize(c.clusters);
        for (auto& list : index.lists_) list = ReadVector<Id>(in, base.size());
        Require(in.peek() == EOF, "unexpected trailing index data");
        index.built_ = true; index.Validate(); return index;
    }
 private:
    static constexpr uint64_t kMagic = 0x31564e4e49534855ULL;
    const Vectors& base_;
    Config config_;
    WeakSinnamon projector_;
    size_t dim_;
    std::vector<float> centroids_;
    std::vector<std::vector<Id>> lists_;
    bool built_ = false;
    static void Normalize(float* row, size_t dim) {
        float norm = std::sqrt(Dot(row, row, dim));
        if (norm > 0) for (size_t j = 0; j < dim; ++j) row[j] /= norm;
    }
    void Validate() const {
        Require(centroids_.size() == size_t(config_.clusters) * dim_, "invalid centroid shape");
        for (float c : centroids_) Require(std::isfinite(c), "nonfinite centroid");
        std::vector<uint8_t> seen(base_.size(), 0);
        for (const auto& list : lists_) for (Id id : list) {
            Require(id < base_.size() && !seen[id], "invalid/duplicate IVF document"); seen[id] = 1;
        }
        Require(std::all_of(seen.begin(), seen.end(), [](uint8_t x) { return x == 1; }), "IVF omitted documents");
    }
};
}  // namespace paper_baselines
