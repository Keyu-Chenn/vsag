// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "simd/basic_func.h"
#include "simd/fp32_simd.h"

namespace paper_baselines {

using Id = uint32_t;
using Hit = std::pair<float, Id>;  // ascending distance, then local ID
using Nearest = std::priority_queue<Hit, std::vector<Hit>, std::greater<Hit>>;
using Worst = std::priority_queue<Hit>;

inline void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline float Dot(const float* a, const float* b, size_t dim) {
    return vsag::FP32ComputeIP(a, b, dim);
}

// Match VSAG's SparseQuantizer code layout: [length][{term, value}...].
// The length word is owned by the data-cell in VSAG; each baseline Sparse
// stores the same interleaved payload contiguously.
struct SparseEntry {
    Id id;
    float value;
};
static_assert(sizeof(SparseEntry) == 2 * sizeof(uint32_t), "unexpected sparse entry padding");
static_assert(offsetof(SparseEntry, value) == sizeof(uint32_t), "unexpected sparse entry layout");

struct Sparse {
    std::vector<SparseEntry> entries;

    Sparse() = default;
    Sparse(std::initializer_list<Id> ids, std::initializer_list<float> values) {
        Require(ids.size() == values.size(), "sparse initializer shape mismatch");
        auto value = values.begin();
        for (Id id : ids) entries.push_back({id, *value++});
    }
    void Push(Id id, float value) { entries.push_back({id, value}); }
    size_t size() const { return entries.size(); }
    bool empty() const { return entries.empty(); }

    Sparse Keep(float fraction) const {
        Require(fraction > 0 && fraction <= 1, "sparse retention must be in (0,1]");
        if (fraction == 1 || entries.empty()) return *this;
        std::vector<size_t> order(entries.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            float av = std::abs(entries[a].value), bv = std::abs(entries[b].value);
            return av != bv ? av > bv : entries[a].id < entries[b].id;
        });
        order.resize(std::max<size_t>(1, std::ceil(entries.size() * fraction)));
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return entries[a].id < entries[b].id;
        });
        Sparse output;
        for (auto i : order) output.entries.push_back(entries[i]);
        return output;
    }
};

// Non-owning query view supporting both the baseline's interleaved sparse
// representation and VSAG's public SparseVector split IDs/values layout.
// This keeps the public-API adapter from adding a representation-conversion
// allocation that HybridIndex itself does not require.
class SparseView {
 public:
    SparseView(const Sparse& sparse)
        : entries_(sparse.entries.data()), size_(sparse.entries.size()) {
    }

    SparseView(const Id* ids, const float* values, size_t size)
        : ids_(ids), values_(values), size_(size) {
        Require(size == 0 || (ids != nullptr && values != nullptr),
                "nonempty sparse view requires IDs and values");
    }

    size_t size() const {
        return size_;
    }

    Id id(size_t offset) const {
        return entries_ == nullptr ? ids_[offset] : entries_[offset].id;
    }

    float value(size_t offset) const {
        return entries_ == nullptr ? values_[offset] : entries_[offset].value;
    }

    Sparse Keep(float fraction) const {
        Require(fraction > 0 && fraction <= 1, "sparse retention must be in (0,1]");
        Sparse owned;
        owned.entries.reserve(size_);
        for (size_t i = 0; i < size_; ++i) owned.entries.push_back({id(i), value(i)});
        return owned.Keep(fraction);
    }

 private:
    const SparseEntry* entries_{nullptr};
    const Id* ids_{nullptr};
    const float* values_{nullptr};
    size_t size_{0};
};

inline float SparseDot(const Sparse& a, const Sparse& b) {
    size_t i = 0, j = 0;
    float sum = 0;
    while (i < a.entries.size() && j < b.entries.size()) {
        if (a.entries[i].id < b.entries[j].id) ++i;
        else if (a.entries[i].id > b.entries[j].id) ++j;
        else { sum += a.entries[i++].value * b.entries[j++].value; }
    }
    return sum;
}

inline float SparseDot(const SparseView& a, const Sparse& b) {
    size_t i = 0, j = 0;
    float sum = 0;
    while (i < a.size() && j < b.entries.size()) {
        if (a.id(i) < b.entries[j].id) ++i;
        else if (a.id(i) > b.entries[j].id) ++j;
        else { sum += a.value(i++) * b.entries[j++].value; }
    }
    return sum;
}

struct Vectors {
    size_t dim = 0;
    std::vector<float> dense;
    std::vector<Sparse> sparse;
    std::vector<int64_t> labels;
    size_t size() const { return sparse.size(); }
    const float* at(Id id) const { return dense.data() + static_cast<size_t>(id) * dim; }
    void Validate() const {
        Require(size() > 0 && size() < std::numeric_limits<Id>::max(), "empty/oversized dataset");
        Require(dim > 0 && dense.size() == size() * dim && labels.size() == size(), "vector shape mismatch");
        for (float x : dense) Require(std::isfinite(x), "nonfinite dense value");
        for (const auto& s : sparse) {
            for (size_t j = 0; j < s.entries.size(); ++j) {
                Require(j == 0 || s.entries[j-1].id < s.entries[j].id,
                        "sparse IDs must be sorted and unique");
                Require(std::isfinite(s.entries[j].value), "nonfinite sparse value");
            }
        }
        auto sorted = labels;
        std::sort(sorted.begin(), sorted.end());
        Require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "duplicate labels");
    }
    uint64_t Fingerprint() const {
        uint64_t hash = 14695981039346656037ULL;
        auto mix = [&](const void* data, size_t count) {
            auto bytes = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < count; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
        };
        mix(&dim, sizeof(dim));
        mix(dense.data(), dense.size() * sizeof(float));
        mix(labels.data(), labels.size() * sizeof(int64_t));
        for (const auto& s : sparse) {
            uint64_t n = s.entries.size();
            mix(&n, sizeof(n));
            // Preserve the original logical fingerprint (all IDs, then all values)
            // so kernel/layout alignment does not silently detach an index from
            // the same corpus.
            for (const auto& entry : s.entries) mix(&entry.id, sizeof(entry.id));
            for (const auto& entry : s.entries) mix(&entry.value, sizeof(entry.value));
        }
        return hash;
    }
};

struct Stats {
    uint64_t dense = 0, sparse = 0, dense_expansions = 0, hybrid_expansions = 0;
    uint64_t transition = 0, scanned = 0, lists = 0;
    Stats& operator+=(const Stats& s) {
        dense += s.dense; sparse += s.sparse; dense_expansions += s.dense_expansions;
        hybrid_expansions += s.hybrid_expansions; transition += s.transition;
        scanned += s.scanned; lists += s.lists; return *this;
    }
};

// Query scorer shared by all three added baselines. Dense scoring calls the
// same runtime-dispatched VSAG FP32 kernels used by FlattenDataCell, including
// its four-vector SIMD kernel. Sparse scoring uses the same interleaved
// {term,value} layout and merge order as VSAG's SparseQuantizer.
class AlignedScorer {
 public:
    enum class Metric { Dense, Hybrid };

    AlignedScorer(const Vectors& base, const float* dense_query, SparseView sparse_query,
                  float alpha, Stats& stats,
                  const std::vector<Sparse>* sparse_base = nullptr)
        : base_(base), dense_query_(dense_query), sparse_query_(sparse_query), alpha_(alpha),
          stats_(stats), sparse_base_(sparse_base == nullptr ? &base.sparse : sparse_base) {
        Require(alpha >= 0 && alpha <= 1 && std::isfinite(alpha), "alpha outside [0,1]");
        Require(sparse_base_->size() == base.size(), "sparse scorer/base size mismatch");
    }

    float Score(Id id, Metric metric) const {
        float result = 0;
        ScoreBatch(&id, 1, &result, metric);
        return result;
    }

    void ScoreBatch(const Id* ids, size_t count, float* distances, Metric metric) const {
        ScoreBatchWithSparse(ids, count, distances, metric, [&](Id id) {
            return SparseDot(sparse_query_, (*sparse_base_)[id]);
        });
    }

    // A scan can prepare its query once and supply an equivalent sparse dot
    // product. Other baselines retain the original merge implementation.
    template<class SparseDotFunction>
    void ScoreBatchWithSparse(const Id* ids, size_t count, float* distances, Metric metric,
                             const SparseDotFunction& sparse_dot) const {
        if (count == 0) return;
        for (size_t i = 0; i < count; ++i) Require(ids[i] < base_.size(), "score ID out of range");

        const bool need_dense = metric == Metric::Dense || alpha_ != 0;
        if (need_dense) {
            DenseInnerProducts(ids, count, distances,
                               metric == Metric::Hybrid && alpha_ != 1);
            stats_.dense += count;
        } else {
            std::fill_n(distances, count, 0.0F);
        }

        if (metric == Metric::Dense) {
            for (size_t i = 0; i < count; ++i) distances[i] = 1.0F - distances[i];
            return;
        }

        const bool need_sparse = alpha_ != 1;
        if (need_sparse) stats_.sparse += count;
        for (size_t i = 0; i < count; ++i) {
            const float sparse_ip = need_sparse ? sparse_dot(ids[i]) : 0.0F;
            distances[i] = 1.0F - (alpha_ * distances[i] + (1.0F - alpha_) * sparse_ip);
        }
    }

 private:
    void DenseInnerProducts(const Id* ids, size_t count, float* products,
                            bool prefetch_sparse) const {
        constexpr size_t kPrefetchStride = 4;
        // VSAG's batch-4 SIMD ABI accumulates into its result references;
        // FlattenDataCell likewise zeroes this output buffer before invoking it.
        std::fill_n(products, count, 0.0F);
        for (size_t i = 0; i < std::min(count, kPrefetchStride); ++i)
            Prefetch(ids[i], prefetch_sparse);
        size_t i = 0;
        for (; i + 3 < count; i += 4) {
            for (size_t j = 0; j < 4; ++j)
                if (i + j + kPrefetchStride < count)
                    Prefetch(ids[i + j + kPrefetchStride], prefetch_sparse);
            vsag::FP32ComputeIPBatch4(dense_query_, base_.dim,
                                      base_.at(ids[i]), base_.at(ids[i + 1]),
                                      base_.at(ids[i + 2]), base_.at(ids[i + 3]),
                                      products[i], products[i + 1], products[i + 2], products[i + 3]);
        }
        for (; i < count; ++i) products[i] = vsag::FP32ComputeIP(dense_query_, base_.at(ids[i]), base_.dim);
    }

    void Prefetch(Id id, bool prefetch_sparse) const {
        vsag::Prefetch(base_.at(id));
        const auto& sparse = (*sparse_base_)[id];
        if (prefetch_sparse && !sparse.entries.empty()) vsag::Prefetch(sparse.entries.data());
    }

    const Vectors& base_;
    const float* dense_query_;
    SparseView sparse_query_;
    float alpha_;
    Stats& stats_;
    const std::vector<Sparse>* sparse_base_;
};

inline float Distance(const Vectors& base, const float* dense, SparseView sparse,
                      Id id, float alpha, Stats* stats = nullptr) {
    float d = 0, s = 0;
    if (alpha != 0) { d = Dot(dense, base.at(id), base.dim); if (stats) ++stats->dense; }
    if (alpha != 1) { s = SparseDot(sparse, base.sparse[id]); if (stats) ++stats->sparse; }
    return 1 - (alpha * d + (1 - alpha) * s);
}

inline float Distance(const Vectors& base, const float* dense, const Sparse& sparse,
                      Id id, float alpha, Stats* stats = nullptr) {
    return Distance(base, dense, SparseView(sparse), id, alpha, stats);
}

inline std::vector<Hit> Sorted(Worst heap, size_t limit) {
    while (heap.size() > limit) heap.pop();
    std::vector<Hit> result(heap.size());
    for (size_t i = result.size(); i > 0; --i) { result[i-1] = heap.top(); heap.pop(); }
    return result;
}

inline void Offer(Worst& heap, Hit hit, size_t limit) {
    if (heap.size() < limit || hit < heap.top()) {
        heap.push(hit);
        if (heap.size() > limit) heap.pop();
    }
}

template <class T> void Write(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    Require(out.good(), "index write failed");
}
template <class T> T Read(std::istream& in) {
    T value{}; in.read(reinterpret_cast<char*>(&value), sizeof(value));
    Require(in.good(), "truncated index"); return value;
}
template <class T> void WriteVector(std::ostream& out, const std::vector<T>& values) {
    Write<uint64_t>(out, values.size());
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    Require(out.good(), "index vector write failed");
}
template <class T> std::vector<T> ReadVector(std::istream& in, uint64_t max_size) {
    auto size = Read<uint64_t>(in);
    Require(size <= max_size, "invalid index vector length");
    std::vector<T> values(size);
    in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
    Require(in.good(), "truncated index vector"); return values;
}
inline void Header(std::ostream& out, uint64_t magic, const Vectors& base) {
    Write(out, magic); Write(out, base.Fingerprint()); Write<uint64_t>(out, base.size());
}
inline void CheckHeader(std::istream& in, uint64_t magic, const Vectors& base) {
    Require(Read<uint64_t>(in) == magic, "index magic/version mismatch");
    Require(Read<uint64_t>(in) == base.Fingerprint(), "index does not match input vectors/labels");
    Require(Read<uint64_t>(in) == base.size(), "index corpus size mismatch");
}

}  // namespace paper_baselines
