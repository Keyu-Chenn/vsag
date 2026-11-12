// Copyright 2026 the vsag project. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common.h"
#include <unordered_map>

namespace paper_baselines {

// The approximate document-graph variant evaluated in LADR Sec. 5.4.
// IDs in these CSR rows are input-corpus row IDs, not HNSW insertion IDs.
class LadrGraph {
 public:
    std::vector<uint64_t> offsets;
    std::vector<Id> neighbors;

    size_t size() const { return offsets.empty() ? 0 : offsets.size() - 1; }
    std::pair<const Id*, size_t> Row(Id id, size_t limit) const {
        Require(id < size(), "LADR graph ID out of range");
        return {neighbors.empty() ? nullptr : neighbors.data() + offsets[id],
                std::min<uint64_t>(limit, offsets[id+1] - offsets[id])};
    }
    void Validate(size_t count) const {
        Require(offsets.size() == count + 1 && offsets.front() == 0 &&
                offsets.back() == neighbors.size(), "invalid LADR graph offsets");
        for (Id id = 0; id < count; ++id) {
            Require(offsets[id] <= offsets[id+1], "decreasing LADR graph offsets");
            std::vector<Id> row(neighbors.begin() + offsets[id], neighbors.begin() + offsets[id+1]);
            std::sort(row.begin(), row.end());
            Require(std::adjacent_find(row.begin(), row.end()) == row.end(), "duplicate LADR edge");
            for (Id other : row) Require(other < count && other != id, "invalid/self LADR edge");
        }
    }
    static LadrGraph FromRows(const std::vector<std::vector<Id>>& rows) {
        LadrGraph graph;
        graph.offsets.push_back(0);
        for (const auto& row : rows) {
            graph.neighbors.insert(graph.neighbors.end(), row.begin(), row.end());
            graph.offsets.push_back(graph.neighbors.size());
        }
        graph.Validate(rows.size());
        return graph;
    }
    void Save(const Vectors& base, const std::string& path) const {
        Validate(base.size());
        std::ofstream out(path, std::ios::binary);
        Header(out, kMagic, base);
        WriteVector(out, offsets);
        WriteVector(out, neighbors);
        Require(out.good(), "LADR graph write failed");
    }
    static LadrGraph Load(const Vectors& base, const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        CheckHeader(in, kMagic, base);
        LadrGraph graph;
        graph.offsets = ReadVector<uint64_t>(in, base.size() + 1);
        Require(graph.offsets.size() == base.size() + 1, "LADR graph row count mismatch");
        graph.neighbors = ReadVector<Id>(in, base.size() * 128);
        Require(in.peek() == EOF, "unexpected trailing LADR graph data");
        graph.Validate(base.size());
        return graph;
    }

    // Read the whole native input before calling this conversion so the caller
    // can report source-index I/O separately from graph conversion time.
    static LadrGraph FromNativeHNSW(const Vectors& base, const std::vector<char>& data) {
        static_assert(sizeof(size_t) == 8, "native VSAG HNSW parser requires 64-bit size_t");
        auto get = [&](size_t offset, auto& value) {
            Require(offset <= data.size() && sizeof(value) <= data.size() - offset,
                    "truncated native HNSW index");
            std::memcpy(&value, data.data() + offset, sizeof(value));
        };
        size_t offset0 = 0, capacity = 0, count = 0, row_bytes = 0, label_offset = 0;
        size_t vector_offset = 0, max_m = 0, max_m0 = 0, m = 0;
        get(0, offset0); get(8, capacity); get(16, count); get(24, row_bytes);
        get(32, label_offset); get(40, vector_offset);
        // Older VSAG snapshots used an int64 entrypoint; current ones use uint32.
        // This is the same header discrimination as DenseEntryHNSWGraph.
        get(56, max_m); get(64, max_m0); get(72, m);
        size_t header_size = 96;
        if (!(max_m > 0 && max_m <= 100000 && m == max_m && max_m0 >= m)) {
            get(60, max_m); get(68, max_m0); get(76, m);
            header_size = 100;
        }
        Require(offset0 == 0 && count == base.size() && capacity >= count,
                "native HNSW size/offset mismatch");
        Require(m > 0 && m <= 128 && max_m == m && max_m0 >= m && max_m0 <= 128,
                "native HNSW graph must have at most 128 L0 neighbors");
        Require(row_bytes >= 4 + max_m0 * sizeof(Id) && label_offset <= row_bytes &&
                sizeof(int64_t) <= row_bytes - label_offset && vector_offset < label_offset,
                "invalid native HNSW row layout");
        Require(header_size <= data.size() && count <= (data.size() - header_size) / row_bytes,
                "truncated native HNSW rows");

        std::unordered_map<int64_t, Id> label_to_row;
        label_to_row.reserve(count);
        for (Id row = 0; row < count; ++row)
            Require(label_to_row.emplace(base.labels[row], row).second, "duplicate base label");
        std::vector<Id> insertion_to_row(count);
        std::vector<uint8_t> mapped(count, 0);
        std::vector<uint16_t> degrees(count);
        LadrGraph graph;
        graph.offsets.assign(count + 1, 0);
        for (Id internal = 0; internal < count; ++internal) {
            const size_t position = header_size + size_t(internal) * row_bytes;
            uint16_t degree = 0; int64_t label = 0;
            get(position, degree); get(position + label_offset, label);
            Require((static_cast<unsigned char>(data[position + 2]) & 1) == 0,
                    "deleted native HNSW rows are unsupported in this static experiment");
            Require(degree <= max_m0, "native HNSW neighbor count exceeds capacity");
            auto found = label_to_row.find(label);
            Require(found != label_to_row.end() && !mapped[found->second], "native HNSW label mismatch");
            Id row = found->second; mapped[row] = 1;
            insertion_to_row[internal] = row;
            degrees[internal] = degree;
            graph.offsets[row + 1] = degree;
        }
        std::partial_sum(graph.offsets.begin(), graph.offsets.end(), graph.offsets.begin());
        graph.neighbors.resize(graph.offsets.back());
        for (Id internal = 0; internal < count; ++internal) {
            const Id row = insertion_to_row[internal];
            const size_t position = header_size + size_t(internal) * row_bytes + 4;
            for (size_t j = 0; j < degrees[internal]; ++j) {
                Id other = 0; get(position + j * sizeof(Id), other);
                Require(other < count, "native HNSW neighbor ID out of range");
                graph.neighbors[graph.offsets[row] + j] = insertion_to_row[other];
            }
        }
        graph.Validate(count);
        return graph;
    }
 private:
    static constexpr uint64_t kMagic = 0x3147505244414c55ULL;
};

class Ladr {
 public:
    struct Workspace {
        std::vector<uint32_t> seen, expanded;
        uint32_t generation = 0;
        std::vector<Id> pending;
        std::vector<float> distances;
        std::vector<Hit> scored;
        size_t rounds = 0;
        void Reset(size_t count) {
            if (seen.size() != count) {
                seen.assign(count, 0); expanded.assign(count, 0); generation = 0;
            }
            if (++generation == 0) {
                std::fill(seen.begin(), seen.end(), 0);
                std::fill(expanded.begin(), expanded.end(), 0);
                generation = 1;
            }
            pending.clear(); distances.clear(); scored.clear(); rounds = 0;
        }
    };
    Ladr(const Vectors& base, const LadrGraph& graph) : base_(base), graph_(graph) {
        Require(graph.size() == base.size(), "LADR graph/base mismatch");
    }

    // Algorithms 1/2 in arXiv:2307.16779. Lexical seed retrieval is performed
    // by the caller INSIDE its search timer. alpha only affects the separately
    // labelled final hybrid rerank; graph exploration is always dense-scored.
    std::vector<Hit> Search(const float* qd, const Sparse& qs, float alpha, size_t k,
                           const std::vector<Id>& seeds, size_t depth, size_t neighbors,
                           bool adaptive, bool hybrid_rerank, Workspace& work, Stats& stats) const {
        Require(k > 0 && k <= base_.size() && depth > 0 && neighbors > 0,
                "invalid LADR search settings");
        Require(alpha >= 0 && alpha <= 1, "invalid LADR alpha");
        work.Reset(base_.size());
        Worst frontier;
        AlignedScorer scorer(base_, qd, qs, alpha, stats);
        auto enqueue = [&](Id id) {
            Require(id < base_.size(), "invalid LADR seed/neighbor");
            if (work.seen[id] != work.generation) {
                work.seen[id] = work.generation;
                work.pending.push_back(id);
            }
        };
        auto expand = [&](Id id) {
            if (work.expanded[id] == work.generation) return;
            work.expanded[id] = work.generation;
            ++stats.dense_expansions;
            auto [row, count] = graph_.Row(id, neighbors);
            for (size_t j = 0; j < count; ++j) enqueue(row[j]);
        };
        auto score_pending = [&] {
            work.distances.resize(work.pending.size());
            scorer.ScoreBatch(work.pending.data(), work.pending.size(), work.distances.data(),
                              AlignedScorer::Metric::Dense);
            for (size_t j = 0; j < work.pending.size(); ++j) {
                Hit hit(work.distances[j], work.pending[j]);
                work.scored.push_back(hit);
                if (adaptive) Offer(frontier, hit, depth);
            }
            work.pending.clear();
        };
        for (Id id : seeds) enqueue(id);
        if (!adaptive) {
            for (Id id : seeds) expand(id);
            if (!work.pending.empty()) { score_pending(); work.rounds = 1; }
        } else {
            score_pending();
            while (!frontier.empty()) {
                // A whole round expands the CURRENT top-c set before any of
                // its neighbors are scored. This is not an HNSW beam search.
                auto current = frontier;
                std::vector<Id> best;
                while (!current.empty()) { best.push_back(current.top().second); current.pop(); }
                std::reverse(best.begin(), best.end());
                for (Id id : best) expand(id);
                if (work.pending.empty()) break;
                score_pending();
                ++work.rounds;
            }
        }
        stats.scanned += work.scored.size();
        Worst result;
        for (auto hit : work.scored) {
            if (hybrid_rerank) {
                const float sparse_distance = 1.0F - SparseDot(qs, base_.sparse[hit.second]);
                hit.first = alpha * hit.first + (1.0F - alpha) * sparse_distance;
                ++stats.sparse;
            }
            Offer(result, hit, k);
        }
        return Sorted(result, k);
    }
 private:
    const Vectors& base_;
    const LadrGraph& graph_;
};
}  // namespace paper_baselines
