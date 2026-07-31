// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// UHG graph construction for parameter tuning (bk, query_prune_ratio).
//
// Based on 706_uhg_exp6, but only processes a subset of points (default 10000)
// to speed up parameter sweeps.  Adds a --brute_force mode that skips the
// HGRAPH+SINDI candidate generation and instead computes exact hybrid
// neighbors against the full dataset, yielding a ground-truth graph for
// quality comparison.

#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct HybridResult {
    int64_t id;
    float dense_score;
    float sparse_score;
    float hybrid_score;

    void
    ComputeHybridScore(float alpha) {
        hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score;
    }

    bool
    operator>(const HybridResult& other) const {
        return hybrid_score > other.hybrid_score;
    }
};

// ---------------------------------------------------------------------------
// Sparse vector parsing
// ---------------------------------------------------------------------------

std::vector<vsag::SparseVector>
ParseSparseVectors(const std::vector<uint8_t>& blob) {
    std::vector<vsag::SparseVector> result;
    size_t offset = 0;

    while (offset < blob.size()) {
        if (offset + sizeof(uint32_t) > blob.size()) break;

        uint32_t len;
        std::memcpy(&len, &blob[offset], sizeof(uint32_t));
        offset += sizeof(uint32_t);

        vsag::SparseVector vec;
        vec.len_ = len;

        if (len > 0) {
            size_t ids_bytes = len * sizeof(uint32_t);
            if (offset + ids_bytes > blob.size()) break;

            vec.ids_ = new uint32_t[len];
            std::memcpy(vec.ids_, &blob[offset], ids_bytes);
            offset += ids_bytes;

            size_t vals_bytes = len * sizeof(float);
            if (offset + vals_bytes > blob.size()) {
                delete[] vec.ids_;
                break;
            }

            vec.vals_ = new float[len];
            std::memcpy(vec.vals_, &blob[offset], vals_bytes);
            offset += vals_bytes;
        } else {
            vec.ids_ = nullptr;
            vec.vals_ = nullptr;
        }

        result.push_back(vec);
    }

    return result;
}

void
FreeSparseVectors(std::vector<vsag::SparseVector>& sparse_vectors) {
    for (auto& vec : sparse_vectors) {
        delete[] vec.ids_;
        delete[] vec.vals_;
        vec.ids_ = nullptr;
        vec.vals_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Distance / score computation
// ---------------------------------------------------------------------------

float
ComputeDenseIP(const float* vec1, const float* vec2, int64_t dim) {
    float score = 0;
    for (int64_t i = 0; i < dim; i++) {
        score += vec1[i] * vec2[i];
    }
    return score;
}

float
ComputeSparseIP(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2) {
    float score = 0.0f;
    uint32_t i = 0, j = 0;
    while (i < vec1.len_ && j < vec2.len_) {
        if (vec1.ids_[i] == vec2.ids_[j]) {
            score += vec1.vals_[i] * vec2.vals_[j];
            ++i;
            ++j;
        } else if (vec1.ids_[i] < vec2.ids_[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return score;
}

float
ComputeHybridDistance(int64_t id_a,
                      int64_t id_b,
                      const std::vector<float>& train_dense,
                      const std::vector<vsag::SparseVector>& train_sparse,
                      int64_t dense_dim,
                      float alpha) {
    int64_t num_train = static_cast<int64_t>(train_sparse.size());
    if (id_a < 0 || id_b < 0 || id_a >= num_train || id_b >= num_train) {
        return std::numeric_limits<float>::max();
    }

    float dense_score = ComputeDenseIP(train_dense.data() + id_a * dense_dim,
                                       train_dense.data() + id_b * dense_dim,
                                       dense_dim);
    float sparse_score = ComputeSparseIP(train_sparse[id_a], train_sparse[id_b]);

    return -(alpha * dense_score + (1.0f - alpha) * sparse_score);
}

// ---------------------------------------------------------------------------
// Refine helpers (identical to 706)
// ---------------------------------------------------------------------------

std::vector<int64_t>
PruneNeighborsByRNG(int64_t query_id,
                     const std::vector<int64_t>& candidates,
                     const std::vector<float>& train_dense,
                     const std::vector<vsag::SparseVector>& train_sparse,
                     int64_t dense_dim,
                     int max_degree,
                     float alpha_eval,
                     float alpha_rng) {
    struct Candidate {
        int64_t id;
        float distance;
    };

    std::vector<Candidate> sorted_candidates;
    sorted_candidates.reserve(candidates.size());
    std::unordered_set<int64_t> seen;
    seen.reserve(candidates.size());

    int64_t num_train = static_cast<int64_t>(train_sparse.size());
    for (int64_t cid : candidates) {
        if (cid == query_id || cid < 0 || cid >= num_train || !seen.insert(cid).second) {
            continue;
        }

        float distance = ComputeHybridDistance(
            query_id, cid, train_dense, train_sparse, dense_dim, alpha_eval);
        sorted_candidates.push_back({cid, distance});
    }

    std::sort(sorted_candidates.begin(), sorted_candidates.end(),
              [](const auto& a, const auto& b) { return a.distance < b.distance; });

    std::vector<int64_t> result;
    result.reserve(max_degree);

    for (const auto& candidate : sorted_candidates) {
        if (static_cast<int>(result.size()) >= max_degree) {
            break;
        }

        bool dominated = false;
        for (int64_t selected_id : result) {
            float selected_candidate_distance = ComputeHybridDistance(
                selected_id, candidate.id, train_dense, train_sparse, dense_dim, alpha_eval);
            if (selected_candidate_distance < alpha_rng * candidate.distance) {
                dominated = true;
                break;
            }
        }

        if (!dominated) {
            result.push_back(candidate.id);
        }
    }

    return result;
}

void
AddNeighborForced(std::vector<int64_t>& neighbors, int64_t neighbor_id, int max_degree) {
    if (std::find(neighbors.begin(), neighbors.end(), neighbor_id) != neighbors.end()) {
        return;
    }

    if (static_cast<int>(neighbors.size()) < max_degree) {
        neighbors.push_back(neighbor_id);
    } else if (!neighbors.empty()) {
        neighbors[0] = neighbor_id;
    }
}

void
AddReverseEdgesWithPrune(std::vector<std::vector<int64_t>>& graph,
                         const std::vector<float>& train_dense,
                         const std::vector<vsag::SparseVector>& train_sparse,
                         int64_t dense_dim,
                         int max_degree,
                         float alpha_eval,
                         float alpha_rng) {
    int64_t n = static_cast<int64_t>(graph.size());
    std::vector<std::vector<int64_t>> reverse_candidates(n);

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j : graph[i]) {
            if (j >= 0 && j < n) {
                reverse_candidates[j].push_back(i);
            }
        }
    }

    int64_t refined_count = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : refined_count)
    for (int64_t j = 0; j < n; j++) {
        if (reverse_candidates[j].empty() && static_cast<int>(graph[j].size()) <= max_degree) {
            continue;
        }

        std::unordered_set<int64_t> merged(graph[j].begin(), graph[j].end());
        for (int64_t reverse_id : reverse_candidates[j]) {
            merged.insert(reverse_id);
        }

        graph[j] = PruneNeighborsByRNG(j,
                                       std::vector<int64_t>(merged.begin(), merged.end()),
                                       train_dense,
                                       train_sparse,
                                       dense_dim,
                                       max_degree,
                                       alpha_eval,
                                       alpha_rng);
        refined_count++;
    }

    std::cout << "[Refine] Reverse-edge RNG refined nodes: " << refined_count << " / " << n
              << std::endl;
}

void
EnsureConnectivity(std::vector<std::vector<int64_t>>& graph,
                   const std::vector<float>& train_dense,
                   const std::vector<vsag::SparseVector>& train_sparse,
                   int64_t dense_dim,
                   int max_degree,
                   float alpha_eval) {
    int n = static_cast<int>(graph.size());
    if (n == 0) {
        return;
    }

    int entry = 0;
    std::vector<bool> visited(n, false);
    std::vector<int> stack;
    stack.push_back(entry);
    visited[entry] = true;
    int visited_count = 1;

    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        for (int64_t nb : graph[cur]) {
            if (nb >= 0 && nb < static_cast<int64_t>(graph.size()) && !visited[nb]) {
                visited[nb] = true;
                visited_count++;
                stack.push_back(static_cast<int>(nb));
            }
        }
    }

    std::cout << "[Refine] Reachable from entry " << entry << ": " << visited_count << " / " << n
              << std::endl;

    if (visited_count == n) {
        std::cout << "[Refine] Graph is already fully connected." << std::endl;
        return;
    }

    int fixed_count = 0;
    for (int i = 0; i < n; i++) {
        if (visited[i]) {
            continue;
        }

        float best_distance = std::numeric_limits<float>::max();
        int best_j = entry;
        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                continue;
            }

            float distance = ComputeHybridDistance(
                i, j, train_dense, train_sparse, dense_dim, alpha_eval);
            if (distance < best_distance) {
                best_distance = distance;
                best_j = j;
            }
        }

        AddNeighborForced(graph[i], best_j, max_degree);
        AddNeighborForced(graph[best_j], i, max_degree);

        visited[i] = true;
        visited_count++;
        stack.push_back(i);
        while (!stack.empty()) {
            int cur = stack.back();
            stack.pop_back();
            for (int64_t nb : graph[cur]) {
                if (nb >= 0 && nb < static_cast<int64_t>(graph.size()) && !visited[nb]) {
                    visited[nb] = true;
                    visited_count++;
                    stack.push_back(static_cast<int>(nb));
                }
            }
        }

        fixed_count++;
    }

    std::cout << "[Refine] Fixed disconnected nodes: " << fixed_count << std::endl;
    std::cout << "[Refine] Final reachable: " << visited_count << " / " << n << std::endl;
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

void
PrintGraphStatistics(const std::string& name,
                     const std::vector<std::vector<int64_t>>& graph) {
    if (graph.empty()) {
        std::cout << name << ": empty graph" << std::endl;
        return;
    }

    double avg_neighbors = 0.0;
    int max_count = 0;
    int min_count = INT_MAX;
    for (const auto& neighbors : graph) {
        int count = static_cast<int>(neighbors.size());
        avg_neighbors += count;
        max_count = std::max(max_count, count);
        min_count = std::min(min_count, count);
    }

    std::cout << name << ":\n";
    std::cout << "  avg neighbors: " << avg_neighbors / graph.size() << "\n";
    std::cout << "  max neighbors: " << max_count << "\n";
    std::cout << "  min neighbors: " << min_count << "\n";
}

void
SaveNeighborsToHDF5(const std::string& filename,
                    const std::vector<std::vector<int64_t>>& all_neighbors,
                    const std::vector<int64_t>& point_ids) {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);

        size_t max_neighbors = 0;
        for (const auto& neighbors : all_neighbors) {
            max_neighbors = std::max(max_neighbors, neighbors.size());
        }

        std::cout << "Max neighbors count: " << max_neighbors << std::endl;

        hsize_t dims[2] = {all_neighbors.size(), max_neighbors};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = file.createDataSet("neighbors",
                                                  H5::PredType::NATIVE_INT64,
                                                  dataspace);

        std::vector<int64_t> flat_data(all_neighbors.size() * max_neighbors, -1);
        for (size_t i = 0; i < all_neighbors.size(); i++) {
            for (size_t j = 0; j < all_neighbors[i].size(); j++) {
                flat_data[i * max_neighbors + j] = all_neighbors[i][j];
            }
        }

        dataset.write(flat_data.data(), H5::PredType::NATIVE_INT64);

        hsize_t count_dims[1] = {all_neighbors.size()};
        H5::DataSpace count_space(1, count_dims);
        H5::DataSet count_dataset = file.createDataSet("neighbor_counts",
                                                        H5::PredType::NATIVE_INT64,
                                                        count_space);

        std::vector<int64_t> counts;
        for (const auto& neighbors : all_neighbors) {
            counts.push_back(static_cast<int64_t>(neighbors.size()));
        }
        count_dataset.write(counts.data(), H5::PredType::NATIVE_INT64);

        // Save the sampled point ids so the consumer knows which rows
        // in the full dataset these neighbors correspond to.
        hsize_t id_dims[1] = {point_ids.size()};
        H5::DataSpace id_space(1, id_dims);
        H5::DataSet id_dataset = file.createDataSet("point_ids",
                                                     H5::PredType::NATIVE_INT64,
                                                     id_space);
        id_dataset.write(point_ids.data(), H5::PredType::NATIVE_INT64);

        std::cout << "Saved to " << filename << std::endl;

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        throw;
    }
}

std::string
GetDatasetName(const std::string& h5_file) {
    std::string filename = h5_file;
    size_t last_slash = filename.find_last_of('/');
    if (last_slash != std::string::npos) {
        filename = filename.substr(last_slash + 1);
    }
    size_t last_dot = filename.find_last_of('.');
    return (last_dot != std::string::npos) ? filename.substr(0, last_dot) : filename;
}

bool
FileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Config {
    std::string h5_file;
    std::string output_file = "uhg_param_graph.h5";
    std::string index_dir = "./indexes";
    int k = 32;
    int bk = 500;
    float alpha_step = 0.1f;
    int num_points = 10000;
    int threads = 8;
    bool rebuild = false;
    float term_prune_ratio = 0.0f;
    float query_prune_ratio = 0.5f;
    int refine_max_degree = 64;
    float refine_alpha = 0.5f;
    float refine_rng = 1.2f;
    bool skip_connectivity = false;
    bool skip_refine = false;
    bool brute_force = false;
    unsigned seed = 42;
};

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nUHG graph construction for parameter tuning (bk, query_prune_ratio).\n"
              << "Processes a random subset of points to speed up sweeps.\n"
              << "\nOptions:\n"
              << "  -k <int>                   Top-k neighbors per alpha (default: 32)\n"
              << "  --bk <int>                 Candidate pool size (default: 500)\n"
              << "  --alpha_step <float>       Step for alpha sampling (default: 0.1)\n"
              << "  --output <path>            Output HDF5 file (default: uhg_param_graph.h5)\n"
              << "  --num_points <int>         Points to sample and process (default: 10000)\n"
              << "  --index_dir <path>         Index cache directory (default: ./indexes)\n"
              << "  --threads <int>            Number of threads (default: 8)\n"
               << "  --query_prune_ratio <float>\n"
               << "                              SINDI query pruning ratio (default: 0.5)\n"
               << "  --term_prune_ratio <float>\n"
               << "                              SINDI term pruning ratio (default: 0.0)\n"
              << "  --refine_max_degree <int>  Max degree after refine (default: 64)\n"
              << "  --refine_alpha <float>     Alpha for refine distance (default: 0.5)\n"
              << "  --refine_rng <float>       RNG prune scale (default: 1.2)\n"
              << "  --skip_connectivity        Skip DFS connectivity fallback\n"
              << "  --skip_refine              Skip the entire refine stage\n"
              << "  --brute_force              Skip index-based candidate generation;\n"
              << "                              compute exact hybrid neighbors against all\n"
              << "                              N points (ground-truth graph). Ignores --bk\n"
              << "                              and --query_prune_ratio.\n"
              << "  --seed <uint>              Random seed for sampling (default: 42)\n"
              << "  --rebuild                  Force rebuild indices\n"
              << "  --help, -h                 Show this help\n"
              << "\nExamples:\n"
              << "  # Index-based, sweep bk=200, no refine\n"
              << "  " << program_name
              << " nq.hdf5 --bk 200 --skip_refine --index_dir ./indexes\n"
              << "  # Brute-force ground-truth graph on 5000 points\n"
              << "  " << program_name
              << " nq.hdf5 --brute_force --num_points 5000 --skip_refine\n"
              << std::endl;
}

Config
ParseCommandLine(int argc, char** argv) {
    Config config;

    if (argc < 2) {
        PrintUsage(argv[0]);
        exit(1);
    }

    // Check for --help/-h anywhere before treating argv[1] as a file.
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            exit(0);
        }
    }

    config.h5_file = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            exit(0);
        } else if (arg == "-k" && i + 1 < argc) {
            config.k = std::atoi(argv[++i]);
        } else if (arg == "--bk" && i + 1 < argc) {
            config.bk = std::atoi(argv[++i]);
        } else if (arg == "--alpha_step" && i + 1 < argc) {
            config.alpha_step = std::atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--num_points" && i + 1 < argc) {
            config.num_points = std::atoi(argv[++i]);
        } else if (arg == "--index_dir" && i + 1 < argc) {
            config.index_dir = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::atoi(argv[++i]);
        } else if (arg == "--rebuild") {
            config.rebuild = true;
        } else if (arg == "--term_prune_ratio" && i + 1 < argc) {
            config.term_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--query_prune_ratio" && i + 1 < argc) {
            config.query_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--refine_max_degree" && i + 1 < argc) {
            config.refine_max_degree = std::atoi(argv[++i]);
        } else if (arg == "--refine_alpha" && i + 1 < argc) {
            config.refine_alpha = std::atof(argv[++i]);
        } else if (arg == "--refine_rng" && i + 1 < argc) {
            config.refine_rng = std::atof(argv[++i]);
        } else if (arg == "--skip_connectivity") {
            config.skip_connectivity = true;
        } else if (arg == "--skip_refine") {
            config.skip_refine = true;
        } else if (arg == "--brute_force") {
            config.brute_force = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }

    if (config.k <= 0 || config.bk <= 0 || config.alpha_step <= 0.0f ||
        config.refine_max_degree <= 0) {
        std::cerr << "Error: k, bk, alpha_step, and refine_max_degree must be positive"
                  << std::endl;
        exit(1);
    }
    if (config.refine_alpha < 0.0f || config.refine_alpha > 1.0f) {
        std::cerr << "Error: refine_alpha must be between 0.0 and 1.0" << std::endl;
        exit(1);
    }
    if (config.num_points <= 0) {
        std::cerr << "Error: num_points must be positive" << std::endl;
        exit(1);
    }

    return config;
}

// ---------------------------------------------------------------------------
// Neighbor generation: index-based (same as 706)
// ---------------------------------------------------------------------------

struct GenResult {
    std::vector<int64_t> neighbors;
    int64_t dense_us{0};
    int64_t sparse_us{0};
    int64_t merge_us{0};
};

GenResult
GenerateNeighborsIndex(int64_t i,
                       const std::vector<float>& train_dense,
                       const std::vector<vsag::SparseVector>& train_sparse,
                       int64_t dense_dim,
                       int64_t num_train,
                       const std::shared_ptr<vsag::Index>& dense_index,
                       const std::shared_ptr<vsag::Index>& sparse_index,
                       const std::string& dense_search_params,
                       const std::string& sparse_search_params,
                       int bk,
                       int k,
                       const std::vector<float>& alpha_values) {
    GenResult gr;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dense_dim)
        ->Float32Vectors(train_dense.data() + i * dense_dim)
        ->SparseVectors(train_sparse.data() + i)
        ->Owner(false);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto dense_results = dense_index->KnnSearch(query, bk, dense_search_params).value();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto sparse_results = sparse_index->KnnSearch(query, bk, sparse_search_params).value();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::unordered_map<int64_t, HybridResult> candidate_map;
    candidate_map.reserve(static_cast<size_t>(bk) * 2);

    for (int j = 0; j < dense_results->GetDim(); j++) {
        int64_t id = dense_results->GetIds()[j];
        if (id == i) continue;
        float ds = 1.0f - dense_results->GetDistances()[j];
        candidate_map[id] = {id, ds, 0.0f, 0.0f};
    }

    for (int j = 0; j < sparse_results->GetDim(); j++) {
        int64_t id = sparse_results->GetIds()[j];
        if (id == i) continue;
        float ss = 1.0f - sparse_results->GetDistances()[j];
        auto it = candidate_map.find(id);
        if (it != candidate_map.end()) {
            it->second.sparse_score = ss;
        } else {
            candidate_map[id] = {id, 0.0f, ss, 0.0f};
        }
    }

    for (auto& [cid, c] : candidate_map) {
        if (c.dense_score == 0.0f) {
            c.dense_score = ComputeDenseIP(train_dense.data() + i * dense_dim,
                                           train_dense.data() + cid * dense_dim,
                                           dense_dim);
        }
        if (c.sparse_score == 0.0f) {
            c.sparse_score = ComputeSparseIP(train_sparse[i], train_sparse[cid]);
        }
    }

    std::vector<HybridResult> candidates;
    candidates.reserve(candidate_map.size());
    for (auto& [cid, c] : candidate_map) {
        candidates.push_back(c);
    }

    std::unordered_set<int64_t> merged_neighbors;

    for (float alpha : alpha_values) {
        for (auto& c : candidates) {
            c.ComputeHybridScore(alpha);
        }

        int actual_k = std::min(k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(),
                          candidates.begin() + actual_k,
                          candidates.end(),
                          std::greater<HybridResult>());

        for (int j = 0; j < actual_k; j++) {
            merged_neighbors.insert(candidates[j].id);
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    gr.dense_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    gr.sparse_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    gr.merge_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    gr.neighbors.assign(merged_neighbors.begin(), merged_neighbors.end());
    std::sort(gr.neighbors.begin(), gr.neighbors.end());
    return gr;
}

// ---------------------------------------------------------------------------
// Neighbor generation: brute-force (exact)
// ---------------------------------------------------------------------------

GenResult
GenerateNeighborsBruteForce(int64_t i,
                            const std::vector<float>& train_dense,
                            const std::vector<vsag::SparseVector>& train_sparse,
                            int64_t dense_dim,
                            int64_t num_train,
                            int k,
                            const std::vector<float>& alpha_values) {
    GenResult gr;
    const float* qi_dense = train_dense.data() + i * dense_dim;
    const vsag::SparseVector& qi_sparse = train_sparse[i];

    // Pre-compute dense and sparse scores against all points once.
    std::vector<float> dense_scores(num_train);
    std::vector<float> sparse_scores(num_train);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int64_t j = 0; j < num_train; j++) {
        if (j == i) {
            dense_scores[j] = std::numeric_limits<float>::lowest();
            sparse_scores[j] = std::numeric_limits<float>::lowest();
            continue;
        }
        dense_scores[j] = ComputeDenseIP(qi_dense, train_dense.data() + j * dense_dim, dense_dim);
        sparse_scores[j] = ComputeSparseIP(qi_sparse, train_sparse[j]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::unordered_set<int64_t> merged_neighbors;

    for (float alpha : alpha_values) {
        std::vector<std::pair<int64_t, float>> scored;
        scored.reserve(num_train);
        for (int64_t j = 0; j < num_train; j++) {
            if (j == i) continue;
            float hybrid = alpha * dense_scores[j] + (1.0f - alpha) * sparse_scores[j];
            scored.emplace_back(j, hybrid);
        }

        int actual_k = std::min(k, static_cast<int>(scored.size()));
        std::partial_sort(scored.begin(),
                          scored.begin() + actual_k,
                          scored.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });

        for (int j = 0; j < actual_k; j++) {
            merged_neighbors.insert(scored[j].first);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    gr.dense_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    gr.merge_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    gr.neighbors.assign(merged_neighbors.begin(), merged_neighbors.end());
    std::sort(gr.neighbors.begin(), gr.neighbors.end());
    return gr;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int
main(int argc, char** argv) {
    vsag::init();

    Config config = ParseCommandLine(argc, argv);

    try {
        std::string dataset_name = GetDatasetName(config.h5_file);

        std::cout << "\n========== UHG Parameter-Tuning Graph Construction ==========\n";
        std::cout << "Input: " << config.h5_file << "\n";
        std::cout << "Output: " << config.output_file << "\n";
        std::cout << "Dataset: " << dataset_name << "\n";
        std::cout << "Mode: " << (config.brute_force ? "BRUTE FORCE" : "INDEX-BASED") << "\n";
        std::cout << "k: " << config.k << "\n";
        std::cout << "bk: " << config.bk << (config.brute_force ? "  (ignored in brute_force)" : "") << "\n";
        std::cout << "alpha_step: " << config.alpha_step << "\n";
        std::cout << "num_points: " << config.num_points << "\n";
        std::cout << "threads: " << config.threads << "\n";
        if (!config.brute_force) {
            std::cout << "query_prune_ratio: " << config.query_prune_ratio << "\n";
            std::cout << "term_prune_ratio: " << config.term_prune_ratio << "\n";
        }
        std::cout << "refine_max_degree: " << config.refine_max_degree << "\n";
        std::cout << "refine_alpha: " << config.refine_alpha << "\n";
        std::cout << "refine_rng: " << config.refine_rng << "\n";
        std::cout << "skip_connectivity: " << (config.skip_connectivity ? "true" : "false") << "\n";
        std::cout << "skip_refine: " << (config.skip_refine ? "true" : "false") << "\n";
        std::cout << "seed: " << config.seed << "\n";
        std::cout << "rebuild: " << (config.rebuild ? "true" : "false") << "\n";

        /******************* 1. Load Dataset *****************/
        std::cout << "\nLoading dataset..." << std::endl;
        H5::H5File file(config.h5_file, H5F_ACC_RDONLY);

        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();
        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims);
        int64_t num_train = static_cast<int64_t>(train_dims[0]);
        int64_t dense_dim = static_cast<int64_t>(train_dims[1]);

        std::vector<float> train_dense(num_train * dense_dim);
        train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
        hsize_t train_sparse_size = train_sparse_dataset.getSpace().getSimpleExtentNpoints();
        std::vector<uint8_t> train_sparse_blob(train_sparse_size);
        train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto train_sparse = ParseSparseVectors(train_sparse_blob);

        // train_labels are loaded but only used for index build; in brute_force
        // mode we work with row offsets directly.
        std::vector<int64_t> train_labels;
        if (!config.brute_force || !config.skip_refine) {
            H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
            train_labels.resize(num_train);
            train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);
        }

        if (static_cast<int64_t>(train_sparse.size()) != num_train) {
            throw std::runtime_error("train_sparse count does not match train dense count");
        }

        std::cout << "Dataset: train=" << num_train << ", dim=" << dense_dim << std::endl;

        /******************* 2. Sample points *****************/
        int num_process = std::min(config.num_points, static_cast<int>(num_train));
        std::mt19937 rng(config.seed);
        std::vector<int64_t> point_ids;
        if (num_process < num_train) {
            std::vector<int64_t> all_ids(num_train);
            std::iota(all_ids.begin(), all_ids.end(), 0);
            std::shuffle(all_ids.begin(), all_ids.end(), rng);
            point_ids.assign(all_ids.begin(), all_ids.begin() + num_process);
        } else {
            point_ids.resize(num_train);
            std::iota(point_ids.begin(), point_ids.end(), 0);
            num_process = static_cast<int>(num_train);
        }
        std::cout << "Sampled " << num_process << " / " << num_train << " points" << std::endl;

        /******************* 3. Build/Load Indices (index mode only) *****************/
        vsag::Resource* resource = nullptr;
        vsag::Engine* engine = nullptr;
        std::shared_ptr<vsag::Index> dense_index;
        std::shared_ptr<vsag::Index> sparse_index;
        std::string dense_search_params;
        std::string sparse_search_params;

        if (!config.brute_force) {
            std::string dense_index_path =
                config.index_dir + "/701_" + dataset_name + "_dense_hgraph.index";
            std::string sparse_index_path =
                config.index_dir + "/701_" + dataset_name + "_sparse_sindi.index";

            bool need_build_dense = config.rebuild || !FileExists(dense_index_path);
            bool need_build_sparse = config.rebuild || !FileExists(sparse_index_path);

            auto base = vsag::Dataset::Make();
            base->NumElements(num_train)
                ->Dim(dense_dim)
                ->Ids(train_labels.data())
                ->Float32Vectors(train_dense.data())
                ->SparseVectors(train_sparse.data())
                ->Owner(false);

            std::string hgraph_build_params = R"(
            {
                "dtype": "float32",
                "metric_type": "ip",
                "dim": )" + std::to_string(dense_dim) + R"(,
                "index_param": {
                    "base_quantization_type": "sq8",
                    "max_degree": 64,
                    "ef_construction": 200
                }
            })";

            resource = new vsag::Resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
            engine = new vsag::Engine(resource);
            dense_index = engine->CreateIndex("hgraph", hgraph_build_params).value();

            if (need_build_dense) {
                std::cout << "\nBuilding HGRAPH..." << std::endl;
                auto build_start = std::chrono::high_resolution_clock::now();
                if (!dense_index->Build(base).has_value()) {
                    std::cerr << "Failed to build HGRAPH" << std::endl;
                    return -1;
                }
                std::cout << "HGRAPH built in "
                          << std::chrono::duration<double>(
                                 std::chrono::high_resolution_clock::now() - build_start).count()
                          << "s" << std::endl;

                std::ofstream out_stream(dense_index_path);
                dense_index->Serialize(out_stream);
                out_stream.close();
                std::cout << "HGRAPH saved to " << dense_index_path << std::endl;
            } else {
                std::cout << "\nLoading HGRAPH from " << dense_index_path << std::endl;
                dense_index = engine->CreateIndex("hgraph", hgraph_build_params).value();
                std::ifstream in_stream(dense_index_path);
                dense_index->Deserialize(in_stream);
                in_stream.close();
                std::cout << "HGRAPH loaded (" << dense_index->GetNumElements() << " vectors)"
                          << std::endl;
            }

            std::string sindi_build_params = R"(
            {
                "dtype": "sparse",
                "metric_type": "ip",
                "index_param": {"use_reorder": true}
            })";

            sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();

            if (need_build_sparse) {
                std::cout << "\nBuilding SINDI..." << std::endl;
                auto build_start = std::chrono::high_resolution_clock::now();
                if (!sparse_index->Build(base).has_value()) {
                    std::cerr << "Failed to build SINDI" << std::endl;
                    return -1;
                }
                std::cout << "SINDI built in "
                          << std::chrono::duration<double>(
                                 std::chrono::high_resolution_clock::now() - build_start).count()
                          << "s" << std::endl;

                std::ofstream out_stream(sparse_index_path);
                sparse_index->Serialize(out_stream);
                out_stream.close();
                std::cout << "SINDI saved to " << sparse_index_path << std::endl;
            } else {
                std::cout << "\nLoading SINDI from " << sparse_index_path << std::endl;
                sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();
                std::ifstream in_stream(sparse_index_path);
                sparse_index->Deserialize(in_stream);
                in_stream.close();
                std::cout << "SINDI loaded (" << sparse_index->GetNumElements() << " vectors)"
                          << std::endl;
            }

            int ef_search = std::max(config.bk, 100);
            dense_search_params =
                R"({"hgraph": {"ef_search": )" + std::to_string(ef_search) + R"(}})";
            sparse_search_params = R"({"sindi": {"term_prune_ratio": )" +
                                   std::to_string(config.term_prune_ratio) +
                                   R"(, "query_prune_ratio": )" +
                                   std::to_string(config.query_prune_ratio) + "}}";
            std::cout << "sparse_search_params: " << sparse_search_params << std::endl;
        }

        /******************* 4. Generate neighbors *****************/
        std::vector<float> alpha_values;
        for (float a = 0.0f; a <= 1.0f + config.alpha_step / 2; a += config.alpha_step) {
            alpha_values.push_back(a);
        }
        std::cout << "\nAlpha values: ";
        for (float a : alpha_values) std::cout << a << " ";
        std::cout << std::endl;

        std::vector<std::vector<int64_t>> all_neighbors(num_process);

        omp_set_num_threads(config.threads);

        auto gen_start = std::chrono::high_resolution_clock::now();
        std::atomic<int> processed_count{0};
        std::atomic<int64_t> total_dense_us{0};
        std::atomic<int64_t> total_sparse_us{0};
        std::atomic<int64_t> total_merge_us{0};

#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < num_process; idx++) {
            int64_t i = point_ids[idx];

            GenResult gr;
            if (config.brute_force) {
                gr = GenerateNeighborsBruteForce(i,
                                                 train_dense,
                                                 train_sparse,
                                                 dense_dim,
                                                 num_train,
                                                 config.k,
                                                 alpha_values);
            } else {
                gr = GenerateNeighborsIndex(i,
                                            train_dense,
                                            train_sparse,
                                            dense_dim,
                                            num_train,
                                            dense_index,
                                            sparse_index,
                                            dense_search_params,
                                            sparse_search_params,
                                            config.bk,
                                            config.k,
                                            alpha_values);
            }

            all_neighbors[idx] = std::move(gr.neighbors);
            total_dense_us.fetch_add(gr.dense_us, std::memory_order_relaxed);
            total_sparse_us.fetch_add(gr.sparse_us, std::memory_order_relaxed);
            total_merge_us.fetch_add(gr.merge_us, std::memory_order_relaxed);

            int done = processed_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % 100 == 0) {
#pragma omp critical
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(now - gen_start).count();
                    double eta = elapsed / done * (num_process - done);
                    std::cout << "  [" << done << "/" << num_process << "] "
                              << elapsed << "s, eta:" << eta << "s" << std::endl;
                }
            }
        }

        auto gen_end = std::chrono::high_resolution_clock::now();
        double gen_seconds = std::chrono::duration<double>(gen_end - gen_start).count();
        std::cout << "\nGeneration time: " << gen_seconds << "s" << std::endl;
        std::cout << "  (" << num_process << " points, "
                  << (config.brute_force ? "brute_force" : "index-based") << " mode)" << std::endl;
        std::cout << "  "
                  << (config.brute_force ? "score computation" : "dense search")
                  << " total (thread-sum): " << total_dense_us.load() / 1e6 << "s" << std::endl;
        std::cout << "  sparse search total (thread-sum): " << total_sparse_us.load() / 1e6 << "s"
                  << std::endl;
        std::cout << "  merge total (thread-sum): " << total_merge_us.load() / 1e6 << "s"
                  << std::endl;

        /******************* 5. Refine (optional) *****************/
        PrintGraphStatistics("\nStatistics before refine", all_neighbors);

        if (config.skip_refine) {
            std::cout << "\nRefine stage skipped (--skip_refine)." << std::endl;
            std::cout << "Refine time: 0s" << std::endl;
        } else {
            auto refine_start = std::chrono::high_resolution_clock::now();
            AddReverseEdgesWithPrune(all_neighbors,
                                     train_dense,
                                     train_sparse,
                                     dense_dim,
                                     config.refine_max_degree,
                                     config.refine_alpha,
                                     config.refine_rng);
            if (!config.skip_connectivity) {
                EnsureConnectivity(all_neighbors,
                                   train_dense,
                                   train_sparse,
                                   dense_dim,
                                   config.refine_max_degree,
                                   config.refine_alpha);
            }
            auto refine_end = std::chrono::high_resolution_clock::now();
            double refine_seconds = std::chrono::duration<double>(refine_end - refine_start).count();
            std::cout << "\nRefine time: " << refine_seconds << "s" << std::endl;
            PrintGraphStatistics("\nStatistics after refine", all_neighbors);
            std::cout << "Total construction time: " << (gen_seconds + refine_seconds) << "s"
                      << std::endl;
        }

        /******************* 6. Save *****************/
        std::cout << "\nSaving to " << config.output_file << std::endl;
        SaveNeighborsToHDF5(config.output_file, all_neighbors, point_ids);

        std::cout << "\n========== Done ==========" << std::endl;

        FreeSparseVectors(train_sparse);

        // shared_ptr indices auto-release; just shutdown the engine.
        if (engine) {
            engine->Shutdown();
            delete engine;
        }
        if (resource) {
            delete resource;
        }

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
