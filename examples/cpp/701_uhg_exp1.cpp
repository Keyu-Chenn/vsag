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

#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "uhg_mixed_alpha_utils.h"

// Simple NPY file reader for int64 arrays
std::vector<int64_t>
ReadNPYInt64(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open npy file: " + path);
    }

    char magic[8];
    file.read(magic, 8);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Invalid npy file format: " + path);
    }

    uint8_t major = static_cast<uint8_t>(magic[6]);
    uint32_t header_len;
    if (major == 1) {
        uint16_t hlen;
        file.read(reinterpret_cast<char*>(&hlen), 2);
        header_len = hlen;
    } else if (major == 2) {
        file.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("Unsupported npy version: " + std::to_string(major));
    }

    std::string header(header_len, '\0');
    file.read(header.data(), header_len);

    shape.clear();
    size_t shape_pos = header.find("'shape':");
    if (shape_pos != std::string::npos) {
        size_t paren_start = header.find("(", shape_pos);
        size_t paren_end = header.find(")", paren_start);
        if (paren_start != std::string::npos && paren_end != std::string::npos) {
            std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);
            std::stringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                if (!token.empty()) {
                    shape.push_back(std::stoll(token));
                }
            }
        }
    }

    if (shape.empty()) {
        throw std::runtime_error("Failed to parse shape from npy header");
    }

    int64_t total = 1;
    for (auto s : shape) total *= s;

    std::vector<int64_t> data(total);
    file.read(reinterpret_cast<char*>(data.data()), total * sizeof(int64_t));
    return data;
}

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

float
CalculateRecall(const std::vector<int64_t>& results, const std::vector<int64_t>& ground_truth) {
    std::unordered_set<int64_t> gt_set(ground_truth.begin(), ground_truth.end());
    int hit_count = 0;
    for (auto id : results) {
        if (gt_set.count(id) > 0) {
            hit_count++;
        }
    }
    return static_cast<float>(hit_count) / ground_truth.size();
}

std::vector<int>
ParseSearchPoints(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::atoi(token.c_str()));
        }
    }
    return values;
}

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nOptions:\n"
              << "  -k, --topk <int>           Number of final top results (default: 10)\n"
              << "  --bk_dense <int>           Candidates from dense index (default: 100)\n"
              << "  --bk_sparse <int>          Candidates from sparse index (default: 100)\n"
              << "  --bk <int>                 Set both bk_dense and bk_sparse\n"
              << "  --alpha <float>            Weight for dense score (default: 0.5)\n"
              << "  --num_queries <int>        Number of queries to process (default: all)\n"
              << "  --threads <int>            Query-level parallel search threads (default: 1)\n"
              << "  --ef_search <int>          ef_search for hnsw (default: 200)\n"
              << "  --search_points <csv>      Run bk_dense=bk_sparse=ef_search points after one load\n"
              << "  --query_prune_ratio <float>\n"
              << "                              SINDI query pruning ratio, [0,0.9], 0=no pruning (default: 0)\n"
              << "  --term_prune_ratio <float>\n"
              << "                              SINDI term pruning ratio, [0,0.9], 0=no pruning (default: 0)\n"
              << "  --rebuild                  Force rebuild indexes\n"
              << "  --index_dir <path>         Directory to save/load indexes (default: ./indexes)\n"
              << "  --gt_dir <path>            Directory containing ground truth npy files (required)\n"
              << "  --mixed_alpha_file <path> Per-query alpha float32 npy; use with --mixed_gt_file\n"
              << "  --mixed_gt_file <path>    Ground truth int64 npy for mixed per-query alphas\n"
              << "  --help, -h                 Show this help message\n"
              << "\nIndex files: 701_dense_hnsw.index, 701_sparse_sindi.index\n"
              << "\nExample:\n"
              << "  " << program_name << " nq.h5 -k 10 --bk 200 --alpha 0.5 --gt_dir ./ground_truth/nq\n"
              << std::endl;
}

struct SearchParams {
    std::string h5_file;
    int k = 10;
    int bk_dense = 100;
    int bk_sparse = 100;
    float alpha = 0.5;
    int num_queries = -1;
    int threads = 1;
    int ef_search = 200;
    std::vector<int> search_points;
    float query_prune_ratio = 0.0f;
    float term_prune_ratio = 0.0f;
    bool rebuild = false;
    std::string index_dir = "./indexes";
    std::string gt_dir = "";
    std::string mixed_alpha_file;
    std::string mixed_gt_file;
};

SearchParams
ParseCommandLine(int argc, char** argv) {
    SearchParams params;

    if (argc < 2) {
        PrintUsage(argv[0]);
        exit(1);
    }

    std::string arg1 = argv[1];
    if (arg1 == "--help" || arg1 == "-h") {
        PrintUsage(argv[0]);
        exit(0);
    }

    params.h5_file = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            exit(0);
        } else if ((arg == "-k" || arg == "--topk") && i + 1 < argc) {
            params.k = std::atoi(argv[++i]);
        } else if (arg == "--bk" && i + 1 < argc) {
            int bk = std::atoi(argv[++i]);
            params.bk_dense = bk;
            params.bk_sparse = bk;
        } else if (arg == "--bk_dense" && i + 1 < argc) {
            params.bk_dense = std::atoi(argv[++i]);
        } else if (arg == "--bk_sparse" && i + 1 < argc) {
            params.bk_sparse = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            params.alpha = std::atof(argv[++i]);
        } else if (arg == "--num_queries" && i + 1 < argc) {
            params.num_queries = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            params.threads = std::atoi(argv[++i]);
        } else if (arg == "--ef_search" && i + 1 < argc) {
            params.ef_search = std::atoi(argv[++i]);
        } else if (arg == "--search_points" && i + 1 < argc) {
            params.search_points = ParseSearchPoints(argv[++i]);
        } else if (arg == "--query_prune_ratio" && i + 1 < argc) {
            params.query_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--term_prune_ratio" && i + 1 < argc) {
            params.term_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--rebuild") {
            params.rebuild = true;
        } else if (arg == "--index_dir" && i + 1 < argc) {
            params.index_dir = argv[++i];
        } else if (arg == "--gt_dir" && i + 1 < argc) {
            params.gt_dir = argv[++i];
        } else if (arg == "--mixed_alpha_file" && i + 1 < argc) {
            params.mixed_alpha_file = argv[++i];
        } else if (arg == "--mixed_gt_file" && i + 1 < argc) {
            params.mixed_gt_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }

    if (params.k <= 0 || params.bk_dense <= 0 || params.bk_sparse <= 0) {
        std::cerr << "Error: k, bk_dense, bk_sparse must be positive\n";
        exit(1);
    }
    if (params.threads <= 0) {
        std::cerr << "Error: threads must be positive\n";
        exit(1);
    }
    for (int point : params.search_points) {
        if (point <= 0) {
            std::cerr << "Error: search_points must be positive\n";
            exit(1);
        }
    }
    if (params.alpha < 0.0 || params.alpha > 1.0) {
        std::cerr << "Error: alpha must be between 0.0 and 1.0\n";
        exit(1);
    }
    return params;
}

bool
FileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
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

int
main(int argc, char** argv) {
    vsag::init();

    SearchParams params = ParseCommandLine(argc, argv);

    try {
        std::string dataset_name = GetDatasetName(params.h5_file);

        // Index paths with dataset name
        std::string dense_index_path = params.index_dir + "/701_" + dataset_name + "_dense_hnsw.index";
        std::string sparse_index_path = params.index_dir + "/701_" + dataset_name + "_sparse_sindi.index";

        bool need_build_dense = params.rebuild || !FileExists(dense_index_path);
        bool need_build_sparse = params.rebuild || !FileExists(sparse_index_path);
        bool need_build = need_build_dense || need_build_sparse;

        // Create index directory if needed
        if (need_build) {
            std::string cmd = "mkdir -p " + params.index_dir;
            system(cmd.c_str());
        }

        // Data containers (only loaded when needed)
        std::vector<float> train_dense;
        std::vector<vsag::SparseVector> train_sparse;
        std::vector<int64_t> train_labels;
        std::vector<float> test_dense;
        std::vector<vsag::SparseVector> test_sparse;
        int64_t num_train = 0;
        int64_t num_test = 0;
        int64_t dense_dim = 0;

        // Load data from HDF5 (only train data when rebuilding, test data always needed)
        std::cout << "Loading data from " << params.h5_file << std::endl;
        H5::H5File file(params.h5_file, H5F_ACC_RDONLY);

        // Always load test data (for queries)
        {
            H5::DataSet test_dataset = file.openDataSet("test");
            H5::DataSpace test_dataspace = test_dataset.getSpace();
            hsize_t test_dims[2];
            test_dataspace.getSimpleExtentDims(test_dims);
            num_test = test_dims[0];
            dense_dim = test_dims[1];

            test_dense.resize(num_test * dense_dim);
            test_dataset.read(test_dense.data(), H5::PredType::NATIVE_FLOAT);

            H5::DataSet test_sparse_dataset = file.openDataSet("test_sparse");
            H5::DataSpace test_sparse_dataspace = test_sparse_dataset.getSpace();
            hsize_t test_sparse_size = test_sparse_dataspace.getSimpleExtentNpoints();

            std::vector<uint8_t> test_sparse_blob(test_sparse_size);
            test_sparse_dataset.read(test_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
            test_sparse = ParseSparseVectors(test_sparse_blob);

            std::cout << "Test data loaded: test=" << num_test << ", dim=" << dense_dim << std::endl;
        }

        // Load train data only when rebuilding
        if (need_build) {
            std::cout << "Loading train data for index building..." << std::endl;

            H5::DataSet train_dataset = file.openDataSet("train");
            H5::DataSpace train_dataspace = train_dataset.getSpace();
            hsize_t train_dims[2];
            train_dataspace.getSimpleExtentDims(train_dims);
            num_train = train_dims[0];

            train_dense.resize(num_train * dense_dim);
            train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

            H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
            H5::DataSpace train_sparse_dataspace = train_sparse_dataset.getSpace();
            hsize_t train_sparse_size = train_sparse_dataspace.getSimpleExtentNpoints();

            std::vector<uint8_t> train_sparse_blob(train_sparse_size);
            train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
            train_sparse = ParseSparseVectors(train_sparse_blob);

            H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
            train_labels.resize(num_train);
            train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);

            std::cout << "Train data loaded: train=" << num_train << std::endl;
        }

        // Load ground truth
        const bool mixed_alpha_mode = uhg_mixed_alpha::ValidateMixedFiles(
            params.mixed_alpha_file, params.mixed_gt_file);
        std::vector<float> mixed_alphas;
        if (mixed_alpha_mode) {
            std::vector<int64_t> alpha_shape;
            mixed_alphas = uhg_mixed_alpha::ReadNPYFloat32(
                params.mixed_alpha_file, alpha_shape);
        }
        std::ostringstream gt_file_ss;
        if (mixed_alpha_mode) {
            gt_file_ss << params.mixed_gt_file;
        } else {
            gt_file_ss << params.gt_dir << "/" << dataset_name << "_ground_truth_alpha_"
                       << std::fixed << std::setprecision(1) << params.alpha << ".npy";
        }
        std::string gt_file_path = gt_file_ss.str();

        std::cout << "Loading ground truth from " << gt_file_path << std::endl;
        std::vector<int64_t> gt_shape;
        std::vector<int64_t> ground_truth = ReadNPYInt64(gt_file_path, gt_shape);

        if (gt_shape.size() != 2) {
            std::cerr << "Error: Ground truth must be 2D array (num_queries, k)\n";
            return -1;
        }
        int64_t num_gt_queries = gt_shape[0];
        int64_t k_gt = gt_shape[1];

        std::cout << "Ground truth loaded: queries=" << num_gt_queries << ", k=" << k_gt << std::endl;

        // Build/Load Dense Index (HNSW)
        std::string hnsw_build_params = R"(
        {
            "dtype": "float32",
            "metric_type": "ip",
            "dim": )" + std::to_string(dense_dim) + R"(,
            "hnsw": {
                "max_degree": 64,
                "ef_construction": 200
            }
        })";

        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);

        auto dense_index = engine.CreateIndex("hnsw", hnsw_build_params).value();

        if (need_build_dense) {
            std::cout << "Building HNSW index..." << std::endl;
            auto base_dataset = vsag::Dataset::Make();
            base_dataset->NumElements(num_train)
                ->Dim(dense_dim)
                ->Ids(train_labels.data())
                ->Float32Vectors(train_dense.data())
                ->Owner(false);

            auto build_start = std::chrono::high_resolution_clock::now();
            if (!dense_index->Build(base_dataset).has_value()) {
                std::cerr << "Failed to build HNSW index" << std::endl;
                return -1;
            }
            auto build_end = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double>(build_end - build_start).count();
            std::cout << "HNSW built in " << build_time << "s" << std::endl;

            std::ofstream out_stream(dense_index_path);
            auto serialize_result = dense_index->Serialize(out_stream);
            out_stream.close();
            if (!serialize_result.has_value()) {
                std::cerr << "Failed to save HNSW index: " << serialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "HNSW saved to " << dense_index_path << std::endl;
        } else {
            std::cout << "Loading HNSW from " << dense_index_path << std::endl;
            dense_index = nullptr;
            dense_index = engine.CreateIndex("hnsw", hnsw_build_params).value();
            std::ifstream in_stream(dense_index_path);
            auto deserialize_result = dense_index->Deserialize(in_stream);
            in_stream.close();
            if (!deserialize_result.has_value()) {
                std::cerr << "Failed to load HNSW index: " << deserialize_result.error().message << std::endl;
                return -1;
            }
            num_train = dense_index->GetNumElements();
            std::cout << "HNSW loaded (" << num_train << " vectors)" << std::endl;
        }

        // Build/Load Sparse Index (SINDI)
        std::string sindi_build_params = R"(
        {
            "dtype": "sparse",
            "metric_type": "ip",
            "index_param": {
                "use_reorder": true
            }
        })";

        auto sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();

        if (need_build_sparse) {
            std::cout << "Building SINDI index..." << std::endl;
            auto base_sparse_dataset = vsag::Dataset::Make();
            base_sparse_dataset->NumElements(num_train)
                ->Ids(train_labels.data())
                ->SparseVectors(train_sparse.data())
                ->Owner(false);

            auto build_start = std::chrono::high_resolution_clock::now();
            if (!sparse_index->Build(base_sparse_dataset).has_value()) {
                std::cerr << "Failed to build SINDI index" << std::endl;
                return -1;
            }
            auto build_end = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double>(build_end - build_start).count();
            std::cout << "SINDI built in " << build_time << "s" << std::endl;

            std::ofstream out_stream(sparse_index_path);
            auto serialize_result = sparse_index->Serialize(out_stream);
            out_stream.close();
            if (!serialize_result.has_value()) {
                std::cerr << "Failed to save SINDI index: " << serialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "SINDI saved to " << sparse_index_path << std::endl;
        } else {
            std::cout << "Loading SINDI from " << sparse_index_path << std::endl;
            sparse_index = nullptr;
            sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();
            std::ifstream in_stream(sparse_index_path);
            auto deserialize_result = sparse_index->Deserialize(in_stream);
            in_stream.close();
            if (!deserialize_result.has_value()) {
                std::cerr << "Failed to load SINDI index: " << deserialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "SINDI loaded (" << sparse_index->GetNumElements() << " vectors)" << std::endl;
        }

        // Hybrid Search (using index distances for rerank)
        int actual_num_queries = (params.num_queries > 0) ? std::min(params.num_queries, (int)num_test) : (int)num_test;
        if (gt_shape[0] < actual_num_queries ||
            (mixed_alpha_mode && static_cast<int>(mixed_alphas.size()) < actual_num_queries)) {
            throw std::runtime_error("Mixed alpha/ground truth rows are fewer than num_queries");
        }

        std::ostringstream sparse_ss;
        sparse_ss << std::fixed << std::setprecision(1)
                  << R"({"sindi": {"query_prune_ratio": )" << params.query_prune_ratio
                  << R"(, "term_prune_ratio": )" << params.term_prune_ratio
                  << "}}";
        std::string sparse_search_params = sparse_ss.str();

        struct CandidateDistances {
            bool has_dense{false};
            bool has_sparse{false};
            float dense_dist{0.0f};
            float sparse_dist{0.0f};
        };

        std::vector<int> search_points = params.search_points;
        if (search_points.empty()) {
            search_points.push_back(params.bk_dense);
        }

        for (int point : search_points) {
            const int effective_bk_dense = params.search_points.empty() ? params.bk_dense : point;
            const int effective_bk_sparse = params.search_points.empty() ? params.bk_sparse : point;
            const int effective_ef_search = params.search_points.empty() ? params.ef_search : point;
            std::vector<float> query_recalls(actual_num_queries, 0.0f);
            std::string dense_search_params = R"({"hnsw": {"ef_search": )" +
                std::to_string(effective_ef_search) + "}}";

            auto search_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(params.threads) schedule(dynamic)
            for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {
                const float query_alpha =
                    mixed_alpha_mode ? mixed_alphas[query_idx] : params.alpha;
                if (params.threads == 1 && (query_idx + 1) % 100 == 0) {
                    std::cout << "Processing query " << query_idx + 1 << "/"
                              << actual_num_queries << " (point=" << point << ")" << std::endl;
                }

                auto query = vsag::Dataset::Make();
                query->NumElements(1)
                    ->Dim(dense_dim)
                    ->Float32Vectors(test_dense.data() + query_idx * dense_dim)
                    ->SparseVectors(test_sparse.data() + query_idx)
                    ->Owner(false);

                auto dense_results = dense_index->KnnSearch(
                    query, effective_bk_dense, dense_search_params).value();
                auto sparse_results = sparse_index->KnnSearch(
                    query, effective_bk_sparse, sparse_search_params).value();

                std::unordered_map<int64_t, CandidateDistances> candidate_distances;
                candidate_distances.reserve(dense_results->GetDim() + sparse_results->GetDim());

                const auto* dense_ids = dense_results->GetIds();
                const auto* dense_search_dists = dense_results->GetDistances();
                for (int i = 0; i < dense_results->GetDim(); i++) {
                    auto& candidate = candidate_distances[dense_ids[i]];
                    candidate.has_dense = true;
                    candidate.dense_dist = dense_search_dists[i];
                }

                const auto* sparse_ids = sparse_results->GetIds();
                const auto* sparse_search_dists = sparse_results->GetDistances();
                for (int i = 0; i < sparse_results->GetDim(); i++) {
                    auto& candidate = candidate_distances[sparse_ids[i]];
                    candidate.has_sparse = true;
                    candidate.sparse_dist = sparse_search_dists[i];
                }

                std::vector<int64_t> missing_dense_ids;
                std::vector<int64_t> missing_sparse_ids;
                missing_dense_ids.reserve(candidate_distances.size());
                missing_sparse_ids.reserve(candidate_distances.size());
                for (const auto& [candidate_id, distances] : candidate_distances) {
                    if (!distances.has_dense) missing_dense_ids.emplace_back(candidate_id);
                    if (!distances.has_sparse) missing_sparse_ids.emplace_back(candidate_id);
                }

                if (!missing_dense_ids.empty()) {
                    auto dense_dists_result = dense_index->CalDistanceById(
                        test_dense.data() + query_idx * dense_dim,
                        missing_dense_ids.data(),
                        static_cast<int64_t>(missing_dense_ids.size())).value();
                    const auto* dense_dists = dense_dists_result->GetDistances();
                    for (size_t i = 0; i < missing_dense_ids.size(); ++i) {
                        auto& candidate = candidate_distances[missing_dense_ids[i]];
                        candidate.has_dense = true;
                        candidate.dense_dist = dense_dists[i];
                    }
                }

                if (!missing_sparse_ids.empty()) {
                    auto sparse_query_ds = vsag::Dataset::Make();
                    sparse_query_ds->NumElements(1)
                        ->SparseVectors(test_sparse.data() + query_idx)
                        ->Owner(false);
                    auto sparse_dists_result = sparse_index->CalDistanceById(
                        sparse_query_ds,
                        missing_sparse_ids.data(),
                        static_cast<int64_t>(missing_sparse_ids.size())).value();
                    const auto* sparse_dists = sparse_dists_result->GetDistances();
                    for (size_t i = 0; i < missing_sparse_ids.size(); ++i) {
                        auto& candidate = candidate_distances[missing_sparse_ids[i]];
                        candidate.has_sparse = true;
                        candidate.sparse_dist = sparse_dists[i];
                    }
                }

                std::vector<std::pair<int64_t, float>> hybrid_results;
                hybrid_results.reserve(candidate_distances.size());
                for (const auto& [candidate_id, distances] : candidate_distances) {
                    float dense_score = 1.0f - distances.dense_dist;
                    float sparse_score = 1.0f - distances.sparse_dist;
                    float hybrid_score =
                        query_alpha * dense_score + (1.0f - query_alpha) * sparse_score;
                    hybrid_results.emplace_back(candidate_id, hybrid_score);
                }

                int actual_k = std::min(params.k, (int)hybrid_results.size());
                std::partial_sort(hybrid_results.begin(), hybrid_results.begin() + actual_k,
                                  hybrid_results.end(),
                                  [](const auto& a, const auto& b) { return a.second > b.second; });

                std::vector<int64_t> search_results;
                for (int i = 0; i < actual_k; i++) {
                    search_results.push_back(hybrid_results[i].first);
                }

                int gt_k = std::min(params.k, (int)k_gt);
                std::vector<int64_t> gt(ground_truth.begin() + query_idx * k_gt,
                                       ground_truth.begin() + query_idx * k_gt + gt_k);
                query_recalls[query_idx] = CalculateRecall(search_results, gt);
            }

            auto search_end = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(search_end - search_start).count();
            double qps = actual_num_queries / elapsed_seconds;

            double total_recall = 0.0;
            for (float recall : query_recalls) total_recall += recall;
            float avg_recall = static_cast<float>(total_recall / actual_num_queries);

            std::cout << "\n========== Results ==========\n";
            std::cout << "Dataset: " << params.h5_file << "\n";
            std::cout << "k: " << params.k << "\n";
            std::cout << "bk_dense: " << effective_bk_dense << "\n";
            std::cout << "bk_sparse: " << effective_bk_sparse << "\n";
            std::cout << "alpha: " << params.alpha << "\n";
            std::cout << "mixed_alpha: " << (mixed_alpha_mode ? "true" : "false") << "\n";
            std::cout << "ef_search: " << effective_ef_search << "\n";
            std::cout << "query_prune_ratio: " << params.query_prune_ratio << "\n";
            std::cout << "term_prune_ratio: " << params.term_prune_ratio << "\n";
            std::cout << "num_queries: " << actual_num_queries << "\n";
            std::cout << "threads: " << params.threads << "\n";
            std::cout << "rebuild: " << (params.rebuild ? "true" : "false") << "\n";
            std::cout << "index_dir: " << params.index_dir << "\n";
            std::cout << "gt_file: " << gt_file_path << "\n";
            std::cout << "----------------------------\n";
            std::cout << "Recall: " << avg_recall << "\n";
            std::cout << "QPS: " << qps << "\n";
            std::cout << "============================\n";
        }

        // Cleanup
        if (need_build) {
            FreeSparseVectors(train_sparse);
        }
        FreeSparseVectors(test_sparse);
        engine.Shutdown();

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
