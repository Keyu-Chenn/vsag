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
#include <unordered_set>
#include <vector>
#include <cmath>
#include <nlohmann/json.hpp>

#include "uhg_mixed_alpha_utils.h"

// ============================================================
// 工具函数
// ============================================================

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
    return static_cast<float>(hit_count) / static_cast<float>(ground_truth.size());
}


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

bool
FileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
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

// ============================================================
// 参数解析
// ============================================================

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nOptions:\n"
              << "  --method <string>          hybrid_index method: uhg/uhgs/uhgh/auto (default: auto)\n"
              << "  -k, --topk <int>           Number of final top results (default: 10)\n"
              << "  --sindi_bk <int>           Sindi recall count for entry points (default: 100)\n"
              << "  --dense_entry_bk <int>     Dense entry recall count for uhgh/auto (default: 100)\n"
              << "  --dense_entry_ef_search <int>\n"
              << "                              ef_search for dense entry graph (default: 200)\n"
              << "  --sindi_query_prune_ratio <float>\n"
              << "                              SINDI query pruning ratio, [0,0.9], 0=no pruning (default: 0)\n"
              << "  --sindi_term_prune_ratio <float>\n"
              << "                              SINDI term pruning ratio, [0,0.9], 0=no pruning (default: 0)\n"
              << "  --ef_search <int>          ef_search for hybrid index (default: 200)\n"
              << "  --search_points <csv>      Run all coupled search-width points after one index load\n"
              << "  --alpha <float>            Dense score weight (default: 0.5)\n"
              << "  --build_alpha <float>      Dense weight used to build the main graph (default: alpha)\n"
              << "  --graph_path <path>        Precomputed UHG graph HDF5 for the main hybrid graph\n"
              << "  --sindi_index_path <path>  Existing SINDI index to load into hybrid_index\n"
              << "  --dense_entry_hnsw_graph_path <path>\n"
              << "                              Existing dense HNSW index; load graph only for UHGH\n"
              << "  --disable_sindi            Do not build internal SINDI helper\n"
              << "  --disable_dense_entry      Do not build internal dense-entry helper\n"
              << "  --hybrid_prune_scale <float>\n"
              << "                              Sparse-IP upper-bound scale; smaller is more aggressive, 1=Cauchy bound (default: 1.0)\n"
              << "  --disable_hybrid_pruning   Disable dense-first sparse upper-bound pruning\n"
              << "  --max_hops <int>           Max neighbor expansion hops, 0=no limit (default: 0)\n"
              << "  --num_queries <int>        Number of queries to process, -1 for all (default: -1)\n"
              << "  --threads <int>            Query-level parallel search threads (default: 1)\n"
              << "  --rebuild                  Force rebuild indexes\n"
              << "  --index_dir <path>         Directory for index cache (default: /tbase-project/vsag/scripts/UHG/data/index)\n"
              << "  --hybrid_index_path <path> Explicit hybrid_index path, overrides index_dir\n"
              << "  --gt_dir <path>            Directory with ground truth npy files (required)\n"
              << "  --mixed_alpha_file <path> Per-query alpha float32 npy; use with --mixed_gt_file\n"
              << "  --mixed_gt_file <path>    Ground truth int64 npy for mixed per-query alphas\n"
              << "  --help, -h                 Show this help message\n"
              << "\nMethods:\n"
              << "  uhg:   Main hybrid graph search\n"
              << "  uhgs:  Internal SINDI entry points + main hybrid graph\n"
              << "  uhgh:  Internal dense entry graph + main hybrid graph\n"
              << "  auto:  alpha<=0.5 => uhgs, alpha>0.5 => uhgh\n"
              << "\nIndex file: 703_{dataset}_hybrid_index.index\n"
              << std::endl;
}

struct SearchParams {
    std::string h5_file;
    std::string method = "auto";
    int k = 10;
    int sindi_bk = 100;
    int dense_entry_bk = 100;
    int dense_entry_ef_search = 200;
    float sindi_query_prune_ratio = 0.0f;
    float sindi_term_prune_ratio = 0.0f;
    int ef_search = 200;
    std::vector<int> search_points;
    float alpha = 0.5f;
    float build_alpha = -1.0f;
    std::string graph_path;
    std::string sindi_index_path;
    std::string dense_entry_hnsw_graph_path;
    bool enable_sindi = true;
    bool enable_dense_entry = true;
    bool enable_hybrid_pruning = true;
    float hybrid_prune_scale = 1.0f;
    int max_hops = 0;
    int num_queries = -1;
    int threads = 1;
    bool rebuild = false;
    std::string index_dir = "/tbase-project/vsag/scripts/UHG/data/index";
    std::string hybrid_index_path;
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
        } else if (arg == "--method" && i + 1 < argc) {
            params.method = argv[++i];
            if (params.method != "uhg" && params.method != "uhgs" &&
                params.method != "uhgh" && params.method != "auto") {
                std::cerr << "Error: method must be one of uhg/uhgs/uhgh/auto\n";
                exit(1);
            }
        } else if ((arg == "-k" || arg == "--topk") && i + 1 < argc) {
            params.k = std::atoi(argv[++i]);
        } else if (arg == "--sindi_bk" && i + 1 < argc) {
            params.sindi_bk = std::atoi(argv[++i]);
        } else if ((arg == "--dense_entry_bk" || arg == "--hnsw_bk") && i + 1 < argc) {
            params.dense_entry_bk = std::atoi(argv[++i]);
        } else if ((arg == "--dense_entry_ef_search" || arg == "--hnsw_ef_search") &&
                   i + 1 < argc) {
            params.dense_entry_ef_search = std::atoi(argv[++i]);
        } else if (arg == "--sindi_query_prune_ratio" && i + 1 < argc) {
            params.sindi_query_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--sindi_term_prune_ratio" && i + 1 < argc) {
            params.sindi_term_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--ef_search" && i + 1 < argc) {
            params.ef_search = std::atoi(argv[++i]);
        } else if (arg == "--search_points" && i + 1 < argc) {
            params.search_points = ParseSearchPoints(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            params.alpha = std::atof(argv[++i]);
        } else if (arg == "--build_alpha" && i + 1 < argc) {
            params.build_alpha = std::atof(argv[++i]);
        } else if (arg == "--graph_path" && i + 1 < argc) {
            params.graph_path = argv[++i];
        } else if (arg == "--sindi_index_path" && i + 1 < argc) {
            params.sindi_index_path = argv[++i];
        } else if ((arg == "--dense_entry_hnsw_graph_path" ||
                    arg == "--dense_entry_hnsw_index_path" ||
                    arg == "--hnsw_index_path") &&
                   i + 1 < argc) {
            params.dense_entry_hnsw_graph_path = argv[++i];
        } else if (arg == "--disable_sindi") {
            params.enable_sindi = false;
        } else if (arg == "--disable_dense_entry") {
            params.enable_dense_entry = false;
        } else if (arg == "--disable_hybrid_pruning") {
            params.enable_hybrid_pruning = false;
        } else if (arg == "--hybrid_prune_scale" && i + 1 < argc) {
            params.hybrid_prune_scale = std::atof(argv[++i]);
        } else if (arg == "--max_hops" && i + 1 < argc) {
            params.max_hops = std::atoi(argv[++i]);
        } else if (arg == "--num_queries" && i + 1 < argc) {
            params.num_queries = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            params.threads = std::atoi(argv[++i]);
        } else if (arg == "--rebuild") {
            params.rebuild = true;
        } else if (arg == "--index_dir" && i + 1 < argc) {
            params.index_dir = argv[++i];
        } else if (arg == "--hybrid_index_path" && i + 1 < argc) {
            params.hybrid_index_path = argv[++i];
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

    if (params.k <= 0 || params.sindi_bk <= 0 || params.dense_entry_bk <= 0 ||
        params.dense_entry_ef_search <= 0 || params.ef_search <= 0) {
        std::cerr << "Error: k, sindi_bk, dense_entry_bk, dense_entry_ef_search, ef_search must be positive\n";
        exit(1);
    }
    if (params.alpha < 0.0f || params.alpha > 1.0f) {
        std::cerr << "Error: alpha must be in [0, 1]\n";
        exit(1);
    }
    if (params.build_alpha >= 0.0f and params.build_alpha > 1.0f) {
        std::cerr << "Error: build_alpha must be in [0, 1]\n";
        exit(1);
    }
    if (params.sindi_query_prune_ratio < 0.0f || params.sindi_query_prune_ratio > 0.9f) {
        std::cerr << "Error: sindi_query_prune_ratio must be in [0, 0.9]\n";
        exit(1);
    }
    if (params.sindi_term_prune_ratio < 0.0f || params.sindi_term_prune_ratio > 0.9f) {
        std::cerr << "Error: sindi_term_prune_ratio must be in [0, 0.9]\n";
        exit(1);
    }
    if (params.hybrid_prune_scale < 0.0f) {
        std::cerr << "Error: hybrid_prune_scale must be >= 0\n";
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
    if (params.max_hops < 0) {
        std::cerr << "Error: max_hops must be >= 0\n";
        exit(1);
    }
    if (!params.graph_path.empty() && !FileExists(params.graph_path)) {
        std::cerr << "Error: graph_path does not exist: " << params.graph_path << "\n";
        exit(1);
    }
    if (!params.sindi_index_path.empty() && !FileExists(params.sindi_index_path)) {
        std::cerr << "Error: sindi_index_path does not exist: " << params.sindi_index_path << "\n";
        exit(1);
    }
    if (!params.dense_entry_hnsw_graph_path.empty() &&
        !FileExists(params.dense_entry_hnsw_graph_path)) {
        std::cerr << "Error: dense_entry_hnsw_graph_path does not exist: "
                  << params.dense_entry_hnsw_graph_path << "\n";
        exit(1);
    }

    return params;
}

std::string
GetDatasetName(const std::string& file_path) {
    std::string filename = file_path;
    size_t last_slash = filename.find_last_of('/');
    if (last_slash != std::string::npos) {
        filename = filename.substr(last_slash + 1);
    }
    size_t last_dot = filename.find_last_of('.');
    return (last_dot != std::string::npos) ? filename.substr(0, last_dot) : filename;
}

// ============================================================
// main
// ============================================================

int
main(int argc, char** argv) {
    vsag::init();

    SearchParams params = ParseCommandLine(argc, argv);

    try {
        std::string dataset_name = GetDatasetName(params.h5_file);

        // Index paths
        std::string hybrid_index_path = params.hybrid_index_path.empty()
                                            ? params.index_dir + "/703_" + dataset_name + "_hybrid_index.index"
                                            : params.hybrid_index_path;

        bool need_build_hybrid = params.rebuild || !FileExists(hybrid_index_path);
        bool need_build = need_build_hybrid;

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

        /******************* 1. 加载数据集 *****************/
        std::cout << "Opening " << params.h5_file << std::endl;
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
            hsize_t test_sparse_size = test_sparse_dataset.getSpace().getSimpleExtentNpoints();
            std::vector<uint8_t> test_sparse_blob(test_sparse_size);
            test_sparse_dataset.read(test_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
            test_sparse = ParseSparseVectors(test_sparse_blob);

            std::cout << "Test data loaded: " << num_test << " queries, dim=" << dense_dim << std::endl;
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
            hsize_t train_sparse_size = train_sparse_dataset.getSpace().getSimpleExtentNpoints();
            std::vector<uint8_t> train_sparse_blob(train_sparse_size);
            train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
            train_sparse = ParseSparseVectors(train_sparse_blob);

            H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
            train_labels.resize(num_train);
            train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);

            std::cout << "Train data loaded: " << num_train << " vectors" << std::endl;
        }

        /******************* 2. 构建/加载 unified hybrid_index *****************/
        float effective_build_alpha = params.build_alpha >= 0.0f ? params.build_alpha : params.alpha;
        nlohmann::json hybrid_build_param_json = {
            {"dtype", "float32"},
            {"metric_type", "ip"},
            {"dim", dense_dim},
            {"index_param",
             {
                 {"sparse_dtype", "float32"},
                 {"sparse_metric_type", "ip"},
                 {"sparse_dim", 30000},
                 {"alpha", effective_build_alpha},
                 {"ef_construction", 200},
                 {"max_degree", 64},
                 {"enable_sindi", params.enable_sindi},
                 {"enable_dense_entry", params.enable_dense_entry},
             }},
        };
        if (!params.graph_path.empty()) {
            hybrid_build_param_json["index_param"]["graph_path"] = params.graph_path;
        }
        if (!params.sindi_index_path.empty()) {
            hybrid_build_param_json["index_param"]["sindi_index_path"] = params.sindi_index_path;
        }
        if (!params.dense_entry_hnsw_graph_path.empty()) {
            hybrid_build_param_json["index_param"]["dense_entry_hnsw_graph_path"] =
                params.dense_entry_hnsw_graph_path;
        }
        std::string hybrid_build_params = hybrid_build_param_json.dump();

        auto hybrid_index = vsag::Factory::CreateIndex("hybrid_index", hybrid_build_params).value();

        if (need_build_hybrid) {
            std::cout << "Building hybrid_index..." << std::endl;
            auto base_dataset = vsag::Dataset::Make();
            base_dataset->NumElements(num_train)
                ->Dim(dense_dim)
                ->Ids(train_labels.data())
                ->Float32Vectors(train_dense.data())
                ->SparseVectors(train_sparse.data())
                ->Owner(false);

            auto build_start = std::chrono::high_resolution_clock::now();
            if (!hybrid_index->Build(base_dataset).has_value()) {
                std::cerr << "Failed to build hybrid_index" << std::endl;
                return -1;
            }
            auto build_end = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double>(build_end - build_start).count();
            std::cout << "Hybrid_index built in " << build_time << "s" << std::endl;

            std::ofstream out_stream(hybrid_index_path);
            auto serialize_result = hybrid_index->Serialize(out_stream);
            out_stream.close();
            if (!serialize_result.has_value()) {
                std::cerr << "Failed to save hybrid_index: " << serialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "Hybrid_index saved to " << hybrid_index_path << std::endl;
        } else {
            std::cout << "Loading hybrid_index from " << hybrid_index_path << std::endl;
            hybrid_index = nullptr;
            hybrid_index = vsag::Factory::CreateIndex("hybrid_index", hybrid_build_params).value();
            std::ifstream in_stream(hybrid_index_path);
            auto deserialize_result = hybrid_index->Deserialize(in_stream);
            in_stream.close();
            if (!deserialize_result.has_value()) {
                std::cerr << "Failed to load hybrid_index: " << deserialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "Hybrid_index loaded (" << hybrid_index->GetNumElements() << " vectors)" << std::endl;
        }

        /******************* 4. 加载 ground truth *****************/
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
            std::cerr << "Error: Ground truth must be 2D array\n";
            return -1;
        }
        int64_t k_gt = gt_shape[1];
        std::cout << "Ground truth loaded: k=" << k_gt << std::endl;

        /******************* 5. 检索实验 *****************/
        int actual_num_queries = (params.num_queries > 0)
            ? std::min(params.num_queries, (int)num_test)
            : (int)num_test;
        if (gt_shape[0] < actual_num_queries ||
            (mixed_alpha_mode && static_cast<int>(mixed_alphas.size()) < actual_num_queries)) {
            throw std::runtime_error("Mixed alpha/ground truth rows are fewer than num_queries");
        }

        std::vector<int> search_points = params.search_points;
        if (search_points.empty()) {
            search_points.push_back(params.ef_search);
        }

        for (int point : search_points) {
            const int effective_ef_search = params.search_points.empty() ? params.ef_search : point;
            const int effective_sindi_bk = params.search_points.empty() ? params.sindi_bk : point;
            const int effective_dense_entry_bk =
                params.search_points.empty() ? params.dense_entry_bk : point;
            const int effective_dense_entry_ef_search =
                params.search_points.empty() ? params.dense_entry_ef_search : point;
            std::vector<float> query_recalls(actual_num_queries, 0.0f);

            auto search_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(params.threads) schedule(dynamic)
            for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {
                const float query_alpha =
                    mixed_alpha_mode ? mixed_alphas[query_idx] : params.alpha;
                if (params.threads == 1 && (query_idx + 1) % 100 == 0) {
                    std::cout << "Processing query " << query_idx + 1 << "/"
                              << actual_num_queries << " (point=" << point << ")" << std::endl;
                }

                nlohmann::json unified_search_param_json = {
                    {"method", params.method},
                    {"alpha", query_alpha},
                    {"ef_search", effective_ef_search},
                    {"enable_hybrid_pruning", params.enable_hybrid_pruning},
                    {"hybrid_prune_scale", params.hybrid_prune_scale},
                    {"sindi_bk", effective_sindi_bk},
                    {"sindi_query_prune_ratio", params.sindi_query_prune_ratio},
                    {"sindi_term_prune_ratio", params.sindi_term_prune_ratio},
                    {"dense_entry_bk", effective_dense_entry_bk},
                    {"dense_entry_ef_search", effective_dense_entry_ef_search}
                };
                if (params.max_hops > 0) {
                    unified_search_param_json["max_hops"] = params.max_hops;
                }

                auto query_ds = vsag::Dataset::Make();
                query_ds->NumElements(1)
                    ->Dim(dense_dim)
                    ->Float32Vectors(test_dense.data() + query_idx * dense_dim)
                    ->SparseVectors(test_sparse.data() + query_idx)
                    ->Owner(false);

                auto hybrid_result = hybrid_index->KnnSearch(
                    query_ds, params.k, unified_search_param_json.dump()).value();

                int result_num = hybrid_result->GetDim();
                std::vector<int64_t> search_results(
                    hybrid_result->GetIds(), hybrid_result->GetIds() + result_num);

                int gt_k = std::min(params.k, (int)k_gt);
                std::vector<int64_t> gt(
                    ground_truth.begin() + query_idx * k_gt,
                    ground_truth.begin() + query_idx * k_gt + gt_k);
                query_recalls[query_idx] = CalculateRecall(search_results, gt);
            }

            auto search_end = std::chrono::high_resolution_clock::now();
            double elapsed_seconds =
                std::chrono::duration<double>(search_end - search_start).count();
            double qps = static_cast<double>(actual_num_queries) / elapsed_seconds;

            double total_recall = 0.0;
            for (float recall : query_recalls) total_recall += recall;
            float avg_recall = static_cast<float>(total_recall / actual_num_queries);

            /******************* 6. 输出结果 *****************/
            std::cout << "\n========== Results ==========\n";
            std::cout << "Method: " << params.method << "\n";
            std::cout << "k: " << params.k << "\n";
            std::cout << "sindi_bk: " << effective_sindi_bk << "\n";
            std::cout << "dense_entry_bk: " << effective_dense_entry_bk << "\n";
            std::cout << "dense_entry_ef_search: " << effective_dense_entry_ef_search << "\n";
            std::cout << "sindi_query_prune_ratio: " << params.sindi_query_prune_ratio << "\n";
            std::cout << "sindi_term_prune_ratio: " << params.sindi_term_prune_ratio << "\n";
            std::cout << "ef_search: " << effective_ef_search << "\n";
            std::cout << "alpha: " << params.alpha << "\n";
            std::cout << "mixed_alpha: " << (mixed_alpha_mode ? "true" : "false") << "\n";
            std::cout << "build_alpha: " << effective_build_alpha << "\n";
            std::cout << "graph_path: "
                      << (params.graph_path.empty() ? "(none)" : params.graph_path) << "\n";
            std::cout << "sindi_index_path: "
                      << (params.sindi_index_path.empty() ? "(none)" : params.sindi_index_path)
                      << "\n";
            std::cout << "dense_entry_hnsw_graph_path: "
                      << (params.dense_entry_hnsw_graph_path.empty()
                              ? "(none)"
                              : params.dense_entry_hnsw_graph_path)
                      << "\n";
            std::cout << "enable_sindi: " << (params.enable_sindi ? "true" : "false") << "\n";
            std::cout << "enable_dense_entry: "
                      << (params.enable_dense_entry ? "true" : "false") << "\n";
            std::cout << "enable_hybrid_pruning: "
                      << (params.enable_hybrid_pruning ? "true" : "false") << "\n";
            std::cout << "hybrid_prune_scale: " << params.hybrid_prune_scale << "\n";
            std::cout << "max_hops: " << params.max_hops << "\n";
            std::cout << "num_queries: " << actual_num_queries << "\n";
            std::cout << "threads: " << params.threads << "\n";
            std::cout << "rebuild: " << (params.rebuild ? "true" : "false") << "\n";
            std::cout << "index_dir: " << params.index_dir << "\n";
            std::cout << "----------------------------\n";
            std::cout << "Recall: " << avg_recall << "\n";
            std::cout << "QPS: " << qps << "\n";
            std::cout << "============================\n";
        }

        /******************* 7. 清理资源 *****************/
        if (need_build) {
            FreeSparseVectors(train_sparse);
        }
        FreeSparseVectors(test_sparse);

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
        return 1;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
