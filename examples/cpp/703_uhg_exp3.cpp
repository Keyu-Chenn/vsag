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

// ============================================================
// 参数解析
// ============================================================

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nOptions:\n"
              << "  --method <string>          Search method: 'uhg' or 'uhgs' (default: uhgs)\n"
              << "  -k, --topk <int>           Number of final top results (default: 10)\n"
              << "  --sindi_bk <int>           Sindi recall count for entry points (default: 100)\n"
              << "  --sindi_query_prune_ratio <float>\n"
              << "                              SINDI query pruning ratio, [0,0.9], 0=no pruning (default: 0)\n"
              << "  --sindi_term_prune_ratio <float>\n"
              << "                              SINDI term pruning ratio, [0,0.9], 0=no pruning (default: 0)\n"
              << "  --ef_search <int>          ef_search for hybrid index (default: 200)\n"
              << "  --alpha <float>            Dense score weight (default: 0.5)\n"
              << "  --hybrid_prune_scale <float>\n"
              << "                              Prune scale for hybrid search, 0=sparse-only, smaller=more aggressive (default: 1.0)\n"
              << "  --max_hops <int>           Max neighbor expansion hops, 0=no limit (default: 0)\n"
              << "  --num_queries <int>        Number of queries to process, -1 for all (default: -1)\n"
              << "  --rebuild                  Force rebuild indexes\n"
              << "  --index_dir <path>         Directory for index cache (default: /tbase-project/vsag/scripts/UHG/data/index)\n"
              << "  --gt_dir <path>            Directory with ground truth npy files (required)\n"
              << "  --help, -h                 Show this help message\n"
              << "\nMethods:\n"
              << "  uhg:   Hybrid index search with single entry point (baseline)\n"
              << "  uhgs:  Sindi recall + hybrid index search (sindi results as entry points,\n"
              << "         reuse sindi sparse distances via ExtraInfos)\n"
              << "\nIndex files: 703_{dataset}_sindi.index, 703_{dataset}_hybrid_index.index\n"
              << std::endl;
}

struct SearchParams {
    std::string h5_file;
    std::string method = "uhgs";
    int k = 10;
    int sindi_bk = 100;
    float sindi_query_prune_ratio = 0.0f;
    float sindi_term_prune_ratio = 0.0f;
    int ef_search = 200;
    float alpha = 0.5f;
    float hybrid_prune_scale = 1.0f;
    int max_hops = 0;
    int num_queries = -1;
    bool rebuild = false;
    std::string index_dir = "/tbase-project/vsag/scripts/UHG/data/index";
    std::string gt_dir = "";
};

SearchParams
ParseCommandLine(int argc, char** argv) {
    SearchParams params;

    if (argc < 2) {
        PrintUsage(argv[0]);
        exit(1);
    }

    params.h5_file = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            exit(0);
        } else if (arg == "--method" && i + 1 < argc) {
            params.method = argv[++i];
            if (params.method != "uhg" && params.method != "uhgs") {
                std::cerr << "Error: method must be 'uhg' or 'uhgs'\n";
                exit(1);
            }
        } else if ((arg == "-k" || arg == "--topk") && i + 1 < argc) {
            params.k = std::atoi(argv[++i]);
        } else if (arg == "--sindi_bk" && i + 1 < argc) {
            params.sindi_bk = std::atoi(argv[++i]);
        } else if (arg == "--sindi_query_prune_ratio" && i + 1 < argc) {
            params.sindi_query_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--sindi_term_prune_ratio" && i + 1 < argc) {
            params.sindi_term_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--ef_search" && i + 1 < argc) {
            params.ef_search = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            params.alpha = std::atof(argv[++i]);
        } else if (arg == "--hybrid_prune_scale" && i + 1 < argc) {
            params.hybrid_prune_scale = std::atof(argv[++i]);
        } else if (arg == "--max_hops" && i + 1 < argc) {
            params.max_hops = std::atoi(argv[++i]);
        } else if (arg == "--num_queries" && i + 1 < argc) {
            params.num_queries = std::atoi(argv[++i]);
        } else if (arg == "--rebuild") {
            params.rebuild = true;
        } else if (arg == "--index_dir" && i + 1 < argc) {
            params.index_dir = argv[++i];
        } else if (arg == "--gt_dir" && i + 1 < argc) {
            params.gt_dir = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }

    if (params.k <= 0 || params.sindi_bk <= 0 || params.ef_search <= 0) {
        std::cerr << "Error: k, sindi_bk, ef_search must be positive\n";
        exit(1);
    }
    if (params.alpha < 0.0f || params.alpha > 1.0f) {
        std::cerr << "Error: alpha must be in [0, 1]\n";
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
    if (params.max_hops < 0) {
        std::cerr << "Error: max_hops must be >= 0\n";
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
        std::string sindi_index_path = params.index_dir + "/703_" + dataset_name + "_sindi.index";
        std::string hybrid_index_path = params.index_dir + "/703_" + dataset_name + "_hybrid_index.index";

        bool need_build_sindi = params.rebuild || !FileExists(sindi_index_path);
        bool need_build_hybrid = params.rebuild || !FileExists(hybrid_index_path);
        bool need_build = need_build_sindi || need_build_hybrid;

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

        /******************* 2. 构建/加载 sindi 索引 *****************/
        std::string sindi_build_params = R"(
        {
            "dtype": "sparse",
            "metric_type": "ip",
            "index_param": {
                "use_reorder": false
            }
        })";

        auto sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();

        if (need_build_sindi) {
            std::cout << "Building sindi index..." << std::endl;
            auto base_sparse_dataset = vsag::Dataset::Make();
            base_sparse_dataset->NumElements(num_train)
                ->Ids(train_labels.data())
                ->SparseVectors(train_sparse.data())
                ->Owner(false);

            auto build_start = std::chrono::high_resolution_clock::now();
            if (!sparse_index->Build(base_sparse_dataset).has_value()) {
                std::cerr << "Failed to build sindi index" << std::endl;
                return -1;
            }
            auto build_end = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double>(build_end - build_start).count();
            std::cout << "Sindi built in " << build_time << "s" << std::endl;

            std::ofstream out_stream(sindi_index_path);
            auto serialize_result = sparse_index->Serialize(out_stream);
            out_stream.close();
            if (!serialize_result.has_value()) {
                std::cerr << "Failed to save sindi index: " << serialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "Sindi saved to " << sindi_index_path << std::endl;
        } else {
            std::cout << "Loading sindi from " << sindi_index_path << std::endl;
            sparse_index = nullptr;
            sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();
            std::ifstream in_stream(sindi_index_path);
            auto deserialize_result = sparse_index->Deserialize(in_stream);
            in_stream.close();
            if (!deserialize_result.has_value()) {
                std::cerr << "Failed to load sindi index: " << deserialize_result.error().message << std::endl;
                return -1;
            }
            num_train = sparse_index->GetNumElements();
            std::cout << "Sindi loaded (" << num_train << " vectors)" << std::endl;
        }

        /******************* 3. 构建/加载 hybrid_index *****************/
        std::string hybrid_build_params =
            R"({
                "dtype": "float32",
                "metric_type": "ip",
                "dim": )" + std::to_string(dense_dim) + R"(,
                "index_param": {
                    "sparse_dtype": "float32",
                    "sparse_metric_type": "ip",
                    "sparse_dim": 30000,
                    "alpha": )" + std::to_string(params.alpha) + R"(,
                    "ef_construction": 200,
                    "max_degree": 64
                }
            })";

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
        std::ostringstream gt_file_ss;
        gt_file_ss << params.gt_dir << "/" << dataset_name << "_ground_truth_alpha_"
                   << std::fixed << std::setprecision(1) << params.alpha << ".npy";
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

        float total_recall = 0.0f;

        auto search_start = std::chrono::high_resolution_clock::now();

        for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {
            if ((query_idx + 1) % 100 == 0) {
                std::cout << "Processing query " << query_idx + 1 << "/" << actual_num_queries << std::endl;
            }

            if (params.method == "uhgs") {
                // UHGS: sindi 召回作为入口点
                auto query_sparse_ds = vsag::Dataset::Make();
                query_sparse_ds->NumElements(1)
                    ->SparseVectors(test_sparse.data() + query_idx)
                    ->Owner(false);

                nlohmann::json sindi_search_param_json = {
                    {"sindi", {
                        {"query_prune_ratio", params.sindi_query_prune_ratio},
                        {"term_prune_ratio", params.sindi_term_prune_ratio}
                    }}
                };
                auto sindi_search_params = sindi_search_param_json.dump();
                auto sindi_result = sparse_index->KnnSearch(
                    query_sparse_ds, params.sindi_bk, sindi_search_params).value();

                int sindi_num = sindi_result->GetDim();
                const int64_t* sindi_ids = sindi_result->GetIds();
                std::vector<int64_t> entry_points(sindi_ids, sindi_ids + sindi_num);

                // hybrid_index 搜索，传入 entry_points 和 ExtraInfos
                nlohmann::json search_param_json = {
                    {"alpha", params.alpha},
                    {"ef_search", params.ef_search},
                    {"hybrid_prune_scale", params.hybrid_prune_scale},
                    {"entry_points", entry_points}
                };
                if (params.max_hops > 0) {
                    search_param_json["max_hops"] = params.max_hops;
                }
                std::string hybrid_search_params = search_param_json.dump();

                auto query_ds = vsag::Dataset::Make();
                query_ds->NumElements(1)
                    ->Dim(dense_dim)
                    ->Float32Vectors(test_dense.data() + query_idx * dense_dim)
                    ->SparseVectors(test_sparse.data() + query_idx)
                    ->ExtraInfos(sindi_result->GetExtraInfos())
                    ->ExtraInfoSize(sindi_result->GetExtraInfoSize())
                    ->Owner(false);

                auto hybrid_result = hybrid_index->KnnSearch(
                    query_ds, params.k, hybrid_search_params).value();

                int result_num = hybrid_result->GetDim();
                std::vector<int64_t> search_results(
                    hybrid_result->GetIds(),
                    hybrid_result->GetIds() + result_num);

                int gt_k = std::min(params.k, (int)k_gt);
                std::vector<int64_t> gt(
                    ground_truth.begin() + query_idx * k_gt,
                    ground_truth.begin() + query_idx * k_gt + gt_k);
                total_recall += CalculateRecall(search_results, gt);

            } else {
                // UHG: 单入口点 baseline
                nlohmann::json search_param_json = {
                    {"alpha", params.alpha},
                    {"ef_search", params.ef_search},
                    {"hybrid_prune_scale", params.hybrid_prune_scale},
                    {"entry_point", 0}
                };
                if (params.max_hops > 0) {
                    search_param_json["max_hops"] = params.max_hops;
                }
                std::string hybrid_search_params = search_param_json.dump();

                auto query_ds = vsag::Dataset::Make();
                query_ds->NumElements(1)
                    ->Dim(dense_dim)
                    ->Float32Vectors(test_dense.data() + query_idx * dense_dim)
                    ->SparseVectors(test_sparse.data() + query_idx)
                    ->Owner(false);

                auto hybrid_result = hybrid_index->KnnSearch(
                    query_ds, params.k, hybrid_search_params).value();

                int result_num = hybrid_result->GetDim();
                std::vector<int64_t> search_results(
                    hybrid_result->GetIds(),
                    hybrid_result->GetIds() + result_num);

                int gt_k = std::min(params.k, (int)k_gt);
                std::vector<int64_t> gt(
                    ground_truth.begin() + query_idx * k_gt,
                    ground_truth.begin() + query_idx * k_gt + gt_k);
                total_recall += CalculateRecall(search_results, gt);
            }
        }

        auto search_end = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(search_end - search_start).count();
        double qps = static_cast<double>(actual_num_queries) / elapsed_seconds;

        float avg_recall = total_recall / static_cast<float>(actual_num_queries);

        /******************* 6. 输出结果 *****************/
        std::cout << "\n========== Results ==========\n";
        std::cout << "Method: " << params.method << "\n";
        std::cout << "k: " << params.k << "\n";
        std::cout << "sindi_bk: " << params.sindi_bk << "\n";
        std::cout << "sindi_query_prune_ratio: " << params.sindi_query_prune_ratio << "\n";
        std::cout << "sindi_term_prune_ratio: " << params.sindi_term_prune_ratio << "\n";
        std::cout << "ef_search: " << params.ef_search << "\n";
        std::cout << "alpha: " << params.alpha << "\n";
        std::cout << "hybrid_prune_scale: " << params.hybrid_prune_scale << "\n";
        std::cout << "max_hops: " << params.max_hops << "\n";
        std::cout << "num_queries: " << actual_num_queries << "\n";
        std::cout << "rebuild: " << (params.rebuild ? "true" : "false") << "\n";
        std::cout << "index_dir: " << params.index_dir << "\n";
        std::cout << "----------------------------\n";
        std::cout << "Recall: " << avg_recall << "\n";
        std::cout << "QPS: " << qps << "\n";
        std::cout << "============================\n";

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
