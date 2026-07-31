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

// Dense-only (HNSW) baseline for UHG experiments.
// Candidates come from HNSW only, then reranked by hybrid score (alpha * dense + (1-alpha) * sparse).
// Uses raw sparse vectors for the sparse score, so no SINDI index is needed.

#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "uhg_mixed_alpha_utils.h"

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
            if (offset + vals_bytes > blob.size()) { delete[] vec.ids_; break; }
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
        if (gt_set.count(id) > 0) hit_count++;
    }
    return static_cast<float>(hit_count) / ground_truth.size();
}

float
ComputeSparseIP(const vsag::SparseVector& lhs, const vsag::SparseVector& rhs) {
    float score = 0.0f;
    uint32_t i = 0;
    uint32_t j = 0;
    while (i < lhs.len_ && j < rhs.len_) {
        if (lhs.ids_[i] == rhs.ids_[j]) {
            score += lhs.vals_[i] * rhs.vals_[j];
            ++i;
            ++j;
        } else if (lhs.ids_[i] < rhs.ids_[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return score;
}

bool
LabelsAreIdentity(const std::vector<int64_t>& labels) {
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] != static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

std::unordered_map<int64_t, int64_t>
BuildLabelToOffset(const std::vector<int64_t>& labels) {
    std::unordered_map<int64_t, int64_t> label_to_offset;
    label_to_offset.reserve(labels.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        label_to_offset.emplace(labels[i], static_cast<int64_t>(i));
    }
    return label_to_offset;
}

int64_t
ResolveLabelOffset(int64_t label,
                   bool labels_are_identity,
                   int64_t num_train,
                   const std::unordered_map<int64_t, int64_t>& label_to_offset) {
    if (labels_are_identity) {
        if (label >= 0 && label < num_train) {
            return label;
        }
    } else {
        auto it = label_to_offset.find(label);
        if (it != label_to_offset.end()) {
            return it->second;
        }
    }
    throw std::runtime_error("Candidate id not found in train_labels: " + std::to_string(label));
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

std::vector<int64_t>
ReadNPYInt64(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open npy file: " + path);
    char magic[8];
    file.read(magic, 8);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY")
        throw std::runtime_error("Invalid npy file format: " + path);
    uint8_t major = static_cast<uint8_t>(magic[6]);
    uint32_t header_len;
    if (major == 1) { uint16_t hlen; file.read(reinterpret_cast<char*>(&hlen), 2); header_len = hlen; }
    else if (major == 2) { file.read(reinterpret_cast<char*>(&header_len), 4); }
    else throw std::runtime_error("Unsupported npy version: " + std::to_string(major));
    std::string header(header_len, '\0');
    file.read(header.data(), header_len);
    shape.clear();
    size_t shape_pos = header.find("'shape':");
    if (shape_pos != std::string::npos) {
        size_t ps = header.find("(", shape_pos), pe = header.find(")", ps);
        if (ps != std::string::npos && pe != std::string::npos) {
            std::string ss = header.substr(ps + 1, pe - ps - 1);
            std::stringstream sss(ss);
            std::string tok;
            while (std::getline(sss, tok, ',')) {
                tok.erase(0, tok.find_first_not_of(" \t"));
                tok.erase(tok.find_last_not_of(" \t") + 1);
                if (!tok.empty()) shape.push_back(std::stoll(tok));
            }
        }
    }
    if (shape.empty()) throw std::runtime_error("Failed to parse shape from npy header");
    int64_t total = 1;
    for (auto s : shape) total *= s;
    std::vector<int64_t> data(total);
    file.read(reinterpret_cast<char*>(data.data()), total * sizeof(int64_t));
    return data;
}

bool
FileExists(const std::string& path) { std::ifstream f(path); return f.good(); }

std::string
GetDatasetName(const std::string& h5_file) {
    std::string filename = h5_file;
    size_t ls = filename.find_last_of('/');
    if (ls != std::string::npos) filename = filename.substr(ls + 1);
    size_t ld = filename.find_last_of('.');
    return (ld != std::string::npos) ? filename.substr(0, ld) : filename;
}

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nDense-only (HNSW) baseline for hybrid retrieval.\n"
              << "Candidates from HNSW only, reranked by hybrid score.\n"
              << "\nOptions:\n"
              << "  -k, --topk <int>           Number of final top results (default: 100)\n"
              << "  --bk <int>                 Candidates from HNSW before rerank (default: same as k)\n"
              << "  --alpha <float>            Dense score weight (default: 0.5)\n"
              << "  --num_queries <int>        Number of queries to process (default: all)\n"
              << "  --threads <int>            Query-level parallel search threads (default: 1)\n"
              << "  --ef_search <int>          ef_search for HNSW (default: 200)\n"
              << "  --search_points <csv>      Run bk=ef_search points after one data/index load\n"
              << "  --build_only               Build/save HNSW and exit without search\n"
              << "  --rebuild                  Force rebuild indexes\n"
              << "  --index_dir <path>         Directory to save/load indexes (default: ./indexes)\n"
              << "  --gt_dir <path>            Directory containing ground truth npy files (required unless --build_only)\n"
              << "  --mixed_alpha_file <path> Per-query alpha float32 npy; use with --mixed_gt_file\n"
              << "  --mixed_gt_file <path>    Ground truth int64 npy for mixed per-query alphas\n"
              << "  --help, -h                 Show this help message\n"
              << "\nIndex file: 701_{dataset}_dense_hnsw.index\n"
              << std::endl;
}

struct SearchParams {
    std::string h5_file;
    int k = 100;
    int bk = -1;  // -1 means same as k
    float alpha = 0.5f;
    int num_queries = -1;
    int threads = 1;
    int ef_search = 200;
    std::vector<int> search_points;
    bool build_only = false;
    bool rebuild = false;
    std::string index_dir = "./indexes";
    std::string gt_dir = "";
    std::string mixed_alpha_file;
    std::string mixed_gt_file;
};

SearchParams
ParseCommandLine(int argc, char** argv) {
    SearchParams params;
    if (argc < 2) { PrintUsage(argv[0]); exit(1); }
    std::string arg1 = argv[1];
    if (arg1 == "--help" || arg1 == "-h") { PrintUsage(argv[0]); exit(0); }
    params.h5_file = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { PrintUsage(argv[0]); exit(0); }
        else if ((arg == "-k" || arg == "--topk") && i + 1 < argc) params.k = std::atoi(argv[++i]);
        else if (arg == "--bk" && i + 1 < argc) params.bk = std::atoi(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc) params.alpha = std::atof(argv[++i]);
        else if (arg == "--num_queries" && i + 1 < argc) params.num_queries = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) params.threads = std::atoi(argv[++i]);
        else if (arg == "--ef_search" && i + 1 < argc) params.ef_search = std::atoi(argv[++i]);
        else if (arg == "--search_points" && i + 1 < argc)
            params.search_points = ParseSearchPoints(argv[++i]);
        else if (arg == "--build_only") params.build_only = true;
        else if (arg == "--rebuild") params.rebuild = true;
        else if (arg == "--index_dir" && i + 1 < argc) params.index_dir = argv[++i];
        else if (arg == "--gt_dir" && i + 1 < argc) params.gt_dir = argv[++i];
        else if (arg == "--mixed_alpha_file" && i + 1 < argc) params.mixed_alpha_file = argv[++i];
        else if (arg == "--mixed_gt_file" && i + 1 < argc) params.mixed_gt_file = argv[++i];
        else { std::cerr << "Unknown argument: " << arg << std::endl; PrintUsage(argv[0]); exit(1); }
    }
    if (params.k <= 0) { std::cerr << "Error: k must be positive\n"; exit(1); }
    if (params.threads <= 0) { std::cerr << "Error: threads must be positive\n"; exit(1); }
    for (int point : params.search_points) {
        if (point <= 0) { std::cerr << "Error: search_points must be positive\n"; exit(1); }
    }
    if (params.alpha < 0.0f || params.alpha > 1.0f) { std::cerr << "Error: alpha must be [0,1]\n"; exit(1); }
    return params;
}

int
main(int argc, char** argv) {
    vsag::init();
    SearchParams params = ParseCommandLine(argc, argv);

        try {
        std::string dataset_name = GetDatasetName(params.h5_file);
        std::string dense_index_path = params.index_dir + "/701_" + dataset_name + "_dense_hnsw.index";

        bool need_build_dense = params.rebuild || !FileExists(dense_index_path);

        if (need_build_dense) { std::string cmd = "mkdir -p " + params.index_dir; system(cmd.c_str()); }

        std::vector<float> train_dense;
        std::vector<vsag::SparseVector> train_sparse;
        std::vector<int64_t> train_labels;
        std::vector<float> test_dense;
        std::vector<vsag::SparseVector> test_sparse;
        int64_t num_train = 0;
        int64_t num_test = 0;
        int64_t dense_dim = 0;

        std::cout << "Loading data from " << params.h5_file << std::endl;
        H5::H5File file(params.h5_file, H5F_ACC_RDONLY);

        if (!params.build_only) {
            H5::DataSet test_dataset = file.openDataSet("test");
            H5::DataSpace test_dataspace = test_dataset.getSpace();
            hsize_t test_dims[2];
            test_dataspace.getSimpleExtentDims(test_dims);
            num_test = test_dims[0];
            dense_dim = test_dims[1];
            test_dense.resize(num_test * dense_dim);
            test_dataset.read(test_dense.data(), H5::PredType::NATIVE_FLOAT);

            H5::DataSet test_sparse_ds = file.openDataSet("test_sparse");
            hsize_t test_sparse_size = test_sparse_ds.getSpace().getSimpleExtentNpoints();
            std::vector<uint8_t> test_sparse_blob(test_sparse_size);
            test_sparse_ds.read(test_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
            test_sparse = ParseSparseVectors(test_sparse_blob);

            std::cout << "Test data loaded: test=" << num_test << ", dim=" << dense_dim << std::endl;
        }

        {
            std::cout << "Loading train labels..." << std::endl;
            H5::DataSet train_labels_ds = file.openDataSet("train_labels");
            hsize_t train_label_dims[1];
            train_labels_ds.getSpace().getSimpleExtentDims(train_label_dims);
            num_train = static_cast<int64_t>(train_label_dims[0]);
            train_labels.resize(num_train);
            train_labels_ds.read(train_labels.data(), H5::PredType::NATIVE_INT64);
        }

        if (!params.build_only) {
            std::cout << "Loading train sparse data for raw rerank..." << std::endl;
            H5::DataSet train_sparse_ds = file.openDataSet("train_sparse");
            hsize_t train_sparse_size = train_sparse_ds.getSpace().getSimpleExtentNpoints();
            std::vector<uint8_t> train_sparse_blob(train_sparse_size);
            train_sparse_ds.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
            train_sparse = ParseSparseVectors(train_sparse_blob);
            if (static_cast<int64_t>(train_sparse.size()) != num_train) {
                throw std::runtime_error("train_sparse count does not match train_labels count");
            }
            std::cout << "Train sparse loaded: train=" << num_train << std::endl;
        }

        bool labels_are_identity = LabelsAreIdentity(train_labels);
        std::unordered_map<int64_t, int64_t> label_to_offset;
        if (!labels_are_identity) {
            label_to_offset = BuildLabelToOffset(train_labels);
        }

        if (need_build_dense) {
            std::cout << "Loading train dense data for HNSW build..." << std::endl;
            H5::DataSet train_dataset = file.openDataSet("train");
            H5::DataSpace train_dataspace = train_dataset.getSpace();
            hsize_t train_dims[2];
            train_dataspace.getSimpleExtentDims(train_dims);
            if (params.build_only) {
                dense_dim = static_cast<int64_t>(train_dims[1]);
            }
            if (static_cast<int64_t>(train_dims[0]) != num_train ||
                (!params.build_only && static_cast<int64_t>(train_dims[1]) != dense_dim)) {
                throw std::runtime_error("train dense shape does not match labels/test dim");
            }
            train_dense.resize(num_train * dense_dim);
            train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);
            std::cout << "Train dense loaded: train=" << num_train << std::endl;
        }

        if (params.build_only && !need_build_dense) {
            std::cout << "Build-only requested and HNSW already exists: " << dense_index_path
                      << std::endl;
            return 0;
        }

        if (params.build_only) {
            std::string hnsw_build_params = R"({"dtype":"float32","metric_type":"ip","dim":)" +
                std::to_string(dense_dim) +
                R"(,"hnsw":{"max_degree":64,"ef_construction":200}})";
            vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
            vsag::Engine engine(&resource);
            auto dense_index = engine.CreateIndex("hnsw", hnsw_build_params).value();

            std::cout << "Building HNSW..." << std::endl;
            auto base = vsag::Dataset::Make();
            base->NumElements(num_train)->Dim(dense_dim)->Ids(train_labels.data())
                ->Float32Vectors(train_dense.data())->Owner(false);
            auto t0 = std::chrono::high_resolution_clock::now();
            dense_index->Build(base);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::cout << "HNSW built in " << std::chrono::duration<double>(t1 - t0).count()
                      << "s" << std::endl;
            std::ofstream out(dense_index_path);
            dense_index->Serialize(out);
            out.close();
            std::cout << "HNSW saved to " << dense_index_path << std::endl;
            std::cout << "Build-only complete. Exiting before search." << std::endl;
            engine.Shutdown();
            return 0;
        }

        // Load fixed-alpha or per-query mixed-alpha ground truth.
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
        if (gt_shape.size() != 2) { std::cerr << "Error: GT must be 2D\n"; return -1; }
        int64_t k_gt = gt_shape[1];

        // Build/Load Dense Index (HNSW)
        std::string hnsw_build_params = R"({"dtype":"float32","metric_type":"ip","dim":)" +
            std::to_string(dense_dim) +
            R"(,"hnsw":{"max_degree":64,"ef_construction":200}})";
        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);
        auto dense_index = engine.CreateIndex("hnsw", hnsw_build_params).value();

        if (need_build_dense) {
            std::cout << "Building HNSW..." << std::endl;
            auto base = vsag::Dataset::Make();
            base->NumElements(num_train)->Dim(dense_dim)->Ids(train_labels.data())
                ->Float32Vectors(train_dense.data())->Owner(false);
            auto t0 = std::chrono::high_resolution_clock::now();
            dense_index->Build(base);
            auto t1 = std::chrono::high_resolution_clock::now();
            std::cout << "HNSW built in " << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
            std::ofstream out(dense_index_path);
            dense_index->Serialize(out);
            out.close();
            std::cout << "HNSW saved to " << dense_index_path << std::endl;
        } else {
            std::cout << "Loading HNSW from " << dense_index_path << std::endl;
            dense_index = nullptr;
            dense_index = engine.CreateIndex("hnsw", hnsw_build_params).value();
            std::ifstream in(dense_index_path);
            dense_index->Deserialize(in);
            in.close();
            int64_t dense_count = dense_index->GetNumElements();
            if (dense_count != num_train) {
                throw std::runtime_error("HNSW index count does not match train_labels count");
            }
            std::cout << "HNSW loaded (" << dense_count << " vectors)" << std::endl;
        }

        // Search: HNSW retrieves bk candidates -> hybrid rerank -> top-k
        int actual_num_queries =
            (params.num_queries > 0) ? std::min(params.num_queries, (int)num_test) : (int)num_test;
        if (gt_shape[0] < actual_num_queries ||
            (mixed_alpha_mode && static_cast<int>(mixed_alphas.size()) < actual_num_queries)) {
            throw std::runtime_error("Mixed alpha/ground truth rows are fewer than num_queries");
        }
        std::vector<int> search_points = params.search_points;
        if (search_points.empty()) {
            search_points.push_back((params.bk > 0) ? params.bk : params.k);
        }

        for (int effective_bk : search_points) {
            const int effective_ef_search =
                params.search_points.empty() ? params.ef_search : effective_bk;
            std::vector<float> query_recalls(actual_num_queries, 0.0f);
            std::string dense_search_params =
                R"({"hnsw": {"ef_search": )" + std::to_string(effective_ef_search) + "}}";

            auto search_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(params.threads) schedule(dynamic)
            for (int qi = 0; qi < actual_num_queries; qi++) {
                const float query_alpha = mixed_alpha_mode ? mixed_alphas[qi] : params.alpha;
                if (params.threads == 1 && (qi + 1) % 100 == 0)
                    std::cout << "Processing query " << qi + 1 << "/" << actual_num_queries
                              << " (point=" << effective_bk << ")" << std::endl;

                auto query = vsag::Dataset::Make();
                query->NumElements(1)->Dim(dense_dim)
                    ->Float32Vectors(test_dense.data() + qi * dense_dim)
                    ->SparseVectors(test_sparse.data() + qi)
                    ->Owner(false);

                auto dense_results =
                    dense_index->KnnSearch(query, effective_bk, dense_search_params).value();
                int result_num = dense_results->GetDim();
                std::vector<int64_t> candidate_ids(dense_results->GetIds(),
                                                   dense_results->GetIds() + result_num);

                const float* dense_dists = dense_results->GetDistances();
                std::vector<std::pair<int64_t, float>> hybrid_results;
                hybrid_results.reserve(result_num);
                for (int i = 0; i < result_num; i++) {
                    int64_t offset = ResolveLabelOffset(candidate_ids[i],
                                                        labels_are_identity,
                                                        num_train,
                                                        label_to_offset);
                    float dense_score = 1.0f - dense_dists[i];
                    float sparse_score = ComputeSparseIP(test_sparse[qi], train_sparse[offset]);
                    float hybrid = query_alpha * dense_score + (1.0f - query_alpha) * sparse_score;
                    hybrid_results.emplace_back(candidate_ids[i], hybrid);
                }

                int actual_k = std::min(params.k, (int)hybrid_results.size());
                std::partial_sort(hybrid_results.begin(), hybrid_results.begin() + actual_k,
                                  hybrid_results.end(),
                                  [](const auto& a, const auto& b) { return a.second > b.second; });

                std::vector<int64_t> search_results;
                for (int i = 0; i < actual_k; i++)
                    search_results.push_back(hybrid_results[i].first);

                int gt_k = std::min(params.k, (int)k_gt);
                std::vector<int64_t> gt(ground_truth.begin() + qi * k_gt,
                                        ground_truth.begin() + qi * k_gt + gt_k);
                query_recalls[qi] = CalculateRecall(search_results, gt);
            }

            auto search_end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(search_end - search_start).count();
            double qps = static_cast<double>(actual_num_queries) / elapsed;
            double total_recall = 0.0;
            for (float recall : query_recalls) {
                total_recall += recall;
            }
            float avg_recall = static_cast<float>(total_recall / actual_num_queries);

            std::cout << "\n========== Results ==========\n";
            std::cout << "Method: dense_only\n";
            std::cout << "k: " << params.k << "\n";
            std::cout << "bk: " << effective_bk << "\n";
            std::cout << "alpha: " << params.alpha << "\n";
            std::cout << "mixed_alpha: " << (mixed_alpha_mode ? "true" : "false") << "\n";
            std::cout << "ef_search: " << effective_ef_search << "\n";
            std::cout << "num_queries: " << actual_num_queries << "\n";
            std::cout << "threads: " << params.threads << "\n";
            std::cout << "----------------------------\n";
            std::cout << "Recall: " << avg_recall << "\n";
            std::cout << "QPS: " << qps << "\n";
            std::cout << "============================\n";
        }

        FreeSparseVectors(train_sparse);
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
