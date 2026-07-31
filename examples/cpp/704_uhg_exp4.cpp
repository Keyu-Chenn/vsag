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
#include <omp.h>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct HybridResult {
    int64_t id;
    float dense_score;
    float sparse_score;
    float hybrid_score;

    void ComputeHybridScore(float alpha) {
        hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score;
    }

    bool operator>(const HybridResult& other) const {
        return hybrid_score > other.hybrid_score;
    }
};

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

float
ComputeDenseIP(const float* vec1, const float* vec2, int dim) {
    float score = 0;
    for (int i = 0; i < dim; i++) {
        score += vec1[i] * vec2[i];
    }
    return score;
}

float
ComputeSparseIP(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2) {
    float score = 0;
    int i = 0, j = 0;
    while (i < vec1.len_ && j < vec2.len_) {
        if (vec1.ids_[i] == vec2.ids_[j]) {
            score += vec1.vals_[i] * vec2.vals_[j];
            i++;
            j++;
        } else if (vec1.ids_[i] < vec2.ids_[j]) {
            i++;
        } else {
            j++;
        }
    }
    return score;
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

void
SaveNeighborsToHDF5(const std::string& filename,
                    const std::vector<std::vector<int64_t>>& all_neighbors,
                    float alpha) {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);

        size_t num_points = all_neighbors.size();
        size_t max_neighbors = 0;
        for (const auto& neighbors : all_neighbors) {
            max_neighbors = std::max(max_neighbors, neighbors.size());
        }

        std::cout << "Max neighbors count: " << max_neighbors << std::endl;

        // Save neighbors array
        hsize_t dims[2] = {num_points, max_neighbors};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = file.createDataSet("neighbors",
                                                  H5::PredType::NATIVE_INT64,
                                                  dataspace);

        std::vector<int64_t> flat_data(num_points * max_neighbors, -1);
        for (size_t i = 0; i < num_points; i++) {
            for (size_t j = 0; j < all_neighbors[i].size(); j++) {
                flat_data[i * max_neighbors + j] = all_neighbors[i][j];
            }
        }

        dataset.write(flat_data.data(), H5::PredType::NATIVE_INT64);

        // Save neighbor counts
        hsize_t count_dims[1] = {num_points};
        H5::DataSpace count_space(1, count_dims);
        H5::DataSet count_dataset = file.createDataSet("neighbor_counts",
                                                        H5::PredType::NATIVE_INT64,
                                                        count_space);

        std::vector<int64_t> counts;
        for (const auto& neighbors : all_neighbors) {
            counts.push_back(neighbors.size());
        }
        count_dataset.write(counts.data(), H5::PredType::NATIVE_INT64);

        // Save alpha value as attribute
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute alpha_attr = dataset.createAttribute("alpha",
                                                            H5::PredType::NATIVE_FLOAT,
                                                            attr_space);
        alpha_attr.write(H5::PredType::NATIVE_FLOAT, &alpha);

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

void
PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nOptions:\n"
              << "  -k <int>                   Top-k neighbors (default: 32)\n"
              << "  --bk <int>                 Candidate pool size (default: 100)\n"
              << "  --dense_ef_search <int>    HNSW ef_search for dense candidates (default: 200)\n"
              << "  --alpha <float>            Dense weight alpha (default: 0.5)\n"
              << "  --query_prune_ratio <float>\n"
              << "                              SINDI query pruning ratio (default: 0.5)\n"
              << "  --output_dir <path>        Output directory (default: /tbase-project/vsag/scripts/UHG/data/fhg)\n"
              << "  --num_points <int>         Points to process (default: all)\n"
              << "  --index_dir <path>         Index cache directory (default: /tbase-project/vsag/scripts/UHG/data/index)\n"
              << "  --threads <int>            Number of threads (default: 8)\n"
              << "  --rebuild                  Force rebuild indices\n"
              << "  --help, -h                 Show this help\n"
              << "\nOutput file: {dataset}_fhg_alpha_{alpha}.h5 in output_dir\n"
              << "\nExample:\n"
              << "  " << program_name << " /tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5 -k 32\n"
              << "  " << program_name << " nq.hdf5 --alpha 0.5 --threads 16\n"
              << std::endl;
}

struct Config {
    std::string h5_file;
    std::string output_dir = "/tbase-project/vsag/scripts/UHG/data/fhg";
    std::string index_dir = "/tbase-project/vsag/scripts/UHG/data/index";
    int k = 32;
    int bk = 100;
    int dense_ef_search = 200;
    float alpha = 0.5f;
    float query_prune_ratio = 0.5f;
    int num_points = -1;
    int threads = 8;
    bool rebuild = false;
};

Config
ParseCommandLine(int argc, char** argv) {
    Config config;

    if (argc < 2) {
        PrintUsage(argv[0]);
        exit(1);
    }

    config.h5_file = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            exit(0);
        } else if (arg == "-k" && i + 1 < argc) {
            config.k = std::atoi(argv[++i]);
            config.bk = 10 * config.k;
        } else if (arg == "--bk" && i + 1 < argc) {
            config.bk = std::atoi(argv[++i]);
        } else if (arg == "--dense_ef_search" && i + 1 < argc) {
            config.dense_ef_search = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            config.alpha = std::atof(argv[++i]);
        } else if (arg == "--query_prune_ratio" && i + 1 < argc) {
            config.query_prune_ratio = std::atof(argv[++i]);
        } else if (arg == "--output_dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--num_points" && i + 1 < argc) {
            config.num_points = std::atoi(argv[++i]);
        } else if (arg == "--index_dir" && i + 1 < argc) {
            config.index_dir = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::atoi(argv[++i]);
        } else if (arg == "--rebuild") {
            config.rebuild = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }

    if (config.k <= 0 || config.bk <= 0 || config.dense_ef_search <= 0) {
        std::cerr << "Error: k, bk, and dense_ef_search must be positive" << std::endl;
        exit(1);
    }
    if (config.dense_ef_search < config.bk) {
        std::cerr << "Error: dense_ef_search must be greater than or equal to bk" << std::endl;
        exit(1);
    }
    if (config.query_prune_ratio < 0.0f || config.query_prune_ratio > 0.9f) {
        std::cerr << "Error: query_prune_ratio must be in [0, 0.9]" << std::endl;
        exit(1);
    }

    return config;
}

int
main(int argc, char** argv) {
    vsag::init();

    Config config = ParseCommandLine(argc, argv);

    try {
        std::string dataset_name = GetDatasetName(config.h5_file);

        // Generate output filename with alpha value (e.g., nq_fhg_alpha_0_5.h5)
        std::ostringstream alpha_ss;
        alpha_ss << std::fixed << std::setprecision(1) << config.alpha;
        std::string alpha_str = alpha_ss.str();
        std::replace(alpha_str.begin(), alpha_str.end(), '.', '_');
        std::string output_file = config.output_dir + "/" + dataset_name + "_fhg_alpha_" + alpha_str + ".h5";

        // Index paths (use 701 prefix to reuse existing indices)
        std::string dense_index_path = config.index_dir + "/701_" + dataset_name + "_dense_hnsw.index";
        std::string sparse_index_path = config.index_dir + "/701_" + dataset_name + "_sparse_sindi.index";

        bool need_build_dense = config.rebuild || !FileExists(dense_index_path);
        bool need_build_sparse = config.rebuild || !FileExists(sparse_index_path);
        bool need_build = need_build_dense || need_build_sparse;

        std::cout << "\n========== FHG (Fixed Hybrid Graph) Generation ==========\n";
        std::cout << "Input: " << config.h5_file << "\n";
        std::cout << "Output: " << output_file << "\n";
        std::cout << "Dataset: " << dataset_name << "\n";
        std::cout << "Index dir: " << config.index_dir << "\n";
        std::cout << "k: " << config.k << "\n";
        std::cout << "bk: " << config.bk << "\n";
        std::cout << "dense_ef_search: " << config.dense_ef_search << "\n";
        std::cout << "alpha: " << config.alpha << "\n";
        std::cout << "query_prune_ratio: " << config.query_prune_ratio << "\n";
        std::cout << "threads: " << config.threads << "\n";
        std::cout << "rebuild: " << (config.rebuild ? "true" : "false") << "\n";

        /******************* 1. Load Dataset *****************/
        std::cout << "\nLoading dataset..." << std::endl;
        H5::H5File file(config.h5_file, H5F_ACC_RDONLY);

        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();
        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims);
        int64_t num_train = train_dims[0];
        int64_t dense_dim = train_dims[1];

        std::vector<float> train_dense(num_train * dense_dim);
        train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
        H5::DataSpace train_sparse_dataspace = train_sparse_dataset.getSpace();
        hsize_t train_sparse_size = train_sparse_dataspace.getSimpleExtentNpoints();

        std::vector<uint8_t> train_sparse_blob(train_sparse_size);
        train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto train_sparse = ParseSparseVectors(train_sparse_blob);

        H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
        std::vector<int64_t> train_labels(num_train);
        train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);

        int num_process = (config.num_points > 0) ? std::min(config.num_points, (int)num_train) : (int)num_train;

        std::cout << "Dataset: train=" << num_train << ", process=" << num_process
                  << ", dim=" << dense_dim << std::endl;

        /******************* 2. Build/Load Indices *****************/
        auto base = vsag::Dataset::Make();
        base->NumElements(num_train)
            ->Dim(dense_dim)
            ->Ids(train_labels.data())
            ->Float32Vectors(train_dense.data())
            ->SparseVectors(train_sparse.data())
            ->Owner(false);

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
            std::cout << "\nBuilding HNSW..." << std::endl;
            auto build_start = std::chrono::high_resolution_clock::now();
            if (!dense_index->Build(base).has_value()) {
                std::cerr << "Failed to build HNSW" << std::endl;
                return -1;
            }
            std::cout << "HNSW built in " << std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - build_start).count() << "s" << std::endl;

            std::ofstream out_stream(dense_index_path);
            auto serialize_result = dense_index->Serialize(out_stream);
            out_stream.close();
            if (!serialize_result.has_value()) {
                std::cerr << "Failed to save HNSW: " << serialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "HNSW saved to " << dense_index_path << std::endl;
        } else {
            std::cout << "\nLoading HNSW from " << dense_index_path << std::endl;
            dense_index = nullptr;
            dense_index = engine.CreateIndex("hnsw", hnsw_build_params).value();
            std::ifstream in_stream(dense_index_path);
            auto deserialize_result = dense_index->Deserialize(in_stream);
            in_stream.close();
            if (!deserialize_result.has_value()) {
                std::cerr << "Failed to load HNSW: " << deserialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "HNSW loaded (" << dense_index->GetNumElements() << " vectors)" << std::endl;
        }

        std::string sindi_build_params = R"(
        {
            "dtype": "sparse",
            "metric_type": "ip",
            "index_param": {"use_reorder": true}
        })";

        auto sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();

        if (need_build_sparse) {
            std::cout << "\nBuilding SINDI..." << std::endl;
            auto build_start = std::chrono::high_resolution_clock::now();
            if (!sparse_index->Build(base).has_value()) {
                std::cerr << "Failed to build SINDI" << std::endl;
                return -1;
            }
            std::cout << "SINDI built in " << std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - build_start).count() << "s" << std::endl;

            std::ofstream out_stream(sparse_index_path);
            auto serialize_result = sparse_index->Serialize(out_stream);
            out_stream.close();
            if (!serialize_result.has_value()) {
                std::cerr << "Failed to save SINDI: " << serialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "SINDI saved to " << sparse_index_path << std::endl;
        } else {
            std::cout << "\nLoading SINDI from " << sparse_index_path << std::endl;
            sparse_index = nullptr;
            sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();
            std::ifstream in_stream(sparse_index_path);
            auto deserialize_result = sparse_index->Deserialize(in_stream);
            in_stream.close();
            if (!deserialize_result.has_value()) {
                std::cerr << "Failed to load SINDI: " << deserialize_result.error().message << std::endl;
                return -1;
            }
            std::cout << "SINDI loaded (" << sparse_index->GetNumElements() << " vectors)" << std::endl;
        }

        /******************* 3. Generate KNNG with Fixed Alpha *****************/
        std::cout << "\nGenerating KNNG with alpha=" << config.alpha << " using " << config.threads << " threads..." << std::endl;

        std::vector<std::vector<int64_t>> all_neighbors(num_process);
        std::string dense_search_params = R"({"hnsw": {"ef_search": )" +
                                          std::to_string(config.dense_ef_search) + R"(}})";
        std::string sparse_search_params = R"({"sindi": {"query_prune_ratio": )" +
                                           std::to_string(config.query_prune_ratio) + "}}";
        std::cout << "sparse_search_params: " << sparse_search_params << std::endl;

        // Set thread count
        omp_set_num_threads(config.threads);

        auto gen_start = std::chrono::high_resolution_clock::now();
        int processed_count = 0;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_process; i++) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(dense_dim)
                ->Float32Vectors(train_dense.data() + i * dense_dim)
                ->SparseVectors(train_sparse.data() + i)
                ->Owner(false);

            auto dense_results = dense_index->KnnSearch(query, config.bk, dense_search_params).value();
            auto sparse_results = sparse_index->KnnSearch(query, config.bk, sparse_search_params).value();

            // Build candidate pool: union of dense & sparse results
            std::unordered_map<int64_t, HybridResult> candidate_map;
            candidate_map.reserve(config.bk * 2);

            for (int j = 0; j < dense_results->GetDim(); j++) {
                int64_t id = dense_results->GetIds()[j];
                if (id == train_labels[i]) continue;
                float ds = 1.0f - dense_results->GetDistances()[j];
                candidate_map[id] = {id, ds, 0.0f, 0.0f};
            }

            for (int j = 0; j < sparse_results->GetDim(); j++) {
                int64_t id = sparse_results->GetIds()[j];
                if (id == train_labels[i]) continue;
                float ss = 1.0f - sparse_results->GetDistances()[j];
                auto it = candidate_map.find(id);
                if (it != candidate_map.end()) {
                    it->second.sparse_score = ss;
                } else {
                    candidate_map[id] = {id, 0.0f, ss, 0.0f};
                }
            }

            // Fill missing scores with exact computation
            for (auto& [cid, c] : candidate_map) {
                if (c.dense_score == 0.0f) {
                    // Find the index of this label in train_labels
                    c.dense_score = ComputeDenseIP(
                        train_dense.data() + i * dense_dim,
                        train_dense.data() + cid * dense_dim, dense_dim);
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

            // Compute hybrid scores with fixed alpha
            for (auto& c : candidates) {
                c.ComputeHybridScore(config.alpha);
            }

            // Sort by hybrid score and take top-k
            std::partial_sort(candidates.begin(),
                              candidates.begin() + std::min(config.k, (int)candidates.size()),
                              candidates.end(), std::greater<HybridResult>());

            all_neighbors[i].reserve(config.k);
            for (int j = 0; j < std::min(config.k, (int)candidates.size()); j++) {
                all_neighbors[i].push_back(candidates[j].id);
            }

            // Progress output (thread-safe)
            #pragma omp atomic
            processed_count++;
            if (processed_count % 1000 == 0) {
                #pragma omp critical
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double>(now - gen_start).count();
                    double eta = elapsed / processed_count * (num_process - processed_count);
                    std::cout << "  [" << processed_count << "/" << num_process << "] " << elapsed << "s, eta:" << eta << "s" << std::endl;
                }
            }
        }

        auto gen_end = std::chrono::high_resolution_clock::now();
        std::cout << "\nGeneration time: " << std::chrono::duration<double>(gen_end - gen_start).count() << "s" << std::endl;

        /******************* 4. Statistics and Save *****************/
        double avg_neighbors = 0.0;
        int max_count = 0, min_count = INT_MAX;
        for (int i = 0; i < num_process; i++) {
            avg_neighbors += all_neighbors[i].size();
            max_count = std::max(max_count, (int)all_neighbors[i].size());
            min_count = std::min(min_count, (int)all_neighbors[i].size());
        }

        std::cout << "\nStatistics:\n";
        std::cout << "  avg neighbors: " << avg_neighbors / num_process << "\n";
        std::cout << "  max neighbors: " << max_count << "\n";
        std::cout << "  min neighbors: " << min_count << "\n";

        // Create output directory if not exists
        std::string mkdir_cmd = "mkdir -p " + config.output_dir;
        system(mkdir_cmd.c_str());

        std::cout << "\nSaving to " << output_file << std::endl;
        SaveNeighborsToHDF5(output_file, all_neighbors, config.alpha);

        std::cout << "\n========== Done ==========\n";

        FreeSparseVectors(train_sparse);
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
