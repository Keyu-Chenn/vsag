#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <fstream>

// 混合检索结果结构
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

// 解析稀疏向量
std::vector<vsag::SparseVector> ParseSparseVectors(const std::vector<uint8_t>& blob) {
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

// 释放稀疏向量内存
void FreeSparseVectors(std::vector<vsag::SparseVector>& sparse_vectors) {
    for (auto& vec : sparse_vectors) {
        delete[] vec.ids_;
        delete[] vec.vals_;
        vec.ids_ = nullptr;
        vec.vals_ = nullptr;
    }
}

// 计算密集向量内积
float ComputeDenseIP(const float* vec1, const float* vec2, int dim) {
    float score = 0;
    for (int i = 0; i < dim; i++) {
        score += vec1[i] * vec2[i];
    }
    return score;
}

// 计算稀疏向量内积
float ComputeSparseIP(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2) {
    float dis = 0;
    for (int m = 0; m < vec1.len_; m++) {
        for (int n = 0; n < vec2.len_; n++) {
            if (vec1.ids_[m] == vec2.ids_[n]) {
                dis += vec1.vals_[m] * vec2.vals_[n];
            }
        }
    }
    return dis;
}

// 计算召回率
float CalculateRecall(const std::vector<int64_t>& results,
                      const std::vector<int64_t>& ground_truth) {
    std::unordered_set<int64_t> gt_set(ground_truth.begin(), ground_truth.end());
    int hit_count = 0;

    for (auto id : results) {
        if (gt_set.count(id) > 0) {
            hit_count++;
        }
    }

    return static_cast<float>(hit_count) / ground_truth.size();
}

// 打印使用说明
void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nOptions:\n"
              << "  -k, --topk <int>           Number of final top results to return (default: 10)\n"
              << "  --bk_dense <int>           Number of candidates to recall from dense index (default: 100)\n"
              << "  --bk_sparse <int>          Number of candidates to recall from sparse index (default: 100)\n"
              << "  --bk <int>                 Set both bk_dense and bk_sparse to the same value\n"
              << "  --alpha <float>            Weight for dense score in hybrid scoring (default: 0.5)\n"
              << "  --num_queries <int>        Number of test queries to process (default: all)\n"
              << "  --ef_search <int>          Search parameter for dense and sparse index (default: 200)\n"
              << "  --help, -h                 Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " dataset.h5 -k 10 --bk 200 --alpha 0.5\n"
              << std::endl;
}

// 解析命令行参数
struct SearchParams {
    std::string h5_file;
    int k = 10;
    int bk_dense = 100;
    int bk_sparse = 100;
    float alpha = 0.5;
    int num_queries = -1;
    int ef_search = 200;
};

SearchParams ParseCommandLine(int argc, char** argv) {
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
        } else if (arg == "--ef_search" && i + 1 < argc) {
            params.ef_search = std::atoi(argv[++i]);
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
    if (params.alpha < 0.0 || params.alpha > 1.0) {
        std::cerr << "Error: alpha must be between 0.0 and 1.0\n";
        exit(1);
    }

    return params;
}

int main(int argc, char** argv) {
    vsag::init();

    SearchParams params = ParseCommandLine(argc, argv);

    try {
        /******************* 1. 加载数据集 *****************/
        H5::H5File file(params.h5_file, H5F_ACC_RDONLY);

        // 加载训练集
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();
        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims);
        int64_t num_train = train_dims[0];
        int64_t dense_dim = train_dims[1];

        std::vector<float> train_dense(num_train * dense_dim);
        train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

        // 加载稀疏向量
        H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
        H5::DataSpace train_sparse_dataspace = train_sparse_dataset.getSpace();
        hsize_t train_sparse_size = train_sparse_dataspace.getSimpleExtentNpoints();

        std::vector<uint8_t> train_sparse_blob(train_sparse_size);
        train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto train_sparse = ParseSparseVectors(train_sparse_blob);

        // 加载标签
        H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
        std::vector<int64_t> train_labels(num_train);
        train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);

        // 加载测试集
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();
        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims);
        int64_t num_test = test_dims[0];

        std::vector<float> test_dense(num_test * dense_dim);
        test_dataset.read(test_dense.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSet test_sparse_dataset = file.openDataSet("test_sparse");
        H5::DataSpace test_sparse_dataspace = test_sparse_dataset.getSpace();
        hsize_t test_sparse_size = test_sparse_dataspace.getSimpleExtentNpoints();

        std::vector<uint8_t> test_sparse_blob(test_sparse_size);
        test_sparse_dataset.read(test_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto test_sparse = ParseSparseVectors(test_sparse_blob);

        // 加载ground truth
        H5::DataSet gt_dataset = file.openDataSet("neighbors");
        H5::DataSpace gt_dataspace = gt_dataset.getSpace();
        hsize_t gt_dims[2];
        gt_dataspace.getSimpleExtentDims(gt_dims);
        int64_t num_gt_queries = gt_dims[0];
        int64_t k_gt = gt_dims[1];

        std::vector<int64_t> ground_truth(num_gt_queries * k_gt);
        gt_dataset.read(ground_truth.data(), H5::PredType::NATIVE_INT64);

        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);

        /******************* 2. 构建密集向量索引（hgraph） *****************/
        std::string hgraph_dense_params = R"(
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

        auto base_dense_dataset = vsag::Dataset::Make();
        base_dense_dataset->NumElements(num_train)
            ->Dim(dense_dim)
            ->Ids(train_labels.data())
            ->Float32Vectors(train_dense.data())
            ->Owner(false);

        auto dense_index = engine.CreateIndex("hgraph", hgraph_dense_params).value();
        if (!dense_index->Build(base_dense_dataset).has_value()) {
            std::cerr << "Failed to build dense index" << std::endl;
            return -1;
        }

        std::cout << "dense index built" << std::endl;

        /******************* 3. 构建稀疏向量索引（hgraph） *****************/
        // 与密集索引的区别：dtype 为 "sparse"，无需指定 dim
        std::string hgraph_sparse_params = R"(
        {
            "dtype": "sparse",
            "metric_type": "ip",
            "index_param": {
                "base_quantization_type": "sparse",
                "max_degree": 64,
                "ef_construction": 200
            }
        })";

        auto base_sparse_dataset = vsag::Dataset::Make();
        base_sparse_dataset->NumElements(num_train)
            ->Ids(train_labels.data())
            ->SparseVectors(train_sparse.data())
            ->Owner(false);

        // 改为 engine.CreateIndex("hgraph", ...) 而非 Factory::CreateIndex("sindi", ...)
        auto sparse_index = engine.CreateIndex("hgraph", hgraph_sparse_params).value();
        if (!sparse_index->Build(base_sparse_dataset).has_value()) {
            std::cerr << "Failed to build sparse hgraph index" << std::endl;
            return -1;
        }

        /******************* 4. 混合检索并计算召回率 *****************/
        int actual_num_queries = (params.num_queries > 0) ?
                                 std::min(params.num_queries, (int)num_test) :
                                 (int)num_test;

        float total_recall = 0.0f;
        float total_dist_compute_dense = 0.0f;
        float total_dist_compute_sparse = 0.0f;
        float total_hops_dense = 0.0f;
        float total_hops_sparse = 0.0f;


        auto search_start = std::chrono::high_resolution_clock::now();


        // 保存第 0 个 query 的 IDs
        std::vector<int64_t> dense_ids_file;
        std::vector<int64_t> sparse_ids_file;
        std::vector<int64_t> rerank_ids_file;


        for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {
            if ((query_idx + 1) % 100 == 0) {
                std::cout << "Processing query " << query_idx + 1
                          << "/" << actual_num_queries << "..." << std::endl;
            }

            // ── 4.1 密集向量召回 ──────────────────────────────────────────
            auto dense_query = vsag::Dataset::Make();
            dense_query->NumElements(1)
                ->Dim(dense_dim)
                ->Float32Vectors(test_dense.data() + query_idx * dense_dim)
                ->Owner(false);

            std::string dense_search_params =
                R"({"hgraph": {"ef_search": )" +
                std::to_string(params.ef_search) + "}}";

            auto dense_results = dense_index->KnnSearch(
                dense_query, params.bk_dense, dense_search_params).value();

            // 获取 dense 索引统计信息
            auto dense_stats_str = dense_results->GetStatistics();
            auto dense_stats_js = nlohmann::json::parse(dense_stats_str);
            if (dense_stats_js.contains("dist_cmp") && !dense_stats_js["dist_cmp"].is_null()) {
                total_dist_compute_dense += dense_stats_js["dist_cmp"].get<float>();
            }
            if (dense_stats_js.contains("hops") && !dense_stats_js["hops"].is_null()) {
                total_hops_dense += dense_stats_js["hops"].get<float>();
            }

            // 保存第 0 个 query 的 dense IDs
            std::vector<int64_t> dense_ids_for_file;
            std::vector<int64_t> sparse_ids_for_file;
            std::vector<int64_t> rerank_ids_for_file;
            if (query_idx == 0) {
                for (int i = 0; i < dense_results->GetDim(); i++) {
                    dense_ids_for_file.push_back(dense_results->GetIds()[i]);
                }
            }

            // ── 4.2 稀疏向量召回（hgraph） ────────────────────────────────
            auto sparse_query = vsag::Dataset::Make();
            sparse_query->NumElements(1)
                ->SparseVectors(test_sparse.data() + query_idx)
                ->Owner(false);

            std::string sparse_search_params =
                R"({"hgraph": {"ef_search": )" +
                std::to_string(params.ef_search) + "}}";

            auto sparse_results = sparse_index->KnnSearch(
                sparse_query, params.bk_sparse, sparse_search_params).value();

            // 获取 sparse 索引统计信息
            auto sparse_stats_str = sparse_results->GetStatistics();
            auto sparse_stats_js = nlohmann::json::parse(sparse_stats_str);
            if (sparse_stats_js.contains("dist_cmp") && !sparse_stats_js["dist_cmp"].is_null()) {
                total_dist_compute_sparse += sparse_stats_js["dist_cmp"].get<float>();
            }
            if (sparse_stats_js.contains("hops") && !sparse_stats_js["hops"].is_null()) {
                total_hops_sparse += sparse_stats_js["hops"].get<float>();
            }

            // ── 4.3 合并两路召回结果（去重） ──────────────────────────────
            std::unordered_set<int64_t> candidate_ids;


            for (int i = 0; i < dense_results->GetDim(); i++) {
                candidate_ids.insert(dense_results->GetIds()[i]);
                if (query_idx == 0) {
                    dense_ids_file.push_back(dense_results->GetIds()[i]);
                }
            }
            for (int i = 0; i < sparse_results->GetDim(); i++) {
                candidate_ids.insert(sparse_results->GetIds()[i]);
                if (query_idx == 0) {
                    sparse_ids_file.push_back(sparse_results->GetIds()[i]);
                }
            }

            // ── 4.4 统一重新计算所有候选的距离并重排序 ────────────────────
            std::vector<HybridResult> results;
            results.reserve(candidate_ids.size());

            for (int64_t id : candidate_ids) {
                HybridResult result;
                result.id = id;

                result.dense_score = ComputeDenseIP(
                    test_dense.data() + query_idx * dense_dim,
                    train_dense.data() + id * dense_dim,
                    dense_dim
                );

                result.sparse_score = ComputeSparseIP(
                    test_sparse[query_idx],
                    train_sparse[id]
                );

                result.ComputeHybridScore(params.alpha);
                results.push_back(result);
            }

            // ── 4.5 partial_sort 取 top-k ─────────────────────────────────
            int actual_k = std::min(params.k, (int)results.size());
            std::partial_sort(results.begin(),
                              results.begin() + actual_k,
                              results.end(),
                              std::greater<HybridResult>());

            std::vector<int64_t> search_results;
            search_results.reserve(actual_k);
            for (int i = 0; i < actual_k; i++) {
                search_results.push_back(results[i].id);
                if (query_idx == 0) {
                    rerank_ids_file.push_back(results[i].id);
                }
            }

            // ── 4.6 计算召回率 ────────────────────────────────────────────
            int gt_k = std::min(params.k, (int)k_gt);
            std::vector<int64_t> gt(
                ground_truth.begin() + query_idx * k_gt,
                ground_truth.begin() + query_idx * k_gt + gt_k);

            total_recall += CalculateRecall(search_results, gt);
        }

        auto search_end = std::chrono::high_resolution_clock::now();
        double elapsed_seconds =
            std::chrono::duration<double>(search_end - search_start).count();
        double qps = actual_num_queries / elapsed_seconds;

        float avg_recall = total_recall / actual_num_queries;
        float avg_dist_compute_dense = total_dist_compute_dense / actual_num_queries;
        float avg_dist_compute_sparse = total_dist_compute_sparse / actual_num_queries;
        float avg_hops_dense = total_hops_dense / actual_num_queries;
        float avg_hops_sparse = total_hops_sparse / actual_num_queries;
        float avg_dist_compute_total = avg_dist_compute_dense + avg_dist_compute_sparse;
        float avg_hops_total = avg_hops_dense + avg_hops_sparse;

        /******************* 5. 输出结果 *****************/
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "k:                    " << params.k                  << std::endl;
        std::cout << "bk_dense:             " << params.bk_dense           << std::endl;
        std::cout << "bk_sparse:            " << params.bk_sparse          << std::endl;
        std::cout << "alpha:                " << params.alpha              << std::endl;
        std::cout << "num_queries:          " << actual_num_queries        << std::endl;
        std::cout << "avg_recall:           " << avg_recall                << std::endl;
        std::cout << "qps:                  " << qps                       << std::endl;
        std::cout << std::endl;
        std::cout << "=== Search Statistics ===" << std::endl;
        std::cout << "avg_dist_compute_dense:   " << avg_dist_compute_dense    << std::endl;
        std::cout << "avg_dist_compute_sparse:  " << avg_dist_compute_sparse   << std::endl;
        std::cout << "avg_dist_compute_total:   " << avg_dist_compute_total    << std::endl;
        std::cout << "avg_hops_dense:           " << avg_hops_dense            << std::endl;
        std::cout << "avg_hops_sparse:          " << avg_hops_sparse           << std::endl;
        std::cout << "avg_hops_total:           " << avg_hops_total            << std::endl;

        /******************* 6. 保存 ID 到文件 *****************/
        std::ofstream outfile("606_hybrid_exp6_baseline_id_check.txt");
        outfile << "# 606_hybrid_exp6 baseline ID check" << std::endl;
        outfile << "# k=" << params.k << ", bk_dense=" << params.bk_dense
                << ", bk_sparse=" << params.bk_sparse << ", alpha=" << params.alpha << std::endl;
        outfile << std::endl;
        outfile << "dense_top" << params.bk_dense << "_ids: ";
        for (size_t i = 0; i < dense_ids_file.size(); i++) {
            outfile << dense_ids_file[i];
            if (i < dense_ids_file.size() - 1) outfile << ",";
        }
        outfile << std::endl;
        outfile << "sparse_top" << params.bk_sparse << "_ids: ";
        for (size_t i = 0; i < sparse_ids_file.size(); i++) {
            outfile << sparse_ids_file[i];
            if (i < sparse_ids_file.size() - 1) outfile << ",";
        }
        outfile << std::endl;
        outfile << "rerank_top" << params.k << "_ids: ";
        for (size_t i = 0; i < rerank_ids_file.size(); i++) {
            outfile << rerank_ids_file[i];
            if (i < rerank_ids_file.size() - 1) outfile << ",";
        }
        outfile << std::endl;
        outfile.close();
        std::cout << "IDs saved to 606_hybrid_exp6_baseline_id_check.txt" << std::endl;

        /******************* 7. 清理资源 *****************/
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

