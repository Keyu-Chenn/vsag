#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <cstring>
#include <chrono>
#include <iomanip>
#include<nlohmann/json.hpp>


// ============================================================
// 工具函数
// ============================================================

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

void FreeSparseVectors(std::vector<vsag::SparseVector>& sparse_vectors) {
    for (auto& vec : sparse_vectors) {
        delete[] vec.ids_;
        delete[] vec.vals_;
        vec.ids_ = nullptr;
        vec.vals_ = nullptr;
    }
}

float CalculateRecall(const std::vector<int64_t>& results,
                      const std::vector<int64_t>& ground_truth) {
    std::unordered_set<int64_t> gt_set(ground_truth.begin(), ground_truth.end());
    int hit_count = 0;
    for (auto id : results) {
        if (gt_set.count(id) > 0) hit_count++;
    }
    return static_cast<float>(hit_count) / static_cast<float>(ground_truth.size());
}

// ============================================================
// 参数解析
// ============================================================

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <h5_file> [options]\n"
              << "\nOptions:\n"
              << "  -k, --topk <int>           最终返回的 top-k 结果数 (default: 10)\n"
              << "  --sindi_bk <int>           sindi 召回数量，作为 hybrid 入口点 (default: 100)\n"
              << "  --ef_search <int>          hybrid index ef_search (default: 200)\n"
              << "  --alpha <float>            dense 分数权重，0~1 (default: 0.5)\n"
              << "  --num_queries <int>        测试查询数量，-1 表示全部 (default: -1)\n"
              << "  --dim <int>                dense 向量维度 (default: 从数据集读取)\n"
              << "  --help, -h                 显示此帮助信息\n"
              << std::endl;
}

struct SearchParams {
    std::string h5_file;
    int k           = 10;
    int sindi_bk    = 100;
    int ef_search   = 200;
    float alpha     = 0.5f;
    int num_queries = -1;
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
        } else if (arg == "--sindi_bk" && i + 1 < argc) {
            params.sindi_bk = std::atoi(argv[++i]);
        } else if (arg == "--ef_search" && i + 1 < argc) {
            params.ef_search = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            params.alpha = std::atof(argv[++i]);
        } else if (arg == "--num_queries" && i + 1 < argc) {
            params.num_queries = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }

    if (params.k <= 0) { std::cerr << "Error: k must be positive\n"; exit(1); }
    if (params.sindi_bk < params.k) { std::cerr << "Error: sindi_bk must be >= k\n"; exit(1); }
    if (params.alpha < 0.0f || params.alpha > 1.0f) { std::cerr << "Error: alpha must be in [0,1]\n"; exit(1); }

    return params;
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    vsag::init();

    SearchParams params = ParseCommandLine(argc, argv);

    try {
        /******************* 1. 加载数据集 *****************/
        H5::H5File file(params.h5_file, H5F_ACC_RDONLY);

        // -------- 训练集 dense --------
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();
        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims);
        int64_t num_train = train_dims[0];
        int64_t dense_dim  = train_dims[1];

        std::vector<float> train_dense(num_train * dense_dim);
        train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

        // -------- 训练集 sparse --------
        H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
        hsize_t train_sparse_size =
            train_sparse_dataset.getSpace().getSimpleExtentNpoints();
        std::vector<uint8_t> train_sparse_blob(train_sparse_size);
        train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto train_sparse = ParseSparseVectors(train_sparse_blob);

        // -------- 训练集 labels --------
        H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
        std::vector<int64_t> train_labels(num_train);
        train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);

        // -------- 测试集 dense --------
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();
        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims);
        int64_t num_test = test_dims[0];

        std::vector<float> test_dense(num_test * dense_dim);
        test_dataset.read(test_dense.data(), H5::PredType::NATIVE_FLOAT);

        // -------- 测试集 sparse --------
        H5::DataSet test_sparse_dataset = file.openDataSet("test_sparse");
        hsize_t test_sparse_size =
            test_sparse_dataset.getSpace().getSimpleExtentNpoints();
        std::vector<uint8_t> test_sparse_blob(test_sparse_size);
        test_sparse_dataset.read(test_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto test_sparse = ParseSparseVectors(test_sparse_blob);

        // -------- ground truth --------
        H5::DataSet gt_dataset = file.openDataSet("neighbors");
        H5::DataSpace gt_dataspace = gt_dataset.getSpace();
        hsize_t gt_dims[2];
        gt_dataspace.getSimpleExtentDims(gt_dims);
        int64_t num_gt_queries = gt_dims[0];
        int64_t k_gt           = gt_dims[1];

        std::vector<int64_t> ground_truth(num_gt_queries * k_gt);
        gt_dataset.read(ground_truth.data(), H5::PredType::NATIVE_INT64);

        std::cout << "Dataset loaded: "
                  << num_train << " train / "
                  << num_test  << " test / "
                  << "dense_dim=" << dense_dim << std::endl;

        /******************* 2. 构建 sindi 索引 *****************/
        std::string sindi_build_params = R"(
        {
            "dtype": "sparse",
            "metric_type": "ip",
            "index_param": {
                "use_reorder": false
            }
        })";

        auto base_sparse_dataset = vsag::Dataset::Make();
        base_sparse_dataset->NumElements(num_train)
            ->Ids(train_labels.data())
            ->SparseVectors(train_sparse.data())
            ->Owner(false);

        auto sparse_index = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();
        if (!sparse_index->Build(base_sparse_dataset).has_value()) {
            std::cerr << "Failed to build sindi index" << std::endl;
            return -1;
        }
        std::cout << "Sindi index built." << std::endl;

        /******************* 3. 构建 hybrid_index *****************/
        // alpha 写入 build 参数
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

        // 构建数据集：dense + sparse + ids
        auto base_dataset = vsag::Dataset::Make();
        base_dataset->NumElements(num_train)
            ->Dim(dense_dim)
            ->Ids(train_labels.data())
            ->Float32Vectors(train_dense.data())
            ->SparseVectors(train_sparse.data())
            ->Owner(false);

        if (!hybrid_index->Build(base_dataset).has_value()) {
            std::cerr << "Failed to build hybrid index" << std::endl;
            return -1;
        }
        std::cout << "Hybrid index built, elements: "
                  << hybrid_index->GetNumElements() << std::endl;

        /******************* 4. 检索：sindi → hybrid_index（以 sindi 结果为入口点） *****************/
        int actual_num_queries = (params.num_queries > 0)
            ? std::min(params.num_queries, (int)num_test)
            : (int)num_test;

        float total_recall = 0.0f;

        auto search_start = std::chrono::high_resolution_clock::now();

        for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {

            if ((query_idx + 1) % 100 == 0) {
                std::cout << "Processing query "
                          << query_idx + 1 << "/" << actual_num_queries
                          << "..." << std::endl;
            }

            /* ---- Step 1: sindi 召回 top sindi_bk，得到入口点 ---- */
            auto query_sparse_ds = vsag::Dataset::Make();
            query_sparse_ds->NumElements(1)
                ->SparseVectors(test_sparse.data() + query_idx)
                ->Owner(false);

            auto sindi_search_params = R"({"sindi": {}})";
            auto sindi_result = sparse_index->KnnSearch(
                query_sparse_ds, params.sindi_bk, sindi_search_params).value();

            int sindi_num = sindi_result->GetDim();
            const int64_t* sindi_ids = sindi_result->GetIds();

            // 收集入口点 id（sindi 返回的是 label id，hybrid index 用同一套 label）
            std::vector<int64_t> entry_points(sindi_ids, sindi_ids + sindi_num);

            /* ---- Step 2: hybrid_index 以 sindi 结果为入口点搜索 ---- */
            // 构造 search_parameters，传入多个入口点
            // 格式参考 hybrid_index 示例中的 entry_point 字段，
            // 这里扩展为 entry_points 数组
            nlohmann::json search_param_json = {
                {"alpha",        params.alpha},
                {"ef_search",    params.ef_search},
                {"entry_points", entry_points}   // 多入口点
            };
            std::string hybrid_search_params = search_param_json.dump();

            auto query_ds = vsag::Dataset::Make();
            query_ds->NumElements(1)
                ->Dim(dense_dim)
                ->Float32Vectors(test_dense.data() + query_idx * dense_dim)
                ->SparseVectors(test_sparse.data() + query_idx)
                ->Owner(false);


            auto hybrid_result = hybrid_index->KnnSearch(
                query_ds, params.k, hybrid_search_params).value();


            /* ---- Step 3: 计算召回率 ---- */
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
        auto search_end = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(
            search_end - search_start).count();
        double qps = static_cast<double>(actual_num_queries) / elapsed_seconds;

        float avg_recall = total_recall / static_cast<float>(actual_num_queries);

        /******************* 5. 输出结果 *****************/
        std::cout << "\n=== Results (sindi → hybrid_index with entry points) ===" << std::endl;
        std::cout << "k:             " << params.k            << std::endl;
        std::cout << "sindi_bk:      " << params.sindi_bk     << std::endl;
        std::cout << "ef_search:     " << params.ef_search     << std::endl;
        std::cout << "alpha:         " << params.alpha         << std::endl;
        std::cout << "num_queries:   " << actual_num_queries   << std::endl;
        std::cout << "avg_recall:    " << avg_recall           << std::endl;
        std::cout << "qps:           " << qps                  << std::endl;
        std::cout << "total_time(s): " << elapsed_seconds      << std::endl;

        /******************* 6. 对比实验：hybrid_index 不使用 sindi 入口点 *****************/
        std::cout << "\n=== Baseline (hybrid_index only, single entry point) ===" << std::endl;

        float total_recall_baseline = 0.0f;
        auto baseline_start = std::chrono::high_resolution_clock::now();

        for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {
            // 使用默认单入口点
            std::string hybrid_search_params_baseline =
                R"({
                    "alpha": )" + std::to_string(params.alpha) + R"(,
                    "ef_search": )" + std::to_string(params.ef_search) + R"(,
                    "entry_point": 0
                })";

            auto query_ds = vsag::Dataset::Make();
            query_ds->NumElements(1)
                ->Dim(dense_dim)
                ->Float32Vectors(test_dense.data() + query_idx * dense_dim)
                ->SparseVectors(test_sparse.data() + query_idx)
                ->Owner(false);

            auto hybrid_result = hybrid_index->KnnSearch(
                query_ds, params.k, hybrid_search_params_baseline).value();

            int result_num = hybrid_result->GetDim();
            std::vector<int64_t> search_results(
                hybrid_result->GetIds(),
                hybrid_result->GetIds() + result_num);

            int gt_k = std::min(params.k, (int)k_gt);
            std::vector<int64_t> gt(
                ground_truth.begin() + query_idx * k_gt,
                ground_truth.begin() + query_idx * k_gt + gt_k);

            // std::cout << "result VS gt: " << std::endl;
            // for (int x = 0; x < params.k; x++) {
            // std::cout << x << ": " << search_results[x] << "  " << gt[x] << std::endl;
            // }
            auto cur_recall = CalculateRecall(search_results, gt);
            // std::cout << std::endl << "query: " << query_idx << std::endl;
            // std::cout << "cur_recall: " << cur_recall << std::endl;
            total_recall_baseline += cur_recall;
        }
        std::cout << "total_recall: " << total_recall_baseline << std::endl;
        auto baseline_end = std::chrono::high_resolution_clock::now();
        double baseline_elapsed = std::chrono::duration<double>(
            baseline_end - baseline_start).count();
        double baseline_qps = static_cast<double>(actual_num_queries) / baseline_elapsed;
        float avg_recall_baseline = total_recall_baseline / static_cast<float>(actual_num_queries);

        std::cout << "avg_recall:    " << avg_recall_baseline  << std::endl;
        std::cout << "qps:           " << baseline_qps         << std::endl;
        std::cout << "total_time(s): " << baseline_elapsed     << std::endl;

        /******************* 7. 对比总结 *****************/
        std::cout << "\n=== Comparison ===" << std::endl;
        std::cout << std::left
                  << std::setw(30) << "Method"
                  << std::setw(15) << "avg_recall"
                  << std::setw(15) << "qps"
                  << std::endl;
        std::cout << std::setw(30) << "sindi + hybrid(multi-ep)"
                  << std::setw(15) << avg_recall
                  << std::setw(15) << qps
                  << std::endl;
        std::cout << std::setw(30) << "hybrid only (single-ep)"
                  << std::setw(15) << avg_recall_baseline
                  << std::setw(15) << baseline_qps
                  << std::endl;

        /******************* 8. 清理资源 *****************/
        FreeSparseVectors(train_sparse);
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
