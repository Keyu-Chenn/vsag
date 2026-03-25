#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <cstring>
#include <chrono>

// ============================================================
// 数据结构
// ============================================================

struct RerankedResult {
    int64_t id;
    float dense_score;
    float sparse_score;
    float hybrid_score;

    void ComputeHybridScore(float alpha) {
        hybrid_score = alpha * dense_score + (1.0f - alpha) * sparse_score;
    }

    bool operator>(const RerankedResult& other) const {
        return hybrid_score > other.hybrid_score;
    }
};

// ============================================================
// 工具函数（与原代码保持一致）
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

float ComputeDenseIP(const float* vec1, const float* vec2, int dim) {
    float score = 0.0f;
    for (int i = 0; i < dim; i++) {
        score += vec1[i] * vec2[i];
    }
    return score;
}

float ComputeSparseIP(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2) {
    float dis = 0.0f;
    for (uint32_t m = 0; m < vec1.len_; m++) {
        for (uint32_t n = 0; n < vec2.len_; n++) {
            if (vec1.ids_[m] == vec2.ids_[n]) {
                dis += vec1.vals_[m] * vec2.vals_[n];
            }
        }
    }
    return dis;
}

float CalculateRecall(const std::vector<int64_t>& results,
                      const std::vector<int64_t>& ground_truth) {
    std::unordered_set<int64_t> gt_set(ground_truth.begin(), ground_truth.end());
    int hit_count = 0;
    for (auto id : results) {
        if (gt_set.count(id) > 0) {
            hit_count++;
        }
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
              << "  --sindi_bk <int>           sindi 一阶段召回数量 (default: 10000)\n"
              << "  --alpha <float>            dense 分数权重，0~1 (default: 0.5)\n"
              << "  --num_queries <int>        测试查询数量，-1 表示全部 (default: -1)\n"
              << "  --help, -h                 显示此帮助信息\n"
              << "\nExample:\n"
              << "  " << program_name << " dataset.h5 -k 10 --sindi_bk 10000 --alpha 0.5\n"
              << std::endl;
}

struct SearchParams {
    std::string h5_file;
    int k         = 10;
    int sindi_bk  = 10000;   // sindi 一阶段召回数
    float alpha   = 0.5f;
    int num_queries = -1;    // -1 表示全部
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

    // 参数校验
    if (params.k <= 0) {
        std::cerr << "Error: k must be positive\n";
        exit(1);
    }
    if (params.sindi_bk < params.k) {
        std::cerr << "Error: sindi_bk must be >= k\n";
        exit(1);
    }
    if (params.alpha < 0.0f || params.alpha > 1.0f) {
        std::cerr << "Error: alpha must be between 0.0 and 1.0\n";
        exit(1);
    }

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
                  << "dim=" << dense_dim << std::endl;

        /******************* 2. 仅构建 sindi（稀疏索引） *****************/
        // 本方案一阶段只用 sindi，不需要 dense index
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

        /******************* 3. 检索：sindi → 精确重排 *****************/
        int actual_num_queries = (params.num_queries > 0)
            ? std::min(params.num_queries, (int)num_test)
            : (int)num_test;

        float total_recall = 0.0f;

        // ---------- 计时开始 ----------
        auto search_start = std::chrono::high_resolution_clock::now();
        // ------------------------------

        for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {

            if ((query_idx + 1) % 100 == 0) {
                std::cout << "Processing query "
                          << query_idx + 1 << "/" << actual_num_queries
                          << "..." << std::endl;
            }

            /* ---- Step 1: sindi 一阶段召回 top sindi_bk ---- */
            auto query_sparse = vsag::Dataset::Make();
            query_sparse->NumElements(1)
                ->SparseVectors(test_sparse.data() + query_idx)
                ->Owner(false);

            auto sindi_search_params = R"({"sindi": {}})";
            auto sindi_results = sparse_index->KnnSearch(
                query_sparse, params.sindi_bk, sindi_search_params).value();

            int sindi_num = sindi_results->GetDim();   // 实际召回数（可能 < sindi_bk）

            /* ---- Step 2: 精确重排（dense + sparse 混合分数） ---- */
            const float* query_dense_vec = test_dense.data() + query_idx * dense_dim;
            const vsag::SparseVector& query_sparse_vec = test_sparse[query_idx];

            std::vector<RerankedResult> candidates;
            candidates.reserve(sindi_num);

            for (int i = 0; i < sindi_num; i++) {
                int64_t cand_id = sindi_results->GetIds()[i];

                RerankedResult r;
                r.id = cand_id;

                // 精确计算 dense 内积
                r.dense_score = ComputeDenseIP(
                    query_dense_vec,
                    train_dense.data() + cand_id * dense_dim,
                    static_cast<int>(dense_dim)
                );

                // 精确计算 sparse 内积
                r.sparse_score = 1 - sindi_results->GetDistances()[i];

                // 混合分数
                r.ComputeHybridScore(params.alpha);
                candidates.push_back(r);
            }

            /* ---- Step 3: partial_sort 取 top k ---- */
            int actual_k = std::min(params.k, (int)candidates.size());
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + actual_k,
                candidates.end(),
                std::greater<RerankedResult>()
            );

            /* ---- Step 4: 提取 top-k ID ---- */
            std::vector<int64_t> search_results;
            search_results.reserve(actual_k);
            for (int i = 0; i < actual_k; i++) {
                search_results.push_back(candidates[i].id);
            }

            /* ---- Step 5: 计算召回率 ---- */
            int gt_k = std::min(params.k, (int)k_gt);
            std::vector<int64_t> gt(
                ground_truth.begin() + query_idx * k_gt,
                ground_truth.begin() + query_idx * k_gt + gt_k
            );

            float recall = CalculateRecall(search_results, gt);
            total_recall += recall;
        }

        // ---------- 计时结束 ----------
        auto search_end = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(
            search_end - search_start).count();
        double qps = static_cast<double>(actual_num_queries) / elapsed_seconds;
        // ------------------------------

        float avg_recall = total_recall / static_cast<float>(actual_num_queries);

        /******************* 4. 输出结果 *****************/
        std::cout << "\n=== Results (sindi → rerank) ===" << std::endl;
        std::cout << "k:             " << params.k            << std::endl;
        std::cout << "sindi_bk:      " << params.sindi_bk     << std::endl;
        std::cout << "alpha:         " << params.alpha         << std::endl;
        std::cout << "num_queries:   " << actual_num_queries   << std::endl;
        std::cout << "avg_recall:    " << avg_recall           << std::endl;
        std::cout << "qps:           " << qps                  << std::endl;
        std::cout << "total_time(s): " << elapsed_seconds      << std::endl;

        /******************* 5. 清理资源 *****************/
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

