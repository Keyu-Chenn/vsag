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

float ComputeDenseIP(const float* vec1, const float* vec2, int dim) {
    float score = 0.0f;
    for (int i = 0; i < dim; i++) {
        score += vec1[i] * vec2[i];
    }
    return score;
}

float ComputeSparseIP(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2) {
    float ip = 0.0f;
    uint32_t i = 0, j = 0;
    while (i < vec1.len_ && j < vec2.len_) {
        if (vec1.ids_[i] == vec2.ids_[j]) {
            ip += vec1.vals_[i] * vec2.vals_[j];
            i++;
            j++;
        } else if (vec1.ids_[i] < vec2.ids_[j]) {
            i++;
        } else {
            j++;
        }
    }
    return ip;
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
              << "  --bk <int>                 一阶段召回数量 (default: 10000)\n"
              << "  --alpha <float>            dense 分数权重，0~1 (default: 0.5)\n"
              << "  --num_queries <int>        测试查询数量，-1 表示全部 (default: -1)\n"
              << "  --exp <int>                实验编号: 1=sindi->dense, 2=hgraph->sparse (default: 1)\n"
              << "  --help, -h                 显示此帮助信息\n"
              << "\nExample:\n"
              << "  " << program_name << " dataset.h5 -k 10 --bk 10000 --alpha 0.5 --exp 1\n"
              << "  " << program_name << " dataset.h5 -k 10 --bk 10000 --alpha 0.5 --exp 2\n"
              << std::endl;
}

struct SearchParams {
    std::string h5_file;
    int k         = 10;
    int bk        = 10000;   // 一阶段召回数
    float alpha   = 0.5f;
    int num_queries = -1;    // -1 表示全部
    int exp       = 1;       // 1=sindi->dense, 2=hgraph->sparse
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
            params.bk = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            params.alpha = std::atof(argv[++i]);
        } else if (arg == "--num_queries" && i + 1 < argc) {
            params.num_queries = std::atoi(argv[++i]);
        } else if (arg == "--exp" && i + 1 < argc) {
            params.exp = std::atoi(argv[++i]);
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
    if (params.bk < params.k) {
        std::cerr << "Error: bk must be >= k\n";
        exit(1);
    }
    if (params.alpha < 0.0f || params.alpha > 1.0f) {
        std::cerr << "Error: alpha must be between 0.0 and 1.0\n";
        exit(1);
    }
    if (params.exp != 1 && params.exp != 2) {
        std::cerr << "Error: exp must be 1 or 2\n";
        exit(1);
    }

    return params;
}

// ============================================================
// 实验1: sindi搜top bk，按dense重排取top k
// ============================================================

void RunExp1(const SearchParams& params,
             const std::vector<float>& train_dense,
             const std::vector<vsag::SparseVector>& train_sparse,
             const std::vector<int64_t>& train_labels,
             const std::vector<float>& test_dense,
             const std::vector<vsag::SparseVector>& test_sparse,
             const std::vector<int64_t>& ground_truth,
             int64_t num_train, int64_t num_test,
             int64_t dense_dim, int64_t k_gt) {

    std::cout << "\n=== Experiment 1: sindi -> dense rerank ===" << std::endl;

    // 构建sindi索引
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
        return;
    }
    std::cout << "Sindi index built." << std::endl;

    int actual_num_queries = (params.num_queries > 0)
        ? std::min(params.num_queries, (int)num_test)
        : (int)num_test;

    float total_recall = 0.0f;
    auto search_start = std::chrono::high_resolution_clock::now();

    for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {
        if ((query_idx + 1) % 100 == 0) {
            std::cout << "Processing query " << query_idx + 1 << "/" << actual_num_queries << std::endl;
        }

        // Step 1: sindi召回 top bk
        auto query_sparse = vsag::Dataset::Make();
        query_sparse->NumElements(1)
            ->SparseVectors(test_sparse.data() + query_idx)
            ->Owner(false);

        auto sindi_search_params = R"({"sindi": {}})";
        auto sindi_results = sparse_index->KnnSearch(
            query_sparse, params.bk, sindi_search_params).value();
        int sindi_num = sindi_results->GetDim();

        // Step 2: 按dense精确重排
        const float* query_dense_vec = test_dense.data() + query_idx * dense_dim;

        std::vector<RerankedResult> candidates;
        candidates.reserve(sindi_num);

        for (int i = 0; i < sindi_num; i++) {
            int64_t cand_id = sindi_results->GetIds()[i];
            RerankedResult r;
            r.id = cand_id;
            r.dense_score = ComputeDenseIP(
                query_dense_vec,
                train_dense.data() + cand_id * dense_dim,
                static_cast<int>(dense_dim)
            );
            // alpha=1时不需要计算sparse_score
            if (std::abs(params.alpha - 1.0f) > 1e-5f) {
                r.sparse_score = 1 - sindi_results->GetDistances()[i];
            } else {
                r.sparse_score = 0.0f;
            }
            r.ComputeHybridScore(params.alpha);
            candidates.push_back(r);
        }

        // Step 3: 取top k
        int actual_k = std::min(params.k, (int)candidates.size());
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + actual_k,
            candidates.end(),
            std::greater<RerankedResult>()
        );

        // Step 4: 计算recall
        std::vector<int64_t> search_results;
        for (int i = 0; i < actual_k; i++) {
            search_results.push_back(candidates[i].id);
        }

        int gt_k = std::min(params.k, (int)k_gt);
        std::vector<int64_t> gt(
            ground_truth.begin() + query_idx * k_gt,
            ground_truth.begin() + query_idx * k_gt + gt_k
        );
        total_recall += CalculateRecall(search_results, gt);
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(search_end - search_start).count();
    double qps = static_cast<double>(actual_num_queries) / elapsed_seconds;
    float avg_recall = total_recall / static_cast<float>(actual_num_queries);

    std::cout << "\n=== Exp1 Results ===" << std::endl;
    std::cout << "k:             " << params.k << std::endl;
    std::cout << "bk:            " << params.bk << std::endl;
    std::cout << "alpha:         " << params.alpha << std::endl;
    std::cout << "num_queries:   " << actual_num_queries << std::endl;
    std::cout << "avg_recall:    " << avg_recall << std::endl;
    std::cout << "qps:           " << qps << std::endl;
    std::cout << "total_time(s): " << elapsed_seconds << std::endl;
}

// ============================================================
// 实验2: hgraph搜top bk，按sparse重排取top k
// ============================================================

void RunExp2(const SearchParams& params,
             const std::vector<float>& train_dense,
             const std::vector<vsag::SparseVector>& train_sparse,
             const std::vector<int64_t>& train_labels,
             const std::vector<float>& test_dense,
             const std::vector<vsag::SparseVector>& test_sparse,
             const std::vector<int64_t>& ground_truth,
             int64_t num_train, int64_t num_test,
             int64_t dense_dim, int64_t k_gt) {

    std::cout << "\n=== Experiment 2: hgraph -> sparse rerank ===" << std::endl;

    // 构建hgraph索引
    std::string hgraph_build_params = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": )" + std::to_string(dense_dim) + R"(,
        "index_param": {
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

    auto dense_index = vsag::Factory::CreateIndex("hgraph", hgraph_build_params).value();
    if (!dense_index->Build(base_dense_dataset).has_value()) {
        std::cerr << "Failed to build hgraph index" << std::endl;
        return;
    }
    std::cout << "Hgraph index built." << std::endl;

    int actual_num_queries = (params.num_queries > 0)
        ? std::min(params.num_queries, (int)num_test)
        : (int)num_test;

    float total_recall = 0.0f;
    auto search_start = std::chrono::high_resolution_clock::now();

    for (int query_idx = 0; query_idx < actual_num_queries; query_idx++) {
        if ((query_idx + 1) % 100 == 0) {
            std::cout << "Processing query " << query_idx + 1 << "/" << actual_num_queries << std::endl;
        }

        // Step 1: hgraph召回 top bk
        const float* query_dense_vec = test_dense.data() + query_idx * dense_dim;
        auto query_dense = vsag::Dataset::Make();
        query_dense->NumElements(1)
            ->Dim(dense_dim)
            ->Float32Vectors(query_dense_vec)
            ->Owner(false);

        // ef_search需要 >= bk才能保证召回足够多的候选
        std::string hgraph_search_params = "{\"hgraph\": {\"ef_search\": " + std::to_string(params.bk) + "}}";
        auto hgraph_results = dense_index->KnnSearch(
            query_dense, params.bk, hgraph_search_params).value();
        int hgraph_num = hgraph_results->GetDim();

        // Step 2: 按sparse精确重排
        const vsag::SparseVector& query_sparse_vec = test_sparse[query_idx];

        std::vector<RerankedResult> candidates;
        candidates.reserve(hgraph_num);

        for (int i = 0; i < hgraph_num; i++) {
            int64_t cand_id = hgraph_results->GetIds()[i];
            RerankedResult r;
            r.id = cand_id;
            // alpha=0时不需要计算dense_score
            if (std::abs(params.alpha - 0.0f) > 1e-5f) {
                r.dense_score = 1 - hgraph_results->GetDistances()[i];
            } else {
                r.dense_score = 0.0f;
            }
            // 精确计算sparse内积
            r.sparse_score = ComputeSparseIP(query_sparse_vec, train_sparse[cand_id]);
            r.ComputeHybridScore(params.alpha);
            candidates.push_back(r);
        }

        // Step 3: 取top k
        int actual_k = std::min(params.k, (int)candidates.size());
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + actual_k,
            candidates.end(),
            std::greater<RerankedResult>()
        );

        // Step 4: 计算recall
        std::vector<int64_t> search_results;
        for (int i = 0; i < actual_k; i++) {
            search_results.push_back(candidates[i].id);
        }

        int gt_k = std::min(params.k, (int)k_gt);
        std::vector<int64_t> gt(
            ground_truth.begin() + query_idx * k_gt,
            ground_truth.begin() + query_idx * k_gt + gt_k
        );
        total_recall += CalculateRecall(search_results, gt);
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(search_end - search_start).count();
    double qps = static_cast<double>(actual_num_queries) / elapsed_seconds;
    float avg_recall = total_recall / static_cast<float>(actual_num_queries);

    std::cout << "\n=== Exp2 Results ===" << std::endl;
    std::cout << "k:             " << params.k << std::endl;
    std::cout << "bk:            " << params.bk << std::endl;
    std::cout << "alpha:         " << params.alpha << std::endl;
    std::cout << "num_queries:   " << actual_num_queries << std::endl;
    std::cout << "avg_recall:    " << avg_recall << std::endl;
    std::cout << "qps:           " << qps << std::endl;
    std::cout << "total_time(s): " << elapsed_seconds << std::endl;
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

        // 训练集 dense
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();
        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims);
        int64_t num_train = train_dims[0];
        int64_t dense_dim  = train_dims[1];

        std::vector<float> train_dense(num_train * dense_dim);
        train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

        // 训练集 sparse
        H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
        hsize_t train_sparse_size = train_sparse_dataset.getSpace().getSimpleExtentNpoints();
        std::vector<uint8_t> train_sparse_blob(train_sparse_size);
        train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto train_sparse = ParseSparseVectors(train_sparse_blob);

        // 训练集 labels
        H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
        std::vector<int64_t> train_labels(num_train);
        train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);

        // 测试集 dense
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();
        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims);
        int64_t num_test = test_dims[0];

        std::vector<float> test_dense(num_test * dense_dim);
        test_dataset.read(test_dense.data(), H5::PredType::NATIVE_FLOAT);

        // 测试集 sparse
        H5::DataSet test_sparse_dataset = file.openDataSet("test_sparse");
        hsize_t test_sparse_size = test_sparse_dataset.getSpace().getSimpleExtentNpoints();
        std::vector<uint8_t> test_sparse_blob(test_sparse_size);
        test_sparse_dataset.read(test_sparse_blob.data(), H5::PredType::NATIVE_UINT8);
        auto test_sparse = ParseSparseVectors(test_sparse_blob);

        // ground truth
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

        /******************* 2. 运行实验 *****************/
        if (params.exp == 1) {
            params.alpha = 1.0f;  // sindi -> dense: 只用dense重排
            RunExp1(params, train_dense, train_sparse, train_labels,
                    test_dense, test_sparse, ground_truth,
                    num_train, num_test, dense_dim, k_gt);
        } else if (params.exp == 2) {
            params.alpha = 0.0f;  // hgraph -> sparse: 只用sparse重排
            RunExp2(params, train_dense, train_sparse, train_labels,
                    test_dense, test_sparse, ground_truth,
                    num_train, num_test, dense_dim, k_gt);
        }

        /******************* 3. 清理资源 *****************/
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