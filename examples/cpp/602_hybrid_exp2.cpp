//
// Created by root on 2026/1/26.
//
// test difference between two-route top k and true hybrid top beta*k

#include <H5Cpp.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
#include<nlohmann/json.hpp>

#include "diskann_logger.h"

struct union_search_res {
    int64_t id;
    float dense_distance;
    float sparse_distance;
    float hybrid_distance;

    void GetHybridDis(float alpha) {
        hybrid_distance = alpha * dense_distance + (1 - alpha) * sparse_distance;
    }

    bool operator<(const union_search_res& other) const {
        return hybrid_distance < other.hybrid_distance;
    }

    bool operator>(const union_search_res& other) const {
        return hybrid_distance > other.hybrid_distance;
    }

};

std::vector<vsag::SparseVector>
ParseSparseVectors(const std::vector<uint8_t>& blob) {
    std::vector<vsag::SparseVector> result;
    size_t offset = 0;

    while (offset < blob.size()) {
        // Read length
        if (offset + sizeof(uint32_t) > blob.size()) {
            break;
        }

        uint32_t len;
        std::memcpy(&len, &blob[offset], sizeof(uint32_t));
        offset += sizeof(uint32_t);

        vsag::SparseVector vec;
        vec.len_ = len;

        if (len > 0) {
            // Read IDs
            size_t ids_bytes = len * sizeof(uint32_t);
            if (offset + ids_bytes > blob.size()) {
                break;
            }

            vec.ids_ = new uint32_t[len];
            std::memcpy(vec.ids_, &blob[offset], ids_bytes);
            offset += ids_bytes;

            // Read Values
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
        if (vec.ids_) {
            delete[] vec.ids_;
            vec.ids_ = nullptr;
        }
        if (vec.vals_) {
            delete[] vec.vals_;
            vec.vals_ = nullptr;
        }
    }
}

float
CalDenseIp(const float* vec1, const float* vec2, int dim) {
    float dis = 0;
    for (int i = 0; i < dim; i++) {
        dis += vec1[i] * vec2[i];
    }
    return dis;
}


float
CalSparseIp(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2) {
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


int
main(int argc, char** argv) {
    vsag::init();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <h5_file_path>" << std::endl;
        return 1;
    }

    std::string h5_file = argv[1];

    try {
        /******************* Load Dataset from HDF5 *****************/
        H5::H5File file(h5_file, H5F_ACC_RDONLY);

        // Read train dense vectors
        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();
        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims);
        int64_t num_train = train_dims[0];
        int64_t dense_dim = train_dims[1];

        std::vector<float> train_dense(num_train * dense_dim);
        train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);

        std::cout << "Loaded train dense: " << num_train << " x " << dense_dim << std::endl;

        // Read train sparse vectors
        H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
        H5::DataSpace train_sparse_dataspace = train_sparse_dataset.getSpace();
        hsize_t train_sparse_size = train_sparse_dataspace.getSimpleExtentNpoints();

        std::vector<uint8_t> train_sparse_blob(train_sparse_size);
        train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);

        auto train_sparse = ParseSparseVectors(train_sparse_blob);
        std::cout << "Loaded train sparse: " << train_sparse.size() << " vectors" << std::endl;

        // Read train labels
        H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
        H5::DataSpace train_labels_dataspace = train_labels_dataset.getSpace();
        hsize_t num_train_labels = train_labels_dataspace.getSimpleExtentNpoints();

        std::vector<int64_t> train_labels(num_train_labels);
        train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);

        std::cout << "Loaded train labels: " << num_train_labels << std::endl;

        // Read test dense vectors
        H5::DataSet test_dataset = file.openDataSet("test");
        H5::DataSpace test_dataspace = test_dataset.getSpace();
        hsize_t test_dims[2];
        test_dataspace.getSimpleExtentDims(test_dims);
        int64_t num_test = test_dims[0];

        std::vector<float> test_dense(num_test * dense_dim);
        test_dataset.read(test_dense.data(), H5::PredType::NATIVE_FLOAT);

        std::cout << "Loaded test dense: " << num_test << " x " << dense_dim << std::endl;

        // Read test sparse vectors
        H5::DataSet test_sparse_dataset = file.openDataSet("test_sparse");
        H5::DataSpace test_sparse_dataspace = test_sparse_dataset.getSpace();
        hsize_t test_sparse_size = test_sparse_dataspace.getSimpleExtentNpoints();

        std::vector<uint8_t> test_sparse_blob(test_sparse_size);
        test_sparse_dataset.read(test_sparse_blob.data(), H5::PredType::NATIVE_UINT8);

        auto test_sparse = ParseSparseVectors(test_sparse_blob);
        std::cout << "Loaded test sparse: " << test_sparse.size() << " vectors" << std::endl;

        /******************* Read Ground Truth *****************/
        H5::DataSet gt_dataset = file.openDataSet("neighbors");
        H5::DataSpace gt_dataspace = gt_dataset.getSpace();
        hsize_t gt_dims[2];
        gt_dataspace.getSimpleExtentDims(gt_dims);
        int64_t num_queries = gt_dims[0];
        int64_t k_gt = gt_dims[1];

        std::vector<int64_t> ground_truth(num_queries * k_gt);
        gt_dataset.read(ground_truth.data(), H5::PredType::NATIVE_INT64);

        std::cout << "Loaded ground truth: " << num_queries << " x " << k_gt << std::endl;

        /******************* Prepare Base Dataset *****************/
        auto base = vsag::Dataset::Make();
        base->NumElements(num_train)
            ->Dim(dense_dim)
            ->Ids(train_labels.data())
            ->Float32Vectors(train_dense.data())
            ->SparseVectors(train_sparse.data())
            ->Owner(false);


        /******************* Prepare Query Dataset *****************/
        auto query = vsag::Dataset::Make();
        query->NumElements(num_test)
             ->Dim(dense_dim)
             ->Float32Vectors(test_dense.data())
             ->SparseVectors(test_sparse.data())
             ->Owner(false);


        //exp: sindi结果作为hgraph入口点

        //params
        int k = 32;
        int tt_num = 1000;

        // Build Dense Index: hgraph
        std::string hgraph_build_parameters = R"(
        {
            "dtype": "float32",
            "metric_type": "ip",
            "dim": 1024,
            "index_param": {
                "base_quantization_type": "sq8",
                "max_degree": 26,
                "ef_construction": 100,
                "alpha":1.2
            }
        }
        )";
        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);
        auto index_hgraph = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

        if (auto build_result_hgraph = index_hgraph->Build(base); build_result_hgraph.has_value()) {
            std::cout << "After Build(), Index HGraph contains: " << index_hgraph->GetNumElements()
                      << std::endl;
        } else if (build_result_hgraph.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "Failed to build index: internalError" << std::endl;
            exit(-1);
        }

        // Build Sparse Index: sindi
        auto sindi_build_params = R"(
        {
            "dtype": "sparse",
            "dim": 128,
            "metric_type": "ip",
            "index_param": {
                "use_reorder": false,
                "term_id_limit": 1000000,
                "doc_prune_ratio": 0.0,
                "window_size": 100000,
                "use_quantization": false
            }
        }
        )";

        auto index_sindi = vsag::Factory::CreateIndex("sindi", sindi_build_params).value();

        if (auto build_result_sindi = index_sindi->Build(base); build_result_sindi.has_value()) {
            std::cout << "After Build(), Sparse Term Index contains: " << index_sindi->GetNumElements()
                      << std::endl;
        } else if (build_result_sindi.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "Failed to build index: internalError" << std::endl;
            exit(-1);
        }

        // read ground truth


        // 处理每个点
        float avg_dist_compute = 0;
        float avg_hops = 0;
        float avg_recall = 0;


        for (int i = 0; i <tt_num; i++) {
            auto q_test = vsag::Dataset::Make();
            q_test->NumElements(1)
                ->Dim(dense_dim)
                ->Float32Vectors(test_dense.data() + i * dense_dim)
                ->SparseVectors(test_sparse.data() + i)
                ->Owner(false);



            /******************* KnnSearch For Sindi Index *****************/
            auto sindi_search_params = R"(
            {
                "sindi": {
                    "query_prune_ratio": 0,
                    "term_prune_ratio": 0,
                    "n_candidate": 0
                }
            }
            )";

            auto result_sparse = index_sindi->KnnSearch(q_test, k, sindi_search_params).value();

            // check sparse result
            for (int j = 0; j < k; j++) {
                auto cur_id = result_sparse->GetIds()[j];
                auto cur_dist = 1 - result_sparse->GetDistances()[j];
                std::cout << "top" << j + 1 << ":" << std::endl;
                std::cout << "id: " << cur_id << std::endl;
                std::cout << "dist: " << cur_dist << std::endl;
                std::cout << std::endl;
            }

            auto sparse_top_id = result_sparse->GetIds()[0];

            /******************* KnnSearch For HGraph Index *****************/
            auto hgraph_search_parameters = R"(
            {
                "hgraph": {
                    "ef_search": 100,
                    "entry_point": )" + std::to_string(sparse_top_id) + R"(
                }
            }
            )";

            // auto hgraph_search_parameters = R"(
            // {
            //     "hgraph": {
            //         "ef_search": 100,
            //         "entry_point": 6325
            //     }
            // }
            // )";

            // std::cout << hgraph_search_parameters << std::endl;
            auto result_dense = index_hgraph->KnnSearch(q_test, k, hgraph_search_parameters).value();
            // std::cout << result_dense->GetStatistics() << std::endl;


            // check dense result
            for (int j = 0; j < k; j++) {
                std::cout << "top" << j + 1 << ":" << std::endl;
                std::cout << "id:";
                std::cout << result_dense->GetIds()[j] << std::endl;
                std::cout << "dist:";
                std::cout << 1 - result_dense->GetDistances()[j] << std::endl;
                std::cout << std::endl;
            }
            auto stats = result_dense->GetStatistics();
            // std::cout << stats << std::endl;
            auto stats_js = nlohmann::json::parse(stats);
            int cur_dist_comp = stats_js["dist_cmp"];
            int cur_hops = stats_js["hops"];
            avg_dist_compute += cur_dist_comp;
            avg_hops += cur_hops;

            // std::cout << hgraph_search_parameters << std::endl;
            // compute recall
            std::vector<int64_t> search_results(result_dense->GetIds(),
                                                    result_dense->GetIds() + k);
            std::vector<int64_t> gt(ground_truth.begin() + i * k_gt,
                                   ground_truth.begin() + i * k_gt + k);

            float recall = CalculateRecall(search_results, gt);
            avg_recall += recall;
        }
        avg_dist_compute /= tt_num;
        avg_hops /= tt_num;
        avg_recall /= tt_num;

        std::cout << "k: " << k << std::endl;
        std::cout << "avg dist compute: " << avg_dist_compute << std::endl;
        std::cout << "avg hops: " << avg_hops << std::endl;
        std::cout << "avg recall: " << avg_recall << std::endl;


        /******************* Cleanup *****************/
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