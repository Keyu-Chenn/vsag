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



        // dense & sparse 各搜top ak个，合并，采样alpha，在每个alpha对集合进行重排，取top k，把这些top k集合合并为最终的邻居all neighbors
        // make sure ak >> k
        int k = 32;
        int ak = 10 * k;


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










        // Get all neighbors
        float avg_total_neighbor_num = 0;
        int tt_num = 80000;

        // cover_rate[bk]：记从dense和sparse两路取top bk得到的并集为A，all_neighbors为B，
        // cover_rate = |A and B| / |B|
        int bk_num = 10 * k;
        std::vector<float> cover_rate(bk_num);



        // 处理每个点
        for (int i = 0; i <tt_num; i++) {
            auto q_train = vsag::Dataset::Make();
            q_train->NumElements(1)
                ->Dim(dense_dim)
                ->Ids(train_labels.data() + i)
                ->Float32Vectors(train_dense.data() + i * dense_dim)
                ->SparseVectors(train_sparse.data() + i)
                ->Owner(false);

            /******************* KnnSearch For HGraph Index *****************/
            auto hgraph_search_parameters = R"(
            {
                "hgraph": {
                    "ef_search": 100
                }
            }
            )";
            auto result_dense = index_hgraph->KnnSearch(q_train, ak, hgraph_search_parameters).value();


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

            auto result_sparse = index_sindi->KnnSearch(q_train, ak, sindi_search_params).value();


            // std::cout << "dense_res_from_hgraph: " << std::endl;
            // for (int z = 0; z < bk_num; z++) {
            //     std::cout << "id: " << result_dense->GetIds()[z] << std::endl;
            //     std::cout << "dense_dis: " << 1 - result_dense->GetDistances()[z] << std::endl;
            // }
            //
            // std::cout << "sparse_res_from_sindi: " << std::endl;
            // for (int z = 0; z < bk_num; z++) {
            //     std::cout << "id: " << result_sparse->GetIds()[z] << std::endl;
            //     std::cout << "sparse_dis: " << 1- result_sparse->GetDistances()[z] << std::endl;
            // }



            // Get dense_sparse_union 合并dense和sparse res的id，距离自己重算
            std::vector<union_search_res> ds_u_res;
            std::unordered_set<int> seen_ids;
            std::unordered_set<int> all_neighbors;

            for (int64_t j = 0; j < result_dense->GetDim(); ++j) {
                auto cur_id = result_dense->GetIds()[j];
                if (seen_ids.insert(cur_id).second) {
                    auto cur_dense_dist = CalDenseIp(train_dense.data() + i * dense_dim, train_dense.data() + cur_id * dense_dim, dense_dim);
                    auto cur_sparse_dist = CalSparseIp(train_sparse[i], train_sparse[cur_id]);
                    ds_u_res.push_back({cur_id, cur_dense_dist, cur_sparse_dist, 0});
                }
            }

            for (int64_t j = 0; j < result_sparse->GetDim(); ++j) {
                auto cur_id = result_sparse->GetIds()[j];
                if (seen_ids.insert(cur_id).second) {
                    auto cur_dense_dist = CalDenseIp(train_dense.data() + i * dense_dim, train_dense.data() + cur_id * dense_dim, dense_dim);
                    auto cur_sparse_dist = CalSparseIp(train_sparse[i], train_sparse[cur_id]);
                    ds_u_res.push_back({cur_id, cur_dense_dist, cur_sparse_dist, 0});
                }
            }






            // Get all neighbors
            for (float x = 0; x <= 1 + 1e-5; x += 0.01) {
                for (auto& item : ds_u_res) {
                    item.GetHybridDis(x);
                }
                std::sort(ds_u_res.begin(), ds_u_res.end(), std::greater<union_search_res>());
                for (int j = 0; j < k; j++) {
                    all_neighbors.insert(ds_u_res[j].id);
                }
            }



            // 输出每个点的all_neighbors

            std::cout << "id: " << i << "    total_neighbor_num: " << all_neighbors.size() << std::endl;
            avg_total_neighbor_num += all_neighbors.size();
            // std::vector<int> alls;
            // for (auto item : all_neighbors) {
            //     alls.push_back(item);
            // }
            // std::sort(alls.begin(), alls.end());
            // std::cout << "{ ";
            // for (auto item : alls) {
            //     std::cout << item << " ";
            // }
            // std::cout << "}" << std::endl;



            // 计算不同bk对应的覆盖率
            std::unordered_set<int> dsu;
            for (int bk = 0; bk < bk_num; bk ++) {
                int denseid = result_dense->GetIds()[bk];
                int sparseid = result_sparse->GetIds()[bk];
                for (auto item : all_neighbors) {
                    if (denseid == item) {
                        dsu.insert(denseid);
                    }
                    if (sparseid == item) {
                        dsu.insert(sparseid);
                    }
                }
                cover_rate[bk] += (float)dsu.size() / all_neighbors.size();
            }

        }

        std::cout << "avg_num: " << avg_total_neighbor_num / tt_num << std::endl;



        std::ofstream fout("601_results_k32.txt");
        fout << "avg_num: " << avg_total_neighbor_num / tt_num << std::endl;
        fout << "cover_rate: " << std::endl;
        for (int bk = 0; bk < bk_num; bk ++) {
            fout << "bk" << bk + 1 << ": " << cover_rate[bk] / tt_num << std::endl;
        }
        fout.close();



        // std::cout << "cover_rate: " << std::endl;
        // for (int bk = 0; bk < bk_num; bk ++) {
        //    std::cout << "bk" << bk + 1 << ": " << cover_rate[bk] / tt_num << std::endl;
        // }











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