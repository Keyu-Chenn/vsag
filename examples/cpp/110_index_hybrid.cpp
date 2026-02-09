
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

#include <vsag/vsag.h>
#include <unordered_set>
#include <iostream>

std::vector<vsag::SparseVector>
GenerateSparseVectors(
    uint32_t count, uint32_t max_dim, uint32_t max_id, float min_val, float max_val, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib_real(min_val, max_val);
    std::uniform_int_distribution<int> distrib_dim(max_dim / 2, max_dim);
    std::uniform_int_distribution<int> distrib_id(0, max_id);

    std::vector<vsag::SparseVector> sparse_vectors(count);
    if (max_dim > max_id) {
        throw std::runtime_error("generate sparse vectors failed, max_dim > max_id");
    }

    for (int i = 0; i < count; i++) {
        sparse_vectors[i].len_ = distrib_dim(rng);
        sparse_vectors[i].ids_ = new uint32_t[sparse_vectors[i].len_];
        sparse_vectors[i].vals_ = new float[sparse_vectors[i].len_];
        std::unordered_set<uint32_t> unique_ids;
        for (int d = 0; d < sparse_vectors[i].len_; d++) {
            auto u_id = distrib_id(rng);
            while (unique_ids.count(u_id) > 0) {
                u_id = distrib_id(rng);
            }
            unique_ids.insert(u_id);
            sparse_vectors[i].ids_[d] = u_id;
            sparse_vectors[i].vals_[d] = distrib_real(rng);
        }

        std::sort(sparse_vectors[i].ids_, sparse_vectors[i].ids_ + sparse_vectors[i].len_);
    }

    return sparse_vectors;
}


int
main(int argc, char** argv) {

    vsag::init();

    /******************* Prepare Dense_base Dataset *****************/
    int64_t num_vectors = 5000;
    int64_t dense_dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> dense_datas(num_vectors * dense_dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dense_dim * num_vectors; ++i) {
        dense_datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dense_dim)
        ->Ids(ids.data())
        ->Float32Vectors(dense_datas.data())
        ->Owner(false);

    /******************* Prepare Sparse_base Dataset *****************/
    int64_t max_dim = 128;
    int64_t max_id = 30000;
    float min_val = 0;
    float max_val = 10;
    int seed_base = 114;

    auto sparse_base_vectors = GenerateSparseVectors(num_vectors, max_dim, max_id, min_val, max_val, seed_base);
    base->SparseVectors(sparse_base_vectors.data());



    /******************* Create Hybrid Index *****************/
    std::string hybrid_index_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "ip",
        "dim": 128,
        "index_param": {
            "sparse_dtype": "float32",
            "saprse_metric_type": "ip",
            "sparse_dim": 10000,
            "alpha": 0.5
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hybrid_index", hybrid_index_build_parameters).value();
    std::cout << "hybrid index built" << std::endl;

    /******************* Build Hybrid Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index Hybrid contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }


    /******************* Prepare Query Dataset *****************/
    int64_t num_query = 1;
    int64_t seed_query = 514;
    std::vector<float> query_vector(num_query * dense_dim);
    for (int64_t i = 0; i < num_query * dense_dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(num_query)->Dim(dense_dim)->Float32Vectors(query_vector.data())->Owner(false);

    auto sparse_query_vectors = GenerateSparseVectors(num_query, max_dim / 2, max_id, min_val, max_val, seed_query);
    query->SparseVectors(sparse_query_vectors.data());


    //test_look
    // auto tt = query->GetSparseVectors();
    // auto num = tt[0].len_;
    // for (int i = 0; i < num; ++i) {
    //     std::cout << i << ": " << tt[0].ids_[i] << "  " << tt[0].vals_[i] << std::endl;
    // }







    /******************* KnnSearch For Hybrid Index *****************/
    auto hybrid_index_search_parameters = R"(
    {
        "alpha": 0.5
    })";
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, hybrid_index_search_parameters).value();

    /******************* Print Search Result *****************/
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
    return 0;
}
