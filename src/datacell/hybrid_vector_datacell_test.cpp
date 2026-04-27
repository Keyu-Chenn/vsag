#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "datacell/flatten_datacell_parameter.h"
#include "datacell/hybrid_vector_datacell.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"

namespace vsag {

namespace {

FlattenInterfaceParamPtr
MakeFlattenParam(const char* quantization_type) {
    auto json = JsonType::Parse(fmt::format(R"({{
        "io_params": {{
            "type": "memory_io"
        }},
        "quantization_params": {{
            "type": "{}"
        }}
    }})",
                                          quantization_type));

    if (std::string(quantization_type) == "sparse") {
        auto param = std::make_shared<SparseVectorDataCellParameter>();
        param->FromJson(json);
        return param;
    }

    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(json);
    return param;
}

float
SparseDistance(const SparseVector& lhs, const SparseVector& rhs) {
    float inner_product = 0.0F;
    uint32_t i = 0;
    uint32_t j = 0;
    while (i < lhs.len_ and j < rhs.len_) {
        if (lhs.ids_[i] < rhs.ids_[j]) {
            ++i;
        } else if (lhs.ids_[i] > rhs.ids_[j]) {
            ++j;
        } else {
            inner_product += lhs.vals_[i] * rhs.vals_[j];
            ++i;
            ++j;
        }
    }
    return 1.0F - inner_product;
}

SparseVector
MakeSparseVector(uint32_t len, uint32_t* ids, float* vals) {
    SparseVector sparse_vector;
    sparse_vector.len_ = len;
    sparse_vector.ids_ = ids;
    sparse_vector.vals_ = vals;
    return sparse_vector;
}

}  // namespace

TEST_CASE("HybridVectorDataCell dense-first sparse pruning", "[ut][HybridVectorDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = 2;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    auto dense_param = MakeFlattenParam("fp32");
    auto sparse_param = MakeFlattenParam("sparse");
    auto data_cell = std::make_shared<HybridVectorDataCell>(dense_param, sparse_param, common_param);
    data_cell->SetHybridWeight(0.5F, 0.5F);
    data_cell->Resize(3);

    std::vector<float> dense_vectors = {
        0.9F, 0.0F,
        0.2F, 0.0F,
        0.1F, 0.0F,
    };

    uint32_t sparse_id0[] = {0};
    float sparse_val0[] = {0.1F};
    uint32_t sparse_id1[] = {0};
    float sparse_val1[] = {0.9F};
    uint32_t sparse_id2[] = {0};
    float sparse_val2[] = {0.1F};
    std::vector<SparseVector> sparse_vectors = {
        MakeSparseVector(1, sparse_id0, sparse_val0),
        MakeSparseVector(1, sparse_id1, sparse_val1),
        MakeSparseVector(1, sparse_id2, sparse_val2),
    };

    const auto dense_size = common_param.dim_ * sizeof(float);
    std::vector<int8_t> hybrid_vectors(dense_vectors.size() * sizeof(float) +
                                       sparse_vectors.size() * sizeof(SparseVector));
    std::memcpy(hybrid_vectors.data(), dense_vectors.data(), dense_vectors.size() * sizeof(float));
    std::memcpy(hybrid_vectors.data() + dense_vectors.size() * sizeof(float),
                sparse_vectors.data(),
                sparse_vectors.size() * sizeof(SparseVector));
    data_cell->BatchInsertVector(hybrid_vectors.data(), sparse_vectors.size());

    std::vector<float> dense_query = {1.0F, 0.0F};
    uint32_t query_sparse_id[] = {0};
    float query_sparse_val[] = {1.0F};
    auto sparse_query = MakeSparseVector(1, query_sparse_id, query_sparse_val);
    std::vector<int8_t> hybrid_query(dense_size + sizeof(SparseVector));
    std::memcpy(hybrid_query.data(), dense_query.data(), dense_size);
    std::memcpy(hybrid_query.data() + dense_size, &sparse_query, sizeof(SparseVector));

    auto computer = data_cell->FactoryComputer(hybrid_query.data());
    InnerIdType ids[] = {0, 1, 2};

    std::vector<float> full_dists(3, 0.0F);
    data_cell->Query(full_dists.data(), computer, ids, 3, allocator.get());

    for (InnerIdType i = 0; i < 3; ++i) {
        const float dense_dist = 1.0F - dense_query[0] * dense_vectors[i * common_param.dim_];
        const float sparse_dist = SparseDistance(sparse_query, sparse_vectors[i]);
        REQUIRE(std::abs(full_dists[i] - 0.5F * (dense_dist + sparse_dist)) < 1e-5F);
    }

    computer->SetSearchLowerBound(0.7F);
    std::vector<float> pruned_dists(3, 0.0F);
    data_cell->Query(pruned_dists.data(), computer, ids, 3, allocator.get());

    REQUIRE(std::abs(pruned_dists[0] - full_dists[0]) < 1e-5F);
    REQUIRE(std::abs(pruned_dists[1] - full_dists[1]) < 1e-5F);
    REQUIRE(pruned_dists[2] > 0.7F);
}

}  // namespace vsag
