#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <vector>

#include "datacell/flatten_datacell_parameter.h"
#include "datacell/hybrid_vector_datacell.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

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
    constexpr std::size_t kDim = 2;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = kDim;
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

    const auto dense_size = kDim * sizeof(float);
    std::vector<int8_t> hybrid_vectors(dense_vectors.size() * sizeof(float) +
                                       sparse_vectors.size() * sizeof(SparseVector));
    std::memcpy(hybrid_vectors.data(), dense_vectors.data(), dense_vectors.size() * sizeof(float));
    std::memcpy(hybrid_vectors.data() + dense_vectors.size() * sizeof(float),
                sparse_vectors.data(),
                sparse_vectors.size() * sizeof(SparseVector));
    data_cell->BatchInsertVector(hybrid_vectors.data(), sparse_vectors.size());

    bool combined_need_release = false;
    const auto* combined_codes = data_cell->GetCodesById(1, combined_need_release);
    REQUIRE(combined_codes != nullptr);
    REQUIRE(combined_need_release);
    uint32_t stored_sparse_len = 0;
    std::memcpy(&stored_sparse_len, combined_codes + dense_size, sizeof(stored_sparse_len));
    REQUIRE(stored_sparse_len == sparse_vectors[1].len_);
    data_cell->Release(combined_codes);

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

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    IOStreamWriter writer(stream);
    data_cell->Serialize(writer);
    auto serialized = stream.str();

    constexpr size_t sparse_norm_count_offset = sizeof(float) * 2;
    uint64_t sparse_norm_count = 0;
    std::memcpy(&sparse_norm_count,
                serialized.data() + sparse_norm_count_offset,
                sizeof(sparse_norm_count));
    REQUIRE(sparse_norm_count == sparse_vectors.size());

    std::istringstream input(serialized, std::ios::binary);
    IOStreamReader reader(input);
    auto restored =
        std::make_shared<HybridVectorDataCell>(dense_param, sparse_param, common_param);
    restored->Deserialize(reader);

    auto restored_computer = restored->FactoryComputer(hybrid_query.data());
    restored_computer->SetSearchLowerBound(0.7F);
    std::vector<float> restored_dists(3, 0.0F);
    restored->Query(restored_dists.data(), restored_computer, ids, 3, allocator.get());
    for (InnerIdType i = 0; i < 3; ++i) {
        REQUIRE(std::abs(restored_dists[i] - pruned_dists[i]) < 1e-5F);
    }
}

TEST_CASE("Hybrid dense reuse rejects stale IDs and different kernels",
          "[ut][HybridVectorDataCell]") {
    HybridDenseDistanceCache cache;
    cache.Reset();
    InnerIdType first[] = {0, 1, 2, 3};
    float scores[] = {0.25F, 0.5F, 0.75F, 1.0F};
    cache.Record(first, scores, 4);
    float result = 0;
    REQUIRE(cache.Find(0, true, result));
    REQUIRE(result == scores[0]);
    REQUIRE_FALSE(cache.Find(0, false, result));

    // Exercise IDs that collide in the bounded cache. Eviction is allowed, but
    // any retained hit must still belong to the requested ID.
    InnerIdType collision[] = {65537, 65536, 65539, 65538};
    float collision_scores[] = {10.0F, 11.0F, 12.0F, 13.0F};
    cache.Record(collision, collision_scores, 4);
    if (cache.Find(0, true, result)) {
        REQUIRE(result == scores[0]);
    }
    REQUIRE(cache.Find(collision[0], true, result));
    REQUIRE(result == collision_scores[0]);
    cache.Reset();
    REQUIRE_FALSE(cache.Find(collision[0], true, result));
}

TEST_CASE("Hybrid dense reuse preserves FP32 results across query and batch boundaries",
          "[ut][HybridVectorDataCell]") {
    constexpr uint64_t dim = 769;
    constexpr InnerIdType count = 29;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = dim;
    common.metric_ = MetricType::METRIC_TYPE_IP;
    auto cell = std::make_shared<HybridVectorDataCell>(
        MakeFlattenParam("fp32"), MakeFlattenParam("sparse"), common);
    cell->Resize(count);
    std::vector<float> dense(count * dim);
    for (size_t i = 0; i < dense.size(); ++i) {
        dense[i] = static_cast<float>(static_cast<int>((i * 97 + 11) % 1021) - 510) / 5100;
    }
    uint32_t sparse_ids[] = {3, 7};
    float sparse_vals[] = {0.2F, 0.3F};
    std::vector<SparseVector> sparse(count, MakeSparseVector(2, sparse_ids, sparse_vals));
    std::vector<int8_t> data(dense.size() * sizeof(float) + sparse.size() * sizeof(SparseVector));
    std::memcpy(data.data(), dense.data(), dense.size() * sizeof(float));
    std::memcpy(data.data() + dense.size() * sizeof(float),
                sparse.data(),
                sparse.size() * sizeof(SparseVector));
    cell->BatchInsertVector(data.data(), count);

    HybridDenseDistanceCache cache;
    std::vector<InnerIdType> ids(count);
    for (InnerIdType i = 0; i < count; ++i) ids[i] = (i * 7) % count;
    for (int query_number = 0; query_number < 3; ++query_number) {
        std::vector<float> query(dim);
        for (size_t i = 0; i < dim; ++i) {
            query[i] =
                static_cast<float>(static_cast<int>((i * 53 + query_number * 193) % 997) - 498) /
                5000;
        }
        auto sparse_query = MakeSparseVector(2, sparse_ids, sparse_vals);
        std::vector<int8_t> hybrid_query(dim * sizeof(float) + sizeof(SparseVector));
        std::memcpy(hybrid_query.data(), query.data(), dim * sizeof(float));
        std::memcpy(hybrid_query.data() + dim * sizeof(float), &sparse_query, sizeof(SparseVector));
        auto control = cell->FactoryComputer(hybrid_query.data());
        auto candidate =
            std::static_pointer_cast<HybridComputer>(cell->FactoryComputer(hybrid_query.data()));
        candidate->SetDenseDistanceCache(&cache);
        auto guide = cell->GetDenseCell()->FactoryComputer(query.data());

        for (int pattern = 0; pattern < 3; ++pattern) {
            cache.Reset();
            if (pattern != 0) {
                // Some hits, some misses, and hits from the wrong reduction kernel.
                std::vector<InnerIdType> guide_ids =
                    pattern == 1 ? std::vector<InnerIdType>{ids[0], ids[3], ids[5], ids[9], ids[1]}
                                 : ids;
                std::vector<float> scores(guide_ids.size());
                cell->GetDenseCell()->Query(
                    scores.data(), guide, guide_ids.data(), guide_ids.size(), allocator.get());
                cache.Record(guide_ids.data(), scores.data(), guide_ids.size());
            }
            for (const float weight : {0.0F, 0.2F, 0.8F, 1.0F}) {
                control->SetHybridWeight(weight, 1 - weight);
                candidate->SetHybridWeight(weight, 1 - weight);
                for (const float bound : {std::numeric_limits<float>::max(), 0.8F}) {
                    control->SetSearchLowerBound(bound);
                    candidate->SetSearchLowerBound(bound);
                    for (InnerIdType size = 1; size <= count; ++size) {
                        std::vector<float> expected(size), actual(size);
                        cell->Query(expected.data(), control, ids.data(), size, allocator.get());
                        cell->Query(actual.data(), candidate, ids.data(), size, allocator.get());
                        INFO("query=" << query_number << " pattern=" << pattern << " size=" << size
                                      << " weight=" << weight << " bound=" << bound);
                        REQUIRE(std::memcmp(expected.data(), actual.data(), size * sizeof(float)) ==
                                0);
                    }
                }
            }
        }
    }
}

}  // namespace vsag
