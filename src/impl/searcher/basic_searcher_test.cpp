
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

#include "basic_searcher.h"

#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "searcher_test.h"
#include "utils/visited_list.h"

using namespace vsag;

namespace {

class CountingComputer : public ComputerInterface {
public:
    void
    SetSearchLowerBound(float lower_bound) override {
        search_lower_bound_ = lower_bound;
    }

    [[nodiscard]] float
    GetSearchLowerBound() const override {
        return search_lower_bound_;
    }

private:
    float search_lower_bound_{std::numeric_limits<float>::max()};
};

class FixedDistanceFlatten : public FlattenInterface {
public:
    FixedDistanceFlatten(std::vector<float> distances, std::vector<uint32_t>* query_counts)
        : distances_(std::move(distances)), query_counts_(query_counts) {
        this->total_count_ = static_cast<InnerIdType>(distances_.size());
        this->max_capacity_ = static_cast<InnerIdType>(distances_.size());
        this->code_size_ = sizeof(float);
    }

    void
    Query(float* result_dists,
          const ComputerInterfacePtr&,
          const InnerIdType* idx,
          InnerIdType id_count,
          Allocator*) override {
        query_counts_->push_back(static_cast<uint32_t>(id_count));
        for (InnerIdType i = 0; i < id_count; ++i) {
            result_dists[i] = distances_.at(idx[i]);
        }
    }

    ComputerInterfacePtr
    FactoryComputer(const void*) override {
        last_computer_ = std::make_shared<CountingComputer>();
        return last_computer_;
    }

    [[nodiscard]] float
    GetLastSearchLowerBound() const {
        return last_computer_ == nullptr ? std::numeric_limits<float>::max()
                                         : last_computer_->GetSearchLowerBound();
    }

    void
    Train(const void*, uint64_t) override {
    }

    void
    InsertVector(const void*, InnerIdType) override {
    }

    void
    BatchInsertVector(const void*, InnerIdType, InnerIdType*) override {
    }

    float
    ComputePairVectors(InnerIdType, InnerIdType) override {
        return 0.0F;
    }

    void
    Prefetch(InnerIdType) override {
    }

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return "fixed_distance";
    }

    [[nodiscard]] MetricType
    GetMetricType() override {
        return MetricType::METRIC_TYPE_L2SQR;
    }

    void
    Resize(InnerIdType) override {
    }

    void
    ExportModel(const FlattenInterfacePtr&) const override {
    }

    bool
    Decode(const uint8_t*, DataType*) override {
        return false;
    }

    [[nodiscard]] const uint8_t*
    GetCodesById(InnerIdType, bool& need_release) const override {
        need_release = false;
        return nullptr;
    }

    void
    Release(const uint8_t*) const override {
    }

    bool
    GetCodesById(InnerIdType id, uint8_t* codes) const override {
        auto value = distances_.at(id);
        std::memcpy(codes, &value, sizeof(value));
        return true;
    }

private:
    std::shared_ptr<CountingComputer> last_computer_{nullptr};
    std::vector<float> distances_;
    std::vector<uint32_t>* query_counts_{nullptr};
};

class StaticGraph : public GraphInterface {
public:
    explicit StaticGraph(std::vector<std::vector<InnerIdType>> neighbors)
        : neighbors_(std::move(neighbors)) {
        this->total_count_ = static_cast<InnerIdType>(neighbors_.size());
        this->max_capacity_ = static_cast<InnerIdType>(neighbors_.size());
        for (const auto& ids : neighbors_) {
            this->maximum_degree_ = std::max<uint32_t>(this->maximum_degree_, ids.size());
        }
    }

    void
    InsertNeighborsById(InnerIdType, const Vector<InnerIdType>&) override {
    }

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override {
        ++copy_calls_;
        neighbor_ids.clear();
        for (auto neighbor : neighbors_.at(id)) {
            neighbor_ids.push_back(neighbor);
        }
    }

    bool
    TryGetNeighborsView(InnerIdType id,
                        const InnerIdType*& neighbor_ids,
                        uint32_t& neighbor_count) const override {
        ++view_calls_;
        const auto& stored_neighbors = neighbors_.at(id);
        neighbor_ids = stored_neighbors.data();
        neighbor_count = static_cast<uint32_t>(stored_neighbors.size());
        return true;
    }

    uint32_t
    GetNeighborSize(InnerIdType id) const override {
        return static_cast<uint32_t>(neighbors_.at(id).size());
    }

    void
    Resize(InnerIdType) override {
    }

    void
    Prefetch(InnerIdType, uint32_t) override {
    }

    [[nodiscard]] uint32_t
    CopyCalls() const {
        return copy_calls_;
    }

    [[nodiscard]] uint32_t
    ViewCalls() const {
        return view_calls_;
    }

private:
    std::vector<std::vector<InnerIdType>> neighbors_;
    mutable uint32_t copy_calls_{0};
    mutable uint32_t view_calls_{0};
};

class RejectIdsFilter : public Filter {
public:
    explicit RejectIdsFilter(std::vector<InnerIdType> rejected_ids)
        : rejected_ids_(std::move(rejected_ids)) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return std::find(rejected_ids_.begin(), rejected_ids_.end(), id) == rejected_ids_.end();
    }

private:
    std::vector<InnerIdType> rejected_ids_;
};

}  // namespace

TEST_CASE("Basic Usage for GraphDataCell (adapter of hnsw)", "[ut][GraphDataCell]") {
    uint32_t M = 32;
    uint32_t data_size = 1000;
    uint32_t ef_construction = 100;
    uint64_t default_max_element = 1;
    uint64_t dim = 960;
    auto vectors = fixtures::generate_vectors(data_size, dim);
    std::vector<int64_t> ids(data_size);
    std::iota(ids.begin(), ids.end(), 0);

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   default_max_element,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();
    for (int64_t i = 0; i < data_size; ++i) {
        auto successful_insert =
            alg_hnsw->addPoint((const void*)(vectors.data() + i * dim), ids[i]);
        REQUIRE(successful_insert == true);
    }

    GraphInterfacePtr graph = std::make_shared<AdaptGraphDataCell>(alg_hnsw);

    for (uint32_t i = 0; i < data_size; i++) {
        auto neighbor_size = graph->GetNeighborSize(i);
        Vector<InnerIdType> neighbor_ids(neighbor_size, allocator.get());
        graph->GetNeighbors(i, neighbor_ids);

        int* data = (int*)alg_hnsw->get_linklist0(i);
        REQUIRE(neighbor_size == alg_hnsw->getListCount((hnswlib::linklistsizeint*)data));

        for (uint32_t j = 0; j < neighbor_size; j++) {
            REQUIRE(neighbor_ids[j] == *(data + j + 1));
        }
    }
}

TEST_CASE("Search with HNSW", "[ut][BasicSearcher]") {
    // data attr
    uint32_t base_size = 1000;
    uint32_t query_size = 100;
    uint64_t dim = 128;

    // build and search attr
    uint32_t M = 16;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t default_max_element = 1;

    // data preparation
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    // hnswlib build
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   default_max_element,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();
    for (int64_t i = 0; i < base_size; ++i) {
        auto successful_insert =
            alg_hnsw->addPoint((const void*)(base_vectors.data() + i * dim), ids[i]);
        REQUIRE(successful_insert == true);
    }

    // graph data cell
    auto graph_data_cell = std::make_shared<AdaptGraphDataCell>(alg_hnsw);

    // vector data cell
    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto fp32_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_temp, "fp32")));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(param_temp, "memory_io")));
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;

    auto vector_data_cell = std::make_shared<
        FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
        fp32_param, io_param, common);
    vector_data_cell->SetQuantizer(
        std::make_shared<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim, allocator.get()));
    vector_data_cell->SetIO(std::make_unique<MemoryIO>(allocator.get()));

    vector_data_cell->Train(base_vectors.data(), base_size);
    vector_data_cell->BatchInsertVector(base_vectors.data(), base_size, ids.data());

    auto init_size = 10;
    auto pool = std::make_shared<VisitedListPool>(
        init_size, allocator.get(), vector_data_cell->TotalCount(), allocator.get());

    auto exception_func = [&](const InnerSearchParam& search_param) -> void {
        // init searcher
        auto searcher = std::make_shared<BasicSearcher>(common);
        {
            // search with empty graph_data_cell
            auto vl = pool->TakeOne();
            Statistics stats;
            auto failed_without_vector = searcher->Search(
                graph_data_cell, nullptr, vl, base_vectors.data(), search_param, nullptr, stats);
            pool->ReturnOne(vl);
            REQUIRE(failed_without_vector->Size() == 0);
        }
        {
            // search with empty vector_data_cell
            auto vl = pool->TakeOne();
            Statistics stats;
            auto failed_without_graph = searcher->Search(
                nullptr, vector_data_cell, vl, base_vectors.data(), search_param, nullptr, stats);
            pool->ReturnOne(vl);
            REQUIRE(failed_without_graph->Size() == 0);
        }
    };

    auto filter_func = [](LabelType id) -> bool { return id % 2 == 0; };
    float range = 0.1F;
    auto f = std::make_shared<BlackListFilter>(filter_func);

    // search param
    InnerSearchParam search_param_temp;
    search_param_temp.ep = fixed_entry_point_id;
    search_param_temp.ef = ef_search;
    search_param_temp.topk = k;
    search_param_temp.is_inner_id_allowed = nullptr;
    search_param_temp.radius = range;

    std::vector<InnerSearchParam> params(4);
    params[0] = search_param_temp;
    params[1] = search_param_temp;
    params[1].is_inner_id_allowed = f;
    params[2] = search_param_temp;
    params[2].search_mode = RANGE_SEARCH;
    params[3] = params[2];
    params[3].is_inner_id_allowed = f;

    for (const auto& search_param : params) {
        exception_func(search_param);
        auto searcher = std::make_shared<BasicSearcher>(common);
        for (int i = 0; i < query_size; i++) {
            std::unordered_set<InnerIdType> valid_set, set;
            auto vl = pool->TakeOne();
            Statistics stats;
            auto result = searcher->Search(graph_data_cell,
                                           vector_data_cell,
                                           vl,
                                           base_vectors.data() + i * dim,
                                           search_param,
                                           (LabelTablePtr) nullptr,
                                           stats);
            pool->ReturnOne(vl);
            auto result_size = result->Size();
            for (int j = 0; j < result_size; j++) {
                set.insert(result->Top().second);
                result->Pop();
            }
            if (search_param.search_mode == KNN_SEARCH) {
                auto valid_result =
                    alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                              base_vectors.data() + i * dim,
                                                              ef_search,
                                                              search_param.is_inner_id_allowed);
                REQUIRE(result_size == valid_result.size());
                for (int j = 0; j < result_size; j++) {
                    valid_set.insert(valid_result.top().second);
                    valid_result.pop();
                }
            } else if (search_param.search_mode == RANGE_SEARCH) {
                auto valid_result =
                    alg_hnsw->searchBaseLayerST<false, false>(fixed_entry_point_id,
                                                              base_vectors.data() + i * dim,
                                                              range,
                                                              ef_search,
                                                              search_param.is_inner_id_allowed);
                REQUIRE(result_size == valid_result.size());
                for (int j = 0; j < result_size; j++) {
                    valid_set.insert(valid_result.top().second);
                    valid_result.pop();
                }
            }

            for (auto id : set) {
                REQUIRE(valid_set.count(id) > 0);
            }
            for (auto id : valid_set) {
                REQUIRE(set.count(id) > 0);
            }
        }
    }
}

TEST_CASE("Hybrid search leaves candidate set unbounded by default", "[ut][BasicSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 1;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto graph = std::make_shared<StaticGraph>(std::vector<std::vector<InnerIdType>>{
        {1, 2, 3},
        {},
        {},
        {},
    });

    std::vector<uint32_t> query_counts;
    auto flatten = std::make_shared<FixedDistanceFlatten>(
        std::vector<float>{0.0F, 0.1F, 0.2F, 0.3F}, &query_counts);

    InnerSearchParam search_param;
    search_param.ep = 0;
    search_param.ef = 2;
    search_param.topk = 2;
    search_param.is_hybrid = true;
    search_param.is_inner_id_allowed =
        std::make_shared<RejectIdsFilter>(std::vector<InnerIdType>{0, 1});

    auto vl = std::make_shared<VisitedList>(graph->MaxCapacity(), allocator.get());
    BasicSearcher searcher(common);
    Statistics stats;
    float query = 0.0F;

    auto result =
        searcher.Search(graph, flatten, vl, &query, search_param, (LabelTablePtr)nullptr, stats);

    REQUIRE(result->Size() == search_param.topk);
    REQUIRE(stats.hops.load(std::memory_order_relaxed) == 4);
}

TEST_CASE("BasicSearcher direct neighbor view matches copied neighbors",
          "[ut][BasicSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 1;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto run_search = [&](bool use_graph_neighbor_view) {
        auto graph = std::make_shared<StaticGraph>(
            std::vector<std::vector<InnerIdType>>{{1, 2}, {3}, {3}, {}});
        std::vector<uint32_t> query_counts;
        auto flatten = std::make_shared<FixedDistanceFlatten>(
            std::vector<float>{0.0F, 0.1F, 0.2F, 0.3F}, &query_counts);

        InnerSearchParam search_param;
        search_param.ep = 0;
        search_param.ef = 4;
        search_param.topk = 4;
        search_param.use_graph_neighbor_view = use_graph_neighbor_view;

        auto vl = std::make_shared<VisitedList>(graph->MaxCapacity(), allocator.get());
        BasicSearcher searcher(common);
        Statistics stats;
        float query = 0.0F;
        auto result =
            searcher.Search(graph,
                            flatten,
                            vl,
                            &query,
                            search_param,
                            (LabelTablePtr)nullptr,
                            stats);

        std::vector<std::pair<float, InnerIdType>> records;
        while (not result->Empty()) {
            records.emplace_back(result->Top());
            result->Pop();
        }
        return std::make_tuple(records, graph->CopyCalls(), graph->ViewCalls());
    };

    const auto [copied_records, copied_calls, unused_view_calls] = run_search(false);
    const auto [view_records, fallback_copy_calls, view_calls] = run_search(true);
    REQUIRE(copied_records == view_records);
    REQUIRE(copied_calls > 0);
    REQUIRE(unused_view_calls == 0);
    REQUIRE(fallback_copy_calls == 0);
    REQUIRE(view_calls > 0);
}

TEST_CASE("Hybrid sparse upper-bound pruning can be disabled", "[ut][BasicSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 1;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto run_search = [&](bool enable_hybrid_pruning) {
        auto graph = std::make_shared<StaticGraph>(std::vector<std::vector<InnerIdType>>{
            {1, 2, 3},
            {},
            {},
            {},
        });
        std::vector<uint32_t> query_counts;
        auto flatten = std::make_shared<FixedDistanceFlatten>(
            std::vector<float>{0.0F, 0.1F, 0.2F, 0.3F}, &query_counts);

        InnerSearchParam search_param;
        search_param.ep = 0;
        search_param.ef = 2;
        search_param.topk = 2;
        search_param.is_hybrid = true;
        search_param.enable_hybrid_pruning = enable_hybrid_pruning;

        auto vl = std::make_shared<VisitedList>(graph->MaxCapacity(), allocator.get());
        BasicSearcher searcher(common);
        Statistics stats;
        float query = 0.0F;
        auto result =
            searcher.Search(graph, flatten, vl, &query, search_param, (LabelTablePtr)nullptr, stats);
        REQUIRE(result->Size() == search_param.topk);
        return flatten->GetLastSearchLowerBound();
    };

    REQUIRE(run_search(true) < std::numeric_limits<float>::max());
    REQUIRE(run_search(false) == std::numeric_limits<float>::max());
}

TEST_CASE("Hybrid search bounds candidate set by explicit search parameter", "[ut][BasicSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 1;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto graph = std::make_shared<StaticGraph>(std::vector<std::vector<InnerIdType>>{
        {1, 2, 3},
        {},
        {},
        {},
    });

    std::vector<uint32_t> query_counts;
    auto flatten = std::make_shared<FixedDistanceFlatten>(
        std::vector<float>{0.0F, 0.1F, 0.2F, 0.3F}, &query_counts);

    InnerSearchParam search_param;
    search_param.ep = 0;
    search_param.ef = 3;
    search_param.topk = 2;
    search_param.is_hybrid = true;
    search_param.hybrid_candidate_set_size = 1;
    search_param.is_inner_id_allowed =
        std::make_shared<RejectIdsFilter>(std::vector<InnerIdType>{0});

    auto vl = std::make_shared<VisitedList>(graph->MaxCapacity(), allocator.get());
    BasicSearcher searcher(common);
    Statistics stats;
    float query = 0.0F;

    auto result =
        searcher.Search(graph, flatten, vl, &query, search_param, (LabelTablePtr)nullptr, stats);

    REQUIRE(result->Size() == search_param.topk);
    REQUIRE(stats.hops.load(std::memory_order_relaxed) == search_param.hybrid_candidate_set_size + 1);
}

TEST_CASE("Hybrid search stops when max hops is reached", "[ut][BasicSearcher]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = 1;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    auto graph = std::make_shared<StaticGraph>(std::vector<std::vector<InnerIdType>>{
        {1, 2, 3},
        {},
        {},
        {},
    });

    std::vector<uint32_t> query_counts;
    auto flatten = std::make_shared<FixedDistanceFlatten>(
        std::vector<float>{0.0F, 0.1F, 0.2F, 0.3F}, &query_counts);

    InnerSearchParam search_param;
    search_param.ep = 0;
    search_param.ef = 3;
    search_param.topk = 2;
    search_param.is_hybrid = true;
    search_param.max_hops = 2;
    search_param.is_inner_id_allowed =
        std::make_shared<RejectIdsFilter>(std::vector<InnerIdType>{0});

    auto vl = std::make_shared<VisitedList>(graph->MaxCapacity(), allocator.get());
    BasicSearcher searcher(common);
    Statistics stats;
    float query = 0.0F;

    auto result =
        searcher.Search(graph, flatten, vl, &query, search_param, (LabelTablePtr)nullptr, stats);

    REQUIRE(result->Size() == search_param.topk);
    REQUIRE(stats.hops.load(std::memory_order_relaxed) == search_param.max_hops);
}

TEST_CASE("Optimize SQ4", "[ut][BasicOptimizer]") {
    // avoid too much slow task logs
    fixtures::logger::LoggerReplacer _;
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);

    // data attr
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    uint32_t base_size = 1000;
    uint64_t dim = 128;
    auto quantizer_type = GENERATE("fp32", "sq4_uniform");

    // build and search attr
    uint32_t M = 16;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    InnerIdType fixed_entry_point_id = 0;
    uint64_t default_max_element = 1;

    // data preparation
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<InnerIdType> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    // vector data cell
    constexpr const char* param_temp = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_temp, quantizer_type)));
    auto io_param =
        IOParameter::GetIOParameterByJson(JsonType::Parse(fmt::format(param_temp, "memory_io")));
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = allocator;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;

    FlattenInterfacePtr vector_data_cell;
    if (quantizer_type == std::string("sq4_uniform")) {
        vector_data_cell = std::make_shared<
            FlattenDataCell<SQ4UniformQuantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            quantizer_param, io_param, common);
    } else {
        vector_data_cell = std::make_shared<
            FlattenDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            quantizer_param, io_param, common);
    }

    vector_data_cell->Train(base_vectors.data(), base_size);
    vector_data_cell->BatchInsertVector(base_vectors.data(), base_size, ids.data());

    // hnswlib build
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator.get());
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   default_max_element,
                                                   allocator.get(),
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    alg_hnsw->init_memory_space();

    for (int64_t i = 0; i < base_size; ++i) {
        alg_hnsw->addPoint((const void*)(base_vectors.data() + i * dim), ids[i]);
    }

    // graph data cell
    auto graph_data_cell = std::make_shared<AdaptGraphDataCell>(alg_hnsw);

    // pool
    auto init_size = 10;
    auto pool = std::make_shared<VisitedListPool>(
        init_size, allocator.get(), vector_data_cell->TotalCount(), allocator.get());

    // search param
    InnerSearchParam search_param;
    search_param.ep = fixed_entry_point_id;
    search_param.ef = ef_search;
    search_param.topk = k;

    // init searcher
    auto searcher = std::make_shared<BasicSearcher>(common);

    // searcher-optimizer
    searcher->SetMockParameters(graph_data_cell, vector_data_cell, pool, search_param, dim, 1000);
    Statistics stats;
    auto loss_before = searcher->MockRun(stats);
    auto optimizer_searcher = std::make_shared<Optimizer<BasicSearcher>>(common);
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_DEPTH_CODE, 1, 3, 1));
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 3, 1));
    optimizer_searcher->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 3, 1));
    float end2end_improvement = optimizer_searcher->Optimize(searcher);
    auto loss_after = searcher->MockRun(stats);
}
