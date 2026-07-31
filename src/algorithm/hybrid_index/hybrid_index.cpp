//
// Created by root on 2026/1/8.
//
#include "hybrid_index.h"
#include <H5Cpp.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <unordered_set>

#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/attribute_inverted_interface.h"
#include "datacell/flatten_datacell.h"
#include "datacell/flatten_interface.h"
#include "datacell/sparse_vector_datacell_parameter.h"
#include "fmt/chrono.h"
#include "impl/heap/standard_heap.h"
#include "impl/logger/logger.h"
#include "impl/pruning_strategy.h"
#include "index_common_param.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "typing.h"
#include "utils/lock_strategy.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"
#include "vsag/constants.h"

namespace vsag {

namespace {

constexpr float DEFAULT_HYBRID_ALPHA_EPSILON = 1e-6F;
constexpr uint32_t HYBRID_INDEX_UNIFIED_SERIALIZE_VERSION = 2;

GraphInterfacePtr
make_memory_graph(int64_t max_degree, const IndexCommonParam& common_param) {
    JsonType json;
    json["max_degree"].SetInt(max_degree);
    JsonType io_json;
    io_json["type"].SetString("memory_io");
    json["io_params"].SetJson(io_json);
    auto graph_params = GraphInterfaceParameter::GetGraphParameterByJson(
        GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, json);
    return GraphInterface::MakeInstance(graph_params, common_param);
}

uint64_t
get_uint_param(const JsonType& json, const std::string& key, uint64_t default_value) {
    if (json.Contains(key)) {
        return static_cast<uint64_t>(json[key].GetInt());
    }
    return default_value;
}

float
get_float_param(const JsonType& json, const std::string& key, float default_value) {
    if (json.Contains(key)) {
        return json[key].GetFloat();
    }
    return default_value;
}

bool
get_bool_param(const JsonType& json, const std::string& key, bool default_value) {
    if (json.Contains(key)) {
        return json[key].GetBool();
    }
    return default_value;
}

std::string
get_string_param(const JsonType& json, const std::string& key, const std::string& default_value) {
    if (json.Contains(key)) {
        return json[key].GetString();
    }
    return default_value;
}

template <typename T>
void
read_one(StreamReader& reader, T& value) {
    reader.Read(reinterpret_cast<char*>(&value), sizeof(value));
}

uint16_t
read_link_count(const char* link_data) {
    uint16_t count = 0;
    std::memcpy(&count, link_data, sizeof(count));
    return count;
}

}  // namespace

class DenseEntryHNSWGraph {
public:
    void
    LoadFromHNSWIndex(const std::string& path,
                      const LabelTablePtr& label_table,
                      InnerIdType expected_total,
                      Allocator* allocator) {
        std::ifstream input(path, std::ios::binary);
        CHECK_ARGUMENT(input.good(), fmt::format("failed to open dense HNSW graph path {}", path));

        IOStreamReader io_reader(input);
        BufferStreamReader reader(
            &io_reader, std::numeric_limits<uint64_t>::max(), allocator);
        auto footer = Footer::Parse(reader);
        if (footer != nullptr and footer->GetMetadata()->EmptyIndex()) {
            return;
        }

        size_t offset_level0 = 0;
        size_t max_elements = 0;
        size_t cur_element_count = 0;
        size_t size_data_per_element = 0;
        size_t label_offset = 0;
        size_t offset_data = 0;
        int max_level = 0;

        read_one(reader, offset_level0);
        read_one(reader, max_elements);
        read_one(reader, cur_element_count);
        read_one(reader, size_data_per_element);
        read_one(reader, label_offset);
        read_one(reader, offset_data);
        read_one(reader, max_level);

        const auto enterpoint_header_pos = reader.GetCursor();
        char enterpoint_header_buffer[sizeof(int64_t) + sizeof(size_t)] = {};
        reader.Read(enterpoint_header_buffer, sizeof(enterpoint_header_buffer));

        InnerIdType parsed_new_entry = 0;
        size_t parsed_new_max_m = 0;
        std::memcpy(&parsed_new_entry, enterpoint_header_buffer, sizeof(parsed_new_entry));
        std::memcpy(&parsed_new_max_m,
                    enterpoint_header_buffer + sizeof(parsed_new_entry),
                    sizeof(parsed_new_max_m));

        auto read_tail_header = [&](size_t cursor,
                                    size_t& max_m0,
                                    size_t& m,
                                    double& mult,
                                    size_t& ef_construction) {
            reader.Seek(cursor);
            read_one(reader, max_m0);
            read_one(reader, m);
            read_one(reader, mult);
            read_one(reader, ef_construction);
        };

        size_t max_m0 = 0;
        size_t m = 0;
        double mult = 0.0;
        size_t ef_construction = 0;
        read_tail_header(enterpoint_header_pos + sizeof(InnerIdType) + sizeof(size_t),
                         max_m0,
                         m,
                         mult,
                         ef_construction);

        const auto sane_degree = [](size_t degree) {
            return degree > 0 and degree <= 100000;
        };
        const bool newer_header =
            sane_degree(parsed_new_max_m) and m == parsed_new_max_m and max_m0 >= m;

        if (newer_header) {
            enterpoint_node_ = parsed_new_entry;
            max_m_ = parsed_new_max_m;
        } else {
            reader.Seek(enterpoint_header_pos);
            int64_t old_entry = 0;
            read_one(reader, old_entry);
            enterpoint_node_ = old_entry < 0 ? std::numeric_limits<InnerIdType>::max()
                                             : static_cast<InnerIdType>(old_entry);
            read_one(reader, max_m_);
            read_one(reader, max_m0);
            read_one(reader, m);
            read_one(reader, mult);
            read_one(reader, ef_construction);
        }

        CHECK_ARGUMENT(cur_element_count <= std::numeric_limits<InnerIdType>::max(),
                       "dense HNSW graph has too many nodes");
        CHECK_ARGUMENT(size_data_per_element > label_offset,
                       "invalid dense HNSW size_data_per_element/label_offset");
        CHECK_ARGUMENT(label_offset + sizeof(LabelType) <= size_data_per_element,
                       "invalid dense HNSW label offset");
        CHECK_ARGUMENT(offset_level0 == 0,
                       fmt::format("unsupported dense HNSW offset_level0 {}", offset_level0));

        num_nodes_ = static_cast<InnerIdType>(cur_element_count);
        max_level_ = max_level;
        max_m0_ = max_m0;
        internal_to_inner_.assign(num_nodes_, kInvalidInnerId);
        deleted_.assign(num_nodes_, 0);
        levels_.assign(num_nodes_, 0);
        level0_neighbor_offsets_.clear();
        level0_neighbor_offsets_.reserve(static_cast<size_t>(num_nodes_) + 1);
        level0_neighbor_offsets_.emplace_back(0);
        level0_neighbor_ids_.clear();
        upper_neighbors_.assign(num_nodes_, {});
        has_deleted_ = false;

        std::vector<char> element_buffer(size_data_per_element);
        uint64_t missing_labels = 0;
        for (InnerIdType internal_id = 0; internal_id < num_nodes_; ++internal_id) {
            reader.Read(element_buffer.data(), size_data_per_element);
            const auto count = read_link_count(element_buffer.data());
            const bool deleted = (static_cast<unsigned char>(element_buffer[2]) & kDeleteMark) != 0;
            deleted_[internal_id] = deleted ? 1 : 0;
            has_deleted_ = has_deleted_ or deleted;

            LabelType label = 0;
            std::memcpy(&label, element_buffer.data() + label_offset, sizeof(label));
            if (label_table->CheckLabel(label)) {
                internal_to_inner_[internal_id] = label_table->GetIdByLabel(label);
            } else {
                ++missing_labels;
            }

            const char* neighbor_data = element_buffer.data() + sizeof(uint32_t);
            for (uint16_t i = 0; i < count; ++i) {
                InnerIdType neighbor = 0;
                std::memcpy(&neighbor, neighbor_data + i * sizeof(InnerIdType), sizeof(neighbor));
                if (neighbor < num_nodes_) {
                    level0_neighbor_ids_.emplace_back(neighbor);
                }
            }
            level0_neighbor_offsets_.emplace_back(level0_neighbor_ids_.size());
        }

        const size_t size_links_per_element = max_m_ * sizeof(InnerIdType) + sizeof(uint32_t);
        CHECK_ARGUMENT(size_links_per_element > sizeof(uint32_t),
                       "invalid dense HNSW size_links_per_element");
        for (InnerIdType internal_id = 0; internal_id < num_nodes_; ++internal_id) {
            uint32_t link_list_size = 0;
            read_one(reader, link_list_size);
            if (link_list_size == 0) {
                continue;
            }
            CHECK_ARGUMENT(link_list_size % size_links_per_element == 0,
                           "invalid dense HNSW upper link list size");
            const uint32_t level_count =
                static_cast<uint32_t>(link_list_size / size_links_per_element);
            levels_[internal_id] = level_count;
            std::vector<char> link_buffer(link_list_size);
            reader.Read(link_buffer.data(), link_list_size);
            upper_neighbors_[internal_id].resize(level_count);
            for (uint32_t level = 1; level <= level_count; ++level) {
                const char* level_data =
                    link_buffer.data() + (level - 1) * size_links_per_element;
                const auto count = read_link_count(level_data);
                auto& neighbors = upper_neighbors_[internal_id][level - 1];
                neighbors.reserve(count);
                const char* neighbor_data = level_data + sizeof(uint32_t);
                for (uint16_t i = 0; i < count; ++i) {
                    InnerIdType neighbor = 0;
                    std::memcpy(&neighbor,
                                neighbor_data + i * sizeof(InnerIdType),
                                sizeof(neighbor));
                    if (neighbor < num_nodes_) {
                        neighbors.emplace_back(neighbor);
                    }
                }
            }
        }

        if (expected_total != 0 and expected_total != num_nodes_) {
            logger::warn("dense HNSW graph node count {} differs from hybrid_index count {}",
                         num_nodes_,
                         expected_total);
        }
        if (missing_labels > 0) {
            logger::warn("dense HNSW graph loaded with {} labels missing in hybrid_index",
                         missing_labels);
        }
        loaded_ = true;
        logger::info("HybridIndex loaded dense-entry HNSW graph from {}, points={}, max_level={}",
                     path,
                     num_nodes_,
                     max_level_);
    }

    [[nodiscard]] bool
    Loaded() const {
        return loaded_;
    }

    [[nodiscard]] std::vector<InnerIdType>
    Search(const float* query,
           const FlattenInterfacePtr& dense_cell,
           uint64_t topk,
           uint64_t ef,
           Allocator* allocator) const {
        if (not loaded_ or num_nodes_ == 0 or
            enterpoint_node_ == std::numeric_limits<InnerIdType>::max()) {
            return {};
        }
        CHECK_ARGUMENT(dense_cell != nullptr, "dense cell should not be null for HNSW entry search");

        auto computer = dense_cell->FactoryComputer(query);
        auto curr_obj = enterpoint_node_;
        if (curr_obj >= num_nodes_ or not HasVector(curr_obj)) {
            return {};
        }

        float curdist = Distance(curr_obj, dense_cell, computer, allocator);
        for (int level = max_level_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (auto candidate : Neighbors(curr_obj, level)) {
                    if (candidate >= num_nodes_ or not HasVector(candidate)) {
                        continue;
                    }
                    const float distance = Distance(candidate, dense_cell, computer, allocator);
                    if (distance < curdist) {
                        curdist = distance;
                        curr_obj = candidate;
                        changed = true;
                    }
                }
            }
        }

        ef = std::max<uint64_t>(ef, topk);
        auto top_candidates = SearchBaseLayer(curr_obj, dense_cell, computer, ef, allocator);
        while (top_candidates.size() > topk) {
            top_candidates.pop();
        }

        std::vector<InnerIdType> entry_points;
        entry_points.reserve(top_candidates.size());
        while (not top_candidates.empty()) {
            const auto internal_id = top_candidates.top().second;
            if (HasVector(internal_id) and not IsDeleted(internal_id)) {
                entry_points.emplace_back(internal_to_inner_[internal_id]);
            }
            top_candidates.pop();
        }
        return entry_points;
    }

    void
    Serialize(StreamWriter& writer) const {
        constexpr uint32_t version = 1;
        StreamWriter::WriteObj(writer, version);
        StreamWriter::WriteObj(writer, loaded_);
        StreamWriter::WriteObj(writer, num_nodes_);
        StreamWriter::WriteObj(writer, max_level_);
        StreamWriter::WriteObj(writer, enterpoint_node_);
        StreamWriter::WriteObj(writer, max_m_);
        StreamWriter::WriteObj(writer, max_m0_);
        StreamWriter::WriteObj(writer, has_deleted_);
        StreamWriter::WriteVector(writer, internal_to_inner_);
        StreamWriter::WriteVector(writer, deleted_);
        StreamWriter::WriteVector(writer, levels_);

        uint64_t level0_size = num_nodes_;
        StreamWriter::WriteObj(writer, level0_size);
        for (InnerIdType internal_id = 0; internal_id < num_nodes_; ++internal_id) {
            const auto neighbors = Neighbors(internal_id, 0);
            const uint64_t degree = neighbors.size;
            StreamWriter::WriteObj(writer, degree);
            if (degree > 0) {
                writer.Write(reinterpret_cast<const char*>(neighbors.data),
                             degree * sizeof(InnerIdType));
            }
        }

        uint64_t upper_size = upper_neighbors_.size();
        StreamWriter::WriteObj(writer, upper_size);
        for (const auto& per_node : upper_neighbors_) {
            uint64_t level_count = per_node.size();
            StreamWriter::WriteObj(writer, level_count);
            for (const auto& neighbors : per_node) {
                StreamWriter::WriteVector(writer, neighbors);
            }
        }
    }

    void
    Deserialize(StreamReader& reader) {
        uint32_t version = 0;
        StreamReader::ReadObj(reader, version);
        CHECK_ARGUMENT(version == 1,
                       fmt::format("unsupported dense-entry HNSW graph version {}", version));
        StreamReader::ReadObj(reader, loaded_);
        StreamReader::ReadObj(reader, num_nodes_);
        StreamReader::ReadObj(reader, max_level_);
        StreamReader::ReadObj(reader, enterpoint_node_);
        StreamReader::ReadObj(reader, max_m_);
        StreamReader::ReadObj(reader, max_m0_);
        StreamReader::ReadObj(reader, has_deleted_);
        StreamReader::ReadVector(reader, internal_to_inner_);
        StreamReader::ReadVector(reader, deleted_);
        StreamReader::ReadVector(reader, levels_);

        uint64_t level0_size = 0;
        StreamReader::ReadObj(reader, level0_size);
        CHECK_ARGUMENT(
            level0_size == num_nodes_,
            fmt::format("invalid dense-entry HNSW level-0 node count {}, expected {}",
                        level0_size,
                        num_nodes_));
        level0_neighbor_offsets_.clear();
        level0_neighbor_offsets_.reserve(level0_size + 1);
        level0_neighbor_offsets_.emplace_back(0);
        level0_neighbor_ids_.clear();
        for (uint64_t internal_id = 0; internal_id < level0_size; ++internal_id) {
            uint64_t degree = 0;
            StreamReader::ReadObj(reader, degree);
            CHECK_ARGUMENT(
                degree <= std::numeric_limits<size_t>::max() - level0_neighbor_ids_.size(),
                "dense-entry HNSW level-0 neighbor count overflow");
            const auto old_size = level0_neighbor_ids_.size();
            level0_neighbor_ids_.resize(old_size + static_cast<size_t>(degree));
            if (degree > 0) {
                reader.Read(reinterpret_cast<char*>(level0_neighbor_ids_.data() + old_size),
                            degree * sizeof(InnerIdType));
            }
            level0_neighbor_offsets_.emplace_back(level0_neighbor_ids_.size());
        }

        uint64_t upper_size = 0;
        StreamReader::ReadObj(reader, upper_size);
        upper_neighbors_.resize(upper_size);
        for (auto& per_node : upper_neighbors_) {
            uint64_t level_count = 0;
            StreamReader::ReadObj(reader, level_count);
            per_node.resize(level_count);
            for (auto& neighbors : per_node) {
                StreamReader::ReadVector(reader, neighbors);
            }
        }
    }

private:
    using CandidateQueue = std::priority_queue<std::pair<float, InnerIdType>>;
    static constexpr unsigned char kDeleteMark = 0x01;
    static constexpr InnerIdType kInvalidInnerId = std::numeric_limits<InnerIdType>::max();

    struct NeighborView {
        const InnerIdType* data{nullptr};
        size_t size{0};

        [[nodiscard]] const InnerIdType*
        begin() const {
            return data;
        }

        [[nodiscard]] const InnerIdType*
        end() const {
            return size == 0 ? data : data + size;
        }
    };

    [[nodiscard]] bool
    HasVector(InnerIdType internal_id) const {
        return internal_id < internal_to_inner_.size() and internal_to_inner_[internal_id] != kInvalidInnerId;
    }

    [[nodiscard]] bool
    IsDeleted(InnerIdType internal_id) const {
        return internal_id < deleted_.size() and deleted_[internal_id] != 0;
    }

    [[nodiscard]] NeighborView
    Neighbors(InnerIdType internal_id, int level) const {
        if (internal_id >= num_nodes_ or level < 0) {
            return {};
        }
        if (level == 0) {
            const auto begin_offset = level0_neighbor_offsets_[internal_id];
            const auto end_offset = level0_neighbor_offsets_[internal_id + 1];
            if (begin_offset == end_offset) {
                return {};
            }
            return {level0_neighbor_ids_.data() + begin_offset, end_offset - begin_offset};
        }
        const auto upper_level = static_cast<size_t>(level - 1);
        if (upper_level >= upper_neighbors_[internal_id].size()) {
            return {};
        }
        const auto& neighbors = upper_neighbors_[internal_id][upper_level];
        return {neighbors.data(), neighbors.size()};
    }

    [[nodiscard]] float
    Distance(InnerIdType internal_id,
             const FlattenInterfacePtr& dense_cell,
             const ComputerInterfacePtr& computer,
             Allocator* allocator) const {
        if (not HasVector(internal_id)) {
            return std::numeric_limits<float>::max();
        }
        float distance = std::numeric_limits<float>::max();
        const auto hybrid_inner_id = internal_to_inner_[internal_id];
        dense_cell->Query(&distance, computer, &hybrid_inner_id, 1, allocator);
        return distance;
    }

    [[nodiscard]] CandidateQueue
    SearchBaseLayer(InnerIdType ep_id,
                    const FlattenInterfacePtr& dense_cell,
                    const ComputerInterfacePtr& computer,
                    uint64_t ef,
                    Allocator* allocator) const {
        CandidateQueue top_candidates;
        CandidateQueue candidate_set;

        thread_local std::vector<uint32_t> visited;
        thread_local uint32_t visited_tag = 0;
        if (visited.size() < num_nodes_) {
            visited.resize(num_nodes_, 0);
        }
        ++visited_tag;
        if (visited_tag == 0) {
            std::fill(visited.begin(), visited.end(), 0);
            visited_tag = 1;
        }

        float lower_bound = std::numeric_limits<float>::max();
        if (HasVector(ep_id) and not IsDeleted(ep_id)) {
            const float distance = Distance(ep_id, dense_cell, computer, allocator);
            lower_bound = distance;
            top_candidates.emplace(distance, ep_id);
            candidate_set.emplace(-distance, ep_id);
        } else {
            candidate_set.emplace(-lower_bound, ep_id);
        }
        visited[ep_id] = visited_tag;

        while (not candidate_set.empty()) {
            const auto current_node_pair = candidate_set.top();
            const float candidate_distance = -current_node_pair.first;
            if (candidate_distance > lower_bound and
                (top_candidates.size() == ef or not has_deleted_)) {
                break;
            }
            candidate_set.pop();

            const auto current_node_id = current_node_pair.second;
            for (auto candidate_id : Neighbors(current_node_id, 0)) {
                if (candidate_id >= num_nodes_ or visited[candidate_id] == visited_tag) {
                    continue;
                }
                visited[candidate_id] = visited_tag;
                if (not HasVector(candidate_id)) {
                    continue;
                }

                const float distance = Distance(candidate_id, dense_cell, computer, allocator);
                if (top_candidates.size() < ef or lower_bound > distance) {
                    candidate_set.emplace(-distance, candidate_id);

                    if (not IsDeleted(candidate_id)) {
                        top_candidates.emplace(distance, candidate_id);
                    }

                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }

                    if (not top_candidates.empty()) {
                        lower_bound = top_candidates.top().first;
                    }
                }
            }
        }
        return top_candidates;
    }

private:
    bool loaded_{false};
    InnerIdType num_nodes_{0};
    int max_level_{0};
    InnerIdType enterpoint_node_{std::numeric_limits<InnerIdType>::max()};
    size_t max_m_{0};
    size_t max_m0_{0};
    bool has_deleted_{false};
    std::vector<InnerIdType> internal_to_inner_;
    std::vector<uint8_t> deleted_;
    std::vector<uint32_t> levels_;
    std::vector<size_t> level0_neighbor_offsets_;
    std::vector<InnerIdType> level0_neighbor_ids_;
    std::vector<std::vector<std::vector<InnerIdType>>> upper_neighbors_;
};

HybridIndex::HybridIndex(const HybridIndexParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      common_param_(common_param),
      searcher_(std::make_unique<BasicSearcher>(common_param)),
      ef_construction_(param->ef_construction),
      max_degree_(param->max_degree),
      dense_entry_ef_construction_(param->dense_entry_ef_construction),
      dense_entry_max_degree_(param->dense_entry_max_degree),
      alpha_(param->alpha),
      enable_sindi_(param->enable_sindi),
      enable_dense_entry_(param->enable_dense_entry),
      default_sindi_bk_(param->default_sindi_bk),
      default_dense_entry_bk_(param->default_dense_entry_bk),
      default_dense_entry_ef_search_(param->default_dense_entry_ef_search),
      auto_sparse_alpha_max_(param->auto_sparse_alpha_max),
      auto_dense_alpha_min_(param->auto_dense_alpha_min),
      search_method_(param->search_method),
      graph_path_(param->graph_path),
      sindi_index_path_(param->sindi_index_path),
      dense_entry_hnsw_graph_path_(param->dense_entry_hnsw_graph_path) {
    const char* dense_param_str =
        R"({
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            }
        }
    )";
    auto dense_param_json = JsonType::Parse(dense_param_str);
    auto dense_param = std::make_shared<FlattenDataCellParameter>();
    dense_param->FromJson(dense_param_json);

    const char* sparse_param_str =
        R"({
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "sparse"
            }
        }
    )";

    JsonType sparse_param_json = JsonType::Parse(sparse_param_str);
    auto sparse_param = std::make_shared<SparseVectorDataCellParameter>();
    sparse_param->FromJson(sparse_param_json);

    hybrid_codes_ = std::make_shared<HybridVectorDataCell>(dense_param, sparse_param, common_param);
    hybrid_codes_->SetHybridWeight(alpha_, 1-alpha_);

    graph_ = make_memory_graph(max_degree_, common_param);
    graph_mutex_ = std::make_shared<EmptyMutex>();
    searcher_->SetMutexArray(graph_mutex_);

    if (enable_dense_entry_ and dense_entry_hnsw_graph_path_.empty()) {
        dense_entry_graph_ = make_memory_graph(dense_entry_max_degree_, common_param);
        dense_entry_graph_mutex_ = std::make_shared<EmptyMutex>();
        dense_entry_searcher_ =
            std::make_unique<BasicSearcher>(common_param, dense_entry_graph_mutex_);
    } else if (enable_dense_entry_) {
        dense_entry_hnsw_graph_ = std::make_shared<DenseEntryHNSWGraph>();
    }

    if (enable_sindi_) {
        sindi_parameter_ = std::make_shared<SINDIParameter>();
        sindi_parameter_->FromJson(param->sindi_json);
        if (sindi_parameter_->use_reorder) {
            logger::warn("HybridIndex forces internal SINDI use_reorder=false to avoid duplicating "
                         "sparse raw vectors");
            sindi_parameter_->use_reorder = false;
        }
        sindi_index_ = std::make_shared<SINDI>(sindi_parameter_, common_param);
        sindi_index_->InitFeatures();
    }
}


ParamPtr
HybridIndex::CheckAndMappingExternalParam(const JsonType& external_param,
                                          const IndexCommonParam& common_param) {

    auto inner_json = external_param;

    auto hybrid_index_parameter = std::make_shared<HybridIndexParameter>();
    hybrid_index_parameter->FromJson(inner_json);

    return hybrid_index_parameter;
}


void
HybridIndex::Train(const DatasetPtr& base) {
    // const auto* base_data_dense = base->GetFloat32Vectors();
    // const auto* base_data_sparse = base->GetSparseVectors();
    // auto num_elements = base->GetNumElements();
    //
    // // 计算 dense 部分的总大小
    // size_t dense_total_size = num_elements * dim_ * sizeof(float);
    //
    // // 计算 sparse 部分的总大小
    // size_t sparse_total_size = 0;
    // for (size_t i = 0; i < num_elements; ++i) {
    //     const auto& sv = base_data_sparse[i];
    //     sparse_total_size += sizeof(uint32_t) +                    // len
    //                         sizeof(uint32_t) * sv.len_ + // ids
    //                         sizeof(float) * sv.len_;      // vals
    // }
    //
    // auto base_data = static_cast<uint8_t*>(allocator_->Allocate(dense_total_size + sparse_total_size));
    // memcpy(base_data, base_data_dense, dense_total_size);
    // // 复制 sparse（逐个序列化）
    // uint8_t* sparse_ptr = base_data + dense_total_size;
    // for (size_t i = 0; i < num_elements; ++i) {
    //     const auto& sv = base_data_sparse[i];
    //     uint32_t len = sv.len_;
    //
    //     // len
    //     std::memcpy(sparse_ptr, &len, sizeof(uint32_t));
    //     sparse_ptr += sizeof(uint32_t);
    //
    //     if (len > 0) {
    //         // ids
    //         std::memcpy(sparse_ptr, sv.ids_, sizeof(uint32_t) * len);
    //         sparse_ptr += sizeof(uint32_t) * len;
    //
    //         // vals
    //         std::memcpy(sparse_ptr, sv.vals_, sizeof(float) * len);
    //         sparse_ptr += sizeof(float) * len;
    //     }
    // }
    //
    // this->hybrid_codes_->Train(base_data, base->GetNumElements());
    // allocator_->Deallocate(base_data);
}

std::vector<int64_t>
HybridIndex::Build(const DatasetPtr& data) {
    // this->Train(data);  train is used when qualification is needed
    graph_->Resize(data->GetNumElements());
    if (dense_entry_graph_ != nullptr) {
        dense_entry_graph_->Resize(data->GetNumElements());
    }
    if (not graph_path_.empty()) {
        load_precomputed_neighbors();
    }
    if (not sindi_index_path_.empty()) {
        load_sindi_index();
    }
    auto failed_ids = this->Add(data);
    if (not dense_entry_hnsw_graph_path_.empty()) {
        load_dense_entry_hnsw_graph();
    }
    return failed_ids;
}

void
HybridIndex::load_precomputed_neighbors() {
    if (precomputed_neighbors_loaded_) {
        return;
    }
    CHECK_ARGUMENT(not graph_path_.empty(), "graph_path should not be empty");
    try {
        H5::H5File file(graph_path_, H5F_ACC_RDONLY);

        H5::DataSet neighbors_dataset = file.openDataSet("neighbors");
        H5::DataSpace neighbors_dataspace = neighbors_dataset.getSpace();
        hsize_t dims[2];
        neighbors_dataspace.getSimpleExtentDims(dims);

        precomputed_num_points_ = static_cast<int64_t>(dims[0]);
        precomputed_max_neighbors_ = static_cast<int64_t>(dims[1]);

        precomputed_neighbors_.resize(precomputed_num_points_ * precomputed_max_neighbors_);
        neighbors_dataset.read(precomputed_neighbors_.data(), H5::PredType::NATIVE_INT64);

        H5::DataSet counts_dataset = file.openDataSet("neighbor_counts");
        precomputed_neighbor_counts_.resize(precomputed_num_points_);
        counts_dataset.read(precomputed_neighbor_counts_.data(), H5::PredType::NATIVE_INT64);

        precomputed_neighbors_loaded_ = true;
        logger::info("HybridIndex loaded precomputed graph from {}, points={}",
                     graph_path_,
                     precomputed_num_points_);
    } catch (H5::Exception& e) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("failed to load hybrid graph_path {}: {}",
                                        graph_path_,
                                        e.getDetailMsg()));
    }
}

void
HybridIndex::load_sindi_index() {
    if (sindi_loaded_from_path_) {
        return;
    }
    CHECK_ARGUMENT(sindi_index_ != nullptr,
                   "sindi_index_path requires internal SINDI to be enabled");
    CHECK_ARGUMENT(not sindi_index_path_.empty(), "sindi_index_path should not be empty");

    std::ifstream input(sindi_index_path_, std::ios::binary);
    CHECK_ARGUMENT(input.good(), fmt::format("failed to open SINDI index {}", sindi_index_path_));
    IOStreamReader reader(input);
    sindi_index_->Deserialize(reader);
    sindi_loaded_from_path_ = true;
    logger::info("HybridIndex loaded internal SINDI from {}", sindi_index_path_);
}

void
HybridIndex::load_dense_entry_hnsw_graph() {
    CHECK_ARGUMENT(enable_dense_entry_,
                   "dense_entry_hnsw_graph_path requires dense entry to be enabled");
    CHECK_ARGUMENT(not dense_entry_hnsw_graph_path_.empty(),
                   "dense_entry_hnsw_graph_path should not be empty");
    if (dense_entry_hnsw_graph_ == nullptr) {
        dense_entry_hnsw_graph_ = std::make_shared<DenseEntryHNSWGraph>();
    }
    if (dense_entry_hnsw_graph_->Loaded()) {
        return;
    }
    dense_entry_hnsw_graph_->LoadFromHNSWIndex(
        dense_entry_hnsw_graph_path_, label_table_, static_cast<InnerIdType>(total_count_),
        allocator_);
}

void
HybridIndex::add_hybrid_graph_point(InnerIdType inner_id, const float* vector) {
    if (precomputed_neighbors_loaded_) {
        CHECK_ARGUMENT(inner_id < precomputed_num_points_,
                       fmt::format("precomputed graph has {} points, but building id {}",
                                   precomputed_num_points_,
                                   inner_id));
        int64_t actual_count = std::min(precomputed_neighbor_counts_[inner_id],
                                        static_cast<int64_t>(max_degree_));
        Vector<InnerIdType> neighbor_vec(allocator_);
        neighbor_vec.reserve(actual_count);

        const int64_t row_offset = inner_id * precomputed_max_neighbors_;
        for (int64_t i = 0; i < actual_count; ++i) {
            int64_t neighbor_id = precomputed_neighbors_[row_offset + i];
            if (neighbor_id >= 0) {
                neighbor_vec.emplace_back(static_cast<InnerIdType>(neighbor_id));
            }
        }
        graph_->InsertNeighborsById(inner_id, neighbor_vec);
        if (entry_point_id_ == std::numeric_limits<InnerIdType>::max()) {
            entry_point_id_ = inner_id;
        }
        return;
    }

    if (graph_->TotalCount() == 0) {
        graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        entry_point_id_ = inner_id;
        return;
    }

    InnerSearchParam search_param;
    search_param.ef = ef_construction_;
    search_param.topk = max_degree_;
    search_param.search_mode = KNN_SEARCH;
    search_param.ep = entry_point_id_;
    search_param.is_hybrid = true;
    search_param.hybrid_dense_weight = alpha_;
    search_param.hybrid_sparse_weight = 1 - alpha_;

    auto vl = std::make_shared<VisitedList>(graph_->max_capacity_, allocator_);
    Statistics discard_stats;
    auto results = searcher_->Search(
        graph_, hybrid_codes_, vl, vector, search_param, (LabelTablePtr)nullptr, discard_stats);
    mutually_connect_new_element(
        inner_id, results, graph_, hybrid_codes_, graph_mutex_, allocator_);
}

void
HybridIndex::add_dense_entry_point(InnerIdType inner_id, const float* dense_vector) {
    if (dense_entry_graph_ == nullptr) {
        return;
    }
    if (dense_entry_graph_->TotalCount() == 0) {
        dense_entry_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        dense_entry_point_id_ = inner_id;
        return;
    }

    InnerSearchParam search_param;
    search_param.ef = dense_entry_ef_construction_;
    search_param.topk = dense_entry_max_degree_;
    search_param.search_mode = KNN_SEARCH;
    search_param.ep = dense_entry_point_id_;

    auto vl = std::make_shared<VisitedList>(dense_entry_graph_->max_capacity_, allocator_);
    Statistics discard_stats;
    auto results = dense_entry_searcher_->Search(dense_entry_graph_,
                                                 hybrid_codes_->GetDenseCell(),
                                                 vl,
                                                 dense_vector,
                                                 search_param,
                                                 (LabelTablePtr)nullptr,
                                                 discard_stats);
    mutually_connect_new_element(inner_id,
                                 results,
                                 dense_entry_graph_,
                                 hybrid_codes_->GetDenseCell(),
                                 dense_entry_graph_mutex_,
                                 allocator_);
}

void
HybridIndex::add_one_point(InnerIdType inner_id, const float* vector) {
    add_hybrid_graph_point(inner_id, vector);
}

void
HybridIndex::reset_graph_visited_pool() {
    if (graph_ == nullptr or graph_->max_capacity_ == 0) {
        graph_visited_pool_.reset();
        return;
    }
    graph_visited_pool_ =
        std::make_shared<VisitedListPool>(1, allocator_, graph_->max_capacity_, allocator_);
}

std::vector<int64_t>
HybridIndex::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    int64_t vec_num = data->GetNumElements();

    int64_t dim = data->GetDim();
    auto labels = data->GetIds();

    auto new_total = total_count_ + vec_num;
    graph_->Resize(new_total);
    // Dense payload has a fixed size, so reserve it once and avoid realloc/mremap in the
    // per-vector insert loop.  Do not resize the whole hybrid cell here: sparse payload is
    // variable-length and SparseVectorDataCell::Resize uses the maximum possible sparse code size
    // rather than the actual payload size.
    hybrid_codes_->GetDenseCell()->Resize(new_total);
    if (dense_entry_graph_ != nullptr) {
        dense_entry_graph_->Resize(new_total);
    }
    if (dense_entry_hnsw_graph_ != nullptr and dense_entry_hnsw_graph_->Loaded()) {
        CHECK_ARGUMENT(false,
                       "Add after loading dense-entry HNSW graph is not supported");
    }
    if (sindi_index_ != nullptr and not sindi_loaded_from_path_) {
        auto sindi_failed_ids = sindi_index_->Add(data);
        CHECK_ARGUMENT(sindi_failed_ids.empty(),
                       "internal SINDI failed to add some vectors; sparse ids would be misaligned");
        failed_ids.insert(failed_ids.end(), sindi_failed_ids.begin(), sindi_failed_ids.end());
    }

    auto dense_vecs = data->GetFloat32Vectors();
    auto sparse_vecs = data->GetSparseVectors();

    const auto dense_bytes = dim * sizeof(float);
    const auto sparse_ref_bytes = sizeof(sparse_vecs[0]);
    auto hybrid_vec = static_cast<uint8_t*>(allocator_->Allocate(dense_bytes + sparse_ref_bytes));

    // insert labels and points
    for (auto offset = 0; offset < vec_num; offset++) {
        auto inner_id = total_count_ + offset;
        label_table_->Insert(inner_id, labels[offset]);
        std::memcpy(hybrid_vec, dense_vecs + offset * dim, dense_bytes);
        std::memcpy(hybrid_vec + dense_bytes, sparse_vecs + offset, sparse_ref_bytes);
        hybrid_codes_->InsertVector(hybrid_vec, inner_id);
        add_one_point(inner_id, reinterpret_cast<const float*>(hybrid_vec));
        add_dense_entry_point(inner_id, dense_vecs + offset * dim);
    }

    allocator_->Deallocate(hybrid_vec);

    this->total_count_ += vec_num;
    reset_graph_visited_pool();
    return failed_ids;
}

HybridIndex::SearchMethod
HybridIndex::resolve_search_method(const JsonType& parsed_search_param, float alpha) const {
    std::string method =
        get_string_param(parsed_search_param, "method", search_method_);
    method = get_string_param(parsed_search_param, "search_method", method);
    std::transform(method.begin(), method.end(), method.begin(), ::tolower);

    if (method == "auto") {
        if (alpha <= auto_sparse_alpha_max_ + DEFAULT_HYBRID_ALPHA_EPSILON) {
            return SearchMethod::UHGS;
        }
        if (alpha > auto_dense_alpha_min_ + DEFAULT_HYBRID_ALPHA_EPSILON) {
            return SearchMethod::UHGH;
        }
        return SearchMethod::UHG;
    }
    if (method == "uhg") {
        return SearchMethod::UHG;
    }
    if (method == "uhgs") {
        return SearchMethod::UHGS;
    }
    if (method == "uhgh") {
        return SearchMethod::UHGH;
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("method must be one of uhg/uhgs/uhgh/auto, got {}",
                                    method));
}

std::vector<InnerIdType>
HybridIndex::get_sindi_entry_points(const DatasetPtr& query,
                                    const JsonType& parsed_search_param,
                                    int64_t k,
                                    const FilterPtr& filter) const {
    CHECK_ARGUMENT(sindi_index_ != nullptr,
                   "hybrid_index method=uhgs requires internal SINDI to be enabled");
    auto uhgs_json = parsed_search_param.Contains("uhgs") ? parsed_search_param["uhgs"] : JsonType();
    uint64_t default_bk = default_sindi_bk_;
    if (default_bk == 0) {
        default_bk = get_uint_param(parsed_search_param, "ef_search", static_cast<uint64_t>(k));
    }
    auto sindi_bk = get_uint_param(uhgs_json, "sindi_bk", default_bk);
    sindi_bk = get_uint_param(parsed_search_param, "sindi_bk", sindi_bk);
    sindi_bk = std::max<uint64_t>(sindi_bk, static_cast<uint64_t>(k));

    JsonType sindi_params;
    sindi_params[INDEX_SINDI].SetJson(JsonType());
    const float query_prune_ratio =
        get_float_param(parsed_search_param,
                        "sindi_query_prune_ratio",
                        get_float_param(uhgs_json, "sindi_query_prune_ratio", DEFAULT_QUERY_PRUNE_RATIO));
    const float term_prune_ratio =
        get_float_param(parsed_search_param,
                        "sindi_term_prune_ratio",
                        get_float_param(uhgs_json, "sindi_term_prune_ratio", DEFAULT_TERM_PRUNE_RATIO));
    const uint64_t n_candidate =
        get_uint_param(parsed_search_param,
                       "sindi_n_candidate",
                       get_uint_param(uhgs_json, "sindi_n_candidate", DEFAULT_N_CANDIDATE));
    sindi_params[INDEX_SINDI][SPARSE_QUERY_PRUNE_RATIO].SetFloat(query_prune_ratio);
    sindi_params[INDEX_SINDI][SPARSE_TERM_PRUNE_RATIO].SetFloat(term_prune_ratio);
    sindi_params[INDEX_SINDI][SPARSE_N_CANDIDATE].SetInt(n_candidate);

    auto sparse_query = Dataset::Make();
    sparse_query->NumElements(1)->SparseVectors(query->GetSparseVectors())->Owner(false);
    auto sindi_result_holder =
        sindi_index_->KnnSearch(sparse_query, static_cast<int64_t>(sindi_bk), sindi_params.Dump(), filter);

    std::vector<InnerIdType> entry_points;
    entry_points.reserve(sindi_result_holder->GetDim());
    for (int64_t i = 0; i < sindi_result_holder->GetDim(); ++i) {
        const auto label = sindi_result_holder->GetIds()[i];
        if (label_table_->CheckLabel(label)) {
            entry_points.emplace_back(label_table_->GetIdByLabel(label));
        }
    }
    return entry_points;
}

std::vector<InnerIdType>
HybridIndex::get_dense_entry_points(const DatasetPtr& query,
                                    const JsonType& parsed_search_param,
                                    int64_t k,
                                    Statistics& stats) const {
    auto uhgh_json = parsed_search_param.Contains("uhgh") ? parsed_search_param["uhgh"] : JsonType();
    uint64_t default_bk = default_dense_entry_bk_;
    if (default_bk == 0) {
        default_bk = get_uint_param(parsed_search_param, "ef_search", static_cast<uint64_t>(k));
    }
    auto entry_bk = get_uint_param(uhgh_json, "entry_bk", default_bk);
    entry_bk = get_uint_param(uhgh_json, "hnsw_bk", entry_bk);
    entry_bk = get_uint_param(parsed_search_param, "dense_entry_bk", entry_bk);
    entry_bk = get_uint_param(parsed_search_param, "hnsw_bk", entry_bk);
    entry_bk = std::max<uint64_t>(entry_bk, static_cast<uint64_t>(k));

    uint64_t default_ef = default_dense_entry_ef_search_ == 0 ? entry_bk
                                                               : default_dense_entry_ef_search_;
    auto entry_ef = get_uint_param(uhgh_json, "entry_ef_search", default_ef);
    entry_ef = get_uint_param(uhgh_json, "hnsw_ef_search", entry_ef);
    entry_ef = get_uint_param(parsed_search_param, "dense_entry_ef_search", entry_ef);
    entry_ef = get_uint_param(parsed_search_param, "hnsw_ef_search", entry_ef);
    entry_ef = std::max(entry_ef, entry_bk);

    if (dense_entry_hnsw_graph_ != nullptr and dense_entry_hnsw_graph_->Loaded()) {
        return dense_entry_hnsw_graph_->Search(query->GetFloat32Vectors(),
                                               hybrid_codes_->GetDenseCell(),
                                               entry_bk,
                                               entry_ef,
                                               allocator_);
    }

    CHECK_ARGUMENT(dense_entry_graph_ != nullptr and dense_entry_searcher_ != nullptr,
                   "hybrid_index method=uhgh requires dense entry graph to be enabled");
    if (dense_entry_graph_->TotalCount() == 0) {
        return {};
    }

    InnerSearchParam search_param;
    search_param.ef = entry_ef;
    search_param.topk = static_cast<int64_t>(entry_bk);
    search_param.search_mode = KNN_SEARCH;
    search_param.ep = dense_entry_point_id_;

    auto vl = std::make_shared<VisitedList>(dense_entry_graph_->max_capacity_, allocator_);
    auto dense_results = dense_entry_searcher_->Search(dense_entry_graph_,
                                                       hybrid_codes_->GetDenseCell(),
                                                       vl,
                                                       query->GetFloat32Vectors(),
                                                       search_param,
                                                       (LabelTablePtr)nullptr,
                                                       stats);

    std::vector<InnerIdType> entry_points;
    entry_points.reserve(dense_results->Size());
    while (not dense_results->Empty()) {
        entry_points.emplace_back(dense_results->Top().second);
        dense_results->Pop();
    }
    return entry_points;
}

DatasetPtr
HybridIndex::graph_knn_search(const DatasetPtr& query,
                              int64_t k,
                              const JsonType& parsed_search_param,
                              float alpha,
                              const std::vector<InnerIdType>& entry_points) const {
    InnerSearchParam search_param;
    search_param.ef = get_uint_param(parsed_search_param, "ef_search", std::max<int64_t>(k, 10));
    search_param.hybrid_candidate_set_size = parsed_search_param.Contains("hybrid_candidate_set_size")
                                                 ? parsed_search_param["hybrid_candidate_set_size"].GetInt()
                                                 : 0;
    search_param.max_hops = parsed_search_param.Contains("max_hops")
                                ? parsed_search_param["max_hops"].GetInt()
                                : 0;
    search_param.topk = k;
    search_param.search_mode = KNN_SEARCH;
    search_param.is_hybrid = true;
    search_param.use_graph_neighbor_view = true;
    search_param.enable_hybrid_pruning =
        get_bool_param(parsed_search_param, "enable_hybrid_pruning", true);
    search_param.hybrid_dense_weight = alpha;
    search_param.hybrid_sparse_weight = 1 - alpha;

    if (not entry_points.empty()) {
        for (const auto inner_id : entry_points) {
            if (inner_id >= total_count_) {
                continue;
            }
            search_param.eps.push_back(inner_id);
        }
        if (search_param.eps.empty()) {
            search_param.ep = entry_point_id_;
        }
    } else if (parsed_search_param.Contains("entry_points") &&
               parsed_search_param["entry_points"].IsArray()) {
        const auto& ep_array = parsed_search_param["entry_points"].GetVector();
        std::unordered_set<InnerIdType> dedup_entry_points;
        for (int i = 0; i < ep_array.size(); ++i) {
            int64_t label_id = ep_array[i];
            if (not label_table_->CheckLabel(label_id)) {
                continue;
            }
            auto inner_id = label_table_->GetIdByLabel(label_id);
            if (dedup_entry_points.insert(inner_id).second) {
                search_param.eps.push_back(inner_id);
            }
        }
        if (search_param.eps.empty()) {
            search_param.ep = entry_point_id_;
        }
    } else if (parsed_search_param.Contains("entry_point")) {
        search_param.ep = parsed_search_param["entry_point"].GetInt();
    } else {
        search_param.ep = entry_point_id_;
    }

    search_param.hybrid_prune_scale = parsed_search_param.Contains("hybrid_prune_scale")
                                          ? parsed_search_param["hybrid_prune_scale"].GetFloat()
                                          : 1.0F;

    auto dense_vector = query->GetFloat32Vectors();
    auto dim = query->GetDim();
    auto sparse_vector = query->GetSparseVectors();
    auto hybrid_vector = (int8_t*)(allocator_->Allocate(dim * sizeof(float) + sizeof(sparse_vector[0])));
    std::memcpy(hybrid_vector, dense_vector, dim * sizeof(float));
    std::memcpy(hybrid_vector + dim * sizeof(float), sparse_vector, sizeof(sparse_vector[0]));

    auto visited_pool = graph_visited_pool_;
    auto vl = visited_pool != nullptr
                  ? visited_pool->TakeOne()
                  : std::make_shared<VisitedList>(graph_->max_capacity_, allocator_);
    Statistics discard_stats;
    auto search_results = searcher_->Search(graph_,
                                            hybrid_codes_,
                                            vl,
                                            hybrid_vector,
                                            search_param,
                                            (LabelTablePtr)nullptr,
                                            discard_stats);
    if (visited_pool != nullptr) {
        visited_pool->ReturnOne(vl);
    }

    allocator_->Deallocate(hybrid_vector);
    int64_t result_size = search_results->Size();
    auto results = Dataset::Make();

    results->NumElements(1)->Dim(result_size)->Owner(true, allocator_);
    auto ids = (int64_t*)allocator_->Allocate(result_size * sizeof(int64_t));
    results->Ids(ids);
    auto dists = (float*)allocator_->Allocate(result_size * sizeof(float));
    results->Distances(dists);

    for (auto j = result_size - 1; j >= 0; --j) {
        if (j < result_size) {
            dists[j] = search_results->Top().first;
            ids[j] = label_table_->GetLabelById(search_results->Top().second);
        }
        search_results->Pop();
    }
    results->Statistics(discard_stats.Dump());
    return results;
}

DatasetPtr
HybridIndex::KnnSearch(const DatasetPtr& query,
                       int64_t k,
                       const std::string& parameters,
                       const FilterPtr& filter) const {
    CHECK_ARGUMENT(query != nullptr, "query should not be null");
    CHECK_ARGUMENT(query->GetNumElements() == 1, "hybrid_index supports one query per search");
    CHECK_ARGUMENT(k > 0, "topk should be positive");

    auto parsed_search_param = parameters.empty() ? JsonType::Parse("{}") : JsonType::Parse(parameters);
    const float search_alpha =
        parsed_search_param.Contains("alpha") ? parsed_search_param["alpha"].GetFloat() : alpha_;
    CHECK_ARGUMENT((0.0F <= search_alpha and search_alpha <= 1.0F),
                   fmt::format("alpha must in [0, 1], got {}", search_alpha));

    auto method = resolve_search_method(parsed_search_param, search_alpha);
    std::vector<InnerIdType> entry_points;

    switch (method) {
        case SearchMethod::UHGS:
            entry_points = get_sindi_entry_points(query, parsed_search_param, k, filter);
            break;
        case SearchMethod::UHGH: {
            Statistics dense_entry_stats;
            entry_points = get_dense_entry_points(query, parsed_search_param, k, dense_entry_stats);
            break;
        }
        case SearchMethod::UHG:
            break;
    }

    return graph_knn_search(query,
                            k,
                            parsed_search_param,
                            search_alpha,
                            entry_points);
}

DatasetPtr
HybridIndex::SearchWithRequest(const SearchRequest& request) const {
    // // support 1 query only
    // auto heap = vsag::DistanceHeap::MakeInstanceBySize<false, true>(this->allocator_, request.topk_);
    //
    // auto dense_query = request.query_->GetFloat32Vectors();
    // auto sparse_query = request.query_->GetSparseVectors();
    //
    // for (int i = 0; i < this->total_count_; i++) {
    //     float dis = 0;
    //     // dense ip distance
    //     float dense_dis = 0;
    //     for (int j = 0; j < this->dim_; j++) {
    //         dense_dis += dense_query[j] * dense_vector_[i * this->dim_ + j];
    //     }
    //
    //     // sparse ip distance
    //     float sparse_dis = 0;
    //     auto cur_sparse_vector = sparse_vector_[i];
    //     auto num_base_len = cur_sparse_vector.len_;
    //     auto num_query_len = sparse_query[0].len_;
    //     int j = 0, k = 0;
    //     while (j < num_base_len && k < num_query_len) {
    //         if (cur_sparse_vector.ids_[j] < sparse_query[0].ids_[k]) j++;
    //         else if (cur_sparse_vector.ids_[j] > sparse_query[0].ids_[k]) k++;
    //         else {
    //             sparse_dis += cur_sparse_vector.vals_[j] * sparse_query[0].vals_[k];
    //             k++;
    //             j++;
    //         }
    //     }
    //
    //     JsonType params = vsag::JsonType::Parse(request.params_str_);
    //     auto search_alpha = params["alpha"].GetFloat();
    //     dis = search_alpha * dense_dis + (1 - search_alpha) * sparse_dis;
    //     heap->Push(dis, i);
    // }
    //
    //
    // auto [results, dists, ids] = create_fast_dataset(static_cast<int64_t>(heap->Size()), allocator_); // structed binding
    // for (int j = 0; j < request.topk_; j++) {
    //     dists[request.topk_ - j - 1] = heap->Top().first;
    //     ids[request.topk_ - j - 1] = heap->Top().second;
    //     heap->Pop();
    // }
    // return results;
    auto data = Dataset::Make();
    return data;
}

float
HybridIndex::CalcDistanceById(const float* vector, int64_t id) const {
    float result = 0.0F;
    return result;
}


void
HybridIndex::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
}

void
HybridIndex::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    this->attr_filter_index_->GetAttribute(0, inner_id, attr);
}


void
HybridIndex::InitFeatures() {

}

DatasetPtr
HybridIndex::RangeSearch(const vsag::DatasetPtr& query,
                        float radius,
                        const std::string& parameters,
                        const vsag::FilterPtr& filter,
                        int64_t limited_size) const {
    auto result = vsag::Dataset::Make();
    return result;

}



void HybridIndex::Serialize(StreamWriter& writer) const {
    label_table_->Serialize(writer);
    hybrid_codes_->Serialize(writer);
    graph_->Serialize(writer);

    StreamWriter::WriteObj(writer, entry_point_id_);
    StreamWriter::WriteObj(writer, ef_construction_);
    StreamWriter::WriteObj(writer, max_degree_);
    StreamWriter::WriteObj(writer, total_count_);
    StreamWriter::WriteObj(writer, alpha_);

    StreamWriter::WriteObj(writer, HYBRID_INDEX_UNIFIED_SERIALIZE_VERSION);
    StreamWriter::WriteObj(writer, enable_sindi_);
    StreamWriter::WriteObj(writer, enable_dense_entry_);
    StreamWriter::WriteObj(writer, dense_entry_ef_construction_);
    StreamWriter::WriteObj(writer, dense_entry_max_degree_);
    StreamWriter::WriteObj(writer, default_sindi_bk_);
    StreamWriter::WriteObj(writer, default_dense_entry_bk_);
    StreamWriter::WriteObj(writer, default_dense_entry_ef_search_);
    StreamWriter::WriteObj(writer, auto_sparse_alpha_max_);
    StreamWriter::WriteObj(writer, auto_dense_alpha_min_);
    StreamWriter::WriteString(writer, search_method_);
    StreamWriter::WriteString(writer, graph_path_);
    StreamWriter::WriteObj(writer, dense_entry_point_id_);

    const bool has_dense_entry_graph = dense_entry_graph_ != nullptr;
    StreamWriter::WriteObj(writer, has_dense_entry_graph);
    if (has_dense_entry_graph) {
        dense_entry_graph_->Serialize(writer);
    }

    const bool has_sindi_index = sindi_index_ != nullptr and sindi_parameter_ != nullptr;
    StreamWriter::WriteObj(writer, has_sindi_index);
    if (has_sindi_index) {
        StreamWriter::WriteString(writer, sindi_parameter_->ToJson().Dump());
        std::stringstream sindi_stream;
        IOStreamWriter sindi_writer(sindi_stream);
        sindi_index_->Serialize(sindi_writer);
        StreamWriter::WriteString(writer, sindi_stream.str());
    }

    StreamWriter::WriteString(writer, sindi_index_path_);
    StreamWriter::WriteString(writer, dense_entry_hnsw_graph_path_);
    const bool has_dense_entry_hnsw_graph =
        dense_entry_hnsw_graph_ != nullptr and dense_entry_hnsw_graph_->Loaded();
    StreamWriter::WriteObj(writer, has_dense_entry_hnsw_graph);
    if (has_dense_entry_hnsw_graph) {
        dense_entry_hnsw_graph_->Serialize(writer);
    }
}

void
HybridIndex::Deserialize(StreamReader& reader) {
    label_table_->Deserialize(reader);
    hybrid_codes_->Deserialize(reader);
    graph_->Deserialize(reader);
    reset_graph_visited_pool();

    StreamReader::ReadObj(reader, entry_point_id_);
    StreamReader::ReadObj(reader, ef_construction_);
    StreamReader::ReadObj(reader, max_degree_);
    StreamReader::ReadObj(reader, total_count_);
    StreamReader::ReadObj(reader, alpha_);

    if (reader.GetCursor() >= reader.Length()) {
        return;
    }

    uint32_t serialize_version = 0;
    StreamReader::ReadObj(reader, serialize_version);
    CHECK_ARGUMENT(serialize_version == 1 or serialize_version == HYBRID_INDEX_UNIFIED_SERIALIZE_VERSION,
                   fmt::format("unsupported hybrid_index serialize version {}",
                               serialize_version));

    StreamReader::ReadObj(reader, enable_sindi_);
    StreamReader::ReadObj(reader, enable_dense_entry_);
    StreamReader::ReadObj(reader, dense_entry_ef_construction_);
    StreamReader::ReadObj(reader, dense_entry_max_degree_);
    StreamReader::ReadObj(reader, default_sindi_bk_);
    StreamReader::ReadObj(reader, default_dense_entry_bk_);
    StreamReader::ReadObj(reader, default_dense_entry_ef_search_);
    StreamReader::ReadObj(reader, auto_sparse_alpha_max_);
    StreamReader::ReadObj(reader, auto_dense_alpha_min_);
    search_method_ = StreamReader::ReadString(reader);
    graph_path_ = StreamReader::ReadString(reader);
    StreamReader::ReadObj(reader, dense_entry_point_id_);

    bool has_dense_entry_graph = false;
    StreamReader::ReadObj(reader, has_dense_entry_graph);
    if (has_dense_entry_graph) {
        if (dense_entry_graph_ == nullptr) {
            dense_entry_graph_ = make_memory_graph(dense_entry_max_degree_, common_param_);
        }
        if (dense_entry_graph_mutex_ == nullptr) {
            dense_entry_graph_mutex_ = std::make_shared<EmptyMutex>();
        }
        if (dense_entry_searcher_ == nullptr) {
            dense_entry_searcher_ =
                std::make_unique<BasicSearcher>(common_param_, dense_entry_graph_mutex_);
        }
        dense_entry_graph_->Deserialize(reader);
    } else {
        dense_entry_graph_.reset();
        dense_entry_searcher_.reset();
        dense_entry_graph_mutex_.reset();
        dense_entry_point_id_ = std::numeric_limits<InnerIdType>::max();
    }

    bool has_sindi_index = false;
    StreamReader::ReadObj(reader, has_sindi_index);
    if (has_sindi_index) {
        const auto sindi_param_string = StreamReader::ReadString(reader);
        sindi_parameter_ = std::make_shared<SINDIParameter>();
        sindi_parameter_->FromJson(JsonType::Parse(sindi_param_string));
        if (sindi_parameter_->use_reorder) {
            logger::warn("HybridIndex forces internal SINDI use_reorder=false to avoid duplicating "
                         "sparse raw vectors");
            sindi_parameter_->use_reorder = false;
        }
        sindi_index_ = std::make_shared<SINDI>(sindi_parameter_, common_param_);
        sindi_index_->InitFeatures();
        const auto sindi_blob = StreamReader::ReadString(reader);
        std::stringstream sindi_stream(sindi_blob);
        IOStreamReader sindi_reader(sindi_stream);
        sindi_index_->Deserialize(sindi_reader);
        sindi_loaded_from_path_ = true;
    } else {
        sindi_parameter_.reset();
        sindi_index_.reset();
        sindi_loaded_from_path_ = false;
    }

    if (serialize_version >= 2 and reader.GetCursor() < reader.Length()) {
        sindi_index_path_ = StreamReader::ReadString(reader);
        dense_entry_hnsw_graph_path_ = StreamReader::ReadString(reader);
        bool has_dense_entry_hnsw_graph = false;
        StreamReader::ReadObj(reader, has_dense_entry_hnsw_graph);
        if (has_dense_entry_hnsw_graph) {
            dense_entry_hnsw_graph_ = std::make_shared<DenseEntryHNSWGraph>();
            dense_entry_hnsw_graph_->Deserialize(reader);
            dense_entry_graph_.reset();
            dense_entry_searcher_.reset();
            dense_entry_graph_mutex_.reset();
            dense_entry_point_id_ = std::numeric_limits<InnerIdType>::max();
        } else {
            dense_entry_hnsw_graph_.reset();
        }
    }

}

}  // namespace vsag
