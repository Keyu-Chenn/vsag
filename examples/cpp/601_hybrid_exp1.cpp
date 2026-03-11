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
        if (offset + sizeof(uint32_t) > blob.size()) {
            break;
        }

        uint32_t len;
        std::memcpy(&len, &blob[offset], sizeof(uint32_t));
        offset += sizeof(uint32_t);

        vsag::SparseVector vec;
        vec.len_ = len;

        if (len > 0) {
            size_t ids_bytes = len * sizeof(uint32_t);
            if (offset + ids_bytes > blob.size()) {
                break;
            }

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

// 保存邻居到HDF5的函数
void SaveNeighborsToHDF5(const std::string& filename, 
                         const std::vector<std::vector<int64_t>>& all_neighbors) {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        
        // 找出最大邻居数（用于确定数组维度）
        size_t max_neighbors = 0;
        for (const auto& neighbors : all_neighbors) {
            max_neighbors = std::max(max_neighbors, neighbors.size());
        }
        
        std::cout << "Max neighbors count: " << max_neighbors << std::endl;
        
        // 创建数据集（num_points x max_neighbors），用-1填充空位
        hsize_t dims[2] = {all_neighbors.size(), max_neighbors};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = file.createDataSet("neighbors", 
                                                  H5::PredType::NATIVE_INT64, 
                                                  dataspace);
        
        // 准备数据（填充-1表示无效邻居）
        std::vector<int64_t> flat_data(all_neighbors.size() * max_neighbors, -1);
        for (size_t i = 0; i < all_neighbors.size(); i++) {
            for (size_t j = 0; j < all_neighbors[i].size(); j++) {
                flat_data[i * max_neighbors + j] = all_neighbors[i][j];
            }
        }
        
        dataset.write(flat_data.data(), H5::PredType::NATIVE_INT64);
        
        // 保存每个点的实际邻居数
        hsize_t count_dims[1] = {all_neighbors.size()};
        H5::DataSpace count_space(1, count_dims);
        H5::DataSet count_dataset = file.createDataSet("neighbor_counts", 
                                                        H5::PredType::NATIVE_INT64, 
                                                        count_space);
        
        std::vector<int64_t> counts;
        for (const auto& neighbors : all_neighbors) {
            counts.push_back(neighbors.size());
        }
        count_dataset.write(counts.data(), H5::PredType::NATIVE_INT64);
        
        std::cout << "Saved neighbors to " << filename << std::endl;
        
    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error when saving: " << e.getDetailMsg() << std::endl;
        throw;
    }
}

void AddReverseEdgesForced(
    std::vector<std::vector<int64_t>>& graph,
    const std::vector<float>& train_dense,
    const std::vector<vsag::SparseVector>& train_sparse,
    int dense_dim,
    int max_degree)
{
    int n = graph.size();

    for (int i = 0; i < n; i++) {
        for (int64_t j : graph[i]) {
            if (j < 0 || j >= n) continue;

            auto& jlist = graph[j];
            bool exists = std::find(
                jlist.begin(), jlist.end(), (int64_t)i) != jlist.end();
            if (exists) continue;

            if ((int)jlist.size() < max_degree) {
                // 出度未满，直接加
                jlist.push_back(i);
            } else {
                // 出度已满，找最远的邻居替换
                // 用alpha=0.5的hybrid距离衡量重要性
                float worst_dist = std::numeric_limits<float>::max();
                int worst_idx = -1;

                for (int k = 0; k < (int)jlist.size(); k++) {
                    float dd = CalDenseIp(
                        train_dense.data() + j * dense_dim,
                        train_dense.data() + jlist[k] * dense_dim,
                        dense_dim);
                    float sd = CalSparseIp(
                        train_sparse[j], train_sparse[jlist[k]]);
                    float hd = 0.5f * dd + 0.5f * sd;
                    // IP距离：越小越远
                    if (hd < worst_dist) {
                        worst_dist = hd;
                        worst_idx = k;
                    }
                }

                // 检查i是否比最差邻居更近
                float i_dd = CalDenseIp(
                    train_dense.data() + j * dense_dim,
                    train_dense.data() + i * dense_dim, dense_dim);
                float i_sd = CalSparseIp(train_sparse[j], train_sparse[i]);
                float i_hd = 0.5f * i_dd + 0.5f * i_sd;

                if (i_hd > worst_dist && worst_idx >= 0) {
                    jlist[worst_idx] = i;
                }
                // 若i比最差邻居还远，跳过（不强制替换）
            }
        }
    }
}

// 新增函数，放在 AddReverseEdgesForced 之前（AddReverseEdgesForced 可以整个删掉）

// inline计算hybrid距离（IP模式下越大越近，转为dist取负）
inline float HybridDist(
    int id_a, int id_b,
    const std::vector<float>& train_dense,
    const std::vector<vsag::SparseVector>& train_sparse,
    int dense_dim, float alpha)
{
    float dd = CalDenseIp(
        train_dense.data() + id_a * dense_dim,
        train_dense.data() + id_b * dense_dim, dense_dim);
    float sd = CalSparseIp(train_sparse[id_a], train_sparse[id_b]);
    // IP越大越近 → dist取负
    return -(alpha * dd + (1.0f - alpha) * sd);
}

// RNG剪枝：去掉被已选邻居"支配"的候选点
// alpha_eval: 用哪个alpha评估距离（建议传0.5或你的目标alpha）
// alpha_rng:  支配阈值系数，>1表示允许保留一定冗余（DiskANN默认1.2）
std::vector<int64_t> PruneNeighborsByRNG(
    int query_id,
    const std::vector<int64_t>& candidates,
    const std::vector<float>& train_dense,
    const std::vector<vsag::SparseVector>& train_sparse,
    int dense_dim,
    int max_degree,
    float alpha_eval = 0.5f,
    float alpha_rng  = 1.2f)
{
    struct Cand {
        int64_t id;
        float   dist; // 越小越近
    };

    // 按距离从近到远排序
    std::vector<Cand> sorted_cands;
    sorted_cands.reserve(candidates.size());
    for (int64_t cid : candidates) {
        if (cid == query_id) continue; // 排除自己
        float d = HybridDist(query_id, (int)cid,
                             train_dense, train_sparse,
                             dense_dim, alpha_eval);
        sorted_cands.push_back({cid, d});
    }
    std::sort(sorted_cands.begin(), sorted_cands.end(),
              [](const Cand& a, const Cand& b){ return a.dist < b.dist; });

    std::vector<int64_t> result;
    result.reserve(max_degree);

    for (auto& cand : sorted_cands) {
        if ((int)result.size() >= max_degree) break;

        bool dominated = false;
        for (int64_t sel_id : result) {
            // dist(selected, cand) < alpha_rng * dist(query, cand) → 被支配
            float dist_sc = HybridDist((int)sel_id, (int)cand.id,
                                       train_dense, train_sparse,
                                       dense_dim, alpha_eval);
            if (dist_sc < alpha_rng * cand.dist) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            result.push_back(cand.id);
        }
    }
    return result;
}


//原来的 AddReverseEdgesForced，替换为：
void AddReverseEdgesWithPrune(
    std::vector<std::vector<int64_t>>& graph,
    const std::vector<float>& train_dense,
    const std::vector<vsag::SparseVector>& train_sparse,
    int dense_dim,
    int max_degree,
    float alpha_eval = 0.5f,
    float alpha_rng  = 1.2f)
{
    int n = (int)graph.size();

    // 第一步：收集每个节点需要接收的反向边候选
    std::vector<std::vector<int64_t>> reverse_candidates(n);
    for (int i = 0; i < n; i++) {
        for (int64_t j : graph[i]) {
            if (j < 0 || j >= n) continue;
            reverse_candidates[j].push_back((int64_t)i);
        }
    }

    // 第二步：合并正向邻居 + 反向候选，重新RNG剪枝
    for (int j = 0; j < n; j++) {
        if (reverse_candidates[j].empty()) continue;

        // 合并去重
        std::unordered_set<int64_t> merged(graph[j].begin(), graph[j].end());
        for (int64_t rc : reverse_candidates[j]) {
            merged.insert(rc);
        }

        // RNG剪枝后写回
        graph[j] = PruneNeighborsByRNG(
            j,
            std::vector<int64_t>(merged.begin(), merged.end()),
            train_dense, train_sparse,
            dense_dim, max_degree,
            alpha_eval, alpha_rng);
    }
}

// ===== DFS 兜底连通 =====
void EnsureConnectivity(
    std::vector<std::vector<int64_t>>& graph,
    const std::vector<float>& train_dense,
    const std::vector<vsag::SparseVector>& train_sparse,
    int dense_dim,
    int max_degree,
    float alpha_eval = 0.5f)
{
    int n = (int)graph.size();
    if (n == 0) return;

    // Step 1: 选择入口节点（这里选 0，也可以选度数最大的）
    int entry = 0;

    // Step 2: DFS 找出所有可达节点
    std::vector<bool> visited(n, false);
    std::vector<int> stack;
    stack.push_back(entry);
    visited[entry] = true;
    int visited_count = 1;

    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        for (int64_t nb : graph[cur]) {
            if (nb >= 0 && nb < n && !visited[nb]) {
                visited[nb] = true;
                visited_count++;
                stack.push_back((int)nb);
            }
        }
    }

    std::cout << "[DFS] Reachable from entry " << entry
              << ": " << visited_count << " / " << n << std::endl;

    if (visited_count == n) {
        std::cout << "[DFS] Graph is already fully connected." << std::endl;
        return;
    }

    // Step 3: 对不可达节点，找已连通集合中最近的节点，强制连边
    int fixed_count = 0;
    for (int i = 0; i < n; i++) {
        if (visited[i]) continue;

        // 在已访问节点中找距离最近的节点（暴力扫描，可优化）
        float best_dist = std::numeric_limits<float>::max();
        int best_j = entry; // 兜底用 entry

        // 为避免全量扫描，只扫已连通节点的样本（可调整采样比例）
        for (int j = 0; j < n; j++) {
            if (!visited[j]) continue;
            float d = HybridDist(i, j, train_dense, train_sparse,
                                 dense_dim, alpha_eval);
            if (d < best_dist) {
                best_dist = d;
                best_j = j;
            }
        }

        // 强制添加双向边
        // i → best_j
        if (std::find(graph[i].begin(), graph[i].end(), (int64_t)best_j)
            == graph[i].end()) {
            if ((int)graph[i].size() < max_degree) {
                graph[i].push_back(best_j);
            } else {
                graph[i][0] = best_j; // 兜底强制替换第一个
            }
        }

        // best_j → i（反向边）
        if (std::find(graph[best_j].begin(), graph[best_j].end(), (int64_t)i)
            == graph[best_j].end()) {
            if ((int)graph[best_j].size() < max_degree) {
                graph[best_j].push_back(i);
            } else {
                graph[best_j][0] = i; // 兜底强制替换
            }
        }

        // 将 i 标记为已连通，并 DFS 扩展
        visited[i] = true;
        visited_count++;
        stack.push_back(i);
        while (!stack.empty()) {
            int cur = stack.back(); stack.pop_back();
            for (int64_t nb : graph[cur]) {
                if (nb >= 0 && nb < n && !visited[nb]) {
                    visited[nb] = true;
                    visited_count++;
                    stack.push_back((int)nb);
                }
            }
        }

        fixed_count++;
    }

    std::cout << "[DFS] Fixed " << fixed_count << " disconnected nodes." << std::endl;
    std::cout << "[DFS] Final reachable: " << visited_count << " / " << n << std::endl;
}


int
main(int argc, char** argv) {
    vsag::init();

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <h5_file_path> [output_neighbors.h5]" << std::endl;
        return 1;
    }

    std::string h5_file = argv[1];
    std::string output_file = (argc >= 3) ? argv[2] : "all_neighbors.h5";

    try {
        /******************* Load Dataset from HDF5 *****************/
        H5::H5File file(h5_file, H5F_ACC_RDONLY);

        H5::DataSet train_dataset = file.openDataSet("train");
        H5::DataSpace train_dataspace = train_dataset.getSpace();
        hsize_t train_dims[2];
        train_dataspace.getSimpleExtentDims(train_dims);
        int64_t num_train = train_dims[0];
        int64_t dense_dim = train_dims[1];

        std::vector<float> train_dense(num_train * dense_dim);
        train_dataset.read(train_dense.data(), H5::PredType::NATIVE_FLOAT);
        std::cout << "Loaded train dense: " << num_train << " x " << dense_dim << std::endl;

        H5::DataSet train_sparse_dataset = file.openDataSet("train_sparse");
        H5::DataSpace train_sparse_dataspace = train_sparse_dataset.getSpace();
        hsize_t train_sparse_size = train_sparse_dataspace.getSimpleExtentNpoints();

        std::vector<uint8_t> train_sparse_blob(train_sparse_size);
        train_sparse_dataset.read(train_sparse_blob.data(), H5::PredType::NATIVE_UINT8);

        auto train_sparse = ParseSparseVectors(train_sparse_blob);
        std::cout << "Loaded train sparse: " << train_sparse.size() << " vectors" << std::endl;

        H5::DataSet train_labels_dataset = file.openDataSet("train_labels");
        H5::DataSpace train_labels_dataspace = train_labels_dataset.getSpace();
        hsize_t num_train_labels = train_labels_dataspace.getSimpleExtentNpoints();

        std::vector<int64_t> train_labels(num_train_labels);
        train_labels_dataset.read(train_labels.data(), H5::PredType::NATIVE_INT64);
        std::cout << "Loaded train labels: " << num_train_labels << std::endl;

        /******************* Prepare Base Dataset *****************/
        auto base = vsag::Dataset::Make();
        base->NumElements(num_train)
            ->Dim(dense_dim)
            ->Ids(train_labels.data())
            ->Float32Vectors(train_dense.data())
            ->SparseVectors(train_sparse.data())
            ->Owner(false);

        int k = 32;
        int ak = 10 * k;

        /******************* Build Dense Index: hgraph *****************/
        std::string hgraph_build_parameters = R"(
        {
            "dtype": "float32",
            "metric_type": "ip",
            "dim": 1024,
            "index_param": {
                "base_quantization_type": "sq8",
                "max_degree": 64,
                "ef_construction": 200,
                "alpha":1.2
            }
        }
        )";
        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine engine(&resource);
        auto index_hgraph = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

        if (auto build_result_hgraph = index_hgraph->Build(base); build_result_hgraph.has_value()) {
            std::cout << "After Build(), Index HGraph contains: " << index_hgraph->GetNumElements() << std::endl;
        } else {
            std::cerr << "Failed to build hgraph index" << std::endl;
            exit(-1);
        }

        /******************* Build Sparse Index: sindi *****************/
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
            std::cout << "After Build(), Sparse Term Index contains: " << index_sindi->GetNumElements() << std::endl;
        } else {
            std::cerr << "Failed to build sindi index" << std::endl;
            exit(-1);
        }

        /******************* Process Each Point *****************/
        float avg_total_neighbor_num = 0;
        int tt_num = std::min(80000, (int)num_train);
        int bk_num = 10 * k;
        std::vector<float> cover_rate(bk_num);
        
        // **新增：存储所有点的邻居**
        std::vector<std::vector<int64_t>> all_points_neighbors(tt_num);
        std::vector<std::vector<int64_t>> all_points_neighbors_0_1(tt_num);
        std::vector<std::vector<int64_t>> all_points_neighbors_1(tt_num);

        int max_neighbor_num = 0;
        for (int i = 0; i < tt_num; i++) {
            if (i % 1000 == 0) {
                std::cout << "Processing point " << i << "/" << tt_num << std::endl;
            }

            auto q_train = vsag::Dataset::Make();
            q_train->NumElements(1)
                ->Dim(dense_dim)
                ->Ids(train_labels.data() + i)
                ->Float32Vectors(train_dense.data() + i * dense_dim)
                ->SparseVectors(train_sparse.data() + i)
                ->Owner(false);

            auto hgraph_search_parameters = R"({"hgraph": {"ef_search": 400}})";
            auto result_dense = index_hgraph->KnnSearch(q_train, ak, hgraph_search_parameters).value();

            auto sindi_search_params = R"({"sindi": {"query_prune_ratio": 0, "term_prune_ratio": 0, "n_candidate": 0}})";
            auto result_sparse = index_sindi->KnnSearch(q_train, ak, sindi_search_params).value();

            // 合并dense和sparse结果
            std::vector<union_search_res> ds_u_res;
            std::unordered_set<int> seen_ids;
            std::unordered_set<int> all_neighbors;
            std::unordered_set<int> neighbors_0_1;
            std::unordered_set<int> neighbors_1;

            for (int64_t j = 0; j < result_dense->GetDim(); ++j) {
                auto cur_id = result_dense->GetIds()[j];
                if (seen_ids.insert(cur_id).second) {
                    auto cur_dense_dist = CalDenseIp(train_dense.data() + i * dense_dim, 
                                                     train_dense.data() + cur_id * dense_dim, dense_dim);
                    auto cur_sparse_dist = CalSparseIp(train_sparse[i], train_sparse[cur_id]);
                    ds_u_res.push_back({cur_id, cur_dense_dist, cur_sparse_dist, 0});
                }
            }

            for (int64_t j = 0; j < result_sparse->GetDim(); ++j) {
                auto cur_id = result_sparse->GetIds()[j];
                if (seen_ids.insert(cur_id).second) {
                    auto cur_dense_dist = CalDenseIp(train_dense.data() + i * dense_dim, 
                                                     train_dense.data() + cur_id * dense_dim, dense_dim);
                    auto cur_sparse_dist = CalSparseIp(train_sparse[i], train_sparse[cur_id]);
                    ds_u_res.push_back({cur_id, cur_dense_dist, cur_sparse_dist, 0});
                }
            }

            // 遍历不同alpha，收集所有top-k邻居
            for (float alpha = 0; alpha <= 1.0 + 1e-5; alpha += 0.01) {
                for (auto& item : ds_u_res) {
                    item.GetHybridDis(alpha);
                }
                std::sort(ds_u_res.begin(), ds_u_res.end(), std::greater<union_search_res>());
                for (int j = 0; j < std::min(k, (int)ds_u_res.size()); j++) {
                    all_neighbors.insert(ds_u_res[j].id);
                    if (std::abs(alpha) < 1e-5) {
                        neighbors_0_1.insert(ds_u_res[j].id);
                    }
                }
                if (std::abs(alpha - 1.0) < 1e-5) {
                    for (int j = 0; j < std::min(2 * k, (int)ds_u_res.size()); j++) {
                        neighbors_1.insert(ds_u_res[j].id);
                        if (neighbors_0_1.size() < 2 * k) {
                            neighbors_0_1.insert(ds_u_res[j].id);
                        }
                    }
                }
            }

            // **保存当前点的邻居到向量中**
            std::vector<int64_t> neighbors_vec(all_neighbors.begin(), all_neighbors.end());
            std::sort(neighbors_vec.begin(), neighbors_vec.end());  // 排序便于后续使用
            all_points_neighbors[i] = neighbors_vec;
            // std::vector<int64_t> cands(all_neighbors.begin(), all_neighbors.end());
            // all_points_neighbors[i] = PruneNeighborsByRNG(
            //     i, cands, train_dense, train_sparse,
            //     dense_dim,
            //     96,     // max_degree，和后面AddReverse保持一致
            //     0.5f,   // alpha_eval
            //     1.2f);  // alpha_rng

            std::vector<int64_t> neighbors_0_1_vec(neighbors_0_1.begin(), neighbors_0_1.end());
            std::sort(neighbors_0_1_vec.begin(), neighbors_0_1_vec.end());
            all_points_neighbors_0_1[i] = neighbors_0_1_vec;

            std::vector<int64_t> neighbors_1_vec(neighbors_1.begin(), neighbors_1.end());
            std::sort(neighbors_1_vec.begin(), neighbors_1_vec.end());
            all_points_neighbors_1[i] = neighbors_1_vec;

            avg_total_neighbor_num += all_neighbors.size();
            max_neighbor_num = all_neighbors.size() > max_neighbor_num ? all_neighbors.size() : max_neighbor_num;

            // 计算覆盖率
            std::unordered_set<int> dsu;
            for (int bk = 0; bk < bk_num; bk++) {
                if (bk < result_dense->GetDim()) {
                    int denseid = result_dense->GetIds()[bk];
                    if (all_neighbors.count(denseid)) {
                        dsu.insert(denseid);
                    }
                }
                if (bk < result_sparse->GetDim()) {
                    int sparseid = result_sparse->GetIds()[bk];
                    if (all_neighbors.count(sparseid)) {
                        dsu.insert(sparseid);
                    }
                }
                cover_rate[bk] += (float)dsu.size() / all_neighbors.size();
            }
        }
        std::cout << "max_neighbor_num: " << max_neighbor_num << std::endl;

        std::cout << "Average neighbors per point: " << avg_total_neighbor_num / tt_num << std::endl;

        // AddReverseEdgesForced(all_points_neighbors, train_dense, train_sparse, dense_dim, 96);
        AddReverseEdgesWithPrune(
            all_points_neighbors,
            train_dense, train_sparse,
            dense_dim,
            96,    // max_degree
            0.5f,  // alpha_eval
            1.2f); // alpha_rng

        // ====== 新增：DFS 兜底连通 ======
        EnsureConnectivity(
            all_points_neighbors,
            train_dense, train_sparse,
            (int)dense_dim,
            96,    // max_degree
            0.5f); // alpha_eval

        /******************* Save Results *****************/
        // 保存邻居到HDF5
        SaveNeighborsToHDF5(output_file, all_points_neighbors);
        SaveNeighborsToHDF5("601_output_neighbors_k32_alpha_0_1.h5", all_points_neighbors_0_1);
        SaveNeighborsToHDF5("601_output_neighbors_k32_alpha_1.h5", all_points_neighbors_1);

        // 保存统计信息到文本文件
        std::ofstream fout("601_results_msmarco_k32.txt");
        fout << "avg_num: " << avg_total_neighbor_num / tt_num << std::endl;
        fout << "cover_rate: " << std::endl;
        for (int bk = 0; bk < bk_num; bk++) {
            fout << "bk" << bk + 1 << ": " << cover_rate[bk] / tt_num << std::endl;
        }
        fout.close();
        std::cout << "Saved statistics to 601_results_msmarco_k32.txt" << std::endl;

        /******************* Cleanup *****************/
        FreeSparseVectors(train_sparse);
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
