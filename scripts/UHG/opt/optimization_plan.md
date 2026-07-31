# UHG 搜索性能优化方案

## 1. 背景与动机

### 1.1 当前现状

通过 `test_uhgh` 实验（`scripts/UHG/results/test_uhgh`）测试了 UHGH 在 NQ 上 `alpha=0.7` 时，`dense_entry_ef_search` 和 `dense_entry_bk` 同时按 `scale × ef_search`（scale ∈ {0.5, 1.0, 1.2, 1.5, 2.0}）缩放的 Recall-QPS 曲线，结论是：

| ef_search | scale=0.5 (ef=50) | scale=1.0 (ef=100) | scale=2.0 (ef=200) |
|---|---|---|---|
| 100 | Recall 0.921 / QPS 165.7 | Recall 0.921 / QPS 165.3 | Recall 0.957 / QPS 114.5 |
| 500 | Recall 0.982 / QPS 53.7 | Recall 0.990 / QPS 40.9 | Recall 0.995 / QPS 28.0 |

- **scale 0.5 vs 1.0 在 ef=100 时 Recall 完全相同**：说明 entry 搜索在 ef=50 已经收敛，更宽的 HNSW entry 搜索对最终 Recall 无贡献
- **scale 1.0 → 2.0 提升 Recall 但 QPS 线性下降**：只是平移曲线，没有 Pareto frontier 改善
- 结论：**entry search 不是瓶颈**，调整 entry 参数无法提升整体性能

### 1.2 优化目标

转向主图搜索循环本身。通读 `hybrid_index` 完整搜索流程后，识别出若干可优化点。本文档按收益预期排序，给出完整方案，并分析每个改动对 5 个 baseline 方法（`hnsw` / `sindi` / `hnsw_sindi` / `fhg` / `uhg`）的影响。

## 2. UHG 搜索流程梳理

### 2.1 调用链

```
HybridIndex::KnnSearch (hybrid_index.cpp:1165)
├─ resolve_search_method: auto/uhg/uhgs/uhgh
├─ [entry 阶段]
│   ├─ UHGS: get_sindi_entry_points → SINDI KnnSearch → vector<InnerIdType>
│   └─ UHGH: get_dense_entry_points → DenseEntryHNSWGraph::Search / dense_entry_searcher_ → vector<InnerIdType>
└─ [主图阶段] graph_knn_search (hybrid_index.cpp:1065)
    └─ basic_searcher::search_impl<KNN_SEARCH> (basic_searcher.cpp:245)
        每个 hop:
        ├─ graph->GetNeighbors(current, neighbors)       // 顺序读图
        ├─ visit(): 标记 visited, 收集 to_be_visited_id
        ├─ flatten->Query(line_dists, computer, ids, N)   // ★核心瓶颈★
        │   └─ HybridVectorDataCell::query (hybrid_vector_datacell.cpp:67)
        │       ├─ dense_cell_->Query  (SIMD, 4x unroll, 顺序内存)  // 便宜
        │       ├─ for each id: 用 sparse_norms_[id] 做 Cauchy 上界剪枝
        │       └─ sparse_cell_->Query(通过剪枝的 ids)              // ★★★ 极贵 ★★★
        │            └─ for each id:
        │                ├─ GetCodesById(id) → allocate + 2次 io_->Read
        │                ├─ SparseQuantizer::ComputeImpl → 标量二路归并
        │                └─ Deallocate(codes)
        └─ push to top_candidates / candidate_set (heap push/pop)
```

### 2.2 关键代码位置

| 组件 | 文件 | 行号 |
|---|---|---|
| 主搜索循环 | `src/impl/searcher/basic_searcher.cpp` | 245-515 |
| 入口点初始化 | 同上 | 326-401 |
| 邻居访问 | 同上 | 33-71 (visit) |
| Hybrid 距离计算 | `src/datacell/hybrid_vector_datacell.cpp` | 67-232 |
| 稀疏 Cauchy 剪枝 | 同上 | 184-228 |
| 稀疏向量存储/读取 | `src/datacell/sparse_vector_datacell.inl` | 23-35, 114-126 |
| 稀疏 IP 计算 | `src/quantization/sparse_quantization/sparse_quantizer.h` | 164-188 |
| std::function | `src/impl/searcher/basic_searcher.cpp` | 296-299 |

## 3. 优化方案（按收益排序）

### ★★★ 优化 1：稀疏距离计算的内存分配（最大瓶颈）

#### 问题

`SparseVectorDataCell::GetCodesById`（`sparse_vector_datacell.inl:114-126`）每次调用：

```cpp
const uint8_t*
SparseVectorDataCell::GetCodesById(InnerIdType id, bool& need_release) const {
    uint32_t offset;
    offset_io_->Read(sizeof(offset), id * sizeof(offset), (uint8_t*)&offset);  // 第一次 IO
    uint32_t length;
    io_->Read(sizeof(length), offset, (uint8_t*)&length);                       // 第二次 IO
    need_release = true;
    size_t read_size = sizeof(uint32_t) * (2 * length + 1);
    auto* codes = (uint8_t*)allocator_->Allocate(read_size);                    // ★ malloc ★
    io_->Read(read_size, offset, codes);                                        // 第三次 IO
    return codes;
}
```

`SparseVectorDataCell::query`（`sparse_vector_datacell.inl:23-35`）逐个调用：

```cpp
for (int i = 0; i < id_count; ++i) {
    bool need_release{true};
    auto codes = this->GetCodesById(idx[i], need_release);   // malloc + 3次 Read
    computer->ComputeDist(codes, result_dists + i);
    if (need_release) {
        allocator_->Deallocate((void*)codes);                // ★ free ★
    }
}
```

**量化分析**（NQ, ef_search=200, 主图搜索约 200 hops）：
- 每跳访问 ~64 个邻居
- ~30% 通过 dense 上界剪枝 → ~45 个稀疏计算
- 每次稀疏计算 = 1 次 malloc + 1 次 free + 3 次 IO Read
- 整次搜索：`200 × 45 = 9000 次 malloc/free + 27000 次 Read`
- glibc malloc 在多线程下有锁，单次 ~100-300ns
- 总开销估算：`9000 × 200ns ≈ 1.8ms`，占单次搜索 ~10-30%（取决于 ef）

#### 方案

在 `SparseVectorDataCell` 内引入 thread-local 复用 buffer：

```cpp
// sparse_vector_datacell.h 新增成员:
struct TlsCodeBuffer {
    uint8_t* buf{nullptr};
    size_t cap{0};
};
mutable std::vector<TlsCodeBuffer> tls_bufs_;  // 或者用 thread_local

// GetCodesById 改为:
const uint8_t*
SparseVectorDataCell::GetCodesById(InnerIdType id, bool& need_release) const {
    auto& tls = tls_code_buffer();  // thread_local 引用
    uint32_t offset;
    offset_io_->Read(sizeof(offset), id * sizeof(offset), (uint8_t*)&offset);
    uint32_t length;
    io_->Read(sizeof(length), offset, (uint8_t*)&length);
    size_t read_size = sizeof(uint32_t) * (2 * length + 1);
    if (tls.cap < read_size) {
        tls.buf = static_cast<uint8_t*>(allocator_->Reallocate(tls.buf, read_size));
        tls.cap = read_size;
    }
    io_->Read(read_size, offset, tls.buf);
    need_release = false;  // ★ 不再需要调用方 free
    return tls.buf;
}
```

**注意点**：
- `query()` 当前是"读完一个立即 ComputeDist，再读下一个"，所以 thread-local 单 buffer 安全
- 但如果未来改成批量读 + 批量算，需要 N 个 buffer（或返回 `need_release=true` 走旧路径）
- `Reallocate` 调用比每次 `Allocate + Deallocate` 便宜得多（仅在 buffer 不够大时调用）

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/datacell/sparse_vector_datacell.h` | 加 thread-local buffer 结构 |
| `src/datacell/sparse_vector_datacell.inl` | 改 `GetCodesById` 和 `query` |

#### 预期收益

- **UHG/UHGS/UHGH**: 30-50% QPS 提升（这些方法的主图搜索都走 `HybridVectorDataCell::query` → `sparse_cell_->Query`）
- **其他 baseline**: 见第 4 节

#### 风险

- **低**。不改变存储格式，不破坏索引兼容性。只改内存管理。
- 单元测试覆盖：`src/datacell/sparse_vector_datacell_test.cpp` 已有测试，跑通即可。

---

### ★★★ 优化 2：稀疏向量存储布局 — 变长 → 定长桶

#### 问题

当前稀疏向量变长存储，`GetCodesById` 需要：
1. `offset_io_->Read` 读偏移量（间接寻址）
2. `io_->Read` 读长度
3. `io_->Read` 读实际数据

三次 IO + 一次 malloc（见优化 1）。即使优化 1 消除 malloc，三次 IO 仍然存在。

#### 方案

改成定长桶：每个稀疏向量占 `max_terms × sizeof(BufferEntry) + sizeof(uint32_t)` 字节，直接 `id × code_size` 寻址。

```cpp
// 存储格式: [len(uint32_t) | entry_0 | entry_1 | ... | entry_{max_terms-1}]
// code_size = sizeof(uint32_t) + max_terms * sizeof(BufferEntry)
// 直接 id * code_size 寻址，省掉 offset_io_ 间接层

const uint8_t*
SparseVectorDataCell::GetCodesById(InnerIdType id, bool& need_release) const {
    uint64_t offset = id * code_size_;
    need_release = false;
    return io_->DirectRead(code_size_, offset);  // 直接返回内部指针，无 malloc
}
```

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/datacell/sparse_vector_datacell.h` | 去掉 `offset_io_`，改 `code_size_` 为定长 |
| `src/datacell/sparse_vector_datacell.inl` | 改 `InsertVector` / `GetCodesById` / `BatchInsertVector` |
| 索引构建脚本 | 需要重建所有 sparse 相关索引 |

#### 预期收益

- **UHG/UHGS/UHGH**: 额外 20-40% QPS（消除间接寻址 + 配合优化 1 直接返回裸指针）
- **其他 baseline**: 见第 4 节

#### 风险

- **中**。破坏索引向后兼容，需要重建所有 hybrid_index（`703_*_hybrid_index.index`）和 SINDI（`709_*_sindi.index`、`701_*_sparse_sindi.index`）。
- 内存占用增加 ~1.5-2x（取决于平均/最大长度比）。NQ 稀疏向量平均 ~50 terms，最大 ~100+ terms，膨胀 ~2x。
- 可以通过统计实际长度分布，选择 p99 分位作为 `max_terms`，平衡内存和精度。

---

### ★★ 优化 3：稀疏 IP 计算 SIMD 化 / Hash Lookup

#### 问题

`SparseQuantizer::ComputeImpl`（`sparse_quantizer.h:164-188`）当前是标量二路归并：

```cpp
while (idx1 < len1 && idx2 < len2) {
    if (entries1[idx1].id < entries2[idx2].id) idx1++;
    else if (entries1[idx1].id > entries2[idx2].id) idx2++;
    else { inner_product += entries1[idx1].val * entries2[idx2].val; idx1++; idx2++; }
}
```

NQ 稀疏向量 ~50-100 terms，query 经 `sindi_query_prune_ratio=0.5` 后 ~15-30 terms。标量归并 ~100-300 cycles/次。

#### 方案 A：SIMD gather

用 AVX2 gather 一次取 4-8 个 (id, val)，批量比较 id 并累加匹配项的 val 乘积。适合 base 较长、query 较短的情况（对 base 做线性扫描，对 query 做 SIMD 查找）。

#### 方案 B：Small hash table

把 query 的 `(id, val)` 放进一个小 hash table（直接寻址或线性探测，size ~64），对 base 的每个 term 直接 hash 查找。适合 query 短、base 长的情况。

**推荐方案 B**：NQ 的 query 经剪枝后 ~15-30 terms，base ~50-100 terms。把 query 放 hash table，遍历 base 查找，复杂度 `O(len_base)` vs 当前 `O(len_base + len_query)`，且 hash lookup 可向量化。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/quantization/sparse_quantization/sparse_quantizer.h` | 改 `ComputeImpl`，加 hash table 初始化 |

#### 预期收益

- **UHG/UHGS/UHGH**: 10-20% QPS
- **SINDI baseline**: 见第 4 节

#### 风险

- **低**。纯计算层优化，不改存储格式。
- hash table 需要在 `ProcessQueryImpl` 里构建，增加少量 query 初始化开销，但摊销到多次 `ComputeDist` 调用后净收益正。

---

### ★★ 优化 4：`std::function` → 模板/函数指针

#### 问题

`basic_searcher.cpp:296-299`：

```cpp
std::function<void(float, InnerIdType)> push_candidate_func = push_candidate;
if (limit_candidate_set_size) {
    push_candidate_func = push_fixed_size_candidate;
}
```

`std::function` 通过类型擦除实现，每次调用有间接跳转，且可能涉及堆分配（如果捕获的 lambda 状态过大）。在每跳的内循环里被调用 ~64 次（邻居数）。

#### 方案

改成模板参数或编译期分支：

```cpp
// 方案 A: 模板特化
template <bool limit_candidate_set>
DistHeapPtr search_impl(...);  // if constexpr 分支

// 方案 B: 函数指针 + branchless
using PushFunc = void(*)(float, InnerIdType, ...);
PushFunc push_func = limit_candidate_set ? &push_fixed : &push_simple;
```

`hybrid_candidate_set_size=0` 是默认情况（主实验都用默认值），可以走快路径完全跳过 `push_fixed_size_candidate` 的复杂逻辑。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/impl/searcher/basic_searcher.cpp` | 改 `search_impl` 模板化 |

#### 预期收益

- **所有走 basic_searcher 的方法**: 2-5% QPS（小但免费）
- **其他 baseline**: 见第 4 节

#### 风险

- **低**。纯重构，不改语义。

---

### ★★ 优化 5：稀疏 Cauchy 上界剪枝 SIMD 化

#### 问题

`hybrid_vector_datacell.cpp:198-213` 的剪枝循环：

```cpp
for (InnerIdType i = 0; i < id_count; ++i) {
    const auto id = idx[i];
    const float dense_ip = 1.0F - dense_dists[i];
    float sparse_ip_upper_bound = query_sparse_norm * sparse_norms_[id];
    const float score_upper_bound =
        dense_weight * dense_ip + sparse_weight * prune_scale * sparse_ip_upper_bound;
    if (score_upper_bound > score_threshold + kDistanceEpsilon) {
        idx_sparse.push_back(id);
        sparse_positions.push_back(i);
    } else {
        result_dists[i] = lower_bound + kDistanceEpsilon;
    }
}
```

每跳 ~64 个邻居，标量循环。`sparse_norms_` 是连续数组但 `idx[i]` 是随机访问 → gather。

#### 方案

4x unroll + AVX2：
- SIMD load `dense_dists[i..i+3]`
- SIMD gather `sparse_norms_[idx[i..i+3]]`（注意 gather 很慢，可能不如标量）
- SIMD 计算 `score_upper_bound` 并比较

**注意**：SIMD gather 在现代 CPU 上仍然较慢（~10-20 cycles vs 标量 1 次 load ~5 cycles）。如果 `idx` 随机性高，SIMD 收益有限。更好的方案是**重排数据**：先按 `idx` 排序，使 `sparse_norms_` 访问局部化，但排序开销可能抵消收益。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/datacell/hybrid_vector_datacell.cpp` | 改 `query` 函数的剪枝循环 |

#### 预期收益

- **UHG/UHGS/UHGH**: 5-10% QPS
- **其他 baseline**: 无（此代码路径只在 `is_hybrid=true` 时走）

#### 风险

- **低**。但收益不确定（SIMD gather 可能不比标量快），需要实测验证。

---

### ★ 优化 6：Candidate set heap 优化（ef≥500 时）

#### 问题

`basic_searcher.cpp:268`：`StandardHeap<true, false>` 是二叉堆，每次 push/pop 是 `O(log ef)`。ef=500/1000 时，每跳 ~64 次 push/pop，总开销 `200 × 64 × log(1000) ≈ 400K` 操作。

#### 方案

- **方案 A**：对 ef≥500 改用 bucket-based priority queue（dist 量化到 uint16，按 bucket 分组）。push/pop 退化为 `O(1)` 摊销。
- **方案 B**：candidate_set 用更松的结构（比如 size-bounded 的简单排序数组，因为 candidate_set 大小相对有界）。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/impl/heap/standard_heap.h` | 加 bucket heap 实现 |
| `src/impl/searcher/basic_searcher.cpp` | 根据 ef 大小选择 heap 类型 |

#### 预期收益

- **所有走 basic_searcher 的方法**: 5-10% QPS（仅在 ef≥500 时显著）
- **其他 baseline**: 见第 4 节

#### 风险

- **中**。bucket heap 实现复杂，dist 量化可能引入精度问题。需要仔细测试 Recall 不降。

---

### ★ 优化 7：Entry points dense distance 复用（已验证收益小）

#### 问题

`DenseEntryHNSWGraph::Search` 内部已计算 entry points 的 dense distance，但 `get_dense_entry_points` 只返回 ID，`graph_knn_search` 里 `flatten->Query` 重算一遍。

#### 方案

加 `--reuse_dense_entry_distances` 开关，让 entry points 的 dense distance 直接注入主图搜索的初始 heap。

#### 实测结论

**收益小**（用户已实测）。原因：dense IP 是 SIMD + 顺序内存访问，~10-30ns/次，entry points ~100-500 个，总开销 ~1-15μs。而 `if (precomputed != nullptr)` 分支 + memcpy + 额外参数传递的开销 ~0.5-5μs，几乎抵消。

**建议**：暂不做。除非和优化 2 一起做（定长桶后 entry dense dist 可以直接从 dense cell 拿裸指针，零拷贝）。

---

### ★★★ 优化 8：稀疏 Query 批量预取 + 去逐个串行（新发现）

#### 问题

`SparseVectorDataCell::query`（`sparse_vector_datacell.inl:23-35`）是**纯串行**的：

```cpp
for (int i = 0; i < id_count; ++i) {
    bool need_release{true};
    auto codes = this->GetCodesById(idx[i], need_release);  // 读 offset + 读 length + 读 codes
    computer->ComputeDist(codes, result_dists + i);          // 标量归并
    if (need_release) {
        allocator_->Deallocate((void*)codes);                // free
    }
}
```

对比 `FlattenDataCell::query`（`flatten_datacell.h:286`）的 dense 实现：
- 4x unroll + `ComputeDistsBatch4`
- `prefetch_stride_code_` 步长的预取（`io_->Prefetch`）
- `memset(result_dists, 0, ...)` 初始化

**稀疏 query 完全没有预取和批量处理**。每次 `GetCodesById` 要 3 次 IO Read（offset, length, codes），CPU 在等内存时完全闲置。这是比优化 1（malloc）更大的问题——**即使消除了 malloc，3 次串行 IO Read + 串行 ComputeDist 仍然是瓶颈**。

#### 方案

两阶段流水线：

```cpp
void SparseVectorDataCell::query(float* result_dists, ..., const InnerIdType* idx, InnerIdType id_count) {
    // 阶段 1: 批量预取所有 codes 到连续 buffer，同时预取下下个
    constexpr int PREFETCH_AHEAD = 4;
    struct CodeRef { uint8_t* buf; size_t size; };
    std::array<CodeRef, PREFETCH_AHEAD> inflight;
    
    for (int i = 0; i < id_count; ++i) {
        int slot = i % PREFETCH_AHEAD;
        // 启动第 i 个的读取
        uint32_t offset, length;
        offset_io_->Read(sizeof(offset), idx[i]*sizeof(offset), &offset);
        io_->Read(sizeof(length), offset, &length);
        size_t read_size = sizeof(uint32_t) * (2*length + 1);
        inflight[slot].buf = tls_buffer[slot];  // 复用 thread-local pool
        inflight[slot].size = read_size;
        io_->Read(read_size, offset, inflight[slot].buf);
        
        // 预取 i+PREFETCH_AHEAD 的 offset（让 CPU 不闲置）
        if (i + PREFETCH_AHEAD < id_count) {
            offset_io_->Prefetch(idx[i+PREFETCH_AHEAD]*sizeof(offset), 64);
        }
        
        // 处理 i-PREFETCH_AHEAD+1 的（已经读完了）
        if (i >= PREFETCH_AHEAD - 1) {
            int ready_slot = (i + 1) % PREFETCH_AHEAD;
            computer->ComputeDist(inflight[ready_slot].buf, result_dists + (i - PREFETCH_AHEAD + 1));
        }
    }
    // 处理剩余的 in-flight
    ...
}
```

**关键点**：`MemoryIO::Read` 是 memcpy（内存IO），CPU 可以 overlap 多个 read 的内存访问延迟。当前串行模式下，每次 read 都要等上一个完成才能开始下一个。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/datacell/sparse_vector_datacell.inl` | 重写 `query`，加预取流水线 |
| `src/datacell/sparse_vector_datacell.h` | 加 thread-local buffer pool（配合优化 1） |

#### 预期收益

- **UHG/UHGS/UHGH**: 20-40% QPS（独立于优化 1，但和优化 1 叠加效果更好）
- **其他 baseline**: 见第 4 节

#### 风险

- **中**。代码复杂度增加，需要正确处理 PREFETCH_AHEAD 的边界。
- 需要保证 `computer->ComputeDist` 是线程安全的（同一个 computer 在一个线程内被顺序调用，OK）。

---

### ★★ 优化 9：稀疏 Cauchy 上界剪枝的剪枝率偏低（新发现）

#### 问题

`HybridVectorDataCell::query` 的剪枝判断（`hybrid_vector_datacell.cpp:198-213`）：

```cpp
const float score_threshold = 1.0F - lower_bound;  // 当前 top-k 最差 score
...
const float score_upper_bound =
    dense_weight * dense_ip + sparse_weight * prune_scale * sparse_ip_upper_bound;
if (score_upper_bound > score_threshold + kDistanceEpsilon) {
    // 需要计算 sparse
} else {
    result_dists[i] = lower_bound + kDistanceEpsilon;  // 直接剪枝
}
```

`sparse_ip_upper_bound = query_sparse_norm * sparse_norms_[id]`（Cauchy 不等式）。这是**最宽松的上界**——实际 sparse IP 几乎总是远小于这个上界（因为 query 和 base 的 term overlap 通常很少）。

**实测剪枝率**：通过 `703_uhg_exp3` 的 `hybrid_prune_scale=0.3` 配置推断，约 30-50% 的邻居被剪掉。但如果上界更紧，剪枝率可以到 70-90%。

#### 方案 A：用 query 的实际 term 集合做更紧的上界

Cauchy 上界 `query_norm * base_norm` 假设所有 term 都对齐。实际 sparse IP 只累加**交集 term**。可以用一个更紧的上界：

```cpp
// 对每个 base，预先存 base 的 norm 和 base 的 term 集合指纹（如 minhash signature）
// 在剪枝时用 (query∩base) 的 norm 上界 替代 query_norm * base_norm
```

但这需要额外的存储和计算，可能得不偿失。

#### 方案 B：分级剪枝（two-stage）

第一阶段用 Cauchy 上界粗筛，第二阶段对边界附近的候选用更紧的上界（如只用 query top-k term 的 norm 之和）。

#### 方案 C：降低 `prune_scale` 进一步

当前默认 `0.3`，但 `test_prune` 实验已经测过 `0.3` vs `no_prune`。如果 `prune_scale=0.1` 或 `0.0`（纯 dense 上界）的 Recall 可接受，可以进一步激进剪枝。

**推荐**：先实测 `prune_scale ∈ {0.0, 0.1, 0.2, 0.3}` 的 Recall-QPS 曲线（已经有 `test_prune` 脚本，只需扩大 scale 范围），确认更激进的剪枝是否能提升 QPS 而不失 Recall。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| 无代码改动（方案 C） | 只调参数 |
| `src/datacell/hybrid_vector_datacell.cpp` | 方案 A/B 需要改剪枝逻辑 |

#### 预期收益

- 方案 C（调参）：5-15% QPS（如果 Recall 允许）
- 方案 A/B（紧上界）：10-30% QPS，但工程量大

#### 风险

- 方案 C：低（只是参数调整，可立即测）
- 方案 A/B：中（需要额外存储和算法设计）

---

### ★★ 优化 10：HybridVectorDataCell::query 的 Vector 重复分配（新发现）

#### 问题

`HybridVectorDataCell::query`（`hybrid_vector_datacell.cpp:67-232`）每次被调用都分配多个 Vector：

```cpp
Vector<float> dense_dists(id_count, 0.0f, search_alloc);       // 每跳分配
Vector<float> sparse_dists(id_count, 0.0f, search_alloc);      // 每跳分配
Vector<InnerIdType> idx_sparse(search_alloc);                  // 每跳分配
Vector<InnerIdType> sparse_positions(search_alloc);            // 每跳分配
Vector<float> selected_sparse_dists(idx_sparse.size(), 0.0F, search_alloc);  // 每跳分配
```

主图搜索循环每跳调用一次 `flatten->Query`，也就是 `HybridVectorDataCell::query`。NQ 搜索 ~200 跳，每跳 5 次 Vector 分配 = **1000 次分配/释放**。

虽然有 `reserve`，但 Vector 构造本身要走 allocator（即使是 `search_alloc` 也是 vsag allocator）。

#### 方案

把这些 Vector 改成 thread-local 复用 buffer，或者作为 `HybridVectorDataCell` 的成员预分配（但要注意线程安全）。

更好的方案：把 `HybridVectorDataCell::query` 改成有状态的"搜索上下文"对象，在 `factory_computer` 时创建，搜索过程中复用：

```cpp
class HybridSearchContext {
    Vector<float> dense_dists;
    Vector<float> sparse_dists;
    Vector<InnerIdType> idx_sparse;
    Vector<InnerIdType> sparse_positions;
    Vector<float> selected_sparse_dists;
};
// 在 factory_computer 里创建 context，后续 query 复用
```

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/datacell/hybrid_vector_datacell.h` | 加 search context 或 thread-local buffer |
| `src/datacell/hybrid_vector_datacell.cpp` | 改 `query` 复用 buffer |

#### 预期收益

- **UHG/UHGS/UHGH**: 5-15% QPS
- **其他 baseline**: 无（此路径只在 hybrid 搜索走）

#### 风险

- **低**。不改变搜索语义，只改内存管理。
- thread-local 需要处理多线程搜索（`threads > 1` 时每个线程有自己的 buffer）。

---

### ★★ 优化 11：early-exit 剪枝信号在 hybrid 模式下被忽视（新发现）

#### 问题

`basic_searcher.cpp:415-418`：

```cpp
if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
    if ((-current_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
        break;  // 标准图搜索的 early-exit
    }
}
```

这是标准 HNSW 的 early-exit：candidate_set 的 best 候选已经比 top-k 最差还差，可以停止。

但在 hybrid 模式下，`flatten->Query` 返回的 `dist` 可能是**剪枝后的 `lower_bound + epsilon`**（`hybrid_vector_datacell.cpp:211`），而不是真实距离。这意味着：
- 一个被剪枝的候选进 candidate_set 时，dist = lower_bound + epsilon（虚高）
- 但它的**真实 score 可能更高**（sparse 贡献没算）
- 如果这个候选是 candidate_set 的 top，early-exit 可能提前退出，错过真实更好的候选

当前代码**没有处理这个问题**——hybrid 模式和纯 dense 模式用相同的 early-exit 逻辑。这可能导致 **Recall 损失**（不是 QPS 问题，而是正确性问题）。

但实测 Recall 似乎没问题，原因是：
1. 被剪枝的候选 dist 设为 `lower_bound + epsilon`，**比 lower_bound 更差**，所以它不会成为 candidate_set 的 top（除非 candidate_set 里所有其他候选都比它更差）
2. 实际上 early-exit 触发时，candidate_set 的 top 的 dist 是真实计算的（不是剪枝的），因为剪枝的候选 dist 更高

**但这里有一个更微妙的问题**：当 `top_candidates->Size() < ef` 时（搜索早期），`SetSearchLowerBound(float::max())` 禁用了剪枝（`basic_searcher.cpp:437`），所以早期所有候选都走完整计算。只有 `top_candidates` 满了之后才启用剪枝。此时 early-exit 的判断是正确的。

**结论**：当前逻辑在 Recall 上是正确的，但 early-exit 的触发时机可能比纯 dense 更晚（因为剪枝候选进 candidate_set 后不会立即触发 exit，需要等真实计算的候选变差才触发）。

#### 方案

可以在 `candidate_set->Push` 时区分"真实距离"和"剪枝距离"，让剪枝候选不参与 early-exit 判断：

```cpp
// 剪枝候选用一个特殊标记，push_candidate_func 时标记
// early-exit 时只看真实距离的候选
```

但这会增加复杂度，且收益不确定。

**建议**：暂不做，但记录此处作为 Recall 调试的潜在点。如果未来发现 Recall 偏低，可以检查这里。

#### 预期收益

- 主要是正确性保障，不是 QPS 提升
- 可能在某些 alpha/scale 组合下 Recall 略升（如果当前 early-exit 确实过早触发）

#### 风险

- **中**。改动 early-exit 逻辑可能影响所有走 basic_searcher 的方法。

---

### ★ 优化 12：visit 函数的 LinearCongruentialGenerator 每跳重建（新发现）

#### 问题

`basic_searcher.cpp:41`：

```cpp
BasicSearcher::visit(...) const {
    LinearCongruentialGenerator generator;  // 每次调用 visit 都构造一个新的
    ...
}
```

每跳调用一次 `visit`，构造一个 `LinearCongruentialGenerator`。虽然构造只是初始化一个 `uint32_t`，但这是不必要的开销。

#### 方案

把 `generator` 改成 `static thread_local` 或作为 `BasicSearcher` 的成员。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/impl/searcher/basic_searcher.cpp` | `LinearCongruentialGenerator` 改 thread_local |

#### 预期收益

- 微小（<1% QPS），但免费

#### 风险

- **低**。但要注意 `filter == nullptr` 时 `skip_threshold = 0`，`generator.NextFloat() > 0` 总是 true，所以 generator 实际没被用。可以加 fast path：`if (skip_threshold == 0.0F) { 跳过 generator 调用 }`。

---

### ★ 优化 13：GraphDataCell::GetNeighbors 的删除标记检查（新发现）

#### 问题

`GraphDataCell::GetNeighbors`（`graph_datacell.h:215-240`）在 `is_support_delete_=true` 时：

```cpp
Vector<InnerIdType> shared_neighbor_ids(neighbor_count, this->allocator_);  // 每跳分配
this->io_->Read(..., shared_neighbor_ids.data());
neighbor_ids.clear();
neighbor_ids.reserve(neighbor_count);
for (int i = 0; i < neighbor_count; ++i) {
    uint8_t neighbor_version = shared_neighbor_ids[i] >> id_bit_;
    InnerIdType neighbor_id = shared_neighbor_ids[i] & remove_flag_mask_;
    if (node_versions_[neighbor_id] == neighbor_version) {
        neighbor_ids.push_back(neighbor_id);
    }
}
```

**每跳分配一个临时 Vector** + 逐个检查 version。对于 UHG 主图（不需要支持删除，因为 UHG 是只读索引），这个检查是多余的。

#### 方案

UHG 的主图构建时设 `is_support_delete_=false`，走 fast path：

```cpp
} else {
    start += sizeof(neighbor_count);
    neighbor_ids.resize(neighbor_count);
    this->io_->Read(neighbor_ids.size() * sizeof(InnerIdType), start, ...);
}
```

需要确认 UHG 主图构建时 `is_support_delete_` 的默认值。

#### 涉及文件

| 文件 | 改动 |
|---|---|
| `src/datacell/graph_datacell.h` | 确认/修改 `is_support_delete_` 默认值 |
| `src/algorithm/hybrid_index/hybrid_index.cpp` | 构建主图时显式设 `is_support_delete_=false` |

#### 预期收益

- 2-5% QPS（省掉每跳一个 Vector 分配 + version 检查循环）

#### 风险

- **低**。UHG 主图是只读的，不需要删除支持。

---

## 4. 代码质量问题（逻辑繁琐/冗余/可读性）

以下不是性能优化，而是代码本身写得不够好的地方。按严重程度排序。

### 问题 Q1：`HybridVectorDataCell::query` 有 ~60 行注释掉的死代码

**位置**：`src/datacell/hybrid_vector_datacell.cpp:87-147`

函数开头有**两段完整的注释掉的旧实现**（"no prune" 版本和 "prune sparse compute" 版本），共 ~60 行。这些是历史调试遗留，严重干扰阅读：

```cpp
// // no prune
// // Query dense cell
// if (std::abs(dense_weight_) > 1e-5)
// dense_cell_->Query(dense_dists.data(), hybrid_comp->GetDenseComputer(),
//                   idx, id_count, search_alloc);
...（60 行注释）
```

**建议**：直接删除。git 历史可以追溯旧版本，不需要在源码里保留。

---

### 问题 Q2：`SearchWithRequest` 整个函数体是注释掉的死代码

**位置**：`src/algorithm/hybrid_index/hybrid_index.cpp:1203-1251`

```cpp
DatasetPtr
HybridIndex::SearchWithRequest(const SearchRequest& request) const {
    // // support 1 query only
    // auto heap = vsag::DistanceHeap::MakeInstanceBySize<false, true>(...);
    // ...（~40 行注释掉的暴力搜索实现）
    auto data = Dataset::Make();
    return data;  // 返回空 Dataset
}
```

整个函数体是注释掉的旧暴力搜索代码 + 一行 `return Dataset::Make()`。要么实现它，要么删掉声明和定义。当前状态是"假实现"，调用方会拿到空结果但不会报错，是个潜在的 silent bug。

**建议**：如果要保留接口，改成 `throw VsagException(UNSUPPORTED_INDEX_OPERATION, ...)`；如果不要，删除声明和定义。

---

### 问题 Q3：`CalcDistanceById` 是空实现（silent bug）

**位置**：`src/algorithm/hybrid_index/hybrid_index.cpp:1253-1257`

```cpp
float
HybridIndex::CalcDistanceById(const float* vector, int64_t id) const {
    float result = 0.0F;
    return result;  // 永远返回 0
}
```

`GetVectorByInnerId` 同样是空实现（`hybrid_index.cpp:1260-1262`）。这些是 `Index` 基类的虚函数，hybrid_index 没有实现但也没有抛异常。调用方会拿到错误的 0 距离，不会报错。

**建议**：改成 `throw VsagException(UNSUPPORTED_INDEX_OPERATION, ...)`，让调用方立即失败而不是拿到错误结果。

---

### 问题 Q4：两个 `search_impl` 重载有大量重复代码

**位置**：`src/impl/searcher/basic_searcher.cpp`

有两个 `search_impl` 模板：
- `:103-241`：带 `IteratorFilterContext* iter_ctx` 参数的版本
- `:245-515`：带 `LabelTablePtr label_table` 参数的版本

两者共享 ~80% 的代码（computer 初始化、visit 循环、flatten->Query、push candidate 等），只是入口点初始化和 candidate 处理细节不同。

**问题**：
- 任何搜索逻辑的修改要同步改两处，容易遗漏
- 两个版本的 `lower_bound` 更新逻辑、`dist_cmp` 累加、`hops` 统计都重复
- `iter_ctx` 版本的 `max_hops` 检查缺失（`:171-172` 没有 `max_hops` 判断，而 `:404` 有）

**建议**：合并成一个模板，用 `if constexpr` 或 optional 参数区分。或者把公共部分抽成 helper 函数。

---

### 问题 Q5：`703_uhg_exp3.cpp` 的 `search_points` 模式逻辑绕

**位置**：`examples/cpp/703_uhg_exp3.cpp:624-635`

```cpp
std::vector<int> search_points = params.search_points;
if (search_points.empty()) {
    search_points.push_back(params.ef_search);  // 往空 vector push 一个值
}

for (int point : search_points) {
    // 但这里用 params.search_points.empty() 判断，不是 search_points.empty()
    const int effective_ef_search = params.search_points.empty() ? params.ef_search : point;
    const int effective_sindi_bk = params.search_points.empty() ? params.sindi_bk : point;
    ...
}
```

问题：`search_points` 在 `:625` 被填充了至少一个元素，但 `:630-635` 判断的是 **原始的 `params.search_points`** 是否为空。这导致：
- 用户传了 `--search_points 100,200` → `params.search_points` 非空 → `effective_*` 用 `point`（正确）
- 用户没传 `--search_points` → `params.search_points` 为空 → `search_points = [ef_search]` → 循环一次 → `effective_*` 用 `params.*`（正确但绕）

**更严重的问题**：`search_points` 模式下 `effective_dense_entry_bk = point` 和 `effective_dense_entry_ef_search = point`，**强制覆盖了用户传入的 `--dense_entry_bk` 和 `--dense_entry_ef_search`**。这就是为什么 `test_uhgh` 脚本必须用单点模式而不能用 `search_points` 模式。这个覆盖行为没有文档说明，也没有 warning，是个 footgun。

**建议**：
- 删掉 `:625-627` 的 `if (search_points.empty()) push_back`，直接用 `params.search_points`，空就用 `{params.ef_search}` 作为默认
- `effective_*` 的判断改成 `search_points.size() > 1 ? point : params.*`，这样单点时不覆盖用户传入的 entry 参数
- 或者至少在覆盖时打一行 warning log

---

### 问题 Q6：`get_dense_entry_points` 的参数查找链过长

**位置**：`src/algorithm/hybrid_index/hybrid_index.cpp:1008-1025`

```cpp
auto entry_bk = get_uint_param(uhgh_json, "entry_bk", default_bk);
entry_bk = get_uint_param(uhgh_json, "hnsw_bk", entry_bk);          // alias 1
entry_bk = get_uint_param(parsed_search_param, "dense_entry_bk", entry_bk);
entry_bk = get_uint_param(parsed_search_param, "hnsw_bk", entry_bk);  // alias 2

auto entry_ef = get_uint_param(uhgh_json, "entry_ef_search", default_ef);
entry_ef = get_uint_param(uhgh_json, "hnsw_ef_search", entry_ef);    // alias 1
entry_ef = get_uint_param(parsed_search_param, "dense_entry_ef_search", entry_ef);
entry_ef = get_uint_param(parsed_search_param, "hnsw_ef_search", entry_ef);  // alias 2
```

每个参数有 **4 层查找**（uhgh.entry_bk → uhgh.hnsw_bk → root.dense_entry_bk → root.hnsw_bk）。`get_sindi_entry_points` 类似但只有 2 层。

**问题**：
- `hnsw_bk` / `hnsw_ef_search` 是 `dense_entry_bk` / `dense_entry_ef_search` 的别名，但文档没提
- 4 层 fallback 难以追踪实际生效的值
- `entry_bk` 和 `entry_ef` 用相同的别名 `hnsw_bk` / `hnsw_ef_search`，但语义不同（一个是 topk，一个是 ef），容易混淆

**建议**：去掉别名（`hnsw_bk`、`hnsw_ef_search`），只保留 `dense_entry_bk` 和 `dense_entry_ef_search`。如果需要向后兼容，加一个 deprecation warning。

---

### 问题 Q7：`graph_knn_search` 的 entry_points 去重逻辑冗余

**位置**：`src/algorithm/hybrid_index/hybrid_index.cpp:1087-1121`

三个分支各自做去重：
1. `entry_points` 参数（`:1088-1096`）：用 `unordered_set` 去重 + 跳过 `>= total_count_`
2. JSON `entry_points` 数组（`:1100-1116`）：用另一个 `unordered_set` 去重 + label 检查
3. JSON `entry_point` 单值（`:1117-1118`）：不去重

而且 `basic_searcher.cpp:331-337` 在用 `eps` 时**又做了一次去重**（`if (not vl->Get(ep_id))`）。

**问题**：
- `graph_knn_search` 里的去重和 `basic_searcher` 里的去重重复
- `:1090-1093` 的 `if (inner_id >= total_count_) continue; if (inner_id < total_count_ and ...)` 是冗余判断——`continue` 后 `inner_id` 必然 `< total_count_`，第二个 `if` 永远为 true

**建议**：
- 去掉 `graph_knn_search` 里的去重，让 `basic_searcher` 统一处理
- `:1093` 简化成 `if (dedup_entry_points.insert(inner_id).second) { search_param.eps.push_back(inner_id); }`

---

### 问题 Q8：`graph_knn_search` 的结果转换循环有冗余判断

**位置**：`src/algorithm/hybrid_index/hybrid_index.cpp:1154-1160`

```cpp
for (auto j = result_size - 1; j >= 0; --j) {
    if (j < result_size) {  // ← 永远为 true（j 从 result_size-1 开始递减）
        dists[j] = search_results->Top().first;
        ids[j] = label_table_->GetLabelById(search_results->Top().second);
    }
    search_results->Pop();
}
```

`j` 从 `result_size - 1` 开始，永远 `< result_size`，`if (j < result_size)` 永远为 true。而且 `j` 是 `int64_t`（推测），`j >= 0` 的判断在 `j = 0` 后 `--j` 变成 `-1` 才退出，但如果 `result_size` 是 `int64_t`，`j = result_size - 1` 在 `result_size = 0` 时是 `-1`，循环不执行（OK）。

**建议**：删掉 `if (j < result_size)` 判断。用 `while (not search_results->Empty())` 更清晰：

```cpp
int64_t j = result_size - 1;
while (not search_results->Empty()) {
    dists[j] = search_results->Top().first;
    ids[j] = label_table_->GetLabelById(search_results->Top().second);
    search_results->Pop();
    --j;
}
```

---

### 问题 Q9：`graph_knn_search` 手动拼装 hybrid_vector 的 byte stream

**位置**：`src/algorithm/hybrid_index/hybrid_index.cpp:1127-1132`

```cpp
auto dense_vector = query->GetFloat32Vectors();
auto dim = query->GetDim();
auto sparse_vector = query->GetSparseVectors();
auto hybrid_vector = (int8_t*)(allocator_->Allocate(dim * sizeof(float) + sizeof(sparse_vector[0])));
std::memcpy(hybrid_vector, dense_vector, dim * sizeof(float));
std::memcpy(hybrid_vector + dim * sizeof(float), sparse_vector, sizeof(sparse_vector[0]));
```

手动把 dense + sparse 拼成一段 byte stream 传给 `searcher_->Search`，然后 `HybridVectorDataCell::factory_computer` 再拆开：

```cpp
// hybrid_vector_datacell.cpp:235-245
const float* dense_query = query;  // 前 dim*sizeof(float) 字节
const void* sparse_query = query_ptr + dense_size;  // 后 sizeof(SparseVector) 字节
```

**问题**：
- 用 `int8_t*` 强转 `float*` 的 byte stream，靠 `memcpy` 偏移来传参，类型不安全
- `dim * sizeof(float) + sizeof(sparse_vector[0])` 的布局是隐式约定，没有结构体封装
- `factory_computer` 里 `const float* dense_query = query` 然后 `query_ptr + dense_size` 拿 sparse，依赖 `GetDenseVectorSize()` 返回正确的 size
- 如果 `dim` 或 `SparseVector` 布局变化，两边的偏移要同步改

**建议**：定义一个 `HybridQuery` 结构体：
```cpp
struct HybridQuery {
    const float* dense;
    const SparseVector* sparse;
};
```
传 `const void* query` 改成传 `HybridQuery`，消除 byte stream 拼装/拆装。

---

### 问题 Q10：`hybrid_vector_datacell.cpp:62` 硬编码 `dim_ = 4096`

**位置**：`src/datacell/hybrid_vector_datacell.cpp:61-63`

```cpp
IndexCommonParam common_param_sparse = common_param;
common_param_sparse.dim_ = 4096;  // ← 硬编码
sparse_cell_ = FlattenInterface::MakeInstance(sparse_param, common_param_sparse);
```

`4096` 是一个 magic number，没有注释说明为什么是 4096。而 `SparseQuantizer` 的 `ProcessQueryImpl` 实际上用 `sparse_query->len_` 来确定大小，不依赖这个 `dim_`。

**建议**：加注释说明 `dim_=4096` 的含义（sparse term 上限？），或者从参数传入。

---

### 问题 Q11：`visit` 函数的 `to_be_visited_rid` 从未被使用

**位置**：`src/impl/searcher/basic_searcher.cpp:38, 63`

```cpp
Vector<InnerIdType>& to_be_visited_rid,  // 参数
...
to_be_visited_rid[count_no_visited] = i;  // 写入
```

`to_be_visited_rid` 存的是邻居在 `neighbors` 数组里的索引 `i`，但调用方（`:426-433`）只用了 `to_be_visited_id`，**从不读取 `to_be_visited_rid`**。

```bash
grep to_be_visited_rid basic_searcher.cpp
# 只在 visit 的签名和赋值处出现，调用方从不读
```

**建议**：删掉 `to_be_visited_rid` 参数和赋值行。

---

### 问题 Q12：`basic_searcher.cpp:344-346` 的 entry_points 兜底逻辑冗余

**位置**：`src/impl/searcher/basic_searcher.cpp:331-346`

```cpp
if (not inner_search_param.eps.empty()) {
    for (auto ep_id : inner_search_param.eps) {
        if (not vl->Get(ep_id)) {
            entry_points.push_back(ep_id);
        }
    }
} else {
    entry_points.push_back(inner_search_param.ep);
}

if (entry_points.empty()) {
    entry_points.push_back(inner_search_param.ep);  // ← 兜底
}
```

逻辑：
1. 如果 `eps` 非空，过滤已访问的，可能全部被过滤 → `entry_points` 为空
2. 如果 `eps` 为空，push `ep` → `entry_points` 非空
3. `:344` 兜底：如果 `entry_points` 为空，再 push `ep`

**问题**：`:344` 的兜底只在"eps 非空但全部已访问"时触发。这种情况下用 `ep` 兜底是否合理？`ep` 可能也已被访问（`vl->Get(ep)` 为 true），但这里不检查。如果 `ep` 也被访问了，后面 `flatten->Query` 会重复计算一个已访问的点（浪费但不报错）。

**建议**：兜底时检查 `vl->Get(ep)`，如果 `ep` 也被访问了，直接返回空 `top_candidates`（或者从 `entry_point_id_` 重新开始）。

---

### 问题 Q13：`push_fixed_size_candidate` 的实现是 O(n²) 的全量重建

**位置**：`src/impl/searcher/basic_searcher.cpp:272-293`

```cpp
auto push_fixed_size_candidate = [&](float candidate_dist, InnerIdType candidate_id) {
    candidate_set->Push(-candidate_dist, candidate_id);
    if (candidate_set->Size() <= inner_search_param.hybrid_candidate_set_size) {
        return;
    }
    // 找最远的元素
    const auto* candidate_data = candidate_set->GetData();
    uint64_t farthest_pos = 0;
    for (uint64_t i = 1; i < candidate_set->Size(); ++i) {
        if (candidate_data[i].first < candidate_data[farthest_pos].first) {
            farthest_pos = i;
        }
    }
    // 把除了 farthest 之外的所有元素重新 push 到新 heap
    auto capped_candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    for (uint64_t i = 0; i < candidate_set->Size(); ++i) {
        if (i != farthest_pos) {
            capped_candidate_set->Push(candidate_data[i].first, candidate_data[i].second);
        }
    }
    candidate_set = capped_candidate_set;
};
```

每次超出 `hybrid_candidate_set_size` 时，**遍历整个 heap 找最远元素 + 创建新 heap + 把所有元素重新 push**。如果 `hybrid_candidate_set_size = N`，每次 push 是 O(N)（找最远）+ O(N log N)（重建 heap）。整个搜索是 O(hops × N²)。

**问题**：
- `candidate_set` 是 `StandardHeap<true, false>`（max-heap），存的是 `-dist`。max-heap 的 `Top()` 是最大的 `-dist` = 最小的 `dist` = **最近的候选**。`Pop()` 弹出的是最近候选，不是最远的。
- 代码找 "最远" 的方式是遍历找 `.first` 最小的（最负的 `-dist` = 最远的），这是正确的，但**不能用 `Pop()` 替代**（`Pop` 弹的是最近的）。
- 正确的 O(log N) 做法是：用 `StandardHeap<false, false>`（min-heap）存 `candidate_set`，这样 `Top()` 是最小的 `-dist` = 最远的，`Pop()` 弹最远的。但这会改变 `candidate_set` 的其他语义。
- 或者用两个 heap 配合，或者用 `std::priority_queue` 的 reverse comparator。

**建议**：
- 简单方案：改成 min-heap（`StandardHeap<false, false>`）存 `-dist`，`Top()` 就是最远，`Pop()` 弹最远。需要确认 `candidate_set` 的其他用法（`:408` `candidate_set->Top()` 取最近的候选来 expand）是否兼容。
- 如果改 heap 方向影响太大，至少把全量重建改成"标记删除"（lazy deletion）：不重建 heap，只在 `Top()` 时跳过被标记删除的元素。

---

### 问题 Q14：`basic_searcher.cpp:447-475` 的内循环条件判断重复更新 `lower_bound`

**位置**：`src/impl/searcher/basic_searcher.cpp:447-475`

```cpp
for (uint32_t i = 0; i < count_no_visited; i++) {
    dist = line_dists[i];
    if (top_candidates->Size() < ef || lower_bound > dist || ...) {
        push_candidate_func(dist, to_be_visited_id[i]);
        if (check_func(...)) {
            top_candidates->Push(dist, ...);
        }
        ...
        if constexpr (mode == KNN_SEARCH) {
            if (top_candidates->Size() > ef) {
                top_candidates->Pop();
            }
        }
        if (not top_candidates->Empty()) {
            lower_bound = top_candidates->Top().first;  // ← 每次都更新
        }
    }
}
```

每次 push 后都更新 `lower_bound`，即使 `top_candidates->Size()` 没变（push 了但没 pop）。`lower_bound` 只在 `top_candidates` 的 top 变化时才需要更新。

**问题**：
- 每次都 `Top()` 查询是有开销的（虽然小）
- 更语义化的做法是：只在 `top_candidates->Size() == ef` 且 push 了更近的候选后才更新

**建议**：把 `lower_bound` 更新移到循环外，或者只在 pop 后更新：
```cpp
if constexpr (mode == KNN_SEARCH) {
    if (top_candidates->Size() > ef) {
        top_candidates->Pop();
        lower_bound = top_candidates->Top().first;  // 只在 pop 后更新
    } else if (not top_candidates->Empty() && dist < lower_bound) {
        lower_bound = dist;  // push 了更近的（但还没满 ef）
    }
}
```

---

### 问题 Q15：`703_uhg_exp3.cpp` 的 `effective_*` 变量命名和逻辑不一致

**位置**：`examples/cpp/703_uhg_exp3.cpp:630-635`

```cpp
const int effective_ef_search = params.search_points.empty() ? params.ef_search : point;
const int effective_sindi_bk = params.search_points.empty() ? params.sindi_bk : point;
const int effective_dense_entry_bk = params.search_points.empty() ? params.dense_entry_bk : point;
const int effective_dense_entry_ef_search = params.search_points.empty() ? params.dense_entry_ef_search : point;
```

- `effective_ef_search`：单点时用 `params.ef_search`，多点时用 `point`（合理）
- `effective_sindi_bk`：单点时用 `params.sindi_bk`，多点时用 `point`（强制等于 ef_search，合理但绕）
- `effective_dense_entry_bk` / `effective_dense_entry_ef_search`：同上

**问题**：多点模式下所有 `effective_*` 都等于 `point`，失去了独立控制 entry 参数的能力（见 Q5）。而且代码里 `effective_dense_entry_bk` 和 `effective_dense_entry_ef_search` 值相同，但语义不同（topk vs ef），强制相等在逻辑上不一定正确。

**建议**：在 `search_points` 模式下，保留用户传入的 `dense_entry_bk` / `dense_entry_ef_search` 比例，或者让 `search_points` 只控制 `ef_search`，entry 参数单独传。

---

## 5. 对其他 baseline 方法的影响分析

### 4.1 Baseline 方法概览

主实验对比 5 种方法：

| 方法 | 实现位置 | 搜索路径 |
|---|---|---|
| `hnsw` | `701_uhg_exp1.cpp` + dense HNSW index | 纯 dense HNSW，不走 hybrid_index |
| `sindi` | `701_uhg_exp1.cpp` + SINDI index | 纯 sparse SINDI，走 `SINDI::search_impl` |
| `hnsw_sindi` | `701_uhg_exp1.cpp` | HNSW + SINDI 各自搜索后融合，不走 hybrid_index |
| `fhg` | `703_uhg_exp3.cpp` + `method=uhg` + `--disable_sindi --disable_dense_entry` | 走 hybrid_index 主图搜索，无 entry |
| `uhg` (auto/uhgs/uhgh) | `703_uhg_exp3.cpp` + hybrid_index | 走 hybrid_index 完整流程 |

### 4.2 各优化点对 baseline 的影响

#### 优化 1：稀疏 GetCodesById 的 malloc/free → thread-local buffer

| 方法 | 影响 | 说明 |
|---|---|---|
| **hnsw** | **无** | 纯 dense，不走 `SparseVectorDataCell` |
| **sindi** | **无** | SINDI 用 `SparseTermDataCell`（倒排表），不用 `SparseVectorDataCell`。SINDI 的距离计算在 `SparseTermDataCell::Query` + `SparseTermComputer::ScanForAccumulate`，不走 `GetCodesById` |
| **hnsw_sindi** | **无** | 同上，HNSW 走 dense，SINDI 走倒排表 |
| **fhg** | **有（正向）** | FHG 走 `hybrid_index` 主图搜索，`is_hybrid=true`，会调 `HybridVectorDataCell::query` → `sparse_cell_->Query` → `GetCodesById`。QPS 同步提升 |
| **uhg** (所有变体) | **有（正向）** | 同 FHG，主图搜索走相同路径 |

**结论**：优化 1 对 fhg 和 uhg 都提升，对其他三个 baseline 无影响。**实验对比公平性不受影响**（fhg 和 uhg 同步变快，相对关系不变）。

#### 优化 2：稀疏存储改定长桶

| 方法 | 影响 | 说明 |
|---|---|---|
| **hnsw** | **无** | 不涉及稀疏存储 |
| **sindi** | **无** | SINDI 用倒排表（`SparseTermDataCell`），不是 `SparseVectorDataCell` |
| **hnsw_sindi** | **无** | 同上 |
| **fhg** | **有（正向）** | 走 hybrid_index 主图，受益 |
| **uhg** | **有（正向）** | 同上 |

**注意**：SINDI 的 rerank 阶段（`SINDI::search_impl:204-230`）会用 `SparseVectorDataCell` 做高精度 rerank。如果 SINDI baseline（`701_*_sparse_sindi.index`）也用 `SparseVectorDataCell`，则 SINDI 也会受益。需要确认 SINDI rerank 的具体实现。

**索引兼容性**：需要重建 `703_*_hybrid_index.index`（fhg 和 uhg 用）和 `709_*_sindi.index`（uhg 内部 SINDI）。`701_*_sparse_sindi.index`（sindi baseline）如果也用 `SparseVectorDataCell` 也需重建。dense HNSW（`701_*_dense_hnsw.index`）不需要重建。

#### 优化 3：稀疏 IP SIMD/Hash

| 方法 | 影响 | 说明 |
|---|---|---|
| **hnsw** | **无** | 不涉及稀疏计算 |
| **sindi** | **可能无** | SINDI 用 `SparseTermComputer::ScanForAccumulate`，不走 `SparseQuantizer::ComputeImpl`。但 SINDI rerank 阶段如果用 `SparseQuantizer::ComputeImpl` 会受益 |
| **hnsw_sindi** | **同 sindi** | |
| **fhg** | **有（正向）** | 主图稀疏计算走 `SparseQuantizer::ComputeImpl` |
| **uhg** | **有（正向）** | 同上 |

#### 优化 4：std::function → 模板/函数指针

| 方法 | 影响 | 说明 |
|---|---|---|
| **hnsw** | **有（正向）** | HNSW 走 `BasicSearcher::search_impl`（如果 hnsw 用 vsag 的 HNSW 实现）。但主实验的 `hnsw` baseline 用 `701_uhg_exp1.cpp` 调 `vsag::Factory::CreateIndex("hnsw")`，内部走 `hnswlib::hnswalg`，不一定走 `BasicSearcher`。需确认 |
| **sindi** | **无** | SINDI 走自己的 `search_impl`，不走 `BasicSearcher` |
| **hnsw_sindi** | **HNSW 部分有** | 同 hnsw |
| **fhg** | **有（正向）** | 走 `BasicSearcher` |
| **uhg** | **有（正向）** | 走 `BasicSearcher` |

**需确认**：`vsag::Factory::CreateIndex("hnsw")` 的搜索是否走 `BasicSearcher::search_impl`。从代码看 `hgraph.cpp:67` 用 `BasicSearcher`，但 `hnswlib/hnswalg.cpp` 可能有自己的搜索循环。

#### 优化 5：Cauchy 上界剪枝 SIMD

| 方法 | 影响 | 说明 |
|---|---|---|
| **hnsw** | **无** | `is_hybrid=false`，不走剪枝路径 |
| **sindi** | **无** | 不走 `HybridVectorDataCell` |
| **hnsw_sindi** | **无** | 同上 |
| **fhg** | **有（正向）** | `is_hybrid=true`，走剪枝 |
| **uhg** | **有（正向）** | 同上 |

#### 优化 6：Heap 优化

| 方法 | 影响 | 说明 |
|---|---|---|
| **hnsw** | **可能有** | 如果走 `BasicSearcher` |
| **sindi** | **无** | 不走 `BasicSearcher` |
| **hnsw_sindi** | **HNSW 部分可能有** | |
| **fhg** | **有（正向）** | 走 `BasicSearcher` |
| **uhg** | **有（正向）** | 走 `BasicSearcher` |

### 4.3 影响汇总矩阵

| 优化 | hnsw | sindi | hnsw_sindi | fhg | uhg | 索引兼容性 |
|---|---|---|---|---|---|---|
| 1. malloc→tls | 无 | 无* | 无* | ✅提升 | ✅提升 | 兼容 |
| 2. 定长桶 | 无 | 无* | 无* | ✅提升 | ✅提升 | **需重建** |
| 3. SIMD/hash IP | 无 | 无* | 无* | ✅提升 | ✅提升 | 兼容 |
| 4. std::function→模板 | ? | 无 | ? | ✅提升 | ✅提升 | 兼容 |
| 5. Cauchy SIMD | 无 | 无 | 无 | ✅提升 | ✅提升 | 兼容 |
| 6. Heap 优化 | ? | 无 | ? | ✅提升(ef≥500) | ✅提升(ef≥500) | 兼容 |
| 8. 稀疏批量预取 | 无 | 无* | 无* | ✅提升 | ✅提升 | 兼容 |
| 9. 更紧上界剪枝 | 无 | 无 | 无 | ✅提升 | ✅提升 | 兼容 |
| 10. Vector 复用 | 无 | 无 | 无 | ✅提升 | ✅提升 | 兼容 |
| 11. early-exit 检查 | ? | 无 | ? | ? | ? | 兼容（Recall 影响） |
| 12. LCG 复用 | ? | 无 | ? | ✅微提 | ✅微提 | 兼容 |
| 13. GetNeighbors 删除标记 | ? | 无 | ? | ✅提升 | ✅提升 | 兼容 |

`?` = 需确认是否走 `BasicSearcher`；`*` = SINDI rerank 阶段可能受益（需确认）

### 4.4 实验对比公平性

- **优化 1、3、4、5、6、8、9、10、12、13**：不破坏索引兼容性，fhg 和 uhg 同步变快，其他 baseline 不变或同步变快。**相对对比关系不受影响**。
- **优化 2**：破坏索引兼容性，需要重建所有 hybrid_index。fhg 和 uhg 同步受益。**相对对比关系不受影响**，但需要重跑所有 fhg 和 uhg 实验（baseline 不需要重跑，因为 hnsw/sindi/hnsw_sindi 的索引不受影响）。
- **优化 11**：可能影响 Recall，需要单独验证。

## 5. 实施建议

### 5.1 优先级

| 优先级 | 优化 | 原因 |
|---|---|---|
| **P0** | 优化 1 (malloc→tls) | 收益最大、改动最小、风险最低、不破坏兼容性 |
| **P0** | 优化 8 (稀疏批量预取) | 串行 IO 是最大瓶颈之一、改动中等、不破坏兼容性 |
| **P0** | 优化 10 (Vector 复用) | 每跳 5 次分配、改动小、不破坏兼容性 |
| **P1** | 优化 3 (SIMD/hash IP) | 收益较大、改动小、不破坏兼容性 |
| **P1** | 优化 4 (std::function→模板) | 收益小但免费、改动极小、不破坏兼容性 |
| **P1** | 优化 13 (GetNeighbors 删除标记) | 只读图走 fast path、改动小 |
| **P1** | 优化 9C (调 prune_scale) | 仅调参，立即测，无代码改动 |
| **P2** | 优化 2 (定长桶) | 收益大但破坏兼容性、需要重建索引、工程量大 |
| **P3** | 优化 5 (Cauchy SIMD) | 收益不确定（SIMD gather 可能不比标量快） |
| **P3** | 优化 6 (Heap 优化) | 收益仅在 ef≥500 时显著，实现复杂 |
| **P3** | 优化 11 (early-exit 检查) | 主要是正确性，不是 QPS |
| **跳过** | 优化 7 (entry dense 复用) | 实测收益小 |
| **跳过** | 优化 12 (LCG 复用) | 收益 <1% |

### 5.2 验证方案

每个优化实施后：

1. **正确性验证**：跑现有单元测试 `./build/tests/unittests`，确保不破坏现有功能
2. **性能验证**：在 NQ 上跑 `run_uhg.sh nq`（全量 query，5 个 alpha，5 个 ef 点），对比优化前后的 Recall-QPS
   - Recall 应完全一致（优化 1/3/4/5/6/8/10/12/13 不改搜索语义，只改实现效率）
   - 优化 9C（调 prune_scale）会改变 Recall，需记录曲线
   - QPS 应提升
3. **Baseline 对比验证**：跑 `run_baselines.sh nq`，确认 baseline 的 QPS 不变（优化 1/3/4/5/6/8/10/12/13）或同步提升（优化 2）

### 5.3 预期总体收益

| 优化组合 | 预期 QPS 提升（UHG） | 索引兼容性 |
|---|---|---|
| 仅优化 1 | 30-50% | 兼容 |
| 优化 1 + 8 + 10 | 50-80% | 兼容 |
| 优化 1 + 3 + 4 + 8 + 10 + 13 | 60-95% | 兼容 |
| 上面 + 优化 2 + 9 | 80-130% | **需重建索引** |

## 6. 附录

### 6.1 关键文件清单

```
src/impl/searcher/basic_searcher.cpp          # 主搜索循环（优化 4、6、11、12）
src/impl/searcher/basic_searcher.h            # 搜索器接口
src/impl/inner_search_param.h                 # 搜索参数结构
src/datacell/hybrid_vector_datacell.cpp       # Hybrid 距离计算（优化 5、9、10）
src/datacell/hybrid_vector_datacell.h         # Hybrid datacell 接口
src/datacell/sparse_vector_datacell.h         # 稀疏存储（优化 1、2、8）
src/datacell/sparse_vector_datacell.inl       # 稀疏存储实现（优化 1、2、8）
src/quantization/sparse_quantization/sparse_quantizer.h  # 稀疏 IP 计算（优化 3）
src/datacell/graph_datacell.h                 # 图数据cell（优化 13）
src/algorithm/hybrid_index/hybrid_index.cpp   # hybrid_index 主逻辑
src/algorithm/sindi/sindi.cpp                 # SINDI 搜索（baseline）
src/datacell/sparse_term_datacell.cpp         # SINDI 倒排表（baseline）
```

### 6.2 Baseline 索引文件清单

```
hnsw:        data/index/701_<dataset>_dense_hnsw.index         # 不受优化 1-13 影响
sindi:       data/index/701_<dataset>_sparse_sindi.index       # 不受优化 1、3-13 影响；优化 2 可能影响
hnsw_sindi:  同上两个索引
fhg:         data/index/703_<dataset>_fhg_hybrid_index.index   # 优化 1、3-6 兼容；优化 2 需重建
uhg:         data/index_hybrid_union/703_<dataset>_hybrid_index.index  # 同 fhg
uhg 内部 SINDI: data/index/709_<dataset>_sindi.index           # 优化 2 需重建
```

### 6.3 实验数据参考（test_uhgh, NQ, alpha=0.7）

```
scale=0.5:
  ef=100  Recall 0.921  QPS 165.7
  ef=200  Recall 0.946  QPS 120.9
  ef=300  Recall 0.967  QPS 83.8
  ef=500  Recall 0.982  QPS 53.7
  ef=1000 Recall 0.993  QPS 29.2

scale=1.0 (主实验默认):
  ef=100  Recall 0.921  QPS 165.3
  ef=200  Recall 0.968  QPS 89.9
  ef=300  Recall 0.981  QPS 64.8
  ef=500  Recall 0.990  QPS 40.9
  ef=1000 Recall 0.996  QPS 22.2

scale=2.0:
  ef=100  Recall 0.957  QPS 114.5
  ef=200  Recall 0.983  QPS 62.3
  ef=300  Recall 0.990  QPS 43.9
  ef=500  Recall 0.995  QPS 28.0
  ef=1000 Recall 0.998  QPS 15.3
```

结论：entry scale 调整只平移曲线，无 Pareto 改善。确认 entry search 非瓶颈，优化重心应放在主图搜索循环。
