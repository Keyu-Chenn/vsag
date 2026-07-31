# UHG 搜索代码定向优化设计

## 1. 文档目的

本文档细化以下四项只与 UHG/hybrid index 搜索相关的优化：

1. 重做原 `optimization_plan.md` 的“优化 1”：消除 sparse code 查询中的逐向量内存分配和复制。
2. 重做原 `optimization_plan.md` 的“优化 10”：消除 `HybridVectorDataCell::query` 每个 hop 的临时 Vector 分配。
3. 优化 UHG 自己的 dense-entry HNSW 适配层，不修改 baseline HNSW 的实现。
4. 让 `HybridComputer` 持有一次搜索生命周期内可复用的 scratch，并作为优化 1、优化 10 和后续 hybrid pruning 的统一搜索上下文。

分析基线为同目录的 `MAIN_EXPERIMENT_RUNBOOK.md`、`optimization_plan.md`，以及当前
`hybrid_index`、`HybridVectorDataCell`、`SparseVectorDataCell` 和 `BasicSearcher` 实现。

本文档是实现设计，不包含源码修改。目标是先明确接口、内存生命周期、兼容性、测试方法和分阶段落地顺序，再决定是否实施。

## 2. 范围和非目标

### 2.1 范围

涉及的主要文件：

```text
src/datacell/sparse_vector_datacell.h
src/datacell/sparse_vector_datacell.inl
src/datacell/hybrid_vector_datacell.h
src/datacell/hybrid_vector_datacell.cpp
src/algorithm/hybrid_index/hybrid_index.cpp
src/algorithm/hybrid_index/dense_entry_hnsw_graph.h   # 若按建议拆出内部类
src/algorithm/hybrid_index/dense_entry_hnsw_graph.cpp # 若按建议拆出内部类
src/impl/searcher/basic_searcher.cpp              # 仅在复用 entry distance 时可能涉及
src/impl/inner_search_param.h                     # 仅在复用 entry distance 时可能涉及
```

测试文件：

```text
src/datacell/sparse_vector_datacell_test.cpp
src/datacell/hybrid_vector_datacell_test.cpp
src/impl/searcher/basic_searcher_test.cpp
```

### 2.2 非目标

- 不修改 baseline HNSW 的 `hnswlib` 搜索逻辑。
- 不修改 baseline SINDI 的倒排搜索逻辑。
- 不改变 UHG 图构建算法、邻居集合或 auto 路由规则。
- 不在本方案中引入新的近似打分、距离量化或 Recall-QPS trade-off。
- 第一阶段不改变 sparse code 的序列化格式。

### 2.3 对实验方法的影响范围

| 改动 | UHG/UHGS/UHGH | FHG | HNSW | SINDI | HNSW+SINDI |
|---|---:|---:|---:|---:|---:|
| sparse zero-copy | 提升 | 提升 | 无 | 无 | 无 |
| HybridComputer scratch | 提升 | 提升 | 无 | 无 | 无 |
| dense-entry 适配层 | 仅使用外部 dense HNSW 文件的 UHGH 提升 | 无 | 无 | 无 | 无 |
| dense-entry distance 复用 | UHGH 提升，可覆盖两种 dense-entry 路径 | 无 | 无 | 无 | 无 |

前两项属于 hybrid 主图公共路径，因此会同时改善 FHG；后两项是 UHG/UHGH 专属优化，更有利于改善 UHG 相对 baseline 的性能。

## 3. 当前热点和约束

### 3.1 当前 sparse code 查询

`SparseVectorDataCell::GetCodesById` 当前执行：

```text
offset_io_ 读取 offset
io_        读取 length
allocator  Allocate(read_size)
io_        把完整 code 复制到临时 buffer
计算 sparse IP
allocator  Deallocate(buffer)
```

UHG 的 sparse cell 使用 `memory_io`。`MemoryIO::DirectReadImpl` 已经能够直接返回 `start_ + offset`，并将 `need_release` 设为 false。因此 UHG 搜索没有必要把 code 再复制到一个临时 buffer。

### 3.2 当前 HybridVectorDataCell 临时内存

`HybridVectorDataCell::query` 一进入函数就创建：

```text
dense_dists
sparse_dists
```

进入 pruning 分支后还会创建：

```text
idx_sparse
sparse_positions
selected_sparse_dists
```

这些 Vector 在每次邻居扩展、即每个 hop 中重复创建和释放。主图最大 degree 为 64，单次分配不大，但调用次数很多，而且 pruning 分支中的全量 `sparse_dists` 实际没有使用。

### 3.3 当前 dense-entry HNSW 适配层

`DenseEntryHNSWGraph` 是 `hybrid_index.cpp` 内部的 UHG 专用类，只有配置并加载
`dense_entry_hnsw_graph_path` 时才走这条路径。当前实现具有以下特征：

- level-0 邻接使用 `vector<vector<InnerIdType>>`；
- upper-level 邻接使用三层 vector；
- base-layer 搜索逐个邻居调用一次 `dense_cell->Query(..., count=1)`；
- 已经计算出的 entry dense distance 最终被丢弃，只返回 ID；
- 主图初始化再次对所有 entry 计算 dense+sparse 完整距离。

UHGH 还有一条 fallback：`dense_entry_graph_ + dense_entry_searcher_`。这条路径不使用
`DenseEntryHNSWGraph`，因此不会从本适配层的 CSR/batch 改动中受益，但它的搜索结果堆中
同样已经带有 dense distance，可以参与后面的 entry distance 复用。

这里的修改可以完全限制在 hybrid index 内，不需要修改 baseline HNSW。

## 4. 优化 1：Sparse code 零拷贝访问

### 4.1 对原 TLS buffer 方案的修正

不建议在通用 `GetCodesById` 中使用单个 thread-local buffer，原因如下：

1. `ComputePairVectors` 会先连续获取 `codes1` 和 `codes2`，然后才计算距离。单 TLS buffer 会使第二次读取覆盖第一次结果。
2. `GetCodesById` 是公共接口，不能只依据当前 `query()` 中“读取后立即计算”的调用方式设计生命周期。
3. thread-local buffer 的释放时机、allocator 归属和多索引共存都比较复杂。
4. UHG 使用 MemoryIO，本身已有稳定的直接读取能力，不需要额外 buffer。

因此第一选择应当是 **DirectRead/zero-copy**，TLS 或 buffer pool 只作为非连续 IO 的 fallback，而不是默认路径。

### 4.2 目标行为

对于 UHG 的 `SparseVectorDataCell<SparseQuantizer, MemoryIO>`：

```text
读取 offset                 小型随机访问
直接读取 code header 指针   不复制
从 header 得到 length
直接返回完整 code 指针      不复制、不分配
ComputeDist
无需释放
```

对于 `MemoryBlockIO` 等可能跨 block 的存储：

- code 位于单个连续 block 时直接返回内部指针；
- code 跨 block 时允许 IO 层分配临时 buffer，并通过 `need_release=true` 通知调用方；
- 调用方统一使用 `io_->Release(codes)`，不能直接假设由 `allocator_->Deallocate` 释放。

### 4.3 建议接口和伪代码

保持现有接口不变：

```cpp
const uint8_t*
GetCodesById(InnerIdType id, bool& need_release) const override;
```

建议实现逻辑：

```cpp
const uint8_t*
SparseVectorDataCell::GetCodesById(InnerIdType id, bool& need_release) const {
    uint32_t offset = 0;
    offset_io_->Read(sizeof(offset),
                     static_cast<uint64_t>(id) * sizeof(offset),
                     reinterpret_cast<uint8_t*>(&offset));

    bool header_need_release = false;
    const uint8_t* header = io_->Read(sizeof(uint32_t), offset, header_need_release);
    CHECK_ARGUMENT(header != nullptr, "invalid sparse code header");

    uint32_t length = 0;
    std::memcpy(&length, header, sizeof(length));
    if (header_need_release) {
        io_->Release(header);
    }

    const uint64_t read_size =
        sizeof(uint32_t) + static_cast<uint64_t>(length) * sizeof(BufferEntry);

    need_release = false;
    const uint8_t* codes = io_->Read(read_size, offset, need_release);
    CHECK_ARGUMENT(codes != nullptr, "invalid sparse code range");
    return codes;
}
```

这里仍然会调用两次 DirectRead：一次读取 header，一次校验并返回完整范围。对于 MemoryIO 两次都只是边界检查和指针计算，没有 code memcpy。

正式实现不能只照抄伪代码，还要补齐以下校验：

- `id < total_count_`；
- `offset + sizeof(uint32_t)` 不越界；
- `length` 不超过 quantizer/dimension 允许的最大 term 数；
- `sizeof(uint32_t) + length * sizeof(BufferEntry)` 不发生整数溢出；
- 完整 `read_size` 位于 IO 的有效范围内。

如果后续愿意为 MemoryIO 增加只读 span API，可以进一步变为一次指针解析，但不应作为第一阶段的前置条件。

### 4.4 释放契约必须统一

当前 `SparseVectorDataCell::query` 在 `need_release=true` 时直接调用：

```cpp
allocator_->Deallocate(codes);
```

建议统一改成：

```cpp
if (need_release) {
    io_->Release(codes);
}
```

同样需要检查并修正：

- `SparseVectorDataCell::query`；
- `SparseVectorDataCell::ComputePairVectors`；
- 其他直接调用 `GetCodesById(..., need_release)` 的位置。

这可以保证 MemoryIO、MemoryBlockIO 和未来其他 IO 的所有权语义一致。

### 4.5 可变索引和指针有效期

MemoryIO 在扩容时会 `Reallocate(start_)`，可能使旧指针失效。因此 zero-copy 依赖以下约束：

- 一次 `ComputeDist` 期间不能并发触发同一个 sparse cell 的扩容；
- UHG 主实验中的 hybrid index 在 Build/Deserialize 完成后按只读索引使用，满足该条件；
- 如果未来支持 search 与 Add 并发，应在 sparse cell 上增加读写锁或 immutable/frozen 状态；
- mutable 模式可以保留复制 fallback，immutable 模式启用 zero-copy fast path。

推荐显式引入只读状态，而不是依赖隐含约定：

```cpp
bool immutable_{false};

void SetImmutable();
```

第一阶段如果不想扩大接口，可先只在当前 UHG 的静态使用约束下启用，并增加注释和并发测试。

### 4.6 第二阶段：批量解析和预取

单纯把同步 `Read` 放进环形 buffer 不会自动形成 IO overlap。正确的内存预取方式是：

```text
阶段 1：批量解析本 hop 所有 sparse offsets/code pointers
阶段 2：对未来若干个 code pointer 发出 CPU prefetch
阶段 3：按原始顺序计算 sparse IP
```

可复用 HybridComputer scratch 保存：

```cpp
Vector<const uint8_t*> sparse_code_ptrs;
Vector<uint8_t> sparse_code_need_release;
```

但当前 `SparseVectorDataCell::Query` 只接收 `hybrid_comp->GetSparseComputer()`，并不能反向取得
外层 `HybridComputer`。因此不能只增加上述字段而不设计传参。若实施该可选阶段，推荐增加一个
不进入通用 `FlattenInterface` 的扩展接口：

```cpp
struct SparseReadScratch {
    explicit SparseReadScratch(Allocator* allocator)
        : code_ptrs(allocator), need_release(allocator) {
    }

    Vector<const uint8_t*> code_ptrs;
    Vector<uint8_t> need_release;
};

class SparseQueryWithScratchInterface {
public:
    virtual ~SparseQueryWithScratchInterface() = default;

    virtual void QueryWithReadScratch(
        float* result_dists,
        const ComputerInterfacePtr& sparse_computer,
        const InnerIdType* ids,
        InnerIdType count,
        SparseReadScratch& scratch) = 0;
};
```

`SparseVectorDataCell` 实现该扩展；`HybridVectorDataCell` 在构造时把 `sparse_cell_` cross-cast
并缓存为 `SparseQueryWithScratchInterface`，每个 hop 显式传入
`hybrid_comp->GetOrCreateScratch(search_alloc).sparse_read_scratch_`。普通 sparse caller 仍走现有 `Query`，不修改
baseline 使用的通用 Flatten 接口。若扩展不可用，则直接回退现有标量 Query。

伪代码：

```cpp
for (i = 0; i < count; ++i) {
    refs[i].codes = GetCodesById(ids[i], refs[i].need_release);
}

for (i = 0; i < min(prefetch_ahead, count); ++i) {
    PrefetchLines(refs[i].codes, prefetch_depth);
}

for (i = 0; i < count; ++i) {
    if (i + prefetch_ahead < count) {
        PrefetchLines(refs[i + prefetch_ahead].codes, prefetch_depth);
    }
    computer->ComputeDist(refs[i].codes, result_dists + i);
    if (refs[i].need_release) {
        io_->Release(refs[i].codes);
    }
}
```

该阶段必须在 zero-copy 完成后再做，并通过 perf 的 cache-miss、cycles 和 QPS 结果决定是否保留。

UHG 当前使用 `MemoryIO`，所有 code ref 都是 non-owning pointer，因此第一版 batch/prefetch
建议只对 `MemoryIO` 启用。对于可能返回 `need_release=true` 的 IO，继续保留“取一个、计算、
立即 Release”的标量 fallback。若以后确实需要批量持有跨 block 临时对象，必须使用 RAII guard，
保证中途异常时已经取得的 buffer 也会被释放，不能仅用两组裸指针数组承担所有权。

### 4.7 兼容性

- 不改变 sparse code 格式。
- 不改变 hybrid index 序列化格式。
- 已有 `703_*_hybrid_index.index` 可以继续加载。
- 理论距离和 Recall 应完全不变。

### 4.8 测试要求

必须补充以下测试：

1. MemoryIO 下 `need_release=false`。
2. MemoryBlockIO 单 block code 下 `need_release=false`。
3. 人工构造跨 block code，验证 `need_release=true` 和 Release。
4. `ComputePairVectors` 连续持有两份 code，验证不会发生覆盖。
5. Serialize/Deserialize 后距离完全一致。
6. 多线程并发只读 Query。
7. SafeAllocator 统计单次大批量 Query 的 Allocate 次数显著下降。

## 5. 优化 10：由 HybridComputer 持有搜索级 scratch

原“优化 10”和“HybridComputer 持有每次搜索的 scratch”本质上应当合并实现。只增加 thread-local Vector 或 datacell 成员 Vector 都不合适。

### 5.1 为什么 scratch 应属于 HybridComputer

`HybridComputer` 由 `FactoryComputer(query)` 为每次搜索创建，并由 `BasicSearcher::search_impl` 在搜索结束时释放。因此它天然满足：

- 一个 query/一次搜索独占；
- 不同搜索线程不共享；
- 生命周期覆盖 entry 初始化和全部主图 hops；
- 已经持有 dense computer、sparse computer、query norm、weight 和 lower bound；
- 不需要使用 thread-local，也不需要给 datacell 加锁。

### 5.2 建议数据结构

```cpp
class HybridSearchScratch {
public:
    explicit HybridSearchScratch(Allocator* allocator)
        : allocator_(allocator),
          sparse_dists_(allocator),
          sparse_ids_(allocator),
          sparse_positions_(allocator),
          selected_sparse_dists_(allocator),
          sparse_read_scratch_(allocator) {
    }

    void EnsureNeighborCapacity(uint32_t count);
    void ClearSelection();

    Allocator* allocator_{nullptr};

    // no-prune 分支需要
    Vector<float> sparse_dists_;

    // pruning 分支需要
    Vector<InnerIdType> sparse_ids_;
    Vector<uint32_t> sparse_positions_;
    Vector<float> selected_sparse_dists_;

    // sparse batch resolve/prefetch 的可选第二阶段
    SparseReadScratch sparse_read_scratch_;
};
```

`SparseReadScratch` 只是 P2 batch/prefetch 的接口载体；P0/P1 的 result buffer 复用和 selection
scratch 不依赖它。第一轮实现可以先不加入该字段，避免为未经 perf 证明的预取路径扩大接口。

`HybridComputer` 新增：

```cpp
class HybridComputer : public ComputerInterface {
public:
    HybridSearchScratch& GetOrCreateScratch(Allocator* allocator);

private:
    std::optional<HybridSearchScratch> scratch_;
};
```

这里推荐 `optional` 而不是 `unique_ptr`，可以避免每次搜索再为 scratch 对象本身走一次全局
`new/delete`；真正可能增长的数组仍由 `Vector` 和 search allocator 管理。

### 5.3 allocator 处理

`FactoryComputer` 当前不知道 `BasicSearcher` 最终传入的 `search_alloc`，因此 scratch 应在第一次 `query(..., allocator)` 时延迟创建：

```cpp
HybridSearchScratch&
HybridComputer::GetOrCreateScratch(Allocator* allocator) {
    if (not scratch_.has_value()) {
        scratch_.emplace(allocator);
    }
    CHECK_ARGUMENT(scratch_->allocator_ == allocator,
                   "one HybridComputer cannot switch search allocator");
    return *scratch_;
}
```

传给该函数的必须是 `allocator == nullptr ? allocator_ : allocator` 解析后的
`search_alloc`，而不是原始可空参数。一次 BasicSearcher 搜索使用固定 allocator，因此这个
约束合理。不要让 Vector 在创建后切换 allocator。

### 5.4 重写 query 的内存流

核心优化是直接复用调用方的 `result_dists` 保存 dense distance，不再创建 `dense_dists`。

#### 5.4.1 Dense-only 分支

```cpp
if (std::abs(dense_weight) <= kWeightEpsilon) {
    std::fill_n(result_dists, id_count, 0.0F);
} else {
    dense_cell_->Query(result_dists, dense_computer, idx, id_count, search_alloc);
    for (i = 0; i < id_count; ++i) {
        result_dists[i] *= dense_weight;
    }
}
```

临时 Vector 数量：0。

#### 5.4.2 不启用 hybrid pruning 的分支

```cpp
if (std::abs(dense_weight) > kWeightEpsilon) {
    dense_cell_->Query(result_dists, dense_computer, idx, id_count, search_alloc);
} else {
    // 防止后面的 0 * 未初始化值产生 NaN。
    std::fill_n(result_dists, id_count, 0.0F);
}

scratch.sparse_dists_.resize(id_count);
sparse_cell_->Query(scratch.sparse_dists_.data(),
                    sparse_computer,
                    idx,
                    id_count,
                    search_alloc);

for (i = 0; i < id_count; ++i) {
    result_dists[i] = dense_weight * result_dists[i]
                    + sparse_weight * scratch.sparse_dists_[i];
}
```

第一次 hop 可能扩容一次，后续 hop 不再分配。

#### 5.4.3 启用 hybrid pruning 的分支

```cpp
dense_cell_->Query(result_dists, dense_computer, idx, id_count, search_alloc);

scratch.ClearSelection();
scratch.EnsureNeighborCapacity(id_count);

for (i = 0; i < id_count; ++i) {
    const float dense_dist = result_dists[i];
    const float dense_ip = 1.0F - dense_dist;
    const float sparse_ub = query_sparse_norm * sparse_norms_[idx[i]];
    const float score_ub = dense_weight * dense_ip
                         + sparse_weight * prune_scale * sparse_ub;

    if (score_ub > score_threshold + epsilon) {
        scratch.sparse_ids_.push_back(idx[i]);
        scratch.sparse_positions_.push_back(i);
    } else {
        result_dists[i] = lower_bound + epsilon;
    }
}

const auto selected_count = scratch.sparse_ids_.size();
if (selected_count > 0) {
    scratch.selected_sparse_dists_.resize(selected_count);

    sparse_cell_->Query(scratch.selected_sparse_dists_.data(),
                        sparse_computer,
                        scratch.sparse_ids_.data(),
                        selected_count,
                        search_alloc);

    for (i = 0; i < selected_count; ++i) {
        const auto pos = scratch.sparse_positions_[i];
        const float dense_dist = result_dists[pos];
        result_dists[pos] = dense_weight * dense_dist
                          + sparse_weight * scratch.selected_sparse_dists_[i];
    }
}
```

注意：只有被选中计算 sparse 的位置仍保留 dense distance；被剪枝位置已经被写为 `lower_bound + epsilon`，两者不会冲突。

### 5.5 容量策略

主图邻居数通常不超过 64，但代码不应硬编码 64。建议：

```cpp
if (capacity < id_count) {
    new_capacity = NextPowerOfTwo(id_count);
    reserve(new_capacity);
}
```

原则：

- scratch Vector 只增长不缩小；
- 每个 hop 只 clear/resize，不释放 capacity；
- 不对每个 query 预分配很大的 `ef_search`，因为 scratch 按单批邻居数而不是 ef 分配；
- entry 批量距离计算可能一次传入 100–1000 个 ID，scratch 会扩展到 entry pool 大小，然后主图继续复用。

如果后续把 entry pool 与 graph neighbor scratch 分开，可以避免 entry pool 使主图 scratch 长期保留过大 capacity；第一阶段没有必要提前复杂化。

### 5.6 不建议的替代方案

#### datacell 成员 buffer

`HybridVectorDataCell` 被多个搜索线程共享，成员 buffer 需要锁或每线程分片，不合适。

#### 全局/thread-local scratch

- allocator 归属不清晰；
- 多个索引共用线程时可能保留超大 buffer；
- 嵌套查询和递归调用难以保证安全；
- 生命周期不如 query computer 清晰。

#### 每个 hop 使用栈上定长数组

当前 degree 为 64，但接口支持其他 degree；硬编码数组容易溢出或产生额外 fallback 分支。

### 5.7 兼容性和正确性

- 不改变索引格式。
- 不改变距离计算顺序的主要逻辑。
- 不改变剪枝条件。
- 已有 hybrid index 不需要重建。
- 结果应与改动前一致；浮点表达式应保持相同结合顺序，避免无意产生边界差异。

### 5.8 测试和统计

新增或扩展测试：

1. dense-only、sparse-only、no-prune、prune 四个分支。
2. 连续多次 Query，第一次扩容后 allocator 分配次数不再随 hop 增长。
3. 同一个 datacell 上多线程创建不同 HybridComputer 并查询。
4. search allocator 与 index allocator 不同时正常释放。
5. id_count 从 0、1、63、64、65 到较大 entry pool 的边界测试。
6. Serialize/Deserialize 前后结果一致。

建议给 HybridComputer 增加仅用于统计的计数，默认关闭详细日志：

```text
hybrid_dense_evaluated
hybrid_sparse_evaluated
hybrid_sparse_pruned
scratch_growth_count
scratch_peak_capacity
```

## 6. UHG dense-entry HNSW 适配层优化

本节只优化 `hybrid_index.cpp` 中的 `DenseEntryHNSWGraph`，不修改 `src/algorithm/hnswlib`。
CSR 和邻居 batch 仅作用于配置了 `dense_entry_hnsw_graph_path` 的 UHGH；内建 fallback
entry graph 继续使用现有 `BasicSearcher + FlattenDataCell` 路径。

为了能独立测试，建议先把内部类原样拆到
`src/algorithm/hybrid_index/dense_entry_hnsw_graph.{h,cpp}`。该提交只移动代码、不改行为，
并继续保持类位于 hybrid index 模块内；如果不愿增加文件，也可以通过 HybridIndex 集成测试，
但很难覆盖 CSR 边界、v1/v2 兼容和 scalar/batch 对照。

### 6.1 分阶段目标

| 阶段 | 改动 | 是否改变搜索结果 | 是否需要新序列化版本 |
|---|---|---:|---:|
| A | 外部 dense HNSW 邻居 distance 批量计算 | 算法不变，允许末位浮点差异 | 否 |
| B | 外部 dense HNSW level-0 邻接改成 flat CSR | 否 | 建议是 |
| C | 两种 UHGH entry 路径都返回 ID + dense distance | 否 | 否 |
| D | 主图 entry 初始化复用 dense distance | 算法不变，允许末位浮点差异 | 否，但涉及搜索接口 |

应先实施 A 并测量，再决定 B；C/D 应一起评估，避免只增加接口复杂度而没有收益。

### 6.2 阶段 A：批量计算邻居 dense distance

当前 base-layer 内循环对每个邻居调用：

```cpp
dense_cell->Query(&distance, computer, &hybrid_inner_id, 1, allocator);
```

建议每次展开一个 HNSW 节点时：

1. 收集未访问且有效的 internal IDs；
2. 映射到 hybrid inner IDs；
3. 对 dense vector 发出预取；
4. 一次调用 `dense_cell->Query(..., count=N)`；
5. 按原邻居顺序更新 candidate/top heaps。

建议增加每次 entry search 独占的 context：

```cpp
struct DenseEntrySearchContext {
    Vector<InnerIdType> internal_ids;
    Vector<InnerIdType> hybrid_ids;
    Vector<float> dists;

    explicit DenseEntrySearchContext(Allocator* allocator)
        : internal_ids(allocator), hybrid_ids(allocator), dists(allocator) {
    }

    void EnsureCapacity(uint32_t max_degree);
    void ClearBatch();
};
```

base-layer 伪代码：

```cpp
context.ClearBatch();
for (auto candidate_internal_id : Neighbors(current_node_id, 0)) {
    if (candidate_internal_id >= num_nodes_ ||
        visited[candidate_internal_id] == visited_tag) {
        continue;
    }

    visited[candidate_internal_id] = visited_tag;
    if (!HasVector(candidate_internal_id)) {
        continue;
    }

    context.internal_ids.push_back(candidate_internal_id);
    context.hybrid_ids.push_back(internal_to_inner_[candidate_internal_id]);
}

context.dists.resize(context.hybrid_ids.size());
dense_cell->Query(context.dists.data(),
                  computer,
                  context.hybrid_ids.data(),
                  context.hybrid_ids.size(),
                  allocator);

for (i = 0; i < context.internal_ids.size(); ++i) {
    const auto candidate_internal_id = context.internal_ids[i];
    const float distance = context.dists[i];
    // 保持当前 heap 更新逻辑
}
```

该改法复用 `FlattenDataCell` 已有的 batch4 和 prefetch 路径。为了保持行为一致，heap 更新仍按原始邻居顺序执行。

`count=1` 的 scalar kernel 和 `count>=4` 的 batch4 kernel 可能使用不同的 SIMD 累加顺序。
因此这里的“不改变结果”指邻居集合、访问顺序和 heap 规则不变，不承诺 distance 位级一致。
单元测试应使用数值容差，并单独检查 top entry 集合、顺序、Recall 和临界 tie；如果某个数据集
在 tie 附近发生可见 Recall 波动，应保留 scalar 开关，而不是把它当作纯内存重构直接合入。

upper-level greedy search 也可以批量计算当前节点在该层的全部邻居，然后选择最小 distance；不过 upper level 占比通常较小，可以放在第二步。

### 6.3 阶段 B：level-0 邻接改成 flat CSR

当前：

```cpp
std::vector<std::vector<InnerIdType>> level0_neighbors_;
```

建议：

```cpp
std::vector<uint64_t> level0_offsets_;       // size = num_nodes + 1
std::vector<InnerIdType> level0_neighbors_;  // flat payload
```

节点 `id` 的邻居范围：

```cpp
begin = level0_offsets_[id];
end   = level0_offsets_[id + 1];
```

为避免依赖 `std::span` 或额外分配，可以定义轻量视图：

```cpp
struct NeighborView {
    const InnerIdType* data{nullptr};
    uint32_t size{0};
};
```

优点：

- 去掉每个节点一个 vector 对象和独立分配；
- 邻接数据连续，减少 pointer chasing；
- 序列化和反序列化可以一次写入大块数据；
- entry search 访问邻居时更容易预取。

不建议直接使用 `num_nodes × max_m0` 固定槽位，因为 max_m0 可能明显大于平均 degree，会增加常驻内存。

upper-level 节点远少于 level-0，可以分两步处理：

- 第一阶段暂时保留当前 upper-level vector；
- 如果 profile 证明 upper-level 碎片显著，再增加 `upper_level_offsets` 和 flat payload。

### 6.4 CSR 构建过程

从 baseline HNSW 文件加载时当前逐节点读取邻居。改成两遍或增量构建均可：

#### 方案一：两遍构建

1. 第一遍记录每个节点 `count`，构建 offsets；
2. 根据最后一个 offset 一次性分配 flat neighbors；
3. 第二遍填充邻居。

由于输入流不方便回退，通常需要临时保存原始 level-0 link block，内存收益不一定好。

#### 方案二：单遍 append，推荐

HNSW level-0 element 本来就按 internal ID 顺序读取，不需要先构造
`vector<vector<id>>` 再 compact：

```cpp
level0_offsets_.reserve(num_nodes_ + 1);
level0_offsets_.push_back(0);

for (InnerIdType internal_id = 0; internal_id < num_nodes_; ++internal_id) {
    // 读取当前 element_buffer 和 count
    for (auto neighbor : parsed_neighbors) {
        if (neighbor < num_nodes_) {
            level0_neighbors_.push_back(neighbor);
        }
    }
    level0_offsets_.push_back(level0_neighbors_.size());
}
```

这样只有 flat payload 自身的增长。可以依据 `num_nodes_ ×` 实测平均 degree 做保守 reserve，
但不要直接 reserve `num_nodes_ × max_m0_`，否则可能制造不必要的峰值内存。

读取旧 v1 内部序列化时，也只需复用一个“当前节点邻居”临时 Vector，读完当前节点就 append
到 flat payload 并更新 offset，不需要同时保留整份旧表示。

### 6.5 阶段 C：返回 entry distance

定义：

```cpp
struct DenseEntryCandidate {
    InnerIdType inner_id;
    float dense_distance;
};
```

`DenseEntryHNSWGraph::Search` 返回：

```cpp
std::vector<DenseEntryCandidate>
Search(...);
```

第一版应保持当前 `priority_queue` pop 的顺序和相同 distance 下的 ID tie-break，不要顺手改为
“从近到远排序”。entry 初始化通常对顺序不敏感，但 candidate-set 上限和浮点 tie 可能放大
顺序变化。若后续希望改成近到远，应作为独立算法实验。

这一步本身不会提高主图速度，只有阶段 D 消费这些 distance 后才有收益。因此 C/D 应放在同一实验分支中。

fallback 的 `dense_entry_searcher_` 已返回包含 `(distance, inner_id)` 的 heap，也应在转换成
entry list 时填充同一个 `DenseEntryCandidate`，否则同为 UHGH 却只有外部 HNSW 路径能复用。

### 6.6 阶段 D：主图 entry 初始化复用 dense distance

当前 BasicSearcher 对所有 entry 调用一次完整：

```cpp
flatten->Query(ep_dists, hybrid_computer, entry_points, entry_count, alloc);
```

UHGH 已经拥有每个 entry 的 dense distance，可以只计算 sparse distance并合成：

```text
hybrid_distance = dense_weight  × precomputed_dense_distance
                + sparse_weight × sparse_distance
```

#### 推荐接口设计

不要把 dense-entry 细节硬编码进通用 `FlattenInterface::Query`。建议给 `HybridVectorDataCell` 增加显式方法：

```cpp
void QueryWithPrecomputedDense(
    float* result_dists,
    const ComputerInterfacePtr& computer,
    const InnerIdType* idx,
    const float* dense_dists,
    InnerIdType id_count,
    Allocator* allocator);
```

该函数：

- 不调用 dense cell；
- 计算 entry 的 sparse distance；
- 使用搜索 alpha 合成完整 hybrid distance；
- 使用 HybridComputer scratch；
- entry 初始化阶段不启用 sparse pruning，因为 lower bound 尚未稳定。

为了让 BasicSearcher 使用完整预计算距离，建议给 `InnerSearchParam` 增加通用而非 dense-specific 的 seed：

```cpp
struct SearchSeed {
    InnerIdType id;
    float distance;
};

std::vector<SearchSeed> precomputed_seeds;
```

BasicSearcher entry 初始化规则：

```text
precomputed_seeds 非空：直接使用其中完整 distance
否则 eps 非空：保持现有 flatten->Query
否则使用 ep
```

HybridIndex 的 UHGH 路径负责：

1. dense-entry HNSW 返回 ID+dense distance；
2. `HybridVectorDataCell::QueryWithPrecomputedDense` 生成完整 hybrid distance；
3. 填充 `precomputed_seeds`；
4. BasicSearcher 不再重复计算 entry dense distance。

这种设计让 BasicSearcher 只认识“已打分 seed”，不认识 dense/sparse 组件，避免 generic searcher 与 hybrid datacell 耦合。

`precomputed_seeds` 是默认空的通用 plumbing，不是对 baseline searcher 算法的优化。HNSW、
SINDI 等 baseline 不设置它时必须走现有分支，不能增加额外打分、分配或条件内循环。为了与
当前多 entry 初始化严格对齐，seed 分支还必须：

1. 按输入顺序去重并标记 visited；
2. 所有 seed 都进入 candidate set，只有通过 filter 的 seed 进入 top candidates；
3. 保持 `ef` 截断、`hybrid_candidate_set_size` 和 distance/ID tie-break；
4. seed 为空时退回 `eps/ep`，不能返回空结果；
5. 明确定义统计：完整 hybrid seed 计算应计入 UHG 总 distance 数，但不能在 entry 和主图统计中重复计数。

这里同样不能要求 bitwise distance 一致：复用的是 entry 搜索阶段 kernel 得到的 dense distance，
而旧路径会按 entry list 再做一次 batch Query，两个 kernel/分组的浮点累加顺序可能不同。验收
应使用距离容差、entry 集合和端到端 Recall，而不是 `memcmp(float)`。

#### Computer 重复创建问题

如果 HybridIndex 为 entry 评分创建一个 HybridComputer，而 BasicSearcher 随后又为主图创建另一个 HybridComputer，会重复编码 sparse query，并使 scratch 无法贯穿整个搜索。

更完整但改动更大的方案是让 BasicSearcher 接收可选的预构建 computer：

```cpp
Search(...,
       const ComputerInterfacePtr& prepared_computer,
       ...);
```

建议分阶段处理：

- D1：允许创建两个 computer，先验证 distance 复用本身是否有可测收益；
- D2：只有 D1 有收益时，再让 HybridIndex 创建一次 computer，并传入 BasicSearcher，使 entry 和主图共用 scratch。

不要一开始同时扩大太多接口，否则很难定位收益来源。

### 6.7 visited 和 heap 内存

当前 dense-entry base-layer 使用 thread-local `visited`。它避免每 query 分配，但一个线程搜索多个大索引时会保留最大索引规模的数组。

相比改 visited，先给两个 `priority_queue` 的底层容器 reserve 是更低风险的改动。当前
`top_candidates` 最多保留 `ef` 个元素，`candidate_set` 的量级也通常与 `ef` 接近：

```cpp
using Candidate = std::pair<float, InnerIdType>;
using CandidateBuffer = Vector<Candidate>;
using CandidateQueue =
    std::priority_queue<Candidate, CandidateBuffer, std::less<Candidate>>;

CandidateQueue MakeCandidateQueue(uint64_t capacity, Allocator* allocator) {
    CandidateBuffer buffer(allocator);
    buffer.reserve(capacity);
    return CandidateQueue(std::less<Candidate>{}, std::move(buffer));
}
```

必须保留当前 `std::pair` 的 lexicographic 比较，不能换成只比较 distance 的 comparator，
否则相同 distance 下的 ID tie-break 会变化。`top_candidates` reserve `ef`；
`candidate_set` 可先 reserve `ef` 并记录 peak，确认不足后再调容量策略。这项可以与 batch
distance 同阶段做独立开关，通常比 visited pool 和 CSR 更容易验证。

第一阶段可以保留 thread-local visited，避免同时改变太多变量。后续可考虑：

- index-owned VisitedListPool；
- query context 从 pool 获取 visited list；
- visited tag 回绕和多索引切换的专项测试。

visited pool 应排在 batch distance、heap reserve 和 flat adjacency 之后。

### 6.8 序列化兼容

`DenseEntryHNSWGraph` 当前内部序列化版本为 1。引入 CSR 后建议升级为版本 2：

```text
version=1: 读取旧 vector-by-vector 格式，加载时 compact 成 CSR
version=2: 直接读取 offsets + flat neighbors
```

兼容策略：

- 新代码必须能读取已有 UHG hybrid index 中的 dense-entry v1；
- 读取 v1 后只改变内存表示，不要求立刻重建索引；
- 新建或重新序列化的 index 写 v2；
- 外层 hybrid index 序列化版本不必仅因为内部 dense-entry 子版本变化而升级，但应增加兼容测试。

阶段 A、C、D 不改变 dense-entry 序列化格式；只有阶段 B 需要版本 2。

### 6.9 dense-entry 测试要求

当前 DenseEntryHNSWGraph 缺少独立测试，应至少增加：

1. 从小型 HNSW index 加载并验证节点、层级和邻居一致。
2. scalar 与 batch 的 entry 集合/顺序一致，distance 在约定容差内；单独覆盖相等距离 tie。
3. v1 deserialize 后搜索结果一致。
4. v2 serialize/deserialize 后搜索结果一致。
5. CSR NeighborView 的 0 degree、最大 degree 和末节点边界。
6. 含 deleted/missing label 节点的兼容行为。
7. entry_bk、entry_ef 的边界：`topk < ef`、`topk == ef`、`topk > k`。
8. distance 复用开关前后逐 query 结果对照，并确认总体 Recall 无统计显著下降。

建议增加阶段统计：

```text
dense_entry_upper_dist_cmp
dense_entry_base_dist_cmp
dense_entry_hops
dense_entry_batch_calls
dense_entry_elapsed_us
dense_entry_rerank_elapsed_us
```

## 7. 四项改动的组合关系

```text
Sparse DirectRead
      │
      ├── 降低所有 exact sparse distance 的读取/分配成本
      │
HybridComputer scratch
      │
      ├── 复用 sparse selection / result / code-ref buffer
      ├── 支持 sparse batch resolve + prefetch
      └── 支持 entry rerank 与主图复用搜索上下文

Dense-entry batch + CSR
      │
      ├── 降低 UHGH entry HNSW 本身的成本
      └── 返回 ID + dense distance
                  │
                  └── QueryWithPrecomputedDense
                         └── 避免 UHGH entry dense distance 重算
```

优化 1 和 scratch 互补，但 zero-copy 不依赖 scratch；dense-entry batch/CSR 也可以独立实施。应通过分阶段 A/B 测试避免把多个收益混在一起。

## 8. 推荐实施顺序

### Phase 0：仅增加统计

先获得：

```text
每 query entry 时间
每 query main graph 时间
exact sparse 次数
sparse prune 次数
hybrid query 调用次数
scratch/allocator 分配次数
dense-entry distance 次数
```

否则无法验证原计划中 30–50% 等收益估算。

### Phase 1：Sparse DirectRead

- 修改 GetCodesById；
- 统一 Release 契约；
- 不改格式；
- 单独跑 UHG/FHG A/B。

### Phase 2：HybridComputer scratch

- 先完成 result buffer 复用和 Vector capacity 复用；
- 不同时改变剪枝公式；
- 检查 Recall 完全一致；
- 统计分配次数和 QPS。

### Phase 3：Sparse batch resolve/prefetch

- 建立在 DirectRead 和 scratch 上；
- 通过 hybrid/sparse 扩展接口显式传递 `SparseReadScratch`，不修改通用 Flatten 接口；
- 用 perf 验证 LLC miss、cycles 和 QPS；
- 无收益则不保留复杂流水线。

### Phase 4：Dense-entry batch distance

- 暂不改存储和接口；
- scalar 与 batch 路径做开关 A/B；
- heap reserve 使用单独开关，避免与 batch 收益混在一起；
- 只跑配置外部 `dense_entry_hnsw_graph_path` 的 UHGH。

### Phase 5：Dense-entry level-0 CSR

- 加内部序列化 v2；
- 兼容读取 v1；
- 测 index load time、RSS、entry QPS。

### Phase 6：Dense distance 复用

- 返回 scored entry candidates；
- 外部 HNSW 和 fallback dense-entry 两条路径统一返回结构；
- 增加 QueryWithPrecomputedDense；
- 先允许两个 computer 验证收益；
- 有明确收益后再共用 prepared computer/scratch。

## 9. 性能验证矩阵

### 9.1 微基准

| 对象 | 变量 | 指标 |
|---|---|---|
| SparseVectorDataCell | id_count=1/16/32/64/256/1000 | ns/vector、allocs、bytes copied |
| HybridVectorDataCell | prune/no-prune、不同通过率 | ns/batch、allocs、exact sparse count |
| DenseEntryHNSWGraph | entry_ef=100/200/500/1000 | entry us、dist cmp、cache miss、RSS |

### 9.2 端到端

至少跑：

```text
dataset: nq, msmarco
alpha:   0.3(UHGS), 0.7(UHGH), mixed-alpha
ef:      100, 300, 500, 1000
threads: 1，另补并发吞吐测试
```

判断标准：

- 本文四项原则上不改变搜索语义；batch SIMD 允许末位距离差异，Recall 不应出现统计显著下降；
- 不能只比较相同 ef 的单点，应同时检查完整 Recall-QPS 曲线；
- UHG 和 FHG 都需验证 sparse/scratch 改动；
- dense-entry 改动只需验证 UHGH，确认 baseline 文件和结果未变化；
- 每项至少重复 3 次，固定 CPU、NUMA、warmup 和数据缓存状态。

## 10. 验收条件

### 10.1 Sparse DirectRead

- MemoryIO 搜索期每个 sparse vector 不再 Allocate/Deallocate code buffer；
- 不复制完整 sparse code；
- 跨 block fallback 正确；
- 所有 sparse distance 测试通过。

### 10.2 HybridComputer scratch

- 每个 hop 不再创建新的 dense/sparse/selection Vector capacity；
- pruning 分支不再创建未使用的全量 sparse_dists；
- 多线程搜索无共享 scratch；
- search allocator 生命周期正确。

### 10.3 Dense-entry batch/CSR

- 不修改 baseline HNSW 源码和索引；
- entry 集合和顺序与原实现一致；
- scalar/batch distance 在约定浮点容差内，tie 行为经过单独验证；
- 新代码可加载 dense-entry v1；
- level-0 邻接不再是 per-node vector allocation；
- batch 路径确实进入 FlattenDataCell 的批量/prefetch 实现。

### 10.4 Dense distance 复用

- UHGH entry 初始化不再重复计算已有 dense distance；
- 完整 hybrid entry distance 与原实现一致；
- 如果端到端收益落在测试噪声范围内，则不合入 D2 的 prepared-computer 接口扩展。

## 11. 最终建议

四项中推荐优先级为：

```text
P0  Sparse DirectRead + Release 契约修正
P0  HybridComputer scratch + result buffer 复用
P1  Dense-entry 邻居 batch distance + heap reserve（分别 A/B）
P1  Dense-entry level-0 CSR（profile 证明 pointer chasing 后再做）
P2  Sparse batch resolve/prefetch（以 perf 结果决定）
P2  Dense-entry distance 复用（以独立 A/B 结果决定是否扩大接口）
```

这个优先级只是在本文请求的四项内部排序，不代表 dense-entry 是 UHG 的全局第一瓶颈。
已有实验显示扩大 entry `ef/bk` 主要是在移动 Recall-QPS 曲线，entry 阶段占比仍应先由计时
确认；如果占比很低，P1 的 batch/CSR 即使局部收益明显，端到端收益也可能很小。

不建议实施原方案中的通用单 TLS code buffer，也不建议为了消除 offset lookup 把 sparse vector 改成定长桶。当前 UHG 的 MemoryIO 和一次搜索一个 HybridComputer 的生命周期已经提供了更简单、更安全的优化基础。
