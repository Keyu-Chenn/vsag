# Hybrid Index 使用说明

本文档介绍 vsag 库中 Hybrid Index 的设计原理、使用方法及实验对比方案。

---

## 1. 概述

Hybrid Index 是一种用于**混合向量检索**的索引结构，同时支持 dense 向量和 sparse 向量的混合距离计算。适用于需要结合语义相似性（dense）和关键词匹配（sparse）的场景，如多模态检索、文本+语义混合搜索等。

### 核心特性

- **混合距离计算**: 支持 `alpha * dense_dist + (1-alpha) * sparse_dist` 的加权组合
- **动态权重调整**: 搜索时可动态设置 `alpha` 参数，无需重建索引
- **图索引结构**: 基于 HNSW/HGraph 的图结构，支持高效近似搜索
- **稀疏距离优化**: 支持预计算稀疏距离表，减少在线计算开销

---

## 2. 架构设计

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────┐
│                     HybridIndex                          │
│  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │  HybridVector   │  │      GraphInterface         │   │
│  │    DataCell     │  │    (HNSW-like graph)        │   │
│  │ ┌───────┬─────┐ │  └─────────────────────────────┘   │
│  │ │ Dense │Sparse│ │                                    │
│  │ │ Cell  │ Cell │ │  ┌─────────────────────────────┐   │
│  │ └───────┴─────┘ │  │      BasicSearcher           │   │
│  └─────────────────┘  │    (Graph traversal +        │   │
│                       │     hybrid pruning)          │   │
│                       └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**文件位置**:
- `src/algorithm/hybrid_index/hybrid_index.h/cpp`: 主索引类
- `src/datacell/hybrid_vector_datacell.h/cpp`: 混合向量存储与距离计算
- `src/impl/searcher/basic_searcher.cpp`: 图搜索算法（含混合剪枝）

### 2.2 混合距离计算

```cpp
// HybridComputer 计算混合距离
hybrid_dist = alpha * dense_dist + (1 - alpha) * sparse_dist

// 其中:
// dense_dist: dense 向量的内积距离 (1 - IP)
// sparse_dist: sparse 向量的内积距离 (1 - IP)
```

### 2.3 搜索剪枝策略

Hybrid Index 在图遍历过程中采用**动态剪枝策略**以减少稀疏距离计算:

1. **Dense First**: 先计算 dense 距离（计算成本低）
2. **Threshold Pruning**: 使用当前 top-k 的下界距离作为阈值
3. **Sparse Skip**: 若 `alpha * dense_dist + (1-alpha) * upper_bound` 已超过阈值，跳过 sparse 计算

注：也可以用稀疏距离去prune稠密距离

相关参数:
- `hybrid_prune_scale`: 剪枝阈值缩放因子，控制剪枝激进程度

---

## 3. 使用方法

### 3.1 创建索引

```cpp
#include <vsag/vsag.h>

vsag::init();

// 构建参数
std::string build_params = R"(
{
    "dtype": "float32",
    "metric_type": "ip",
    "dim": 1024,
    "index_param": {
        "sparse_dtype": "float32",
        "sparse_metric_type": "ip",
        "sparse_dim": 10000,
        "alpha": 0.5,           // 建图时使用的权重
        "max_degree": 32,       // 图最大度数
        "ef_construction": 200  // 建图 ef 参数
    }
}
)";

auto index = vsag::Factory::CreateIndex("hybrid_index", build_params).value();
```

### 3.2 构建索引

```cpp
// 准备数据
int64_t num_vectors = 10000;
int64_t dim = 1024;
std::vector<float> dense_data(num_vectors * dim);
std::vector<vsag::SparseVector> sparse_data(num_vectors);
std::vector<int64_t> ids(num_vectors);

// ... 填充数据 ...

auto dataset = vsag::Dataset::Make();
dataset->NumElements(num_vectors)
       ->Dim(dim)
       ->Ids(ids.data())
       ->Float32Vectors(dense_data.data())
       ->SparseVectors(sparse_data.data())
       ->Owner(false);

index->Build(dataset);
```
注：实际构图是在InsertNeighbors函数中直接读取提前构建的混合图的邻接表

### 3.3 搜索

```cpp
// 准备查询
auto query = vsag::Dataset::Make();
query->NumElements(1)
      ->Dim(dim)
      ->Float32Vectors(query_dense.data())
      ->SparseVectors(query_sparse.data())
      ->Owner(false);

// 搜索参数
std::string search_params = R"(
{
    "alpha": 0.5,            // 搜索时的权重（可与建图时不同）
    "ef_search": 100,        // 搜索 ef 参数
    "entry_point": 0,        // 入口点
    "hybrid_prune_scale": 1.0  // 剪枝缩放因子
}
)";

int64_t topk = 10;
auto result = index->KnnSearch(query, topk, search_params).value();

// 读取结果
for (int64_t i = 0; i < result->GetDim(); ++i) {
    std::cout << "id: " << result->GetIds()[i]
              << ", dist: " << result->GetDistances()[i] << std::endl;
}
```


---

## 4. 实验设计: Hybrid Index vs Two-Route Baseline

### 4.1 实验背景

对比 Hybrid Index 与 Two-Route Baseline 在混合检索场景下的性能差异。

**Two-Route Baseline**:
- Hgraph做稠密和稀疏向量的两路检索，合并后重排

**Hybrid Index**:
- 单次图遍历，混合距离引导搜索，无需重排

### 4.2 实验脚本

实验脚本位于 `vsag/examples/` 目录:

| 文件                                 | 功能                         |
|------------------------------------|----------------------------|
| `examples/cpp/603_hybrid_exp3.cpp` | Two-Route Baseline 实验      |
| `examples/cpp/605_hybrid_exp5.cpp` | Hybrid Index 实验            |


### 4.3 实验运行

```bash
# 编译
make release
/tbase-project/vsag/build-release/examples/cpp/603_hybrid_exp3 /tbase-project/vsag/scripts/hybrid_index/data/models/data/msmarco_gt/msmarco_alpha_0_5_k_200.hdf5 -k 100 --bk 200 --alpha 0.5 --ef_search 200
/tbase-project/vsag/build-release/examples/cpp/605_hybrid_exp5 /tbase-project/vsag/scripts/hybrid_index/data/models/data/msmarco_gt/msmarco_alpha_0_5_k_200.hdf5 -k 100 --sindi_bk 200 --ef_search 200 --alpha 0.5 --hybrid_prune_scale 1
```

参数说明:
- `-k`: 最终返回的 top-k 数量
- `--bk`: 第一阶段召回数量（候选集大小）
- `--alpha`: 混合权重（实验1/2中固定为极端值）



## 5. 文件索引

### 核心源码

- `src/algorithm/hybrid_index/hybrid_index.h` - Hybrid Index 类定义
- `src/algorithm/hybrid_index/hybrid_index.cpp` - Hybrid Index 实现
- `src/datacell/hybrid_vector_datacell.h` - 混合向量存储类
- `src/datacell/hybrid_vector_datacell.cpp` - 混合距离计算实现
- `src/impl/searcher/basic_searcher.cpp` - 图搜索算法（含混合剪枝）
- `src/impl/inner_search_param.h` - 搜索参数定义

### 示例代码

- `examples/cpp/110_index_hybrid.cpp` - Hybrid Index 基本使用示例
- `examples/cpp/601-608_hybrid_exp*.cpp` - 混合检索实验代码
