# UHG 实验流程

## 1. 背景

**UHG (Unified Hybrid Graph)** 是一种混合向量检索索引，同时利用 dense 向量（语义相似性）和 sparse 向量（关键词匹配），通过单次图遍历完成混合距离搜索，无需两路检索+重排。

混合距离公式：

```
hybrid_dist = alpha * dense_dist + (1 - alpha) * sparse_dist
```

### 对比方法

| 方法 | 说明 |
|------|------|
| **HNSW+SINDI** | Two-Route Baseline：稠密/稀疏分别检索，合并候选集后重排 |
| **FHG** | Fixed Hybrid Graph：固定 alpha 建图的混合索引 |
| **UHG** | 多 alpha 联合建图的混合索引 |
| **UHGS** | UHG + SINDI 引导入口点的搜索策略 |
| **Refined UHG** | UHG + 反向边 + RNG 剪枝 + 连通性修复（最终方法） |
| **Refined UHGS** | Refined UHG + SINDI 引导入口点 |

### 消融实验设计

- **UHG vs Refined UHG**：验证 refine 阶段（反向边 + RNG 剪枝 + 连通修复）的效果
- **UHG vs UHGS**：验证 SINDI 引导入口点的效果
- **UHG vs FHG**：验证多 alpha 联合建图 vs 固定 alpha 建图的效果
- **UHG vs HNSW+SINDI**：验证单图混合搜索 vs 两路检索+重排的效果

---

## 2. 数据准备

### 2.1 输入数据格式

HDF5 文件，包含以下数据集：

```
train           - 稠密训练向量 (N × dim)
test            - 稠密查询向量 (Q × dim)
train_sparse    - 稀疏训练向量 (N × sparse_dim)
test_sparse     - 稀疏查询向量 (Q × sparse_dim)
train_labels    - 训练集标签
test_labels     - 查询集标签
```

### 2.2 已有数据集

| 数据集 | HDF5 路径 |
|--------|-----------|
| NQ | `scripts/UHG/data/hdf5/nq.hdf5` |
| MSMARCO | `scripts/UHG/data/hdf5/msmarco.hdf5` |
| HotpotQA | `scripts/UHG/data/hdf5/hotpotqa.hdf5` |
| Quora | `scripts/UHG/data/hdf5/quora.hdf5` |
| Webis-Touche2020 | `scripts/UHG/data/hdf5/webis-touche2020.hdf5` |

---

## 3. 预处理

### 3.1 计算距离矩阵

计算稠密和稀疏的 train-by-test 距离矩阵（耗时，大矩阵）：

```bash
python scripts/UHG/scripts/compute_distances.py \
  --input scripts/UHG/data/hdf5/nq.hdf5 \
  --output scripts/UHG/data/distances \
  --dense-metric ip \
  --batch-size 10000 \
  --sparse-batch-size 10000
```

输出：

```
scripts/UHG/data/distances/<dataset>_dense_distances.npy
scripts/UHG/data/distances/<dataset>_sparse_distances.npy
```

### 3.2 计算 Ground Truth

基于距离矩阵，按 alpha 值计算精确混合距离的 top-k ground truth：

```bash
# k=100
python scripts/UHG/scripts/compute_ground_truth.py \
  --distances scripts/UHG/data/distances \
  --output scripts/UHG/data/ground_truth/nq \
  --dataset nq --topk 100

# k=1000
python scripts/UHG/scripts/compute_ground_truth.py \
  --distances scripts/UHG/data/distances \
  --output scripts/UHG/data/ground_truth/nq_1000 \
  --dataset nq --topk 1000
```

输出格式：`<dataset>_ground_truth_alpha_<alpha>.npy`，alpha 从 0.0 到 1.0 步长 0.1。

---

## 4. 构建辅助索引

701/702/706 实验会自动构建或加载缓存的辅助索引：

| 索引 | 缓存文件名 | 用途 |
|------|-----------|------|
| Dense HGRAPH | `701_<dataset>_dense_hgraph.index` | 702/706 图生成时获取 dense 候选 |
| Dense HNSW | `701_<dataset>_dense_hnsw.index` | 701 基线搜索 |
| Sparse SINDI | `701_<dataset>_sparse_sindi.index` | 701 基线搜索 + 702/706 图生成 |

索引缓存目录：`scripts/UHG/data/index/`

---

## 5. 生成混合图（离线）

### 5.1 FHG 图（固定 alpha）

FHG 图在建图时使用固定 alpha 值，已预计算并存储在 `scripts/UHG/data/fhg/` 目录。

### 5.2 UHG 图（多 alpha 联合）— 消融实验用

使用 **702_uhg_exp2** 生成基础 UHG 图（无 refine）：

```bash
./build-release/examples/cpp/702_uhg_exp2 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --index_dir scripts/UHG/data/index \
  --threads 16 \
  -k 32 \
  --bk 500 \
  --alpha_step 0.1 \
  --output scripts/UHG/data/uhg/nq_uhg_v2.h5
```

流程：
1. 加载 dense HGRAPH 和 sparse SINDI 辅助索引
2. 对每个基点，分别获取 dense 和 sparse 候选池
3. 对候选并集，在多个 alpha 值（0.0, 0.1, ..., 1.0）下评估
4. 合并所有 alpha 下出现在 top-k 的邻居
5. 输出 HDF5 文件（`neighbors` + `neighbor_counts`）

### 5.3 Refined UHG 图（最终方法）

使用 **706_uhg_exp6** 生成 refined UHG 图：

```bash
./build-release/examples/cpp/706_uhg_exp6 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --index_dir scripts/UHG/data/index \
  --threads 16 \
  -k 32 \
  --bk 500 \
  --alpha_step 0.1 \
  --refine_max_degree 64 \
  --refine_alpha 0.5 \
  --refine_rng 1.2 \
  --output scripts/UHG/data/uhg/nq_uhg_refined.h5
```

流程：
1. 同 702 的候选获取和多 alpha 评估
2. 合并邻居
3. **Refine 阶段**（相比 702 新增）：
   - 添加反向边
   - RNG 风格剪枝（`refine_rng` 控制剪枝激进程度）
   - 连通性修复
4. 输出 HDF5 文件

Refine 关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `refine_max_degree` | 64 | refine 后最大度数 |
| `refine_alpha` | 0.5 | refine 时使用的混合权重 |
| `refine_rng` | 1.2 | RNG 剪枝阈值 |

---

## 6. 构建 Hybrid Index

703 实验加载预计算的图到 HybridIndex 中。

**重要限制**：当前 `HybridIndex::add_one_point()` 中的图路径是硬编码的。若需重建索引，需：

1. 修改 `src/algorithm/hybrid_index/hybrid_index.cpp` 中的 `h5_file` 路径
2. 重新编译 `make release`
3. 运行 703 并加 `--rebuild` 参数
4. 将生成的索引复制到目标缓存路径

**推荐做法**：使用已有的序列化索引缓存，不传 `--rebuild`。

索引缓存目录：

| 场景 | 目录 |
|------|------|
| 基础实验（UHG/FHG） | `scripts/UHG/data/index/` |
| Refined NQ | `scripts/UHG/data/index_refined/` |
| Refined HotpotQA | `scripts/UHG/data/index_refined_hotpotqa/` |
| Refined Quora | `scripts/UHG/data/index_refined_quora/` |

---

## 7. 运行搜索实验

### 7.1 4-Method 对比实验（基础版，含消融）

对比 HNSW+SINDI / FHG / UHG / UHGS 四种方法：

**HotpotQA k=100：**

```bash
# 单个 alpha
bash scripts/UHG/scripts/run_hotpotqa_4methods.sh 0.3

# 全 alpha 扫描
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  bash scripts/UHG/scripts/run_hotpotqa_4methods.sh "$alpha"
done
```

**NQ k=100：**

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  bash scripts/UHG/scripts/run_nq_4methods.sh "$alpha"
done
```

**Quora k=100：**

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  bash scripts/UHG/scripts/run_quora_4methods.sh "$alpha"
done
```

脚本内部方法映射：

| 方法标签 | 二进制 | 索引来源 |
|----------|--------|----------|
| `hnsw_sindi` | 701_uhg_exp1 | 独立 HNSW + SINDI |
| `fhg` | 703_uhg_exp3 | 复制 `703_<dataset>_fhg_hybrid_index.index` |
| `uhg` | 703_uhg_exp3 | 复制 `703_<dataset>_uhg_hybrid_index.index` |
| `uhgs` | 703_uhg_exp3 --method uhgs | 同 UHG 索引 + SINDI 入口点 |

输出：`scripts/UHG/results/<dataset>_4methods/alpha_<alpha>.txt`

### 7.2 Refined UHG 实验（最终方法）

**NQ Refined k=100：**

```bash
bash scripts/UHG/scripts/run_exp_nq_refined_uhg.sh
```

**HotpotQA Refined k=100：**

```bash
bash scripts/UHG/scripts/run_exp_hotpotqa_refined_uhg.sh
bash scripts/UHG/scripts/run_exp_hotpotqa_refined_uhg_extra.sh
```

这些脚本使用 refined 索引目录，运行 UHG 和 UHGS 方法，遍历多个 alpha 和 ef_search 值。

### 7.3 k=1000 实验

```bash
# NQ k=1000
bash scripts/UHG/scripts/run_nq_4methods_k1000.sh 0.3

# HotpotQA k=1000
bash scripts/UHG/scripts/run_hotpotqa_4methods_k1000.sh 0.3
```

### 7.4 Smoke Test

正式实验前先跑小规模验证：

```bash
# 701 基线
./build-release/examples/cpp/701_uhg_exp1 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --gt_dir scripts/UHG/data/ground_truth/nq \
  --index_dir scripts/UHG/data/index \
  --alpha 0.3 --num_queries 10 -k 100 --bk 500 --ef_search 500

# 703 UHG
./build-release/examples/cpp/703_uhg_exp3 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --method uhg -k 100 --alpha 0.3 \
  --gt_dir scripts/UHG/data/ground_truth/nq \
  --index_dir scripts/UHG/data/index \
  --hybrid_prune_scale 0.5 --max_hops 0 \
  --sindi_bk 100 --ef_search 500 --num_queries 10

# 703 UHGS
./build-release/examples/cpp/703_uhg_exp3 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --method uhgs -k 100 --alpha 0.3 \
  --gt_dir scripts/UHG/data/ground_truth/nq \
  --index_dir scripts/UHG/data/index \
  --hybrid_prune_scale 0.7 --max_hops 0 \
  --sindi_bk 1000 --ef_search 500 --num_queries 10
```

---

## 8. 绘图

解析结果文本文件，生成 Recall-QPS 曲线图：

```bash
# HotpotQA 4-method
python scripts/UHG/scripts/plot_qps_recall_hotpotqa_4methods.py

# NQ 4-method
python scripts/UHG/scripts/plot_qps_recall_nq_4methods.py

# Quora 4-method
python scripts/UHG/scripts/plot_qps_recall_quora_4methods.py
```

输出：PNG 和 PDF 格式的 Recall-QPS 曲线图。

---

## 9. 实验二进制功能总结

| 二进制 | 功能 | 输出 | 是否注册到 CMake |
|--------|------|------|------------------|
| **701_uhg_exp1** | HNSW+SINDI 两路基线搜索 | Recall/QPS | 是 |
| **702_uhg_exp2** | 基础 UHG 图生成（无 refine） | HDF5 图文件 | 是 |
| **703_uhg_exp3** | UHG/UHGS/FHG 搜索实验 | Recall/QPS | 是 |
| **704_uhg_exp4** | — | — | 否 |
| **705_uhg_exp5** | — | — | 否 |
| **706_uhg_exp6** | Refined UHG 图生成 | HDF5 图文件 | 是 |
| **707_uhg_exp7** | — | — | 否 |

---

## 10. 关键注意事项

1. **必须从 `/tbase-project/vsag` 目录运行**，脚本使用相对路径
2. **不要并行运行同一数据集的 4-method 脚本**，会因索引文件复制产生竞争
3. **703 的 `--rebuild` 有风险**：图路径硬编码在源码中，重建前需确认路径正确
4. **先跑 smoke test**（`--num_queries 10`），再跑全量实验
5. **Refine 是最终方法**，UHG（无 refine）作为消融实验的对比项

---

## 11. 完整实验执行清单

### 11.1 编译

```bash
cd /tbase-project/vsag
make release
```

### 11.2 预处理（如已有可跳过）

```bash
# 距离矩阵
python scripts/UHG/scripts/compute_distances.py \
  --input scripts/UHG/data/hdf5/<dataset>.hdf5 \
  --output scripts/UHG/data/distances \
  --dense-metric ip

# Ground truth
python scripts/UHG/scripts/compute_ground_truth.py \
  --distances scripts/UHG/data/distances \
  --output scripts/UHG/data/ground_truth/<dataset> \
  --dataset <dataset> --topk 100
```

### 11.3 生成图（如已有可跳过）

```bash
# UHG 图（消融用）
./build-release/examples/cpp/702_uhg_exp2 \
  scripts/UHG/data/hdf5/<dataset>.hdf5 \
  --index_dir scripts/UHG/data/index \
  --threads 16 -k 32 --bk 500 --alpha_step 0.1 \
  --output scripts/UHG/data/uhg/<dataset>_uhg_v2.h5

# Refined UHG 图（最终方法）
./build-release/examples/cpp/706_uhg_exp6 \
  scripts/UHG/data/hdf5/<dataset>.hdf5 \
  --index_dir scripts/UHG/data/index \
  --threads 16 -k 32 --bk 500 --alpha_step 0.1 \
  --refine_max_degree 64 --refine_alpha 0.5 --refine_rng 1.2 \
  --output scripts/UHG/data/uhg/<dataset>_uhg_refined.h5
```

### 11.4 运行搜索实验

```bash
# 4-method 基础对比（含消融：UHG vs FHG vs HNSW+SINDI）
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  bash scripts/UHG/scripts/run_<dataset>_4methods.sh "$alpha"
done

# Refined UHG 实验（最终方法）
bash scripts/UHG/scripts/run_exp_<dataset>_refined_uhg.sh
```

### 11.5 绘图

```bash
python scripts/UHG/scripts/plot_qps_recall_<dataset>_4methods.py
```

---

## 12. 目录结构

```
scripts/UHG/
├── EXPERIMENT_MANUAL.md          实验操作手册
├── data/
│   ├── hdf5/                     输入 HDF5 数据集
│   ├── distances/                预计算距离矩阵 (.npy)
│   ├── ground_truth/             精确 ground truth (.npy)
│   │   ├── nq/                   NQ k=100
│   │   ├── nq_1000/              NQ k=1000
│   │   ├── hotpotqa/             HotpotQA k=100
│   │   ├── hotpotqa_1000/        HotpotQA k=1000
│   │   ├── quora/                Quora k=100
│   │   ├── quora_1000/           Quora k=1000
│   │   ├── webis-touche2020/     Webis k=100
│   │   └── webis-touche2020_1000/ Webis k=1000
│   ├── index/                    基础索引缓存（UHG/FHG/HNSW/SINDI）
│   ├── index_refined/            Refined NQ 索引缓存
│   ├── index_refined_hotpotqa/   Refined HotpotQA 索引缓存
│   ├── index_refined_quora/      Refined Quora 索引缓存
│   ├── fhg/                      FHG 预计算图 (.h5)
│   ├── uhg/                      UHG 预计算图 (.h5)
│   └── raw/                      原始 parquet 数据
├── scripts/
│   ├── compute_distances.py      距离矩阵计算
│   ├── compute_ground_truth.py   Ground truth 计算
│   ├── run_<dataset>_4methods.sh 4-method 实验脚本
│   ├── run_exp_<dataset>_refined_uhg.sh Refined 实验脚本
│   ├── plot_qps_recall_<dataset>_4methods.py 绘图脚本
│   └── analyze_topk_overlap.py   Top-k 重叠分析
├── results/
│   ├── <dataset>_4methods/       4-method 结果 (alpha_*.txt + 图)
│   └── topk_overlap/             Top-k 重叠分析结果
└── logs/                         构建日志
```
