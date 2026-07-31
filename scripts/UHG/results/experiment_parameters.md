# UHG 实验参数设置汇总

本文档汇总了三类实验的参数设置，数据来源为各实验目录下的结果文件头注释（`#` 开头）及运行脚本（`scripts/UHG/scripts/hybrid_union/`）。

---

## 一、全局公共设置

以下设置在三类实验中保持一致：

| 配置项 | 值 |
|---|---|
| 仓库根目录 | `/tbase-project/vsag` |
| 数据目录 | `scripts/UHG/data` |
| 数据集 | `nq`, `hotpotqa`, `msmarco` |
| top-k | `100` |
| query 数 | `-1`（全量 test query） |
| threads | `1` |
| 指标 | Recall@100, QPS |
| Ground truth 路径 | `scripts/UHG/data/ground_truth/<dataset>/<dataset>_ground_truth_alpha_<alpha>.npy` |
| Ground truth 公式 | `score = alpha * dense_inner_product + (1 - alpha) * sparse_inner_product` |
| build_alpha | `0.5` |

---

## 二、主实验（results/main）

### 2.1 实验概述

主实验对比 5 种方法在 3 个数据集 × 5 个 alpha 下的 Recall-QPS 表现。

| 配置项 | 值 |
|---|---|
| 结果目录 | `scripts/UHG/results/main` |
| 方法 | `hnsw`, `sindi`, `hnsw_sindi`, `fhg`, `uhg` |
| alpha | `0.3`, `0.4`, `0.5`, `0.6`, `0.7` |

### 2.2 各方法参数

#### 2.2.1 HNSW（Dense-only 基线）

| 配置项 | 值 |
|---|---|
| 二进制 | `build-release/examples/cpp/708_uhg_exp8` |
| 脚本 | `scripts/UHG/scripts/hybrid_union/main/run_baselines.sh` |
| bk = ef_search | `100, 200, 300, 500, 800, 1000` |
| 索引目录 | `scripts/UHG/data/index`（`701_<dataset>_dense_hnsw.index`） |
| 高 recall 补点 | `1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000`（按需追加，target recall > 0.95） |

**流程**：HNSW 检索 top-bk dense 候选 → sparse 距离用原始 sparse 向量重新计算 → `alpha * dense_score + (1-alpha) * sparse_score` 重排取 top-100。

#### 2.2.2 SINDI（Sparse-only 基线）

| 配置项 | 值 |
|---|---|
| 二进制 | `build-release/examples/cpp/709_uhg_exp9` |
| 脚本 | `scripts/UHG/scripts/hybrid_union/main/run_baselines.sh` |
| bk | `100, 200, 300, 500, 800, 1000` |
| 索引目录 | `scripts/UHG/data/index`（`701_<dataset>_sparse_sindi.index`） |
| 高 recall 补点 | `1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000`（按需追加） |

**流程**：SINDI 检索 top-bk sparse 候选 → dense 距离用原始 dense 向量重新计算 → 按 hybrid score 重排取 top-100。

#### 2.2.3 HNSW+SINDI（融合基线）

| 配置项 | 值 |
|---|---|
| 二进制 | `build-release/examples/cpp/701_uhg_exp1` |
| 脚本 | `scripts/UHG/scripts/hybrid_union/main/run_baselines.sh` |
| bk = ef_search | `100, 150, 200, 300, 500` |
| 索引目录 | `scripts/UHG/data/index` |

**流程**：HNSW 检索 top-bk dense + SINDI 检索 top-bk sparse → 合并候选 → 复用已有距离，缺失侧补算 → 按 hybrid score 重排取 top-100。

#### 2.2.4 FHG

| 配置项 | 值 |
|---|---|
| 二进制 | `build-release/examples/cpp/703_uhg_exp3` |
| 脚本 | `scripts/UHG/scripts/hybrid_union/main/run_fhg.sh` |
| method | `uhg`（直接在 FHG 主图上搜索） |
| ef_search | `100, 200, 300, 500, 800, 1000` |
| hybrid_prune_scale | `1`（不剪枝） |
| max_hops | `0`（不限制） |
| sindi_bk | `100` |
| build_alpha | `0.5` |
| 图构建参数 | `bk=100 query_prune_ratio=0.9 refine=none fixed_alpha=0.5`（主图邻居数 k=64） |
| 图路径 | `scripts/UHG/data/fhg/<dataset>_fhg_alpha_0_5.h5` |
| 索引路径 | `scripts/UHG/data/index/703_<dataset>_fhg_hybrid_index.index` |
| 特殊标志 | `--disable_sindi --disable_dense_entry`（不使用 sparse/dense entry） |

#### 2.2.5 UHG（本文方法）

| 配置项 | 值 |
|---|---|
| 二进制 | `build-release/examples/cpp/703_uhg_exp3` |
| 脚本 | `scripts/UHG/scripts/hybrid_union/main/run_uhg.sh` |
| method | `auto` |
| ef_search | `100, 200, 300, 500, 1000` |
| sindi_bk | `= ef_search` |
| dense_entry_bk | `= ef_search` |
| dense_entry_ef_search | `= ef_search` |
| hybrid_prune_scale | `0.3` |
| max_hops | `0`（不限制） |
| sindi_query_prune_ratio | `0.5` |
| sindi_term_prune_ratio | `0` |
| build_alpha | `0.5` |
| 图构建参数 | `bk=100 query_prune_ratio=0.9 refine=none merge_max_degree=64` |
| UHG 主图路径 | `scripts/UHG/data/uhg/<dataset>_uhg.h5` |
| Dense entry HNSW | `scripts/UHG/data/index/701_<dataset>_dense_hnsw.index` |
| Hybrid index 路径 | `scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index` |
| SINDI index | 不使用外部 703 SINDI（index 内部已嵌入） |

**auto 入口策略规则**：

| alpha | 实际方法 | 入口策略 |
|---:|---|---|
| 0.3, 0.4, 0.5 | `uhgs` | SINDI 找 sparse entry ids，再搜 UHG 主图 |
| 0.6, 0.7 | `uhgh` | Dense-entry HNSW 找 dense entry，再搜 UHG 主图 |

### 2.3 主实验结果文件结构

```
results/main/
├── hotpotqa/
│   ├── hnsw/alpha_{0.3,0.4,0.5,0.6,0.7}.txt
│   ├── sindi/alpha_{...}.txt
│   ├── hnsw_sindi/alpha_{...}.txt
│   ├── fhg/alpha_{...}.txt
│   ├── uhg/alpha_{...}.txt
│   └── plots/
├── msmarco/  (同上)
├── nq/       (同上)
└── logs/
```

---

## 三、消融实验 1 — 入口策略（results/ablation）

### 3.1 实验意图

将入口策略与 alpha 解耦：对**每一个** alpha 都跑全量三种方法（`uhg`/`uhgs`/`uhgh`），观察固定 alpha 下不同入口策略的差异。三个方法加载**同一个** UHG `hybrid_index`，命令行只有 `--method` 不同，确保差异只来自入口策略。

### 3.2 参数设置

| 配置项 | 值 |
|---|---|
| 结果目录 | `scripts/UHG/results/ablation` |
| 二进制 | `build-release/examples/cpp/703_uhg_exp3` |
| 脚本 | `scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh` |
| 数据集 | `nq`, `hotpotqa`（msmarco 目录为空） |
| 方法 | `uhg`, `uhgs`, `uhgh` |
| alpha | `0.3`, `0.4`, `0.5`, `0.6`, `0.7` |
| top-k | `100` |
| num_queries | `-1`（全量） |
| threads | `1` |
| ef_search | `100, 200, 300, 500, 1000` |
| sindi_bk | `= ef_search` |
| dense_entry_bk | `= ef_search` |
| dense_entry_ef_search | `= ef_search` |
| hybrid_prune_scale | `0.5` |
| max_hops | `0`（不限制） |
| sindi_query_prune_ratio | `0.5` |
| sindi_term_prune_ratio | `0` |
| build_alpha | `0.5` |
| 固定索引 | `scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index` |

### 3.3 与主实验的关键差异

| 差异项 | 主实验 UHG | 消融实验 1 |
|---|---|---|
| method | `auto`（按 alpha 自适应） | 显式指定 `uhg` / `uhgs` / `uhgh` |
| hybrid_prune_scale | `0.3` | `0.5` |
| 传参方式 | 传 `--graph_path`、`--dense_entry_hnsw_graph_path` 等构建期参数 | 不传构建期参数，直接 `Deserialize` 加载已建好的 hybrid_index |
| 索引 | 每次调用时加载/可能 rebuild | 三种方法加载同一索引，无 rebuild |

### 3.4 三种入口策略说明

| 方法 | 入口策略 | 行为 |
|---|---|---|
| `uhg` | 无特殊入口 | 直接从 UHG 主图默认入口点搜索 |
| `uhgs` | Sparse entry | 使用 index 内部 SINDI 检索 top `sindi_bk`，取结果 ids 作为 UHG 主图入口点 |
| `uhgh` | Dense entry | 使用 index 内部 dense-entry HNSW 检索，找到 dense entry 后搜 UHG 主图 |

### 3.5 结果文件结构

```
results/ablation/
├── hotpotqa/
│   ├── uhg/alpha_{0.3,...,0.7}.txt
│   ├── uhgs/alpha_{...}.txt
│   └── uhgh/alpha_{...}.txt
├── nq/  (同上)
├── msmarco/  (空)
└── logs/
```

---

## 四、消融实验 2 — 剪枝强度（results/test_prune）

### 4.1 实验意图

固定 `alpha` 和 `--method`，只切换 `hybrid_prune_scale`，衡量不同剪枝强度对 Recall-QPS 的影响。两组/三组配置加载**同一个** UHG `hybrid_index`，命令行只有 `--hybrid_prune_scale` 不同。

### 4.2 hybrid_prune_scale 语义

| hybrid_prune_scale | 含义 |
|---:|---|
| 0 | 仅 sparse（最激进） |
| < 1 | 更激进剪枝 |
| 1 | 不剪枝（baseline） |

### 4.3 参数设置

| 配置项 | 值 |
|---|---|
| 结果目录 | `scripts/UHG/results/test_prune` |
| 二进制 | `build-release/examples/cpp/703_uhg_exp3` |
| 脚本 | `scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh` |
| 数据集 | `nq`（hotpotqa、msmarco 目录为空） |
| alpha | `0.5`（固定） |
| method | `uhgs`（固定） |
| top-k | `100` |
| num_queries | `-1`（全量） |
| threads | `1` |
| ef_search | `100, 200, 300, 500, 1000` |
| sindi_bk | `= ef_search` |
| dense_entry_bk | `= ef_search` |
| dense_entry_ef_search | `= ef_search` |
| hybrid_prune_scale | `0.3`, `0.5`, `1.0`（变量，三个取值） |
| max_hops | `0`（不限制） |
| sindi_query_prune_ratio | `0.5` |
| sindi_term_prune_ratio | `0` |
| build_alpha | `0.5` |
| 固定索引 | `scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index` |

### 4.4 实际运行的 prune_scale 取值

脚本默认 `PRUNE_SCALES_TEXT="0.5 1.0"`，即默认只对比 0.5（剪枝）与 1.0（不剪枝）。结果目录中实际存在三个 prune_scale 子目录（`prune_0.3`、`prune_0.5`、`prune_1.0`），其中 `prune_0.3` 通过环境变量 `PRUNE_SCALES_TEXT="0.3"` 单独追加运行。

| prune_scale | 含义 | 运行方式 |
|---:|---|---|
| 0.3 | 激进剪枝（与主实验一致） | 环境变量覆盖单独运行 |
| 0.5 | 中等剪枝（与消融实验 1 一致） | 脚本默认 |
| 1.0 | 不剪枝（baseline） | 脚本默认 |

### 4.5 结果文件结构

```
results/test_prune/
├── nq/
│   ├── prune_0.3/alpha_0.5.txt
│   ├── prune_0.5/alpha_0.5.txt
│   └── prune_1.0/alpha_0.5.txt
├── hotpotqa/  (空)
└── logs/
```

---

## 五、三组实验参数对比总表

| 参数 | 主实验 UHG | 消融 1（入口策略） | 消融 2（剪枝强度） |
|---|---|---|---|
| 数据集 | nq, hotpotqa, msmarco | nq, hotpotqa | nq |
| 方法 | `auto` | `uhg` / `uhgs` / `uhgh` | `uhgs`（固定） |
| alpha | 0.3, 0.4, 0.5, 0.6, 0.7 | 0.3, 0.4, 0.5, 0.6, 0.7 | 0.5（固定） |
| ef_search | 100, 200, 300, 500, 1000 | 100, 200, 300, 500, 1000 | 100, 200, 300, 500, 1000 |
| hybrid_prune_scale | **0.3** | **0.5** | **0.3 / 0.5 / 1.0**（变量） |
| sindi_bk | = ef_search | = ef_search | = ef_search |
| dense_entry_bk | = ef_search | = ef_search | = ef_search |
| dense_entry_ef_search | = ef_search | = ef_search | = ef_search |
| max_hops | 0 | 0 | 0 |
| sindi_query_prune_ratio | 0.5 | 0.5 | 0.5 |
| sindi_term_prune_ratio | 0 | 0 | 0 |
| build_alpha | 0.5 | 0.5 | 0.5 |
| 索引 | 703_<ds>_hybrid_index.index | 同左（同一索引） | 同左（同一索引） |
| 传构建期参数 | 是（graph_path 等） | 否（Deserialize 加载） | 否（Deserialize 加载） |
| 变量 | 无（全部固定） | `--method` | `--hybrid_prune_scale` |
| 二进制 | 703_uhg_exp3 | 703_uhg_exp3 | 703_uhg_exp3 |

---

## 六、索引与数据产物路径

| 类型 | 路径 | 使用方法 |
|---|---|---|
| HDF5 原始数据 | `scripts/UHG/data/hdf5/<dataset>.hdf5` | 所有方法 |
| Dense HNSW 索引 | `scripts/UHG/data/index/701_<dataset>_dense_hnsw.index` | hnsw, hnsw_sindi, uhg dense entry |
| Sparse SINDI 索引 | `scripts/UHG/data/index/701_<dataset>_sparse_sindi.index` | sindi, hnsw_sindi |
| UHG 主图 | `scripts/UHG/data/uhg/<dataset>_uhg.h5` | uhg |
| FHG 图 | `scripts/UHG/data/fhg/<dataset>_fhg_alpha_0_5.h5` | fhg |
| FHG hybrid index | `scripts/UHG/data/index/703_<dataset>_fhg_hybrid_index.index` | fhg |
| UHG hybrid index | `scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index` | uhg, ablation, test_prune |
| Ground truth | `scripts/UHG/data/ground_truth/<dataset>/<dataset>_ground_truth_alpha_<alpha>.npy` | 所有方法 |

---

## 七、二进制汇总

| 二进制 | 用途 | 使用实验 |
|---|---|---|
| `701_uhg_exp1` | HNSW+SINDI 融合基线 | 主实验 |
| `703_uhg_exp3` | UHG / FHG / 消融 / 剪枝实验 | 主实验, 消融 1, 消融 2 |
| `708_uhg_exp8` | HNSW dense-only 基线 | 主实验 |
| `709_uhg_exp9` | SINDI sparse-only 基线 | 主实验 |

---

## 八、结果行格式

### 主实验

```
hnsw       bk=<bk> ef_search=<bk> Recall: <recall> QPS: <qps>
sindi      bk=<bk> Recall: <recall> QPS: <qps>
hnsw_sindi bk=<bk> ef_search=<bk> Recall: <recall> QPS: <qps>
fhg        ef_search=<ef> prune_scale=1 threads=1 Recall: <recall> QPS: <qps>
uhg        method=auto ef_search=<ef> prune_scale=0.3 sindi_qp=0.5 threads=1 Recall: <recall> QPS: <qps>
```

### 消融实验 1

```
<method> ef_search=<ef> prune_scale=0.5 sindi_qp=0.5 threads=1 Recall: <recall> QPS: <qps>
```

### 消融实验 2

```
uhgs ef_search=<ef> prune_scale=<scale> sindi_qp=0.5 threads=1 Recall: <recall> QPS: <qps>
```
