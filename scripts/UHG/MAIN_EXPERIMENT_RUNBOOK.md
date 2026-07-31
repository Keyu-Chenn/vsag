# UHG 主实验从零运行手册（NQ 示例）

本文档以 `nq` 为例，说明在只有一份原始 HDF5 文件时，如何生成主实验所需的所有中间产物并跑完五个方法；末尾还给出复用主实验索引的四组补充实验。要跑其他数据集，把命令里的 `nq` 替换成对应数据集名即可。

五组实验的关系如下：

| 实验 | 改变的变量 | 默认对比范围 | 结果目录 |
|---|---|---|---|
| `main` | 每次运行使用一个固定 alpha，对比五种检索方法 | 5 methods × 5 alpha | `results/main` |
| `mixed_alpha` | 同一次运行中每条 query 使用不同 alpha | 5 methods | `results/mixed_alpha` |
| `mixed_alpha_zt` | 每条 query 使用 `[0.3,0.7]` 截断正态分布的 alpha，对比五个 top-k | 5 datasets × 5 methods × 5 top-k | `mixed_alpha_zt/results` |
| `ablation` | 补跑直接 UHG，并与主实验的 auto 入口结果按相同 ef 比较 Recall | alpha `0.3`: UHG/UHGS；alpha `0.7`: UHG/UHGH | `results/ablation` + `results/main` |
| `test_prune` | 固定 UHG index、mixed-alpha queries 和 `auto` 路由，只改变混合剪枝配置 | prune scale `0.3` / `no_prune` | `results/test_prune` |
| `comp_hops` | 固定 alpha，对比有无外部多入口时 UHG 主图的 hops 和距离计算次数 | alpha `0.3`: UHGS/UHG；alpha `0.7`: UHGH/UHG | `results/comp_hops` |
| `comp_prune_dist` | 固定 UHG index 和 uniform mixed-alpha queries，对比剪枝前后的实际 dense/sparse 距离计算次数 | prune scale `0.3` / `no_prune` | `results/comp_prune_dist` |

主实验目录固定为：

```text
HDF5 输入： /tbase-project/vsag/scripts/UHG/data/hdf5
中间产物： /tbase-project/vsag/scripts/UHG/data
运行脚本： /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main
实验结果： /tbase-project/vsag/scripts/UHG/results/main
```

下面所有命令都使用绝对路径，不依赖提前设置环境变量。

## 1. 准备 HDF5

把原始 HDF5 放到固定路径：

```text
/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5
```

HDF5 至少需要包含：

```text
train, train_labels, train_sparse
test, test_labels, test_sparse
```

检查输入：

```bash
ls "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5"
```

## 2. 创建目录

```bash
mkdir -p \
  "/tbase-project/vsag/scripts/UHG/data/fhg" \
  "/tbase-project/vsag/scripts/UHG/data/uhg" \
  "/tbase-project/vsag/scripts/UHG/data/index" \
  "/tbase-project/vsag/scripts/UHG/data/index_hybrid_union" \
  "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq" \
  "/tbase-project/vsag/scripts/UHG/results/main/logs/ground_truth" \
  "/tbase-project/vsag/scripts/UHG/results/main/logs/index" \
  "/tbase-project/vsag/scripts/UHG/results/main/logs/construction" \
  "/tbase-project/vsag/scripts/UHG/results/main/logs/cache" \
  "/tbase-project/vsag/scripts/UHG/results/main"
```

## 3. 构建二进制

如果这些二进制已经存在，可以跳过本节。

```bash
cmake -S "/tbase-project/vsag" -B "/tbase-project/vsag/build-release" -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build "/tbase-project/vsag/build-release" \
  --target 701_uhg_exp1 703_uhg_exp3 704_uhg_exp4 706_uhg_exp6 708_uhg_exp8 709_uhg_exp9 \
  -j16
```

检查二进制：

```bash
ls /tbase-project/vsag/build-release/examples/cpp/701_uhg_exp1
ls /tbase-project/vsag/build-release/examples/cpp/703_uhg_exp3
ls /tbase-project/vsag/build-release/examples/cpp/704_uhg_exp4
ls /tbase-project/vsag/build-release/examples/cpp/706_uhg_exp6
ls /tbase-project/vsag/build-release/examples/cpp/708_uhg_exp8
ls /tbase-project/vsag/build-release/examples/cpp/709_uhg_exp9
```

## 4. 从 HDF5 生成 ground truth

所有搜索实验都需要 exact ground truth：

```text
/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_<alpha>.npy
```

计算公式是：

```text
score = alpha * dense_inner_product + (1 - alpha) * sparse_inner_product
```

这是精确全量计算，会比较耗时。正式主实验使用 `--num-queries -1` 跑完整 test queries；只做链路验证时可以临时改成 `--num-queries 100`。

先检查 Python 依赖：

```bash
python3 -c "import h5py, numpy, scipy"
```

生成 NQ ground truth：

```bash
python3 "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/generate_ground_truth_from_hdf5.py" \
  --hdf5 "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  --output-dir "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq" \
  --alphas 0.3 0.4 0.5 0.6 0.7 \
  --topk 100 \
  --num-queries -1 \
  --query-chunk 32 \
  --train-chunk 50000 \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/ground_truth/nq_ground_truth.log"
```

检查 ground truth：

```bash
ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.3.npy"
ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.4.npy"
ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.5.npy"
ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.6.npy"
ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.7.npy"
```

## 5. 生成 baseline 索引

这一步从 HDF5 生成 baseline、FHG/UHG 离线构图和 UHG dense-entry 共用的 dense HNSW，以及 SINDI baseline 和 FHG/UHG 离线构图会用到的 sparse SINDI：

```text
/tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index
/tbase-project/vsag/scripts/UHG/data/index/701_nq_sparse_sindi.index
```

命令使用 `--build_only`，只触发索引构建并落盘，不加载 ground truth，也不跑验证查询。后面的正式搜索会复用这些索引。

构建 dense HNSW：

```bash
/tbase-project/vsag/build-release/examples/cpp/708_uhg_exp8 \
  "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  --index_dir "/tbase-project/vsag/scripts/UHG/data/index" \
  --build_only \
  --rebuild \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/index/nq_dense_hnsw_build.log"
```

构建 sparse SINDI：

`709_uhg_exp9` 默认使用 `use_reorder=true`，输出 baseline 用的 `701_<dataset>_sparse_sindi.index`。UHG 专用的 no-reorder `709_<dataset>_sindi.index` 在 8.2 里单独构建。

```bash
/tbase-project/vsag/build-release/examples/cpp/709_uhg_exp9 \
  "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  --index_dir "/tbase-project/vsag/scripts/UHG/data/index" \
  --build_only \
  --rebuild \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/index/nq_701_sindi_reorder_build.log"
```

检查 baseline 索引：

```bash
ls "/tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index"
ls "/tbase-project/vsag/scripts/UHG/data/index/701_nq_sparse_sindi.index"
```

## 6. 构建 FHG 图

FHG 图输出到：

```text
/tbase-project/vsag/scripts/UHG/data/fhg/nq_fhg_alpha_0_5.h5
```

构图参数为 `alpha=0.5, k=64, bk=100, dense_ef_search=200, query_prune_ratio=0.9, threads=1`，不执行 refine。dense candidate index 复用第 5 步的 HNSW。HNSW 自身使用 `max_degree=64, ef_construction=200` 构建；生成 FHG dense candidates 时单独使用 `ef_search=200` 搜索并返回 `bk=100` 个候选。如果 HNSW 不存在，704 会使用相同的 HNSW 构建参数为全部训练数据构建并保存：

```text
/tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index
```

```bash
/tbase-project/vsag/build-release/examples/cpp/704_uhg_exp4 \
  "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  -k 64 \
  --bk 100 \
  --dense_ef_search 200 \
  --alpha 0.5 \
  --query_prune_ratio 0.9 \
  --output_dir "/tbase-project/vsag/scripts/UHG/data/fhg" \
  --index_dir "/tbase-project/vsag/scripts/UHG/data/index" \
  --threads 1 \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/construction/nq_fhg_build.log"
```

检查 FHG 产物：

```bash
ls "/tbase-project/vsag/scripts/UHG/data/fhg/nq_fhg_alpha_0_5.h5"
ls "/tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index"
```

## 7. 构建 UHG 图

UHG 图输出到：

```text
/tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg.h5
```

构图算法使用第 5 步的 HNSW 和 baseline SINDI 分别召回 `bk=100` 个候选，在 `alpha=0.0,0.1,...,1.0` 下各取 top-32，按邻居进入各 alpha top-k 的频次合并，并裁剪到 `merge_max_degree=64`。参数固定为 `k=32, bk=100, dense_ef_search=200, alpha_step=0.1, query_prune_ratio=0.9, merge_max_degree=64, threads=1`，并跳过 refine。HNSW 自身使用 `max_degree=64, ef_construction=200` 构建；生成 UHG dense candidates 时单独使用 `ef_search=200`。

```bash
/tbase-project/vsag/build-release/examples/cpp/706_uhg_exp6 \
  "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  -k 32 \
  --bk 100 \
  --dense_ef_search 200 \
  --alpha_step 0.1 \
  --query_prune_ratio 0.9 \
  --merge_max_degree 64 \
  --skip_refine \
  --output "/tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg.h5" \
  --index_dir "/tbase-project/vsag/scripts/UHG/data/index" \
  --threads 1 \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/construction/nq_uhg_build.log"
```

检查 UHG 产物：

```bash
ls "/tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg.h5"
```

`bk=100`、HNSW candidate index、`dense_ef_search=200`、alpha-frequency merge 和 `skip_refine` 是当前主实验的固定构图配置。旧 Fever 日志使用 HGraph，只保留作历史结果参考：
`results/main/logs/construction/fever_fhg_build.log` 和
`results/main/logs/construction/fever_uhg_build.log`。

## 8. 构建 hybrid_index index

搜索脚本最终加载的是 hybrid_index index，不是直接搜索 `.h5` 图文件。重建 FHG/UHG 图之后必须重建对应 hybrid_index 文件。

### 8.1 FHG hybrid_index index

输出到：

```text
/tbase-project/vsag/scripts/UHG/data/index/703_nq_fhg_hybrid_index.index
```

这个文件就是 FHG 搜索用的 index。推荐在正式搜索前显式运行下面这条命令构建它。`run_fhg.sh` 在文件不存在时也会由 `703_uhg_exp3` 触发构建，但第一条搜索点会先花时间建 index，建完前结果文件里不会出现 Recall/QPS；为了流程清楚，主流程按下面命令先构建。

```bash
/tbase-project/vsag/build-release/examples/cpp/703_uhg_exp3 \
  "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  --method uhg \
  -k 100 \
  --alpha 0.5 \
  --gt_dir "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq" \
  --graph_path "/tbase-project/vsag/scripts/UHG/data/fhg/nq_fhg_alpha_0_5.h5" \
  --hybrid_index_path "/tbase-project/vsag/scripts/UHG/data/index/703_nq_fhg_hybrid_index.index" \
  --index_dir "/tbase-project/vsag/scripts/UHG/data/index" \
  --disable_sindi \
  --disable_dense_entry \
  --build_alpha 0.5 \
  --disable_hybrid_pruning \
  --max_hops 0 \
  --sindi_bk 100 \
  --ef_search 100 \
  --num_queries 1 \
  --threads 1 \
  --rebuild \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/cache/nq_fhg_cache_build.log"
```

这条命令的建库过程：

1. `--rebuild` 强制重建 `/tbase-project/vsag/scripts/UHG/data/index/703_nq_fhg_hybrid_index.index`，即使旧 index 已经存在也不会直接复用。
2. `703_uhg_exp3` 打开 `/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5`。建库时读取 `train`、`train_sparse`、`train_labels`；最后做 1 条查询验证时读取 `test`、`test_sparse`。
3. 程序创建一个 `hybrid_index`。主图不在这条命令里重新构图，而是从 `--graph_path "/tbase-project/vsag/scripts/UHG/data/fhg/nq_fhg_alpha_0_5.h5"` 加载第 6 步已经生成的 FHG 图。
4. 因为传了 `--disable_sindi` 和 `--disable_dense_entry`，这个 FHG hybrid_index 只包含 FHG 主图和用于最终距离计算的 dense/sparse base 数据，不包含 internal SINDI 和 dense-entry HNSW。
5. 构建完成后序列化保存到 `/tbase-project/vsag/scripts/UHG/data/index/703_nq_fhg_hybrid_index.index`。
6. `--num_queries 1` 只表示保存 index 后用 1 条 query 跑一次 `method=uhg` 检索并输出 Recall/QPS，方便确认 index 能被搜索；它不是建库规模参数，建库仍然使用完整 `train` 集合。`--disable_hybrid_pruning` 关闭的是这次验证查询的 hybrid sparse 上界剪枝，不影响 index 构建。

### 8.2 UHG hybrid_index index

输出到：

```text
/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index
```

这里不是说 UHG 不需要 SINDI。UHG hybrid_index 需要 SINDI 来支持 sparse-entry 搜索；主流程先用 `709_uhg_exp9 --reorder false --build_only` 单独构建一份 UHG 专用 SINDI：

```text
/tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index
```

然后在构建 UHG hybrid_index 时把这份文件传给 `703_uhg_exp3 --sindi_index_path`。这样 `hybrid_index.Build()` 会反序列化已有 SINDI，并把它序列化进最终的 UHG hybrid_index index，而不是在 hybrid_index 构建过程中重新对 `train_sparse` 建 internal SINDI。

注意不要复用第 5 步的 `701_nq_sparse_sindi.index`。`701_*_sparse_sindi.index` 是 baseline 用的 SINDI，默认 `use_reorder=true`；UHG hybrid_index 需要 `use_reorder=false`，避免和 `hybrid_codes` 中已经保存的 sparse 原始数据重复。

UHG SINDI 的触发条件：

- 先用 `709_uhg_exp9 --reorder false --build_only` 构建 `/tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index`。
- 构建 hybrid_index 时不传 `--disable_sindi`，因此 `enable_sindi=true`。
- 构建 hybrid_index 时传 `--sindi_index_path "/tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index"`，因此 `hybrid_index.Build()` 会加载外部 SINDI，并跳过内部 SINDI 重建。
- 如果 `/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index` 已经存在且不加 `--rebuild`，程序只会反序列化 hybrid_index 里已经保存的 SINDI，不会重新加载或构建。

先构建 UHG 专用 no-reorder SINDI：

```bash
/tbase-project/vsag/build-release/examples/cpp/709_uhg_exp9 \
  "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  --index_dir "/tbase-project/vsag/scripts/UHG/data/index" \
  --reorder false \
  --build_only \
  --rebuild \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/index/nq_709_sindi_no_reorder_build.log"
```

检查 UHG 专用 SINDI：

```bash
ls "/tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index"
```

这条命令传给 `hybrid_index` 的构建参数是：

```text
dtype=float32
metric_type=ip
dim=<从 nq.hdf5 的 train 读取>
sparse_dtype=float32
sparse_metric_type=ip
sparse_dim=30000
alpha=0.5
build_alpha=0.5
ef_construction=200
max_degree=64
enable_sindi=true
enable_dense_entry=true
graph_path=/tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg.h5
dense_entry_hnsw_graph_path=/tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index
sindi_index_path=/tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index
```

`--build_alpha 0.5` 是建库用的 dense 权重；`--alpha` 是这次查询验证用的 dense 权重。这里两者都设为 `0.5`。

`709_nq_sindi.index` 的 SINDI 构建参数是：

```text
term_id_limit=1000000
window_size=100000
doc_prune_ratio=0
use_reorder=false
```

下面命令里的 `--sindi_bk 100`、`--sindi_query_prune_ratio 0.5`、`--sindi_term_prune_ratio 0` 是 sparse-entry 的搜索参数，不是 `709_nq_sindi.index` 的建库参数。

```bash
/tbase-project/vsag/build-release/examples/cpp/703_uhg_exp3 \
  "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  --graph_path "/tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg.h5" \
  --sindi_index_path "/tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index" \
  --dense_entry_hnsw_graph_path "/tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index" \
  --hybrid_index_path "/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index" \
  --alpha 0.5 \
  --build_alpha 0.5 \
  --method auto \
  -k 100 \
  --ef_search 100 \
  --sindi_bk 100 \
  --dense_entry_bk 100 \
  --dense_entry_ef_search 100 \
  --hybrid_prune_scale 0.3 \
  --max_hops 0 \
  --sindi_query_prune_ratio 0.5 \
  --sindi_term_prune_ratio 0 \
  --index_dir "/tbase-project/vsag/scripts/UHG/data/index_hybrid_union" \
  --gt_dir "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq" \
  --num_queries 1 \
  --threads 1 \
  --rebuild \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/cache/nq_uhg_cache_build.log"
```

这条命令的建库过程：

1. `--rebuild` 强制重建 `/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index`，即使旧 index 已经存在也不会直接复用。
2. `703_uhg_exp3` 打开 `/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5`。建库时读取 `train`、`train_sparse`、`train_labels`；最后做 1 条查询验证时读取 `test`、`test_sparse`。
3. 程序创建一个 `hybrid_index`。主图不在这条命令里重新构图，而是从 `--graph_path "/tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg.h5"` 加载第 7 步已经生成的 UHG 图。
4. 因为传了 `--sindi_index_path "/tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index"`，`hybrid_index.Build()` 会加载这份 no-reorder SINDI，不会在 hybrid_index 构建过程中重新构建 SINDI。
5. 因为传了 `--dense_entry_hnsw_graph_path "/tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index"`，dense-entry 部分会加载第 5 步生成的 dense HNSW。
6. `hybrid_index.Build()` 把每个 base 向量的 dense 和 sparse 数据写入 hybrid code，同时把 UHG 主图、加载进来的 no-reorder SINDI、dense-entry HNSW 图结构组织成一个统一索引。这里的 dense-entry HNSW 主要是入口搜索用的 HNSW 图拓扑、层级、入口点和 label 映射；搜索时的 dense 距离仍然用 hybrid code 里的 dense 向量计算。
7. 构建完成后序列化保存到 `/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index`。
8. `--num_queries 1` 只表示保存 index 后用 1 条 query 跑一次 `method=auto` 检索并输出 Recall/QPS，方便确认 index 能被搜索；它不是建库规模参数，建库仍然使用完整 `train` 集合。

`--method auto` 的路由规则（`alpha=0.5` 走 UHGS）：

| alpha | auto 实际方法 | 行为 |
|---:|---|---|
| 0.3, 0.4, 0.5 | `uhgs` | 先用 SINDI 找 sparse entry ids，再搜 UHG 主图 |
| 0.6, 0.7 | `uhgh` | 先用 dense-entry HNSW 找 dense entry，再搜 UHG 主图 |

默认阈值 `auto_sparse_alpha_max_=0.5`、`auto_dense_alpha_min_=0.5`；`alpha<=0.5` 走 UHGS，`alpha>0.5` 走 UHGH。

检查 hybrid_index index：

```bash
ls "/tbase-project/vsag/scripts/UHG/data/index/703_nq_fhg_hybrid_index.index"
ls "/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index"
```

## 9. 运行搜索实验

搜索结果写到：

```text
/tbase-project/vsag/scripts/UHG/results/main/nq/<method>/alpha_<alpha>.txt
```

跑 NQ 的三个 baseline：

```bash
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/run_baselines_nq.log"
```

其中纯 `sindi` 和 `hnsw_sindi` baseline 默认都使用
`query_prune_ratio=0.5, term_prune_ratio=0`，与 UHGS 的 sparse-entry 搜索参数对齐，
可通过环境变量 `SINDI_QUERY_PRUNE_RATIO` 和 `SINDI_TERM_PRUNE_RATIO` 临时覆盖。
HNSW baseline 不使用这两个参数。

默认搜索参数点为：HNSW 使用 `100 200 300 500 800 1000`，SINDI 使用
`1000 1500 2000 3000 5000 10000`，HNSW+SINDI 使用 `100 150 200 300 500`。

跑 NQ 的 FHG：

```bash
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_fhg.sh" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/run_fhg_nq.log"
```

`run_fhg.sh` 固定传入 `--disable_hybrid_pruning`，关闭 hybrid sparse 上界剪枝；
`ef_search` 控制的标准图搜索候选队列剪枝仍然保留。
FHG 的默认 `ef_search` 点为 `100 200 300 500 1000`，与 UHG 保持一致。

跑 NQ 的 UHG：

```bash
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/main/logs/run_uhg_nq.log"
```

正式运行 UHG 前先按 8.2 显式构建 `/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index`。这个 cache 已存在时，`run_uhg.sh` 会直接加载 index 里已经内嵌的 no-reorder SINDI。不要依赖 `run_uhg.sh` 首次运行时自动构建 UHG hybrid_index；脚本本身不负责先生成并传入 `709_nq_sindi.index`。

只跑某一个 baseline 方法：

```bash
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh" nq hnsw
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh" nq sindi
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh" nq hnsw_sindi
```

## 10. 结果检查

确认 NQ 每个方法都有 5 个 alpha 结果文件：

```bash
find "/tbase-project/vsag/scripts/UHG/results/main/nq/hnsw" -maxdepth 1 -type f -name 'alpha_*.txt' | wc -l
find "/tbase-project/vsag/scripts/UHG/results/main/nq/sindi" -maxdepth 1 -type f -name 'alpha_*.txt' | wc -l
find "/tbase-project/vsag/scripts/UHG/results/main/nq/hnsw_sindi" -maxdepth 1 -type f -name 'alpha_*.txt' | wc -l
find "/tbase-project/vsag/scripts/UHG/results/main/nq/fhg" -maxdepth 1 -type f -name 'alpha_*.txt' | wc -l
find "/tbase-project/vsag/scripts/UHG/results/main/nq/uhg" -maxdepth 1 -type f -name 'alpha_*.txt' | wc -l
```

确认结果文件里有指标行：

```bash
rg -c 'Recall:' "/tbase-project/vsag/scripts/UHG/results/main/nq"/*/alpha_*.txt
```

检查错误：

```bash
rg -n 'ERROR|NO_METRICS|Failed|Error:' "/tbase-project/vsag/scripts/UHG/results/main/nq" "/tbase-project/vsag/scripts/UHG/results/main/logs"
```

生成 NQ 结果汇总 TSV：

```bash
awk '
  /^hnsw |^sindi |^hnsw_sindi |^fhg |^uhg / {
    path = FILENAME
    sub(/^.*results\/main\//, "", path)
    split(path, p, "/")
    dataset = p[1]
    method = p[2]
    alpha = p[3]
    sub(/^alpha_/, "", alpha)
    sub(/\.txt$/, "", alpha)
    param = ""
    recall = ""
    qps = ""
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^ef_search=/) { split($i, a, "="); param = a[2] }
      if ($i ~ /^bk=/ && param == "") { split($i, a, "="); param = a[2] }
      if ($i == "Recall:") { recall = $(i + 1) }
      if ($i == "QPS:") { qps = $(i + 1) }
    }
    print dataset "\t" method "\t" alpha "\t" param "\t" recall "\t" qps
  }
' "/tbase-project/vsag/scripts/UHG/results/main/nq"/*/alpha_*.txt > "/tbase-project/vsag/scripts/UHG/results/main/nq_summary.tsv"
```

汇总文件：

```text
/tbase-project/vsag/scripts/UHG/results/main/nq_summary.tsv
```

## 11. 从零产物清单

按本文档执行后，NQ 会生成：

```text
ground truth:
  /tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_<alpha>.npy

baseline indexes:
  /tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index
  /tbase-project/vsag/scripts/UHG/data/index/701_nq_sparse_sindi.index

shared dense HNSW index (baseline, offline graph construction, and dense entry):
  /tbase-project/vsag/scripts/UHG/data/index/701_nq_dense_hnsw.index

UHG sparse-entry SINDI:
  /tbase-project/vsag/scripts/UHG/data/index/709_nq_sindi.index

graphs:
  /tbase-project/vsag/scripts/UHG/data/fhg/nq_fhg_alpha_0_5.h5
  /tbase-project/vsag/scripts/UHG/data/uhg/nq_uhg.h5

hybrid_index indexes:
  /tbase-project/vsag/scripts/UHG/data/index/703_nq_fhg_hybrid_index.index
  /tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index

results:
  /tbase-project/vsag/scripts/UHG/results/main/nq/<method>/alpha_<alpha>.txt
```

## 12. Mixed-alpha 实验：每条 query 使用不同 alpha

这个实验与前面的主实验相互独立。主实验在一次搜索中让所有 query 共用一个
`alpha`；mixed-alpha 实验为每条 query 分配一个不同的 alpha。alpha 在
`[0.3, 0.7]` 上均匀分布，使用固定随机种子 `42` 打乱，因此五个方法使用完全
相同且可复现的 query-alpha 对应关系。

实验脚本位于独立目录：

```text
/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha
```

这个实验复用前面已经构建的 HNSW、SINDI、FHG、UHG 和 hybrid_index，不需要
重新构图或重建索引。

### 12.1 重新编译搜索程序

四个搜索程序新增了两个可选参数：

```text
--mixed_alpha_file <path>  每条 query 的 alpha，float32 npy，shape=(query_count,)
--mixed_gt_file <path>     mixed-alpha exact ground truth，int64 npy，shape=(query_count, topk)
```

两个参数必须同时传入。两个参数都不传时，程序继续使用原来的 `--alpha` 和
`--gt_dir`，所以前面的主实验命令和行为不变。

编译：

```bash
cmake --build "/tbase-project/vsag/build-release" \
  --target 701_uhg_exp1 703_uhg_exp3 708_uhg_exp8 709_uhg_exp9 \
  -j16
```

### 12.2 生成均匀 alpha 和 exact ground truth

下面的命令读取 NQ 全部 test queries，为每条 query 生成一个不同的 alpha，
并计算对应的 exact top-100：

```bash
python3 "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha/prepare_mixed_alpha.py" \
  --hdf5 "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5" \
  --output-dir "/tbase-project/vsag/scripts/UHG/data/mixed_alpha/nq" \
  --alpha-min 0.3 \
  --alpha-max 0.7 \
  --seed 42 \
  --topk 100 \
  --num-queries -1 \
  --query-chunk 32 \
  --train-chunk 50000
```

产物为：

```text
/tbase-project/vsag/scripts/UHG/data/mixed_alpha/nq/alphas.npy
/tbase-project/vsag/scripts/UHG/data/mixed_alpha/nq/ground_truth.npy
/tbase-project/vsag/scripts/UHG/data/mixed_alpha/nq/metadata.json
```

其中第 `i` 条 query 使用 `alphas.npy[i]`，并与
`ground_truth.npy[i]` 一一对应。alpha 使用均匀分位点生成后再打乱；对于
1000 条 query，1000 个 alpha 互不相同，均值为 `0.5`。

检查产物：

```bash
python3 -c 'import json, numpy as np; p="/tbase-project/vsag/scripts/UHG/data/mixed_alpha/nq"; a=np.load(p+"/alphas.npy"); g=np.load(p+"/ground_truth.npy"); print(json.load(open(p+"/metadata.json"))); print("alpha", a.shape, a.min(), a.max(), a.mean()); print("gt", g.shape)'
```

只做链路验证时，可以在准备命令中临时使用 `--num-queries 100`；正式实验需要
重新使用 `--num-queries -1` 生成完整产物。

### 12.3 运行五个方法

确认第 5～8 节的索引和图已经存在后运行：

```bash
mkdir -p "/tbase-project/vsag/scripts/UHG/results/mixed_alpha"

bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha/run_mixed_alpha.sh" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/mixed_alpha/run_nq.log"
```

脚本依次运行：

```text
hnsw, sindi, hnsw_sindi, fhg, uhg
```

搜索参数点与主实验保持一致。FHG 继续使用固定 `build_alpha=0.5` 的图，并关闭
hybrid sparse 上界剪枝；UHG
继续使用 `method=auto` 和 `build_alpha=0.5`，但每条 query 的搜索 alpha 来自
`alphas.npy`。因此 UHG 会对 `alpha<=0.5` 的 query 使用 UHGS，对
`alpha>0.5` 的 query 使用 UHGH。

纯 `sindi` 和 `hnsw_sindi` baseline 同样默认使用
`query_prune_ratio=0.5, term_prune_ratio=0`，与主实验和 UHGS 对齐；参数可通过
`SINDI_QUERY_PRUNE_RATIO` 和 `SINDI_TERM_PRUNE_RATIO` 临时覆盖。

其中 SINDI 的默认 `bk` 点为 `1000 1500 2000 3000 5000 10000`；FHG 和 UHG 的默认
搜索点均为 `100 200 300 500 1000`。

结果单独写入：

```text
/tbase-project/vsag/scripts/UHG/results/mixed_alpha/nq/hnsw.txt
/tbase-project/vsag/scripts/UHG/results/mixed_alpha/nq/sindi.txt
/tbase-project/vsag/scripts/UHG/results/mixed_alpha/nq/hnsw_sindi.txt
/tbase-project/vsag/scripts/UHG/results/mixed_alpha/nq/fhg.txt
/tbase-project/vsag/scripts/UHG/results/mixed_alpha/nq/uhg.txt
```

不会覆盖 `/tbase-project/vsag/scripts/UHG/results/main` 下的现有主实验结果。

检查结果：

```bash
rg -n 'Recall:|ERROR|NO_METRICS' "/tbase-project/vsag/scripts/UHG/results/mixed_alpha/nq"
```

生成五种方法的交互式 Recall-QPS HTML 图和对应 TSV：

```bash
python3 "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha/plot_mixed_alpha_html.py" nq
```

输出为：

```text
/tbase-project/vsag/scripts/UHG/results/mixed_alpha/mixed_alpha_qps_recall.html
/tbase-project/vsag/scripts/UHG/results/mixed_alpha/mixed_alpha_qps_recall_points.tsv
```

HTML 默认使用线性 QPS 轴，可在页面中切换为对数轴，并支持悬停查看搜索参数、
Recall 和 QPS。脚本默认也可以不传数据集名，此时会自动发现结果目录下已有的
数据集。若某个方法的结果文件尚无指标行，脚本会给出警告并绘制其余已有方法。

如果只想快速验证搜索链路，可以限制 query 数和每种方法的单个参数点：

```bash
NUM_QUERIES=100 \
HNSW_POINTS_TEXT=100 \
SINDI_POINTS_TEXT=100 \
HNSW_SINDI_POINTS_TEXT=100 \
FHG_POINTS_TEXT=100 \
UHG_POINTS_TEXT=100 \
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha/run_mixed_alpha.sh" nq
```

### 12.4 截断正态 Mixed-alpha：Recall@10/@20/@50/@100/@200

这组实验与 12.2～12.3 的均匀分布 mixed-alpha 实验相互独立。实验脚本、
新生成的 alpha/ground truth、日志和结果全部放在：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt
```

目标矩阵为：

```text
datasets = nq hotpotqa msmarco dbpedia-entity fever
methods = hnsw sindi hnsw_sindi fhg uhg
top-k = 10 20 50 100 200
num_queries = -1
threads = 1
```

完整矩阵会生成 `5 datasets × 5 methods × 5 top-k = 125` 个结果文件和 `705`
个 Recall/QPS 参数点。当前按要求暂不运行 `msmarco` 搜索，因此默认运行其余四个
数据集，生成 100 个结果文件和 564 个 Recall/QPS 参数点；`msmarco` 的 alpha/GT
仍按 12.4.2 节准备。

#### 12.4.1 Alpha 分布定义

正态分布本身没有有限边界，因此“`0.3-0.7` 的正态分布”固定定义为截断正态：

```text
X ~ Normal(mu=0.5, sigma=0.1)
alpha = X conditioned on 0.3 <= X <= 0.7
seed = 42
```

生成器不直接随机抽样再裁剪，而是取截断正态分布的中点分位数，再用 seed `42`
打乱 query-alpha 对应关系。这与 12.2 节均匀分布使用“分层分位点后打乱”的方式
一致，可以避免有限样本的分布偏移和裁剪造成的 `0.3/0.7` 边界堆积，并保证结果
完全可复现。正式运行使用全量 test queries。

每条 query 的 exact ground truth 仍按下式计算：

```text
score = alpha * dense_inner_product + (1 - alpha) * sparse_inner_product
```

#### 12.4.2 数据和 exact ground truth

每个数据集只计算一份 exact top-200：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/data/gt_top200/<dataset>/alphas.npy
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/data/gt_top200/<dataset>/ground_truth.npy
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/data/gt_top200/<dataset>/metadata.json
```

Recall@10、Recall@20、Recall@50、Recall@100 和 Recall@200 全部读取这一份
`alphas.npy` 和 `ground_truth.npy`。每次搜索仍传入各自独立的 `-k`，因此每条
query 分别只使用 GT 的前 10、20、50、100 或 200 个 id。校验脚本要求 GT shape
严格为 `(query_count, 200)`，无需再计算或派生其他宽度的 GT。

按数据集串行生成五套 alpha 和 GT：

```bash
bash "/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/prepare_all.sh"
```

默认支持断点续跑。完整预检命令为：

```bash
python3 "/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/validate_inputs.py"
```

数据准备日志写到：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/logs/prepare/<dataset>.log
```

#### 12.4.3 五个 top-k 的搜索参数

不同 Recall 使用不同的 `-k` 和对应的搜索参数点，所有参数点都不小于 top-k：

| Recall | HNSW | SINDI | HNSW+SINDI | FHG | UHG |
|---:|---|---|---|---|---|
| 10 | `10 20 30 50 80 100` | `100 150 200 300 500 1000` | `10 15 20 30 50 100` | `10 20 30 50 80 100` | `10 20 30 50 80 100` |
| 20 | `20 40 60 100 160 200` | `200 300 400 600 1000 2000` | `20 30 40 60 100 200` | `20 40 60 100 160 200` | `20 40 60 100 160 200` |
| 50 | `50 100 150 250 400 500` | `500 750 1000 1500 2500 5000` | `50 75 100 150 250` | `50 100 150 250 500` | `50 100 150 250 500` |
| 100 | `100 200 300 500 800 1000` | `1000 1500 2000 3000 5000 10000` | `100 150 200 300 500` | `100 200 300 500 1000` | `100 200 300 500 1000` |
| 200 | `200 400 600 1000 1600 2000` | `2000 3000 4000 6000 10000 20000` | `200 300 400 600 1000` | `200 400 600 1000 2000` | `200 400 600 1000 2000` |

Recall@100 使用第 9 节主实验的默认参数点；Recall@50 使用仓库中已有的
Recall@50 mixed-alpha 参数网格；Recall@10、Recall@20 和 Recall@200 与已完成
的均匀 mixed-alpha 实验保持一致。

方法配置也保持不变：

```text
SINDI/HNSW+SINDI:
  query_prune_ratio=0.5
  term_prune_ratio=0

FHG:
  build_alpha=0.5
  enable_sindi=false
  enable_dense_entry=false
  enable_hybrid_pruning=false
  max_hops=0

UHG:
  method=auto
  build_alpha=0.5
  hybrid_prune_scale=0.3
  max_hops=0
  alpha<=0.5 -> UHGS
  alpha>0.5 -> UHGH
```

#### 12.4.4 复用索引和 hybrid index 生命周期

实验直接复用第 5～7 节已有的以下产物，不重建、也不删除这些文件：

```text
data/index/701_<dataset>_dense_hnsw.index
data/index/701_<dataset>_sparse_sindi.index
data/index/709_<dataset>_sindi.index
data/fhg/<dataset>_fhg_alpha_0_5.h5
data/uhg/<dataset>_uhg.h5
```

由于磁盘空间有限，完整实验严格按数据集串行运行：

1. 依次运行 HNSW、SINDI、HNSW+SINDI 的五个 top-k。
2. 需要 FHG 时才构建
   `data/index/703_<dataset>_fhg_hybrid_index.index`。
3. 连续完成 FHG 的 Recall@10/@20/@50/@100/@200。
4. 只删除 `data/index/703_<dataset>_fhg_hybrid_index.index`。
5. 需要 UHG 时才构建
   `data/index_hybrid_union/703_<dataset>_hybrid_index.index`。
6. 连续完成 UHG 的 Recall@10/@20/@50/@100/@200。
7. 只删除 `data/index_hybrid_union/703_<dataset>_hybrid_index.index`。
8. 进入下一个数据集并重新构建对应 hybrid index。

删除函数只允许固定目录
`/tbase-project/vsag/scripts/UHG/data`、五个数据集白名单和上面两种精确路径模板；
会拒绝目录、符号链接、非普通文件和其他路径。每次构建前还会清理上述白名单内
由中断遗留的其他 FHG/UHG hybrid index，确保不会同时保留两份 hybrid index。
若新 index 构建后运行失败或收到信号，退出 trap 也只删除当轮新建的精确文件。

每次实际构建 FHG/UHG hybrid index 之前，调度脚本还会读取
`results/results_integration/construction/index_size.txt` 中对应数据集和方法的
实测索引大小，与当时磁盘可用空间比较，并默认额外预留 `2 GiB`。若预计索引大小
加安全余量无法容纳，则拒绝开始构建，不会留下因空间不足产生的半成品索引。

#### 12.4.5 运行、断点续跑和结果

准备好 alpha/GT 后，可以先打印完整命令计划；这个命令不搜索、不构建或删除索引：

```bash
DRY_RUN=1 \
bash "/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/run_all.sh"
```

正式串行运行：

```bash
bash "/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/run_all.sh"
```

结果和日志分别写入：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/results/recall_<10|20|50|100|200>/<dataset>/<method>.txt
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/results/logs/recall_<k>/<dataset>/<method>.log
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/results/logs/runner/recall_<k>/<dataset>_<method>.log
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/results/logs/index/<dataset>_<fhg|uhg>_build.log
```

结果文件通过临时文件原子写入。默认 `RESUME=1`；只有分布元数据、top-k、
query 数、线程数、参数点列表、方法配置和全部 Recall/QPS 点都匹配时才跳过。

全套成功后严格检查 125 个结果文件和 705 个参数点，并生成：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/results/mixed_alpha_zt_points.tsv
```

也可以手工执行严格汇总：

```bash
python3 "/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/summarize_results.py" --strict
```

完整的独立说明见：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/README.md
```

当前进度（2026-07-28）：

- 五个数据集的截断正态 alpha 和 exact top-200 GT 已全部生成并通过校验。
- `nq`、`hotpotqa`、`dbpedia-entity`、`fever` 的五种方法、五个 top-k 已全部完成，
  共 100 个结果文件和 564 个 Recall/QPS 参数点。
- 按当前实验要求暂不运行 `msmarco` 搜索；其 alpha/GT 已准备好，结果文件数为 0。
- 每个 FHG/UHG 实验完成后均已删除对应的临时 hybrid index，当前无此类 index
  残留。

## 13. Ablation 实验：UHG 入口策略消融

这个实验只补跑直接搜索主图的 `method=uhg`，不再重复运行 UHGS/UHGH。最终复用
主实验 `method=auto` 的结果进行两组比较：

| alpha | Ablation 新跑 | 主实验复用 | 指标 |
|---:|---|---|---|
| `0.3` | UHG | auto 实际走 UHGS | 相同 `ef_search` 的 Recall |
| `0.7` | UHG | auto 实际走 UHGH | 相同 `ef_search` 的 Recall |

默认运行 `nq`、`hotpotqa`、`msmarco`、`dbpedia-entity`，每个数据集只跑 2 个 alpha 和 5 个
`ef_search` 点（`100 200 300 500 1000`）。参数与主实验保持一致：

```text
top-k=100
num_queries=-1
threads=1
build_alpha=0.5
enable_hybrid_pruning=true
hybrid_prune_scale=0.3
max_hops=0
sindi_query_prune_ratio=0.5
sindi_term_prune_ratio=0
sindi_bk=dense_entry_bk=dense_entry_ef_search=ef_search
```

实验不重建索引。运行 NQ 前确认下面的主实验产物存在：

```bash
ls "/tbase-project/vsag/build-release/examples/cpp/703_uhg_exp3"
ls "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5"
ls "/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index"
for alpha in 0.3 0.7; do
  ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_${alpha}.npy"
done
ls "/tbase-project/vsag/scripts/UHG/results/main/nq/uhg/alpha_0.3.txt"
ls "/tbase-project/vsag/scripts/UHG/results/main/nq/uhg/alpha_0.7.txt"
```

运行 NQ 的直接 UHG：

```bash
mkdir -p "/tbase-project/vsag/scripts/UHG/results/ablation/logs"

bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/ablation/logs/run_ablation_nq.log"
```

运行全部四个数据集：

```bash
mkdir -p "/tbase-project/vsag/scripts/UHG/results/ablation/logs"

bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_all.sh" \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/ablation/logs/run_ablation_all.log"
```

快速验证：

```bash
RESULT_DIR=/tmp/vsag-ablation-smoke \
NUM_QUERIES=100 \
POINTS_TEXT=100 \
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh" nq
```

新跑结果写入：

```text
/tbase-project/vsag/scripts/UHG/results/ablation/<dataset>/uhg/alpha_0.3.txt
/tbase-project/vsag/scripts/UHG/results/ablation/<dataset>/uhg/alpha_0.7.txt
```

按相同 `ef_search` 汇总 Recall 并画图：

```bash
rg -c 'Recall:' "/tbase-project/vsag/scripts/UHG/results/ablation/nq/uhg"/alpha_{0.3,0.7}.txt
rg -n 'ERROR|NO_METRICS|Failed|Error:' "/tbase-project/vsag/scripts/UHG/results/ablation"

python3 "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/plot_ablation.py" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/ablation/logs/plot_ablation_nq.log"
```

比较脚本只读取 ablation 的直接 UHG 结果和主实验对应的 auto 结果，并检查两侧
都是 `top-k=100`、全量 query、`build_alpha=0.5` 和
`hybrid_prune_scale=0.3`。输出 `nq_ablation_recall_by_ef.png` 和
`nq_ablation_recall_by_ef.tsv`。

更完整的结果汇总和画图说明见
`/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/RUNBOOK.md`。

## 14. Test-prune 实验：混合搜索剪枝消融

这个实验同样固定使用第 8.2 节的 UHG `hybrid_index` 和第 12.2 节生成的
mixed-alpha queries/ground truth。每条 query 使用 `alphas.npy` 中自己的 alpha，
并固定使用 `method=auto`：`alpha<=0.5` 走 UHGS，`alpha>0.5` 走 UHGH。
实验对比以下两种混合剪枝配置：

| `enable_hybrid_pruning` | `hybrid_prune_scale` | 设置 |
|---|---:|---|
| `true` | `0.3` | 将 sparse Cauchy 上界缩放为 `0.3`，剪枝最激进，可能损失召回 |
| `false` | 忽略 | `no_prune`：关闭 dense-first sparse 上界剪枝，作为真正的 no-prune 对照 |

`hybrid_prune_scale` 只缩放候选 sparse inner product 的范数上界。值越小越激进；
`scale=0` 表示剪枝判断只看 dense 项，不是只计算 sparse 分数，未被剪掉的候选仍按
完整 dense+sparse 距离计算。默认不传 `--disable_hybrid_pruning`，即
`enable_hybrid_pruning=true`；传入该开关后只关闭这层 sparse 上界剪枝，标准图搜索的
`ef_search` 候选队列剪枝仍然保留。

默认运行 `nq`、`hotpotqa`、`msmarco`、`dbpedia-entity`；运行前需要为相应数据集准备
`data/mixed_alpha/<dataset>/alphas.npy` 和 `ground_truth.npy`。每组剪枝配置使用
`ef_search=100 200 300 500 1000`；`sindi_bk`、`dense_entry_bk` 和
`dense_entry_ef_search` 都随 `ef_search` 取相同值。实验只加载已有 index，不会
重建索引。

运行 NQ 的两个默认剪枝配置：

```bash
mkdir -p "/tbase-project/vsag/scripts/UHG/results/test_prune/logs"

bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/test_prune/logs/run_test_prune_nq.log"
```

运行全部四个数据集：

```bash
mkdir -p "/tbase-project/vsag/scripts/UHG/results/test_prune/logs"

bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_all.sh" \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/test_prune/logs/run_test_prune_all.log"
```

只跑 NQ 的一个剪枝配置（第二个位置参数也接受 `no_prune`）：

```bash
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" nq 0.3
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" nq no_prune
```

也可以通过环境变量改变 method、剪枝配置和 `ef_search` 点。例如：

```bash
METHOD=auto \
PRUNE_SCALES_TEXT="0.3 no_prune" \
POINTS_TEXT="200 500" \
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" nq
```

快速验证默认的两个剪枝配置：

```bash
NUM_QUERIES=100 \
POINTS_TEXT=100 \
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" nq
```

结果写入：

```text
/tbase-project/vsag/scripts/UHG/results/test_prune/<dataset>/prune_<scale|no_prune>/alpha_mixed.txt
```

检查和画图：

```bash
rg -c 'Recall:' "/tbase-project/vsag/scripts/UHG/results/test_prune/nq"/*/alpha_mixed.txt
rg -n 'ERROR|NO_METRICS|Failed|Error:' "/tbase-project/vsag/scripts/UHG/results/test_prune"

python3 "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/plot_test_prune.py" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/test_prune/logs/plot_test_prune_nq.log"
```

更完整的环境变量、结果文件和画图说明见
`/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/README.md`。

## 15. Comp-hops 实验：UHG 多入口搜索开销对比

这组实验参照第 13 节 ablation 的固定 alpha 设置，但不复用旧的 Recall 结果。实验在
同一个 UHG `hybrid_index` 上重新运行有入口和无外部多入口两种方法，比较 UHG 主图
搜索阶段的平均 hops 和平均距离计算次数，同时记录 Recall@100 以确认结果质量。

这里的“无入口点 UHG”是相对于 UHGS/UHGH 的外部多入口召回而言。直接
`method=uhg` 仍然需要从 UHG 图自身保存的默认单入口点开始搜索，并不是在没有任何
初始节点的情况下搜索。

### 15.1 实验矩阵

默认运行除 `msmarco` 外的四个数据集：

```text
nq
hotpotqa
dbpedia-entity
fever
```

每个数据集执行以下四组固定-alpha搜索：

| alpha | 有入口方法 | 无外部多入口对照 | 说明 |
|---:|---|---|---|
| `0.3` | `uhgs` | `uhg` | UHGS 先用 internal no-reorder SINDI 生成 sparse entry ids |
| `0.7` | `uhgh` | `uhg` | UHGH 先用 dense-entry HNSW 生成 dense entry ids |

这里显式使用 `method=uhgs` 和 `method=uhgh`，不使用 `method=auto`。在默认路由阈值下，
它们分别等价于 `auto` 在 `alpha=0.3` 和 `alpha=0.7` 时选择的方法，但显式方法名能让
结果文件和汇总表的含义更清楚。

四个数据集、四组方法/alpha 组合、五个搜索宽度共应产生 `4 × 4 × 5 = 80` 个指标点。
本实验不使用 uniform mixed-alpha 或 truncated-normal mixed-alpha 数据，也不按 alpha
区间进一步拆分 query。

### 15.2 输入和固定参数

Recall@100 使用主实验的固定-alpha exact ground truth：

```text
/tbase-project/vsag/scripts/UHG/data/ground_truth/<dataset>/<dataset>_ground_truth_alpha_0.3.npy
/tbase-project/vsag/scripts/UHG/data/ground_truth/<dataset>/<dataset>_ground_truth_alpha_0.7.npy
```

所有对比使用相同的 UHG 图、hybrid index、完整 test queries 和以下参数：

```text
top-k=100
num_queries=-1
threads=1
build_alpha=0.5
ef_search=100 200 300 500 1000
enable_hybrid_pruning=true
hybrid_prune_scale=0.3
max_hops=0
sindi_query_prune_ratio=0.5
sindi_term_prune_ratio=0
sindi_bk=dense_entry_bk=dense_entry_ef_search=ef_search
```

在同一个 `ef_search` 点，有入口方法和直接 UHG 使用相同的 UHG 主图搜索宽度。
`sindi_bk`、`dense_entry_bk` 和 `dense_entry_ef_search` 随 `ef_search` 一起变化，与主实验
和第 13 节 ablation 的参数保持一致。对于 `method=uhg`，这些入口搜索参数不会被使用。

### 15.3 hops 和距离计算次数的统计口径

`703_uhg_exp3` 需要按 query 读取 `hybrid_index.KnnSearch()` 返回结果中的
`GetStatistics()`，并在每个搜索宽度结束后输出：

```text
Recall
QPS
AvgHops
AvgDistCmp
```

其中：

- `AvgHops` 是每条 query 在 UHG 主图中从候选队列弹出并展开节点的平均次数。
- `AvgDistCmp` 是每条 query 在 UHG 主图初始化入口点和访问未访问邻居时执行 hybrid
  distance 计算的平均次数。
- UHGS 传给 UHG 主图的多个 sparse entry ids，其主图初始 hybrid distance 计算计入
  `AvgDistCmp`。
- UHGH 传给 UHG 主图的多个 dense entry ids，其主图初始 hybrid distance 计算同样计入
  `AvgDistCmp`。
- UHGS 在进入 UHG 主图之前执行的 SINDI 检索开销不计入这两个指标。
- UHGH 在进入 UHG 主图之前执行的 dense-entry HNSW 检索开销不计入这两个指标。

因此，这组数据回答的是“入口点如何改变后续 UHG 主图遍历”，不是“包含入口召回在内
的端到端总计算量”。QPS 只作为辅助指标，核心对比是相同 `ef_search` 下的 Recall、
`AvgHops` 和 `AvgDistCmp`。

统计功能应通过 `703_uhg_exp3` 的可选参数启用；未启用时保持已有主实验输出格式不变。
query 级 hops 和距离计算次数先分别保存，再在并行查询循环结束后统一归并，避免
OpenMP 线程之间的数据竞争。

### 15.4 Hybrid index 构建和磁盘生命周期

UHG hybrid index 严格按照第 8.2 节构建，使用：

```text
UHG graph:
/tbase-project/vsag/scripts/UHG/data/uhg/<dataset>_uhg.h5

UHG no-reorder SINDI:
/tbase-project/vsag/scripts/UHG/data/index/709_<dataset>_sindi.index

dense-entry HNSW:
/tbase-project/vsag/scripts/UHG/data/index/701_<dataset>_dense_hnsw.index

hybrid index output:
/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index
```

不能把 hybrid index 写到 `results/comp_hops` 或其他临时路径。由于磁盘空间不足，
整个实验任何时刻最多只允许存在一个 hybrid index，四个数据集必须严格串行执行：

```text
检查当前 hybrid index 清单和剩余磁盘空间
→ 按第 8.2 节构建当前数据集的 UHG hybrid index
→ 运行 alpha=0.3 的 UHGS 和 UHG
→ 运行 alpha=0.7 的 UHGH 和 UHG
→ 校验当前数据集四个结果文件均完整且没有错误
→ 删除当前数据集的 UHG hybrid index
→ 验证该 index 已不存在
→ 再开始下一个数据集
```

运行器在构建新 index 前必须检查
`data/index/703_*_fhg_hybrid_index.index` 和
`data/index_hybrid_union/703_*_hybrid_index.index`。如果发现不属于本次运行的已有
hybrid index，应停止并报告，不能自动删除。正常结束、失败或收到中断信号时，只能
删除本次实验明确创建的当前 UHG hybrid index，并且删除目标必须与上面的标准路径
完全匹配。

### 15.5 结果目录和完整性检查

除 hybrid index 外，实验脚本、日志、原始结果、汇总和图片都放在：

```text
/tbase-project/vsag/scripts/UHG/results/comp_hops
```

计划使用以下结果布局：

```text
results/comp_hops/
├── README.md
├── run_comp_hops.sh
├── summarize.py
├── logs/
│   ├── index/
│   └── search/
├── <dataset>/
│   ├── uhgs/alpha_0.3.txt
│   ├── uhgh/alpha_0.7.txt
│   └── uhg/
│       ├── alpha_0.3.txt
│       └── alpha_0.7.txt
├── comp_hops_points.tsv
└── plots/
```

每个结果文件必须恰好包含五个搜索宽度的 Recall、QPS、`AvgHops` 和 `AvgDistCmp`。
四个数据集应产生 16 个结果文件和 80 个指标点。汇总表按数据集、alpha、方法和
`ef_search` 保存原始指标，并计算入口方法相对直接 UHG 的变化比例：

```text
hops_reduction = (UHG_AvgHops - Entry_AvgHops) / UHG_AvgHops
dist_cmp_reduction = (UHG_AvgDistCmp - Entry_AvgDistCmp) / UHG_AvgDistCmp
```

正值表示入口方法减少了 UHG 主图工作量，负值表示入口方法增加了 UHG 主图工作量。
只有在当前数据集的所有结果文件通过数量检查、字段检查和错误扫描后，才可以删除
当前 hybrid index 并继续下一个数据集。

## 16. Comp-prune-dist 实验：混合剪枝距离计算次数对比

这组实验沿用第 14 节 `test_prune` 的 uniform mixed-alpha、`method=auto` 和两种剪枝
配置，但需要重新运行搜索，不能直接复用旧的 Recall-QPS 结果。实验目标是测量
hybrid sparse 上界剪枝实际跳过了多少 sparse distance 计算，以及它对总计算量和
Recall@100 的影响。

### 16.1 为什么不能直接使用现有 `dist_cmp`

当前 `BasicSearcher` 的 `dist_cmp` 在候选 id 交给
`HybridVectorDataCell::Query()` 时按候选数量累加。混合剪枝发生在 `Query()` 内部：
程序先计算 dense distance，再根据 dense score 和 sparse Cauchy 上界决定是否继续
计算 sparse distance。因此，一个被剪枝而没有计算 sparse distance 的候选仍会被
现有 `dist_cmp` 计数。

所以现有 `dist_cmp` 表示“UHG 主图送入 hybrid distance 流程的候选数”，不能单独
回答“实际执行了多少次 dense/sparse distance 计算”。本实验保留这个字段作为
候选规模参考，同时增加实际计算分量的独立计数器。

### 16.2 实验矩阵和输入

默认运行除 `msmarco` 外的四个数据集：

```text
nq
hotpotqa
dbpedia-entity
fever
```

每个数据集使用第 12.2 节生成的 uniform mixed-alpha 和 exact top-100 ground truth：

```text
/tbase-project/vsag/scripts/UHG/data/mixed_alpha/<dataset>/alphas.npy
/tbase-project/vsag/scripts/UHG/data/mixed_alpha/<dataset>/ground_truth.npy
```

搜索固定使用 `method=auto`，因此每条 query 的路由仍为：

```text
alpha<=0.5 -> UHGS
alpha>0.5  -> UHGH
```

同一个数据集的两组实验加载同一个 UHG hybrid index、使用相同的 query 顺序和
ground truth，只改变下面的混合剪枝配置：

| 配置名 | `enable_hybrid_pruning` | `hybrid_prune_scale` | 含义 |
|---|---|---:|---|
| `prune_0.3` | `true` | `0.3` | 使用主实验的激进 sparse 上界剪枝 |
| `prune_no_prune` | `false` | 忽略 | 传 `--disable_hybrid_pruning`，完全关闭这层上界剪枝 |

`no_prune` 只关闭 hybrid sparse 上界剪枝，不关闭由 `ef_search` 控制的标准图搜索候选
队列剪枝。

四个数据集、两种剪枝配置、五个搜索宽度共应产生 `4 × 2 × 5 = 40` 个指标点。本实验
不按 alpha 区间或 UHGS/UHGH 路由进一步拆分 query。

### 16.3 固定搜索参数

除剪枝配置外，其余参数与第 14 节 `test_prune` 保持一致：

```text
top-k=100
num_queries=-1
threads=1
method=auto
build_alpha=0.5
ef_search=100 200 300 500 1000
max_hops=0
sindi_query_prune_ratio=0.5
sindi_term_prune_ratio=0
sindi_bk=dense_entry_bk=dense_entry_ef_search=ef_search
```

统计次数本身是确定性的，因此不要求为计数重复运行多轮；QPS 仍作为辅助指标，不作为
本实验的核心结论。

### 16.4 新增统计字段和计数位置

统计功能应由 `703_uhg_exp3` 的可选参数启用，未启用时保持已有主实验和
`test_prune` 的输出格式不变。每条 query 使用独立的运行时计数器，并在每个搜索宽度
结束后输出：

```text
Recall
QPS
AvgHops
AvgCandidateDistCmp
AvgDenseDistCmp
AvgSparseDistCmp
AvgSparseSkipped
SparseSkipRatio
AvgTotalComponentDistCmp
```

字段定义如下：

| 字段 | 定义 |
|---|---|
| `AvgHops` | 每条 query 在 UHG 主图中弹出并展开节点的平均次数 |
| `AvgCandidateDistCmp` | 现有 `dist_cmp` 的平均值，即进入 hybrid distance 流程的候选数 |
| `AvgDenseDistCmp` | 实际调用 dense distance 计算的候选数平均值 |
| `AvgSparseDistCmp` | 实际调用 sparse distance 计算的候选数平均值 |
| `AvgSparseSkipped` | 经 sparse 上界判断后没有执行 sparse distance 的候选数平均值 |
| `SparseSkipRatio` | `SparseSkipped / DenseDistCmp` |
| `AvgTotalComponentDistCmp` | `AvgDenseDistCmp + AvgSparseDistCmp` |

计数必须放在 `HybridVectorDataCell::Query()` 的实际执行分支中：

- 每次调用 `dense_cell_->Query(..., id_count, ...)` 时，把对应 `id_count` 加入
  dense 计数。
- 无剪枝路径调用 `sparse_cell_->Query(..., id_count, ...)` 时，把 `id_count` 加入
  sparse 计数。
- 剪枝路径只对 `selected_sparse_ids` 调用 sparse query，因此把
  `selected_sparse_ids.size()` 加入 sparse 计数。
- 剪枝路径中未进入 `selected_sparse_ids` 的候选加入 `SparseSkipped`。
- 初始化阶段因候选队列尚未填满而使用完整距离计算时，也要如实计入 dense 和 sparse
  次数，但不能计入 `SparseSkipped`。

不能仅在 `BasicSearcher` 外层根据候选数推算 sparse 次数，因为只有
`HybridVectorDataCell::Query()` 知道每个 batch 中有多少候选真正通过了上界判断。
这些计数是 query 局部状态，不需要写入或改变 hybrid index 的序列化格式。

在本实验的 alpha 范围 `[0.3,0.7]` 内，dense 和 sparse 权重都非零，因此
`AvgSparseDistCmp` 也可视为真正完成 dense+sparse 混合距离计算的候选数量。

### 16.5 统计范围和结果解释

与第 15 节相同，本实验只统计 UHG 主图搜索：

- 不统计 UHGS 生成入口点时的 SINDI 检索。
- 不统计 UHGH 生成入口点时的 dense-entry HNSW 检索。
- 统计 UHGS/UHGH 入口 ids 进入 UHG 主图以后产生的 dense/sparse distance 计算。

两种剪枝配置使用完全相同的入口方法和入口参数，所以该范围可以隔离 hybrid sparse
上界剪枝对 UHG 主图内部计算量的影响。

每个 `ef_search` 点计算：

```text
sparse_compute_reduction =
    (NoPrune_AvgSparseDistCmp - Prune_AvgSparseDistCmp)
    / NoPrune_AvgSparseDistCmp

total_compute_reduction =
    (NoPrune_AvgTotalComponentDistCmp - Prune_AvgTotalComponentDistCmp)
    / NoPrune_AvgTotalComponentDistCmp
```

正值表示剪枝减少了对应计算量。汇总时同时保留 Recall 差值，不能只报告计算节省而
忽略近似剪枝可能造成的召回损失。

需要生成以下三类对比：

1. 相同 `ef_search` 下 `AvgSparseDistCmp` 和 `AvgTotalComponentDistCmp` 的对比。
2. 相同 `ef_search` 下 Recall 和距离计算节省率的对比。
3. Recall-`AvgSparseDistCmp` 曲线，观察达到相近 Recall 时两种配置的实际计算量。

### 16.6 Hybrid index 生命周期

UHG hybrid index 仍然严格按第 8.2 节构建到标准路径：

```text
/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index
```

不能写入 `results/comp_prune_dist`。独立运行本实验时必须严格串行：

```text
确认当前不存在其他 hybrid index
→ 构建当前数据集的 UHG hybrid index
→ 运行 prune_0.3 的五个搜索宽度
→ 运行 prune_no_prune 的五个搜索宽度
→ 校验当前数据集两个结果文件
→ 删除当前数据集的 UHG hybrid index
→ 验证 index 已不存在
→ 再处理下一个数据集
```

任何时刻最多只允许存在一个 hybrid index。发现不属于本次运行的已有 hybrid index
时必须停止并报告，不能自动删除；正常结束、失败或收到中断信号时，只能删除本实验
明确创建且路径完全匹配的当前 UHG hybrid index。

第 15 节 `comp_hops` 和本实验使用相同格式的 UHG hybrid index。如果两组实验连续
执行，可以在每个数据集的 index 构建完成后，先完成 `comp_hops`，再完成
`comp_prune_dist`，两边结果全部校验通过后只删除一次 index。这样仍然满足单 index
限制，并可避免为同一个数据集重复构建大型 index。

### 16.7 结果目录和完整性检查

除 hybrid index 外，实验脚本、日志、原始结果、汇总和图片都放在：

```text
/tbase-project/vsag/scripts/UHG/results/comp_prune_dist
```

计划使用以下结果布局：

```text
results/comp_prune_dist/
├── README.md
├── run_comp_prune_dist.sh
├── summarize.py
├── logs/
│   ├── index/
│   └── search/
├── <dataset>/
│   ├── prune_0.3/alpha_mixed.txt
│   └── prune_no_prune/alpha_mixed.txt
├── comp_prune_dist_points.tsv
└── plots/
```

每个结果文件必须恰好包含五个搜索宽度的完整指标。四个数据集应产生 8 个结果文件和
40 个指标点。除结果数量、字段和错误扫描外，还必须检查以下计数不变量：

```text
AvgDenseDistCmp >= AvgSparseDistCmp
AvgSparseSkipped >= 0
prune_no_prune: AvgSparseSkipped = 0
prune_no_prune: AvgDenseDistCmp = AvgSparseDistCmp
prune_0.3: AvgSparseDistCmp + AvgSparseSkipped = AvgDenseDistCmp
```

由于剪枝可能改变候选得分、后续搜索顺序和最终遍历路径，两组配置的 hops、
`AvgCandidateDistCmp` 和 `AvgDenseDistCmp` 不一定完全相同。因此校验应针对每组内部的
计数关系，不能错误地要求剪枝与不剪枝访问完全相同的候选集合。

只有在当前数据集的两个结果文件通过数量检查、字段检查、计数不变量检查和错误扫描
后，才可以删除当前 hybrid index 并继续下一个数据集。
