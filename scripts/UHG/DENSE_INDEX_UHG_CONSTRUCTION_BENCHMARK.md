# HNSW 与 HGraph 构建 UHG 的时间对比

更新时间：2026-07-11（CST）

## 1. 实验目的与当前状态

本实验考察在 NQ 数据集上，使用 HNSW 或 HGraph 作为 dense candidate index 时：

1. dense candidate index 本身的构建时间和文件大小；
2. UHG generation 阶段的耗时；
3. HGraph 是否值得替代当前主实验使用的 HNSW。

当前已经完成 NQ 的全量 dense index 构建，以及前 100,000 个训练点的 UHG
generation 对照。实验进程已经结束；原始日志在删除前均包含正常完成标记，关键数字
已整理到本文。尚未完成全量 UHG、召回质量对比和其他数据集复现。2026-07-11 已按要求
删除本轮日志、新索引和两份 100,000 点对照图。

## 2. 数据与实验参数

- 数据集：NQ
- 训练点数：2,681,468
- dense 维度：1,024
- 距离：inner product（`ip`）
- graph `max_degree`：64
- dense `ef_construction`：200
- dense `ef_search`：200
- UHG `k`：32
- candidate pool `bk`：100
- alpha：0.0 到 1.0，步长 0.1
- SINDI `query_prune_ratio`：0.9
- UHG generation 线程数：1
- UHG 样本：训练集前 100,000 点
- refine：跳过

HNSW 使用：

```json
{"dtype":"float32","metric_type":"ip","dim":1024,
 "hnsw":{"max_degree":64,"ef_construction":200}}
```

HGraph 使用：

```json
{"dtype":"float32","metric_type":"ip","dim":1024,
 "index_param":{"base_quantization_type":"sq8","max_degree":64,
                "ef_construction":200}}
```

这里有一个重要限制：HGraph 的 base quantization 是 `sq8`，HNSW 的图内向量存储
配置与之不同。因此当前结果是两套实际工程配置的对比，不是只替换 graph algorithm、
保持数据表示完全相同的严格微基准。尤其是索引大小和构建速度，都会受到量化配置影响。

## 3. 实验结果

### 3.1 Dense candidate index

| 方法 | ef_construction | 构建时间 | 文件大小 | 相对 HNSW efc=200 |
|---|---:|---:|---:|---:|
| HGraph | 200 | 390.405 s（6.51 min） | 3,593,370,385 B（3.35 GiB） | 快 25.18 倍 |
| HNSW | 200 | 9,829.32 s（2.730 h） | 12,410,223,202 B（11.56 GiB） | 基准 |
| HNSW | 100 | 3,720.725 s（1.034 h） | 12,410,223,202 B（11.56 GiB） | 补充实验 |

在 `ef_construction=200` 的已有结果中，HGraph 的构建时间比 HNSW 少 96.03%，
索引文件小约 71.04%。新补跑的 HNSW `ef_construction=100` 明显快于
`ef_construction=200`，但仍是本次 HGraph 构建时间的 9.53 倍。

HNSW `ef_construction=200` 来自之前的 dedicated in-memory benchmark；HGraph 和
HNSW `ef_construction=100` 是 2026-07-11 的运行。各次运行没有记录或固定底层构建
线程数、CPU 频率及机器负载，因此这些构建时间应视为工程测量，而非严格隔离的
CPU microbenchmark。

### 3.2 UHG generation（100,000 点）

该阶段复用已经构建好的 dense index 和同一个 SINDI，只统计日志中的
`Generation time`，不包含 HDF5 加载、索引反序列化和输出序列化。

| 方法 | Generation time | Dense search | Sparse search | Merge | 平均邻居数 |
|---|---:|---:|---:|---:|---:|
| HGraph | 344.940 s | 201.269 s | 110.665 s | 31.654 s | 52.0119 |
| HNSW | 617.039 s | 466.695 s | 115.379 s | 33.430 s | 52.0073 |

HGraph 的 UHG generation 比 HNSW 少 44.10% 时间，即约快 1.79 倍。Sparse search
和 merge 时间接近，主要差距来自 dense search：HGraph 为 201.269 秒，HNSW 为
466.695 秒，HGraph 在该阶段约快 2.32 倍。

两种方法生成的平均邻居数基本一致，但这不能代替召回率或图质量评估。

### 3.3 全量线性估计

如果按 `2,681,468 / 100,000` 对 generation time 做简单线性外推：

| 方法 | 全量 generation 估计 | 加上 efc=200 dense index 构建 |
|---|---:|---:|
| HGraph | 9,249.46 s（2.569 h） | 2.678 h |
| HNSW | 16,545.70 s（4.596 h） | 7.326 h |

这只是规划后续实验所需资源的估计，不是实测全量结果。全量运行可能受缓存、内存、
数据分布和长时间机器负载变化影响。

## 4. 当前结论

在当前 NQ 工程配置下，HGraph 同时表现出更低的 dense index 构建时间、更小的索引
文件和更快的 UHG dense candidate generation。100,000 点样本中，UHG generation
约快 1.79 倍；按已有 `ef_construction=200` 记录计算，dense index 构建约快 25.18 倍。

但目前还不能直接决定主实验全面切换到 HGraph，原因包括：

1. HGraph 使用 `sq8`，不是与 HNSW 完全相同的数据表示；
2. 尚未比较 HGraph 与 HNSW 生成的 UHG 的 recall/QPS 或最终检索质量；
3. UHG 只运行了前 100,000 点，且跳过 refine；
4. 只验证了 NQ；
5. dense index 构建实验没有严格固定和记录底层并发及机器负载。

建议下一步先在同一批 query 和 ground truth 上比较两份 100,000 点图的质量。如果质量
可接受，再做 NQ 全量；否则应先对齐 quantization，并扫描 HGraph 的构建/搜索参数。

## 5. 复现方式

以下命令根据日志和程序参数整理。运行前应确认目标索引路径，避免 `--rebuild` 覆盖
主实验正在使用的索引。

### 5.1 HGraph dense index 构建

`706_uhg_exp6` 会先为全部训练点构建 dense index，再仅为 `--num_points` 指定的点生成
UHG。使用 1 个点可以用很小的额外开销触发并计时完整 dense index 构建。

```bash
build-release/examples/cpp/706_uhg_exp6 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  -k 32 --bk 100 \
  --dense_index_type hgraph \
  --dense_ef_construction 200 --dense_ef_search 200 \
  --alpha_step 0.1 --query_prune_ratio 0.9 \
  --merge_max_degree 64 --skip_refine \
  --num_points 1 --threads 1 \
  --index_dir scripts/UHG/data/index \
  --output /tmp/nq_uhg_hgraph_build_smoke.h5
```

如果 `701_nq_dense_hgraph.index` 已经存在，程序会直接加载它；需要重新测量时必须先把
输出放到独立 index directory，或明确使用 `--rebuild`。

### 5.2 100,000 点 UHG generation

将 `--dense_index_type` 分别设为 `hgraph` 和 `hnsw`，其余参数保持一致：

```bash
build-release/examples/cpp/706_uhg_exp6 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  -k 32 --bk 100 \
  --dense_index_type hgraph \
  --dense_ef_construction 200 --dense_ef_search 200 \
  --alpha_step 0.1 --query_prune_ratio 0.9 \
  --merge_max_degree 64 --skip_refine \
  --num_points 100000 --threads 1 \
  --index_dir scripts/UHG/data/index \
  --output /tmp/nq_uhg_hgraph_ef200_bk100_100k.h5
```

```bash
build-release/examples/cpp/706_uhg_exp6 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  -k 32 --bk 100 \
  --dense_index_type hnsw \
  --dense_ef_construction 200 --dense_ef_search 200 \
  --alpha_step 0.1 --query_prune_ratio 0.9 \
  --merge_max_degree 64 --skip_refine \
  --num_points 100000 --threads 1 \
  --index_dir scripts/UHG/data/index \
  --output /tmp/nq_uhg_hnsw_efc200_efs200_bk100_100k.h5
```

### 5.3 HNSW ef_construction=100 补充实验

```bash
build-release/examples/cpp/708_uhg_exp8 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --ef_construction 100 --build_only --rebuild \
  --index_dir scripts/UHG/data/index_benchmark/efc100
```

## 6. 实验文件盘点与清理建议

### 6.1 日志状态

以下本轮日志已于 2026-07-11 删除，关键数字保留在本文第 3 节：

```text
results/main/logs/construction/dense_index_benchmark/nq_hgraph_efc200_build.log
results/main/logs/construction/dense_index_benchmark/nq_uhg_hgraph_ef200_bk100_100k.log
results/main/logs/construction/dense_index_benchmark/nq_uhg_hnsw_efc200_efs200_bk100_100k.log
results/main/logs/construction/dense_index_benchmark/nq_hnsw_efc100_full_build.log
```

旧的 HNSW efc=200 dedicated 记录 `results/construction_time/nq/nq_hnsw.log` 仍保留；
它不是本轮新日志。

### 6.2 索引状态

| 文件 | 大小 | 状态 |
|---|---:|---|
| `data/index/701_nq_dense_hgraph.index` | 3.35 GiB | 本轮新产物，已于 2026-07-11 删除 |
| `data/index_benchmark/efc100/701_nq_dense_hnsw.index` | 11.56 GiB | 本轮新产物，已于 2026-07-11 删除 |
| `data/index/701_nq_dense_hnsw.index` | 11.56 GiB | 旧的主实验共享索引，仍保留 |

两个新索引删除后，空的 `data/index_benchmark/efc100` 和 `data/index_benchmark`
目录也已移除。

### 6.3 `/tmp` 临时产物

2026-07-11 已清理以下 smoke、10k/FHG 临时样本和临时日志，共回收约 22 MiB：

```text
/tmp/nq_uhg_hnsw_smoke.h5
/tmp/nq_uhg_hgraph_build_smoke.h5
/tmp/nq_uhg_hnsw_ef200_bk100_10k.h5
/tmp/nq_uhg_hnsw_ef200_bk200_10k.h5
/tmp/nq_uhg_hnsw_ef200_bk100_10k.log
/tmp/nq_uhg_hnsw_ef200_bk200_10k.log
/tmp/nq_fhg_hnsw_ef200_bk100_10k.log
/tmp/nq_fhg_hnsw_ef200_bk200_10k.log
/tmp/fhg_bk_test/
/tmp/fhg_hnsw_ef200_bk100_10k/
/tmp/fhg_hnsw_ef200_bk200_10k/
```

以下两个 100,000 点 UHG HDF5 合计约 99.2 MiB，已于 2026-07-11 删除：

```text
/tmp/nq_uhg_hgraph_ef200_bk100_100k.h5
/tmp/nq_uhg_hnsw_efc200_efs200_bk100_100k.h5
```

本次盘点没有发现仍在运行的相关进程，也没有发现明显的零字节或构建中断索引。

### 6.4 仓库根目录的无关垃圾文件

盘点期间还发现 3 个 2026-07-10 生成的未跟踪零字节文件：

```text
0.5$
alpha_t,
texttt{ef_search}$}{
```

这些名字像是 shell 误解析 LaTeX 文本后创建的文件，不属于有效源码、日志或实验产物。
它们已于 2026-07-11 清理。
