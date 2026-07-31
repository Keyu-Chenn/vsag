# exp_7: 701-710 实验说明

这个目录记录 `examples/cpp/701_uhg_exp1.cpp` 到 `710_uhg_exp10.cpp` 在当前 UHG 主实验里的分工。主实验保留三类总入口脚本，并为 `nq/`、`hotpotqa/`、`msmarco/`、`fever/`、`dbpedia-entity/` 提供数据集专用脚本：

- `../run_baselines.sh`: HNSW、SINDI、HNSW+SINDI 三个 baseline。
- `../run_fhg.sh`: FHG 搜索实验。
- `../run_uhg.sh`: UHG 搜索实验，`method=auto`，其中 alpha `<= 0.5` 时实际走 UHGS，UHGS entry 阶段的 SINDI query prune 设为 `0.5`。
- `../<dataset>/run_all.sh`: 只运行该数据集的 baselines、FHG 和 UHG。
- `../<dataset>/run_baselines.sh`、`../<dataset>/run_fhg.sh`、`../<dataset>/run_uhg.sh`: 只运行该数据集的对应实验。

所有新脚本和当前 703 搜索入口都不再包含旧的 distance table 相关参数。

## 默认实验设置

| 项目 | 默认值 |
|---|---|
| 数据集 | `nq hotpotqa msmarco fever dbpedia-entity` |
| alpha | `0.3 0.4 0.5 0.6 0.7` |
| top-k | `100` |
| query 数 | `-1`，即使用程序默认的全量 test query |
| 检索线程 | `1`，通过 `THREADS=N` 开 query-level 并发 |
| 数据目录 | `/tbase-project/vsag/scripts/UHG/data` |
| 结果目录 | `/tbase-project/vsag/scripts/UHG/results/main` |

Ground truth:

```text
/tbase-project/vsag/scripts/UHG/data/ground_truth/<dataset>/<dataset>_ground_truth_alpha_<alpha>.npy
```

结果:

```text
/tbase-project/vsag/scripts/UHG/results/main/<dataset>/<method>/alpha_<alpha>.txt
```

## 701-710 定位

| 编号 | 二进制 | 当前定位 | 是否进入 exp_7 主脚本 |
|---|---|---|---|
| 701 | `701_uhg_exp1` | HNSW+SINDI 双路召回 baseline，合并 dense/sparse 候选后按 hybrid score 重排。 | 是，`run_baselines.sh` 的 `hnsw_sindi` |
| 702 | `702_uhg_exp2` | 早期 UHG multi-alpha 图构建，未做后续 refine。 | 否，保留作历史构图入口 |
| 703 | `703_uhg_exp3` | 统一 hybrid_index 搜索入口，支持 `uhg/uhgs/uhgh/auto`，也是 FHG/UHG 搜索主入口。 | 是，`run_fhg.sh` 和 `run_uhg.sh` |
| 704 | `704_uhg_exp4` | FHG fixed-alpha 图构建，输出 `{dataset}_fhg_alpha_<alpha>.h5`。 | 否，搜索脚本使用已构建的 FHG graph/hybrid_index |
| 705 | `705_uhg_exp5` | 旧版 HNSW+SINDI baseline，功能和 701 重叠，参数更少。 | 否，优先使用 701 |
| 706 | `706_uhg_exp6` | 当前 UHG 图构建入口；主实验使用多 alpha merge，不执行 refine。 | 否，搜索脚本使用已构建的 UHG graph |
| 707 | `707_uhg_exp7` | 旧版 hybrid graph 搜索入口，已被 703 覆盖。 | 否，优先使用 703 |
| 708 | `708_uhg_exp8` | Dense-only HNSW baseline，HNSW 召回后用 raw sparse 补分并重排。 | 是，`run_baselines.sh` 的 `hnsw` |
| 709 | `709_uhg_exp9` | Sparse-only SINDI baseline，SINDI 召回后用 raw dense 补分并重排。 | 是，`run_baselines.sh` 的 `sindi` |
| 710 | `710_uhg_exp10` | UHG 构图参数调参工具，默认采样 10000 点，支持 index-based 和 brute-force graph 对比。 | 否，保留作调参/质量诊断 |

## 脚本入口

从仓库根目录或任意目录都可以运行，脚本内部使用绝对默认路径：

```bash
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_fhg.sh
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh
```

只跑一个数据集：

```bash
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh nq
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_fhg.sh nq
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh nq
```

等价的数据集专用脚本：

```bash
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/nq/run_all.sh
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/hotpotqa/run_all.sh
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/msmarco/run_all.sh
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/fever/run_all.sh
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/dbpedia-entity/run_all.sh
```

只跑一个 baseline 方法：

```bash
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh nq hnsw
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh nq sindi
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh nq hnsw_sindi
bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/nq/run_baselines.sh hnsw
```

常用环境变量覆盖：

```bash
ALPHAS_TEXT="0.3 0.5" POINTS_TEXT="100 300 500" \
  bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh nq

RESULT_DIR=/tmp/uhg_exp7_smoke NUM_QUERIES=100 \
  bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh nq hnsw

THREADS=8 \
  bash /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh nq
```

`THREADS` 只并行不同 query；单条 query 的检索参数和重排逻辑不变。`THREADS=1` 的结果和旧单线程 QPS 可直接比较，`THREADS>1` 表示多请求并发吞吐，结果行会写出线程数。

## 主实验参数

### Baselines

`run_baselines.sh` 包含三个方法：

| 方法 | 二进制 | 参数点 |
|---|---|---|
| `hnsw` | `708_uhg_exp8` | `bk = ef_search = 100 200 300 500 800 1000` |
| `sindi` | `709_uhg_exp9` | `bk = 1000 1500 2000 3000 5000 10000` |
| `hnsw_sindi` | `701_uhg_exp1` | `bk = ef_search = 100 150 200 300 500` |

纯 `sindi` 和 `hnsw_sindi` baseline 默认都使用
`query_prune_ratio=0.5, term_prune_ratio=0`，与 UHGS sparse-entry 对齐，
可以通过 `SINDI_QUERY_PRUNE_RATIO` 和 `SINDI_TERM_PRUNE_RATIO` 覆盖。

### FHG

`run_fhg.sh` 使用 `703_uhg_exp3`，输入为 FHG graph 和 FHG hybrid_index cache:

```text
/tbase-project/vsag/scripts/UHG/data/fhg/<dataset>_fhg_alpha_0_5.h5
/tbase-project/vsag/scripts/UHG/data/index/703_<dataset>_fhg_hybrid_index.index
```

固定参数：

```text
构图参数：dense HNSW, bk=100, dense_ef_search=200, query_prune_ratio=0.9, alpha=0.5，不执行 refine
method = uhg
enable_hybrid_pruning = false (`--disable_hybrid_pruning`)
max_hops = 0
ef_search = 100 200 300 500 1000
```

### UHG

`run_uhg.sh` 使用 `703_uhg_exp3`，输入为 UHG graph、dense-entry HNSW index 和 UHG hybrid_index cache。外部 `703_<dataset>_sindi.index` 不再作为主实验输入；UHGS 需要的 SINDI 会从 HDF5 构建并序列化进 hybrid_index cache，搜索时直接从 cache 反序列化 internal SINDI。

```text
/tbase-project/vsag/scripts/UHG/data/uhg/<dataset>_uhg.h5
/tbase-project/vsag/scripts/UHG/data/index/701_<dataset>_dense_hnsw.index
/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index
```

固定参数：

```text
构图参数：dense HNSW, bk=100, dense_ef_search=200, query_prune_ratio=0.9, merge_max_degree=64, skip_refine
method = auto
build_alpha = 0.5
hybrid_prune_scale = 0.5
sindi_query_prune_ratio = 0.5
sindi_term_prune_ratio = 0
ef_search = sindi_bk = dense_entry_bk = dense_entry_ef_search
points = 100 200 300 500 1000
```

`method=auto` 的实际分支：

| alpha | 实际方法 |
|---:|---|
| `0.3`, `0.4`, `0.5` | `uhgs` |
| `0.6`, `0.7` | `uhgh` |

## 清理建议

可以保留作为主线的内容：

- `main/exp_7/README.md`
- `main/run_baselines.sh`
- `main/run_fhg.sh`
- `main/run_uhg.sh`
- `main/nq/`, `main/hotpotqa/`, `main/msmarco/`
- `data/hdf5`, `data/ground_truth`, `data/index`, `data/index_hybrid_union`, `data/uhg`

不影响主实验、可后续归档或清理的内容：

- 父目录中旧的合并脚本、补点脚本、一次性调参脚本。
- `__pycache__` 和历史 `.log`。
- `test_prune` 中围绕旧 distance table 叙述的说明和日志。
- 705/707 对应的旧实验入口说明；源码可保留，但不作为主实验调用入口。
