# Hybrid 剪枝消融实验（test_prune）

本目录用于评估 UHG 混合搜索中 sparse 上界剪枝对 Recall-QPS 的影响。所有配置使用同一份 mixed-alpha query：每条 query 从 `alphas.npy` 读取自己的 alpha，并用对应的 `ground_truth.npy` 评估。实验固定 `--method=auto`，因此 `alpha<=0.5` 的 query 使用 UHGS，`alpha>0.5` 的 query 使用 UHGH；对比时只改变 `hybrid_prune_scale` 或关闭 hybrid pruning。

## 实验意图

混合剪枝先计算 dense inner product，再用 sparse query/base 范数的 Cauchy 上界判断是否值得计算 sparse inner product。`hybrid_prune_scale` 缩放这个 sparse 上界（见 `703_uhg_exp3 --help`）：

| hybrid_prune_scale | 含义 |
|---:|---|
| `0` | 上界判断忽略 sparse 项，剪枝最激进 |
| `< 1` | 缩小 sparse Cauchy 上界，更激进，可能损失召回 |
| `1` | 使用完整 sparse Cauchy 上界，属于保守剪枝，但仍会剪枝 |
| `> 1` | 放大上界，比 `1` 更保守 |

`scale=0` 不是「只按 sparse 计算」：它只影响是否继续计算 sparse 分数的上界判断；未被剪掉的候选仍按完整 dense+sparse 距离计算。真正关闭这层剪枝必须传 `--disable_hybrid_pruning`。该开关不会关闭标准图搜索由 `ef_search` 控制的候选队列剪枝。

本实验默认对比 `0.3`（激进剪枝）和 `no_prune`（传 `--disable_hybrid_pruning`）两个配置，以 `no_prune` 作为不使用混合上界剪枝的 baseline。

关键设计：两组配置加载**同一个** UHG `hybrid_index`（已内嵌 SINDI + dense-entry HNSW + UHG 主图），并读取**同一份** mixed-alpha/ground-truth 文件，不重建 index；搜索时只改变 `enable_hybrid_pruning` / `hybrid_prune_scale`。

## 默认参数

| 配置项 | 默认值 |
|---|---|
| query alpha | `data/mixed_alpha/<dataset>/alphas.npy` 中的逐 query alpha |
| method | `auto`（`alpha<=0.5` 走 UHGS，`alpha>0.5` 走 UHGH） |
| hybrid pruning | `0.3`（激进剪枝）、`no_prune`（关闭） |
| 数据集 | `nq`, `hotpotqa`, `msmarco`, `dbpedia-entity` |
| top-k | `100` |
| query 数 | 全量 test query（`-1`） |
| ef_search 点 | `100 200 300 500 1000` |
| `sindi_bk` | `= ef_search` |
| `dense_entry_bk` | `= ef_search` |
| `dense_entry_ef_search` | `= ef_search` |
| `max_hops` | `0`（不限制） |
| `sindi_query_prune_ratio` | `0.5` |
| `sindi_term_prune_ratio` | `0` |
| `build_alpha` | `0.5` |
| threads | `1` |

以上默认值均可通过环境变量覆盖（见下文）。

## 目录约定

```text
脚本目录： /tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune
实验结果： /tbase-project/vsag/scripts/UHG/results/test_prune
```

## 前置依赖

不重建任何索引，全部依赖主实验产物。确认以下文件存在：

```bash
# 二进制
ls /tbase-project/vsag/build-release/examples/cpp/703_uhg_exp3

# 数据与索引
ls /tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5
ls /tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index

# mixed-alpha query 权重和逐 query exact ground truth
ls /tbase-project/vsag/scripts/UHG/data/mixed_alpha/nq/alphas.npy
ls /tbase-project/vsag/scripts/UHG/data/mixed_alpha/nq/ground_truth.npy
```

若 `703_<dataset>_hybrid_index.index` 缺失，先按 `MAIN_EXPERIMENT_RUNBOOK.md` 第 8.2 步构建。若 mixed-alpha 文件缺失，先按第 12.2 步生成；正式实验应使用 `--num-queries -1` 的完整产物。

## 运行

脚本入口：

```text
/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh
```

参数约定：

| 位置参数 | 含义 | 可选值 |
|---|---|---|
| `$1` | 数据集；不传则跑全部 | `nq`, `hotpotqa`, `msmarco`, `dbpedia-entity` |
| `$2` | 剪枝配置；不传则跑全部 | 数值 prune scale 或 `no_prune` |

跑 NQ 的两个默认剪枝配置：

```bash
cd /tbase-project/vsag
bash scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh nq \
  2>&1 | tee scripts/UHG/results/test_prune/logs/run_test_prune_nq.log
```

跑全部数据集：

```bash
cd /tbase-project/vsag
mkdir -p scripts/UHG/results/test_prune/logs
bash scripts/UHG/scripts/hybrid_union/test_prune/run_all.sh \
  2>&1 | tee scripts/UHG/results/test_prune/logs/run_test_prune_all.log
```

只跑 NQ 的某一个剪枝配置：

```bash
bash scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh nq 0.3
bash scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh nq no_prune
```

或使用数据集 wrapper：

```bash
bash scripts/UHG/scripts/hybrid_union/test_prune/nq/run_test_prune.sh
bash scripts/UHG/scripts/hybrid_union/test_prune/hotpotqa/run_test_prune.sh
bash scripts/UHG/scripts/hybrid_union/test_prune/msmarco/run_test_prune.sh
bash scripts/UHG/scripts/hybrid_union/test_prune/dbpedia-entity/run_test_prune.sh
```

### 通过环境变量自定义

```bash
# 换 method / 剪枝配置组合；no_prune 表示关闭混合上界剪枝
METHOD=auto PRUNE_SCALES_TEXT="0.3 no_prune" \
  bash scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh nq

# 只跑指定 ef_search 点
POINTS_TEXT="200 500" \
  bash scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh nq
```

## 结果路径

```text
scripts/UHG/results/test_prune/<dataset>/prune_<scale|no_prune>/alpha_mixed.txt
```

例如：

```text
scripts/UHG/results/test_prune/nq/prune_0.3/alpha_mixed.txt
scripts/UHG/results/test_prune/nq/prune_no_prune/alpha_mixed.txt
```

结果行格式：

```text
auto ef_search=200 pruning=true prune_scale=0.3 sindi_qp=0.5 threads=1 Recall: 0.8123 QPS: 456.7
```

## 结果检查

```bash
# 每个数据集默认应有 2 个剪枝配置目录
ls scripts/UHG/results/test_prune/nq/

# 每个文件应包含 5 条 ef_search 点的指标行
rg -c 'Recall:' scripts/UHG/results/test_prune/nq/*/alpha_mixed.txt

# 检查错误
rg -n 'ERROR|NO_METRICS|Failed|Error:' scripts/UHG/results/test_prune
```

## 画图

画 QPS-Recall 曲线（每个数据集一张 mixed-alpha 图，按剪枝配置分曲线）：

```bash
cd /tbase-project/vsag
python3 scripts/UHG/scripts/hybrid_union/test_prune/plot_test_prune.py nq \
  2>&1 | tee scripts/UHG/results/test_prune/logs/plot_test_prune_nq.log
```

全部数据集：

```bash
python3 scripts/UHG/scripts/hybrid_union/test_prune/plot_test_prune.py
```

输出：

```text
scripts/UHG/results/test_prune/plots/nq_prune_ablation_qps_recall.png
scripts/UHG/results/test_prune/plots/nq_prune_ablation_summary.tsv
scripts/UHG/results/test_prune/plots/hotpotqa_prune_ablation_qps_recall.png
scripts/UHG/results/test_prune/plots/hotpotqa_prune_ablation_summary.tsv
scripts/UHG/results/test_prune/plots/msmarco_prune_ablation_qps_recall.png
scripts/UHG/results/test_prune/plots/msmarco_prune_ablation_summary.tsv
```

## 产物清单

按本文档执行后，NQ 会生成：

```text
结果文件（2 个 = 2 剪枝配置 × 1 份 mixed-alpha queries）:
  scripts/UHG/results/test_prune/nq/prune_0.3/alpha_mixed.txt
  scripts/UHG/results/test_prune/nq/prune_no_prune/alpha_mixed.txt

画图:
  scripts/UHG/results/test_prune/plots/nq_prune_ablation_qps_recall.png
  scripts/UHG/results/test_prune/plots/nq_prune_ablation_summary.tsv

日志:
  scripts/UHG/results/test_prune/logs/run_test_prune_nq.log
```

## 语法检查

```bash
bash -n scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh
bash -n scripts/UHG/scripts/hybrid_union/test_prune/run_all.sh
python3 -m py_compile scripts/UHG/scripts/hybrid_union/test_prune/plot_test_prune.py
python3 -m py_compile scripts/UHG/scripts/hybrid_union/test_prune/plot_test_prune_fig8_html.py
```
