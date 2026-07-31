# UHG 入口策略消融实验运行手册

这个实验只补跑“无额外入口”的直接 UHG 搜索，并复用主实验已经生成的 `auto`
结果进行比较：

| alpha | Ablation 新跑 | 主实验复用 | 比较方式 |
|---:|---|---|---|
| `0.3` | `method=uhg` | `method=auto`，实际走 UHGS | 相同 `ef_search` 比较 Recall |
| `0.7` | `method=uhg` | `method=auto`，实际走 UHGH | 相同 `ef_search` 比较 Recall |

不会在 ablation 中重复运行 UHGS/UHGH。三个关键配置保持一致：同一个 UHG
`hybrid_index`、同一个 alpha、`hybrid_prune_scale=0.3`。

## 1. 前置依赖

直接 UHG 搜索需要：

```bash
ls "/tbase-project/vsag/build-release/examples/cpp/703_uhg_exp3"
ls "/tbase-project/vsag/scripts/UHG/data/hdf5/nq.hdf5"
ls "/tbase-project/vsag/scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index"
ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.3.npy"
ls "/tbase-project/vsag/scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.7.npy"
```

最终比较还需要主实验的两个结果文件：

```bash
ls "/tbase-project/vsag/scripts/UHG/results/main/nq/uhg/alpha_0.3.txt"
ls "/tbase-project/vsag/scripts/UHG/results/main/nq/uhg/alpha_0.7.txt"
```

主实验目录名是 `uhg`，但文件头中的 `method=auto` 表示 `alpha=0.3` 实际使用
UHGS、`alpha=0.7` 实际使用 UHGH。

## 2. Ablation 搜索参数

| 配置项 | 值 |
|---|---|
| 数据集 | `nq`, `hotpotqa`, `msmarco`, `dbpedia-entity` |
| method | 仅 `uhg` |
| alpha | 仅 `0.3`, `0.7` |
| top-k | `100` |
| query 数 | 全量 test query（`-1`） |
| ef_search | `100 200 300 500 1000` |
| `hybrid_prune_scale` | `0.3` |
| `max_hops` | `0` |
| `sindi_query_prune_ratio` | `0.5` |
| `sindi_term_prune_ratio` | `0` |
| `build_alpha` | `0.5` |
| threads | `1` |

虽然直接 UHG 不使用 SINDI/dense-entry 入口，脚本仍让 `sindi_bk`、
`dense_entry_bk` 和 `dense_entry_ef_search` 等于当前 `ef_search`，以保持命令参数
与主实验一致；这些入口参数在 `method=uhg` 分支中不会生效。

实验只加载已有 index，不传 `--rebuild`，不会重建索引。

## 3. 运行

运行 NQ：

```bash
mkdir -p "/tbase-project/vsag/scripts/UHG/results/ablation/logs"

bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/ablation/logs/run_ablation_nq.log"
```

运行全部四个数据集：

```bash
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_all.sh" \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/ablation/logs/run_ablation_all.log"
```

默认只产生两个新结果文件：

```text
/tbase-project/vsag/scripts/UHG/results/ablation/nq/uhg/alpha_0.3.txt
/tbase-project/vsag/scripts/UHG/results/ablation/nq/uhg/alpha_0.7.txt
```

快速验证：

```bash
RESULT_DIR=/tmp/vsag-ablation-smoke \
NUM_QUERIES=100 POINTS_TEXT=100 \
bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh" nq
```

## 4. 按相同 ef_search 比较 Recall

运行：

```bash
python3 "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/plot_ablation.py" nq \
  2>&1 | tee "/tbase-project/vsag/scripts/UHG/results/ablation/logs/plot_ablation_nq.log"
```

脚本只读取以下组合：

```text
alpha=0.3: ablation/uhg/alpha_0.3.txt  vs  main/uhg/alpha_0.3.txt (UHGS)
alpha=0.7: ablation/uhg/alpha_0.7.txt  vs  main/uhg/alpha_0.7.txt (UHGH)
```

它以 `ef_search` 为 key 取交集，保证只比较相同 `ef_search` 的 Recall；同时检查
两侧结果都是 `top-k=100`、全量 query、`build_alpha=0.5` 和
`hybrid_prune_scale=0.3`，避免快速验证结果或旧的 ablation `scale=0.5` 结果被误用。

输出：

```text
/tbase-project/vsag/scripts/UHG/results/ablation/plots/nq_ablation_recall_by_ef.png
/tbase-project/vsag/scripts/UHG/results/ablation/plots/nq_ablation_recall_by_ef.tsv
```

TSV 字段为：

```text
dataset, alpha, ef_search, uhg_recall, entry_method, entry_recall,
entry_minus_uhg_recall
```

检查搜索结果：

```bash
rg -c 'Recall:' "/tbase-project/vsag/scripts/UHG/results/ablation/nq/uhg"/alpha_{0.3,0.7}.txt
rg -n 'ERROR|NO_METRICS|Failed|Error:' "/tbase-project/vsag/scripts/UHG/results/ablation"
```
