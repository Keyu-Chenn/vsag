# Mixed-alpha Recall@10 / Recall@20 / Recall@200

本目录是独立的 mixed-alpha 实验目录；运行脚本、日志、结果和汇总均位于：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha
```

目标矩阵为 5 个数据集 × 5 个方法 × 3 个 top-k：

- 数据集：`nq hotpotqa msmarco dbpedia-entity fever`
- 方法：`hnsw sindi hnsw_sindi fhg uhg`
- 指标：`Recall@10 Recall@20 Recall@200`
- 全量 query、`threads=1`
- 每条 query 的 alpha 在 `[0.3, 0.7]` 上均匀分布，seed 为 `42`

`Recall@10` 和 `Recall@20` 使用
`data/mixed_alpha/<dataset>/ground_truth.npy`（exact top-100）；
`Recall@200` 使用
`data/mixed_alpha_gt_1000/<dataset>/ground_truth.npy`（exact top-1000）。
两套输入的 `alphas.npy` 已由预检要求逐元素相同。

## 参数点

| Recall | HNSW | SINDI | HNSW+SINDI | FHG | UHG |
|---:|---|---|---|---|---|
| 10 | 10 20 30 50 80 100 | 100 150 200 300 500 1000 | 10 15 20 30 50 100 | 10 20 30 50 80 100 | 10 20 30 50 80 100 |
| 20 | 20 40 60 100 160 200 | 200 300 400 600 1000 2000 | 20 30 40 60 100 200 | 20 40 60 100 160 200 | 20 40 60 100 160 200 |
| 200 | 200 400 600 1000 1600 2000 | 2000 3000 4000 6000 10000 20000 | 200 300 400 600 1000 | 200 400 600 1000 2000 | 200 400 600 1000 2000 |

所有参数点都不小于对应 top-k。FHG 固定使用 `build_alpha=0.5`，
关闭 hybrid sparse 上界剪枝；UHG 使用 `method=auto`、
`build_alpha=0.5` 和 `hybrid_prune_scale=0.3`。

## 磁盘安全与执行顺序

`run_all.sh` 按数据集串行执行：

1. 运行 HNSW、SINDI、HNSW+SINDI 的三个 Recall。
2. 确认该数据集的 UHG hybrid index 不存在；若存在，只按白名单删除该文件。
3. 构建或复用该数据集的 FHG hybrid index。
4. 连续完成 FHG 的 Recall@10/@20/@200。
5. 只删除
   `data/index/703_<dataset>_fhg_hybrid_index.index`。
6. 构建该数据集的 UHG hybrid index；构建时复用
   `709_<dataset>_sindi.index`（no-reorder）和
   `701_<dataset>_dense_hnsw.index`。
7. 连续完成 UHG 的 Recall@10/@20/@200。
8. 只删除
   `data/index_hybrid_union/703_<dataset>_hybrid_index.index`。
9. 进入下一个数据集。

NQ 开始前两个 hybrid index 都已存在，因此默认先精确删除暂时不用的 NQ UHG
hybrid index，复用 NQ FHG index 完成三个 Recall；删除 FHG 后再重新构建 UHG。
进入 FHG/UHG 阶段后，任一时刻至多保留一个 hybrid index。DRY_RUN 会模拟索引
存在状态，使打印出来的“复用、删除、重建”顺序与正式运行一致。

删除函数只接受上述两个精确路径模板、固定数据目录和数据集白名单；
会拒绝目录和符号链接。若新 hybrid index 构建后任务失败或被中断，脚本会删除
该轮新建的精确 index 以释放磁盘；正在复用的已有 index 若实验失败则保留以便续跑，
成功完成对应方法的三个 Recall 后删除。为保证单 index 生命周期而主动删除的另一个
hybrid index，则会等轮到该方法时重新构建。

FHG/UHG 的 `.h5` 图、HNSW/SINDI baseline index 和
`709_*_sindi.index` 均不会删除。

## 运行

只检查输入：

```bash
python3 /tbase-project/vsag/scripts/UHG/mixed_alpha/validate_inputs.py
```

打印完整命令但不运行实验、不构建或删除索引：

```bash
DRY_RUN=1 bash /tbase-project/vsag/scripts/UHG/mixed_alpha/run_all.sh
```

正式运行：

```bash
bash /tbase-project/vsag/scripts/UHG/mixed_alpha/run_all.sh
```

结果默认支持断点续跑：一个结果文件包含预期数量的 Recall/QPS 点时会跳过。
设置 `RESUME=0` 可强制重跑并原子替换该结果文件。
若某个数据集某种 hybrid 方法的三个目标 Recall 已全部完成，续跑会在构建前
识别这一状态，不会为了“检查结果”重新生成大索引；若对应 hybrid index 仍残留，
只会按上述白名单精确删除它。

也可以只选择数据集、Recall 或方法。例如：

```bash
DATASETS_TEXT="nq hotpotqa" \
RECALLS_TEXT="10 20" \
METHODS_TEXT="hnsw sindi hnsw_sindi" \
bash /tbase-project/vsag/scripts/UHG/mixed_alpha/run_all.sh
```

不要单独用 `run_search.sh` 启动 FHG/UHG，除非对应 hybrid index 已经存在；
该脚本不会构建或删除索引。

## 结果

单项结果：

```text
results/recall_<10|20|200>/<dataset>/<method>.txt
```

原始日志：

```text
results/logs/recall_<10|20|200>/<dataset>/<method>.log
results/logs/index/<dataset>_<fhg|uhg>_build.log
results/logs/runner/recall_<10|20|200>/<dataset>_<method>.log
```

## 交互式 QPS-Recall 图

生成包含 Recall@10、Recall@20 和 Recall@200 的交互式 HTML：

```bash
python3 /tbase-project/vsag/scripts/UHG/mixed_alpha/plot_results.py
```

输出：

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha/results/mixed_alpha_qps_recall.html
```

页面使用与 `results/mixed_alpha/mixed_alpha_qps_recall.html` 相同的单行面板、
Recall `0.7–1.0` 和对数 QPS `10¹–10³`；可切换 Recall@10、Recall@20、
Recall@200，并支持悬停参数点、打印 PDF 和 SVG 导出。

全套实验成功后会严格检查全部 75 个方法结果和 435 个参数点，并生成：

```text
results/mixed_alpha_points.tsv
```
