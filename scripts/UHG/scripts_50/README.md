# UHG Recall@50 实验脚本

本目录是 `scripts/UHG/scripts/hybrid_union` 四组实验的 Recall@50 独立版本。搜索 top-k 默认改为 `50`，并使用针对 Recall@50 下调后的搜索参数点；alpha、路由、剪枝和索引配置保持不变。结果统一写入 `scripts/UHG/results_50`。

现有 fixed-alpha 和 mixed-alpha ground truth 都包含 exact top-100。搜索程序在 `-k 50` 时只读取每行前 50 个 ID，因此本实验直接复用 `scripts/UHG/data/ground_truth` 和 `scripts/UHG/data/mixed_alpha`，不需要重新生成 GT，也不会覆盖 Recall@100 数据。

默认搜索参数点：

| 方法 | 参数点 |
|---|---|
| HNSW | `50 100 150 250 400 500` |
| SINDI | `500 750 1000 1500 2500 5000` |
| HNSW+SINDI | `50 75 100 150 250` |
| FHG/UHG | `50 100 150 250 500` |

## 运行顺序

以 NQ 为例，依次运行四组实验：

```bash
bash /tbase-project/vsag/scripts/UHG/scripts_50/hybrid_union/main/nq/run_all.sh
bash /tbase-project/vsag/scripts/UHG/scripts_50/hybrid_union/mixed_alpha/run_mixed_alpha.sh nq
bash /tbase-project/vsag/scripts/UHG/scripts_50/hybrid_union/ablation/run_ablation.sh nq
bash /tbase-project/vsag/scripts/UHG/scripts_50/hybrid_union/test_prune/run_test_prune.sh nq
```

也可以顺序运行上述四条命令并保存日志：

```bash
bash /tbase-project/vsag/scripts/UHG/scripts_50/hybrid_union/run_four_experiments.sh nq
```

Ablation 的画图脚本会读取本目录生成的 Recall@50 main 结果进行比较，所以应先完成 main。索引、FHG/UHG 图和 mixed-alpha 输入仍复用 `scripts/UHG/data` 下的现有产物。

## 结果目录

```text
scripts/UHG/results_50/main
scripts/UHG/results_50/mixed_alpha
scripts/UHG/results_50/ablation
scripts/UHG/results_50/test_prune
```

所有搜索脚本仍允许通过环境变量覆盖参数；默认 `K=50`。正式 Recall@50 实验不建议覆盖 `K`。
