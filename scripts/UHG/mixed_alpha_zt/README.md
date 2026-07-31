# Truncated-normal mixed-alpha experiment plan

This directory contains the data-preparation scripts, experiment runner, logs,
results, and validation tools for the normal-distribution mixed-alpha suite.
The suite is intentionally separate from the existing uniform-distribution
experiment in `scripts/UHG/mixed_alpha`.

Current completion status:

- Truncated-normal alpha assignments and exact top-200 ground truth are
  complete for all five datasets.
- ANN search is complete for `nq`, `hotpotqa`, `dbpedia-entity`, and `fever`.
- `msmarco` ANN search is intentionally deferred; no MSMARCO result file has
  been produced.
- The current validated result set contains 100 result files and 564
  Recall/QPS points.
- All temporary FHG/UHG hybrid indexes were deleted after their corresponding
  experiments.

## Experiment matrix

- Prepared datasets: `nq hotpotqa msmarco dbpedia-entity fever`
- Default ANN datasets: `nq hotpotqa dbpedia-entity fever`
- Deferred ANN dataset: `msmarco` (can be selected later with
  `DATASETS_TEXT=msmarco`)
- Methods: `hnsw sindi hnsw_sindi fhg uhg`
- Metrics: `Recall@10 Recall@20 Recall@50 Recall@100 Recall@200`
- Full test query set
- `threads=1`
- Serial execution only
- Full-matrix result files: `5 datasets × 5 methods × 5 recalls = 125`
- Full-matrix Recall/QPS points: `705`
- Current result files without MSMARCO: `4 × 5 × 5 = 100`
- Current Recall/QPS points without MSMARCO: `564`

## Alpha distribution

The phrase "normal distribution on 0.3-0.7" is implemented as:

```text
X ~ Normal(mu=0.5, sigma=0.1)
alpha = X conditioned on 0.3 <= X <= 0.7
seed = 42
```

The generator uses midpoint quantiles of the truncated distribution and then
shuffles them with seed 42. This mirrors the stratified construction used by
the uniform mixed-alpha experiment, avoids random finite-sample imbalance, and
does not create clipped endpoint spikes.

## Exact ground truth

Exact top-200 is computed once for every dataset:

```text
data/gt_top200/<dataset>/alphas.npy
data/gt_top200/<dataset>/ground_truth.npy
data/gt_top200/<dataset>/metadata.json
```

Recall@10, Recall@20, Recall@50, Recall@100, and Recall@200 all read this same
alpha/GT pair. Each search uses its own `-k`, so it consumes only the first
10/20/50/100/200 ground-truth ids for each query.

Prepare all datasets serially:

```bash
bash /tbase-project/vsag/scripts/UHG/mixed_alpha_zt/prepare_all.sh
```

Preparation supports resume by default. It writes logs to:

```text
/tbase-project/vsag/scripts/UHG/mixed_alpha_zt/logs/prepare
```

## Search parameter points

| Recall | HNSW | SINDI | HNSW+SINDI | FHG | UHG |
|---:|---|---|---|---|---|
| 10 | 10 20 30 50 80 100 | 100 150 200 300 500 1000 | 10 15 20 30 50 100 | 10 20 30 50 80 100 | 10 20 30 50 80 100 |
| 20 | 20 40 60 100 160 200 | 200 300 400 600 1000 2000 | 20 30 40 60 100 200 | 20 40 60 100 160 200 | 20 40 60 100 160 200 |
| 50 | 50 100 150 250 400 500 | 500 750 1000 1500 2500 5000 | 50 75 100 150 250 | 50 100 150 250 500 | 50 100 150 250 500 |
| 100 | 100 200 300 500 800 1000 | 1000 1500 2000 3000 5000 10000 | 100 150 200 300 500 | 100 200 300 500 1000 | 100 200 300 500 1000 |
| 200 | 200 400 600 1000 1600 2000 | 2000 3000 4000 6000 10000 20000 | 200 300 400 600 1000 | 200 400 600 1000 2000 | 200 400 600 1000 2000 |

Every search point is at least its corresponding top-k. Recall@100 uses the
main-experiment parameter points; Recall@50 uses the repository's existing
Recall@50 mixed-alpha grid; the other rows follow the completed uniform
mixed-alpha experiment.

## Shared immutable inputs

The suite reuses these existing artifacts under `scripts/UHG/data`:

- HDF5 datasets
- `701_<dataset>_dense_hnsw.index`
- `701_<dataset>_sparse_sindi.index`
- `709_<dataset>_sindi.index`
- FHG `.h5` graphs
- UHG `.h5` graphs

It does not rebuild or delete those artifacts.

FHG uses `build_alpha=0.5` and disables hybrid sparse upper-bound pruning.
UHG uses `method=auto`, `build_alpha=0.5`,
`hybrid_prune_scale=0.3`, and the same auto routing rule as the main runbook:

```text
alpha <= 0.5 -> UHGS
alpha > 0.5  -> UHGH
```

## Hybrid index lifecycle

For each dataset, `run_all.sh` performs these steps serially:

1. Run HNSW, SINDI, and HNSW+SINDI for all five recalls.
2. Build the FHG hybrid index only when FHG results are incomplete.
3. Run FHG for all five recalls.
4. Delete only
   `scripts/UHG/data/index/703_<dataset>_fhg_hybrid_index.index`.
5. Build the UHG hybrid index only when UHG results are incomplete.
6. Run UHG for all five recalls.
7. Delete only
   `scripts/UHG/data/index_hybrid_union/703_<dataset>_hybrid_index.index`.
8. Continue to the next dataset.

The deletion helper requires the canonical shared data directory, an explicit
five-dataset allowlist, an exact path template, and a regular non-symlink file.
It refuses any other deletion target. If a newly built hybrid index is followed
by a failure or signal, the trap deletes only that exact new hybrid index.
Before each build it also removes any stale allowlisted FHG/UHG hybrid index
from an interrupted earlier run, so two hybrid indexes cannot coexist.

Immediately before each FHG/UHG build, the runner reads the corresponding
measured size from
`scripts/UHG/results/results_integration/construction/index_size.txt`, compares
it with the current free space, and reserves an additional 2 GiB by default.
It refuses to start a build when the estimate plus reserve does not fit.

## Run and resume

After data preparation and validation:

```bash
python3 /tbase-project/vsag/scripts/UHG/mixed_alpha_zt/validate_inputs.py
```

Print the command plan without running searches or changing indexes:

```bash
DRY_RUN=1 \
bash /tbase-project/vsag/scripts/UHG/mixed_alpha_zt/run_all.sh
```

Run the complete suite:

```bash
bash /tbase-project/vsag/scripts/UHG/mixed_alpha_zt/run_all.sh
```

Results are atomically written to:

```text
results/recall_<10|20|50|100|200>/<dataset>/<method>.txt
```

Raw and runner logs are written below `results/logs`. Completed result files
are detected by their full metadata header, expected point count, and exact
point order, so the default `RESUME=1` safely skips them.

At successful completion, strict validation writes:

```text
results/mixed_alpha_zt_points.tsv
```

## Interactive QPS-Recall plot

Generate one self-contained HTML covering all completed top-k values:

```bash
python3 /tbase-project/vsag/scripts/UHG/mixed_alpha_zt/plot_results.py
```

Output:

```text
results/mixed_alpha_zt_qps_recall.html
```

The page matches the single-row panel layout, Recall range `0.7-1.0`, and
log-QPS range `10^1-10^3` of
`results/mixed_alpha/mixed_alpha_qps_recall.html`. Recall@K is selectable;
hover details, PDF printing, and SVG export remain available.
