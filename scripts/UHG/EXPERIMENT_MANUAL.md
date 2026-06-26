# UHG Experiment Manual for Agents

This manual explains how to run the UHG experiments in this repository. It is written for an automated agent that needs to reproduce, extend, or debug the experiments without relying on unstated context.

## 1. Start Here

Always run UHG commands from the repository root:

```bash
cd /tbase-project/vsag
```

The shell scripts under `scripts/UHG/scripts/` call binaries through relative paths such as `./build-release/examples/cpp/703_uhg_exp3`. If the current directory is not `/tbase-project/vsag`, the scripts will fail or run the wrong binary.

Build the release binaries before running experiments:

```bash
make release
```

The required example binaries are registered in `examples/cpp/CMakeLists.txt`:

```text
701_uhg_exp1
702_uhg_exp2
703_uhg_exp3
706_uhg_exp6
```

After building, verify that these exist:

```bash
ls build-release/examples/cpp/701_uhg_exp1
ls build-release/examples/cpp/702_uhg_exp2
ls build-release/examples/cpp/703_uhg_exp3
ls build-release/examples/cpp/706_uhg_exp6
```

## 2. Directory Map

Important paths:

```text
scripts/UHG/data/hdf5/              Input dense+sparse HDF5 datasets
scripts/UHG/data/distances/         Precomputed dense/sparse distance matrices
scripts/UHG/data/ground_truth/      Exact hybrid ground truth by dataset and alpha
scripts/UHG/data/index/             Default serialized index cache
scripts/UHG/data/index_refined/     Refined NQ serialized index cache
scripts/UHG/data/index_refined_hotpotqa/ Refined HotpotQA serialized index cache
scripts/UHG/data/fhg/               FHG precomputed graph HDF5 files
scripts/UHG/data/uhg/               UHG precomputed graph HDF5 files
scripts/UHG/results/                Experiment text outputs and generated figures
scripts/UHG/scripts/                Shell drivers, preprocessing scripts, plotting scripts
```

Input HDF5 files are expected to contain at least:

```text
train
test
train_sparse
test_sparse
train_labels
test_labels
```

Ground truth files are named like this:

```text
scripts/UHG/data/ground_truth/<gt_subdir>/<dataset>_ground_truth_alpha_<alpha>.npy
```

Examples:

```text
scripts/UHG/data/ground_truth/nq/nq_ground_truth_alpha_0.3.npy
scripts/UHG/data/ground_truth/hotpotqa_1000/hotpotqa_ground_truth_alpha_0.3.npy
```

## 3. What Each Binary Does

### 701_uhg_exp1: HNSW+SINDI baseline

Use this for the two-route baseline. It searches a dense HNSW index and a sparse SINDI index independently, merges candidate ids, computes exact dense and sparse distances for the merged candidates, reranks by hybrid score, and reports Recall/QPS.

Index cache names:

```text
701_<dataset>_dense_hnsw.index
701_<dataset>_sparse_sindi.index
```

Current defaults:

```text
k = 10
bk_dense = 100
bk_sparse = 100
ef_search = 200
alpha = 0.5
```

The shell scripts usually pass `--bk` and `--ef_search` explicitly, so these defaults mostly matter for direct manual runs.

Smoke test:

```bash
./build-release/examples/cpp/701_uhg_exp1 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --gt_dir scripts/UHG/data/ground_truth/nq \
  --index_dir scripts/UHG/data/index \
  --alpha 0.3 \
  --num_queries 10 \
  -k 100 \
  --bk 500 \
  --ef_search 500
```

### 702_uhg_exp2: offline UHG graph generator

Use this to generate a precomputed UHG graph HDF5 file without the extra refine stage in 706. It does not report Recall/QPS and is not a query experiment.

It uses dense HGRAPH and sparse SINDI helper indexes, collects dense/sparse candidates for each base point, evaluates the candidate union over sampled alpha values, and writes `neighbors` plus `neighbor_counts` to an HDF5 file.

Index cache names used by 702:

```text
701_<dataset>_dense_hgraph.index
701_<dataset>_sparse_sindi.index
```

Current defaults:

```text
k = 32
bk = 500
query_prune_ratio = 0.5
term_prune_ratio = 0.0
alpha_step = 0.1
threads = 8
```

Example for generating an NQ UHG graph:

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

### 703_uhg_exp3: UHG/UHGS/FHG search experiments

Use this for graph-based hybrid search. It creates or loads a `sindi` index and a `hybrid_index`, then runs one of two methods:

```text
--method uhg   Single-entry hybrid graph search. It passes entry_point=0.
--method uhgs  SINDI-guided hybrid graph search. It first gets SINDI candidates and passes them as entry_points.
```

Index cache names:

```text
703_<dataset>_sindi.index
703_<dataset>_hybrid_index.index
```

For FHG and UHG comparisons, the shell scripts copy prebuilt graph-index files onto the generic `703_<dataset>_hybrid_index.index` path before calling `703_uhg_exp3`:

```text
FHG source: 703_<dataset>_fhg_hybrid_index.index
UHG source: 703_<dataset>_uhg_hybrid_index.index
Runtime generic target: 703_<dataset>_hybrid_index.index
```

This copy only overwrites the generic runtime cache, not the source FHG/UHG cache.

Smoke test for UHG:

```bash
./build-release/examples/cpp/703_uhg_exp3 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --method uhg \
  -k 100 \
  --alpha 0.3 \
  --gt_dir scripts/UHG/data/ground_truth/nq \
  --index_dir scripts/UHG/data/index \
  --hybrid_prune_scale 0.5 \
  --max_hops 0 \
  --sindi_bk 100 \
  --ef_search 500 \
  --num_queries 10
```

Smoke test for UHGS:

```bash
./build-release/examples/cpp/703_uhg_exp3 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --method uhgs \
  -k 100 \
  --alpha 0.3 \
  --gt_dir scripts/UHG/data/ground_truth/nq \
  --index_dir scripts/UHG/data/index \
  --hybrid_prune_scale 0.7 \
  --max_hops 0 \
  --sindi_bk 1000 \
  --ef_search 500 \
  --num_queries 10
```

### 706_uhg_exp6: offline refined UHG graph generator

Use this to generate a precomputed UHG graph HDF5 file. It does not report Recall/QPS and is not a query experiment.

It performs these steps:

```text
1. Load train dense and train sparse vectors.
2. Load or build dense HGRAPH and sparse SINDI helper indexes.
3. For each base point, retrieve dense and sparse candidate pools.
4. Evaluate the candidate union over alpha values 0.0, 0.1, ..., 1.0 by default.
5. Merge neighbors that appear in the top-k set for any sampled alpha.
6. Refine the graph with reverse edges, RNG-style pruning, and optional connectivity repair.
7. Save an HDF5 file with datasets neighbors and neighbor_counts.
```

Index cache names used by 706:

```text
701_<dataset>_dense_hgraph.index
701_<dataset>_sparse_sindi.index
```

Current defaults:

```text
k = 32
bk = 500
query_prune_ratio = 0.5
term_prune_ratio = 0.0
alpha_step = 0.1
refine_max_degree = 64
refine_alpha = 0.5
refine_rng = 1.2
threads = 8
```

Example for generating an NQ refined graph:

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

Important limitation: `706_uhg_exp6` only writes the graph HDF5 file. Current `HybridIndex::add_one_point()` does not accept a graph path from command line. It reads a hardcoded HDF5 path in `src/algorithm/hybrid_index/hybrid_index.cpp`.

At the time of writing, the active hardcoded path is:

```text
/tbase-project/vsag/scripts/UHG/data/uhg/hotpotqa_uhg_refined.h5
```

Therefore, do not use `--rebuild` with `703_uhg_exp3` for another dataset unless you first update the hardcoded graph path, rebuild, and intentionally regenerate the serialized index. The safest way to reproduce existing refined experiments is to use the existing serialized caches in `data/index_refined/` and `data/index_refined_hotpotqa/` without `--rebuild`.

## 4. Preprocessing and Ground Truth

Most data needed for the checked-in experiments already exists. Only regenerate preprocessing artifacts when they are missing or when input data changes.

### Compute dense/sparse distance matrices

Use `compute_distances.py` to produce memmapped `.npy` files under `scripts/UHG/data/distances/`:

```bash
python scripts/UHG/scripts/compute_distances.py \
  --input scripts/UHG/data/hdf5/nq.hdf5 \
  --output scripts/UHG/data/distances \
  --dense-metric ip \
  --batch-size 10000 \
  --sparse-batch-size 10000
```

Output names follow this pattern:

```text
<dataset>_dense_distances.npy
<dataset>_sparse_distances.npy
```

This step is expensive for large datasets because it computes train-by-test distances.

### Compute ground truth

Use `compute_ground_truth.py` after distance matrices exist:

```bash
python scripts/UHG/scripts/compute_ground_truth.py \
  --distances scripts/UHG/data/distances \
  --output scripts/UHG/data/ground_truth/nq \
  --dataset nq \
  --topk 100
```

For Recall@1000:

```bash
python scripts/UHG/scripts/compute_ground_truth.py \
  --distances scripts/UHG/data/distances \
  --output scripts/UHG/data/ground_truth/nq_1000 \
  --dataset nq \
  --topk 1000
```

The script currently has built-in dataset configs for `nq`, `msmarco`, and `hotpotqa`. Existing Quora and Webis ground-truth files should be reused unless the script is extended with their train/query sizes.

## 5. Running Main Experiments

Run one smoke test first before launching full sweeps. Full sweeps can take a long time and may overwrite result text files.

Do not run multiple 4-method scripts for the same dataset and `INDEX_DIR` in parallel. These scripts copy FHG/UHG source indexes into the shared generic path `703_<dataset>_hybrid_index.index`; parallel runs can race and mix methods.

Most result directories already exist in this workspace. If a script fails while opening its output file, create the matching result directory first, for example:

```bash
mkdir -p scripts/UHG/results/hotpotqa_4methods
```

### HotpotQA k=100 4-method experiment

One alpha:

```bash
bash scripts/UHG/scripts/run_hotpotqa_4methods.sh 0.3
```

All paper alpha values:

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash scripts/UHG/scripts/run_hotpotqa_4methods.sh "$alpha"; done
```

Output:

```text
scripts/UHG/results/hotpotqa_4methods/alpha_<alpha>.txt
```

Methods inside this script:

```text
hnsw_sindi  701 baseline
fhg         703 with FHG hybrid index copied into generic cache path
uhg         703 with UHG hybrid index copied into generic cache path
uhgs        703 UHGS, using SINDI entry points
```

### NQ k=1000 4-method experiment

One alpha:

```bash
bash scripts/UHG/scripts/run_nq_4methods_k1000.sh 0.3
```

All alpha values:

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash scripts/UHG/scripts/run_nq_4methods_k1000.sh "$alpha"; done
```

Output:

```text
scripts/UHG/results/nq_4methods_k1000/alpha_<alpha>.txt
```

### HotpotQA k=1000 4-method experiment

One alpha:

```bash
bash scripts/UHG/scripts/run_hotpotqa_4methods_k1000.sh 0.3
```

All alpha values:

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash scripts/UHG/scripts/run_hotpotqa_4methods_k1000.sh "$alpha"; done
```

Output:

```text
scripts/UHG/results/hotpotqa_4methods_k1000/alpha_<alpha>.txt
```

### Quora k=100 4-method experiment

One alpha:

```bash
bash scripts/UHG/scripts/run_quora_4methods.sh 0.3
```

All alpha values:

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash scripts/UHG/scripts/run_quora_4methods.sh "$alpha"; done
```

Output:

```text
scripts/UHG/results/quora_4methods/alpha_<alpha>.txt
```

Optional arguments:

```text
bash scripts/UHG/scripts/run_quora_4methods.sh <alpha> <k> <gt_subdir> <result_subdir>
```

Example for k=1000 if matching GT exists:

```bash
bash scripts/UHG/scripts/run_quora_4methods.sh 0.3 1000 quora_1000 quora_4methods_k1000
```

### Webis k=100 4-method experiment

One alpha:

```bash
bash scripts/UHG/scripts/run_webis_4methods.sh 0.3
```

All alpha values:

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash scripts/UHG/scripts/run_webis_4methods.sh "$alpha"; done
```

Output:

```text
scripts/UHG/results/webis-touche2020_4methods/alpha_<alpha>.txt
```

Optional arguments:

```text
bash scripts/UHG/scripts/run_webis_4methods.sh <alpha> <k> <gt_subdir> <result_subdir>
```

## 6. Running Refined UHG Experiments

Refined experiments compare old baselines with refined UHG/UHGS curves. Prefer using the existing serialized refined indexes. Do not pass `--rebuild` unless you have updated the hardcoded graph path in `HybridIndex` and rebuilt the project.

### NQ refined k=100

```bash
bash scripts/UHG/scripts/run_exp_nq_refined_uhg.sh
```

Output:

```text
scripts/UHG/results/nq_refined_uhg/refined_uhg_uhgs.txt
```

This script uses:

```text
DATASET=scripts/UHG/data/hdf5/nq.hdf5
GT_DIR=scripts/UHG/data/ground_truth/nq
INDEX_DIR=scripts/UHG/data/index_refined
```

It runs `uhg` and `uhgs` for alpha values `0.1` to `0.9` and `ef_search` values `100,200,300,500`.

### HotpotQA refined k=100

Base refined points:

```bash
bash scripts/UHG/scripts/run_exp_hotpotqa_refined_uhg.sh
```

Extra high-ef points:

```bash
bash scripts/UHG/scripts/run_exp_hotpotqa_refined_uhg_extra.sh
```

Outputs:

```text
scripts/UHG/results/hotpotqa_refined_uhg/refined_uhg_uhgs.txt
scripts/UHG/results/hotpotqa_refined_uhg/refined_uhg_uhgs_extra.txt
```

These scripts use:

```text
DATASET=scripts/UHG/data/hdf5/hotpotqa.hdf5
GT_DIR=scripts/UHG/data/ground_truth/hotpotqa
INDEX_DIR=scripts/UHG/data/index_refined_hotpotqa
```

## 7. Older Single-Method Scripts

These scripts run one method across alpha values and print results directly to stdout. Use them for debugging or regenerating old-style logs.

```bash
bash scripts/UHG/scripts/run_exp_nq_hnsw_sindi.sh
bash scripts/UHG/scripts/run_exp_nq_uhg.sh
bash scripts/UHG/scripts/run_exp_nq_uhgs.sh
bash scripts/UHG/scripts/run_exp_hotpotqa_hnsw_sindi.sh
bash scripts/UHG/scripts/run_exp_hotpotqa_uhg.sh
bash scripts/UHG/scripts/run_exp_hotpotqa_uhgs.sh
```

If a persistent log is needed, redirect stdout and stderr explicitly:

```bash
bash scripts/UHG/scripts/run_exp_nq_uhg.sh > scripts/UHG/results/nq_uhg_stdout.log 2>&1
```

## 8. Plotting

Plot scripts parse result text files and write PNG/PDF outputs.

HotpotQA 4-method k=100:

```bash
python scripts/UHG/scripts/plot_qps_recall_hotpotqa.py
```

Outputs:

```text
scripts/UHG/results/hotpotqa_4methods_final.png
scripts/UHG/results/hotpotqa_4methods_final.pdf
```

NQ refined:

```bash
python scripts/UHG/scripts/plot_qps_recall_nq_refined.py
```

Outputs:

```text
scripts/UHG/results/nq_refined_uhg/nq_refined_4methods_final.png
scripts/UHG/results/nq_refined_uhg/nq_refined_4methods_final.pdf
```

HotpotQA refined:

```bash
python scripts/UHG/scripts/plot_qps_recall_hotpotqa_refined.py
```

Outputs:

```text
scripts/UHG/results/hotpotqa_refined_uhg/hotpotqa_refined_4methods_final.png
scripts/UHG/results/hotpotqa_refined_uhg/hotpotqa_refined_4methods_final.pdf
```

Avoid `plot_qps_recall.py` and `plot_qps_recall_nq_v2.py` for fresh experiments unless you intentionally update their hardcoded data arrays. They do not parse the current result text files.

## 9. Result Format and Validation

The C++ binaries print a result block containing:

```text
Recall: <value>
QPS: <value>
```

Most shell scripts extract these two lines and write compact records like:

```text
uhgs ef_search=500 sindi_bk=1000 prune_scale=0.7 Recall: 0.986928 QPS: 54.0088
```

After running a script, validate that the output file contains method records:

```bash
grep -E "^(hnsw_sindi|fhg|uhg|uhgs)" scripts/UHG/results/hotpotqa_4methods/alpha_0.3.txt
```

If a result line is missing `Recall:` or `QPS:`, inspect the full command output. The script may have hidden an error because it pipes command output through `grep`.

## 10. Experiment Archiving

Every experiment run should leave enough artifacts for another agent to reproduce and audit it later. Do not rely only on terminal history.

Create a per-run archive directory before a non-trivial run:

```bash
RUN_DIR="scripts/UHG/results/runs/$(date +%Y%m%d_%H%M%S)_<dataset>_<experiment_name>"
mkdir -p "$RUN_DIR"
```

Save the exact script or command set used for the run:

```bash
cp scripts/UHG/scripts/run_hotpotqa_4methods.sh "$RUN_DIR/"
```

For ad-hoc command lines, write a small runnable shell script and save it in `RUN_DIR` before running it. The script should include the dataset path, `gt_dir`, `index_dir`, `alpha`, `k`, `bk`, `ef_search`, pruning parameters, and output paths.

Save full stdout/stderr, not only the compact result file:

```bash
bash scripts/UHG/scripts/run_hotpotqa_4methods.sh 0.3 2>&1 | tee "$RUN_DIR/run.log"
```

Save the generated result text files and figures into the same archive directory:

```bash
cp scripts/UHG/results/hotpotqa_4methods/alpha_0.3.txt "$RUN_DIR/"
cp scripts/UHG/results/hotpotqa_4methods_final.png "$RUN_DIR/" 2>/dev/null || true
cp scripts/UHG/results/hotpotqa_4methods_final.pdf "$RUN_DIR/" 2>/dev/null || true
```

For 702/706 graph-construction runs, always save the construction log because it contains construction cost and graph statistics:

```bash
./build-release/examples/cpp/706_uhg_exp6 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --index_dir scripts/UHG/data/index \
  --threads 16 \
  -k 32 \
  --bk 500 \
  --output scripts/UHG/data/uhg/nq_uhg_refined.h5 \
  2>&1 | tee "$RUN_DIR/construct_graph.log"
```

The construction log should be checked for at least these fields:

```text
Generation time
dense search total
sparse search total
merge total
Statistics before refine
Refine time
Statistics after refine
Saved to <graph_file>
```

Also save the generated graph file path and file metadata:

```bash
ls -lh scripts/UHG/data/uhg/nq_uhg_refined.h5 | tee "$RUN_DIR/graph_file.txt"
```

If an index is rebuilt, save the index directory listing and the exact graph HDF5 path used by `HybridIndex`:

```bash
ls -lh scripts/UHG/data/index_refined | tee "$RUN_DIR/index_files.txt"
grep -n "std::string h5_file" src/algorithm/hybrid_index/hybrid_index.cpp | tee "$RUN_DIR/hybrid_index_graph_path.txt"
```

At the end of a run, write a short summary file:

```text
Dataset:
Method(s):
k:
Alpha values:
Index directory:
Ground-truth directory:
Graph file, if any:
Command or script:
Main result files:
Construction time, if any:
Notes and failures:
```

## 11. Rebuilding Indexes Safely

Most experiments should load existing index caches. Rebuilding can be very expensive and can accidentally use the wrong graph file.

For 701 baseline indexes, rebuilding is usually safe:

```bash
./build-release/examples/cpp/701_uhg_exp1 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --gt_dir scripts/UHG/data/ground_truth/nq \
  --index_dir scripts/UHG/data/index \
  --alpha 0.3 \
  -k 100 \
  --bk 500 \
  --ef_search 500 \
  --rebuild
```

For 703 `hybrid_index`, rebuilding is not safe unless the intended graph HDF5 path is active in `src/algorithm/hybrid_index/hybrid_index.cpp`. The current implementation has no command-line graph path. If rebuilding UHG/FHG hybrid indexes, use this sequence:

```text
1. Generate or choose the correct graph HDF5 file under scripts/UHG/data/uhg/ or scripts/UHG/data/fhg/.
2. Edit the hardcoded h5_file in src/algorithm/hybrid_index/hybrid_index.cpp.
3. Run make release.
4. Run 703_uhg_exp3 with --rebuild and a dedicated --index_dir.
5. Rename or copy the generated 703_<dataset>_hybrid_index.index to the intended source cache name.
6. Restore the source code if the hardcoded path was only temporary.
```

Do not overwrite checked-in or shared index caches unless explicitly requested.

## 12. Common Failure Modes

Wrong current directory:

```text
Symptom: ./build-release/examples/cpp/703_uhg_exp3: No such file or directory
Fix: run from /tbase-project/vsag.
```

Missing release binary:

```text
Symptom: build-release/examples/cpp/701_uhg_exp1 does not exist
Fix: run make release with VSAG_ENABLE_EXAMPLES=ON.
```

Missing ground truth:

```text
Symptom: Cannot open npy file: ..._ground_truth_alpha_X.X.npy
Fix: use the correct gt_dir for k=100 vs k=1000, or regenerate ground truth.
```

Wrong FHG/UHG runtime index:

```text
Symptom: FHG and UHG results look identical or unexpectedly stale
Fix: inspect scripts that copy 703_<dataset>_fhg_hybrid_index.index or 703_<dataset>_uhg_hybrid_index.index to 703_<dataset>_hybrid_index.index.
```

Accidental wrong graph on rebuild:

```text
Symptom: rebuilt NQ refined index uses HotpotQA graph or crashes
Fix: do not rebuild 703 until the hardcoded graph path in HybridIndex is set correctly.
```

## 13. Recommended Agent Workflow

Use this order for a complete, low-risk experiment session:

```text
1. Read this manual and identify the target dataset, k, alpha range, and method set.
2. Confirm that the needed HDF5 data and ground-truth files exist.
3. Build release binaries with make release if build-release is missing or stale.
4. Run one 10-query smoke test with 701 and/or 703.
5. Create a per-run archive directory and save the exact script or command list.
6. Run the desired shell driver for one alpha with stdout/stderr captured by `tee`.
7. Inspect the result text file for Recall/QPS lines.
8. Run the full alpha sweep only after the one-alpha run succeeds.
9. Save full logs, result text files, plots, graph-construction logs, and graph/index metadata.
10. Generate plots only after all required result files exist.
11. Do not pass --rebuild for 703 unless the graph path issue has been handled.
12. Report exact commands, changed files, result files, construction time, and any skipped steps.
```

## 14. Minimal Command Cheat Sheet

Build:

```bash
make release
```

HotpotQA k=100 full sweep:

```bash
for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do bash scripts/UHG/scripts/run_hotpotqa_4methods.sh "$alpha"; done
python scripts/UHG/scripts/plot_qps_recall_hotpotqa.py
```

NQ refined:

```bash
bash scripts/UHG/scripts/run_exp_nq_refined_uhg.sh
python scripts/UHG/scripts/plot_qps_recall_nq_refined.py
```

HotpotQA refined:

```bash
bash scripts/UHG/scripts/run_exp_hotpotqa_refined_uhg.sh
bash scripts/UHG/scripts/run_exp_hotpotqa_refined_uhg_extra.sh
python scripts/UHG/scripts/plot_qps_recall_hotpotqa_refined.py
```

Generate a refined graph with 706:

```bash
./build-release/examples/cpp/706_uhg_exp6 \
  scripts/UHG/data/hdf5/nq.hdf5 \
  --index_dir scripts/UHG/data/index \
  --threads 16 \
  -k 32 \
  --bk 500 \
  --output scripts/UHG/data/uhg/nq_uhg_refined.h5
```

Remember: the 706 graph output is not automatically consumed by 703 unless `HybridIndex` is rebuilt with the matching hardcoded graph path or an existing serialized index already contains that graph.
