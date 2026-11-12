# UniGraph

This repository contains the implementation and experiment code for
**UniGraph: A Unified Graph-based Index for Dynamic Hybrid Dense-Sparse Vector
Retrieval**.

UniGraph supports hybrid retrieval with query-dependent dense-sparse weights by
combining graph neighborhoods built at multiple weights into one bounded-degree
index. This repository is forked from [VSAG](https://github.com/antgroup/vsag);
the UniGraph implementation is integrated into the VSAG source tree, and the
reproducible experiment entry point is
[`experiments/UniGraph`](experiments/UniGraph/).

## Experiments

The experiment runner covers:

- mixed-weight and fixed-weight Recall-QPS comparisons (Figures 5 and 6);
- entry-point and hybrid-pruning ablations (Figures 8 and 9);
- fixed-weight graph versus multi-weight UniGraph (Table V).

The experiments use five BGE-M3 dense-sparse datasets: Natural Questions,
HotpotQA, MS MARCO, FEVER, and DBpedia-Entity.

## Requirements

- Linux
- CMake 3.18 or newer
- a C++17 compiler, Make, and OpenMP
- Python 3

Create the Python environment from the repository root:

```bash
python3 -m venv experiments/UniGraph/.venv
experiments/UniGraph/.venv/bin/pip install -r experiments/UniGraph/requirements.txt
```

## Prepare the data

Place the five HDF5 datasets at:

```text
experiments/UniGraph/data/hdf5/
├── nq.hdf5
├── hotpotqa.hdf5
├── msmarco.hdf5
├── fever.hdf5
└── dbpedia-entity.hdf5
```

The required HDF5 layout is described in
[`experiments/UniGraph/DATA.md`](experiments/UniGraph/DATA.md). The experiment
workloads and ground-truth arrays are included in the repository.

## Build

Build the UniGraph and baseline executables:

```bash
bash experiments/UniGraph/build.sh
```

To select the number of parallel build jobs:

```bash
JOBS=16 bash experiments/UniGraph/build.sh
```

## Build the indexes

Check the inputs and build all indexes:

```bash
python3 experiments/UniGraph/run.py indexes check --datasets all
python3 experiments/UniGraph/run.py indexes build --datasets all
```

The index build is resumable. To build only one dataset, replace `all` with its
name, for example `nq`.

## Run the experiments

Validate the complete experiment setup:

```bash
python3 experiments/UniGraph/run.py check
```

Run all main experiments for Figures 5, 6, 8, and 9:

```bash
python3 experiments/UniGraph/run.py main run
```

Individual experiment suites can also be run separately:

```bash
python3 experiments/UniGraph/run.py main run --suite mixed_alpha
python3 experiments/UniGraph/run.py main run --suite fixed_alpha
python3 experiments/UniGraph/run.py main run --suite ablation1
python3 experiments/UniGraph/run.py main run --suite ablation2
```

Run the multi-weight graph experiment for Table V:

```bash
python3 experiments/UniGraph/run.py multi-weight run
```

For a quick single-dataset run:

```bash
python3 experiments/UniGraph/run.py main run \
  --suite mixed_alpha --datasets nq --methods uhg
```

Generate the figures after completing the main experiments:

```bash
python3 experiments/UniGraph/run.py plot --suite all --x-min 0.7 --x-max 1.0
```

Generated logs, summaries, and figures are written under
`experiments/UniGraph/runs/`.

## License

This project follows the original VSAG [Apache License 2.0](LICENSE).
