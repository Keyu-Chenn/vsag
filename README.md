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

The experiment runner covers two groups:

- **Main experiments:** mixed-weight and fixed-weight Recall-QPS comparisons.
- **Ablation studies:** entry-point selection, hybrid pruning, and fixed-weight
  versus multi-weight graph construction.

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

### Main experiments

Run both mixed-alpha and fixed-alpha experiments:

```bash
python3 experiments/UniGraph/run.py main run
```

Run either main experiment separately:

```bash
python3 experiments/UniGraph/run.py main run --suite mixed_alpha
python3 experiments/UniGraph/run.py main run --suite fixed_alpha
```

For a quick single-dataset run:

```bash
python3 experiments/UniGraph/run.py main run \
  --suite mixed_alpha --datasets nq --methods uhg
```

### Ablation studies

Run all three ablation studies:

```bash
python3 experiments/UniGraph/run.py ablation run
```

Run an individual ablation:

```bash
python3 experiments/UniGraph/run.py ablation run --suite entry_point
python3 experiments/UniGraph/run.py ablation run --suite hybrid_pruning
python3 experiments/UniGraph/run.py ablation run --suite multi_weight
```

### Generate figures

Generate the main-experiment and ablation figures:

```bash
python3 experiments/UniGraph/run.py plot --group all --x-min 0.7 --x-max 1.0
```

This command reads the completed results without rerunning the experiments.
`--group all` renders both main-experiment and ablation figures, while
`--x-min` and `--x-max` set the displayed Recall range for the main curves.
The `multi_weight` ablation produces a summary table instead of a figure.

Generated logs, summaries, and figures are written under
`experiments/UniGraph/runs/main/` and `experiments/UniGraph/runs/ablation/`.

## License

This project follows the original VSAG [Apache License 2.0](LICENSE).
