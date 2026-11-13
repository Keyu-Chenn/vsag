# UniGraph experiments

This directory contains the scripts, configurations, and ground truth used to
reproduce the experiments in the UniGraph paper.

## Structure

```text
experiments/UniGraph/
├── run.py           experiment entry point
├── build.sh         build UniGraph and baseline programs
├── requirements.txt Python dependencies
├── DATA.md          input data format
├── INDEX_LAYOUT.md  generated index layout
├── scripts/         experiment and plotting scripts
├── data/            ground truth and external HDF5 inputs
├── index/           locally generated indexes
├── bin/             locally built programs
└── runs/            generated logs and results
```

## Setup

From the repository root, create the Python environment and build the programs:

```bash
python3 -m venv experiments/UniGraph/.venv
experiments/UniGraph/.venv/bin/pip install -r experiments/UniGraph/requirements.txt
bash experiments/UniGraph/build.sh
```

Prepare the five HDF5 datasets according to [DATA.md](DATA.md), then build the
indexes:

```bash
python3 experiments/UniGraph/run.py indexes check --datasets all
python3 experiments/UniGraph/run.py indexes build --datasets all
```

## Run

Check the complete setup:

```bash
python3 experiments/UniGraph/run.py check
```

### Main experiments

Run both mixed-alpha and fixed-alpha experiments:

```bash
python3 experiments/UniGraph/run.py main run
```

Run one main suite or a small dataset subset when needed:

```bash
python3 experiments/UniGraph/run.py main run --suite mixed_alpha
python3 experiments/UniGraph/run.py main run --suite fixed_alpha
python3 experiments/UniGraph/run.py main run \
  --suite mixed_alpha --datasets nq --methods uhg
```

### Ablation studies

Run all three ablation studies or select one:

```bash
python3 experiments/UniGraph/run.py ablation run
python3 experiments/UniGraph/run.py ablation run --suite entry_point
python3 experiments/UniGraph/run.py ablation run --suite hybrid_pruning
python3 experiments/UniGraph/run.py ablation run --suite multi_weight
```

### Generate figures

Generate the figures:

```bash
python3 experiments/UniGraph/run.py plot --group all --x-min 0.7 --x-max 1.0
```

Generated files are written under `runs/main/` and `runs/ablation/`.
