# Experiment data

The compact ground-truth `.npy` arrays are stored in Git. The large HDF5
payloads are external: prepare the five BGE-M3 dense--sparse datasets listed
below and place them under `experiments/UniGraph/data/hdf5/` using the layout
in this document.

| Dataset | Directory name | Queries |
|---|---|---:|
| Natural Questions | `nq` | 1,000 |
| HotpotQA | `hotpotqa` | 1,000 |
| MS MARCO | `msmarco` | 1,000 |
| FEVER | `fever` | 1,000 |
| DBpedia-Entity | `dbpedia-entity` | 467 |

## HDF5 schema

Create one file named `data/hdf5/<dataset>.hdf5` for each dataset. Each file
must contain these six datasets:

| HDF5 dataset | Type and shape | Meaning |
|---|---|---|
| `train` | `float32 [N, 1024]` | corpus dense vectors |
| `train_labels` | `int64 [N]` | corpus labels returned by search |
| `train_sparse` | `uint8 [bytes]` | serialized corpus sparse vectors |
| `test` | `float32 [Q, 1024]` | query dense vectors |
| `test_labels` | `int64 [Q]` | query labels |
| `test_sparse` | `uint8 [bytes]` | serialized query sparse vectors |

Sparse vectors are concatenated in row order. Every vector is encoded as:

```text
[uint32 nnz][nnz × uint32 term_id][nnz × float32 value]
```

The number and order of serialized sparse vectors must match the corresponding
dense and label rows. HDF5 root attributes, chunk sizes, and compression are not
used by the experiment programs.

## Required layout

```text
data/
├── hdf5/
│   ├── nq.hdf5
│   ├── hotpotqa.hdf5
│   ├── msmarco.hdf5
│   ├── fever.hdf5
│   └── dbpedia-entity.hdf5
├── ground_truth/<dataset>/
│   └── <dataset>_ground_truth_alpha_<alpha>.npy
├── mixed_alpha_gt_1000/<dataset>/
│   ├── alphas.npy
│   └── ground_truth.npy
└── mixed_0_1/<dataset>/
    ├── alphas.npy
    └── ground_truth.npy
```

The fixed-alpha experiment uses `alpha` values `0.3`, `0.4`, `0.5`, `0.6`, and
`0.7` for NQ, MS MARCO, and DBpedia-Entity. The entry-point ablation additionally
uses only `0.3` and `0.7` for HotpotQA and FEVER. Every `alphas.npy` is
`float32 [Q]`, and every ground-truth array is `int64 [Q, K]`.
`mixed_alpha_gt_1000` contains top-1000 results and is shared by the main
mixed-weight experiment and the pruning ablation. Its alpha values use a
uniform distribution over `[0.3, 0.7]` with seed 42. `mixed_0_1` contains
top-100 exact full-corpus results; its query-specific alpha values span `[0,1]`
as shuffled midpoint quantiles with seed 42, with ties ordered by descending
score and then ascending label. The committed alpha arrays define the workloads
exactly, so no ground-truth regeneration is needed to reproduce the searches.

After placing the HDF5 inputs and building the executables, verify the index
builder prerequisites and generate the indexes:

```bash
python3 experiments/UniGraph/run.py indexes check
python3 experiments/UniGraph/run.py indexes build --datasets all
```

After preparing the HDF5 files and indexes, validate the complete data and index
layout without running any searches:

```bash
python3 experiments/UniGraph/run.py check
```
