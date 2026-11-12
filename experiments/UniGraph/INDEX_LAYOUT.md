# UniGraph index construction and layout

Indexes are generated locally from the external HDF5 files and are ignored by
Git. They do not need to be downloaded or uploaded.

## Build

First build the executables and place the HDF5 inputs as described in
`DATA.md`. Then run:

```bash
python3 experiments/UniGraph/run.py indexes check
python3 experiments/UniGraph/run.py indexes plan --datasets nq
python3 experiments/UniGraph/run.py indexes build --datasets nq
python3 experiments/UniGraph/run.py indexes build --datasets all
```

`--datasets` accepts `all` or a comma-separated subset of `nq`, `hotpotqa`,
`msmarco`, `fever`, and `dbpedia-entity`. The default construction budget is
32 threads, matching the paper run; it can be changed with `--threads`.

The builder runs each dataset in this dependency order:

1. dense HNSW and reordered/non-reordered SINDI guide indexes;
2. the fixed-weight FHG graph and the multi-weight UHG graph;
3. deployable FHG and UHG packages;
4. naive and optimized Hybrid-HNSW baselines;
5. the LADR graph extracted from dense HNSW.

Existing non-empty outputs are skipped. Each new component is built in
`index/.staging/` and moved into place only after successful completion. Raw
logs are written under `runs/index-build/`.

## Generated files

For every `<dataset>`, the builder produces:

```text
index/
├── guide-indexes/
│   ├── 701_<dataset>_dense_hnsw.index
│   ├── 701_<dataset>_sparse_sindi.index
│   └── 709_<dataset>_sindi.index
├── deployable-indexes/<dataset>/
│   ├── fhg.index
│   └── uhg.index
├── baseline-indexes/<dataset>/
│   ├── naive_hybrid_hnsw.index
│   └── optimized_hybrid_hnsw.index
├── ladr/<dataset>/graph.index
└── graphs/
    ├── fhg/<dataset>_fhg_alpha_0_5.h5
    └── uhg/<dataset>_uhg.h5
```

The files under `graphs/` and the non-reordered `709_*_sindi.index` are retained
construction intermediates used to resume or repackage an index. Search does
not read them directly.

Table V uses graph-only search and directly reuses
`deployable-indexes/<dataset>/fhg.index` and `uhg.index`. The historical
`index/multi_weight/` copies are not read by the current code.

After construction, validate all search inputs without running an experiment:

```bash
python3 experiments/UniGraph/run.py check
```
