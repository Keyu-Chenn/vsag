# Tech Paper Skeleton: Hybrid Graph Index for Dense-Sparse Retrieval

## 1. Paper-type Positioning

- **Type:** Technique Paper
- **Rationale:** Dense-sparse hybrid retrieval is an established retrieval setting; the paper's main contribution is a new **hybrid graph index construction method** that builds graph edges for dense+sparse objects under hybrid similarity, rather than only improving search-time entry routing.

## 2. Corrected Core Thesis

The core work is not merely guided search over an existing graph. The core work is the construction of a **hybrid graph index** whose neighborhood structure is designed for dense-sparse hybrid similarity.

The construction prototype is in:

- `examples/cpp/601_hybrid_exp1.cpp`

The baseline two-route retrieval experiment is in:

- `examples/cpp/603_hybrid_exp3.cpp`

The later guided-entry search experiment over the constructed hybrid graph is in:

- `examples/cpp/605_hybrid_exp5.cpp`

The revised paper thesis should be:

> Existing dense-sparse hybrid retrieval systems usually maintain separate dense and sparse indexes, then merge and rerank candidates. This two-route design does not build a graph structure that directly reflects hybrid similarity. We propose a hybrid graph index that constructs graph neighbors from dense and sparse candidate pools, selects neighbors under a family of hybrid weights, and refines the graph with pruning and connectivity repair. The resulting index supports direct hybrid graph traversal and improves the recall-efficiency trade-off over two-route baselines.

## 3. Thinking Template

| Stage | Your content |
|---|---|
| Research background | Modern retrieval systems combine dense semantic vectors and sparse lexical vectors because the two representations capture complementary evidence. Dense vectors capture semantic similarity but may miss exact lexical matches; sparse vectors preserve keyword-level evidence but are high-dimensional and expensive to search. Hybrid retrieval is therefore important for RAG, web search, and production retrieval systems, but efficient approximate search under hybrid similarity remains difficult. |
| Limitation 1 | Existing two-route hybrid retrieval builds separate dense and sparse indexes, searches them independently, merges candidates, and reranks by a hybrid score. This pipeline does not construct a neighborhood graph that directly reflects hybrid similarity. |
| Limitation 2 | A neighbor that is useful under hybrid similarity may not be top-ranked by dense-only or sparse-only search alone. Therefore, two-route candidate merging can miss or underrepresent graph edges that are important for hybrid traversal. |
| Limitation 3 | Building a graph for one fixed dense-sparse weight `alpha` can be brittle, because hybrid retrieval may need to work across different dense/sparse trade-offs. A useful hybrid graph should preserve neighborhoods across a range of hybrid weights, while keeping degree and connectivity under control. |
| Key Idea / Our Goal | Construct a hybrid graph index by generating dense and sparse candidate pools, evaluating candidates under multiple hybrid weights, and retaining graph neighbors that cover the hybrid neighborhood structure across alpha values. |
| Challenge 1 | How can the construction algorithm obtain a candidate pool rich enough to contain useful hybrid neighbors without doing full brute-force search over all points? |
| Challenge 2 | How can the graph select neighbors that remain useful across different dense-sparse weights instead of overfitting to one fixed alpha? |
| Challenge 3 | How can the constructed graph maintain practical graph-index properties, including bounded degree, reciprocal reachability, and global connectivity? |
| Methodology topic sentence | We build a hybrid graph index through candidate generation from dense and sparse retrievers, all-alpha hybrid neighbor selection, and graph refinement with reverse-edge insertion, RNG-style pruning, and connectivity repair. |
| Module A (addresses Challenge 1) | **Dense-Sparse Candidate Generation.** For every point, the construction procedure queries a dense HGraph and a sparse SINDI index, unions their top candidates, and recomputes exact dense and sparse inner products for the union. |
| Module B (addresses Challenge 2) | **All-alpha Hybrid Neighbor Selection.** For each point, the method scans `alpha` from 0 to 1, computes `alpha * dense_score + (1-alpha) * sparse_score`, and inserts the top-k candidates at each alpha into the point's hybrid neighbor set. |
| Module C (addresses Challenge 3) | **Graph Refinement and Export.** The method adds reverse-edge candidates, applies RNG-style pruning to bound degree and remove dominated neighbors, repairs disconnected components, and exports the final adjacency lists to HDF5 for `HybridIndex` construction. |
| Contribution 1 | We identify that two-route hybrid retrieval lacks a graph topology aligned with hybrid similarity and can miss neighbors that matter for hybrid traversal. (Section 1 and Section 3) |
| Contribution 2 | We propose an all-alpha hybrid graph construction method that combines dense and sparse candidate generation with hybrid neighbor selection across the dense-sparse weight space. (Section 4) |
| Contribution 3 | We design graph refinement steps, including reverse-edge augmentation, RNG-style pruning, and connectivity repair, to make the hybrid graph usable as an ANN index. (Section 4) |
| Contribution 4 | We evaluate the resulting hybrid graph index against two-route baselines such as `hgraph + hgraph` and `hgraph + sindi`, and study guided-entry search as an extension over the constructed hybrid graph. (Section 5) |

## 4. Methodology Outline

### 4.1 Overview

The method section should present the hybrid graph index as a construction-time contribution with a search-time payoff.

The pipeline is:

1. Build auxiliary dense and sparse indexes for construction-time candidate generation.
2. For each data point, retrieve dense candidates from HGraph and sparse candidates from SINDI.
3. Recompute exact dense and sparse scores for the union candidate pool.
4. Sweep `alpha` from sparse-dominant to dense-dominant settings.
5. Insert the top-k hybrid candidates under each alpha into the point's neighbor set.
6. Refine the graph with reverse edges, RNG-style pruning, and connectivity repair.
7. Export the final graph as adjacency lists consumed by `HybridIndex`.

This makes `601_hybrid_exp1.cpp` the main methodology anchor.

### 4.2 Construction-time Candidate Generation

This subsection addresses Challenge 1.

Implementation anchors in `601_hybrid_exp1.cpp`:

- Lines 503-526 build dense HGraph.
- Lines 528-551 build sparse SINDI.
- Lines 578-582 retrieve dense and sparse candidates for each data point.
- Lines 584-609 union candidates and recompute exact dense/sparse scores.

Key technical point:

> The auxiliary dense and sparse indexes are not the final retrieval system. They are construction-time tools used to cheaply approximate a rich candidate pool for hybrid graph edge selection.

Suggested text:

The construction algorithm avoids full all-pairs hybrid distance computation by using two complementary candidate generators. Dense HGraph proposes semantic neighbors, while SINDI proposes lexical neighbors. Their union forms a candidate set that is much smaller than the whole dataset but more expressive than either modality alone.

### 4.3 All-alpha Hybrid Neighbor Selection

This subsection addresses Challenge 2.

Implementation anchors:

- `union_search_res::GetHybridDis(alpha)` computes the weighted hybrid score.
- Lines 611-631 sweep `alpha` from 0 to 1 with step 0.01.
- Lines 616-618 sort candidates by hybrid score and insert top-k into the all-alpha neighbor set.

Core formula:

```text
s_h(u, v; alpha) = alpha * s_dense(u, v) + (1 - alpha) * s_sparse(u, v)
```

The paper should emphasize that this is not a single-alpha graph. The construction aggregates neighbors that become top-k under different dense-sparse trade-offs, producing a graph that covers a family of hybrid similarity functions.

Possible terminology:

- all-alpha hybrid graph;
- alpha-cover neighbor set;
- hybrid Pareto-style neighborhood;
- multi-weight hybrid graph construction.

Recommended name: **Alpha-Cover Hybrid Graph**.

### 4.4 Graph Refinement: Reverse Edges, RNG Pruning, and Connectivity Repair

This subsection addresses Challenge 3.

Implementation anchors:

- `PruneNeighborsByRNG()` lines 246-295.
- `AddReverseEdgesWithPrune()` lines 300-338.
- `EnsureConnectivity()` lines 341-443.
- Lines 678-693 apply reverse-edge pruning and DFS connectivity repair.

The construction initially produces directed neighbor sets. To make the graph suitable for ANN traversal, the method refines it through three steps:

1. **Reverse-edge augmentation:** if `u` selects `v`, then `u` becomes a candidate neighbor of `v`.
2. **RNG-style pruning:** merged forward and reverse candidates are pruned to remove dominated neighbors and control degree.
3. **Connectivity repair:** disconnected nodes are linked back to the reachable component using nearest connected nodes under hybrid distance.

Key message:

> The graph is not just a bag of hybrid top-k neighbors. It is refined into a bounded-degree, traversable ANN graph.

### 4.5 HybridIndex Integration

Implementation anchors:

- `HybridIndex::add_one_point()` in `src/algorithm/hybrid_index/hybrid_index.cpp` reads `601_output_neighbors_k32.h5`.
- It loads `neighbors` and `neighbor_counts` datasets.
- It inserts the precomputed adjacency list through `graph_->InsertNeighborsById()`.

This subsection should explain the division between the experimental construction pipeline and the index implementation:

- `601_hybrid_exp1.cpp` constructs and exports the graph topology.
- `HybridIndex` loads this topology, stores dense+sparse vectors through `HybridVectorDataCell`, and performs hybrid graph search.

### 4.6 Search over the Hybrid Graph

Search should be presented as a consequence of the index, not the main contribution.

Basic search:

- query contains dense and sparse components;
- `HybridIndex` computes weighted hybrid scores;
- graph traversal returns top-k hybrid nearest neighbors.

Extension experiment:

- `605_hybrid_exp5.cpp` uses SINDI or HGraph results as entry points;
- this can be framed as an optional query-time routing enhancement over the constructed hybrid graph.

## 5. Suggested Paper Section Skeleton

### Section 1: Introduction

Recommended flow:

1. Dense and sparse retrieval are complementary.
2. Hybrid retrieval is important, but efficient ANN under hybrid similarity is hard.
3. Existing two-route systems search dense and sparse indexes independently, then merge and rerank.
4. The deeper limitation is that two-route systems do not build a graph topology aligned with hybrid similarity.
5. A hybrid graph should contain neighbors that are useful across dense-dominant, sparse-dominant, and balanced hybrid weights.
6. We propose a hybrid graph index construction method that uses dense/sparse candidate generation, all-alpha neighbor selection, and graph refinement.
7. Summarize contributions.

### Section 2: Background and Problem Definition

Subsections:

- Dense vector retrieval.
- Sparse vector retrieval.
- Hybrid score definition.
- Graph-based ANNS.
- Problem: construct a bounded-degree graph that supports efficient top-k retrieval under hybrid similarity.

Problem formulation:

```text
Given object x_i = (d_i, s_i), construct graph G = (V, E) such that greedy/beam search over G
can retrieve top-k neighbors under s_h(q, x; alpha) efficiently for hybrid queries q.
```

### Section 3: Motivation

Subsections:

1. **Why two-route retrieval is insufficient.** It retrieves candidates but does not create hybrid graph structure.
2. **Why single-modality graphs are insufficient.** Dense graph misses sparse-only lexical neighbors; sparse graph misses semantic neighbors.
3. **Why single-alpha graph construction is brittle.** The best edge set changes as `alpha` changes.
4. **Observation from construction experiment.** Neighbor sets collected over alpha form a richer hybrid topology than dense-only or sparse-only neighbors.

Suggested figures:

- Figure 1: two-route retrieval vs hybrid graph index.
- Figure 2: alpha changes the nearest-neighbor ordering.
- Figure 3: all-alpha neighbor selection creates alpha-cover edges.

### Section 4: Hybrid Graph Index Construction

Subsections:

1. Construction overview.
2. Dense-sparse candidate generation.
3. All-alpha hybrid neighbor selection.
4. Reverse-edge augmentation.
5. RNG-style pruning.
6. Connectivity repair.
7. HybridIndex integration and search.

Suggested algorithm blocks:

#### Algorithm 1: Hybrid Graph Construction

```text
Input: dense vectors D, sparse vectors S, top-k k, candidate budget ak, alpha grid A
Output: hybrid graph G

1. Build dense HGraph over D
2. Build sparse SINDI over S
3. for each point x_i:
4.     C_d <- dense HGraph top-ak neighbors of x_i
5.     C_s <- SINDI top-ak neighbors of x_i
6.     C <- C_d union C_s
7.     for each candidate c in C:
8.         compute dense score s_d(i, c)
9.         compute sparse score s_s(i, c)
10.    N_i <- empty set
11.    for alpha in A:
12.        rank C by alpha * s_d + (1-alpha) * s_s
13.        insert top-k candidates into N_i
14.    E_i <- N_i
15. Add reverse-edge candidates
16. Apply RNG-style pruning with max degree M
17. Repair disconnected components
18. return G
```

#### Algorithm 2: RNG-style Hybrid Pruning

```text
Input: point x_i, candidate set C, max degree M, evaluation alpha alpha_eval
Output: pruned neighbor list N_i

1. Sort C by hybrid distance to x_i
2. N_i <- empty
3. for candidate c in sorted C:
4.     if |N_i| = M: break
5.     dominated <- false
6.     for selected neighbor n in N_i:
7.         if dist(n, c) < alpha_rng * dist(i, c):
8.             dominated <- true
9.             break
10.    if not dominated:
11.        insert c into N_i
12. return N_i
```

### Section 5: Experiments

Subsections:

#### 5.1 Setup

Report:

- datasets;
- dense dimension;
- sparse dimension and average nnz;
- number of base/query vectors;
- hardware;
- `k`, `ak`, alpha grid, max degree, `ef_search`;
- construction output file and graph degree statistics.

#### 5.2 Baselines

Recommended baselines:

1. Dense-only HGraph.
2. Sparse-only SINDI or sparse HGraph.
3. Two-route `hgraph + hgraph` with merge/rerank.
4. Two-route `hgraph + sindi` with merge/rerank.
5. Single-alpha hybrid graph, if available.
6. Proposed all-alpha hybrid graph index.
7. Proposed hybrid graph with guided entry search, as an extension.

#### 5.3 Main Results

Primary comparison:

- proposed hybrid graph vs `hgraph + hgraph`;
- proposed hybrid graph + SINDI entry vs `hgraph + sindi`.

Metrics:

- Recall@k;
- QPS;
- average graph hops;
- average distance computations;
- index build time;
- graph degree / memory.

#### 5.4 Construction Analysis

A paper centered on graph construction needs construction-specific analysis:

- average all-alpha neighbor count before pruning;
- max neighbor count before pruning;
- degree distribution after pruning;
- connectivity before and after repair;
- coverage of dense/sparse candidate budgets;
- effect of alpha grid step size;
- effect of `k` and `ak`.

#### 5.5 Ablation Study

Recommended ablations:

1. Dense candidates only.
2. Sparse candidates only.
3. Dense+sparse union without alpha sweep.
4. Alpha sweep without RNG pruning.
5. Alpha sweep with RNG pruning.
6. With vs without reverse-edge augmentation.
7. With vs without connectivity repair.
8. Basic hybrid graph search vs guided-entry hybrid graph search.

### Section 6: Related Work

Organize as:

1. Dense graph-based ANNS: HNSW, NSG, DiskANN, HGraph.
2. Sparse retrieval: inverted indexes, WAND-style algorithms, SINDI.
3. Dense-sparse hybrid retrieval: score fusion, RRF, two-route retrieval.
4. Hybrid graph / multi-vector graph indexes: methods such as DEG and dense-sparse graph-based hybrid retrieval.
5. Graph pruning and construction heuristics: RNG pruning, reverse edges, connectivity repair.

### Section 7: Conclusion

Key conclusion:

> Hybrid retrieval should not only combine dense and sparse scores at reranking time; it should also construct an index topology that reflects hybrid similarity. The proposed hybrid graph index builds such a topology through dense-sparse candidate generation, all-alpha neighbor selection, and graph refinement, leading to better retrieval trade-offs than two-route baselines.

## 6. Self-consistency Checks

- **Check 1 Limitations -> Key Idea:** Pass. The limitations all concern the lack of a hybrid-aware graph topology; the key idea directly constructs such a topology.
- **Check 2 Key Idea -> Challenges:** Pass. Constructing an all-alpha hybrid graph naturally creates candidate-generation, alpha-coverage, and graph-quality challenges.
- **Check 3 Challenges -> Methodology:** Pass. Candidate generation, all-alpha selection, and graph refinement map one-to-one to the three challenges.
- **Check 4 Methodology -> Contributions:** Pass. Contributions cover the motivation, construction algorithm, graph refinement, and experimental validation.

## 7. Severity Summary

- **0 CRITICAL, 2 MAJOR, 1 MINOR.**

Top fixes first:

1. **MAJOR:** Move the construction method from experimental example code into a clean method description. The paper should not describe implementation artifacts such as hard-coded output paths as if they were algorithmic requirements.
2. **MAJOR:** Decide the final naming of the method. Recommended names: **Alpha-Cover Hybrid Graph**, **All-Alpha Hybrid Graph**, or **HybridGraphIndex**.
3. **MINOR:** Standardize notation around score vs distance. The construction code uses IP scores where larger is better, while pruning uses negative hybrid score as a distance where smaller is better.

## 8. Next Suggested Skill

Use `intro-drafter` next to turn this corrected construction-centered skeleton into a six-paragraph Introduction outline.
