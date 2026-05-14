# Guided Hybrid Graph Search for Dense-Sparse Retrieval: Paper Outline

## 1. Paper Positioning

**Paper type:** Technique paper.

This work is best positioned as a technique paper rather than a new-problem paper. The task,
dense-sparse hybrid retrieval, is already well motivated by search and RAG workloads. The main
contribution is a better retrieval method: replacing the conventional two-route retrieval pipeline
with a unified hybrid graph and modality-aware entry-point guided graph traversal.

## 2. Core Thesis

Existing hybrid retrieval systems usually search dense and sparse indexes separately and merge
candidates afterward. This two-route design is simple but loses promising hybrid neighbors,
duplicates search cost, and cannot exploit graph traversal under the true hybrid similarity. We
propose a unified hybrid graph index with sparse/dense-aware entry routing, enabling graph search
directly under hybrid similarity and improving the recall-efficiency trade-off over `hgraph + hgraph`
and `hgraph + sindi` baselines.

In this repository, the baseline is represented by `examples/cpp/603_hybrid_exp3.cpp`, while the
new hybrid graph experiment is represented by `examples/cpp/605_hybrid_exp5.cpp`.

## 3. Thinking Template

| Stage | Content |
|---|---|
| Research background | Dense retrieval captures semantic similarity, sparse retrieval captures lexical matching. Hybrid retrieval improves robustness and accuracy in RAG/search, but efficient hybrid ANN remains difficult because dense and sparse vectors have very different dimensions, distributions, and computation costs. |
| Limitation 1 | Two-route hybrid retrieval, e.g. dense HGraph + sparse SINDI/HGraph, retrieves candidates separately and merges them later. This may miss objects that are not top-ranked in either individual modality but are strong under the combined hybrid score. |
| Limitation 2 | Separate dense/sparse indexes duplicate search work and introduce candidate-set tuning complexity: `bk_dense`, `bk_sparse`, `ef_search`, `alpha`, and reranking budget interact non-trivially. |
| Limitation 3 | A naive unified hybrid graph can search under hybrid distance, but single/default entry-point search may be inefficient or unstable because the entry point is not adapted to the query's sparse/dense evidence. |
| Key idea / goal | Build a unified hybrid graph for dense+sparse objects and use modality-aware entry points, especially SINDI or HGraph results, to guide hybrid graph traversal under the true weighted hybrid similarity. |
| Challenge 1 | How to represent dense+sparse objects in one graph while preserving the weighted hybrid score used by retrieval. |
| Challenge 2 | How to choose effective graph entry points so hybrid search starts near promising regions rather than relying on a single generic entry. |
| Challenge 3 | How to show the benefit is not merely from adding SINDI, but from the hybrid graph itself. |
| Methodology topic sentence | We design a hybrid graph retrieval framework that combines unified hybrid distance computation, external modality-aware entry routing, and controlled comparisons against two-route baselines. |
| Module A | Unified Hybrid Graph Index: use `HybridIndex` and `HybridVectorDataCell` to store dense and sparse vectors and compute `alpha * dense_score + (1 - alpha) * sparse_score` during graph search. |
| Module B | Entry-point Guided Hybrid Traversal: use SINDI or HGraph top candidates as multiple `entry_points` for the hybrid graph, replacing single-entry search with query-aware graph initialization. |
| Module C | Pruning and efficiency controls: expose `ef_search`, `sindi_bk`, `hybrid_prune_scale`, SINDI query/term pruning, and report `dist_cmp`, `hops`, QPS, and recall. |
| Contribution 1 | Identify the candidate-loss and duplicated-computation problem in two-route dense+sparse hybrid retrieval. |
| Contribution 2 | Propose a unified hybrid graph index with weighted dense+sparse distance computation and multi-entry guided graph search. |
| Contribution 3 | Experimentally show that hybrid graph beats `hgraph + hgraph`, and `sindi + hybrid graph` beats `hgraph + sindi`, under recall/QPS/hops/distance-computation metrics. |

## 4. Candidate Titles

1. **Guided Hybrid Graph Search for Efficient Dense-Sparse Retrieval**
2. **A Unified Hybrid Graph Index for Dense-Sparse Approximate Nearest Neighbor Search**
3. **From Two-Route Retrieval to Guided Hybrid Graph Search**
4. **Entry-Guided Hybrid Graph Search for Dense-Sparse Vector Retrieval**

The first title is the recommended one because it clearly highlights the two central ideas:
guided entry routing and hybrid graph search.

## 5. Abstract Outline

The abstract can follow this four-sentence structure:

1. Dense-sparse hybrid retrieval is important for modern search and RAG because dense and sparse
   representations capture complementary semantic and lexical signals.
2. Existing two-route methods search dense and sparse indexes separately, then merge and rerank
   candidates; this design is inefficient and can miss true hybrid nearest neighbors.
3. We propose a unified hybrid graph index with modality-aware multi-entry routing, enabling graph
   traversal directly under hybrid similarity.
4. Experiments show that hybrid graph outperforms `hgraph + hgraph`, and `sindi + hybrid graph`
   outperforms `hgraph + sindi` in recall-efficiency trade-offs.

## 6. Section-by-Section Outline

### 1. Introduction

Recommended six-paragraph flow:

1. **Background.** Dense and sparse retrieval are complementary. Dense retrieval captures semantic
   similarity; sparse retrieval captures exact lexical matching. Hybrid retrieval is increasingly
   important in RAG, web search, recommendation, and knowledge retrieval.
2. **Existing practice.** The common solution is two-route retrieval: build one dense index and one
   sparse index, retrieve candidates independently, merge the two candidate sets, and rerank by a
   weighted hybrid score. This corresponds to `603_hybrid_exp3.cpp`.
3. **Limitation: candidate loss.** A true hybrid top-k neighbor may not be top-ranked in either
   dense-only or sparse-only retrieval, so it may never enter the reranking set.
4. **Limitation: duplicated work and parameter complexity.** Two-route retrieval performs two
   searches, requires candidate budgets for both routes, and must recompute hybrid scores for the
   merged candidate set.
5. **Our approach.** We build a unified hybrid graph where traversal uses the weighted hybrid score.
   We further use dense or sparse retrieval results as query-aware entry points, allowing the graph
   search to start from promising regions.
6. **Contributions.** Summarize the three contributions: problem observation, guided hybrid graph
   method, and experimental validation.

### 2. Background and Problem Definition

#### 2.1 Dense, Sparse, and Hybrid Vector Retrieval

Define dense vector `q_d`, sparse vector `q_s`, and database object `x = (x_d, x_s)`.

For an inner-product score formulation:

```text
s_h(q, x) = alpha * s_d(q_d, x_d) + (1 - alpha) * s_s(q_s, x_s)
```

If the implementation uses distance internally, keep one consistent direction throughout the paper
and explain whether larger or smaller values are better.

#### 2.2 Graph-based Approximate Nearest Neighbor Search

Introduce HNSW/HGraph-style graph search:

- nodes represent database vectors;
- edges connect approximate neighbors;
- search starts from one or more entry points;
- `ef_search` controls the candidate queue size;
- graph hops and distance computations determine efficiency.

#### 2.3 Baseline: Two-route Hybrid Retrieval

Describe the baseline pipeline from `603_hybrid_exp3.cpp`:

1. Build dense HGraph.
2. Build sparse SINDI or sparse HGraph.
3. Search both indexes independently.
4. Union candidate IDs.
5. Recompute dense and sparse scores for all candidates.
6. Rerank by the weighted hybrid score.

Recommended figure: **Figure 1. Two-route hybrid retrieval pipeline.**

### 3. Motivation

#### Observation 1: Separate Top-k Candidate Sets Are Not Enough

A point can be moderately good in both dense and sparse spaces but not appear in either individual
top-b candidate set. Such a point may still have a high hybrid score. This causes candidate loss in
two-route retrieval.

#### Observation 2: Hybrid Graph Traversal Directly Optimizes the Target Similarity

Instead of retrieving by single-modality scores and reranking afterward, a unified hybrid graph uses
the hybrid score during traversal. This aligns the search process with the final ranking objective.

#### Observation 3: Entry-point Quality Strongly Affects Hybrid Graph Search

Single-entry hybrid graph search may start far from the query-specific relevant region. SINDI or
HGraph can provide sparse- or dense-aware starting nodes, which are then used as multiple entry
points for hybrid graph traversal.

Recommended figure: **Figure 2. Candidate loss and guided hybrid graph intuition.**

### 4. Method: Guided Hybrid Graph Search

#### 4.1 Unified Hybrid Graph Representation

The unified index stores dense and sparse components together. In the current implementation:

- `HybridIndex` is implemented in `src/algorithm/hybrid_index/hybrid_index.cpp`.
- `HybridVectorDataCell` stores dense and sparse vector components.
- `SetHybridWeight(alpha, 1 - alpha)` configures the weighted hybrid score.
- Search is performed over a graph while computing hybrid distances/scores.

This differs from two-route retrieval because hybrid scoring is part of graph traversal, not merely a
post-processing reranking step.

#### 4.2 Modality-aware Entry-point Routing

The method supports two guided variants:

1. **Dense-guided Hybrid Graph:** HGraph retrieves dense candidates, then uses them as entry points
   for hybrid graph search.
2. **Sparse-guided Hybrid Graph:** SINDI retrieves sparse candidates, then uses them as entry points
   for hybrid graph search.

In `605_hybrid_exp5.cpp`, SINDI results are converted into `entry_points` and passed to
`hybrid_index->KnnSearch()`.

This transforms SINDI/HGraph from an independent retrieval route into a routing module for the
unified hybrid graph.

#### 4.3 Search Algorithm

Pseudo-code:

```text
Input: query q = (q_d, q_s), top-k, alpha, ef_search, entry budget b
Output: top-k objects ranked by hybrid score

1. E <- EntryRetriever(q_d or q_s, b)
2. Initialize graph search candidate queue with E
3. while candidate queue is not exhausted:
4.     pop the most promising candidate
5.     visit its graph neighbors
6.     compute hybrid score using dense and sparse components
7.     update candidate queue and top-k result queue
8. return top-k results
```

#### 4.4 Pruning and Efficiency Controls

The implementation exposes several knobs:

- `ef_search`: graph search breadth;
- `sindi_bk`: number of SINDI entry candidates;
- `hgraph_bk`: number of HGraph entry candidates;
- `hybrid_prune_scale`: hybrid sparse-table pruning control;
- `sindi_query_prune_ratio`: query-side SINDI pruning;
- `sindi_term_prune_ratio`: term-side SINDI pruning.

These should be presented as efficiency optimizations rather than the primary contribution.

### 5. Experiments

#### 5.1 Experimental Setup

Report:

- datasets and data format;
- number of base vectors and queries;
- dense dimension;
- sparse dimension and average non-zero elements;
- hardware;
- index parameters;
- query parameters.

The current H5 fields used in the experiments include:

- `train`;
- `train_sparse`;
- `train_labels`;
- `test`;
- `test_sparse`;
- `neighbors`.

#### 5.2 Metrics

Use:

- Recall@k;
- QPS;
- total search time;
- average distance computations, `dist_cmp`;
- average graph hops, `hops`;
- optionally index build time and memory.

#### 5.3 Baselines

Recommended baselines:

1. **HGraph + HGraph:** dense graph plus sparse/dense graph two-route merge.
2. **HGraph + SINDI:** dense HGraph plus sparse SINDI two-route merge.
3. **HybridGraph single-entry:** unified hybrid graph without guided entry points.
4. **HGraph -> HybridGraph:** dense-guided hybrid graph.
5. **SINDI -> HybridGraph:** sparse-guided hybrid graph.

#### 5.4 Main Results

Main claim:

> Guided hybrid graph consistently achieves a better recall-efficiency trade-off than two-route
> baselines. The improvement remains whether the entry signal comes from dense HGraph or sparse
> SINDI, showing that the gain comes from searching a unified hybrid graph rather than merely using
> a stronger sparse retriever.

Recommended plots/tables:

1. Recall vs QPS curve.
2. Recall vs `ef_search`.
3. QPS at fixed recall.
4. Average `dist_cmp` comparison.
5. Average `hops` comparison.

#### 5.5 Ablation Study

Recommended ablations:

1. Single-entry vs multi-entry hybrid graph.
2. SINDI entry vs HGraph entry.
3. Vary `sindi_bk` / `hgraph_bk`.
4. Vary `alpha`.
5. Vary `hybrid_prune_scale`.
6. With and without sparse vector truncation/pruning.

#### 5.6 Cost Analysis

Analyze:

- build time;
- memory;
- latency breakdown;
- dense/sparse distance computation count;
- graph hops.

### 6. Related Work

Organize related work into four groups:

1. **Dense vector ANNS:** HNSW, NSG, DiskANN, HGraph.
2. **Sparse retrieval:** inverted index, WAND-style algorithms, SINDI.
3. **Dense-sparse hybrid retrieval:** two-route retrieval, score fusion, RRF, OneSparse.
4. **Graph-based hybrid search:** ACORN, DEG, dense-sparse hybrid graph methods.

Useful writing references:

- **Efficient and Effective Retrieval of Dense-Sparse Hybrid Vectors using Graph-based Approximate
  Nearest Neighbor Search.** Most relevant narrative template: two-route retrieval is inefficient,
  unified graph is promising, sparse computation must be optimized.
- **DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation Graph.** Useful for
  writing about weighted hybrid similarity, alpha, and graph design.
- **ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data.**
  Useful for explaining why modifying graph traversal is better than post-filtering/post-reranking.
- **OneSparse.** Useful for motivating unified index design over isolated multi-index pipelines.

### 7. Conclusion

Conclude with three messages:

1. Two-route retrieval is not ideal for dense-sparse hybrid search.
2. Unified hybrid graph enables direct optimization under hybrid similarity.
3. Entry-guided traversal makes hybrid graph search practical and efficient.

## 7. Figures to Prepare

1. **Figure 1: Two-route baseline vs guided hybrid graph.**
   - Left: dense index + sparse index -> union -> rerank.
   - Right: SINDI/HGraph entry retriever -> multi-entry hybrid graph traversal.
2. **Figure 2: Candidate loss example.**
   - Show a point that is not top-ranked in either modality but ranks high by hybrid score.
3. **Figure 3: Hybrid graph search workflow.**
   - Entry retrieval, entry point initialization, hybrid graph traversal, top-k output.
4. **Figure 4: Recall-QPS curves.**
   - Compare all baselines and proposed variants.
5. **Figure 5: Ablation of entry budget and alpha.**

## 8. Experimental Table Templates

### Table 1. Dataset Statistics

| Dataset | #Base | #Query | Dense dim | Sparse dim | Avg. sparse nnz | Ground truth |
|---|---:|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | top-k hybrid brute force |

### Table 2. Main Results

| Method | Recall@k | QPS | Avg. dist_cmp | Avg. hops | Total time |
|---|---:|---:|---:|---:|---:|
| HGraph + HGraph | TBD | TBD | TBD | TBD | TBD |
| HGraph + SINDI | TBD | TBD | TBD | TBD | TBD |
| HybridGraph single-entry | TBD | TBD | TBD | TBD | TBD |
| HGraph -> HybridGraph | TBD | TBD | TBD | TBD | TBD |
| SINDI -> HybridGraph | TBD | TBD | TBD | TBD | TBD |

### Table 3. Ablation Study

| Variant | Recall@k | QPS | Avg. dist_cmp | Avg. hops | Observation |
|---|---:|---:|---:|---:|---|
| single entry | TBD | TBD | TBD | TBD | baseline unified graph |
| multi-entry, dense-guided | TBD | TBD | TBD | TBD | effect of dense routing |
| multi-entry, sparse-guided | TBD | TBD | TBD | TBD | effect of sparse routing |
| no pruning | TBD | TBD | TBD | TBD | pruning effect |
| tuned pruning | TBD | TBD | TBD | TBD | best efficiency setting |

## 9. Important Risk / Open Issue

The current `HybridIndex::add_one_point()` implementation appears to read precomputed neighbors
from a fixed H5 path:

```text
/tbase-project/vsag/build-release/examples/cpp/601_output_neighbors_k32.h5
```

If the paper claims a full index construction algorithm, this part must be formalized:

- How are these neighbors generated?
- Which distance is used to generate them?
- Is the construction offline or integrated into the index?
- What is the construction cost?
- Does the graph construction use hybrid distance, dense distance, sparse distance, or all-alpha
  neighbor candidates?

Until this is clarified, the safer paper claim is:

> We propose guided search over a constructed hybrid graph.

rather than:

> We propose a complete end-to-end hybrid graph construction algorithm.

## 10. Self-consistency Check

- **Limitations -> Key idea:** Pass. The two-route retrieval limitations naturally motivate unified
  hybrid graph search.
- **Key idea -> Challenges:** Pass. Unified scoring, entry selection, and controlled comparison are
  direct challenges of implementing the idea.
- **Challenges -> Methodology:** Pass. `HybridIndex`, multi-entry routing, and ablations map to the
  three challenges.
- **Methodology -> Contributions:** Pass. The proposed method and experimental claims are supported
  by the planned sections.

## 11. Next Writing Step

The recommended next step is to write a six-paragraph Introduction outline, then prepare the first
three figures:

1. two-route vs guided hybrid graph;
2. candidate loss motivation;
3. guided hybrid graph search workflow.
