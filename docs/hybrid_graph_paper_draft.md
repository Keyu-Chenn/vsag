# Alpha-Cover Hybrid Graph Index for Dense-Sparse Retrieval

## Abstract

Dense-sparse hybrid retrieval is widely used in modern search and retrieval-augmented generation systems because dense embeddings and sparse lexical vectors provide complementary evidence. Existing hybrid retrieval systems typically follow a two-route design: they search a dense index and a sparse index independently, merge the returned candidates, and rerank them using a weighted hybrid score. This design is simple, but it does not construct an index topology that is aligned with the final hybrid similarity; consequently, candidates that are only moderately strong in each individual modality but strong under the combined score may be missed or underrepresented. We propose an Alpha-Cover Hybrid Graph, a graph index construction method that builds dense-sparse candidate pools, selects neighbors across a range of hybrid weights, and refines the resulting graph with reverse-edge augmentation, relative-neighborhood-style pruning, and connectivity repair. The resulting index supports direct hybrid graph traversal under dense-sparse similarity and provides a foundation for query-time guided entry search. Experiments compare the proposed index with two-route `HGraph + HGraph` and `HGraph + SINDI` baselines using recall, throughput, graph hops, and distance-computation metrics.

## 1. Introduction

Dense and sparse retrieval models capture different forms of relevance. Dense embeddings encode semantic similarity and can retrieve conceptually related objects even when the query and the target use different surface forms. Sparse vectors, in contrast, preserve lexical and term-level evidence, which is important for exact matching, entity names, rare terms, and other cases where semantic embeddings may over-generalize. Production retrieval systems therefore increasingly combine the two representations in web search, recommendation, question answering, and retrieval-augmented generation. The central technical question is not whether dense and sparse signals should be combined, but how to search efficiently under the combined similarity.

The common solution is two-route hybrid retrieval. A dense approximate nearest-neighbor index retrieves candidates according to dense similarity, while a sparse index retrieves candidates according to lexical similarity. The two candidate sets are then merged and reranked by a hybrid score such as

```text
s_h(q, x; alpha) = alpha * s_d(q_d, x_d) + (1 - alpha) * s_s(q_s, x_s),
```

where `alpha` controls the dense-sparse trade-off. This pipeline is easy to implement because it reuses mature dense and sparse indexes. However, it treats hybrid retrieval as a post-processing problem: the index structures themselves remain single-modality structures, and the hybrid score only appears after candidate generation.

This separation creates a structural limitation. A strong hybrid neighbor does not have to be among the top dense-only candidates or the top sparse-only candidates. It may be moderately close in both spaces and become highly ranked only after the two signals are combined. If such a point is absent from both individual candidate sets, reranking cannot recover it. Increasing the dense and sparse candidate budgets can reduce this failure mode, but it also increases latency, duplicated search work, and parameter-tuning complexity. More importantly, larger candidate pools still do not create a graph topology whose edges reflect hybrid similarity.

A natural alternative is to build a unified graph over dense-sparse objects and traverse it directly with the hybrid score. Yet a naive unified graph is not sufficient. If the graph is constructed only from dense neighbors, sparse-only evidence may be unreachable. If it is constructed only from sparse neighbors, semantic neighborhoods may be fragmented. If it is constructed for one fixed `alpha`, the graph may overfit to one dense-sparse trade-off and become brittle when applications adjust the weight. A practical hybrid graph must therefore preserve useful neighborhoods across sparse-dominant, balanced, and dense-dominant scoring regimes, while still maintaining bounded degree and connectivity for efficient graph search.

We propose an Alpha-Cover Hybrid Graph index construction method. During construction, auxiliary dense and sparse indexes are used only as candidate generators. For each data point, the method unions dense and sparse candidate pools, recomputes exact dense and sparse scores within this pool, and sweeps a grid of `alpha` values from sparse-dominant to dense-dominant settings. Candidates that become top-ranked under any weight are retained as alpha-cover neighbors. The initial directed neighbor sets are then refined through reverse-edge augmentation, relative-neighborhood-style pruning, and connectivity repair, producing a bounded-degree graph that can be consumed by a hybrid graph index.

This paper makes four contributions. First, we identify a topology mismatch in two-route dense-sparse retrieval: the final score is hybrid, but the candidate-generation structures are not. Second, we propose alpha-cover neighbor selection, which constructs graph edges that cover a family of dense-sparse hybrid weights rather than a single fixed weight. Third, we refine the constructed graph with reverse edges, pruning, and connectivity repair so that it is suitable for approximate nearest-neighbor traversal. Fourth, we evaluate the resulting hybrid graph index against two-route baselines and study guided entry search as a query-time extension over the constructed graph.

## 2. Background and Problem Definition

### 2.1 Dense, Sparse, and Hybrid Similarity

Each database object is represented by a pair `x = (x_d, x_s)`, where `x_d` is a dense vector and `x_s` is a sparse vector. Likewise, a query is represented as `q = (q_d, q_s)`. Dense similarity `s_d(q_d, x_d)` measures semantic proximity, while sparse similarity `s_s(q_s, x_s)` measures lexical or term-level overlap. In this draft, we use an inner-product score convention where larger values indicate better matches.

The hybrid score is defined as a weighted sum:

```text
s_h(q, x; alpha) = alpha * s_d(q_d, x_d) + (1 - alpha) * s_s(q_s, x_s), alpha in [0, 1].
```

When `alpha = 1`, retrieval is dense-only. When `alpha = 0`, retrieval is sparse-only. Intermediate values define different dense-sparse trade-offs. In practical systems, this weight may be chosen per workload, per dataset, or even per query family. An index that only works well for one fixed value can therefore be fragile.

### 2.2 Graph-based Approximate Nearest-Neighbor Search

Graph-based approximate nearest-neighbor search builds a graph whose nodes are database objects and whose edges connect nearby objects. At query time, the search starts from one or more entry points, explores neighboring nodes, and maintains a candidate queue ordered by the query-to-node score or distance. Parameters such as graph degree and `ef_search` control the trade-off between search quality and cost.

Graph search is effective when the graph topology is aligned with the target similarity. If neighbors in the graph are also likely to be neighbors under the query scoring function, greedy or beam-style traversal can quickly move toward high-quality candidates. Conversely, if the graph was built under a different similarity, search may spend many hops in irrelevant regions or fail to reach the true neighborhood.

### 2.3 Two-route Hybrid Retrieval Baseline

The two-route baseline builds separate dense and sparse indexes. At query time, it performs the following steps:

1. search the dense index for `b_d` candidates;
2. search the sparse index for `b_s` candidates;
3. union the two candidate sets;
4. recompute dense and sparse scores for each candidate;
5. rank candidates by `s_h(q, x; alpha)` and return the top results.

This baseline is strong and modular, but its candidate-generation stage is not optimized for the hybrid score. The dense route only knows `s_d`; the sparse route only knows `s_s`; the hybrid score is applied after both searches have already discarded most objects. This motivates constructing a graph whose edges are selected with hybrid similarity in mind.

### 2.4 Problem Statement

Given a set of dense-sparse objects `X = {x_i = (x_{i,d}, x_{i,s})}`, our goal is to construct a bounded-degree graph `G = (V, E)` over the objects such that graph traversal can efficiently retrieve top-`k` objects under `s_h(q, x; alpha)` for hybrid queries. The graph should satisfy three requirements:

- **Hybrid awareness:** edges should reflect dense-sparse hybrid similarity rather than only one modality.
- **Alpha coverage:** the graph should preserve useful neighbors across a range of `alpha` values.
- **Traversability:** the graph should have controlled degree and sufficient connectivity for efficient approximate search.

## 3. Motivation

### 3.1 Candidate Loss in Two-route Retrieval

Two-route retrieval assumes that a good hybrid candidate will be retrieved by at least one individual modality. This assumption is not always true. Consider a candidate that is not in the top dense candidates because several objects are semantically closer, and not in the top sparse candidates because several objects share more exact terms. If the candidate is consistently strong across both modalities, its weighted hybrid score may still be high. Such a candidate can be important for hybrid retrieval precisely because it balances semantic and lexical evidence, but it may never enter the reranking set.

This failure is not just a matter of final ranking. It affects what structure an index can exploit. Two-route systems do not create edges between objects that are close under hybrid similarity. They only provide a temporary query-time candidate set. As a result, they cannot use hybrid neighborhoods to guide traversal from one promising object to another.

### 3.2 Single-modality Graphs Do Not Encode Hybrid Neighborhoods

A dense graph connects objects according to dense similarity. It is effective for semantic search, but it may omit lexical neighbors that matter when sparse evidence is important. A sparse graph has the opposite weakness: it can preserve exact-match structure but may fragment semantically related objects. Searching either graph and reranking afterward still relies on a graph topology that was built for a different objective.

The desired structure is a graph where edges can represent dense-dominant, sparse-dominant, and balanced hybrid neighborhoods. Such a graph gives search a chance to move through the same similarity landscape used by the final ranking function.

### 3.3 Single-alpha Construction Is Brittle

One might build a graph using a fixed hybrid weight, such as `alpha = 0.5`. This improves over purely single-modality construction because the selected edges use both signals. However, the nearest-neighbor ordering can change as `alpha` changes. A neighbor that is essential for sparse-heavy retrieval may be absent from a balanced graph; a semantic neighbor that matters when dense evidence dominates may be pruned away when construction uses a sparse-heavy weight.

This observation motivates alpha coverage. Instead of selecting neighbors for only one weight, construction should collect candidates that become important under any weight in a predefined alpha grid. The resulting graph is not tied to one application-specific setting and can support multiple dense-sparse trade-offs.

## 4. Alpha-Cover Hybrid Graph Construction

### 4.1 Overview

The proposed construction method turns dense and sparse retrievers into construction-time candidate generators. They are not the final two-route retrieval system. Their role is to produce a compact but expressive candidate pool for each data point. The method then evaluates these candidates using exact dense and sparse scores, selects alpha-cover neighbors, and refines the graph for traversal.

The construction pipeline is:

1. build an auxiliary dense graph index over dense vectors;
2. build an auxiliary sparse index over sparse vectors;
3. for each data point, retrieve dense and sparse candidate pools;
4. union the pools and recompute dense and sparse scores for each candidate;
5. sweep an alpha grid and retain candidates that appear in the top-`k` under any alpha;
6. add reverse-edge candidates;
7. prune neighbors to control degree and remove dominated edges;
8. repair disconnected components;
9. export the adjacency lists for hybrid graph search.

The key distinction from two-route retrieval is that dense and sparse indexes are used offline to construct a hybrid topology, rather than online as independent retrieval routes whose results are merely merged.

### 4.2 Dense-Sparse Candidate Generation

Full all-pairs hybrid graph construction is prohibitively expensive for large datasets. For every point, comparing against all other points under many alpha values would require both dense and sparse score computation at dataset scale. We avoid this cost by using two complementary candidate generators.

For a point `x_i`, the dense candidate generator returns a set `C_d(i)` of objects that are close to `x_i` in dense space. The sparse candidate generator returns a set `C_s(i)` of objects that are close in sparse space. The candidate pool is their union:

```text
C(i) = C_d(i) union C_s(i).
```

The union is typically much smaller than the full dataset, but it contains candidates proposed by both semantic and lexical evidence. After the union is formed, the method recomputes exact dense and sparse scores for every candidate in `C(i)`. This step is important because the auxiliary indexes may use approximation or pruning; graph construction should rank the candidate pool using consistent exact scores.

### 4.3 Alpha-Cover Neighbor Selection

Given the candidate pool `C(i)`, the method constructs a neighbor set that covers a family of hybrid weights. Let `A = {0, delta, 2delta, ..., 1}` be an alpha grid. For each `alpha in A`, candidates are ranked by

```text
s_h(x_i, x_j; alpha) = alpha * s_d(x_i, x_j) + (1 - alpha) * s_s(x_i, x_j).
```

The top-`k` candidates for each alpha are inserted into an unordered alpha-cover set:

```text
N_i = union_{alpha in A} TopK(C(i), s_h(x_i, ·; alpha)).
```

This set keeps candidates that are important for sparse-dominant, balanced, or dense-dominant scoring. It can be larger than `k`, because different alpha values may select different neighbors. The purpose is not to fix the final graph degree at this stage, but to avoid prematurely discarding candidates that are useful under some hybrid trade-off.

Alpha-cover selection can be interpreted as a practical approximation to a multi-weight hybrid neighborhood. Rather than solving for all possible breakpoints where the hybrid ranking changes continuously with alpha, the method samples a grid and retains candidates that become top-ranked at sampled weights. The grid step controls a construction-time trade-off: smaller steps provide finer coverage but increase sorting and candidate-retention cost.

### 4.4 Graph Refinement

The initial alpha-cover neighbor sets are directed and may have uneven degree. Some nodes may be selected by many other nodes, while others may have too few incoming edges. Directly using these sets as a search graph can lead to poor traversability. We therefore refine the graph in three steps.

First, reverse-edge augmentation adds reciprocal candidates. If `x_i` selects `x_j` as an alpha-cover neighbor, then `x_i` is also considered as a candidate neighbor for `x_j`. This improves reachability because graph search needs paths that can move through the topology in both directions.

Second, relative-neighborhood-style pruning controls degree. Candidate neighbors are considered in order of hybrid distance or score. A candidate can be removed if an already selected neighbor provides a shorter route to it, making the direct edge less useful for navigation. This pruning follows the intuition used by graph-based ANN methods: retain diverse navigational neighbors rather than many redundant close points from the same local region.

Third, connectivity repair links disconnected components back to the main reachable component. A graph that has good local neighbors but disconnected components is not usable for global ANN traversal. Connectivity repair identifies unreachable nodes or components and adds edges to nearby reachable nodes under the hybrid evaluation score. This step turns the alpha-cover neighbor collection into a connected, bounded-degree search graph.

### 4.5 HybridIndex Integration and Search

After construction, the graph is exported as adjacency lists and loaded by the hybrid index. The index stores dense and sparse vector components together and configures the hybrid weight used during search. At query time, graph traversal evaluates candidates using the weighted hybrid score rather than a single-modality score.

This integration separates two concerns. The construction pipeline decides which graph edges should exist. The search index stores vectors, loads the graph topology, and performs query-time traversal. Guided entry search can then be added as an extension: dense or sparse retrievers provide query-aware entry points, and the hybrid graph continues traversal under the hybrid score. In the paper, this guided search should be presented as an enhancement over the constructed graph, not as a replacement for hybrid graph construction.

## 5. Experimental Plan

### 5.1 Research Questions

The experiments should answer four questions.

1. Does the alpha-cover hybrid graph improve the recall-efficiency trade-off over two-route baselines?
2. How much do graph refinement steps contribute to search quality and efficiency?
3. How sensitive is the method to construction parameters such as candidate budget, alpha-grid step, and maximum degree?
4. Does guided entry search further improve traversal over the constructed hybrid graph?

### 5.2 Baselines

The main baselines are:

- dense-only HGraph;
- sparse-only SINDI or sparse HGraph;
- two-route `HGraph + HGraph` with merge and reranking;
- two-route `HGraph + SINDI` with merge and reranking;
- single-entry hybrid graph search;
- alpha-cover hybrid graph search;
- alpha-cover hybrid graph with dense- or sparse-guided entry points.

### 5.3 Metrics

The evaluation should report Recall@`k`, QPS, total query time, average distance computations, average graph hops, construction time, graph degree statistics, and memory footprint. For a construction-centered paper, graph statistics are especially important: the paper should report average alpha-cover neighbor count before pruning, degree distribution after pruning, and connectivity before and after repair.

### 5.4 Ablation Studies

The ablation study should isolate the effect of each design choice:

- dense candidates only versus sparse candidates only versus dense-sparse union;
- single-alpha selection versus alpha-cover selection;
- with and without reverse-edge augmentation;
- with and without pruning;
- with and without connectivity repair;
- single-entry search versus guided-entry search;
- different alpha-grid steps, candidate budgets, and maximum degrees.

## 6. Related Work Placeholder

This section should be organized around five lines of work: dense graph-based approximate nearest-neighbor search, sparse retrieval and inverted-index pruning, dense-sparse hybrid retrieval, graph-based hybrid retrieval, and graph construction/pruning heuristics. The final draft should compare against HNSW, NSG, DiskANN, HGraph, SINDI, two-route score-fusion systems, DEG-style hybrid graph methods, ACORN-style traversal modifications, and relative-neighborhood pruning techniques.

## 7. Conclusion Placeholder

Dense-sparse hybrid retrieval should not only combine scores at reranking time. It should also construct an index topology aligned with the hybrid similarity used by the final ranking function. The Alpha-Cover Hybrid Graph follows this principle by generating dense-sparse candidate pools, selecting neighbors across hybrid weights, and refining the resulting topology into a traversable graph. This construction enables direct hybrid graph search and provides a foundation for guided entry routing.
