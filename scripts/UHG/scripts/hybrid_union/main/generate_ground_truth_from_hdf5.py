#!/usr/bin/env python3
"""Generate exact hybrid ground-truth npy files from a VSAG hybrid HDF5 file."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact top-k ground truth for alpha*dense_ip + (1-alpha)*sparse_ip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", required=True, type=Path, help="Input <dataset>.hdf5 file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output ground-truth directory.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--topk", type=int, default=100, help="Number of exact neighbors per query.")
    parser.add_argument("--num-queries", type=int, default=-1, help="-1 means all test queries.")
    parser.add_argument("--query-chunk", type=int, default=32, help="Queries processed together.")
    parser.add_argument("--train-chunk", type=int, default=50000, help="Train rows processed together.")
    return parser.parse_args()


def dataset_name(hdf5: Path) -> str:
    return hdf5.stem


def parse_sparse_blob(blob: np.ndarray, count: int) -> sparse.csr_matrix:
    raw = memoryview(blob)
    offset = 0
    indptr = np.empty(count + 1, dtype=np.int64)
    indptr[0] = 0

    lengths = np.empty(count, dtype=np.uint32)
    nnz = 0
    for i in range(count):
        length = int(np.frombuffer(raw[offset : offset + 4], dtype=np.uint32, count=1)[0])
        lengths[i] = length
        offset += 4 + length * 8
        nnz += length
        indptr[i + 1] = nnz

    indices = np.empty(nnz, dtype=np.uint32)
    data = np.empty(nnz, dtype=np.float32)
    offset = 0
    cursor = 0
    for length in lengths:
        length_int = int(length)
        offset += 4
        if length_int:
            end = cursor + length_int
            byte_count = length_int * 4
            indices[cursor:end] = np.frombuffer(
                raw[offset : offset + byte_count], dtype=np.uint32, count=length_int
            )
            offset += byte_count
            data[cursor:end] = np.frombuffer(
                raw[offset : offset + byte_count], dtype=np.float32, count=length_int
            )
            offset += byte_count
            cursor = end

    dim = int(indices.max()) + 1 if nnz else 0
    return sparse.csr_matrix((data, indices, indptr), shape=(count, dim), dtype=np.float32)


def query_ranges(total: int, chunk: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


def update_topk(
    current_scores: np.ndarray,
    current_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k = current_scores.shape[1]
    take = min(k, scores.shape[1])
    part = np.argpartition(-scores, take - 1, axis=1)[:, :take]
    part_scores = np.take_along_axis(scores, part, axis=1)
    part_ids = labels[part]

    merged_scores = np.concatenate([current_scores, part_scores], axis=1)
    merged_ids = np.concatenate([current_ids, part_ids], axis=1)
    keep = np.argpartition(-merged_scores, k - 1, axis=1)[:, :k]
    kept_scores = np.take_along_axis(merged_scores, keep, axis=1)
    kept_ids = np.take_along_axis(merged_ids, keep, axis=1)
    order = np.argsort(-kept_scores, axis=1)
    return (
        np.take_along_axis(kept_scores, order, axis=1),
        np.take_along_axis(kept_ids, order, axis=1),
    )


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be positive")
    if args.query_chunk <= 0 or args.train_chunk <= 0:
        raise ValueError("--query-chunk and --train-chunk must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = dataset_name(args.hdf5)

    with h5py.File(args.hdf5, "r") as h5:
        train = h5["train"]
        test = h5["test"]
        train_labels = h5["train_labels"][:].astype(np.int64, copy=False)
        num_train = int(train.shape[0])
        num_test = int(test.shape[0])
        dense_dim = int(train.shape[1])
        num_queries = num_test if args.num_queries <= 0 else min(args.num_queries, num_test)

        if args.topk > num_train:
            raise ValueError(f"--topk={args.topk} exceeds train size {num_train}")

        print(f"Input: {args.hdf5}")
        print(f"Dataset: {name}, train={num_train}, queries={num_queries}, dim={dense_dim}")
        print("Loading sparse vectors into CSR...")
        train_sparse = parse_sparse_blob(h5["train_sparse"][:], num_train)
        test_sparse = parse_sparse_blob(h5["test_sparse"][:], num_test)[:num_queries]
        if train_sparse.shape[1] != test_sparse.shape[1]:
            sparse_dim = max(train_sparse.shape[1], test_sparse.shape[1])
            train_sparse = train_sparse.asformat("csr")
            test_sparse = test_sparse.asformat("csr")
            train_sparse.resize((train_sparse.shape[0], sparse_dim))
            test_sparse.resize((test_sparse.shape[0], sparse_dim))

        top_scores = {
            alpha: np.full((num_queries, args.topk), -np.inf, dtype=np.float32)
            for alpha in args.alphas
        }
        top_ids = {
            alpha: np.full((num_queries, args.topk), -1, dtype=np.int64)
            for alpha in args.alphas
        }

        total_train_chunks = math.ceil(num_train / args.train_chunk)
        for chunk_idx, train_start in enumerate(range(0, num_train, args.train_chunk), start=1):
            train_end = min(train_start + args.train_chunk, num_train)
            print(f"Train chunk {chunk_idx}/{total_train_chunks}: rows {train_start}:{train_end}")
            train_dense = train[train_start:train_end].astype(np.float32, copy=False)
            train_sparse_chunk_t = train_sparse[train_start:train_end].transpose().tocsr()
            labels = train_labels[train_start:train_end]

            for query_start, query_end in query_ranges(num_queries, args.query_chunk):
                query_dense = test[query_start:query_end].astype(np.float32, copy=False)
                dense_scores = query_dense @ train_dense.T
                sparse_scores = (test_sparse[query_start:query_end] @ train_sparse_chunk_t).toarray()

                for alpha in args.alphas:
                    scores = alpha * dense_scores + (1.0 - alpha) * sparse_scores
                    rows = slice(query_start, query_end)
                    top_scores[alpha][rows], top_ids[alpha][rows] = update_topk(
                        top_scores[alpha][rows],
                        top_ids[alpha][rows],
                        scores.astype(np.float32, copy=False),
                        labels,
                    )

    for alpha in args.alphas:
        alpha_text = f"{alpha:.1f}"
        output = args.output_dir / f"{name}_ground_truth_alpha_{alpha_text}.npy"
        np.save(output, top_ids[alpha])
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
