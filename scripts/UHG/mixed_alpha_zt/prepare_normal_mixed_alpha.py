#!/usr/bin/env python3
"""Prepare truncated-normal per-query alphas and exact hybrid ground truth."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import h5py
import numpy as np

MAIN_DIR = (
    Path(__file__).resolve().parent.parent / "scripts" / "hybrid_union" / "main"
)
sys.path.insert(0, str(MAIN_DIR))

from generate_ground_truth_from_hdf5 import (  # noqa: E402
    parse_sparse_blob,
    query_ranges,
    update_topk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate stratified truncated-normal per-query alphas and exact "
            "mixed-alpha ground truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hdf5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha-min", type=float, default=0.3)
    parser.add_argument("--alpha-max", type=float, default=0.7)
    parser.add_argument("--normal-mean", type=float, default=0.5)
    parser.add_argument("--normal-stddev", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument(
        "--num-queries", type=int, default=-1, help="-1 means all test queries."
    )
    parser.add_argument("--query-chunk", type=int, default=32)
    parser.add_argument("--train-chunk", type=int, default=50000)
    return parser.parse_args()


def make_truncated_normal_alphas(
    count: int,
    alpha_min: float,
    alpha_max: float,
    normal_mean: float,
    normal_stddev: float,
    seed: int,
) -> np.ndarray:
    """Return deterministic midpoint quantiles from a bounded normal distribution."""
    if count <= 0:
        raise ValueError("count must be positive")
    if normal_stddev <= 0:
        raise ValueError("normal_stddev must be positive")

    distribution = NormalDist(mu=normal_mean, sigma=normal_stddev)
    lower_cdf = distribution.cdf(alpha_min)
    upper_cdf = distribution.cdf(alpha_max)
    if not lower_cdf < upper_cdf:
        raise ValueError("truncated normal interval has no probability mass")

    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
    probabilities = lower_cdf + (upper_cdf - lower_cdf) * quantiles
    alphas = np.fromiter(
        (distribution.inv_cdf(float(value)) for value in probabilities),
        dtype=np.float64,
        count=count,
    )
    np.random.default_rng(seed).shuffle(alphas)
    return alphas.astype(np.float32)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha_min < args.alpha_max <= 1.0:
        raise ValueError("Require 0 <= --alpha-min < --alpha-max <= 1")
    if not args.alpha_min <= args.normal_mean <= args.alpha_max:
        raise ValueError("--normal-mean must lie inside the alpha interval")
    if args.normal_stddev <= 0:
        raise ValueError("--normal-stddev must be positive")
    if args.topk <= 0 or args.query_chunk <= 0 or args.train_chunk <= 0:
        raise ValueError("--topk, --query-chunk and --train-chunk must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.hdf5, "r") as h5:
        train = h5["train"]
        test = h5["test"]
        train_labels = h5["train_labels"][:].astype(np.int64, copy=False)
        num_train = int(train.shape[0])
        num_test = int(test.shape[0])
        num_queries = num_test if args.num_queries <= 0 else min(args.num_queries, num_test)
        if num_queries <= 0:
            raise ValueError("No test queries selected")
        if args.topk > num_train:
            raise ValueError(f"--topk={args.topk} exceeds train size {num_train}")

        alphas = make_truncated_normal_alphas(
            num_queries,
            args.alpha_min,
            args.alpha_max,
            args.normal_mean,
            args.normal_stddev,
            args.seed,
        )
        alpha_path = args.output_dir / "alphas.npy"

        print(f"Input: {args.hdf5}")
        print(f"train={num_train}, queries={num_queries}, topk={args.topk}")
        print(
            "truncated normal alpha: "
            f"bounds=[{args.alpha_min}, {args.alpha_max}], "
            f"mu={args.normal_mean}, sigma={args.normal_stddev}, "
            f"sample_min={alphas.min():.6f}, sample_max={alphas.max():.6f}, "
            f"sample_mean={alphas.mean():.6f}, seed={args.seed}"
        )
        print("Loading sparse vectors into CSR...")
        train_sparse = parse_sparse_blob(h5["train_sparse"][:], num_train)
        test_sparse = parse_sparse_blob(h5["test_sparse"][:], num_test)[:num_queries]
        if train_sparse.shape[1] != test_sparse.shape[1]:
            sparse_dim = max(train_sparse.shape[1], test_sparse.shape[1])
            train_sparse.resize((train_sparse.shape[0], sparse_dim))
            test_sparse.resize((test_sparse.shape[0], sparse_dim))

        top_scores = np.full((num_queries, args.topk), -np.inf, dtype=np.float32)
        top_ids = np.full((num_queries, args.topk), -1, dtype=np.int64)
        total_train_chunks = math.ceil(num_train / args.train_chunk)

        for chunk_idx, train_start in enumerate(
            range(0, num_train, args.train_chunk), start=1
        ):
            train_end = min(train_start + args.train_chunk, num_train)
            print(
                f"Train chunk {chunk_idx}/{total_train_chunks}: "
                f"rows {train_start}:{train_end}"
            )
            train_dense = train[train_start:train_end].astype(np.float32, copy=False)
            train_sparse_chunk_t = (
                train_sparse[train_start:train_end].transpose().tocsr()
            )
            labels = train_labels[train_start:train_end]

            for query_start, query_end in query_ranges(
                num_queries, args.query_chunk
            ):
                query_dense = test[query_start:query_end].astype(
                    np.float32, copy=False
                )
                dense_scores = query_dense @ train_dense.T
                sparse_scores = (
                    test_sparse[query_start:query_end] @ train_sparse_chunk_t
                ).toarray()
                query_alphas = alphas[query_start:query_end, None]
                scores = (
                    query_alphas * dense_scores
                    + (1.0 - query_alphas) * sparse_scores
                )
                rows = slice(query_start, query_end)
                top_scores[rows], top_ids[rows] = update_topk(
                    top_scores[rows],
                    top_ids[rows],
                    scores.astype(np.float32, copy=False),
                    labels,
                )

    gt_path = args.output_dir / "ground_truth.npy"
    alpha_tmp = args.output_dir / "alphas.tmp.npy"
    gt_tmp = args.output_dir / "ground_truth.tmp.npy"
    metadata_tmp = args.output_dir / "metadata.tmp.json"
    np.save(alpha_tmp, alphas)
    np.save(gt_tmp, top_ids)
    metadata = {
        "distribution": "truncated_normal",
        "generation": "stratified_midpoint_quantiles_then_seeded_shuffle",
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "normal_mean": args.normal_mean,
        "normal_stddev": args.normal_stddev,
        "seed": args.seed,
        "num_queries": num_queries,
        "topk": args.topk,
        "alpha_file": alpha_path.name,
        "ground_truth_file": gt_path.name,
    }
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    alpha_tmp.replace(alpha_path)
    gt_tmp.replace(gt_path)
    metadata_tmp.replace(args.output_dir / "metadata.json")
    print(f"Saved {alpha_path}")
    print(f"Saved {gt_path}")
    print(f"Saved {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
