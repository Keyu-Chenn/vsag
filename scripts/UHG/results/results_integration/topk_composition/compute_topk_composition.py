#!/usr/bin/env python3
"""Compute top-k composition: how much of hybrid top-k comes from dense / sparse top-k.

For each dataset and each target alpha (0.0, 0.3, 0.5, 0.7, 1.0), loads:
  - hybrid ground truth:  alpha_<a>.npy   (score = a*dense + (1-a)*sparse)
  - dense   ground truth:  alpha_1.0.npy   (score = dense_ip)
  - sparse  ground truth:  alpha_0.0.npy   (score = sparse_ip)

For every query, partitions the hybrid top-k into four disjoint sets:
  both:        in hybrid ∩ dense ∩ sparse
  dense_only:  in hybrid ∩ dense \ sparse
  sparse_only: in hybrid ∩ sparse \ dense
  neither:     in hybrid \ (dense ∪ sparse)

Outputs the mean fraction (over all queries) for each segment, plus per-query
raw data, to a JSON file and an optional CSV.

Usage:
    python3 compute_topk_composition.py
    python3 compute_topk_composition.py --gt-root /path/to/ground_truth
    python3 compute_topk_composition.py --output-dir /path/to/output
    python3 compute_topk_composition.py --topk 100 --alphas 0.0 0.3 0.5 0.7 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASETS = ["nq", "msmarco", "hotpotqa"]
DATASET_LABELS = {
    "nq": "NQ",
    "msmarco": "MS MARCO",
    "hotpotqa": "HotpotQA",
}
SEGMENTS = ["both", "dense_only", "sparse_only", "neither"]


def compute_composition(
    hybrid_ids: np.ndarray,
    dense_ids: np.ndarray,
    sparse_ids: np.ndarray,
    topk: int,
) -> dict[str, np.ndarray]:
    """Compute per-query composition fractions.

    Returns dict mapping segment name to float array of shape (num_queries,).
    """
    num_queries = hybrid_ids.shape[0]
    k = min(topk, hybrid_ids.shape[1], dense_ids.shape[1], sparse_ids.shape[1])

    both = np.zeros(num_queries, dtype=np.float64)
    dense_only = np.zeros(num_queries, dtype=np.float64)
    sparse_only = np.zeros(num_queries, dtype=np.float64)
    neither = np.zeros(num_queries, dtype=np.float64)

    for q in range(num_queries):
        h = set(hybrid_ids[q, :k].tolist())
        d = set(dense_ids[q, :k].tolist())
        s = set(sparse_ids[q, :k].tolist())

        hd = h & d
        hs = h & s

        both[q] = len(hd & hs)
        dense_only[q] = len(hd - hs)
        sparse_only[q] = len(hs - hd)
        neither[q] = len(h - hd - hs)

    return {
        "both": both / k,
        "dense_only": dense_only / k,
        "sparse_only": sparse_only / k,
        "neither": neither / k,
    }


def load_gt(gt_dir: Path, dataset: str, alpha: str) -> np.ndarray:
    fname = gt_dir / dataset / f"{dataset}_ground_truth_alpha_{alpha}.npy"
    if not fname.exists():
        raise FileNotFoundError(f"Ground truth not found: {fname}")
    return np.load(fname)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute top-k composition of hybrid results vs dense/sparse baselines."
    )
    parser.add_argument(
        "--gt-root",
        default="/tbase-project/vsag/scripts/UHG/data/ground_truth",
        help="Root directory containing per-dataset ground truth .npy files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tbase-project/vsag/scripts/UHG/results/results_integration/topk_composition",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Top-k to use for the composition analysis.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=["0.0", "0.3", "0.5", "0.7", "1.0"],
        help="Target alpha values to analyze.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Datasets to process.",
    )
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dense_alpha = "1.0"
    sparse_alpha = "0.0"

    results: dict = {
        "topk": args.topk,
        "alphas": args.alphas,
        "datasets": args.datasets,
        "segments": SEGMENTS,
        "data": {},
    }

    csv_lines = ["dataset,alpha,num_queries,both,dense_only,sparse_only,neither"]

    for dataset in args.datasets:
        print(f"Processing {dataset} ...")
        dense_ids = load_gt(gt_root, dataset, dense_alpha)
        sparse_ids = load_gt(gt_root, dataset, sparse_alpha)
        num_queries = dense_ids.shape[0]

        results["data"][dataset] = {"num_queries": num_queries, "alphas": {}}

        for alpha in args.alphas:
            hybrid_ids = load_gt(gt_root, dataset, alpha)
            comp = compute_composition(hybrid_ids, dense_ids, sparse_ids, args.topk)

            means = {seg: float(np.mean(comp[seg])) for seg in SEGMENTS}
            stds = {seg: float(np.std(comp[seg])) for seg in SEGMENTS}

            results["data"][dataset]["alphas"][alpha] = {
                "mean": means,
                "std": stds,
                "per_query": {seg: comp[seg].tolist() for seg in SEGMENTS},
            }

            print(
                f"  alpha={alpha}: "
                f"both={means['both']:.4f}  "
                f"dense_only={means['dense_only']:.4f}  "
                f"sparse_only={means['sparse_only']:.4f}  "
                f"neither={means['neither']:.4f}"
            )

            csv_lines.append(
                f"{dataset},{alpha},{num_queries},"
                f"{means['both']:.6f},{means['dense_only']:.6f},"
                f"{means['sparse_only']:.6f},{means['neither']:.6f}"
            )

    json_path = output_dir / "topk_composition.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote JSON: {json_path}")

    csv_path = output_dir / "topk_composition.csv"
    with csv_path.open("w") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"Wrote CSV:  {csv_path}")


if __name__ == "__main__":
    main()
