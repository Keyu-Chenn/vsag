#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


DEFAULT_GT_ROOT = Path("/tbase-project/vsag/scripts/UHG/data/ground_truth")
DEFAULT_OUTPUT = Path("/tbase-project/vsag/scripts/UHG/results/topk_overlap/nq_topk_overlap.txt")
DEFAULT_ALPHAS = [i / 10.0 for i in range(11)]
DEFAULT_KS = [10, 20, 50, 100]
DEFAULT_ALPHA_K = 100


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def alpha_label(alpha):
    return f"{alpha:.1f}"


def ground_truth_path(gt_dir, dataset, alpha):
    return gt_dir / f"{dataset}_ground_truth_alpha_{alpha_label(alpha)}.npy"


def load_topk(gt_dir, dataset, alpha):
    path = ground_truth_path(gt_dir, dataset, alpha)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    return np.load(path, mmap_mode="r")


def mean_topk_overlap(left, right, k):
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"Query count mismatch: {left.shape[0]} vs {right.shape[0]}")
    if left.shape[1] < k or right.shape[1] < k:
        raise ValueError(
            f"k={k} exceeds available topk: left={left.shape[1]}, right={right.shape[1]}"
        )

    overlaps = []
    for query_id in range(left.shape[0]):
        left_set = set(left[query_id, :k].tolist())
        right_set = set(right[query_id, :k].tolist())
        overlaps.append(len(left_set & right_set) / k)
    return float(np.mean(overlaps))


def analyze(dataset, gt_dir, alphas, ks, alpha_k):
    dense = load_topk(gt_dir, dataset, 1.0)
    sparse = load_topk(gt_dir, dataset, 0.0)

    max_k = max(max(ks), alpha_k)
    if dense.shape[1] < max_k or sparse.shape[1] < max_k:
        raise ValueError(
            f"max k={max_k} exceeds dense/sparse ground truth topk: "
            f"dense={dense.shape[1]}, sparse={sparse.shape[1]}"
        )

    lines = []

    print(f"Dataset: {dataset}")
    print(f"Ground truth dir: {gt_dir}")
    print(f"Queries: {dense.shape[0]}")
    print()
    lines.extend(
        [
            f"Dataset: {dataset}",
            f"Ground truth dir: {gt_dir}",
            f"Queries: {dense.shape[0]}",
            "",
        ]
    )

    print("Dense topk vs Sparse topk overlap")
    print("k\toverlap")
    lines.append("Dense topk vs Sparse topk overlap")
    lines.append("k\toverlap")
    for k in ks:
        overlap = mean_topk_overlap(dense, sparse, k)
        print(f"{k}\t{overlap:.6f}")
        lines.append(f"{k}\t{overlap:.6f}")

    print()
    print("Alpha topk overlap with Dense topk and Sparse topk")
    print("alpha\ttopk\talpha_dense_overlap\talpha_sparse_overlap")
    lines.extend(
        [
            "",
            "Alpha topk overlap with Dense topk and Sparse topk",
            "alpha\ttopk\talpha_dense_overlap\talpha_sparse_overlap",
        ]
    )
    for alpha in alphas:
        alpha_topk = load_topk(gt_dir, dataset, alpha)
        dense_overlap = mean_topk_overlap(alpha_topk, dense, alpha_k)
        sparse_overlap = mean_topk_overlap(alpha_topk, sparse, alpha_k)
        print(
            f"{alpha_label(alpha)}\t{alpha_k}\t{dense_overlap:.6f}\t{sparse_overlap:.6f}"
        )
        lines.append(
            f"{alpha_label(alpha)}\t{alpha_k}\t{dense_overlap:.6f}\t{sparse_overlap:.6f}"
        )

    return lines


def write_txt(lines, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    print()
    print(f"Saved TXT: {output}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze topk set overlap between dense, sparse, and alpha-weighted "
            "hybrid ground truth."
        )
    )
    parser.add_argument("--dataset", default="nq", help="Dataset name. Default: nq")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing {dataset}_ground_truth_alpha_*.npy. "
            "Default: /tbase-project/vsag/scripts/UHG/data/ground_truth/{dataset}"
        ),
    )
    parser.add_argument(
        "--alphas",
        default=",".join(alpha_label(alpha) for alpha in DEFAULT_ALPHAS),
        help="Comma-separated alpha values. Default: 0.0,0.1,...,1.0",
    )
    parser.add_argument(
        "--ks",
        default=",".join(str(k) for k in DEFAULT_KS),
        help="Comma-separated k values. Default: 10,20,50,100",
    )
    parser.add_argument(
        "--alpha-k",
        type=int,
        default=DEFAULT_ALPHA_K,
        help="Topk used for alpha-vs-dense/sparse overlap. Default: 100",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output TXT path. Default: {DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    gt_dir = args.gt_dir or (DEFAULT_GT_ROOT / args.dataset)
    lines = analyze(
        dataset=args.dataset,
        gt_dir=gt_dir,
        alphas=parse_float_list(args.alphas),
        ks=parse_int_list(args.ks),
        alpha_k=args.alpha_k,
    )
    write_txt(lines, args.output)


if __name__ == "__main__":
    main()
