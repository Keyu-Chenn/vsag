#!/usr/bin/env python3
"""Validate shared indexes/graphs and truncated-normal mixed-alpha inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from prepare_normal_mixed_alpha import make_truncated_normal_alphas


DEFAULT_DATASETS = ("nq", "hotpotqa", "msmarco", "dbpedia-entity", "fever")
ALPHA_MIN = 0.3
ALPHA_MAX = 0.7
NORMAL_MEAN = 0.5
NORMAL_STDDEV = 0.1
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-data-dir",
        type=Path,
        default=Path("/tbase-project/vsag/scripts/UHG/data"),
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    return parser.parse_args()


def require_file(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing: {path}")


def load_array(path: Path, errors: list[str]) -> np.ndarray | None:
    if not path.is_file():
        errors.append(f"missing: {path}")
        return None
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cannot load {path}: {exc}")
        return None


def load_metadata(path: Path, errors: list[str]) -> dict[str, object] | None:
    if not path.is_file():
        errors.append(f"missing: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"invalid JSON {path}: {exc}")
        return None


def validate_root(
    root: Path,
    dataset: str,
    query_count: int,
    expected_topk: int,
    expected_alphas: np.ndarray,
    errors: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    dataset_root = root / dataset
    alpha_path = dataset_root / "alphas.npy"
    gt_path = dataset_root / "ground_truth.npy"
    alphas = load_array(alpha_path, errors)
    ground_truth = load_array(gt_path, errors)
    metadata = load_metadata(dataset_root / "metadata.json", errors)

    if alphas is not None:
        if alphas.dtype != np.float32 or alphas.shape != (query_count,):
            errors.append(
                f"invalid alpha array {alpha_path}: "
                f"shape={alphas.shape} dtype={alphas.dtype}"
            )
        elif not np.array_equal(alphas, expected_alphas):
            errors.append(f"alpha assignment does not match configured normal: {alpha_path}")

    if ground_truth is not None:
        if ground_truth.dtype != np.int64 or ground_truth.shape != (
            query_count,
            expected_topk,
        ):
            errors.append(
                f"invalid GT array {gt_path}: "
                f"shape={ground_truth.shape} dtype={ground_truth.dtype}"
            )

    if metadata is not None:
        expected_metadata = {
            "distribution": "truncated_normal",
            "alpha_min": ALPHA_MIN,
            "alpha_max": ALPHA_MAX,
            "normal_mean": NORMAL_MEAN,
            "normal_stddev": NORMAL_STDDEV,
            "seed": SEED,
            "num_queries": query_count,
            "topk": expected_topk,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                errors.append(
                    f"metadata mismatch {dataset_root}: "
                    f"{key}={metadata.get(key)!r}, expected={expected!r}"
                )

    return alphas, ground_truth


def main() -> None:
    args = parse_args()
    shared_data_dir = args.shared_data_dir.resolve()
    experiment_root = args.experiment_root.resolve()
    errors: list[str] = []

    for dataset in args.datasets:
        hdf5_path = shared_data_dir / "hdf5" / f"{dataset}.hdf5"
        if not hdf5_path.is_file():
            errors.append(f"missing: {hdf5_path}")
            continue
        try:
            with h5py.File(hdf5_path, "r") as h5:
                query_count = int(h5["test"].shape[0])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cannot read test shape from {hdf5_path}: {exc}")
            continue

        for path in (
            shared_data_dir / "index" / f"701_{dataset}_dense_hnsw.index",
            shared_data_dir / "index" / f"701_{dataset}_sparse_sindi.index",
            shared_data_dir / "index" / f"709_{dataset}_sindi.index",
            shared_data_dir / "fhg" / f"{dataset}_fhg_alpha_0_5.h5",
            shared_data_dir / "uhg" / f"{dataset}_uhg.h5",
        ):
            require_file(path, errors)

        expected_alphas = make_truncated_normal_alphas(
            query_count,
            ALPHA_MIN,
            ALPHA_MAX,
            NORMAL_MEAN,
            NORMAL_STDDEV,
            SEED,
        )
        validate_root(
            experiment_root / "data" / "gt_top200",
            dataset,
            query_count,
            200,
            expected_alphas,
            errors,
        )

        print(
            f"input OK: dataset={dataset} queries={query_count} "
            "distribution=truncated_normal GT@10/20/50/100/200=top200"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"All normal mixed-alpha inputs OK: {len(args.datasets)} datasets")


if __name__ == "__main__":
    main()
