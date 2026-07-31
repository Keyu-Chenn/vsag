#!/usr/bin/env python3
"""Validate all immutable inputs for the mixed-alpha Recall@10/20/200 suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DATASETS = ("nq", "hotpotqa", "msmarco", "dbpedia-entity", "fever")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/tbase-project/vsag/scripts/UHG/data"),
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    return parser.parse_args()


def load_metadata(path: Path, errors: list[str]) -> dict[str, object] | None:
    if not path.is_file():
        errors.append(f"missing: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report malformed experiment input
        errors.append(f"invalid JSON {path}: {exc}")
        return None


def load_array(path: Path, errors: list[str]) -> np.ndarray | None:
    if not path.is_file():
        errors.append(f"missing: {path}")
        return None
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - report malformed experiment input
        errors.append(f"cannot load {path}: {exc}")
        return None


def require_file(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing: {path}")


def validate_mixed_root(
    root: Path,
    dataset: str,
    query_count: int,
    required_topk: int,
    errors: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    dataset_root = root / dataset
    alphas = load_array(dataset_root / "alphas.npy", errors)
    gt = load_array(dataset_root / "ground_truth.npy", errors)
    metadata = load_metadata(dataset_root / "metadata.json", errors)

    if alphas is not None:
        if alphas.dtype != np.float32 or alphas.ndim != 1:
            errors.append(
                f"expected float32 alpha vector: {dataset_root / 'alphas.npy'}, "
                f"got shape={alphas.shape} dtype={alphas.dtype}"
            )
        elif alphas.shape[0] != query_count:
            errors.append(
                f"alpha/query mismatch for {dataset}: "
                f"{alphas.shape[0]} != {query_count}"
            )
        elif alphas.size:
            if float(alphas.min()) < 0.3 - 1e-6 or float(alphas.max()) > 0.7 + 1e-6:
                errors.append(f"alpha outside [0.3, 0.7]: {dataset_root}")
            if abs(float(alphas.mean()) - 0.5) > 1e-6:
                errors.append(f"alpha mean is not 0.5: {dataset_root}")

    if gt is not None:
        if gt.dtype != np.int64 or gt.ndim != 2:
            errors.append(
                f"expected int64 2D ground truth: {dataset_root / 'ground_truth.npy'}, "
                f"got shape={gt.shape} dtype={gt.dtype}"
            )
        else:
            if gt.shape[0] != query_count:
                errors.append(
                    f"GT/query mismatch for {dataset}: {gt.shape[0]} != {query_count}"
                )
            if gt.shape[1] < required_topk:
                errors.append(
                    f"GT too narrow for Recall@{required_topk}: "
                    f"{dataset_root / 'ground_truth.npy'} shape={gt.shape}"
                )

    if metadata is not None:
        if int(metadata.get("seed", -1)) != 42:
            errors.append(f"metadata seed is not 42: {dataset_root}")
        if int(metadata.get("topk", -1)) < required_topk:
            errors.append(
                f"metadata topk is below {required_topk}: {dataset_root}"
            )

    return alphas, gt


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    errors: list[str] = []

    for dataset in args.datasets:
        hdf5_path = data_dir / "hdf5" / f"{dataset}.hdf5"
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
            data_dir / "index" / f"701_{dataset}_dense_hnsw.index",
            data_dir / "index" / f"701_{dataset}_sparse_sindi.index",
            data_dir / "index" / f"709_{dataset}_sindi.index",
            data_dir / "fhg" / f"{dataset}_fhg_alpha_0_5.h5",
            data_dir / "uhg" / f"{dataset}_uhg.h5",
        ):
            require_file(path, errors)

        alphas_100, _ = validate_mixed_root(
            data_dir / "mixed_alpha", dataset, query_count, 20, errors
        )
        alphas_1000, _ = validate_mixed_root(
            data_dir / "mixed_alpha_gt_1000",
            dataset,
            query_count,
            200,
            errors,
        )
        if (
            alphas_100 is not None
            and alphas_1000 is not None
            and alphas_100.shape == alphas_1000.shape
            and not np.array_equal(alphas_100, alphas_1000)
        ):
            errors.append(
                f"mixed-alpha assignments differ between GT roots: {dataset}"
            )

        print(
            f"input OK: dataset={dataset} queries={query_count} "
            "GT@10/20=top100 GT@200=top1000"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"All mixed-alpha inputs OK: {len(args.datasets)} datasets")


if __name__ == "__main__":
    main()
