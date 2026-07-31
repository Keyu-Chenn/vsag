#!/usr/bin/env python3
"""Validate mixed-alpha result coverage and write one Recall/QPS TSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_DATASETS = ("nq", "hotpotqa", "msmarco", "dbpedia-entity", "fever")
DEFAULT_METHODS = ("hnsw", "sindi", "hnsw_sindi", "fhg", "uhg")
DEFAULT_RECALLS = (10, 20, 200)
EXPECTED_POINTS = {
    10: {
        "hnsw": (10, 20, 30, 50, 80, 100),
        "sindi": (100, 150, 200, 300, 500, 1000),
        "hnsw_sindi": (10, 15, 20, 30, 50, 100),
        "fhg": (10, 20, 30, 50, 80, 100),
        "uhg": (10, 20, 30, 50, 80, 100),
    },
    20: {
        "hnsw": (20, 40, 60, 100, 160, 200),
        "sindi": (200, 300, 400, 600, 1000, 2000),
        "hnsw_sindi": (20, 30, 40, 60, 100, 200),
        "fhg": (20, 40, 60, 100, 160, 200),
        "uhg": (20, 40, 60, 100, 160, 200),
    },
    200: {
        "hnsw": (200, 400, 600, 1000, 1600, 2000),
        "sindi": (2000, 3000, 4000, 6000, 10000, 20000),
        "hnsw_sindi": (200, 300, 400, 600, 1000),
        "fhg": (200, 400, 600, 1000, 2000),
        "uhg": (200, 400, 600, 1000, 2000),
    },
}
POINT_RE = re.compile(r"\b(?:ef_search|bk|point)=(\d+)\b")
RECALL_RE = re.compile(r"\bRecall:\s*([0-9.eE+-]+)")
QPS_RE = re.compile(r"\bQPS:\s*([0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent / "results"
    parser.add_argument("--result-root", type=Path, default=default_root)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--recalls", nargs="+", type=int, default=DEFAULT_RECALLS
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: <result-root>/mixed_alpha_points.tsv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.result_root / "mixed_alpha_points.tsv"
    rows: list[tuple[int, str, str, int, float, float]] = []
    errors: list[str] = []

    for recall_at in args.recalls:
        if recall_at not in EXPECTED_POINTS:
            errors.append(f"unsupported recall value: {recall_at}")
            continue
        for dataset in args.datasets:
            for method in args.methods:
                path = (
                    args.result_root
                    / f"recall_{recall_at}"
                    / dataset
                    / f"{method}.txt"
                )
                if not path.is_file():
                    errors.append(f"missing: {path}")
                    continue
                found: dict[int, tuple[float, float]] = {}
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.startswith(f"{method} "):
                        continue
                    point_match = POINT_RE.search(line)
                    recall_match = RECALL_RE.search(line)
                    qps_match = QPS_RE.search(line)
                    if not (point_match and recall_match and qps_match):
                        continue
                    point = int(point_match.group(1))
                    found[point] = (
                        float(recall_match.group(1)),
                        float(qps_match.group(1)),
                    )
                expected = EXPECTED_POINTS[recall_at][method]
                if tuple(found) != expected:
                    errors.append(
                        f"point mismatch: {path}; expected={expected}, "
                        f"found={tuple(found)}"
                    )
                for point, (recall, qps) in found.items():
                    rows.append((recall_at, dataset, method, point, recall, qps))

    if args.strict and errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    for error in errors:
        print(f"WARNING: {error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("recall_at\tdataset\tmethod\tpoint\trecall\tqps\n")
        for row in rows:
            stream.write("\t".join(map(str, row)) + "\n")
    print(f"Wrote {len(rows)} points: {output}")


if __name__ == "__main__":
    main()
