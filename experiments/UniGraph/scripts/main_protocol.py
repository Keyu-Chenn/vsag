#!/usr/bin/env python3
"""Shared protocol for the final UniGraph Recall--QPS experiments."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import tempfile


DATASETS = ("nq", "hotpotqa", "msmarco", "fever", "dbpedia-entity")
FIXED_DATASETS = ("nq", "msmarco", "dbpedia-entity")
QUERY_COUNT = {name: (467 if name == "dbpedia-entity" else 1000) for name in DATASETS}

ORIGINAL_METHODS = ("hnsw", "sindi", "hnsw_sindi", "uhg")
ADDED_METHODS = ("naive_hybrid_hnsw", "optimized_hybrid_hnsw")
ORIGINAL_POINTS = {
    "hnsw": (100, 200, 300, 500, 800, 1000),
    "sindi": (1000, 1500, 2000, 3000, 5000, 10000),
    "hnsw_sindi": (100, 150, 200, 300, 500),
    "uhg": (100, 200, 300, 500, 1000),
}
ADDED_HNSW_POINTS = ORIGINAL_POINTS["hnsw"]
LADR_POINTS = (1, 2, 5, 10, 20, 50, 100, 200)
ABLATION_SUITES = frozenset(("entry_point", "hybrid_pruning"))
ORIGINAL_BINARIES = {
    "hnsw": "708_uhg_exp8",
    "sindi": "709_uhg_exp9",
    "hnsw_sindi": "701_uhg_exp1",
    "uhg": "703_uhg_exp3",
}

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RECALL_RE = re.compile(rf"^Recall:\s*({NUMBER})\s*$")
QPS_RE = re.compile(rf"^QPS:\s*({NUMBER})\s*$")
SEARCH_SECONDS_RE = re.compile(rf"^SearchSeconds:\s*({NUMBER})\s*$")
QPS_SCOPE_RE = re.compile(r"^QPSScope:\s*(\S+)\s*$")
NUM_QUERIES_RE = re.compile(r"^num_queries:\s*(\d+)\s*$")
METRIC_SCHEMA_RE = re.compile(r"^MetricSchema:\s*(\d+)\s*$")


def allowed_cpus() -> tuple[int, ...]:
    """Return the logical CPUs available to this process."""
    if hasattr(os, "sched_getaffinity"):
        cpus = tuple(sorted(os.sched_getaffinity(0)))
    else:
        cpus = tuple(range(os.cpu_count() or 1))
    if not cpus:
        raise RuntimeError("the process has no available CPUs")
    return cpus


def resolve_cpu(requested: int | None) -> int:
    """Choose an allowed CPU, or validate an explicit selection."""
    cpus = allowed_cpus()
    if requested is None:
        return cpus[0]
    if requested not in cpus:
        ranges = f"{cpus[0]}-{cpus[-1]}" if len(cpus) > 1 else str(cpus[0])
        raise ValueError(
            f"CPU {requested} is unavailable to this process "
            f"(allowed logical CPUs span {ranges}; count={len(cpus)})"
        )
    return requested


@dataclasses.dataclass(frozen=True)
class Job:
    suite: str
    dataset: str
    method: str
    k: int
    alpha: str = "mixed"
    variant: str = ""
    search_sparse_keep: float | None = None

    @property
    def points(self) -> tuple[int, ...]:
        if self.suite in ABLATION_SUITES:
            return (300,)
        if self.method == "ladr_adaptive_hybrid":
            return LADR_POINTS
        source = (ADDED_HNSW_POINTS if self.method in ADDED_METHODS
                  else ORIGINAL_POINTS[self.method])
        return tuple(point * self.k // 100 for point in source)

    @property
    def cache_kind(self) -> str | None:
        return "uhg" if self.method == "uhg" else None


def gt_paths(job: Job, data: Path) -> tuple[Path | None, Path]:
    if job.alpha == "mixed":
        root = data / "mixed_alpha_gt_1000" / job.dataset
        return root / "alphas.npy", root / "ground_truth.npy"
    return (
        None,
        data / "ground_truth" / job.dataset /
        f"{job.dataset}_ground_truth_alpha_{job.alpha}.npy",
    )


def index_path(args: argparse.Namespace, job: Job) -> Path:
    if job.method in ADDED_METHODS:
        return args.baseline_index_root / job.dataset / f"{job.method}.index"
    if job.method == "sindi":
        return args.original_index_dir / f"701_{job.dataset}_sparse_sindi.index"
    if job.method in {"hnsw", "hnsw_sindi"}:
        return args.original_index_dir / f"701_{job.dataset}_dense_hnsw.index"
    if job.method == "uhg":
        return args.deployable_index_dir / job.dataset / "uhg.index"
    raise ValueError(f"unknown method: {job.method}")


def point_list_argument(points: int | tuple[int, ...] | list[int]) -> str:
    if isinstance(points, int):
        return str(points)
    if not points:
        raise ValueError("search point list must not be empty")
    return ",".join(str(point) for point in points)


def baseline_search_command(
        args: argparse.Namespace, job: Job,
        points: int | tuple[int, ...] | list[int]) -> list[str]:
    alphas, truth = gt_paths(job, args.data_dir)
    command = [
        str(args.baseline_bin), "search",
        "--data", str(args.data_dir / "hdf5" / f"{job.dataset}.hdf5"),
        "--method", job.method,
        "--index", str(index_path(args, job)),
        "--gt", str(truth),
        "--k", str(job.k),
        "--points", point_list_argument(points),
    ]
    command += ["--alphas", str(alphas)] if alphas else ["--alpha", job.alpha]
    if job.method == "optimized_hybrid_hnsw":
        if job.search_sparse_keep is None:
            raise ValueError("optimized Hybrid HNSW is missing sparse_keep")
        command += [
            "--tau-dense", str(args.tau_dense),
            "--tau-hybrid", str(args.tau_hybrid),
            "--sparse-keep", str(job.search_sparse_keep),
        ]
    return command


def original_search_command(
        args: argparse.Namespace, job: Job,
        points: int | tuple[int, ...] | list[int], cache: Path | None) -> list[str]:
    binary = args.original_bin_dir / ORIGINAL_BINARIES[job.method]
    command = [str(binary), str(args.data_dir / "hdf5" / f"{job.dataset}.hdf5")]
    alphas, truth = gt_paths(job, args.data_dir)
    if alphas:
        command += [
            "--mixed_alpha_file", str(alphas),
            "--mixed_gt_file", str(truth),
            "--alpha", "0.5",
        ]
    else:
        command += ["--gt_dir", str(truth.parent), "--alpha", job.alpha]
    command += ["-k", str(job.k), "--num_queries", "-1", "--threads", "1"]

    if job.cache_kind is None:
        command += ["--index_dir", str(args.original_index_dir)]
        command += [
            "--bk_list" if job.method == "sindi" else "--search_points",
            point_list_argument(points),
        ]
        if job.method != "hnsw":
            command += ["--query_prune_ratio", "0.5", "--term_prune_ratio", "0"]
        return command

    if cache is None:
        raise ValueError(f"missing packaged index for {job.dataset}/{job.method}")
    command += [
        "--hybrid_index_path", str(cache),
        "--index_dir", str(cache.parent),
        "--build_alpha", "0.5", "--max_hops", "0",
    ]
    mode = "uhg" if job.variant == "unguided" else "auto"
    scale = "1.0" if job.variant == "prune_no_prune" else "0.3"
    command += [
        "--method", mode,
        "--hybrid_prune_scale", scale,
        "--sindi_query_prune_ratio", "0.5",
        "--sindi_term_prune_ratio", "0",
    ]
    if job.variant == "prune_no_prune":
        command += ["--disable_hybrid_pruning"]
    if job.suite in ABLATION_SUITES:
        if not isinstance(points, int):
            if len(points) != 1:
                raise ValueError("ablation requires exactly one search point")
            points = points[0]
        command += [
            "--ef_search", str(points), "--sindi_bk", str(points),
            "--dense_entry_bk", str(points),
            "--dense_entry_ef_search", str(points),
        ]
    else:
        command += ["--search_points", point_list_argument(points)]
    return command


def npy_info(path: Path) -> tuple[str, tuple[int, ...]]:
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"invalid NPY magic: {path}")
        major, _minor = struct.unpack("BB", stream.read(2))
        if major == 1:
            header_size = struct.unpack("<H", stream.read(2))[0]
        elif major in (2, 3):
            header_size = struct.unpack("<I", stream.read(4))[0]
        else:
            raise ValueError(f"unsupported NPY version {major}: {path}")
        header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
    if header.get("fortran_order") is not False:
        raise ValueError(f"Fortran-order NPY is unsupported: {path}")
    return str(header["descr"]), tuple(int(value) for value in header["shape"])


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=".partial-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def command_display(command: list[str], cpus: tuple[int, ...]) -> str:
    return f"taskset -c {','.join(map(str, cpus))} " + shlex.join(command)


def parse_search_metrics(
        text: str, added: bool, expected_queries: int,
        expected_points: tuple[int, ...] | list[int] | None = None,
        original_method: str | None = None) -> list[dict]:
    points = tuple(expected_points) if expected_points is not None else None
    expected_count = len(points) if points is not None else 1
    if added:
        metrics = [
            json.loads(line) for line in text.splitlines()
            if line.startswith('{"method"')
        ]
        if len(metrics) != expected_count:
            raise ValueError(
                f"expected {expected_count} JSON metrics, found {len(metrics)}")
    else:
        lines = [line.strip() for line in text.splitlines()]
        values = {
            "recall": [float(match.group(1)) for line in lines
                       if (match := RECALL_RE.fullmatch(line))],
            "qps": [float(match.group(1)) for line in lines
                    if (match := QPS_RE.fullmatch(line))],
            "search_seconds": [float(match.group(1)) for line in lines
                               if (match := SEARCH_SECONDS_RE.fullmatch(line))],
            "qps_scope": [match.group(1) for line in lines
                          if (match := QPS_SCOPE_RE.fullmatch(line))],
            "queries": [int(match.group(1)) for line in lines
                        if (match := NUM_QUERIES_RE.fullmatch(line))],
            "metric_schema": [int(match.group(1)) for line in lines
                              if (match := METRIC_SCHEMA_RE.fullmatch(line))],
        }
        if any(len(found) != expected_count for found in values.values()):
            counts = {name: len(found) for name, found in values.items()}
            raise ValueError(
                f"expected {expected_count} complete metrics, found {counts}")
        metrics = [
            {name: found[index] for name, found in values.items()}
            for index in range(expected_count)
        ]
        point_patterns = {
            "hnsw": re.compile(r"^bk:\s*(\d+)\s*$"),
            "sindi": re.compile(r"^bk:\s*(\d+)\s*$"),
            "hnsw_sindi": re.compile(r"^bk_dense:\s*(\d+)\s*$"),
            "uhg": re.compile(r"^ef_search:\s*(\d+)\s*$"),
        }
        if points is not None and original_method in point_patterns:
            pattern = point_patterns[original_method]
            reported = [
                int(match.group(1)) for line in lines
                if (match := pattern.fullmatch(line))
            ]
            if reported != list(points):
                raise ValueError(f"reported points {reported}; expected {list(points)}")
            for metric, point in zip(metrics, points):
                metric["point"] = point

    if points is not None and added:
        reported = [int(metric.get("point", -1)) for metric in metrics]
        if reported != list(points):
            raise ValueError(f"reported points {reported}; expected {list(points)}")

    for metric in metrics:
        if metric.get("metric_schema") != 2 or metric.get("qps_scope") != "query_to_topk":
            raise ValueError("unsupported QPS metric definition")
        if (added and metric.get("method") in ADDED_METHODS and
                metric.get("api_scope") != "dataset_json_knnsearch"):
            raise ValueError("hybrid-HNSW did not use the public Dataset/JSON API")
        if metric.get("queries") != expected_queries:
            raise ValueError(
                f"reported {metric.get('queries')} queries; expected {expected_queries}")
        qps = float(metric.get("qps", math.nan))
        seconds = float(metric.get("search_seconds", math.nan))
        if (not math.isfinite(qps) or qps <= 0 or
                not math.isfinite(seconds) or seconds <= 0 or
                not math.isclose(
                    qps, expected_queries / seconds,
                    rel_tol=1e-8, abs_tol=1e-10)):
            raise ValueError("QPS is inconsistent with query count/search_seconds")
    return metrics
