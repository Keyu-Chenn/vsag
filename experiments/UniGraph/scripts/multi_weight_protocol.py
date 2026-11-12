#!/usr/bin/env python3
"""Execution protocol for the fixed- versus multi-weight graph comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import main_protocol as common


ROOT = Path(__file__).resolve().parent.parent
ENV_PYTHON = ROOT / ".venv" / "bin" / "python"

try:
    import numpy as np
except ModuleNotFoundError:
    if ENV_PYTHON.is_file() and Path(sys.executable) != ENV_PYTHON:
        os.execv(str(ENV_PYTHON), [str(ENV_PYTHON), *sys.argv])
    raise


REPO = ROOT.parents[1]
BINARY = ROOT / "bin" / "703_uhg_exp3"
DATA_ROOT = ROOT / "data"
INDEX_ROOT = ROOT / "index" / "deployable-indexes"
WORKLOAD_ROOT = DATA_ROOT / "mixed_0_1"

DATASETS = ("nq", "hotpotqa", "msmarco", "fever", "dbpedia-entity")
TOPOLOGIES = ("fhg", "uhg")
CPU = common.resolve_cpu(None)
K = 100
UHG_KG = 32
BLOCK_PATTERN = re.compile(
    r"========== Results ==========\n(.*?)============================",
    re.DOTALL,
)


def configure_cpu(requested: int | None) -> int:
    global CPU
    CPU = common.resolve_cpu(requested)
    return CPU


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, with_hash: bool = False) -> dict:
    stat = path.stat()
    value = {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if with_hash:
        value["sha256"] = sha256(path)
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_path(dataset: str, topology: str) -> Path:
    # Table V uses graph-only search, so the canonical FHG/UHG packages already
    # contain the exact two graph payloads needed by this comparison.
    if topology == "fhg":
        return INDEX_ROOT / dataset / "fhg.index"
    if topology == "uhg":
        return INDEX_ROOT / dataset / "uhg.index"
    raise ValueError(f"unknown topology: {topology}")


def workload_paths(_workload: str, dataset: str) -> tuple[Path, Path]:
    root = WORKLOAD_ROOT / dataset
    return root / "alphas.npy", root / "ground_truth.npy"


def preflight(
        run_root: Path, datasets: tuple[str, ...], workloads: tuple[str, ...],
        points: tuple[int, ...], query_limit: int, repetitions: int) -> None:
    required = [BINARY]
    details: dict[str, dict] = {}
    for dataset in datasets:
        required.extend([
            DATA_ROOT / "hdf5" / f"{dataset}.hdf5",
            package_path(dataset, "fhg"),
            package_path(dataset, "uhg"),
        ])
        for workload in workloads:
            required.extend(workload_paths(workload, dataset))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    for dataset in datasets:
        details[dataset] = {
            "fhg_package": file_info(package_path(dataset, "fhg")),
            "uhg_package": file_info(package_path(dataset, "uhg")),
            "workloads": {},
        }
        for workload in workloads:
            alphas, truth = workload_paths(workload, dataset)
            alpha_array = np.load(alphas, mmap_mode="r")
            truth_array = np.load(truth, mmap_mode="r")
            if alpha_array.ndim != 1 or truth_array.ndim != 2:
                raise RuntimeError(f"invalid workload shapes: {workload}/{dataset}")
            if len(alpha_array) != truth_array.shape[0] or truth_array.shape[1] < K:
                raise RuntimeError(f"incompatible workload arrays: {workload}/{dataset}")
            details[dataset]["workloads"][workload] = {
                "alphas": file_info(alphas, True),
                "ground_truth": file_info(truth, True),
                "alpha_min": float(np.min(alpha_array)),
                "alpha_max": float(np.max(alpha_array)),
                "queries": int(len(alpha_array)),
                "ground_truth_k": int(truth_array.shape[1]),
            }

    write_json(run_root / "preflight.json", {
        "schema": 1,
        "created_utc": utc_now(),
        "binary": file_info(BINARY, True),
        "design": {
            "topologies": list(TOPOLOGIES),
            "uhg_configuration": {"kg": UHG_KG, "max_degree": 64},
            "datasets": list(datasets),
            "workloads": list(workloads),
            "points": list(points),
            "query_limit": query_limit,
            "repetitions": repetitions,
            "k": K,
            "threads": 1,
            "cpu": CPU,
            "method": "uhg (graph only)",
            "guided_entry": False,
            "hybrid_pruning": False,
            "max_hops": 0,
        },
        "inputs": details,
    })
    print(f"Preflight passed: {run_root}", flush=True)


def command(
        dataset: str, workload: str, topology: str,
        points: tuple[int, ...], query_limit: int) -> list[str]:
    package = package_path(dataset, topology)
    alphas, truth = workload_paths(workload, dataset)
    return [
        str(BINARY), str(DATA_ROOT / "hdf5" / f"{dataset}.hdf5"),
        "--mixed_alpha_file", str(alphas),
        "--mixed_gt_file", str(truth),
        "--alpha", "0.5", "-k", str(K),
        "--num_queries", str(query_limit), "--threads", "1",
        "--hybrid_index_path", str(package),
        "--index_dir", str(package.parent),
        "--build_alpha", "0.5", "--max_hops", "0", "--method", "uhg",
        "--disable_hybrid_pruning",
        "--search_points", ",".join(map(str, points)),
    ]


def run_logged(argv: list[str], log: Path) -> tuple[str, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    executed = ["taskset", "-c", str(CPU), *argv]
    print("START " + " ".join(executed), flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("# command: " + " ".join(executed) + "\n")
        stream.write("# utc: " + utc_now() + "\n")
        stream.flush()
        process = subprocess.Popen(
            executed, stdout=stream, stderr=subprocess.STDOUT, text=True)
        while True:
            try:
                return_code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                print(f"  running {elapsed:.0f}s log_bytes={log.stat().st_size}", flush=True)
    elapsed = time.monotonic() - started
    text = log.read_text(encoding="utf-8", errors="replace")
    if return_code:
        raise RuntimeError(
            f"command failed rc={return_code}:\n" + "\n".join(text.splitlines()[-80:]))
    print(f"DONE {log} ({elapsed:.1f}s)", flush=True)
    return text, elapsed


def field(block: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s+(.+)$", block, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing field {name}")
    return match.group(1).strip()


def parse_metrics(text: str, expected: tuple[int, ...]) -> list[dict]:
    blocks = BLOCK_PATTERN.findall(text)
    if len(blocks) != len(expected):
        raise RuntimeError(f"expected {len(expected)} result blocks, got {len(blocks)}")
    metrics = [{
        "point": int(field(block, "ef_search")),
        "recall": float(field(block, "Recall")),
        "qps": float(field(block, "QPS")),
        "search_seconds": float(field(block, "SearchSeconds")),
        "queries": int(field(block, "num_queries")),
        "qps_scope": field(block, "QPSScope"),
    } for block in blocks]
    if tuple(item["point"] for item in metrics) != expected:
        raise RuntimeError("search point order mismatch")
    return metrics


def measurement_path(
        run_root: Path, dataset: str, workload: str,
        topology: str, repetition: int) -> Path:
    return (run_root / "measurements" / dataset / workload / topology /
            f"repeat_{repetition}.json")


def measured_run(
        run_root: Path, dataset: str, workload: str, topology: str,
        repetition: int, points: tuple[int, ...], query_limit: int) -> None:
    output = measurement_path(run_root, dataset, workload, topology, repetition)
    if output.is_file():
        record = json.loads(output.read_text(encoding="utf-8"))
        if record.get("cpu") != CPU:
            raise RuntimeError(
                f"CPU mismatch: {output} used {record.get('cpu')}, "
                f"current run requested {CPU}; choose a new --run-root"
            )
        print(f"SKIP {dataset}/{workload}/{topology}/r{repetition}", flush=True)
        return
    ordered = points if repetition % 2 else tuple(reversed(points))
    argv = command(dataset, workload, topology, ordered, query_limit)
    log = (run_root / "logs" / dataset / workload / topology /
           f"repeat_{repetition}.log")
    text, elapsed = run_logged(argv, log)
    write_json(output, {
        "schema": 1,
        "created_utc": utc_now(),
        "dataset": dataset,
        "workload": workload,
        "topology": topology,
        "repetition": repetition,
        "points": list(ordered),
        "query_limit": query_limit,
        "cpu": CPU,
        "command": argv,
        "process_wall_seconds": elapsed,
        "metrics": parse_metrics(text, ordered),
        "log": file_info(log, True),
    })
