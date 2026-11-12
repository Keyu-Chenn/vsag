#!/usr/bin/env python3
"""Run the final UniGraph search experiments used by Figures 5, 6, 8, and 9."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import main_protocol as protocol


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs" / "main"

DATASETS = protocol.DATASETS
FIXED_DATASETS = protocol.FIXED_DATASETS
SUITES = ("mixed_alpha", "fixed_alpha", "ablation1", "ablation2")
METHODS = (
    "hnsw",
    "sindi",
    "hnsw_sindi",
    "naive_hybrid_hnsw",
    "optimized_hybrid_hnsw",
    "uhg",
    "ladr_adaptive_hybrid",
)
METHOD_FILENAMES = {
    "hnsw": "hnsw",
    "sindi": "sindi",
    "hnsw_sindi": "two_route",
    "naive_hybrid_hnsw": "naive_hybrid",
    "optimized_hybrid_hnsw": "opt_hybrid",
    "uhg": "unigraph",
    "ladr_adaptive_hybrid": "ladr",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def make_job(suite: str, dataset: str, method: str, k: int,
             alpha: str = "mixed") -> protocol.Job:
    if method == "optimized_hybrid_hnsw":
        return protocol.Job(
            suite, dataset, method, k, alpha, "no_pruning", 1.0
        )
    return protocol.Job(suite, dataset, method, k, alpha)


def all_jobs(suite: str | None = None) -> list[protocol.Job]:
    selected: list[protocol.Job] = []
    suites = SUITES if suite is None else (suite,)
    for name in suites:
        if name == "mixed_alpha":
            selected.extend(
                make_job(name, dataset, method, k)
                for dataset in DATASETS
                for k in (50, 100, 200)
                for method in METHODS
            )
        elif name == "fixed_alpha":
            selected.extend(
                make_job(name, dataset, method, 100, alpha)
                for dataset in FIXED_DATASETS
                for alpha in ("0.3", "0.4", "0.5", "0.6", "0.7")
                for method in METHODS
            )
        elif name == "ablation1":
            selected.extend(
                protocol.Job(name, dataset, "uhg", 100, alpha, variant)
                for dataset in DATASETS
                for alpha in ("0.3", "0.7")
                for variant in ("unguided", "guided")
            )
        elif name == "ablation2":
            selected.extend(
                protocol.Job(name, dataset, "uhg", 100, "mixed", variant)
                for dataset in DATASETS
                for variant in ("prune_no_prune", "prune_0.3")
            )
        else:
            raise ValueError(f"unknown suite: {name}")
    return selected


def points(job: protocol.Job) -> tuple[int, ...]:
    if job.method == "ladr_adaptive_hybrid":
        return (1, 2, 5, 10, 20, 50, 100, 200)
    return job.points


def configure(run_root: Path, cpu: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=run_root.resolve(),
        data_dir=ROOT / "data",
        original_index_dir=ROOT / "index" / "guide-indexes",
        deployable_index_dir=ROOT / "index" / "deployable-indexes",
        baseline_index_root=ROOT / "index" / "baseline-indexes",
        ladr_index_root=ROOT / "index" / "ladr",
        original_bin_dir=ROOT / "bin",
        baseline_bin=ROOT / "bin" / "uhg_paper_baselines",
        extra_bin=ROOT / "bin" / "uhg_extra_baselines",
        result_root=run_root.resolve() / "results",
        log_root=run_root.resolve() / "logs",
        h5ls=REPO / "build-release" / "hdf5" / "install" / "bin" / "h5ls",
        search_cpu=protocol.resolve_cpu(cpu),
        build_threads=32,
        tau_dense=0.8,
        tau_hybrid=1.0,
    )


def gt_paths(args: SimpleNamespace, job: protocol.Job) -> tuple[Path | None, Path]:
    return protocol.gt_paths(job, args.data_dir)


def ladr_index(args: SimpleNamespace, dataset: str) -> Path:
    return args.ladr_index_root / dataset / "graph.index"


def command_for(args: SimpleNamespace, job: protocol.Job) -> list[str]:
    sequence = points(job)
    alphas, truth = gt_paths(args, job)
    if job.method == "ladr_adaptive_hybrid":
        command = [
            str(args.extra_bin), "search",
            "--method", "ladr_adaptive_hybrid",
            "--data", str(args.data_dir / "hdf5" / f"{job.dataset}.hdf5"),
            "--index", str(ladr_index(args, job.dataset)),
            "--gt", str(truth),
            "--k", str(job.k),
            "--points", protocol.point_list_argument(sequence),
        ]
        command += ["--alphas", str(alphas)] if alphas else ["--alpha", job.alpha]
        command += [
            "--sindi-index",
            str(args.original_index_dir / f"701_{job.dataset}_sparse_sindi.index"),
            "--seeds", "200", "--neighbors", "128",
            "--query-prune-ratio", "0.5",
        ]
        return command
    if job.method in protocol.ADDED_METHODS:
        return protocol.baseline_search_command(args, job, sequence)
    cache = protocol.index_path(args, job) if job.cache_kind else None
    command = protocol.original_search_command(
        args,
        job,
        300 if job.suite.startswith("ablation") else sequence,
        cache,
    )
    if job.method == "sindi":
        command += ["--reorder", "true"]
    return command


def result_path(args: SimpleNamespace, job: protocol.Job) -> Path:
    if job.suite == "mixed_alpha":
        return (args.result_root / job.suite / job.dataset / f"k_{job.k}" /
                f"{METHOD_FILENAMES[job.method]}.json")
    if job.suite == "fixed_alpha":
        return (args.result_root / job.suite / job.dataset / f"alpha_{job.alpha}" /
                f"{METHOD_FILENAMES[job.method]}.json")
    if job.suite == "ablation1":
        return (args.result_root / job.suite / job.dataset / f"alpha_{job.alpha}" /
                f"{job.variant}.json")
    return args.result_root / job.suite / job.dataset / f"{job.variant}.json"


def log_path(args: SimpleNamespace, job: protocol.Job) -> Path:
    return args.log_root / result_path(args, job).relative_to(args.result_root).with_suffix(".log")


def parse_metrics(text: str, job: protocol.Job,
                  sequence: tuple[int, ...] | None = None) -> list[dict]:
    sequence = points(job) if sequence is None else sequence
    metrics = protocol.parse_search_metrics(
        text,
        job.method not in protocol.ORIGINAL_METHODS,
        protocol.QUERY_COUNT[job.dataset],
        sequence,
        job.method,
    )
    if job.method not in protocol.ORIGINAL_METHODS:
        counts = [metric["returned_counts"] for metric in metrics]
    else:
        counts = [
            json.loads(line.split(": ", 1)[1])
            for line in text.splitlines()
            if line.startswith("ReturnedCounts: ")
        ]
    if len(counts) != len(metrics):
        raise ValueError("missing per-query returned counts")
    for metric, returned in zip(metrics, counts):
        if len(returned) != protocol.QUERY_COUNT[job.dataset]:
            raise ValueError("wrong number of per-query returned counts")
        if any(type(value) is not int or not 0 <= value <= job.k for value in returned):
            raise ValueError("invalid returned count")
        if not math.isfinite(metric["recall"]) or not 0 <= metric["recall"] <= 1:
            raise ValueError("invalid recall")
        metric.update(
            returned_counts=returned,
            short_query_count=sum(value < job.k for value in returned),
            total_missing=sum(job.k - value for value in returned),
        )
    return metrics


def required_index(args: SimpleNamespace, job: protocol.Job) -> list[Path]:
    if job.method == "ladr_adaptive_hybrid":
        return [
            ladr_index(args, job.dataset),
            args.original_index_dir / f"701_{job.dataset}_sparse_sindi.index",
        ]
    if job.method == "hnsw_sindi":
        return [
            protocol.index_path(args, job),
            args.original_index_dir / f"701_{job.dataset}_sparse_sindi.index",
        ]
    return [protocol.index_path(args, job)]


def select_jobs(suite: str, datasets: set[str] | None,
                methods: set[str] | None) -> list[protocol.Job]:
    selected = all_jobs(None if suite == "all" else suite)
    if datasets:
        selected = [job for job in selected if job.dataset in datasets]
    if methods:
        selected = [job for job in selected if job.method in methods]
    if not selected:
        raise ValueError("selection contains no jobs")
    return selected


def check(args: SimpleNamespace, selected: list[protocol.Job]) -> dict:
    missing: set[str] = set()
    invalid: list[str] = []
    binaries = {
        args.original_bin_dir / "701_uhg_exp1",
        args.original_bin_dir / "703_uhg_exp3",
        args.original_bin_dir / "708_uhg_exp8",
        args.original_bin_dir / "709_uhg_exp9",
        args.baseline_bin,
        args.extra_bin,
    }
    for binary in binaries:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            missing.add(str(binary))
    checked_arrays: set[Path] = set()
    for job in selected:
        data = args.data_dir / "hdf5" / f"{job.dataset}.hdf5"
        if not data.is_file():
            missing.add(str(data))
        alphas, truth = gt_paths(args, job)
        for path, expected_dtype in ((truth, "i8"), (alphas, "f4")):
            if path is None or path in checked_arrays:
                continue
            checked_arrays.add(path)
            if not path.is_file():
                missing.add(str(path))
                continue
            try:
                dtype, shape = protocol.npy_info(path)
                expected_queries = protocol.QUERY_COUNT[job.dataset]
                if expected_dtype not in dtype or shape[0] != expected_queries:
                    invalid.append(f"{path}: dtype={dtype}, shape={shape}")
                if path == truth and (len(shape) != 2 or shape[1] < job.k):
                    invalid.append(f"{path}: insufficient ground-truth width {shape}")
            except Exception as error:
                invalid.append(f"{path}: {error}")
        for index in required_index(args, job):
            if not index.is_file() or index.stat().st_size == 0:
                missing.add(str(index))
    report = {
        "passed": not missing and not invalid,
        "cpu": args.search_cpu,
        "jobs": len(selected),
        "parameter_points": sum(len(points(job)) for job in selected),
        "suites": sorted({job.suite for job in selected}),
        "datasets": sorted({job.dataset for job in selected}),
        "methods": sorted({job.method for job in selected}),
        "missing": sorted(missing),
        "invalid": invalid,
    }
    return report


def run_logged(command: list[str], destination: Path, cpu: int,
               threads: int = 1) -> tuple[str, float]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = destination
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = destination.with_name(
            f"{destination.stem}.attempt-{attempt}{destination.suffix}"
        )
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["LD_LIBRARY_PATH"] = str(REPO / "build-release" / "src")
    started = time.monotonic()
    displayed = protocol.command_display(command, (cpu,))
    print(f"START {displayed}", flush=True)
    with candidate.open("x", encoding="utf-8") as stream:
        stream.write(f"# command: {displayed}\n# utc: {utc_now()}\n")
        stream.flush()
        process = subprocess.run(
            ["taskset", "-c", str(cpu), *command],
            cwd=REPO,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - started
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}); inspect {candidate}")
    print(f"DONE {candidate} ({elapsed:.1f}s)", flush=True)
    return candidate.read_text(encoding="utf-8", errors="replace"), elapsed


def verify_result(args: SimpleNamespace, job: protocol.Job) -> dict:
    path = result_path(args, job)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["job"] != dataclasses.asdict(job):
        raise ValueError(f"job mismatch: {path}")
    if record.get("cpu") != args.search_cpu:
        raise ValueError(
            f"CPU mismatch: {path} used {record.get('cpu')}, "
            f"current run requested {args.search_cpu}"
        )
    if record["command"] != command_for(args, job):
        raise ValueError(f"command mismatch: {path}")
    log = Path(record["raw_log"])
    if not log.is_file():
        raise ValueError(f"missing raw log: {log}")
    if parse_metrics(log.read_text(encoding="utf-8", errors="replace"), job) != record["metrics"]:
        raise ValueError(f"metric/log mismatch: {path}")
    return record


def run_one(args: SimpleNamespace, job: protocol.Job) -> None:
    output = result_path(args, job)
    if output.exists():
        verify_result(args, job)
        print(f"SKIP {output}", flush=True)
        return
    command = command_for(args, job)
    text, elapsed = run_logged(command, log_path(args, job), args.search_cpu)
    log = sorted(
        log_path(args, job).parent.glob(log_path(args, job).stem + "*.log"),
        key=lambda path: path.stat().st_mtime_ns,
    )[-1]
    record = {
        "schema": 1,
        "created_utc": utc_now(),
        "job": dataclasses.asdict(job),
        "command": command,
        "metrics": parse_metrics(text, job),
        "warmup_runs": 0,
        "repetitions": 1,
        "cpu": args.search_cpu,
        "wall_seconds": elapsed,
        "raw_log": str(log.resolve()),
    }
    protocol.atomic_json(output, record)


def summarize(args: SimpleNamespace) -> list[dict]:
    fields = (
        "suite", "dataset", "method", "k", "alpha", "variant", "point",
        "recall", "qps_median", "qps_min", "qps_max",
        "search_seconds_median", "repetitions", "short_query_count",
        "total_missing", "result",
    )
    rows: list[dict] = []
    for job in all_jobs():
        path = result_path(args, job)
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        for metric in record["metrics"]:
            row = dataclasses.asdict(job)
            row.update(
                point=metric["point"],
                recall=metric["recall"],
                qps_median=metric["qps"],
                qps_min=metric["qps"],
                qps_max=metric["qps"],
                search_seconds_median=metric["search_seconds"],
                repetitions=1,
                short_query_count=metric["short_query_count"],
                total_missing=metric["total_missing"],
                result=str(path.resolve()),
            )
            rows.append(row)
    args.run_root.mkdir(parents=True, exist_ok=True)
    temporary = args.run_root / "summary.tsv.partial"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.run_root / "summary.tsv")
    return rows


def verify(args: SimpleNamespace, selected: list[protocol.Job]) -> dict:
    missing: list[str] = []
    errors: list[dict] = []
    for job in selected:
        path = result_path(args, job)
        if not path.is_file():
            missing.append(str(path.relative_to(args.run_root)))
            continue
        try:
            verify_result(args, job)
        except Exception as error:
            errors.append({"result": str(path), "error": str(error)})
    return {
        "passed": not missing and not errors,
        "expected_jobs": len(selected),
        "completed_jobs": len(selected) - len(missing) - len(errors),
        "missing": missing,
        "errors": errors,
    }


def parse_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "plan", "run", "verify", "summarize"))
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    parser.add_argument("--datasets", help="comma-separated dataset names")
    parser.add_argument("--methods", help="comma-separated method names")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--cpu", type=int,
        help="logical CPU used for single-threaded timing (default: first allowed CPU)",
    )
    options = parser.parse_args()
    datasets = parse_set(options.datasets)
    methods = parse_set(options.methods)
    unknown_datasets = (datasets or set()) - set(DATASETS)
    unknown_methods = (methods or set()) - set(METHODS)
    if unknown_datasets or unknown_methods:
        parser.error(
            f"unknown datasets={sorted(unknown_datasets)}, methods={sorted(unknown_methods)}"
        )
    try:
        args = configure(options.run_root, options.cpu)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    selected = select_jobs(options.suite, datasets, methods)
    if options.stage in {"check", "plan"}:
        report = check(args, selected)
        if options.stage == "plan":
            report["commands"] = [
                {
                    "result": str(result_path(args, job)),
                    "command": command_for(args, job),
                }
                for job in selected
            ]
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if report["passed"] else 1)
    args.run_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.run_root / "execution.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if options.stage == "summarize":
            rows = summarize(args)
            print(f"wrote {len(rows)} rows to {args.run_root / 'summary.tsv'}")
            return
        if options.stage == "verify":
            report = verify(args, selected)
            protocol.atomic_json(args.run_root / "audit.json", report)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            raise SystemExit(0 if report["passed"] else 1)
        report = check(args, selected)
        if not report["passed"]:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            raise SystemExit("preflight failed")
        for job in selected:
            run_one(args, job)
            summarize(args)
        audit = verify(args, selected)
        protocol.atomic_json(args.run_root / "audit.json", audit)
        if not audit["passed"]:
            raise SystemExit("run completed with audit failures")


if __name__ == "__main__":
    main()
