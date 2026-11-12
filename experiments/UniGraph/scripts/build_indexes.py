#!/usr/bin/env python3
"""Build every reusable index required by the UniGraph paper experiments."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Callable

import main_protocol as protocol


ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[1]
DATA_ROOT = ROOT / "data" / "hdf5"
INDEX_ROOT = ROOT / "index"
BIN_ROOT = ROOT / "bin"
LOG_ROOT = ROOT / "runs" / "index-build"
STAGING_ROOT = INDEX_ROOT / ".staging"

DEFAULT_THREADS = 32
DATASETS = protocol.DATASETS


@dataclasses.dataclass(frozen=True)
class Task:
    name: str
    output: Path
    dependencies: tuple[str, ...]
    command: Callable[[Path, int], tuple[list[str], Path]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def binary(name: str) -> Path:
    return BIN_ROOT / name


def outputs(dataset: str) -> dict[str, Path]:
    guide = INDEX_ROOT / "guide-indexes"
    deployable = INDEX_ROOT / "deployable-indexes" / dataset
    baseline = INDEX_ROOT / "baseline-indexes" / dataset
    return {
        "hnsw": guide / f"701_{dataset}_dense_hnsw.index",
        "sindi_reorder": guide / f"701_{dataset}_sparse_sindi.index",
        "sindi_no_reorder": guide / f"709_{dataset}_sindi.index",
        "fhg_graph": INDEX_ROOT / "graphs" / "fhg" / f"{dataset}_fhg_alpha_0_5.h5",
        "uhg_graph": INDEX_ROOT / "graphs" / "uhg" / f"{dataset}_uhg.h5",
        "fhg_package": deployable / "fhg.index",
        "uhg_package": deployable / "uhg.index",
        "naive_hybrid_hnsw": baseline / "naive_hybrid_hnsw.index",
        "optimized_hybrid_hnsw": baseline / "optimized_hybrid_hnsw.index",
        "ladr_graph": INDEX_ROOT / "ladr" / dataset / "graph.index",
    }


def tasks(dataset: str) -> list[Task]:
    source = DATA_ROOT / f"{dataset}.hdf5"
    target = outputs(dataset)
    guide = INDEX_ROOT / "guide-indexes"

    def hnsw(stage: Path, threads: int) -> tuple[list[str], Path]:
        command = [
            str(binary("708_uhg_exp8")), str(source),
            "--index_dir", str(stage),
            "--build-threads", str(threads),
            "--parallel-warmup", "10000",
            "--build_only",
        ]
        return command, stage / target["hnsw"].name

    def sindi(stage: Path, _threads: int, reorder: bool) -> tuple[list[str], Path]:
        command = [
            str(binary("709_uhg_exp9")), str(source),
            "--index_dir", str(stage),
            "--threads", "1",
            "--reorder", str(reorder).lower(),
            "--build_only",
        ]
        key = "sindi_reorder" if reorder else "sindi_no_reorder"
        return command, stage / target[key].name

    def fhg_graph(stage: Path, threads: int) -> tuple[list[str], Path]:
        command = [
            str(binary("704_uhg_exp4")), str(source),
            "-k", "64",
            "--bk", "100",
            "--dense_ef_search", "200",
            "--alpha", "0.5",
            "--query_prune_ratio", "0.9",
            "--output_dir", str(stage),
            "--index_dir", str(guide),
            "--threads", str(threads),
        ]
        return command, stage / target["fhg_graph"].name

    def uhg_graph(stage: Path, threads: int) -> tuple[list[str], Path]:
        generated = stage / target["uhg_graph"].name
        command = [
            str(binary("706_uhg_exp6")), str(source),
            "-k", "32",
            "--bk", "100",
            "--dense_ef_search", "200",
            "--alpha_step", "0.1",
            "--query_prune_ratio", "0.9",
            "--term_prune_ratio", "0",
            "--merge_max_degree", "64",
            "--skip_refine",
            "--output", str(generated),
            "--index_dir", str(guide),
            "--threads", str(threads),
        ]
        return command, generated

    def package(stage: Path, _threads: int, kind: str) -> tuple[list[str], Path]:
        generated = stage / f"{kind}.index"
        graph = target[f"{kind}_graph"]
        command = [
            str(binary("703_uhg_exp3")), str(source),
            "--hybrid_index_path", str(generated),
            "--index_dir", str(target[f"{kind}_package"].parent),
            "--graph_path", str(graph),
            "--gt_dir", str(ROOT / "data" / "ground_truth" / dataset),
            "--alpha", "0.5",
            "--build_alpha", "0.5",
            "-k", "100",
            "--ef_search", "100",
            "--sindi_bk", "100",
            "--num_queries", "1",
            "--threads", "1",
        ]
        if kind == "fhg":
            command += [
                "--method", "uhg",
                "--disable_sindi",
                "--disable_dense_entry",
                "--disable_hybrid_pruning",
            ]
        else:
            command += [
                "--method", "auto",
                "--hybrid_prune_scale", "0.3",
                "--sindi_query_prune_ratio", "0.5",
                "--sindi_term_prune_ratio", "0",
                "--dense_entry_bk", "100",
                "--dense_entry_ef_search", "100",
                "--sindi_index_path", str(target["sindi_no_reorder"]),
                "--dense_entry_hnsw_graph_path", str(target["hnsw"]),
            ]
        command += ["--build_only"]
        return command, generated

    def baseline(stage: Path, threads: int, method: str) -> tuple[list[str], Path]:
        generated = stage / f"{method}.index"
        command = [
            str(binary("uhg_paper_baselines")), "build",
            "--data", str(source),
            "--method", method,
            "--index", str(generated),
            "--build-threads", str(threads),
            "--m", "64",
            "--ef-construction", "200",
            "--ef-hybrid", "200",
            "--alpha-build", "0.5",
            "--seed", "42",
            "--parallel-warmup", "10000",
        ]
        return command, generated

    def ladr(stage: Path, threads: int) -> tuple[list[str], Path]:
        generated = stage / "graph.index"
        command = [
            str(binary("uhg_extra_baselines")), "build",
            "--method", "ladr_adaptive",
            "--data", str(source),
            "--index", str(generated),
            "--build-threads", str(threads),
            "--hnsw-index", str(target["hnsw"]),
        ]
        return command, generated

    return [
        Task("hnsw", target["hnsw"], (), hnsw),
        Task(
            "sindi_reorder", target["sindi_reorder"], (),
            lambda stage, threads: sindi(stage, threads, True),
        ),
        Task(
            "sindi_no_reorder", target["sindi_no_reorder"], (),
            lambda stage, threads: sindi(stage, threads, False),
        ),
        Task(
            "fhg_graph", target["fhg_graph"], ("hnsw", "sindi_reorder"),
            fhg_graph,
        ),
        Task(
            "uhg_graph", target["uhg_graph"], ("hnsw", "sindi_reorder"),
            uhg_graph,
        ),
        Task(
            "fhg_package", target["fhg_package"], ("fhg_graph",),
            lambda stage, threads: package(stage, threads, "fhg"),
        ),
        Task(
            "uhg_package", target["uhg_package"],
            ("uhg_graph", "hnsw", "sindi_no_reorder"),
            lambda stage, threads: package(stage, threads, "uhg"),
        ),
        Task(
            "naive_hybrid_hnsw", target["naive_hybrid_hnsw"], (),
            lambda stage, threads: baseline(stage, threads, "naive_hybrid_hnsw"),
        ),
        Task(
            "optimized_hybrid_hnsw", target["optimized_hybrid_hnsw"], (),
            lambda stage, threads: baseline(
                stage, threads, "optimized_hybrid_hnsw"),
        ),
        Task("ladr_graph", target["ladr_graph"], ("hnsw",), ladr),
    ]


def is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def required_binaries() -> tuple[Path, ...]:
    names = (
        "703_uhg_exp3", "704_uhg_exp4", "706_uhg_exp6",
        "708_uhg_exp8", "709_uhg_exp9",
        "uhg_paper_baselines", "uhg_extra_baselines",
    )
    return tuple(binary(name) for name in names)


def prerequisites(datasets: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for executable in required_binaries():
        if not executable.is_file() or not os.access(executable, os.X_OK):
            missing.append(str(executable))
    for dataset in datasets:
        source = DATA_ROOT / f"{dataset}.hdf5"
        if not is_complete(source):
            missing.append(str(source))
    return missing


def status(datasets: tuple[str, ...], threads: int, include_commands: bool) -> dict:
    rows = []
    for dataset in datasets:
        for task in tasks(dataset):
            row: dict[str, object] = {
                "dataset": dataset,
                "task": task.name,
                "complete": is_complete(task.output),
                "output": str(task.output),
                "dependencies": list(task.dependencies),
            }
            if include_commands:
                preview = STAGING_ROOT / f"{dataset}-{task.name}"
                command, generated = task.command(preview, threads)
                row["command"] = command
                row["staged_output"] = str(generated)
            rows.append(row)
    missing = prerequisites(datasets)
    return {
        "passed": not missing,
        "datasets": list(datasets),
        "threads": threads,
        "missing_prerequisites": missing,
        "complete_tasks": sum(bool(row["complete"]) for row in rows),
        "total_tasks": len(rows),
        "tasks": rows,
    }


def next_log(dataset: str, task: str) -> Path:
    directory = LOG_ROOT / dataset
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task}.log"
    attempt = 1
    while path.exists():
        attempt += 1
        path = directory / f"{task}.attempt-{attempt}.log"
    return path


def run_command(command: list[str], log: Path, threads: int) -> None:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    library = str(REPO / "build-release" / "src")
    current_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        f"{library}:{current_library_path}" if current_library_path else library
    )
    display = shlex.join(command)
    print(f"START {display}", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        stream.write(f"# command: {display}\n# utc: {utc_now()}\n")
        stream.flush()
        process = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        raise RuntimeError(
            f"command failed with exit code {process.returncode}; inspect {log}"
        )


def build_task(dataset: str, task: Task, completed: dict[str, Path], threads: int) -> None:
    if is_complete(task.output):
        print(f"SKIP {dataset}/{task.name}: {task.output}", flush=True)
        completed[task.name] = task.output
        return
    if task.output.exists() and not task.output.is_file():
        raise RuntimeError(f"output path exists but is not a file: {task.output}")
    for dependency in task.dependencies:
        path = completed.get(dependency)
        if path is None or not is_complete(path):
            raise RuntimeError(
                f"{dataset}/{task.name} is missing dependency {dependency}"
            )

    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f"{dataset}-{task.name}-", dir=STAGING_ROOT) as temporary:
        stage = Path(temporary)
        command, generated = task.command(stage, threads)
        log = next_log(dataset, task.name)
        environment_threads = 1 if task.name.startswith("sindi_") else threads
        run_command(command, log, environment_threads)
        if not is_complete(generated):
            raise RuntimeError(
                f"{dataset}/{task.name} did not produce a non-empty file: {generated}"
            )
        task.output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(generated, task.output)
    completed[task.name] = task.output
    print(f"DONE {dataset}/{task.name}: {task.output}", flush=True)


def build(datasets: tuple[str, ...], threads: int) -> None:
    missing = prerequisites(datasets)
    if missing:
        raise FileNotFoundError(
            "missing build prerequisites:\n" + "\n".join(missing)
        )
    for dataset in datasets:
        completed: dict[str, Path] = {}
        for task in tasks(dataset):
            build_task(dataset, task, completed, threads)
    final = status(datasets, threads, False)
    if final["complete_tasks"] != final["total_tasks"]:
        raise RuntimeError("index build ended with incomplete tasks")
    print(
        f"All {final['total_tasks']} index tasks are complete for "
        f"{len(datasets)} dataset(s).",
        flush=True,
    )


def parse_datasets(value: str) -> tuple[str, ...]:
    if value.strip() == "all":
        return DATASETS
    selected = tuple(dict.fromkeys(
        item.strip() for item in value.split(",") if item.strip()
    ))
    unknown = set(selected) - set(DATASETS)
    if not selected or unknown:
        raise argparse.ArgumentTypeError(
            f"expected all or a comma-separated subset of {','.join(DATASETS)}"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "plan", "build"))
    parser.add_argument(
        "--datasets", type=parse_datasets, default=DATASETS,
        help="all or a comma-separated subset (default: all)",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help=f"construction thread budget (default: {DEFAULT_THREADS})",
    )
    options = parser.parse_args()
    if not 1 <= options.threads <= 96:
        parser.error("--threads must be between 1 and 96")

    datasets = tuple(options.datasets)
    if options.stage == "build":
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        lock_path = LOG_ROOT / "execution.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"another index build holds {lock_path}"
                ) from error
            build(datasets, options.threads)
        return
    report = status(datasets, options.threads, options.stage == "plan")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
