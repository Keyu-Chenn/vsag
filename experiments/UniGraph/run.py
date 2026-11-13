#!/usr/bin/env python3
"""Single entry point for the UniGraph paper experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
MAIN_RUN = ROOT / "runs" / "main"
ABLATION_RUN = ROOT / "runs" / "ablation"
ENV_PYTHON = ROOT / ".venv" / "bin" / "python"

MAIN_SUITES = ("mixed_alpha", "fixed_alpha")
ABLATION_SUITES = ("entry_point", "hybrid_pruning", "multi_weight")
SEARCH_STAGES = ("check", "plan", "run", "verify", "summarize")
COMMON_ABLATION_STAGES = ("check", "run", "summarize")


def execute(script: str, arguments: list[str]) -> None:
    python = ENV_PYTHON if ENV_PYTHON.is_file() else Path(sys.executable)
    command = [str(python), str(SCRIPTS / script), *arguments]
    subprocess.run(command, cwd=ROOT.parents[1], check=True)


def add_search_arguments(
        parser: argparse.ArgumentParser, suites: tuple[str, ...],
        stages: tuple[str, ...] = SEARCH_STAGES,
        allow_filters: bool = True) -> None:
    parser.add_argument("stage", nargs="?", choices=stages, default="check")
    parser.add_argument("--suite", choices=("all", *suites), default="all")
    if allow_filters:
        parser.add_argument("--datasets", help="comma-separated dataset subset")
        parser.add_argument("--methods", help="comma-separated method subset")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--cpu", type=int,
        help="logical CPU used for single-threaded timing",
    )


def search_arguments(
        options: argparse.Namespace, group: str, default_root: Path,
        suite: str | None = None) -> list[str]:
    arguments = [
        options.stage,
        "--group", group,
        "--suite", suite or options.suite,
        "--run-root", str((options.run_root or default_root).resolve()),
    ]
    if getattr(options, "datasets", None):
        arguments += ["--datasets", options.datasets]
    if getattr(options, "methods", None):
        arguments += ["--methods", options.methods]
    if options.cpu is not None:
        arguments += ["--cpu", str(options.cpu)]
    return arguments


def multi_weight_arguments(
        options: argparse.Namespace, default_root: Path) -> list[str]:
    arguments = [
        options.stage,
        "--run-root", str((options.run_root or default_root).resolve()),
    ]
    if options.cpu is not None:
        arguments += ["--cpu", str(options.cpu)]
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check", help="validate all experiment inputs")

    indexes = commands.add_parser("indexes", help="check or build experiment indexes")
    indexes.add_argument("stage", nargs="?", choices=("check", "plan", "build"),
                         default="check")
    indexes.add_argument("--datasets", default="all",
                         help="all or a comma-separated dataset subset")
    indexes.add_argument("--threads", type=int, default=32)

    main_experiments = commands.add_parser(
        "main", help="run mixed-alpha and fixed-alpha experiments")
    add_search_arguments(main_experiments, MAIN_SUITES)

    ablations = commands.add_parser(
        "ablation", help="run entry-point, hybrid-pruning, and multi-weight ablations")
    add_search_arguments(
        ablations,
        ABLATION_SUITES,
        stages=COMMON_ABLATION_STAGES,
        allow_filters=False,
    )

    plot = commands.add_parser("plot", help="render main and ablation figures")
    plot.add_argument("--group", choices=("all", "main", "ablation"), default="all")
    plot.add_argument("--x-min", type=float, default=0.0)
    plot.add_argument("--x-max", type=float, default=1.0)

    options = parser.parse_args()

    if options.command == "check":
        execute("main.py", [
            "check", "--group", "main", "--suite", "all",
            "--run-root", str(MAIN_RUN),
        ])
        execute("main.py", [
            "check", "--group", "ablation", "--suite", "all",
            "--run-root", str(ABLATION_RUN),
        ])
        execute("multi_weight.py", [
            "check", "--run-root", str(ABLATION_RUN),
        ])
        return

    if options.command == "indexes":
        execute("build_indexes.py", [
            options.stage,
            "--datasets", options.datasets,
            "--threads", str(options.threads),
        ])
        return

    if options.command == "main":
        execute("main.py", search_arguments(options, "main", MAIN_RUN))
        return

    if options.command == "ablation":
        includes_search = options.suite in {
            "all", "entry_point", "hybrid_pruning",
        }
        includes_multi_weight = options.suite in {"all", "multi_weight"}
        if includes_multi_weight and options.stage not in COMMON_ABLATION_STAGES:
            parser.error(
                f"ablation stage {options.stage!r} is unavailable for multi_weight; "
                f"choose one of {COMMON_ABLATION_STAGES}"
            )
        if includes_search:
            search_suite = "all" if options.suite == "all" else options.suite
            execute(
                "main.py",
                search_arguments(
                    options, "ablation", ABLATION_RUN, suite=search_suite),
            )
        if includes_multi_weight:
            execute(
                "multi_weight.py",
                multi_weight_arguments(options, ABLATION_RUN),
            )
        return

    groups = ("main", "ablation") if options.group == "all" else (options.group,)
    for group in groups:
        run_root = MAIN_RUN if group == "main" else ABLATION_RUN
        execute("plot.py", [
            "--run-root", str(run_root),
            "--suite", group,
            "--x-min", str(options.x_min),
            "--x-max", str(options.x_max),
        ])


if __name__ == "__main__":
    main()
