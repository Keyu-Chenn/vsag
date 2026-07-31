#!/usr/bin/env python3
"""Integrate UHG main-experiment results into a single main_results.txt.

Scans ``results/main/<dataset>/<method>/alpha_*.txt`` for every dataset, method
and alpha, parses the ``Recall:`` / ``QPS:`` metric lines (same regex used by
``plot_qps_recall_one_dataset.py``) and writes a consolidated, human-readable
report to ``results/results_integration/main_results.txt``.

The report contains:
  1. A header with generation metadata and a coverage matrix.
  2. A full fixed-width table of every parsed operating point.
  3. A TSV block (tab-separated) for easy downstream parsing.
  4. A best-point summary: for each (dataset, alpha, method) the operating
     point with the highest Recall (ties broken by highest QPS).

Usage:
    python3 integrate_main_results.py
    python3 integrate_main_results.py --results-root /path/to/results/main
    python3 integrate_main_results.py --output /path/to/main_results.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import csv
except ImportError:  # pragma: no cover
    csv = None


METHOD_ORDER = ["hnsw", "sindi", "hnsw_sindi", "fhg", "uhg"]
METHOD_LABELS = {
    "hnsw": "HNSW",
    "sindi": "SINDI",
    "hnsw_sindi": "HNSW+SINDI",
    "fhg": "FHG",
    "uhg": "UHG",
}
DATASET_ORDER = ["nq", "msmarco", "hotpotqa"]
DATASET_LABELS = {
    "nq": "NQ",
    "msmarco": "MS MARCO",
    "hotpotqa": "HotpotQA",
}
EXPECTED_ALPHAS = ["0.3", "0.4", "0.5", "0.6", "0.7"]

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(
    r"\b(?P<name>ef_search|bk|sindi_bk|dense_entry_bk)=(?P<value>[-+0-9.eE]+)"
)


@dataclass(frozen=True)
class Point:
    dataset: str
    method: str
    alpha: str
    param: str
    recall: float
    qps: float
    source: Path


@dataclass
class DatasetReport:
    dataset: str
    methods: dict[str, dict[str, list[Point]]] = field(default_factory=dict)


def alpha_sort_key(alpha: str) -> tuple[int, float | str]:
    try:
        return (0, float(alpha))
    except ValueError:
        return (1, alpha)


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def dataset_sort_key(dataset: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(dataset), dataset)
    except ValueError:
        return (len(DATASET_ORDER), dataset)


def parse_param(line: str) -> str:
    params = {match.group("name"): match.group("value") for match in PARAM_RE.finditer(line)}
    if "ef_search" in params:
        return f"ef={params['ef_search']}"
    if "bk" in params:
        return f"bk={params['bk']}"
    if "sindi_bk" in params:
        return f"sindi_bk={params['sindi_bk']}"
    if "dense_entry_bk" in params:
        return f"dense_bk={params['dense_entry_bk']}"
    return ""


def parse_result_file(dataset: str, method: str, path: Path) -> list[Point]:
    alpha = path.stem.removeprefix("alpha_")
    points: list[Point] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#"):
            continue
        metric_match = METRIC_RE.search(line)
        if metric_match is None:
            continue
        try:
            recall = float(metric_match.group("recall"))
            qps = float(metric_match.group("qps"))
        except ValueError:
            continue
        if recall < 0 or qps <= 0:
            continue
        points.append(
            Point(
                dataset=dataset,
                method=method,
                alpha=alpha,
                param=parse_param(line),
                recall=recall,
                qps=qps,
                source=path,
            )
        )
    return points


def load_all_points(result_root: Path) -> list[Point]:
    points: list[Point] = []
    for dataset_dir in sorted(result_root.iterdir(), key=lambda p: dataset_sort_key(p.name)):
        if not dataset_dir.is_dir() or dataset_dir.name in {"logs", "plots", "__pycache__"}:
            continue
        dataset = dataset_dir.name
        for method_dir in sorted(dataset_dir.iterdir(), key=lambda p: method_sort_key(p.name)):
            if not method_dir.is_dir() or method_dir.name == "plots":
                continue
            method = method_dir.name
            for result_file in sorted(method_dir.glob("alpha_*.txt")):
                points.extend(parse_result_file(dataset, method, result_file))
    return points


def group_points(points: list[Point]) -> dict[str, DatasetReport]:
    reports: dict[str, DatasetReport] = {}
    for point in points:
        report = reports.setdefault(point.dataset, DatasetReport(dataset=point.dataset))
        method_map = report.methods.setdefault(point.method, {})
        method_map.setdefault(point.alpha, []).append(point)
    return reports


def best_point(points: list[Point]) -> Point:
    return max(points, key=lambda p: (p.recall, p.qps))


def fmt_float(value: float, width: int = 10) -> str:
    text = f"{value:.6g}"
    if len(text) > width:
        text = f"{value:.4g}"
    return text.rjust(width)


def render_header(
    result_root: Path,
    output_path: Path,
    points: list[Point],
    reports: dict[str, DatasetReport],
) -> list[str]:
    datasets = sorted(reports.keys(), key=dataset_sort_key)
    methods_present = sorted(
        {m for report in reports.values() for m in report.methods.keys()},
        key=method_sort_key,
    )
    alphas_present = sorted(
        {p.alpha for p in points}, key=alpha_sort_key
    )
    lines: list[str] = []
    bar = "=" * 80
    lines.append(bar)
    lines.append("UHG Main Experiment Results (Integrated)")
    lines.append(bar)
    lines.append(f"Generated:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source root:  {result_root}")
    lines.append(f"Output file:  {output_path}")
    lines.append(f"Datasets:     {', '.join(DATASET_LABELS.get(d, d) for d in datasets)}")
    lines.append(
        f"Methods:      {', '.join(METHOD_LABELS.get(m, m) for m in methods_present)}"
    )
    lines.append(f"Alphas:       {', '.join(alphas_present)}")
    lines.append(f"Total points: {len(points)}")
    lines.append("")
    lines.append("Coverage matrix (number of operating points):")
    header = f"  {'dataset':<10} " + " ".join(
        f"{METHOD_LABELS.get(m, m):>12}" for m in methods_present
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for dataset in datasets:
        report = reports[dataset]
        cells = []
        for method in methods_present:
            count = sum(len(pts) for pts in report.methods.get(method, {}).values())
            cells.append(f"{count:>12}" if count else "          -")
        lines.append(f"  {DATASET_LABELS.get(dataset, dataset):<10} " + " ".join(cells))
    lines.append(bar)
    lines.append("")
    return lines


def render_full_table(points: list[Point]) -> list[str]:
    lines: list[str] = []
    lines.append("Section 1: Full Results Table (all operating points)")
    lines.append("=" * 80)
    lines.append("")
    header = (
        f"{'dataset':<10} {'alpha':<6} {'method':<12} "
        f"{'param':<10} {'recall':>12} {'qps':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for point in sorted(
        points,
        key=lambda p: (
            dataset_sort_key(p.dataset),
            alpha_sort_key(p.alpha),
            method_sort_key(p.method),
            -p.recall,
        ),
    ):
        lines.append(
            f"{DATASET_LABELS.get(point.dataset, point.dataset):<10} "
            f"{point.alpha:<6} "
            f"{METHOD_LABELS.get(point.method, point.method):<12} "
            f"{point.param:<10} "
            f"{fmt_float(point.recall, 12)} "
            f"{fmt_float(point.qps, 12)}"
        )
    lines.append("")
    return lines


def render_tsv_block(points: list[Point]) -> list[str]:
    lines: list[str] = []
    lines.append("Section 2: TSV Block (tab-separated, for downstream parsing)")
    lines.append("=" * 80)
    lines.append("# dataset\talpha\tmethod\tparam\trecall\tqps\tsource")
    for point in sorted(
        points,
        key=lambda p: (
            dataset_sort_key(p.dataset),
            alpha_sort_key(p.alpha),
            method_sort_key(p.method),
            -p.recall,
        ),
    ):
        lines.append(
            f"{point.dataset}\t{point.alpha}\t{point.method}\t"
            f"{point.param}\t{point.recall:.8g}\t{point.qps:.8g}\t{point.source}"
        )
    lines.append("")
    return lines


def render_best_summary(reports: dict[str, DatasetReport]) -> list[str]:
    lines: list[str] = []
    lines.append("Section 3: Best-point Summary")
    lines.append("(highest Recall per (dataset, alpha, method); ties broken by highest QPS)")
    lines.append("=" * 80)
    lines.append("")
    datasets = sorted(reports.keys(), key=dataset_sort_key)
    for dataset in datasets:
        report = reports[dataset]
        alphas = sorted(
            {a for am in report.methods.values() for a in am.keys()},
            key=alpha_sort_key,
        )
        methods = sorted(report.methods.keys(), key=method_sort_key)
        lines.append(f"--- {DATASET_LABELS.get(dataset, dataset)} ---")
        header = (
            f"  {'alpha':<6} {'method':<12} {'param':<10} "
            f"{'recall':>12} {'qps':>12}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for alpha in alphas:
            for method in methods:
                pts = report.methods.get(method, {}).get(alpha)
                if not pts:
                    continue
                best = best_point(pts)
                lines.append(
                    f"  {alpha:<6} "
                    f"{METHOD_LABELS.get(method, method):<12} "
                    f"{best.param:<10} "
                    f"{fmt_float(best.recall, 12)} "
                    f"{fmt_float(best.qps, 12)}"
                )
            lines.append("")
        lines.append("")
    return lines


def write_report(points: list[Point], output_path: Path, result_root: Path) -> None:
    reports = group_points(points)
    lines: list[str] = []
    lines.extend(render_header(result_root, output_path, points, reports))
    lines.extend(render_full_table(points))
    lines.extend(render_tsv_block(points))
    lines.extend(render_best_summary(reports))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate UHG main-experiment results from "
            "results/main/<dataset>/<method>/alpha_*.txt into a single "
            "main_results.txt report."
        )
    )
    parser.add_argument(
        "--results-root",
        default="/tbase-project/vsag/scripts/UHG/results/main",
        help="Root directory of the main experiment results.",
    )
    parser.add_argument(
        "--output",
        default="/tbase-project/vsag/scripts/UHG/results/results_integration/main_results.txt",
        help="Output path for the integrated main_results.txt.",
    )
    args = parser.parse_args()

    result_root = Path(args.results_root).resolve()
    if not result_root.is_dir():
        raise SystemExit(f"Results root not found: {result_root}")

    points = load_all_points(result_root)
    if not points:
        raise SystemExit(
            f"No Recall/QPS points found under {result_root}. "
            "Run experiments first or check result file format."
        )

    output_path = Path(args.output).resolve()
    write_report(points, output_path, result_root)

    print(f"Parsed points: {len(points)}")
    print(f"Datasets:      {len(group_points(points))}")
    print(f"Wrote report:  {output_path}")


if __name__ == "__main__":
    main()
