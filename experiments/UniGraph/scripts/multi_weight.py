#!/usr/bin/env python3
"""Measure kg=32 UHG/FHG QPS at Recall@100 targets 0.90/0.95/0.99."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import multi_weight_protocol as base


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
RUN_ROOT = EXPERIMENT_ROOT / "runs" / "ablation"
WORKLOAD = "mixed_0_1"
TARGETS = (0.90, 0.95, 0.99)
REPETITIONS = 3
PROBE_QUERY_LIMIT = 200
PROBE_ROUNDS = (
    ("probe-to-50k", (100, 200, 300, 500, 700, 1000, 1500, 2000, 3000,
                      5000, 7500, 10000)),
    ("probe-15k-30k", (15000, 20000, 30000)),
    ("probe-50k-100k", (50000, 75000, 100000)),
    ("probe-150k-200k", (150000, 200000)),
)


def load_metrics(root: Path, dataset: str, topology: str) -> list[dict]:
    path = base.measurement_path(root, dataset, WORKLOAD, topology, 1)
    record = json.loads(path.read_text(encoding="utf-8"))
    return sorted(record["metrics"], key=lambda row: row["point"])


def target_is_covered(metrics: list[dict], target: float) -> bool:
    if not metrics:
        return False
    recalls = [row["recall"] for row in metrics]
    return min(recalls) <= target <= max(recalls)


def run_probe() -> dict[tuple[str, str], list[dict]]:
    merged = {(dataset, topology): []
              for dataset in base.DATASETS for topology in base.TOPOLOGIES}
    for round_name, points in PROBE_ROUNDS:
        pending = [
            (dataset, topology)
            for dataset in base.DATASETS
            for topology in base.TOPOLOGIES
            if not target_is_covered(merged[(dataset, topology)], max(TARGETS))
            and (not merged[(dataset, topology)] or
                 max(row["recall"] for row in merged[(dataset, topology)]) < max(TARGETS))
        ]
        if not pending:
            break
        round_root = RUN_ROOT / round_name
        datasets = tuple(dataset for dataset in base.DATASETS
                         if any(item[0] == dataset for item in pending))
        base.preflight(round_root, datasets, (WORKLOAD,), points,
                       PROBE_QUERY_LIMIT, 1)
        for dataset_index, dataset in enumerate(datasets):
            order = list(base.TOPOLOGIES)
            if dataset_index % 2:
                order.reverse()
            for topology in order:
                if (dataset, topology) not in pending:
                    continue
                base.measured_run(round_root, dataset, WORKLOAD, topology, 1,
                                  points, PROBE_QUERY_LIMIT)
        for dataset, topology in pending:
            merged[(dataset, topology)].extend(
                load_metrics(round_root, dataset, topology))
            merged[(dataset, topology)].sort(key=lambda row: row["point"])
    return merged


def select_points(metrics: list[dict], target: float) -> set[int]:
    ordered = sorted(metrics, key=lambda row: row["point"])
    for high_index, row in enumerate(ordered):
        if row["recall"] >= target:
            low_index = max(0, high_index - 1)
            start = max(0, low_index - 1)
            stop = min(len(ordered), high_index + 2)
            result = {ordered[index]["point"] for index in range(start, stop)}
            # A 200-query probe can barely cross 0.99 while the full 1000-query
            # recall lands just below it.  Keep one upper guard point in that
            # boundary case so interpolation never relies on extrapolation.
            if high_index == len(ordered) - 1:
                result.add(int(row["point"] * 1.5))
            return result
    return {row["point"] for row in ordered[-3:]}


def choose_full_points(
        probe: dict[tuple[str, str], list[dict]]) -> dict[str, dict[str, tuple[int, ...]]]:
    selected: dict[str, dict[str, tuple[int, ...]]] = {}
    coverage = []
    for dataset in base.DATASETS:
        selected[dataset] = {}
        for topology in base.TOPOLOGIES:
            points: set[int] = set()
            for target in TARGETS:
                metrics = probe[(dataset, topology)]
                covered = target_is_covered(metrics, target)
                coverage.append({
                    "dataset": dataset,
                    "topology": topology,
                    "target_recall": target,
                    "covered": covered,
                    "max_probe_recall": max(row["recall"] for row in metrics),
                    "max_probe_point": max(row["point"] for row in metrics),
                })
                points.update(select_points(metrics, target))
            selected[dataset][topology] = tuple(sorted(points))
    base.write_json(RUN_ROOT / "probe_coverage.json", coverage)
    base.write_json(RUN_ROOT / "selected_points.json", selected)
    return selected


def measured_bracket_points(metrics: list[dict], target: float) -> set[int]:
    ordered = sorted(metrics, key=lambda row: row["recall"])
    for left, right in zip(ordered, ordered[1:]):
        if left["recall"] <= target <= right["recall"]:
            return {left["point"], right["point"]}
    raise RuntimeError(
        f"full-query scan does not bracket recall {target}: "
        f"{min(row['recall'] for row in ordered):.6f}-"
        f"{max(row['recall'] for row in ordered):.6f}")


def run_full(
        selected: dict[str, dict[str, tuple[int, ...]]]
        ) -> dict[str, dict[str, tuple[int, ...]]]:
    union_points = tuple(sorted({
        point
        for dataset_points in selected.values()
        for topology_points in dataset_points.values()
        for point in topology_points
    }))
    base.preflight(RUN_ROOT, base.DATASETS, (WORKLOAD,), union_points, -1, REPETITIONS)
    manifest = json.loads((RUN_ROOT / "preflight.json").read_text(encoding="utf-8"))
    manifest["design"]["target_recalls"] = list(TARGETS)
    manifest["design"]["full_scan_points"] = {
        dataset: {topology: list(points) for topology, points in values.items()}
        for dataset, values in selected.items()
    }
    base.write_json(RUN_ROOT / "preflight.json", manifest)

    timing_points: dict[str, dict[str, tuple[int, ...]]] = {}
    for dataset_index, dataset in enumerate(base.DATASETS):
        first_order = list(base.TOPOLOGIES)
        if (dataset_index + 1) % 2:
            first_order.reverse()
        for topology in first_order:
            base.measured_run(RUN_ROOT, dataset, WORKLOAD, topology, 1,
                              selected[dataset][topology], -1)

        timing_points[dataset] = {}
        for topology in base.TOPOLOGIES:
            first = json.loads(base.measurement_path(
                RUN_ROOT, dataset, WORKLOAD, topology, 1
            ).read_text(encoding="utf-8"))
            points: set[int] = set()
            for target in TARGETS:
                points.update(measured_bracket_points(first["metrics"], target))
            timing_points[dataset][topology] = tuple(sorted(points))

        for repetition in range(2, REPETITIONS + 1):
            order = list(base.TOPOLOGIES)
            if (dataset_index + repetition) % 2:
                order.reverse()
            for topology in order:
                base.measured_run(RUN_ROOT, dataset, WORKLOAD, topology,
                                  repetition, timing_points[dataset][topology], -1)

    manifest = json.loads((RUN_ROOT / "preflight.json").read_text(encoding="utf-8"))
    manifest["design"]["repeated_timing_points"] = {
        dataset: {topology: list(points) for topology, points in values.items()}
        for dataset, values in timing_points.items()
    }
    base.write_json(RUN_ROOT / "preflight.json", manifest)
    base.write_json(RUN_ROOT / "timing_points.json", timing_points)
    return timing_points


def interpolate_qps(curve: list[dict], target: float) -> tuple[float, tuple[int, int]] | None:
    ordered = sorted(curve, key=lambda row: row["recall"])
    for left, right in zip(ordered, ordered[1:]):
        if left["recall"] <= target <= right["recall"]:
            if right["recall"] == left["recall"]:
                return max(left["qps_median"], right["qps_median"]), (
                    left["point"], right["point"])
            fraction = ((target - left["recall"]) /
                        (right["recall"] - left["recall"]))
            qps = left["qps_median"] + fraction * (
                right["qps_median"] - left["qps_median"])
            return qps, (left["point"], right["point"])
    return None


def summarize(selected: dict[str, dict[str, tuple[int, ...]]]) -> None:
    curves = []
    for dataset in base.DATASETS:
        for topology in base.TOPOLOGIES:
            points = selected[dataset][topology]
            by_point = {point: [] for point in points}
            for repetition in range(1, REPETITIONS + 1):
                record = json.loads(base.measurement_path(
                    RUN_ROOT, dataset, WORKLOAD, topology, repetition
                ).read_text(encoding="utf-8"))
                for metric in record["metrics"]:
                    if metric["point"] in by_point:
                        by_point[metric["point"]].append(metric)
            for point in points:
                metrics = by_point[point]
                curves.append({
                    "dataset": dataset,
                    "workload": WORKLOAD,
                    "topology": topology,
                    "point": point,
                    "recall": statistics.median(row["recall"] for row in metrics),
                    "qps_median": statistics.median(row["qps"] for row in metrics),
                    "qps_min": min(row["qps"] for row in metrics),
                    "qps_max": max(row["qps"] for row in metrics),
                })

    summary = RUN_ROOT / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    with (summary / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)

    targets = []
    for dataset in base.DATASETS:
        dataset_rows = [row for row in curves if row["dataset"] == dataset]
        topology_curves = {
            topology: [row for row in dataset_rows if row["topology"] == topology]
            for topology in base.TOPOLOGIES
        }
        for target in TARGETS:
            fhg = interpolate_qps(topology_curves["fhg"], target)
            uhg = interpolate_qps(topology_curves["uhg"], target)
            targets.append({
                "dataset": dataset,
                "workload": WORKLOAD,
                "target_recall": target,
                "fhg_recall": target if fhg else None,
                "uhg_recall": target if uhg else None,
                "fhg_qps": fhg[0] if fhg else None,
                "uhg_qps": uhg[0] if uhg else None,
                "qps_gain_pct": (100.0 * (uhg[0] / fhg[0] - 1.0)
                                 if fhg and uhg else None),
                "fhg_bracket_ef": (f"{fhg[1][0]}-{fhg[1][1]}" if fhg else None),
                "uhg_bracket_ef": (f"{uhg[1][0]}-{uhg[1][1]}" if uhg else None),
            })
    with (summary / "target_recall_qps.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)

    lines = [
        "# kg=32 UHG versus FHG at fixed Recall targets",
        "",
        "Graph-only search; alpha uniformly covers [0,1]; three timing repetitions;",
        "QPS is interpolated only within each measured curve.",
        "",
        "| Dataset | Recall | FHG QPS | UHG QPS | QPS gain | FHG ef bracket | UHG ef bracket |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in targets:
        if row["qps_gain_pct"] is None:
            lines.append(
                f"| {row['dataset']} | {row['target_recall']:.2f} | unsupported | "
                "unsupported | unsupported | -- | -- |")
        else:
            lines.append(
                f"| {row['dataset']} | {row['target_recall']:.2f} | "
                f"{row['fhg_qps']:.2f} | {row['uhg_qps']:.2f} | "
                f"{row['qps_gain_pct']:+.2f}% | {row['fhg_bracket_ef']} | "
                f"{row['uhg_bracket_ef']} |")
    (summary / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    base.write_json(RUN_ROOT / "complete.json", {
        "schema": 1,
        "created_utc": base.utc_now(),
        "target_recalls": list(TARGETS),
        "results": base.file_info(summary / "RESULTS.md", True),
    })
    print((summary / "RESULTS.md").read_text(encoding="utf-8"), flush=True)


def run_all() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    probe = run_probe()
    selected = choose_full_points(probe)
    print("Selected full-run points:", selected, flush=True)
    timing_points = run_full(selected)
    summarize(timing_points)


def check_only() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    points = tuple(sorted({point for _, values in PROBE_ROUNDS for point in values}))
    base.preflight(
        RUN_ROOT,
        base.DATASETS,
        (WORKLOAD,),
        points,
        -1,
        REPETITIONS,
    )
    print(f"Preflight passed: {len(base.DATASETS)} datasets, 2 graph topologies")


def summarize_existing() -> None:
    path = RUN_ROOT / "timing_points.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed timing selection: {path}")
    timing_points = json.loads(path.read_text(encoding="utf-8"))
    summarize(timing_points)


def main() -> None:
    global RUN_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "run", "summarize"))
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--cpu", type=int,
        help="logical CPU used for single-threaded timing (default: first allowed CPU)",
    )
    args = parser.parse_args()
    try:
        base.configure_cpu(args.cpu)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    RUN_ROOT = args.run_root.resolve()
    if args.stage == "check":
        check_only()
    elif args.stage == "summarize":
        summarize_existing()
    else:
        run_all()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", flush=True)
        raise
