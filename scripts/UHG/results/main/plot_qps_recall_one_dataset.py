#!/usr/bin/env python3
"""Plot QPS-Recall curves for one UHG result dataset."""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    AutoMinorLocator,
    FixedLocator,
    FuncFormatter,
    LogFormatterMathtext,
    LogLocator,
    MaxNLocator,
)

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
METHOD_MARKERS = {
    "hnsw": "o",
    "sindi": "s",
    "hnsw_sindi": "^",
    "fhg": "D",
    "uhg": "P",
}
METHOD_COLORS = {
    "hnsw": "#4C78A8",
    "sindi": "#F58518",
    "hnsw_sindi": "#54A24B",
    "fhg": "#B279A2",
    "uhg": "#E45756",
}
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "dbpedia-entity": "DBpedia-Entity",
}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"\b(?P<name>ef_search|bk|sindi_bk|dense_entry_bk)=(?P<value>[-+0-9.eE]+)")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#0F172A",
            "axes.labelsize": 10.5,
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 12.0,
            "axes.titleweight": "semibold",
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "grid.color": "#CBD5E1",
            "grid.linewidth": 0.65,
            "legend.fontsize": 10.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "xtick.color": "#334155",
            "xtick.labelsize": 9.5,
            "ytick.color": "#334155",
            "ytick.labelsize": 9.5,
        }
    )


@dataclass(frozen=True)
class Point:
    dataset: str
    method: str
    alpha: str
    recall: float
    qps: float
    param: str
    source: Path


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
        metric_match = METRIC_RE.search(line)
        if metric_match is None:
            continue
        recall = float(metric_match.group("recall"))
        qps = float(metric_match.group("qps"))
        if recall < 0 or qps <= 0:
            continue
        points.append(
            Point(
                dataset=dataset,
                method=method,
                alpha=alpha,
                recall=recall,
                qps=qps,
                param=parse_param(line),
                source=path,
            )
        )
    return points


def load_points(result_root: Path, dataset: str, methods: set[str] | None) -> list[Point]:
    dataset_dir = result_root / dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset result directory not found: {dataset_dir}")

    points: list[Point] = []
    for method_dir in sorted(dataset_dir.iterdir()):
        if not method_dir.is_dir() or method_dir.name == "plots":
            continue
        method = method_dir.name
        if methods is not None and method not in methods:
            continue
        for result_file in sorted(method_dir.glob("alpha_*.txt")):
            points.extend(parse_result_file(dataset, method, result_file))
    return points


def write_csv(points: list[Point], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "alpha", "method", "recall", "qps", "param", "source"])
        for point in sorted(
            points,
            key=lambda p: (alpha_sort_key(p.alpha), method_sort_key(p.method), p.recall, -p.qps),
        ):
            writer.writerow(
                [
                    point.dataset,
                    point.alpha,
                    point.method,
                    f"{point.recall:.8g}",
                    f"{point.qps:.8g}",
                    point.param,
                    str(point.source),
                ]
            )


def display_dataset_name(dataset: str) -> str:
    return DATASET_LABELS.get(dataset.lower(), dataset)


def format_recall_tick(value: float, _: int) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def get_x_axis(points: list[Point]) -> tuple[tuple[float, float], list[float]]:
    recalls = [point.recall for point in points if point.recall >= 0]
    if not recalls:
        return (0.7, 1.01), [0.7, 0.8, 0.9, 1.0]

    max_recall = min(1.0, max(recalls))
    lower = 0.7
    upper_tick = max(0.8, min(1.0, math.ceil((max_recall + 0.01) * 20.0) / 20.0))
    if max_recall >= 0.90:
        upper_tick = 1.0

    step = 0.1 if upper_tick - lower <= 0.85 else 0.2
    first_tick = math.ceil((lower - 1e-9) / step) * step
    ticks: list[float] = []
    tick = first_tick
    while tick <= upper_tick + 1e-9:
        ticks.append(round(tick, 2))
        tick += step
    if upper_tick == 1.0 and all(abs(tick - 1.0) > 1e-9 for tick in ticks):
        ticks.append(1.0)

    return (lower, min(1.01, upper_tick + 0.01)), ticks


def get_log_ticks(lower: float, upper: float) -> list[float]:
    start_exp = math.floor(math.log10(lower))
    end_exp = math.ceil(math.log10(upper))
    return [
        10.0**exp
        for exp in range(start_exp, end_exp + 1)
        if lower <= 10.0**exp <= upper * 1.001
    ]


def plot_alpha(
    ax: plt.Axes,
    points: list[Point],
    alpha: str,
    recall_at: int,
    log_qps: bool,
    x_limits: tuple[float, float],
    x_ticks: list[float],
    y_limits: tuple[float, float] | None,
    y_ticks: list[float] | None = None,
) -> None:
    methods = sorted({point.method for point in points}, key=method_sort_key)
    for method in methods:
        method_points = sorted(
            [point for point in points if point.method == method],
            key=lambda p: (p.recall, p.qps),
        )
        if not method_points:
            continue
        color = METHOD_COLORS.get(method, "#64748B")
        ax.plot(
            [point.recall for point in method_points],
            [point.qps for point in method_points],
            color=color,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2.1 if method == "uhg" else 1.9,
            markersize=5.3,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.25,
            label=METHOD_LABELS.get(method, method),
            solid_capstyle="round",
            zorder=3 if method == "uhg" else 2,
        )

    ax.set_title(f"alpha = {alpha}", loc="left", pad=8)
    ax.set_xlabel(f"Recall@{recall_at}")
    ax.set_ylabel("QPS")
    ax.set_xlim(*x_limits)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(format_recall_tick))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    if log_qps:
        ax.set_yscale("log")
    if y_ticks:
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))
        if log_qps:
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    elif not log_qps:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if log_qps:
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        )
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="-", alpha=0.72)
    ax.grid(True, which="minor", axis="y", linestyle="-", linewidth=0.4, alpha=0.28)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#CBD5E1")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.0, width=0.5)


def cleanup_old_outputs(output_dir: Path) -> None:
    for path in output_dir.glob("qps_recall_alpha_*.png"):
        path.unlink()
    for filename in ("qps_recall_overview.png", "qps_recall_points.csv"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def get_y_axis(points: list[Point], log_qps: bool) -> tuple[tuple[float, float], list[float] | None]:
    qps_values = [point.qps for point in points if point.qps > 0]
    if not qps_values:
        return (1.0, 10.0), [1.0, 10.0] if log_qps else None

    min_qps = min(qps_values)
    max_qps = max(qps_values)
    if log_qps:
        lower = 10.0
        upper = max(max_qps * 1.35, lower * 1.5)
        return (lower, upper), get_log_ticks(lower, upper)

    lower = min_qps / 1.08
    upper = max_qps * 1.12
    return (lower, upper), None


def make_alpha_axes(alpha_count: int) -> tuple[plt.Figure, list[plt.Axes]]:
    if alpha_count == 5:
        fig = plt.figure(figsize=(15.0, 8.4), constrained_layout=True)
        grid = fig.add_gridspec(2, 6)
        slots = [
            (0, slice(0, 2)),
            (0, slice(2, 4)),
            (0, slice(4, 6)),
            (1, slice(1, 3)),
            (1, slice(3, 5)),
        ]
        return fig, [fig.add_subplot(grid[row, col]) for row, col in slots]

    cols = min(3, alpha_count)
    if alpha_count == 4:
        cols = 2
    rows = (alpha_count + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.2 * cols, 3.95 * rows),
        constrained_layout=True,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax in axes_list[alpha_count:]:
        ax.remove()
    return fig, axes_list[:alpha_count]


def save_plot(
    points: list[Point],
    output_dir: Path,
    dataset: str,
    recall_at: int,
    log_qps: bool,
) -> Path | None:
    configure_plot_style()
    alphas = sorted({point.alpha for point in points}, key=alpha_sort_key)
    if not alphas:
        return None
    fig, axes_list = make_alpha_axes(len(alphas))
    x_limits, x_ticks = get_x_axis(points)
    y_limits, y_ticks = get_y_axis(points, log_qps)

    for ax, alpha in zip(axes_list, alphas):
        alpha_points = [point for point in points if point.alpha == alpha]
        plot_alpha(
            ax,
            alpha_points,
            alpha,
            recall_at=recall_at,
            log_qps=log_qps,
            x_limits=x_limits,
            x_ticks=x_ticks,
            y_limits=y_limits,
            y_ticks=y_ticks,
        )

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=min(len(labels), 5),
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
    fig.suptitle(
        f"{display_dataset_name(dataset)} QPS vs Recall@{recall_at}",
        fontsize=15.5,
        fontweight="bold",
        y=1.065,
    )

    output_path = output_dir / "qps_recall.png"
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return output_path


def main(
    default_results_root: Path | None = None,
    default_recall_at: int = 100,
) -> None:
    if default_results_root is None:
        default_results_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Plot QPS-Recall curves from <results-root>/<dataset>/<method>/alpha_*.txt."
    )
    parser.add_argument("dataset", help="Dataset name, for example: nq")
    parser.add_argument(
        "--results-root",
        default=str(default_results_root),
        help="Result root directory. Default: directory containing this script.",
    )
    parser.add_argument(
        "--recall-at",
        type=int,
        default=default_recall_at,
        help=f"Recall cutoff used in axis labels. Default: {default_recall_at}.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional method filter, for example: --methods hnsw sindi fhg uhg",
    )
    parser.add_argument(
        "--linear-qps",
        action="store_true",
        help="Use linear QPS axis instead of the default log scale.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write parsed points to qps_recall_points.csv.",
    )
    args = parser.parse_args()

    if args.recall_at <= 0:
        parser.error("--recall-at must be a positive integer")

    result_root = Path(args.results_root).resolve()
    methods = set(args.methods) if args.methods else None
    points = load_points(result_root, args.dataset, methods)
    if not points:
        raise SystemExit(
            f"No Recall/QPS points found under {result_root / args.dataset}. "
            "Run experiments first or check result file format."
        )

    output_dir = result_root / args.dataset / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_outputs(output_dir)

    csv_path = None
    if args.write_csv:
        csv_path = output_dir / "qps_recall_points.csv"
        write_csv(points, csv_path)

    log_qps = not args.linear_qps
    output_path = save_plot(
        points,
        output_dir,
        args.dataset,
        recall_at=args.recall_at,
        log_qps=log_qps,
    )

    print(f"Parsed points: {len(points)}")
    if csv_path is not None:
        print(f"Wrote CSV: {csv_path}")
    if output_path is not None:
        print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
