#!/usr/bin/env python3
"""Plot QPS-Recall curves for the UHG max_hops ablation.

Reads:
  results/test_uhgs/test_max_hops/<dataset>/hops_<N|ef>/alpha_<alpha>.txt

Produces, per dataset, a figure with one subplot per alpha, comparing different
--max_hops values on the UHG hybrid_index (method=auto). max_hops=0 is the
no-limit baseline used by the main experiment; "ef" means max_hops was set
equal to ef_search for each point.

The max_hops label is derived from the directory name (hops_0, hops_50, ...,
hops_ef), not from individual result lines, so the "ef" configuration is
treated as a single curve even though the actual hop cap varies per ef point.

Usage:
  python3 plot_max_hops.py                # all datasets found under RESULT_ROOT
  python3 plot_max_hops.py nq             # one dataset
  python3 plot_max_hops.py nq hotpotqa    # selected datasets
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path("/tbase-project/vsag")
RESULT_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "test_uhgs" / "test_max_hops"
OUTPUT_DIR = RESULT_ROOT / "plots"

# Ordered palette for max_hops labels. 0 (no limit) gets a distinct baseline
# colour; fixed integer caps share a sequential blue ramp; "ef" (max_hops =
# ef_search) gets a contrasting orange.
HOPS_COLORS = {
    "0": "#111827",
    "50": "#9ECBE0",
    "100": "#6BAED6",
    "200": "#4292C6",
    "500": "#08519C",
    "ef": "#F58518",
}
HOPS_MARKERS = {
    "0": "D",
    "50": "o",
    "100": "o",
    "200": "o",
    "500": "o",
    "ef": "^",
}
DEFAULT_COLOR = "#54A24B"
DEFAULT_MARKER = "s"

DATASET_LABELS = {"nq": "NQ", "hotpotqa": "HotpotQA", "msmarco": "MS MARCO"}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"\b(?P<name>ef_search|bk)=(?P<value>[-+0-9.eE]+)")


@dataclass(frozen=True)
class Point:
    dataset: str
    hops_label: str  # "0", "50", "100", "200", "500", "ef"
    alpha: str
    recall: float
    qps: float
    ef: str


def hops_sort_key(label: str) -> tuple[int, float | str]:
    """Sort key: fixed integers first (ascending), then "ef" last."""
    if label == "ef":
        return (2, label)
    try:
        return (0, float(label))
    except ValueError:
        return (1, label)


def hops_label_text(label: str) -> str:
    if label == "0":
        return "max_hops=0 (no limit)"
    if label == "ef":
        return "max_hops=ef_search"
    return f"max_hops={label}"


def parse_ef(line: str) -> str:
    params = {m.group("name"): m.group("value") for m in PARAM_RE.finditer(line)}
    if "ef_search" in params:
        return params["ef_search"]
    if "bk" in params:
        return params["bk"]
    return ""


def parse_result_file(dataset: str, hops_label: str, path: Path) -> list[Point]:
    alpha = path.stem.removeprefix("alpha_")
    points: list[Point] = []
    for line in path.read_text(errors="replace").splitlines():
        m = METRIC_RE.search(line)
        if m is None:
            continue
        recall = float(m.group("recall"))
        qps = float(m.group("qps"))
        if recall < 0 or qps <= 0:
            continue
        points.append(
            Point(dataset, hops_label, alpha, recall, qps, parse_ef(line))
        )
    return points


def load_points(dataset: str) -> list[Point]:
    dataset_dir = RESULT_ROOT / dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset result directory not found: {dataset_dir}")
    points: list[Point] = []
    for hops_dir in sorted(dataset_dir.glob("hops_*")):
        if not hops_dir.is_dir():
            continue
        hops_label = hops_dir.name.removeprefix("hops_")
        for result_file in sorted(hops_dir.glob("alpha_*.txt")):
            points.extend(parse_result_file(dataset, hops_label, result_file))
    return points


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#0F172A",
            "axes.labelsize": 10.0,
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 11.0,
            "axes.titleweight": "semibold",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "grid.color": "#CBD5E1",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.fontsize": 8.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "xtick.color": "#334155",
            "xtick.labelsize": 9.0,
            "ytick.color": "#334155",
            "ytick.labelsize": 9.0,
        }
    )


def _alpha_sort_key(alpha: str) -> tuple[int, float | str]:
    try:
        return (0, float(alpha))
    except ValueError:
        return (1, alpha)


def plot_dataset(points: list[Point], dataset: str) -> Path:
    alphas = sorted({p.alpha for p in points}, key=_alpha_sort_key)
    if not alphas:
        print(f"No data for {dataset}")
        return Path()

    n = len(alphas)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, alpha in enumerate(alphas):
        ax = axes_flat[idx]
        for hops_label in sorted(
            {p.hops_label for p in points if p.alpha == alpha},
            key=hops_sort_key,
        ):
            pts = sorted(
                [p for p in points if p.alpha == alpha and p.hops_label == hops_label],
                key=lambda p: p.recall,
            )
            if not pts:
                continue
            is_baseline = hops_label == "0"
            ax.plot(
                [p.recall for p in pts],
                [p.qps for p in pts],
                marker=HOPS_MARKERS.get(hops_label, DEFAULT_MARKER),
                color=HOPS_COLORS.get(hops_label, DEFAULT_COLOR),
                label=hops_label_text(hops_label),
                linewidth=1.6 if is_baseline else 1.2,
                markersize=5,
                linestyle="--" if is_baseline else "-",
            )
        ax.set_title(f"alpha={alpha}")
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("QPS")
        ax.grid(True, which="major")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if idx == 0:
            ax.legend(loc="best", framealpha=0.9)

    for idx in range(len(alphas), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)} - UHG max_hops Ablation",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUT_DIR / f"{dataset}_max_hops_qps_recall.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def write_summary(points: list[Point], dataset: str) -> None:
    out_path = OUTPUT_DIR / f"{dataset}_max_hops_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["dataset", "alpha", "max_hops", "ef_search", "recall", "qps"])
        for p in sorted(
            points,
            key=lambda p: (
                _alpha_sort_key(p.alpha),
                hops_sort_key(p.hops_label),
                p.recall,
            ),
        ):
            writer.writerow(
                [
                    p.dataset,
                    p.alpha,
                    p.hops_label,
                    p.ef,
                    f"{p.recall:.8g}",
                    f"{p.qps:.8g}",
                ]
            )
    print(f"Saved: {out_path}")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets to plot. Defaults to every dataset directory under "
        "results/test_uhgs/test_max_hops.",
    )
    args = parser.parse_args()

    datasets = args.datasets
    if not datasets:
        datasets = sorted(
            d.name for d in RESULT_ROOT.iterdir() if d.is_dir() and d.name != "plots"
        )

    for dataset in datasets:
        points = load_points(dataset)
        if not points:
            print(f"No points found for {dataset}, skipping")
            continue
        plot_dataset(points, dataset)
        write_summary(points, dataset)


if __name__ == "__main__":
    main()
