#!/usr/bin/env python3
"""Plot QPS-Recall curves for the hybrid-pruning ablation.

Reads:
  results_50/test_prune/<dataset>/prune_<scale>/alpha_mixed.txt

Produces, per dataset, a figure comparing different hybrid_prune_scale values
and the true "no_prune" configuration on the mixed-alpha queries.

Usage:
  python3 plot_test_prune.py                       # all datasets
  python3 plot_test_prune.py nq                    # one dataset
  python3 plot_test_prune.py nq hotpotqa           # selected datasets
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
RESULT_ROOT = REPO_ROOT / "scripts" / "UHG" / "results_50" / "test_prune"
OUTPUT_DIR = RESULT_ROOT / "plots"

# Ordered palette for pruning configurations. Keys are the canonical directory
# suffixes (e.g. "0.3", "1.0", "no_prune"). Unknown values use a default colour.
SCALE_COLORS = {
    "0.3": "#E45756",
    "0.5": "#F58518",
    "1.0": "#4C78A8",
    "no_prune": "#6B7280",
}
SCALE_MARKERS = {
    "0.3": "^",
    "0.5": "s",
    "1.0": "o",
    "no_prune": "D",
}
DEFAULT_COLOR = "#54A24B"
DEFAULT_MARKER = "^"

DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "dbpedia-entity": "DBpedia-Entity",
}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"\b(?P<name>ef_search|bk)=(?P<value>[-+0-9.eE]+)")


@dataclass(frozen=True)
class Point:
    dataset: str
    scale: str
    alpha: str
    recall: float
    qps: float
    param: str


def scale_sort_key(scale: str) -> tuple[int, float | str]:
    try:
        return (0, float(scale))
    except ValueError:
        return (1, scale)


def canonical_scale(scale: str) -> str:
    if scale.lower() in {"no_prune", "no-prune", "off", "none", "disabled"}:
        return "no_prune"
    return scale


def scale_label(scale: str) -> str:
    if scale == "no_prune":
        return "no prune"
    try:
        v = float(scale)
        if v == 1.0:
            return f"prune_scale={v:g} (Cauchy bound)"
        if v <= 0.0:
            return f"prune_scale={v:g} (most aggressive)"
        if v < 1.0:
            return f"prune_scale={v:g} (aggressive)"
        return f"prune_scale={v:g} (conservative)"
    except ValueError:
        return f"prune_scale={scale}"


def parse_param(line: str) -> str:
    params = {m.group("name"): m.group("value") for m in PARAM_RE.finditer(line)}
    if "ef_search" in params:
        return f"ef={params['ef_search']}"
    if "bk" in params:
        return f"bk={params['bk']}"
    return ""


def parse_result_file(dataset: str, scale: str, path: Path) -> list[Point]:
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
            Point(dataset, scale, alpha, recall, qps, parse_param(line))
        )
    return points


def load_points(dataset: str) -> list[Point]:
    dataset_dir = RESULT_ROOT / dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset result directory not found: {dataset_dir}")
    scale_dirs: dict[str, Path] = {}
    for scale_dir in sorted(dataset_dir.glob("prune_*")):
        if not scale_dir.is_dir():
            continue
        scale = canonical_scale(scale_dir.name.removeprefix("prune_"))
        canonical_name = f"prune_{scale}"
        if scale not in scale_dirs or scale_dir.name == canonical_name:
            scale_dirs[scale] = scale_dir

    points: list[Point] = []
    for scale, scale_dir in scale_dirs.items():
        for result_file in sorted(scale_dir.glob("alpha_mixed.txt")):
            points.extend(parse_result_file(dataset, scale, result_file))
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
            "legend.fontsize": 9.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "xtick.color": "#334155",
            "xtick.labelsize": 9.0,
            "ytick.color": "#334155",
            "ytick.labelsize": 9.0,
        }
    )


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
        for scale in sorted(
            {p.scale for p in points if p.alpha == alpha}, key=scale_sort_key
        ):
            pts = sorted(
                [p for p in points if p.alpha == alpha and p.scale == scale],
                key=lambda p: p.recall,
            )
            if not pts:
                continue
            ax.plot(
                [p.recall for p in pts],
                [p.qps for p in pts],
                marker=SCALE_MARKERS.get(scale, DEFAULT_MARKER),
                color=SCALE_COLORS.get(scale, DEFAULT_COLOR),
                label=scale_label(scale),
                linewidth=1.4,
                markersize=5,
            )
        ax.set_title("mixed alpha" if alpha == "mixed" else f"alpha={alpha}")
        ax.set_xlabel("Recall@50")
        ax.set_ylabel("QPS")
        ax.grid(True, which="major")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if idx == 0:
            ax.legend(loc="best", framealpha=0.9)

    for idx in range(len(alphas), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)} - Hybrid Pruning Ablation",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUT_DIR / f"{dataset}_prune_ablation_qps_recall.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _alpha_sort_key(alpha: str) -> tuple[int, float | str]:
    try:
        return (0, float(alpha))
    except ValueError:
        return (1, alpha)


def write_summary(points: list[Point], dataset: str) -> None:
    out_path = OUTPUT_DIR / f"{dataset}_prune_ablation_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["dataset", "alpha", "prune_config", "param", "recall", "qps"])
        for p in sorted(
            points,
            key=lambda p: (
                _alpha_sort_key(p.alpha),
                scale_sort_key(p.scale),
                p.recall,
            ),
        ):
            writer.writerow(
                [p.dataset, p.alpha, p.scale, p.param, f"{p.recall:.8g}", f"{p.qps:.8g}"]
            )
    print(f"Saved: {out_path}")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["nq", "hotpotqa", "msmarco", "dbpedia-entity"],
        help="Datasets to plot (default: all four)",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        points = load_points(dataset)
        if not points:
            print(f"No points found for {dataset}, skipping")
            continue
        plot_dataset(points, dataset)
        write_summary(points, dataset)


if __name__ == "__main__":
    main()
