#!/usr/bin/env python3
"""Plot QPS-Recall curves for the UHGH scaled dense-entry sweep.

Reads:
  results/test_uhgh/<dataset>/scale_<S>/alpha_<alpha>.txt

Each line in the result file corresponds to one main ef_search point with
dense_entry_ef_search = dense_entry_bk = S * ef_search. The figure overlays
different ENTRY_SCALE values on a single Recall-QPS plot, so you can see which
scale gives the best Recall-QPS frontier.

Usage:
  python3 plot_test_uhgh.py                # all datasets found under RESULT_ROOT
  python3 plot_test_uhgh.py nq             # one dataset
  python3 plot_test_uhgh.py nq hotpotqa    # selected datasets
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
RESULT_ROOT = Path(
    os.environ.get(
        "RESULT_ROOT",
        REPO_ROOT / "scripts" / "UHG" / "results" / "test_uhgh",
    )
)
RECALL_AT = int(os.environ.get("RECALL_AT", "100"))
OUTPUT_DIR = RESULT_ROOT / "plots"

DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "fever": "FEVER",
    "dbpedia-entity": "DBpedia-entity",
}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(
    r"\b(?P<name>ef_search|dense_entry_ef|dense_entry_bk|entry_scale)=(?P<value>[-+0-9.eE]+)"
)

# Palette for entry scales. 0.5 is the new default; 1.0 is the main-experiment
# behaviour (entry_ef = entry_bk = ef_search); additional scales fall back.
SCALE_COLORS = {
    "0.25": "#9ECBE0",
    "0.5": "#08519C",
    "0.7": "#2B8CBE",
    "0.75": "#4292C6",
    "1.0": "#111827",
    "1.2": "#54A24B",
    "1.5": "#F58518",
    "2.0": "#E45756",
}
SCALE_MARKERS = {
    "0.25": "o",
    "0.5": "o",
    "0.7": "s",
    "0.75": "s",
    "1.0": "D",
    "1.2": "P",
    "1.5": "^",
    "2.0": "^",
}
DEFAULT_COLOR = "#54A24B"
DEFAULT_MARKER = "P"


@dataclass(frozen=True)
class Point:
    dataset: str
    alpha: str
    scale: float
    main_ef: int
    entry_ef: int
    entry_bk: int
    recall: float
    qps: float


def scale_sort_key(scale: float) -> float:
    return scale


def scale_label_text(scale: float) -> str:
    if scale == 1.0:
        return "entry_ef=ef_search (main-exp default)"
    return f"entry_ef = {scale:g} × ef_search"


def parse_line(line: str) -> dict[str, float] | None:
    params = {m.group("name"): m.group("value") for m in PARAM_RE.finditer(line)}
    if "ef_search" not in params:
        return None
    out: dict[str, float] = {}
    for k in ("ef_search", "dense_entry_ef", "dense_entry_bk"):
        if k in params:
            out[k] = int(float(params[k]))
    if "entry_scale" in params:
        out["entry_scale"] = float(params["entry_scale"])
    return out


def parse_result_file(dataset: str, scale: float, path: Path) -> list[Point]:
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
        parsed = parse_line(line)
        if parsed is None or "ef_search" not in parsed:
            continue
        points.append(
            Point(
                dataset,
                alpha,
                scale,
                int(parsed["ef_search"]),
                int(parsed.get("dense_entry_ef", round(scale * int(parsed["ef_search"])))),
                int(parsed.get("dense_entry_bk", round(scale * int(parsed["ef_search"])))),
                recall,
                qps,
            )
        )
    return points


def load_points(dataset: str) -> list[Point]:
    dataset_dir = RESULT_ROOT / dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset result directory not found: {dataset_dir}")
    points: list[Point] = []
    for scale_dir in sorted(dataset_dir.glob("scale_*")):
        if not scale_dir.is_dir():
            continue
        try:
            scale = float(scale_dir.name.removeprefix("scale_"))
        except ValueError:
            continue
        for result_file in sorted(scale_dir.glob("alpha_*.txt")):
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
        nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), squeeze=False
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
            scale_key = f"{scale:g}"
            is_main_exp = scale == 1.0
            ax.plot(
                [p.recall for p in pts],
                [p.qps for p in pts],
                marker=SCALE_MARKERS.get(scale_key, DEFAULT_MARKER),
                color=SCALE_COLORS.get(scale_key, DEFAULT_COLOR),
                label=scale_label_text(scale),
                linewidth=1.6 if is_main_exp else 1.3,
                markersize=5,
                linestyle="--" if is_main_exp else "-",
            )
            # annotate ef_search values next to each point for readability
            for p in pts:
                ax.annotate(
                    f"ef={p.main_ef}",
                    (p.recall, p.qps),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6.5,
                    color=SCALE_COLORS.get(scale_key, DEFAULT_COLOR),
                    alpha=0.85,
                )
        ax.set_title(f"alpha={alpha}")
        ax.set_xlabel(f"Recall@{RECALL_AT}")
        ax.set_ylabel("QPS")
        ax.grid(True, which="major")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if idx == 0:
            ax.legend(loc="best", framealpha=0.9)

    for idx in range(len(alphas), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)} - UHGH dense-entry scale sweep "
        f"(Recall@{RECALL_AT})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = OUTPUT_DIR / f"{dataset}_uhgh_entry_scale_qps_recall.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def write_summary(points: list[Point], dataset: str) -> None:
    out_path = OUTPUT_DIR / f"{dataset}_uhgh_entry_scale_summary.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "dataset",
                "alpha",
                "entry_scale",
                "ef_search",
                "dense_entry_ef",
                "dense_entry_bk",
                "recall",
                "qps",
            ]
        )
        for p in sorted(
            points,
            key=lambda p: (
                _alpha_sort_key(p.alpha),
                scale_sort_key(p.scale),
                p.main_ef,
            ),
        ):
            writer.writerow(
                [
                    p.dataset,
                    p.alpha,
                    f"{p.scale:g}",
                    p.main_ef,
                    p.entry_ef,
                    p.entry_bk,
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
        "results/test_uhgh.",
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
