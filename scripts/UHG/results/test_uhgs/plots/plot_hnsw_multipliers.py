#!/usr/bin/env python3
"""Plot NQ and HotpotQA Recall-QPS curves for UHGH HNSW 1x/2x.

Each dataset gets one PNG with alpha=0.6 and 0.7 subplots. HNSW 1x comes
from the main experiment's method=auto results; auto routes both alphas to
UHGH. HNSW 2x comes from test_uhgs/uhgh_hnsw_2x. This script only produces
PNG images and rejects incomplete or mismatched ef_search sweeps.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path("/tbase-project/vsag")
TEST_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "test_uhgs"
MAIN_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "main"
PLOTS_DIR = TEST_ROOT / "plots"
DEFAULT_DATASETS = ("nq", "hotpotqa")
ALPHAS = ("0.6", "0.7")
DATASET_LABELS = {"nq": "NQ", "hotpotqa": "HotpotQA"}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
EF_RE = re.compile(r"\bef_search=(?P<ef>\d+)")
DENSE_ENTRY_BK_RE = re.compile(r"\bdense_entry_bk=(?P<value>\d+)")
DENSE_ENTRY_EF_RE = re.compile(r"\bdense_entry_ef_search=(?P<value>\d+)")


class PlotInputError(RuntimeError):
    """Raised when a required result is absent, incomplete, or inconsistent."""


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    multiplier: int
    result_subdir: str
    color: str
    marker: str
    required_method: str
    from_main: bool = False


@dataclass(frozen=True)
class Point:
    ef_search: int
    recall: float
    qps: float


SERIES = (
    Series(
        key="hnsw_1x",
        label="HNSW 1x",
        multiplier=1,
        result_subdir="uhg",
        color="#4C78A8",
        marker="o",
        required_method="method=auto",
        from_main=True,
    ),
    Series(
        key="hnsw_2x",
        label="HNSW 2x",
        multiplier=2,
        result_subdir="uhgh_hnsw_2x",
        color="#E45756",
        marker="s",
        required_method="method=uhgh",
    ),
)


def result_path(dataset: str, alpha: str, series: Series) -> Path:
    root = MAIN_ROOT if series.from_main else TEST_ROOT
    return root / dataset / series.result_subdir / f"alpha_{alpha}.txt"


def require_config(path: Path, dataset: str, alpha: str, series: Series) -> None:
    text = path.read_text(errors="replace")
    required = (
        series.required_method,
        f"dataset={dataset}",
        f"alpha={alpha}",
        "k=100",
        "num_queries=-1",
        "build_alpha=0.5",
        "hybrid_prune_scale=0.3",
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise PlotInputError(
            f"Unexpected or stale configuration in {path}; missing: "
            + ", ".join(missing)
        )


def parse_result(
    path: Path, dataset: str, alpha: str, series: Series
) -> list[Point]:
    if not path.is_file():
        raise PlotInputError(f"Required result file not found: {path}")
    require_config(path, dataset, alpha, series)

    points: list[Point] = []
    seen_efs: set[int] = set()
    for line in path.read_text(errors="replace").splitlines():
        metric_match = METRIC_RE.search(line)
        ef_match = EF_RE.search(line)
        if metric_match is None or ef_match is None:
            continue

        ef_search = int(ef_match.group("ef"))
        expected_dense_entry = series.multiplier * ef_search
        bk_match = DENSE_ENTRY_BK_RE.search(line)
        dense_ef_match = DENSE_ENTRY_EF_RE.search(line)
        if bk_match is not None and int(bk_match.group("value")) != expected_dense_entry:
            raise PlotInputError(
                f"Unexpected dense_entry_bk in {path}: ef_search={ef_search}, "
                f"expected {expected_dense_entry}, found {bk_match.group('value')}"
            )
        if (
            dense_ef_match is not None
            and int(dense_ef_match.group("value")) != expected_dense_entry
        ):
            raise PlotInputError(
                f"Unexpected dense_entry_ef_search in {path}: ef_search={ef_search}, "
                f"expected {expected_dense_entry}, found {dense_ef_match.group('value')}"
            )

        recall = float(metric_match.group("recall"))
        qps = float(metric_match.group("qps"))
        if not 0.0 <= recall <= 1.0 or qps <= 0.0:
            raise PlotInputError(f"Invalid Recall/QPS point in {path}: {line}")
        if ef_search in seen_efs:
            raise PlotInputError(f"Duplicate ef_search={ef_search} in {path}")
        seen_efs.add(ef_search)
        points.append(Point(ef_search=ef_search, recall=recall, qps=qps))

    if not points:
        raise PlotInputError(f"No Recall/QPS points found in {path}")
    return sorted(points, key=lambda point: point.recall)


def load_points(dataset: str) -> dict[tuple[str, str], list[Point]]:
    all_points: dict[tuple[str, str], list[Point]] = {}
    for alpha in ALPHAS:
        expected_efs: set[int] | None = None
        for series in SERIES:
            path = result_path(dataset, alpha, series)
            points = parse_result(path, dataset, alpha, series)
            current_efs = {point.ef_search for point in points}
            if expected_efs is None:
                expected_efs = current_efs
            elif current_efs != expected_efs:
                raise PlotInputError(
                    f"Incomplete or mismatched ef_search points for {dataset} alpha={alpha}: "
                    f"expected {sorted(expected_efs)}, found {sorted(current_efs)} in {path}"
                )
            all_points[(alpha, series.key)] = points
    return all_points


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
            "ytick.color": "#334155",
        }
    )


def plot_dataset(
    dataset: str,
    points: dict[tuple[str, str], list[Point]],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(9.0, 3.8), squeeze=False)

    for index, alpha in enumerate(ALPHAS):
        ax = axes[0][index]
        for series in SERIES:
            series_points = points[(alpha, series.key)]
            ax.plot(
                [point.recall for point in series_points],
                [point.qps for point in series_points],
                color=series.color,
                marker=series.marker,
                linewidth=1.6,
                markersize=5.5,
                label=series.label,
            )
        ax.set_title(f"alpha={alpha}")
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("QPS (log10 scale)")
        ax.set_yscale("log", base=10)
        ax.grid(True, which="both")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(loc="best", framealpha=0.92)

    dataset_label = DATASET_LABELS.get(dataset, dataset)
    fig.suptitle(
        f"{dataset_label} - UHGH Dense-entry HNSW Candidates",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{dataset}_hnsw_multiplier_qps_recall_log.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved: {output}")
    return output


def main() -> None:
    configure_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Datasets to plot (default: nq hotpotqa)",
    )
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args()

    failures: list[str] = []
    for dataset in args.datasets:
        try:
            points = load_points(dataset)
        except PlotInputError as error:
            failures.append(f"{dataset}: {error}")
            continue
        plot_dataset(dataset, points, args.output_dir)

    if failures:
        raise SystemExit("Cannot plot incomplete datasets:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
