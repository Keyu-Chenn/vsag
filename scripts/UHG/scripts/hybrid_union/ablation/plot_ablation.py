#!/usr/bin/env python3
"""Compare direct UHG recall with the main auto-routed entry method at equal ef_search.

For each dataset this reads:
  alpha=0.3: results/ablation/<dataset>/uhg/alpha_0.3.txt (UHG)
             results/main/<dataset>/uhg/alpha_0.3.txt     (UHGS via auto)
  alpha=0.7: results/ablation/<dataset>/uhg/alpha_0.7.txt (UHG)
             results/main/<dataset>/uhg/alpha_0.7.txt     (UHGH via auto)
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
ABLATION_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "ablation"
MAIN_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "main"
OUTPUT_DIR = ABLATION_ROOT / "plots"

COMPARISONS = {"0.3": "uhgs", "0.7": "uhgh"}
METHOD_LABELS = {"uhg": "UHG (direct)", "uhgs": "UHGS (main auto)", "uhgh": "UHGH (main auto)"}
METHOD_COLORS = {"uhg": "#4C78A8", "uhgs": "#F58518", "uhgh": "#54A24B"}
METHOD_MARKERS = {"uhg": "o", "uhgs": "s", "uhgh": "^"}
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "dbpedia-entity": "DBpedia-Entity",
}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
EF_RE = re.compile(r"\bef_search=(?P<ef>\d+)")


@dataclass(frozen=True)
class Metric:
    recall: float
    qps: float


@dataclass(frozen=True)
class ComparisonRow:
    dataset: str
    alpha: str
    ef_search: int
    entry_method: str
    uhg_recall: float
    entry_recall: float
    recall_delta: float


def parse_result_file(path: Path) -> dict[int, Metric]:
    metrics: dict[int, Metric] = {}
    for line in path.read_text(errors="replace").splitlines():
        metric_match = METRIC_RE.search(line)
        ef_match = EF_RE.search(line)
        if metric_match is None or ef_match is None:
            continue
        recall = float(metric_match.group("recall"))
        qps = float(metric_match.group("qps"))
        if recall < 0 or qps <= 0:
            continue
        metrics[int(ef_match.group("ef"))] = Metric(recall, qps)
    if not metrics:
        raise SystemExit(f"No Recall/QPS points found in {path}")
    return metrics


def require_config(path: Path, *tokens: str) -> None:
    text = path.read_text(errors="replace")
    missing = [token for token in tokens if token not in text]
    if missing:
        raise SystemExit(
            f"Result file has stale or unexpected configuration: {path}; "
            f"missing {', '.join(missing)}"
        )


def load_rows(dataset: str, ablation_root: Path, main_root: Path) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for alpha, entry_method in COMPARISONS.items():
        direct_path = ablation_root / dataset / "uhg" / f"alpha_{alpha}.txt"
        main_path = main_root / dataset / "uhg" / f"alpha_{alpha}.txt"
        for path in (direct_path, main_path):
            if not path.is_file():
                raise SystemExit(f"Required result file not found: {path}")

        require_config(
            direct_path,
            "method=uhg",
            "k=100",
            "num_queries=-1",
            "build_alpha=0.5",
            "hybrid_prune_scale=0.3",
        )
        require_config(
            main_path,
            "method=auto",
            "k=100",
            "num_queries=-1",
            "build_alpha=0.5",
            "hybrid_prune_scale=0.3",
        )
        direct = parse_result_file(direct_path)
        entry = parse_result_file(main_path)
        common_efs = sorted(set(direct) & set(entry))
        if not common_efs:
            raise SystemExit(
                f"No common ef_search points between {direct_path} and {main_path}"
            )
        if set(direct) != set(entry):
            print(
                f"Warning: comparing only common ef_search points for {dataset} alpha={alpha}: "
                f"{common_efs}"
            )

        for ef_search in common_efs:
            direct_metric = direct[ef_search]
            entry_metric = entry[ef_search]
            rows.append(
                ComparisonRow(
                    dataset=dataset,
                    alpha=alpha,
                    ef_search=ef_search,
                    entry_method=entry_method,
                    uhg_recall=direct_metric.recall,
                    entry_recall=entry_metric.recall,
                    recall_delta=entry_metric.recall - direct_metric.recall,
                )
            )
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#CBD5E1",
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "legend.fontsize": 9.0,
            "savefig.bbox": "tight",
        }
    )


def plot_dataset(rows: list[ComparisonRow], dataset: str, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), squeeze=False)
    for idx, (alpha, entry_method) in enumerate(COMPARISONS.items()):
        ax = axes[0][idx]
        alpha_rows = sorted(
            (row for row in rows if row.alpha == alpha), key=lambda row: row.ef_search
        )
        efs = [row.ef_search for row in alpha_rows]
        ax.plot(
            efs,
            [row.uhg_recall for row in alpha_rows],
            marker=METHOD_MARKERS["uhg"],
            color=METHOD_COLORS["uhg"],
            label=METHOD_LABELS["uhg"],
        )
        ax.plot(
            efs,
            [row.entry_recall for row in alpha_rows],
            marker=METHOD_MARKERS[entry_method],
            color=METHOD_COLORS[entry_method],
            label=METHOD_LABELS[entry_method],
        )
        ax.set_title(f"alpha={alpha}")
        ax.set_xlabel("ef_search")
        ax.set_ylabel("Recall@100")
        ax.set_xticks(efs)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.legend(loc="best")

    fig.suptitle(f"{DATASET_LABELS.get(dataset, dataset)} - Entry Strategy Recall")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_ablation_recall_by_ef.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def write_summary(rows: list[ComparisonRow], dataset: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_ablation_recall_by_ef.tsv"
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "dataset",
                "alpha",
                "ef_search",
                "uhg_recall",
                "entry_method",
                "entry_recall",
                "entry_minus_uhg_recall",
            ]
        )
        for row in sorted(rows, key=lambda row: (float(row.alpha), row.ef_search)):
            writer.writerow(
                [
                    row.dataset,
                    row.alpha,
                    row.ef_search,
                    f"{row.uhg_recall:.8g}",
                    row.entry_method,
                    f"{row.entry_recall:.8g}",
                    f"{row.recall_delta:.8g}",
                ]
            )
    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    configure_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["nq", "hotpotqa", "msmarco", "dbpedia-entity"],
        help="Datasets to compare (default: all four)",
    )
    parser.add_argument("--ablation-root", type=Path, default=ABLATION_ROOT)
    parser.add_argument("--main-root", type=Path, default=MAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    for dataset in args.datasets:
        rows = load_rows(dataset, args.ablation_root, args.main_root)
        plot_dataset(rows, dataset, args.output_dir)
        write_summary(rows, dataset, args.output_dir)


if __name__ == "__main__":
    main()
