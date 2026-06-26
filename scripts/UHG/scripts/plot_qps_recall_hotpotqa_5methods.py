#!/usr/bin/env python3
import argparse
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib import rcParams


DEFAULT_BASE_RESULT_DIR = Path("/tbase-project/vsag/scripts/UHG/results/hotpotqa_4methods")
DEFAULT_UHGH_RESULT_DIR = Path("/tbase-project/vsag/scripts/UHG/results/hotpotqa_uhgh")
DEFAULT_OUTPUT_PREFIX = Path(
    "/tbase-project/vsag/scripts/UHG/results/hotpotqa_5methods/recall_qps_hotpotqa_5methods"
)

ALPHAS = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
BASE_METHODS = ["hnsw_sindi", "fhg", "uhg_refined", "uhgs_refined"]
UHGH_METHOD = "uhgh"
METHODS = BASE_METHODS + [UHGH_METHOD]
BASE_METHOD_RE = "|".join(re.escape(method) for method in BASE_METHODS)
BASE_LINE_RE = re.compile(
    rf"^({BASE_METHOD_RE})\b.*Recall:\s*([0-9.]+)\s*QPS:\s*([0-9.]+)"
)
UHGH_LINE_RE = re.compile(
    rf"^({UHGH_METHOD})\b.*Recall:\s*([0-9.]+)\s*QPS:\s*([0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Recall@100-QPS curves for HotpotQA 4-method results plus UHGH."
    )
    parser.add_argument(
        "--base-result-dir",
        type=Path,
        default=DEFAULT_BASE_RESULT_DIR,
        help=(
            "Directory containing existing hotpotqa_4methods alpha_*.txt files. "
            f"Default: {DEFAULT_BASE_RESULT_DIR}"
        ),
    )
    parser.add_argument(
        "--uhgh-result-dir",
        type=Path,
        default=DEFAULT_UHGH_RESULT_DIR,
        help=(
            "Directory containing UHGH alpha_*.txt files. "
            f"Default: {DEFAULT_UHGH_RESULT_DIR}"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Output path prefix without extension. "
            f"Default: {DEFAULT_OUTPUT_PREFIX}"
        ),
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.70,
        help="Only show points with recall >= this value when enough points remain.",
    )
    parser.add_argument(
        "--min-visible-points",
        type=int,
        default=3,
        help="Minimum points to keep per curve after filtering by recall.",
    )
    return parser.parse_args()


def empty_data():
    return {alpha: {method: [] for method in METHODS} for alpha in ALPHAS}


def read_method_results(data, result_dir, methods, line_re):
    for alpha in ALPHAS:
        path = result_dir / f"alpha_{alpha}.txt"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                match = line_re.search(line)
                if not match:
                    continue
                method, recall, qps = match.groups()
                if method in methods:
                    data[alpha][method].append((float(recall), float(qps)))


def read_results(base_result_dir, uhgh_result_dir):
    data = empty_data()
    read_method_results(data, base_result_dir, BASE_METHODS, BASE_LINE_RE)
    read_method_results(data, uhgh_result_dir, [UHGH_METHOD], UHGH_LINE_RE)
    return data


def select_visible_points(points, min_recall, min_visible_points):
    points = sorted(set(points))
    visible = [point for point in points if point[0] >= min_recall]
    if len(visible) >= min_visible_points:
        return visible
    return points[-min(len(points), min_visible_points) :]


def configure_plot_style():
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["STIXGeneral"]
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 11
    rcParams["axes.labelsize"] = 12
    rcParams["axes.titlesize"] = 13
    rcParams["legend.fontsize"] = 9
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10


def plot(data, output_prefix, min_recall, min_visible_points):
    styles = {
        "hnsw_sindi": {
            "label": "HNSW+SINDI",
            "marker": "o",
            "color": "#a0b8cf",
            "lw": 1.3,
            "ms": 4,
            "alpha": 0.75,
            "zorder": 1,
        },
        "fhg": {
            "label": "FHG",
            "marker": "D",
            "color": "#c9b68e",
            "lw": 1.3,
            "ms": 4,
            "alpha": 0.75,
            "zorder": 1,
        },
        "uhg_refined": {
            "label": "UHG",
            "marker": "s",
            "color": "#e06040",
            "lw": 2.0,
            "ms": 5,
            "alpha": 0.9,
            "zorder": 5,
        },
        "uhgs_refined": {
            "label": "UHGS",
            "marker": "^",
            "color": "#2050a0",
            "lw": 2.5,
            "ms": 7,
            "alpha": 1.0,
            "zorder": 10,
        },
        "uhgh": {
            "label": "UHGH",
            "marker": "P",
            "color": "#2a9d55",
            "lw": 2.2,
            "ms": 6,
            "alpha": 0.95,
            "zorder": 8,
        },
    }

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    for idx, alpha in enumerate(ALPHAS):
        ax = axes[idx // 3][idx % 3]
        for method in METHODS:
            points = select_visible_points(
                data[alpha][method], min_recall, min_visible_points
            )
            if not points:
                continue
            style = styles[method]
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                f'{style["marker"]}-',
                color=style["color"],
                label=style["label"],
                markersize=style["ms"],
                linewidth=style["lw"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                markeredgecolor="white",
                markeredgewidth=0.5 if method in {"uhgs_refined", "uhgh"} else 0.3,
            )

        ax.set_title(r"$\alpha$ = " + alpha, fontweight="bold")
        ax.set_xlabel("Recall@100")
        ax.set_ylabel("QPS")
        ax.set_xlim(max(0.0, min_recall - 0.02), 1.005)
        ax.grid(True, alpha=0.15, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        fontsize=11,
        framealpha=0.9,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.suptitle(
        "QPS vs Recall@100 on HotpotQA Dataset",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved to {png_path} and {pdf_path}")


def main():
    args = parse_args()
    data = read_results(args.base_result_dir, args.uhgh_result_dir)
    if not any(data[alpha][method] for alpha in ALPHAS for method in METHODS):
        raise SystemExit(
            f"No result lines found in {args.base_result_dir} or {args.uhgh_result_dir}"
        )
    if not any(data[alpha][UHGH_METHOD] for alpha in ALPHAS):
        raise SystemExit(f"No UHGH result lines found in {args.uhgh_result_dir}")

    configure_plot_style()
    plot(data, args.output_prefix, args.min_recall, args.min_visible_points)


if __name__ == "__main__":
    main()
