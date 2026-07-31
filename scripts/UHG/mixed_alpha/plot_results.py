#!/usr/bin/env python3
"""Plot the uniform mixed-alpha multi-k experiment."""

from pathlib import Path
import sys


UHG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UHG_ROOT))

from plot_mixed_alpha_multik import PlotConfig, generate_plot  # noqa: E402


if __name__ == "__main__":
    experiment_root = Path(__file__).resolve().parent
    generate_plot(
        PlotConfig(
            input_tsv=experiment_root / "results" / "mixed_alpha_points.tsv",
            output_html=experiment_root
            / "results"
            / "mixed_alpha_qps_recall.html",
            title="Uniform Mixed Alpha QPS–Recall",
            subtitle=(
                "Per-query α uses the uniform distribution on [0.3, 0.7]; "
                "QPS is shown on a logarithmic scale by default."
            ),
            download_stem="mixed_alpha_qps_recall",
            default_recall=20,
        )
    )
