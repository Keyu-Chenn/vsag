#!/usr/bin/env python3
"""Plot Recall@20 QPS-Recall curves for one UHG result dataset."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_shared_plotter():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "main"
        / "plot_qps_recall_one_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("_uhg_qps_recall_plotter", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load shared plotter: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    load_shared_plotter().main(
        default_results_root=Path(__file__).resolve().parent,
        default_recall_at=20,
    )
