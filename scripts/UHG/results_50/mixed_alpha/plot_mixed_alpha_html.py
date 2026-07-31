#!/usr/bin/env python3
"""Generate the Recall@50 mixed-alpha QPS-Recall HTML plot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_plotter():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts_50"
        / "hybrid_union"
        / "mixed_alpha"
        / "plot_mixed_alpha_html.py"
    )
    spec = importlib.util.spec_from_file_location("_uhg_mixed_alpha_plotter_50", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load mixed-alpha plotter: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    load_plotter().main()
