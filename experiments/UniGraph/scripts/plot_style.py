"""Labels, colors, and SVG markers shared by the paper plots."""

from __future__ import annotations

import html
import math


DATASET_ORDER = ("nq", "hotpotqa", "msmarco", "fever", "dbpedia-entity")
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "fever": "FEVER",
    "dbpedia-entity": "DBpedia-Entity",
}

METHOD_LABELS = {
    "hnsw": "HNSW",
    "sindi": "SINDI",
    "hnsw_sindi": "HNSW+SINDI",
    "uhg": "UHG",
    "naive_hybrid_hnsw": "Naive Hybrid HNSW",
}
METHOD_COLORS = {
    "hnsw": "#9CA3AF",
    "sindi": "#4C78A8",
    "hnsw_sindi": "#59A14F",
    "uhg": "#E45756",
    "naive_hybrid_hnsw": "#F28E2B",
}
METHOD_MARKERS = {
    "hnsw": "circle",
    "sindi": "square",
    "hnsw_sindi": "plus",
    "uhg": "pentagon",
    "naive_hybrid_hnsw": "diamond",
}


def marker_svg(
        kind: str, x: float, y: float, color: str, size: float,
        tooltip: str | None = None) -> str:
    half = size / 2
    title = f"<title>{html.escape(tooltip)}</title>" if tooltip else ""
    common = f'fill="{color}" stroke="#111111" stroke-width="0.55"'
    if kind == "circle":
        shape = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{half:.2f}" {common}/>'
    elif kind == "square":
        shape = (
            f'<rect x="{x-half:.2f}" y="{y-half:.2f}" width="{size:.2f}" '
            f'height="{size:.2f}" {common}/>'
        )
    elif kind in {"triangle", "triangle-down"}:
        if kind == "triangle":
            points = (
                f"{x:.2f},{y-half:.2f} {x-half:.2f},{y+half:.2f} "
                f"{x+half:.2f},{y+half:.2f}"
            )
        else:
            points = (
                f"{x-half:.2f},{y-half:.2f} {x+half:.2f},{y-half:.2f} "
                f"{x:.2f},{y+half:.2f}"
            )
        shape = f'<polygon points="{points}" {common}/>'
    elif kind == "pentagon":
        points = []
        for index in range(5):
            angle = -math.pi / 2 + 2 * math.pi * index / 5
            points.append(
                f"{x + math.cos(angle)*half:.2f},{y + math.sin(angle)*half:.2f}")
        shape = f'<polygon points="{" ".join(points)}" {common}/>'
    elif kind == "diamond":
        points = (
            f"{x:.2f},{y-half:.2f} {x+half:.2f},{y:.2f} "
            f"{x:.2f},{y+half:.2f} {x-half:.2f},{y:.2f}"
        )
        shape = f'<polygon points="{points}" {common}/>'
    else:
        lines = (
            ((x-half, y, x+half, y), (x, y-half, x, y+half))
            if kind == "plus"
            else ((x-half, y-half, x+half, y+half),
                  (x-half, y+half, x+half, y-half))
        )
        shape = "".join(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>'
            for x1, y1, x2, y2 in lines
        )
    css_class = ' class="data-marker"' if tooltip else ""
    return f"<g{css_class}>{title}{shape}</g>"


def nice_log_upper(value: float) -> float:
    exponent = math.floor(math.log10(value))
    scale = 10 ** exponent
    normalized = value / scale
    return next(candidate * scale for candidate in (1, 2, 3, 5, 10)
                if normalized <= candidate)
