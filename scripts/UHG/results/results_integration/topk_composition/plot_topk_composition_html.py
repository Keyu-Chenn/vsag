#!/usr/bin/env python3
"""Generate a standalone HTML/SVG figure for top-k composition.

The HTML is self-contained and can be opened directly in a browser. Use the
"Print PDF" button, or the browser print dialog, to save the SVG figure as PDF.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DATASET_ORDER = ["nq", "msmarco", "hotpotqa"]
DATASET_LABELS = {
    "nq": "NQ",
    "msmarco": "MS MARCO",
    "hotpotqa": "HotpotQA",
}
SEGMENT_ORDER = ["sparse_only", "both", "dense_only", "neither"]
SEGMENT_LABELS = {
    "sparse_only": "Sparse only",
    "both": "Both",
    "dense_only": "Dense only",
    "neither": "Neither",
}
SEGMENT_COLORS = {
    "sparse_only": "#EE8E7A",
    "both": "#78BEA8",
    "dense_only": "#78A6C8",
    "neither": "#D9DCE3",
}
SEGMENT_TEXT_COLORS = {
    "sparse_only": "#1F2328",
    "both": "#1F2328",
    "dense_only": "#1F2328",
    "neither": "#222222",
}


def load_data(json_path: Path) -> dict:
    with json_path.open() as f:
        return json.load(f)


def build_payload(data: dict, title: str) -> dict:
    datasets = [dataset for dataset in DATASET_ORDER if dataset in data["data"]]
    row_count = len(data["alphas"])
    row_height = 38
    row_gap = 25
    panel_top = 162
    figure_height = panel_top + row_count * row_height + max(row_count - 1, 0) * row_gap + 86
    page_height_in = round(7.4 * figure_height / 1200, 2)
    return {
        "title": title,
        "topk": data["topk"],
        "alphas": data["alphas"],
        "height": figure_height,
        "pageHeightIn": page_height_in,
        "datasets": datasets,
        "datasetLabels": DATASET_LABELS,
        "segmentOrder": SEGMENT_ORDER,
        "segmentLabels": SEGMENT_LABELS,
        "segmentColors": SEGMENT_COLORS,
        "segmentTextColors": SEGMENT_TEXT_COLORS,
        "data": {
            dataset: {
                alpha: data["data"][dataset]["alphas"][alpha]["mean"]
                for alpha in data["alphas"]
            }
            for dataset in datasets
        },
    }


def build_html(payload: dict) -> str:
    payload_json = html.escape(json.dumps(payload, ensure_ascii=False), quote=False)
    title = html.escape(payload["title"])
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --ink: #1f2328;
  --page-bg: #f5f6f8;
  --paper: #ffffff;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
  padding: 24px;
  background: var(--page-bg);
  color: var(--ink);
  font-family: "Times New Roman", Times, "Nimbus Roman No9 L", serif;
}
.toolbar {
  width: min(1240px, 100%);
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
button {
  appearance: none;
  border: 1px solid #c8ced6;
  border-radius: 6px;
  padding: 7px 13px;
  background: #ffffff;
  color: #1f2328;
  font: 14px "Times New Roman", Times, "Nimbus Roman No9 L", serif;
  cursor: pointer;
  box-shadow: 0 4px 14px rgba(31, 35, 40, 0.08);
}
button:hover {
  background: #f7f8fa;
}
.page {
  width: min(1240px, 100%);
  background: var(--paper);
  border: 1px solid #dde2e8;
  padding: 16px;
  box-shadow: 0 22px 60px rgba(31, 35, 40, 0.12);
}
svg {
  display: block;
  width: 100%;
  height: auto;
}
@page {
  size: 7.4in __PAGE_HEIGHT_IN__in;
  margin: 0.05in;
}
@media print {
  body {
    display: block;
    min-height: auto;
    padding: 0;
    background: #ffffff;
  }
  .toolbar {
    display: none;
  }
  .page {
    width: 100%;
    padding: 0;
    border: 0;
    box-shadow: none;
  }
}
</style>
</head>
<body>
<div class="toolbar">
  <button type="button" onclick="window.print()">Print PDF</button>
  <button type="button" onclick="downloadSvg()">Save SVG</button>
</div>
<main class="page" aria-label="Top-k composition figure">
  <svg id="figure" viewBox="0 0 1200 __FIGURE_HEIGHT__" role="img"
       aria-labelledby="figure-title figure-desc"></svg>
</main>
<script id="figure-data" type="application/json">__PAYLOAD__</script>
<script>
const FIGURE = JSON.parse(document.getElementById("figure-data").textContent);
const SVG_NS = "http://www.w3.org/2000/svg";

function node(name, attrs = {}, text = null) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    element.setAttribute(key, value);
  }
  if (text !== null) element.textContent = text;
  return element;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function labelColor(segment) {
  return FIGURE.segmentTextColors?.[segment] || "#111111";
}

function segmentColor(segment) {
  return FIGURE.segmentColors[segment] || "#999999";
}

function draw() {
  const svg = document.getElementById("figure");
  const width = 1200;
  const height = FIGURE.height;
  const margin = { top: 34, right: 38, bottom: 58, left: 82 };
  const panelGap = 54;
  const legendY = 82;
  const panelTop = 162;
  const panelTitleY = 134;
  const barHeight = 38;
  const rowGap = 25;
  const panelWidth =
    (width - margin.left - margin.right - panelGap * (FIGURE.datasets.length - 1)) /
    FIGURE.datasets.length;
  const axisY = panelTop + FIGURE.alphas.length * barHeight +
    (FIGURE.alphas.length - 1) * rowGap + 28;

  svg.replaceChildren();

  const defs = node("defs");
  const style = node("style");
  style.textContent = `
    text {
      font-family: "Times New Roman", Times, "Nimbus Roman No9 L", serif;
      fill: #1f2328;
    }
    .title { font-size: 24px; font-weight: 700; }
    .panel-title { font-size: 20px; font-weight: 700; }
    .legend { font-size: 15px; fill: #272b30; }
    .tick { font-size: 14px; fill: #000000; }
    .alpha { font-size: 17px; fill: #272b30; }
    .axis-label { font-size: 16px; fill: #272b30; }
    .bar-label { font-size: 15px; font-weight: 700; }
  `;
  defs.appendChild(style);

  svg.appendChild(defs);
  svg.appendChild(node("title", { id: "figure-title" }, FIGURE.title));
  svg.appendChild(node("desc", { id: "figure-desc" },
    `Stacked source composition of hybrid top-${FIGURE.topk} results.`));

  svg.appendChild(node("text", {
    x: width / 2,
    y: margin.top,
    "text-anchor": "middle",
    class: "title"
  }, FIGURE.title));

  const legendStep = 96;
  const swatchWidth = 24;
  const legendItems = FIGURE.segmentOrder.map((segment) => ({
    segment,
    label: FIGURE.segmentLabels[segment]
  }));
  const legendWidth = (legendItems.length - 1) * legendStep + swatchWidth;
  const legendStartX = (width - legendWidth) / 2;
  for (const [legendIndex, item] of legendItems.entries()) {
    const legendX = legendStartX + legendIndex * legendStep;
    svg.appendChild(node("rect", {
      x: legendX,
      y: legendY - 13,
      width: swatchWidth,
      height: 13,
      rx: 2,
      fill: segmentColor(item.segment),
      stroke: "none"
    }));
    svg.appendChild(node("text", {
      x: legendX + swatchWidth / 2,
      y: legendY + 18,
      "text-anchor": "middle",
      class: "legend"
    }, item.label));
  }

  FIGURE.datasets.forEach((dataset, datasetIndex) => {
    const panelX = margin.left + datasetIndex * (panelWidth + panelGap);
    const panelLabel = FIGURE.datasetLabels[dataset] || dataset;

    svg.appendChild(node("text", {
      x: panelX + panelWidth / 2,
      y: panelTitleY,
      "text-anchor": "middle",
      class: "panel-title"
    }, panelLabel));

    for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
      const x = panelX + panelWidth * tick;
      svg.appendChild(node("line", {
        x1: x,
        y1: panelTop - 10,
        x2: x,
        y2: axisY,
        stroke: tick === 0 || tick === 1 ? "#9aa3ad" : "#e7eaee",
        "stroke-width": tick === 0 || tick === 1 ? 1 : 0.85
      }));
      svg.appendChild(node("text", {
        x,
        y: axisY + 24,
        "text-anchor": "middle",
        class: "tick"
      }, formatPercent(tick)));
    }

    svg.appendChild(node("line", {
      x1: panelX,
      y1: axisY,
      x2: panelX + panelWidth,
      y2: axisY,
      stroke: "#7b848e",
      "stroke-width": 1
    }));

    FIGURE.alphas.forEach((alpha, alphaIndex) => {
      const y = panelTop + alphaIndex * (barHeight + rowGap);
      const clipId = `clip-${dataset}-${alphaIndex}`;
      const clip = node("clipPath", { id: clipId });
      clip.appendChild(node("rect", {
        x: panelX,
        y,
        width: panelWidth,
        height: barHeight,
        rx: 6,
        ry: 6
      }));
      defs.appendChild(clip);

      if (datasetIndex === 0) {
        svg.appendChild(node("text", {
          x: panelX - 18,
          y: y + barHeight / 2 + 6,
          "text-anchor": "end",
          class: "alpha"
        }, `\\u03b1 = ${alpha}`));
      }

      svg.appendChild(node("rect", {
        x: panelX,
        y,
        width: panelWidth,
        height: barHeight,
        rx: 6,
        ry: 6,
        fill: "#ffffff",
        stroke: "#9aa3ad",
        "stroke-width": 0.9
      }));

      let cursor = panelX;
      for (const segment of FIGURE.segmentOrder) {
        const value = FIGURE.data[dataset][alpha][segment];
        const segmentWidth = panelWidth * value;
        const rect = node("rect", {
          x: cursor,
          y,
          width: Math.max(segmentWidth, 0),
          height: barHeight,
          fill: segmentColor(segment),
          "clip-path": `url(#${clipId})`
        });
        rect.appendChild(node("title", {},
          `${panelLabel}, \\u03b1=${alpha}, ${FIGURE.segmentLabels[segment]}: ${(value * 100).toFixed(1)}%`));
        svg.appendChild(rect);

        if (segmentWidth >= 28) {
          svg.appendChild(node("text", {
            x: cursor + segmentWidth / 2,
            y: y + barHeight / 2 + 5,
            "text-anchor": "middle",
            class: "bar-label",
            style: `fill: ${labelColor(segment)}`
          }, formatPercent(value)));
        }
        cursor += segmentWidth;
      }
    });
  });

}

function downloadSvg() {
  const svg = document.getElementById("figure").cloneNode(true);
  svg.setAttribute("xmlns", SVG_NS);
  const blob = new Blob([new XMLSerializer().serializeToString(svg)], {
    type: "image/svg+xml;charset=utf-8"
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "topk_composition.svg";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

draw();
</script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", title)
        .replace("__PAYLOAD__", payload_json)
        .replace("__FIGURE_HEIGHT__", str(payload["height"]))
        .replace("__PAGE_HEIGHT_IN__", str(payload["pageHeightIn"]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML/SVG top-k composition figure."
    )
    parser.add_argument(
        "--json",
        default="/tbase-project/vsag/scripts/UHG/results/results_integration/topk_composition/topk_composition.json",
        help="Path to topk_composition.json.",
    )
    parser.add_argument(
        "--output",
        default="/tbase-project/vsag/scripts/UHG/results/results_integration/topk_composition/topk_composition.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--title",
        default="Hybrid Top-100 Source Composition",
        help="Figure title.",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise SystemExit(f"JSON not found: {json_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(load_data(json_path), args.title)
    output_path.write_text(build_html(payload), encoding="utf-8")
    print(f"Wrote HTML figure: {output_path}")


if __name__ == "__main__":
    main()
