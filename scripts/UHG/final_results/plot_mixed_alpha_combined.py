#!/usr/bin/env python3
"""Combine Recall@50/@100/@200 mixed-alpha results into one HTML/SVG figure."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path


REPO_ROOT = Path("/tbase-project/vsag")
RESULT_ROOTS = {
    50: REPO_ROOT / "scripts" / "UHG" / "results_50" / "mixed_alpha",
    100: REPO_ROOT / "scripts" / "UHG" / "results" / "mixed_alpha",
    200: REPO_ROOT
    / "scripts"
    / "UHG"
    / "mixed_alpha"
    / "results"
    / "recall_200",
}
OUTPUT_PATH = Path(__file__).with_name("mixed_alpha_recall50_100.html")
SUMMARY_PATH = Path(__file__).with_name("mixed_alpha_recall50_100_points.tsv")

DATASET_ORDER = ["nq", "hotpotqa", "msmarco", "fever", "dbpedia-entity"]
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "fever": "FEVER",
    "dbpedia-entity": "DBpedia-Entity",
}
METHOD_ORDER = ["hnsw", "sindi", "hnsw_sindi", "fhg", "uhg"]
METHOD_LABELS = {
    "hnsw": "HNSW",
    "sindi": "SINDI",
    "hnsw_sindi": "HNSW+SINDI",
    "fhg": "FHG",
    "uhg": "UHG",
}
METHOD_COLORS = {
    "hnsw": "#9CA3AF",
    "sindi": "#4C78A8",
    "hnsw_sindi": "#59A14F",
    "fhg": "#B07AA1",
    "uhg": "#E45756",
}
METHOD_MARKERS = {
    "hnsw": "circle",
    "sindi": "square",
    "hnsw_sindi": "cross",
    "fhg": "triangle",
    "uhg": "pentagon",
}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"\b(?P<name>point|ef_search|bk)=(?P<value>[-+0-9.eE]+)")
K_RE = re.compile(r"\bk=(?P<k>\d+)")


def parse_result_file(
    recall_at: int,
    dataset: str,
    method: str,
    path: Path,
) -> list[dict]:
    points: list[dict] = []
    observed_k: int | None = None
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        k_match = K_RE.search(line)
        if k_match is not None:
            observed_k = int(k_match.group("k"))

        metric_match = METRIC_RE.search(line)
        if metric_match is None:
            continue
        recall = float(metric_match.group("recall"))
        qps = float(metric_match.group("qps"))
        if not math.isfinite(recall) or not math.isfinite(qps):
            raise SystemExit(f"Non-finite metric at {path}:{line_number}")
        if recall < 0 or recall > 1 or qps <= 0:
            raise SystemExit(f"Invalid metric at {path}:{line_number}")

        param_match = PARAM_RE.search(line)
        points.append(
            {
                "recall_at": recall_at,
                "dataset": dataset,
                "method": method,
                "param_name": param_match.group("name") if param_match else "",
                "param_value": param_match.group("value") if param_match else "",
                "recall": round(recall, 8),
                "qps": round(qps, 8),
                "source": str(path),
            }
        )

    if observed_k != recall_at:
        raise SystemExit(f"Expected k={recall_at}, found k={observed_k}: {path}")
    if not points:
        raise SystemExit(f"No Recall/QPS points found: {path}")
    return points


def load_points() -> list[dict]:
    points: list[dict] = []
    for recall_at, result_root in RESULT_ROOTS.items():
        if not result_root.is_dir():
            raise SystemExit(f"Result directory not found: {result_root}")
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                path = result_root / dataset / f"{method}.txt"
                if not path.is_file():
                    raise SystemExit(f"Result file not found: {path}")
                points.extend(parse_result_file(recall_at, dataset, method, path))
    return points


def build_payload(points: list[dict]) -> dict:
    grouped = {
        str(recall_at): {
            dataset: {method: [] for method in METHOD_ORDER}
            for dataset in DATASET_ORDER
        }
        for recall_at in RESULT_ROOTS
    }
    for point in points:
        grouped[str(point["recall_at"])][point["dataset"]][point["method"]].append(
            {
                "paramName": point["param_name"],
                "paramValue": point["param_value"],
                "recall": point["recall"],
                "qps": point["qps"],
            }
        )

    for recall_at in RESULT_ROOTS:
        for dataset in DATASET_ORDER:
            for method in METHOD_ORDER:
                grouped[str(recall_at)][dataset][method].sort(
                    key=lambda row: (row["recall"], row["qps"])
                )

    return {
        "recallAts": list(RESULT_ROOTS),
        "datasets": DATASET_ORDER,
        "datasetLabels": DATASET_LABELS,
        "methods": METHOD_ORDER,
        "methodLabels": METHOD_LABELS,
        "methodColors": METHOD_COLORS,
        "methodMarkers": METHOD_MARKERS,
        "points": grouped,
    }


def write_summary(points: list[dict]) -> None:
    with SUMMARY_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "recall_at",
                "dataset",
                "method",
                "param_name",
                "param_value",
                "recall",
                "qps",
                "source",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for point in sorted(
            points,
            key=lambda row: (
                row["recall_at"],
                DATASET_ORDER.index(row["dataset"]),
                METHOD_ORDER.index(row["method"]),
                row["recall"],
            ),
        ):
            writer.writerow(point)


def build_html(payload: dict) -> str:
    payload_json = html.escape(json.dumps(payload, separators=(",", ":")), quote=False)
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mixed-alpha Recall@50, Recall@100, and Recall@200</title>
<style>
:root { color-scheme: light; --paper: #fff; --page: #f5f5f5; --ink: #111; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 20px; background: var(--page); color: var(--ink);
  font-family: "Times New Roman", Times, serif;
}
.toolbar {
  width: min(1480px, 100%); margin: 0 auto 12px; display: flex;
  justify-content: flex-end; gap: 8px;
}
button {
  appearance: none; border: 1px solid #c7c7c7; border-radius: 4px; padding: 7px 12px;
  background: #fff; color: #111; font: 14px "Times New Roman", Times, serif; cursor: pointer;
}
button:hover { background: #f7f7f7; }
.page {
  width: min(1480px, 100%); margin: 0 auto; padding: 10px; background: var(--paper);
  border: 1px solid #ddd;
}
svg { display: block; width: 100%; height: auto; }
@page { size: 11in 7.3in; margin: 0.08in; }
@media print {
  body { padding: 0; background: #fff; }
  .toolbar { display: none; }
  .page { width: 100%; padding: 0; border: 0; }
}
</style>
</head>
<body>
<div class="toolbar">
  <button type="button" onclick="window.print()">Print PDF</button>
  <button type="button" onclick="downloadSvg()">Save SVG</button>
</div>
<main class="page" aria-label="Combined mixed-alpha QPS-Recall figure">
  <svg id="figure" viewBox="0 0 1480 924" role="img"
       aria-labelledby="figure-title figure-desc"></svg>
</main>
<script id="plot-data" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById("plot-data").textContent);
const SVG_NS = "http://www.w3.org/2000/svg";

function node(name, attrs = {}, text = null) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== null) element.textContent = text;
  return element;
}

function xScale(recall, plotLeft, plotWidth) {
  return plotLeft + ((recall - 0.7) / 0.3) * plotWidth;
}

function yScale(qps, yMin, yMax, plotTop, plotHeight) {
  const logMin = Math.log10(yMin);
  const logMax = Math.log10(yMax);
  return plotTop + (logMax - Math.log10(qps)) / (logMax - logMin) * plotHeight;
}

function logTicks(yMin, yMax) {
  const ticks = [];
  const start = Math.floor(Math.log10(yMin));
  const end = Math.ceil(Math.log10(yMax));
  for (let exp = start; exp <= end; exp++) {
    const value = Math.pow(10, exp);
    if (value >= yMin * 0.999 && value <= yMax * 1.001) ticks.push(value);
  }
  return ticks;
}

function formatYTick(value) {
  if (value >= 1000) return "10³";
  if (value >= 100) return "10²";
  if (value >= 10) return "10¹";
  return "10⁰";
}

function drawMarker(parent, type, x, y, color, size = 5.2, tooltip = null) {
  const group = node("g", tooltip ? { class: "data-marker" } : {});
  if (tooltip) group.appendChild(node("title", {}, tooltip));
  if (type === "square") {
    group.appendChild(node("rect", {
      x: x - size / 2, y: y - size / 2, width: size, height: size,
      fill: color, stroke: "#111111", "stroke-width": 0.5,
    }));
  } else if (type === "circle") {
    group.appendChild(node("circle", {
      cx: x, cy: y, r: size / 2, fill: color, stroke: "#111111", "stroke-width": 0.5,
    }));
  } else if (type === "triangle") {
    group.appendChild(node("polygon", {
      points: `${x},${y - size / 2} ${x - size / 2},${y + size / 2} ${x + size / 2},${y + size / 2}`,
      fill: color, stroke: "#111111", "stroke-width": 0.5,
    }));
  } else if (type === "pentagon") {
    const points = [];
    for (let i = 0; i < 5; i++) {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / 5;
      points.push(`${x + Math.cos(angle) * size / 2},${y + Math.sin(angle) * size / 2}`);
    }
    group.appendChild(node("polygon", {
      points: points.join(" "), fill: color, stroke: "#111111", "stroke-width": 0.5,
    }));
  } else {
    group.appendChild(node("line", {
      x1: x - size / 2, y1: y - size / 2, x2: x + size / 2, y2: y + size / 2,
      stroke: color, "stroke-width": 2,
    }));
    group.appendChild(node("line", {
      x1: x - size / 2, y1: y + size / 2, x2: x + size / 2, y2: y - size / 2,
      stroke: color, "stroke-width": 2,
    }));
  }
  parent.appendChild(group);
}

function drawLegend(svg, width) {
  const y = 51;
  const labelWidths = { hnsw: 56, sindi: 52, hnsw_sindi: 92, fhg: 40, uhg: 42 };
  const gap = 33;
  const itemWidths = DATA.methods.map((method) => 22 + labelWidths[method]);
  const totalWidth = itemWidths.reduce((sum, value) => sum + value, 0) +
    gap * (DATA.methods.length - 1);
  let x = width / 2 - totalWidth / 2;
  DATA.methods.forEach((method, index) => {
    drawMarker(svg, DATA.methodMarkers[method], x + 8, y - 3, DATA.methodColors[method], 9);
    svg.appendChild(node("text", { x: x + 21, y, class: "legend" }, DATA.methodLabels[method]));
    x += itemWidths[index] + gap;
  });
}

function panelYBounds(dataset) {
  const qpsValues = [];
  DATA.recallAts.forEach((recallAt) => {
    DATA.methods.forEach((method) => {
      DATA.points[String(recallAt)][dataset][method].forEach((point) => qpsValues.push(point.qps));
    });
  });
  const minQps = Math.min(...qpsValues);
  const maxQps = Math.max(...qpsValues);
  return {
    min: Math.pow(10, Math.floor(Math.log10(minQps))),
    max: Math.pow(10, Math.ceil(Math.log10(maxQps * 1.12))),
  };
}

function drawPanel(svg, recallAt, dataset, panelIndex, panelX, panelY, panelWidth, panelHeight) {
  const plotLeft = panelX + 42;
  const plotTop = panelY + 12;
  const plotWidth = panelWidth - 50;
  const plotHeight = panelHeight - 66;
  const plotBottom = plotTop + plotHeight;
  const bounds = panelYBounds(dataset);
  const clipId = `clip-mixed-${recallAt}-${dataset}`;

  const clip = node("clipPath", { id: clipId });
  clip.appendChild(node("rect", { x: plotLeft, y: plotTop, width: plotWidth, height: plotHeight }));
  svg.querySelector("defs").appendChild(clip);
  svg.appendChild(node("rect", {
    x: plotLeft, y: plotTop, width: plotWidth, height: plotHeight,
    fill: "#ffffff", stroke: "#111111", "stroke-width": 1,
  }));

  [0.7, 0.8, 0.9, 1.0].forEach((tick) => {
    const x = xScale(tick, plotLeft, plotWidth);
    svg.appendChild(node("line", {
      x1: x, y1: plotTop, x2: x, y2: plotBottom,
      class: tick === 0.7 || tick === 1.0 ? "axis-grid-edge" : "grid-line",
    }));
    svg.appendChild(node("text", {
      x, y: plotBottom + 13, "text-anchor": "middle", class: "tick",
    }, tick.toFixed(1)));
  });

  logTicks(bounds.min, bounds.max).forEach((tick) => {
    const y = yScale(tick, bounds.min, bounds.max, plotTop, plotHeight);
    svg.appendChild(node("line", {
      x1: plotLeft, y1: y, x2: plotLeft + plotWidth, y2: y, class: "grid-line",
    }));
    svg.appendChild(node("text", {
      x: plotLeft - 7, y: y + 3, "text-anchor": "end", class: "tick",
    }, formatYTick(tick)));
  });

  const seriesGroup = node("g", { "clip-path": `url(#${clipId})` });
  svg.appendChild(seriesGroup);
  DATA.methods.forEach((method) => {
    const points = DATA.points[String(recallAt)][dataset][method];
    const color = DATA.methodColors[method];
    const coords = points.map((point) => [
      xScale(point.recall, plotLeft, plotWidth),
      yScale(point.qps, bounds.min, bounds.max, plotTop, plotHeight),
    ]);
    seriesGroup.appendChild(node("polyline", {
      points: coords.map(([x, y]) => `${x},${y}`).join(" "),
      fill: "none", stroke: color, "stroke-width": method === "uhg" ? 2.4 : 1.2,
    }));
    points.forEach((point, index) => {
      const parameter = point.paramName
        ? `${point.paramName}=${point.paramValue}`
        : "parameter unavailable";
      const tooltip = `${DATA.methodLabels[method]}\n${parameter}\nRecall@${recallAt}=${point.recall}\nQPS=${point.qps}`;
      drawMarker(
        seriesGroup,
        DATA.methodMarkers[method],
        coords[index][0],
        coords[index][1],
        color,
        method === "uhg" ? 7.4 : 5.4,
        tooltip
      );
    });
  });

  svg.appendChild(node("text", {
    x: plotLeft + plotWidth / 2, y: plotBottom + 29,
    "text-anchor": "middle", class: "axis-label",
  }, `Recall@${recallAt}`));
  svg.appendChild(node("text", {
    x: panelX + 9, y: plotTop + plotHeight / 2,
    transform: `rotate(-90 ${panelX + 9} ${plotTop + plotHeight / 2})`,
    "text-anchor": "middle", class: "axis-label",
  }, "QPS"));
  svg.appendChild(node("text", {
    x: plotLeft + plotWidth / 2, y: panelY + panelHeight - 3,
    "text-anchor": "middle", class: "panel-label",
  }, `(${String.fromCharCode(97 + panelIndex)}) ${DATA.datasetLabels[dataset]}`));
}

function draw() {
  const svg = document.getElementById("figure");
  const width = 1480;
  const height = 924;
  const panelWidth = 260;
  const panelHeight = 252;
  const colGap = 16;
  const rowGap = 36;
  const startX = (width - DATA.datasets.length * panelWidth -
    (DATA.datasets.length - 1) * colGap) / 2;
  const startY = 82;
  svg.replaceChildren();

  const defs = node("defs");
  const style = node("style");
  style.textContent = `
    .figure-heading { font: bold 18px "Times New Roman", Times, serif; fill: #111111; }
    .legend { font: 17px "Times New Roman", Times, serif; fill: #111111; }
    .tick { font: 11px "Times New Roman", Times, serif; fill: #111111; }
    .axis-label { font: 13px "Times New Roman", Times, serif; fill: #111111; }
    .panel-label { font: 15px "Times New Roman", Times, serif; fill: #111111; }
    .grid-line { stroke: #8f8f8f; stroke-width: 0.75; stroke-dasharray: 2 2; }
    .axis-grid-edge { stroke: #111111; stroke-width: 0.9; }
    .data-marker { cursor: crosshair; }
  `;
  defs.appendChild(style);
  svg.appendChild(defs);
  svg.appendChild(node("title", { id: "figure-title" },
    "Mixed-alpha QPS-Recall curves for Recall@50, Recall@100, and Recall@200"));
  svg.appendChild(node("desc", { id: "figure-desc" },
    "Three rows compare five retrieval methods on five datasets under independently assigned query alpha values."));
  svg.appendChild(node("rect", { x: 0, y: 0, width, height, fill: "#ffffff" }));
  svg.appendChild(node("text", {
    x: width / 2, y: 21, "text-anchor": "middle", class: "figure-heading",
  }, "QPS–Recall Performance under Mixed Alpha"));
  drawLegend(svg, width);

  DATA.recallAts.forEach((recallAt, rowIndex) => {
    const rowY = startY + rowIndex * (panelHeight + rowGap);
    DATA.datasets.forEach((dataset, colIndex) => {
      drawPanel(
        svg,
        recallAt,
        dataset,
        rowIndex * DATA.datasets.length + colIndex,
        startX + colIndex * (panelWidth + colGap),
        rowY,
        panelWidth,
        panelHeight
      );
    });
  });
}

function downloadSvg() {
  const svg = document.getElementById("figure");
  const source = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "mixed_alpha_recall50_100_200.svg";
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
    return template.replace("__PAYLOAD__", payload_json)


def main() -> None:
    points = load_points()
    payload = build_payload(points)
    OUTPUT_PATH.write_text(build_html(payload))
    write_summary(points)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
