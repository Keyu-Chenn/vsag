#!/usr/bin/env python3
"""Build a self-contained interactive QPS-Recall HTML for multi-k mixed alpha."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


PREFERRED_DATASET_ORDER = ("nq", "hotpotqa", "msmarco", "fever", "dbpedia-entity")
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "fever": "FEVER",
    "dbpedia-entity": "DBpedia-Entity",
}
METHOD_ORDER = ("hnsw", "sindi", "hnsw_sindi", "fhg", "uhg")
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
METHOD_PARAMETERS = {
    "hnsw": "ef_search",
    "sindi": "bk",
    "hnsw_sindi": "point",
    "fhg": "ef_search",
    "uhg": "ef_search",
}
REQUIRED_COLUMNS = {
    "recall_at",
    "dataset",
    "method",
    "point",
    "recall",
    "qps",
}


@dataclass(frozen=True)
class PlotConfig:
    input_tsv: Path
    output_html: Path
    title: str
    subtitle: str
    download_stem: str
    default_recall: int | None = None


def dataset_sort_key(dataset: str) -> tuple[int, int | str]:
    if dataset in PREFERRED_DATASET_ORDER:
        return (0, PREFERRED_DATASET_ORDER.index(dataset))
    return (1, dataset)


def read_points(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input TSV does not exist: {path}")

    points: list[dict[str, object]] = []
    seen: set[tuple[int, str, str, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing TSV columns in {path}: {missing}")

        for line_number, row in enumerate(reader, 2):
            try:
                recall_at = int(row["recall_at"])
                dataset = row["dataset"]
                method = row["method"]
                point = int(row["point"])
                recall = float(row["recall"])
                qps = float(row["qps"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row at {path}:{line_number}: {row}") from exc

            if method not in METHOD_ORDER:
                raise ValueError(f"Unknown method at {path}:{line_number}: {method}")
            if recall_at <= 0 or point <= 0:
                raise ValueError(f"Non-positive k/point at {path}:{line_number}")
            if not math.isfinite(recall) or not 0.0 <= recall <= 1.0:
                raise ValueError(f"Invalid recall at {path}:{line_number}: {recall}")
            if not math.isfinite(qps) or qps <= 0.0:
                raise ValueError(f"Invalid QPS at {path}:{line_number}: {qps}")

            key = (recall_at, dataset, method, point)
            if key in seen:
                raise ValueError(f"Duplicate point at {path}:{line_number}: {key}")
            seen.add(key)
            points.append(
                {
                    "recall_at": recall_at,
                    "dataset": dataset,
                    "method": method,
                    "point": point,
                    "recall": recall,
                    "qps": qps,
                }
            )

    if not points:
        raise ValueError(f"No data rows found in {path}")
    return points


def validate_coverage(points: list[dict[str, object]]) -> None:
    recalls = sorted({int(point["recall_at"]) for point in points})
    datasets = sorted(
        {str(point["dataset"]) for point in points}, key=dataset_sort_key
    )
    combinations = {
        (
            int(point["recall_at"]),
            str(point["dataset"]),
            str(point["method"]),
        )
        for point in points
    }
    missing = [
        (recall_at, dataset, method)
        for recall_at in recalls
        for dataset in datasets
        for method in METHOD_ORDER
        if (recall_at, dataset, method) not in combinations
    ]
    if missing:
        preview = ", ".join(
            f"Recall@{recall_at}/{dataset}/{method}"
            for recall_at, dataset, method in missing[:10]
        )
        raise ValueError(f"Incomplete plot coverage ({len(missing)} missing): {preview}")


def build_payload(points: list[dict[str, object]], config: PlotConfig) -> dict:
    recalls = sorted({int(point["recall_at"]) for point in points})
    datasets = sorted(
        {str(point["dataset"]) for point in points}, key=dataset_sort_key
    )
    default_recall = config.default_recall
    if default_recall is None:
        default_recall = 100 if 100 in recalls else recalls[0]
    if default_recall not in recalls:
        raise ValueError(
            f"Default Recall@{default_recall} is absent; available values: {recalls}"
        )

    grouped = {
        str(recall_at): {
            dataset: {method: [] for method in METHOD_ORDER}
            for dataset in datasets
        }
        for recall_at in recalls
    }
    point_counts = {str(recall_at): 0 for recall_at in recalls}
    for point in points:
        recall_at = str(point["recall_at"])
        dataset = str(point["dataset"])
        method = str(point["method"])
        grouped[recall_at][dataset][method].append(
            {
                "point": int(point["point"]),
                "recall": round(float(point["recall"]), 9),
                "qps": round(float(point["qps"]), 9),
            }
        )
        point_counts[recall_at] += 1

    for recall_at in grouped.values():
        for dataset in recall_at.values():
            for method_points in dataset.values():
                method_points.sort(key=lambda row: (row["recall"], -row["qps"]))

    return {
        "title": config.title,
        "subtitle": config.subtitle,
        "downloadStem": config.download_stem,
        "sourceTsv": str(config.input_tsv.resolve()),
        "recallValues": recalls,
        "defaultRecall": default_recall,
        "datasets": datasets,
        "datasetLabels": {
            dataset: DATASET_LABELS.get(dataset, dataset) for dataset in datasets
        },
        "methods": list(METHOD_ORDER),
        "methodLabels": METHOD_LABELS,
        "methodColors": METHOD_COLORS,
        "methodMarkers": METHOD_MARKERS,
        "methodParameters": METHOD_PARAMETERS,
        "pointCounts": point_counts,
        "totalPoints": len(points),
        "figureHeight": 376,
        "points": grouped,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__DOCUMENT_TITLE__</title>
<style>
:root { color-scheme: light; --paper: #fff; --page: #f5f5f5; --ink: #111; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 18px; background: var(--page); color: var(--ink);
  font-family: "Times New Roman", Times, serif;
}
.toolbar {
  width: min(1480px, 100%); margin: 0 auto 10px; display: flex;
  align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
}
.controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
label { font-size: 14px; }
select, button {
  appearance: none; border: 1px solid #bdbdbd; border-radius: 4px;
  padding: 7px 11px; background: #fff; color: #111;
  font: 14px "Times New Roman", Times, serif;
}
button { cursor: pointer; }
button:hover, select:hover { background: #f8f8f8; }
.status { color: #555; font-size: 13px; }
.page {
  width: min(1480px, 100%); margin: 0 auto; padding: 8px; background: var(--paper);
  border: 1px solid #d7d7d7;
}
svg { display: block; width: 100%; height: auto; }
@page { size: 11in 3.5in; margin: 0.08in; }
@media print {
  body { padding: 0; background: #fff; }
  .toolbar { display: none; }
  .page { width: 100%; padding: 0; border: 0; }
}
</style>
</head>
<body>
<div class="toolbar">
  <div class="controls">
    <label for="recall-select">Metric</label>
    <select id="recall-select" aria-label="Recall top-k"></select>
  </div>
  <div class="controls">
    <span id="status" class="status"></span>
    <button type="button" id="print-button">Print PDF</button>
    <button type="button" id="save-button">Save SVG</button>
  </div>
</div>
<main class="page" aria-label="Mixed-alpha QPS-Recall figure">
  <svg id="figure" viewBox="0 0 1480 __FIGURE_HEIGHT__" role="img"
       aria-labelledby="figure-title figure-desc"></svg>
</main>
<script id="plot-data" type="application/json">__PLOT_DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("plot-data").textContent);
const SVG_NS = "http://www.w3.org/2000/svg";
const STATE = {
  recallAt: String(DATA.defaultRecall),
};

function node(name, attrs = {}, text = null) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== null) element.textContent = text;
  return element;
}

function xScale(recall, xMin, xMax, plotLeft, plotWidth) {
  return plotLeft + ((recall - xMin) / (xMax - xMin)) * plotWidth;
}

function yScale(qps, bounds, plotTop, plotHeight) {
  const logMin = Math.log10(bounds.min);
  const logMax = Math.log10(bounds.max);
  return plotTop + (logMax - Math.log10(qps)) / (logMax - logMin) * plotHeight;
}

function superscript(value) {
  const digits = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
                  "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"};
  return String(value).split("").map((digit) => digits[digit]).join("");
}

function formatLogTick(value) {
  return `10${superscript(Math.round(Math.log10(value)))}`;
}

function drawMarker(target, type, x, y, color, size = 5.4, tooltip = null) {
  const group = node("g", tooltip ? {class: "data-marker"} : {});
  if (tooltip) group.appendChild(node("title", {}, tooltip));
  if (type === "square") {
    group.appendChild(node("rect", {
      x: x - size / 2, y: y - size / 2, width: size, height: size,
      fill: color, stroke: "#111", "stroke-width": 0.5,
    }));
  } else if (type === "circle") {
    group.appendChild(node("circle", {
      cx: x, cy: y, r: size / 2, fill: color, stroke: "#111", "stroke-width": 0.5,
    }));
  } else if (type === "triangle") {
    group.appendChild(node("polygon", {
      points: `${x},${y - size / 2} ${x - size / 2},${y + size / 2} ${x + size / 2},${y + size / 2}`,
      fill: color, stroke: "#111", "stroke-width": 0.5,
    }));
  } else if (type === "pentagon") {
    const points = [];
    for (let index = 0; index < 5; index++) {
      const angle = -Math.PI / 2 + (2 * Math.PI * index) / 5;
      points.push(`${x + Math.cos(angle) * size / 2},${y + Math.sin(angle) * size / 2}`);
    }
    group.appendChild(node("polygon", {
      points: points.join(" "), fill: color, stroke: "#111", "stroke-width": 0.5,
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
  target.appendChild(group);
}

function pointsFor(recallAt) {
  return DATA.points[String(recallAt)];
}

function recallTicks() {
  return [0.7, 0.8, 0.9, 1.0];
}

function globalQpsBounds() {
  return {min: 10, max: 1000};
}

function qpsTicks(bounds) {
  const ticks = [];
  const start = Math.round(Math.log10(bounds.min));
  const end = Math.round(Math.log10(bounds.max));
  for (let exponent = start; exponent <= end; exponent++) ticks.push(10 ** exponent);
  return ticks;
}

function drawLegend(svg, width) {
  const y = 51;
  const labelWidths = {hnsw: 56, sindi: 52, hnsw_sindi: 92, fhg: 40, uhg: 42};
  const gap = 33;
  const widths = DATA.methods.map((method) => 22 + labelWidths[method]);
  const total = widths.reduce((sum, width) => sum + width, 0) + gap * (widths.length - 1);
  let x = width / 2 - total / 2;
  DATA.methods.forEach((method, index) => {
    drawMarker(svg, DATA.methodMarkers[method], x + 8, y - 4,
               DATA.methodColors[method], 9);
    svg.appendChild(node("text", {x: x + 21, y, class: "legend"},
                         DATA.methodLabels[method]));
    x += widths[index] + gap;
  });
}

function drawPanel(svg, recallAt, dataset, panelIndex, panelX, panelY, panelWidth,
                   panelHeight, recallDomain, qpsDomain) {
  const plotLeft = panelX + 44;
  const plotTop = panelY + 13;
  const plotWidth = panelWidth - 54;
  const plotHeight = panelHeight - 80;
  const plotBottom = plotTop + plotHeight;
  const clipId = `clip-${recallAt}-${dataset}`;
  const defs = svg.querySelector("defs");
  const clip = node("clipPath", {id: clipId});
  clip.appendChild(node("rect", {
    x: plotLeft, y: plotTop, width: plotWidth, height: plotHeight,
  }));
  defs.appendChild(clip);

  svg.appendChild(node("rect", {
    x: plotLeft, y: plotTop, width: plotWidth, height: plotHeight,
    fill: "#fff", stroke: "#111", "stroke-width": 1,
  }));

  recallTicks().forEach((tick) => {
    const x = xScale(tick, recallDomain.min, recallDomain.max, plotLeft, plotWidth);
    svg.appendChild(node("line", {
      x1: x, y1: plotTop, x2: x, y2: plotBottom,
      class: "grid-line",
    }));
    svg.appendChild(node("text", {
      x, y: plotBottom + 13, "text-anchor": "middle", class: "tick",
    }, tick.toFixed(1)));
  });

  qpsTicks(qpsDomain).forEach((tick) => {
    const y = yScale(tick, qpsDomain, plotTop, plotHeight);
    svg.appendChild(node("line", {
      x1: plotLeft, y1: y, x2: plotLeft + plotWidth, y2: y,
      class: "grid-line",
    }));
    svg.appendChild(node("text", {
      x: plotLeft - 7, y: y + 3, "text-anchor": "end", class: "tick",
    }, formatLogTick(tick)));
  });

  const seriesGroup = node("g", {"clip-path": `url(#${clipId})`});
  svg.appendChild(seriesGroup);
  DATA.methods.forEach((method) => {
    const points = pointsFor(recallAt)[dataset][method];
    const color = DATA.methodColors[method];
    const coordinates = points.map((point) => [
      xScale(point.recall, recallDomain.min, recallDomain.max, plotLeft, plotWidth),
      yScale(point.qps, qpsDomain, plotTop, plotHeight),
    ]);
    seriesGroup.appendChild(node("polyline", {
      points: coordinates.map(([x, y]) => `${x},${y}`).join(" "),
      fill: "none", stroke: color,
      "stroke-width": method === "uhg" ? 2.4 : 1.25,
    }));
    points.forEach((point, index) => {
      const tooltip = `${DATA.datasetLabels[dataset]} · ${DATA.methodLabels[method]}
${DATA.methodParameters[method]}=${point.point}
Recall@${recallAt}=${point.recall}
QPS=${point.qps}`;
      drawMarker(seriesGroup, DATA.methodMarkers[method],
                 coordinates[index][0], coordinates[index][1], color,
                 method === "uhg" ? 7.4 : 5.6, tooltip);
    });
  });

  svg.appendChild(node("text", {
    x: plotLeft + plotWidth / 2, y: plotBottom + 29,
    "text-anchor": "middle", class: "axis-label",
  }, `Recall@${recallAt}`));
  svg.appendChild(node("text", {
    x: panelX + 11, y: plotTop + plotHeight / 2,
    transform: `rotate(-90 ${panelX + 11} ${plotTop + plotHeight / 2})`,
    "text-anchor": "middle", class: "axis-label",
  }, "QPS"));
  svg.appendChild(node("text", {
    x: plotLeft + plotWidth / 2, y: panelY + panelHeight - 5,
    "text-anchor": "middle", class: "panel-label",
  }, `(${String.fromCharCode(97 + panelIndex)}) ${DATA.datasetLabels[dataset]}`));
}

function draw() {
  const svg = document.getElementById("figure");
  const width = 1480;
  const panelWidth = 274;
  const panelHeight = 274;
  const colGap = 18;
  const startY = 80;
  const height = DATA.figureHeight;
  const usedWidth = panelWidth * DATA.datasets.length +
    colGap * (DATA.datasets.length - 1);
  const startX = (width - usedWidth) / 2;
  const recallDomain = {min: 0.7, max: 1.0};
  const qpsDomain = globalQpsBounds();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();

  const defs = node("defs");
  const style = node("style");
  style.textContent = `
    .figure-heading { font: bold 18px "Times New Roman", Times, serif; fill: #111; }
    .figure-subtitle { font: 13px "Times New Roman", Times, serif; fill: #444; }
    .legend { font: 17px "Times New Roman", Times, serif; fill: #111; }
    .tick { font: 11px "Times New Roman", Times, serif; fill: #111; }
    .axis-label { font: 13px "Times New Roman", Times, serif; fill: #111; }
    .panel-label { font: 15px "Times New Roman", Times, serif; fill: #111; }
    .grid-line { stroke: #929292; stroke-width: 0.72; stroke-dasharray: 2 2; }
    .data-marker { cursor: crosshair; }
  `;
  defs.appendChild(style);
  svg.appendChild(defs);
  svg.appendChild(node("title", {id: "figure-title"},
                       `${DATA.title}, Recall@${STATE.recallAt}`));
  svg.appendChild(node("desc", {id: "figure-desc"},
                       "Five retrieval methods evaluated with one alpha per query."));
  svg.appendChild(node("rect", {x: 0, y: 0, width, height, fill: "#fff"}));
  svg.appendChild(node("text", {
    x: width / 2, y: 21, "text-anchor": "middle", class: "figure-heading",
  }, `${DATA.title} · Recall@${STATE.recallAt}`));
  drawLegend(svg, width);
  DATA.datasets.forEach((dataset, columnIndex) => {
    drawPanel(
      svg,
      Number(STATE.recallAt),
      dataset,
      columnIndex,
      startX + columnIndex * (panelWidth + colGap),
      startY,
      panelWidth,
      panelHeight,
      recallDomain,
      qpsDomain
    );
  });

  document.getElementById("status").textContent =
    `${DATA.datasets.length} datasets · ${DATA.pointCounts[STATE.recallAt]} points`;
}

function downloadSvg() {
  const svg = document.getElementById("figure");
  const source = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([source], {type: "image/svg+xml;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${DATA.downloadStem}_recall_${STATE.recallAt}.svg`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

const recallSelect = document.getElementById("recall-select");
DATA.recallValues.forEach((recallAt) => {
  const option = document.createElement("option");
  option.value = String(recallAt);
  option.textContent = `Recall@${recallAt}`;
  if (recallAt === DATA.defaultRecall) option.selected = true;
  recallSelect.appendChild(option);
});
recallSelect.addEventListener("change", (event) => {
  STATE.recallAt = event.target.value;
  draw();
});
document.getElementById("print-button").addEventListener("click", () => window.print());
document.getElementById("save-button").addEventListener("click", downloadSvg);
draw();
</script>
</body>
</html>
"""


def build_html(payload: dict) -> str:
    payload_json = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    return (
        HTML_TEMPLATE.replace("__DOCUMENT_TITLE__", str(payload["title"]))
        .replace("__FIGURE_HEIGHT__", str(payload["figureHeight"]))
        .replace("__PLOT_DATA__", payload_json)
    )


def generate_plot(config: PlotConfig) -> dict:
    points = read_points(config.input_tsv)
    validate_coverage(points)
    payload = build_payload(points, config)
    config.output_html.parent.mkdir(parents=True, exist_ok=True)
    config.output_html.write_text(build_html(payload), encoding="utf-8")
    print(
        f"Wrote {payload['totalPoints']} points, "
        f"{len(payload['datasets'])} datasets, "
        f"Recall@{'/'.join(map(str, payload['recallValues']))}: "
        f"{config.output_html}"
    )
    return payload
