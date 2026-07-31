#!/usr/bin/env python3
"""Generate a Fig. 8 style HTML/SVG plot from UHG main results."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


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
PARAM_RE = re.compile(
    r"\b(?P<name>ef_search|bk|sindi_bk|dense_entry_bk)=(?P<value>[-+0-9.eE]+)"
)


def parse_param(line: str) -> str:
    params = {match.group("name"): match.group("value") for match in PARAM_RE.finditer(line)}
    if "ef_search" in params:
        return f"ef={params['ef_search']}"
    if "bk" in params:
        return f"bk={params['bk']}"
    if "sindi_bk" in params:
        return f"sindi_bk={params['sindi_bk']}"
    if "dense_entry_bk" in params:
        return f"dense_bk={params['dense_entry_bk']}"
    return ""


def load_raw_points(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            method_dir = results_root / dataset / method
            for result_file in sorted(method_dir.glob("alpha_*.txt")):
                alpha = result_file.stem.removeprefix("alpha_")
                for line in result_file.read_text(errors="replace").splitlines():
                    match = METRIC_RE.search(line)
                    if match is None:
                        continue
                    try:
                        recall = float(match.group("recall"))
                        qps = float(match.group("qps"))
                    except ValueError:
                        continue
                    if recall < 0.6 or qps <= 0:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "alpha": alpha,
                            "method": method,
                            "recall": recall,
                            "qps": qps,
                            "param": parse_param(line),
                        }
                    )
    return rows


def load_points(path: Path) -> list[dict]:
    lines = path.read_text(errors="replace").splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("# dataset\talpha\tmethod\tparam\trecall\tqps\tsource"):
            header_index = index
            break
    if header_index is None:
        raise SystemExit(f"TSV header not found: {path}")

    rows: list[dict] = []
    reader = csv.DictReader(
        [lines[header_index].lstrip("# ")] + lines[header_index + 1 :],
        delimiter="\t",
    )
    for row in reader:
        if not row or not row.get("dataset"):
            continue
        dataset = row["dataset"]
        method = row["method"]
        if dataset not in DATASET_ORDER or method not in METHOD_ORDER:
            continue
        try:
            recall = float(row["recall"])
            qps = float(row["qps"])
        except (TypeError, ValueError):
            continue
        if recall < 0.6 or qps <= 0:
            continue
        rows.append(
            {
                "dataset": dataset,
                "alpha": row["alpha"],
                "method": method,
                "recall": recall,
                "qps": qps,
                "param": row.get("param", ""),
            }
        )
    return rows


def build_payload(points: list[dict]) -> dict:
    alphas = sorted({point["alpha"] for point in points}, key=lambda value: float(value))
    grouped: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for dataset in DATASET_ORDER:
        grouped[dataset] = {alpha: {method: [] for method in METHOD_ORDER} for alpha in alphas}
    for point in points:
        grouped[point["dataset"]][point["alpha"]][point["method"]].append(
            {
                "recall": round(point["recall"], 6),
                "qps": round(point["qps"], 6),
                "param": point["param"],
            }
        )

    for dataset in DATASET_ORDER:
        for alpha in alphas:
            for method in METHOD_ORDER:
                grouped[dataset][alpha][method].sort(key=lambda row: (row["recall"], row["qps"]))

    return {
        "datasets": DATASET_ORDER,
        "datasetLabels": DATASET_LABELS,
        "alphas": alphas,
        "methods": METHOD_ORDER,
        "methodLabels": METHOD_LABELS,
        "methodColors": METHOD_COLORS,
        "methodMarkers": METHOD_MARKERS,
        "points": grouped,
    }


def build_html(payload: dict) -> str:
    payload_json = html.escape(json.dumps(payload, separators=(",", ":")), quote=False)
    figure_height = 80 + len(payload["datasets"]) * 253 + (len(payload["datasets"]) - 1) * 35 + 22
    print_width = min(10.05, round(8.2 * 1480 / figure_height, 2))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Main Results Fig. 8 Style</title>
<style>
:root {{
  color-scheme: light;
  --page: #f5f5f5;
  --paper: #ffffff;
  --ink: #111111;
}}
* {{
  box-sizing: border-box;
}}
body {{
  margin: 0;
  min-height: 100vh;
  padding: 20px;
  background: var(--page);
  color: var(--ink);
  font-family: "Times New Roman", Times, serif;
}}
.toolbar {{
  width: min(1480px, 100%);
  margin: 0 auto 12px;
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}}
button {{
  appearance: none;
  border: 1px solid #c7c7c7;
  border-radius: 4px;
  padding: 7px 12px;
  background: #ffffff;
  color: #111111;
  font: 14px "Times New Roman", Times, serif;
  cursor: pointer;
}}
button:hover {{
  background: #f7f7f7;
}}
.page {{
  width: min(1480px, 100%);
  margin: 0 auto;
  padding: 10px;
  background: var(--paper);
  border: 1px solid #dddddd;
}}
svg {{
  display: block;
  width: 100%;
  height: auto;
}}
@page {{
  size: 11in 8.5in;
  margin: 0.08in;
}}
@media print {{
  html,
  body {{
    width: 100%;
    height: 100%;
    overflow: hidden;
  }}
  body {{
    min-height: 0;
    padding: 0;
    background: #ffffff;
  }}
  .toolbar {{
    display: none;
  }}
  .page {{
    width: {print_width}in;
    max-width: 100%;
    margin: 0 auto;
    padding: 0;
    border: 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
}}
</style>
</head>
<body>
<div class="toolbar">
  <button type="button" onclick="window.print()">Print PDF</button>
  <button type="button" onclick="downloadSvg()">Save SVG</button>
</div>
<main class="page" aria-label="Main results figure">
  <svg id="figure" viewBox="0 0 1480 {figure_height}" role="img"
       aria-labelledby="figure-title figure-desc"></svg>
</main>
<script id="plot-data" type="application/json">{payload_json}</script>
<script>
const DATA = JSON.parse(document.getElementById("plot-data").textContent);
const SVG_NS = "http://www.w3.org/2000/svg";

function node(name, attrs = {{}}, text = null) {{
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {{
    element.setAttribute(key, value);
  }}
  if (text !== null) element.textContent = text;
  return element;
}}

function xScale(recall, plotLeft, plotWidth) {{
  return plotLeft + ((recall - 0.7) / 0.3) * plotWidth;
}}

function yScale(qps, yMin, yMax, plotTop, plotHeight) {{
  const logMin = Math.log10(yMin);
  const logMax = Math.log10(yMax);
  return plotTop + (logMax - Math.log10(qps)) / (logMax - logMin) * plotHeight;
}}

function logTicks(yMin, yMax) {{
  const ticks = [];
  const start = Math.floor(Math.log10(yMin));
  const end = Math.ceil(Math.log10(yMax));
  for (let exp = start; exp <= end; exp++) {{
    const value = Math.pow(10, exp);
    if (value >= yMin * 0.999 && value <= yMax * 1.001) ticks.push(value);
  }}
  return ticks;
}}

function formatYTick(value) {{
  if (value >= 1000) return "10³";
  if (value >= 100) return "10²";
  if (value >= 10) return "10¹";
  return "10⁰";
}}

function drawMarker(svg, type, x, y, color, size = 5.2) {{
  if (type === "square") {{
    svg.appendChild(node("rect", {{
      x: x - size / 2,
      y: y - size / 2,
      width: size,
      height: size,
      fill: color,
      stroke: "#111111",
      "stroke-width": 0.5,
    }}));
  }} else if (type === "circle") {{
    svg.appendChild(node("circle", {{
      cx: x,
      cy: y,
      r: size / 2,
      fill: color,
      stroke: "#111111",
      "stroke-width": 0.5,
    }}));
  }} else if (type === "triangle") {{
    const points = [
      `${{x}},${{y - size / 2}}`,
      `${{x - size / 2}},${{y + size / 2}}`,
      `${{x + size / 2}},${{y + size / 2}}`,
    ].join(" ");
    svg.appendChild(node("polygon", {{
      points,
      fill: color,
      stroke: "#111111",
      "stroke-width": 0.5,
    }}));
  }} else if (type === "pentagon") {{
    const points = [];
    for (let i = 0; i < 5; i++) {{
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / 5;
      points.push(`${{x + Math.cos(angle) * size / 2}},${{y + Math.sin(angle) * size / 2}}`);
    }}
    svg.appendChild(node("polygon", {{
      points: points.join(" "),
      fill: color,
      stroke: "#111111",
      "stroke-width": 0.5,
    }}));
  }} else {{
    svg.appendChild(node("line", {{
      x1: x - size / 2,
      y1: y - size / 2,
      x2: x + size / 2,
      y2: y + size / 2,
      stroke: color,
      "stroke-width": 2,
    }}));
    svg.appendChild(node("line", {{
      x1: x - size / 2,
      y1: y + size / 2,
      x2: x + size / 2,
      y2: y - size / 2,
      stroke: color,
      "stroke-width": 2,
    }}));
  }}
}}

function drawLegend(svg, width) {{
  const y = 51;
  const labelWidths = {{
    hnsw: 56,
    sindi: 52,
    hnsw_sindi: 92,
    fhg: 40,
    uhg: 42,
  }};
  const gap = 33;
  const itemWidths = DATA.methods.map((method) => 22 + labelWidths[method]);
  const totalWidth = itemWidths.reduce((sum, value) => sum + value, 0) + gap * (DATA.methods.length - 1);
  let x = width / 2 - totalWidth / 2;

  DATA.methods.forEach((method, index) => {{
    const color = DATA.methodColors[method];
    drawMarker(svg, DATA.methodMarkers[method], x + 8, y - 3, color, 9);
    svg.appendChild(node("text", {{
      x: x + 21,
      y: y,
      class: "legend",
    }}, DATA.methodLabels[method]));
    x += itemWidths[index] + gap;
  }});
}}

function panelYBounds(dataset, alpha) {{
  const qpsValues = [];
  DATA.methods.forEach((method) => {{
    DATA.points[dataset][alpha][method].forEach((point) => qpsValues.push(point.qps));
  }});
  const maxQps = Math.max(...qpsValues, 10);
  const upper = Math.pow(10, Math.ceil(Math.log10(maxQps * 1.12)));
  return {{ min: 10, max: upper }};
}}

function drawPanel(svg, dataset, alpha, panelIndex, panelX, panelY, panelWidth, panelHeight) {{
  const plotLeft = panelX + 44;
  const plotTop = panelY + 13;
  const plotWidth = panelWidth - 54;
  const plotHeight = panelHeight - 59;
  const plotBottom = plotTop + plotHeight;
  const bounds = panelYBounds(dataset, alpha);
  const clipId = `clip-${{dataset}}-${{alpha.replace(".", "_")}}`;

  const defs = svg.querySelector("defs");
  const clip = node("clipPath", {{ id: clipId }});
  clip.appendChild(node("rect", {{
    x: plotLeft,
    y: plotTop,
    width: plotWidth,
    height: plotHeight,
  }}));
  defs.appendChild(clip);

  svg.appendChild(node("rect", {{
    x: plotLeft,
    y: plotTop,
    width: plotWidth,
    height: plotHeight,
    fill: "#ffffff",
    stroke: "#111111",
    "stroke-width": 1,
  }}));

  [0.7, 0.8, 0.9, 1.0].forEach((tick) => {{
    const x = xScale(tick, plotLeft, plotWidth);
    svg.appendChild(node("line", {{
      x1: x,
      y1: plotTop,
      x2: x,
      y2: plotBottom,
      class: tick === 0.7 || tick === 1.0 ? "axis-grid-edge" : "grid-line",
    }}));
    svg.appendChild(node("text", {{
      x,
      y: plotBottom + 13,
      "text-anchor": "middle",
      class: "tick",
    }}, tick.toFixed(1)));
  }});

  logTicks(bounds.min, bounds.max).forEach((tick) => {{
    const y = yScale(tick, bounds.min, bounds.max, plotTop, plotHeight);
    svg.appendChild(node("line", {{
      x1: plotLeft,
      y1: y,
      x2: plotLeft + plotWidth,
      y2: y,
      class: "grid-line",
    }}));
    svg.appendChild(node("text", {{
      x: plotLeft - 7,
      y: y + 3,
      "text-anchor": "end",
      class: "tick",
    }}, formatYTick(tick)));
  }});

  const seriesGroup = node("g", {{ "clip-path": `url(#${{clipId}})` }});
  svg.appendChild(seriesGroup);

  DATA.methods.forEach((method) => {{
    const points = DATA.points[dataset][alpha][method];
    if (!points.length) return;
    const color = DATA.methodColors[method];
    const coords = points.map((point) => [
      xScale(point.recall, plotLeft, plotWidth),
      yScale(point.qps, bounds.min, bounds.max, plotTop, plotHeight),
    ]);
    seriesGroup.appendChild(node("polyline", {{
      points: coords.map(([x, y]) => `${{x}},${{y}}`).join(" "),
      fill: "none",
      stroke: color,
      "stroke-width": method === "uhg" ? 2.4 : 1.2,
    }}));
    coords.forEach(([x, y]) => drawMarker(seriesGroup, DATA.methodMarkers[method], x, y, color, method === "uhg" ? 7.4 : 5.4));
  }});

  svg.appendChild(node("text", {{
    x: plotLeft + plotWidth / 2,
    y: plotBottom + 29,
    "text-anchor": "middle",
    class: "axis-label",
  }}, "Recall"));
  svg.appendChild(node("text", {{
    x: panelX + 11,
    y: plotTop + plotHeight / 2,
    transform: `rotate(-90 ${{panelX + 11}} ${{plotTop + plotHeight / 2}})`,
    "text-anchor": "middle",
    class: "axis-label",
  }}, "QPS"));
  svg.appendChild(node("text", {{
    x: plotLeft + plotWidth / 2,
    y: plotBottom + 54,
    "text-anchor": "middle",
    class: "panel-label",
  }}, `(${{String.fromCharCode(97 + panelIndex)}}) ${{DATA.datasetLabels[dataset]}} α=${{alpha}}`));
}}

function draw() {{
  const svg = document.getElementById("figure");
  const width = 1480;
  const height = {figure_height};
  const panelWidth = 274;
  const panelHeight = 253;
  const colGap = 18;
  const rowGap = 35;
  const startX = (width - DATA.alphas.length * panelWidth - (DATA.alphas.length - 1) * colGap) / 2;
  const startY = 80;

  svg.replaceChildren();

  const defs = node("defs");
  const style = node("style");
  style.textContent = `
    .figure-heading {{ font: bold 18px "Times New Roman", Times, serif; fill: #111111; }}
    .legend {{ font: 17px "Times New Roman", Times, serif; fill: #111111; }}
    .tick {{ font: 11px "Times New Roman", Times, serif; fill: #111111; }}
    .axis-label {{ font: 13px "Times New Roman", Times, serif; fill: #111111; }}
    .panel-label {{ font: 15px "Times New Roman", Times, serif; fill: #111111; }}
    .grid-line {{ stroke: #8f8f8f; stroke-width: 0.75; stroke-dasharray: 2 2; }}
    .axis-grid-edge {{ stroke: #111111; stroke-width: 0.9; }}
  `;
  defs.appendChild(style);
  svg.appendChild(defs);

  svg.appendChild(node("title", {{ id: "figure-title" }}, "QPS–Recall Performance under Fixed Alpha"));
  svg.appendChild(node("desc", {{ id: "figure-desc" }}, "Fig. 8 style QPS versus Recall curves for all UHG main-result methods across datasets and alpha values."));
  svg.appendChild(node("rect", {{ x: 0, y: 0, width, height, fill: "#ffffff" }}));

  svg.appendChild(node("text", {{
    x: width / 2, y: 21, "text-anchor": "middle", class: "figure-heading",
  }}, "QPS–Recall Performance under Fixed Alpha"));
  drawLegend(svg, width);

  let panelIndex = 0;
  DATA.datasets.forEach((dataset, rowIndex) => {{
    DATA.alphas.forEach((alpha, colIndex) => {{
      drawPanel(
        svg,
        dataset,
        alpha,
        panelIndex,
        startX + colIndex * (panelWidth + colGap),
        startY + rowIndex * (panelHeight + rowGap),
        panelWidth,
        panelHeight
      );
      panelIndex += 1;
    }});
  }});
}}

function downloadSvg() {{
  const svg = document.getElementById("figure");
  const source = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([source], {{ type: "image/svg+xml;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "main_results_fig8.svg";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}}

draw();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional integrated main_results.txt path (overrides --results-root).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).parent,
        help="Raw results/main directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("main_results_fig8.html"),
        help="Output HTML path.",
    )
    args = parser.parse_args()

    points = load_points(args.input) if args.input else load_raw_points(args.results_root)
    if not points:
        source = args.input if args.input else args.results_root
        raise SystemExit(f"No valid Recall/QPS points found in {source}")
    payload = build_payload(points)
    args.output.write_text(build_html(payload))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
