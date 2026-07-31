#!/usr/bin/env python3
"""Plot UHG against UHG-S/UHG-H at a fixed, matched ef_search value."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


REPO_ROOT = Path("/tbase-project/vsag")
RESULT_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "ablation"
MAIN_RESULT_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "main"
OUTPUT_PATH = RESULT_ROOT / "ablation_fig8.html"

DATASET_ORDER = ["nq", "hotpotqa", "msmarco", "fever", "dbpedia-entity"]
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "fever": "FEVER",
    "dbpedia-entity": "DBpedia-Entity",
}
METHOD_ORDER = ["uhg", "uhgs", "uhgh"]
METHOD_LABELS = {
    "uhg": "UHG",
    "uhgs": "UHG-S",
    "uhgh": "UHG-H",
}
METHOD_COLORS = {
    "uhg": "#4B5563",
    "uhgs": "#0072B2",
    "uhgh": "#CC79A7",
}
METHOD_MARKERS = {
    "uhg": "pentagon",
    "uhgs": "square",
    "uhgh": "triangle",
}
COMPARISONS = {"0.3": "uhgs", "0.7": "uhgh"}

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"\b(?P<name>ef_search|bk)=(?P<value>[-+0-9.eE]+)")


def parse_param(line: str) -> str:
    params = {match.group("name"): match.group("value") for match in PARAM_RE.finditer(line)}
    if "ef_search" in params:
        return f"ef={params['ef_search']}"
    if "bk" in params:
        return f"bk={params['bk']}"
    return ""


def parse_result_file(dataset: str, method: str, path: Path) -> list[dict]:
    alpha = path.stem.removeprefix("alpha_")
    points: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        metric_match = METRIC_RE.search(line)
        if metric_match is None:
            continue
        recall = float(metric_match.group("recall"))
        qps = float(metric_match.group("qps"))
        if recall < 0.6 or qps <= 0:
            continue
        points.append(
            {
                "dataset": dataset,
                "method": method,
                "alpha": alpha,
                "recall": round(recall, 6),
                "qps": round(qps, 6),
                "param": parse_param(line),
            }
        )
    return points


def require_config(path: Path, *tokens: str) -> None:
    text = path.read_text(errors="replace")
    missing = [token for token in tokens if token not in text]
    if missing:
        raise SystemExit(
            f"Result file has stale or unexpected configuration: {path}; "
            f"missing {', '.join(missing)}"
        )


def load_points(
    result_root: Path,
    main_result_root: Path,
    expected_k: int,
) -> list[dict]:
    points: list[dict] = []
    for dataset in DATASET_ORDER:
        for alpha, entry_method in COMPARISONS.items():
            direct_path = result_root / dataset / "uhg" / f"alpha_{alpha}.txt"
            main_path = main_result_root / dataset / "uhg" / f"alpha_{alpha}.txt"
            if not direct_path.is_file() or not main_path.is_file():
                continue
            common_config = (
                f"k={expected_k}",
                "num_queries=-1",
                "build_alpha=0.5",
                "hybrid_prune_scale=0.3",
            )
            require_config(direct_path, "method=uhg", *common_config)
            require_config(main_path, "method=auto", *common_config)
            points.extend(parse_result_file(dataset, "uhg", direct_path))
            points.extend(parse_result_file(dataset, entry_method, main_path))
    return points


def build_payload(points: list[dict], target_ef: str) -> dict:
    datasets = [dataset for dataset in DATASET_ORDER if any(p["dataset"] == dataset for p in points)]
    alphas = sorted({point["alpha"] for point in points}, key=lambda value: float(value))
    grouped = {
        dataset: {alpha: {method: [] for method in METHOD_ORDER} for alpha in alphas}
        for dataset in datasets
    }
    for point in points:
        if point["dataset"] not in grouped:
            continue
        grouped[point["dataset"]][point["alpha"]][point["method"]].append(
            {
                "recall": point["recall"],
                "qps": point["qps"],
                "param": point["param"],
            }
        )

    for dataset in datasets:
        for alpha in alphas:
            for method in METHOD_ORDER:
                grouped[dataset][alpha][method].sort(key=lambda row: (row["recall"], row["qps"]))

    return {
        "datasets": datasets,
        "datasetLabels": DATASET_LABELS,
        "alphas": alphas,
        "methods": METHOD_ORDER,
        "methodLabels": METHOD_LABELS,
        "methodColors": METHOD_COLORS,
        "methodMarkers": METHOD_MARKERS,
        "targetEf": target_ef,
        "points": grouped,
    }


def write_summary(points: list[dict], output: Path) -> None:
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "alpha", "method", "param", "recall", "qps"],
            delimiter="\t",
        )
        writer.writeheader()
        for point in sorted(
            points,
            key=lambda p: (DATASET_ORDER.index(p["dataset"]), float(p["alpha"]), METHOD_ORDER.index(p["method"]), p["recall"]),
        ):
            writer.writerow(point)


def build_html(payload: dict) -> str:
    payload_json = html.escape(json.dumps(payload, separators=(",", ":")), quote=False)
    row_count = len(payload["alphas"])
    figure_height = 80 + row_count * 274 + max(row_count - 1, 0) * 35 + 22
    # Fit the full-width SVG on one PDF page, including margins and a small
    # allowance for browser print rounding.
    page_height = round(10.84 * figure_height / 1480 + 0.25, 2)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recall Improvement from Entry-Point Selection</title>
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
  size: 11in {page_height}in;
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
    width: 100%;
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
<main class="page" aria-label="Ablation results figure">
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

function efValue(point) {{
  return Number(point.param.split("=").pop());
}}

function yScale(value, minValue, maxValue, plotTop, plotHeight) {{
  return plotTop + (maxValue - value) / (maxValue - minValue) * plotHeight;
}}

function localRecallBounds(firstRecall, secondRecall) {{
  const lowerValue = Math.min(firstRecall, secondRecall);
  const upperValue = Math.max(firstRecall, secondRecall);
  const difference = Math.max(upperValue - lowerValue, 0.1);
  const rawStep = difference / 4;
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const scaled = rawStep / power;
  const step = (scaled <= 1.5 ? 1 : scaled <= 3.5 ? 2 : scaled <= 7 ? 5 : 10) * power;
  const min = Math.max(0, Math.floor((lowerValue - difference * 0.2) / step) * step);
  const max = Math.min(100, Math.ceil((upperValue + difference * 0.35) / step) * step);
  const ticks = [];
  for (let tick = min; tick <= max + step * 0.5; tick += step) {{
    ticks.push(Number(tick.toFixed(8)));
  }}
  return {{ min, max, ticks }};
}}

function comparisonMethod(alpha) {{
  return alpha === "0.3" ? "uhgs" : "uhgh";
}}

function drawMarker(svg, type, x, y, color, size = 5.5) {{
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
  }} else {{
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
  }}
}}

function drawLegend(svg, width) {{
  const y = 51;
  const methods = DATA.methods;
  const labelWidths = {{ uhg: 44, uhgs: 56, uhgh: 58 }};
  const gap = 36;
  const itemWidths = methods.map((method) => 22 + labelWidths[method]);
  const totalWidth = itemWidths.reduce((sum, value) => sum + value, 0) + gap * (methods.length - 1);
  let x = width / 2 - totalWidth / 2;

  methods.forEach((method, index) => {{
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

function matchedPoint(dataset, alpha, method) {{
  return DATA.points[dataset][alpha][method]
    .find((point) => String(efValue(point)) === String(DATA.targetEf));
}}

function drawPanel(svg, dataset, alpha, panelIndex, panelX, panelY, panelWidth, panelHeight) {{
  const improvedMethod = comparisonMethod(alpha);
  const baseline = matchedPoint(dataset, alpha, "uhg");
  const improved = matchedPoint(dataset, alpha, improvedMethod);
  if (!baseline || !improved) return;

  const baselineRecall = baseline.recall * 100;
  const improvedRecall = improved.recall * 100;
  const bounds = localRecallBounds(baselineRecall, improvedRecall);
  const plotLeft = panelX + 50;
  const plotTop = panelY + 13;
  const plotWidth = panelWidth - 60;
  const plotHeight = 205;
  const plotBottom = plotTop + plotHeight;

  svg.appendChild(node("rect", {{
    x: plotLeft,
    y: plotTop,
    width: plotWidth,
    height: plotHeight,
    fill: "#ffffff",
    stroke: "#111111",
    "stroke-width": 1,
  }}));

  bounds.ticks.forEach((tick) => {{
    const y = yScale(tick, bounds.min, bounds.max, plotTop, plotHeight);
    svg.appendChild(node("line", {{
      x1: plotLeft,
      y1: y,
      x2: plotLeft + plotWidth,
      y2: y,
      class: tick === bounds.min ? "baseline" : "grid-line",
    }}));
    svg.appendChild(node("text", {{
      x: plotLeft - 10,
      y: y + 4,
      "text-anchor": "end",
      class: "tick",
    }}, Number.isInteger(tick) ? String(tick) : tick.toFixed(1)));
  }});

  const barWidth = 72;
  const bars = [
    {{
      point: baseline,
      recall: baselineRecall,
      method: "uhg",
      center: Math.round(plotLeft + plotWidth * 0.31),
    }},
    {{
      point: improved,
      recall: improvedRecall,
      method: improvedMethod,
      center: Math.round(plotLeft + plotWidth * 0.69),
    }},
  ];
  bars.forEach((bar) => {{
    const y = yScale(bar.recall, bounds.min, bounds.max, plotTop, plotHeight);
    const barLeft = bar.center - barWidth / 2;
    svg.appendChild(node("rect", {{
      x: barLeft,
      y,
      width: barWidth,
      height: plotBottom - y,
      fill: DATA.methodColors[bar.method],
      stroke: "#111111",
      "stroke-width": 0.7,
    }}));
    svg.appendChild(node("text", {{
      x: bar.center,
      y: y - 7,
      "text-anchor": "middle",
      class: "bar-value",
    }}, bar.recall.toFixed(1)));
    svg.appendChild(node("text", {{
      x: bar.center,
      y: plotBottom + 8,
      "text-anchor": "middle",
      "dominant-baseline": "hanging",
      class: "bar-label",
    }}, DATA.methodLabels[bar.method]));
  }});

  svg.appendChild(node("polyline", {{
    points: `${{plotLeft - 4}},${{plotBottom - 8}} ${{plotLeft + 4}},${{plotBottom - 4}} ${{plotLeft - 4}},${{plotBottom}}`,
    fill: "none",
    stroke: "#111111",
    "stroke-width": 1.2,
  }}));
  svg.appendChild(node("text", {{
    x: panelX + 12,
    y: plotTop + plotHeight / 2,
    transform: `rotate(-90 ${{panelX + 12}} ${{plotTop + plotHeight / 2}})`,
    "text-anchor": "middle",
    class: "axis-label",
  }}, "Recall (%)"));
  svg.appendChild(node("text", {{
    x: plotLeft + plotWidth / 2,
    y: plotBottom + 61,
    "text-anchor": "middle",
    class: "panel-label",
  }}, `(${{String.fromCharCode(97 + panelIndex)}}) ${{DATA.datasetLabels[dataset]}} α=${{alpha}}`));
}}

function draw() {{
  const svg = document.getElementById("figure");
  const width = 1480;
  const height = {figure_height};
  const panelWidth = 274;
  const panelHeight = 274;
  const colGap = 0;
  const rowGap = 35;
  const startX = (width - DATA.datasets.length * panelWidth - (DATA.datasets.length - 1) * colGap) / 2;
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
    .bar-label {{ font: 12px "Times New Roman", Times, serif; fill: #111111; }}
    .bar-value {{ font: bold 13px "Times New Roman", Times, serif; fill: #111111; }}
    .grid-line {{ stroke: #8f8f8f; stroke-width: 0.75; stroke-dasharray: 2 2; }}
    .axis-grid-edge {{ stroke: #111111; stroke-width: 0.9; }}
    .baseline {{ stroke: #111111; stroke-width: 1.1; }}
  `;
  defs.appendChild(style);
  svg.appendChild(defs);

  svg.appendChild(node("title", {{ id: "figure-title" }}, "Recall Improvement from Entry-Point Selection"));
  svg.appendChild(node("desc", {{ id: "figure-desc" }}, "Recall comparison at a fixed, matched ef_search value."));
  svg.appendChild(node("rect", {{ x: 0, y: 0, width, height, fill: "#ffffff" }}));

  svg.appendChild(node("text", {{
    x: width / 2, y: 21, "text-anchor": "middle", class: "figure-heading",
  }}, "Recall Improvement from Entry-Point Selection"));
  drawLegend(svg, width);

  let panelIndex = 0;
  DATA.alphas.forEach((alpha, rowIndex) => {{
    DATA.datasets.forEach((dataset, colIndex) => {{
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
  link.download = "ablation_fig8.svg";
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--main-results-root", type=Path, default=MAIN_RESULT_ROOT)
    parser.add_argument("--k", type=int, default=100, help="Expected Recall@k value (default: 100)")
    parser.add_argument("--ef", default="300", help="ef_search value to plot (default: 300)")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULT_ROOT / "ablation_fig8_points.tsv",
    )
    args = parser.parse_args()

    points = load_points(args.results_root, args.main_results_root, args.k)
    if not points:
        raise SystemExit(f"No ablation points found under {args.results_root}")

    payload = build_payload(points, args.ef)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload))
    write_summary(points, args.summary)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
