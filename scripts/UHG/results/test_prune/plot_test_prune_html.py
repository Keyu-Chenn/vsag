#!/usr/bin/env python3
"""Compare QPS with and without hybrid pruning at a matched ef_search."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path


REPO_ROOT = Path("/tbase-project/vsag")
RESULT_ROOT = REPO_ROOT / "scripts" / "UHG" / "results" / "test_prune"
OUTPUT_PATH = RESULT_ROOT / "test_prune_fig8.html"

DATASET_ORDER = ["nq", "hotpotqa", "msmarco", "fever", "dbpedia-entity"]
DATASET_LABELS = {
    "nq": "NQ",
    "hotpotqa": "HotpotQA",
    "msmarco": "MS MARCO",
    "fever": "FEVER",
    "dbpedia-entity": "DBpedia-Entity",
}
SCALE_COLORS = {
    "no_prune": "#9CA3AF",
    "1.0": "#4B5563",
    "0.5": "#0072B2",
    "0.3": "#E45756",
}
SCALE_MARKERS = {
    "no_prune": "diamond",
    "1.0": "circle",
    "0.5": "square",
    "0.3": "pentagon",
}
SCALE_ORDER = ["no_prune", "0.3", "0.5", "1.0"]

METRIC_RE = re.compile(
    r"\bRecall:\s*(?P<recall>[-+0-9.eE]+)\s+QPS:\s*(?P<qps>[-+0-9.eE]+)"
)
PARAM_RE = re.compile(r"\bef_search=(?P<value>[-+0-9.eE]+)")


def scale_sort_key(scale: str) -> tuple[int, float | str]:
    if scale in SCALE_ORDER:
        return (0, SCALE_ORDER.index(scale))
    try:
        return (1, float(scale))
    except ValueError:
        return (2, scale)


def scale_label(scale: str) -> str:
    if scale == "no_prune":
        return "No pruning"
    if scale == "1.0":
        return "Pruning (scale=1.0, Cauchy)"
    return f"Pruning (scale={scale})"


def parse_result_file(dataset: str, scale: str, path: Path) -> list[dict]:
    alpha = path.stem.removeprefix("alpha_")
    points: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        metric_match = METRIC_RE.search(line)
        if metric_match is None:
            continue
        param_match = PARAM_RE.search(line)
        ef_search = param_match.group("value") if param_match else ""
        recall = float(metric_match.group("recall"))
        qps = float(metric_match.group("qps"))
        if recall <= 0 or qps <= 0:
            continue
        points.append(
            {
                "dataset": dataset,
                "alpha": alpha,
                "scale": scale,
                "ef": ef_search,
                "recall": round(recall, 6),
                "qps": round(qps, 6),
            }
        )
    return points


def canonical_scale(scale: str) -> str:
    if scale.lower() in {"no_prune", "no-prune", "off", "none", "disabled"}:
        return "no_prune"
    return scale


def load_points(result_root: Path) -> list[dict]:
    points: list[dict] = []
    for dataset in DATASET_ORDER:
        dataset_dir = result_root / dataset
        if not dataset_dir.is_dir():
            continue
        scale_dirs: dict[str, Path] = {}
        for scale_dir in sorted(dataset_dir.glob("prune_*")):
            if not scale_dir.is_dir():
                continue
            scale = canonical_scale(scale_dir.name.removeprefix("prune_"))
            canonical_name = f"prune_{scale}"
            if scale not in scale_dirs or scale_dir.name == canonical_name:
                scale_dirs[scale] = scale_dir
        for scale, scale_dir in scale_dirs.items():
            for path in sorted(scale_dir.glob("alpha_mixed.txt")):
                points.extend(parse_result_file(dataset, scale, path))
    return points


def add_speedups(points: list[dict]) -> list[dict]:
    baseline: dict[tuple[str, str, str], float] = {}
    for point in points:
        if point["scale"] == "no_prune":
            baseline[(point["dataset"], point["alpha"], point["ef"])] = point["qps"]

    enriched: list[dict] = []
    for point in points:
        base = baseline.get((point["dataset"], point["alpha"], point["ef"]))
        row = dict(point)
        row["speedup"] = round(point["qps"] / base, 6) if base else None
        row["gain_pct"] = round((point["qps"] / base - 1.0) * 100.0, 3) if base else None
        enriched.append(row)
    return enriched


def build_payload(points: list[dict], target_ef: str) -> dict:
    selected_points = [point for point in points if point["ef"] == target_ef]
    datasets = [dataset for dataset in DATASET_ORDER if any(p["dataset"] == dataset for p in selected_points)]
    alphas = sorted({point["alpha"] for point in selected_points})
    scales = sorted({point["scale"] for point in selected_points}, key=scale_sort_key)
    grouped = {
        dataset: {alpha: {scale: [] for scale in scales} for alpha in alphas}
        for dataset in datasets
    }
    for point in selected_points:
        if point["dataset"] not in grouped:
            continue
        grouped[point["dataset"]][point["alpha"]][point["scale"]].append(
            {
                "recall": point["recall"],
                "qps": point["qps"],
                "ef": point["ef"],
                "speedup": point["speedup"],
                "gain_pct": point["gain_pct"],
            }
        )

    for dataset in datasets:
        for alpha in alphas:
            for scale in scales:
                grouped[dataset][alpha][scale].sort(key=lambda row: (row["recall"], row["qps"]))

    return {
        "datasets": datasets,
        "datasetLabels": DATASET_LABELS,
        "alphas": alphas,
        "scales": scales,
        "scaleLabels": {scale: scale_label(scale) for scale in scales},
        "scaleColors": {scale: SCALE_COLORS.get(scale, "#59A14F") for scale in scales},
        "scaleMarkers": {scale: SCALE_MARKERS.get(scale, "triangle") for scale in scales},
        "targetEf": target_ef,
        "points": grouped,
    }


def write_summary(points: list[dict], output: Path) -> None:
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "alpha", "scale", "ef", "recall", "qps", "speedup", "gain_pct"],
            delimiter="\t",
        )
        writer.writeheader()
        for point in sorted(
            points,
            key=lambda p: (
                DATASET_ORDER.index(p["dataset"]),
                p["alpha"],
                scale_sort_key(p["scale"]),
                int(p["ef"]) if p["ef"].isdigit() else 0,
            ),
        ):
            writer.writerow(point)


def build_html(payload: dict) -> str:
    payload_json = html.escape(json.dumps(payload, separators=(",", ":")), quote=False)
    figure_width = 1480
    figure_height = 396
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QPS Improvement from Hybrid Pruning</title>
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
  width: min({figure_width}px, 100%);
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
  width: min({figure_width}px, 100%);
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
  size: 11in 3.4in;
  margin: 0.08in;
}}
@media print {{
  body {{
    padding: 0;
    background: #ffffff;
  }}
  .toolbar {{
    display: none;
  }}
  .page {{
    width: 100%;
    padding: 0;
    border: 0;
  }}
}}
</style>
</head>
<body>
<div class="toolbar">
  <button type="button" onclick="window.print()">Print PDF</button>
  <button type="button" onclick="downloadSvg()">Save SVG</button>
</div>
<main class="page" aria-label="Hybrid pruning ablation figure">
  <svg id="figure" viewBox="0 0 {figure_width} {figure_height}" role="img"
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

function extent(values, paddingRatio = 0.08) {{
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const pad = Math.max((maxValue - minValue) * paddingRatio, 1e-6);
  return [minValue - pad, maxValue + pad];
}}

function niceTicks(minValue, maxValue, count = 5) {{
  const span = maxValue - minValue;
  if (span <= 0) return [minValue];
  const rawStep = span / Math.max(count - 1, 1);
  const pow10 = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const scaled = rawStep / pow10;
  const niceScaled = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  const step = niceScaled * pow10;
  const start = Math.ceil(minValue / step) * step;
  const ticks = [];
  for (let tick = start; tick <= maxValue + step * 0.5; tick += step) {{
    ticks.push(Number(tick.toFixed(8)));
  }}
  return ticks;
}}

function xScale(value, minValue, maxValue, left, width) {{
  return left + ((value - minValue) / (maxValue - minValue)) * width;
}}

function yScale(value, minValue, maxValue, top, height) {{
  return top + (maxValue - value) / (maxValue - minValue) * height;
}}

function drawMarker(svg, type, x, y, color, size = 6.5) {{
  if (type === "circle") {{
    svg.appendChild(node("circle", {{
      cx: x,
      cy: y,
      r: size / 2,
      fill: color,
      stroke: "#111111",
      "stroke-width": 0.5,
    }}));
  }} else if (type === "diamond") {{
    const points = [
      `${{x}},${{y - size / 2}}`,
      `${{x + size / 2}},${{y}}`,
      `${{x}},${{y + size / 2}}`,
      `${{x - size / 2}},${{y}}`,
    ].join(" ");
    svg.appendChild(node("polygon", {{
      points,
      fill: color,
      stroke: "#111111",
      "stroke-width": 0.5,
    }}));
  }} else if (type === "square") {{
    svg.appendChild(node("rect", {{
      x: x - size / 2,
      y: y - size / 2,
      width: size,
      height: size,
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
  }}
}}

function drawLegend(svg, width) {{
  const scales = ["no_prune", "0.3"].filter((scale) => DATA.scales.includes(scale));
  const y = 51;
  const labelWidths = {{ "no_prune": 78, "0.3": 155 }};
  const itemWidths = scales.map((scale) => 23 + labelWidths[scale]);
  const gap = 38;
  const totalWidth = itemWidths.reduce((sum, value) => sum + value, 0) + gap * (scales.length - 1);
  let x = width / 2 - totalWidth / 2;
  scales.forEach((scale, index) => {{
    const color = DATA.scaleColors[scale];
    svg.appendChild(node("rect", {{
      x, y: y - 11, width: 16, height: 11,
      fill: color, stroke: "#111111", "stroke-width": 0.5,
    }}));
    svg.appendChild(node("text", {{
      x: x + 23,
      y,
      class: "legend",
    }}, DATA.scaleLabels[scale]));
    x += itemWidths[index] + gap;
  }});
}}

function flattenPoints(dataset, alpha) {{
  const points = [];
  DATA.scales.forEach((scale) => {{
    DATA.points[dataset][alpha][scale].forEach((point) => points.push({{ ...point, scale }}));
  }});
  return points;
}}

function drawAxes(svg, opts) {{
  svg.appendChild(node("rect", {{
    x: opts.left,
    y: opts.top,
    width: opts.width,
    height: opts.height,
    fill: "#ffffff",
    stroke: "#111111",
    "stroke-width": 1,
  }}));
  opts.xTicks.forEach((tick) => {{
    const x = xScale(tick, opts.xMin, opts.xMax, opts.left, opts.width);
    svg.appendChild(node("line", {{
      x1: x,
      y1: opts.top,
      x2: x,
      y2: opts.top + opts.height,
      class: tick === opts.xTicks[0] || tick === opts.xTicks[opts.xTicks.length - 1] ? "axis-grid-edge" : "grid-line",
    }}));
    svg.appendChild(node("text", {{
      x,
      y: opts.top + opts.height + 13,
      "text-anchor": "middle",
      class: "tick",
    }}, opts.xFormat(tick)));
  }});
  opts.yTicks.forEach((tick) => {{
    const y = yScale(tick, opts.yMin, opts.yMax, opts.top, opts.height);
    svg.appendChild(node("line", {{
      x1: opts.left,
      y1: y,
      x2: opts.left + opts.width,
      y2: y,
      class: "grid-line",
    }}));
    svg.appendChild(node("text", {{
      x: opts.left - 8,
      y: y + 3,
      "text-anchor": "end",
      class: "tick",
    }}, opts.yFormat(tick)));
  }});
  svg.appendChild(node("text", {{
    x: opts.left + opts.width / 2,
    y: opts.top + opts.height + 29,
    "text-anchor": "middle",
    class: "axis-label",
  }}, opts.xLabel));
  svg.appendChild(node("text", {{
    x: opts.left - 33,
    y: opts.top + opts.height / 2,
    transform: `rotate(-90 ${{opts.left - 33}} ${{opts.top + opts.height / 2}})`,
    "text-anchor": "middle",
    class: "axis-label",
  }}, opts.yLabel));
}}

function drawGainPlot(svg, dataset, alpha, panelIndex, panelX, panelY, panelWidth, panelHeight) {{
  const points = flattenPoints(dataset, alpha).filter((point) => point.scale !== "no_prune" && point.gain_pct !== null);
  const efValues = [...new Set(points.map((point) => Number(point.ef)))].sort((a, b) => a - b);
  const gains = points.map((point) => point.gain_pct);
  let [yMin, yMax] = extent(gains, 0.18);
  yMin = Math.min(0, yMin);
  const opts = {{
    left: panelX + 44,
    top: panelY + 13,
    width: panelWidth - 54,
    height: panelHeight - 59,
    xMin: Math.min(...efValues),
    xMax: Math.max(...efValues),
    yMin,
    yMax,
    xTicks: efValues,
    yTicks: niceTicks(yMin, yMax, 6),
    xFormat: (value) => value.toString(),
    yFormat: (value) => `${{value.toFixed(1)}}`,
    xLabel: "ef_search",
    yLabel: "QPS Gain (%)",
  }};
  drawAxes(svg, opts);

  DATA.scales.filter((scale) => scale !== "no_prune").forEach((scale) => {{
    const scalePoints = DATA.points[dataset][alpha][scale].filter((point) => point.gain_pct !== null);
    if (!scalePoints.length) return;
    const coords = scalePoints.map((point) => [
      xScale(Number(point.ef), opts.xMin, opts.xMax, opts.left, opts.width),
      yScale(point.gain_pct, opts.yMin, opts.yMax, opts.top, opts.height),
    ]);
    const color = DATA.scaleColors[scale];
    svg.appendChild(node("polyline", {{
      points: coords.map(([x, y]) => `${{x}},${{y}}`).join(" "),
      fill: "none",
      stroke: color,
      "stroke-width": scale === "0.3" ? 2.5 : 1.65,
    }}));
    coords.forEach(([x, y]) => drawMarker(svg, DATA.scaleMarkers[scale], x, y, color, scale === "0.3" ? 8 : 6.4));
  }});

  svg.appendChild(node("text", {{
    x: panelX + panelWidth / 2,
    y: panelY + panelHeight - 5,
    "text-anchor": "middle",
    class: "panel-label",
  }}, `(${{String.fromCharCode(97 + panelIndex)}}) ${{DATA.datasetLabels[dataset]}} α=${{alpha}}`));
}}

function localQpsBounds(noPruneQps, prunedQps) {{
  const lowerValue = Math.min(noPruneQps, prunedQps);
  const upperValue = Math.max(noPruneQps, prunedQps);
  const difference = Math.max(upperValue - lowerValue, 0.1);
  const rawStep = difference / 4;
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const scaled = rawStep / power;
  const step = (scaled <= 1.5 ? 1 : scaled <= 3.5 ? 2 : scaled <= 7 ? 5 : 10) * power;
  const min = Math.floor((lowerValue - difference * 0.2) / step) * step;
  const max = Math.ceil((upperValue + difference * 0.35) / step) * step;
  const ticks = [];
  for (let tick = min; tick <= max + step * 0.5; tick += step) {{
    ticks.push(Number(tick.toFixed(8)));
  }}
  return {{ min, max, ticks }};
}}

function drawQpsPanel(svg, dataset, panelIndex, panelX, panelY, panelWidth) {{
  const alpha = DATA.alphas[0];
  const noPrune = DATA.points[dataset][alpha].no_prune[0];
  const pruned = DATA.points[dataset][alpha]["0.3"][0];
  if (!noPrune || !pruned) return;

  const plotLeft = panelX + 50;
  const plotTop = panelY + 13;
  const plotWidth = panelWidth - 60;
  const plotHeight = 205;
  const plotBottom = plotTop + plotHeight;
  const bounds = localQpsBounds(noPrune.qps, pruned.qps);

  svg.appendChild(node("rect", {{
    x: plotLeft, y: plotTop, width: plotWidth, height: plotHeight,
    fill: "#ffffff", stroke: "#111111", "stroke-width": 1,
  }}));
  bounds.ticks.forEach((tick) => {{
    const y = yScale(tick, bounds.min, bounds.max, plotTop, plotHeight);
    svg.appendChild(node("line", {{
      x1: plotLeft, y1: y, x2: plotLeft + plotWidth, y2: y,
      class: tick === bounds.min ? "baseline" : "grid-line",
    }}));
    svg.appendChild(node("text", {{
      x: plotLeft - 10, y: y + 4, "text-anchor": "end", class: "tick",
    }}, Number.isInteger(tick) ? String(tick) : tick.toFixed(1)));
  }});

  const barWidth = 72;
  const bars = [
    {{ point: noPrune, scale: "no_prune", label: "No pruning", center: Math.round(plotLeft + plotWidth * 0.31) }},
    {{ point: pruned, scale: "0.3", label: "Pruning", center: Math.round(plotLeft + plotWidth * 0.69) }},
  ];
  bars.forEach((bar) => {{
    const y = yScale(bar.point.qps, bounds.min, bounds.max, plotTop, plotHeight);
    const barLeft = bar.center - barWidth / 2;
    svg.appendChild(node("rect", {{
      x: barLeft, y, width: barWidth, height: plotBottom - y,
      fill: DATA.scaleColors[bar.scale], stroke: "#111111", "stroke-width": 0.7,
    }}));
    svg.appendChild(node("text", {{
      x: barLeft + barWidth / 2, y: y - 7, "text-anchor": "middle", class: "bar-value",
    }}, bar.point.qps.toFixed(1)));
    svg.appendChild(node("text", {{
      x: barLeft + barWidth / 2, y: plotBottom + 8,
      "text-anchor": "middle", "dominant-baseline": "hanging", class: "bar-label",
    }}, bar.label));
  }});

  svg.appendChild(node("polyline", {{
    points: `${{plotLeft - 4}},${{plotBottom - 8}} ${{plotLeft + 4}},${{plotBottom - 4}} ${{plotLeft - 4}},${{plotBottom}}`,
    fill: "none", stroke: "#111111", "stroke-width": 1.2,
  }}));
  svg.appendChild(node("text", {{
    x: panelX + 12, y: plotTop + plotHeight / 2,
    transform: `rotate(-90 ${{panelX + 12}} ${{plotTop + plotHeight / 2}})`,
    "text-anchor": "middle", class: "axis-label",
  }}, "QPS"));
  svg.appendChild(node("text", {{
    x: plotLeft + plotWidth / 2, y: plotBottom + 61,
    "text-anchor": "middle", class: "panel-label",
  }}, `(${{String.fromCharCode(97 + panelIndex)}}) ${{DATA.datasetLabels[dataset]}} (ef_search=${{DATA.targetEf}})`));
}}

function draw() {{
  const svg = document.getElementById("figure");
  const width = {figure_width};
  const height = {figure_height};
  const panelWidth = 274;
  const colGap = 0;
  const panelCount = DATA.datasets.length;
  const startX = (width - panelCount * panelWidth - (panelCount - 1) * colGap) / 2;
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

  svg.appendChild(node("title", {{ id: "figure-title" }}, "QPS with vs. without Hybrid Pruning"));
  svg.appendChild(node("desc", {{ id: "figure-desc" }}, "Grouped QPS comparison with direct percentage and absolute gains from pruning."));
  svg.appendChild(node("rect", {{ x: 0, y: 0, width, height, fill: "#ffffff" }}));

  svg.appendChild(node("text", {{
    x: width / 2, y: 21, "text-anchor": "middle", class: "figure-heading",
  }}, "QPS with vs. without Hybrid Pruning"));
  drawLegend(svg, width);
  DATA.datasets.forEach((dataset, index) => {{
    drawQpsPanel(
      svg,
      dataset,
      index,
      startX + index * (panelWidth + colGap),
      startY,
      panelWidth
    );
  }});
}}

function downloadSvg() {{
  const svg = document.getElementById("figure");
  const source = new XMLSerializer().serializeToString(svg);
  const blob = new Blob([source], {{ type: "image/svg+xml;charset=utf-8" }});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "test_prune_fig8.svg";
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
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=RESULT_ROOT / "test_prune_fig8_points.tsv")
    parser.add_argument("--ef", default="300", help="ef_search value to plot (default: 300)")
    args = parser.parse_args()

    points = add_speedups(load_points(args.results_root))
    if not points:
        raise SystemExit(f"No prune-test points found under {args.results_root}")
    payload = build_payload(points, args.ef)
    if not payload["datasets"]:
        raise SystemExit(f"No matched pruning results found for ef_search={args.ef}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload))
    write_summary(points, args.summary)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
