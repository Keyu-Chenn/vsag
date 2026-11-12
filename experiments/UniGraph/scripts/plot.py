#!/usr/bin/env python3
"""Render Figures 5, 6, 8, and 9 from the canonical search summary."""
from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
import shutil
import subprocess

from plot_helpers import render_html, write_pdf_from_svg
from plot_style import (DATASET_LABELS, DATASET_ORDER, METHOD_COLORS as OLD_COLORS,
                        METHOD_LABELS as OLD_LABELS, METHOD_MARKERS as OLD_MARKERS,
                        marker_svg, nice_log_upper)

METHODS = (
    "hnsw", "sindi", "hnsw_sindi", "naive_hybrid_hnsw",
    "optimized_hybrid_hnsw", "uhg", "ladr_adaptive_hybrid",
)

LABELS = {key: OLD_LABELS.get(key, key) for key in METHODS}
LABELS.update(
    optimized_hybrid_hnsw="Opt Hybrid HNSW",
    ladr_adaptive_hybrid="LADR",
)
COLORS = {key: OLD_COLORS.get(key, "#111111") for key in METHODS}
COLORS.update(
    optimized_hybrid_hnsw="#17A6B6",
    ladr_adaptive_hybrid="#2CA02C",
)
MARKERS = {key: OLD_MARKERS.get(key, "circle") for key in METHODS}
MARKERS.update(
    optimized_hybrid_hnsw="cross",
    ladr_adaptive_hybrid="square",
)


def text(x, y, value, size=14, anchor="middle", extra=""):
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" {extra}>'
            f'{html.escape(str(value))}</text>')


def opening(width, height, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            '<style>text {font-family:"Times New Roman","DejaVu Serif",serif;fill:#111111}'
            '.grid-line {stroke:#929292;stroke-width:0.65;stroke-dasharray:2 3;opacity:.6}</style>',
            f'<title>{html.escape(title)}</title>',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
            text(width/2, 24, title, 20, extra='class="figure-heading"')]


def export(svg, output, stem, html_max_width=1180):
    output.mkdir(parents=True, exist_ok=True)
    (output / (stem+".svg")).write_text(svg+"\n")
    document = render_html(svg, stem+".svg")
    if html_max_width != 1180:
        document = document.replace("max-width: 1180px", f"max-width: {html_max_width}px")
    if stem == "ablation1":
        document = document.replace("max-width: 1180px", "max-width: 1480px")
    (output / (stem+".html")).write_text(document)
    write_pdf_from_svg(svg, output / (stem+".pdf"))
    if converter := shutil.which("pdftoppm"):
        subprocess.run([converter, "-png", "-singlefile", "-r", "92",
                        str(output / (stem+".pdf")), str(output / stem)], check=True)
    print(output / (stem+".pdf"))


def load(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        for name in ("recall", "qps_median", "qps_min", "qps_max"):
            row[name] = float(row[name])
            if not math.isfinite(row[name]):
                raise ValueError(f"nonfinite {name}")
        for name in ("k", "point", "repetitions"):
            row[name] = int(row[name])
        if not 0 <= row["recall"] <= 1 or row["qps_min"] <= 0 or row["repetitions"] != 1:
            raise ValueError("invalid or incomplete search summary")
    return rows


def curves(rows, suite, x_min, x_max):
    if suite == "mixed_alpha":
        panels = [(ds, k, "mixed") for k in (50, 100, 200) for ds in DATASET_ORDER]
        title = "QPS–Recall Performance under Mixed Alpha"
    else:
        panels = [(ds, 100, alpha) for ds in ("nq", "msmarco", "dbpedia-entity")
                  for alpha in ("0.3", "0.4", "0.5", "0.6", "0.7")]
        title = "QPS–Recall Performance under Fixed Alpha"
    grouped = {}
    for ds, k, alpha in panels:
        for method in METHODS:
            selected = [row for row in rows if row["suite"] == suite and row["dataset"] == ds
                        and row["k"] == k and row["alpha"] == alpha and row["method"] == method
                        and row["variant"] in {"", "no_pruning"}]
            if not selected:
                raise ValueError(f"missing curve: {suite}/{ds}/{k}/{alpha}/{method}")
            grouped[(ds,k,alpha,method)] = sorted(selected, key=lambda row: (row["recall"], row["qps_median"]))
    visible = [row for group in grouped.values() for row in group if x_min <= row["recall"] <= x_max]
    if not visible:
        raise ValueError("no points in selected recall range")
    y_min = 10**math.floor(math.log10(min(row["qps_median"] for row in visible)))
    y_max = nice_log_upper(max(row["qps_median"] for row in visible)*1.08)
    width, height = 1600, 1080
    parts = opening(width, height, title)
    for index in range(0, len(METHODS), 5):
        line = METHODS[index:index+5]
        widths = [32+len(LABELS[method])*7.4 for method in line]
        x = (width-sum(widths)-25*(len(line)-1))/2
        y = 52+27*(index//5)
        for method, item_width in zip(line, widths):
            parts.append(marker_svg(MARKERS[method], x+7, y-4, COLORS[method], 8))
            parts.append(text(x+20, y, LABELS[method], 15, "start"))
            x += item_width+25
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    for index, (ds, k, alpha) in enumerate(panels):
        px, py = 38+308*(index%5), 137+305*(index//5)
        left, top, w, h = px+43, py+12, 241, 198
        xs = lambda value: left+(value-x_min)/(x_max-x_min)*w
        ys = lambda value: top+(math.log10(y_max)-math.log10(value))/(math.log10(y_max)-math.log10(y_min))*h
        parts.append(f'<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="#ffffff" stroke="#111111" stroke-width="0.9"/>')
        step = .1 if x_max-x_min <= .5 else .2
        for tick in range(math.ceil(x_min/step-1e-9), math.floor(x_max/step+1e-9)+1):
            value = tick*step
            x = xs(value)
            parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+h}" class="grid-line"/>')
            parts.append(text(x, top+h+14, f"{value:.1f}", 11))
        for exponent in range(math.ceil(math.log10(y_min)), math.floor(math.log10(y_max))+1):
            y = ys(10**exponent)
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+w}" y2="{y:.2f}" class="grid-line"/>')
            parts.append(text(left-7, y+3.5, "10"+str(exponent).translate(superscripts), 11, "end"))
        for method in tuple(m for m in METHODS if m != "uhg")+("uhg",):
            data = [row for row in grouped[(ds,k,alpha,method)] if x_min <= row["recall"] <= x_max]
            if not data:
                continue
            coordinates = [(xs(row["recall"]), ys(row["qps_median"])) for row in data]
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x,y in coordinates)
            line_width = 2.2 if method == "uhg" else 1.7 if method == "optimized_hybrid_hnsw" else 1.25
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{COLORS[method]}" stroke-width="{line_width}" stroke-linejoin="round"/>')
            for row, (x,y) in zip(data, coordinates):
                tooltip = (f"{LABELS[method]}\npoint={row['point']}\nRecall@{k}={row['recall']:.6f}"
                           f"\nQPS={row['qps_median']:.3f}\n1 run; no warm-up")
                parts.append(marker_svg(MARKERS[method], x, y, COLORS[method], 6.7 if method == "uhg" else 5.4, tooltip))
        parts.append(text(left+w/2, top+h+31, f"Recall@{k}", 13))
        parts.append(text(px+10, top+h/2, "QPS", 13,
                          extra=f'transform="rotate(-90 {px+10} {top+h/2})"'))
        suffix = f", α={alpha}" if suite == "fixed_alpha" else ""
        parts.append(text(left+w/2, py+266, f"({chr(97+index)}) {DATASET_LABELS[ds]}{suffix}", 14))
    parts.append(text(width/2, height-15, f"One run, no warm-up; one pinned CPU; Recall range [{x_min:g}, {x_max:g}].", 12))
    return "".join(parts+["</svg>"])


def single(rows, suite, dataset, variant, alpha):
    found = [row for row in rows if row["suite"] == suite and row["dataset"] == dataset
             and row["variant"] == variant and row["alpha"] == alpha and row["point"] == 300]
    if len(found) != 1:
        raise ValueError(f"expected one ablation point: {suite}/{dataset}/{variant}/{alpha}")
    return found[0]


def entry_ablation(rows):
    width, height = 1480, 420
    panel_width, column_gap = 700, 40
    start_x = (width - panel_width * 2 - column_gap) / 2
    panel_y = 65
    plot_height = 220
    methods = (
        ("UHG", "pentagon", "#4B5563"),
        ("UHGS", "square", "#0072B2"),
        ("UHGH", "triangle", "#CC79A7"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="figure-title figure-desc">',
        '<style>'
        'text{font-family:"Times New Roman",Times,serif;fill:#111111}'
        '.legend{font-size:25px}.tick{font-size:19px;fill:#333333}'
        '.axis-label{font-size:22px}.panel-label{font-size:24px}'
        '.dataset-label{font-size:20px}.point-value{font-size:18px;font-weight:bold}'
        '.baseline-value{fill:#30343b}.delta-label{font-size:16px;font-weight:bold}'
        '.delta-box{stroke:#d8d8d8;stroke-width:.7}.connector{stroke-width:5;'
        'stroke-linecap:round;opacity:.58}.row-band{fill:#f4f5f7}'
        '.row-separator{stroke:#dddddd;stroke-width:.7}'
        '.grid-line{stroke:#b5b8bc;stroke-width:.8;stroke-dasharray:3 3}'
        '.grid-edge{stroke:#94979b;stroke-width:.85}'
        '.axis-line{stroke:#111111;stroke-width:1}'
        '</style>',
        '<title id="figure-title">Recall Improvement from Entry-Point Selection — Dumbbell Plot</title>',
        '<desc id="figure-desc">Dumbbell comparison showing UHG recall and the improved recall at ef=300.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    label_widths = (58, 74, 76)
    item_widths = tuple(29 + value for value in label_widths)
    legend_gap = 36
    legend_x = width / 2 - (sum(item_widths) + legend_gap * 2) / 2
    for (label, marker, color), item_width in zip(methods, item_widths):
        parts.append(marker_svg(marker, legend_x + 9, 38, color, 12))
        parts.append(f'<text x="{legend_x + 28:.2f}" y="42" class="legend">{label}</text>')
        legend_x += item_width + legend_gap

    for panel, alpha in enumerate(("0.3", "0.7")):
        pairs = [(single(rows, "ablation1", dataset, "unguided", alpha),
                  single(rows, "ablation1", dataset, "guided", alpha))
                 for dataset in DATASET_ORDER]
        panel_x = start_x + panel * (panel_width + column_gap)
        plot_left = panel_x + 148
        plot_top = panel_y + 20
        plot_width = panel_width - 180
        plot_bottom = plot_top + plot_height
        row_height = plot_height / len(pairs)
        minimum = 80 if panel == 0 else 85
        ticks = (80, 85, 90, 95, 100) if panel == 0 else (85, 90, 95, 100)
        guided_label, guided_marker, guided_color = methods[panel + 1]

        def x_scale(value):
            return plot_left + (value - minimum) / (100 - minimum) * plot_width

        parts.append(f'<rect x="{plot_left:.2f}" y="{plot_top:.2f}" width="{plot_width:.2f}" '
                     f'height="{plot_height:.2f}" fill="#fbfbfb"/>')
        for index in range(len(pairs)):
            if index % 2 == 0:
                parts.append(f'<rect x="{panel_x + 4:.2f}" y="{plot_top + index * row_height:.2f}" '
                             f'width="{panel_width - 8:.2f}" height="{row_height:.2f}" class="row-band"/>')

        for tick in ticks:
            x = x_scale(tick)
            css_class = "grid-edge" if tick in {minimum, 100} else "grid-line"
            parts.append(f'<line x1="{x:.2f}" y1="{plot_top:.2f}" x2="{x:.2f}" '
                         f'y2="{plot_bottom:.2f}" class="{css_class}"/>')
            parts.append(f'<text x="{x:.2f}" y="{plot_bottom + 24:.2f}" '
                         f'text-anchor="middle" class="tick">{tick}</text>')

        for index, (unguided, guided) in enumerate(pairs):
            y = plot_top + row_height * (index + .5)
            baseline_recall = unguided["recall"] * 100
            guided_recall = guided["recall"] * 100
            baseline_x, guided_x = x_scale(baseline_recall), x_scale(guided_recall)
            delta = guided_recall - baseline_recall
            delta_x = (baseline_x + guided_x) / 2
            dataset_label = html.escape(DATASET_LABELS[DATASET_ORDER[index]])
            parts.append(f'<text x="{plot_left - 14:.2f}" y="{y + 6:.2f}" text-anchor="end" '
                         f'class="dataset-label">{dataset_label}</text>')
            parts.append(f'<line x1="{baseline_x:.2f}" y1="{y:.2f}" x2="{guided_x:.2f}" '
                         f'y2="{y:.2f}" stroke="{guided_color}" class="connector"/>')
            parts.append(f'<rect x="{delta_x - 36:.2f}" y="{y - 25:.2f}" width="72" '
                         'height="19" rx="4" fill="#ffffff" class="delta-box"/>')
            parts.append(f'<text x="{delta_x:.2f}" y="{y - 11:.2f}" text-anchor="middle" '
                         f'fill="{guided_color}" class="delta-label">+{delta:.1f}%</text>')
            baseline_tip = (f"{DATASET_LABELS[DATASET_ORDER[index]]} — UHG\n"
                            f"alpha={alpha}\nef=300\nRecall@100={unguided['recall']:.6f}")
            guided_tip = (f"{DATASET_LABELS[DATASET_ORDER[index]]} — {guided_label}\n"
                          f"alpha={alpha}\nef=300\nRecall@100={guided['recall']:.6f}")
            parts.append(marker_svg("pentagon", baseline_x, y, "#4B5563", 13, baseline_tip))
            parts.append(marker_svg(guided_marker, guided_x, y, guided_color, 13, guided_tip))
            parts.append(f'<text x="{baseline_x - 11:.2f}" y="{y + 6:.2f}" text-anchor="end" '
                         f'class="point-value baseline-value">{baseline_recall:.1f}</text>')
            parts.append(f'<text x="{guided_x + 11:.2f}" y="{y + 6:.2f}" text-anchor="start" '
                         f'fill="{guided_color}" class="point-value">{guided_recall:.1f}</text>')
            if index < len(pairs) - 1:
                separator_y = plot_top + row_height * (index + 1)
                parts.append(f'<line x1="{panel_x + 4:.2f}" y1="{separator_y:.2f}" '
                             f'x2="{panel_x + panel_width - 4:.2f}" y2="{separator_y:.2f}" '
                             'class="row-separator"/>')

        parts.append(f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" '
                     f'x2="{plot_left + plot_width:.2f}" y2="{plot_bottom:.2f}" class="axis-line"/>')
        for tick in ticks:
            x = x_scale(tick)
            parts.append(f'<line x1="{x:.2f}" y1="{plot_bottom:.2f}" x2="{x:.2f}" '
                         f'y2="{plot_bottom + 6:.2f}" class="axis-line"/>')
        parts.append(f'<text x="{plot_left + plot_width / 2:.2f}" y="{plot_bottom + 48:.2f}" '
                     'text-anchor="middle" class="axis-label">Recall (%)</text>')
        parts.append(f'<text x="{plot_left + plot_width / 2:.2f}" y="{plot_bottom + 83:.2f}" '
                     f'text-anchor="middle" class="panel-label">({chr(97 + panel)}) α={alpha}</text>')

    parts.append('</svg>')
    return "".join(parts)


def pruning_ablation(rows):
    pairs = [(single(rows,"ablation2",ds,"prune_no_prune","mixed"),
              single(rows,"ablation2",ds,"prune_0.3","mixed")) for ds in DATASET_ORDER]
    width, height = 900, 520
    panel_x, panel_y, panel_width = 50, 80, 800
    plot_left, plot_top = panel_x + 64, panel_y + 13
    plot_width, plot_height = panel_width - 82, 320
    plot_bottom = plot_top + plot_height
    values = [row["qps_median"] for pair in pairs for row in pair]
    raw_step = max(values) / 6
    power = 10 ** math.floor(math.log10(raw_step))
    scaled = raw_step / power
    step = next(multiplier * power for multiplier in (1, 2, 5, 10)
                if multiplier >= scaled)
    span = max(values) - min(values)
    lower = math.floor((min(values) - span * .15) / step) * step
    upper = math.ceil((max(values) + span * .03) / step) * step
    if lower == upper:
        lower, upper = max(0, lower - step), upper + step
    ticks = []
    tick = lower
    while tick <= upper + step * .5:
        ticks.append(tick)
        tick += step

    def y_scale(value):
        return plot_top + (upper - value) / (upper - lower) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="figure-title figure-desc">',
        '<style>'
        'text {font-family:"Times New Roman",Times,serif;fill:#111111}'
        '.legend{font-size:27px}.tick{font-size:23px}.axis-label{font-size:25px}'
        '.bar-label{font-size:23px}.bar-value{font-size:20px;font-weight:bold}'
        '.grid-line{stroke:#8f8f8f;stroke-width:.75;stroke-dasharray:2 2}'
        '.baseline{stroke:#111111;stroke-width:1.1}'
        '</style>',
        '<title id="figure-title">QPS with vs. without Hybrid Pruning</title>',
        '<desc id="figure-desc">Grouped QPS comparison for five datasets with and without hybrid pruning.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    legend = (("No pruning", "#9CA3AF", 101),
              ("Pruning (scale=0.3)", "#E45756", 178))
    legend_gap = 38
    legend_x = width / 2 - (sum(item[2] for item in legend) + legend_gap) / 2
    for label, color, item_width in legend:
        parts.append(f'<rect x="{legend_x:.2f}" y="40" width="16" height="11" '
                     f'fill="{color}" stroke="#111111" stroke-width=".5"/>')
        parts.append(f'<text x="{legend_x + 23:.2f}" y="51" class="legend">'
                     f'{html.escape(label)}</text>')
        legend_x += item_width + legend_gap

    parts.append(f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" '
                 f'height="{plot_height}" fill="#ffffff"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
                 f'y2="{plot_bottom}" class="baseline"/>')
    for tick in ticks:
        y = y_scale(tick)
        css_class = "baseline" if math.isclose(tick, lower) else "grid-line"
        if not math.isclose(tick, upper):
            parts.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" '
                         f'y2="{y:.2f}" class="{css_class}"/>')
        label = f"{tick:g}"
        parts.append(f'<text x="{plot_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                     f'class="tick">{label}</text>')

    group_width = plot_width / len(pairs)
    bar_width = min(72, group_width * .26)
    for index, (without, pruned) in enumerate(pairs):
        center = plot_left + group_width * (index + .5)
        bars = ((without, center - bar_width * .62, "#9CA3AF", "No pruning"),
                (pruned, center + bar_width * .62, "#E45756", "Pruning (scale=0.3)"))
        for row, bar_center, color, label in bars:
            y = y_scale(row["qps_median"])
            bar_left = bar_center - bar_width / 2
            tooltip = (f"{DATASET_LABELS[DATASET_ORDER[index]]} — {label}\n"
                       f"QPS={row['qps_median']:.3f}\nRecall@100={row['recall']:.6f}")
            parts.append(f'<rect x="{bar_left:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                         f'height="{plot_bottom - y:.2f}" fill="{color}" stroke="#111111" '
                         f'stroke-width=".7"><title>{html.escape(tooltip)}</title></rect>')
            parts.append(f'<text x="{bar_center:.2f}" y="{y - 7:.2f}" text-anchor="middle" '
                         f'class="bar-value">{row["qps_median"]:.1f}</text>')
        parts.append(f'<text x="{center:.2f}" y="{plot_bottom + 12}" text-anchor="middle" '
                     f'dominant-baseline="hanging" class="bar-label">'
                     f'{html.escape(DATASET_LABELS[DATASET_ORDER[index]])}</text>')

    parts.append(f'<polyline points="{plot_left - 4},{plot_bottom - 8} '
                 f'{plot_left + 4},{plot_bottom - 4} {plot_left - 4},{plot_bottom}" '
                 'fill="none" stroke="#111111" stroke-width="1.2"/>')
    axis_x, axis_y = panel_x + 19, plot_top + plot_height / 2
    parts.append(f'<text x="{axis_x}" y="{axis_y}" transform="rotate(-90 {axis_x} {axis_y})" '
                 'text-anchor="middle" class="axis-label">QPS</text>')
    parts.append('</svg>')
    return "".join(parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--suite",choices=("all","mixed_alpha","fixed_alpha","ablation1","ablation2"),default="all")
    parser.add_argument("--x-min",type=float,default=0.)
    parser.add_argument("--x-max",type=float,default=1.)
    args=parser.parse_args()
    if not 0 <= args.x_min < args.x_max <= 1:
        parser.error("require 0 <= x-min < x-max <= 1")
    rows=load(args.run_root/"summary.tsv")
    suites=("mixed_alpha","fixed_alpha","ablation1","ablation2") if args.suite=="all" else (args.suite,)
    for suite in suites:
        svg=(curves(rows,suite,args.x_min,args.x_max) if suite in {"mixed_alpha","fixed_alpha"}
             else entry_ablation(rows) if suite=="ablation1" else pruning_ablation(rows))
        suffix="" if args.x_min==0 and args.x_max==1 else f"_recall_{args.x_min:g}_{args.x_max:g}"
        export(svg,args.run_root/"figures",suite+suffix)


if __name__=="__main__":
    main()
