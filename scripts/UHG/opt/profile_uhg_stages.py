#!/usr/bin/env python3
"""Profile UHG query stages without modifying the search implementation.

The script records user-space CPU-clock samples with DWARF call stacks, then
classifies samples by the HybridIndex function present in each call stack:

  get_dense_entry_points -> UHGH dense HNSW entry search
  get_sindi_entry_points -> UHGS SINDI entry search
  graph_knn_search       -> main hybrid graph search

Only samples in a HybridIndex::KnnSearch call tree are used in the query-stage
percentages. Index loading, HDF5 reading, and recall evaluation are excluded.

Example:
  python3 scripts/UHG/opt/profile_uhg_stages.py \
    --perf-data /tmp/nq_uhgh.data \
    --log /tmp/nq_uhgh.log \
    --report /tmp/nq_uhgh.report.txt \
    -- build-release/examples/cpp/703_uhg_exp3 \
       scripts/UHG/data/hdf5/nq.hdf5 \
       --hybrid_index_path scripts/UHG/data/index_hybrid_union/703_nq_hybrid_index.index \
       --gt_dir scripts/UHG/data/ground_truth/nq \
       --method uhgh --alpha 0.6 -k 50 \
       --ef_search 150 --dense_entry_bk 150 --dense_entry_ef_search 150 \
       --hybrid_prune_scale 0.3 --num_queries 1000 --threads 1

Analyze an existing recording:
  python3 scripts/UHG/opt/profile_uhg_stages.py \
    --analyze-only /tmp/nq_uhgh.data --log /tmp/nq_uhgh.log
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional, TextIO


STAGE_MARKERS = (
    ("UHGH entry HNSW", "HybridIndex::get_dense_entry_points"),
    ("UHGS entry SINDI", "HybridIndex::get_sindi_entry_points"),
    ("main hybrid graph", "HybridIndex::graph_knn_search"),
)
QUERY_MARKER = "HybridIndex::KnnSearch"
QUERY_UNCLASSIFIED = "query routing/other"

HEADER_RE = re.compile(
    r"^\S.*:\s+([0-9]+)\s+([A-Za-z0-9_.:-]+):\s*$"
)
FRAME_RE = re.compile(r"^\s+[0-9a-fA-F]+\s+(.+?)\s+\((.*?)\)\s*$")
OFFSET_RE = re.compile(r"\+0x[0-9a-fA-F]+$")
METRIC_RE = re.compile(r"^(Recall|QPS)(?:@[^: ]+)?\s*[:=]\s*(.+?)\s*$")


@dataclass
class StageStats:
    period_ns: int = 0
    samples: int = 0
    leaf_period_ns: collections.Counter[str] = field(
        default_factory=collections.Counter
    )
    leaf_caller_period_ns: collections.Counter[tuple[str, str]] = field(
        default_factory=collections.Counter
    )

    def add(self, period_ns: int, leaf: str, caller: str) -> None:
        self.period_ns += period_ns
        self.samples += 1
        self.leaf_period_ns[leaf] += period_ns
        self.leaf_caller_period_ns[(leaf, caller)] += period_ns


@dataclass
class ProfileStats:
    event: Optional[str] = None
    all_period_ns: int = 0
    all_samples: int = 0
    query_period_ns: int = 0
    query_samples: int = 0
    stages: dict[str, StageStats] = field(default_factory=dict)
    query_leaf_period_ns: collections.Counter[str] = field(
        default_factory=collections.Counter
    )
    query_leaf_caller_period_ns: collections.Counter[tuple[str, str]] = field(
        default_factory=collections.Counter
    )


def normalize_symbol(raw: str) -> str:
    return OFFSET_RE.sub("", raw.strip())


def classify_stack(symbols: Iterable[str]) -> Optional[str]:
    symbols = tuple(symbols)
    for stage, marker in STAGE_MARKERS:
        if any(marker in symbol for symbol in symbols):
            return stage
    if any(QUERY_MARKER in symbol for symbol in symbols):
        return QUERY_UNCLASSIFIED
    return None


def consume_sample(
    stats: ProfileStats,
    period: int,
    event: str,
    frames: list[str],
) -> None:
    if not frames:
        return
    if stats.event is None:
        stats.event = event
    elif stats.event != event:
        raise RuntimeError(
            f"multiple perf events are not supported: {stats.event!r}, {event!r}"
        )

    stats.all_period_ns += period
    stats.all_samples += 1
    stage = classify_stack(frames)
    if stage is None:
        return

    leaf = frames[0]
    caller = next(
        (symbol for symbol in frames[1:] if symbol != leaf),
        "[no caller]",
    )
    stats.query_period_ns += period
    stats.query_samples += 1
    stats.query_leaf_period_ns[leaf] += period
    stats.query_leaf_caller_period_ns[(leaf, caller)] += period
    stats.stages.setdefault(stage, StageStats()).add(period, leaf, caller)


def parse_perf_script(stream: TextIO) -> ProfileStats:
    stats = ProfileStats()
    period: Optional[int] = None
    event: Optional[str] = None
    frames: list[str] = []

    def flush() -> None:
        nonlocal period, event, frames
        if period is not None and event is not None:
            consume_sample(stats, period, event, frames)
        period = None
        event = None
        frames = []

    for line in stream:
        if not line.strip():
            flush()
            continue
        header = HEADER_RE.match(line)
        if header:
            flush()
            period = int(header.group(1))
            event = header.group(2)
            continue
        frame = FRAME_RE.match(line)
        if frame and period is not None:
            frames.append(normalize_symbol(frame.group(1)))
    flush()
    return stats


def read_metrics(log_path: Optional[pathlib.Path]) -> dict[str, str]:
    metrics: dict[str, str] = {}
    if log_path is None or not log_path.is_file():
        return metrics
    for line in log_path.read_text(errors="replace").splitlines():
        match = METRIC_RE.match(line.strip())
        if match:
            metrics[match.group(1)] = match.group(2)
    return metrics


def format_ms(period_ns: int) -> str:
    return f"{period_ns / 1_000_000.0:.1f}"


def append_hotspots(
    lines: list[str],
    title: str,
    counter: collections.Counter[str],
    denominator: int,
    top_n: int,
) -> None:
    lines.extend(("", title))
    if denominator == 0 or not counter:
        lines.append("  no samples")
        return
    for symbol, period in counter.most_common(top_n):
        percent = 100.0 * period / denominator
        lines.append(f"  {percent:6.2f}%  {format_ms(period):>9} ms  {symbol}")


def append_call_edges(
    lines: list[str],
    title: str,
    counter: collections.Counter[tuple[str, str]],
    denominator: int,
    top_n: int,
) -> None:
    lines.extend(("", title))
    if denominator == 0 or not counter:
        lines.append("  no samples")
        return
    for (leaf, caller), period in counter.most_common(top_n):
        percent = 100.0 * period / denominator
        lines.append(
            f"  {percent:6.2f}%  {format_ms(period):>9} ms  "
            f"{leaf} <- {caller}"
        )


def make_report(
    stats: ProfileStats,
    command: Optional[list[str]],
    metrics: dict[str, str],
    top_n: int,
) -> str:
    if stats.event is None or stats.all_samples == 0:
        raise RuntimeError("perf script produced no usable samples")
    if not stats.event.startswith("cpu-clock"):
        raise RuntimeError(
            f"expected a cpu-clock recording, found {stats.event!r}; "
            "absolute millisecond values would be invalid"
        )
    if stats.query_samples == 0:
        raise RuntimeError(
            "no HybridIndex query samples found; use a release binary with symbols "
            "and record DWARF call stacks"
        )

    lines = [
        "UHG query stage profile",
        "=======================",
    ]
    if command:
        lines.append(f"command: {shlex.join(command)}")
    if metrics:
        lines.append(
            "metrics: "
            + " ".join(f"{name}={value}" for name, value in metrics.items())
        )
    query_coverage = 100.0 * stats.query_period_ns / stats.all_period_ns
    lines.extend(
        (
            f"event: {stats.event}",
            f"all sampled CPU: {format_ms(stats.all_period_ns)} ms "
            f"({stats.all_samples} samples)",
            f"query sampled CPU: {format_ms(stats.query_period_ns)} ms "
            f"({stats.query_samples} samples, {query_coverage:.2f}% of process CPU)",
            "",
            "Query stages (percentages use query CPU as denominator):",
        )
    )

    known_stage_names = [stage for stage, _ in STAGE_MARKERS] + [
        QUERY_UNCLASSIFIED
    ]
    for stage in known_stage_names:
        stage_stats = stats.stages.get(stage, StageStats())
        percent = 100.0 * stage_stats.period_ns / stats.query_period_ns
        lines.append(
            f"  {percent:6.2f}%  {format_ms(stage_stats.period_ns):>9} ms  "
            f"{stage} ({stage_stats.samples} samples)"
        )

    append_hotspots(
        lines,
        "Top query leaf functions (exclusive sampled CPU):",
        stats.query_leaf_period_ns,
        stats.query_period_ns,
        top_n,
    )
    append_call_edges(
        lines,
        "Top query leaf/caller pairs (where the hotspot came from):",
        stats.query_leaf_caller_period_ns,
        stats.query_period_ns,
        top_n,
    )
    for stage in known_stage_names:
        stage_stats = stats.stages.get(stage)
        if stage_stats is None or stage_stats.period_ns == 0:
            continue
        append_hotspots(
            lines,
            f"Top leaf functions inside {stage}:",
            stage_stats.leaf_period_ns,
            stage_stats.period_ns,
            top_n,
        )

    lines.extend(
        (
            "",
            "Notes:",
            "  - Times are sampled user-space CPU time, not per-query wall time.",
            "  - Use --threads 1 when comparing the stage percentages with QPS.",
            "  - Sampling changes runtime slightly; use normal experiment QPS for final claims.",
        )
    )
    return "\n".join(lines) + "\n"


def run_perf(
    command: list[str],
    perf_data: pathlib.Path,
    log_path: pathlib.Path,
    frequency: int,
) -> None:
    perf = shutil.which("perf")
    if perf is None:
        raise RuntimeError("perf is not installed or not in PATH")
    if not command:
        raise RuntimeError("missing command after --")
    executable = pathlib.Path(command[0])
    if "/" in command[0] and not executable.exists():
        raise RuntimeError(f"executable does not exist: {command[0]}")

    perf_data.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    perf_command = [
        perf,
        "record",
        "-o",
        str(perf_data),
        "-e",
        "cpu-clock:u",
        "-F",
        str(frequency),
        "--call-graph",
        "dwarf,16384",
        "--",
        *command,
    ]
    with log_path.open("w") as log:
        result = subprocess.run(
            perf_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(
            f"perf or benchmark exited with status {result.returncode}\n{tail}"
        )


def analyze_perf(perf_data: pathlib.Path) -> ProfileStats:
    perf = shutil.which("perf")
    if perf is None:
        raise RuntimeError("perf is not installed or not in PATH")
    if not perf_data.is_file():
        raise RuntimeError(f"perf data does not exist: {perf_data}")

    process = subprocess.Popen(
        [perf, "script", "-i", str(perf_data), "--demangle"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    stats = parse_perf_script(process.stdout)
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"perf script exited with status {return_code}: {stderr.strip()}"
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile UHGH/UHGS/main-graph query stages with perf call stacks"
    )
    parser.add_argument(
        "--perf-data",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/uhg_stage_profile.data"),
        help="output perf.data path",
    )
    parser.add_argument(
        "--analyze-only",
        type=pathlib.Path,
        help="skip recording and analyze this existing cpu-clock perf.data",
    )
    parser.add_argument(
        "--log",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/uhg_stage_profile.log"),
        help="benchmark/perf output log",
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        help="also write the text report to this path",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=499,
        help="cpu-clock samples per second (default: 499)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="number of leaf hotspots per section (default: 12)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="benchmark command, placed after --",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if args.frequency <= 0:
        parser.error("--frequency must be positive")
    if args.top <= 0:
        parser.error("--top must be positive")
    if args.analyze_only is None and not args.command:
        parser.error("provide a benchmark command after --, or use --analyze-only")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.analyze_only is None:
            run_perf(args.command, args.perf_data, args.log, args.frequency)
            perf_data = args.perf_data
            command = args.command
        else:
            perf_data = args.analyze_only
            command = None
        stats = analyze_perf(perf_data)
        report = make_report(
            stats,
            command,
            read_metrics(args.log),
            args.top,
        )
        sys.stdout.write(report)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report)
        return 0
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
