#!/usr/bin/env python3
"""Single entry point for the final UniGraph paper experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
MAIN_RUN = ROOT / "runs" / "main"
ENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def execute(script: str, arguments: list[str]) -> None:
    python = ENV_PYTHON if ENV_PYTHON.is_file() else Path(sys.executable)
    command = [str(python), str(SCRIPTS / script), *arguments]
    subprocess.run(command, cwd=ROOT.parents[1], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "indexes", "main", "multi-weight", "plot"),
        help="experiment group or utility",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "check":
        if args.arguments:
            parser.error("check does not accept additional arguments")
        execute("main.py", ["check"])
        execute("multi_weight.py", ["check"])
        return
    if args.command == "main":
        execute("main.py", args.arguments or ["check"])
        return
    if args.command == "indexes":
        execute("build_indexes.py", args.arguments or ["check"])
        return
    if args.command == "multi-weight":
        execute("multi_weight.py", args.arguments or ["check"])
        return
    plot_arguments = ["--run-root", str(MAIN_RUN), *args.arguments]
    execute("plot.py", plot_arguments)


if __name__ == "__main__":
    main()
