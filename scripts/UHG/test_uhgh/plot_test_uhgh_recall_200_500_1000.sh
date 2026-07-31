#!/usr/bin/env bash
# Plot entry-scale QPS-Recall curves for the high-recall UHGH sweeps.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
UHG_ROOT="$REPO_ROOT/scripts/UHG"
PLOTTER="$UHG_ROOT/results/test_uhgh/plot_test_uhgh.py"
RECALLS_TEXT="${RECALLS_TEXT:-200 500 1000}"
DATASETS_TEXT="${DATASETS_TEXT:-nq}"

read -r -a RECALLS <<< "$RECALLS_TEXT"
read -r -a DATASETS <<< "$DATASETS_TEXT"

for recall in "${RECALLS[@]}"; do
    result_root="$UHG_ROOT/results_${recall}/test_uhgh"
    echo "Plot Recall@$recall from $result_root"
    RESULT_ROOT="$result_root" RECALL_AT="$recall" \
        python3 "$PLOTTER" "${DATASETS[@]}"
done

