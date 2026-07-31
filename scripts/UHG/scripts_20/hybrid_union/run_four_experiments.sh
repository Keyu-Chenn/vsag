#!/usr/bin/env bash
# Run fixed-alpha main, mixed-alpha, entry ablation, and test-prune serially.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/scripts/UHG/results_20}"
DATASET="${1:-nq}"

case "$DATASET" in
    nq|hotpotqa|msmarco|fever|dbpedia-entity) ;;
    *)
        echo "Unknown dataset: $DATASET (valid: nq hotpotqa msmarco fever dbpedia-entity)"
        exit 1
        ;;
esac

mkdir -p \
    "$RESULT_ROOT/main/logs" \
    "$RESULT_ROOT/mixed_alpha/logs" \
    "$RESULT_ROOT/ablation/logs" \
    "$RESULT_ROOT/test_prune/logs"

echo "[$(date '+%F %T')] START $DATASET main (Recall@20)"
RESULT_DIR="$RESULT_ROOT/main" \
    "$SCRIPT_DIR/main/$DATASET/run_all.sh" \
    2>&1 | tee "$RESULT_ROOT/main/logs/run_all_${DATASET}.log"

echo "[$(date '+%F %T')] START $DATASET mixed_alpha (Recall@20)"
RESULT_DIR="$RESULT_ROOT/mixed_alpha" \
    "$SCRIPT_DIR/mixed_alpha/run_mixed_alpha.sh" "$DATASET" \
    2>&1 | tee "$RESULT_ROOT/mixed_alpha/logs/run_${DATASET}.log"

echo "[$(date '+%F %T')] START $DATASET ablation (Recall@20)"
RESULT_DIR="$RESULT_ROOT/ablation" \
    "$SCRIPT_DIR/ablation/run_ablation.sh" "$DATASET" \
    2>&1 | tee "$RESULT_ROOT/ablation/logs/run_ablation_${DATASET}.log"

echo "[$(date '+%F %T')] START $DATASET test_prune (Recall@20)"
RESULT_DIR="$RESULT_ROOT/test_prune" \
    "$SCRIPT_DIR/test_prune/run_test_prune.sh" "$DATASET" \
    2>&1 | tee "$RESULT_ROOT/test_prune/logs/run_test_prune_${DATASET}.log"

echo "[$(date '+%F %T')] DONE $DATASET four Recall@20 experiments"
