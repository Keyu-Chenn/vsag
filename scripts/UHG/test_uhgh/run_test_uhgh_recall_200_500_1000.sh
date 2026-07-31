#!/usr/bin/env bash
# Run the UHGH dense-entry scale comparison at Recall@200/500/1000.
# The underlying experiment uses one process per ef_search point so that
# dense_entry_ef_search and dense_entry_bk are not overwritten by
# 703_uhg_exp3's --search_points mode.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
UHG_ROOT="$REPO_ROOT/scripts/UHG"
DATA_DIR="${DATA_DIR:-$UHG_ROOT/data}"
GT_ROOT="${GT_ROOT:-$DATA_DIR/ground_truth_1000}"
GT_VALIDATOR="${GT_VALIDATOR:-$GT_ROOT/validate_ground_truth.py}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
BASE_RUNNER="$UHG_ROOT/results/test_uhgh/run_test_uhgh.sh"

RECALLS_TEXT="${RECALLS_TEXT:-200 500 1000}"
DATASETS_TEXT="${DATASETS_TEXT:-nq}"
ENTRY_SCALE_TEXT="${ENTRY_SCALE_TEXT:-0.5 0.7 1.0 1.2 1.5}"
ALPHA="${ALPHA:-0.7}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"

read -r -a RECALLS <<< "$RECALLS_TEXT"
read -r -a DATASETS <<< "$DATASETS_TEXT"

current_stage="initialization"
on_error() {
    local status=$?
    echo "[$(date '+%F %T')] FAILED status=$status stage=$current_stage" >&2
    exit "$status"
}
trap on_error ERR

if [ ! -x "$BASE_RUNNER" ]; then
    echo "Missing executable base runner: $BASE_RUNNER" >&2
    exit 1
fi
if [ ! -f "$GT_VALIDATOR" ]; then
    echo "Missing GT validator: $GT_VALIDATOR" >&2
    exit 1
fi
if [ "${#RECALLS[@]}" -eq 0 ] || [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "No recalls or datasets configured" >&2
    exit 1
fi

# Validate every prerequisite before starting QPS measurements.
for dataset in "${DATASETS[@]}"; do
    current_stage="preflight dataset=$dataset"
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    hybrid_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"
    if [ ! -f "$h5_file" ]; then
        echo "Missing HDF5: $h5_file" >&2
        exit 1
    fi
    if [ ! -f "$hybrid_index" ]; then
        echo "Missing UHG hybrid index: $hybrid_index" >&2
        exit 1
    fi
    python3 "$GT_VALIDATOR" "$dataset" \
        --topk 1000 \
        --num-queries "$NUM_QUERIES" \
        --fixed-root "$GT_ROOT" \
        --hdf5-root "$DATA_DIR/hdf5" \
        --alphas "$ALPHA" \
        --fixed-only
done

echo "============================================================"
echo "UHGH dense-entry scale comparison"
echo "recalls:      ${RECALLS[*]}"
echo "datasets:     ${DATASETS[*]}"
echo "alpha:        $ALPHA"
echo "entry scales: $ENTRY_SCALE_TEXT"
echo "threads:      $THREADS"
echo "queries:      $NUM_QUERIES"
echo "============================================================"

for recall in "${RECALLS[@]}"; do
    case "$recall" in
        200)
            points_text="${POINTS_200_TEXT:-200 400 600 1000 2000}"
            ;;
        500)
            points_text="${POINTS_500_TEXT:-500 1000 1500 2500 5000}"
            ;;
        1000)
            points_text="${POINTS_1000_TEXT:-1000 2000 3000 5000 10000}"
            ;;
        *)
            echo "Unsupported recall: $recall (valid: 200 500 1000)" >&2
            exit 2
            ;;
    esac

    result_dir="$UHG_ROOT/results_${recall}/test_uhgh"
    log_dir="$result_dir/logs"
    mkdir -p "$log_dir"
    current_stage="Recall@$recall"

    echo "============================================================"
    echo "[$(date '+%F %T')] START Recall@$recall points=[$points_text]"
    echo "============================================================"

    K="$recall" \
    METHOD="uhgh" \
    ALPHA="$ALPHA" \
    POINTS_TEXT="$points_text" \
    ENTRY_SCALE_TEXT="$ENTRY_SCALE_TEXT" \
    DATASETS_TEXT="$DATASETS_TEXT" \
    NUM_QUERIES="$NUM_QUERIES" \
    THREADS="$THREADS" \
    DATA_DIR="$DATA_DIR" \
    GT_ROOT="$GT_ROOT" \
    GT_VALIDATOR="$GT_VALIDATOR" \
    INDEX_HYBRID_DIR="$INDEX_HYBRID_DIR" \
    RESULT_DIR="$result_dir" \
        bash "$BASE_RUNNER" all 2>&1 | tee "$log_dir/run_test_uhgh.log"

    if rg -n 'ERROR|NO_METRICS|Failed|Error:' "$result_dir" \
        -g '*.txt' -g '*.log'; then
        echo "Recall@$recall produced error markers; inspect $result_dir" >&2
        exit 1
    fi
    echo "[$(date '+%F %T')] DONE Recall@$recall results=$result_dir"
done

current_stage="complete"
echo "[$(date '+%F %T')] ALL UHGH RECALL SWEEPS COMPLETE"
