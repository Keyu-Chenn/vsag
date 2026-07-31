#!/usr/bin/env bash
# Serially generate truncated-normal alphas and exact GT for all five datasets.
#
# This script only prepares experiment data. It never runs an ANN experiment
# and never builds or deletes an index.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_ROOT/scripts/UHG/mixed_alpha_zt}"
SHARED_DATA_DIR="${SHARED_DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco dbpedia-entity fever}"
NUM_QUERIES="${NUM_QUERIES:--1}"
QUERY_CHUNK="${QUERY_CHUNK:-32}"
TRAIN_CHUNK="${TRAIN_CHUNK:-50000}"
PREPARE_RESUME="${PREPARE_RESUME:-1}"

GENERATOR="$EXPERIMENT_ROOT/prepare_normal_mixed_alpha.py"
VALIDATOR="$EXPERIMENT_ROOT/validate_inputs.py"
DATA_ROOT="$EXPERIMENT_ROOT/data"
LOG_ROOT="$EXPERIMENT_ROOT/logs/prepare"

read -r -a DATASETS <<< "$DATASETS_TEXT"

if [ "$PREPARE_RESUME" != "0" ] && [ "$PREPARE_RESUME" != "1" ]; then
    echo "PREPARE_RESUME must be 0 or 1" >&2
    exit 2
fi

run_logged() {
    local log_file="$1" status
    shift
    mkdir -p "$(dirname "$log_file")"
    set +e
    "$@" 2>&1 | tee "$log_file"
    status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Command failed with status $status; log: $log_file" >&2
        return "$status"
    fi
}

for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        nq|hotpotqa|msmarco|dbpedia-entity|fever) ;;
        *)
            echo "Unsupported dataset: $dataset" >&2
            exit 2
            ;;
    esac

    if [ "$PREPARE_RESUME" = "1" ] &&
       python3 "$VALIDATOR" \
           --shared-data-dir "$SHARED_DATA_DIR" \
           --experiment-root "$EXPERIMENT_ROOT" \
           --datasets "$dataset" \
           >/dev/null 2>&1; then
        echo "Skip completed normal alpha/GT data: $dataset"
        continue
    fi

    log_file="$LOG_ROOT/${dataset}.log"
    run_logged "$log_file" \
        python3 "$GENERATOR" \
            --hdf5 "$SHARED_DATA_DIR/hdf5/${dataset}.hdf5" \
            --output-dir "$DATA_ROOT/gt_top200/$dataset" \
            --alpha-min 0.3 \
            --alpha-max 0.7 \
            --normal-mean 0.5 \
            --normal-stddev 0.1 \
            --seed 42 \
            --topk 200 \
            --num-queries "$NUM_QUERIES" \
            --query-chunk "$QUERY_CHUNK" \
            --train-chunk "$TRAIN_CHUNK"

    python3 "$VALIDATOR" \
        --shared-data-dir "$SHARED_DATA_DIR" \
        --experiment-root "$EXPERIMENT_ROOT" \
        --datasets "$dataset"
done

python3 "$VALIDATOR" \
    --shared-data-dir "$SHARED_DATA_DIR" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --datasets "${DATASETS[@]}"

echo "All truncated-normal alpha/exact-top-200 GT data prepared: $DATA_ROOT"
