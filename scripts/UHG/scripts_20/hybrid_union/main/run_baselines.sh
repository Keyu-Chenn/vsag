#!/usr/bin/env bash
# recall@20 fixed-alpha baseline experiments: HNSW, SINDI, and HNSW+SINDI.
#
# Usage:
#   bash run_baselines.sh                         # all datasets, all baselines
#   bash run_baselines.sh nq                      # one dataset, all baselines
#   bash run_baselines.sh nq hnsw                 # one dataset, one baseline
#   bash run_baselines.sh all sindi               # all datasets, one baseline
#   bash nq/run_baselines.sh hnsw                 # dataset-specific wrapper
#
# Results:
#   $RESULT_DIR/<dataset>/<method>/alpha_<alpha>.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/index}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results_20/main}"

BIN_HNSW="${BIN_HNSW:-$REPO_ROOT/build-release/examples/cpp/708_uhg_exp8}"
BIN_SINDI="${BIN_SINDI:-$REPO_ROOT/build-release/examples/cpp/709_uhg_exp9}"
BIN_HNSW_SINDI="${BIN_HNSW_SINDI:-$REPO_ROOT/build-release/examples/cpp/701_uhg_exp1}"

K="${K:-20}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco fever dbpedia-entity}"
METHODS_TEXT="${METHODS_TEXT:-hnsw sindi hnsw_sindi}"
ALPHAS_TEXT="${ALPHAS_TEXT:-0.3 0.4 0.5 0.6 0.7}"
HNSW_POINTS_TEXT="${HNSW_POINTS_TEXT:-20 40 60 100 160 200}"
SINDI_POINTS_TEXT="${SINDI_POINTS_TEXT:-200 300 400 600 1000 2000}"
HNSW_SINDI_POINTS_TEXT="${HNSW_SINDI_POINTS_TEXT:-20 30 40 60 100 200}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"

read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a METHODS <<< "$METHODS_TEXT"
read -r -a ALPHAS <<< "$ALPHAS_TEXT"
read -r -a HNSW_POINTS <<< "$HNSW_POINTS_TEXT"
read -r -a SINDI_POINTS <<< "$SINDI_POINTS_TEXT"
read -r -a HNSW_SINDI_POINTS <<< "$HNSW_SINDI_POINTS_TEXT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
    DATASETS=("$1")
fi
if [ "$#" -ge 2 ] && [ "$2" != "all" ]; then
    METHODS=("$2")
fi

REBUILD_ARGS=()
if [ "${REBUILD:-0}" = "1" ]; then
    REBUILD_ARGS=(--rebuild)
fi

capture_metrics() {
    local output status summary
    output="$("$@" 2>&1)"
    status=$?
    summary="$(printf '%s\n' "$output" | awk '
        /^(Recall|QPS)/ {
            if (line == "") {
                line = $0
            } else {
                print line " " $0
                line = ""
            }
        }
        END {
            if (line != "") {
                print line
            }
        }
    ')"

    if [ "$status" -ne 0 ]; then
        printf 'ERROR status=%s\n%s' "$status" "$output"
    elif [ -n "$summary" ]; then
        printf '%s' "$summary"
    else
        printf 'NO_METRICS\n%s' "$output"
    fi
}

join_points() {
    local IFS=,
    printf '%s' "$*"
}

METRIC_LINES=()
load_metric_lines() {
    local output="$1"
    local expected="$2"
    local line
    mapfile -t METRIC_LINES <<< "$output"
    if [ "${#METRIC_LINES[@]}" -ne "$expected" ]; then
        return 1
    fi
    for line in "${METRIC_LINES[@]}"; do
        if [[ ! "$line" =~ ^Recall:.*QPS: ]]; then
            return 1
        fi
    done
}

binary_ready() {
    local bin="$1"
    if [ ! -x "$bin" ]; then
        echo "Skip: binary not executable: $bin"
        return 1
    fi
    return 0
}

write_header() {
    local outfile="$1"
    local method="$2"
    local dataset="$3"
    local alpha="$4"
    {
        echo "# recall@20 fixed-alpha baseline"
        echo "# method=$method dataset=$dataset alpha=$alpha k=$K num_queries=$NUM_QUERIES threads=$THREADS"
        echo "# data_dir=$DATA_DIR"
        echo "# index_dir=$INDEX_DIR"
        echo "# result_dir=$RESULT_DIR/$dataset/$method"
    } > "$outfile"
}

run_hnsw() {
    local dataset="$1"
    local h5_file="$2"
    local gt_dir="$3"
    binary_ready "$BIN_HNSW" || return 0

    for alpha in "${ALPHAS[@]}"; do
        local gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
        local method_dir="$RESULT_DIR/$dataset/hnsw"
        local outfile="$method_dir/alpha_${alpha}.txt"
        mkdir -p "$method_dir"
        if [ ! -f "$gt_file" ]; then
            echo "Skip $dataset hnsw alpha=$alpha: GT not found: $gt_file"
            continue
        fi
        write_header "$outfile" "hnsw" "$dataset" "$alpha"
        echo "--- $dataset hnsw alpha=$alpha ---"

        local result points_csv i point
        points_csv="$(join_points "${HNSW_POINTS[@]}")"
        result="$(capture_metrics \
            "$BIN_HNSW" "$h5_file" \
            --gt_dir "$gt_dir" \
            --index_dir "$INDEX_DIR" \
            --alpha "$alpha" \
            -k "$K" \
            --search_points "$points_csv" \
            --num_queries "$NUM_QUERIES" \
            --threads "$THREADS" \
            ${REBUILD_ARGS[@]+"${REBUILD_ARGS[@]}"})"
        if ! load_metric_lines "$result" "${#HNSW_POINTS[@]}"; then
            printf 'ERROR: expected %d HNSW metric lines\n%s\n' \
                "${#HNSW_POINTS[@]}" "$result" | tee -a "$outfile"
            continue
        fi
        for i in "${!HNSW_POINTS[@]}"; do
            point="${HNSW_POINTS[$i]}"
            echo "hnsw bk=$point ef_search=$point threads=$THREADS ${METRIC_LINES[$i]}" | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
}

run_sindi() {
    local dataset="$1"
    local h5_file="$2"
    local gt_dir="$3"
    binary_ready "$BIN_SINDI" || return 0

    for alpha in "${ALPHAS[@]}"; do
        local gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
        local method_dir="$RESULT_DIR/$dataset/sindi"
        local outfile="$method_dir/alpha_${alpha}.txt"
        mkdir -p "$method_dir"
        if [ ! -f "$gt_file" ]; then
            echo "Skip $dataset sindi alpha=$alpha: GT not found: $gt_file"
            continue
        fi
        write_header "$outfile" "sindi" "$dataset" "$alpha"
        echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO" >> "$outfile"
        echo "--- $dataset sindi alpha=$alpha ---"

        local result points_csv i point
        points_csv="$(join_points "${SINDI_POINTS[@]}")"
        result="$(capture_metrics \
            "$BIN_SINDI" "$h5_file" \
            --gt_dir "$gt_dir" \
            --index_dir "$INDEX_DIR" \
            --alpha "$alpha" \
            -k "$K" \
            --bk_list "$points_csv" \
            --query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
            --term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
            --num_queries "$NUM_QUERIES" \
            --threads "$THREADS" \
            ${REBUILD_ARGS[@]+"${REBUILD_ARGS[@]}"})"
        if ! load_metric_lines "$result" "${#SINDI_POINTS[@]}"; then
            printf 'ERROR: expected %d SINDI metric lines\n%s\n' \
                "${#SINDI_POINTS[@]}" "$result" | tee -a "$outfile"
            continue
        fi
        for i in "${!SINDI_POINTS[@]}"; do
            point="${SINDI_POINTS[$i]}"
            echo "sindi bk=$point query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS ${METRIC_LINES[$i]}" | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
}

run_hnsw_sindi() {
    local dataset="$1"
    local h5_file="$2"
    local gt_dir="$3"
    binary_ready "$BIN_HNSW_SINDI" || return 0

    for alpha in "${ALPHAS[@]}"; do
        local gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
        local method_dir="$RESULT_DIR/$dataset/hnsw_sindi"
        local outfile="$method_dir/alpha_${alpha}.txt"
        mkdir -p "$method_dir"
        if [ ! -f "$gt_file" ]; then
            echo "Skip $dataset hnsw_sindi alpha=$alpha: GT not found: $gt_file"
            continue
        fi
        write_header "$outfile" "hnsw_sindi" "$dataset" "$alpha"
        echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO" >> "$outfile"
        echo "--- $dataset hnsw_sindi alpha=$alpha ---"

        local result points_csv i point
        points_csv="$(join_points "${HNSW_SINDI_POINTS[@]}")"
        result="$(capture_metrics \
            "$BIN_HNSW_SINDI" "$h5_file" \
            --gt_dir "$gt_dir" \
            --index_dir "$INDEX_DIR" \
            --alpha "$alpha" \
            -k "$K" \
            --search_points "$points_csv" \
            --query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
            --term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
            --num_queries "$NUM_QUERIES" \
            --threads "$THREADS" \
            ${REBUILD_ARGS[@]+"${REBUILD_ARGS[@]}"})"
        if ! load_metric_lines "$result" "${#HNSW_SINDI_POINTS[@]}"; then
            printf 'ERROR: expected %d HNSW+SINDI metric lines\n%s\n' \
                "${#HNSW_SINDI_POINTS[@]}" "$result" | tee -a "$outfile"
            continue
        fi
        for i in "${!HNSW_SINDI_POINTS[@]}"; do
            point="${HNSW_SINDI_POINTS[$i]}"
            echo "hnsw_sindi bk=$point ef_search=$point query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS ${METRIC_LINES[$i]}" | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
}

for dataset in "${DATASETS[@]}"; do
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    gt_dir="$DATA_DIR/ground_truth/$dataset"

    if [ ! -f "$h5_file" ]; then
        echo "Skip $dataset: hdf5 not found: $h5_file"
        continue
    fi
    if [ ! -d "$gt_dir" ]; then
        echo "Skip $dataset: GT directory not found: $gt_dir"
        continue
    fi

    for method in "${METHODS[@]}"; do
        echo "============================================"
        echo "  Dataset: $dataset  Method: $method"
        echo "============================================"
        case "$method" in
            hnsw)
                run_hnsw "$dataset" "$h5_file" "$gt_dir"
                ;;
            sindi)
                run_sindi "$dataset" "$h5_file" "$gt_dir"
                ;;
            hnsw_sindi)
                run_hnsw_sindi "$dataset" "$h5_file" "$gt_dir"
                ;;
            *)
                echo "Unknown baseline method: $method (valid: hnsw sindi hnsw_sindi)"
                ;;
        esac
    done
done

echo "Baseline experiments complete. Results: $RESULT_DIR"
