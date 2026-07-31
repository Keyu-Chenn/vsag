#!/usr/bin/env bash
# Run one mixed-alpha (dataset, Recall@K, method) search job.
#
# Usage:
#   bash run_search.sh <dataset> <10|20|200> <hnsw|sindi|hnsw_sindi|fhg|uhg>
#
# This script never builds or deletes an index. Hybrid indexes must already
# exist; run_all.sh owns their lifecycle.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_ROOT/scripts/UHG/mixed_alpha}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/index}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
RESULT_ROOT="${RESULT_ROOT:-$EXPERIMENT_ROOT/results}"

BIN_HNSW="${BIN_HNSW:-$REPO_ROOT/build-release/examples/cpp/708_uhg_exp8}"
BIN_SINDI="${BIN_SINDI:-$REPO_ROOT/build-release/examples/cpp/709_uhg_exp9}"
BIN_HNSW_SINDI="${BIN_HNSW_SINDI:-$REPO_ROOT/build-release/examples/cpp/701_uhg_exp1}"
BIN_HYBRID="${BIN_HYBRID:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

DATASET="${1:-}"
K="${2:-}"
METHOD="${3:-}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
RESUME="${RESUME:-1}"
CHECK_COMPLETE_ONLY="${CHECK_COMPLETE_ONLY:-0}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.3}"
MAX_HOPS="${MAX_HOPS:-0}"

usage() {
    echo "Usage: $0 <dataset> <10|20|200> <hnsw|sindi|hnsw_sindi|fhg|uhg>" >&2
}

case "$DATASET" in
    nq|hotpotqa|msmarco|dbpedia-entity|fever) ;;
    *)
        usage
        echo "Unsupported dataset: ${DATASET:-<empty>}" >&2
        exit 2
        ;;
esac

case "$METHOD" in
    hnsw|sindi|hnsw_sindi|fhg|uhg) ;;
    *)
        usage
        echo "Unsupported method: ${METHOD:-<empty>}" >&2
        exit 2
        ;;
esac

if [ "$RESUME" != "0" ] && [ "$RESUME" != "1" ]; then
    echo "RESUME must be 0 or 1" >&2
    exit 2
fi
if [ "$CHECK_COMPLETE_ONLY" != "0" ] && [ "$CHECK_COMPLETE_ONLY" != "1" ]; then
    echo "CHECK_COMPLETE_ONLY must be 0 or 1" >&2
    exit 2
fi

case "$K" in
    10)
        MIXED_ROOT="${MIXED_ROOT:-$DATA_DIR/mixed_alpha}"
        HNSW_POINTS_TEXT="${HNSW_POINTS_TEXT:-10 20 30 50 80 100}"
        SINDI_POINTS_TEXT="${SINDI_POINTS_TEXT:-100 150 200 300 500 1000}"
        HNSW_SINDI_POINTS_TEXT="${HNSW_SINDI_POINTS_TEXT:-10 15 20 30 50 100}"
        FHG_POINTS_TEXT="${FHG_POINTS_TEXT:-10 20 30 50 80 100}"
        UHG_POINTS_TEXT="${UHG_POINTS_TEXT:-10 20 30 50 80 100}"
        ;;
    20)
        MIXED_ROOT="${MIXED_ROOT:-$DATA_DIR/mixed_alpha}"
        HNSW_POINTS_TEXT="${HNSW_POINTS_TEXT:-20 40 60 100 160 200}"
        SINDI_POINTS_TEXT="${SINDI_POINTS_TEXT:-200 300 400 600 1000 2000}"
        HNSW_SINDI_POINTS_TEXT="${HNSW_SINDI_POINTS_TEXT:-20 30 40 60 100 200}"
        FHG_POINTS_TEXT="${FHG_POINTS_TEXT:-20 40 60 100 160 200}"
        UHG_POINTS_TEXT="${UHG_POINTS_TEXT:-20 40 60 100 160 200}"
        ;;
    200)
        MIXED_ROOT="${MIXED_ROOT:-$DATA_DIR/mixed_alpha_gt_1000}"
        HNSW_POINTS_TEXT="${HNSW_POINTS_TEXT:-200 400 600 1000 1600 2000}"
        SINDI_POINTS_TEXT="${SINDI_POINTS_TEXT:-2000 3000 4000 6000 10000 20000}"
        HNSW_SINDI_POINTS_TEXT="${HNSW_SINDI_POINTS_TEXT:-200 300 400 600 1000}"
        FHG_POINTS_TEXT="${FHG_POINTS_TEXT:-200 400 600 1000 2000}"
        UHG_POINTS_TEXT="${UHG_POINTS_TEXT:-200 400 600 1000 2000}"
        ;;
    *)
        usage
        echo "Unsupported Recall@K: ${K:-<empty>}" >&2
        exit 2
        ;;
esac

case "$METHOD" in
    hnsw) POINTS_TEXT="$HNSW_POINTS_TEXT" ;;
    sindi) POINTS_TEXT="$SINDI_POINTS_TEXT" ;;
    hnsw_sindi) POINTS_TEXT="$HNSW_SINDI_POINTS_TEXT" ;;
    fhg) POINTS_TEXT="$FHG_POINTS_TEXT" ;;
    uhg) POINTS_TEXT="$UHG_POINTS_TEXT" ;;
esac
read -r -a POINTS <<< "$POINTS_TEXT"
if [ "${#POINTS[@]}" -eq 0 ]; then
    echo "No search points configured for $METHOD Recall@$K" >&2
    exit 2
fi
for point in "${POINTS[@]}"; do
    if ! [[ "$point" =~ ^[1-9][0-9]*$ ]] || [ "$point" -lt "$K" ]; then
        echo "Invalid search point for Recall@$K: $point" >&2
        exit 2
    fi
done

H5_FILE="$DATA_DIR/hdf5/${DATASET}.hdf5"
MIXED_DIR="$MIXED_ROOT/$DATASET"
ALPHA_FILE="$MIXED_DIR/alphas.npy"
GT_FILE="$MIXED_DIR/ground_truth.npy"
DENSE_HNSW="$INDEX_DIR/701_${DATASET}_dense_hnsw.index"
BASELINE_SINDI="$INDEX_DIR/701_${DATASET}_sparse_sindi.index"
FHG_GRAPH="$DATA_DIR/fhg/${DATASET}_fhg_alpha_0_5.h5"
UHG_GRAPH="$DATA_DIR/uhg/${DATASET}_uhg.h5"
FHG_INDEX="$INDEX_DIR/703_${DATASET}_fhg_hybrid_index.index"
UHG_INDEX="$INDEX_HYBRID_DIR/703_${DATASET}_hybrid_index.index"

OUT_DIR="$RESULT_ROOT/recall_${K}/$DATASET"
OUT_FILE="$OUT_DIR/$METHOD.txt"
RAW_LOG="$RESULT_ROOT/logs/recall_${K}/$DATASET/${METHOD}.log"

require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "Missing required file: $path" >&2
        exit 2
    fi
}

require_executable() {
    local path="$1"
    if [ ! -x "$path" ]; then
        echo "Missing executable: $path" >&2
        exit 2
    fi
}

result_is_complete() {
    local count i
    local -a found_points
    [ -f "$OUT_FILE" ] || return 1
    grep -Fqx \
        "# experiment=mixed_alpha dataset=$DATASET method=$METHOD recall_at=$K" \
        "$OUT_FILE" || return 1
    grep -Fqx \
        "# distribution=uniform alpha_min=0.3 alpha_max=0.7 seed=42" \
        "$OUT_FILE" || return 1
    grep -Fqx "# k=$K num_queries=$NUM_QUERIES threads=$THREADS" "$OUT_FILE" || return 1
    grep -Fqx "# points=$POINTS_TEXT" "$OUT_FILE" || return 1
    grep -Fqx "# mixed_alpha_file=$ALPHA_FILE" "$OUT_FILE" || return 1
    grep -Fqx "# mixed_gt_file=$GT_FILE" "$OUT_FILE" || return 1
    if [ "$METHOD" = "sindi" ] ||
       [ "$METHOD" = "hnsw_sindi" ] ||
       [ "$METHOD" = "uhg" ]; then
        grep -Fqx \
            "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO" \
            "$OUT_FILE" || return 1
    fi
    if [ "$METHOD" = "fhg" ]; then
        grep -Fqx \
            "# build_alpha=0.5 hybrid_pruning=false max_hops=$MAX_HOPS" \
            "$OUT_FILE" || return 1
    elif [ "$METHOD" = "uhg" ]; then
        grep -Fqx \
            "# method=auto build_alpha=0.5 hybrid_prune_scale=$HYBRID_PRUNE_SCALE max_hops=$MAX_HOPS" \
            "$OUT_FILE" || return 1
        grep -Fqx \
            "# auto_rule=alpha<=0.5:uhgs,alpha>0.5:uhgh" \
            "$OUT_FILE" || return 1
    fi
    if grep -Eq 'ERROR|NO_METRICS|Failed|Error:' "$OUT_FILE"; then
        return 1
    fi
    count="$(awk -v method="$METHOD" '
        $1 == method && /Recall:/ && /QPS:/ { count++ }
        END { print count + 0 }
    ' "$OUT_FILE")"
    [ "$count" -eq "${#POINTS[@]}" ] || return 1
    mapfile -t found_points < <(awk -v method="$METHOD" '
        $1 == method && /Recall:/ && /QPS:/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^(ef_search|bk|point)=[0-9]+$/) {
                    split($i, value, "=")
                    print value[2]
                    break
                }
            }
        }
    ' "$OUT_FILE")
    [ "${#found_points[@]}" -eq "${#POINTS[@]}" ] || return 1
    for i in "${!POINTS[@]}"; do
        [ "${found_points[$i]}" = "${POINTS[$i]}" ] || return 1
    done
}

if [ "$CHECK_COMPLETE_ONLY" = "1" ]; then
    if result_is_complete; then
        echo "Complete result: $OUT_FILE"
        exit 0
    fi
    echo "Incomplete result: $OUT_FILE" >&2
    exit 3
fi

if [ "$RESUME" = "1" ] && result_is_complete; then
    echo "Skip completed result: $OUT_FILE"
    exit 0
fi

for path in "$H5_FILE" "$ALPHA_FILE" "$GT_FILE" "$DENSE_HNSW" "$BASELINE_SINDI"; do
    require_file "$path"
done
case "$METHOD" in
    hnsw) require_executable "$BIN_HNSW" ;;
    sindi) require_executable "$BIN_SINDI" ;;
    hnsw_sindi) require_executable "$BIN_HNSW_SINDI" ;;
    fhg)
        require_executable "$BIN_HYBRID"
        require_file "$FHG_GRAPH"
        require_file "$FHG_INDEX"
        ;;
    uhg)
        require_executable "$BIN_HYBRID"
        require_file "$UHG_GRAPH"
        require_file "$UHG_INDEX"
        ;;
esac

mkdir -p "$OUT_DIR" "$(dirname "$RAW_LOG")"
: > "$RAW_LOG"
TMP_RESULT="$(mktemp "$OUT_DIR/.${METHOD}.txt.tmp.XXXXXX")"
cleanup_tmp() {
    if [ -n "${TMP_RESULT:-}" ] && [ -e "$TMP_RESULT" ]; then
        rm -f -- "$TMP_RESULT"
    fi
}
trap cleanup_tmp EXIT

join_points() {
    local IFS=,
    printf '%s' "$*"
}

run_binary() {
    local status
    set +e
    "$@" 2>&1 | tee -a "$RAW_LOG"
    status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Search command failed with status $status; raw log: $RAW_LOG" >&2
        return "$status"
    fi
}

{
    echo "# experiment=mixed_alpha dataset=$DATASET method=$METHOD recall_at=$K"
    echo "# distribution=uniform alpha_min=0.3 alpha_max=0.7 seed=42"
    echo "# k=$K num_queries=$NUM_QUERIES threads=$THREADS"
    echo "# points=$POINTS_TEXT"
    echo "# mixed_alpha_file=$ALPHA_FILE"
    echo "# mixed_gt_file=$GT_FILE"
    if [ "$METHOD" = "sindi" ] || [ "$METHOD" = "hnsw_sindi" ] || [ "$METHOD" = "uhg" ]; then
        echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
    fi
    if [ "$METHOD" = "fhg" ]; then
        echo "# build_alpha=0.5 hybrid_pruning=false max_hops=$MAX_HOPS"
    elif [ "$METHOD" = "uhg" ]; then
        echo "# method=auto build_alpha=0.5 hybrid_prune_scale=$HYBRID_PRUNE_SCALE max_hops=$MAX_HOPS"
        echo "# auto_rule=alpha<=0.5:uhgs,alpha>0.5:uhgh"
    fi
} > "$TMP_RESULT"

POINTS_CSV="$(join_points "${POINTS[@]}")"
case "$METHOD" in
    hnsw)
        run_binary \
            "$BIN_HNSW" "$H5_FILE" \
            --mixed_alpha_file "$ALPHA_FILE" \
            --mixed_gt_file "$GT_FILE" \
            --index_dir "$INDEX_DIR" \
            --alpha 0.5 -k "$K" \
            --search_points "$POINTS_CSV" \
            --num_queries "$NUM_QUERIES" --threads "$THREADS"
        ;;
    sindi)
        run_binary \
            "$BIN_SINDI" "$H5_FILE" \
            --mixed_alpha_file "$ALPHA_FILE" \
            --mixed_gt_file "$GT_FILE" \
            --index_dir "$INDEX_DIR" \
            --alpha 0.5 -k "$K" --bk_list "$POINTS_CSV" \
            --query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
            --term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
            --num_queries "$NUM_QUERIES" --threads "$THREADS"
        ;;
    hnsw_sindi)
        run_binary \
            "$BIN_HNSW_SINDI" "$H5_FILE" \
            --mixed_alpha_file "$ALPHA_FILE" \
            --mixed_gt_file "$GT_FILE" \
            --index_dir "$INDEX_DIR" \
            --alpha 0.5 -k "$K" \
            --search_points "$POINTS_CSV" \
            --query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
            --term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
            --num_queries "$NUM_QUERIES" --threads "$THREADS"
        ;;
    fhg)
        run_binary \
            "$BIN_HYBRID" "$H5_FILE" \
            --mixed_alpha_file "$ALPHA_FILE" \
            --mixed_gt_file "$GT_FILE" \
            --graph_path "$FHG_GRAPH" \
            --hybrid_index_path "$FHG_INDEX" \
            --index_dir "$INDEX_DIR" \
            --method uhg --alpha 0.5 --build_alpha 0.5 -k "$K" \
            --search_points "$POINTS_CSV" \
            --disable_sindi --disable_dense_entry --disable_hybrid_pruning \
            --max_hops "$MAX_HOPS" \
            --num_queries "$NUM_QUERIES" --threads "$THREADS"
        ;;
    uhg)
        run_binary \
            "$BIN_HYBRID" "$H5_FILE" \
            --mixed_alpha_file "$ALPHA_FILE" \
            --mixed_gt_file "$GT_FILE" \
            --graph_path "$UHG_GRAPH" \
            --dense_entry_hnsw_graph_path "$DENSE_HNSW" \
            --hybrid_index_path "$UHG_INDEX" \
            --index_dir "$INDEX_HYBRID_DIR" \
            --method auto --alpha 0.5 --build_alpha 0.5 -k "$K" \
            --search_points "$POINTS_CSV" \
            --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
            --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
            --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
            --max_hops "$MAX_HOPS" \
            --num_queries "$NUM_QUERIES" --threads "$THREADS"
        ;;
esac

mapfile -t METRIC_LINES < <(awk '
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
' "$RAW_LOG")

if [ "${#METRIC_LINES[@]}" -ne "${#POINTS[@]}" ]; then
    echo "Expected ${#POINTS[@]} Recall/QPS pairs, got ${#METRIC_LINES[@]}: $RAW_LOG" >&2
    exit 1
fi
for i in "${!POINTS[@]}"; do
    if [[ ! "${METRIC_LINES[$i]}" =~ ^Recall:.*QPS: ]]; then
        echo "Malformed metric pair: ${METRIC_LINES[$i]}" >&2
        exit 1
    fi
    case "$METHOD" in
        sindi)
            echo "$METHOD bk=${POINTS[$i]} query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS ${METRIC_LINES[$i]}" >> "$TMP_RESULT"
            ;;
        hnsw_sindi)
            echo "$METHOD point=${POINTS[$i]} query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS ${METRIC_LINES[$i]}" >> "$TMP_RESULT"
            ;;
        *)
            echo "$METHOD ef_search=${POINTS[$i]} threads=$THREADS ${METRIC_LINES[$i]}" >> "$TMP_RESULT"
            ;;
    esac
done

mv -- "$TMP_RESULT" "$OUT_FILE"
TMP_RESULT=""
echo "Completed: $OUT_FILE"
