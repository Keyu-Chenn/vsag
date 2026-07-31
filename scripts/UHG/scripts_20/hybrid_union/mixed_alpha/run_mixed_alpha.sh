#!/usr/bin/env bash
# Run the five methods at Recall@20 with one alpha per query.
# Usage:
#   bash run_mixed_alpha.sh nq             # all five methods
#   bash run_mixed_alpha.sh nq sindi       # one method
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/index}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results_20/mixed_alpha}"

BIN_HNSW="${BIN_HNSW:-$REPO_ROOT/build-release/examples/cpp/708_uhg_exp8}"
BIN_SINDI="${BIN_SINDI:-$REPO_ROOT/build-release/examples/cpp/709_uhg_exp9}"
BIN_HNSW_SINDI="${BIN_HNSW_SINDI:-$REPO_ROOT/build-release/examples/cpp/701_uhg_exp1}"
BIN_HYBRID="${BIN_HYBRID:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

DATASET="${1:-nq}"
METHOD="${2:-all}"
K="${K:-20}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
HNSW_POINTS_TEXT="${HNSW_POINTS_TEXT:-20 40 60 100 160 200}"
SINDI_POINTS_TEXT="${SINDI_POINTS_TEXT:-200 300 400 600 1000 2000}"
HNSW_SINDI_POINTS_TEXT="${HNSW_SINDI_POINTS_TEXT:-20 30 40 60 100 200}"
FHG_POINTS_TEXT="${FHG_POINTS_TEXT:-20 40 60 100 160 200}"
UHG_POINTS_TEXT="${UHG_POINTS_TEXT:-20 40 60 100 160 200}"

case "$METHOD" in
    all|hnsw|sindi|hnsw_sindi|fhg|uhg) ;;
    *)
        echo "Unknown method: $METHOD (valid: all hnsw sindi hnsw_sindi fhg uhg)"
        exit 1
        ;;
esac

read -r -a HNSW_POINTS <<< "$HNSW_POINTS_TEXT"
read -r -a SINDI_POINTS <<< "$SINDI_POINTS_TEXT"
read -r -a HNSW_SINDI_POINTS <<< "$HNSW_SINDI_POINTS_TEXT"
read -r -a FHG_POINTS <<< "$FHG_POINTS_TEXT"
read -r -a UHG_POINTS <<< "$UHG_POINTS_TEXT"

H5_FILE="$DATA_DIR/hdf5/${DATASET}.hdf5"
MIXED_DIR="$DATA_DIR/mixed_alpha/$DATASET"
ALPHA_FILE="$MIXED_DIR/alphas.npy"
GT_FILE="$MIXED_DIR/ground_truth.npy"
FHG_GRAPH="$DATA_DIR/fhg/${DATASET}_fhg_alpha_0_5.h5"
UHG_GRAPH="$DATA_DIR/uhg/${DATASET}_uhg.h5"
FHG_INDEX="$INDEX_DIR/703_${DATASET}_fhg_hybrid_index.index"
UHG_INDEX="$INDEX_HYBRID_DIR/703_${DATASET}_hybrid_index.index"
DENSE_ENTRY_HNSW="$INDEX_DIR/701_${DATASET}_dense_hnsw.index"
BASELINE_SINDI="$INDEX_DIR/701_${DATASET}_sparse_sindi.index"
OUT_DIR="$RESULT_DIR/$DATASET"

for path in "$H5_FILE" "$ALPHA_FILE" "$GT_FILE" "$DENSE_ENTRY_HNSW" "$BASELINE_SINDI"; do
    if [ ! -f "$path" ]; then
        echo "Missing required file: $path"
        exit 1
    fi
done
for bin in "$BIN_HNSW" "$BIN_SINDI" "$BIN_HNSW_SINDI" "$BIN_HYBRID"; do
    if [ ! -x "$bin" ]; then
        echo "Missing executable: $bin"
        exit 1
    fi
done
mkdir -p "$OUT_DIR"

capture_metrics() {
    local output status summary
    output="$("$@" 2>&1)"
    status=$?
    summary="$(printf '%s\n' "$output" | awk '
        /^(Recall|QPS)/ {
            if (line == "") line = $0
            else { print line " " $0; line = "" }
        }
        END { if (line != "") print line }
    ')"
    if [ "$status" -ne 0 ]; then
        printf 'ERROR status=%s\n%s' "$status" "$output"
    elif [ -n "$summary" ]; then
        printf '%s' "$summary"
    else
        printf 'NO_METRICS\n%s' "$output"
    fi
}

write_header() {
    local method="$1" outfile="$2"
    {
        echo "# experiment=mixed_alpha method=$method dataset=$DATASET"
        echo "# distribution=uniform alpha_min=0.3 alpha_max=0.7 seed=42"
        echo "# k=$K num_queries=$NUM_QUERIES threads=$THREADS"
        echo "# mixed_alpha_file=$ALPHA_FILE"
        echo "# mixed_gt_file=$GT_FILE"
        if [ "$method" = "sindi" ] || [ "$method" = "hnsw_sindi" ]; then
            echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
        fi
        if [ "$method" = "fhg" ]; then
            echo "# hybrid_pruning=false"
        fi
    } > "$outfile"
}

run_baseline() {
    local method="$1" bin="$2" outfile="$OUT_DIR/$1.txt"
    shift 2
    local -a points=("$@")
    write_header "$method" "$outfile"
    for point in "${points[@]}"; do
        local -a args=(
            "$bin" "$H5_FILE"
            --mixed_alpha_file "$ALPHA_FILE"
            --mixed_gt_file "$GT_FILE"
            --index_dir "$INDEX_DIR"
            --alpha 0.5 -k "$K" --bk "$point"
            --num_queries "$NUM_QUERIES" --threads "$THREADS"
        )
        if [ "$method" != "sindi" ]; then
            args+=(--ef_search "$point")
        fi
        if [ "$method" = "sindi" ] || [ "$method" = "hnsw_sindi" ]; then
            args+=(--query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO"
                   --term_prune_ratio "$SINDI_TERM_PRUNE_RATIO")
        fi
        result="$(capture_metrics "${args[@]}")"
        if [ "$method" = "sindi" ]; then
            echo "$method point=$point query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS $result" | tee -a "$outfile"
        elif [ "$method" = "hnsw_sindi" ]; then
            echo "$method point=$point query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS $result" | tee -a "$outfile"
        else
            echo "$method point=$point threads=$THREADS $result" | tee -a "$outfile"
        fi
    done
}

run_hybrid() {
    local method="$1" outfile="$OUT_DIR/$1.txt" graph index
    shift
    local -a points=("$@")
    local -a method_args
    if [ "$method" = "fhg" ]; then
        graph="$FHG_GRAPH"
        index="$FHG_INDEX"
        method_args=(--method uhg --disable_sindi --disable_dense_entry --disable_hybrid_pruning)
    else
        graph="$UHG_GRAPH"
        index="$UHG_INDEX"
        method_args=(--method auto --dense_entry_hnsw_graph_path "$DENSE_ENTRY_HNSW"
                     --hybrid_prune_scale 0.3
                     --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO"
                     --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO")
    fi
    local -a required_paths=("$graph" "$index")
    if [ "$method" = "uhg" ]; then
        required_paths+=("$DENSE_ENTRY_HNSW")
    fi
    for path in "${required_paths[@]}"; do
        if [ ! -f "$path" ]; then
            echo "Missing required $method file: $path"
            exit 1
        fi
    done
    write_header "$method" "$outfile"
    for point in "${points[@]}"; do
        local -a args=(
            "$BIN_HYBRID" "$H5_FILE"
            --mixed_alpha_file "$ALPHA_FILE"
            --mixed_gt_file "$GT_FILE"
            --graph_path "$graph"
            --hybrid_index_path "$index"
            --index_dir "$(dirname "$index")"
            --alpha 0.5 --build_alpha 0.5 -k "$K"
            --ef_search "$point" --sindi_bk "$point"
            --dense_entry_bk "$point" --dense_entry_ef_search "$point"
            --max_hops 0 --num_queries "$NUM_QUERIES" --threads "$THREADS"
            "${method_args[@]}"
        )
        result="$(capture_metrics "${args[@]}")"
        echo "$method ef_search=$point threads=$THREADS $result" | tee -a "$outfile"
    done
}

case "$METHOD" in
    all)
        run_baseline hnsw "$BIN_HNSW" "${HNSW_POINTS[@]}"
        run_baseline sindi "$BIN_SINDI" "${SINDI_POINTS[@]}"
        run_baseline hnsw_sindi "$BIN_HNSW_SINDI" "${HNSW_SINDI_POINTS[@]}"
        run_hybrid fhg "${FHG_POINTS[@]}"
        run_hybrid uhg "${UHG_POINTS[@]}"
        ;;
    hnsw)
        run_baseline hnsw "$BIN_HNSW" "${HNSW_POINTS[@]}"
        ;;
    sindi)
        run_baseline sindi "$BIN_SINDI" "${SINDI_POINTS[@]}"
        ;;
    hnsw_sindi)
        run_baseline hnsw_sindi "$BIN_HNSW_SINDI" "${HNSW_SINDI_POINTS[@]}"
        ;;
    fhg)
        run_hybrid fhg "${FHG_POINTS[@]}"
        ;;
    uhg)
        run_hybrid uhg "${UHG_POINTS[@]}"
        ;;
esac

echo "Mixed-alpha experiments complete: $OUT_DIR"
