#!/usr/bin/env bash
# Force UHGS at alpha=0.3/0.4/0.5 with:
#
#   sindi_bk = 3 * ef_search
#
# Existing UHG hybrid indexes are loaded as-is; this script never rebuilds an
# index. Each search point is run separately because --search_points couples
# sindi_bk and ef_search to the same value.
#
# Usage:
#   bash /tbase-project/vsag/scripts/UHG/results/test_uhgs/run_test_uhgs_sindi_3x.sh
#   bash /tbase-project/vsag/scripts/UHG/results/test_uhgs/run_test_uhgs_sindi_3x.sh nq
#
# Results:
#   <RESULT_DIR>/<dataset>/uhgs_sindi_3x/alpha_<0.3|0.4|0.5>.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results/test_uhgs}"
MAIN_RESULT_DIR="${MAIN_RESULT_DIR:-$REPO_ROOT/scripts/UHG/results/main}"
BIN="${BIN:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

K="${K:-100}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.3}"
MAX_HOPS="${MAX_HOPS:-0}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco fever}"
POINTS_TEXT="${POINTS_TEXT:-100 200 300 500 1000}"

METHOD="uhgs"
RESULT_VARIANT="uhgs_sindi_3x"
BUILD_ALPHA="0.5"
SINDI_BK_MULTIPLIER=3
ALPHAS=(0.3 0.4 0.5)
read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a POINTS <<< "$POINTS_TEXT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
    DATASETS=("$1")
fi
if [ "$#" -ge 2 ]; then
    echo "Unexpected argument: this experiment fixes alpha to 0.3, 0.4, and 0.5"
    exit 1
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

if [ ! -x "$BIN" ]; then
    echo "Missing binary: $BIN"
    exit 1
fi
if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "No datasets configured"
    exit 1
fi
if [ "${#POINTS[@]}" -eq 0 ]; then
    echo "No search points configured"
    exit 1
fi
for ef in "${POINTS[@]}"; do
    if [[ ! "$ef" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid ef_search point (expected a positive integer): $ef"
        exit 1
    fi
done

for dataset in "${DATASETS[@]}"; do
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    gt_dir="$DATA_DIR/ground_truth/$dataset"
    uhg_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"
    method_dir="$RESULT_DIR/$dataset/$RESULT_VARIANT"

    if [ ! -f "$h5_file" ]; then
        echo "Skip $dataset: hdf5 not found: $h5_file"
        continue
    fi
    if [ ! -d "$gt_dir" ]; then
        echo "Skip $dataset: ground-truth directory not found: $gt_dir"
        continue
    fi
    if [ ! -f "$uhg_index" ]; then
        echo "Skip $dataset: UHG hybrid_index not found: $uhg_index"
        echo "  Build it first via MAIN_EXPERIMENT_RUNBOOK.md step 8.2"
        continue
    fi

    mkdir -p "$method_dir"
    echo "============================================"
    echo "  Dataset: $dataset  Method: $METHOD"
    echo "  SINDI candidates: sindi_bk=${SINDI_BK_MULTIPLIER}*ef_search"
    echo "  Alphas: ${ALPHAS[*]}  Search points: ${POINTS[*]}"
    echo "============================================"

    for alpha in "${ALPHAS[@]}"; do
        gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
        outfile="$method_dir/alpha_${alpha}.txt"
        standard_uhgs_result="$RESULT_DIR/$dataset/uhgs/alpha_${alpha}.txt"
        main_auto_result="$MAIN_RESULT_DIR/$dataset/uhg/alpha_${alpha}.txt"

        if [ ! -f "$gt_file" ]; then
            echo "Skip $dataset $METHOD alpha=$alpha: ground truth not found: $gt_file"
            continue
        fi

        {
            echo "# UHGS with SINDI entry candidates set to 3x ef_search"
            echo "# method=$METHOD dataset=$dataset alpha=$alpha k=$K num_queries=$NUM_QUERIES threads=$THREADS"
            echo "# fixed index=$uhg_index (load only; no rebuild)"
            echo "# standard UHGS comparison=$standard_uhgs_result (sindi_bk=ef_search)"
            echo "# main auto comparison=$main_auto_result"
            echo "# build_alpha=$BUILD_ALPHA hybrid_prune_scale=$HYBRID_PRUNE_SCALE max_hops=$MAX_HOPS"
            echo "# sindi_bk=${SINDI_BK_MULTIPLIER}*ef_search"
            echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
            echo "# search_points=${POINTS[*]}"
        } > "$outfile"

        echo "--- $dataset $METHOD alpha=$alpha sindi_bk=${SINDI_BK_MULTIPLIER}*ef_search ---"
        for ef in "${POINTS[@]}"; do
            sindi_bk=$((ef * SINDI_BK_MULTIPLIER))
            result="$(capture_metrics \
                "$BIN" "$h5_file" \
                --hybrid_index_path "$uhg_index" \
                --alpha "$alpha" \
                --build_alpha "$BUILD_ALPHA" \
                --method "$METHOD" \
                -k "$K" \
                --ef_search "$ef" \
                --sindi_bk "$sindi_bk" \
                --dense_entry_bk "$ef" \
                --dense_entry_ef_search "$ef" \
                --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
                --max_hops "$MAX_HOPS" \
                --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
                --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
                --index_dir "$INDEX_HYBRID_DIR" \
                --gt_dir "$gt_dir" \
                --num_queries "$NUM_QUERIES" \
                --threads "$THREADS")"

            echo "$METHOD ef_search=$ef sindi_bk=$sindi_bk prune_scale=$HYBRID_PRUNE_SCALE sindi_qp=$SINDI_QUERY_PRUNE_RATIO threads=$THREADS $result" \
                | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
done

echo "UHGS SINDI-3x experiments complete. Results: $RESULT_DIR"
