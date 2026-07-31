#!/usr/bin/env bash
# Ablation: compare direct UHG search with the main experiment's entry strategy.
#
# This script only runs direct UHG search on the SAME pre-built UHG hybrid_index
# (data/index_hybrid_union/703_<dataset>_hybrid_index.index, built in main
# runbook step 8.2) at alpha=0.3 and alpha=0.7. Final reporting compares:
#
#   alpha=0.3: ablation UHG vs main experiment UHGS (auto route)
#   alpha=0.7: ablation UHG vs main experiment UHGH (auto route)
#
# Usage:
#   bash run_ablation.sh                       # all datasets, direct UHG only
#   bash run_ablation.sh nq                    # one dataset, direct UHG only
#   bash nq/run_ablation.sh                    # dataset-specific wrapper
#
# Results:
#   $RESULT_DIR/<dataset>/<method>/alpha_<alpha>.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results/ablation}"
BIN="${BIN:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

K="${K:-100}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
HYBRID_PRUNE_SCALE=0.3
MAX_HOPS="${MAX_HOPS:-0}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco dbpedia-entity}"
POINTS_TEXT="${POINTS_TEXT:-100 200 300 500 1000}"

read -r -a DATASETS <<< "$DATASETS_TEXT"
METHODS=(uhg)
ALPHAS=(0.3 0.7)
read -r -a POINTS <<< "$POINTS_TEXT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
    DATASETS=("$1")
fi
if [ "$#" -ge 2 ]; then
    echo "Unexpected method argument: this ablation only runs method=uhg"
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

for dataset in "${DATASETS[@]}"; do
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    gt_dir="$DATA_DIR/ground_truth/$dataset"
    uhg_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"

    if [ ! -f "$h5_file" ]; then
        echo "Skip $dataset: hdf5 not found: $h5_file"
        continue
    fi
    if [ ! -d "$gt_dir" ]; then
        echo "Skip $dataset: GT directory not found: $gt_dir"
        continue
    fi
    if [ ! -f "$uhg_index" ]; then
        echo "Skip $dataset: UHG hybrid_index not found: $uhg_index"
        echo "  Build it first via MAIN_EXPERIMENT_RUNBOOK.md step 8.2"
        continue
    fi

    for method in "${METHODS[@]}"; do
        method_dir="$RESULT_DIR/$dataset/$method"
        mkdir -p "$method_dir"
        echo "============================================"
        echo "  Dataset: $dataset  Method: $method (ablation)"
        echo "============================================"

        for alpha in "${ALPHAS[@]}"; do
            gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
            outfile="$method_dir/alpha_${alpha}.txt"
            if [ ! -f "$gt_file" ]; then
                echo "Skip $dataset $method alpha=$alpha: GT not found: $gt_file"
                continue
            fi

            {
                echo "# ablation: entry strategy"
                echo "# method=$method dataset=$dataset alpha=$alpha k=$K num_queries=$NUM_QUERIES threads=$THREADS"
                echo "# fixed index: $uhg_index (SINDI + dense-entry + UHG main graph embedded at build time)"
                echo "# direct UHG baseline; compare with main auto result at the same alpha and ef_search"
                echo "# build-time paths (graph/sindi/dense_entry) are NOT passed: index already has them"
                echo "# build_alpha=0.5 hybrid_prune_scale=$HYBRID_PRUNE_SCALE max_hops=$MAX_HOPS"
                echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
                echo "# result_dir=$method_dir"
            } > "$outfile"

            echo "--- $dataset $method alpha=$alpha ---"
            for ef in "${POINTS[@]}"; do
                result="$(capture_metrics \
                    "$BIN" "$h5_file" \
                    --hybrid_index_path "$uhg_index" \
                    --alpha "$alpha" \
                    --build_alpha 0.5 \
                    --method "$method" \
                    -k "$K" \
                    --ef_search "$ef" \
                    --sindi_bk "$ef" \
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
                echo "$method ef_search=$ef prune_scale=$HYBRID_PRUNE_SCALE sindi_qp=$SINDI_QUERY_PRUNE_RATIO threads=$THREADS $result" | tee -a "$outfile"
            done
            echo "Done: $outfile"
        done
    done
done

echo "Ablation experiments complete. Results: $RESULT_DIR"
