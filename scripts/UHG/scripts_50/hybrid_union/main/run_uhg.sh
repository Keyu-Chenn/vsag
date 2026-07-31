#!/usr/bin/env bash
# recall@50 fixed-alpha UHG search experiments.
#
# Default METHOD=auto:
#   alpha <= 0.5 -> UHGS
#   alpha > 0.5 -> UHGH
#
# Usage:
#   bash run_uhg.sh
#   bash run_uhg.sh nq
#   bash nq/run_uhg.sh
#
# Results:
#   $RESULT_DIR/<dataset>/uhg/alpha_<alpha>.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/index}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results_50/main}"
BIN="${BIN:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"
UHG_GRAPH_DIR="${UHG_GRAPH_DIR:-$DATA_DIR/uhg}"
UHG_GRAPH_SUFFIX="${UHG_GRAPH_SUFFIX:-uhg}"

K="${K:-50}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
METHOD="${METHOD:-auto}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.3}"
MAX_HOPS="${MAX_HOPS:-0}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco fever dbpedia-entity}"
ALPHAS_TEXT="${ALPHAS_TEXT:-0.3 0.4 0.5 0.6 0.7}"
POINTS_TEXT="${POINTS_TEXT:-50 100 150 250 500}"

read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a ALPHAS <<< "$ALPHAS_TEXT"
read -r -a POINTS <<< "$POINTS_TEXT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
    DATASETS=("$1")
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

if [ ! -x "$BIN" ]; then
    echo "Missing binary: $BIN"
    exit 1
fi

for dataset in "${DATASETS[@]}"; do
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    gt_dir="$DATA_DIR/ground_truth/$dataset"
    uhg_graph="$UHG_GRAPH_DIR/${dataset}_${UHG_GRAPH_SUFFIX}.h5"
    dense_entry_hnsw="$INDEX_DIR/701_${dataset}_dense_hnsw.index"
    uhg_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"
    method_dir="$RESULT_DIR/$dataset/uhg"

    if [ ! -f "$h5_file" ]; then
        echo "Skip $dataset: hdf5 not found: $h5_file"
        continue
    fi
    if [ ! -d "$gt_dir" ]; then
        echo "Skip $dataset: GT directory not found: $gt_dir"
        continue
    fi
    if [ ! -f "$uhg_graph" ]; then
        echo "Skip $dataset: UHG graph not found: $uhg_graph"
        continue
    fi
    if [ ! -f "$dense_entry_hnsw" ]; then
        echo "Skip $dataset: dense-entry HNSW index not found: $dense_entry_hnsw"
        continue
    fi

    mkdir -p "$method_dir" "$INDEX_HYBRID_DIR"
    echo "============================================"
    echo "  Dataset: $dataset  Method: uhg ($METHOD)"
    echo "============================================"

    for alpha in "${ALPHAS[@]}"; do
        gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
        outfile="$method_dir/alpha_${alpha}.txt"
        if [ ! -f "$gt_file" ]; then
            echo "Skip $dataset uhg alpha=$alpha: GT not found: $gt_file"
            continue
        fi

        {
            echo "# recall@50 fixed-alpha UHG"
            echo "# method=$METHOD dataset=$dataset alpha=$alpha k=$K num_queries=$NUM_QUERIES threads=$THREADS"
            echo "# auto rule: alpha<=0.5 -> uhgs, alpha>0.5 -> uhgh"
            echo "# graph_build: bk=100 query_prune_ratio=0.9 refine=none merge_max_degree=64"
            echo "# build_alpha=0.5"
            echo "# hybrid_prune_scale=$HYBRID_PRUNE_SCALE max_hops=$MAX_HOPS"
            echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
            echo "# graph_path=$uhg_graph"
            echo "# sindi_index_path=(none; external 703 SINDI is not used)"
            echo "# dense_entry_hnsw_graph_path=$dense_entry_hnsw"
            echo "# hybrid_index_path=$uhg_index"
            echo "# result_dir=$method_dir"
        } > "$outfile"

        echo "--- $dataset uhg alpha=$alpha ---"
        points_csv="$(join_points "${POINTS[@]}")"
        result="$(capture_metrics \
            "$BIN" "$h5_file" \
            --graph_path "$uhg_graph" \
            --dense_entry_hnsw_graph_path "$dense_entry_hnsw" \
            --hybrid_index_path "$uhg_index" \
            --alpha "$alpha" \
            --build_alpha 0.5 \
            --method "$METHOD" \
            -k "$K" \
            --search_points "$points_csv" \
            --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
            --max_hops "$MAX_HOPS" \
            --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
            --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
            --index_dir "$INDEX_HYBRID_DIR" \
            --gt_dir "$gt_dir" \
            --num_queries "$NUM_QUERIES" \
            --threads "$THREADS" \
            ${REBUILD_ARGS[@]+"${REBUILD_ARGS[@]}"})"
        if ! load_metric_lines "$result" "${#POINTS[@]}"; then
            printf 'ERROR: expected %d UHG metric lines\n%s\n' \
                "${#POINTS[@]}" "$result" | tee -a "$outfile"
            continue
        fi
        for i in "${!POINTS[@]}"; do
            ef="${POINTS[$i]}"
            echo "uhg method=$METHOD ef_search=$ef prune_scale=$HYBRID_PRUNE_SCALE sindi_qp=$SINDI_QUERY_PRUNE_RATIO threads=$THREADS ${METRIC_LINES[$i]}" | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
done

echo "UHG experiments complete. Results: $RESULT_DIR"
