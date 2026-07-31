#!/usr/bin/env bash
# recall@20 fixed-alpha FHG search experiments.
#
# Usage:
#   bash run_fhg.sh
#   bash run_fhg.sh nq
#   bash nq/run_fhg.sh
#
# Results:
#   $RESULT_DIR/<dataset>/fhg/alpha_<alpha>.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/index}"
FHG_GRAPH_DIR="${FHG_GRAPH_DIR:-$DATA_DIR/fhg}"
FHG_INDEX_DIR="${FHG_INDEX_DIR:-$INDEX_DIR}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results_20/main}"
BIN="${BIN:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

K="${K:-20}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
MAX_HOPS="${MAX_HOPS:-0}"
SINDI_BK="${SINDI_BK:-100}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco fever dbpedia-entity}"
ALPHAS_TEXT="${ALPHAS_TEXT:-0.3 0.4 0.5 0.6 0.7}"
POINTS_TEXT="${POINTS_TEXT:-20 40 60 100 160 200}"

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
    fhg_graph="$FHG_GRAPH_DIR/${dataset}_fhg_alpha_0_5.h5"
    fhg_index="$FHG_INDEX_DIR/703_${dataset}_fhg_hybrid_index.index"
    method_dir="$RESULT_DIR/$dataset/fhg"

    if [ ! -f "$h5_file" ]; then
        echo "Skip $dataset: hdf5 not found: $h5_file"
        continue
    fi
    if [ ! -d "$gt_dir" ]; then
        echo "Skip $dataset: GT directory not found: $gt_dir"
        continue
    fi
    if [ ! -f "$fhg_graph" ]; then
        echo "Skip $dataset: FHG graph not found: $fhg_graph"
        continue
    fi

    mkdir -p "$method_dir" "$FHG_INDEX_DIR"
    echo "============================================"
    echo "  Dataset: $dataset  Method: fhg"
    echo "============================================"

    for alpha in "${ALPHAS[@]}"; do
        gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
        outfile="$method_dir/alpha_${alpha}.txt"
        if [ ! -f "$gt_file" ]; then
            echo "Skip $dataset fhg alpha=$alpha: GT not found: $gt_file"
            continue
        fi

        {
            echo "# recall@20 fixed-alpha FHG"
            echo "# method=uhg dataset=$dataset alpha=$alpha k=$K num_queries=$NUM_QUERIES threads=$THREADS"
            echo "# graph_build: bk=100 query_prune_ratio=0.9 refine=none fixed_alpha=0.5"
            echo "# build_alpha=0.5"
            echo "# graph_path=$fhg_graph"
            echo "# hybrid_index_path=$fhg_index"
            echo "# hybrid_pruning=false max_hops=$MAX_HOPS sindi_bk=$SINDI_BK"
            echo "# result_dir=$method_dir"
        } > "$outfile"

        echo "--- $dataset fhg alpha=$alpha ---"
        points_csv="$(join_points "${POINTS[@]}")"
        result="$(capture_metrics \
            "$BIN" "$h5_file" \
            --method uhg \
            -k "$K" \
            --alpha "$alpha" \
            --build_alpha 0.5 \
            --gt_dir "$gt_dir" \
            --graph_path "$fhg_graph" \
            --hybrid_index_path "$fhg_index" \
            --index_dir "$FHG_INDEX_DIR" \
            --disable_sindi \
            --disable_dense_entry \
            --disable_hybrid_pruning \
            --max_hops "$MAX_HOPS" \
            --sindi_bk "$SINDI_BK" \
            --search_points "$points_csv" \
            --num_queries "$NUM_QUERIES" \
            --threads "$THREADS" \
            ${REBUILD_ARGS[@]+"${REBUILD_ARGS[@]}"})"
        if ! load_metric_lines "$result" "${#POINTS[@]}"; then
            printf 'ERROR: expected %d FHG metric lines\n%s\n' \
                "${#POINTS[@]}" "$result" | tee -a "$outfile"
            continue
        fi
        for i in "${!POINTS[@]}"; do
            ef="${POINTS[$i]}"
            echo "fhg ef_search=$ef hybrid_pruning=false threads=$THREADS ${METRIC_LINES[$i]}" | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
done

echo "FHG experiments complete. Results: $RESULT_DIR"
