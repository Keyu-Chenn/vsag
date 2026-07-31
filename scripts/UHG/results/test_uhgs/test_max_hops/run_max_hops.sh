#!/usr/bin/env bash
# Test the effect of --max_hops on UHG search.
#
# Loads the pre-built UHG hybrid_index (data/index_hybrid_union/703_<dataset>_hybrid_index.index)
# and varies --max_hops while keeping every other search parameter fixed. No index
# is rebuilt. max_hops caps the number of neighbour-expansion iterations in the
# main hybrid graph search; 0 means no limit (the main-experiment default).
#
# The special token "ef" means max_hops is set equal to ef_search for each
# search point, i.e. the hop budget scales with the candidate queue width.
# Because max_hops varies per ef point, the "ef" configuration runs each
# ef_search point as a separate process; fixed max_hops values use the
# --search_points batch mode (one process per alpha, loads index once).
#
# Fixed search settings (aligned with MAIN_EXPERIMENT_RUNBOOK.md step 9 / run_uhg.sh):
#   method              = auto  (alpha<=0.5 => UHGS, alpha>0.5 => UHGH)
#   build_alpha         = 0.5
#   hybrid_prune_scale  = 0.3
#   sindi_query_prune_ratio = 0.5
#   sindi_term_prune_ratio  = 0
#   ef_search points    = 100 200 300 500 1000  (sindi_bk/dense_entry_* tied to ef_search)
#
# Usage:
#   bash run_max_hops.sh                       # all datasets, all max_hops, all alphas
#   bash run_max_hops.sh nq                    # one dataset
#   bash run_max_hops.sh nq 0                  # one dataset, one max_hops value
#   bash run_max_hops.sh nq ef                 # one dataset, max_hops=ef_search
#   bash run_max_hops.sh nq 0 0.3              # one dataset, one max_hops, one alpha
#
#   DATASETS_TEXT="nq" MAX_HOPS_TEXT="0 50 100 ef" ALPHAS_TEXT="0.3 0.7" \
#     bash run_max_hops.sh
#
# Results:
#   <RESULT_DIR>/<dataset>/hops_<N|ef>/alpha_<alpha>.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results/test_uhgs/test_max_hops}"
BIN="${BIN:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

K="${K:-100}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
METHOD="${METHOD:-auto}"
BUILD_ALPHA="${BUILD_ALPHA:-0.5}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.3}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
DATASETS_TEXT="${DATASETS_TEXT:-nq}"
ALPHAS_TEXT="${ALPHAS_TEXT:-0.3 0.7}"
# max_hops configurations:
#   0   = no limit (main-experiment baseline)
#   50  = aggressive hop cap, large Recall loss
#   100 = moderate cap
#   200 = near-baseline on NQ at ef_search<=300
#   500 = effectively unlimited for ef_search<=500
#   ef  = max_hops equals ef_search (hop budget scales with candidate width)
MAX_HOPS_TEXT="${MAX_HOPS_TEXT:-0 50 100 200 500 ef}"
POINTS_TEXT="${POINTS_TEXT:-100 200 300 500 1000}"

read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a ALPHAS <<< "$ALPHAS_TEXT"
read -r -a MAX_HOPS_VALUES <<< "$MAX_HOPS_TEXT"
read -r -a POINTS <<< "$POINTS_TEXT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
    DATASETS=("$1")
fi
if [ "$#" -ge 2 ] && [ "$2" != "all" ]; then
    MAX_HOPS_VALUES=("$2")
fi
if [ "$#" -ge 3 ] && [ "$3" != "all" ]; then
    ALPHAS=("$3")
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
if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "No datasets configured"
    exit 1
fi
if [ "${#MAX_HOPS_VALUES[@]}" -eq 0 ]; then
    echo "No max_hops values configured"
    exit 1
fi
if [ "${#POINTS[@]}" -eq 0 ]; then
    echo "No search points configured"
    exit 1
fi

points_csv="$(join_points "${POINTS[@]}")"

for dataset in "${DATASETS[@]}"; do
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    gt_dir="$DATA_DIR/ground_truth/$dataset"
    uhg_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"

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

    echo "============================================"
    echo "  Dataset: $dataset  Method: $METHOD"
    echo "  Alphas: ${ALPHAS[*]}  max_hops: ${MAX_HOPS_VALUES[*]}"
    echo "  Search points: ${POINTS[*]}"
    echo "============================================"

    for max_hops_token in "${MAX_HOPS_VALUES[@]}"; do
        # "ef" means max_hops = ef_search (varies per point); anything else is
        # a fixed integer hop cap.
        is_ef_mode=0
        if [[ "$max_hops_token" == "ef" || "$max_hops_token" == "EF" ]]; then
            is_ef_mode=1
            hops_label="ef"
        else
            hops_label="$max_hops_token"
        fi
        hops_dir="$RESULT_DIR/$dataset/hops_${hops_label}"
        mkdir -p "$hops_dir"

        for alpha in "${ALPHAS[@]}"; do
            gt_file="$gt_dir/${dataset}_ground_truth_alpha_${alpha}.npy"
            outfile="$hops_dir/alpha_${alpha}.txt"

            if [ ! -f "$gt_file" ]; then
                echo "Skip $dataset max_hops=$hops_label alpha=$alpha: ground truth not found: $gt_file"
                continue
            fi

            {
                echo "# UHG max_hops ablation"
                echo "# method=$METHOD dataset=$dataset alpha=$alpha max_hops=$hops_label k=$K num_queries=$NUM_QUERIES threads=$THREADS"
                echo "# auto rule: alpha<=0.5 -> uhgs, alpha>0.5 -> uhgh"
                echo "# fixed index=$uhg_index (load only; no rebuild)"
                echo "# build_alpha=$BUILD_ALPHA hybrid_prune_scale=$HYBRID_PRUNE_SCALE"
                if [ "$is_ef_mode" -eq 1 ]; then
                    echo "# max_hops=ef_search (hop budget scales with ef_search per point)"
                else
                    echo "# max_hops=$hops_label (fixed hop cap; 0=no limit)"
                fi
                echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
                echo "# search_points=$points_csv (ef_search=sindi_bk=dense_entry_bk=dense_entry_ef_search)"
                echo "# result_dir=$hops_dir"
            } > "$outfile"

            echo "--- $dataset $METHOD alpha=$alpha max_hops=$hops_label ---"

            if [ "$is_ef_mode" -eq 1 ]; then
                # max_hops=ef_search: run each ef point as a separate process
                # because max_hops varies per point (cannot use --search_points).
                for ef in "${POINTS[@]}"; do
                    result="$(capture_metrics \
                        "$BIN" "$h5_file" \
                        --hybrid_index_path "$uhg_index" \
                        --alpha "$alpha" \
                        --build_alpha "$BUILD_ALPHA" \
                        --method "$METHOD" \
                        -k "$K" \
                        --ef_search "$ef" \
                        --sindi_bk "$ef" \
                        --dense_entry_bk "$ef" \
                        --dense_entry_ef_search "$ef" \
                        --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
                        --max_hops "$ef" \
                        --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
                        --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
                        --index_dir "$INDEX_HYBRID_DIR" \
                        --gt_dir "$gt_dir" \
                        --num_queries "$NUM_QUERIES" \
                        --threads "$THREADS")"

                    if ! load_metric_lines "$result" 1; then
                        printf 'ERROR: expected 1 metric line for max_hops=ef(%s) alpha=%s\n%s\n' \
                            "$ef" "$alpha" "$result" | tee -a "$outfile"
                        continue
                    fi
                    echo "$METHOD ef_search=$ef max_hops=ef(=$ef) prune_scale=$HYBRID_PRUNE_SCALE sindi_qp=$SINDI_QUERY_PRUNE_RATIO threads=$THREADS ${METRIC_LINES[0]}" \
                        | tee -a "$outfile"
                done
            else
                # Fixed max_hops: use --search_points to batch all ef points in
                # one process (index loaded once).
                result="$(capture_metrics \
                    "$BIN" "$h5_file" \
                    --hybrid_index_path "$uhg_index" \
                    --alpha "$alpha" \
                    --build_alpha "$BUILD_ALPHA" \
                    --method "$METHOD" \
                    -k "$K" \
                    --search_points "$points_csv" \
                    --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
                    --max_hops "$hops_label" \
                    --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
                    --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
                    --index_dir "$INDEX_HYBRID_DIR" \
                    --gt_dir "$gt_dir" \
                    --num_queries "$NUM_QUERIES" \
                    --threads "$THREADS")"

                if ! load_metric_lines "$result" "${#POINTS[@]}"; then
                    printf 'ERROR: expected %d metric lines for max_hops=%s alpha=%s\n%s\n' \
                        "${#POINTS[@]}" "$hops_label" "$alpha" "$result" | tee -a "$outfile"
                    continue
                fi
                for i in "${!POINTS[@]}"; do
                    ef="${POINTS[$i]}"
                    echo "$METHOD ef_search=$ef max_hops=$hops_label prune_scale=$HYBRID_PRUNE_SCALE sindi_qp=$SINDI_QUERY_PRUNE_RATIO threads=$THREADS ${METRIC_LINES[$i]}" \
                        | tee -a "$outfile"
                done
            fi
            echo "Done: $outfile"
        done
    done
done

echo "max_hops ablation complete. Results: $RESULT_DIR"
