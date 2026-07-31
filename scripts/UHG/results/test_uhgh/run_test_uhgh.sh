#!/usr/bin/env bash
# Test UHGH with the dense-entry HNSW ef_search and entry_bk tied to the main
# graph ef_search via a fixed scale factor.
#
# The main experiment ties dense_entry_bk = dense_entry_ef_search = ef_search
# (scale = 1.0, see 703_uhg_exp3.cpp:629-635 search_points mode). This script
# instead sets both to ENTRY_SCALE * ef_search (default 0.5), so a smaller /
# cheaper HNSW entry search feeds fewer entry points into the main hybrid graph
# search. The goal is to find the sweet spot where entry search cost is reduced
# without hurting final Recall.
#
# Because the binary's --search_points mode forcibly overwrites
# dense_entry_bk/dense_entry_ef_search to ef_search (see 703_uhg_exp3.cpp:632-635),
# we use single-point --ef_search mode (one process per ef point) and explicitly
# pass --dense_entry_ef_search and --dense_entry_bk.
#
# Fixed:
#   method              = uhgh  (force dense-entry path, no auto routing)
#   alpha               = 0.7   (dense-dominant, the regime where UHGH is used)
#   build_alpha         = 0.5
#   hybrid_prune_scale  = 0.3
#   sindi_query_prune_ratio = 0.5  (irrelevant for uhgh but kept for param parity)
#   sindi_term_prune_ratio  = 0
#   max_hops            = 0   (no limit)
#
# Varied:
#   ef_search                              (main hybrid graph ef)  default: 100 200 300 500 1000
#   ENTRY_SCALE  (= dense_entry_ef/ef = dense_entry_bk/ef)         default: 0.5
#                Multiple scales can be passed to compare; each scale produces
#                its own result subdirectory and curve.
#
# No index is rebuilt; loads data/index_hybrid_union/703_<dataset>_hybrid_index.index.
#
# Usage:
#   bash run_test_uhgh.sh                       # nq, scale=0.5, all ef points
#   bash run_test_uhgh.sh nq                    # one dataset
#   bash run_test_uhgh.sh nq 500                # one dataset, one main ef_search
#   ENTRY_SCALE_TEXT="0.5 1.0" bash run_test_uhgh.sh nq   # compare two scales
#
# Results:
#   <RESULT_DIR>/<dataset>/scale_<S>/alpha_<alpha>.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
GT_ROOT="${GT_ROOT:-$DATA_DIR/ground_truth}"
GT_VALIDATOR="${GT_VALIDATOR:-}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results/test_uhgh}"
BIN="${BIN:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

K="${K:-100}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
METHOD="${METHOD:-uhgh}"
ALPHA="${ALPHA:-0.7}"
BUILD_ALPHA="${BUILD_ALPHA:-0.5}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.3}"
MAX_HOPS="${MAX_HOPS:-0}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
DATASETS_TEXT="${DATASETS_TEXT:-nq}"
ENTRY_SCALE_TEXT="${ENTRY_SCALE_TEXT:-0.5}"
POINTS_TEXT="${POINTS_TEXT:-100 200 300 500 1000}"

read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a ENTRY_SCALES <<< "$ENTRY_SCALE_TEXT"
read -r -a POINTS <<< "$POINTS_TEXT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
    DATASETS=("$1")
fi
# Optional 2nd positional arg: a single main ef_search value
if [ "$#" -ge 2 ] && [ "$2" != "all" ]; then
    POINTS=("$2")
fi

# Normalise a scale token to a fixed <d>.<d> form for directory names.
normalise_scale() {
    local s="$1"
    if [[ "$s" == *.* ]]; then
        if [[ "$s" == .* ]]; then
            s="0${s}"
        fi
    else
        s="${s}.0"
    fi
    # strip trailing .0 for integer-valued scales? keep canonical <d>.<d>
    printf '%s' "$s"
}

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
if [ "${#ENTRY_SCALES[@]}" -eq 0 ] || [ "${#POINTS[@]}" -eq 0 ]; then
    echo "No entry scales / ef_search points configured"
    exit 1
fi

total_processes=$(( ${#ENTRY_SCALES[@]} * ${#POINTS[@]} ))
echo "============================================"
echo "  Method: $METHOD  Alpha: $ALPHA"
echo "  Datasets: ${DATASETS[*]}"
echo "  Entry scales (entry_ef = entry_bk = scale * ef): ${ENTRY_SCALES[*]}"
echo "  Main ef_search: ${POINTS[*]}"
echo "  Total processes per dataset: $total_processes"
echo "============================================"

for dataset in "${DATASETS[@]}"; do
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    gt_dir="$GT_ROOT/$dataset"
    uhg_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"

    if [ ! -f "$h5_file" ]; then
        echo "Skip $dataset: hdf5 not found: $h5_file"
        continue
    fi
    if [ ! -d "$gt_dir" ]; then
        echo "Skip $dataset: ground-truth directory not found: $gt_dir"
        continue
    fi
    if [ -n "$GT_VALIDATOR" ]; then
        if ! python3 "$GT_VALIDATOR" "$dataset" \
            --topk "$K" \
            --num-queries "$NUM_QUERIES" \
            --fixed-root "$GT_ROOT" \
            --hdf5-root "$DATA_DIR/hdf5" \
            --alphas "$ALPHA" \
            --fixed-only; then
            echo "Skip $dataset: fixed-alpha GT is not valid for Recall@$K"
            continue
        fi
    fi
    if [ ! -f "$uhg_index" ]; then
        echo "Skip $dataset: UHG hybrid_index not found: $uhg_index"
        echo "  Build it first via MAIN_EXPERIMENT_RUNBOOK.md step 8.2"
        continue
    fi

    gt_file="$gt_dir/${dataset}_ground_truth_alpha_${ALPHA}.npy"
    if [ ! -f "$gt_file" ]; then
        echo "Skip $dataset: ground truth for alpha=$ALPHA not found: $gt_file"
        continue
    fi

    for raw_scale in "${ENTRY_SCALES[@]}"; do
        scale_label="$(normalise_scale "$raw_scale")"
        scale_dir="$RESULT_DIR/$dataset/scale_${scale_label}"
        mkdir -p "$scale_dir"
        outfile="$scale_dir/alpha_${ALPHA}.txt"

        {
            echo "# UHGH dense-entry HNSW scaled sweep"
            echo "# method=$METHOD dataset=$dataset alpha=$ALPHA k=$K num_queries=$NUM_QUERIES threads=$THREADS"
            echo "# fixed index=$uhg_index (load only; no rebuild)"
            echo "# build_alpha=$BUILD_ALPHA hybrid_prune_scale=$HYBRID_PRUNE_SCALE max_hops=$MAX_HOPS"
            echo "# dense_entry_ef_search = dense_entry_bk = ${raw_scale} * ef_search (scale=${raw_scale})"
            echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
            echo "# ef_search points: ${POINTS[*]}"
            echo "# result_dir=$scale_dir"
        } > "$outfile"

        echo "--- $dataset $METHOD alpha=$ALPHA entry_scale=$raw_scale ---"
        for ef in "${POINTS[@]}"; do
            # entry_ef = entry_bk = round(scale * ef); use integer arithmetic via awk.
            # entry_ef may legitimately exceed the main ef_search: the HNSW entry
            # search and the main hybrid graph search are independent widths.
            entry_val="$(awk -v s="$raw_scale" -v e="$ef" 'BEGIN { v = s * e; r = int(v + 0.5); if (r < 1) r = 1; print r }')"

            result="$(capture_metrics \
                "$BIN" "$h5_file" \
                --hybrid_index_path "$uhg_index" \
                --alpha "$ALPHA" \
                --build_alpha "$BUILD_ALPHA" \
                --method "$METHOD" \
                -k "$K" \
                --ef_search "$ef" \
                --dense_entry_bk "$entry_val" \
                --dense_entry_ef_search "$entry_val" \
                --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
                --max_hops "$MAX_HOPS" \
                --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
                --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
                --index_dir "$INDEX_HYBRID_DIR" \
                --gt_dir "$gt_dir" \
                --num_queries "$NUM_QUERIES" \
                --threads "$THREADS")"

            if [[ "$result" != Recall:* ]]; then
                printf 'ERROR: ef_search=%s entry_ef=bk=%s\n%s\n' \
                    "$ef" "$entry_val" "$result" | tee -a "$outfile"
                continue
            fi
            echo "$METHOD ef_search=$ef dense_entry_ef=$entry_val dense_entry_bk=$entry_val entry_scale=$raw_scale prune_scale=$HYBRID_PRUNE_SCALE threads=$THREADS $result" \
                | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
done

echo "UHGH scaled dense-entry sweep complete. Results: $RESULT_DIR"
