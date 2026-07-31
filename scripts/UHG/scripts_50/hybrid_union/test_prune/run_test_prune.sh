#!/usr/bin/env bash
# Ablation: hybrid upper-bound pruning configurations.
#
# Uses the same per-query mixed alphas and exact ground truth, fixes --method,
# then varies --hybrid_prune_scale or disables the upper-bound pruning entirely
# to measure its effect on Recall-QPS. All configurations load the
# SAME pre-built UHG hybrid_index (data/index_hybrid_union/703_<dataset>_hybrid_index.index),
# so no index is rebuilt.
#
# hybrid_prune_scale semantics (see 703_uhg_exp3 --help):
#   0   = sparse contribution omitted from the upper-bound test (most aggressive)
#   <1  = scaled sparse Cauchy upper bound (more aggressive, possibly approximate)
#   1   = full sparse Cauchy upper bound (conservative, but still prunes)
#   no_prune = pass --disable_hybrid_pruning (true no-prune baseline for this layer)
#
# Defaults (overridable via env):
#   query alpha        = data/mixed_alpha/<dataset>/alphas.npy
#   method             = auto (alpha<=0.5 => uhgs, alpha>0.5 => uhgh)
#   pruning configs    = 0.3 no_prune
#
# Usage:
#   bash run_test_prune.sh                       # all datasets, all pruning configs
#   bash run_test_prune.sh nq                    # one dataset, all pruning configs
#   bash run_test_prune.sh nq 0.3                # one dataset, one prune scale
#   bash run_test_prune.sh nq no_prune           # one dataset, pruning disabled
#   bash nq/run_test_prune.sh                    # dataset-specific wrapper
#
# Results:
#   $RESULT_DIR/<dataset>/prune_<scale|no_prune>/alpha_mixed.txt
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
MIXED_ALPHA_DIR="${MIXED_ALPHA_DIR:-$DATA_DIR/mixed_alpha}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/scripts/UHG/results_50/test_prune}"
BIN="${BIN:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

K="${K:-50}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
METHOD="${METHOD:-auto}"
MAX_HOPS="${MAX_HOPS:-0}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco dbpedia-entity}"
PRUNE_SCALES_TEXT="${PRUNE_SCALES_TEXT:-0.3 no_prune}"
POINTS_TEXT="${POINTS_TEXT:-50 100 150 250 500}"

read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a PRUNE_SCALES <<< "$PRUNE_SCALES_TEXT"
read -r -a POINTS <<< "$POINTS_TEXT"

if [ "$#" -ge 1 ] && [ "$1" != "all" ]; then
    DATASETS=("$1")
fi
if [ "$#" -ge 2 ] && [ "$2" != "all" ]; then
    PRUNE_SCALES=("$2")
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

# Normalise a prune-scale token to a fixed <d>.<d> form for directory names.
#   1   -> 1.0
#   0.5 -> 0.5
#   .5  -> 0.5
normalise_scale() {
    local s="$1"
    if [[ "$s" == *.* ]]; then
        if [[ "$s" == .* ]]; then
            s="0${s}"
        fi
    else
        s="${s}.0"
    fi
    printf '%s' "$s"
}

# Canonicalise the true no-prune configuration as "no_prune". Keep the older
# aliases usable for ad-hoc command lines and existing automation.
normalise_prune_config() {
    local config="${1,,}"
    case "$config" in
        no_prune|no-prune|off|none|disabled)
            printf 'no_prune'
            ;;
        *)
            normalise_scale "$config"
            ;;
    esac
}

if [ ! -x "$BIN" ]; then
    echo "Missing binary: $BIN"
    exit 1
fi

for dataset in "${DATASETS[@]}"; do
    h5_file="$DATA_DIR/hdf5/${dataset}.hdf5"
    mixed_alpha_file="$MIXED_ALPHA_DIR/$dataset/alphas.npy"
    mixed_gt_file="$MIXED_ALPHA_DIR/$dataset/ground_truth.npy"
    uhg_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"

    if [ ! -f "$h5_file" ]; then
        echo "Skip $dataset: hdf5 not found: $h5_file"
        continue
    fi
    if [ ! -f "$mixed_alpha_file" ]; then
        echo "Skip $dataset: mixed-alpha query weights not found: $mixed_alpha_file"
        echo "  Generate them first via MAIN_EXPERIMENT_RUNBOOK.md step 12.2"
        continue
    fi
    if [ ! -f "$mixed_gt_file" ]; then
        echo "Skip $dataset: mixed-alpha ground truth not found: $mixed_gt_file"
        echo "  Generate it first via MAIN_EXPERIMENT_RUNBOOK.md step 12.2"
        continue
    fi
    if [ ! -f "$uhg_index" ]; then
        echo "Skip $dataset: UHG hybrid_index not found: $uhg_index"
        echo "  Build it first via MAIN_EXPERIMENT_RUNBOOK.md step 8.2"
        continue
    fi

    for raw_config in "${PRUNE_SCALES[@]}"; do
        prune_config="$(normalise_prune_config "$raw_config")"
        scale="$prune_config"
        enable_hybrid_pruning=true
        prune_args=()
        if [ "$prune_config" = "no_prune" ]; then
            # The scale is ignored when pruning is disabled, but pass the
            # default value so logs and direct invocations remain reproducible.
            scale="1.0"
            enable_hybrid_pruning=false
            prune_args=(--hybrid_prune_scale "$scale" --disable_hybrid_pruning)
        else
            prune_args=(--hybrid_prune_scale "$scale")
        fi

        scale_dir="$RESULT_DIR/$dataset/prune_${prune_config}"
        mkdir -p "$scale_dir"
        echo "============================================"
        echo "  Dataset: $dataset  Method: $METHOD  alpha=mixed  pruning=$enable_hybrid_pruning  prune_scale=$scale"
        echo "============================================"

        outfile="$scale_dir/alpha_mixed.txt"
        {
            echo "# ablation: hybrid upper-bound pruning configurations"
            echo "# method=$METHOD dataset=$dataset alpha=mixed k=$K num_queries=$NUM_QUERIES threads=$THREADS"
            echo "# mixed_alpha_file=$mixed_alpha_file"
            echo "# mixed_gt_file=$mixed_gt_file"
            echo "# fixed index: $uhg_index (SINDI + dense-entry + UHG main graph embedded at build time)"
            echo "# only enable_hybrid_pruning/hybrid_prune_scale vary; index is identical (load, no rebuild)"
            echo "# build_alpha=0.5 enable_hybrid_pruning=$enable_hybrid_pruning hybrid_prune_scale=$scale max_hops=$MAX_HOPS"
            echo "# sindi_query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO sindi_term_prune_ratio=$SINDI_TERM_PRUNE_RATIO"
            echo "# prune semantics: 0=dense-only bound, <1=aggressive scaled bound, 1=full Cauchy bound, no_prune=no upper-bound pruning"
            echo "# result_dir=$scale_dir"
        } > "$outfile"

        echo "--- $dataset $METHOD alpha=mixed pruning=$enable_hybrid_pruning prune_scale=$scale ---"
        for ef in "${POINTS[@]}"; do
            result="$(capture_metrics \
                "$BIN" "$h5_file" \
                --hybrid_index_path "$uhg_index" \
                --build_alpha 0.5 \
                --mixed_alpha_file "$mixed_alpha_file" \
                --mixed_gt_file "$mixed_gt_file" \
                --method "$METHOD" \
                -k "$K" \
                --ef_search "$ef" \
                --sindi_bk "$ef" \
                --dense_entry_bk "$ef" \
                --dense_entry_ef_search "$ef" \
                "${prune_args[@]}" \
                --max_hops "$MAX_HOPS" \
                --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
                --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
                --index_dir "$INDEX_HYBRID_DIR" \
                --num_queries "$NUM_QUERIES" \
                --threads "$THREADS")"
            echo "$METHOD ef_search=$ef pruning=$enable_hybrid_pruning prune_scale=$scale sindi_qp=$SINDI_QUERY_PRUNE_RATIO threads=$THREADS $result" | tee -a "$outfile"
        done
        echo "Done: $outfile"
    done
done

echo "Hybrid-pruning ablation complete. Results: $RESULT_DIR"
