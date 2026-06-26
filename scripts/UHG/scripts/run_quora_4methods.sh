#!/usr/bin/env bash
# Quora 4-method experiment, Recall@100.
#
# Usage:
#   bash scripts/UHG/scripts/run_quora_4methods.sh        # run all alpha values
#   bash scripts/UHG/scripts/run_quora_4methods.sh 0.3    # run one alpha
set -euo pipefail

DATASET="/tbase-project/vsag/scripts/UHG/data/hdf5/quora.hdf5"
GT_DIR="/tbase-project/vsag/scripts/UHG/data/ground_truth/quora"
INDEX_DIR="/tbase-project/vsag/scripts/UHG/data/index"
REFINED_INDEX_DIR="/tbase-project/vsag/scripts/UHG/data/index_refined_quora"
RESULT_DIR="/tbase-project/vsag/scripts/UHG/results/quora_4methods"
K=100

if [ "$#" -gt 0 ]; then
    ALPHAS=("$1")
else
    ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
fi

run_and_capture() {
    "$@" 2>&1 | grep -E "^(Recall|QPS)" | paste - -
}

use_fhg_index() {
    cp "$INDEX_DIR/703_quora_fhg_hybrid_index.index" "$INDEX_DIR/703_quora_hybrid_index.index"
}

mkdir -p "$RESULT_DIR"

for ALPHA in "${ALPHAS[@]}"; do
    OUTFILE="$RESULT_DIR/alpha_${ALPHA}.txt"
    {
        echo "# Quora 4-method experiment, alpha=$ALPHA, k=$K, Recall@$K"
        echo "# hnsw_sindi/fhg index_dir=$INDEX_DIR"
        echo "# refined uhg/uhgs index_dir=$REFINED_INDEX_DIR"
        echo "# FHG uses 703_quora_fhg_hybrid_index.index"
    } > "$OUTFILE"

    echo "=== Quora Recall@$K alpha=$ALPHA ==="

    echo "--- HNSW+SINDI ---"
    for bk in 100 150 200 300 500; do
        result=$(run_and_capture \
            ./build-release/examples/cpp/701_uhg_exp1 "$DATASET" \
            --gt_dir "$GT_DIR" \
            --index_dir "$INDEX_DIR" \
            --alpha "$ALPHA" \
            --num_queries 1000 \
            -k "$K" \
            --bk "$bk" \
            --ef_search "$bk")
        echo "hnsw_sindi bk=$bk ef_search=$bk $result" | tee -a "$OUTFILE"
    done

    echo "--- FHG ---"
    use_fhg_index
    for ef in 100 200 300 500 800 1000; do
        result=$(run_and_capture \
            ./build-release/examples/cpp/703_uhg_exp3 "$DATASET" \
            --method uhg \
            -k "$K" \
            --alpha "$ALPHA" \
            --gt_dir "$GT_DIR" \
            --index_dir "$INDEX_DIR" \
            --hybrid_prune_scale 1 \
            --max_hops 0 \
            --sindi_bk 100 \
            --ef_search "$ef")
        echo "fhg ef_search=$ef prune_scale=1 $result" | tee -a "$OUTFILE"
    done

    echo "--- Refined UHG ---"
    for ef in 100 200 300 500 800 1000; do
        result=$(run_and_capture \
            ./build-release/examples/cpp/703_uhg_exp3 "$DATASET" \
            --method uhg \
            -k "$K" \
            --alpha "$ALPHA" \
            --gt_dir "$GT_DIR" \
            --index_dir "$REFINED_INDEX_DIR" \
            --hybrid_prune_scale 0.5 \
            --max_hops 0 \
            --sindi_bk 100 \
            --ef_search "$ef")
        echo "uhg_refined ef_search=$ef prune_scale=0.5 $result" | tee -a "$OUTFILE"
    done

    echo "--- Refined UHGS ---"
    for ef in 100 200 300 500 800 1000; do
        result=$(run_and_capture \
            ./build-release/examples/cpp/703_uhg_exp3 "$DATASET" \
            --method uhgs \
            -k "$K" \
            --alpha "$ALPHA" \
            --gt_dir "$GT_DIR" \
            --index_dir "$REFINED_INDEX_DIR" \
            --hybrid_prune_scale 0.5 \
            --max_hops 0 \
            --sindi_bk "$ef" \
            --ef_search "$ef")
        echo "uhgs_refined ef_search=$ef sindi_bk=$ef prune_scale=0.5 $result" | tee -a "$OUTFILE"
    done

    echo "=== Done: alpha=$ALPHA, saved to $OUTFILE ==="
done
