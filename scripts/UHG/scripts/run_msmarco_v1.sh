#!/usr/bin/env bash
# MSMARCO FHG experiment, Recall@100.
#
# Usage:
#   bash scripts/UHG/scripts/run_msmarco_v1.sh        # run all alpha values
#   bash scripts/UHG/scripts/run_msmarco_v1.sh 0.5    # run one alpha
#   NUM_QUERIES=10 bash scripts/UHG/scripts/run_msmarco_v1.sh 0.5
set -euo pipefail

DATASET="/tbase-project/vsag/scripts/UHG/data/hdf5/msmarco.hdf5"
GT_DIR="/tbase-project/vsag/scripts/UHG/data/ground_truth/msmarco"
INDEX_DIR="/tbase-project/vsag/scripts/UHG/data/index"
RESULT_DIR="/tbase-project/vsag/scripts/UHG/results/msmarco_v1"
K=100
NUM_QUERIES="${NUM_QUERIES:-1000}"

if [ "$#" -gt 0 ]; then
    ALPHAS=("$@")
else
    ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
fi

if [ -n "${POINTS:-}" ]; then
    read -r -a SEARCH_POINTS <<< "$POINTS"
else
    SEARCH_POINTS=(1000 1500 2000 2500)
fi

run_and_capture() {
    "$@" 2>&1 | grep -E "^(Recall|QPS)" | paste - -
}

use_fhg_index() {
    cp "$INDEX_DIR/703_msmarco_fhg_hybrid_index.index" \
       "$INDEX_DIR/703_msmarco_hybrid_index.index"
}

mkdir -p "$RESULT_DIR"

for ALPHA in "${ALPHAS[@]}"; do
    OUTFILE="$RESULT_DIR/alpha_${ALPHA}.txt"
    {
        echo "# MSMARCO FHG experiment, alpha=$ALPHA, k=$K, Recall@$K"
        echo "# index_dir=$INDEX_DIR"
        echo "# FHG uses 703_msmarco_fhg_hybrid_index.index"
        echo "# num_queries=$NUM_QUERIES"
        echo "# points=${SEARCH_POINTS[*]}"
    } > "$OUTFILE"

    echo "=== MSMARCO FHG Recall@$K alpha=$ALPHA ==="

    use_fhg_index
    for ef in "${SEARCH_POINTS[@]}"; do
        result=$(run_and_capture \
            ./build-release/examples/cpp/703_uhg_exp3 "$DATASET" \
            --method uhg \
            -k "$K" \
            --alpha "$ALPHA" \
            --gt_dir "$GT_DIR" \
            --index_dir "$INDEX_DIR" \
            --hybrid_prune_scale 1 \
            --max_hops 0 \
            --num_queries "$NUM_QUERIES" \
            --sindi_bk 100 \
            --ef_search "$ef")
        echo "fhg ef_search=$ef prune_scale=1 $result" | tee -a "$OUTFILE"
    done

    echo "=== Done: alpha=$ALPHA, saved to $OUTFILE ==="
done
