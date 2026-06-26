#!/usr/bin/env bash
# HotpotQA UHGH experiment with 707_uhg_exp7, Recall@100.
#
# Usage:
#   bash scripts/UHG/scripts/run_hotpotqa_uhgh.sh          # run all alpha values
#   bash scripts/UHG/scripts/run_hotpotqa_uhgh.sh 0.5      # run one alpha
#   NUM_QUERIES=100 bash scripts/UHG/scripts/run_hotpotqa_uhgh.sh 0.5
set -euo pipefail

DATASET="${DATASET:-/tbase-project/vsag/scripts/UHG/data/hdf5/hotpotqa.hdf5}"
GT_DIR="${GT_DIR:-/tbase-project/vsag/scripts/UHG/data/ground_truth/hotpotqa}"
DENSE_INDEX_DIR="${DENSE_INDEX_DIR:-/tbase-project/vsag/scripts/UHG/data/index}"
HYBRID_INDEX_DIR="${HYBRID_INDEX_DIR:-/tbase-project/vsag/scripts/UHG/data/index_refined_hotpotqa}"
HNSW_INDEX_PATH="${HNSW_INDEX_PATH:-$DENSE_INDEX_DIR/701_hotpotqa_dense_hnsw.index}"
HYBRID_INDEX_PATH="${HYBRID_INDEX_PATH:-$HYBRID_INDEX_DIR/703_hotpotqa_hybrid_index.index}"
RESULT_DIR="${RESULT_DIR:-/tbase-project/vsag/scripts/UHG/results/hotpotqa_uhgh}"
K="${K:-100}"
NUM_QUERIES="${NUM_QUERIES:-1000}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.5}"
MAX_HOPS="${MAX_HOPS:-0}"

if [ "$#" -gt 0 ]; then
    ALPHAS=("$@")
else
    ALPHAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
fi

if [ -n "${POINTS:-}" ]; then
    read -r -a SEARCH_POINTS <<< "$POINTS"
else
    SEARCH_POINTS=(100 200 300 500 800 1000)
fi

run_and_capture() {
    "$@" 2>&1 | grep -E "^(Recall|QPS)" | paste - -
}

mkdir -p "$RESULT_DIR"

for ALPHA in "${ALPHAS[@]}"; do
    OUTFILE="$RESULT_DIR/alpha_${ALPHA}.txt"
    {
        echo "# HotpotQA UHGH experiment, alpha=$ALPHA, k=$K, Recall@$K"
        echo "# dataset=$DATASET"
        echo "# gt_dir=$GT_DIR"
        echo "# dense_index=$HNSW_INDEX_PATH"
        echo "# hybrid_index=$HYBRID_INDEX_PATH"
        echo "# num_queries=$NUM_QUERIES"
        echo "# hybrid_prune_scale=$HYBRID_PRUNE_SCALE"
        echo "# max_hops=$MAX_HOPS"
        echo "# points=${SEARCH_POINTS[*]}"
    } > "$OUTFILE"

    echo "=== HotpotQA UHGH Recall@$K alpha=$ALPHA ==="

    for point in "${SEARCH_POINTS[@]}"; do
        result=$(run_and_capture \
            ./build-release/examples/cpp/707_uhg_exp7 "$DATASET" \
            -k "$K" \
            --alpha "$ALPHA" \
            --gt_dir "$GT_DIR" \
            --index_dir "$HYBRID_INDEX_DIR" \
            --dense_index_dir "$DENSE_INDEX_DIR" \
            --hnsw_index_path "$HNSW_INDEX_PATH" \
            --hybrid_index_path "$HYBRID_INDEX_PATH" \
            --hnsw_bk "$point" \
            --hnsw_ef_search "$point" \
            --ef_search "$point" \
            --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
            --max_hops "$MAX_HOPS" \
            --num_queries "$NUM_QUERIES")
        echo "uhgh hnsw_bk=$point hnsw_ef_search=$point ef_search=$point prune_scale=$HYBRID_PRUNE_SCALE max_hops=$MAX_HOPS $result" | tee -a "$OUTFILE"
    done

    echo "=== Done: alpha=$ALPHA, saved to $OUTFILE ==="
done
