#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/tbase-project/vsag}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-release}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/scripts/hybrid_index/data/models/data/msmarco_gt}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/scripts/hybrid_index/results/compare_603_605_msmarco_$(date +%Y%m%d_%H%M%S)}"

TOPK="${TOPK:-100}"
NUM_QUERIES="${NUM_QUERIES:--1}"

BASELINE_BK="${BASELINE_BK:-400}"
BASELINE_EF="${BASELINE_EF:-400}"

SINDI_FAST_BK="${SINDI_FAST_BK:-200}"
SINDI_FAST_EF="${SINDI_FAST_EF:-200}"
SINDI_RECALL_BK="${SINDI_RECALL_BK:-400}"
SINDI_RECALL_EF="${SINDI_RECALL_EF:-400}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-1}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"

BIN_603="${BUILD_DIR}/examples/cpp/603_hybrid_exp3"
BIN_605="${BUILD_DIR}/examples/cpp/605_hybrid_exp5"

if [[ -n "${ALPHAS_OVERRIDE:-}" ]]; then
    read -r -a ALPHAS <<< "${ALPHAS_OVERRIDE}"
else
    ALPHAS=(0_2 0_3 0_4 0_5 0_6 0_7 0_8)
fi

mkdir -p "${OUT_DIR}"

if [[ ! -x "${BIN_603}" ]]; then
    echo "Missing executable: ${BIN_603}" >&2
    exit 1
fi

if [[ ! -x "${BIN_605}" ]]; then
    echo "Missing executable: ${BIN_605}" >&2
    exit 1
fi

SUMMARY_CSV="${OUT_DIR}/summary.csv"
echo "method,alpha,file,topk,num_queries,bk,ef_search,hybrid_prune_scale,sindi_query_prune_ratio,sindi_term_prune_ratio,avg_recall,qps,total_time,avg_dist_compute,avg_hops,log_file" > "${SUMMARY_CSV}"

alpha_float() {
    local alpha_tag="$1"
    echo "${alpha_tag/_/.}"
}

extract_metric() {
    local log_file="$1"
    local key="$2"
    awk -F ':' -v key="${key}" '$1 ~ key {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}' "${log_file}"
}

extract_table_metric() {
    local log_file="$1"
    local method_prefix="$2"
    local column="$3"

    awk -v method_prefix="${method_prefix}" -v column="${column}" '
        $1 == method_prefix {
            if (column == "qps") {
                print $NF;
            } else if (column == "avg_hops") {
                print $(NF - 1);
            } else if (column == "avg_dist_compute") {
                print $(NF - 2);
            } else if (column == "avg_recall") {
                print $(NF - 3);
            }
            exit;
        }
    ' "${log_file}"
}

append_summary() {
    local method="$1"
    local alpha="$2"
    local data_file="$3"
    local bk="$4"
    local ef="$5"
    local log_file="$6"
    local table_method="${7:-}"
    local avg_recall qps total_time avg_dist_compute avg_hops

    if [[ -n "${table_method}" ]]; then
        avg_recall="$(extract_table_metric "${log_file}" "${table_method}" "avg_recall")"
        avg_dist_compute="$(extract_table_metric "${log_file}" "${table_method}" "avg_dist_compute")"
        avg_hops="$(extract_table_metric "${log_file}" "${table_method}" "avg_hops")"
        qps="$(extract_table_metric "${log_file}" "${table_method}" "qps")"
        total_time="$(extract_metric "${log_file}" "total_time")"
    else
        avg_recall="$(extract_metric "${log_file}" "avg_recall")"
        qps="$(extract_metric "${log_file}" "qps")"
        total_time="$(extract_metric "${log_file}" "total_time")"
        avg_dist_compute="$(extract_metric "${log_file}" "avg_dist_compute")"
        avg_hops="$(extract_metric "${log_file}" "avg_hops")"
    fi

    echo "${method},${alpha},${data_file},${TOPK},${NUM_QUERIES},${bk},${ef},${HYBRID_PRUNE_SCALE},${SINDI_QUERY_PRUNE_RATIO},${SINDI_TERM_PRUNE_RATIO},${avg_recall},${qps},${total_time},${avg_dist_compute},${avg_hops},${log_file}" >> "${SUMMARY_CSV}"
}

run_and_log() {
    local log_file="$1"
    shift
    echo "Running: $*"
    "$@" 2>&1 | tee "${log_file}"
}

echo "Output directory: ${OUT_DIR}"
echo "Summary CSV:      ${SUMMARY_CSV}"
echo "TOPK=${TOPK}, NUM_QUERIES=${NUM_QUERIES}"

for alpha_tag in "${ALPHAS[@]}"; do
    alpha="$(alpha_float "${alpha_tag}")"
    data_file="${DATA_DIR}/msmarco_alpha_${alpha_tag}_k_200.hdf5"

    if [[ ! -f "${data_file}" ]]; then
        echo "Missing data file: ${data_file}" >&2
        exit 1
    fi

    echo "================================================================"
    echo "alpha=${alpha} file=${data_file}"
    echo "================================================================"

    baseline_log="${OUT_DIR}/603_alpha_${alpha_tag}_bk${BASELINE_BK}_ef${BASELINE_EF}.log"
    run_and_log "${baseline_log}" \
        "${BIN_603}" "${data_file}" \
        -k "${TOPK}" \
        --bk "${BASELINE_BK}" \
        --alpha "${alpha}" \
        --ef_search "${BASELINE_EF}" \
        --num_queries "${NUM_QUERIES}"
    append_summary "603_baseline" "${alpha}" "${data_file}" "${BASELINE_BK}" "${BASELINE_EF}" "${baseline_log}"

    fast_log="${OUT_DIR}/605_alpha_${alpha_tag}_fast_sindi${SINDI_FAST_BK}_ef${SINDI_FAST_EF}.log"
    run_and_log "${fast_log}" \
        "${BIN_605}" "${data_file}" \
        -k "${TOPK}" \
        --sindi_bk "${SINDI_FAST_BK}" \
        --alpha "${alpha}" \
        --ef_search "${SINDI_FAST_EF}" \
        --hybrid_prune_scale "${HYBRID_PRUNE_SCALE}" \
        --sindi_query_prune_ratio "${SINDI_QUERY_PRUNE_RATIO}" \
        --sindi_term_prune_ratio "${SINDI_TERM_PRUNE_RATIO}" \
        --num_queries "${NUM_QUERIES}"
    append_summary "605_sindi_fast" "${alpha}" "${data_file}" "${SINDI_FAST_BK}" "${SINDI_FAST_EF}" "${fast_log}" "sindi"

    recall_log="${OUT_DIR}/605_alpha_${alpha_tag}_recall_sindi${SINDI_RECALL_BK}_ef${SINDI_RECALL_EF}.log"
    run_and_log "${recall_log}" \
        "${BIN_605}" "${data_file}" \
        -k "${TOPK}" \
        --sindi_bk "${SINDI_RECALL_BK}" \
        --alpha "${alpha}" \
        --ef_search "${SINDI_RECALL_EF}" \
        --hybrid_prune_scale "${HYBRID_PRUNE_SCALE}" \
        --sindi_query_prune_ratio "${SINDI_QUERY_PRUNE_RATIO}" \
        --sindi_term_prune_ratio "${SINDI_TERM_PRUNE_RATIO}" \
        --num_queries "${NUM_QUERIES}"
    append_summary "605_sindi_recall" "${alpha}" "${data_file}" "${SINDI_RECALL_BK}" "${SINDI_RECALL_EF}" "${recall_log}" "sindi"
done

echo "Done. Summary: ${SUMMARY_CSV}"
