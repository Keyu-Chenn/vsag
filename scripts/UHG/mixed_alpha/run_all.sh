#!/usr/bin/env bash
# Run mixed-alpha Recall@10, Recall@20, and Recall@200 for five datasets and
# five methods while keeping at most one newly-built hybrid index on disk.
#
# This is the only script in this directory that builds/deletes hybrid indexes.
# Usage:
#   bash /tbase-project/vsag/scripts/UHG/mixed_alpha/run_all.sh
#   DRY_RUN=1 bash /tbase-project/vsag/scripts/UHG/mixed_alpha/run_all.sh
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_ROOT/scripts/UHG/mixed_alpha}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/index}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
RESULT_ROOT="${RESULT_ROOT:-$EXPERIMENT_ROOT/results}"

BIN_HYBRID="${BIN_HYBRID:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"
DATASETS_TEXT="${DATASETS_TEXT:-nq hotpotqa msmarco dbpedia-entity fever}"
RECALLS_TEXT="${RECALLS_TEXT:-10 20 200}"
METHODS_TEXT="${METHODS_TEXT:-hnsw sindi hnsw_sindi fhg uhg}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.3}"
MAX_HOPS="${MAX_HOPS:-0}"

CANONICAL_DATA_DIR="/tbase-project/vsag/scripts/UHG/data"
SEARCH_SCRIPT="$EXPERIMENT_ROOT/run_search.sh"
VALIDATOR="$EXPERIMENT_ROOT/validate_inputs.py"
SUMMARIZER="$EXPERIMENT_ROOT/summarize_results.py"

read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a RECALLS <<< "$RECALLS_TEXT"
read -r -a METHODS <<< "$METHODS_TEXT"
declare -A VIRTUAL_INDEX_PRESENT=()

ACTIVE_CREATED_INDEX=""
cleanup_new_index_on_exit() {
    local status="$?"
    trap - EXIT INT TERM
    if [ "$status" -ne 0 ] && [ -n "$ACTIVE_CREATED_INDEX" ]; then
        echo "Cleaning newly-created hybrid index after failure: $ACTIVE_CREATED_INDEX" >&2
        set +e
        safe_delete_active_index
    fi
    exit "$status"
}

cleanup_new_index_on_signal() {
    local status="$1"
    trap - EXIT INT TERM
    if [ -n "$ACTIVE_CREATED_INDEX" ]; then
        echo "Cleaning newly-created hybrid index after signal: $ACTIVE_CREATED_INDEX" >&2
        set +e
        safe_delete_active_index
    fi
    exit "$status"
}

trap cleanup_new_index_on_exit EXIT
trap 'cleanup_new_index_on_signal 130' INT
trap 'cleanup_new_index_on_signal 143' TERM

is_allowed_dataset() {
    case "$1" in
        nq|hotpotqa|msmarco|dbpedia-entity|fever) return 0 ;;
        *) return 1 ;;
    esac
}

wants_method() {
    local requested="$1" value
    for value in "${METHODS[@]}"; do
        if [ "$value" = "$requested" ]; then
            return 0
        fi
    done
    return 1
}

require_file() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "Missing required file: $path" >&2
        exit 2
    fi
}

require_executable() {
    local path="$1"
    if [ ! -x "$path" ]; then
        echo "Missing executable: $path" >&2
        exit 2
    fi
}

print_command() {
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
}

run_logged() {
    local log_file="$1" status
    shift
    if [ "$DRY_RUN" = "1" ]; then
        print_command "$@"
        return 0
    fi
    mkdir -p "$(dirname "$log_file")"
    set +e
    "$@" 2>&1 | tee "$log_file"
    status="${PIPESTATUS[0]}"
    set -e
    if [ "$status" -ne 0 ]; then
        echo "Command failed with status $status; log: $log_file" >&2
        return "$status"
    fi
}

hybrid_index_path() {
    local dataset="$1" kind="$2"
    case "$kind" in
        fhg) printf '%s/index/703_%s_fhg_hybrid_index.index' "$DATA_DIR" "$dataset" ;;
        uhg) printf '%s/index_hybrid_union/703_%s_hybrid_index.index' "$DATA_DIR" "$dataset" ;;
        *)
            echo "Unsupported hybrid kind: $kind" >&2
            return 2
            ;;
    esac
}

initialize_virtual_index_state() {
    local dataset kind path
    for dataset in "${DATASETS[@]}"; do
        for kind in fhg uhg; do
            path="$(hybrid_index_path "$dataset" "$kind")"
            if [ -f "$path" ]; then
                VIRTUAL_INDEX_PRESENT["$path"]=1
            else
                VIRTUAL_INDEX_PRESENT["$path"]=0
            fi
        done
    done
}

hybrid_index_exists() {
    local path="$1"
    if [ "$DRY_RUN" = "1" ]; then
        [ "${VIRTUAL_INDEX_PRESENT["$path"]:-0}" = "1" ]
    else
        [ -f "$path" ]
    fi
}

mark_virtual_index_present() {
    local path="$1"
    if [ "$DRY_RUN" = "1" ]; then
        VIRTUAL_INDEX_PRESENT["$path"]=1
    fi
}

mark_virtual_index_absent() {
    local path="$1"
    if [ "$DRY_RUN" = "1" ]; then
        VIRTUAL_INDEX_PRESENT["$path"]=0
    fi
}

validate_deletion_target() {
    local path="$1" dataset="$2" kind="$3" expected
    if [ "$DATA_DIR" != "$CANONICAL_DATA_DIR" ]; then
        echo "Refusing hybrid deletion with non-canonical DATA_DIR: $DATA_DIR" >&2
        return 2
    fi
    case "$kind" in
        fhg) expected="$CANONICAL_DATA_DIR/index/703_${dataset}_fhg_hybrid_index.index" ;;
        uhg) expected="$CANONICAL_DATA_DIR/index_hybrid_union/703_${dataset}_hybrid_index.index" ;;
        *) return 2 ;;
    esac
    if [ "$path" != "$expected" ]; then
        echo "Refusing unexpected deletion target: $path" >&2
        return 2
    fi
    if [ -L "$path" ] || [ -d "$path" ]; then
        echo "Refusing to delete symlink/directory: $path" >&2
        return 2
    fi
    if [ -e "$path" ] && [ ! -f "$path" ]; then
        echo "Refusing to delete non-regular file: $path" >&2
        return 2
    fi
}

delete_hybrid_index() {
    local dataset="$1" kind="$2" path
    path="$(hybrid_index_path "$dataset" "$kind")"
    validate_deletion_target "$path" "$dataset" "$kind"
    if [ "$DRY_RUN" = "1" ]; then
        if hybrid_index_exists "$path"; then
            print_command rm -- "$path"
            mark_virtual_index_absent "$path"
        else
            echo "Hybrid index already absent: $path"
        fi
        return 0
    fi
    if [ -e "$path" ]; then
        rm -- "$path"
        echo "Deleted hybrid index: $path"
    else
        echo "Hybrid index already absent: $path"
    fi
}

safe_delete_active_index() {
    local dataset kind
    for dataset in "${DATASETS[@]}"; do
        for kind in fhg uhg; do
            if [ "$ACTIVE_CREATED_INDEX" = "$(hybrid_index_path "$dataset" "$kind")" ]; then
                delete_hybrid_index "$dataset" "$kind"
                ACTIVE_CREATED_INDEX=""
                return 0
            fi
        done
    done
    echo "Refusing unknown active index cleanup: $ACTIVE_CREATED_INDEX" >&2
    return 2
}

validate_selection() {
    local value
    if [ "$DRY_RUN" != "0" ] && [ "$DRY_RUN" != "1" ]; then
        echo "DRY_RUN must be 0 or 1" >&2
        exit 2
    fi
    for value in "${DATASETS[@]}"; do
        if ! is_allowed_dataset "$value"; then
            echo "Unsupported dataset: $value" >&2
            exit 2
        fi
    done
    for value in "${RECALLS[@]}"; do
        case "$value" in
            10|20|200) ;;
            *)
                echo "Unsupported Recall@K: $value" >&2
                exit 2
                ;;
        esac
    done
    for value in "${METHODS[@]}"; do
        case "$value" in
            hnsw|sindi|hnsw_sindi|fhg|uhg) ;;
            *)
                echo "Unsupported method: $value" >&2
                exit 2
                ;;
        esac
    done
}

preflight() {
    local dataset
    require_file "$SEARCH_SCRIPT"
    require_file "$VALIDATOR"
    require_file "$SUMMARIZER"
    require_executable "$REPO_ROOT/build-release/examples/cpp/701_uhg_exp1"
    require_executable "$BIN_HYBRID"
    require_executable "$REPO_ROOT/build-release/examples/cpp/708_uhg_exp8"
    require_executable "$REPO_ROOT/build-release/examples/cpp/709_uhg_exp9"

    python3 "$VALIDATOR" --data-dir "$DATA_DIR" --datasets "${DATASETS[@]}"
    for dataset in "${DATASETS[@]}"; do
        require_file "$INDEX_DIR/709_${dataset}_sindi.index"
    done

    echo "Hybrid index inventory before run:"
    find "$INDEX_DIR" "$INDEX_HYBRID_DIR" -maxdepth 1 -type f \
        \( -name '703_*_fhg_hybrid_index.index' -o -name '703_*_hybrid_index.index' \) \
        -printf '  %p %k KiB\n' | sort
    df -h "$DATA_DIR"
    initialize_virtual_index_state
}

build_fhg_index() {
    local dataset="$1" path h5 graph alpha_file gt_file log_file
    path="$(hybrid_index_path "$dataset" fhg)"
    if hybrid_index_exists "$path"; then
        echo "Reuse existing FHG hybrid index: $path"
        return 0
    fi
    h5="$DATA_DIR/hdf5/${dataset}.hdf5"
    graph="$DATA_DIR/fhg/${dataset}_fhg_alpha_0_5.h5"
    alpha_file="$DATA_DIR/mixed_alpha/$dataset/alphas.npy"
    gt_file="$DATA_DIR/mixed_alpha/$dataset/ground_truth.npy"
    log_file="$RESULT_ROOT/logs/index/${dataset}_fhg_build.log"

    ACTIVE_CREATED_INDEX="$path"
    run_logged "$log_file" \
        "$BIN_HYBRID" "$h5" \
        --mixed_alpha_file "$alpha_file" \
        --mixed_gt_file "$gt_file" \
        --method uhg -k 10 --alpha 0.5 --build_alpha 0.5 \
        --graph_path "$graph" \
        --hybrid_index_path "$path" \
        --index_dir "$INDEX_DIR" \
        --disable_sindi --disable_dense_entry --disable_hybrid_pruning \
        --max_hops 0 --sindi_bk 10 --ef_search 10 \
        --num_queries 1 --threads 1 --rebuild
    if [ "$DRY_RUN" != "1" ]; then
        require_file "$path"
    else
        mark_virtual_index_present "$path"
    fi
}

build_uhg_index() {
    local dataset="$1" path h5 graph sindi dense alpha_file gt_file log_file
    path="$(hybrid_index_path "$dataset" uhg)"
    if hybrid_index_exists "$path"; then
        echo "Reuse existing UHG hybrid index: $path"
        return 0
    fi
    h5="$DATA_DIR/hdf5/${dataset}.hdf5"
    graph="$DATA_DIR/uhg/${dataset}_uhg.h5"
    sindi="$INDEX_DIR/709_${dataset}_sindi.index"
    dense="$INDEX_DIR/701_${dataset}_dense_hnsw.index"
    alpha_file="$DATA_DIR/mixed_alpha/$dataset/alphas.npy"
    gt_file="$DATA_DIR/mixed_alpha/$dataset/ground_truth.npy"
    log_file="$RESULT_ROOT/logs/index/${dataset}_uhg_build.log"

    ACTIVE_CREATED_INDEX="$path"
    run_logged "$log_file" \
        "$BIN_HYBRID" "$h5" \
        --mixed_alpha_file "$alpha_file" \
        --mixed_gt_file "$gt_file" \
        --graph_path "$graph" \
        --sindi_index_path "$sindi" \
        --dense_entry_hnsw_graph_path "$dense" \
        --hybrid_index_path "$path" \
        --index_dir "$INDEX_HYBRID_DIR" \
        --alpha 0.5 --build_alpha 0.5 --method auto -k 10 \
        --ef_search 10 --sindi_bk 10 \
        --dense_entry_bk 10 --dense_entry_ef_search 10 \
        --hybrid_prune_scale "$HYBRID_PRUNE_SCALE" \
        --max_hops 0 \
        --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO" \
        --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO" \
        --num_queries 1 --threads 1 --rebuild
    if [ "$DRY_RUN" != "1" ]; then
        require_file "$path"
    else
        mark_virtual_index_present "$path"
    fi
}

run_search_job() {
    local dataset="$1" recall_at="$2" method="$3"
    local log_file="$RESULT_ROOT/logs/runner/recall_${recall_at}/${dataset}_${method}.log"
    run_logged "$log_file" \
        env REPO_ROOT="$REPO_ROOT" \
            EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
            DATA_DIR="$DATA_DIR" \
            RESULT_ROOT="$RESULT_ROOT" \
            NUM_QUERIES="$NUM_QUERIES" \
            THREADS="$THREADS" \
            RESUME="$RESUME" \
            SINDI_QUERY_PRUNE_RATIO="$SINDI_QUERY_PRUNE_RATIO" \
            SINDI_TERM_PRUNE_RATIO="$SINDI_TERM_PRUNE_RATIO" \
            HYBRID_PRUNE_SCALE="$HYBRID_PRUNE_SCALE" \
            MAX_HOPS="$MAX_HOPS" \
            bash "$SEARCH_SCRIPT" "$dataset" "$recall_at" "$method"
}

hybrid_results_complete() {
    local dataset="$1" method="$2" recall_at
    [ "$RESUME" = "1" ] || return 1
    for recall_at in "${RECALLS[@]}"; do
        if ! env REPO_ROOT="$REPO_ROOT" \
            EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
            DATA_DIR="$DATA_DIR" \
            RESULT_ROOT="$RESULT_ROOT" \
            NUM_QUERIES="$NUM_QUERIES" \
            THREADS="$THREADS" \
            RESUME="$RESUME" \
            CHECK_COMPLETE_ONLY=1 \
            SINDI_QUERY_PRUNE_RATIO="$SINDI_QUERY_PRUNE_RATIO" \
            SINDI_TERM_PRUNE_RATIO="$SINDI_TERM_PRUNE_RATIO" \
            HYBRID_PRUNE_SCALE="$HYBRID_PRUNE_SCALE" \
            MAX_HOPS="$MAX_HOPS" \
            bash "$SEARCH_SCRIPT" "$dataset" "$recall_at" "$method" \
            >/dev/null 2>&1; then
            return 1
        fi
    done
}

ensure_other_hybrid_index_absent() {
    local dataset="$1" active_kind="$2" other_kind other_path
    case "$active_kind" in
        fhg) other_kind=uhg ;;
        uhg) other_kind=fhg ;;
        *)
            echo "Unsupported active hybrid kind: $active_kind" >&2
            return 2
            ;;
    esac
    other_path="$(hybrid_index_path "$dataset" "$other_kind")"
    if hybrid_index_exists "$other_path"; then
        echo "Remove inactive $other_kind index before $active_kind: $other_path"
        delete_hybrid_index "$dataset" "$other_kind"
    fi
}

run_hybrid_method() {
    local dataset="$1" method="$2" recall_at path
    path="$(hybrid_index_path "$dataset" "$method")"
    ACTIVE_CREATED_INDEX=""
    if hybrid_results_complete "$dataset" "$method"; then
        echo "Skip completed $method results for $dataset; no index rebuild needed"
        delete_hybrid_index "$dataset" "$method"
        return 0
    fi
    ensure_other_hybrid_index_absent "$dataset" "$method"
    if [ "$method" = "fhg" ]; then
        build_fhg_index "$dataset"
    else
        build_uhg_index "$dataset"
    fi
    for recall_at in "${RECALLS[@]}"; do
        run_search_job "$dataset" "$recall_at" "$method"
    done
    delete_hybrid_index "$dataset" "$method"
    ACTIVE_CREATED_INDEX=""
    if [ "$DRY_RUN" != "1" ] && [ -e "$path" ]; then
        echo "Hybrid index deletion verification failed: $path" >&2
        exit 1
    fi
}

validate_selection
preflight

echo "Mixed-alpha suite"
echo "Datasets: ${DATASETS[*]}"
echo "Recall values: ${RECALLS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Results: $RESULT_ROOT"
echo "Hybrid policy: FHG build/run/delete, then UHG build/run/delete, per dataset"

for dataset in "${DATASETS[@]}"; do
    echo "================================================================"
    echo "Dataset: $dataset"
    echo "================================================================"

    for method in hnsw sindi hnsw_sindi; do
        if wants_method "$method"; then
            for recall_at in "${RECALLS[@]}"; do
                run_search_job "$dataset" "$recall_at" "$method"
            done
        fi
    done

    if wants_method fhg; then
        run_hybrid_method "$dataset" fhg
    fi
    if wants_method uhg; then
        run_hybrid_method "$dataset" uhg
    fi
done

if [ "$DRY_RUN" = "1" ]; then
    print_command \
        python3 "$SUMMARIZER" \
        --result-root "$RESULT_ROOT" \
        --datasets "${DATASETS[@]}" \
        --methods "${METHODS[@]}" \
        --recalls "${RECALLS[@]}" \
        --strict
else
    python3 "$SUMMARIZER" \
        --result-root "$RESULT_ROOT" \
        --datasets "${DATASETS[@]}" \
        --methods "${METHODS[@]}" \
        --recalls "${RECALLS[@]}" \
        --strict
fi

trap - EXIT INT TERM
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry-run plan completed; no experiment/index mutation was performed"
else
    echo "All mixed-alpha experiments completed: $RESULT_ROOT"
fi
