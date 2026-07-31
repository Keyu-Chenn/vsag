#!/usr/bin/env bash
# Add larger search-parameter points for main experiments whose best Recall is
# not strictly greater than TARGET_RECALL. New points are appended to the
# existing result files, one at a time, until the target is reached or the
# method-specific parameter cap is reached.
#
# Usage:
#   bash run_supplement_until_recall.sh
#   bash run_supplement_until_recall.sh nq
#   bash run_supplement_until_recall.sh nq sindi
#   DRY_RUN=1 bash run_supplement_until_recall.sh nq hnsw
#
# Positional filters:
#   $1: dataset name or "all" (default: all existing datasets)
#   $2: hnsw, sindi, hnsw_sindi, fhg, uhg, or "all" (default: all)
#
# Main options (environment variables):
#   TARGET_RECALL=0.8          Stop only when Recall is greater than this value.
#   PARAM_CAP=20000            Override the cap for every method.
#   HNSW_PARAM_CAP=20000       HNSW bk/ef_search cap.
#   SINDI_PARAM_CAP=20000      SINDI bk cap.
#   HNSW_SINDI_PARAM_CAP=5000  HNSW+SINDI bk/ef_search cap.
#   FHG_PARAM_CAP=5000         FHG ef_search cap.
#   UHG_PARAM_CAP=5000         UHG ef_search/entry-search cap.
#   PARAM_STEPS_TEXT="..."     Ordered candidate parameters before the cap.
#   DRY_RUN=1                  Print planned commands without running or appending.
#
# Search options use the same defaults as the main experiment scripts and can
# also be overridden: K, NUM_QUERIES, THREADS, SINDI_QUERY_PRUNE_RATIO,
# SINDI_TERM_PRUNE_RATIO, HYBRID_PRUNE_SCALE, MAX_HOPS, and METHOD.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/tbase-project/vsag}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/scripts/UHG/data}"
INDEX_DIR="${INDEX_DIR:-$DATA_DIR/index}"
INDEX_HYBRID_DIR="${INDEX_HYBRID_DIR:-$DATA_DIR/index_hybrid_union}"
FHG_GRAPH_DIR="${FHG_GRAPH_DIR:-$DATA_DIR/fhg}"
UHG_GRAPH_DIR="${UHG_GRAPH_DIR:-$DATA_DIR/uhg}"
RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR}"
LOG_DIR="${LOG_DIR:-$RESULT_DIR/logs/supplement}"

BIN_HNSW="${BIN_HNSW:-$REPO_ROOT/build-release/examples/cpp/708_uhg_exp8}"
BIN_SINDI="${BIN_SINDI:-$REPO_ROOT/build-release/examples/cpp/709_uhg_exp9}"
BIN_HNSW_SINDI="${BIN_HNSW_SINDI:-$REPO_ROOT/build-release/examples/cpp/701_uhg_exp1}"
BIN_HYBRID="${BIN_HYBRID:-$REPO_ROOT/build-release/examples/cpp/703_uhg_exp3}"

K="${K:-100}"
NUM_QUERIES="${NUM_QUERIES:--1}"
THREADS="${THREADS:-1}"
TARGET_RECALL="${TARGET_RECALL:-0.8}"
DRY_RUN="${DRY_RUN:-0}"
SINDI_QUERY_PRUNE_RATIO="${SINDI_QUERY_PRUNE_RATIO:-0.5}"
SINDI_TERM_PRUNE_RATIO="${SINDI_TERM_PRUNE_RATIO:-0}"
HYBRID_PRUNE_SCALE="${HYBRID_PRUNE_SCALE:-0.3}"
MAX_HOPS="${MAX_HOPS:-0}"
METHOD="${METHOD:-auto}"
FHG_SINDI_BK="${FHG_SINDI_BK:-100}"
UHG_GRAPH_SUFFIX="${UHG_GRAPH_SUFFIX:-uhg}"

GLOBAL_PARAM_CAP="${PARAM_CAP:-}"
HNSW_PARAM_CAP="${HNSW_PARAM_CAP:-${GLOBAL_PARAM_CAP:-20000}}"
SINDI_PARAM_CAP="${SINDI_PARAM_CAP:-${GLOBAL_PARAM_CAP:-20000}}"
HNSW_SINDI_PARAM_CAP="${HNSW_SINDI_PARAM_CAP:-${GLOBAL_PARAM_CAP:-5000}}"
FHG_PARAM_CAP="${FHG_PARAM_CAP:-${GLOBAL_PARAM_CAP:-5000}}"
UHG_PARAM_CAP="${UHG_PARAM_CAP:-${GLOBAL_PARAM_CAP:-5000}}"
PARAM_STEPS_TEXT="${PARAM_STEPS_TEXT:-800 1000 1500 2000 3000 5000 8000 10000 15000 20000 30000 50000 80000 100000}"

FILTER_DATASET="${1:-all}"
FILTER_METHOD="${2:-all}"
METHODS=(hnsw sindi hnsw_sindi fhg uhg)
read -r -a PARAM_STEPS <<< "$PARAM_STEPS_TEXT"

mkdir -p "$LOG_DIR"
OVERALL_LOG="$LOG_DIR/supplement_until_recall.log"

TASKS_UNDER_TARGET=0
TASKS_REACHED=0
TASKS_CAPPED=0
TASKS_FAILED=0
POINTS_RUN=0

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$OVERALL_LOG"
}

format_cmd() {
    printf '%q ' "$@"
}

is_number() {
    awk -v value="$1" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/) }'
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

recall_above_target() {
    awk -v recall="$1" -v target="$TARGET_RECALL" \
        'BEGIN { exit !(recall + 0 > target + 0) }'
}

max_recall() {
    awk '
        /Recall:/ {
            for (i = 1; i <= NF; i++) {
                if ($i == "Recall:") {
                    value = $(i + 1) + 0
                    if (!found || value > best) {
                        best = value
                    }
                    found = 1
                }
            }
        }
        END {
            if (found) {
                printf "%.10g\n", best
            }
        }
    ' "$1"
}

max_parameter() {
    local key="$1"
    local outfile="$2"
    awk -v key="$key" '
        /Recall:/ {
            for (i = 1; i <= NF; i++) {
                split($i, pair, "=")
                if (pair[1] == key && pair[2] ~ /^[0-9]+$/) {
                    value = pair[2] + 0
                    if (!found || value > best) {
                        best = value
                    }
                    found = 1
                }
            }
        }
        END {
            if (found) {
                printf "%d\n", best
            }
        }
    ' "$outfile"
}

parameter_exists() {
    local key="$1"
    local parameter="$2"
    local outfile="$3"
    awk -v key="$key" -v target="$parameter" '
        /Recall:/ {
            for (i = 1; i <= NF; i++) {
                split($i, pair, "=")
                if (pair[1] == key && pair[2] + 0 == target + 0) {
                    found = 1
                }
            }
        }
        END { exit !found }
    ' "$outfile"
}

method_selected() {
    local method="$1"
    [ "$FILTER_METHOD" = "all" ] || [ "$FILTER_METHOD" = "$method" ]
}

parameter_key() {
    case "$1" in
        sindi) printf 'bk\n' ;;
        *) printf 'ef_search\n' ;;
    esac
}

parameter_cap() {
    case "$1" in
        hnsw) printf '%s\n' "$HNSW_PARAM_CAP" ;;
        sindi) printf '%s\n' "$SINDI_PARAM_CAP" ;;
        hnsw_sindi) printf '%s\n' "$HNSW_SINDI_PARAM_CAP" ;;
        fhg) printf '%s\n' "$FHG_PARAM_CAP" ;;
        uhg) printf '%s\n' "$UHG_PARAM_CAP" ;;
    esac
}

method_binary() {
    case "$1" in
        hnsw) printf '%s\n' "$BIN_HNSW" ;;
        sindi) printf '%s\n' "$BIN_SINDI" ;;
        hnsw_sindi) printf '%s\n' "$BIN_HNSW_SINDI" ;;
        fhg|uhg) printf '%s\n' "$BIN_HYBRID" ;;
    esac
}

check_file() {
    local label="$1"
    local path="$2"
    if [ ! -f "$path" ]; then
        log "[skip] missing $label: $path"
        return 1
    fi
    return 0
}

check_prerequisites() {
    local dataset="$1"
    local method="$2"
    local alpha="$3"
    local bin h5 gt_file
    bin="$(method_binary "$method")"
    h5="$DATA_DIR/hdf5/${dataset}.hdf5"
    gt_file="$DATA_DIR/ground_truth/$dataset/${dataset}_ground_truth_alpha_${alpha}.npy"

    if [ ! -x "$bin" ]; then
        log "[skip] binary is not executable: $bin"
        return 1
    fi
    check_file hdf5 "$h5" || return 1
    check_file ground_truth "$gt_file" || return 1

    case "$method" in
        hnsw)
            check_file dense_hnsw "$INDEX_DIR/701_${dataset}_dense_hnsw.index" || return 1
            ;;
        sindi)
            check_file sparse_sindi "$INDEX_DIR/701_${dataset}_sparse_sindi.index" || return 1
            ;;
        hnsw_sindi)
            check_file dense_hnsw "$INDEX_DIR/701_${dataset}_dense_hnsw.index" || return 1
            check_file sparse_sindi "$INDEX_DIR/701_${dataset}_sparse_sindi.index" || return 1
            ;;
        fhg)
            check_file fhg_graph "$FHG_GRAPH_DIR/${dataset}_fhg_alpha_0_5.h5" || return 1
            check_file fhg_hybrid_index "$INDEX_DIR/703_${dataset}_fhg_hybrid_index.index" || return 1
            ;;
        uhg)
            check_file uhg_graph "$UHG_GRAPH_DIR/${dataset}_${UHG_GRAPH_SUFFIX}.h5" || return 1
            check_file dense_hnsw "$INDEX_DIR/701_${dataset}_dense_hnsw.index" || return 1
            check_file uhg_hybrid_index "$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index" || return 1
            ;;
    esac
    return 0
}

CMD=()
RESULT_PREFIX=""

build_command() {
    local dataset="$1"
    local method="$2"
    local alpha="$3"
    local parameter="$4"
    local h5="$DATA_DIR/hdf5/${dataset}.hdf5"
    local gt_dir="$DATA_DIR/ground_truth/$dataset"

    case "$method" in
        hnsw)
            CMD=("$BIN_HNSW" "$h5"
                --gt_dir "$gt_dir"
                --index_dir "$INDEX_DIR"
                --alpha "$alpha"
                -k "$K"
                --bk "$parameter"
                --ef_search "$parameter"
                --num_queries "$NUM_QUERIES"
                --threads "$THREADS")
            RESULT_PREFIX="hnsw bk=$parameter ef_search=$parameter threads=$THREADS"
            ;;
        sindi)
            CMD=("$BIN_SINDI" "$h5"
                --gt_dir "$gt_dir"
                --index_dir "$INDEX_DIR"
                --alpha "$alpha"
                -k "$K"
                --bk "$parameter"
                --query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO"
                --term_prune_ratio "$SINDI_TERM_PRUNE_RATIO"
                --num_queries "$NUM_QUERIES"
                --threads "$THREADS")
            RESULT_PREFIX="sindi bk=$parameter query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS"
            ;;
        hnsw_sindi)
            CMD=("$BIN_HNSW_SINDI" "$h5"
                --gt_dir "$gt_dir"
                --index_dir "$INDEX_DIR"
                --alpha "$alpha"
                -k "$K"
                --bk "$parameter"
                --ef_search "$parameter"
                --query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO"
                --term_prune_ratio "$SINDI_TERM_PRUNE_RATIO"
                --num_queries "$NUM_QUERIES"
                --threads "$THREADS")
            RESULT_PREFIX="hnsw_sindi bk=$parameter ef_search=$parameter query_prune_ratio=$SINDI_QUERY_PRUNE_RATIO term_prune_ratio=$SINDI_TERM_PRUNE_RATIO threads=$THREADS"
            ;;
        fhg)
            local fhg_graph="$FHG_GRAPH_DIR/${dataset}_fhg_alpha_0_5.h5"
            local fhg_index="$INDEX_DIR/703_${dataset}_fhg_hybrid_index.index"
            CMD=("$BIN_HYBRID" "$h5"
                --method uhg
                -k "$K"
                --alpha "$alpha"
                --build_alpha 0.5
                --gt_dir "$gt_dir"
                --graph_path "$fhg_graph"
                --hybrid_index_path "$fhg_index"
                --index_dir "$INDEX_DIR"
                --disable_sindi
                --disable_dense_entry
                --disable_hybrid_pruning
                --max_hops "$MAX_HOPS"
                --sindi_bk "$FHG_SINDI_BK"
                --ef_search "$parameter"
                --num_queries "$NUM_QUERIES"
                --threads "$THREADS")
            RESULT_PREFIX="fhg ef_search=$parameter hybrid_pruning=false threads=$THREADS"
            ;;
        uhg)
            local uhg_graph="$UHG_GRAPH_DIR/${dataset}_${UHG_GRAPH_SUFFIX}.h5"
            local dense_entry_hnsw="$INDEX_DIR/701_${dataset}_dense_hnsw.index"
            local uhg_index="$INDEX_HYBRID_DIR/703_${dataset}_hybrid_index.index"
            CMD=("$BIN_HYBRID" "$h5"
                --graph_path "$uhg_graph"
                --dense_entry_hnsw_graph_path "$dense_entry_hnsw"
                --hybrid_index_path "$uhg_index"
                --alpha "$alpha"
                --build_alpha 0.5
                --method "$METHOD"
                -k "$K"
                --ef_search "$parameter"
                --sindi_bk "$parameter"
                --dense_entry_bk "$parameter"
                --dense_entry_ef_search "$parameter"
                --hybrid_prune_scale "$HYBRID_PRUNE_SCALE"
                --max_hops "$MAX_HOPS"
                --sindi_query_prune_ratio "$SINDI_QUERY_PRUNE_RATIO"
                --sindi_term_prune_ratio "$SINDI_TERM_PRUNE_RATIO"
                --index_dir "$INDEX_HYBRID_DIR"
                --gt_dir "$gt_dir"
                --num_queries "$NUM_QUERIES"
                --threads "$THREADS")
            RESULT_PREFIX="uhg method=$METHOD ef_search=$parameter prune_scale=$HYBRID_PRUNE_SCALE sindi_qp=$SINDI_QUERY_PRUNE_RATIO threads=$THREADS"
            ;;
    esac
}

run_and_capture() {
    local run_log="$1"
    shift
    local output status summary

    if [ "$DRY_RUN" = "1" ]; then
        {
            printf '[dry-run] '
            format_cmd "$@"
            printf '\n'
        } | tee -a "$run_log"
        printf 'DRY_RUN\n'
        return 0
    fi

    output="$("$@" 2>&1)"
    status=$?
    {
        printf '[%s] ' "$(date '+%F %T')"
        format_cmd "$@"
        printf '\n%s\n\n' "$output"
    } >> "$run_log"

    if [ "$status" -ne 0 ]; then
        printf 'command failed with status=%s; see %s\n' "$status" "$run_log" >&2
        return 1
    fi

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
    if [[ ! "$summary" =~ Recall:[[:space:]]+[-+0-9.eE]+[[:space:]]+QPS:[[:space:]]+[-+0-9.eE]+ ]]; then
        printf 'no Recall/QPS pair captured; see %s\n' "$run_log" >&2
        return 1
    fi
    printf '%s\n' "$summary"
}

run_task() {
    local dataset="$1"
    local method="$2"
    local alpha="$3"
    local outfile="$4"
    local best current_parameter key cap run_log result new_recall

    best="$(max_recall "$outfile")"
    if [ -z "$best" ]; then
        log "[skip] no valid Recall in $outfile"
        return
    fi
    if recall_above_target "$best"; then
        return
    fi

    key="$(parameter_key "$method")"
    current_parameter="$(max_parameter "$key" "$outfile")"
    if [ -z "$current_parameter" ]; then
        log "[skip] no $key parameter found in $outfile"
        return
    fi
    cap="$(parameter_cap "$method")"
    TASKS_UNDER_TARGET=$((TASKS_UNDER_TARGET + 1))

    log "[task] dataset=$dataset method=$method alpha=$alpha best_recall=$best current_$key=$current_parameter target_gt=$TARGET_RECALL cap=$cap"
    if [ "$current_parameter" -ge "$cap" ]; then
        log "[capped] dataset=$dataset method=$method alpha=$alpha already at cap=$cap with best_recall=$best"
        TASKS_CAPPED=$((TASKS_CAPPED + 1))
        return
    fi
    if ! check_prerequisites "$dataset" "$method" "$alpha"; then
        TASKS_FAILED=$((TASKS_FAILED + 1))
        return
    fi

    run_log="$LOG_DIR/${dataset}_${method}_alpha_${alpha}.log"
    local candidates=("${PARAM_STEPS[@]}" "$cap")
    local parameter
    for parameter in "${candidates[@]}"; do
        if ! is_positive_integer "$parameter"; then
            log "[skip] invalid parameter in PARAM_STEPS_TEXT: $parameter"
            continue
        fi
        if [ "$parameter" -le "$current_parameter" ] || [ "$parameter" -gt "$cap" ]; then
            continue
        fi
        if parameter_exists "$key" "$parameter" "$outfile"; then
            current_parameter="$parameter"
            continue
        fi

        build_command "$dataset" "$method" "$alpha" "$parameter"
        log "[run] dataset=$dataset method=$method alpha=$alpha $key=$parameter"
        if ! result="$(run_and_capture "$run_log" "${CMD[@]}")"; then
            log "[failed] dataset=$dataset method=$method alpha=$alpha $key=$parameter; see $run_log"
            TASKS_FAILED=$((TASKS_FAILED + 1))
            return
        fi
        if [ "$DRY_RUN" = "1" ]; then
            current_parameter="$parameter"
            continue
        fi

        printf '%s %s\n' "$RESULT_PREFIX" "$result" | tee -a "$outfile"
        POINTS_RUN=$((POINTS_RUN + 1))
        current_parameter="$parameter"
        new_recall="$(printf '%s\n' "$result" | awk '{for (i=1; i<=NF; i++) if ($i=="Recall:") print $(i+1)}')"
        if recall_above_target "$new_recall"; then
            log "[reached] dataset=$dataset method=$method alpha=$alpha recall=$new_recall $key=$parameter"
            TASKS_REACHED=$((TASKS_REACHED + 1))
            return
        fi
    done

    if [ "$DRY_RUN" = "1" ]; then
        log "[dry-run] dataset=$dataset method=$method alpha=$alpha planned through cap=$cap"
        return
    fi

    best="$(max_recall "$outfile")"
    if recall_above_target "$best"; then
        TASKS_REACHED=$((TASKS_REACHED + 1))
    else
        log "[capped] dataset=$dataset method=$method alpha=$alpha best_recall=$best cap=$cap"
        TASKS_CAPPED=$((TASKS_CAPPED + 1))
    fi
}

validate_options() {
    if ! is_number "$TARGET_RECALL"; then
        printf 'TARGET_RECALL must be a non-negative number: %s\n' "$TARGET_RECALL" >&2
        exit 2
    fi
    local cap
    for cap in "$HNSW_PARAM_CAP" "$SINDI_PARAM_CAP" "$HNSW_SINDI_PARAM_CAP" "$FHG_PARAM_CAP" "$UHG_PARAM_CAP"; do
        if ! is_positive_integer "$cap"; then
            printf 'All parameter caps must be positive integers: %s\n' "$cap" >&2
            exit 2
        fi
    done
    if [ "$FILTER_METHOD" != "all" ]; then
        local valid=0 method
        for method in "${METHODS[@]}"; do
            if [ "$FILTER_METHOD" = "$method" ]; then
                valid=1
            fi
        done
        if [ "$valid" -ne 1 ]; then
            printf 'Unknown method: %s (valid: hnsw sindi hnsw_sindi fhg uhg all)\n' "$FILTER_METHOD" >&2
            exit 2
        fi
    fi
}

main() {
    validate_options
    log "[start] target_recall_gt=$TARGET_RECALL dry_run=$DRY_RUN dataset=$FILTER_DATASET method=$FILTER_METHOD caps=hnsw:$HNSW_PARAM_CAP,sindi:$SINDI_PARAM_CAP,hnsw_sindi:$HNSW_SINDI_PARAM_CAP,fhg:$FHG_PARAM_CAP,uhg:$UHG_PARAM_CAP"

    local dataset_dir dataset method method_dir outfile alpha found_dataset=0
    for dataset_dir in "$RESULT_DIR"/*; do
        [ -d "$dataset_dir" ] || continue
        dataset="$(basename "$dataset_dir")"
        case "$dataset" in
            logs|trash|plots|__pycache__) continue ;;
        esac
        if [ "$FILTER_DATASET" != "all" ] && [ "$FILTER_DATASET" != "$dataset" ]; then
            continue
        fi
        found_dataset=1

        for method in "${METHODS[@]}"; do
            method_selected "$method" || continue
            method_dir="$dataset_dir/$method"
            [ -d "$method_dir" ] || continue
            for outfile in "$method_dir"/alpha_*.txt; do
                [ -f "$outfile" ] || continue
                alpha="$(basename "$outfile")"
                alpha="${alpha#alpha_}"
                alpha="${alpha%.txt}"
                run_task "$dataset" "$method" "$alpha" "$outfile"
            done
        done
    done

    if [ "$found_dataset" -eq 0 ]; then
        log "[finish] no matching dataset directory found"
        exit 1
    fi
    log "[finish] under_target=$TASKS_UNDER_TARGET reached=$TASKS_REACHED capped=$TASKS_CAPPED failed=$TASKS_FAILED points_run=$POINTS_RUN"
}

main "$@"
