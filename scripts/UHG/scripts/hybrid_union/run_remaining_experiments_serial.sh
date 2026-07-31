#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  "/tbase-project/vsag/scripts/UHG/results/main/logs" \
  "/tbase-project/vsag/scripts/UHG/results/mixed_alpha" \
  "/tbase-project/vsag/scripts/UHG/results/ablation/logs" \
  "/tbase-project/vsag/scripts/UHG/results/test_prune/logs"

echo "[$(date '+%F %T')] START hotpotqa"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh" hotpotqa \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_baselines_hotpotqa.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_fhg.sh" hotpotqa \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_fhg_hotpotqa.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh" hotpotqa \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_uhg_hotpotqa.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha/run_mixed_alpha.sh" hotpotqa \
  > "/tbase-project/vsag/scripts/UHG/results/mixed_alpha/run_hotpotqa.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh" hotpotqa \
  > "/tbase-project/vsag/scripts/UHG/results/ablation/logs/run_ablation_hotpotqa.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" hotpotqa \
  > "/tbase-project/vsag/scripts/UHG/results/test_prune/logs/run_test_prune_hotpotqa.log" 2>&1 &
wait "$!"

echo "[$(date '+%F %T')] DONE hotpotqa"
echo "[$(date '+%F %T')] START msmarco"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh" msmarco \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_baselines_msmarco.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_fhg.sh" msmarco \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_fhg_msmarco.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh" msmarco \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_uhg_msmarco.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha/run_mixed_alpha.sh" msmarco \
  > "/tbase-project/vsag/scripts/UHG/results/mixed_alpha/run_msmarco.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh" msmarco \
  > "/tbase-project/vsag/scripts/UHG/results/ablation/logs/run_ablation_msmarco.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" msmarco \
  > "/tbase-project/vsag/scripts/UHG/results/test_prune/logs/run_test_prune_msmarco.log" 2>&1 &
wait "$!"

echo "[$(date '+%F %T')] DONE msmarco"
echo "[$(date '+%F %T')] START fever"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_baselines.sh" fever \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_baselines_fever.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_fhg.sh" fever \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_fhg_fever.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/main/run_uhg.sh" fever \
  > "/tbase-project/vsag/scripts/UHG/results/main/logs/run_uhg_fever.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/mixed_alpha/run_mixed_alpha.sh" fever \
  > "/tbase-project/vsag/scripts/UHG/results/mixed_alpha/run_fever.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/ablation/run_ablation.sh" fever \
  > "/tbase-project/vsag/scripts/UHG/results/ablation/logs/run_ablation_fever.log" 2>&1 &
wait "$!"

nohup bash "/tbase-project/vsag/scripts/UHG/scripts/hybrid_union/test_prune/run_test_prune.sh" fever \
  > "/tbase-project/vsag/scripts/UHG/results/test_prune/logs/run_test_prune_fever.log" 2>&1 &
wait "$!"

echo "[$(date '+%F %T')] DONE fever"
echo "[$(date '+%F %T')] ALL EXPERIMENTS DONE"
