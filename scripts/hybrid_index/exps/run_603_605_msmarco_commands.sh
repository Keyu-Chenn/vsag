#!/usr/bin/env bash

mkdir -p /tbase-project/vsag/scripts/hybrid_index/results/603_605_msmarco_logs
rm -f /tbase-project/vsag/scripts/hybrid_index/results/603_605_msmarco_logs/*.log

/tbase-project/vsag/build-release/examples/cpp/605_hybrid_exp5 /tbase-project/vsag/scripts/hybrid_index/data/models/data/msmarco_gt/msmarco_alpha_0_3_k_200.hdf5 -k 100 --sindi_bk 100 --alpha 0.3 --ef_search 100 --hybrid_prune_scale 0.6 --sindi_query_prune_ratio 0 --sindi_term_prune_ratio 0 2>&1 | tee -a /tbase-project/vsag/scripts/hybrid_index/results/603_605_msmarco_logs/605_alpha_0_3.log
/tbase-project/vsag/build-release/examples/cpp/605_hybrid_exp5 /tbase-project/vsag/scripts/hybrid_index/data/models/data/msmarco_gt/msmarco_alpha_0_3_k_200.hdf5 -k 100 --sindi_bk 200 --alpha 0.3 --ef_search 200 --hybrid_prune_scale 0.6 --sindi_query_prune_ratio 0 --sindi_term_prune_ratio 0 2>&1 | tee -a /tbase-project/vsag/scripts/hybrid_index/results/603_605_msmarco_logs/605_alpha_0_3.log
/tbase-project/vsag/build-release/examples/cpp/605_hybrid_exp5 /tbase-project/vsag/scripts/hybrid_index/data/models/data/msmarco_gt/msmarco_alpha_0_3_k_200.hdf5 -k 100 --sindi_bk 300 --alpha 0.3 --ef_search 300 --hybrid_prune_scale 0.6 --sindi_query_prune_ratio 0 --sindi_term_prune_ratio 0 2>&1 | tee -a /tbase-project/vsag/scripts/hybrid_index/results/603_605_msmarco_logs/605_alpha_0_3.log
/tbase-project/vsag/build-release/examples/cpp/605_hybrid_exp5 /tbase-project/vsag/scripts/hybrid_index/data/models/data/msmarco_gt/msmarco_alpha_0_3_k_200.hdf5 -k 100 --sindi_bk 400 --alpha 0.3 --ef_search 400 --hybrid_prune_scale 0.6 --sindi_query_prune_ratio 0 --sindi_term_prune_ratio 0 2>&1 | tee -a /tbase-project/vsag/scripts/hybrid_index/results/603_605_msmarco_logs/605_alpha_0_3.log
