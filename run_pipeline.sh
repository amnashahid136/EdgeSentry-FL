#!/usr/bin/env bash
# Full EdgeSentry pipeline: data -> federated training -> compression -> benchmark
# -> explainability -> multi-seed studies. Everything lands in results/ and
# saved_models/, and the website reads it from there.
#
#   ./run_pipeline.sh              full run  (~1 h on a laptop CPU)
#   QUICK=1 ./run_pipeline.sh      1 seed, trimmed DP sweep (~20 min)
set -e
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=3
if [ -n "$QUICK" ]; then SEEDS="0"; DP_SIGMAS="0 0.1 0.3"; else SEEDS="0 1 2"; DP_SIGMAS="0 0.1 0.3 1.0"; fi

echo "== 1/10 dataset (leak-free temporal split) =="; python3 data/prepare_dataset.py
echo "== 2/10 official FedAvg training ==";            python3 federated/simulate_federated.py --rounds 20 --local_epochs 2 --seed 0
echo "== 3/10 INT8 quantization ==";                    python3 model/compress_model.py
echo "== 4/10 structured pruning (federated fine-tune) =="; python3 model/prune_model.py --prune_ratio 0.5 --finetune_rounds 4
echo "== 5/10 benchmark ==";                            python3 benchmarking/benchmark.py
echo "== 6/10 explainability + faithfulness report =="; python3 model/explainability.py
echo "== 7/10 FedAvg vs local-only vs centralized (seeds: $SEEDS) =="; python3 experiments/fl_comparison.py --seeds $SEEDS
echo "== 8/10 fair compression study ==";               python3 experiments/compression_study.py --seeds $SEEDS
echo "== 9/10 DP-FedAvg sweep ==";                      python3 federated/compare_dp.py --seeds 0 --sigmas $DP_SIGMAS
echo "== 10/10 leave-one-severity-out ==";              python3 experiments/generalization.py --seeds 0
echo; echo "Done. Start the website:  ./run_local_demo.sh   or   docker compose up --build"
