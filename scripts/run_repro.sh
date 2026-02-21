#!/usr/bin/env bash
# Run a sequence of seeds (sequential) and save logs under runs/repro
set -euo pipefail

EPOCHS=${1:-300}
OUTDIR=${2:-runs/repro}
SEEDS=${3:-"42 52 62"}

mkdir -p "${OUTDIR}"

for seed in ${SEEDS}; do
  echo "=== Starting seed ${seed} ===" | tee -a "${OUTDIR}/run.log"
  bash scripts/run_single.sh "${EPOCHS}" "${seed}" "${OUTDIR}/seed_${seed}"
done

echo "All runs complete. Logs in ${OUTDIR}"