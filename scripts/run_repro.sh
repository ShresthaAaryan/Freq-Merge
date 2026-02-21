#!/usr/bin/env bash
# Run a sequence of seeds (sequential) and save logs under runs/repro
set -euo pipefail

EPOCHS=${1:-300}
OUTDIR=${2:-runs/repro}
SEEDS=${3:-"42 52 62"}
VAL_DIR=${4:-}

mkdir -p "${OUTDIR}"

for seed in ${SEEDS}; do
  echo "=== Starting seed ${seed} ===" | tee -a "${OUTDIR}/run.log"
  if [ -n "${VAL_DIR}" ]; then
    bash scripts/run_single.sh "${EPOCHS}" "${seed}" "${OUTDIR}/seed_${seed}" "${VAL_DIR}"
  else
    bash scripts/run_single.sh "${EPOCHS}" "${seed}" "${OUTDIR}/seed_${seed}"
  fi
done

echo "All runs complete. Logs in ${OUTDIR}"