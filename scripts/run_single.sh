#!/usr/bin/env bash
# Run a single training with sensible defaults and logging
set -euo pipefail

EPOCHS=${1:-300}
SEED=${2:-42}
OUTDIR=${3:-runs/seed_${SEED}}

mkdir -p "${OUTDIR}"
LOGFILE="${OUTDIR}/train_seed_${SEED}.log"

echo "Running train.py --epochs ${EPOCHS} --seed ${SEED}"
python train.py --epochs "${EPOCHS}" --seed "${SEED}" 2>&1 | tee "${LOGFILE}"
