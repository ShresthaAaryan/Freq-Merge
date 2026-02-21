#!/usr/bin/env bash
# Run a single training with sensible defaults and logging
set -euo pipefail

EPOCHS=${1:-300}
SEED=${2:-42}
OUTDIR=${3:-runs/seed_${SEED}}
VAL_DIR=${4:-}

mkdir -p "${OUTDIR}"
LOGFILE="${OUTDIR}/train_seed_${SEED}.log"

echo "Running train.py --epochs ${EPOCHS} --seed ${SEED}"
if [ -n "${VAL_DIR}" ]; then
	echo "Using custom val dir: ${VAL_DIR}"
	python train.py --epochs "${EPOCHS}" --seed "${SEED}" --val_data_dir "${VAL_DIR}" 2>&1 | tee "${LOGFILE}"
else
	python train.py --epochs "${EPOCHS}" --seed "${SEED}" 2>&1 | tee "${LOGFILE}"
fi
