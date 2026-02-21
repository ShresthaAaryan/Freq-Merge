#!/usr/bin/env bash
# Run a single training with sensible defaults and logging
set -euo pipefail

# Ensure repo root is on PYTHONPATH and use repo as working dir
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

EPOCHS=${1:-300}
SEED=${2:-42}
OUTDIR=${3:-runs/seed_${SEED}}

mkdir -p "${OUTDIR}"
LOGFILE="${OUTDIR}/train_seed_${SEED}.log"

echo "Running train.py --epochs ${EPOCHS} --seed ${SEED} (cwd=${REPO_ROOT})"
python train.py --epochs "${EPOCHS}" --seed "${SEED}" 2>&1 | tee "${LOGFILE}"
