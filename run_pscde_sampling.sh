#!/usr/bin/env bash
# run_pscde_sampling.sh
#
# Runs the diffusion model sampling on the PSCDE test set.
#
# Usage:
#   bash run_pscde_sampling.sh --ckpt <path/to/checkpoint.ckpt> [--steps N] [--batch_size N] [--max_samples N]
#   All arguments are forwarded to run_pscde_sampling.py.

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_pscde_sampling.py"
OUTPUT_DIR="${SCRIPT_DIR}/results/PSCDE"

# ── Defaults ──────────────────────────────────────────────────────────────────
STEPS=50
BATCH_SIZE=4
DEVICE=cuda
SEED=42

echo "============================================================"
echo " PSCDE Sampling"
echo " Output  : ${OUTPUT_DIR}"
echo " Steps   : ${STEPS}  |  Batch size: ${BATCH_SIZE}  |  Device: ${DEVICE}"
echo "============================================================"

cd "${SCRIPT_DIR}"

python "${PYTHON_SCRIPT}" \
    --steps       "${STEPS}" \
    --batch_size  "${BATCH_SIZE}" \
    --device      "${DEVICE}" \
    --seed        "${SEED}" \
    --output      "${OUTPUT_DIR}" \
    "$@"

echo ""
echo "Sampling complete. Results saved to: ${OUTPUT_DIR}/"
