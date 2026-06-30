#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_ROOT="/data/zhaozy/qiaoxiaozhen/data/TTA-PT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main_bpre.py \
  --config configs \
  --data-root "${DATA_ROOT}" \
  --wandb-log \
  --datasets caltech101 \
  --backbone RN50
