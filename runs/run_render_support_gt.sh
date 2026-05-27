#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/choheeseung/workspace/vlm-privacy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT_JSON="${PROJECT_ROOT}/data/Biv-priv-seg/support_set.json"
IMAGE_DIR="${PROJECT_ROOT}/data/Biv-priv-seg/support_images"
OUTPUT_DIR="${REPO_ROOT}/artifacts/support_gt_overlay"

python "${REPO_ROOT}/scripts/render_support_gt.py" \
  --support_json "${SUPPORT_JSON}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --mask_color red \
  --mask_alpha 88 \
  --bbox_color red \
  --bbox_width 5
