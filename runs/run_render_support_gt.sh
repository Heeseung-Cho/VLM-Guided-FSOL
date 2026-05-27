#!/usr/bin/env bash
set -euo pipefail

if [ -z "${PROJECT_ROOT:-}" ]; then
  echo "PROJECT_ROOT must point to a directory containing data/Biv-priv-seg/{support_set.json,support_images/}." >&2
  echo "  export PROJECT_ROOT=/path/to/dataset_root" >&2
  exit 1
fi

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
