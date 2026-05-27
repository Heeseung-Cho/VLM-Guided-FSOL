#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <llm_model> <gdino_checkpoint> <sam_checkpoint>"
  exit 1
fi

LLM_MODEL="$1"
GDINO_CHECKPOINT="$2"
SAM_CHECKPOINT="$3"

PROJECT_ROOT="${PROJECT_ROOT:-/home/choheeseung/workspace/vlm-privacy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPPORT_DIR="${PROJECT_ROOT}/data/Biv-priv-seg/support_images"
SUPPORT_JSON="${PROJECT_ROOT}/data/Biv-priv-seg/support_set.json"
OUTPUT_ROOT="${REPO_ROOT}/artifacts/support_inference"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}/stage1" "${OUTPUT_ROOT}/stage1_fam" "${OUTPUT_ROOT}/stage2" "${OUTPUT_ROOT}/stage3" "${OUTPUT_ROOT}/stage4_sam"

python "${REPO_ROOT}/semantic/run_stage1_semantic.py" \
  --query_dir "${SUPPORT_DIR}" \
  --json_path "${SUPPORT_JSON}" \
  --output_path "${OUTPUT_ROOT}/stage1/stage1_semantic.json" \
  --llm_model "${LLM_MODEL}" \
  --device cuda \
  --llm_max_new_tokens 180 \
  --llm_decoding_mode deterministic \
  --llm_max_pixels 448 \
  --family_config "${REPO_ROOT}/config/family_category_route4_v1.json" \
  --query_prompt_path "${REPO_ROOT}/prompts/active/semantic_query_route4_v1.txt" \
  --null_policy skip \
  --save_raw_text

python "${REPO_ROOT}/scripts/expand_stage1_to_family.py" \
  --input_path "${OUTPUT_ROOT}/stage1/stage1_semantic.json" \
  --output_path "${OUTPUT_ROOT}/stage1_fam/stage1_semantic.json" \
  --family_config "${REPO_ROOT}/config/family_category_route4_v1.json"

python "${REPO_ROOT}/semantic/run_stage2_detection.py" \
  --stage1_path "${OUTPUT_ROOT}/stage1_fam/stage1_semantic.json" \
  --output_path "${OUTPUT_ROOT}/stage2/stage2_detection_gdino_ft.json" \
  --config_path "${REPO_ROOT}/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py" \
  --checkpoint_path "${GDINO_CHECKPOINT}" \
  --device cuda \
  --box_threshold 0.20 \
  --text_threshold 0.10 \
  --proposal_nms_iou 0.50 \
  --max_candidates 5

python "${REPO_ROOT}/semantic/run_stage3_minimal.py" \
  --json_path "${SUPPORT_JSON}" \
  --stage1_path "${OUTPUT_ROOT}/stage1_fam/stage1_semantic.json" \
  --stage2_path "${OUTPUT_ROOT}/stage2/stage2_detection_gdino_ft.json" \
  --output_dir "${OUTPUT_ROOT}/stage3" \
  --prompt_path "${REPO_ROOT}/prompts/active/stage3_l0_enriched.txt" \
  --per_image_prompt_path "${REPO_ROOT}/prompts/active/stage3_per_image_norank.txt" \
  --ocr_prompt_path "${REPO_ROOT}/prompts/active/semantic_image_description.txt" \
  --family_config "${REPO_ROOT}/config/family_category_route4_v1.json" \
  --llm_model "${LLM_MODEL}" \
  --device cuda \
  --llm_decoding_mode deterministic \
  --llm_max_pixels 3584 \
  --proposal_score_threshold 0.40 \
  --enriched_context \
  --per_image_mode \
  --document_ocr \
  --ocr_doc_route_only \
  --max_new_tokens 1024 \
  --eval

python "${REPO_ROOT}/scripts/add_sam_masks.py" \
  --det_path "${OUTPUT_ROOT}/stage3/query_submission.json" \
  --gt_path "${SUPPORT_JSON}" \
  --image_dir "${SUPPORT_DIR}" \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --sam_model_type vit_h \
  --device cuda \
  --output_path "${OUTPUT_ROOT}/stage4_sam/query_submission_segm.json"

echo "Saved support inference outputs under ${OUTPUT_ROOT}"
