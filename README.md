# semantic_vlm_privacy_extabs

Minimal code snapshot aligned with the extended abstract:

- [VizWiz_Workshop_2026-3.pdf](./VizWiz_Workshop_2026-3.pdf)
- Title: `VLM-Guided Detection and Rematching for Private Object Localization`

This repo keeps only the runtime path used by the abstract:

1. Stage 1: VLM predicts a privacy supercategory and detector-friendly semantic cues
2. Family expansion: expand the predicted supercategory to its allowed category set
3. Stage 2: Grounding DINO generates candidate boxes from the Stage-1 cue list
4. Stage 3: the same VLM jointly rematches all detected candidates on the full image
5. SAM converts the final detections into segmentation masks

The repo intentionally excludes training code, support-reference matching workflows, and legacy calibration branches.

## Repository Layout

- `semantic/run_stage1_semantic.py`
- `scripts/expand_stage1_to_family.py`
- `semantic/run_stage2_detection.py`
- `semantic/run_stage3_minimal.py`
- `scripts/add_sam_masks.py`
- `scripts/render_support_gt.py`
- `prompts/active/semantic_query_route4_v1.txt`
- `prompts/active/stage3_per_image_norank.txt`
- `prompts/active/semantic_image_description.txt`
- `config/family_category_route4_v1.json`
- `runs/run_extended_abstract.sh`
- `runs/run_render_support_gt.sh`

## Prerequisites

- Conda env: `psi`
- Python path rooted at this repo
- Query images and COCO-style metadata json
- Grounding DINO checkpoint
- SAM checkpoint
- Qwen3-VL model path or Hugging Face model id

This repo does not ship datasets or model weights.

## End-to-End Run

Run everything inside `psi`:

```bash
source /home/choheeseung/miniconda3/etc/profile.d/conda.sh
conda activate psi

bash runs/run_extended_abstract.sh \
  /path/to/query_images \
  /path/to/query_set_images_info.json \
  /path/to/output_root \
  Qwen/Qwen3-VL-8B-Instruct \
  /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  /path/to/sam_vit_h_4b8939.pth
```

Outputs:

- `output_root/stage1/stage1_semantic.json`
- `output_root/stage1_fam/stage1_semantic.json`
- `output_root/stage2/stage2_detection_gdino_ft.json`
- `output_root/stage3/query_submission.json`
- `output_root/stage4_sam/query_submission_segm.json`

## Notes

- Stage 1 uses the route4 supercategory taxonomy in `config/family_category_route4_v1.json`.
- Stage 3 is the full-image joint rematching path (`--per_image_mode`), which is the abstract-aligned Phase-2 inference path.
- The detector config is local to this repo. Checkpoint paths are passed at runtime instead of being baked into the config.

## Support GT Overlay

To render the 16 support images with ground-truth bbox and mask overlays only:

```bash
source /home/choheeseung/miniconda3/etc/profile.d/conda.sh
conda activate psi

bash runs/run_render_support_gt.sh
```

Outputs are written to `artifacts/support_gt_overlay/`.

## Support Inference

To run the full abstract pipeline on the 16 support images:

```bash
source /home/choheeseung/miniconda3/etc/profile.d/conda.sh
conda activate psi

bash runs/run_support_inference.sh \
  Qwen/Qwen3-VL-8B-Instruct \
  /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  /path/to/sam_vit_h_4b8939.pth
```

Outputs are written to `artifacts/support_inference/`, including:

- `stage1/stage1_semantic.json`
- `stage2/stage2_detection_gdino_ft.json`
- `stage3/query_submission.json`
- `stage3/detection_metrics.json`
- `stage4_sam/query_submission_segm.json`

## Citation

If you use this code snapshot, cite the workshop abstract and keep the code/paper terminology consistent with the two-phase VLM-guided detection and rematching pipeline.
