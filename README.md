# VLM-Guided Detection and Rematching for Private Object Localization

This repository provides the supplementary inference code for the extended
abstract:

> VLM-Guided Detection and Rematching for Private Object Localization

The released code is a compact snapshot of the runtime pipeline used in the
abstract. It is intended to document and reproduce the inference procedure, not
to provide a full training framework or a general-purpose detection library.

## Method Overview

The pipeline localizes privacy-sensitive objects through a two-stage
vision-language reasoning process followed by mask generation:

1. **Semantic query generation.** A VLM predicts a privacy supercategory and
   detector-friendly textual cues for each query image.
2. **Supercategory expansion.** The predicted supercategory is expanded into
   its allowed leaf categories.
3. **Candidate detection.** Grounding DINO proposes candidate bounding boxes
   using the semantic cues from Stage 1.
4. **VLM rematching.** The VLM jointly reviews all detected candidates in the
   full image and selects the final privacy-object localization result.
5. **Mask conversion.** SAM converts the selected bounding boxes into
   segmentation masks.

This repository intentionally excludes training code, internal experiment
branches, legacy calibration scripts, and support-reference matching variants
that are not part of the released abstract pipeline.

## Repository Contents

- `semantic/run_stage1_semantic.py`: VLM semantic query generation.
- `scripts/expand_stage1_to_categories.py`: supercategory-to-leaf-category expansion.
- `semantic/run_stage2_detection.py`: Grounding DINO candidate detection.
- `semantic/run_stage3_minimal.py`: VLM candidate rematching.
- `scripts/add_sam_masks.py`: SAM-based mask generation.
- `scripts/render_support_gt.py`: support-set ground-truth overlay utility.
- `config/supercategory.json`: privacy supercategory taxonomy.
- `prompts/active/`: prompts used by the released inference path.
- `runs/run_extended_abstract.sh`: end-to-end query-set runner.
- `runs/run_support_inference.sh`: support-set inference runner.
- `runs/run_render_support_gt.sh`: support-set overlay runner.
- `requirements.txt`: Python package versions used by the authors.
- `THIRD_PARTY.md`: notes on external checkpoints and assets.

## External Assets

The repository does not include datasets, model weights, or third-party source
trees. The following assets must be provided by the user:

- query images and a COCO-style metadata JSON file;
- a Grounding DINO checkpoint compatible with the provided MMDetection config
  (the authors used `groundingdino_swint_ogc_mmdet-822d7e9d.pth`);
- a SAM checkpoint, such as `sam_vit_h_4b8939.pth`;
- a Qwen3-VL model path or Hugging Face model identifier (the authors used
  `Qwen/Qwen3-VL-8B-Instruct`).

### Downloading the detector and segmenter checkpoints

The release scripts expect the checkpoint *paths* as command-line arguments,
so any local directory works. The authors used:

```bash
mkdir -p checkpoints

# Segment Anything (ViT-H, ~2.4 GB). Official Meta release:
#   https://github.com/facebookresearch/segment-anything#model-checkpoints
wget -O checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Grounding DINO Swin-T (MMDetection port, ~659 MB). Released via the
# MMDetection model zoo; the checkpoint is the one paired with
# configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py shipped in this repo.
# See https://github.com/open-mmlab/mmdetection/blob/main/configs/grounding_dino/
# for the current download URL of `groundingdino_swint_ogc_mmdet-822d7e9d.pth`
# (or fetch it once via `mim download` after installing the OpenMMLab stack).
```

`Qwen/Qwen3-VL-8B-Instruct` is downloaded automatically from the Hugging Face
hub on first use; alternatively, point `--llm_model` at a locally mirrored
copy.

The released support-set utilities assume a `BIV-Priv-Seg`-style layout at
`${PROJECT_ROOT}/data/Biv-priv-seg/{support_set.json,support_images/}`.
`PROJECT_ROOT` has no default and must be exported before running the
`run_support_*` scripts.

## Environment

The experiments were run in a Conda environment named `psi` with Python 3.10
and CUDA 12.8. The pinned versions in `requirements.txt` reflect the authors'
working setup; other versions may work but were not tested.

```bash
conda create -n psi python=3.10 -y
conda activate psi

# 1. PyTorch + torchvision matched to your local CUDA. Adjust the index URL
#    if you are not on CUDA 12.8.
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 2. OpenMMLab stack. mmcv 2.x must be installed via mim, not from PyPI.
pip install -U openmim
mim install "mmcv==2.1.0" "mmengine==0.10.7" "mmdet==3.3.0"

# 3. Remaining Python dependencies (Qwen3-VL backend, SAM, COCO eval, etc.).
pip install -r requirements.txt
```

The released scripts expect CUDA execution by default. Qwen3-VL-8B, Grounding
DINO Swin-T, and SAM ViT-H together require roughly 24 GB of GPU memory in
the configuration used by the authors.

## End-to-End Inference

Run the released query-set pipeline from the repository root:

```bash
conda activate psi

bash runs/run_extended_abstract.sh \
  /path/to/query_images \
  /path/to/query_set_images_info.json \
  /path/to/output_root \
  Qwen/Qwen3-VL-8B-Instruct \
  /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  /path/to/sam_vit_h_4b8939.pth
```

The runner writes the following outputs:

- `output_root/stage1/stage1_semantic.json`
- `output_root/stage1_fam/stage1_semantic.json`
- `output_root/stage2/stage2_detection_gdino_ft.json`
- `output_root/stage3/query_submission.json`
- `output_root/stage4_sam/query_submission_segm.json`

## Support-Set Utilities

To render support-image ground-truth bounding boxes and masks:

```bash
conda activate psi

PROJECT_ROOT=/path/to/project_or_dataset_root \
bash runs/run_render_support_gt.sh
```

To run the released inference pipeline on the support images:

```bash
conda activate psi

PROJECT_ROOT=/path/to/project_or_dataset_root \
bash runs/run_support_inference.sh \
  Qwen/Qwen3-VL-8B-Instruct \
  /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  /path/to/sam_vit_h_4b8939.pth
```

The support inference runner writes outputs under
`artifacts/support_inference/`, including:

- `stage1/stage1_semantic.json`
- `stage2/stage2_detection_gdino_ft.json`
- `stage3/query_submission.json`
- `stage3/detection_metrics.json`
- `stage4_sam/query_submission_segm.json`

## Reproducibility Notes

- Stage 1 uses the privacy supercategory taxonomy in
  `config/supercategory.json`.
- Stage 3 uses the full-image joint rematching path enabled by
  `--per_image_mode`.
- Checkpoint paths are passed at runtime and are not hard-coded into the
  detector config.
- Because the repository does not redistribute data or weights, exact
  reproduction requires matching the external assets used in the abstract.

## Citation

If this supplementary code is useful for your work, please cite the accompanying
extended abstract:

```bibtex
@misc{cho2026vlmguidedprivacy,
  title = {VLM-Guided Detection and Rematching for Private Object Localization},
  author = {Cho, Heeseung and Park, Yuna and Kim, Esther and Kim, Jiho and Kim, Hyojoo and Wallraven, Christian and Oh, Junhyoung},
  year = {2026},
  note = {Extended abstract supplementary code}
}
```

Please update the citation entry with the final workshop proceedings metadata
when it becomes available.

## License

The code in this repository is released under the Apache License 2.0;
see [LICENSE](LICENSE) for the full text. External components — Grounding
DINO, Segment Anything, MMDetection, and Qwen3-VL — are governed by their
respective upstream licenses, which the user must consult separately.
