# VLM-Guided Detection and Rematching for Private Object Localization

Inference code for the extended abstract *VLM-Guided Detection and Rematching
for Private Object Localization*.

## Method

**Phase 1 — Detection.** A VLM (Qwen3-VL-8B) examines the query image and
produces two outputs: semantic cues (detector-friendly noun phrases) and a
predicted supercategory among four PII groups — *document* (8 categories),
*health* (5), *card* (2), *tattoo* (1). All categories in the predicted group,
together with the cues, are passed as text prompts to an open-vocabulary
detector (Grounding DINO), which proposes candidate boxes over a wider search
space than fixed category names alone.

| SC | Category |
|---|---|
| Document (8) | Bills or receipt, Bank statement, Transcript, <br> Letters with address, Local newspaper, <br> Medical record document, Doctors prescription, <br> Mortgage or investment report |
| Health (5) | Pregnancy test, Pregnancy test box, <br> Condom box, Condom with plastic bag, <br> Empty pill bottle |
| Card (2) | Credit or debit card, Business card |
| Tattoo sleeve (1) | Tattoo sleeve |

**Phase 2 — Rematching.** The query image, annotated with all proposed boxes,
is presented to the VLM in a single call with the Phase-1 context and a scene
caption. The VLM rematches each proposal against the closed set of categories
within the predicted supercategory, accepting boxes that match a target
category and rejecting the rest. Accepted boxes are passed to SAM for
segmentation.

## Setup

```bash
conda create -n fsol python=3.10 -y
conda activate fsol

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -U openmim
mim install "mmcv==2.1.0" "mmengine==0.10.7" "mmdet==3.3.0"
pip install -r requirements.txt
```

Checkpoints (paths are passed at runtime, so any directory works):

```bash
# SAM ViT-H
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Grounding DINO Swin-T: groundingdino_swint_ogc_mmdet-822d7e9d.pth
# from the MMDetection model zoo
# (https://github.com/open-mmlab/mmdetection/blob/main/configs/grounding_dino/)
```

`Qwen/Qwen3-VL-8B-Instruct` is fetched from the Hugging Face hub on first use.

## Run

```bash
bash runs/run_all.sh \
  <query_images_dir> \
  <query_set_images_info.json> \
  <output_root> \
  Qwen/Qwen3-VL-8B-Instruct \
  <groundingdino_checkpoint> \
  <sam_checkpoint>
```

Final predictions are written to
`<output_root>/stage4_sam/query_submission_segm.json`.

## License

Apache License 2.0 (see [LICENSE](LICENSE)). External components (Grounding
DINO, Segment Anything, MMDetection, Qwen3-VL) retain their upstream licenses.
