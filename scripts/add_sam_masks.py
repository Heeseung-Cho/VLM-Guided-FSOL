"""Add SAM segmentation masks to an existing Stage-3 detection submission.

Reads COCO-format detection submission (bbox xywh), runs SAM ViT-H with each box
as prompt, and writes a new submission with a 'segmentation' (COCO RLE) field
per detection. Optionally runs pycocotools segm eval.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from segment_anything import SamPredictor, sam_model_registry
import torch


def _load_sam(checkpoint: str, model_type: str, device: str) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    sam.to(device=device)
    return SamPredictor(sam)


def _mask_to_coco_polygon(mask: np.ndarray) -> list[list[float]]:
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        polygon = contour.reshape(-1, 2).flatten().astype(float).tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--det_path', required=True, help='Stage-3 query_submission.json (bbox xywh)')
    ap.add_argument('--gt_path', required=True, help='COCO GT json (for image_id -> file_name, size)')
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--sam_checkpoint', required=True)
    ap.add_argument('--sam_model_type', default='vit_h')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output_path', required=True)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt_path).read_text())
    img_info = {img['id']: img for img in gt['images']}

    dets = json.loads(Path(args.det_path).read_text())
    by_image: dict[int, list[dict]] = defaultdict(list)
    for d in dets:
        by_image[d['image_id']].append(d)

    predictor = _load_sam(args.sam_checkpoint, args.sam_model_type, args.device)
    image_dir = Path(args.image_dir)
    out: list[dict] = []
    n_images = len(by_image)
    for idx, (image_id, image_dets) in enumerate(sorted(by_image.items()), start=1):
        info = img_info.get(image_id)
        if info is None:
            print(f'[skip] image_id={image_id} not in GT', file=sys.stderr)
            continue
        path = image_dir / info['file_name']
        if not path.is_file():
            print(f'[skip] image not found: {path}', file=sys.stderr)
            continue
        img = np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
        predictor.set_image(img)
        for d in image_dets:
            x, y, w, h = d['bbox']
            box = np.array([x, y, x + w, y + h], dtype=np.float32)[None, :]
            with torch.inference_mode():
                masks, _, _ = predictor.predict(
                    box=box, point_coords=None, point_labels=None, multimask_output=False,
                )
            mask = masks[0]
            entry = dict(d)
            poly = _mask_to_coco_polygon(mask)
            x0, y0, w0, h0 = d['bbox']
            if not poly:
                # SAM mask too small/sparse to yield a valid polygon (all contours < 3 points).
                # Fall back to bbox-derived rectangle polygon so pycocotools can evaluate.
                poly = [[float(x0), float(y0), float(x0 + w0), float(y0),
                         float(x0 + w0), float(y0 + h0), float(x0), float(y0 + h0)]]
                area_val = float(w0 * h0)
            else:
                # COCO segm convention: area = pixel count of the binary mask
                area_val = float(mask.sum())
            # Submission convention: bbox in integer pixel coordinates [x, y, w, h]
            entry['bbox'] = [int(round(x0)), int(round(y0)),
                             int(round(w0)), int(round(h0))]
            entry['segmentation'] = poly
            entry['area'] = area_val
            out.append(entry)
        if hasattr(predictor, 'reset_image'):
            predictor.reset_image()
        if idx % 10 == 0 or idx == n_images:
            print(f'[{idx}/{n_images}] image_id={image_id} dets={len(image_dets)}', flush=True)

    # Submission output is now strictly real detections only. Missing image_ids are
    # simply absent from the file; do not emit any dummy or empty-placeholder rows.
    real_only = [x for x in out if x.get('score', 0) > 0]
    real_present_ids = {x['image_id'] for x in real_only}
    all_image_ids = sorted({im['id'] for im in gt['images']})

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_path).write_text(json.dumps(real_only))
    print(
        f'Saved real-only submission: {len(real_only)} detections, '
        f'{len(real_present_ids)}/{len(all_image_ids)} unique image_ids to {args.output_path}'
    )


if __name__ == '__main__':
    main()
