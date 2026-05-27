from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import mmcv
import numpy as np
import torch
from torchvision import ops

from common.model_loaders import load_groundingdino_model, load_sam_model
from common.text_utils import preprocess_caption


@dataclass
class DetectionCandidate:
    score: float
    label_text: str
    category_id: int
    xyxy: list[float]


def dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        lowered = item.lower()
        if lowered in seen:
            continue
        deduped.append(item)
        seen.add(lowered)
    return deduped


def load_support_image_paths(support_json_path: str | Path, support_dir: str | Path) -> list[str]:
    payload = json.loads(Path(support_json_path).read_text())
    support_dir = Path(support_dir)
    return [str((support_dir / image_info['file_name']).resolve()) for image_info in payload['images']]


class GroundingDinoLocalizer:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda') -> None:
        self.model = load_groundingdino_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)

    def detect(
        self,
        image_path: str,
        cue_text: str,
        categories_dict: dict[str, int],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_iou: float = 0.8,
        max_dets: int = 100,
    ) -> list[DetectionCandidate]:
        if not cue_text or cue_text == 'No output found':
            return []
        image = mmcv.imread(image_path, channel_order='rgb')
        prompt = preprocess_caption(cue_text)
        label_texts = [cls.strip() for cls in cue_text.split(',') if cls.strip()]
        if not label_texts:
            return []

        result = self.model(inputs=image, texts=[prompt])
        if isinstance(result, list):
            result = result[0]
        if not isinstance(result, dict) or 'predictions' not in result:
            return []
        predictions = result['predictions']
        if not predictions or not isinstance(predictions[0], dict):
            return []
        first_pred = predictions[0]
        boxes = first_pred.get('bboxes', [])
        scores = first_pred.get('scores', [])
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)
        if len(boxes) == 0 or len(scores) == 0:
            return []

        top_indices = torch.argsort(scores, descending=True)[:max_dets]
        keep_indices_relative = ops.nms(boxes[top_indices], scores[top_indices], iou_threshold=nms_iou)
        final_indices = top_indices[keep_indices_relative]

        detections: list[DetectionCandidate] = []
        for idx in final_indices.tolist():
            score = float(scores[idx].item())
            if score < box_threshold or score < text_threshold:
                continue
            label_text = label_texts[idx] if idx < len(label_texts) else label_texts[0]
            if label_text not in categories_dict:
                matched = next((cls for cls in label_texts if label_text in cls), None)
                if matched is None or matched not in categories_dict:
                    continue
                label_text = matched
            detections.append(
                DetectionCandidate(
                    score=score,
                    label_text=label_text,
                    category_id=categories_dict[label_text],
                    xyxy=[float(v) for v in boxes[idx].tolist()],
                )
            )
        return detections


class SamSegmenter:
    def __init__(self, checkpoint_path: str, model_type: str = 'vit_h', device: str = 'cuda') -> None:
        self.predictor = load_sam_model(model_type=model_type, checkpoint=checkpoint_path, device=device)

    @staticmethod
    def _mask_to_coco_polygon(mask: np.ndarray) -> list[list[float]]:
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for contour in contours:
            contour = contour.reshape(-1, 2)
            polygon = contour.flatten().astype(float).tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
        return polygons

    def segment(self, image_path: str, detections: Sequence[DetectionCandidate], image_id: int) -> list[dict[str, Any]]:
        if not detections:
            return []
        image = mmcv.imread(image_path, channel_order='rgb')
        image_np = np.array(image)
        results: list[dict[str, Any]] = []
        try:
            with torch.inference_mode():
                self.predictor.set_image(image_np)
                for det in detections:
                    xyxy = np.array(det.xyxy, dtype=np.float32)[None, :]
                    masks, _, _ = self.predictor.predict(
                        box=xyxy,
                        point_coords=None,
                        point_labels=None,
                        multimask_output=False,
                    )
                    x1, y1, x2, y2 = det.xyxy
                    results.append({
                        'image_id': image_id,
                        'score': det.score,
                        'category_id': det.category_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': float((x2 - x1) * (y2 - y1)),
                        'segmentation': self._mask_to_coco_polygon(masks[0]),
                        'label_text': det.label_text,
                    })
        finally:
            if hasattr(self.predictor, 'reset_image'):
                self.predictor.reset_image()
        return results


__all__ = [
    'DetectionCandidate',
    'GroundingDinoLocalizer',
    'SamSegmenter',
    'dedupe_preserve_order',
    'load_support_image_paths',
]
