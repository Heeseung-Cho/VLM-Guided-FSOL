from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mmcv
import torch
from PIL import Image

from common.detection import (
    DetectionCandidate,
    GroundingDinoLocalizer,
    SamSegmenter,
    dedupe_preserve_order,
    load_support_image_paths,
)
from common.text_utils import preprocess_caption
from common.vlm import SwiftVLMCaller, release_torch_runtime
from semantic.supercategory_config import (
    canonicalize_supercategory_name,
    get_category_supercategory,
    render_prompt_with_supercategory_config,
)

NEGATIVE_VALUES = {'yes', 'true', '1'}
LOW_SIGNAL_PROMPTS = {
    'blurry', 'white', 'black', 'small', 'large', 'background', 'table', 'wooden table',
    'wooden surface', 'surface', 'floor', 'room', 'photo', 'image', 'object'
}
PRIORITY_SUPERCATEGORY_TERMS = (
    'document', 'paper', 'receipt', 'newspaper', 'card', 'bottle', 'prescription',
    'record', 'statement', 'report', 'transcript', 'test', 'box', 'sleeve', 'letter'
)
DESCRIPTIVE_SPLIT_PATTERNS = (
    r'\bsuggesting\b',
    r'\bshowing\b',
    r'\bwith\b',
    r'\bsitting\b',
    r'\blying\b',
    r'\bplaced\b',
    r'\bon top of\b',
    r'\bin the\b',
    r'\bon the\b',
    r'\bpartially\b',
    r'\bnext to\b',
)


@dataclass
class SemanticCue:
    raw_text: str = ''
    supercategory: str = ''
    route_type: str = ''
    route_confidence: str = ''
    categories: list[str] = field(default_factory=list)
    summary: str = ''
    proposal_prompts: list[str] = field(default_factory=list)
    null_likely: bool = False
    text_hint_raw: str = ''
    text_hint_summary: str = ''
    text_hint_tokens: list[str] = field(default_factory=list)


@dataclass
class SupportReferenceCrop:
    image_id: int
    category_name: str
    crop_path: str


class SemanticController:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        decoding_mode: str = 'deterministic',
        seed: int | None = None,
        max_pixels: int = 448,
        instruction: str | None = None,
        query_only_instruction: str | None = None,
        client: SwiftVLMCaller | None = None,
    ) -> None:
        if not query_only_instruction:
            raise ValueError(
                'query_only_instruction is required. Pass --query_prompt_path '
                'to load a prompt template from prompts/active/.'
            )
        self.client = client or SwiftVLMCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            seed=seed,
            max_pixels=max_pixels,
        )
        self.instruction = render_prompt_with_supercategory_config(instruction) if instruction else ''
        self.query_only_instruction = render_prompt_with_supercategory_config(query_only_instruction)

    def infer_query_only(self, query_image_path: str) -> SemanticCue:
        raw_text = self.client.generate(query_image_path, instruction=self.query_only_instruction)
        return _parse_semantic_cue(raw_text)

    def infer_query_only_with_raw(self, query_image_path: str) -> tuple[SemanticCue, str]:
        raw_text = self.client.generate(query_image_path, instruction=self.query_only_instruction)
        return _parse_semantic_cue(raw_text), raw_text


def detect_free_text(
    localizer: GroundingDinoLocalizer,
    image_path: str,
    cue_text: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    max_dets: int = 50,
) -> list[DetectionCandidate]:
    if not cue_text:
        return []
    image = mmcv.imread(image_path, channel_order='rgb')
    prompt = preprocess_caption(cue_text)
    label_texts = [label.strip() for label in cue_text.split(',') if label.strip()]
    if not label_texts:
        return []
    try:
        result = localizer.model(inputs=image, texts=[prompt])
    except RuntimeError as exc:
        if 'selected index k out of range' in str(exc):
            return []
        raise
    if isinstance(result, list):
        result = result[0]
    predictions = result.get('predictions', []) if isinstance(result, dict) else []
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
    detections: list[DetectionCandidate] = []
    for idx in top_indices.tolist():
        score = float(scores[idx].item())
        if score < box_threshold or score < text_threshold:
            continue
        label_text = label_texts[idx] if idx < len(label_texts) else label_texts[0]
        detections.append(
            DetectionCandidate(
                score=score,
                label_text=label_text,
                category_id=-1,
                xyxy=[float(v) for v in boxes[idx].tolist()],
            )
        )
    return detections


def _parse_semantic_cue(raw_text: str) -> SemanticCue:
    raw_categories = _extract_tag(raw_text, 'categories')
    categories = _normalize_category_list(raw_categories)
    # Prefer explicit <route_type>; fall back to category-to-supercategory derivation when absent.
    explicit_route = canonicalize_supercategory_name(_extract_tag(raw_text, 'route_type'))
    if explicit_route and explicit_route.lower() not in {'none', 'null', ''}:
        route_type = explicit_route
    elif categories:
        route_type = get_category_supercategory(categories[0])
    else:
        route_type = ''
    supercategory = route_type
    summary = _extract_tag(raw_text, 'summary')
    cue = _extract_tag(raw_text, 'cue')
    null_text = _extract_tag(raw_text, 'null').strip().lower()
    if null_text in NEGATIVE_VALUES:
        null_likely = True
    else:
        null_likely = not categories or not route_type
    # proposal_prompts = VLM cue + summary + category names (strong noun anchors for G-DINO).
    proposal_prompts = _normalize_prompt_list(cue, summary, categories=categories)
    return SemanticCue(
        raw_text=raw_text,
        supercategory=supercategory,
        route_type=route_type,
        route_confidence=_normalize_route_confidence(_extract_tag(raw_text, 'route_confidence')),
        categories=categories,
        summary=summary,
        proposal_prompts=proposal_prompts,
        null_likely=null_likely,
    )


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ''
    return match.group(1).strip()


def _extract_category_text(raw_text: str) -> str:
    """Extract category name tolerating prompt-format drift.

    Handles three observed VLM output patterns:
    - <category>X</category> (well-formed)
    - <X> (tag-as-name drift, e.g. `<pregnancy test box>`)
    - plain X (no tags)
    """
    tagged = _extract_tag(raw_text, 'category')
    if tagged:
        return tagged
    stripped = (raw_text or '').strip()
    if not stripped:
        return ''
    single_tag = re.fullmatch(r'\s*<\s*([^<>/]+?)\s*/?>\s*', stripped)
    if single_tag:
        return single_tag.group(1).strip()
    return stripped


def _normalize_route_confidence(text: str) -> str:
    normalized = _normalize_phrase(text)
    if normalized in {'high', 'medium', 'low'}:
        return normalized
    return ''


def _normalize_category_list(raw_text: str) -> list[str]:
    items = []
    for value in re.split(r'[,;\n]+', raw_text or ''):
        normalized = canonicalize_supercategory_name(value)
        normalized_phrase = _normalize_phrase(normalized)
        if not normalized_phrase or normalized_phrase in {'none', 'null', 'empty', 'n a'}:
            continue
        if normalized not in items:
            items.append(normalized)
    return items[:4]


def _normalize_prompt_list(
    cue: str,
    *extra_sources: str,
    categories: list[str] | None = None,
) -> list[str]:
    """Build detector-friendly proposal_prompts = VLM cue + summary + category names."""
    items: list[str] = []
    values: list[str] = [cue, *extra_sources]
    for value in values:
        if not value:
            continue
        if value == cue:
            items.extend(part.strip() for part in cue.split(',') if part.strip())
        else:
            items.append(value.strip())
    if categories:
        items.extend(c.strip() for c in categories if c and c.strip())

    deduped = []
    for item in dedupe_preserve_order(items):
        normalized = item.strip()
        lowered = normalized.lower()
        if lowered in {'empty', 'none', 'no', 'n/a'}:
            continue
        if lowered in LOW_SIGNAL_PROMPTS:
            continue
        if len(lowered.split()) == 1 and lowered not in PRIORITY_SUPERCATEGORY_TERMS and lowered in {'white', 'black', 'blurry', 'small', 'large'}:
            continue
        deduped.append(normalized)

    ranked = sorted(deduped, key=_prompt_priority)
    return ranked[:5]


def _prompt_priority(prompt: str) -> tuple[int, int]:
    lowered = prompt.lower()
    if any(term in lowered for term in PRIORITY_SUPERCATEGORY_TERMS):
        return (0, len(prompt))
    if len(lowered.split()) <= 2:
        return (1, len(prompt))
    return (2, len(prompt))


def _should_run_detection(null_likely: bool, null_policy: str) -> bool:
    if null_policy == 'ignore':
        return True
    if null_policy == 'skip':
        return not null_likely
    if null_policy == 'strict':
        return not null_likely
    raise ValueError(f'Unknown null_policy: {null_policy}')


def _save_candidate_crop(query_image_path: str, xyxy: Sequence[float]) -> str:
    with Image.open(query_image_path) as src_image:
        image = src_image.convert('RGB')
        width, height = image.size
        x1, y1, x2, y2 = xyxy
        left_f, right_f = sorted((float(x1), float(x2)))
        top_f, bottom_f = sorted((float(y1), float(y2)))
        left = max(0, min(width - 1, int(round(left_f))))
        top = max(0, min(height - 1, int(round(top_f))))
        right = max(left + 1, min(width, int(round(right_f))))
        bottom = max(top + 1, min(height, int(round(bottom_f))))

        crop_w = right - left
        crop_h = bottom - top
        min_side = 28
        max_aspect_ratio = 50.0

        if crop_w < min_side:
            pad = min_side - crop_w
            left = max(0, left - pad // 2)
            right = min(width, right + (pad - pad // 2))
        if crop_h < min_side:
            pad = min_side - crop_h
            top = max(0, top - pad // 2)
            bottom = min(height, bottom + (pad - pad // 2))

        crop_w = right - left
        crop_h = bottom - top
        if crop_w / max(crop_h, 1) > max_aspect_ratio:
            target_h = min(height, max(min_side, int(round(crop_w / max_aspect_ratio))))
            extra = max(0, target_h - crop_h)
            top = max(0, top - extra // 2)
            bottom = min(height, bottom + (extra - extra // 2))
        elif crop_h / max(crop_w, 1) > max_aspect_ratio:
            target_w = min(width, max(min_side, int(round(crop_h / max_aspect_ratio))))
            extra = max(0, target_w - crop_w)
            left = max(0, left - extra // 2)
            right = min(width, right + (extra - extra // 2))

        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)

        crop = image.crop((left, top, right, bottom))
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                crop.save(tmp.name)
                return tmp.name
        finally:
            crop.close()
            image.close()
            release_torch_runtime()


def _build_support_reference_crops(
    support_json_path: str | Path,
    support_dir: str | Path,
    reference_source: str = 'crop',
) -> list[SupportReferenceCrop]:
    payload = json.loads(Path(support_json_path).read_text())
    support_dir = Path(support_dir)
    categories_by_id = {int(cat['id']): cat['name'].replace('_', ' ') for cat in payload.get('categories', [])}
    images_by_id = {int(image_info['id']): image_info for image_info in payload.get('images', [])}
    references: list[SupportReferenceCrop] = []
    for ann in payload.get('annotations', []):
        image_id = int(ann['image_id'])
        image_info = images_by_id.get(image_id)
        if image_info is None:
            continue
        category_name = categories_by_id.get(int(ann['category_id']))
        bbox = ann.get('bbox', [])
        if not category_name or len(bbox) != 4:
            continue
        image_path = support_dir / image_info['file_name']
        x, y, w, h = bbox
        if reference_source == 'full_image':
            reference_path = str(image_path)
        else:
            reference_path = _save_candidate_crop(str(image_path), [x, y, x + w, y + h])
        references.append(SupportReferenceCrop(image_id=image_id, category_name=category_name, crop_path=reference_path))
    references.sort(key=lambda item: item.image_id)
    return references


def _build_reference_match_instruction(
    base_instruction: str,
    support_references: Sequence[SupportReferenceCrop],
    allowed_categories: Sequence[str],
) -> str:
    support_lines = [f'{idx}. {entry.category_name}' for idx, entry in enumerate(support_references, start=1)]
    allowed_text = ', '.join(allowed_categories) if allowed_categories else 'all support categories'
    return '\n'.join([
        base_instruction,
        '',
        'The images are ordered as: support reference images first, then one candidate crop last.',
        'Support reference labels in order:',
        *support_lines,
        f'Allowed categories: {allowed_text}',
    ])


def _normalize_phrase(text: str) -> str:
    phrase = (text or '').strip().strip('[](){}')
    phrase = phrase.replace('_', ' ')
    phrase = re.sub(r'^\d+\.\s*', '', phrase)
    phrase = re.sub(r'\s+', ' ', phrase)
    lowered = phrase.lower()
    for pattern in DESCRIPTIVE_SPLIT_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            phrase = phrase[:match.start()].strip(' ,.;:-')
            break
    phrase = re.sub(r'[.;:]+$', '', phrase.strip())
    return phrase.lower()


def should_run_detection(null_likely: bool, null_policy: str) -> bool:
    return _should_run_detection(null_likely, null_policy)


__all__ = [
    'DetectionCandidate',
    'GroundingDinoLocalizer',
    'SamSegmenter',
    'SemanticCue',
    'SemanticController',
    'detect_free_text',
    'load_support_image_paths',
    'should_run_detection',
]
