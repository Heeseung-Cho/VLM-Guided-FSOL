#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render support images with GT bbox and mask overlays.')
    p.add_argument('--support_json', required=True)
    p.add_argument('--image_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--mask_color', default='red')
    p.add_argument('--mask_alpha', type=int, default=88)
    p.add_argument('--bbox_color', default='red')
    p.add_argument('--bbox_width', type=int, default=5)
    return p.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _segmentation_polygons(segmentation: object) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    if not isinstance(segmentation, list):
        return polygons
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        polygons.append([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)])
    return polygons


def main() -> None:
    args = parse_args()
    support = _load_json(Path(args.support_json))
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in support.get('annotations', []):
        anns_by_image[int(ann['image_id'])].append(ann)

    bbox_rgb = ImageColor.getrgb(args.bbox_color)
    mask_rgb = ImageColor.getrgb(args.mask_color)
    mask_rgba = (*mask_rgb, args.mask_alpha)

    for image_info in support.get('images', []):
        image_id = int(image_info['id'])
        image_path = image_dir / image_info['file_name']
        if not image_path.is_file():
            print(f'[skip] missing image: {image_path}')
            continue

        base = ImageOps.exif_transpose(Image.open(image_path)).convert('RGBA')
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        base_draw = ImageDraw.Draw(base)

        for ann in anns_by_image.get(image_id, []):
            for pts in _segmentation_polygons(ann.get('segmentation')):
                overlay_draw.polygon(pts, fill=mask_rgba, outline=bbox_rgb)
            x, y, w, h = ann['bbox']
            base_draw.rectangle((x, y, x + w, y + h), outline=bbox_rgb, width=args.bbox_width)

        rendered = Image.alpha_composite(base, overlay).convert('RGB')
        out_path = output_dir / f"support_{image_id}_{Path(image_info['file_name']).stem}.png"
        rendered.save(out_path)
        print(f'saved {out_path}')


if __name__ == '__main__':
    main()
