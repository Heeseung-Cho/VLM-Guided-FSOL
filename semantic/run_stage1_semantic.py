#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from pprint import pformat

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.vlm import SwiftVLMCaller, release_torch_runtime
from common.vlm import _resolve_device_map
from semantic.supercategory_config import (
    get_active_supercategory_config_path,
    get_supercategory_names,
    set_active_supercategory_config,
)
from semantic.semantic_gdino_sam import SemanticController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 1 query-only semantic inference.')
    parser.add_argument('--query_dir', required=True)
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm_max_new_tokens', type=int, default=128)
    parser.add_argument('--llm_decoding_mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm_seed', type=int, default=None)
    parser.add_argument('--llm_max_pixels', type=int, default=448)
    parser.add_argument('--supercategory_config', '--supercategory-config', dest='supercategory_config', default=None)
    parser.add_argument('--query_prompt_path', default=None)
    parser.add_argument('--null_policy', choices=['strict', 'skip', 'ignore'], default='ignore')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--image_id', type=int, default=None)
    parser.add_argument('--runtime_stats_jsonl', default=None)
    parser.add_argument('--save_raw_text', action='store_true')
    parser.add_argument('--save_global_caption', action='store_true')
    parser.add_argument('--global_caption_prompt_path', default=None)
    parser.add_argument('--cuda_cleanup_interval', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    set_active_supercategory_config(args.supercategory_config)
    if not args.query_prompt_path:
        raise ValueError('--query_prompt_path is required (e.g. prompts/active/semantic_query.txt)')
    query_prompt_text = Path(args.query_prompt_path).read_text().strip()
    global_caption_prompt_text = None
    if args.global_caption_prompt_path:
        global_caption_prompt_text = Path(args.global_caption_prompt_path).read_text().strip()

    dataset = json.loads(Path(args.json_path).read_text())
    images = dataset['images']
    if args.image_id is not None:
        images = [img for img in images if img['id'] == args.image_id]
    if args.limit is not None:
        images = images[:args.limit]

    shared_vlm = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        device=args.device,
    )
    controller = SemanticController(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        query_only_instruction=query_prompt_text,
        client=shared_vlm,
    )
    request_config = getattr(shared_vlm, 'request_config', None)
    runtime_config = {
        'parsed_args': {
            'llm_model': args.llm_model,
            'device': args.device,
            'llm_max_new_tokens': args.llm_max_new_tokens,
            'llm_decoding_mode': args.llm_decoding_mode,
            'llm_seed': args.llm_seed,
            'llm_max_pixels': args.llm_max_pixels,
            'supercategory_config': get_active_supercategory_config_path(),
        },
        'vlm_caller': {
            'model_path': shared_vlm.model_path,
            'max_new_tokens': shared_vlm.max_new_tokens,
            'decoding_mode': shared_vlm.decoding_mode,
            'seed': shared_vlm.seed,
            'max_pixels': shared_vlm.max_pixels,
            'device': shared_vlm.device,
            'resolved_device_map': _resolve_device_map(shared_vlm.device),
        },
        'supercategory_spec': {
            'config_path': get_active_supercategory_config_path(),
            'supercategory_names': get_supercategory_names(),
        },
        'request_config': {
            'max_tokens': getattr(request_config, 'max_tokens', None),
            'temperature': getattr(request_config, 'temperature', None),
            'top_k': getattr(request_config, 'top_k', None),
            'top_p': getattr(request_config, 'top_p', None),
            'seed': getattr(request_config, 'seed', None),
            'repetition_penalty': getattr(request_config, 'repetition_penalty', None),
        },
    }
    print('[stage1] runtime_config')
    print(pformat(runtime_config, sort_dicts=False))

    outputs: list[dict[str, object]] = []
    runtime_stats_path = Path(args.runtime_stats_jsonl).resolve() if args.runtime_stats_jsonl else None
    runtime_stats_fh = runtime_stats_path.open('w') if runtime_stats_path else None
    progress = tqdm(images, desc='stage1 semantic', unit='image')
    try:
        for index, image_info in enumerate(progress, start=1):
            query_image_path = str((Path(args.query_dir) / image_info['file_name']).resolve())
            image_start = time.perf_counter()
            if args.save_raw_text:
                semantic, raw_text = controller.infer_query_only_with_raw(query_image_path)
            else:
                semantic = controller.infer_query_only(query_image_path)
                raw_text = None
            global_caption = ''
            global_caption_raw_text = None
            if args.save_global_caption:
                if not global_caption_prompt_text:
                    raise ValueError('--save_global_caption requires --global_caption_prompt_path')
                global_caption_raw_text = shared_vlm.generate(query_image_path, instruction=global_caption_prompt_text)
                global_caption = _extract_caption_from_raw(global_caption_raw_text)
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed_sec = time.perf_counter() - image_start
            record = {
                'image_id': image_info['id'],
                'query_image_path': query_image_path,
                'null_policy': args.null_policy,
                'semantic_supercategory': semantic.supercategory,
                'route_type': semantic.route_type,
                'route_confidence': semantic.route_confidence,
                'semantic_categories': semantic.categories,
                'proposal_prompts': semantic.proposal_prompts,
                'null_likely': semantic.null_likely,
            }
            if raw_text is not None:
                record['semantic_raw_text'] = raw_text
            if args.save_global_caption:
                record['global_caption'] = global_caption
                record['global_caption_raw_text'] = global_caption_raw_text or ''
            runtime_stats = {
                'image_index': index,
                'image_id': image_info['id'],
                'elapsed_sec': round(elapsed_sec, 3),
                'semantic_supercategory': semantic.supercategory,
                'route_type': semantic.route_type,
                'route_confidence': semantic.route_confidence,
                'semantic_categories': semantic.categories,
                'null_likely': semantic.null_likely,
                'prompt_count': len(semantic.proposal_prompts),
            }
            if raw_text is not None:
                runtime_stats['raw_text_char_len'] = len(raw_text)
                runtime_stats['raw_text_word_len'] = len(raw_text.split())
            if args.save_global_caption:
                runtime_stats['global_caption'] = global_caption
                runtime_stats['global_caption_char_len'] = len(global_caption)
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                runtime_stats['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_memory_reserved_mb'] = round(torch.cuda.memory_reserved(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_max_memory_allocated_mb'] = round(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2), 1)
                torch.cuda.reset_peak_memory_stats(args.device)
            outputs.append(record)
            progress.set_postfix({
                'image_id': image_info['id'],
                'route': semantic.route_type or '-',
                'categories': ','.join(semantic.categories[:2]) or semantic.supercategory or '-',
                'null': semantic.null_likely,
                'sec': f'{elapsed_sec:.1f}',
            })
            message = (
                f"Stage1 image_id={image_info['id']} categories={semantic.categories or [semantic.supercategory]} "
                f"route={semantic.route_type or '-'} route_conf={semantic.route_confidence or '-'} "
                f"null={semantic.null_likely} prompts={len(semantic.proposal_prompts)} "
                f"elapsed_sec={elapsed_sec:.1f}"
            )
            if args.save_global_caption and global_caption:
                message += f" global_caption='{global_caption[:80]}'"
            if 'gpu_memory_reserved_mb' in runtime_stats:
                message += (
                    f" gpu_alloc_mb={runtime_stats['gpu_memory_allocated_mb']:.1f}"
                    f" gpu_reserved_mb={runtime_stats['gpu_memory_reserved_mb']:.1f}"
                    f" gpu_peak_mb={runtime_stats['gpu_max_memory_allocated_mb']:.1f}"
                )
            tqdm.write(message)
            if runtime_stats_fh is not None:
                runtime_stats_fh.write(json.dumps(runtime_stats) + '\n')
                runtime_stats_fh.flush()
            if args.cuda_cleanup_interval > 0 and index % args.cuda_cleanup_interval == 0:
                release_torch_runtime()
        progress.close()
    finally:
        if runtime_stats_fh is not None:
            runtime_stats_fh.close()

    payload = {
        'config': vars(args),
        'records': outputs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tqdm.write(f'Saved Stage 1 outputs to: {output_path}')


def _extract_caption_from_raw(raw_text: str) -> str:
    start_tag = '<caption>'
    end_tag = '</caption>'
    lower = raw_text.lower()
    start = lower.find(start_tag)
    end = lower.find(end_tag)
    if start != -1 and end != -1 and end > start:
        return raw_text[start + len(start_tag):end].strip()
    return raw_text.strip()


if __name__ == '__main__':
    main()
