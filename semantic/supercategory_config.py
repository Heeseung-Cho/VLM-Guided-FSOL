from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_SUPERCATEGORY_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'config' / 'supercategory.json'

_ACTIVE_SUPERCATEGORY_CONFIG_PATH = DEFAULT_SUPERCATEGORY_CONFIG_PATH
_ACTIVE_SUPERCATEGORY_CONFIG: dict | None = None


def _normalize_text(text: str) -> str:
    return ' '.join((text or '').strip().replace('_', ' ').lower().split())


def _validate_supercategory_config(payload: dict, path: Path) -> dict:
    supercategories = payload.get('supercategories')
    if supercategories is None:
        # Legacy key kept for one-time backward compat with older configs.
        supercategories = payload.get('families')
    if not isinstance(supercategories, dict) or not supercategories:
        raise ValueError(f'Invalid supercategory config at {path}: missing non-empty "supercategories" object')
    normalized_names: dict[str, str] = {}
    normalized_families: dict[str, dict[str, object]] = {}
    for supercategory_name, spec in supercategories.items():
        categories: list[str]
        description: str | None = None
        if isinstance(spec, list):
            categories = spec
        elif isinstance(spec, dict):
            categories = spec.get('categories')
            description = spec.get('description')
            if description is not None and not isinstance(description, str):
                raise ValueError(
                    f'Invalid supercategory spec for {supercategory_name!r} at {path}: "description" must be str|null'
                )
        else:
            raise ValueError(
                f'Invalid supercategory spec for {supercategory_name!r} at {path}: expected list[str] or object'
            )
        if not isinstance(categories, list):
            raise ValueError(
                f'Invalid supercategory spec for {supercategory_name!r} at {path}: expected "categories" list[str]'
            )
        normalized_supercategory = _normalize_text(supercategory_name)
        if normalized_supercategory in normalized_names:
            raise ValueError(f'Duplicate normalized supercategory name {supercategory_name!r} in {path}')
        normalized_names[normalized_supercategory] = supercategory_name
        normalized_families[supercategory_name] = {
            'categories': categories,
            'description': description.strip() if isinstance(description, str) else None,
        }
    payload['supercategories'] = normalized_families
    return payload


def load_supercategory_config(path: str | Path | None = None) -> dict:
    config_path = Path(path or DEFAULT_SUPERCATEGORY_CONFIG_PATH).resolve()
    payload = json.loads(config_path.read_text(encoding='utf-8'))
    payload = _validate_supercategory_config(payload, config_path)
    payload['_config_path'] = str(config_path)
    return payload


def set_active_supercategory_config(path: str | Path | None = None) -> dict:
    global _ACTIVE_SUPERCATEGORY_CONFIG_PATH, _ACTIVE_SUPERCATEGORY_CONFIG
    _ACTIVE_SUPERCATEGORY_CONFIG_PATH = Path(path or DEFAULT_SUPERCATEGORY_CONFIG_PATH).resolve()
    _ACTIVE_SUPERCATEGORY_CONFIG = load_supercategory_config(_ACTIVE_SUPERCATEGORY_CONFIG_PATH)
    return _ACTIVE_SUPERCATEGORY_CONFIG


def get_active_supercategory_config() -> dict:
    global _ACTIVE_SUPERCATEGORY_CONFIG
    if _ACTIVE_SUPERCATEGORY_CONFIG is None:
        _ACTIVE_SUPERCATEGORY_CONFIG = load_supercategory_config(_ACTIVE_SUPERCATEGORY_CONFIG_PATH)
    return _ACTIVE_SUPERCATEGORY_CONFIG


def get_active_supercategory_config_path() -> str:
    return str(get_active_supercategory_config().get('_config_path', DEFAULT_SUPERCATEGORY_CONFIG_PATH))


def get_supercategory_names() -> list[str]:
    return list(get_active_supercategory_config()['supercategories'].keys())


def canonicalize_supercategory_name(name: str) -> str:
    normalized_name = _normalize_text(name)
    if not normalized_name:
        return ''
    for canonical_name in get_supercategory_names():
        if _normalize_text(canonical_name) == normalized_name:
            return canonical_name
    return name.strip()


def get_supercategory_categories(name: str) -> list[str]:
    canonical_name = canonicalize_supercategory_name(name)
    spec = get_active_supercategory_config()['supercategories'].get(canonical_name, {})
    categories = spec.get('categories', []) if isinstance(spec, dict) else []
    return [str(category) for category in categories if str(category).strip()]


def get_all_categories() -> list[str]:
    categories: list[str] = []
    for supercategory_name in get_supercategory_names():
        for category in get_supercategory_categories(supercategory_name):
            if category not in categories:
                categories.append(category)
    return categories


def get_category_supercategory(category_name: str) -> str:
    normalized_target = _normalize_text(category_name)
    if not normalized_target:
        return ''
    for supercategory_name in get_supercategory_names():
        for category in get_supercategory_categories(supercategory_name):
            if _normalize_text(category) == normalized_target:
                return supercategory_name
    return ''


def get_supercategory_description(name: str) -> str | None:
    canonical_name = canonicalize_supercategory_name(name)
    spec = get_active_supercategory_config()['supercategories'].get(canonical_name, {})
    if not isinstance(spec, dict):
        return None
    description = spec.get('description')
    if not isinstance(description, str):
        return None
    stripped = description.strip()
    return stripped or None


def render_supercategory_description_block() -> str:
    lines = []
    for supercategory_name in get_supercategory_names():
        description = get_supercategory_description(supercategory_name)
        if description:
            lines.append(f'- {supercategory_name}: {description}')
    return '\n'.join(lines)


def render_category_description_block() -> str:
    payload = get_active_supercategory_config()
    category_descriptions = payload.get('category_descriptions', {})
    lines = []
    for category in get_all_categories():
        description = None
        if isinstance(category_descriptions, dict):
            description = category_descriptions.get(category)
        if isinstance(description, str) and description.strip():
            lines.append(f'- {category}: {description.strip()}')
    return '\n'.join(lines)


def render_prompt_with_supercategory_config(prompt_text: str) -> str:
    supercategory_list = ', '.join(get_supercategory_names())
    supercategory_descriptions = render_supercategory_description_block()
    category_list = ', '.join(get_all_categories())
    category_descriptions = render_category_description_block()
    if '{{supercategory_list}}' in prompt_text:
        prompt_text = prompt_text.replace('{{supercategory_list}}', supercategory_list)
    if '{{route_list}}' in prompt_text:
        prompt_text = prompt_text.replace('{{route_list}}', supercategory_list)
    if '{{supercategory_descriptions}}' in prompt_text:
        prompt_text = prompt_text.replace('{{supercategory_descriptions}}', supercategory_descriptions)
    if '{{route_descriptions}}' in prompt_text:
        prompt_text = prompt_text.replace('{{route_descriptions}}', supercategory_descriptions)
    if '{{category_list}}' in prompt_text:
        prompt_text = prompt_text.replace('{{category_list}}', category_list)
    if '{{category_descriptions}}' in prompt_text:
        prompt_text = prompt_text.replace('{{category_descriptions}}', category_descriptions)
    pattern = re.compile(r'<supercategory>one .*? chosen from:.*?</supercategory>', flags=re.IGNORECASE | re.DOTALL)
    replacement = f'<supercategory>one mid-level supercategory chosen from: {supercategory_list}</supercategory>'
    if pattern.search(prompt_text):
        prompt_text = pattern.sub(replacement, prompt_text, count=1)
    if '{{supercategory_descriptions}}' in prompt_text:
        prompt_text = prompt_text.replace('{{supercategory_descriptions}}', '')
    prompt_text = re.sub(
        r'\nSupercategory guidance:\n(?:[ \t]*\n)?(?=\nRules:)',
        '\n',
        prompt_text,
        flags=re.IGNORECASE,
    )
    if pattern.search(prompt_text) or '{{supercategory_list}}' in prompt_text:
        return prompt_text
    return '\n'.join([prompt_text.rstrip(), '', f'Allowed supercategory set: {supercategory_list}'])
