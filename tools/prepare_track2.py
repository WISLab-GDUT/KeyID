#!/usr/bin/env python3
"""Prepare Track2 metadata and ordered reference images in a new dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from keyid.core.dataset import validate_sample_id
from keyid.core.errors import KeyIDError
from keyid.stages._dataset_prompt import resolve_captions_for_sample


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def _source_image(value: str, image_root: Path, sample_id: str) -> Path:
    raw = Path(value)
    candidates = [image_root / raw]
    parts = raw.parts
    if 'images' in parts:
        candidates.append(image_root.joinpath(*parts[parts.index('images') + 1:]))
    matches = {path.resolve() for path in candidates if path.is_file()}
    if not matches:
        matches = {path.resolve() for path in image_root.rglob(raw.name) if path.is_file()}
    if len(matches) != 1:
        raise ValueError(f'{sample_id}: expected exactly one source image for {value!r}, found {len(matches)}')
    source = matches.pop()
    if source.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f'{sample_id}: unsupported image suffix: {source.suffix}')
    return source


def _category(value: str, label: str | None, sample_id: str) -> str:
    parts = Path(value).parts
    if 'images' in parts and parts.index('images') + 2 < len(parts):
        category = parts[parts.index('images') + 1]
    elif label:
        category = label
    elif len(parts) > 1 and not Path(value).is_absolute() and parts[0] not in {'.', '..'}:
        category = parts[0]
    else:
        raise ValueError(f'{sample_id}: reference needs a category directory or class_label')
    category = category.strip().lower()
    validate_sample_id(category)
    return category


def prepare(metadata: Path, image_root: Path, output: Path) -> Path:
    metadata, image_root, output = metadata.resolve(), image_root.resolve(), output.resolve()
    payload = json.loads(metadata.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not payload:
        raise ValueError('metadata must be a non-empty ID-keyed Track2 dictionary')
    if not image_root.is_dir():
        raise ValueError('image_root must be a directory containing the source images')
    if output == image_root or image_root.is_relative_to(output) or metadata.is_relative_to(output):
        raise ValueError('output must be separate from the source dataset')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be absent or an empty directory')
    converted, copies = {}, []
    for sample_id, entry in payload.items():
        validate_sample_id(sample_id)
        if not isinstance(entry, dict):
            raise ValueError(f'{sample_id}: expected a metadata object')
        if not isinstance(entry.get('global_prompt'), str) or not entry['global_prompt'].strip():
            raise ValueError(f'{sample_id}: expected a non-empty global_prompt')
        resolve_captions_for_sample(entry, 'temporal_captions', sample_id=sample_id)
        paths = entry.get('img_paths')
        if not isinstance(paths, list) or not all(isinstance(path, str) and path.strip() for path in paths):
            raise ValueError(f'{sample_id}: expected img_paths as a list of non-empty strings')
        labels = entry.get('class_label')
        if labels is not None and (not isinstance(labels, list) or len(labels) != len(paths)
                                   or not all(isinstance(label, str) for label in labels)):
            raise ValueError(f'{sample_id}: class_label must contain one string per reference')
        item = dict(entry)
        item['img_paths'], item['class_label'] = [], []
        for index, value in enumerate(paths):
            category = _category(value, labels[index] if labels else None, sample_id)
            source = _source_image(value, image_root, sample_id)
            relative = Path('images') / category / sample_id / f'ref_{index:03d}{source.suffix.lower()}'
            item['img_paths'].append(relative.as_posix())
            item['class_label'].append(category)
            copies.append((source, relative))
        converted[sample_id] = item
    # Complete validation before creating the destination.
    output.mkdir(parents=True, exist_ok=True)
    for source, relative in copies:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    result = output / 'eval.json'
    result.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.metadata, args.image_root, args.output_dir)
    except (OSError, ValueError, TypeError, KeyError, KeyIDError) as exc:
        parser.exit(1, f'{exc}\n')
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
