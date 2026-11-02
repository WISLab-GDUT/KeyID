#!/usr/bin/env python3
"""Normalize legacy Track1 portrait metadata and copy references into a new dataset."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from keyid.core.dataset import validate_sample_id


#: ``id001.webp`` / ``id0001.webp`` -> ``1``.  The leading zeros are optional and
#: the captured digits are normalised through ``int`` so both forms collide
#: correctly and are reported as duplicates.
IMAGE_ID_RE = re.compile(r"^id0*(\d+)\.webp$")


def extract_image_id(img_path: str) -> str:
    image_name = Path(str(img_path)).name
    match = IMAGE_ID_RE.match(image_name)
    if not match:
        raise ValueError(f"Cannot extract numeric image id from img path: {img_path!r}")
    return str(int(match.group(1)))


def convert_eval_list(samples: object) -> dict:
    if not isinstance(samples, list):
        raise TypeError(f"Expected eval.json to contain a JSON list, got {type(samples).__name__}.")

    converted: dict[str, dict] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected item {index} to be a JSON object.")
        if "img" not in sample:
            raise KeyError(f"Missing 'img' field in item {index}.")

        image_id = extract_image_id(sample["img"])
        if image_id in converted:
            raise ValueError(f"Duplicate image id found: {image_id}")
        converted[image_id] = sample

    # Numeric order, so ids 2 and 10 do not sort as "10" < "2".
    return dict(sorted(converted.items(), key=lambda item: int(item[0])))


def prepare(metadata: Path, image_root: Path, output: Path, *, category: str = 'person') -> Path:
    """Prepare the portrait-only Track1 benchmark; never modify the source tree."""
    if category not in {'person', 'man', 'woman'}:
        raise ValueError('Track1 portrait category must be person, man or woman')
    metadata, image_root, output = metadata.resolve(), image_root.resolve(), output.resolve()
    payload = json.loads(metadata.read_text(encoding='utf-8'))
    if isinstance(payload, list):
        payload = convert_eval_list(payload)
    if not isinstance(payload, dict) or not payload:
        raise ValueError('metadata must be a non-empty Track1 list or ID-keyed dictionary')
    if not image_root.is_dir():
        raise ValueError('image_root must be a directory containing the source images')
    if output == image_root or image_root.is_relative_to(output) or metadata.is_relative_to(output):
        raise ValueError('output must be separate from the source dataset')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('output must be absent or an empty directory')
    converted, copies = {}, []
    # Validate every entry before copying anything.
    for sample_id, entry in payload.items():
        validate_sample_id(sample_id)
        if not isinstance(entry, dict) or not isinstance(entry.get('img'), str) or not entry['img'].strip():
            raise ValueError(f'{sample_id}: expected a non-empty legacy img string')
        if not isinstance(entry.get('prompt'), str) or not entry['prompt'].strip():
            raise ValueError(f'{sample_id}: expected a non-empty prompt')
        raw = Path(entry['img'])
        candidates = {p.resolve() for p in (image_root / raw, image_root / raw.name) if p.is_file()}
        if len(candidates) != 1:
            raise ValueError(f'{sample_id}: expected exactly one source image, found {len(candidates)}')
        source = candidates.pop()
        if source.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
            raise ValueError(f'{sample_id}: unsupported image suffix')
        relative = Path('images') / category / (sample_id + source.suffix.lower())
        item = dict(entry)
        # Preserve the prompt and all other fields, but omit the stale source path.
        item.pop('img')
        item['img_paths'], item['class_label'] = [relative.as_posix()], [category]
        converted[sample_id] = item
        copies.append((source, relative))
    output.mkdir(parents=True, exist_ok=True)
    for source, relative in copies:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    result = output / 'metadata.json'
    result.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--category', choices=['person', 'man', 'woman'], default='person')
    args = parser.parse_args()
    try:
        result = prepare(args.metadata, args.image_root, args.output_dir, category=args.category)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
