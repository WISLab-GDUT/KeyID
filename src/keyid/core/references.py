from __future__ import annotations

from pathlib import Path
from typing import Iterable

HUMAN_CATEGORIES = frozenset({"man", "woman", "person"})


def image_category(path_value: str) -> str:

    parts = Path(str(path_value)).parts
    if not parts:
        return ""
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = lowered.index("images")
        if index + 1 < len(parts):
            return lowered[index + 1]
    return lowered[0]


def is_human_reference_image(path_value: str) -> bool:
    return image_category(path_value) in HUMAN_CATEGORIES


def human_reference_indices(img_paths: Iterable[str]) -> list[int]:

    return [index for index, path in enumerate(img_paths) if is_human_reference_image(str(path))]


def human_reference_images(img_paths: Iterable[str]) -> list[str]:

    return [str(path) for path in img_paths if is_human_reference_image(str(path))]
