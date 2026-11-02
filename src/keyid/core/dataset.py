from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .errors import DatasetError

_FORBIDDEN_ID_CHARS = ("/", "\\", "\x00")


def validate_sample_id(sample_id: str) -> str:

    if not isinstance(sample_id, str):
        raise DatasetError(f"sample_id must be a string, got {type(sample_id).__name__}")
    if not sample_id.strip():
        raise DatasetError("sample_id must not be empty")
    if sample_id in (".", ".."):
        raise DatasetError(f"sample_id {sample_id!r} is not a usable directory name")
    for char in _FORBIDDEN_ID_CHARS:
        if char in sample_id:
            raise DatasetError(f"sample_id {sample_id!r} contains a forbidden character {char!r}")
    if Path(sample_id).is_absolute():
        raise DatasetError(f"sample_id {sample_id!r} must not be an absolute path")
    parts = Path(sample_id).parts
    if any(part == ".." for part in parts):
        raise DatasetError(f"sample_id {sample_id!r} contains path traversal")
    return sample_id


@dataclass
class DatasetItem:


    sample_id: str
    raw: dict[str, Any]
    prompt: str
    img_paths: tuple[str, ...]
    root: Path
    search_roots: tuple[Path, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:

        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "img_paths": list(self.img_paths),
            "fields": dict(self.raw),
        }

    def resolve_image(self, path_value: str) -> Path:

        if not isinstance(path_value, str):
            raise DatasetError(f"{self.sample_id}: img_paths entries must be strings")
        candidate = Path(path_value)
        if candidate.is_absolute():
            if candidate.exists():
                return candidate
            raise DatasetError(f"{self.sample_id}: image not found: {candidate}")

        direct = self.root / candidate
        if direct.exists():
            return direct

        return self._search_by_basename(path_value)

    def _search_by_basename(self, path_value: str) -> Path:
        if not self.search_roots:
            raise DatasetError(
                f"{self.sample_id}: image not found: {self.root / path_value}"
            )
        basename = Path(path_value).name
        matches = [match for root in self.search_roots if root.is_dir() for match in root.rglob(basename)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise DatasetError(f"{self.sample_id}: ambiguous image path {path_value!r}: {matches}")
        raise DatasetError(f"{self.sample_id}: image not found: {path_value}")

    def resolved_images(self) -> list[Path]:
        if not self.img_paths:
            raise DatasetError(f"{self.sample_id}: img_paths is empty")
        return [self.resolve_image(value) for value in self.img_paths]


@dataclass
class Dataset:
    format: str
    root: Path
    metadata: Path
    items: dict[str, DatasetItem] = field(default_factory=dict)

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items.values())

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self.items

    def select(self, sample_ids: list[str] | None = None) -> "Dataset":

        if not sample_ids:
            return self
        wanted = set(sample_ids)
        missing = sorted(wanted - set(self.items))
        if missing:
            raise DatasetError(f"requested sample ids not present in the dataset: {missing}")
        return Dataset(
            format=self.format,
            root=self.root,
            metadata=self.metadata,
            items={sid: item for sid, item in self.items.items() if sid in wanted},
        )

    def sample_ids(self) -> list[str]:
        return list(self.items)

    def search_roots_of(self) -> list[Path]:

        if not self.items:
            return []
        first = next(iter(self.items.values()))
        return list(first.search_roots)


def load_dataset(
    metadata: Path,
    root: Path,
    *,
    format: str = "opens2v",
    search_roots: tuple[Path, ...] = (),
) -> Dataset:

    metadata = Path(metadata)
    root = Path(root)
    if format != "opens2v":
        raise DatasetError(f"unsupported dataset format {format!r}; KeyID currently supports 'opens2v'")
    if not metadata.is_file():
        raise DatasetError(f"dataset metadata not found: {metadata}")
    if not root.is_dir():
        raise DatasetError(f"dataset root not found: {root}")

    try:
        with open(metadata, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"dataset metadata is not valid JSON: {metadata}: {exc}") from exc
    except OSError as exc:
        raise DatasetError(f"cannot read dataset metadata {metadata}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DatasetError(
            f"OpenS2V metadata must be a mapping of sample_id -> object, got {type(payload).__name__}"
        )

    items: dict[str, DatasetItem] = {}
    for raw_id, entry in payload.items():
        sample_id = validate_sample_id(raw_id)
        if not isinstance(entry, Mapping):
            raise DatasetError(f"{sample_id}: dataset entry must be an object")
        prompt = entry.get("prompt", "")
        if prompt is None:
            prompt = ""
        if not isinstance(prompt, str):
            raise DatasetError(f"{sample_id}: prompt must be a string, got {type(prompt).__name__}")
        img_paths = entry.get("img_paths", [])
        if img_paths is None:
            img_paths = []
        if not isinstance(img_paths, list) or not all(isinstance(item, str) for item in img_paths):
            raise DatasetError(f"{sample_id}: img_paths must be a list of strings")
        items[sample_id] = DatasetItem(
            sample_id=sample_id,
            raw=dict(entry),
            prompt=prompt,
            img_paths=tuple(img_paths),
            root=root,
            search_roots=tuple(search_roots),
        )

    return Dataset(format=format, root=root, metadata=metadata, items=items)
