from __future__ import annotations

from typing import Any, Mapping, Sequence

from keyid.core.references import (
    HUMAN_CATEGORIES, human_reference_images, image_category, is_human_reference_image,
)


DEFAULT_RELAY_SIGMA = 0.1448
DEFAULT_RELAY_WINDOW_OFFSET = 2
DEFAULT_RELAY_MIN_WINDOW = 0
DEFAULT_RELAY_CHUNK_SIZE = 64


class DatasetPromptError(ValueError):
    pass


def validate_prompt_fields(fields: Any, *, where: str) -> tuple[str, ...]:

    if fields is None:
        return ()
    if isinstance(fields, str) or not isinstance(fields, (list, tuple)):
        raise ValueError(f"{where}: prompt_fields must be a list of field names, got {fields!r}")
    names = tuple(str(field).strip() for field in fields)
    if not names or any(not name for name in names):
        raise ValueError(f"{where}: prompt_fields must list at least one non-empty field name")
    return names


def validate_caption_field(field: Any, *, where: str) -> str | None:
    if field is None:
        return None
    name = str(field).strip()
    if not name:
        raise ValueError(f"{where}: temporal_caption_field must be a non-empty field name")
    return name


def first_non_empty_text(source: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, str | None]:

    for field in fields:
        value = source.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text, field
    return "", None


def resolve_original_prompt(
    fields_source: Mapping[str, Any],
    prompt_fields: Sequence[str],
    *,
    sample_id: str,
) -> tuple[str, str]:

    if not prompt_fields:
        raise ValueError(
            "prompt_fields is not configured; the stage cannot tell which dataset "
            "field holds the prompt"
        )
    text, field = first_non_empty_text(fields_source, prompt_fields)
    if not text:
        raise DatasetPromptError(
            f"{sample_id}: none of the fields {list(prompt_fields)} hold a non-empty "
            "prompt"
        )
    return text, str(field)


def _caption_description(caption: Any, *, sample_id: str) -> str:

    if isinstance(caption, str):
        description = caption.strip()
        if not description:
            raise DatasetPromptError(f"{sample_id}: temporal caption has an empty description")
        return description
    if not isinstance(caption, Mapping):
        raise DatasetPromptError(
            f"{sample_id}: temporal caption must be an object or a string, "
            f"got {type(caption).__name__}"
        )
    description, _ = first_non_empty_text(caption, ("description", "prompt", "caption", "text"))
    if not description:
        raise DatasetPromptError(f"{sample_id}: temporal caption has an empty description: {caption!r}")
    return description


def _caption_interval(caption: Any, index: int, total: int, *, sample_id: str) -> tuple[float, float]:

    if isinstance(caption, Mapping) and "start" in caption and "end" in caption:
        try:
            return float(caption.get("start", 0.0)), float(caption.get("end", 0.0))
        except (TypeError, ValueError) as exc:
            raise DatasetPromptError(f"{sample_id}: caption interval is not numeric: {caption!r}") from exc
    if total <= 0:
        raise DatasetPromptError(f"{sample_id}: cannot divide an empty caption list")
    step = 1.0 / total
    return index * step, (index + 1) * step


def parse_temporal_captions(
    raw_value: Any,
    *,
    sample_id: str,
) -> list[dict[str, Any]]:

    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise DatasetPromptError(
            f"{sample_id}: temporal captions must be a list, got {type(raw_value).__name__}"
        )

    captions: list[dict[str, Any]] = []
    total = len(raw_value)
    for index, caption in enumerate(raw_value):
        start, end = _caption_interval(caption, index, total, sample_id=sample_id)
        description = _caption_description(caption, sample_id=sample_id)
        if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0 and start < end):


            raise DatasetPromptError(
                f"{sample_id}: invalid temporal caption interval [{start}, {end}) for {description!r}"
            )
        captions.append({"start": start, "end": end, "description": description})

    captions.sort(key=lambda caption: caption["start"])
    return captions


def resolve_captions_for_sample(
    fields_source: Mapping[str, Any],
    caption_field: str | None,
    *,
    sample_id: str,
) -> list[dict[str, Any]]:

    if caption_field is None:
        return []
    if caption_field not in fields_source:
        raise DatasetPromptError(
            f"{sample_id}: temporal_caption_field={caption_field!r} is configured but the "
            "dataset entry has no such field"
        )
    captions = parse_temporal_captions(fields_source.get(caption_field), sample_id=sample_id)
    if not captions:
        raise DatasetPromptError(
            f"{sample_id}: temporal_caption_field={caption_field!r} is empty; configured "
            "temporal captions must not be dropped silently"
        )
    return captions


def with_temporal_captions(global_prompt: str, captions: Sequence[Mapping[str, Any]]) -> str:

    return global_prompt + temporal_prompt_text(captions)


def temporal_prompt_text(captions: Sequence[Mapping[str, Any]]) -> str:

    return "".join(" " + str(caption["description"]) for caption in captions)


def build_relay_spec(
    global_prompt: str,
    captions: Sequence[Mapping[str, Any]],
    *,
    prompt_field: str | None,
    sigma: float = DEFAULT_RELAY_SIGMA,
    window_offset: int = DEFAULT_RELAY_WINDOW_OFFSET,
    min_window: int = DEFAULT_RELAY_MIN_WINDOW,
) -> dict[str, Any]:

    spec: dict[str, Any] = {
        "global_prompt": global_prompt,
        "local_prompts": [" " + str(caption["description"]) for caption in captions],
        "intervals": [
            {"start": float(caption["start"]), "end": float(caption["end"])} for caption in captions
        ],
        "prompt_field": prompt_field,
        "sigma": float(sigma),
        "window_offset": int(window_offset),
        "min_window": int(min_window),
    }
    return spec


def full_prompt_from_relay_spec(relay_spec: Mapping[str, Any]) -> str:

    return str(relay_spec["global_prompt"]) + "".join(
        str(local) for local in relay_spec.get("local_prompts") or []
    )
