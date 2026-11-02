from __future__ import annotations

import hashlib
from typing import Any


SEED_MODULUS = 2**31 - 1

_SEPARATOR = "\x1f"


def derive_seed(
    base_seed: int,
    *,
    stage: str,
    sample_id: str,
    extra: Any = "",
) -> int:

    payload = _SEPARATOR.join(
        (str(int(base_seed)), str(stage), str(sample_id), str(extra or ""))
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


def resolve_seed(
    base_seed: int,
    *,
    stage: str,
    sample_id: str,
    mode: str = "per_item",
    extra: Any = "",
) -> int:

    if mode == "fixed":
        return int(base_seed) % SEED_MODULUS
    if mode != "per_item":
        raise ValueError(f"unknown seed mode {mode!r}; expected 'fixed' or 'per_item'")
    return derive_seed(base_seed, stage=stage, sample_id=sample_id, extra=extra)
