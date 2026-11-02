from __future__ import annotations

from typing import Mapping

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


AUTHORITATIVE_VARIABLES = (CUDA_VISIBLE_DEVICES, "PLATFORM", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def parse_device(device: str | None) -> tuple[str | None, int | None]:

    if device is None:
        return (None, None)
    text = str(device).strip()
    if not text:
        return (None, None)
    name, _, index = text.partition(":")
    name = name.strip().lower()
    if not index:
        return (name, None)
    try:
        return (name, int(index))
    except ValueError:
        return (name, None)


def visible_devices_for(device: str | None) -> str | None:

    name, index = parse_device(device)
    if name == "cuda" and index is not None:
        return str(index)
    return None


def runtime_device(device: str | None, *, fallback: str = "cuda") -> str:

    name, index = parse_device(device)
    if name is None:
        return fallback
    if name == "cuda":
        return "cuda:0" if index is not None else "cuda"
    return name if index is None else f"{name}:{index}"


def check_device_environment_conflict(
    device: str | None,
    environment: Mapping[str, str],
    *,
    where: str,
) -> str | None:

    declared = environment.get(CUDA_VISIBLE_DEVICES)
    if declared is None:
        return None
    declared = str(declared).strip()
    expected = visible_devices_for(device)
    if expected is not None and declared != expected:
        return (
            f"{where}: environment.{CUDA_VISIBLE_DEVICES}={declared!r} contradicts "
            f"device={device!r} (which selects physical GPU {expected}). "
            "Keep one of them, or set device to cuda:{expected}."
        )
    if expected is None and str(device or "").strip().lower() in ("cpu",):
        return (
            f"{where}: device is 'cpu' but environment.{CUDA_VISIBLE_DEVICES}={declared!r} "
            "still exposes GPUs; remove one of the two."
        )
    return None
