from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Mapping

from keyid.core.errors import StageError

DEFAULT_BASE_URL = "https://api.openai.com/v1"


SENSITIVE_RESPONSE_KEYS = ("api_key", "authorization", "token")


def build_client(
    credentials: Mapping[str, Any],
    endpoint: Mapping[str, Any] | None = None,
    *,
    timeout: float | None = None,
    params: Mapping[str, Any] | None = None,
):

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise StageError(
            "the 'openai' package is not installed in this stage's interpreter; "
            "install it or point stages.<id>.python at an environment that has it"
        ) from exc

    endpoint = endpoint or {}
    params = params or {}
    api_key = params.get("api_key") if "api_key" in params else credentials.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise StageError(
            "set a non-empty YAML string in stages.<id>.params.api_key or "
            "credentials.openai.api_key; an explicitly configured params.api_key "
            "takes precedence and must not be empty (use a gitignored "
            "configs/*.local.yaml copy for real keys)"
        )
    base_url = (
        params.get("base_url") if "base_url" in params
        else endpoint.get("base_url") or DEFAULT_BASE_URL
    )
    if not isinstance(base_url, str) or not base_url.strip():
        raise StageError(
            "stages.<id>.params.base_url (or endpoints.openai.base_url) "
            "must be a non-empty YAML string"
        )

    kwargs: dict[str, Any] = {"api_key": api_key.strip(), "base_url": base_url.strip()}
    if timeout is not None:
        kwargs["timeout"] = float(timeout)
    proxy = credentials.get("proxy") or credentials.get("http_proxy")
    if proxy:


        import httpx

        kwargs["http_client"] = httpx.Client(proxy=str(proxy), timeout=timeout or 600.0)
    return OpenAI(**kwargs)


def image_to_data_url(path: Path) -> str:

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def response_summary(response: Any) -> dict[str, Any]:

    summary: dict[str, Any] = {}
    for attribute in ("id", "model", "status"):
        value = getattr(response, attribute, None)
        if isinstance(value, (str, int, float, bool)):
            summary[attribute] = value
    usage = getattr(response, "usage", None)
    if usage is not None:
        for attribute in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(usage, attribute, None)
            if isinstance(value, (int, float)):
                summary.setdefault("usage", {})[attribute] = int(value)
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        summary["output_chars"] = len(text)
    return summary


def read_text_resource(override: Any, packaged: Path) -> str:

    path = Path(override) if override else packaged
    if not path.is_file():
        raise StageError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_after_marker(text: str, marker: str) -> str:

    if marker and marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def require_text(value: Any, *, what: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise StageError(f"{what} is empty")
    return text
