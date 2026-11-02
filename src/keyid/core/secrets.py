from __future__ import annotations

import re
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import urlsplit

REDACTED = "***REDACTED***"


SENSITIVE_KEY_PATTERN = re.compile(
    r"api_key|apikey|api-key|token|secret|password|passwd|credential|cookie"
    r"|authorization|auth|proxy",
    re.IGNORECASE,
)


PROXY_ENVIRONMENT_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


SENSITIVE_ENV_PATTERN = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH|CREDENTIAL|COOKIE", re.IGNORECASE
)

def expand_url_secrets(url: str) -> Iterator[str]:

    text = str(url).strip()
    if not text:
        return
    yield text
    if "://" not in text:
        return
    try:
        parts = urlsplit(text)
    except ValueError:
        return
    if parts.password:
        yield parts.password
        if parts.username:
            yield f"{parts.username}:{parts.password}"


def is_secret_key(key: str) -> bool:

    return bool(SENSITIVE_KEY_PATTERN.search(str(key)))


def is_sensitive_environment_key(key: str) -> bool:
    return bool(SENSITIVE_ENV_PATTERN.search(str(key)))


def iter_leaf_strings(node: Any) -> Iterator[str]:

    if isinstance(node, Mapping):
        for value in node.values():
            yield from iter_leaf_strings(value)
    elif isinstance(node, (list, tuple, set)):
        for value in node:
            yield from iter_leaf_strings(value)
    elif isinstance(node, str) and node.strip():
        yield node


def iter_sensitive_values(node: Any, *, assume_sensitive: bool = False) -> Iterator[str]:

    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from iter_sensitive_values(
                value, assume_sensitive=assume_sensitive or is_secret_key(str(key))
            )
    elif isinstance(node, (list, tuple, set)):
        for value in node:
            yield from iter_sensitive_values(value, assume_sensitive=assume_sensitive)
    elif isinstance(node, str) and node.strip() and assume_sensitive:


        yield from expand_url_secrets(node)


def iter_environment_secrets(environment: Mapping[str, Any]) -> Iterator[str]:

    for key, value in environment.items():
        name = str(key)
        text = "" if value is None else str(value)
        if not text:
            continue
        if name.upper() in PROXY_ENVIRONMENT_KEYS:
            yield from expand_url_secrets(text)
            continue
        if is_sensitive_environment_key(name):
            yield text


def iter_config_secrets(
    credentials: Mapping[str, Mapping[str, Any]] | None = None,
    params: Iterable[Mapping[str, Any]] = (),
    environment: Mapping[str, Any] | None = None,
) -> Iterator[str]:

    for namespace in (credentials or {}).values():
        yield from iter_sensitive_values(namespace, assume_sensitive=True)
    for param_block in params:
        yield from iter_sensitive_values(param_block)
    if environment:
        yield from iter_environment_secrets(environment)


def redact(node: Any, key: str = "", *, in_credentials: bool = False) -> Any:

    if isinstance(node, Mapping):
        subtree_sensitive = in_credentials or str(key) == "credentials"
        return {
            child_key: redact(value, str(child_key), in_credentials=subtree_sensitive)
            for child_key, value in node.items()
        }
    if isinstance(node, list):
        return [redact(item, key, in_credentials=in_credentials) for item in node]
    if not isinstance(node, str):
        return node
    if node == "":
        return node
    if in_credentials or is_secret_key(key):
        return REDACTED
    return node


def scrub_values(node: Any) -> Any:

    from .logging import scrub_text

    if isinstance(node, Mapping):
        return {key: scrub_values(value) for key, value in node.items()}
    if isinstance(node, list):
        return [scrub_values(value) for value in node]
    if isinstance(node, tuple):
        return tuple(scrub_values(value) for value in node)
    if isinstance(node, str):
        return scrub_text(node)
    return node


def register_secrets(
    credentials: Mapping[str, Mapping[str, Any]] | None = None,
    params: Iterable[Mapping[str, Any]] = (),
    environment: Mapping[str, Any] | None = None,
) -> int:

    from .logging import register_secret

    count = 0
    for value in iter_config_secrets(credentials=credentials, params=params, environment=environment):
        register_secret(value)
        count += 1
    return count
