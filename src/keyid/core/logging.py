from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import IO, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path(__file__).resolve().parents[1]

FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)s "
    "| run=%(keyid_run)s | stage=%(keyid_stage)s | item=%(keyid_item)s "
    "| batch=%(keyid_batch)s | %(keyid_location)s | %(message)s"
)
DATEFMT = "%Y-%m-%d %H:%M:%S"


EXTERNAL_STDOUT = "external/stdout"
EXTERNAL_STDERR = "external/stderr"

_run_id: contextvars.ContextVar[str] = contextvars.ContextVar("keyid_run_id", default="-")
_stage_id: contextvars.ContextVar[str] = contextvars.ContextVar("keyid_stage_id", default="-")
_item_id: contextvars.ContextVar[str] = contextvars.ContextVar("keyid_item_id", default="-")
_batch_no: contextvars.ContextVar[str] = contextvars.ContextVar("keyid_batch_no", default="-")

_SECRETS: list[str] = []
_SECRETS_LOCK = threading.Lock()

REDACTED = "***REDACTED***"


_FALLBACK_RUN = "-"
_FALLBACK_STAGE = "-"


def now_iso() -> str:

    return datetime.now().astimezone().isoformat()


def set_run_context(run_id: str) -> None:

    global _FALLBACK_RUN
    _FALLBACK_RUN = run_id or "-"
    _run_id.set(run_id)


def set_stage_context(stage_id: str) -> None:

    global _FALLBACK_STAGE
    _FALLBACK_STAGE = stage_id or "-"
    _stage_id.set(stage_id)


def reset_context() -> None:

    global _FALLBACK_RUN, _FALLBACK_STAGE
    _FALLBACK_RUN = "-"
    _FALLBACK_STAGE = "-"
    _run_id.set("-")
    _stage_id.set("-")
    _item_id.set("-")
    _batch_no.set("-")


def is_configured() -> bool:
    return any(getattr(handler, "keyid_handler", False) for handler in logging.getLogger().handlers)


def ensure_configured(
    level: str = "INFO",
    *,
    console: bool = True,
    files: Sequence[Path] = (),
    run_id: str | None = None,
) -> None:

    if is_configured():
        if run_id is not None:
            set_run_context(run_id)
        for path in files:
            add_file_handler(Path(path))
        return
    configure(level, console=console, files=files, run_id=run_id)


def add_file_handler(path: Path, level: str | None = None) -> logging.Handler:

    handler = _make_handler(path=Path(path))
    if level is not None:
        handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logging.getLogger().addHandler(handler)
    return handler


@contextlib.contextmanager
def stage_context(stage_id: str) -> Iterator[None]:
    token = _stage_id.set(stage_id)
    try:
        yield
    finally:
        _stage_id.reset(token)


@contextlib.contextmanager
def item_context(sample_id: str, batch: str | None = None) -> Iterator[None]:
    item_token = _item_id.set(sample_id)
    batch_token = _batch_no.set(batch) if batch is not None else None
    try:
        yield
    finally:
        _item_id.reset(item_token)
        if batch_token is not None:
            _batch_no.reset(batch_token)


def register_secret(value: object) -> None:

    if not isinstance(value, str) or len(value) < 8:
        return
    with _SECRETS_LOCK:
        if value not in _SECRETS:
            _SECRETS.append(value)


def _scrub(value: object) -> object:
    if not isinstance(value, str):
        return value
    return scrub_text(value)


def scrub_text(text: str) -> str:

    if not text:
        return text
    with _SECRETS_LOCK:
        secrets = tuple(_SECRETS)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return text


def registered_secrets() -> tuple[str, ...]:
    with _SECRETS_LOCK:
        return tuple(_SECRETS)


def _scrub_args(args: object) -> object:
    if isinstance(args, dict):
        return {key: _scrub_args(value) for key, value in args.items()}
    if isinstance(args, (list, tuple)):
        return type(args)(_scrub_args(value) for value in args)
    return _scrub(args)


def _location(record: logging.LogRecord) -> str:
    origin = getattr(record, "origin", None)
    if origin:
        return str(origin)
    pathname = record.pathname or "<unknown>"
    try:
        path = Path(pathname).resolve()
    except OSError:
        return f"{pathname}:{record.lineno}"
    for base in (CODE_ROOT, PROJECT_ROOT):
        try:
            return f"{path.relative_to(base)}:{record.lineno}"
        except ValueError:
            continue
    return f"{pathname}:{record.lineno}"


class KeyIDFormatter(logging.Formatter):


    def __init__(self) -> None:
        super().__init__(fmt=FORMAT, datefmt=DATEFMT)

    def format(self, record: logging.LogRecord) -> str:
        for attribute, default in (
            ("keyid_run", "-"),
            ("keyid_stage", "-"),
            ("keyid_item", "-"),
            ("keyid_batch", "-"),
        ):
            if getattr(record, attribute, None) is None:
                setattr(record, attribute, default)
        if not getattr(record, "keyid_location", None):
            record.keyid_location = _location(record)


        return scrub_text(super().format(record))


def _context_or_fallback(variable: contextvars.ContextVar[str], fallback: str) -> str:
    value = variable.get()
    return fallback if value in ("", "-") else value


class _ContextFilter(logging.Filter):


    def filter(self, record: logging.LogRecord) -> bool:


        record.keyid_run = _context_or_fallback(_run_id, _FALLBACK_RUN)
        record.keyid_stage = _context_or_fallback(_stage_id, _FALLBACK_STAGE)
        record.keyid_item = _item_id.get()
        record.keyid_batch = _batch_no.get()
        record.keyid_location = _location(record)
        record.msg = _scrub_args(record.msg)
        if record.args:
            record.args = _scrub_args(record.args)
        return True


def _make_handler(stream: IO[str] | None = None, path: Path | None = None) -> logging.Handler:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    else:


        handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(KeyIDFormatter())
    handler.addFilter(_ContextFilter())
    handler.keyid_handler = True
    return handler


def configure(
    level: str = "INFO",
    *,
    console: bool = True,
    files: Sequence[Path] = (),
    run_id: str | None = None,
) -> list[logging.Handler]:

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    resolved = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(resolved)
    if run_id is not None:
        set_run_context(run_id)

    handlers: list[logging.Handler] = []
    if console:
        handlers.append(_make_handler(stream=sys.stderr))
    for path in files:
        handlers.append(_make_handler(path=Path(path)))
    for handler in handlers:
        root.addHandler(handler)
    return handlers


def debug(logger: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    logger.debug(message, *args, stacklevel=2, **kwargs)


def info(logger: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    logger.info(message, *args, stacklevel=2, **kwargs)


def warning(logger: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    logger.warning(message, *args, stacklevel=2, **kwargs)


def error(logger: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    logger.error(message, *args, stacklevel=2, **kwargs)


def exception(logger: logging.Logger, message: str, *args: object, **kwargs: object) -> None:
    logger.exception(message, *args, stacklevel=2, **kwargs)


class _StreamPump(threading.Thread):


    def __init__(self, stream, logger: logging.Logger, level: int, origin: str) -> None:
        super().__init__(daemon=True, name=f"keyid-log-pump-{origin}")
        self._stream = stream
        self._logger = logger
        self._level = level
        self._origin = origin

    def run(self) -> None:
        try:
            while True:


                raw = self._stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                line = line.rstrip("\n")
                if line:
                    self._logger.log(self._level, line, extra={"origin": self._origin})
        except Exception:
            self._logger.debug("log pump for %s stopped", self._origin, exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                self._stream.close()


def capture_stream(stream, logger: logging.Logger, *, level: int = logging.INFO, origin: str) -> _StreamPump:

    pump = _StreamPump(stream, logger, level, origin)
    pump.start()
    return pump


def capture_external_logging(logger: logging.Logger | None = None) -> None:

    target = logger or logging.getLogger()
    for handler in list(target.handlers):
        if getattr(handler, "keyid_handler", False):
            continue
        target.removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()
