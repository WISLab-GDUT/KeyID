from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import SCHEMA_VERSION, atomic_write_json, read_json
from .secrets import register_secrets, scrub_values
from .contract import (
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    BatchStage,
    ItemResult,
    StageContext,
    WorkItem,
    normalize_results,
)
from .devices import runtime_device
from .errors import EXIT_FATAL, EXIT_OK, EXIT_PARTIAL, StageError
from .logging import (
    capture_external_logging,
    configure,
    item_context,
    now_iso,
    set_stage_context,
)
from .registry import StageRegistry
from .results import STAGE_FATAL, StageResult

logger = logging.getLogger("keyid.stage_worker")


PATH_OUTPUT_KEYS = frozenset({"path", "image", "video", "audio", "manifest", "frames", "keyframes"})
PATH_OUTPUT_SUFFIXES = ("_path", "_paths", "_dir", "_dirs", "_file", "_files")


LEGAL_EXIT_CODES = (EXIT_OK, EXIT_PARTIAL)


def read_stdin_payload(stream=None) -> dict[str, Any]:

    stream = stream if stream is not None else sys.stdin
    raw = stream.read()
    if not raw or not raw.strip():
        raise StageError(
            "no runtime payload on stdin -- the stage worker must be started by "
            "`keyid run`, which pipes the invocation in"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageError(f"runtime payload on stdin is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageError(f"runtime payload must be an object, got {type(payload).__name__}")
    _validate_payload(payload)
    return payload


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise StageError(
            f"unsupported invocation schema_version {payload.get('schema_version')!r}"
        )
    for key in ("run_id", "run_dir", "stage_id", "stage_type", "attempt_log"):
        if not payload.get(key):
            raise StageError(f"runtime payload is missing {key!r}")


def load_invocation(path: Path) -> dict[str, Any]:

    payload = read_json(path, default=None)
    if not isinstance(payload, dict):
        raise StageError(f"invocation is missing or not an object: {path}")
    return payload


def _register_payload_secrets(payload: Mapping[str, Any]) -> int:

    return register_secrets(
        credentials=payload.get("credentials") or {},
        params=[payload.get("params") or {}],
        environment=payload.get("environment") or {},
    )


def _build_work_items(payload: Mapping[str, Any], run_dir: Path, stage_id: str) -> list[WorkItem]:
    items_dir = Path(run_dir) / "stages" / stage_id / "items"
    work_items: list[WorkItem] = []
    for entry in payload.get("items") or []:
        sample_id = str(entry["sample_id"])
        item_dir = items_dir / sample_id
        item_dir.mkdir(parents=True, exist_ok=True)
        work_items.append(
            WorkItem(
                sample_id=sample_id,
                raw=dict(entry.get("raw") or {}),
                inputs={str(k): dict(v) for k, v in (entry.get("inputs") or {}).items()},
                item_dir=item_dir,
            )
        )
    return work_items


def _build_context(payload: Mapping[str, Any]) -> StageContext:
    requested = str(payload["device"]) if payload.get("device") else None
    return StageContext(
        run_id=str(payload["run_id"]),
        run_dir=Path(payload["run_dir"]),
        stage_id=str(payload["stage_id"]),
        stage_type=str(payload["stage_type"]),
        attempt=int(payload.get("attempt", 1)),
        params=dict(payload.get("params") or {}),
        credentials={str(k): dict(v) for k, v in (payload.get("credentials") or {}).items()},
        endpoints={str(k): dict(v) for k, v in (payload.get("endpoints") or {}).items()},
        environment={str(k): str(v) for k, v in (payload.get("environment") or {}).items()},

        device=runtime_device(requested),
        device_requested=requested,
        concurrency=int(payload.get("concurrency", 1)),
        python=Path(payload.get("python") or sys.executable),
        logger=logger,


        needs=tuple(str(name) for name in (payload.get("needs") or ())),
        resuming=bool(payload.get("resuming", False)),
        dataset_root=(Path(payload["dataset"]["root"]) if payload.get("dataset", {}).get("root") else None),
        image_search_roots=tuple(
            Path(entry) for entry in (payload.get("dataset", {}).get("image_search_roots") or [])
        ),
    )


def _looks_like_path_output(key: str) -> bool:
    return key in PATH_OUTPUT_KEYS or key.endswith(PATH_OUTPUT_SUFFIXES)


def _validate_path_outputs(result: ItemResult, context: StageContext) -> ItemResult:

    stage_dir = context.stage_dir.resolve()
    for key, value in list(result.outputs.items()):
        if not _looks_like_path_output(key):
            continue
        values: Sequence[Any]
        if isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = [value]
        for candidate in values:
            if not isinstance(candidate, str) or not candidate:
                continue
            if Path(candidate).is_absolute():
                resolved = Path(candidate).resolve()
            else:
                resolved = (stage_dir / candidate).resolve()
            try:
                resolved.relative_to(stage_dir)
            except ValueError:
                return ItemResult.failed(
                    result.sample_id,
                    "StageOutputOutOfBounds",
                    f"output {key}={candidate!r} resolves outside {stage_dir}",
                )
            if not resolved.exists():
                return ItemResult.failed(
                    result.sample_id,
                    "MissingStageOutput",
                    f"output {key}={candidate!r} does not exist",
                )
    return result


def _write(document: StageResult, path: Path, *, final: bool = False) -> None:

    document.finished_at = now_iso() if final else None

    atomic_write_json(path, scrub_values(document.to_dict()))


def run_stage(payload: Mapping[str, Any]) -> int:
    run_dir = Path(payload["run_dir"])
    stage_id = str(payload["stage_id"])
    attempt_log = Path(payload["attempt_log"])

    configure(
        str(payload.get("log_level", "INFO")),
        console=True,
        files=[attempt_log],
        run_id=str(payload["run_id"]),
    )
    set_stage_context(stage_id)
    registered = _register_payload_secrets(payload)
    capture_external_logging()

    logger.info(
        "stage process started: type=%s attempt=%d items=%d device=%s secrets=%d",
        payload["stage_type"],
        int(payload.get("attempt", 1)),
        len(payload.get("items") or []),
        payload.get("device") or "default",
        registered,
    )

    registry = StageRegistry()
    for package in payload.get("stage_packages") or ["keyid.stages"]:
        registry.discover(str(package))
    stage_cls = registry.get(str(payload["stage_type"]))
    context = _build_context(payload)
    stage: BatchStage = stage_cls()

    result_path = run_dir / "stages" / stage_id / "result.json"
    document = StageResult.load_or_start(
        result_path,
        stage_id,
        context.stage_type,
        attempt=context.attempt,
    )


    document.clear_items()


    for sample_id, entry in (payload.get("preseeded_items") or {}).items():
        document.set_item(str(sample_id), entry)

    work_items = _build_work_items(payload, run_dir, stage_id)
    batch_size = max(1, int((payload.get("batch") or {}).get("size", 1)))
    checkpoint_every = max(1, int((payload.get("batch") or {}).get("checkpoint_every", 1)))

    try:
        try:
            stage.setup(context)


            capture_external_logging()
        except Exception as exc:
            logger.exception("stage setup failed: %s", stage_id)
            document.status = STAGE_FATAL
            document.error = {"type": exc.__class__.__name__, "message": str(exc)}
            _write(document, result_path, final=True)
            return EXIT_FATAL

        try:
            _process_all(
                stage,
                context,
                document,
                work_items,
                result_path,
                batch_size=batch_size,
                checkpoint_every=checkpoint_every,
            )
        except Exception as exc:
            logger.exception("stage failed while processing: %s", stage_id)
            document.status = STAGE_FATAL
            document.error = {"type": exc.__class__.__name__, "message": str(exc)}
            _write(document, result_path, final=True)
            return EXIT_FATAL
    finally:
        try:
            stage.close(context)
        except Exception:
            logger.exception("stage close failed: %s", stage_id)

    document.status = document.derive_status()
    _write(document, result_path, final=True)
    logger.info(
        "stage finished: status=%s summary=%s",
        document.status,
        json.dumps(document.summary(), ensure_ascii=False),
    )

    if document.has_problems():
        return EXIT_PARTIAL
    return EXIT_OK


def _process_all(
    stage: BatchStage,
    context: StageContext,
    document: StageResult,
    work_items: Sequence[WorkItem],
    result_path: Path,
    *,
    batch_size: int,
    checkpoint_every: int,
) -> None:
    results = stage.process_stream(list(work_items), context)
    if results is not None:
        _process_stream(results, context, document, work_items, result_path,
                        checkpoint_every=checkpoint_every)
        return
    total_batches = max(1, (len(work_items) + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, len(work_items), batch_size), start=1):
        batch = list(work_items[start : start + batch_size])
        label = f"{batch_index}/{total_batches}"
        logger.info(
            "batch started: %d item(s) [%s]",
            len(batch),
            ", ".join(item.sample_id for item in batch),
        )
        with item_context("-", batch=label):
            results = _run_batch(stage, batch, context)
        for item, result in zip(batch, results):
            validated = _validate_path_outputs(result, context)
            document.set_item(item.sample_id, validated.to_dict())
        if batch_index % checkpoint_every == 0 or batch_index == total_batches:
            _write(document, result_path)
        logger.info(
            "batch finished: %d succeeded, %d failed",
            sum(1 for r in results if r.status == STATUS_SUCCEEDED),
            sum(1 for r in results if r.status == STATUS_FAILED),
        )


def _process_stream(results: Iterable[ItemResult], context: StageContext, document: StageResult,
                    work_items: Sequence[WorkItem], result_path: Path, *,
                    checkpoint_every: int) -> None:

    expected = {item.sample_id for item in work_items}
    seen: set[str] = set()
    logger.info("stream started: %d item(s), concurrency=%d", len(work_items), context.concurrency)
    try:
        for result in results:
            if not isinstance(result, ItemResult):
                raise StageError(f"process_stream returned {type(result).__name__}, expected ItemResult")
            if result.sample_id not in expected:
                raise StageError(f"process_stream returned unknown sample {result.sample_id!r}")
            if result.sample_id in seen:
                raise StageError(f"process_stream returned two results for {result.sample_id!r}")
            validated = _validate_path_outputs(result, context)
            seen.add(result.sample_id)
            document.set_item(result.sample_id, validated.to_dict())
            if len(seen) % checkpoint_every == 0 or len(seen) == len(expected):
                _write(document, result_path)
            logger.info("stream item finished: %s status=%s (%d/%d)", result.sample_id,
                        validated.status, len(seen), len(expected))
        missing = sorted(expected - seen)
        if missing:
            raise StageError(f"process_stream did not return results for: {missing}")
    finally:

        close = getattr(results, "close", None)
        if close is not None:
            close()


def _run_batch(
    stage: BatchStage,
    batch: Sequence[WorkItem],
    context: StageContext,
) -> list[ItemResult]:
    raw_results = stage.process_batch(list(batch), context)
    return normalize_results(batch, raw_results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m keyid.core.stage_worker",
        description="Internal: run one KeyID stage in its own process.",
    )
    parser.add_argument(
        "invocation",
        type=Path,
        nargs="?",
        help="redacted audit copy of the invocation (the live payload arrives on stdin)",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="print discovered stage types and exit (does not load models, reads no stdin)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_types:
        registry = StageRegistry()
        registry.discover()
        for entry in registry.describe():
            print(json.dumps(entry, ensure_ascii=False))
        return EXIT_OK

    if args.invocation is None:
        build_parser().error("an invocation path is required unless --list-types is given")

    try:
        payload = read_stdin_payload()
    except Exception:
        print("keyid: cannot read the runtime payload from stdin", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return EXIT_FATAL
    return run_stage(payload)


if __name__ == "__main__":
    raise SystemExit(main())
