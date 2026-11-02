from __future__ import annotations

import contextvars
import logging
import threading
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from .errors import StageError

STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_SKIPPED_UPSTREAM_FAILED = "skipped_upstream_failed"

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class StageContext:


    run_id: str
    run_dir: Path
    stage_id: str
    stage_type: str
    attempt: int
    params: dict[str, Any]
    credentials: dict[str, dict[str, Any]]


    endpoints: dict[str, dict[str, Any]]
    environment: dict[str, str]


    device: str | None
    concurrency: int
    python: Path
    logger: logging.Logger


    needs: tuple[str, ...] = ()
    resuming: bool = False


    dataset_root: Path | None = None
    image_search_roots: tuple[Path, ...] = ()


    device_requested: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


    def resolve_images(self, item: WorkItem) -> list[Path]:

        from .dataset import DatasetItem

        if self.dataset_root is None:
            raise StageError(
                f"{self.stage_id}: the runner did not pass dataset_root, so img_paths "
                "cannot be resolved"
            )
        entry = DatasetItem(
            sample_id=item.sample_id,
            raw=dict(item.raw.get("fields") or {}),
            prompt=str(item.raw.get("prompt") or ""),
            img_paths=tuple(item.raw.get("img_paths") or ()),
            root=Path(self.dataset_root),
            search_roots=tuple(self.image_search_roots),
        )
        return entry.resolved_images()

    def require_single_upstream(self) -> str:

        if len(self.needs) != 1:
            declared = ", ".join(repr(name) for name in self.needs) or "none"
            raise StageError(
                f"{self.stage_id}: this stage reads its inputs from exactly one upstream, "
                f"but the resolved dependencies are {declared}. "
                "Declare a single `needs:` entry for it in the pipeline YAML."
            )
        return self.needs[0]

    def input_path(self, namespace: str, relative: str | Path) -> Path:

        candidate = Path(relative)
        if candidate.is_absolute():
            return candidate
        return Path(self.run_dir) / "stages" / namespace / candidate

    def input_media(self, item: "WorkItem", namespace: str, key: str) -> Path:

        try:
            relative = item.require_input(namespace, key)
        except KeyError as exc:
            raise StageError(str(exc)) from None
        path = self.input_path(namespace, str(relative))
        if not path.exists():
            raise StageError(
                f"{item.sample_id}: upstream {namespace} declared {key}={relative!r} "
                f"but {path} is missing"
            )
        return path

    def resolve_image(self, item: WorkItem, index: int = 0) -> Path:
        paths = self.resolve_images(item)
        if not paths:
            raise StageError(f"{item.sample_id}: dataset item has no img_paths")
        return paths[index if index < len(paths) else 0]

    def resolve_human_image(self, item: "WorkItem", index: int = 0) -> Path:

        from .references import human_reference_indices

        indices = human_reference_indices(item.raw.get("img_paths") or ())
        if not 0 <= index < len(indices):
            raise StageError(
                f"{item.sample_id}: human reference index {index} is out of range "
                f"for {len(indices)} human reference(s)"
            )
        return self.resolve_images(item)[indices[index]]


    @property
    def stage_dir(self) -> Path:
        return Path(self.run_dir) / "stages" / self.stage_id

    @property
    def items_dir(self) -> Path:
        return self.stage_dir / "items"

    def item_dir(self, sample_id: str) -> Path:
        path = self.items_dir / sample_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: str | Path) -> str:

        candidate = Path(path)
        try:
            return str(candidate.resolve().relative_to(self.stage_dir.resolve()))
        except ValueError:
            raise StageError(
                f"{self.stage_id}: output {candidate} is outside the stage directory "
                f"{self.stage_dir}; stage outputs must live under items/<sample_id>/"
            ) from None


    def param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def require_param(self, name: str) -> Any:
        if name not in self.params:
            raise StageError(f"{self.stage_id}: missing required param {name!r}")
        value = self.params[name]
        if value is None:
            raise StageError(f"{self.stage_id}: param {name!r} is null")
        return value


@dataclass
class WorkItem:


    sample_id: str
    raw: dict[str, Any]
    inputs: dict[str, dict[str, Any]]
    item_dir: Path

    def input(self, namespace: str) -> dict[str, Any]:

        return dict(self.inputs.get(namespace, {}))

    def require_input(self, namespace: str, key: str) -> Any:
        namespace_outputs = self.inputs.get(namespace) or {}
        if key not in namespace_outputs:
            raise KeyError(
                f"{self.sample_id}: upstream {namespace!r} has no output {key!r} "
                f"(available: {sorted(namespace_outputs)})"
            )
        return namespace_outputs[key]


@dataclass
class ItemResult:
    sample_id: str
    status: str
    outputs: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    reason: str | None = None

    @classmethod
    def succeeded(cls, sample_id: str, **outputs: Any) -> "ItemResult":
        return cls(sample_id=sample_id, status=STATUS_SUCCEEDED, outputs=dict(outputs))

    @classmethod
    def skipped(cls, sample_id: str, reason: str) -> "ItemResult":

        return cls(sample_id=sample_id, status=STATUS_SKIPPED, reason=reason)

    @classmethod
    def failed(
        cls,
        sample_id: str,
        error_type: str,
        message: str,
        *,
        stage: str | None = None,
    ) -> "ItemResult":
        error: dict[str, Any] = {"type": error_type, "message": message}
        if stage:
            error["where"] = stage
        return cls(sample_id=sample_id, status=STATUS_FAILED, error=error)

    @classmethod
    def from_exception(cls, sample_id: str, exc: BaseException) -> "ItemResult":
        message = str(exc).strip() or exc.__class__.__name__
        return cls.failed(sample_id, exc.__class__.__name__, message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status}
        if self.outputs:
            payload["outputs"] = self.outputs
        if self.error:
            payload["error"] = self.error
        if self.reason:
            payload["reason"] = self.reason
        return payload


def guard(
    item: WorkItem,
    action: Callable[[], ItemResult],
    logger: logging.Logger,
) -> ItemResult:

    from .logging import item_context

    with item_context(item.sample_id):
        try:
            return action()
        except Exception as exc:
            logger.exception("item failed: %s", item.sample_id)
            return ItemResult.failed(
                item.sample_id,
                exc.__class__.__name__,
                str(exc).strip() or exc.__class__.__name__,
            )
        except BaseException:
            logger.exception("item aborted: %s", item.sample_id)
            raise


def require_upstream_output(
    item: "WorkItem",
    namespace: str,
    key: str,
    *,
    what: str,
) -> Any:

    value = (item.inputs.get(namespace) or {}).get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            f"upstream stage {namespace!r} did not provide {key!r} ({what}); "
            f"available: {sorted(item.inputs.get(namespace) or {})}"
        )
    return value


def map_concurrently(
    items: Sequence[WorkItem],
    worker: Callable[[WorkItem], ItemResult],
    *,
    concurrency: int,
    logger: logging.Logger,
) -> list[ItemResult]:

    items = list(items)
    if not items:
        return []
    if concurrency <= 1:
        return [worker(item) for item in items]


    base_context = contextvars.copy_context()
    results: dict[str, ItemResult] = {}
    lock = threading.Lock()

    def run(item: WorkItem) -> None:
        try:
            result = base_context.copy().run(worker, item)
        except Exception as exc:
            logger.exception("worker crashed outside guard: %s", item.sample_id)
            result = ItemResult.from_exception(item.sample_id, exc)
        with lock:
            results[item.sample_id] = result

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="keyid-stage") as pool:
        list(pool.map(run, items))
    return [results[item.sample_id] for item in items]


def format_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


class BatchStage(ABC):


    type: str = ""

    needs: tuple[str, ...] = ()

    description: str = ""

    requires_human_reference: bool = False

    @classmethod
    def skip_reason(cls, raw: Mapping[str, Any]) -> str | None:

        from .references import human_reference_indices

        if cls.requires_human_reference and not human_reference_indices(raw.get("img_paths") or ()):
            return "no_human_reference"
        return None

    @classmethod
    def can_reuse(cls, raw: Mapping[str, Any], outputs: Mapping[str, Any],
                  params: Mapping[str, Any]) -> bool:

        return True


    def setup(self, context: StageContext) -> None:
        pass

    @abstractmethod
    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        pass

    def process_stream(self, items: list[WorkItem],
                       context: StageContext) -> Iterable[ItemResult] | None:

        return None

    def close(self, context: StageContext) -> None:
        pass


    @classmethod
    def default_needs(cls) -> tuple[str, ...]:
        return tuple(cls.needs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} type={self.type!r}>"


def normalize_results(items: Iterable[WorkItem], results: Iterable[ItemResult]) -> list[ItemResult]:

    expected = [item.sample_id for item in items]
    seen: dict[str, ItemResult] = {}
    for result in results:
        if not isinstance(result, ItemResult):
            raise StageError(f"process_batch returned {type(result).__name__}, expected ItemResult")
        if result.sample_id in seen:
            raise StageError(f"process_batch returned two results for {result.sample_id!r}")
        seen[result.sample_id] = result

    missing = [sample_id for sample_id in expected if sample_id not in seen]
    if missing:
        raise StageError(f"process_batch did not return results for: {missing}")
    return [seen[sample_id] for sample_id in expected]
