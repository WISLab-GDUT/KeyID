from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .artifacts import SCHEMA_VERSION
from .contract import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SKIPPED_UPSTREAM_FAILED,
    STATUS_SUCCEEDED,
)
from .logging import now_iso

STAGE_SUCCEEDED = "succeeded"
STAGE_PARTIAL = "partial"
STAGE_FATAL = "fatal"
STAGE_EMPTY = "empty"


UNSUCCESSFUL_STATUSES = (STATUS_FAILED, STATUS_SKIPPED, STATUS_SKIPPED_UPSTREAM_FAILED)


@dataclass
class StageResult:
    stage_id: str
    stage_type: str
    status: str = STAGE_EMPTY
    attempt: int = 1
    started_at: str = field(default_factory=now_iso)
    finished_at: str | None = None
    items: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: dict[str, Any] | None = None


    def set_item(self, sample_id: str, payload: Mapping[str, Any]) -> None:
        self.items[sample_id] = dict(payload)

    def clear_items(self) -> None:
        self.items.clear()

    def item_status(self, sample_id: str) -> str | None:
        entry = self.items.get(sample_id)
        return None if entry is None else str(entry.get("status", ""))

    def succeeded_items(self) -> dict[str, dict[str, Any]]:
        return {
            sample_id: entry
            for sample_id, entry in self.items.items()
            if entry.get("status") == STATUS_SUCCEEDED
        }


    def summary(self) -> dict[str, int]:
        counts = {
            "total": len(self.items),
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "reused": 0,
        }
        for entry in self.items.values():
            status = entry.get("status")
            if status == STATUS_SUCCEEDED:
                counts["succeeded"] += 1
                if entry.get("reused"):
                    counts["reused"] += 1
            elif status == STATUS_FAILED:
                counts["failed"] += 1
            elif status in (STATUS_SKIPPED, STATUS_SKIPPED_UPSTREAM_FAILED):
                counts["skipped"] += 1
        return counts

    def derive_status(self) -> str:
        if self.status == STAGE_FATAL:
            return STAGE_FATAL
        counts = self.summary()
        if counts["total"] == 0:
            return STAGE_EMPTY
        if self.has_problems():
            return STAGE_PARTIAL
        return STAGE_SUCCEEDED

    def has_problems(self) -> bool:


        return any(
            entry.get("status") not in (STATUS_SUCCEEDED, STATUS_SKIPPED)
            for entry in self.items.values()
        )


    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": self.stage_id,
            "stage_type": self.stage_type,
            "status": self.status,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary(),
            "items": self.items,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageResult":
        return cls(
            stage_id=str(payload.get("stage", "")),
            stage_type=str(payload.get("stage_type", "")),
            status=str(payload.get("status", STAGE_EMPTY)),
            attempt=int(payload.get("attempt", 1)),
            started_at=str(payload.get("started_at", now_iso())),
            finished_at=payload.get("finished_at"),
            items={str(key): dict(value) for key, value in (payload.get("items") or {}).items()},
            error=payload.get("error"),
        )


    @classmethod
    def start(
        cls,
        stage_id: str,
        stage_type: str,
        *,
        attempt: int,
    ) -> "StageResult":
        return cls(
            stage_id=stage_id,
            stage_type=stage_type,
            status=STAGE_EMPTY,
            attempt=attempt,
            started_at=now_iso(),
        )

    @classmethod
    def load_or_start(
        cls,
        path,
        stage_id: str,
        stage_type: str,
        *,
        attempt: int,
    ) -> "StageResult":

        from .artifacts import read_json

        payload = read_json(path, default=None)
        if isinstance(payload, Mapping) and payload.get("items") is not None:
            document = cls.from_dict(payload)
            document.stage_id = stage_id
            document.stage_type = stage_type
            document.attempt = attempt
            document.status = STAGE_EMPTY
            document.finished_at = None
            document.error = None
            return document
        return cls.start(stage_id, stage_type, attempt=attempt)


def merge_item_entries(
    target: dict[str, dict[str, Any]],
    additions: Iterable[tuple[str, Mapping[str, Any]]],
) -> None:
    for sample_id, payload in additions:
        target[sample_id] = dict(payload)
