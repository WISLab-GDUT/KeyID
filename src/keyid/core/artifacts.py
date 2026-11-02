from __future__ import annotations

import contextlib
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from .errors import ArtifactError

SCHEMA_VERSION = 1


RUN_JSON = "run.json"
CONFIG_SNAPSHOT = "config.snapshot.yaml"
CONFIGS_DIR = "configs"
PIPELINE_LOG = "pipeline.log"
STAGES_DIR = "stages"
RESULT_JSON = "result.json"
ITEMS_DIR = "items"
LOGS_DIR = "logs"


def now_stamp() -> str:

    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def new_run_id() -> str:

    return f"{now_stamp()}_{secrets.token_hex(3)}"


def atomic_write_json(path: Path, data: Any) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise ArtifactError(f"cannot write JSON artifact {path}: {exc}") from exc


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc


@dataclass(frozen=True)
class RunPaths:


    root: Path

    @property
    def run_json(self) -> Path:
        return self.root / RUN_JSON

    @property
    def config_snapshot(self) -> Path:
        return self.root / CONFIG_SNAPSHOT

    @property
    def config_snapshots_dir(self) -> Path:
        return self.root / CONFIGS_DIR

    @property
    def pipeline_log(self) -> Path:
        return self.root / PIPELINE_LOG

    @property
    def stages_dir(self) -> Path:
        return self.root / STAGES_DIR

    def stage_dir(self, stage_id: str) -> Path:
        return self.stages_dir / stage_id

    def stage_logs_dir(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / LOGS_DIR

    def stage_result(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / RESULT_JSON

    def stage_items_dir(self, stage_id: str) -> Path:
        return self.stage_dir(stage_id) / ITEMS_DIR

    def item_dir(self, stage_id: str, sample_id: str) -> Path:
        return self.stage_items_dir(stage_id) / sample_id

    def next_attempt_log(self, stage_id: str) -> Path:

        logs_dir = self.stage_logs_dir(stage_id)
        logs_dir.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (logs_dir / f"attempt-{attempt:03d}.log").exists():
            attempt += 1
        return logs_dir / f"attempt-{attempt:03d}.log"

    def next_config_snapshot(self) -> Path:

        snapshots_dir = self.config_snapshots_dir
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (snapshots_dir / f"attempt-{attempt:03d}.yaml").exists():
            attempt += 1
        return snapshots_dir / f"attempt-{attempt:03d}.yaml"

    def attempt_log(self, stage_id: str, attempt: int) -> Path:
        return self.stage_logs_dir(stage_id) / f"attempt-{attempt:03d}.log"

    def iter_stage_dirs(self) -> Iterator[Path]:
        if not self.stages_dir.is_dir():
            return
        for child in sorted(self.stages_dir.iterdir()):
            if child.is_dir():
                yield child

    def count_attempts(self, stage_id: str) -> int:
        logs_dir = self.stage_logs_dir(stage_id)
        if not logs_dir.is_dir():
            return 0
        return sum(1 for _ in logs_dir.glob("attempt-*.log"))

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "run_json": str(self.run_json),
            "pipeline_log": str(self.pipeline_log),
            "stages_dir": str(self.stages_dir),
        }


def create_run_dir(output_root: Path, run_id: str | None = None) -> RunPaths:

    output_root = Path(output_root)
    runs_dir = output_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = new_run_id()
    root = runs_dir / run_id
    if root.exists():
        raise ArtifactError(f"run directory already exists: {root}")
    root.mkdir(parents=True)
    return RunPaths(root=root)


def open_run_dir(run_dir: Path) -> RunPaths:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ArtifactError(f"run directory does not exist: {run_dir}")
    if not (run_dir / RUN_JSON).exists():
        raise ArtifactError(f"not a KeyID run directory (no {RUN_JSON}): {run_dir}")
    return RunPaths(root=run_dir)


def prepare_work_dir(work_dir: Path, *, create: bool = True) -> tuple[RunPaths, bool]:

    root = Path(work_dir)
    run = RunPaths(root=root)
    if not root.exists():
        if create:
            root.mkdir(parents=True)
        return run, False
    if not root.is_dir():
        raise ArtifactError(f"runtime.work_dir is not a directory: {root}")
    if run.run_json.exists():
        return run, True
    try:
        nonempty = next(root.iterdir(), None) is not None
    except OSError as exc:
        raise ArtifactError(f"cannot inspect runtime.work_dir {root}: {exc}") from exc
    if nonempty:
        raise ArtifactError(
            f"runtime.work_dir is non-empty but is not a KeyID run directory "
            f"(no {RUN_JSON}): {root}"
        )
    return run, False


def artifact_exists(run_dir: Path, relative: str) -> bool:

    candidate = (Path(run_dir) / relative).resolve()
    try:
        candidate.relative_to(Path(run_dir).resolve())
    except ValueError:
        return False
    return candidate.exists()
