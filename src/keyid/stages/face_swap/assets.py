from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from keyid.core.errors import StageError


@dataclass
class StagingReport:
    linked: list[str] = field(default_factory=list)
    copied: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.linked) + len(self.copied) + len(self.present)

    def describe(self) -> str:
        return (
            f"{self.total} file(s): {len(self.present)} already staged, "
            f"{len(self.linked)} hard-linked, {len(self.copied)} copied"
        )


def _same_file(source: Path, target: Path) -> bool:

    if not target.exists():
        return False
    try:
        return target.stat().st_size == source.stat().st_size
    except OSError:
        return False


def stage_assets(source_dir: Path, target_dir: Path) -> StagingReport:

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.is_dir():
        raise StageError(f"face_swap: assets_dir does not exist: {source_dir}")

    report = StagingReport()
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_dir)
        target = target_dir / relative
        if _same_file(source, target):
            report.present.append(str(relative))
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
            report.linked.append(str(relative))
        except OSError:

            shutil.copy2(source, target)
            report.copied.append(str(relative))
    if report.total == 0:
        raise StageError(f"face_swap: assets_dir is empty: {source_dir}")
    return report
