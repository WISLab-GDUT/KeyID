from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import graph as graph_module
from .artifacts import (
    SCHEMA_VERSION,
    RunPaths,
    atomic_write_json,
    create_run_dir,
    open_run_dir,
    prepare_work_dir,
    read_json,
)
from .config import PipelineConfig, iter_secrets, redact_secrets
from .contract import STATUS_SKIPPED, STATUS_SKIPPED_UPSTREAM_FAILED, STATUS_SUCCEEDED
from .dataset import Dataset, DatasetItem, load_dataset
from .errors import EXIT_FATAL, EXIT_OK, EXIT_PARTIAL, ArtifactError, ConfigError
from .devices import visible_devices_for
from .logging import (
    EXTERNAL_STDERR,
    EXTERNAL_STDOUT,
    ensure_configured,
    info,
    now_iso,
    register_secret,
    scrub_text,
    stage_context,
)
from .registry import StageRegistry, build_default_registry, package_parent_dirs
from .results import STAGE_EMPTY, STAGE_FATAL, STAGE_SUCCEEDED, StageResult

logger = logging.getLogger("keyid.runner")

STAGE_WORKER_MODULE = "keyid.core.stage_worker"


@dataclass
class RunOutcome:
    run_paths: RunPaths
    exit_code: int
    stage_statuses: dict[str, str]
    fatal_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == EXIT_OK


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _stage_dir_of(run: RunPaths, stage_id: str) -> Path:
    return run.stage_dir(stage_id)


def _output_path(run: RunPaths, stage_id: str, relative: str) -> Path:

    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    return _stage_dir_of(run, stage_id) / candidate


def _iter_output_paths(outputs: Mapping[str, Any]) -> Iterable[str]:
    for value in outputs.values():
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield item


def _outputs_exist(run: RunPaths, stage_id: str, outputs: Mapping[str, Any]) -> bool:

    from .stage_worker import _looks_like_path_output

    for key, value in outputs.items():
        if not _looks_like_path_output(key):
            continue
        for relative in _iter_output_paths({key: value}):
            path = _output_path(run, stage_id, relative)
            if not path.exists():
                return False
    return True


def reconcile_exit_code(return_code: int, document: Mapping[str, Any] | None) -> tuple[str, str | None]:

    if return_code not in (EXIT_OK, EXIT_PARTIAL):
        if return_code < 0:
            return STAGE_FATAL, f"stage process was killed by signal {-return_code}"
        return STAGE_FATAL, f"stage process exited with unexpected status {return_code}"

    if not isinstance(document, Mapping):
        return STAGE_FATAL, "stage process exited without writing a readable result.json"

    status = str(document.get("status", ""))
    if status == STAGE_FATAL:
        return STAGE_FATAL, str((document.get("error") or {}).get("message") or "stage reported fatal")

    if document.get("finished_at") is None:
        return STAGE_FATAL, "result.json is not in a terminal state (finished_at is null)"

    summary = document.get("summary") or {}
    try:
        failed = int(summary.get("failed") or 0)
        skipped = int(summary.get("skipped") or 0)
    except (TypeError, ValueError):
        return STAGE_FATAL, "result.json has a malformed summary"

    items = document.get("items")
    if items is not None and not isinstance(items, Mapping):
        return STAGE_FATAL, "result.json has a malformed items mapping"
    if isinstance(items, Mapping):
        reported_partial = bool(failed or StageResult.from_dict(document).has_problems())
    else:

        reported_partial = bool(failed or skipped)
    if reported_partial != (return_code == EXIT_PARTIAL):
        return STAGE_FATAL, (
            f"exit status {return_code} disagrees with result.json "
            f"(failed={failed}, skipped={skipped})"
        )
    return status, None


class Runner:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        registry: StageRegistry | None = None,
        dataset: Dataset | None = None,
    ) -> None:
        self.config = config
        self.registry = registry or build_default_registry(config.runtime.stage_packages)
        self.dataset = dataset


    def load_dataset(self) -> Dataset:
        if self.dataset is not None:
            return self.dataset
        dataset_node = self.config.raw.get("dataset") or {}
        raw_search_roots = dataset_node.get("image_search_roots") or []
        search_roots = tuple(
            Path(entry) if Path(entry).is_absolute() else self.config.base_dir / entry
            for entry in raw_search_roots
        )
        self.dataset = load_dataset(
            self.config.dataset.metadata,
            self.config.dataset.root,
            format=self.config.dataset.format,
            search_roots=search_roots,
        )
        return self.dataset

    def build_plan(
        self,
        *,
        target: str | None = None,
        only: str | None = None,
    ) -> graph_module.ExecutionPlan:
        default_needs = {
            stage_type: self.registry.get(stage_type).default_needs()
            for stage_type in (
                stage.type for stage in self.config.stages.values()
            )
            if stage_type in self.registry
        }
        return graph_module.build_plan(
            self.config.stages,
            default_needs=default_needs,
            target=target,
            only=only,
        )

    def plan_payload(
        self,
        *,
        target: str | None = None,
        only: str | None = None,
    ) -> dict[str, Any]:
        plan = self.build_plan(target=target, only=only)
        return {
            "scope": plan.scope_reason,
            "order": plan.describe(),
        }


    def collect_inputs(
        self,
        stage_id: str,
        needs: Sequence[str],
        run: RunPaths,
        sample_id: str,
    ) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:

        inputs: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        skipped: list[str] = []
        for dependency in needs:
            payload = read_json(run.stage_result(dependency), default=None)
            entry = ((payload or {}).get("items") or {}).get(sample_id) or {}
            if entry.get("status") == STATUS_SUCCEEDED:
                inputs[dependency] = dict(entry.get("outputs") or {})
            elif entry.get("status") == STATUS_SKIPPED:
                skipped.append(dependency)
            else:
                failed.append(dependency)
        return inputs, failed, skipped


    def run(
        self,
        *,
        target: str | None = None,
        only: str | None = None,
        resume: str | Path | None = None,
        force_stages: Sequence[str] = (),
        dry_run: bool = False,
        run_id: str | None = None,
        console: bool = True,
    ) -> RunOutcome:
        plan = self.build_plan(target=target, only=only)

        unknown_forced = [sid for sid in force_stages if sid not in plan.nodes]
        if unknown_forced:
            raise ConfigError(f"--force-stage names unknown stages: {unknown_forced}")

        if resume is not None:
            run = open_run_dir(Path(resume))
            resuming = True
            directory_source = "cli_resume"
        elif self.config.runtime.work_dir is not None:
            run, resuming = prepare_work_dir(
                self.config.runtime.work_dir, create=not dry_run
            )
            directory_source = "config_work_dir"
        elif dry_run:

            planned_id = run_id or self.config.runtime.run_id or "(generated)"
            run = RunPaths(root=self.config.runtime.output_root / "runs" / planned_id)
            resuming = False
            directory_source = "new_run"
        else:
            run = create_run_dir(self.config.runtime.output_root, run_id or self.config.runtime.run_id)
            resuming = False
            directory_source = "new_run"

        forced = set(force_stages)

        dataset = self.load_dataset()
        scope_items = list(dataset)

        self._validate_dependencies(plan, run, resuming, only)

        if dry_run:
            for stage_id in plan.order:
                stage_config = self.config.stage(stage_id)
                if not Path(stage_config.python).exists():
                    raise ConfigError(
                        f"stage {stage_id}: interpreter not found: {stage_config.python}"
                    )
            return RunOutcome(run_paths=run, exit_code=EXIT_OK, stage_statuses={})

        secrets = list(iter_secrets(self.config))
        for secret in secrets:
            register_secret(secret)

        self._write_run_metadata(run, plan, dataset, resume, directory_source)


        ensure_configured(
            self.config.runtime.log_level,
            console=True,
            files=[run.pipeline_log],
            run_id=run.root.name,
        )
        info(
            logger,
            "run %s started: %d stage(s) in scope (directory=%s)",
            run.root.name,
            len(plan.order),
            directory_source,
        )

        stage_statuses: dict[str, str] = {}
        for stage_id in plan.order:
            stage_config = self.config.stage(stage_id)
            with stage_context(stage_id):
                status = self._run_stage(
                    run, plan, stage_id, scope_items, forced=forced, resuming=resuming
                )
            stage_statuses[stage_id] = status
            self._update_run_json(run, stage_statuses, exit_code=None)
            if status == STAGE_FATAL:
                info(logger, "stage %s failed fatally; stopping the pipeline", stage_id)
                self._update_run_json(run, stage_statuses, exit_code=EXIT_FATAL)
                return RunOutcome(
                    run_paths=run,
                    exit_code=EXIT_FATAL,
                    stage_statuses=stage_statuses,
                    fatal_error=f"stage {stage_id} failed fatally",
                )
            if not self.config.runtime.continue_on_item_error and status not in (
                STAGE_SUCCEEDED,
                STAGE_EMPTY,
            ):


                info(
                    logger,
                    "stage %s ended %s and continue_on_item_error is false; stopping here",
                    stage_id,
                    status,
                )
                exit_code = EXIT_PARTIAL
                self._update_run_json(run, stage_statuses, exit_code=exit_code)
                return RunOutcome(run_paths=run, exit_code=exit_code, stage_statuses=stage_statuses)

        exit_code = self._aggregate_exit_code(run, plan.order, expected_items=len(scope_items))
        self._update_run_json(run, stage_statuses, exit_code=exit_code)
        return RunOutcome(run_paths=run, exit_code=exit_code, stage_statuses=stage_statuses)


    def _validate_dependencies(
        self,
        plan: graph_module.ExecutionPlan,
        run: RunPaths,
        resuming: bool,
        only: str | None,
    ) -> None:

        if only is not None and not resuming:
            needed = plan.nodes[only].needs
            if needed and not all(run.stage_result(dep).exists() for dep in needed):
                raise ConfigError(
                    f"--only {only} needs inputs from {list(needed)}; use --resume <run_dir> "
                    "or a populated runtime.work_dir so KeyID can resolve them"
                )

        outside = [
            dependency
            for stage_id in plan.order
            for dependency in plan.nodes[stage_id].needs
            if dependency not in plan.order
        ]
        for dependency in sorted(set(outside)):
            if not run.stage_result(dependency).exists():
                raise ConfigError(
                    f"stage {dependency!r} is outside this run's scope and has no results in "
                    f"{run.root} -- enable it, or resume a run that contains it"
                )

    def _write_run_metadata(
        self,
        run: RunPaths,
        plan: graph_module.ExecutionPlan,
        dataset: Dataset,
        resume: str | Path | None,
        directory_source: str,
    ) -> None:
        existing = read_json(run.run_json, default=None) or {}
        snapshot = self.config.redacted_snapshot()
        if not run.config_snapshot.exists():
            _write_yaml(run.config_snapshot, snapshot)

        attempt_snapshot = run.next_config_snapshot()
        _write_yaml(attempt_snapshot, snapshot)
        relative_snapshot = str(attempt_snapshot.relative_to(run.root))
        snapshot_history = existing.get("config_snapshots") or []
        if not isinstance(snapshot_history, list):
            snapshot_history = []
        snapshot_history = [*snapshot_history, relative_snapshot]

        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run.root.name,
            "created_at": existing.get("created_at", now_iso()),
            "updated_at": now_iso(),
            "execution": int(existing.get("execution", 0) or 0) + 1,
            "config_path": str(self.config.path),
            "config_snapshot": relative_snapshot,
            "config_snapshots": snapshot_history,
            "directory_source": directory_source,
            "resumed_from": str(resume) if resume is not None else existing.get("resumed_from"),
            "dataset": {
                "format": self.config.dataset.format,
                "root": str(self.config.dataset.root),
                "metadata": str(self.config.dataset.metadata),
                "samples": len(dataset),
            },
            "scope": {"reason": plan.scope_reason, "order": plan.describe()},
            "stages": existing.get("stages", {}),
            "exit_code": None,
        }
        atomic_write_json(run.run_json, payload)

    def _update_run_json(
        self,
        run: RunPaths,
        stage_statuses: Mapping[str, str],
        *,
        exit_code: int | None,
    ) -> None:
        payload = read_json(run.run_json, default={}) or {}
        stages: dict[str, Any] = dict(payload.get("stages") or {})
        for stage_id, status in stage_statuses.items():
            result_payload = read_json(run.stage_result(stage_id), default=None) or {}
            entry = dict(stages.get(stage_id) or {})
            entry.update(
                {
                    "type": self.config.stages[stage_id].type,
                    "status": status,
                    "attempt": result_payload.get("attempt"),
                    "attempts": run.count_attempts(stage_id),
                    "result": str(run.stage_result(stage_id).relative_to(run.root)),
                    "summary": result_payload.get("summary"),
                    "finished_at": result_payload.get("finished_at"),
                }
            )
            stages[stage_id] = entry
        payload["stages"] = stages
        payload["updated_at"] = now_iso()
        if exit_code is not None:
            payload["exit_code"] = exit_code
        atomic_write_json(run.run_json, payload)

    def _run_stage(
        self,
        run: RunPaths,
        plan: graph_module.ExecutionPlan,
        stage_id: str,
        scope_items: Sequence[DatasetItem],
        *,
        forced: set[str],
        resuming: bool,
    ) -> str:
        stage_config = self.config.stage(stage_id)
        needs = plan.nodes[stage_id].needs
        previous = read_json(run.stage_result(stage_id), default=None)
        stage_cls = self.registry.get(stage_config.type)

        pending: list[dict[str, Any]] = []
        preseeded: dict[str, dict[str, Any]] = {}
        reused = skipped = 0

        for item in scope_items:
            raw = item.to_public_dict()
            reason = stage_cls.skip_reason(raw)
            if reason is not None:
                preseeded[item.sample_id] = {"status": STATUS_SKIPPED, "reason": reason}
                skipped += 1
                info(logger, "item %s skipped normally: %s", item.sample_id, reason)
                continue


            inputs, failed_upstream, skipped_upstream = self.collect_inputs(
                stage_id, needs, run, item.sample_id
            )
            if failed_upstream:
                preseeded[item.sample_id] = {
                    "status": STATUS_SKIPPED_UPSTREAM_FAILED,
                    "reason": "upstream_failed",
                    "upstream_failed": sorted(failed_upstream),
                }
                skipped += 1
                continue
            if skipped_upstream:
                preseeded[item.sample_id] = {
                    "status": STATUS_SKIPPED,
                    "reason": "upstream_skipped",
                    "upstream_skipped": sorted(skipped_upstream),
                }
                skipped += 1
                continue

            previous_entry = ((previous or {}).get("items") or {}).get(item.sample_id) or {}
            previous_outputs = previous_entry.get("outputs") or {}
            if (
                stage_id not in forced
                and previous_entry.get("status") == STATUS_SUCCEEDED
                and _outputs_exist(run, stage_id, previous_outputs)
                and stage_cls.can_reuse(raw, previous_outputs, stage_config.params)
            ):
                preseeded[item.sample_id] = {**previous_entry, "reused": True}
                reused += 1
                continue

            pending.append(
                {
                    "sample_id": item.sample_id,
                    "raw": item.to_public_dict(),
                    "inputs": inputs,
                }
            )

        info(
            logger,
            "stage %s: %d to process, %d reused, %d skipped (needs=%s)",
            stage_id,
            len(pending),
            reused,
            skipped,
            list(needs),
        )

        if not pending:
            document = StageResult.load_or_start(
                run.stage_result(stage_id),
                stage_id,
                stage_config.type,
                attempt=max(1, run.count_attempts(stage_id)),
            )
            document.clear_items()
            for sample_id, entry in preseeded.items():
                document.set_item(sample_id, entry)
            document.status = document.derive_status()
            document.finished_at = now_iso()
            atomic_write_json(run.stage_result(stage_id), document.to_dict())
            info(logger, "stage %s: nothing to do (all items reused or skipped)", stage_id)
            return document.status

        attempt = run.count_attempts(stage_id) + 1
        attempt_log = run.attempt_log(stage_id, attempt)
        invocation_path = run.stage_dir(stage_id) / "invocation.json"

        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run.root.name,
            "run_dir": str(run.root),
            "stage_id": stage_id,
            "stage_type": stage_config.type,
            "attempt": attempt,


            "needs": list(needs),
            "attempt_log": str(attempt_log),
            "log_level": self.config.runtime.log_level,
            "device": stage_config.device,
            "concurrency": stage_config.concurrency,
            "python": str(stage_config.python),
            "batch": {
                "size": stage_config.batch.size,
                "checkpoint_every": stage_config.batch.checkpoint_every,
            },
            "params": stage_config.params,
            "credentials": self.config.credentials,


            "endpoints": self.config.endpoints,
            "environment": self.config.subprocess_environment(stage_config),
            "resuming": resuming,
            "dataset": {
                "root": str(self.config.dataset.root),
                "metadata": str(self.config.dataset.metadata),
                "image_search_roots": [str(path) for path in self.load_dataset().search_roots_of()],
            },
            "stage_packages": list(self.config.runtime.stage_packages),
            "items": pending,
            "preseeded_items": preseeded,
        }


        atomic_write_json(invocation_path, self._audit_invocation(payload))

        attempt_log.parent.mkdir(parents=True, exist_ok=True)
        _append_line(attempt_log, f"# attempt {attempt} of stage {stage_id} started {now_iso()}")

        return_code = self._spawn(run, stage_id, stage_config, payload, attempt_log, attempt)

        document_payload = read_json(run.stage_result(stage_id), default=None)
        if not isinstance(document_payload, dict):
            document_payload = None

        status, reason = reconcile_exit_code(return_code, document_payload)
        summary = (document_payload or {}).get("summary") or {}
        info(
            logger,
            "stage %s finished: status=%s succeeded=%s failed=%s skipped=%s reused=%s (exit=%d)",
            stage_id,
            status,
            summary.get("succeeded"),
            summary.get("failed"),
            summary.get("skipped"),
            summary.get("reused"),
            return_code,
        )

        if status == STAGE_FATAL and (document_payload or {}).get("status") != STAGE_FATAL:


            info(logger, "stage %s: %s", stage_id, reason)
            document = StageResult.start(
                stage_id,
                stage_config.type,
                attempt=attempt,
            )
            document.status = STAGE_FATAL
            document.error = {
                "type": "StageProcessError",
                "message": reason or "stage process failed",
                "checkpointed_items_discarded": len((document_payload or {}).get("items") or {}),
            }
            document.finished_at = now_iso()
            atomic_write_json(run.stage_result(stage_id), document.to_dict())
            return STAGE_FATAL
        return status

    def _audit_invocation(self, payload: Mapping[str, Any]) -> dict[str, Any]:

        audit = redact_secrets(dict(payload))
        serialised = json_dumps(audit)
        scrubbed = scrub_text(serialised)
        if scrubbed == serialised:
            return audit
        return json.loads(scrubbed)

    def _stage_environment(self, stage_id: str, stage_config) -> dict[str, str]:

        env = dict(os.environ)
        env.update(self.config.subprocess_environment(stage_config))
        import_paths = [_src_root(), *package_parent_dirs(self.config.runtime.stage_packages)]
        env["PYTHONPATH"] = _prepend_pythonpath(env.get("PYTHONPATH"), import_paths)
        env["PYTHONUNBUFFERED"] = "1"
        env["KEYID_RUN_ID"] = env.get("KEYID_RUN_ID", "")
        env["KEYID_STAGE_ID"] = stage_id

        visible = visible_devices_for(stage_config.device)
        if visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = visible
        elif "cpu" == str(stage_config.device or "").strip().lower():
            env.pop("CUDA_VISIBLE_DEVICES", None)
        return env

    def _spawn(
        self,
        run: RunPaths,
        stage_id: str,
        stage_config,
        payload: Mapping[str, Any],
        attempt_log: Path,
        attempt: int,
    ) -> int:
        python = str(stage_config.python)
        if not Path(python).exists():
            raise ConfigError(f"stage {stage_id}: interpreter not found: {python}")

        env = self._stage_environment(stage_id, stage_config)
        env["KEYID_RUN_ID"] = run.root.name
        audit_path = run.stage_dir(stage_id) / "invocation.json"


        command = [python, "-m", STAGE_WORKER_MODULE, str(audit_path)]
        info(logger, "launching stage process: %s", " ".join(command))
        info(logger, "interpreter: %s", python)
        info(logger, "device: %s (CUDA_VISIBLE_DEVICES=%s)", stage_config.device or "default",
             env.get("CUDA_VISIBLE_DEVICES", "<inherited>"))

        pipeline_log = run.pipeline_log
        pipeline_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(run.root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
            )
        except OSError as exc:
            raise ArtifactError(f"cannot start stage process for {stage_id}: {exc}") from exc

        pumps = [


            _LogTee(process.stdout, pipeline_log, EXTERNAL_STDOUT, stream_out=sys.stderr),
            _LogTee(process.stderr, pipeline_log, EXTERNAL_STDERR, stream_out=sys.stderr),
            _PayloadWriter(process.stdin, payload),
        ]
        for pump in pumps:
            pump.start()
        return_code = process.wait()
        for pump in pumps:
            pump.join(timeout=30)

        with open(attempt_log, "a", encoding="utf-8") as handle:
            handle.write(f"# attempt {attempt} of stage {stage_id} exited {return_code} at {now_iso()}\n")
        return return_code

    def _aggregate_exit_code(
        self,
        run: RunPaths,
        order: Sequence[str],
        *,
        expected_items: int,
    ) -> int:

        problems = False
        for stage_id in order:
            payload = read_json(run.stage_result(stage_id), default=None)
            if not isinstance(payload, dict):
                return EXIT_FATAL
            status = str(payload.get("status", ""))
            if status == STAGE_FATAL:
                return EXIT_FATAL
            summary = payload.get("summary") or {}
            try:
                total = int(summary.get("total") or 0)
                failed = int(summary.get("failed") or 0)
                skipped = int(summary.get("skipped") or 0)
            except (TypeError, ValueError):
                return EXIT_FATAL
            if total == 0 and expected_items > 0:


                return EXIT_FATAL
            if failed or StageResult.from_dict(payload).has_problems():
                problems = True
        return EXIT_PARTIAL if problems else EXIT_OK


class _LogTee(threading.Thread):


    def __init__(self, pipe, log_path: Path, origin: str, *, stream_out=None) -> None:
        super().__init__(daemon=True, name=f"keyid-tee-{origin}")
        self._pipe = pipe
        self._log_path = log_path
        self._origin = origin
        self._stream_out = stream_out

    def run(self) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as log_file:
                for line in iter(self._pipe.readline, ""):


                    line = scrub_text(line)
                    if self._stream_out is not None:
                        self._stream_out.write(line)
                        self._stream_out.flush()
                    log_file.write(line)
                    log_file.flush()
        except Exception:
            pass
        finally:
            try:
                self._pipe.close()
            except Exception:
                pass


class _PayloadWriter(threading.Thread):


    def __init__(self, pipe, payload: Mapping[str, Any]) -> None:
        super().__init__(daemon=True, name="keyid-payload-writer")
        self._pipe = pipe
        self._payload = payload
        self.error: str | None = None

    def run(self) -> None:
        try:
            self._pipe.write(json_dumps(self._payload))
            self._pipe.flush()
        except (BrokenPipeError, OSError) as exc:


            self.error = str(exc)
        finally:
            try:
                self._pipe.close()
            except Exception:
                pass


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def json_dumps(payload: Any) -> str:

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepend_pythonpath(existing: str | None, additions: Sequence[Path]) -> str:
    parts = [str(path) for path in additions]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:

    run = open_run_dir(Path(run_dir))
    run_payload = read_json(run.run_json, default={}) or {}
    stages: dict[str, Any] = {}
    totals = {"total": 0, "succeeded": 0, "failed": 0, "skipped": 0, "reused": 0}

    for stage_dir in run.iter_stage_dirs():
        stage_id = stage_dir.name
        payload = read_json(stage_dir / "result.json", default=None)
        if payload is None:
            stages[stage_id] = {"status": "not_started"}
            continue
        summary = payload.get("summary") or {}
        stages[stage_id] = {
            "type": payload.get("stage_type"),
            "status": payload.get("status"),
            "attempt": payload.get("attempt"),
            "summary": summary,
            "finished_at": payload.get("finished_at"),
        }
        for key in totals:
            totals[key] += int(summary.get(key) or 0)

    return {
        "run_id": run_payload.get("run_id", run.root.name),
        "run_dir": str(run.root),
        "created_at": run_payload.get("created_at"),
        "updated_at": run_payload.get("updated_at"),
        "exit_code": run_payload.get("exit_code"),
        "dataset": run_payload.get("dataset"),
        "scope": run_payload.get("scope"),
        "stages": stages,
        "totals": totals,
    }
