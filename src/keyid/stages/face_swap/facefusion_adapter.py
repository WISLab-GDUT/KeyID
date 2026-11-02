from __future__ import annotations

import io
import math
import os
import time
from contextlib import contextmanager, redirect_stderr
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

from keyid.core.errors import StageError


def _normalize_extra_args(extra_args: Iterable[str] | None) -> list[str]:
    return [str(item) for item in (extra_args or [])]


@contextmanager
def _facefusion_directory(root: Path | None):

    if root is None:
        yield
        return
    previous = Path.cwd()
    try:
        os.chdir(root)
        yield
    finally:
        os.chdir(previous)


def _in_facefusion_directory(method):
    @wraps(method)
    def scoped(self, *args, **kwargs):
        with _facefusion_directory(self.facefusion_root):
            return method(self, *args, **kwargs)

    return scoped


def _runtime_path_args(extra_args: list[str], work_dir: Path) -> list[str]:

    result = list(extra_args)
    defaults = {
        "--config-path": Path.cwd() / "facefusion.ini",
        "--temp-path": work_dir / "tmp",
        "--jobs-path": work_dir / "jobs",
    }
    for flag, default in defaults.items():
        seen = False
        for index, value in enumerate(result):
            if value == flag:
                seen = True
                if index + 1 < len(result) and not result[index + 1].startswith("--"):
                    result[index + 1] = str(Path(result[index + 1]).resolve())
            elif value.startswith(flag + "="):
                seen = True
                result[index] = flag + "=" + str(Path(value.split("=", 1)[1]).resolve())
        if not seen:
            result.extend([flag, str(default)])
    return result


def _normalize_processors(processors: Iterable[str] | None) -> list[str]:
    normalized = [str(item) for item in (processors or [])]
    if not normalized:
        raise StageError("face_swap: at least one processor must be configured")
    return normalized


def available_execution_providers() -> list[str]:
    from facefusion.execution import get_available_execution_providers

    return list(get_available_execution_providers())


def resolve_execution_providers(
    requested: Sequence[str],
    *,
    allow_cpu_fallback: bool,
) -> list[str]:

    if not requested:
        raise StageError("face_swap: execution_providers must list at least one provider")

    available = available_execution_providers()
    missing = [provider for provider in requested if provider not in available]
    if missing and not allow_cpu_fallback:
        raise StageError(
            f"face_swap: execution providers {missing} are unavailable "
            f"(available: {available}). Install the matching runtime, choose an available "
            "provider in YAML, or set allow_cpu_fallback: true to accept CPU inference."
        )

    selected = [provider for provider in requested if provider in available]
    if not selected:
        if not allow_cpu_fallback:
            raise StageError(
                f"face_swap: none of {list(requested)} are available (available: {available})"
            )


        selected = ["cpu"]
    if "cpu" not in selected and "cpu" in available:

        selected.append("cpu")
    return selected


def _parse_facefusion_args(argv: list[str]) -> dict:

    import sys

    from facefusion.program import create_program

    original_argv = sys.argv[:]
    diagnostics = io.StringIO()
    try:


        sys.argv = [original_argv[0], *argv]
        with redirect_stderr(diagnostics):
            program = create_program()
            return vars(program.parse_args(argv))
    except SystemExit as exc:
        lines = diagnostics.getvalue().strip().splitlines()
        detail = lines[-1] if lines else "the parser exited without an error message"
        raise StageError(
            f"facefusion argument parsing failed (exit={exc.code}, cwd={Path.cwd()}): {detail}"
        ) from exc
    finally:
        sys.argv = original_argv


def build_processing_args(
    source_path: str | None,
    target_path: str,
    output_path: str,
    processors: Sequence[str],
    execution_providers: Sequence[str],
    extra_args: Iterable[str] | None = None,
) -> dict:
    argv = [
        "headless-run",
        "--execution-providers",
        *execution_providers,
        "--processors",
        *_normalize_processors(processors),
        "-t",
        target_path,
        "-o",
        output_path,
    ]
    if source_path:
        argv.extend(["-s", source_path])
    argv.extend(_normalize_extra_args(extra_args))
    return _parse_facefusion_args(argv)


def _get_processor_modules(processors: Iterable[str] | None) -> list[ModuleType]:
    from facefusion.processors.core import get_processors_modules

    return list(get_processors_modules(_normalize_processors(processors)))


def _post_process_modules(processor_modules: Iterable[ModuleType]) -> None:
    for processor_module in processor_modules:
        processor_module.post_process()


VIDEO_MEMORY_STRATEGIES = ("strict", "moderate", "tolerant")
DEFAULT_VIDEO_MEMORY_STRATEGY = "tolerant"

STRATEGY_FLAG = "--video-memory-strategy"


def validate_video_memory_strategy(value: object) -> str:
    strategy = str(value).strip().lower()
    if strategy not in VIDEO_MEMORY_STRATEGIES:
        raise StageError(
            f"video_memory_strategy must be one of {list(VIDEO_MEMORY_STRATEGIES)}, "
            f"got {value!r}"
        )
    return strategy


def reject_strategy_in_extra_args(extra_args: Sequence[str]) -> None:

    for index, argument in enumerate(extra_args):
        if argument == STRATEGY_FLAG:
            raise StageError(
                f"extra_args contains {STRATEGY_FLAG}; configure model residency through "
                "the dedicated `video_memory_strategy` parameter instead "
                f"(allowed: {list(VIDEO_MEMORY_STRATEGIES)})"
            )
        if argument.startswith(f"{STRATEGY_FLAG}="):
            raise StageError(
                f"extra_args contains {STRATEGY_FLAG}=...; configure model residency through "
                "the dedicated `video_memory_strategy` parameter instead "
                f"(allowed: {list(VIDEO_MEMORY_STRATEGIES)})"
            )


class SampleRejected(Exception):
    pass


class FaceSwapPipeline:


    def __init__(
        self,
        *,
        processors: Sequence[str],
        execution_providers: Sequence[str],
        extra_args: Iterable[str] | None = None,
        video_memory_strategy: str = DEFAULT_VIDEO_MEMORY_STRATEGY,
        facefusion_root: str | Path | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        self.processors = _normalize_processors(processors)
        self.execution_providers = list(execution_providers)
        self.extra_args = _normalize_extra_args(extra_args)
        self.facefusion_root = Path(facefusion_root).resolve() if facefusion_root else None
        if work_dir is not None:
            self.extra_args = _runtime_path_args(self.extra_args, Path(work_dir).resolve())
        self.video_memory_strategy = validate_video_memory_strategy(video_memory_strategy)
        reject_strategy_in_extra_args(self.extra_args)
        self.environment_ready = False
        self.requested_output_video_fps = None


    @_in_facefusion_directory
    def prepare_environment(self) -> None:

        from facefusion import core, logger, state_manager

        self.processor_modules = _get_processor_modules(self.processors)


        args = build_processing_args(
            source_path=__file__,
            target_path=__file__,
            output_path=str(Path(__file__).resolve().parent),
            processors=self.processors,
            execution_providers=self.execution_providers,
            extra_args=self.extra_args,
        )
        from facefusion.args import apply_args

        self.requested_output_video_fps = args.get("output_video_fps")
        apply_args(args, state_manager.init_item)


        state_manager.set_item("video_memory_strategy", self.video_memory_strategy)
        logger.init(state_manager.get_item("log_level"))

        if not core.pre_check():
            raise StageError("facefusion pre_check failed: needs Python >= 3.10, curl and ffmpeg")
        if not core.common_pre_check():
            raise StageError("facefusion common_pre_check failed")
        for processor_module in self.processor_modules:
            if not processor_module.pre_check():
                raise StageError(f"{processor_module.__name__}: pre_check failed")

        self.environment_ready = True

    @_in_facefusion_directory
    def run(self, *, source_path: str, target_path: str, output_path: str) -> None:

        if not self.environment_ready:
            raise StageError("facefusion: run() called before prepare_environment()")

        from facefusion import state_manager
        from facefusion.workflows import image_to_video

        self._reject_bad_sample(source_path=source_path, target_path=target_path)

        state_manager.set_item("source_paths", [source_path])
        state_manager.set_item("target_path", target_path)
        state_manager.set_item("output_path", output_path)


        state_manager.set_item("output_video_fps", self._output_video_fps(target_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)


        for processor_module in self.processor_modules:
            if not processor_module.pre_process("output"):
                raise SampleRejected(
                    f"{processor_module.__name__} rejected this sample "
                    "(no face in the reference image, or the draft video is unusable)"
                )

        error_code = image_to_video.process(time.time())
        if error_code:
            raise RuntimeError(f"facefusion image_to_video failed with code {error_code}")
        if not Path(output_path).exists():
            raise RuntimeError(f"facefusion reported success but wrote no file at {output_path}")


    @_in_facefusion_directory
    def close(self) -> None:

        modules = getattr(self, "processor_modules", None)
        if modules:
            from facefusion import state_manager

            previous = state_manager.get_item("video_memory_strategy")
            try:
                state_manager.set_item("video_memory_strategy", "strict")
                _post_process_modules(modules)
            finally:
                state_manager.set_item("video_memory_strategy", previous)
        self.environment_ready = False


    def _output_video_fps(self, target_path: str) -> float:
        from facefusion.normalizer import normalize_fps
        from facefusion.vision import detect_video_fps

        fps = normalize_fps(self.requested_output_video_fps) or detect_video_fps(target_path)
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise SampleRejected(f"cannot determine a valid output FPS for draft video: {target_path}")
        return float(fps)

    @staticmethod
    def _reject_bad_sample(*, source_path: str, target_path: str) -> None:

        from facefusion.filesystem import is_image, is_video

        if not is_image(source_path):
            raise SampleRejected(f"reference image is missing or unreadable: {source_path}")
        if not is_video(target_path):
            raise SampleRejected(f"draft video is missing or unreadable: {target_path}")
