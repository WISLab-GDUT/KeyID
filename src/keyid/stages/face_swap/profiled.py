from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from keyid.core.artifacts import atomic_write_json, read_json
from keyid.core.contract import ItemResult, StageContext, WorkItem, guard
from keyid.core.errors import StageError

from .facefusion_adapter import (
    DEFAULT_VIDEO_MEMORY_STRATEGY,
    FaceSwapPipeline,
    SampleRejected,
    reject_strategy_in_extra_args,
    resolve_execution_providers,
    validate_video_memory_strategy,
)
from .stage import FaceSwapStage

SELECTOR_FLAGS = (
    "--face-selector-mode",
    "--reference-face-distance",
    "--reference-face-position",
)


def reject_selector_in_extra_args(extra_args: Sequence[str]) -> None:

    for argument in extra_args:
        for flag in SELECTOR_FLAGS:
            if argument == flag or argument.startswith(f"{flag}="):
                raise StageError(
                    f"profiled face_swap controls {flag} itself; remove it from extra_args "
                    "and use reference_face_distance/reference_face_margin where applicable"
                )


class StrictReferenceSelector:


    def __init__(self, *, max_distance: float, min_margin: float) -> None:
        self.max_distance = float(max_distance)
        self.min_margin = float(min_margin)
        if not 0 <= self.max_distance <= 1:
            raise StageError("reference_face_distance must be between 0 and 1")
        if not 0 <= self.min_margin <= 1:
            raise StageError("reference_face_margin must be between 0 and 1")
        self._lock = threading.RLock()
        self._reference_face: Any | None = None
        self._active_anchor: str | None = None
        self._stats: dict[str, Any] = {}
        self.reset_stats()

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = {
                "frames_seen": 0,
                "matched": 0,
                "no_faces": 0,
                "no_embeddings": 0,
                "rejected_distance": 0,
                "rejected_ambiguous": 0,
                "multi_face_frames": 0,
                "best_distance_sum": 0.0,
                "best_distance_min": None,
                "best_distance_max": None,
            }

    def activate(self, candidates: Sequence[Path]) -> tuple[Path, list[dict[str, str]]]:

        from facefusion.face_analyser import get_many_faces
        from facefusion.face_selector import sort_and_filter_faces
        from facefusion.vision import read_static_image

        rejected: list[dict[str, str]] = []
        for candidate in candidates:
            vision_frame = read_static_image(str(candidate))
            if vision_frame is None:
                rejected.append({"anchor": str(candidate), "reason": "unreadable image"})
                continue
            faces = sort_and_filter_faces(get_many_faces([vision_frame]))
            if len(faces) != 1:
                rejected.append(
                    {
                        "anchor": str(candidate),
                        "reason": f"expected exactly one detected face, got {len(faces)}",
                    }
                )
                continue
            face = faces[0]
            if getattr(face, "embedding_norm", None) is None:
                rejected.append({"anchor": str(candidate), "reason": "detected face has no embedding"})
                continue
            with self._lock:
                self._reference_face = face
                self._active_anchor = str(candidate)
                self.reset_stats()
            return candidate, rejected
        reasons = "; ".join(entry["reason"] for entry in rejected) or "no candidates"
        raise SampleRejected(f"none of the face_binding anchors is usable: {reasons}")

    def deactivate(self) -> None:
        with self._lock:
            self._reference_face = None
            self._active_anchor = None

    @staticmethod
    def _distance(left: Any, right: Any) -> float | None:
        left_embedding = getattr(left, "embedding_norm", None)
        right_embedding = getattr(right, "embedding_norm", None)
        if left_embedding is None or right_embedding is None:
            return None

        raw = 1.0 - float(left_embedding.dot(right_embedding))
        return max(0.0, min(1.0, raw / 2.0))

    def select_faces(self, reference_vision_frame: Any, target_vision_frame: Any) -> list[Any]:

        del reference_vision_frame
        from facefusion.face_analyser import get_many_faces

        with self._lock:
            reference_face = self._reference_face
        if reference_face is None:
            raise RuntimeError("profiled face selector used without an active anchor")

        target_faces = list(get_many_faces([target_vision_frame]) or [])
        with self._lock:
            self._stats["frames_seen"] += 1
            if len(target_faces) > 1:
                self._stats["multi_face_frames"] += 1
        if not target_faces:
            with self._lock:
                self._stats["no_faces"] += 1
            return []

        distances = [
            (distance, face)
            for face in target_faces
            if (distance := self._distance(face, reference_face)) is not None
        ]
        if not distances:
            with self._lock:
                self._stats["no_embeddings"] += 1
            return []
        distances.sort(key=lambda pair: pair[0])
        best_distance, best_face = distances[0]
        with self._lock:
            self._stats["best_distance_sum"] += best_distance
            previous_min = self._stats["best_distance_min"]
            previous_max = self._stats["best_distance_max"]
            self._stats["best_distance_min"] = (
                best_distance if previous_min is None else min(previous_min, best_distance)
            )
            self._stats["best_distance_max"] = (
                best_distance if previous_max is None else max(previous_max, best_distance)
            )
        if best_distance >= self.max_distance:
            with self._lock:
                self._stats["rejected_distance"] += 1
            return []
        if len(distances) > 1 and distances[1][0] - best_distance < self.min_margin:
            with self._lock:
                self._stats["rejected_ambiguous"] += 1
            return []
        with self._lock:
            self._stats["matched"] += 1
        return [best_face]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._stats)
            result["anchor"] = self._active_anchor
        measured = result["frames_seen"] - result["no_faces"] - result["no_embeddings"]
        result["best_distance_mean"] = (
            result.pop("best_distance_sum") / measured if measured > 0 else None
        )
        return result


class ProfiledFaceSwapPipeline(FaceSwapPipeline):


    def __init__(
        self,
        *,
        processors: Sequence[str],
        execution_providers: Sequence[str],
        extra_args: Iterable[str] | None,
        video_memory_strategy: str,
        reference_face_distance: float,
        reference_face_margin: float,
        minimum_matched_frames: int,
        minimum_match_ratio: float,
        facefusion_root: str | Path | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            processors=processors,
            execution_providers=execution_providers,
            extra_args=extra_args,
            video_memory_strategy=video_memory_strategy,
            facefusion_root=facefusion_root,
            work_dir=work_dir,
        )
        self.selector = StrictReferenceSelector(
            max_distance=reference_face_distance,
            min_margin=reference_face_margin,
        )
        self.minimum_matched_frames = int(minimum_matched_frames)
        self.minimum_match_ratio = float(minimum_match_ratio)
        if self.minimum_matched_frames < 0:
            raise StageError("minimum_matched_frames must be non-negative")
        if not 0 <= self.minimum_match_ratio <= 1:
            raise StageError("minimum_match_ratio must be between 0 and 1")
        self._original_selectors: list[tuple[Any, Any]] = []

    def prepare_environment(self) -> None:
        super().prepare_environment()
        patched = 0
        for module in self.processor_modules:
            if hasattr(module, "select_faces"):
                self._original_selectors.append((module, module.select_faces))
                module.select_faces = self.selector.select_faces
                patched += 1
        if not patched:
            raise StageError(
                "none of the configured FaceFusion processors exposes select_faces; "
                "profiled swapping requires a face-selecting processor"
            )

    def run_profiled(
        self,
        *,
        source_path: str,
        target_path: str,
        output_path: str,
        anchor_paths: Sequence[Path],
    ) -> dict[str, Any]:
        if not self.environment_ready:
            raise StageError("facefusion: run_profiled() called before prepare_environment()")
        selected_anchor, rejected_anchors = self.selector.activate(anchor_paths)
        try:
            from facefusion import state_manager

            state_manager.set_item("face_selector_mode", "reference")
            state_manager.set_item("reference_face_distance", self.selector.max_distance)
            super().run(
                source_path=source_path,
                target_path=target_path,
                output_path=output_path,
            )
            diagnostics = self.selector.snapshot()
            diagnostics["selected_anchor"] = str(selected_anchor)
            diagnostics["rejected_anchors"] = rejected_anchors
            seen = int(diagnostics["frames_seen"])
            matched = int(diagnostics["matched"])
            ratio = matched / seen if seen else 0.0
            diagnostics["match_ratio"] = ratio
            diagnostics["accepted"] = (
                matched >= self.minimum_matched_frames
                and ratio >= self.minimum_match_ratio
            )
            diagnostics["required"] = {
                "minimum_matched_frames": self.minimum_matched_frames,
                "minimum_match_ratio": self.minimum_match_ratio,
            }
            return diagnostics
        finally:
            self.selector.deactivate()

    def close(self) -> None:
        try:
            super().close()
        finally:
            for module, original in self._original_selectors:
                module.select_faces = original
            self._original_selectors.clear()
            self.selector.deactivate()


class ProfiledFaceSwapStage(FaceSwapStage):
    type = "facefusion.profiled_video_swap"
    needs = ("draft_video", "face_binding")
    description = "swap each bound identity into at most one matched draft face per frame"

    def setup(self, context: StageContext) -> None:
        self.processors = [str(processor) for processor in context.require_param("processors")]
        self.extra_args = [str(item) for item in (context.param("extra_args") or [])]
        strategy = validate_video_memory_strategy(
            context.param("video_memory_strategy", DEFAULT_VIDEO_MEMORY_STRATEGY)
        )
        reject_strategy_in_extra_args(self.extra_args)
        reject_selector_in_extra_args(self.extra_args)

        self.facefusion_root = self._prepare_imports(context)
        self._stage_assets(context)
        providers = resolve_execution_providers(
            [str(provider) for provider in context.require_param("execution_providers")],
            allow_cpu_fallback=bool(context.param("allow_cpu_fallback", False)),
        )
        self.keep_intermediate_videos = bool(context.param("keep_intermediate_videos", False))
        distance = float(context.param("reference_face_distance", 0.30))
        margin = float(context.param("reference_face_margin", 0.05))
        minimum_matched_frames = int(context.param("minimum_matched_frames", 1))
        minimum_match_ratio = float(context.param("minimum_match_ratio", 0.0))

        context.logger.info("profiled facefusion execution providers: %s", providers)
        context.logger.info(
            "profiled selector: distance<%.3f margin>=%.3f min_frames=%d "
            "min_ratio=%.3f processors=%s strategy=%s",
            distance,
            margin,
            minimum_matched_frames,
            minimum_match_ratio,
            self.processors,
            strategy,
        )
        self.pipeline = ProfiledFaceSwapPipeline(
            processors=self.processors,
            execution_providers=providers,
            extra_args=self.extra_args,
            video_memory_strategy=strategy,
            reference_face_distance=distance,
            reference_face_margin=margin,
            minimum_matched_frames=minimum_matched_frames,
            minimum_match_ratio=minimum_match_ratio,
            facefusion_root=self.facefusion_root,
            work_dir=context.stage_dir,
        )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        if not items:
            return []
        if any(self.skip_reason(item.raw) is None for item in items) and not self.pipeline.environment_ready:
            self.pipeline.prepare_environment()
            context.logger.info(
                "profiled facefusion environment ready; replaying across %d item(s)", len(items)
            )
        return [
            guard(item, lambda item=item: self._process_profiled(item, context), context.logger)
            for item in items
        ]

    @classmethod
    def can_reuse(cls, raw, outputs, params) -> bool:
        count = outputs.get("subject_count")
        return type(count) is int and count == len(cls._source_indices(raw))

    def _process_profiled(self, item: WorkItem, context: StageContext) -> ItemResult:
        reason = self.skip_reason(item.raw)
        if reason is not None:
            return ItemResult.skipped(item.sample_id, reason)
        target_video = context.input_media(item, "draft_video", "video")
        bindings_path = context.input_media(item, "face_binding", "bindings_file")
        manifest = read_json(bindings_path, default=None)
        subjects = self._validate_manifest(manifest, item.sample_id)
        output_path = item.item_dir / "video.mp4"
        diagnostics_path = item.item_dir / "selector_diagnostics.json"
        prompt = self._prompt(item)

        if not subjects:
            raise SampleRejected(f"{item.sample_id}: human references exist but bindings contain no subjects")
        human_indices = set(self._source_indices(item.raw))
        if {subject["source_image_index"] for subject in subjects} != human_indices:
            raise SampleRejected(
                f"{item.sample_id}: binding sources must match the original human-reference indices"
            )

        source_images = context.resolve_images(item)
        binding_root = bindings_path.parents[2]
        current_video = target_video
        temporary_videos: list[Path] = []
        reports: list[dict[str, Any]] = []
        for ordinal, subject in enumerate(subjects, start=1):
            source_index = subject["source_image_index"]
            if source_index >= len(source_images):
                raise ValueError(
                    f"{item.sample_id}: binding source_image_index={source_index} exceeds "
                    f"the dataset's {len(source_images)} img_paths"
                )
            source_image = source_images[source_index]
            anchors = [
                self._resolve_anchor(binding_root, candidate["anchor_crop"])
                for candidate in subject["candidates"]
            ]
            is_last = ordinal == len(subjects)
            next_video = output_path if is_last else item.item_dir / f"swap_{ordinal:02d}.mp4"
            if next_video.exists():
                next_video.unlink()
            subject_report: dict[str, Any] = {
                "subject_id": subject["subject_id"],
                "source_image_index": source_index,
                "source_image": source_image.name,
                "role": subject["role"],
                "status": "running",
            }
            reports.append(subject_report)
            atomic_write_json(
                diagnostics_path,
                {"schema_version": 1, "sample_id": item.sample_id, "subjects": reports},
            )
            try:
                selector_report = self.pipeline.run_profiled(
                    source_path=str(source_image),
                    target_path=str(current_video),
                    output_path=str(next_video),
                    anchor_paths=anchors,
                )
            except Exception as exc:
                subject_report["status"] = "failed"
                subject_report["error"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc).strip() or exc.__class__.__name__,
                }
                atomic_write_json(
                    diagnostics_path,
                    {"schema_version": 1, "sample_id": item.sample_id, "subjects": reports},
                )
                raise
            subject_report["selector"] = self._relative_selector_report(
                selector_report, binding_root
            )
            if not selector_report["accepted"]:
                subject_report["status"] = "failed"
                atomic_write_json(
                    diagnostics_path,
                    {"schema_version": 1, "sample_id": item.sample_id, "subjects": reports},
                )
                required = selector_report["required"]
                raise SampleRejected(
                    "profiled selector matched too few frames: "
                    f"matched={selector_report['matched']}, seen={selector_report['frames_seen']}, "
                    f"ratio={selector_report['match_ratio']:.3f}, "
                    f"required_frames={required['minimum_matched_frames']}, "
                    f"required_ratio={required['minimum_match_ratio']:.3f}"
                )
            subject_report["status"] = "succeeded"
            atomic_write_json(
                diagnostics_path,
                {"schema_version": 1, "sample_id": item.sample_id, "subjects": reports},
            )
            if not is_last:
                temporary_videos.append(next_video)
                current_video = next_video

        if not self.keep_intermediate_videos:
            for path in temporary_videos:
                if path.exists():
                    path.unlink()
        return ItemResult.succeeded(
            item.sample_id,
            video=context.relative(output_path),
            diagnostics_file=context.relative(diagnostics_path),
            subject_count=len(subjects),
            source_images=[source_images[subject["source_image_index"]].name for subject in subjects],
            processors=list(self.processors),
            prompt=prompt,
        )

    @staticmethod
    def _validate_manifest(value: Any, sample_id: str) -> list[dict[str, Any]]:
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise ValueError(f"{sample_id}: face_binding manifest is missing schema_version=1")
        if str(value.get("sample_id")) != sample_id:
            raise ValueError(
                f"{sample_id}: face_binding manifest belongs to {value.get('sample_id')!r}"
            )
        subjects = value.get("subjects")
        if not isinstance(subjects, list):
            raise ValueError(f"{sample_id}: face_binding subjects must be a list")
        normalized: list[dict[str, Any]] = []
        seen_subjects: set[str] = set()
        seen_sources: set[int] = set()
        for subject in subjects:
            if not isinstance(subject, Mapping):
                raise ValueError(f"{sample_id}: every bound subject must be an object")
            subject_id = str(subject.get("subject_id") or "").strip()
            source_index = subject.get("source_image_index")
            role = str(subject.get("role") or "").strip()
            candidates = subject.get("candidates")
            if not subject_id or subject_id in seen_subjects:
                raise ValueError(f"{sample_id}: missing or duplicate subject_id {subject_id!r}")
            if type(source_index) is not int or source_index < 0 or source_index in seen_sources:
                raise ValueError(f"{sample_id}: invalid or duplicate source_image_index {source_index!r}")
            if not role:
                raise ValueError(f"{sample_id}: subject {subject_id} has no role")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{sample_id}: subject {subject_id} has no anchor candidates")
            clean_candidates: list[dict[str, Any]] = []
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise ValueError(f"{sample_id}: subject {subject_id} candidate is not an object")
                anchor_crop = candidate.get("anchor_crop")
                if not isinstance(anchor_crop, str) or not anchor_crop.strip():
                    raise ValueError(f"{sample_id}: subject {subject_id} candidate lacks anchor_crop")
                clean_candidates.append(dict(candidate))
            seen_subjects.add(subject_id)
            seen_sources.add(source_index)
            normalized.append(
                {
                    "subject_id": subject_id,
                    "source_image_index": source_index,
                    "role": role,
                    "candidates": clean_candidates,
                }
            )
        return normalized

    @staticmethod
    def _resolve_anchor(binding_root: Path, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise ValueError(f"face_binding anchor must be relative, got {relative!r}")
        resolved = (binding_root / candidate).resolve()
        try:
            resolved.relative_to(binding_root.resolve())
        except ValueError:
            raise ValueError(f"face_binding anchor escapes its stage directory: {relative!r}") from None
        if not resolved.is_file():
            raise ValueError(f"face_binding anchor is missing: {relative!r}")
        return resolved

    @staticmethod
    def _copy_or_link(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    @staticmethod
    def _relative_selector_report(report: Mapping[str, Any], binding_root: Path) -> dict[str, Any]:
        clean = dict(report)
        for key in ("anchor", "selected_anchor"):
            value = clean.get(key)
            if value:
                try:
                    clean[key] = str(Path(str(value)).resolve().relative_to(binding_root.resolve()))
                except ValueError:
                    clean[key] = Path(str(value)).name
        rejected = []
        for entry in clean.get("rejected_anchors") or []:
            item = dict(entry)
            value = item.get("anchor")
            if value:
                try:
                    item["anchor"] = str(
                        Path(str(value)).resolve().relative_to(binding_root.resolve())
                    )
                except ValueError:
                    item["anchor"] = Path(str(value)).name
            rejected.append(item)
        clean["rejected_anchors"] = rejected
        return clean
