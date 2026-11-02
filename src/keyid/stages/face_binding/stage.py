from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from keyid.core.artifacts import atomic_write_json
from keyid.core.contract import (
    BatchStage,
    ItemResult,
    StageContext,
    WorkItem,
    guard,
    map_concurrently,
    require_upstream_output,
)
from keyid.core.errors import StageError
from keyid.core.secrets import scrub_values

from .refinement import FaceRefiner, crop_bounds

from .._dataset_prompt import HUMAN_CATEGORIES, image_category, parse_temporal_captions
from .._openai_client import (
    build_client,
    image_to_data_url,
    read_text_resource,
    require_text,
    response_summary,
)

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "face_binding.txt"
CONFIDENCES = frozenset({"high", "medium", "low"})
VISIBILITIES = frozenset({"visible", "occluded", "absent"})


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def normalized_bbox(value: Any, *, field: str) -> list[float]:

    if not isinstance(value, list) or len(value) != 4 or not all(_finite_number(v) for v in value):
        raise ValueError(f"{field} must be four finite normalized coordinates")
    x1, y1, x2, y2 = (float(v) for v in value)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"{field} is not a valid normalized xyxy box: {value!r}")
    return [x1, y1, x2, y2]


def parse_json_object(text: str) -> dict[str, Any]:

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("face binding response must be a JSON object")
    return value


def _spread(values: Sequence[float], limit: int) -> list[float]:

    unique = sorted({min(1.0, max(0.0, float(value))) for value in values})
    if len(unique) <= limit:
        return unique
    if limit <= 1:
        return [unique[len(unique) // 2]]
    indexes = {round(index * (len(unique) - 1) / (limit - 1)) for index in range(limit)}
    return [unique[index] for index in sorted(indexes)]


class FaceBindingStage(BatchStage):
    type = "openai.face_binding"
    requires_human_reference = True
    needs = ("draft_video", "draft_prompt", "head_extract")
    description = "bind ordered person references to inspectable face anchors in the draft"

    def setup(self, context: StageContext) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise StageError(
                "openai.face_binding needs OpenCV in its stage interpreter; point the stage "
                "at an environment containing both openai and cv2"
            ) from exc

        self.cv2 = cv2
        self.model_name = str(context.require_param("model_name"))
        self.detail = str(context.param("image_detail", "high"))
        self.reasoning_effort = str(context.param("reasoning_effort", "low"))
        self.concurrency = max(1, int(context.param("concurrency", context.concurrency)))
        self.max_attempts = max(1, int(context.param("max_attempts", 2)))
        self.max_candidates = max(1, int(context.param("max_candidates_per_subject", 3)))
        self.max_sample_frames = max(1, int(context.param("max_sample_frames", 8)))
        self.max_evidence_images = max(2, int(context.param("max_evidence_images", 12)))
        self.padding = float(context.param("bbox_padding", 0.30))
        if not 0 <= self.padding <= 1:
            raise StageError("openai.face_binding: bbox_padding must be between 0 and 1")
        raw_padding_candidates = context.param("bbox_padding_candidates")
        if raw_padding_candidates is None:
            raw_padding_candidates = [self.padding]
        if (
            not isinstance(raw_padding_candidates, (list, tuple))
            or not raw_padding_candidates
            or not all(
                _finite_number(value) and 0 <= float(value) <= 1
                for value in raw_padding_candidates
            )
        ):
            raise StageError(
                "openai.face_binding: bbox_padding_candidates must be a non-empty "
                "list of finite numbers between 0 and 1"
            )
        self.padding_candidates = tuple(
            dict.fromkeys(float(value) for value in raw_padding_candidates)
        )
        self.overlap_limit = float(context.param("cross_subject_iou_limit", 0.55))
        if not 0 <= self.overlap_limit <= 1:
            raise StageError("openai.face_binding: cross_subject_iou_limit must be between 0 and 1")

        raw_confidences = context.param("accepted_confidences", ["high", "medium"])
        if not isinstance(raw_confidences, (list, tuple)):
            raise StageError("openai.face_binding: accepted_confidences must be a list")
        self.accepted_confidences = {str(value).strip().lower() for value in raw_confidences}
        if not self.accepted_confidences or not self.accepted_confidences <= CONFIDENCES:
            raise StageError(
                f"openai.face_binding: accepted_confidences must be a non-empty subset of {sorted(CONFIDENCES)}"
            )

        raw_positions = context.param("sample_positions", [0.0, 0.25, 0.5, 0.75, 1.0])
        if not isinstance(raw_positions, (list, tuple)) or not raw_positions:
            raise StageError("openai.face_binding: sample_positions must be a non-empty list")
        if not all(_finite_number(value) and 0 <= float(value) <= 1 for value in raw_positions):
            raise StageError("openai.face_binding: sample_positions must contain numbers in [0, 1]")
        self.sample_positions = [float(value) for value in raw_positions]
        self.temporal_caption_field = str(
            context.param("temporal_caption_field", "temporal_captions")
        ).strip()

        self.rule_prompt = read_text_resource(context.param("rule_prompt_path"), PROMPT_PATH)
        if not bool(context.param("send_images_to_api", True)):
            raise StageError(
                "openai.face_binding cannot run with send_images_to_api=false; disable the stage "
                "or explicitly allow the configured endpoint to receive the reference/frame images"
            )
        minimum_size = context.param("face_min_size_px", 16)
        if type(minimum_size) is not int or minimum_size < 1:
            raise StageError("face_binding: face_min_size_px must be a positive integer")
        refinement = {
            "person_overlap": context.param("person_face_overlap", 0.85),
            "roi_padding": context.param("person_roi_padding", 0.05),
            "duplicate_iou": context.param("face_duplicate_iou", 0.50),
            "anchor_iou": context.param("anchor_face_iou", 0.40),
        }
        for name, value in refinement.items():
            if not _finite_number(value) or not (0 <= value <= 1) or (name != "roi_padding" and value == 0):
                raise StageError(f"face_binding: invalid refinement setting {name}")
        self.detector = self._build_detector(context)
        self.refiner = FaceRefiner(
            self.cv2, self.detector, padding=self.padding, max_candidates=self.max_candidates,
            minimum_size=minimum_size, cross_subject_iou=self.overlap_limit,
            padding_candidates=self.padding_candidates, **refinement,
        )
        self.client = build_client(
            context.credentials.get("openai", {}),
            context.endpoints.get("openai", {}),
            timeout=context.param("timeout"),
            params=context.params,
        )
        context.logger.info(
            "face_binding ready: model=%s frames<=%d candidates<=%d concurrency=%d "
            "bbox_padding_candidates=%s",
            self.model_name,
            self.max_sample_frames,
            self.max_candidates,
            self.concurrency,
            list(self.padding_candidates),
        )

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return map_concurrently(
            items,
            lambda item: guard(item, lambda: self._process_one(item, context), context.logger),
            concurrency=self.concurrency,
            logger=context.logger,
        )

    def close(self, context: StageContext) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            try:
                client.close()
            except Exception:
                context.logger.debug("closing face_binding API client failed", exc_info=True)
        detector = getattr(self, "detector", None)
        if detector is not None:
            detector.close()
        context.logger.info("face_binding closed")

    @staticmethod
    def _detector_configuration(params: Mapping[str, Any]) -> dict[str, Any]:
        defaults = {
            "face_detector_model_path": None, "face_detector_size": [640, 640],
            "face_detector_score": 0.60, "face_detector_nms_threshold": 0.40,
            "face_detector_threads": 2, "face_detector_providers": ["cpu"],
            "allow_cpu_fallback": False, "face_min_size_px": 16,
            "person_face_overlap": 0.85, "person_roi_padding": 0.05,
            "face_duplicate_iou": 0.50, "anchor_face_iou": 0.40,
            "cross_subject_iou_limit": 0.55, "bbox_padding": 0.30,
            "bbox_padding_candidates": None, "max_candidates_per_subject": 3,
        }
        return {key: params.get(key, value) for key, value in defaults.items()}

    @classmethod
    def can_reuse(cls, raw, outputs, params) -> bool:


        return outputs.get("binding_revision") == 2

    def _build_detector(self, context: StageContext):
        from .detector import LocalFaceDetector

        return LocalFaceDetector(context)

    @staticmethod
    def _debug_json(path: Path, value: Any) -> None:
        atomic_write_json(path, scrub_values(value))

    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        reason = self.skip_reason(item.raw)
        if reason is not None:
            return ItemResult.skipped(item.sample_id, reason)
        target_video = context.input_media(item, "draft_video", "video")
        prompt = str(
            require_upstream_output(item, "draft_prompt", "prompt", what="the draft prompt")
        ).strip()
        references = self._human_references(item, context)
        if not references:
            raise ValueError(f"{item.sample_id}: human references could not be resolved")
        if len(references) >= self.max_evidence_images:
            raise ValueError(
                f"{item.sample_id}: {len(references)} human references leave no room for a draft "
                f"frame under max_evidence_images={self.max_evidence_images}"
            )
        frame_budget = min(self.max_sample_frames, self.max_evidence_images - len(references))
        frames, video_info = self._extract_frames(
            target_video, item.item_dir / "evidence", self._positions(item), frame_budget,
        )
        if not frames:
            raise ValueError(f"{item.sample_id}: no readable frame could be sampled from {target_video}")
        contact_sheet_path = self._contact_sheet(frames, item.item_dir / "contact_sheet.jpg")
        semantic_path = item.item_dir / "semantic_binding.json"
        detections_path = item.item_dir / "face_detections.json"
        histories, summaries = [], []
        feedback, last_error = "", ""
        binding = None
        overlay_files: list[Path] = []
        for attempt in range(1, self.max_attempts + 1):
            response = self.client.responses.create(
                model=self.model_name,
                input=self._build_input(item, prompt, references, frames, feedback),
                reasoning={"effort": self.reasoning_effort},
            )
            summaries.append(response_summary(response))
            entry = {"attempt": attempt, "status": "pending"}
            histories.append(entry)
            try:
                payload = parse_json_object(require_text(response.output_text, what="face binding response"))
                semantic = self._validate_binding(payload, references, frames)
                entry["semantic_binding"] = semantic
                self._debug_json(semantic_path, {
                    "schema_version": 1, "sample_id": item.sample_id,
                    "run_attempt": context.attempt,
                    "sampled_frames": [
                        {"frame_id": f["frame_id"], "evidence_image": context.relative(f["path"])}
                        for f in frames
                    ],
                    "attempts": histories,
                })
                refined, diagnostics = self.refiner.refine(
                    semantic, frames, item.item_dir / "debug_crops" / f"attempt_{attempt:02d}",
                    context.relative,
                )
                entry["detections"] = diagnostics
                overlay_files = self._detection_overlays(
                    item, context, semantic, frames, diagnostics, attempt,
                )
                self._debug_json(detections_path, {
                    "schema_version": 1, "sample_id": item.sample_id,
                    "run_attempt": context.attempt,
                    "attempts": [
                        {"attempt": h["attempt"], "status": h["status"],
                         "diagnostics": h.get("detections"), "error": h.get("error")}
                        for h in histories
                    ],
                })
                if diagnostics["status"] != "succeeded":
                    reasons = [
                        f"{s['subject_id']} {c['frame_id']}: {c.get('reason', 'rejected')}"
                        for s in diagnostics["subjects"] for c in s["candidates"]
                        if c["status"] != "accepted"
                    ]
                    raise ValueError(
                        "; ".join(diagnostics["errors"]) + "; " + "; ".join(reasons[:12])
                    )
                entry["status"] = "succeeded"
                binding = refined
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                entry.update(status="rejected", error=last_error)
                feedback = (
                    "Your previous person binding failed validation/local face refinement: "
                    f"{last_error}. Correct the PERSON assignment/box or select another supplied "
                    "frame. Do not invent a face box and do not bind a different person."
                )
                context.logger.warning(
                    "face binding rejected for %s (attempt %d/%d): %s",
                    item.sample_id, attempt, self.max_attempts, last_error,
                )
            finally:
                self._debug_json(semantic_path, {
                    "schema_version": 1, "sample_id": item.sample_id,
                    "run_attempt": context.attempt,
                    "sampled_frames": [
                        {"frame_id": f["frame_id"], "evidence_image": context.relative(f["path"])}
                        for f in frames
                    ],
                    "attempts": [
                        {k: v for k, v in h.items() if k != "detections"} for h in histories
                    ],
                })
                self._debug_json(detections_path, {
                    "schema_version": 1, "sample_id": item.sample_id,
                    "run_attempt": context.attempt,
                    "attempts": [
                        {"attempt": h["attempt"], "status": h["status"],
                         "diagnostics": h.get("detections"), "error": h.get("error")}
                        for h in histories
                    ],
                })
        if binding is None:
            raise ValueError(
                f"{item.sample_id}: no reliable face binding after "
                f"{self.max_attempts} attempt(s): {last_error}; inspect semantic_binding.json "
                "and face_detections.json"
            )
        manifest, anchor_files, _ = self._materialize(
            item, context, binding, references, frames, video_info, prompt, summaries,
            overlay_files=overlay_files,
        )
        self._debug_json(item.item_dir / "bindings.json", manifest)
        context.logger.info(
            "local face binding accepted: %s subjects=%d anchors=%d",
            item.sample_id, len(binding["subjects"]), len(anchor_files),
        )
        return ItemResult.succeeded(
            item.sample_id,
            bindings_file=context.relative(item.item_dir / "bindings.json"),
            semantic_binding_file=context.relative(semantic_path),
            face_detections_file=context.relative(detections_path),
            contact_sheet_file=context.relative(contact_sheet_path),
            evidence_files=[context.relative(frame["path"]) for frame in frames],
            overlay_files=[context.relative(path) for path in overlay_files],
            anchor_files=[context.relative(path) for path in anchor_files],
            subject_count=len(manifest["subjects"]),
            candidate_count=len(anchor_files),
            sampled_frame_indices=[frame["frame_index"] for frame in frames],
            model=self.model_name, binding_revision=2,
            detector_config=self._detector_configuration(context.params),
        )

    def _detection_overlays(self, item, context, semantic, frames, diagnostics, attempt):
        directory = item.item_dir / "overlays" / f"attempt_{attempt:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        records = {entry["frame_id"]: entry for entry in diagnostics["frames"]}
        selected = {
            (c["frame_id"], c["face_id"])
            for s in diagnostics["subjects"] for c in s["candidates"]
            if c["status"] == "accepted"
        }
        output = []
        for frame in frames:
            image = self.cv2.imread(str(frame["path"]))
            height, width = image.shape[:2]
            for subject in semantic["subjects"]:
                for candidate in subject["candidates"]:
                    if candidate["frame_id"] != frame["frame_id"]:
                        continue
                    x1, y1, x2, y2 = self._pixel_box(candidate["person_bbox"], width, height)
                    self.cv2.rectangle(image, (x1, y1), (x2, y2), (80, 220, 80), 2)
                    self.cv2.putText(image, subject["subject_id"], (x1, max(20, y1-4)),
                                     self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)
            for face in records[frame["frame_id"]]["detections"]:
                x1, y1, x2, y2 = self._pixel_box(face["face_bbox"], width, height)
                is_selected = (frame["frame_id"], face["face_id"]) in selected
                color = (0, 220, 255) if is_selected else (255, 140, 40)
                self.cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if is_selected else 1)
                self.cv2.putText(image, f"{face['face_id'].split(':')[-1]} {face['detector_score']:.2f}",
                                 (x1, min(height-2, y2+16)), self.cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                for x, y in face["landmarks"]:
                    self.cv2.circle(image, (round(x*width), round(y*height)), 2, color, -1)
            path = directory / f"{frame['frame_id']}.jpg"
            if not self.cv2.imwrite(str(path), image, [self.cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"cannot write detection overlay: {path}")
            output.append(path)
        return output

    def _human_references(self, item: WorkItem, context: StageContext) -> list[dict[str, Any]]:
        raw_paths = [str(value) for value in (item.raw.get("img_paths") or ())]
        resolved = context.resolve_images(item)
        head_prompts = list((item.input("head_extract").get("head_prompts") or []))
        references: list[dict[str, Any]] = []
        human_order = 0
        for source_index, (raw_path, path) in enumerate(zip(raw_paths, resolved, strict=False)):
            if image_category(raw_path) not in HUMAN_CATEGORIES:
                continue
            description = str(head_prompts[human_order]).strip() if human_order < len(head_prompts) else ""
            references.append(
                {
                    "source_image_index": source_index,
                    "reference_order": human_order,
                    "raw_path": raw_path,
                    "path": path,
                    "head_description": description,
                }
            )
            human_order += 1
        return references

    def _positions(self, item: WorkItem) -> list[float]:
        positions = list(self.sample_positions)
        fields = item.raw.get("fields") or {}
        if self.temporal_caption_field and self.temporal_caption_field in fields:
            captions = parse_temporal_captions(
                fields.get(self.temporal_caption_field), sample_id=item.sample_id
            )
            for caption in captions:
                positions.extend((float(caption["start"]), float(caption["end"])))
        return positions

    def _extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        positions: Sequence[float],
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = self.cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot open draft video: {video_path}")
            frame_count = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(self.cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_count <= 0 or width <= 0 or height <= 0:
                raise ValueError(
                    f"draft video has invalid metadata: frames={frame_count} size={width}x{height}"
                )
            selected_positions = _spread(positions, limit)
            indices = sorted({round(position * (frame_count - 1)) for position in selected_positions})
            frames: list[dict[str, Any]] = []
            for frame_index in indices:
                capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok or image is None:
                    continue
                frame_id = f"frame_{frame_index:06d}"
                path = output_dir / f"{frame_id}.jpg"
                if not self.cv2.imwrite(str(path), image, [self.cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"failed to write sampled frame: {path}")
                frames.append(
                    {
                        "frame_id": frame_id,
                        "frame_index": frame_index,
                        "time_sec": frame_index / fps if fps > 0 else None,
                        "path": path,
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                    }
                )
            return frames, {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_sec": (frame_count - 1) / fps if fps > 0 else None,
            }
        finally:
            capture.release()

    def _build_input(
        self,
        item: WorkItem,
        prompt: str,
        references: list[dict[str, Any]],
        frames: list[dict[str, Any]],
        feedback: str,
    ) -> list[dict[str, Any]]:
        fields = item.raw.get("fields") or {}
        context_payload = {
            "sample_id": item.sample_id,
            "video_prompt": prompt,
            "temporal_captions": fields.get(self.temporal_caption_field, []),
            "subjects": [
                {
                    "source_image_index": ref["source_image_index"],
                    "reference_order": ref["reference_order"],
                    "head_description": ref["head_description"],
                }
                for ref in references
            ],
            "sampled_frames": [
                {
                    "frame_id": frame["frame_id"],
                    "frame_index": frame["frame_index"],
                    "time_sec": frame["time_sec"],
                }
                for frame in frames
            ],
            "validation_feedback": feedback,
        }
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": "TASK CONTEXT:\n" + json.dumps(context_payload, ensure_ascii=False),
            }
        ]
        for ref in references:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "IDENTITY REFERENCE "
                        f"source_image_index={ref['source_image_index']} "
                        f"reference_order={ref['reference_order']}"
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(ref["path"]),
                    "detail": self.detail,
                }
            )
        for frame in frames:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"DRAFT FRAME frame_id={frame['frame_id']} "
                        f"frame_index={frame['frame_index']} time_sec={frame['time_sec']}"
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(frame["path"]),
                    "detail": self.detail,
                }
            )
        return [
            {"role": "developer", "content": [{"type": "input_text", "text": self.rule_prompt}]},
            {"role": "user", "content": content},
        ]

    def _validate_binding(
        self,
        payload: Mapping[str, Any],
        references: list[dict[str, Any]],
        frames: list[dict[str, Any]],
    ) -> dict[str, Any]:
        subjects = payload.get("subjects")
        expected = {ref["source_image_index"] for ref in references}
        frame_map = {frame["frame_id"]: frame for frame in frames}
        if not isinstance(subjects, list) or len(subjects) != len(expected):
            raise ValueError("response must contain exactly one subject for every human reference")

        normalized: list[dict[str, Any]] = []
        seen: set[int] = set()
        for subject in subjects:
            if not isinstance(subject, Mapping):
                raise ValueError("every subject must be an object")
            source_index = subject.get("source_image_index")
            if type(source_index) is not int or source_index not in expected or source_index in seen:
                raise ValueError(f"unknown or duplicate source_image_index: {source_index!r}")
            seen.add(source_index)
            role = str(subject.get("role") or "").strip()
            confidence = str(subject.get("confidence") or "").strip().lower()
            if not role:
                raise ValueError(f"subject {source_index} has no role description")
            if confidence not in CONFIDENCES:
                raise ValueError(f"subject {source_index} has invalid confidence {confidence!r}")
            if confidence not in self.accepted_confidences:
                raise ValueError(f"subject {source_index} binding confidence is {confidence!r}; refusing to guess")
            candidates = subject.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError(f"subject {source_index} candidates must be a list")

            accepted: list[dict[str, Any]] = []
            candidate_frames: set[str] = set()
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise ValueError(f"subject {source_index} candidate must be an object")
                frame_id = str(candidate.get("frame_id") or "")
                if frame_id not in frame_map:
                    raise ValueError(f"subject {source_index} cited unknown frame_id {frame_id!r}")
                visibility = str(candidate.get("face_visibility") or "visible").strip().lower()
                candidate_confidence = str(candidate.get("confidence") or "").strip().lower()
                if visibility not in VISIBILITIES:
                    raise ValueError(f"subject {source_index} has invalid face_visibility {visibility!r}")
                if candidate_confidence not in CONFIDENCES:
                    raise ValueError(f"subject {source_index} has invalid candidate confidence")


                if candidate_confidence not in self.accepted_confidences:
                    continue
                if frame_id in candidate_frames:
                    raise ValueError(f"subject {source_index} repeats candidate frame {frame_id}")
                person_box = normalized_bbox(candidate.get("person_bbox"), field="person_bbox")
                candidate_frames.add(frame_id)
                accepted.append(
                    {
                        "frame_id": frame_id,
                        "frame_index": frame_map[frame_id]["frame_index"],
                        "time_sec": frame_map[frame_id]["time_sec"],
                        "person_bbox": person_box,
                        "face_visibility": visibility,
                        "confidence": candidate_confidence,
                        "evidence": str(candidate.get("evidence") or "").strip(),
                    }
                )
            if not accepted:
                raise ValueError(f"subject {source_index} has no acceptable person candidate")
            normalized.append(
                {
                    "subject_id": f"person_{source_index:02d}",
                    "source_image_index": source_index,
                    "role": role,
                    "confidence": confidence,
                    "candidates": accepted,
                }
            )
        if seen != expected:
            raise ValueError(f"subject indexes differ from references: expected {sorted(expected)}, got {sorted(seen)}")
        normalized.sort(key=lambda subject: subject["source_image_index"])
        return {"subjects": normalized}

    def _materialize(
        self,
        item: WorkItem,
        context: StageContext,
        binding: dict[str, Any],
        references: list[dict[str, Any]],
        frames: list[dict[str, Any]],
        video_info: dict[str, Any],
        prompt: str,
        api_summaries: list[dict[str, Any]],
        overlay_files: list[Path] | None = None,
    ) -> tuple[dict[str, Any], list[Path], list[Path]]:
        frame_map = {frame["frame_id"]: frame for frame in frames}
        ref_map = {ref["source_image_index"]: ref for ref in references}
        anchor_files: list[Path] = []
        for subject in binding["subjects"]:
            subject_dir = item.item_dir / subject["subject_id"]
            subject_dir.mkdir(parents=True, exist_ok=True)
            subject["source_reference"] = ref_map[subject["source_image_index"]]["raw_path"]
            for rank, candidate in enumerate(subject["candidates"], start=1):
                frame = frame_map[candidate["frame_id"]]
                image = self.cv2.imread(str(frame["path"]))
                if image is None:
                    raise OSError(f"cannot reread sampled frame {frame['path']}")
                crop = self._crop(
                    image,
                    candidate["face_bbox"],
                    padding=candidate.get("anchor_padding"),
                )
                anchor_path = subject_dir / f"anchor_{rank:02d}_{candidate['frame_id']}.png"
                if not self.cv2.imwrite(str(anchor_path), crop):
                    raise OSError(f"failed to write anchor crop {anchor_path}")
                candidate["anchor_crop"] = context.relative(anchor_path)
                candidate["rank"] = rank
                anchor_files.append(anchor_path)

        overlay_files = overlay_files or []

        manifest = {
            "schema_version": 1,
            "binding_revision": 2,
            "face_bbox_source": "local_detector",
            "detector": self.detector.info,
            "sample_id": item.sample_id,
            "draft_stage": "draft_video",
            "video": video_info,
            "anchor_padding_candidates": list(self.padding_candidates),
            "prompt": prompt,
            "sampled_frames": [
                {
                    "frame_id": frame["frame_id"],
                    "frame_index": frame["frame_index"],
                    "time_sec": frame["time_sec"],
                    "evidence_image": context.relative(frame["path"]),
                }
                for frame in frames
            ],
            "subjects": binding["subjects"],
            "api": api_summaries,
        }
        return manifest, anchor_files, overlay_files

    def _crop(
        self,
        image: Any,
        box: Sequence[float],
        *,
        padding: float | None = None,
    ) -> Any:
        height, width = image.shape[:2]
        selected_padding = self.padding if padding is None else float(padding)
        left, top, right, bottom = crop_bounds(box, width, height, selected_padding)
        crop = image[top:bottom, left:right]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            raise ValueError(f"face crop is too small after padding: {box!r}")
        return crop

    @staticmethod
    def _pixel_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
        left = max(0, min(width - 1, int(math.floor(box[0] * width))))
        top = max(0, min(height - 1, int(math.floor(box[1] * height))))
        right = max(left + 1, min(width, int(math.ceil(box[2] * width))))
        bottom = max(top + 1, min(height, int(math.ceil(box[3] * height))))
        return left, top, right, bottom

    def _contact_sheet(self, frames: list[dict[str, Any]], output_path: Path) -> Path:
        import numpy

        width = 480
        label_height = 42
        tiles = []
        for frame in frames:
            image = self.cv2.imread(str(frame["path"]))
            if image is None:
                continue
            scale = width / image.shape[1]
            resized = self.cv2.resize(image, (width, max(1, round(image.shape[0] * scale))))
            canvas = numpy.full(
                (resized.shape[0] + label_height, width, 3), 24, dtype=numpy.uint8
            )
            canvas[label_height:, :] = resized
            label = f"{frame['frame_id']}  t={frame['time_sec'] if frame['time_sec'] is not None else '?'}"
            self.cv2.putText(
                canvas,
                label,
                (10, 28),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                self.cv2.LINE_AA,
            )
            tiles.append(canvas)
        if not tiles:
            raise ValueError("cannot create contact sheet without readable sampled frames")
        columns = min(3, len(tiles))
        rows = math.ceil(len(tiles) / columns)
        tile_height = max(tile.shape[0] for tile in tiles)
        sheet = numpy.full((rows * tile_height, columns * width, 3), 16, dtype=numpy.uint8)
        for index, tile in enumerate(tiles):
            row, column = divmod(index, columns)
            sheet[row * tile_height : row * tile_height + tile.shape[0], column * width : (column + 1) * width] = tile
        if not self.cv2.imwrite(str(output_path), sheet, [self.cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"failed to write contact sheet {output_path}")
        return output_path
