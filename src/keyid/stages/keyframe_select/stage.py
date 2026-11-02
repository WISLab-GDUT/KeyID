from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from keyid.core.contract import (
    BatchStage,
    ItemResult,
    StageContext,
    WorkItem,
    guard,
    require_upstream_output,
)
from keyid.core.errors import StageError

FRAME_SUFFIX = ".jpg"


@dataclass
class FrameFaceInfo:
    has_face: bool
    is_target: bool
    similarity: float

    def to_dict(self) -> dict:
        return {"has_face": self.has_face, "is_target": self.is_target, "similarity": self.similarity}


def evenly_spaced_time_pairs(source_frames: int, target_frames: int, keyframe_count: int) -> list[tuple[int, int]]:

    if keyframe_count < 2:
        raise ValueError("keyframes must be at least 2")
    pairs: list[tuple[int, int]] = []
    for index in range(keyframe_count):
        target_idx = round(index * (target_frames - 1) / (keyframe_count - 1))
        source_idx = round(target_idx * (source_frames - 1) / (target_frames - 1))
        pairs.append((source_idx, target_idx))
    return pairs


def nearest_indices(center: int, total: int) -> list[int]:

    indices = [center]
    for distance in range(1, total):
        left = center - distance
        right = center + distance
        if left >= 0:
            indices.append(left)
        if right < total:
            indices.append(right)
    return indices


class KeyframeSelectStage(BatchStage):
    type = "ltx.keyframe_select"
    requires_human_reference = True
    needs = ("face_swap",)
    description = "choose identity-verified keyframes per sample into items/<id>/frames/"


    def setup(self, context: StageContext) -> None:
        import numpy as np


        self.upstream = context.require_single_upstream()
        context.logger.info("reading video, prompt and source index from %r", self.upstream)

        self._np = np
        self.num_frames = int(context.require_param("num_frames"))
        self.keyframe_count = int(context.require_param("keyframes"))
        self.similarity_threshold = float(context.require_param("face_sim_threshold"))
        self.face_model_name = str(context.require_param("face_model_name"))
        self.face_model_root = str(context.param("face_model_root", "~/.insightface"))
        self.det_size = int(context.require_param("face_det_size"))
        self.det_thresh = float(context.require_param("face_det_thresh"))
        self.keep_temp = bool(context.param("keep_temp", False))
        self.allow_cpu_fallback = bool(context.param("allow_cpu_fallback", False))
        self.cpu_face = bool(context.param("cpu_face", False))

        self.app, self.face_providers = self._build_face_app(context)
        context.logger.info("insightface providers: %s", self.face_providers)

    def process_batch(self, items: list[WorkItem], context: StageContext) -> list[ItemResult]:
        return [
            guard(item, lambda item=item: self._process_one(item, context), context.logger)
            for item in items
        ]

    def close(self, context: StageContext) -> None:
        if getattr(self, "app", None) is not None:
            del self.app
        context.logger.info("keyframe_select closed")


    def _prompt(self, item: WorkItem) -> str:

        return str(
            require_upstream_output(
                item, self.upstream, "prompt", what="the composed generation prompt"
            )
        ).strip()

    def _process_one(self, item: WorkItem, context: StageContext) -> ItemResult:
        reason = self.skip_reason(item.raw)
        if reason is not None:
            return ItemResult.skipped(item.sample_id, reason)
        video_path = context.input_media(item, self.upstream, "video")
        from keyid.core.references import human_reference_indices

        source_index = item.input(self.upstream).get("source_image_index")
        if source_index is None:
            reference_path = context.resolve_human_image(item)
        else:
            if type(source_index) is not int or source_index not in human_reference_indices(item.raw.get("img_paths") or ()):
                raise ValueError(
                    f"{self.upstream} source_image_index does not identify a human reference"
                )
            reference_path = context.resolve_images(item)[source_index]


        prompt = self._prompt(item)

        frames_dir = item.item_dir / "frames"
        temp_dir = item.item_dir / "frames_temp"
        extracted = self._extract_frames(video_path, temp_dir)

        reference_embedding = self._reference_embedding(reference_path)
        pairs = evenly_spaced_time_pairs(len(extracted), self.num_frames, self.keyframe_count)

        face_cache: dict[int, FrameFaceInfo] = {}
        used: set[int] = set()
        frames_dir.mkdir(parents=True, exist_ok=True)
        plan: list[dict] = []
        keyframe_paths: list[str] = []

        for anchor_idx, target_idx in pairs:
            chosen_idx, reason, info = self._choose_source_frame(
                anchor_idx, extracted, face_cache, used, reference_embedding
            )
            used.add(chosen_idx)
            selected_path = frames_dir / f"target_{target_idx:03d}{FRAME_SUFFIX}"
            self._copy_frame(extracted[chosen_idx], selected_path)

            relative = context.relative(selected_path)
            keyframe_paths.append(relative)
            plan.append(
                {
                    "target_idx": target_idx,
                    "anchor_source_idx": anchor_idx,
                    "chosen_source_idx": chosen_idx,
                    "image": relative,
                    "reason": reason,
                    **info.to_dict(),
                }
            )

        if not self.keep_temp:
            self._remove_tree(temp_dir)

        return ItemResult.succeeded(
            item.sample_id,
            frames=keyframe_paths,
            keyframe_plan=plan,
            num_frames=self.num_frames,


            keyframe_count=self.keyframe_count,
            face_providers=list(self.face_providers),
            face_similarity_threshold=self.similarity_threshold,
            prompt=prompt,
        )


    def _extract_frames(self, video_path: Path, temp_dir: Path) -> list[Path]:
        import cv2
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        frame_paths: list[Path] = []
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_path = temp_dir / f"{index:05d}{FRAME_SUFFIX}"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"cannot write frame: {frame_path}")
                frame_paths.append(frame_path)
                index += 1
        finally:
            capture.release()

        if not frame_paths:
            raise RuntimeError(f"video yielded no frames: {video_path}")
        return frame_paths

    @staticmethod
    def _copy_frame(source: Path, destination: Path) -> None:
        import shutil

        shutil.copy2(source, destination)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)


    def _build_face_app(self, context: StageContext):
        import cv2
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available = list(ort.get_available_providers())
        if self.cpu_face:
            providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.allow_cpu_fallback:
            context.logger.warning(
                "CUDAExecutionProvider is unavailable (available: %s); falling back to CPU "
                "because allow_cpu_fallback is true",
                available,
            )
            providers = ["CPUExecutionProvider"]
        else:
            raise StageError(
                "keyframe_select: CUDAExecutionProvider is not available "
                f"(available: {available}). Set cpu_face: true to run the detector on CPU, "
                "or allow_cpu_fallback: true to accept the fallback."
            )

        app = FaceAnalysis(name=self.face_model_name, root=self.face_model_root, providers=providers)
        app.prepare(
            ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
            det_size=(self.det_size, self.det_size),
            det_thresh=self.det_thresh,
        )
        return app, providers

    @staticmethod
    def _normalized_embedding(face):
        import numpy as np

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 1e-8 else embedding

    def _detect_reference_faces(self, image):
        import cv2

        faces = self.app.get(image)
        if faces:
            return faces


        for ratio in (0.25, 0.5, 0.75):
            height, width = image.shape[:2]
            pad = max(16, int(round(max(height, width) * ratio)))
            padded = cv2.copyMakeBorder(
                image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            faces = self.app.get(padded)
            if faces:
                return faces
        return []

    def _reference_embedding(self, reference_path: Path):
        import cv2

        image = cv2.imread(str(reference_path))
        if image is None:
            raise RuntimeError(f"cannot read reference image: {reference_path}")
        faces = self._detect_reference_faces(image)
        if not faces:
            raise RuntimeError(f"no face detected in the reference image: {reference_path}")
        faces = sorted(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
            reverse=True,
        )
        embedding = self._normalized_embedding(faces[0])
        if embedding is None:
            raise RuntimeError(f"reference face has no embedding: {reference_path}")
        return embedding

    def _detect_frame_face(self, frame_path: Path, reference_embedding, threshold: float) -> FrameFaceInfo:
        import cv2

        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"cannot read frame: {frame_path}")
        faces = self.app.get(image)
        if not faces:
            return FrameFaceInfo(False, False, -1.0)

        best = -1.0
        for face in faces:
            embedding = self._normalized_embedding(face)
            if embedding is not None:
                best = max(best, float(self._np.dot(reference_embedding, embedding)))
        return FrameFaceInfo(True, best >= threshold, best)

    def _choose_source_frame(
        self,
        anchor_idx: int,
        frame_paths: list[Path],
        face_cache: dict[int, FrameFaceInfo],
        used_indices: set[int],
        reference_embedding,
    ) -> tuple[int, str, FrameFaceInfo]:
        def info(frame_idx: int) -> FrameFaceInfo:
            if frame_idx not in face_cache:
                face_cache[frame_idx] = self._detect_frame_face(
                    frame_paths[frame_idx], reference_embedding, self.similarity_threshold
                )
            return face_cache[frame_idx]

        order = [idx for idx in nearest_indices(anchor_idx, len(frame_paths)) if idx not in used_indices]
        if not order:
            order = nearest_indices(anchor_idx, len(frame_paths))

        first_idx = order[0]
        first_info = info(first_idx)
        if first_info.is_target:
            reason = "anchor_target" if first_idx == anchor_idx else "nearby_unused_target"
            return first_idx, reason, first_info

        if not first_info.has_face:
            for frame_idx in order[1:]:
                frame_info = info(frame_idx)
                if frame_info.is_target:
                    return frame_idx, "nearby_target_after_no_face", frame_info
            for frame_idx in order[1:]:
                frame_info = info(frame_idx)
                if frame_info.has_face:
                    return frame_idx, "nearby_face_no_target", frame_info
            return first_idx, "no_face_anywhere", first_info

        for frame_idx in order[1:]:
            frame_info = info(frame_idx)
            if frame_info.is_target:
                return frame_idx, "nearby_target_after_wrong_id", frame_info
        return first_idx, "wrong_id_no_target_found", first_info
