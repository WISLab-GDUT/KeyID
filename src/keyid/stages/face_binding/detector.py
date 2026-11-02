from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

from keyid.core.contract import StageContext
from keyid.core.errors import StageError


class LocalFaceDetector:
    def __init__(self, context: StageContext) -> None:
        model_path = Path(context.require_param("face_detector_model_path"))
        if not model_path.is_file():
            raise StageError(f"face_binding: detector model not found: {model_path}")
        size = context.param("face_detector_size", [640, 640])
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(
            type(v) is not int or v < 32 or v % 32 for v in size
        ):
            raise StageError("face_binding: face_detector_size must contain two positive multiples of 32")
        self.size = tuple(size)
        self.score = context.param("face_detector_score", 0.60)
        nms = context.param("face_detector_nms_threshold", 0.40)
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in (self.score, nms)):
            raise StageError("face_binding: detector score and NMS threshold must be finite numbers in (0, 1)")
        threads = context.param("face_detector_threads", 2)
        if type(threads) is not int or threads < 1:
            raise StageError("face_binding: face_detector_threads must be a positive integer")
        requested = context.param("face_detector_providers", ["cpu"])
        names = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
        if not isinstance(requested, (list, tuple)) or not requested or any(type(v) is not str or v not in names for v in requested):
            raise StageError("face_binding: face_detector_providers must be a non-empty list of cpu/cuda")
        fallback = bool(context.param("allow_cpu_fallback", False))
        try:
            import onnxruntime as ort
            from insightface.model_zoo.scrfd import SCRFD
        except ImportError as exc:
            raise StageError("face_binding: local detection needs insightface and onnxruntime in the stage interpreter") from exc
        providers = list(dict.fromkeys(names[v] for v in requested))
        available = ort.get_available_providers()
        missing = [v for v in providers if v not in available]
        if missing and not fallback:
            raise StageError(f"face_binding: requested detector providers unavailable: {missing}; available={available}")
        if missing:
            providers = ["CPUExecutionProvider"]
            context.logger.warning("face_binding detector explicitly falling back to CPU")
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.providers = list(self.session.get_providers())
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in self.providers and not fallback:
            raise StageError("face_binding: CUDA provider was requested but failed to initialize; refusing silent CPU fallback")
        self.model = SCRFD(model_file=str(model_path), session=self.session)
        if not self.model.use_kps:
            raise StageError("face_binding: detector must produce five facial landmarks")
        self.model.prepare(
            ctx_id=0 if "CUDAExecutionProvider" in self.providers else -1,
            input_size=self.size, det_thresh=float(self.score), nms_thresh=float(nms),
        )
        self._lock = threading.RLock()
        self.info = {
            "name": "scrfd", "model_path": str(model_path), "size": list(self.size),
            "score_threshold": float(self.score), "providers": self.providers,
        }
        context.logger.info("face_binding detector ready: model=%s size=%s providers=%s", model_path, self.size, self.providers)

    def detect(self, image: Any) -> list[dict[str, Any]]:


        with self._lock:
            boxes, landmarks = self.model.detect(image, input_size=self.size, max_num=0)
        if landmarks is None:
            return []
        return [
            {"bbox": [float(v) for v in box[:4]], "score": float(box[4]),
             "landmarks": [[float(v) for v in point] for point in points]}
            for box, points in zip(boxes, landmarks, strict=True)
        ]

    def close(self) -> None:
        self.model = None
        self.session = None
