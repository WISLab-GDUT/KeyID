from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Sequence


def intersection_fraction(outer: Sequence[float], face: Sequence[float]) -> float:
    area = (face[2] - face[0]) * (face[3] - face[1])
    intersection = max(0.0, min(outer[2], face[2]) - max(outer[0], face[0])) * max(
        0.0, min(outer[3], face[3]) - max(outer[1], face[1])
    )
    return intersection / area if area > 0 else 0.0


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (left[2]-left[0]) * (left[3]-left[1]) + (right[2]-right[0]) * (right[3]-right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def crop_bounds(box: Sequence[float], width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    dx, dy = (box[2]-box[0]) * padding, (box[3]-box[1]) * padding
    return (
        max(0, min(width-1, math.floor((box[0]-dx) * width))),
        max(0, min(height-1, math.floor((box[1]-dy) * height))),
        max(1, min(width, math.ceil((box[2]+dx) * width))),
        max(1, min(height, math.ceil((box[3]+dy) * height))),
    )


class FaceRefiner:
    def __init__(self, cv2: Any, detector: Any, *, padding: float, max_candidates: int,
                 minimum_size: int = 16, person_overlap: float = 0.85,
                 roi_padding: float = 0.05, duplicate_iou: float = 0.50,
                 anchor_iou: float = 0.40, cross_subject_iou: float = 0.55,
                 padding_candidates: Sequence[float] | None = None) -> None:
        self.cv2, self.detector = cv2, detector
        self.padding, self.max_candidates = padding, max_candidates
        self.padding_candidates = tuple(padding_candidates or (padding,))
        self.minimum_size, self.person_overlap = minimum_size, person_overlap
        self.roi_padding, self.duplicate_iou = roi_padding, duplicate_iou
        self.anchor_iou, self.cross_subject_iou = anchor_iou, cross_subject_iou

    def _convert(self, raw: dict, width: int, height: int, origin=(0, 0)) -> tuple[dict | None, str | None]:
        try:
            bbox = [float(v) for v in raw["bbox"]]
            score = float(raw["score"])
            landmarks = [[float(v) for v in p] for p in raw["landmarks"]]
        except (KeyError, TypeError, ValueError):
            return None, "invalid_detector_output"
        if len(bbox) != 4 or len(landmarks) != 5 or any(len(p) != 2 for p in landmarks):
            return None, "invalid_detector_output"
        if not all(math.isfinite(v) for v in [*bbox, score, *(v for p in landmarks for v in p)]):
            return None, "nonfinite_detector_output"
        if not 0 <= score <= 1:
            return None, "invalid_detector_score"
        if score < self.detector.score:
            return None, "low_detector_score"
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            return None, "invalid_face_box"
        ox, oy = origin
        bbox = [max(0, x1+ox), max(0, y1+oy), min(width, x2+ox), min(height, y2+oy)]
        if bbox[2]-bbox[0] < self.minimum_size or bbox[3]-bbox[1] < self.minimum_size:
            return None, "face_too_small"
        points = [[p[0]+ox, p[1]+oy] for p in landmarks]
        tolerance_x, tolerance_y = (bbox[2]-bbox[0]) * 0.25, (bbox[3]-bbox[1]) * 0.25
        if any(not (bbox[0]-tolerance_x <= x <= bbox[2]+tolerance_x and
                    bbox[1]-tolerance_y <= y <= bbox[3]+tolerance_y) for x, y in points):
            return None, "landmarks_outside_face"
        return {
            "face_bbox": [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height],
            "detector_score": score,
            "landmarks": [[x/width, y/height] for x, y in points],
        }, None

    def _matches(self, person_box: Sequence[float], faces: list[dict]) -> list[dict]:
        return [face for face in faces if
                person_box[0] <= (face["face_bbox"][0]+face["face_bbox"][2])/2 <= person_box[2]
                and person_box[1] <= (face["face_bbox"][1]+face["face_bbox"][3])/2 <= person_box[3]
                and intersection_fraction(person_box, face["face_bbox"]) >= self.person_overlap]

    def refine(self, semantic: dict, frames: list[dict], debug_dir: Path, relative) -> tuple[dict, dict]:
        binding = copy.deepcopy(semantic)
        images, records = {}, {}
        diagnostic = {"schema_version": 1, "status": "succeeded", "detector": self.detector.info,
                      "frames": [], "subjects": [], "errors": []}
        for frame in frames:
            frame_id = frame["frame_id"]
            image = self.cv2.imread(str(frame["path"]))
            if image is None:
                raise OSError(f"cannot read detection frame: {frame['path']}")
            images[frame_id] = image
            height, width = image.shape[:2]
            record = {"frame_id": frame_id, "detections": [], "rejected_detections": [], "roi_searches": []}
            records[frame_id] = record
            diagnostic["frames"].append(record)
            self._add_detections(record, self.detector.detect(image), width, height, "full_frame")


        for subject in binding["subjects"]:
            for candidate in subject["candidates"]:
                frame_id = candidate["frame_id"]
                record, image = records[frame_id], images[frame_id]
                if self._matches(candidate["person_bbox"], record["detections"]):
                    continue
                height, width = image.shape[:2]
                left, top, right, bottom = crop_bounds(candidate["person_bbox"], width, height, self.roi_padding)
                record["roi_searches"].append({"subject_id": subject["subject_id"], "pixel_box": [left, top, right, bottom]})
                raw = self.detector.detect(image[top:bottom, left:right])
                self._add_detections(record, raw, width, height, f"person_roi:{subject['subject_id']}", (left, top))

        owners = {}
        for subject in binding["subjects"]:
            for candidate in subject["candidates"]:
                for face in self._matches(candidate["person_bbox"], records[candidate["frame_id"]]["detections"]):
                    owners.setdefault(face["face_id"], set()).add(subject["subject_id"])

        claimed = {}
        for subject in binding["subjects"]:
            accepted = []
            report = {"subject_id": subject["subject_id"], "source_image_index": subject["source_image_index"], "candidates": []}
            diagnostic["subjects"].append(report)
            for candidate in subject["candidates"]:
                frame_id = candidate["frame_id"]
                image, record = images[frame_id], records[frame_id]
                matches = self._matches(candidate["person_bbox"], record["detections"])
                entry = {"frame_id": frame_id, "person_bbox": candidate["person_bbox"],
                         "matching_face_ids": [f["face_id"] for f in matches], "status": "rejected"}
                report["candidates"].append(entry)
                if not matches:
                    entry["reason"] = "no_detected_face_in_person"
                    continue


                face = max(matches, key=lambda f: (f["face_bbox"][2] - f["face_bbox"][0])
                           * (f["face_bbox"][3] - f["face_bbox"][1]))
                if len(owners[face["face_id"]]) != 1:
                    entry["reason"] = "face_claimed_by_multiple_subjects"
                    continue
                if any(box_iou(face["face_bbox"], other) > self.cross_subject_iou for other in claimed.get(frame_id, [])):
                    entry["reason"] = "cross_subject_face_overlap"
                    continue
                height, width = image.shape[:2]
                crop_face = None
                selected_padding = None
                padding_attempts = []
                debug_dir.mkdir(parents=True, exist_ok=True)
                for padding in self.padding_candidates:
                    left, top, right, bottom = crop_bounds(
                        face["face_bbox"], width, height, padding
                    )
                    crop = image[top:bottom, left:right]
                    crop_path = debug_dir / (
                        f"{subject['subject_id']}_{frame_id}_padding_{padding:.3f}.png"
                    )
                    attempt = {
                        "padding": padding,
                        "candidate_crop": relative(crop_path),
                    }
                    padding_attempts.append(attempt)
                    if crop.shape[0] < 16 or crop.shape[1] < 16:
                        attempt["reason"] = "anchor_crop_too_small"
                        continue
                    if not self.cv2.imwrite(str(crop_path), crop):
                        raise OSError(f"cannot write candidate crop: {crop_path}")
                    crop_faces = []
                    for raw in self.detector.detect(crop):
                        converted, reason = self._convert(raw, width, height, (left, top))
                        if converted is not None:
                            crop_faces.append(converted)
                    attempt["detected_face_count"] = len(crop_faces)
                    if len(crop_faces) != 1:
                        attempt["reason"] = "anchor_requires_exactly_one_face"
                        continue
                    if box_iou(crop_faces[0]["face_bbox"], face["face_bbox"]) < self.anchor_iou:
                        attempt["reason"] = "anchor_detection_does_not_match_selected_face"
                        continue
                    attempt["status"] = "accepted"
                    crop_face = crop_faces[0]
                    selected_padding = padding
                    entry["candidate_crop"] = relative(crop_path)
                    break
                entry["padding_attempts"] = padding_attempts
                if crop_face is None or selected_padding is None:
                    last = padding_attempts[-1]
                    entry["candidate_crop"] = last["candidate_crop"]
                    entry["anchor_detected_face_count"] = last.get("detected_face_count", 0)
                    entry["reason"] = last.get("reason", "anchor_requires_exactly_one_face")
                    continue
                entry["anchor_detected_face_count"] = 1
                refined = {**candidate, **face, "face_visibility": "visible", "face_bbox_source": "local_detector",
                           "anchor_detector_score": crop_face["detector_score"],
                           "anchor_padding": selected_padding}
                accepted.append(refined)
                claimed.setdefault(frame_id, []).append(face["face_bbox"])
                entry.update(status="accepted", face_id=face["face_id"], face_bbox=face["face_bbox"],
                             reason="unique_detected_face" if len(matches) == 1 else "largest_of_multiple_faces")
                if len(accepted) >= self.max_candidates:
                    break
            subject["candidates"] = accepted
            if not accepted:
                diagnostic["status"] = "failed"
                diagnostic["errors"].append(f"{subject['subject_id']}: no reliable local face anchor")
        return binding, diagnostic

    def _add_detections(self, record: dict, raw_faces: list[dict], width: int, height: int, source: str, origin=(0, 0)) -> None:
        converted = []
        for raw in raw_faces:
            face, reason = self._convert(raw, width, height, origin)
            if face is None:
                record["rejected_detections"].append({"source": source, "reason": reason})
            else:
                converted.append(face)
        for face in sorted(converted, key=lambda f: tuple(f["face_bbox"])):
            duplicates = [existing for existing in record["detections"]
                          if box_iou(existing["face_bbox"], face["face_bbox"]) >= self.duplicate_iou]
            if duplicates:
                existing = duplicates[0]
                existing["sources"].append(source)
                if face["detector_score"] > existing["detector_score"]:
                    existing.update(face)
                continue
            face.update(face_id=f"{record['frame_id']}:face_{len(record['detections']):03d}", sources=[source])
            record["detections"].append(face)
