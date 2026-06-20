from __future__ import annotations
 
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
 
from guardianeye.utils.config import (
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD,
    POSE_LYING_RATIO, KP_NOSE, KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
    SAFE_ZONES, CAMERA_WIDTH, CAMERA_HEIGHT,
    CAMERA_MOUNT, TOP_LYING_SPREAD,
)
from guardianeye.utils.logger import get_logger
 
log = get_logger(__name__)
 
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.warning("ultralytics not found - running in dummy mode.")
 
 
@dataclass
class PersonState:
    is_lying:     bool  = False
    is_safe_zone: bool  = False
    zone_name:    str   = "danger"
    confidence:   float = 0.0
    bbox:         Tuple[float, float, float, float] = (0, 0, 0, 0)
    keypoints:    np.ndarray = field(default_factory=lambda: np.zeros((17, 3)))
 
 
@dataclass
class AnalysisResult:
    person_detected: bool               = False
    lying_detected:  bool               = False
    is_safe_zone:    bool               = False
    zone_name:       str                = "danger"
    persons:         List[PersonState]  = field(default_factory=list)
 
 
class PoseAnalyzer:
    """YOLOv8n-pose wrapper for lying-down detection and zone classification."""
 
    def __init__(self) -> None:
        self._model = None
        self._dummy = not YOLO_AVAILABLE
        if not self._dummy:
            self._load_model()
 
    def _load_model(self) -> None:
        try:
            log.info("Loading YOLO model: %s", YOLO_MODEL_PATH)
            self._model = YOLO(YOLO_MODEL_PATH)
            self._model.to("cpu")
            log.info("YOLO model loaded.")
        except Exception as e:
            log.error("YOLO model load failed: %s -> switching to dummy mode.", e)
            self._dummy = True
 
    def analyze(self, frame: np.ndarray) -> AnalysisResult:
        if self._dummy:
            return AnalysisResult()
 
        try:
            results = self._model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
            return self._parse_results(results, frame.shape)
        except Exception as e:
            log.error("YOLO inference error: %s", e)
            return AnalysisResult()
 
    def _parse_results(self, results, shape: Tuple[int, ...]) -> AnalysisResult:
        h, w     = shape[:2]
        analysis = AnalysisResult()
 
        if not results or results[0].keypoints is None:
            return analysis
 
        kps_data  = results[0].keypoints.data
        boxes     = results[0].boxes
        n_persons = len(kps_data)
 
        if n_persons == 0:
            return analysis
 
        analysis.person_detected = True
 
        for i in range(n_persons):
            kps       = kps_data[i].cpu().numpy()
            conf      = float(boxes.conf[i].cpu().numpy())
            bbox_xyxy = boxes.xyxy[i].cpu().numpy()
 
            if conf < YOLO_CONF_THRESHOLD:
                continue
            norm_bbox = (
                bbox_xyxy[0] / w, bbox_xyxy[1] / h,
                bbox_xyxy[2] / w, bbox_xyxy[3] / h,
            )
 
            is_lying            = self._check_lying(kps)
            zone_name, is_safe  = self._classify_zone(norm_bbox)
 
            analysis.persons.append(PersonState(
                is_lying=is_lying, is_safe_zone=is_safe,
                zone_name=zone_name, confidence=conf,
                bbox=norm_bbox, keypoints=kps,
            ))
 
        lying_persons = [p for p in analysis.persons if p.is_lying]
        if lying_persons:
            analysis.lying_detected = True
            dangerous = [p for p in lying_persons if not p.is_safe_zone]
            if dangerous:
                analysis.is_safe_zone = False
                analysis.zone_name    = dangerous[0].zone_name
            else:
                analysis.is_safe_zone = True
                analysis.zone_name    = lying_persons[0].zone_name
 
        return analysis
 
    def _check_lying(self, kps: np.ndarray) -> bool:
        if CAMERA_MOUNT == "top":
            return self._check_lying_top(kps)
        return self._check_lying_side(kps)
 
    def _check_lying_side(self, kps: np.ndarray) -> bool:
        """
        Side view: compare nose Y and ankle Y.
        Standing: head_y << ankle_y (ratio < POSE_LYING_RATIO)
        Lying:    head_y ~= ankle_y (ratio >= POSE_LYING_RATIO)
        """
        if kps[KP_NOSE, 2] < 0.3:
            return False
 
        head_y   = kps[KP_NOSE, 1]
        ankle_ys = [kps[idx, 1] for idx in (KP_LEFT_ANKLE, KP_RIGHT_ANKLE)
                    if kps[idx, 2] > 0.3]
 
        if not ankle_ys:
            return False
 
        ankle_y = max(ankle_ys)
        if ankle_y < 1e-3:
            return False
 
        ratio = head_y / ankle_y
        log.debug("side ratio head_y=%.1f ankle_y=%.1f ratio=%.3f", head_y, ankle_y, ratio)
        return ratio >= POSE_LYING_RATIO
 
    def _check_lying_top(self, kps: np.ndarray) -> bool:
        """
        Top-down (ceiling) view: measure how spread out the body is.
        Standing looks compact from above; lying stretches the keypoints
        across the floor. Compares the max distance between any two visible
        keypoints against the body's overall size.
        """
        # Collect visible keypoints (visibility > 0.3)
        pts = [(kps[i, 0], kps[i, 1]) for i in range(len(kps)) if kps[i, 2] > 0.3]
        if len(pts) < 3:
            return False
 
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
 
        # Bounding box of visible keypoints
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
 
        if bw < 1e-3 and bh < 1e-3:
            return False
 
        # Aspect ratio of the keypoint cloud:
        # lying = elongated (one side much longer), standing = roughly square/compact
        longer  = max(bw, bh)
        shorter = min(bw, bh)
        if longer < 1e-3:
            return False
 
        elongation = 1.0 - (shorter / longer)  # 0 = square, ->1 = very elongated
        log.debug("top view bw=%.1f bh=%.1f elongation=%.3f", bw, bh, elongation)
        return elongation >= TOP_LYING_SPREAD
 
    def _classify_zone(
        self, norm_bbox: Tuple[float, float, float, float]
    ) -> Tuple[str, bool]:
        cx = (norm_bbox[0] + norm_bbox[2]) / 2
        cy = (norm_bbox[1] + norm_bbox[3]) / 2
 
        for name, x1, y1, x2, y2 in SAFE_ZONES:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return name, True
 
        return "danger", False
