import numpy as np
import cv2
from typing import Optional, Tuple

from guardianeye.utils.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, DEPTH_CULLING_MM
)
from guardianeye.utils.logger import get_logger

log = get_logger(__name__)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    log.warning("pyrealsense2 not found - running in dummy mode.")


class RealSensePreprocessor:
    """
    Manages D435i stream init, frame capture, and depth culling.
    Falls back to webcam or blank frames when RealSense is unavailable.
    """

    def __init__(self) -> None:
        self._pipeline  = None
        self._align     = None
        self._dummy_cap = None
        self._running   = False
        self._use_dummy = not REALSENSE_AVAILABLE

    def start(self) -> None:
        if self._use_dummy:
            self._start_dummy()
            return

        try:
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color,
                              CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
            cfg.enable_stream(rs.stream.depth,
                              CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16,  CAMERA_FPS)

            profile = self._pipeline.start(cfg)
            self._log_device_info(profile)

            # Align depth frame to color frame coordinate space
            self._align = rs.align(rs.stream.color)

            log.info("Warming up camera (30 frames)...")
            for _ in range(30):
                self._pipeline.wait_for_frames(timeout_ms=2000)

            self._running = True
            log.info("RealSense stream started. Culling threshold: %dmm", DEPTH_CULLING_MM)

        except Exception as e:
            log.warning("RealSense init failed (%s) - switching to dummy mode.", e)
            self._use_dummy = True
            self._start_dummy()

    def stop(self) -> None:
        if self._pipeline and self._running:
            self._pipeline.stop()
        if self._dummy_cap:
            self._dummy_cap.release()
        self._running = False
        log.info("Camera stream stopped.")

    def get_culled_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
            culled_bgr   : depth-culled BGR frame ready for YOLO inference
            depth_visual : JET colormap depth image for debug (can be ignored in production)
        """
        if self._use_dummy:
            return self._get_dummy_frame()

        if not self._running:
            log.error("Pipeline not running. Call start() first.")
            return None, None

        try:
            frameset    = self._pipeline.wait_for_frames(timeout_ms=5000)
            aligned     = self._align.process(frameset)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                log.warning("Invalid frame received.")
                return None, None

            color_img = np.asanyarray(color_frame.get_data())  # (H,W,3) uint8
            depth_img = np.asanyarray(depth_frame.get_data())  # (H,W)   uint16 mm

            return self._apply_culling(color_img, depth_img), self._depth_colormap(depth_img)

        except RuntimeError as e:
            log.error("Frame capture error: %s", e)
            return None, None

    def _apply_culling(self, color: np.ndarray, depth: np.ndarray) -> np.ndarray:
        # depth == 0: measurement failure (glass/reflective surfaces) -> also mask out
        # depth > DEPTH_CULLING_MM: out of region of interest -> mask out
        mask   = (depth == 0) | (depth > DEPTH_CULLING_MM)
        result = color.copy()
        result[mask] = 0
        return result

    def _depth_colormap(self, depth: np.ndarray) -> np.ndarray:
        clipped    = np.clip(depth, 0, DEPTH_CULLING_MM).astype(np.float32)
        normalized = (clipped / DEPTH_CULLING_MM * 255).astype(np.uint8)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    def _start_dummy(self) -> None:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self._dummy_cap = cap
            log.info("Dummy mode: using webcam index 0.")
        else:
            cap.release()
            log.info("Dummy mode: no webcam found, generating blank frames.")
        self._running = True

    def _get_dummy_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._dummy_cap and self._dummy_cap.isOpened():
            ret, frame = self._dummy_cap.read()
            if ret:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                return frame, np.zeros_like(frame)
        blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        return blank, blank

    @staticmethod
    def _log_device_info(profile) -> None:
        try:
            dev = profile.get_device()
            log.info("Device: %s | S/N: %s | FW: %s",
                     dev.get_info(rs.camera_info.name),
                     dev.get_info(rs.camera_info.serial_number),
                     dev.get_info(rs.camera_info.firmware_version))
        except Exception:
            pass

