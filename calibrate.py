"""
GuardianEye - Zone calibration helper

Captures ONE frame from the RealSense camera, overlays a 0.0-1.0 grid,
and saves it as an image file. Open the image to read off the coordinates
for your bed / sofa / danger zones, then put them into config.py SAFE_ZONES.

Usage:
    cd ~
    python3 -m guardianeye.calibrate

Output:
    ~/guardianeye/zone_calibration.jpg
"""

from __future__ import annotations

import os
import numpy as np
import cv2

from guardianeye.vision.realsense import RealSensePreprocessor
from guardianeye.utils.config import SAFE_ZONES
from guardianeye.utils.logger import get_logger

log = get_logger("calibrate")

OUTPUT_PATH = os.path.expanduser("~/guardianeye/zone_calibration.jpg")
GRID_STEP = 0.1  # grid lines every 0.1 (10%)


def draw_grid(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    # Vertical lines + x labels
    step = 0.0
    while step <= 1.0001:
        x = int(step * (w - 1))
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 0), 1)
        cv2.putText(out, f"{step:.1f}", (min(x + 2, w - 30), 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        step += GRID_STEP

    # Horizontal lines + y labels
    step = 0.0
    while step <= 1.0001:
        y = int(step * (h - 1))
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)
        cv2.putText(out, f"{step:.1f}", (2, min(y + 14, h - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        step += GRID_STEP

    # Draw existing SAFE_ZONES as blue rectangles
    for name, x1, y1, x2, y2 in SAFE_ZONES:
        p1 = (int(x1 * (w - 1)), int(y1 * (h - 1)))
        p2 = (int(x2 * (w - 1)), int(y2 * (h - 1)))
        cv2.rectangle(out, p1, p2, (255, 100, 0), 2)
        cv2.putText(out, name, (p1[0] + 4, p1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    return out


def main() -> None:
    log.info("Starting camera for calibration...")
    cam = RealSensePreprocessor()
    cam.start()

    # Grab a few frames to let exposure settle, use the last one
    frame = None
    for _ in range(5):
        culled, _ = cam.get_culled_frame()
        if culled is not None:
            frame = culled

    cam.stop()

    if frame is None:
        log.error("Could not capture a frame. Is the camera connected?")
        return

    annotated = draw_grid(frame)
    cv2.imwrite(OUTPUT_PATH, annotated)
    log.info("Saved calibration image -> %s", OUTPUT_PATH)
    log.info("Open it to read off zone coordinates (green grid = 0.0-1.0).")
    log.info("Blue rectangles show your current SAFE_ZONES from config.py.")


if __name__ == "__main__":
    main()
