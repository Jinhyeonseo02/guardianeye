"""
GuardianEye - Safe zone setter (drag rectangles on the camera image).
 
Captures one frame from the RealSense camera, then lets you drag rectangles
on it to define safe zones (bed, sofa, etc.). Saves to ~/guardianeye/zones.json,
which config.py auto-loads on the next run.
 
Usage (VNC display required):
    cd ~
    python3 -m guardianeye.zone_setter
 
Controls:
    Left-click & drag : draw a rectangle
    Release           : prompt for zone name (in terminal)
    u                 : undo last rectangle
    c                 : clear all
    s                 : save zones.json and quit
    q / ESC           : quit without saving
"""
 
from __future__ import annotations
 
import json
import os
import sys
 
import cv2
import numpy as np
 
from guardianeye.vision.realsense import RealSensePreprocessor
from guardianeye.utils.logger import get_logger
 
log = get_logger("zone_setter")
 
OUTPUT_PATH = os.path.expanduser("~/guardianeye/zones.json")
 
 
class ZoneSetter:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.zones: list[dict] = []   # [{name, x1, y1, x2, y2}, ...] normalized
        self._drag_start = None
        self._drag_end = None
        self._dragging = False
 
    # ---------- mouse ----------
    def _on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._drag_end = (x, y)
            self._dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self._drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._dragging:
            self._dragging = False
            self._drag_end = (x, y)
            self._finalize_rect()
 
    def _finalize_rect(self) -> None:
        if self._drag_start is None or self._drag_end is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = self._drag_end
        if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
            self._drag_start = self._drag_end = None
            return
 
        # Normalize 0..1 and ensure x1<x2, y1<y2
        nx1 = max(0.0, min(x1, x2) / (self.w - 1))
        nx2 = min(1.0, max(x1, x2) / (self.w - 1))
        ny1 = max(0.0, min(y1, y2) / (self.h - 1))
        ny2 = min(1.0, max(y1, y2) / (self.h - 1))
 
        # Ask for name in terminal (cv2 has no text input)
        print(f"\nNew zone drawn: ({nx1:.2f}, {ny1:.2f}) -> ({nx2:.2f}, {ny2:.2f})")
        try:
            name = input("Zone name (e.g. bed / sofa) [empty=cancel]: ").strip()
        except EOFError:
            name = ""
 
        if name:
            self.zones.append({"name": name, "x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2})
            print(f"Added '{name}'.  Total zones: {len(self.zones)}")
        else:
            print("Cancelled.")
        self._drag_start = self._drag_end = None
 
    # ---------- drawing ----------
    def _render(self) -> np.ndarray:
        out = self.frame.copy()
        # existing zones
        for z in self.zones:
            p1 = (int(z["x1"] * (self.w - 1)), int(z["y1"] * (self.h - 1)))
            p2 = (int(z["x2"] * (self.w - 1)), int(z["y2"] * (self.h - 1)))
            cv2.rectangle(out, p1, p2, (0, 200, 255), 2)
            cv2.putText(out, z["name"], (p1[0] + 4, p1[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        # current drag
        if self._dragging and self._drag_start and self._drag_end:
            cv2.rectangle(out, self._drag_start, self._drag_end, (0, 255, 0), 2)
        # help text
        cv2.rectangle(out, (0, 0), (self.w, 26), (0, 0, 0), -1)
        cv2.putText(out, "drag=draw  u=undo  c=clear  s=save  q=quit",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out
 
    # ---------- main loop ----------
    def run(self) -> bool:
        win = "GuardianEye - Zone Setter"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)
 
        while True:
            cv2.imshow(win, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print("Quit without saving.")
                return False
            if key == ord("u") and self.zones:
                removed = self.zones.pop()
                print(f"Removed '{removed['name']}'.  Total: {len(self.zones)}")
            elif key == ord("c"):
                self.zones.clear()
                print("Cleared all zones.")
            elif key == ord("s"):
                self._save()
                cv2.destroyAllWindows()
                return True
 
    def _save(self) -> None:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(self.zones, f, indent=2)
            
        print(f"\nSaved {len(self.zones)} zones to {OUTPUT_PATH}")
        for z in self.zones:
            print(f"  {z['name']:>8}  ({z['x1']:.2f}, {z['y1']:.2f}) -> ({z['x2']:.2f}, {z['y2']:.2f})")
 
 
def main() -> int:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("ERROR: No display detected. Run this from the VNC desktop (not SSH).")
        return 1
 
    log.info("Capturing one frame from the camera...")
    cam = RealSensePreprocessor()
    cam.start()
    frame = None
    for _ in range(5):
        culled, _ = cam.get_culled_frame()
        if culled is not None:
            frame = culled
    cam.stop()
 
    if frame is None:
        log.error("Could not capture a frame.")
        return 1
 
    log.info("Camera frame captured. Opening editor...")
    print("\nDrag rectangles on the image to define safe zones.")
    print("Each zone needs a short name (bed / sofa / etc.).")
 
    setter = ZoneSetter(frame)
    setter.run()
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
			
