"""
GuardianEye - main entry point

Usage:
    python main.py              # normal run
    python main.py --debug      # dummy sensors + key simulation
    python main.py --preview    # show OpenCV preview window
"""

from __future__ import annotations

import argparse
import signal
import time
import threading

import cv2

from guardianeye.utils.logger import get_logger
from guardianeye.utils.config import CAPTURE_INTERVAL_S
from guardianeye.vision.realsense     import RealSensePreprocessor
from guardianeye.vision.pose_analyzer import PoseAnalyzer
from guardianeye.sensors.gpio_manager import SensorManager
from guardianeye.core.state_machine   import GuardianStateMachine, MonitorState
from guardianeye.alert.dispatcher     import AlertDispatcher

log = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GuardianEye - elderly inactivity detection")
    parser.add_argument("--debug",   action="store_true", help="debug mode (key simulation)")
    parser.add_argument("--preview", action="store_true", help="show OpenCV preview window")
    return parser.parse_args()


class GuardianEye:
    """Top-level orchestrator: initializes subsystems and runs the main loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args     = args
        self._shutdown = threading.Event()

        log.info("Initializing GuardianEye...")

        self._camera   = RealSensePreprocessor()
        self._analyzer = PoseAnalyzer()
        self._sensors  = SensorManager()
        self._alerter  = AlertDispatcher()
        self._fsm      = GuardianStateMachine(
            sensor_manager  = self._sensors,
            on_alert        = self._on_alert,
            on_state_change = self._on_state_change,
        )

        self._display_available = self._check_display()
        if self._args.preview and not self._display_available:
            log.warning("No display found - --preview disabled.")

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def run(self) -> None:
        try:
            self._startup()
            self._main_loop()
        finally:
            self._cleanup()

    def _startup(self) -> None:
        log.info("Starting subsystems...")
        self._sensors.start()
        self._camera.start()
        log.info("=== GuardianEye monitoring started ===")

    def _cleanup(self) -> None:
        log.info("Shutting down...")
        self._camera.stop()
        self._sensors.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        log.info("=== GuardianEye stopped ===")

    def _main_loop(self) -> None:
        log.info("Main loop started (interval: %ds)", CAPTURE_INTERVAL_S)

        while not self._shutdown.is_set():
            loop_start = time.time()

            culled_frame, depth_visual = self._camera.get_culled_frame()
            if culled_frame is None:
                log.warning("Frame capture failed - retrying...")
                time.sleep(1)
                continue

            vision_result = self._analyzer.analyze(culled_frame)
            state         = self._fsm.update(vision_result)

            self._log_status(vision_result, state)

            if self._args.preview and self._display_available:
                self._show_preview(culled_frame, depth_visual, vision_result, state)

            if self._args.debug:
                self._handle_debug_keys()

            sleep_time = max(0.0, CAPTURE_INTERVAL_S - (time.time() - loop_start))
            if sleep_time > 0:
                self._shutdown.wait(timeout=sleep_time)

    def _log_status(self, vision, state: MonitorState) -> None:
        temp, humid = self._sensors.get_env()
        rms         = self._sensors.is_sound_raw()
        env_str     = f"{temp:.1f}C {humid:.1f}%" if temp is not None else "n/a"

        if state == MonitorState.WAITING:
            log.info("[%s] person=%s lying=%s zone=%s env=%s rms=%.0f | remaining=%.1fmin",
                     state.name, vision.person_detected, vision.lying_detected,
                     vision.zone_name if vision.lying_detected else "-",
                     env_str, rms, self._fsm.remaining_seconds() / 60)
        else:
            log.info("[%s] person=%s lying=%s zone=%s env=%s rms=%.0f",
                     state.name, vision.person_detected, vision.lying_detected,
                     vision.zone_name if vision.lying_detected else "-",
                     env_str, rms)

    def _show_preview(self, culled, depth_visual, vision, state: MonitorState) -> None:
        state_colors = {
            MonitorState.IDLE:    (0, 255, 0),
            MonitorState.WAITING: (0, 200, 255),
            MonitorState.BUZZING: (0, 100, 255),
            MonitorState.ALERT:   (0, 0, 255),
        }
        color = state_colors.get(state, (255, 255, 255))
        label = f"State: {state.name}"
        if state == MonitorState.WAITING:
            temp, humid = self._sensors.get_env()
            env_str = f" | {temp:.1f}C {humid:.1f}%" if temp is not None else ""
            label += f"  Remaining: {self._fsm.remaining_seconds()/60:.1f}min{env_str}"

        annotated = culled.copy()
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if vision.lying_detected:
            cv2.putText(annotated, f"LYING | {vision.zone_name}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("GuardianEye | Culled Color  +  Depth Map",
                   cv2.hconcat([annotated, depth_visual]))
        cv2.waitKey(1)

    def _handle_debug_keys(self) -> None:
        """Key bindings (requires OpenCV window focus): q = quit"""
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            log.info("Quit requested.")
            self._shutdown.set()

    def _check_display(self) -> bool:
        try:
            import os
            if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                return True
            cv2.namedWindow("_test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_test")
            return True
        except Exception:
            return False

    def _on_alert(self, zone_name: str) -> None:
        self._alerter.dispatch(zone_name)

    def _on_state_change(self, old: MonitorState, new: MonitorState) -> None:
        log.info("State change: %s -> %s", old.name, new.name)

    def _signal_handler(self, signum, frame) -> None:
        log.info("Signal %d received, shutting down.", signum)
        self._shutdown.set()


if __name__ == "__main__":
    GuardianEye(parse_args()).run()
