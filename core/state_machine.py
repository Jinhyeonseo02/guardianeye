from __future__ import annotations

import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Optional

from guardianeye.utils.config import (
    TIMER_DANGER_SILENT, TIMER_DANGER_SOUND,
    TIMER_SAFE_SILENT, TIMER_SAFE_SOUND,
    BUZZER_CONFIRM_WAIT,
)
from guardianeye.utils.logger import get_logger
from guardianeye.sensors.gpio_manager import SensorManager
from guardianeye.vision.pose_analyzer import AnalysisResult

log = get_logger(__name__)


class MonitorState(Enum):
    IDLE    = auto()
    WAITING = auto()
    BUZZING = auto()
    ALERT   = auto()


@dataclass
class MonitorContext:
    state:         MonitorState   = MonitorState.IDLE
    timer_total_s: float          = 0.0
    timer_start:   float          = 0.0
    is_safe_zone:  bool           = False
    zone_name:     str            = "danger"
    buzzer_start:  float          = 0.0
    alert_sent:    bool           = False
    lock:          threading.Lock = field(default_factory=threading.Lock)


class GuardianStateMachine:
    """
    Core 4-stage monitoring state machine.

    State flow:
      IDLE    --[lying detected]----------->  WAITING
      WAITING --[posture cleared]---------->  IDLE     (reset)
      WAITING --[timer expired]------------>  BUZZING
      BUZZING --[posture cleared]---------->  IDLE     (safe)
      BUZZING --[60s no response]---------->  ALERT
      ALERT   --[alert sent]--------------->  IDLE

    Timer duration is multiplied by the DHT environment risk factor:
    extreme heat or cold shortens the timer so an alert fires sooner.
    """

    def __init__(
        self,
        sensor_manager:  SensorManager,
        on_alert:        Callable[[str], None],
        on_state_change: Optional[Callable[[MonitorState, MonitorState], None]] = None,
    ) -> None:
        self._sm        = sensor_manager
        self._alert_cb  = on_alert
        self._change_cb = on_state_change
        self._ctx       = MonitorContext()

    def update(self, vision: AnalysisResult) -> MonitorState:
        with self._ctx.lock:
            s = self._ctx.state
            if   s == MonitorState.IDLE:    self._handle_idle(vision)
            elif s == MonitorState.WAITING: self._handle_waiting(vision)
            elif s == MonitorState.BUZZING: self._handle_buzzing(vision)
            elif s == MonitorState.ALERT:   self._handle_alert()
            return self._ctx.state

    def _handle_idle(self, vision: AnalysisResult) -> None:
        if not vision.lying_detected:
            return

        sound      = self._sm.is_sound_detected()
        is_safe    = vision.is_safe_zone
        base_timer = self._select_timer(is_safe, sound)
        env_mult   = self._sm.get_timer_multiplier()
        duration   = base_timer * env_mult

        self._ctx.is_safe_zone  = is_safe
        self._ctx.zone_name     = vision.zone_name
        self._ctx.timer_total_s = duration
        self._ctx.timer_start   = time.time()

        temp, humid = self._sm.get_env()
        log.info(
            "Lying detected | zone: %s | sound: %s | env: %.1f°C %.1f%% | "
            "timer: %.0fs (base %.0fs x %.2f)",
            vision.zone_name, sound,
            temp or 0, humid or 0,
            duration, base_timer, env_mult,
        )
        self._transition(MonitorState.WAITING)

    def _handle_waiting(self, vision: AnalysisResult) -> None:
        if not vision.lying_detected:
            log.info("Posture cleared -> timer reset.")
            self._reset_to_idle()
            return

        # Re-evaluate environment every cycle and shrink remaining time if needed
        env_mult  = self._sm.get_timer_multiplier()
        remaining = self._ctx.timer_total_s - (time.time() - self._ctx.timer_start)
        log.debug("Timer remaining: %.1f s (env x%.2f)", remaining, env_mult)

        if remaining <= 0:
            temp, humid = self._sm.get_env()
            log.warning(
                "Timer expired! zone: %s | env: %.1f°C %.1f%% -> activating buzzer.",
                self._ctx.zone_name, temp or 0, humid or 0,
            )
            self._ctx.buzzer_start = time.time()
            self._sm.beep_alert(blocking=False)
            self._transition(MonitorState.BUZZING)

    def _handle_buzzing(self, vision: AnalysisResult) -> None:
        if not vision.lying_detected:
            log.info("Posture cleared during buzzer -> safe, resetting.")
            self._sm.beep_stop()
            self._reset_to_idle()
            return

        if time.time() - self._ctx.buzzer_start >= BUZZER_CONFIRM_WAIT:
            log.critical(
                "No response for %ds -> emergency confirmed! zone: %s",
                BUZZER_CONFIRM_WAIT, self._ctx.zone_name,
            )
            self._transition(MonitorState.ALERT)

    def _handle_alert(self) -> None:
        if not self._ctx.alert_sent:
            self._ctx.alert_sent = True
            threading.Thread(
                target=self._alert_cb,
                args=(self._ctx.zone_name,),
                daemon=True,
            ).start()
        self._reset_to_idle()

    @staticmethod
    def _select_timer(is_safe: bool, has_sound: bool) -> float:
        if   not is_safe and not has_sound: return TIMER_DANGER_SILENT
        elif not is_safe and has_sound:     return TIMER_DANGER_SOUND
        elif is_safe     and not has_sound: return TIMER_SAFE_SILENT
        else:                               return TIMER_SAFE_SOUND

    def _reset_to_idle(self) -> None:
        self._ctx.timer_start   = 0.0
        self._ctx.timer_total_s = 0.0
        self._ctx.alert_sent    = False
        self._transition(MonitorState.IDLE)

    def _transition(self, new_state: MonitorState) -> None:
        old = self._ctx.state
        if old == new_state:
            return
        self._ctx.state = new_state
        log.info("State: %s -> %s", old.name, new_state.name)
        if self._change_cb:
            try:
                self._change_cb(old, new_state)
            except Exception as e:
                log.error("State change callback error: %s", e)

    @property
    def state(self) -> MonitorState:
        return self._ctx.state

    def remaining_seconds(self) -> float:
        if self._ctx.state != MonitorState.WAITING:
            return 0.0
        env_mult = self._sm.get_timer_multiplier()
        return max(0.0, self._ctx.timer_total_s - (time.time() - self._ctx.timer_start))
