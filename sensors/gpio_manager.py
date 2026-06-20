from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from guardianeye.utils.config import (
    SOUND_RMS_FORCE, SOUND_RMS_THRESHOLD,
    GPIO_DHT, DHT_MODEL, GPIO_BUZZER, BUZZER_PATTERN, BUZZER_FREQ_HZ,
    SOUND_ENABLED, TEMP_HOT_C, TEMP_COLD_C, HUMID_HIGH, ENV_RISK_MULTIPLIER,
)
from guardianeye.utils.logger import get_logger
from guardianeye.sensors.sound_i2s import I2SSoundDetector

log = get_logger(__name__)

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    RPI_AVAILABLE = True
    log.info("RPi.GPIO loaded.")
except (ImportError, RuntimeError):
    RPI_AVAILABLE = False
    log.warning("RPi.GPIO not available - running in dummy mode.")

try:
    import adafruit_dht
    import board
    _DHT_PIN_OBJ = getattr(board, f"D{GPIO_DHT}")
    _dht_device  = adafruit_dht.DHT11(_DHT_PIN_OBJ) if DHT_MODEL == 11 else adafruit_dht.DHT22(_DHT_PIN_OBJ)
    DHT_AVAILABLE = True
    log.info("adafruit_dht loaded (DHT%d on GPIO %d).", DHT_MODEL, GPIO_DHT)
except Exception as e:
    DHT_AVAILABLE = False
    _dht_device  = None
    log.warning("adafruit_dht not available (%s) - environment sensing disabled.", e)


class SensorManager:
    """Controls Buzzer (GPIO PWM), DHT temp/humidity sensor, and USB mic."""

    def __init__(self) -> None:
        self._running     = False
        self._buzzer_lock = threading.Lock()
        self._pwm         = None
        self._sound       = I2SSoundDetector() if SOUND_ENABLED else None

        # DHT cache: read every 30s to avoid hammering the sensor
        self._dht_lock    = threading.Lock()
        self._last_temp   : Optional[float] = None
        self._last_humid  : Optional[float] = None
        self._dht_timer   : Optional[threading.Thread] = None

    def start(self) -> None:
        if RPI_AVAILABLE:
            GPIO.setup(GPIO_BUZZER, GPIO.OUT, initial=GPIO.LOW)
            self._pwm = GPIO.PWM(GPIO_BUZZER, BUZZER_FREQ_HZ)
            log.info("GPIO ready. Buzzer=%d", GPIO_BUZZER)

        self._running = True
        self._start_dht_polling()

    def stop(self) -> None:
        self._running = False
        if RPI_AVAILABLE:
            GPIO.output(GPIO_BUZZER, GPIO.LOW)
            GPIO.cleanup()
            log.info("GPIO cleaned up.")

    # ------------------------------------------------------------------
    # DHT sensor
    # ------------------------------------------------------------------

    def _start_dht_polling(self) -> None:
        def poll():
            while self._running:
                self._read_dht()
                time.sleep(30)
        t = threading.Thread(target=poll, daemon=True)
        t.start()

    def _read_dht(self) -> None:
        if not DHT_AVAILABLE or _dht_device is None:
            return
        for _ in range(3):
            try:
                temp  = _dht_device.temperature
                humid = _dht_device.humidity
                if temp is not None and humid is not None:
                    with self._dht_lock:
                        self._last_temp  = temp
                        self._last_humid = humid
                    log.debug("DHT read: temp=%.1f°C humid=%.1f%%", temp, humid)
                    return
            except RuntimeError:
                time.sleep(2)
            except Exception as e:
                log.warning("DHT read error: %s", e)
                return
        log.warning("DHT read failed after retries.")

    def get_env(self) -> tuple[Optional[float], Optional[float]]:
        """Return (temperature_C, humidity_%) from last DHT read."""
        with self._dht_lock:
            return self._last_temp, self._last_humid

    def get_timer_multiplier(self) -> float:
        """
        Return a multiplier for the monitoring timer based on room environment.
        Returns ENV_RISK_MULTIPLIER (< 1.0) when the room is dangerously hot or
        cold, so the timer is shortened and an alert fires sooner.
        Returns 1.0 when environment is normal or sensor data is unavailable.
        """
        temp, humid = self.get_env()
        if temp is None:
            return 1.0

        too_hot  = temp >= TEMP_HOT_C
        too_cold = temp <= TEMP_COLD_C
        humid_stress = (humid is not None) and (humid >= HUMID_HIGH)

        if too_cold:
            log.info("ENV: cold risk (%.1f°C) -> timer x%.1f", temp, ENV_RISK_MULTIPLIER)
            return ENV_RISK_MULTIPLIER
        if too_hot:
            multiplier = ENV_RISK_MULTIPLIER * (0.8 if humid_stress else 1.0)
            log.info("ENV: heat risk (%.1f°C humid=%.1f%%) -> timer x%.2f",
                     temp, humid or 0, multiplier)
            return multiplier
        return 1.0

    # ------------------------------------------------------------------
    # Sound
    # ------------------------------------------------------------------

    def is_sound_detected(self) -> bool:
        if SOUND_RMS_FORCE is not None:
            return SOUND_RMS_FORCE >= SOUND_RMS_THRESHOLD
        if not SOUND_ENABLED or self._sound is None:
            return False
        return self._sound.is_sound_detected()

    def is_sound_raw(self) -> float:
        if SOUND_RMS_FORCE is not None:
            return float(SOUND_RMS_FORCE)
        if not SOUND_ENABLED or self._sound is None:
            return 0.0
        return self._sound.last_rms

    # ------------------------------------------------------------------
    # Buzzer
    # ------------------------------------------------------------------

    def beep_alert(self, blocking: bool = True) -> None:
        if blocking:
            self._beep_sequence()
        else:
            threading.Thread(target=self._beep_sequence, daemon=True).start()

    def beep_stop(self) -> None:
        if RPI_AVAILABLE and self._pwm:
            self._pwm.stop()

    def _beep_sequence(self) -> None:
        on_sec, off_sec, repeat = BUZZER_PATTERN
        with self._buzzer_lock:
            for _ in range(repeat):
                if not self._running:
                    break
                if RPI_AVAILABLE and self._pwm:
                    self._pwm.start(50)
                else:
                    log.info("[BUZZER] tone %dHz", BUZZER_FREQ_HZ)
                time.sleep(on_sec)
                if RPI_AVAILABLE and self._pwm:
                    self._pwm.stop()
                time.sleep(off_sec)
