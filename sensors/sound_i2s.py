"""
GuardianEye - USB microphone sound detector.

(Originally written for INMP441 I2S; switched to USB mic due to soldering issues.)
Captures a short audio block via `arecord` and computes its RMS level
to decide whether sound is present.

Setup:
    USB microphone plugged into any USB port.
    Verify with `arecord -l` and set I2S_DEVICE in config.py to its plughw:N card.
"""

from __future__ import annotations

import subprocess
import wave
import io

import numpy as np

from guardianeye.utils.config import (
    I2S_SAMPLE_RATE, I2S_CHANNELS, I2S_DEVICE, I2S_FORMAT,
    SOUND_WINDOW_SEC, SOUND_RMS_THRESHOLD,
)
from guardianeye.utils.logger import get_logger

log = get_logger(__name__)


# Map ALSA format string -> numpy dtype
_DTYPE_MAP = {
    "S16_LE": np.int16,
    "S24_LE": np.int32,   # 24-bit usually padded into 32-bit
    "S32_LE": np.int32,
}


class I2SSoundDetector:
    """Captures audio from the USB mic and reports sound presence via RMS."""

    def __init__(self) -> None:
        self._available = self._check_device()
        self._last_rms = 0.0

    def _check_device(self) -> bool:
        """Verify an ALSA capture device exists."""
        try:
            out = subprocess.run(
                ["arecord", "-l"],
                capture_output=True, text=True, timeout=5,
            )
            if "card" in out.stdout:
                log.info("ALSA capture device found.")
                return True
            log.warning("No ALSA capture device found (arecord -l empty).")
            return False
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            log.warning("arecord not available (%s) - sound detection disabled.", e)
            return False

    def _capture_rms(self) -> float:
        """Record a short block and return its RMS amplitude (DC-removed)."""
        if not self._available:
            return 0.0

        cmd = [
            "arecord",
            "-D", I2S_DEVICE,
            "-c", str(I2S_CHANNELS),
            "-r", str(I2S_SAMPLE_RATE),
            "-f", I2S_FORMAT,
            "-d", str(int(max(1, round(SOUND_WINDOW_SEC)))),
            "-t", "wav",
            "-q",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=SOUND_WINDOW_SEC + 5,
            )
            if proc.returncode != 0 or not proc.stdout:
                log.debug("arecord returned no data.")
                return 0.0

            with wave.open(io.BytesIO(proc.stdout), "rb") as wf:
                frames = wf.readframes(wf.getnframes())

            if not frames:
                return 0.0

            dtype = _DTYPE_MAP.get(I2S_FORMAT, np.int16)
            samples = np.frombuffer(frames, dtype=dtype).astype(np.float64)
            if samples.size == 0:
                return 0.0

            # Remove DC offset, then compute RMS amplitude
            samples = samples - samples.mean()
            rms = float(np.sqrt(np.mean(samples ** 2)))
            return rms

        except subprocess.TimeoutExpired:
            log.warning("Audio capture timed out.")
            return 0.0
        except Exception as e:
            log.error("Audio capture error: %s", e)
            return 0.0

    def is_sound_detected(self) -> bool:
        """
        Capture a block and return True if RMS exceeds the threshold.
        Blocks for ~SOUND_WINDOW_SEC, so call only when setting a timer
        (not every main-loop cycle).
        """
        rms = self._capture_rms()
        self._last_rms = rms
        result = rms >= SOUND_RMS_THRESHOLD
        log.debug("sound RMS=%.1f threshold=%.1f -> %s",
                  rms, SOUND_RMS_THRESHOLD, result)
        return result

    @property
    def last_rms(self) -> float:
        return self._last_rms
