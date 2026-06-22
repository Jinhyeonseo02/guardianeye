# GPIO pin numbers (BCM mode)
GPIO_DHT    = 4   # physical pin 11 (DHT11/22 data pin, previously PIR)
GPIO_BUZZER = 13   # physical pin 33 (moved from 18; I2S now uses 18/SCK)

# DHT sensor model: 11 or 22
DHT_MODEL = 11

# Temperature/humidity thresholds for timer weighting.
# If the room is dangerously hot or cold, the monitoring timer is shortened
# so an alert is triggered sooner.
TEMP_HOT_C  = 24.0   # above this -> heat risk (heatstroke in elderly)
TEMP_COLD_C = 16.0   # below this -> cold risk (hypothermia in elderly)
HUMID_HIGH  = 80.0   # above this + hot -> compound heat stress
# Multiplier applied to timer when extreme environment detected (< 1.0 = shorter)
ENV_RISK_MULTIPLIER = 0.5

# Sound detection via USB microphone.
SOUND_ENABLED = True

# USB mic settings (verified on AB13X USB Audio, card 2)
I2S_SAMPLE_RATE   = 44100
I2S_CHANNELS      = 1
I2S_DEVICE        = "plughw:2"
I2S_FORMAT        = "S16_LE"
SOUND_WINDOW_SEC  = 1.0
SOUND_RMS_THRESHOLD = 150

# Debug: force RMS to this value (None = real measurement)
SOUND_RMS_FORCE = None

# Camera / RealSense
CAMERA_WIDTH        = 640
CAMERA_HEIGHT       = 480
CAMERA_FPS          = 6
DEPTH_CULLING_MM    = 3000
CAPTURE_INTERVAL_S  = 5

# YOLO model
YOLO_MODEL_PATH     = "yolov8n-pose.pt"
YOLO_CONF_THRESHOLD = 0.5
POSE_LYING_RATIO    = 0.6

# Camera mount: "side" or "top" (ceiling).
CAMERA_MOUNT = "top"
TOP_LYING_SPREAD = 0.45

# COCO keypoint indices
KP_NOSE         = 0
KP_LEFT_ANKLE   = 15
KP_RIGHT_ANKLE  = 16

# Safe zones: normalized coords (x1, y1, x2, y2)
_DEFAULT_SAFE_ZONES = [
    ("bed",   0.55, 0.0,  1.0,  0.7),
    ("sofa",  0.0,  0.0,  0.45, 0.5),
]

def _load_zones():
    import json, os
    path = os.path.expanduser("~/guardianeye/zones.json")
    if not os.path.exists(path):
        return _DEFAULT_SAFE_ZONES
    try:
        with open(path) as f:
            data = json.load(f)
        return [(z["name"], z["x1"], z["y1"], z["x2"], z["y2"]) for z in data]
    except Exception:
        return _DEFAULT_SAFE_ZONES

SAFE_ZONES = _load_zones()

# Timers (seconds) — production values for real elderly monitoring deployment
# Demo filming used shortened values (8~30s); these reflect realistic inactivity windows.
TIMER_DANGER_SILENT = 600    # danger zone + silent  (10 min)
TIMER_DANGER_SOUND  = 1200   # danger zone + sound   (20 min)
TIMER_SAFE_SILENT   = 1800   # safe zone   + silent  (30 min)
TIMER_SAFE_SOUND    = 3600   # safe zone   + sound   (60 min)

# With ENV_RISK_MULTIPLIER=0.5 applied on heat/cold, effective minimums:
#   danger+silent+heat -> 300s (5 min)
#   danger+silent+heat+humid -> 240s (4 min)

BUZZER_CONFIRM_WAIT = 60   # seconds to wait after buzzer before sending alert

# Buzzer pattern (on_sec, off_sec, repeat)
BUZZER_PATTERN = (0.5, 0.5, 5)

# Buzzer tone frequency in Hz
BUZZER_FREQ_HZ = 440

# Alert settings
ALERT_WEBHOOK_URL  = ""
ALERT_EMAIL_TO     = ""
ALERT_EMAIL_FROM   = ""
ALERT_EMAIL_PASS   = ""
ALERT_SMTP_HOST    = "smtp.gmail.com"
ALERT_SMTP_PORT    = 587

# Telegram bot alerts
ALERT_TELEGRAM_TOKEN   = "8557719007:AAG6o58UEVsaU--3HpmSW_9YRsRUvjcWPgQ"
ALERT_TELEGRAM_CHAT_ID = "6314466046"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE  = "guardianeye.log"
