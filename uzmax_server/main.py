"""
UzMAX — Unified Medical AI + Robot Control Server
===================================================
Single FastAPI server that combines:
  • Voice AI assistant  (Yandex STT/TTS + OpenAI LLM)
  • Face recognition    (InsightFace embeddings + Qdrant vector store)
  • ESP32 robot control (HAND / HEAD / MOVE via USB serial)

Protocol for ESP32_HAND  (new firmware):
    R 1 90       → right servo #1 to 90°
    L 6 120      → left servo #6 to 120°

Protocol for ESP32_HEAD:
    HEAD LEFT 15       -> move head 15 degrees left
    HEAD RIGHT 15      -> move head 15 degrees right
    HEAD STOP          -> hold current angle
    HEAD CENTER        -> move to 90 degrees
    HEAD SERVO 90      -> set head angle 0..180
    HEAD NEUTRAL 90    -> set custom neutral/angle 0..180
    HEAD LED 255 0 0
    HEAD LED_OFF
    HEAD RAINBOW

Protocol for ESP32_MOVE:
    MOVE FWD 150 / MOVE BACK 150 / MOVE LEFT 120 / MOVE RIGHT 120 / MOVE STOP

Run:
    python uzmax_server/main.py
    # or
    cd uzmax_server
    uvicorn main:app --host 0.0.0.0 --port 5000 --reload

.env keys needed:
    YANDEX_CATALOG_ID
    YANDEX_API_KEY
    OPENAI_API_KEY
    FACE_EMBED_MODEL     (optional, default: buffalo_s)
"""

import os
import re
import json
import gc
import base64
import asyncio
import logging
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

import serial
import serial.tools.list_ports
import numpy as np
try:
    import cv2
except Exception:
    cv2 = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from agent.speech_to_text import YandexSpeechRecognizer, SttStreamingSession, WhisperSttSession, YandexSttSession
from agent.text_to_speech import YandexStreamingSynthesizer, TtsStreamingSession
from agent.llm import OpenAIClient
from agent.face_encoder import FaceEncoder
from agent.face_store import FaceVectorStore
from hospital_robot import get_patients as get_hospital_patients
from hospital_robot import get_visits as get_hospital_visits
from hospital_robot import init_db as init_hospital_robot_db
from hospital_robot import router as hospital_robot_router

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# ── MLX90640 Thermal Camera (optional — only works on RPi with I2C) ──
board = None
busio = None
adafruit_mlx90640 = None
_THERMAL_HW = False
_THERMAL_IMPORT_ERROR = None
THERMAL_INSTALL_COMMAND = (
    f"{sys.executable} -m pip install adafruit-circuitpython-mlx90640 adafruit-blinka"
)
_FACE_CASCADE = None


def _try_import_thermal_hw() -> bool:
    global board, busio, adafruit_mlx90640, _THERMAL_HW, _THERMAL_IMPORT_ERROR

    if os.name == "nt":
        _THERMAL_HW = False
        _THERMAL_IMPORT_ERROR = "Windows does not provide direct I2C access for MLX90640."
        return False

    try:
        import board as board_module
        import busio as busio_module
        import adafruit_mlx90640 as mlx_module
        board = board_module
        busio = busio_module
        adafruit_mlx90640 = mlx_module
        _THERMAL_HW = True
        _THERMAL_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _THERMAL_HW = False
        _THERMAL_IMPORT_ERROR = str(exc)
        return False


class ThermalCamera:
    """Wrapper for MLX90640 32×24 IR sensor. Gracefully disabled when not connected."""

    ROWS, COLS = 24, 32

    def __init__(self):
        self.enabled = False
        self._mlx    = None
        self._frame  = [0.0] * (self.ROWS * self.COLS)
        self._lock   = threading.Lock()
        self.reason  = "not_connected"
        self.message = "MLX90640 sensor is not connected."
        self.initialize()

    def initialize(self) -> bool:
        if self.enabled and self._mlx is not None:
            return True

        self.enabled = False
        self._mlx = None
        self.reason = "not_connected"
        self.message = "MLX90640 sensor is not connected."

        if not _try_import_thermal_hw():
            logger_pre = logging.getLogger(__name__)
            import_error = _THERMAL_IMPORT_ERROR or ""
            if "WINDOWS" in import_error.upper() or "UNABLE TO IDENTIFY THE BOARD" in import_error.upper():
                self.reason = "unsupported_platform"
                self.message = (
                    "MLX90640 is an I2C sensor. This Windows machine has no supported I2C board "
                    "for Blinka. Run the server on the Raspberry Pi wired to the sensor, or use a "
                    "supported USB-I2C bridge such as FT232H."
                )
            else:
                self.reason = "missing_dependencies"
                self.message = (
                    "Python thermal libraries are missing in the server Python environment. "
                    "Run the install command below on the Raspberry Pi, using the same Python "
                    "that starts uzmax_server/main.py, then press Thermal ON again."
                )
            logger_pre.info("[THERMAL] local I2C unavailable: %s", _THERMAL_IMPORT_ERROR)
            return False

        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=100_000)
            self._mlx = adafruit_mlx90640.MLX90640(i2c)
            self._mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
            self.enabled = True
            self.reason = None
            self.message = "MLX90640 ready."
            logging.getLogger(__name__).info("[THERMAL] MLX90640 ready (2 Hz)")
            return True
        except Exception as exc:
            self.reason = "init_failed"
            self.message = str(exc)
            logging.getLogger(__name__).warning("[THERMAL] MLX90640 init failed: %s", exc)
            return False

    def read(self) -> dict:
        """Read one 32×24 frame. Returns dict with stats + flat list."""
        if not self.enabled or self._mlx is None:
            return {
                "ok": False,
                "reason": self.reason or "not_connected",
                "message": self.message,
                "install": THERMAL_INSTALL_COMMAND,
            }
        try:
            with self._lock:
                self._mlx.getFrame(self._frame)
            arr   = np.array(self._frame, dtype=np.float32).reshape((self.ROWS, self.COLS))
            return {
                "ok":     True,
                "frame":  arr.flatten().tolist(),   # 768 float values
                "rows":   self.ROWS,
                "cols":   self.COLS,
                "max":    round(float(arr.max()),  1),
                "min":    round(float(arr.min()),  1),
                "avg":    round(float(arr.mean()), 1),
                "center": round(float(arr[self.ROWS//2, self.COLS//2]), 1),
            }
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}


_thermal = ThermalCamera()
_remote_thermal = {"data": None, "updated_at": 0.0, "source": None}
_thermal_runtime = {"enabled": False, "updated_at": 0.0}
REMOTE_THERMAL_TTL_SECONDS = 8.0
HAND_SERVO_LIMITS = {
    "R": {
        1: (90, 180),
        2: (0, 180),
        3: (90, 180),
        4: (0, 180),
        5: (0, 110),
        6: (0, 110),
    },
    "L": {
        1: (0, 100),
        2: (0, 90),
        3: (0, 90),
        4: (0, 180),
        5: (60, 180),
        6: (0, 90),
    },
}


def _normalize_thermal_payload(payload: dict, source: str = "raspberry_pi") -> dict:
    frame = payload.get("frame")
    expected = ThermalCamera.ROWS * ThermalCamera.COLS
    if not isinstance(frame, list) or len(frame) != expected:
        raise ValueError(f"frame must contain {expected} temperature values")

    arr = np.array(frame, dtype=np.float32).reshape((ThermalCamera.ROWS, ThermalCamera.COLS))
    center_default = arr[ThermalCamera.ROWS // 2, ThermalCamera.COLS // 2]
    return {
        "ok": True,
        "source": source,
        "frame": arr.flatten().round(2).tolist(),
        "rows": ThermalCamera.ROWS,
        "cols": ThermalCamera.COLS,
        "max": round(float(payload.get("max", arr.max())), 1),
        "min": round(float(payload.get("min", arr.min())), 1),
        "avg": round(float(payload.get("avg", arr.mean())), 1),
        "center": round(float(payload.get("center", center_default)), 1),
        "timestamp": time.time(),
    }


def _read_thermal_source() -> dict:
    if not _thermal_runtime.get("enabled"):
        return {
            "ok": False,
            "reason": "disabled",
            "message": "Thermal camera is off. Turn it on from the dashboard.",
            "enabled": False,
        }

    local = _thermal.read()
    if local.get("ok"):
        local["source"] = "local_i2c"
        local["enabled"] = True
        return local

    remote = _remote_thermal.get("data")
    remote_age = time.time() - float(_remote_thermal.get("updated_at") or 0)
    if remote and remote_age <= REMOTE_THERMAL_TTL_SECONDS:
        data = dict(remote)
        data["source"] = _remote_thermal.get("source") or data.get("source") or "raspberry_pi"
        data["remote_age"] = round(remote_age, 2)
        data["enabled"] = True
        return data

    data = dict(local)
    data["enabled"] = True
    if remote:
        data["remote_reason"] = "stale"
        data["remote_age"] = round(remote_age, 2)
        data["message"] = (
            "Remote thermal agent is not sending fresh frames. Start the Raspberry Pi thermal "
            "agent and point it at this server link."
        )
    return data


def _set_thermal_enabled(enabled: bool) -> dict:
    if enabled:
        _thermal.initialize()

    _thermal_runtime["enabled"] = bool(enabled)
    _thermal_runtime["updated_at"] = time.time()
    return {
        "ok": True,
        "enabled": _thermal_runtime["enabled"],
        "hardware_ready": bool(_thermal.enabled),
        "reason": None if _thermal.enabled else _thermal.reason,
        "message": _thermal.message,
        "install": None if _thermal.enabled else THERMAL_INSTALL_COMMAND,
    }

# ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def configured_secret(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    if any(marker in upper for marker in ("YOUR_", "YOUR-", "YOUR", "_HERE", "KEY_HERE")):
        return None
    return value


FOLDER_ID            = configured_secret(os.getenv("YANDEX_CATALOG_ID"))
API_KEY              = configured_secret(os.getenv("YANDEX_API_KEY"))
GEMINI_API_KEY       = configured_secret(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.82"))
FACE_MATCH_MIN_CONFIDENCE = float(os.getenv("FACE_MATCH_MIN_CONFIDENCE", "0.82"))
FACE_MATCH_MARGIN = float(os.getenv("FACE_MATCH_MARGIN", "0.04"))
FACE_LOG_ALL         = os.getenv("FACE_LOG_ALL_COMPARISONS", "false").lower() == "true"
FACE_MIN_WIDTH_PX    = int(os.getenv("FACE_MIN_WIDTH_PX", "120"))
FACE_MIN_BLUR_VAR    = float(os.getenv("FACE_MIN_BLUR_VAR", "85"))
FACE_MIN_SAMPLES     = int(os.getenv("FACE_MIN_SAMPLES", "1"))
ENV_PATH             = Path(".env")
FEVER_THRESHOLD_C    = 37.5
YANDEX_TTS_VOICE     = os.getenv("YANDEX_TTS_VOICE", "yulduz")
YANDEX_TTS_VOICE_UZ  = os.getenv("YANDEX_TTS_VOICE_UZ", YANDEX_TTS_VOICE)
YANDEX_TTS_VOICE_EN  = os.getenv("YANDEX_TTS_VOICE_EN", "john")
YANDEX_TTS_VOICE_RU  = os.getenv("YANDEX_TTS_VOICE_RU", "yulduz_ru")
YANDEX_TTS_SPEED     = float(os.getenv("YANDEX_TTS_SPEED", "1.1"))
YANDEX_TTS_SAMPLE_RATE = int(os.getenv("YANDEX_TTS_SAMPLE_RATE", "48000"))

DOCTOR_DIRECTORY = [
    {
        "id": 1,
        "name": "Qabul shifokori",
        "specialty": "dastlabki ko'rik",
        "speciality": "dastlabki ko'rik",
        "room": "100-xona",
        "work_time": "08:00 - 18:00",
        "use_for": "yangi bemor, isitma, umumiy holatni baholash",
        "keywords": ["qabul", "umumiy", "bosh og'riq", "bosh og'rig'i", "holsizlik", "ko'rik", "shifokor", "doktor", "vrach", "navbat"],
    },
    {
        "id": 2,
        "name": "Infeksionist",
        "specialty": "yuqumli kasalliklar",
        "speciality": "yuqumli kasalliklar",
        "room": "102-xona",
        "work_time": "09:00 - 17:00",
        "use_for": "isitma, yo'tal, tomoq og'rig'i, ich ketishi yoki infeksiya gumoni",
        "keywords": ["infeksionist", "isitma", "yo'tal", "yotal", "tomoq og'rig'i", "tomoq ogrigi", "gripp", "shamollash", "ich ketishi", "infeksiya", "yuqumli"],
    },
    {
        "id": 3,
        "name": "Pediatr",
        "specialty": "bolalar shifokori",
        "speciality": "bolalar shifokori",
        "room": "104-xona",
        "work_time": "09:00 - 15:00",
        "use_for": "18 yoshgacha bo'lgan bemorlar",
        "keywords": ["pediatr", "bola", "bolam", "farzand", "chaqaloq"],
    },
    {
        "id": 4,
        "name": "Laboratoriya",
        "specialty": "tahlillar",
        "speciality": "tahlillar",
        "room": "110-xona",
        "work_time": "08:30 - 16:30",
        "use_for": "PZR, qon tahlili va boshqa tekshiruvlar",
        "keywords": ["laboratoriya", "tahlil", "analiz", "qon", "pzr", "test"],
    },
    {
        "id": 5,
        "name": "Dr. Aliyev",
        "specialty": "Terapevt",
        "speciality": "Terapevt",
        "room": "101-xona",
        "work_time": "09:00 - 17:00",
        "use_for": "isitma, yo'tal, holsizlik, shamollash, bosh og'rig'i va umumiy og'riqlar",
        "keywords": ["terapevt", "isitma", "yo'tal", "yotal", "holsizlik", "gripp", "shamollash", "tomoq og'rig'i", "tomoq ogrigi", "bosh og'rig'i", "bosh ogrigi", "boshim og'riyapti", "boshim ogriyapti", "umumiy og'riq"],
    },
    {
        "id": 6,
        "name": "Dr. Karimova",
        "specialty": "Kardiolog",
        "speciality": "Kardiolog",
        "room": "203-xona",
        "work_time": "10:00 - 16:00",
        "use_for": "yurak, ko'krak og'rig'i, qon bosimi, nafas qisishi va yurak urishi",
        "keywords": ["kardiolog", "yurak", "ko'krak og'rig'i", "kokrak ogrigi", "qon bosimi", "bosim", "nafas qisishi", "taxikardiya", "yurak urishi"],
    },
    {
        "id": 7,
        "name": "Dr. Sobirov",
        "specialty": "Nevrolog",
        "speciality": "Nevrolog",
        "room": "305-xona",
        "work_time": "09:00 - 15:00",
        "use_for": "bosh aylanishi, asab, qo'l-oyoq uvishishi, bel og'rig'i va migren",
        "keywords": ["nevrolog", "bosh aylanishi", "asab", "qo'l uvishishi", "qol uvishishi", "oyoq uvishishi", "bel og'rig'i", "bel ogrigi", "migren", "hushdan ketish"],
    },
    {
        "id": 8,
        "name": "Dr. Rustamov",
        "specialty": "LOR",
        "speciality": "LOR",
        "room": "108-xona",
        "work_time": "08:30 - 14:00",
        "use_for": "quloq, burun, tomoq, angina, burun bitishi va eshitish muammolari",
        "keywords": ["lor", "quloq", "burun", "tomoq", "angina", "burun bitishi", "eshitish", "sinusit"],
    },
    {
        "id": 9,
        "name": "Dr. Saidova",
        "specialty": "Gastroenterolog",
        "speciality": "Gastroenterolog",
        "room": "210-xona",
        "work_time": "09:00 - 16:30",
        "use_for": "qorin og'rig'i, oshqozon, ich ketishi, qabziyat, ko'ngil aynishi va hazm muammolari",
        "keywords": ["gastroenterolog", "qorin og'rig'i", "qorin ogrigi", "oshqozon", "ich ketishi", "qabziyat", "ko'ngil aynishi", "kongil aynishi", "jigar", "hazm"],
    },
]

DOCTOR_MULTILINGUAL_ALIASES = {
    1: [
        "doctor", "physician", "reception", "general checkup", "appointment", "queue",
        "врач", "доктор", "прием", "очередь", "осмотр", "регистратура",
    ],
    2: [
        "infectious disease", "infection", "fever", "cough", "flu", "sore throat", "diarrhea", "cold",
        "инфекционист", "инфекция", "температура", "жар", "кашель", "грипп", "горло", "боль в горле", "понос", "простуда",
    ],
    3: [
        "pediatrician", "child", "baby", "infant", "my child",
        "педиатр", "ребенок", "ребёнок", "малыш", "детский врач",
    ],
    4: [
        "laboratory", "lab", "analysis", "blood test", "pcr", "test",
        "лаборатория", "анализ", "анализ крови", "пцр", "тест",
    ],
    5: [
        "therapist", "general practitioner", "fever", "cough", "weakness", "cold", "headache", "body ache",
        "терапевт", "температура", "кашель", "слабость", "простуда", "головная боль", "ломота",
    ],
    6: [
        "cardiologist", "heart", "chest pain", "blood pressure", "pressure", "shortness of breath", "palpitations", "tachycardia",
        "кардиолог", "сердце", "боль в груди", "давление", "одышка", "сердцебиение", "тахикардия",
    ],
    7: [
        "neurologist", "dizziness", "nerve", "numbness", "hand numbness", "leg numbness", "back pain", "migraine", "fainting",
        "невролог", "головокружение", "нервы", "онемение", "немеет рука", "немеет нога", "боль в спине", "мигрень", "обморок",
    ],
    8: [
        "ent", "ear", "nose", "throat", "tonsillitis", "stuffy nose", "hearing", "sinusitis",
        "лор", "ухо", "нос", "горло", "ангина", "заложен нос", "слух", "синусит",
    ],
    9: [
        "gastroenterologist", "stomach pain", "abdominal pain", "stomach", "diarrhea", "constipation", "nausea", "liver", "digestion",
        "гастроэнтеролог", "живот", "боль в животе", "желудок", "понос", "запор", "тошнота", "печень", "пищеварение",
    ],
}

DOCTOR_SPECIALTY_LABELS = {
    1: {"uz-UZ": "Qabul shifokori", "en-US": "Reception doctor", "ru-RU": "Врач первичного приема"},
    2: {"uz-UZ": "Infeksionist", "en-US": "Infectious disease specialist", "ru-RU": "Инфекционист"},
    3: {"uz-UZ": "Pediatr", "en-US": "Pediatrician", "ru-RU": "Педиатр"},
    4: {"uz-UZ": "Laboratoriya", "en-US": "Laboratory", "ru-RU": "Лаборатория"},
    5: {"uz-UZ": "Terapevt", "en-US": "Therapist", "ru-RU": "Терапевт"},
    6: {"uz-UZ": "Kardiolog", "en-US": "Cardiologist", "ru-RU": "Кардиолог"},
    7: {"uz-UZ": "Nevrolog", "en-US": "Neurologist", "ru-RU": "Невролог"},
    8: {"uz-UZ": "LOR", "en-US": "ENT specialist", "ru-RU": "ЛОР"},
    9: {"uz-UZ": "Gastroenterolog", "en-US": "Gastroenterologist", "ru-RU": "Гастроэнтеролог"},
}

SETTINGS_KEYS = [
    "YANDEX_API_KEY",
    "YANDEX_CATALOG_ID",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "GEMINI_API_KEY",
    "FACE_EMBED_MODEL",
    "FACE_EMBED_DET_SIZE",
    "FACE_MATCH_THRESHOLD",
    "FACE_MATCH_MIN_CONFIDENCE",
    "FACE_MATCH_MARGIN",
    "FACE_MIN_SAMPLES",
    "YANDEX_TTS_VOICE",
    "YANDEX_TTS_VOICE_UZ",
    "YANDEX_TTS_VOICE_EN",
    "YANDEX_TTS_VOICE_RU",
    "YANDEX_TTS_SPEED",
    "YANDEX_TTS_SAMPLE_RATE",
    "STT_PROVIDER",
    "ARDUINO_CLI_PATH",
    "ESP32_FQBN",
]

SETTINGS_DEFAULTS = {
    "OPENAI_MODEL": "gpt-4o-mini",
    "FACE_EMBED_MODEL": "buffalo_s",
    "FACE_EMBED_DET_SIZE": "640,640",
    "FACE_MATCH_THRESHOLD": "0.82",
    "FACE_MATCH_MIN_CONFIDENCE": "0.82",
    "FACE_MATCH_MARGIN": "0.04",
    "FACE_MIN_SAMPLES": "1",
    "YANDEX_TTS_VOICE": "yulduz",
    "YANDEX_TTS_VOICE_UZ": "yulduz",
    "YANDEX_TTS_VOICE_EN": "john",
    "YANDEX_TTS_VOICE_RU": "yulduz_ru",
    "YANDEX_TTS_SPEED": "1.1",
    "YANDEX_TTS_SAMPLE_RATE": "48000",
    "STT_PROVIDER": "yandex_stream",
}


def read_env_values() -> dict:
    values = {key: os.getenv(key, SETTINGS_DEFAULTS.get(key, "")) for key in SETTINGS_KEYS}
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw.strip().lstrip("\ufeff")
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in SETTINGS_KEYS:
                values[key] = value.strip().strip('"').strip("'")
    return values


def write_env_values(new_values: dict) -> None:
    existing = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    seen = set()
    output = []

    for raw in existing:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw:
            output.append(raw)
            continue
        key = raw.split("=", 1)[0].strip()
        if key in SETTINGS_KEYS:
            output.append(f"{key}={str(new_values.get(key, '')).strip()}")
            seen.add(key)
        else:
            output.append(raw)

    if output and output[-1].strip():
        output.append("")
    for key in SETTINGS_KEYS:
        if key not in seen:
            output.append(f"{key}={str(new_values.get(key, '')).strip()}")

    ENV_PATH.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        init_hospital_robot_db()
    except Exception as exc:
        logger.warning("Hospital robot DB init skipped: %s", exc)

    try:
        load_registered_faces()
    except Exception as exc:
        logger.warning("Face bootstrap skipped: %s", exc)
    yield


app = FastAPI(title="UzMAX Unified Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(hospital_robot_router)

os.makedirs("static", exist_ok=True)
os.makedirs("images", exist_ok=True)
DATA_DIR            = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REGISTER_FACES_DIR  = DATA_DIR / "register_faces"
REGISTER_FACES_DIR.mkdir(parents=True, exist_ok=True)
REGISTER_FACES_JSON = REGISTER_FACES_DIR / "registry.json"
DOCTOR_QUEUE_FILE   = DATA_DIR / "doctor_queue.json"
IMAGES_DIR          = Path("images")
PROJECT_ROOT        = Path(__file__).resolve().parent.parent
ARDUINO_CLI_DEFAULT = Path(r"C:\Program Files\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe")
ESP32_FQBN_DEFAULT  = os.getenv("ESP32_FQBN", "esp32:esp32:esp32")

FIRMWARE_SKETCHES = {
    "hand": PROJECT_ROOT / "hand" / "hand.ino",
    "head": PROJECT_ROOT / "head" / "head.ino",
    "move": PROJECT_ROOT / "movements" / "move.ino",
}
FIRMWARE_VERSION_DIR = PROJECT_ROOT / "firmware_versions"

app.mount("/static", StaticFiles(directory="static"), name="static")

face_store = None
face_encoder = None


# ═══════════════════════════════════════════════════════════════════
#  ESP32 SERIAL MANAGER
# ═══════════════════════════════════════════════════════════════════

SERIAL_BAUD = 115200

_devices: dict[str, dict] = {
    "hand": {"ser": None, "port": None, "lock": threading.Lock()},
    "head": {"ser": None, "port": None, "lock": threading.Lock()},
    "move": {"ser": None, "port": None, "lock": threading.Lock()},
}


def _serial_error_message(port: str, exc: Exception) -> str:
    raw = str(exc)
    if "Access is denied" in raw or "PermissionError" in raw:
        return (
            f"{port} is busy. Close Arduino IDE Serial Monitor/Plotter, PuTTY, "
            "PlatformIO monitor, or any other serial app, then retry Auto Connect."
        )
    return raw


def _serial_port_owner(port: str, exclude_device: str | None = None) -> str | None:
    for name, dev in _devices.items():
        if name == exclude_device:
            continue
        ser = dev.get("ser")
        if dev.get("port") == port and ser is not None and ser.is_open:
            return name
    return None


def _serial_connect(device: str, port: str) -> dict:
    owner = _serial_port_owner(port, exclude_device=device)
    if owner:
        return {
            "ok": False,
            "message": f"{port} is already connected as {owner.upper()}. Disconnect it first.",
        }

    dev = _devices[device]
    with dev["lock"]:
        if dev["ser"] and dev["ser"].is_open:
            try:
                dev["ser"].close()
            except Exception:
                pass
        try:
            dev["ser"]  = serial.Serial(port, SERIAL_BAUD, timeout=1)
            dev["port"] = port
            time.sleep(2)
            logger.info("[SERIAL] %s connected → %s", device.upper(), port)
            return {"ok": True, "message": f"Connected to {port}"}
        except Exception as e:
            dev["ser"]  = None
            dev["port"] = None
            message = _serial_error_message(port, e)
            logger.error("[SERIAL] %s failed on %s: %s", device.upper(), port, message)
            return {"ok": False, "message": message}


def _serial_auto_connect() -> dict:
    """Scan serial ports, PING each board, and keep HAND/HEAD/MOVE matches open."""
    found: dict[str, dict] = {}
    errors: list[dict] = []

    for device in _devices:
        _serial_disconnect(device)

    for port_info in serial.tools.list_ports.comports():
        port_name = port_info.device
        ser = None
        try:
            ser = serial.Serial(port_name, SERIAL_BAUD, timeout=1)
            time.sleep(2)
            ser.write(b"PING\n")
            time.sleep(0.35)

            lines = []
            while ser.in_waiting:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    lines.append(line)
            response = "\n".join(lines)

            matched = None
            for device_name in ("HAND", "HEAD", "MOVE"):
                if f"DEVICE:{device_name}" in response:
                    matched = device_name.lower()
                    break

            # Legacy hand sketches sometimes print only this banner and support
            # R/L servo commands, but do not identify as DEVICE:HAND.
            if not matched:
                legacy_hand_markers = (
                    "ESP32 12 SERVO CONTROL READY",
                    "USE: R 1 90",
                    "USE: L 6 0",
                    "ERROR FORMAT MUST BE: R 1 90",
                )
                if any(marker in response.upper() for marker in legacy_hand_markers):
                    matched = "hand"

            if not matched:
                legacy_head_markers = (
                    "DEVICE:HEAD",
                    "HEAD READY",
                    "LED SYSTEM READY",
                    "SERVO TEST READY",
                    "SEND ANGLE: 0 TO 180",
                    "COMMANDS:",
                    "RED, GREEN, BLUE, WHITE, OFF",
                )
                if any(marker in response.upper() for marker in legacy_head_markers):
                    matched = "head"

            if not matched:
                ser.close()
                continue

            dev = _devices[matched]
            with dev["lock"]:
                if dev["ser"] and dev["ser"].is_open:
                    dev["ser"].close()
                dev["ser"] = ser
                dev["port"] = port_name
            found[matched] = {
                "port": port_name,
                "description": port_info.description,
                "response": response,
            }
            logger.info("[SERIAL] Auto-connected %s on %s", matched.upper(), port_name)

        except Exception as exc:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
            errors.append({"port": port_name, "message": _serial_error_message(port_name, exc)})

    return {
        "ok": True,
        "found": found,
        "errors": errors,
        "status": {k: _serial_status(k) for k in _devices},
    }


def _serial_send(device: str, command: str) -> tuple[bool, str]:
    dev  = _devices[device]
    with dev["lock"]:
        ser = dev["ser"]
        if ser is None or not ser.is_open:
            return False, "Not connected"
        try:
            ser.write((command + "\n").encode("utf-8"))
            time.sleep(0.08)
            resp = ""
            while ser.in_waiting:
                resp += ser.readline().decode(errors="ignore").strip() + "\n"
            response = resp.strip() or "OK:NO_RESPONSE"
            response_upper = response.upper()
            accepted = not any(token in response_upper for token in ("ERR", "ERROR", "UNKNOWN"))
            return accepted, response
        except serial.SerialException as exc:
            logger.error("[SERIAL] %s lost: %s", device.upper(), exc)
            dev["ser"] = None
            return False, str(exc)


def _legacy_head_led_command(command: str) -> str | None:
    parts = command.strip().split()
    upper = [part.upper() for part in parts]
    if len(parts) == 5 and upper[:2] == ["HEAD", "LED"]:
        return f"color {parts[2]} {parts[3]} {parts[4]}"
    if upper == ["HEAD", "LED_OFF"]:
        return "off"
    if upper == ["HEAD", "RAINBOW"]:
        return "rainbow"
    if len(parts) == 3 and upper[:2] == ["HEAD", "BRIGHTNESS"]:
        return f"brightness {parts[2]}"
    return None


def _normalize_head_command(command: str) -> str:
    parts = command.strip().split()
    upper = [part.upper() for part in parts]
    if len(parts) == 4 and upper[:2] == ["HEAD", "SERVO"]:
        # Old dashboard format was HEAD SERVO <servo_num> <angle>.
        # Current HEAD firmware has one 180-degree positional servo: HEAD SERVO <angle>.
        return f"HEAD SERVO {parts[3]}"
    return command


def _serial_disconnect(device: str):
    dev = _devices[device]
    with dev["lock"]:
        if dev["ser"] and dev["ser"].is_open:
            dev["ser"].close()
        dev["ser"]  = None
        dev["port"] = None


def _serial_status(device: str) -> dict:
    dev       = _devices[device]
    connected = dev["ser"] is not None and dev["ser"].is_open
    return {"connected": connected, "port": dev["port"]}


def _arduino_cli_path() -> str | None:
    configured = os.getenv("ARDUINO_CLI_PATH", "").strip()
    candidates = [
        configured,
        shutil.which("arduino-cli"),
        str(ARDUINO_CLI_DEFAULT) if ARDUINO_CLI_DEFAULT.exists() else "",
    ]
    return next((path for path in candidates if path and Path(path).exists()), None)


def _run_arduino_command(args: list[str], timeout: int = 180) -> dict:
    cli = _arduino_cli_path()
    if not cli:
        return {
            "ok": False,
            "message": "arduino-cli not found. Install Arduino IDE or set ARDUINO_CLI_PATH in .env.",
            "stdout": "",
            "stderr": "",
        }

    cli_tmp = PROJECT_ROOT / "firmware_tmp" / "cli_tmp"
    cli_tmp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMP"] = str(cli_tmp)
    env["TEMP"] = str(cli_tmp)
    env["TMPDIR"] = str(cli_tmp)

    proc = subprocess.run(
        [cli, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": " ".join([cli, *args]),
    }


def _firmware_compile_upload(target: str, port: str, code: str, fqbn: str, compile_only: bool) -> dict:
    sketch_name = f"UZMAX_{target.upper()}_WEB"
    temp_root = PROJECT_ROOT / "firmware_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_dir = temp_root / f"uzmax_firmware_{target}_{int(time.time() * 1000)}"
    sketch_dir = run_dir / sketch_name
    build_dir = run_dir / "build"
    sketch_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    (sketch_dir / f"{sketch_name}.ino").write_text(code, encoding="utf-8")

    compile_result = _run_arduino_command([
        "compile",
        "--fqbn", fqbn,
        "--build-path", str(build_dir),
        str(sketch_dir),
    ])
    compile_result["workdir"] = str(sketch_dir)
    compile_result["build_dir"] = str(build_dir)
    if compile_only or not compile_result["ok"]:
        return {"ok": compile_result["ok"], "stage": "compile", **compile_result}

    if target in _devices and _serial_status(target).get("port") == port:
        _serial_disconnect(target)

    upload_result = _run_arduino_command(
        [
            "upload",
            "-p", port,
            "--fqbn", fqbn,
            "--input-dir", str(build_dir),
            str(sketch_dir),
        ],
        timeout=240,
    )
    upload_result["workdir"] = str(sketch_dir)
    return {
        "ok": upload_result["ok"],
        "stage": "upload",
        "compile": compile_result,
        **upload_result,
    }


def _firmware_assert_target(target: str) -> str:
    target = str(target or "").lower().strip()
    if target not in FIRMWARE_SKETCHES:
        raise ValueError("Unknown firmware target")
    return target


def _firmware_version_path(target: str) -> Path:
    target = _firmware_assert_target(target)
    return FIRMWARE_VERSION_DIR / f"{target}.json"


def _firmware_read_versions(target: str) -> list[dict]:
    path = _firmware_version_path(target)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        versions = data if isinstance(data, list) else data.get("versions", [])
        return versions if isinstance(versions, list) else []
    except Exception:
        logging.exception("Could not read firmware versions: %s", path)
        return []


def _firmware_write_versions(target: str, versions: list[dict]) -> None:
    path = _firmware_version_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")


def _firmware_canonical_code(target: str) -> str:
    target = _firmware_assert_target(target)
    path = FIRMWARE_SKETCHES[target]
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _firmware_find_version(target: str, version_id: str) -> dict | None:
    for item in _firmware_read_versions(target):
        if str(item.get("id")) == str(version_id):
            return item
    return None


def _firmware_filename(target: str, version: dict | None = None) -> str:
    target = _firmware_assert_target(target)
    suffix = "latest"
    if version:
        suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(version.get("name") or version.get("id") or "saved")).strip("_")[:40] or "saved"
    return f"UZMAX_{target.upper()}_{suffix}.ino"


# ═══════════════════════════════════════════════════════════════════
#  ROBOT CONTROL REST ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/ports")
def list_ports():
    return JSONResponse([
        {"device": p.device, "description": p.description}
        for p in serial.tools.list_ports.comports()
    ])


@app.post("/api/connect")
async def connect_device(payload: dict):
    device = payload.get("device", "").lower()
    port   = payload.get("port", "")
    if device not in _devices:
        return JSONResponse({"ok": False, "message": "Unknown device"}, status_code=400)
    if not port:
        return JSONResponse({"ok": False, "message": "No port specified"}, status_code=400)
    result = await asyncio.to_thread(_serial_connect, device, port)
    return JSONResponse(result)


@app.post("/api/auto-connect")
async def auto_connect_devices():
    result = await asyncio.to_thread(_serial_auto_connect)
    return JSONResponse(result)


@app.post("/api/disconnect")
async def disconnect_device(payload: dict):
    device = payload.get("device", "").lower()
    if device not in _devices:
        return JSONResponse({"ok": False, "message": "Unknown device"}, status_code=400)
    await asyncio.to_thread(_serial_disconnect, device)
    return JSONResponse({"ok": True})


@app.get("/api/robot/status")
def robot_status():
    return JSONResponse({k: _serial_status(k) for k in _devices})


@app.get("/api/firmware/sketches")
def firmware_sketches():
    sketches = {}
    for target, path in FIRMWARE_SKETCHES.items():
        exists = path.exists()
        versions = _firmware_read_versions(target)
        latest = versions[0] if versions else None
        canonical_code = path.read_text(encoding="utf-8") if exists else ""
        sketches[target] = {
            "target": target,
            "path": str(path),
            "exists": exists,
            "code": latest.get("code", "") if latest else canonical_code,
            "canonical_code": canonical_code,
            "latest_version": latest,
            "version_count": len(versions),
        }
    return JSONResponse({
        "ok": True,
        "arduino_cli": _arduino_cli_path(),
        "default_fqbn": ESP32_FQBN_DEFAULT,
        "sketches": sketches,
    })


@app.get("/api/firmware/versions")
def firmware_versions(target: str = "move"):
    try:
        target = _firmware_assert_target(target)
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    versions = _firmware_read_versions(target)
    return JSONResponse({
        "ok": True,
        "target": target,
        "versions": versions,
        "canonical_code": _firmware_canonical_code(target),
    })


@app.post("/api/firmware/versions")
async def firmware_save_version(payload: dict):
    try:
        target = _firmware_assert_target(payload.get("target", "move"))
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)

    code = str(payload.get("code", "")).strip()
    if not code:
        return JSONResponse({"ok": False, "message": "Firmware code is empty"}, status_code=400)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = str(payload.get("name", "")).strip() or f"{target.upper()} {now}"
    fqbn = str(payload.get("fqbn", ESP32_FQBN_DEFAULT)).strip() or ESP32_FQBN_DEFAULT
    notes = str(payload.get("notes", "")).strip()
    item = {
        "id": version_id,
        "target": target,
        "name": name,
        "created_at": now,
        "fqbn": fqbn,
        "notes": notes,
        "code": code,
    }
    versions = _firmware_read_versions(target)
    versions.insert(0, item)
    _firmware_write_versions(target, versions[:200])
    return JSONResponse({"ok": True, "target": target, "version": item, "versions": versions[:200]})


@app.get("/api/firmware/download")
def firmware_download(target: str = "move", version_id: str = "latest"):
    try:
        target = _firmware_assert_target(target)
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)

    version = None
    code = ""
    if version_id == "canonical":
        code = _firmware_canonical_code(target)
    else:
        versions = _firmware_read_versions(target)
        if version_id and version_id != "latest":
            version = _firmware_find_version(target, version_id)
        elif versions:
            version = versions[0]
        code = version.get("code", "") if version else _firmware_canonical_code(target)

    if not code.strip():
        return JSONResponse({"ok": False, "message": "Firmware code is empty"}, status_code=404)

    return Response(
        content=code,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{_firmware_filename(target, version)}"'},
    )


@app.post("/api/firmware/compile")
async def firmware_compile(payload: dict):
    target = str(payload.get("target", "move")).lower()
    code = str(payload.get("code", "")).strip()
    fqbn = str(payload.get("fqbn", ESP32_FQBN_DEFAULT)).strip() or ESP32_FQBN_DEFAULT
    if target not in FIRMWARE_SKETCHES:
        return JSONResponse({"ok": False, "message": "Unknown firmware target"}, status_code=400)
    if not code:
        return JSONResponse({"ok": False, "message": "Firmware code is empty"}, status_code=400)
    result = await asyncio.to_thread(_firmware_compile_upload, target, "", code, fqbn, True)
    return JSONResponse(result)


@app.post("/api/firmware/upload")
async def firmware_upload(payload: dict):
    target = str(payload.get("target", "move")).lower()
    port = str(payload.get("port", "")).strip()
    code = str(payload.get("code", "")).strip()
    fqbn = str(payload.get("fqbn", ESP32_FQBN_DEFAULT)).strip() or ESP32_FQBN_DEFAULT
    if target not in FIRMWARE_SKETCHES:
        return JSONResponse({"ok": False, "message": "Unknown firmware target"}, status_code=400)
    if not port:
        return JSONResponse({"ok": False, "message": "Select a COM port before upload"}, status_code=400)
    if not code:
        return JSONResponse({"ok": False, "message": "Firmware code is empty"}, status_code=400)
    result = await asyncio.to_thread(_firmware_compile_upload, target, port, code, fqbn, False)
    return JSONResponse(result)


@app.post("/api/hand/move")
async def hand_move(payload: dict):
    """
    New firmware protocol: R 1 90  /  L 6 120
    Payload: {hand: 'R'|'L', servo: 1-6, angle: 0-180}
    """
    try:
        hand = str(payload.get("hand", "R")).upper()
        servo = int(payload.get("servo", 1))
        angle = max(0, min(180, int(payload.get("angle", 90))))
    except (TypeError, ValueError):
        return JSONResponse({"ok": False, "message": "Servo and angle must be numbers"}, status_code=400)

    if hand not in ("R", "L") or not (1 <= servo <= 6):
        return JSONResponse({"ok": False, "message": "Invalid hand or servo"}, status_code=400)

    min_angle, max_angle = HAND_SERVO_LIMITS[hand][servo]
    angle = max(min_angle, min(max_angle, angle))

    cmd = f"{hand} {servo} {angle}"
    ok, resp = await asyncio.to_thread(_serial_send, "hand", cmd)
    return JSONResponse({"ok": ok, "sent": cmd, "response": resp,
                         "connected": _serial_status("hand")["connected"]})


@app.post("/api/head/command")
async def head_command(payload: dict):
    """
    HEAD LEFT 15 / HEAD RIGHT 15 / HEAD STOP / HEAD CENTER / HEAD SERVO 90
    HEAD LED r g b / HEAD LED_OFF / HEAD RAINBOW
    Payload: {command: '...'}
    """
    requested = str(payload.get("command", "")).strip()
    if not requested:
        return JSONResponse({"ok": False, "message": "Empty command"}, status_code=400)
    cmd = _normalize_head_command(requested)
    ok, resp = await asyncio.to_thread(_serial_send, "head", cmd)
    fallback = _legacy_head_led_command(cmd)
    if fallback and (not ok or "UNKNOWN" in resp.upper()):
        ok, resp = await asyncio.to_thread(_serial_send, "head", fallback)
        return JSONResponse({
            "ok": ok,
            "sent": fallback,
            "requested": cmd,
            "response": resp,
            "fallback": True,
            "connected": _serial_status("head")["connected"],
        })
    return JSONResponse({"ok": ok, "sent": cmd, "response": resp,
                         "requested": requested if requested != cmd else None,
                         "connected": _serial_status("head")["connected"]})


@app.post("/api/move/command")
async def move_command(payload: dict):
    """
    MOVE FWD 150 / MOVE BACK 150 / MOVE LEFT 120 / MOVE RIGHT 120 / MOVE STOP
    Payload: {command: '...'}
    """
    cmd = str(payload.get("command", "")).strip().upper()
    if not cmd:
        return JSONResponse({"ok": False, "message": "Empty command"}, status_code=400)
    ok, resp = await asyncio.to_thread(_serial_send, "move", cmd)
    return JSONResponse({"ok": ok, "sent": cmd, "response": resp,
                         "connected": _serial_status("move")["connected"]})


@app.post("/api/raw")
async def raw_command(payload: dict):
    """Send any raw command to any device."""
    device  = str(payload.get("device", "hand")).lower()
    command = str(payload.get("command", "")).strip()
    if device not in _devices or not command:
        return JSONResponse({"ok": False, "message": "Bad device or empty command"}, status_code=400)
    ok, resp = await asyncio.to_thread(_serial_send, device, command)
    return JSONResponse({"ok": ok, "sent": command, "response": resp})


@app.get("/api/settings")
async def get_settings():
    """Return editable local runtime settings from uzmax_server/.env."""
    return JSONResponse({
        "ok": True,
        "path": str(ENV_PATH.resolve()),
        "settings": read_env_values(),
    })


@app.post("/api/settings")
async def save_settings(payload: dict):
    """Persist dashboard settings to .env and refresh in-process config."""
    global FOLDER_ID, API_KEY, GEMINI_API_KEY, FACE_MATCH_THRESHOLD, FACE_MATCH_MIN_CONFIDENCE, FACE_MATCH_MARGIN, FACE_MIN_SAMPLES
    global YANDEX_TTS_VOICE, YANDEX_TTS_SPEED, YANDEX_TTS_SAMPLE_RATE
    global face_encoder, face_store

    incoming = payload.get("settings", payload)
    if not isinstance(incoming, dict):
        return JSONResponse({"ok": False, "message": "settings must be an object"}, status_code=400)

    values = read_env_values()
    for key in SETTINGS_KEYS:
        if key in incoming:
            values[key] = str(incoming.get(key, "") or "").strip()

    try:
        threshold = float(values.get("FACE_MATCH_THRESHOLD") or "0.82")
    except ValueError:
        return JSONResponse({"ok": False, "message": "FACE_MATCH_THRESHOLD must be a number"}, status_code=400)
    try:
        min_confidence = float(values.get("FACE_MATCH_MIN_CONFIDENCE") or "0.82")
    except ValueError:
        return JSONResponse({"ok": False, "message": "FACE_MATCH_MIN_CONFIDENCE must be a number"}, status_code=400)
    try:
        match_margin = float(values.get("FACE_MATCH_MARGIN") or "0.04")
    except ValueError:
        return JSONResponse({"ok": False, "message": "FACE_MATCH_MARGIN must be a number"}, status_code=400)
    try:
        face_min_samples = max(1, int(values.get("FACE_MIN_SAMPLES") or "1"))
    except ValueError:
        return JSONResponse({"ok": False, "message": "FACE_MIN_SAMPLES must be an integer"}, status_code=400)
    try:
        tts_speed = float(values.get("YANDEX_TTS_SPEED") or "1.1")
    except ValueError:
        return JSONResponse({"ok": False, "message": "YANDEX_TTS_SPEED must be a number"}, status_code=400)
    try:
        tts_sample_rate = int(values.get("YANDEX_TTS_SAMPLE_RATE") or "48000")
    except ValueError:
        return JSONResponse({"ok": False, "message": "YANDEX_TTS_SAMPLE_RATE must be a number"}, status_code=400)
    if tts_sample_rate not in (16000, 24000, 48000):
        return JSONResponse({"ok": False, "message": "YANDEX_TTS_SAMPLE_RATE must be 16000, 24000, or 48000"}, status_code=400)

    write_env_values(values)

    for key, value in values.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)

    FOLDER_ID = configured_secret(values.get("YANDEX_CATALOG_ID"))
    API_KEY = configured_secret(values.get("YANDEX_API_KEY"))
    GEMINI_API_KEY = configured_secret(values.get("GEMINI_API_KEY"))
    FACE_MATCH_THRESHOLD = threshold
    FACE_MATCH_MIN_CONFIDENCE = min_confidence
    FACE_MATCH_MARGIN = match_margin
    FACE_MIN_SAMPLES = face_min_samples
    YANDEX_TTS_VOICE = values.get("YANDEX_TTS_VOICE") or "yulduz"
    YANDEX_TTS_SPEED = tts_speed
    YANDEX_TTS_SAMPLE_RATE = tts_sample_rate

    # These helpers read environment config when created, so recreate lazily.
    face_encoder = None
    face_store = None

    return JSONResponse({
        "ok": True,
        "path": str(ENV_PATH.resolve()),
        "settings": values,
    })


# ═══════════════════════════════════════════════════════════════════
#  FACE RECOGNITION HELPERS
# ═══════════════════════════════════════════════════════════════════

def get_face_encoder() -> FaceEncoder:
    global face_encoder
    if face_encoder is None:
        face_encoder = FaceEncoder()
    return face_encoder


def get_face_store() -> FaceVectorStore:
    global face_store
    if face_store is None:
        face_store = FaceVectorStore()
    return face_store


def compact_thermal_screening(thermal: dict | None = None) -> dict:
    if not thermal:
        thermal = _read_thermal_source()

    now = datetime.now().isoformat(timespec="seconds")
    if not thermal or not thermal.get("ok"):
        return {
            "time": now,
            "ok": False,
            "source": (thermal or {}).get("source"),
            "reason": (thermal or {}).get("reason") or "thermal_unavailable",
            "message": (thermal or {}).get("message") or "Thermal camera data is not available.",
        }

    max_temp = float(thermal.get("max") or 0)
    center_temp = float(thermal.get("center") or max_temp)
    return {
        "time": now,
        "ok": True,
        "source": thermal.get("source") or "thermal",
        "max_c": round(max_temp, 1),
        "center_c": round(center_temp, 1),
        "avg_c": round(float(thermal.get("avg") or center_temp), 1),
        "min_c": round(float(thermal.get("min") or center_temp), 1),
        "status": "fever_screening" if max_temp >= FEVER_THRESHOLD_C else "normal_screening",
        "note": "Bu MLX90640 orqali dastlabki skrining, klinik tashxis emas.",
    }


def merge_patient_screening(metadata: dict | None, screening: dict) -> dict:
    merged = dict(metadata or {})
    screenings = list(merged.get("thermal_screenings") or [])
    screenings.append(screening)
    merged["thermal_screenings"] = screenings[-20:]
    merged["last_thermal_screening"] = screening
    merged["patient_type"] = "yuqumli_kasalliklar_shifoxonasi_bemori"
    merged["registered_by"] = "UzMAX robot"
    return merged


def attach_screening_to_person(person: dict | None, thermal: dict | None) -> dict | None:
    if not person or not person.get("person_id"):
        return person

    screening = compact_thermal_screening(thermal)
    metadata = merge_patient_screening(person.get("metadata") or {}, screening)
    updated = get_face_store().update_metadata(person["person_id"], metadata)
    return updated or {**person, "metadata": metadata}


def latest_thermal_text(person: dict | None) -> str:
    metadata = (person or {}).get("metadata") or {}
    screening = metadata.get("last_thermal_screening") or {}
    if not screening or not screening.get("ok"):
        return "Harorat hozircha olinmadi."
    max_temp = screening.get("max_c")
    status = screening.get("status")
    if max_temp is None:
        return "Thermal skrining bor, lekin harorat raqami aniq emas."
    note = "isitma ehtimoli bor" if status == "fever_screening" else "isitma belgisi yo'q"
    return f"Thermal skrining: {float(max_temp):.1f}°C, {note}."


def parse_person_name_locally(text: str) -> dict | None:
    cleaned = re.sub(r"[^A-Za-zÀ-žА-Яа-яЁё'\-\s]", " ", text or "").strip()
    if not cleaned:
        return None
    stop_words = {
        "men", "meni", "mening", "ismim", "familiyam", "ism", "familiya",
        "salom", "assalomu", "alaykum", "doktor", "shifokor", "bemor",
        "is", "my", "name", "surname",
    }
    words = [w.strip("'-").capitalize() for w in cleaned.split() if w.strip("'-")]
    names = [w for w in words if w.lower() not in stop_words and len(w) > 1]
    if not names:
        return None
    return {
        "first_name": names[0],
        "last_name": names[1] if len(names) > 1 else "",
        "is_confident": True,
    }


def is_affirmative(text: str) -> bool:
    q = clean_patient_text(text)
    compact = q.replace(" ", "").replace("'", "")
    words = set(q.split())
    phrases = {"ha", "xa", "aha", "yes", "yeah", "ok", "okay", "togri", "to'g'ri", "tasdiqlayman", "tasdiq", "da"}
    return q in phrases or any(word in words for word in phrases) or "to'g'ri" in q or "togri" in q or "togri" in compact


def is_negative(text: str) -> bool:
    q = clean_patient_text(text)
    compact = q.replace(" ", "").replace("'", "")
    words = set(q.split())
    phrases = {"yoq", "yo'q", "no", "notogri", "noto'g'ri", "xato", "net"}
    return q in phrases or any(word in words for word in phrases) or "noto'g'ri" in q or "notogri" in q or "notogri" in compact or compact == "yoq"


def extract_name_change(text: str) -> dict | None:
    q = (text or "").strip()
    low = q.lower()
    triggers = (
        "ismimni o'zgartir", "ismimni ozgartir", "ismimni o‘zgartir",
        "ismimni almashtir", "ismimni yangila", "ismim boshqa",
        "meni ", "deb saqla", "deb yoz", "ismim endi", "mening ismim endi",
        "change my name", "update my name", "my name is now",
        "измени имя", "поменяй имя", "меня зовут",
    )
    if not any(trigger in low for trigger in triggers):
        return None

    candidate = q
    replacements = [
        "ismimni o'zgartir", "ismimni ozgartir", "ismimni o‘zgartir",
        "ismimni almashtir", "ismimni yangila", "ismim boshqa",
        "mening ismim endi", "ismim endi", "meni", "deb saqla", "deb yoz",
        "change my name to", "update my name to", "my name is now",
        "измени имя на", "поменяй имя на", "меня зовут",
    ]
    for phrase in replacements:
        candidate = re.sub(re.escape(phrase), " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"[:=,\-]+", " ", candidate).strip()
    return parse_person_name_locally(candidate)


def clean_patient_text(text: str) -> str:
    cleaned = (text or "").lower()
    cleaned = cleaned.replace("ʻ", "'").replace("ʼ", "'").replace("‘", "'").replace("’", "'").replace("´", "'")
    cleaned = cleaned.replace("`", "'").replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r"[^\w\s'\-]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_chat_lang(lang: str | None, text: str = "") -> str:
    if lang in ("uz-UZ", "en-US", "ru-RU"):
        return lang
    if re.search(r"[А-Яа-яЁё]", text or ""):
        return "ru-RU"
    q = clean_patient_text(text)
    english_markers = (
        "hello", "hi", "doctor", "pain", "fever", "cough", "heart", "stomach",
        "chest", "pressure", "dizzy", "nausea", "throat", "nose", "ear",
    )
    if any(marker in q for marker in english_markers):
        return "en-US"
    return "uz-UZ"


def is_emergency_case(text: str) -> bool:
    q = clean_patient_text(text)
    emergency_keywords = [
        "hushdan ketdim",
        "hushdan ketish",
        "nafas ololmayapman",
        "nafas olmayapman",
        "ko'kragim juda og'riyapti",
        "kokragim juda ogrigyapti",
        "qattiq qon ketmoqda",
        "qon ketmoqda",
        "insult",
        "yurak xuruji",
        "zaharlanish",
        "og'ir jarohat",
        "ogir jarohat",
        "i fainted",
        "fainting",
        "i cannot breathe",
        "can't breathe",
        "severe chest pain",
        "heavy bleeding",
        "stroke",
        "heart attack",
        "poisoning",
        "serious injury",
        "я потерял сознание",
        "я потеряла сознание",
        "не могу дышать",
        "сильная боль в груди",
        "сильное кровотечение",
        "инсульт",
        "сердечный приступ",
        "отравление",
        "тяжелая травма",
        "тяжёлая травма",
    ]
    return any(keyword in q for keyword in emergency_keywords)


def doctor_router_retrieve(patient_text: str) -> tuple[dict | None, int, list[str]]:
    q = clean_patient_text(patient_text)
    if not q:
        return None, 0, []

    best_doctor = None
    best_score = 0
    best_matches = []

    for doctor in DOCTOR_DIRECTORY:
        score = 0
        matches = []
        searchable = [
            doctor.get("name", ""),
            doctor.get("specialty", ""),
            doctor.get("speciality", ""),
            doctor.get("use_for", ""),
            *doctor.get("keywords", []),
            *DOCTOR_MULTILINGUAL_ALIASES.get(doctor.get("id"), []),
        ]
        for keyword in searchable:
            keyword_clean = clean_patient_text(keyword)
            if not keyword_clean:
                continue
            if keyword_clean in q:
                weight = 3 if keyword_clean in (
                    clean_patient_text(doctor.get("name", "")),
                    clean_patient_text(doctor.get("specialty", "")),
                    clean_patient_text(doctor.get("speciality", "")),
                ) else 1
                if doctor.get("id") == 3 and keyword_clean in (
                    "bola", "bolam", "farzand", "chaqaloq", "child", "baby", "infant",
                    "my child", "ребенок", "ребёнок", "малыш",
                ):
                    weight += 4
                score += weight
                matches.append(keyword)
        if score > best_score:
            best_doctor = doctor
            best_score = score
            best_matches = matches

    if best_doctor:
        return best_doctor, best_score, best_matches

    if any(word in q for word in (
        "shifokor", "doktor", "vrach", "navbat", "qayerga boray", "kimga boray",
        "doctor", "appointment", "queue", "where should i go", "which doctor",
        "врач", "доктор", "очередь", "к какому врачу", "куда идти",
    )):
        return DOCTOR_DIRECTORY[0], 1, ["umumiy qabul"]

    return None, 0, []


def load_doctor_queue() -> list[dict]:
    if not DOCTOR_QUEUE_FILE.exists():
        return []
    try:
        data = json.loads(DOCTOR_QUEUE_FILE.read_text(encoding="utf-8"))
        return list(data.get("queue") or [])
    except Exception as exc:
        logger.warning("Could not read doctor queue: %s", exc)
        return []


def save_doctor_queue(queue: list[dict]) -> None:
    DOCTOR_QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOCTOR_QUEUE_FILE.write_text(
        json.dumps({"queue": queue}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def patient_display_name(current_person: dict | None) -> str:
    if not current_person:
        return "Bemor"
    full_name = (current_person.get("full_name") or "").strip()
    if full_name:
        return full_name
    first = (current_person.get("first_name") or "").strip()
    last = (current_person.get("last_name") or "").strip()
    return f"{first} {last}".strip() or "Bemor"


def doctor_specialty_label(doctor: dict, lang: str | None = None) -> str:
    lang = normalize_chat_lang(lang)
    labels = DOCTOR_SPECIALTY_LABELS.get(doctor.get("id"), {})
    return labels.get(lang) or doctor.get("specialty") or doctor.get("speciality") or "Shifokor"


def add_patient_to_doctor_queue(current_person: dict | None, doctor: dict) -> dict:
    queue = load_doctor_queue()
    today = datetime.now().strftime("%Y-%m-%d")
    person_id = (current_person or {}).get("person_id") or ""
    patient_name = patient_display_name(current_person)
    metadata = (current_person or {}).get("metadata") or {}
    patient_phone = metadata.get("phone") or metadata.get("patient_phone") or "Kiritilmagan"

    for item in queue:
        if (
            item.get("date") == today
            and item.get("doctor_id") == doctor.get("id")
            and item.get("status") == "waiting"
            and (
                (person_id and item.get("person_id") == person_id)
                or (not person_id and item.get("patient_name") == patient_name)
            )
        ):
            return item

    today_queue_for_doctor = [
        item for item in queue
        if item.get("doctor_id") == doctor.get("id") and item.get("date") == today
    ]
    queue_item = {
        "queue_number": len(today_queue_for_doctor) + 1,
        "person_id": person_id,
        "patient_name": patient_name,
        "patient_phone": patient_phone,
        "doctor_id": doctor.get("id"),
        "doctor_name": doctor.get("name"),
        "specialty": doctor.get("specialty") or doctor.get("speciality"),
        "room": doctor.get("room"),
        "work_time": doctor.get("work_time"),
        "date": today,
        "status": "waiting",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    queue.append(queue_item)
    save_doctor_queue(queue)
    return queue_item


def build_doctor_routing_result(text: str, current_person: dict | None, lang: str | None = None) -> dict | None:
    if not (text or "").strip():
        return None
    lang = normalize_chat_lang(lang, text)

    if is_emergency_case(text):
        emergency_messages = {
            "uz-UZ": (
                "Assalomu alaykum. Sizda shoshilinch holat belgilari bo'lishi mumkin. "
                "Iltimos, darhol navbatchi shifokorga murojaat qiling yoki 103 raqamiga qo'ng'iroq qiling."
            ),
            "en-US": (
                "Hello. You may have signs of an emergency condition. "
                "Please contact the duty doctor immediately or call 103."
            ),
            "ru-RU": (
                "Здравствуйте. У вас могут быть признаки экстренного состояния. "
                "Пожалуйста, срочно обратитесь к дежурному врачу или позвоните 103."
            ),
        }
        return {
            "status": "emergency",
            "lang": lang,
            "message": emergency_messages.get(lang, emergency_messages["uz-UZ"]),
        }

    doctor, score, matches = doctor_router_retrieve(text)
    if not doctor:
        return None

    queue_item = add_patient_to_doctor_queue(current_person, doctor)
    original_specialty = doctor.get("specialty") or doctor.get("speciality")
    specialty = doctor_specialty_label(doctor, lang)
    advice = basic_patient_advice(text, lang)
    success_messages = {
        "uz-UZ": f"Assalomu alaykum. Sizning shikoyatingiz bo'yicha {specialty} shifokoriga uchrashish tavsiya qilinadi.",
        "en-US": f"Hello. Based on your complaint, a visit to a {specialty} specialist is recommended.",
        "ru-RU": f"Здравствуйте. По вашей жалобе рекомендуется обратиться к специалисту: {specialty}.",
    }
    return {
        "status": "success",
        "lang": lang,
        "message": success_messages.get(lang, success_messages["uz-UZ"]),
        "doctor": {
            "id": doctor.get("id"),
            "name": doctor.get("name"),
            "specialty": specialty,
            "specialty_original": original_specialty,
            "room": doctor.get("room"),
            "work_time": doctor.get("work_time"),
            "use_for": doctor.get("use_for"),
        },
        "queue": {
            "number": queue_item.get("queue_number"),
            "date": queue_item.get("date"),
            "status": queue_item.get("status"),
        },
        "basic_advice": advice,
        "retrieval": {
            "score": score,
            "matches": matches[:6],
        },
    }


def format_doctor_routing_reply(result: dict, current_person: dict | None = None, lang: str | None = None) -> str:
    lang = normalize_chat_lang(lang or result.get("lang"), "")
    if result.get("status") == "emergency":
        return result["message"]

    doctor = result["doctor"]
    queue = result["queue"]
    advice = result.get("basic_advice")
    name = patient_display_name(current_person)
    has_name = name and name != "Bemor"
    if lang == "en-US":
        patient_prefix = f"{name}, " if has_name else ""
        advice_text = f"\nBasic advice: {advice}\n" if advice else ""
        return (
            f"Hello. {patient_prefix}this is not a diagnosis, only guidance to the right doctor.\n\n"
            f"Recommended specialist: {doctor['specialty']}.\n"
            f"Doctor: {doctor['name']}\n"
            f"Room: {doctor['room']}\n"
            f"Working hours: {doctor['work_time']}\n"
            f"Your queue number: {queue['number']}\n"
            f"Date: {queue['date']}\n\n"
            f"{advice_text}"
            "Please go to the indicated room and wait for your turn."
        )
    if lang == "ru-RU":
        patient_prefix = f"{name}, " if has_name else ""
        advice_text = f"\nБазовая рекомендация: {advice}\n" if advice else ""
        return (
            f"Здравствуйте. {patient_prefix}это не диагноз, а только направление к подходящему врачу.\n\n"
            f"Рекомендуемый специалист: {doctor['specialty']}.\n"
            f"Врач: {doctor['name']}\n"
            f"Кабинет: {doctor['room']}\n"
            f"Время работы: {doctor['work_time']}\n"
            f"Ваш номер очереди: {queue['number']}\n"
            f"Дата: {queue['date']}\n\n"
            f"{advice_text}"
            "Пожалуйста, пройдите в указанный кабинет и ожидайте своей очереди."
        )
    patient_prefix = "" if not has_name else f"{name}, "
    advice_text = f"\nAsosiy tavsiya: {advice}\n" if advice else ""
    return (
        f"Assalomu alaykum. {patient_prefix}bu diagnostika emas, sizni to'g'ri shifokorga yo'naltirish uchun tavsiya.\n\n"
        f"Sizga {doctor['specialty']} shifokori tavsiya qilinadi.\n"
        f"Shifokor: {doctor['name']}\n"
        f"Xona: {doctor['room']}\n"
        f"Ish vaqti: {doctor['work_time']}\n"
        f"Navbat raqamingiz: {queue['number']}\n"
        f"Sana: {queue['date']}\n\n"
        f"{advice_text}"
        "Iltimos, belgilangan xonaga boring va navbatingizni kuting."
    )


def route_patient_request(text: str, current_person: dict | None, lang: str | None = None) -> str | None:
    result = build_doctor_routing_result(text, current_person, lang)
    if not result:
        return None
    return format_doctor_routing_reply(result, current_person, lang)


def doctor_public_info(doctor: dict, lang: str | None = None) -> dict:
    return {
        "id": doctor.get("id"),
        "name": doctor.get("name"),
        "specialty": doctor_specialty_label(doctor, lang),
        "room": doctor.get("room"),
        "work_time": doctor.get("work_time"),
        "use_for": doctor.get("use_for"),
    }


# OpenAI function-calling tools. The model decides when to look up a doctor and when
# to actually book a queue slot — routing/booking no longer fires on every keyword.
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_doctor",
            "description": (
                "Bemor shikoyati bo'yicha mos shifokorni topadi. Navbatga YOZMAYDI. "
                "Bemor qaysi shifokorga borishini bilmoqchi bo'lganda yoki shikoyatdan "
                "kerakli mutaxassisni aniqlash uchun ishlating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {
                        "type": "string",
                        "description": "Bemor shikoyati yoki belgilari (uz/ru/en).",
                    }
                },
                "required": ["symptoms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": (
                "Bemorni tanlangan shifokor navbatiga yozadi va navbat raqamini qaytaradi. "
                "Faqat bemor ko'rikka yozilishni xohlaganda yoki rozi bo'lganda chaqiring. "
                "Bemor shunchaki ma'lumot so'rasa, chaqirmang."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_id": {
                        "type": "integer",
                        "description": "Shifokor ID (find_doctor natijasidagi 'id').",
                    }
                },
                "required": ["doctor_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_patient",
            "description": (
                "Yangi (tanilmagan) bemorni bazaga yozadi. Bemor ism (va iloji bo'lsa familiya) "
                "aytib, u to'g'riligini tasdiqlagandan keyin chaqiring. Kamera yuzni ko'rib turishi kerak. "
                "Bitta sifatli yuz kadri yetarli. Agar natijada reason='need_more_samples' qaytsa, "
                "bemordan kameraga bir oz qarab turishini so'rang va keyin qayta chaqiring. "
                "Bemor allaqachon tanilgan bo'lsa chaqirmang."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string", "description": "Bemor ismi."},
                    "last_name": {"type": "string", "description": "Bemor familiyasi (ixtiyoriy)."},
                },
                "required": ["first_name"],
            },
        },
    },
]


# Internal prompts produced by the registration flow that must be spoken verbatim
# (a confirmation question to the patient), not paraphrased by the LLM.
_VERBATIM_PREFIXES = (
    "Sizning ism-familiyangiz ",
    "Tushunarli. Iltimos,",
    "Iltimos, ism va familiyangiz",
    "Ismni tushunmadim.",
    "Yuz namunasi hali kam.",
    "Yuz embedding tayyor emas.",
    "Yuz rasmi tayyor emas.",
)


def verbatim_internal_reply(user_text: str) -> str | None:
    if any((user_text or "").startswith(prefix) for prefix in _VERBATIM_PREFIXES):
        return user_text
    return None


def basic_patient_advice(text: str, lang: str | None = None) -> str | None:
    q = clean_patient_text(text)
    lang = normalize_chat_lang(lang, text)
    advice = None
    if any(word in q for word in ("bosh og'riq", "bosh og'rig'i", "boshim ogriyapti", "boshim og'riyapti", "headache", "head pain", "головная боль", "болит голова")):
        advice = {
            "uz-UZ": "Hozircha suv iching, tinch joyda dam oling, harorat va qon bosimini tekshirtiring. Og'riq kuchli bo'lsa yoki qayt qilishi, hushdan ketish, ko'rish buzilishi bo'lsa, zudlik bilan shifokorga murojaat qiling.",
            "en-US": "For now, drink water, rest in a quiet place, and check temperature and blood pressure. If the pain is severe or comes with vomiting, fainting, or vision changes, seek urgent medical help.",
            "ru-RU": "Пока выпейте воды, отдохните в тихом месте, проверьте температуру и давление. Если боль сильная или есть рвота, обморок, нарушение зрения, срочно обратитесь к врачу.",
        }
    elif any(word in q for word in ("isitma", "fever", "температура", "жар")):
        advice = {
            "uz-UZ": "Ko'p suyuqlik iching, niqob taqing va haroratni kuzating. Harorat 38°C dan oshsa yoki holsizlik kuchaysa, shifokor ko'rigidan o'ting.",
            "en-US": "Drink plenty of fluids, wear a mask, and monitor temperature. If it rises above 38°C or weakness increases, see a doctor.",
            "ru-RU": "Пейте больше жидкости, наденьте маску и следите за температурой. Если она выше 38°C или слабость усиливается, обратитесь к врачу.",
        }
    elif any(word in q for word in ("yo'tal", "yotal", "cough", "кашель")):
        advice = {
            "uz-UZ": "Niqob taqing, iliq suyuqlik iching va boshqa odamlardan masofa saqlang. Nafas qisishi yoki yuqori isitma bo'lsa, shifokorga boring.",
            "en-US": "Wear a mask, drink warm fluids, and keep distance from others. If you have shortness of breath or high fever, see a doctor.",
            "ru-RU": "Наденьте маску, пейте теплую жидкость и держите дистанцию. При одышке или высокой температуре обратитесь к врачу.",
        }
    elif any(word in q for word in ("qorin", "oshqozon", "stomach", "abdominal", "живот", "желудок")):
        advice = {
            "uz-UZ": "Yengil ovqatlaning, suyuqlik iching va og'riq kuchaysa yoki ich ketishi/qon/qusish bo'lsa, shifokorga murojaat qiling.",
            "en-US": "Eat lightly, drink fluids, and see a doctor if pain worsens or diarrhea, blood, or vomiting appears.",
            "ru-RU": "Ешьте легкую пищу, пейте жидкость и обратитесь к врачу, если боль усиливается, есть понос, кровь или рвота.",
        }
    if not advice:
        return None
    return advice.get(lang, advice["uz-UZ"])


def local_direct_response(user_text: str, current_person=None, lang: str | None = None) -> str | None:
    direct_prefixes = (
        "Sizning ism-familiyangiz ",
        "Tushunarli. Iltimos,",
        "Iltimos, ism va familiyangiz",
        "Ismni tushunmadim.",
        "Yuz namunasi hali kam.",
        "Yuz embedding tayyor emas.",
        "Yuz rasmi tayyor emas.",
    )
    if any((user_text or "").startswith(prefix) for prefix in direct_prefixes):
        return user_text
    q = clean_patient_text(user_text)
    internal_markers = (
        "oldingizda",
        "endi siz bu odamni taniysiz",
        "uzmax routing natijasi",
        "bemorga shu yo'nalishni",
    )
    if any(marker in q for marker in internal_markers):
        return None
    lang = normalize_chat_lang(lang, user_text)
    routed = route_patient_request(user_text, current_person, lang)
    if routed:
        return routed
    greetings = {
        "uz-UZ": "Assalomu alaykum. Men UzMAX tibbiy yordamchiman. Sizni nima bezovta qilyapti?",
        "en-US": "Hello. I am UzMAX medical assistant. What is bothering you?",
        "ru-RU": "Здравствуйте. Я медицинский помощник UzMAX. Что вас беспокоит?",
    }
    if any(word in q for word in ("salom", "assalomu", "alaykum", "hello", "hi", "здравствуйте", "привет")):
        return greetings.get(lang, greetings["uz-UZ"])
    return None


def append_patient_note(person: dict | None, text: str) -> dict | None:
    if not person or not person.get("person_id") or not text.strip():
        return person
    metadata = person.get("metadata") or {}
    notes = list(metadata.get("complaints") or [])
    notes.append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "text": text.strip()[:500],
    })
    metadata["complaints"] = notes[-20:]
    updated = get_face_store().update_metadata(person["person_id"], metadata)
    return updated or {**person, "metadata": metadata}


def decode_base64_image(image_data: str):
    if "," in image_data:
        _, image_data = image_data.split(",", 1)
    image_bytes = base64.b64decode(image_data)
    if cv2 is None:
        return image_bytes, None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image_bytes, frame


def detect_faces_in_base64(image_data: str) -> list[dict]:
    global _FACE_CASCADE
    if cv2 is None:
        return []

    _, frame = decode_base64_image(image_data)
    if frame is None:
        return []

    if _FACE_CASCADE is None:
        cascade_names = (
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        )
        cascades = []
        for cascade_name in cascade_names:
            cascade_path = Path(cv2.data.haarcascades) / cascade_name
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                cascades.append(cascade)
        _FACE_CASCADE = cascades

    if not _FACE_CASCADE:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    found: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    min_size = max(40, min(FACE_MIN_WIDTH_PX, 80))
    detection_passes = (
        {"scaleFactor": 1.1, "minNeighbors": 4, "minSize": (min_size, min_size)},
        {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (40, 40)},
    )

    for cascade in _FACE_CASCADE:
        for options in detection_passes:
            faces = cascade.detectMultiScale(gray, **options)
            for (x, y, w, h) in faces:
                box = (int(x), int(y), int(w), int(h))
                if box in seen:
                    continue
                seen.add(box)
                found.append({"x": box[0], "y": box[1], "w": box[2], "h": box[3], "area": box[2] * box[3]})

        if found:
            break

    return sorted(found, key=lambda item: int(item.get("area", 0)), reverse=True)


def largest_face(faces: list[dict]) -> dict | None:
    if not faces:
        return None
    return max(faces, key=lambda item: int(item.get("w", 0)) * int(item.get("h", 0)))


def face_quality(frame, face: dict | None) -> dict:
    if cv2 is None or frame is None or not face:
        return {"ok": False, "reason": "no_face", "blur": 0.0, "min_width": FACE_MIN_WIDTH_PX}
    x = int(face.get("x", 0))
    y = int(face.get("y", 0))
    w = int(face.get("w", 0))
    h = int(face.get("h", 0))
    if w < FACE_MIN_WIDTH_PX:
        return {
            "ok": False,
            "reason": "face_too_small",
            "message": f"Face width is {w}px; need at least {FACE_MIN_WIDTH_PX}px.",
            "blur": 0.0,
            "width": w,
            "min_width": FACE_MIN_WIDTH_PX,
        }
    height, width = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"ok": False, "reason": "empty_crop", "blur": 0.0, "width": w, "min_width": FACE_MIN_WIDTH_PX}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < FACE_MIN_BLUR_VAR:
        return {
            "ok": False,
            "reason": "face_blurry",
            "message": f"Face blur score is {blur:.1f}; need at least {FACE_MIN_BLUR_VAR:.1f}.",
            "blur": round(blur, 1),
            "width": w,
            "min_width": FACE_MIN_WIDTH_PX,
        }
    return {
        "ok": True,
        "reason": "good",
        "blur": round(blur, 1),
        "width": w,
        "min_width": FACE_MIN_WIDTH_PX,
        "area": w * h,
    }


def crop_selected_face_base64(image_data: str, face: dict | None) -> str:
    if cv2 is None or not face:
        return image_data

    _, frame = decode_base64_image(image_data)
    if frame is None:
        return image_data

    height, width = frame.shape[:2]
    x = int(face.get("x", 0))
    y = int(face.get("y", 0))
    w = int(face.get("w", 0))
    h = int(face.get("h", 0))
    if w <= 0 or h <= 0:
        return image_data

    pad_x = int(w * 0.45)
    pad_y = int(h * 0.65)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return image_data

    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return image_data
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def crop_largest_face_base64(image_data: str, faces: list[dict]) -> str:
    return crop_selected_face_base64(image_data, largest_face(faces))


UZ_UNITS = {
    0: "nol",
    1: "bir",
    2: "ikki",
    3: "uch",
    4: "to'rt",
    5: "besh",
    6: "olti",
    7: "yetti",
    8: "sakkiz",
    9: "to'qqiz",
    10: "o'n",
    11: "o'n bir",
    12: "o'n ikki",
    13: "o'n uch",
    14: "o'n to'rt",
    15: "o'n besh",
    16: "o'n olti",
    17: "o'n yetti",
    18: "o'n sakkiz",
    19: "o'n to'qqiz",
}
UZ_TENS = {
    20: "yigirma",
    30: "o'ttiz",
    40: "qirq",
    50: "ellik",
    60: "oltmish",
    70: "yetmish",
    80: "sakson",
    90: "to'qson",
}


def uz_number_to_words(value: int) -> str:
    if value < 0:
        return "minus " + uz_number_to_words(abs(value))
    if value < 20:
        return UZ_UNITS[value]
    if value < 100:
        ten = (value // 10) * 10
        rest = value % 10
        return UZ_TENS[ten] if rest == 0 else f"{UZ_TENS[ten]} {UZ_UNITS[rest]}"
    if value < 1000:
        hundred = value // 100
        rest = value % 100
        prefix = "yuz" if hundred == 1 else f"{UZ_UNITS[hundred]} yuz"
        return prefix if rest == 0 else f"{prefix} {uz_number_to_words(rest)}"
    if value < 1_000_000:
        thousand = value // 1000
        rest = value % 1000
        prefix = "ming" if thousand == 1 else f"{uz_number_to_words(thousand)} ming"
        return prefix if rest == 0 else f"{prefix} {uz_number_to_words(rest)}"
    return str(value)


def uz_ordinal_to_words(value: int) -> str:
    word = uz_number_to_words(value)
    if word.endswith(("i", "a", "o", "u", "o'")):
        return word + "nchi"
    return word + "inchi"


def uz_time_to_words(hour_text: str, minute_text: str) -> str:
    hour = int(hour_text)
    minute = int(minute_text)
    if minute == 0:
        return f"soat {uz_number_to_words(hour)}"
    return f"soat {uz_number_to_words(hour)}dan {uz_number_to_words(minute)} daqiqa o'tdi"


def tts_readable_text(text: str, lang: str = "uz-UZ") -> str:
    """Keep UI text numeric, but send TTS a pronunciation-friendly version."""
    if not text:
        return text
    if not (lang or "").lower().startswith("uz"):
        return text

    spoken = text
    spoken = re.sub(
        r"\b([01]?\d|2[0-3]):([0-5]\d)\s*[-–—]\s*([01]?\d|2[0-3]):([0-5]\d)\b",
        lambda m: f"{uz_time_to_words(m.group(1), m.group(2))}dan {uz_time_to_words(m.group(3), m.group(4)).replace('soat ', '')}gacha",
        spoken,
    )
    spoken = re.sub(
        r"\b([01]?\d|2[0-3]):([0-5]\d)\b",
        lambda m: uz_time_to_words(m.group(1), m.group(2)),
        spoken,
    )
    spoken = re.sub(
        r"\b(\d{1,3})\s*[- ]?\s*xona\b",
        lambda m: f"{uz_ordinal_to_words(int(m.group(1)))} xona",
        spoken,
        flags=re.IGNORECASE,
    )
    spoken = re.sub(
        r"\b(\d{1,4})\s*[- ]?\s*navbat\b",
        lambda m: f"{uz_ordinal_to_words(int(m.group(1)))} navbat",
        spoken,
        flags=re.IGNORECASE,
    )
    spoken = re.sub(
        r"\b(\d{1,2})([.,])(\d)\s*(?:°\s*)?C\b",
        lambda m: f"{uz_number_to_words(int(m.group(1)))} butun {uz_number_to_words(int(m.group(3)))} daraja",
        spoken,
        flags=re.IGNORECASE,
    )
    spoken = re.sub(
        r"\b(\d{1,4})\s*°\b",
        lambda m: f"{uz_number_to_words(int(m.group(1)))} daraja",
        spoken,
    )
    spoken = re.sub(r"\b103\s*(?:ga|ni)?\b", "bir yuz uchga", spoken, flags=re.IGNORECASE)
    spoken = re.sub(r"\b(\d{1,4})\b", lambda m: uz_number_to_words(int(m.group(1))), spoken)
    return spoken


def averaged_embedding(samples: list[dict]) -> list[float]:
    embeddings = [sample.get("embedding") for sample in samples if sample.get("embedding")]
    if not embeddings:
        return []
    arr = np.asarray(embeddings, dtype=np.float32)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm:
        mean = mean / norm
    return mean.tolist()


def save_base64_image(image_data: str) -> str:
    image_bytes, _ = decode_base64_image(image_data)
    for old_file in IMAGES_DIR.glob("face_*.jpg"):
        try:
            old_file.unlink()
        except OSError:
            logger.warning("Could not delete old face snapshot: %s", old_file)
    ts           = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename     = IMAGES_DIR / f"face_{ts}.jpg"
    filename.write_bytes(image_bytes)
    return str(filename)


def persist_registered_face_snapshot(snapshot_path: str | None, person_id: str) -> str | None:
    if not snapshot_path or not person_id:
        return None
    source = Path(snapshot_path)
    if not source.exists():
        logger.warning("Pending registration snapshot does not exist: %s", source)
        return None
    suffix = source.suffix or ".jpg"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", person_id).strip("_") or "person"
    target = REGISTER_FACES_DIR / f"{safe_id}{suffix}"
    try:
        shutil.copy2(source, target)
        return str(target)
    except OSError as exc:
        logger.warning("Could not persist registered face snapshot %s: %s", source, exc)
        return None


def save_registration_pending_snapshot(image_data: str) -> str | None:
    image_bytes, _ = decode_base64_image(image_data)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = REGISTER_FACES_DIR / f"pending_{ts}.jpg"
    try:
        filename.write_bytes(image_bytes)
        for old_file in sorted(REGISTER_FACES_DIR.glob("pending_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[12:]:
            try:
                old_file.unlink()
            except OSError:
                logger.warning("Could not delete old pending face snapshot: %s", old_file)
        return str(filename)
    except OSError as exc:
        logger.warning("Could not save pending registration snapshot: %s", exc)
        return None


def write_registered_face_registry(faces: list[dict]) -> None:
    REGISTER_FACES_DIR.mkdir(parents=True, exist_ok=True)
    REGISTER_FACES_JSON.write_text(json.dumps({"faces": faces}, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_registered_face_registry(person: dict, snapshot_path: str | None = None) -> None:
    if not person or not person.get("person_id"):
        return
    file_name = None
    if snapshot_path:
        snapshot = Path(snapshot_path).resolve()
        register_dir = REGISTER_FACES_DIR.resolve()
        if not snapshot.exists() or snapshot.parent != register_dir:
            logger.warning("Registry write skipped because snapshot is not in register_faces: %s", snapshot)
            return
        file_name = snapshot.name
    try:
        data = json.loads(REGISTER_FACES_JSON.read_text(encoding="utf-8")) if REGISTER_FACES_JSON.exists() else {}
    except Exception:
        data = {}
    faces = list(data.get("faces") or [])
    existing = next((item for item in faces if item.get("person_id") == person.get("person_id")), {})
    entry = {
        "person_id": person.get("person_id"),
        "file": file_name or existing.get("file"),
        "first_name": person.get("first_name", ""),
        "last_name": person.get("last_name", ""),
        "metadata": person.get("metadata") or {},
        "registered_at": existing.get("registered_at") or datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    replaced = False
    for idx, item in enumerate(faces):
        if item.get("person_id") == entry["person_id"]:
            faces[idx] = entry
            replaced = True
            break
    if not replaced:
        faces.append(entry)
    write_registered_face_registry(faces)
    logger.info(
        "Registered face written to registry.json: person_id=%s file=%s name=%s %s",
        entry["person_id"],
        entry["file"],
        entry["first_name"],
        entry["last_name"],
    )


def update_registered_face_registry_name(person: dict) -> None:
    if not person or not person.get("person_id") or not REGISTER_FACES_JSON.exists():
        return
    try:
        data = json.loads(REGISTER_FACES_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read registry.json for name update: %s", exc)
        return
    faces = list(data.get("faces") or [])
    changed = False
    for item in faces:
        if item.get("person_id") == person.get("person_id"):
            item["first_name"] = person.get("first_name", "")
            item["last_name"] = person.get("last_name", "")
            item["metadata"] = person.get("metadata") or item.get("metadata") or {}
            item["updated_at"] = datetime.now().isoformat(timespec="seconds")
            changed = True
            break
    if changed:
        REGISTER_FACES_JSON.write_text(json.dumps({"faces": faces}, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Registry name updated: person_id=%s name=%s", person.get("person_id"), person.get("full_name"))


def load_registered_faces() -> tuple[int, int]:
    if not REGISTER_FACES_JSON.exists():
        return 0, 0
    try:
        data = json.loads(REGISTER_FACES_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to read registry: %s", exc)
        return 0, 0

    items   = data.get("faces", []) if isinstance(data, dict) else []
    loaded  = 0
    skipped = 0

    for item in items:
        file_name  = item.get("file")
        person_id  = item.get("person_id")
        first_name = item.get("first_name", "")
        last_name  = item.get("last_name", "")
        metadata   = item.get("metadata", {})

        if not file_name or not first_name:
            skipped += 1
            continue

        image_path = (REGISTER_FACES_DIR / file_name).resolve()
        if not image_path.exists():
            skipped += 1
            continue

        store = get_face_store()
        existing = store.get_person(person_id) if person_id else None
        if existing:
            store.add_snapshot(person_id, str(image_path))
            loaded += 1
            continue

        try:
            embedding = get_face_encoder().extract_embedding_from_path(str(image_path))
            store.register(
                embedding=embedding,
                first_name=first_name,
                last_name=last_name,
                snapshot_path=str(image_path),
                metadata=metadata,
                person_id=person_id,
            )
            loaded += 1
        except Exception as exc:
            logger.error("Bootstrap face %s: %s", file_name, exc)
            skipped += 1

    logger.info("Face bootstrap done: loaded=%s skipped=%s", loaded, skipped)
    return loaded, skipped


def read_registered_face_registry() -> list[dict]:
    if not REGISTER_FACES_JSON.exists():
        return []
    try:
        data = json.loads(REGISTER_FACES_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read registered faces registry: %s", exc)
        return []
    return list(data.get("faces") or []) if isinstance(data, dict) else []


def safe_person_file_id(person_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", person_id).strip("_") or "person"


def image_suffix_from_data_url(image_data: str) -> str:
    header = image_data.split(",", 1)[0].lower() if "," in image_data else ""
    if "image/png" in header:
        return ".png"
    if "image/webp" in header:
        return ".webp"
    return ".jpg"


def save_patient_face_image(image_data: str | None, person_id: str) -> str | None:
    if not image_data:
        return None
    image_bytes, _ = decode_base64_image(image_data)
    REGISTER_FACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = REGISTER_FACES_DIR / f"{safe_person_file_id(person_id)}_{ts}{image_suffix_from_data_url(image_data)}"
    filename.write_bytes(image_bytes)
    return str(filename)


def split_patient_name(full_name: str) -> tuple[str, str]:
    parts = [part for part in full_name.strip().split() if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def patient_payload_to_person(payload: dict, person_id: str | None = None, existing: dict | None = None) -> dict:
    existing = existing or {}
    metadata = dict(existing.get("metadata") or {})
    incoming_metadata = payload.get("metadata")
    if isinstance(incoming_metadata, dict):
        metadata.update(incoming_metadata)

    first_name = str(payload.get("first_name") or existing.get("first_name") or "").strip()
    last_name = str(payload.get("last_name") or existing.get("last_name") or "").strip()
    full_name = str(payload.get("full_name") or "").strip()
    if full_name and not first_name:
        first_name, last_from_full = split_patient_name(full_name)
        if not last_name:
            last_name = last_from_full

    for key, meta_key in (
        ("age", "age"),
        ("phone", "phone"),
        ("notes", "notes"),
        ("birthday", "birthday"),
        ("date_of_birth", "date_of_birth"),
    ):
        value = payload.get(key)
        if value not in (None, ""):
            metadata[meta_key] = value
        elif key in payload and value == "":
            metadata.pop(meta_key, None)

    if not first_name and not last_name:
        raise ValueError("Patient name is required")

    metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if not existing:
        metadata.setdefault("created_at", metadata["updated_at"])

    return {
        "person_id": person_id or str(uuid.uuid4()),
        "first_name": first_name,
        "last_name": last_name,
        "full_name": f"{first_name} {last_name}".strip(),
        "metadata": metadata,
    }


def registry_entry_for_person(person_id: str) -> dict | None:
    return next((item for item in read_registered_face_registry() if item.get("person_id") == person_id), None)


def remove_registry_file(entry: dict | None) -> str | None:
    file_name = (entry or {}).get("file")
    if not file_name:
        return None
    try:
        target = (REGISTER_FACES_DIR / file_name).resolve()
        if target.exists() and target.is_file() and target.parent == REGISTER_FACES_DIR.resolve():
            target.unlink()
            return str(target)
    except Exception as exc:
        logger.warning("Could not delete registry face image %s: %s", file_name, exc)
    return None


def upsert_patient_record(payload: dict, person_id: str | None = None) -> dict:
    existing_registry = registry_entry_for_person(person_id) if person_id else None
    existing_store = get_face_store().get_person(person_id) if person_id else None
    existing = existing_store or existing_registry or {}
    person = patient_payload_to_person(payload, person_id=person_id, existing=existing)
    image_path = save_patient_face_image(payload.get("image"), person["person_id"])

    vector_status = "unchanged"
    if image_path:
        remove_registry_file(existing_registry)
        try:
            embedding = get_face_encoder().extract_embedding_from_path(image_path)
            get_face_store().delete_person(person["person_id"])
            get_face_store().register(
                embedding=embedding,
                first_name=person["first_name"],
                last_name=person["last_name"],
                snapshot_path=image_path,
                metadata=person["metadata"],
                person_id=person["person_id"],
            )
            vector_status = "updated"
        except Exception as exc:
            vector_status = f"image saved, embedding failed: {exc}"
            logger.warning("Could not update face embedding for %s: %s", person["person_id"], exc)
    else:
        try:
            updated_name = get_face_store().update_name(person["person_id"], person["first_name"], person["last_name"])
            updated_meta = get_face_store().update_metadata(person["person_id"], person["metadata"])
            if updated_name or updated_meta:
                vector_status = "metadata updated"
        except Exception as exc:
            logger.warning("Could not update vector metadata for %s: %s", person["person_id"], exc)

    upsert_registered_face_registry(person, image_path)
    return {"person": person, "image_path": image_path, "vector_status": vector_status}


def delete_patient_record(person_id: str) -> dict:
    deleted = {"registry": False, "image": None, "vectors": 0, "queue": 0}

    faces = read_registered_face_registry()
    kept_faces = []
    removed_entry = None
    for item in faces:
        if item.get("person_id") == person_id:
            removed_entry = item
        else:
            kept_faces.append(item)
    if removed_entry is not None:
        write_registered_face_registry(kept_faces)
        deleted["registry"] = True
        deleted["image"] = remove_registry_file(removed_entry)

    try:
        deleted["vectors"] = get_face_store().delete_person(person_id)
    except Exception as exc:
        logger.warning("Could not delete face vectors for %s: %s", person_id, exc)

    queue = load_doctor_queue()
    queue_name = person_id.removeprefix("queue:") if person_id.startswith("queue:") else None
    kept_queue = [
        item for item in queue
        if item.get("person_id") != person_id and not (queue_name and item.get("patient_name") == queue_name)
    ]
    if len(kept_queue) != len(queue):
        save_doctor_queue(kept_queue)
        deleted["queue"] = len(queue) - len(kept_queue)

    return deleted


def safe_image_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    try:
        path = Path(path_value)
        path = (BASE_DIR / path).resolve() if not path.is_absolute() else path.resolve()
        allowed_dirs = [REGISTER_FACES_DIR.resolve(), IMAGES_DIR.resolve()]
        if not path.exists() or not path.is_file():
            return None
        if not any(path.is_relative_to(directory) for directory in allowed_dirs):
            return None
        return "/api/patients/image?path=" + quote(str(path), safe="")
    except Exception:
        return None


def parse_datetime_value(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except Exception:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except Exception:
            pass
    return None


def metadata_age(metadata: dict) -> int | str | None:
    for key in ("age", "patient_age", "yosh"):
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    for key in ("birth_date", "date_of_birth", "dob", "birthday"):
        born = parse_datetime_value(metadata.get(key))
        if born:
            today = datetime.now().date()
            birth_date = born.date()
            return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return None


def merge_patient_record(records: dict, person_id: str, values: dict) -> dict:
    record = records.setdefault(
        person_id,
        {
            "person_id": person_id,
            "first_name": "",
            "last_name": "",
            "full_name": "",
            "age": None,
            "snapshots": [],
            "image_url": None,
            "metadata": {},
            "face_points": 0,
            "registry": None,
            "queue": [],
            "hospital_patient": None,
            "hospital_visits": [],
            "last_visited_at": None,
            "source": [],
        },
    )
    for key in ("first_name", "last_name", "full_name"):
        if values.get(key):
            record[key] = values[key]
    if values.get("metadata"):
        record["metadata"] = {**record.get("metadata", {}), **values["metadata"]}
    if values.get("age") not in (None, ""):
        record["age"] = values["age"]
    for snapshot in values.get("snapshots") or []:
        if snapshot and snapshot not in record["snapshots"]:
            record["snapshots"].append(snapshot)
    if values.get("image_url"):
        record["image_url"] = values["image_url"]
    if values.get("face_points"):
        record["face_points"] = max(int(record.get("face_points") or 0), int(values["face_points"]))
    for source in values.get("source") or []:
        if source not in record["source"]:
            record["source"].append(source)
    return record


def build_patient_records() -> list[dict]:
    records: dict[str, dict] = {}
    name_index: dict[str, str] = {}

    try:
        people = get_face_store().list_people()
    except Exception as exc:
        logger.warning("Could not list face people: %s", exc)
        people = []

    for person in people:
        person_id = person.get("person_id") or ""
        if not person_id:
            continue
        metadata = person.get("metadata") or {}
        snapshots = person.get("snapshots") or []
        image_url = next((url for url in (safe_image_url(path) for path in snapshots) if url), None)
        record = merge_patient_record(records, person_id, {
            **person,
            "age": metadata_age(metadata),
            "image_url": image_url,
            "source": ["face_id"],
        })
        full_name = (record.get("full_name") or "").strip().lower()
        if full_name:
            name_index[full_name] = person_id

    for entry in read_registered_face_registry():
        person_id = entry.get("person_id") or f"registry:{entry.get('file', '')}"
        image_path = str((REGISTER_FACES_DIR / str(entry.get("file") or "")).resolve()) if entry.get("file") else None
        metadata = entry.get("metadata") or {}
        record = merge_patient_record(records, person_id, {
            "first_name": entry.get("first_name", ""),
            "last_name": entry.get("last_name", ""),
            "full_name": f'{entry.get("first_name", "")} {entry.get("last_name", "")}'.strip(),
            "metadata": metadata,
            "age": metadata_age(metadata),
            "snapshots": [image_path] if image_path else [],
            "image_url": safe_image_url(image_path),
            "source": ["registry"],
        })
        record["registry"] = entry
        if record.get("full_name"):
            name_index[record["full_name"].strip().lower()] = person_id

    for item in load_doctor_queue():
        person_id = item.get("person_id") or ""
        name = (item.get("patient_name") or "").strip()
        if not person_id:
            person_id = name_index.get(name.lower()) or f"queue:{name or item.get('created_at', '')}"
        record = merge_patient_record(records, person_id, {
            "full_name": name,
            "metadata": {"phone": item.get("patient_phone")} if item.get("patient_phone") else {},
            "source": ["doctor_queue"],
        })
        record["queue"].append(item)

    try:
        hospital_patients = get_hospital_patients()
        hospital_visits = get_hospital_visits()
    except Exception as exc:
        logger.warning("Could not load hospital patients/visits: %s", exc)
        hospital_patients, hospital_visits = [], []

    for item in hospital_patients:
        name = (item.get("full_name") or "").strip()
        person_id = name_index.get(name.lower()) or f"hospital:{item.get('id')}"
        record = merge_patient_record(records, person_id, {
            "full_name": name,
            "source": ["hospital_intake"],
        })
        record["hospital_patient"] = item

    for visit in hospital_visits:
        name = (visit.get("full_name") or "").strip()
        person_id = name_index.get(name.lower()) or f"hospital_visit:{name or visit.get('id')}"
        record = merge_patient_record(records, person_id, {
            "full_name": name,
            "source": ["hospital_visits"],
        })
        record["hospital_visits"].append(visit)

    for record in records.values():
        metadata = record.get("metadata") or {}
        if record.get("age") in (None, ""):
            record["age"] = metadata_age(metadata)
        last_candidates = [
            metadata.get("last_seen_at"),
            metadata.get("last_visit_at"),
            metadata.get("updated_at"),
            metadata.get("created_at"),
            (record.get("registry") or {}).get("updated_at"),
            (record.get("registry") or {}).get("registered_at"),
            (record.get("hospital_patient") or {}).get("created_at"),
            *[item.get("created_at") for item in record.get("queue") or []],
            *[item.get("created_at") for item in record.get("hospital_visits") or []],
            *[
                item.get("time")
                for item in (metadata.get("thermal_screenings") or [])
                if isinstance(item, dict)
            ],
        ]
        parsed = [dt for dt in (parse_datetime_value(value) for value in last_candidates) if dt]
        record["last_visited_at"] = max(parsed).isoformat(timespec="seconds") if parsed else None
        record["snapshot_count"] = len(record.get("snapshots") or [])

    return sorted(records.values(), key=lambda item: item.get("last_visited_at") or "", reverse=True)


def reset_face_id_data() -> dict:
    global face_store

    deleted_files = []
    deleted_dirs = []
    errors = []
    warnings = []

    def remove_path(path: Path, *, retries: int = 5) -> bool:
        for attempt in range(retries):
            try:
                if not path.exists():
                    return True
                if path.is_dir():
                    shutil.rmtree(path)
                    deleted_dirs.append(str(path))
                else:
                    path.unlink()
                    deleted_files.append(str(path))
                return True
            except Exception as exc:
                if attempt == retries - 1:
                    warnings.append(f"{path}: {exc}")
                    return False
                gc.collect()
                time.sleep(0.25)
        return False

    store = face_store
    vectors_reset = False
    if store is not None:
        try:
            store.reset()
            vectors_reset = True
        except Exception as exc:
            errors.append(f"vector collection: {exc}")
        try:
            close = getattr(store.client, "close", None)
            if close:
                close()
        except Exception:
            pass
        face_store = None
        gc.collect()

    qdrant_dir = DATA_DIR / "faces" / "qdrant"
    if not remove_path(qdrant_dir):
        # Qdrant can keep storage.sqlite locked briefly after closing. If reset()
        # already dropped the collection, keep the button working and recreate the
        # empty store directory on next restart/use.
        if not vectors_reset:
            errors.append(f"{qdrant_dir}: could not remove locked vector store")
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    try:
        REGISTER_FACES_DIR.mkdir(parents=True, exist_ok=True)
        for item in REGISTER_FACES_DIR.iterdir():
            if item.name == "registry.example.json":
                continue
            if not remove_path(item):
                errors.append(f"{item}: could not remove")
    except Exception as exc:
        errors.append(f"{REGISTER_FACES_DIR}: {exc}")

    try:
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for item in IMAGES_DIR.glob("face_*.jpg"):
            if not remove_path(item):
                errors.append(f"{item}: could not remove")
        for item in IMAGES_DIR.glob("pending_*.jpg"):
            if not remove_path(item):
                errors.append(f"{item}: could not remove")
    except Exception as exc:
        errors.append(f"{IMAGES_DIR}: {exc}")

    face_store = None

    return {
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "warnings": warnings,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════
#  FACE API
# ═══════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/api/faces/reset")
async def reset_faces():
    result = await asyncio.to_thread(reset_face_id_data)
    status_code = 500 if result["errors"] else 200
    return JSONResponse({"ok": not result["errors"], **result}, status_code=status_code)


@app.post("/api/faces")
async def save_face_snapshot(payload: dict):
    image_data = payload.get("image")
    if not image_data:
        return JSONResponse({"ok": False, "error": "image is required"}, status_code=400)
    try:
        filename = save_base64_image(image_data)
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid image"}, status_code=400)
    return JSONResponse({"ok": True, "path": filename})


@app.post("/api/faces/detect")
async def detect_faces(payload: dict):
    """Fast, local-only face detection (OpenCV Haar) for live box drawing.

    No InsightFace embedding / Qdrant lookup, so it can run at a high frame rate. The
    browser calls this frequently for responsive boxes and only calls the heavier
    /api/faces/identify-local when a face is actually present and the cooldown elapsed.
    """
    image_data = payload.get("image")
    if not image_data:
        return JSONResponse({"faces": [], "status": "no_face"})

    detected = await asyncio.to_thread(detect_faces_in_base64, image_data)
    if not detected:
        return JSONResponse({"faces": [], "status": "no_face"})

    selected = largest_face(detected)
    visible_faces = [selected] if selected else []
    _, frame = decode_base64_image(image_data)
    quality = face_quality(frame, selected)
    return JSONResponse({
        "faces": visible_faces,
        "selected_face": selected,
        "quality": quality,
        "status": "present",
        "detected_count": len(detected),
    })


async def identify_faces_local_payload(payload: dict):
    faces   = payload.get("faces", [])
    results = []

    for face in faces:
        image_data = face.get("image")
        if not image_data:
            continue
        client_selected = face.get("selected_face") if isinstance(face.get("selected_face"), dict) else None
        detected_faces = [client_selected] if client_selected else detect_faces_in_base64(image_data)
        if cv2 is not None and not detected_faces:
            results.append({"status": "no_face", "faces": [], "local": True, "provider": "insightface_buffalo_s"})
            continue

        selected_face = client_selected or largest_face(detected_faces)
        visible_faces = [selected_face] if selected_face else []
        _, frame = decode_base64_image(image_data)
        quality = face_quality(frame, selected_face)
        if cv2 is not None and not quality.get("ok"):
            results.append({
                "status": "bad_quality",
                "local": True,
                "provider": "insightface_buffalo_s",
                "reason": quality.get("reason"),
                "message": quality.get("message"),
                "faces": visible_faces,
                "selected_face": selected_face,
                "detected_count": len(detected_faces),
                "quality": quality,
            })
            continue

        face_image_data = crop_selected_face_base64(image_data, selected_face)
        try:
            snapshot_path = save_base64_image(face_image_data)
            embedding     = get_face_encoder().extract_embedding_from_base64(face_image_data)
        except Exception as exc:
            logger.warning("Local face identify failed: %s", exc)
            results.append({
                "status": "error",
                "local": True,
                "provider": "insightface_buffalo_s",
                "message": str(exc),
                "faces": visible_faces,
                "selected_face": selected_face,
                "detected_count": len(detected_faces),
                "quality": quality,
            })
            continue

        store = get_face_store()
        effective_threshold = max(FACE_MATCH_THRESHOLD, FACE_MATCH_MIN_CONFIDENCE)
        match = store.identify(embedding, threshold=effective_threshold, margin=FACE_MATCH_MARGIN)

        if match:
            comps = match.get("comparisons", [])
            for c in (comps if FACE_LOG_ALL else comps[:3]):
                logger.info("Face compare: threshold=%.2f margin=%.2f id=%s name=%s score=%.4f",
                            effective_threshold, FACE_MATCH_MARGIN, c.get("person_id"),
                            c.get("full_name"), c.get("score", 0.0))
            logger.info("Face best: matched=%s reason=%s score=%.4f second=%s margin=%s name=%s",
                        match.get("matched"), match.get("reject_reason"),
                        match.get("score", 0.0), match.get("second_score"),
                        match.get("score_margin"), match.get("full_name"))

        if match and match.get("matched"):
            store.add_snapshot(match["person_id"], snapshot_path)
            results.append({
                "status": "known",
                "local": True,
                "provider": "insightface_buffalo_s",
                "snapshot_path": snapshot_path,
                "person": match,
                "faces": visible_faces,
                "selected_face": selected_face,
                "detected_count": len(detected_faces),
                "quality": quality,
            })
        else:
            pending_snapshot_path = save_registration_pending_snapshot(face_image_data) or snapshot_path
            results.append({
                "status": "unknown",
                "local": True,
                "provider": "insightface_buffalo_s",
                "snapshot_path": pending_snapshot_path,
                "embedding": embedding,
                "faces": visible_faces,
                "selected_face": selected_face,
                "detected_count": len(detected_faces),
                "quality": quality,
                "min_samples": FACE_MIN_SAMPLES,
            })

    return JSONResponse({"faces": results})


@app.post("/api/faces/identify-local")
async def identify_faces_local(payload: dict):
    return await identify_faces_local_payload(payload)


@app.post("/api/faces/identify")
async def identify_faces(payload: dict):
    return await identify_faces_local_payload(payload)


@app.get("/api/patients/full")
async def patients_full():
    records = await asyncio.to_thread(build_patient_records)
    return JSONResponse({"ok": True, "count": len(records), "patients": records})


@app.post("/api/patients")
async def create_patient(payload: dict):
    try:
        result = await asyncio.to_thread(upsert_patient_record, payload, None)
        records = await asyncio.to_thread(build_patient_records)
        patient = next((item for item in records if item.get("person_id") == result["person"]["person_id"]), result["person"])
        return JSONResponse({"ok": True, "patient": patient, "vector_status": result["vector_status"]})
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("Create patient failed")
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)


@app.put("/api/patients/{person_id:path}")
async def update_patient(person_id: str, payload: dict):
    try:
        result = await asyncio.to_thread(upsert_patient_record, payload, person_id)
        records = await asyncio.to_thread(build_patient_records)
        patient = next((item for item in records if item.get("person_id") == result["person"]["person_id"]), result["person"])
        return JSONResponse({"ok": True, "patient": patient, "vector_status": result["vector_status"]})
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("Update patient failed: %s", person_id)
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)


@app.delete("/api/patients/{person_id:path}")
async def delete_patient(person_id: str):
    try:
        deleted = await asyncio.to_thread(delete_patient_record, person_id)
        return JSONResponse({"ok": True, "deleted": deleted})
    except Exception as exc:
        logger.exception("Delete patient failed: %s", person_id)
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)


@app.get("/api/patients/image")
async def patient_image(path: str):
    try:
        image_path = Path(path)
        image_path = (BASE_DIR / image_path).resolve() if not image_path.is_absolute() else image_path.resolve()
        allowed_dirs = [REGISTER_FACES_DIR.resolve(), IMAGES_DIR.resolve()]
        if not image_path.exists() or not image_path.is_file():
            return JSONResponse({"ok": False, "message": "Image not found"}, status_code=404)
        if not any(image_path.is_relative_to(directory) for directory in allowed_dirs):
            return JSONResponse({"ok": False, "message": "Image path is not allowed"}, status_code=403)
        return FileResponse(image_path)
    except Exception as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)


@app.get("/api/doctor/directory")
async def doctor_directory():
    return JSONResponse({"ok": True, "doctors": DOCTOR_DIRECTORY})


@app.get("/api/doctor/queue")
async def doctor_queue():
    return JSONResponse({"ok": True, "queue": load_doctor_queue()})


@app.post("/api/doctor/route")
async def doctor_route(payload: dict):
    person = payload.get("person") if isinstance(payload.get("person"), dict) else None
    patient_name = (payload.get("patient_name") or "").strip()
    patient_phone = (payload.get("patient_phone") or "").strip()
    if not person and (patient_name or patient_phone):
        name_parts = patient_name.split()
        person = {
            "first_name": name_parts[0] if name_parts else patient_name,
            "last_name": " ".join(name_parts[1:]) if len(name_parts) > 1 else "",
            "full_name": patient_name or "Bemor",
            "metadata": {"phone": patient_phone} if patient_phone else {},
        }

    patient_message = (payload.get("patient_message") or payload.get("message") or "").strip()
    lang = normalize_chat_lang(payload.get("lang"), patient_message)
    result = build_doctor_routing_result(patient_message, person, lang)
    if not result:
        no_match_messages = {
            "uz-UZ": "Shikoyat bo'yicha aniq yo'nalish topilmadi. Qabul shifokoriga murojaat qiling.",
            "en-US": "No clear route was found for this complaint. Please contact the reception doctor.",
            "ru-RU": "По этой жалобе точное направление не найдено. Пожалуйста, обратитесь к врачу первичного приема.",
        }
        return JSONResponse({
            "ok": False,
            "status": "no_match",
            "lang": lang,
            "message": no_match_messages.get(lang, no_match_messages["uz-UZ"]),
        })

    return JSONResponse({
        "ok": True,
        **result,
        "reply": format_doctor_routing_reply(result, person, lang),
    })


# ═══════════════════════════════════════════════════════════════════
#  VOICE AI WEBSOCKET
# ═══════════════════════════════════════════════════════════════════

def flush_sentence_buffer(buf: str) -> tuple[str | None, str]:
    m = re.search(r'[.!?…\n]\s', buf)
    if m:
        split_pos = m.end()
        return buf[:split_pos].strip(), buf[split_pos:]
    if len(buf) > 150:
        last_space = buf.rfind(' ', 0, 150)
        if last_space > 30:
            return buf[:last_space].strip(), buf[last_space:]
        return buf.strip(), ""
    return None, buf


def build_system_prompt(current_person: dict | None, onboarding: bool, current_lang: str) -> str:
    if current_lang == "uz-UZ":
        lang_instr = "Faqat o'zbek tilida, lotin yozuvida gapiring. Tabiiy, sodda, og'zaki va hurmatli bo'ling."
    elif current_lang == "en-US":
        lang_instr = "Speak only in English. Keep it natural, simple, spoken, and respectful."
    else:
        lang_instr = "Speak only in Russian. Keep it natural, simple, spoken, and respectful."

    base = (
        "Siz UzMAX robotisiz: yuqumli kasalliklar shifoxonasi uchun aqlli yordamchi. "
        "Vazifangiz: kamerada bemorni ko'rganda salomlashish, yangi bemordan faqat ismini so'rash, "
        "thermal kamera skriningini bemor raqamli kartasiga bog'lash, bemordan nima bezovta qilayotganini so'rash "
        "va kerakli shifokorga yo'naltirish. "
        "Loyiha: 'Yuqumli kasalliklar shifoxonasi uchun aqlli robot yaratish'. "
        "Rahbar: TATU, Azimov Bunyod Raximjonovich. "
        "Javoblar qisqa, jonli, tabiiy va do'stona bo'lsin — xuddi jonli hamshira kabi suhbatlashing. "
        "Salomlashishni faqat suhbat boshida bir marta ayting; keyingi javoblarda salomni ('Assalomu alaykum' va h.k.) takrorlamang. "
        "Bir xil tayyor jumlalarni har safar takrorlamang (masalan 'bu tashxis emas' — agar kerak bo'lsa faqat bir marta ayting). "
        "Suhbat tarixini eslab, avvalgi gaplaringizni so'zma-so'z qaytarmang, tabiiy davom eting. "
        "Odatda 1-2 qisqa gapdan oshmang. "
        "Thermal natijani dastlabki skrining deb ayting, tashxis qo'ymang. "
        "Qaysi shifokor kerakligini aniqlash uchun 'find_doctor' funksiyasidan foydalaning. "
        "Bemorni navbatga faqat u ko'rikka yozilishni xohlasa yoki rozi bo'lsa 'book_appointment' bilan yozing; "
        "shunchaki savol-javobda navbatga yozmang. book_appointment qaytargan shifokor, xona, ish vaqti va "
        "navbat raqamini o'zgartirmay ayting. Bemor oddiy savol bersa (masalan kasalliklar farqi), tibbiy ma'lumotni "
        "qisqa tushuntiring, navbatga yozmang. "
        "Isitma yoki xavotirli belgi bo'lsa, qabul shifokori yoki infeksionistga yo'naltiring. "
        f"Shifokorlar: {json.dumps(DOCTOR_DIRECTORY, ensure_ascii=False)}. "
        "Savol bersangiz, faqat bitta oddiy savol bering. "
        "Mehmonga hurmat bilan murojaat qiling. "
        f"{lang_instr}"
    )

    if onboarding:
        return (
            base
            + " Yangi, tanilmagan bemor bilan tanishyapsiz. Tabiiy va jonli suhbatlashing — qattiq qolip bo'yicha takrorlamang. "
            + "O'zingizni qisqa tanishtiring (yuqumli kasalliklar shifoxonasi uchun UzMAX roboti) va ism-familiyasini so'rang. "
            + "Bemor aytgan ismni tabiiy takrorlab tasdiqlating (masalan: 'Ismingiz Abdulla, to'g'rimi?'). "
            + "Bemor tasdiqlagach (masalan 'ha'), register_patient funksiyasini ism va (bo'lsa) familiya bilan chaqiring. "
            + "Agar bemor boshqa ism aytsa yoki tuzatsa, yangi ismni qabul qiling — avvalgi variantda qotib qolmang. "
            + "Bitta sifatli yuz kadri yetarli; agar register_patient reason='need_more_samples' qaytarsa, bemordan bir oz kameraga qarab turishini so'rang, keyin qayta chaqiring. "
            + "Ro'yxatdan o'tgani haqida faqat register_patient ok=true qaytargandan so'ng ayting, so'ng shikoyatini so'rang."
        )

    if current_person:
        full_name = f'{current_person.get("first_name", "")} {current_person.get("last_name", "")}'.strip()
        metadata  = current_person.get("metadata") or {}
        meta_ctx  = f" Qo'shimcha ma'lumot: {json.dumps(metadata, ensure_ascii=False)}." if metadata else ""
        return (
            base
            + f" Siz bu odamni taniysiz: {full_name}. Agar bu suhbatning boshi bo'lsa, ismi bilan bir marta "
            + "iliq salomlashing va nima bezovta qilayotganini so'rang; aks holda salomsiz tabiiy davom eting. "
            + "Thermal skrining holatini faqat kerak bo'lsa qisqa ayting."
            + meta_ctx
        )

    return base


def local_medical_fallback(user_text: str, current_person=None, lang: str | None = None) -> str:
    """Short offline fallback when the configured LLM provider is unavailable."""
    q = (user_text or "").lower()
    lang = normalize_chat_lang(lang, user_text)
    name = ""
    if current_person:
        first = (current_person.get("first_name") or "").strip()
        if first:
            name = f"{first}, "

    routed = route_patient_request(user_text, current_person, lang)
    if routed:
        return routed

    if lang == "en-US":
        if any(word in q for word in ("hello", "hi")):
            return f"{name}hello. I am UzMAX medical assistant. How can I help you?"
        if any(word in q for word in ("temperature", "fever")):
            return f"{name}temperature is only a screening result. If you have fever, weakness, or pain, please see a doctor."
        if any(word in q for word in ("cough", "throat", "flu")):
            return f"{name}if you have cough or sore throat, wear a mask, drink fluids, and see a doctor."
        return f"{name}the cloud AI key is not working right now, but I am in local mode. Please write your question more briefly."

    if lang == "ru-RU":
        if any(word in q for word in ("здравствуйте", "привет")):
            return f"{name}здравствуйте. Я медицинский помощник UzMAX. Чем могу помочь?"
        if any(word in q for word in ("температура", "жар")):
            return f"{name}температура является только результатом скрининга. При жаре, слабости или боли обратитесь к врачу."
        if any(word in q for word in ("кашель", "горло", "грипп")):
            return f"{name}при кашле или боли в горле наденьте маску, пейте жидкость и пройдите осмотр врача."
        return f"{name}сейчас cloud AI ключ не работает, но я работаю в локальном режиме. Напишите вопрос короче."

    if any(word in q for word in ("salom", "assalomu", "hello", "hi")):
        return f"{name}assalomu alaykum. Men UzMAX tibbiy yordamchiman. Sizga qanday yordam kerak?"
    if any(word in q for word in ("harorat", "temperatura", "isitma", "fever")):
        return f"{name}harorat skrining natijasidir. Agar isitma, holsizlik yoki og'riq bo'lsa, shifokorga murojaat qiling."
    if any(word in q for word in ("yo'tal", "yotal", "cough", "tomoq", "gripp")):
        return f"{name}yo'tal yoki tomoq og'rig'i bo'lsa, niqob taqing, suyuqlik iching va shifokor ko'rigidan o'ting."
    if any(word in q for word in ("doktor", "shifokor", "navbat", "qabul", "infeksionist")):
        return f"{name}avval qabul shifokori ko'radi. Isitma yoki infeksiya belgisi bo'lsa, infeksionistga yo'naltirilasiz."
    return f"{name}hozir cloud AI kaliti ishlamayapti, lekin men lokal rejimdaman. Savolingizni qisqaroq yozing."


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WS /ws/chat connected")

    recognizer  = YandexSpeechRecognizer(folder_id=FOLDER_ID, iam_token=API_KEY)
    synthesizer = YandexStreamingSynthesizer(folder_id=FOLDER_ID, iam_token=API_KEY)
    llm         = OpenAIClient()

    loop = asyncio.get_running_loop()

    current_lang = "uz-UZ"
    current_voice = YANDEX_TTS_VOICE_UZ

    stt_session      = None
    partial_stt_task = None

    live_mode     = False
    allow_interrupt = False

    is_responding     = False
    response_cancelled = asyncio.Event()
    active_tts_session  = None
    active_response_task = None
    response_generation  = 0
    current_person       = None
    pending_registration = None
    last_auto_greet_key  = None
    last_auto_greet_at   = 0.0
    last_screening_at    = {}

    def start_stt():
        nonlocal stt_session, partial_stt_task
        if stt_session:
            try:
                stt_session._chunks.put(None)
            except Exception:
                pass
            stt_session = None
        if partial_stt_task:
            partial_stt_task.cancel()
            partial_stt_task = None

        stt_partial_queue = asyncio.Queue()
        provider = os.getenv("STT_PROVIDER", "whisper").lower()
        if provider in ("yandex_stream", "stream", "streaming", "partial"):
            stt_session = SttStreamingSession(recognizer, 16000, loop, stt_partial_queue, current_lang)
        # Non-streaming modes buffer 16 kHz mono PCM and recognise once on end_speech.
        # They are useful fallbacks, but they do not emit live partial text.
        elif provider == "yandex":
            stt_session = YandexSttSession(recognizer, 16000, loop, stt_partial_queue, current_lang)
        else:
            stt_session = WhisperSttSession(llm, 16000, loop, stt_partial_queue, current_lang)
        stt_session.start()

        async def send_partial_stt():
            try:
                while True:
                    text = await stt_partial_queue.get()
                    if text is None:
                        break
                    await websocket.send_json({"type": "stt_partial", "text": text})
            except (asyncio.CancelledError, Exception):
                pass

        partial_stt_task = asyncio.create_task(send_partial_stt())

    def stop_stt():
        nonlocal stt_session, partial_stt_task
        if stt_session:
            try:
                stt_session._chunks.put(None)
            except Exception:
                pass
            stt_session = None
        if partial_stt_task:
            partial_stt_task.cancel()
            partial_stt_task = None

    def stop_partial_sender():
        nonlocal partial_stt_task
        if partial_stt_task:
            partial_stt_task.cancel()
            partial_stt_task = None

    def force_cancel_response():
        nonlocal is_responding, active_tts_session, active_response_task, response_generation
        response_generation += 1
        response_cancelled.set()
        if active_tts_session is not None:
            active_tts_session.cancel()
            active_tts_session = None
        if active_response_task and not active_response_task.done():
            active_response_task.cancel()
            active_response_task = None
        is_responding = False

    messages = [{"role": "system",
                 "content": build_system_prompt(current_person, onboarding=False, current_lang=current_lang)}]

    async def register_pending_person(name_text: str) -> tuple[dict | None, str | None]:
        nonlocal current_person, pending_registration
        if not pending_registration:
            return None, None
        samples = list(pending_registration.get("samples") or [])
        if not samples and pending_registration.get("embedding"):
            samples = [{
                "embedding": pending_registration.get("embedding"),
                "snapshot_path": pending_registration.get("snapshot_path"),
                "quality": pending_registration.get("quality") or {},
            }]

        candidate = pending_registration.get("candidate_name")
        if candidate:
            full_name = f'{candidate.get("first_name", "")} {candidate.get("last_name", "")}'.strip()
            if is_negative(name_text):
                pending_registration.pop("candidate_name", None)
                return None, "Tushunarli. Iltimos, ism va familiyangizni qayta ayting."
            if not is_affirmative(name_text):
                return None, f"Sizning ism-familiyangiz {full_name}mi? To'g'ri bo'lsa ha, noto'g'ri bo'lsa yo'q deng."
            if len(samples) < FACE_MIN_SAMPLES:
                return None, f"Yuz namunasi hali kam. Iltimos kameraga qarang, kamida {FACE_MIN_SAMPLES} ta yaxshi kadr kerak."
            embedding = averaged_embedding(samples)
            if not embedding:
                return None, "Yuz embedding tayyor emas. Iltimos, kameraga qarab qayta urinib ko'ring."
            best_sample = max(
                samples,
                key=lambda item: (
                    int((item.get("quality") or {}).get("area") or 0),
                    float((item.get("quality") or {}).get("blur") or 0),
                ),
            )
            pending_registration["embedding"] = embedding
            pending_registration["snapshot_path"] = best_sample.get("snapshot_path")
            extracted = candidate
        else:
            try:
                extracted = await llm.extract_person_name(name_text)
            except Exception as exc:
                logger.warning("Name extraction via LLM failed: %s", exc)
                extracted = None
            if not extracted:
                extracted = parse_person_name_locally(name_text)
            if not extracted or not extracted.get("first_name"):
                return None, "Ismni tushunmadim. Iltimos, ism va familiyangizni ayting."
            # last_name is optional: STT for Uzbek is imperfect, so don't loop forever
            # demanding a surname. The confirmation step below lets the patient correct it.
            pending_registration["candidate_name"] = {
                "first_name": extracted.get("first_name", ""),
                "last_name": extracted.get("last_name", ""),
            }
            full_name = f'{extracted.get("first_name", "")} {extracted.get("last_name", "")}'.strip()
            return None, f"Sizning ism-familiyangiz {full_name}mi? To'g'ri bo'lsa ha, noto'g'ri bo'lsa yo'q deng."

        if not pending_registration.get("embedding") or not pending_registration.get("snapshot_path"):
            logger.warning("Pending registration is missing embedding or snapshot_path: %s", pending_registration.keys())
            return None, "Yuz rasmi tayyor emas. Iltimos, kameraga qarab qayta urinib ko'ring."

        metadata = merge_patient_screening(
            {},
            compact_thermal_screening(pending_registration.get("thermal")),
        )
        current_person = get_face_store().register(
            embedding=pending_registration["embedding"],
            first_name=extracted.get("first_name", ""),
            last_name=extracted.get("last_name", ""),
            snapshot_path=pending_registration.get("snapshot_path"),
            metadata=metadata,
        )
        permanent_snapshot = persist_registered_face_snapshot(
            pending_registration.get("snapshot_path"),
            current_person["person_id"],
        )
        if permanent_snapshot:
            get_face_store().add_snapshot(current_person["person_id"], permanent_snapshot)
            current_person["snapshots"] = [permanent_snapshot]
            upsert_registered_face_registry(current_person, permanent_snapshot)
            permanent_path = Path(permanent_snapshot)
            for sample in samples:
                pending_path = Path(sample.get("snapshot_path") or "")
                if pending_path.exists() and pending_path.name.startswith("pending_") and pending_path != permanent_path:
                    try:
                        pending_path.unlink()
                    except OSError:
                        logger.warning("Could not delete pending face snapshot: %s", pending_path)
        else:
            logger.warning(
                "Face registered in vector DB but registry.json was not updated because no permanent snapshot was available: person_id=%s",
                current_person["person_id"],
            )
            return current_person, "Yuz bazaga qo'shildi, lekin rasm registry.json ga yozilmadi."

        pending_registration = None
        return current_person, None

    async def update_current_person_name(text: str) -> tuple[dict | None, str | None]:
        nonlocal current_person
        if not current_person or not current_person.get("person_id"):
            return None, None
        extracted = extract_name_change(text)
        if not extracted or not extracted.get("first_name"):
            return None, None

        updated = get_face_store().update_name(
            current_person["person_id"],
            extracted.get("first_name", ""),
            extracted.get("last_name", ""),
        )
        if not updated:
            return None, "Ismni yangilay olmadim. Iltimos, yuzingiz tanilganidan keyin qayta urinib ko'ring."
        current_person = updated
        update_registered_face_registry_name(updated)
        return updated, None

    async def process_response(user_text: str, generation: int):
        nonlocal is_responding, active_tts_session, active_response_task
        is_responding = True
        response_cancelled.clear()
        messages[0] = {
            "role": "system",
            "content": build_system_prompt(current_person, onboarding=bool(pending_registration),
                                           current_lang=current_lang),
        }

        tts_audio_queue = asyncio.Queue()
        tts_session = TtsStreamingSession(
            synthesizer,
            YANDEX_TTS_SPEED,
            YANDEX_TTS_SAMPLE_RATE,
            loop,
            tts_audio_queue,
            current_voice,
        )
        active_tts_session = tts_session
        tts_session.start()

        def feed_tts(text: str):
            tts_session.feed(tts_readable_text(text, current_lang))

        await websocket.send_json({"type": "response_started", "response_id": generation})

        async def send_tts_audio():
            try:
                while not response_cancelled.is_set() and generation == response_generation:
                    try:
                        audio_chunk = await asyncio.wait_for(tts_audio_queue.get(), timeout=0.02)
                    except asyncio.TimeoutError:
                        continue
                    if audio_chunk is None:
                        break
                    if response_cancelled.is_set() or generation != response_generation:
                        break
                    await websocket.send_json({
                        "type": "tts_chunk",
                        "response_id": generation,
                        "encoding": "linear16",
                        "sample_rate": YANDEX_TTS_SAMPLE_RATE,
                        "audio": base64.b64encode(audio_chunk).decode("ascii"),
                    })
                while not tts_audio_queue.empty():
                    try:
                        tts_audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if generation == response_generation and not response_cancelled.is_set():
                    await websocket.send_json({"type": "tts_end", "response_id": generation})
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("TTS audio send error: %s", e)

        tts_task        = asyncio.create_task(send_tts_audio())
        full_llm_resp   = ""
        sentence_buf    = ""

        async def chat_tool_executor(name: str, args: dict) -> dict:
            nonlocal current_person, pending_registration
            if name == "register_patient":
                if not pending_registration:
                    return {"ok": False, "reason": "no_face",
                            "message": "Yuz hali aniqlanmadi. Bemordan kameraga qarashini so'rang."}
                first_name = str(args.get("first_name", "")).strip()
                last_name = str(args.get("last_name", "")).strip()
                if not first_name:
                    return {"ok": False, "reason": "no_name", "message": "Ism kerak."}
                samples = list(pending_registration.get("samples") or [])
                if not samples and pending_registration.get("embedding"):
                    samples = [{
                        "embedding": pending_registration.get("embedding"),
                        "snapshot_path": pending_registration.get("snapshot_path"),
                        "quality": pending_registration.get("quality") or {},
                    }]
                if len(samples) < FACE_MIN_SAMPLES:
                    return {"ok": False, "reason": "need_more_samples",
                            "have": len(samples), "needed": FACE_MIN_SAMPLES,
                            "message": f"Yuz namunasi kam ({len(samples)}/{FACE_MIN_SAMPLES}). "
                                       "Bemor bir oz kameraga qarab tursin."}
                embedding = averaged_embedding(samples)
                if not embedding:
                    return {"ok": False, "reason": "no_embedding", "message": "Yuz embedding tayyor emas."}
                best_sample = max(samples, key=lambda item: (
                    int((item.get("quality") or {}).get("area") or 0),
                    float((item.get("quality") or {}).get("blur") or 0),
                ))
                snapshot_path = best_sample.get("snapshot_path")
                metadata = merge_patient_screening(
                    {}, compact_thermal_screening(pending_registration.get("thermal")))
                person = get_face_store().register(
                    embedding=embedding,
                    first_name=first_name,
                    last_name=last_name,
                    snapshot_path=snapshot_path,
                    metadata=metadata,
                )
                permanent_snapshot = persist_registered_face_snapshot(snapshot_path, person["person_id"])
                if permanent_snapshot:
                    get_face_store().add_snapshot(person["person_id"], permanent_snapshot)
                    person["snapshots"] = [permanent_snapshot]
                    upsert_registered_face_registry(person, permanent_snapshot)
                    permanent_path = Path(permanent_snapshot)
                    for sample in samples:
                        pending_path = Path(sample.get("snapshot_path") or "")
                        if pending_path.exists() and pending_path.name.startswith("pending_") and pending_path != permanent_path:
                            try:
                                pending_path.unlink()
                            except OSError:
                                logger.warning("Could not delete pending face snapshot: %s", pending_path)
                current_person = person
                pending_registration = None
                return {
                    "ok": True,
                    "full_name": person.get("full_name") or f"{first_name} {last_name}".strip(),
                    "person_id": person.get("person_id"),
                    "thermal": latest_thermal_text(person),
                }
            if name == "find_doctor":
                symptoms = str(args.get("symptoms", "")).strip()
                if is_emergency_case(symptoms):
                    return {
                        "emergency": True,
                        "advice": "Bu shoshilinch holat bo'lishi mumkin. 103 ga qo'ng'iroq qiling "
                                  "yoki navbatchi shifokorga darhol murojaat qiling.",
                    }
                doctor, score, matches = doctor_router_retrieve(symptoms)
                if not doctor:
                    return {"found": False, "hint": "Mos shifokor topilmadi, qabul shifokorini tavsiya qiling."}
                return {
                    "found": True,
                    "doctor": doctor_public_info(doctor, current_lang),
                    "basic_advice": basic_patient_advice(symptoms, current_lang),
                }
            if name == "book_appointment":
                try:
                    doctor_id = int(args.get("doctor_id"))
                except (TypeError, ValueError):
                    return {"ok": False, "error": "doctor_id (butun son) talab qilinadi"}
                doctor = next((d for d in DOCTOR_DIRECTORY if d.get("id") == doctor_id), None)
                if not doctor:
                    return {"ok": False, "error": "Bunday doctor_id yo'q"}
                item = add_patient_to_doctor_queue(current_person, doctor)
                return {
                    "ok": True,
                    "doctor": doctor_public_info(doctor, current_lang),
                    "queue_number": item.get("queue_number"),
                    "room": doctor.get("room"),
                    "work_time": doctor.get("work_time"),
                    "date": item.get("date"),
                }
            return {"error": f"unknown tool {name}"}

        try:
            verbatim = verbatim_internal_reply(user_text)
            if verbatim:
                full_llm_resp = verbatim
                await websocket.send_json({
                    "type": "llm_partial",
                    "text": verbatim,
                    "response_id": generation,
                })
                if not response_cancelled.is_set() and generation == response_generation:
                    feed_tts(verbatim)
            else:
                async for llm_chunk in llm.get_response_stream(
                    messages, tools=CHAT_TOOLS, tool_executor=chat_tool_executor
                ):
                    if response_cancelled.is_set() or generation != response_generation:
                        break
                    full_llm_resp += llm_chunk
                    await websocket.send_json({"type": "llm_partial", "text": llm_chunk,
                                               "response_id": generation})
                    sentence_buf += llm_chunk
                    sentence, sentence_buf = flush_sentence_buffer(sentence_buf)
                    if sentence:
                        feed_tts(sentence)

                if sentence_buf.strip() and not response_cancelled.is_set() and generation == response_generation:
                    feed_tts(sentence_buf.strip())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            error_text = local_medical_fallback(user_text, current_person, current_lang)
            if full_llm_resp:
                error_text = "\n\n" + error_text
            full_llm_resp += error_text
            await websocket.send_json({
                "type": "llm_partial",
                "text": error_text,
                "response_id": generation,
            })
            if not response_cancelled.is_set() and generation == response_generation:
                feed_tts(error_text.strip())

        was_cancelled = response_cancelled.is_set() or generation != response_generation
        if full_llm_resp and not was_cancelled:
            messages.append({"role": "assistant", "content": full_llm_resp})
            await websocket.send_json({
                "type": "llm_done",
                "text": full_llm_resp,
                "response_id": generation,
            })

        try:
            await asyncio.wait_for(tts_session.finish(), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("TTS finish timed out; continuing with text response")
            tts_session.cancel()
            if generation == response_generation and not response_cancelled.is_set():
                await websocket.send_json({"type": "tts_end", "response_id": generation})
        except (asyncio.CancelledError, Exception):
            pass

        try:
            await asyncio.wait_for(tts_task, timeout=3)
        except asyncio.TimeoutError:
            tts_task.cancel()
        except asyncio.CancelledError:
            pass

        if active_tts_session is tts_session:
            active_tts_session = None
        if active_response_task is asyncio.current_task():
            active_response_task = None
        if generation == response_generation:
            is_responding = False

        if live_mode and not was_cancelled:
            await websocket.send_json({"type": "ready_to_listen"})

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                if is_responding and allow_interrupt:
                    force_cancel_response()
                    await websocket.send_json({"type": "interrupt"})
                    start_stt()
                    await websocket.send_json({"type": "stt_ready"})
                if stt_session:
                    stt_session.feed(data["bytes"])

            elif "text" in data:
                msg      = json.loads(data["text"])
                msg_type = msg.get("type")

                if msg_type == "set_language":
                    current_lang = msg.get("lang", "uz-UZ")
                    current_voice = {
                        "uz-UZ": YANDEX_TTS_VOICE_UZ,
                        "en-US": YANDEX_TTS_VOICE_EN,
                        "ru-RU": YANDEX_TTS_VOICE_RU,
                    }.get(current_lang, YANDEX_TTS_VOICE_UZ)

                elif msg_type == "set_settings":
                    live_mode      = msg.get("live_mode", False)
                    allow_interrupt = msg.get("allow_interrupt", False)

                elif msg_type == "start_speech":
                    if is_responding and allow_interrupt:
                        force_cancel_response()
                        await websocket.send_json({"type": "interrupt"})
                    start_stt()
                    await websocket.send_json({"type": "stt_ready"})

                elif msg_type == "interrupt":
                    if is_responding:
                        force_cancel_response()
                        await websocket.send_json({"type": "interrupt"})

                elif msg_type == "person_left":
                    # The patient stepped out of frame. Keep the greeting trackers and
                    # the conversation history (messages) so that if the SAME person
                    # returns shortly we continue the conversation instead of greeting
                    # again. A different person, or the same one after REGREET_GRACE_S,
                    # still gets a fresh greeting (see should_greet below).
                    current_person = None
                    pending_registration = None

                elif msg_type == "face_identity":
                    incoming_person = msg.get("person")
                    incoming_pending = msg.get("pending_registration")
                    thermal_context = msg.get("thermal")

                    if incoming_person:
                        current_person = incoming_person
                        pending_registration = None
                        person_id = current_person.get("person_id")
                        last_at = last_screening_at.get(person_id, 0)
                        if person_id and time.time() - last_at > 45:
                            try:
                                current_person = attach_screening_to_person(current_person, thermal_context)
                                last_screening_at[person_id] = time.time()
                            except Exception as exc:
                                logger.warning("Patient screening metadata update failed: %s", exc)
                    elif incoming_pending:
                        current_person = None
                        if not pending_registration:
                            pending_registration = incoming_pending
                        incoming_samples = list(incoming_pending.get("samples") or [])
                        if not incoming_samples and incoming_pending.get("embedding"):
                            incoming_samples = [{
                                "embedding": incoming_pending.get("embedding"),
                                "snapshot_path": incoming_pending.get("snapshot_path"),
                                "quality": incoming_pending.get("quality") or {},
                            }]
                        samples = list(pending_registration.get("samples") or [])
                        seen_snapshots = {sample.get("snapshot_path") for sample in samples}
                        for sample in incoming_samples:
                            if sample.get("snapshot_path") not in seen_snapshots:
                                samples.append(sample)
                                seen_snapshots.add(sample.get("snapshot_path"))
                        pending_registration["samples"] = samples[-5:]
                        if samples:
                            best_sample = max(
                                samples,
                                key=lambda item: (
                                    int((item.get("quality") or {}).get("area") or 0),
                                    float((item.get("quality") or {}).get("blur") or 0),
                                ),
                            )
                            pending_registration["embedding"] = averaged_embedding(samples)
                            pending_registration["snapshot_path"] = best_sample.get("snapshot_path")
                        pending_registration["thermal"] = thermal_context or pending_registration.get("thermal")
                    else:
                        current_person = None

                    if not is_responding:
                        if current_person:
                            full_name = current_person.get("full_name") or current_person.get("first_name", "")
                            greeting = (
                                f"Kameraga tanish bemor {full_name.strip()} qaytib keldi. "
                                f"{latest_thermal_text(current_person)} "
                                "Uni ismi bilan iliq kutib oling (masalan: 'Sizni qayta ko'rganimdan xursandman'). "
                                "Sessiya tarixini eslang: agar avvalgi shikoyati yoki navbati bo'lsa, qisqa eslatib o'ting "
                                "(masalan o'sha shikoyat davom etyaptimi deb so'rang); aks holda hozir nima bezovta "
                                "qilayotganini so'rang. Qisqa va tabiiy bo'ling."
                            )
                            greet_key = current_person.get("person_id") or full_name
                        elif pending_registration:
                            greeting = (
                                "Oldingizda yangi bemor turibdi. Salom bering va aynan shu mazmunda so'rang: "
                                '"Men yuqumli kasalliklar shifoxonasi uchun UzMAX robotman. '
                                "Iltimos, ism va familiyangizni ayting.\""
                            )
                            greet_key = "unknown_patient"
                        else:
                            greeting = None
                            greet_key = None

                        now = time.time()
                        # Re-greet only a different person, or the same one after a real
                        # absence. A brief disappear+return of the same person within the
                        # grace window continues the conversation without greeting again.
                        REGREET_GRACE_S = 120
                        should_greet = bool(greeting) and (
                            greet_key != last_auto_greet_key or now - last_auto_greet_at > REGREET_GRACE_S
                        )

                        if not should_greet and greeting and greet_key == last_auto_greet_key:
                            # Same person returned within the grace window: don't greet,
                            # just refresh the timer so repeated brief flickers keep
                            # continuing the conversation rather than re-greeting.
                            last_auto_greet_at = now

                        if should_greet:
                            last_auto_greet_key = greet_key
                            last_auto_greet_at = now
                            messages.append({"role": "user", "content": greeting})
                            response_generation += 1
                            active_response_task = asyncio.create_task(
                                process_response(greeting, response_generation)
                            )

                elif msg_type == "text_message":
                    final_text = (msg.get("text") or "").strip()
                    if not final_text:
                        continue
                    if is_responding:
                        force_cancel_response()
                    last_auto_greet_at = time.time()   # keep session fresh: grace runs from last interaction
                    current_person = append_patient_note(current_person, final_text)
                    messages.append({"role": "user", "content": final_text})
                    response_generation += 1
                    active_response_task = asyncio.create_task(
                        process_response(final_text, response_generation)
                    )

                elif msg_type == "end_speech":
                    if not stt_session:
                        continue
                    _t_finish = time.perf_counter()
                    final_text  = await stt_session.finish()
                    logger.info("[STT] end_speech -> final in %.2fs text=%r",
                                time.perf_counter() - _t_finish, (final_text or "")[:80])
                    stt_session = None
                    stop_partial_sender()

                    if not final_text:
                        stt_error = getattr(recognizer, "last_error", None)
                        if stt_error:
                            await websocket.send_json({
                                "type": "chat_error",
                                "message": (
                                    "Yandex SpeechKit STT ishlamadi. Settings bo'limida "
                                    "Yandex API key va catalog/folder ID bir xil Yandex Cloud "
                                    f"folderdan ekanini tekshiring. Xato: {str(stt_error)[:220]}"
                                ),
                            })
                        await websocket.send_json({"type": "stt_empty"})
                        if live_mode:
                            await websocket.send_json({"type": "ready_to_listen"})
                        continue

                    if is_responding:
                        force_cancel_response()

                    await websocket.send_json({"type": "stt_final", "text": final_text})
                    last_auto_greet_at = time.time()   # keep session fresh: grace runs from last interaction
                    current_person = append_patient_note(current_person, final_text)
                    messages.append({"role": "user", "content": final_text})
                    response_generation += 1
                    active_response_task = asyncio.create_task(
                        process_response(final_text, response_generation)
                    )

    except WebSocketDisconnect:
        logger.info("WS /ws/chat disconnected")
    except RuntimeError as e:
        if "Cannot call" not in str(e):
            logger.error("RuntimeError in WS: %s", e)
    except Exception as e:
        logger.error("WS error: %s", e)
    finally:
        stop_stt()
        if is_responding or active_response_task:
            force_cancel_response()


# ═══════════════════════════════════════════════════════════════════
#  THERMAL CAMERA
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/thermal")
async def thermal_snapshot():
    """Single frame from MLX90640. Always returns JSON even when disabled."""
    data = await asyncio.to_thread(_read_thermal_source)
    return JSONResponse(data)


@app.get("/api/thermal/state")
async def thermal_state():
    return JSONResponse({
        "ok": True,
        "enabled": bool(_thermal_runtime.get("enabled")),
        "hardware_ready": bool(_thermal.enabled),
        "reason": None if _thermal.enabled else _thermal.reason,
        "message": _thermal.message,
        "install": None if _thermal.enabled else THERMAL_INSTALL_COMMAND,
    })


@app.post("/api/thermal/state")
async def thermal_set_state(payload: dict):
    enabled = bool(payload.get("enabled"))
    state = await asyncio.to_thread(_set_thermal_enabled, enabled)
    return JSONResponse(state)


@app.post("/api/thermal/push")
async def thermal_push(payload: dict):
    """Accept one 32x24 MLX90640 frame from a Raspberry Pi thermal agent."""
    try:
        source = str(payload.get("source") or "raspberry_pi")
        data = _normalize_thermal_payload(payload, source=source)
    except Exception as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)

    _remote_thermal["data"] = data
    _remote_thermal["updated_at"] = time.time()
    _remote_thermal["source"] = source
    return JSONResponse({"ok": True, "source": source, "max": data["max"]})


@app.websocket("/ws/thermal")
async def thermal_ws(websocket: WebSocket):
    """
    Streams one 32×24 frame every 250 ms (≈4 Hz).
    Payload JSON: {ok, frame:[768 floats], max, min, avg, center, rows, cols}
    When sensor is not connected → {ok:false, reason:'not_connected'}
    """
    await websocket.accept()
    logger.info("WS /ws/thermal connected")
    try:
        while True:
            data = await asyncio.to_thread(_read_thermal_source)
            await websocket.send_json(data)
            await asyncio.sleep(0.25)   # 4 Hz
    except WebSocketDisconnect:
        logger.info("WS /ws/thermal disconnected")
    except Exception as e:
        logger.error("Thermal WS error: %s", e)


# ═══════════════════════════════════════════════════════════════════
#  FRONTEND
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("UZMAX_HOST", "0.0.0.0")
    port = int(os.getenv("UZMAX_PORT", "5000"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_check:
        try:
            port_check.bind((host, port))
        except OSError:
            print(f"ERROR: Port {port} is already in use.")
            print("Stop the existing UzMAX server or set UZMAX_PORT to another port.")
            raise SystemExit(1)

    print("=" * 56)
    print(f"  UzMAX Unified Server  ->  http://127.0.0.1:{port}")
    print(f"  LAN/server link       ->  http://YOUR_SERVER_IP:{port}")
    print("  Medical AI  +  Robot Control (HAND/HEAD/MOVE)")
    print("=" * 56)
    uvicorn.run(app, host=host, port=port, log_level="info")
