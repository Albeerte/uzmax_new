# UzMAX — Hospital Assistant Robot

UzMAX is a hospital reception robot for an infectious-diseases clinic. A single
FastAPI server combines:

- **Voice assistant** — Yandex SpeechKit speech-to-text / text-to-speech with an
  LLM (DeepSeek, OpenAI or Gemini) speaking Uzbek, Russian and English.
- **Face ID** — InsightFace detection + ArcFace embeddings stored in a local
  Qdrant vector database, so returning patients are recognized automatically.
- **Patient check-in** — registration by voice, thermal screening (MLX90640),
  routing to the right doctor and a doctor queue.
- **Robot control** — ESP32 boards for the hands, head and wheels over USB serial,
  with firmware compile/upload from the web UI.

## Requirements

- Windows 10/11 (Linux works for the server; serial port names differ)
- Python 3.11
- A webcam, and optionally an MLX90640 thermal camera and the ESP32 boards
- API keys: Yandex Cloud (SpeechKit) and one LLM provider

## Setup

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `uzmax_server/.env` (it is git-ignored — never commit keys):

```ini
YANDEX_API_KEY=...
YANDEX_CATALOG_ID=...
DEEPSEEK_API_KEY=...        # or OPENAI_API_KEY / GEMINI_API_KEY
FACE_EMBED_MODEL=buffalo_s
FACE_MATCH_THRESHOLD=0.45
FACE_MATCH_MIN_CONFIDENCE=0.45
```

## Run

From the repository root:

```bash
.\.venv\Scripts\python.exe uzmax_server/main.py
```

Open <http://127.0.0.1:5000/>. The server listens on `0.0.0.0:5000`, so other
devices on the same network can use `http://SERVER_IP:5000/`.

On first start InsightFace downloads the `buffalo_s` model (~125 MB) into
`uzmax_server/data/insightface/`.

### Raspberry Pi desktop icon

Use Raspberry Pi OS 64-bit. Install once:

```bash
bash scripts/rpi/install-desktop.sh
```

This adds a **RoboMed** icon to the desktop and the app menu. Double-clicking it
starts the server (if it is not running), then opens the UI full-screen in
Chromium with camera and microphone access allowed. Server output goes to
`~/robomed-server.log`. Press `F11` to leave full-screen.

## Configuration

All keys can be set in `uzmax_server/.env` or from the Settings page in the UI.

| Key | Default | Purpose |
|---|---|---|
| `YANDEX_API_KEY`, `YANDEX_CATALOG_ID` | — | Yandex SpeechKit STT/TTS |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | — | LLM provider (first one set wins, in that order) |
| `DEEPSEEK_MODEL`, `OPENAI_MODEL` | `deepseek-flash`, `gpt-4o-mini` | LLM model |
| `STT_PROVIDER` | `yandex` | `yandex`, `yandex_stream` (live partials) or Whisper |
| `YANDEX_TTS_VOICE_UZ` / `_RU` / `_EN` | `yulduz` / `yulduz_ru` / `john` | TTS voices |
| `YANDEX_TTS_SPEED` | `1.1` | TTS speed |
| `FACE_EMBED_MODEL` | `buffalo_s` | InsightFace model pack |
| `FACE_EMBED_DET_SIZE` | `640,640` | Detector input size |
| `FACE_DET_THRESH` | `0.5` | Minimum face detection confidence |
| `FACE_MATCH_THRESHOLD` | `0.45` | Minimum ArcFace cosine similarity for a match |
| `FACE_MATCH_MARGIN` | `0.04` | Required lead over the next *different* person |
| `SIMPLE_FACE_MODE` | `true` | Register from one good frame instead of `FACE_MIN_SAMPLES` |
| `FACE_MIN_SAMPLES` | `3` | Frames collected before registering (when simple mode is off) |
| `FACE_MIN_WIDTH_PX` | `90` | Minimum face width in pixels |
| `FACE_MIN_BLUR_VAR` | `0` (off) | Minimum sharpness; raise it to reject blurry frames |
| `ARDUINO_CLI_PATH`, `ESP32_FQBN` | — | Firmware compile/upload |

## How face recognition works

1. The browser sends camera frames to `/api/faces/detect` (fast box drawing) and
   `/api/faces/identify-local`.
2. Faces are detected with **InsightFace SCRFD**. OpenCV Haar cascades are used
   only if InsightFace is unavailable.
3. The largest face passes a size check, is cropped and turned
   into a 512-dim **ArcFace** embedding.
4. The embedding is compared with Qdrant and `registry.json`. A match needs a
   score ≥ `FACE_MATCH_THRESHOLD` and a margin over the next person with a
   different name.
5. Unknown faces are registered after the patient says and confirms their name.
   If the face (or the same name with a similar face) already exists, the new
   sample is added to the existing patient instead of creating a duplicate.

Each stored embedding is tagged with the model that produced it
(`embed_model`). If the model changes, embeddings are re-computed from the saved
photos at startup.

Face data lives in `uzmax_server/data/` (git-ignored):

- `faces/qdrant/` — vector database
- `register_faces/registry.json` + `<person_id>.jpg` — registered patients
- `doctor_queue.json`, `hospital_robot.db` — queue and visits

The red **Reset** button in the Face ID panel (`POST /api/faces/reset`) wipes
all face data.

## Main API

| Endpoint | Description |
|---|---|
| `GET /` | Web UI |
| `GET /health` | Health check |
| `WS /ws/chat` | Voice/chat session (STT, LLM, TTS, registration) |
| `POST /api/faces/detect` | Fast face boxes |
| `POST /api/faces/identify-local` | Detect + identify faces |
| `POST /api/faces/reset` | Delete all face data |
| `GET /api/patients/full` | Patients with photos, queue and visits |
| `POST /api/patients`, `PUT/DELETE /api/patients/{id}` | Manage patients |
| `GET /api/doctor/directory`, `GET /api/doctor/queue` | Doctors and queue |
| `POST /api/doctor/route` | Pick a doctor from symptoms |
| `GET /api/thermal`, `WS /ws/thermal` | Thermal camera |
| `GET /api/ports`, `POST /api/connect`, `POST /api/auto-connect` | Serial connection to the ESP32 boards |
| `POST /api/hand/move`, `/api/head/command`, `/api/move/command` | Robot control |
| `POST /api/firmware/compile`, `/api/firmware/upload` | ESP32 firmware |

## ESP32 serial protocol

```text
Hand:  R 1 90 | L 6 120                         (side, servo, angle)
Head:  HEAD LEFT 15 | HEAD RIGHT 15 | HEAD CENTER | HEAD SERVO 90
       HEAD LED 255 0 0 | HEAD LED_OFF | HEAD RAINBOW
Move:  MOVE FWD 150 | MOVE BACK 150 | MOVE LEFT 120 | MOVE RIGHT 120 | MOVE STOP
```

## Tests

```bash
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

```bash
node --test tests/face_session.test.cjs
```

## Project layout

```text
uzmax_server/
  main.py              FastAPI server (voice, Face ID, patients, robot, thermal)
  hospital_robot.py    Patient intake and visits (SQLite)
  thermal.py           MLX90640 thermal camera
  agent/
    face_encoder.py    InsightFace detection + embeddings
    face_store.py      Qdrant face vector store
    llm.py             LLM client (DeepSeek / OpenAI / Gemini / YandexGPT)
    speech_to_text.py  Yandex / Whisper STT
    text_to_speech.py  Yandex TTS
  static/index.html    Web UI
hand/ head/ movements/ ESP32 firmware sketches
firmware_versions/     Firmware version metadata
docs/                  Design notes
tests/                 Python and Node tests
```
