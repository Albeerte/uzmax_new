import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:
    cv2 = None

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel


app = FastAPI(title="UZMAX Hospital Robot Chatbot")
router = APIRouter()

DB_PATH = Path(__file__).resolve().parent / "data" / "hospital_robot.db"

# Simple in-memory conversation state for standalone testing.
sessions: dict[str, dict[str, Any]] = {}


def db_connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            complaint TEXT NOT NULL,
            doctor TEXT NOT NULL,
            room TEXT NOT NULL,
            work_time TEXT NOT NULL,
            queue_no INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
        """
    )

    con.commit()
    con.close()


def register_patient(full_name: str) -> int:
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO patients (full_name, created_at) VALUES (?, ?)",
        (full_name, datetime.now().isoformat(timespec="seconds")),
    )
    con.commit()
    patient_id = cur.lastrowid
    con.close()
    return patient_id


def create_visit(patient_id: int, complaint: str, route: dict[str, str]) -> int:
    today = datetime.now().date().isoformat()

    con = db_connect()
    cur = con.cursor()

    cur.execute(
        """
        SELECT COUNT(*) FROM visits
        WHERE doctor = ? AND substr(created_at, 1, 10) = ?
        """,
        (route["doctor"], today),
    )
    queue_no = cur.fetchone()[0] + 1

    cur.execute(
        """
        INSERT INTO visits
        (patient_id, complaint, doctor, room, work_time, queue_no, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            complaint,
            route["doctor"],
            route["room"],
            route["work_time"],
            queue_no,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    con.commit()
    con.close()
    return queue_no


def detect_human_face(camera_index: int = 0, seconds: int = 3) -> bool:
    """Return True when the selected camera sees at least one frontal face."""
    if cv2 is None:
        return False

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        return False

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False

    start_time = datetime.now()
    try:
        while (datetime.now() - start_time).seconds < seconds:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(60, 60),
            )
            if len(faces) > 0:
                return True
    finally:
        cap.release()

    return False


DOCTOR_RULES = [
    {
        "doctor": "Shoshilinch yordam shifokori",
        "room": "103-xona",
        "work_time": "24/7",
        "keywords": [
            "hushsiz",
            "yiqildim",
            "qon ket",
            "qon ketyapti",
            "nafas ololmayapman",
            "ko'krak og'riq",
            "kokrak ogriq",
            "infarkt",
            "insult",
            "og'ir jarohat",
            "avariya",
            "tez yordam",
            "o'zimni juda yomon",
        ],
    },
    {
        "doctor": "Terapevt",
        "room": "101-xona",
        "work_time": "09:00 - 17:00",
        "keywords": [
            "isitma",
            "harorat",
            "tana qizidi",
            "holsizlik",
            "bosh og'riq",
            "gripp",
            "shamollash",
            "yo'tal",
            "yotal",
            "mazam yo'q",
            "mazam yoq",
        ],
    },
    {
        "doctor": "Pulmonolog",
        "room": "207-xona",
        "work_time": "10:00 - 16:00",
        "keywords": [
            "nafas",
            "o'pka",
            "opka",
            "astma",
            "bronxit",
            "quruq yo'tal",
            "balg'am",
            "balgam",
            "hansirash",
            "nafas qisishi",
        ],
    },
    {
        "doctor": "Kardiolog",
        "room": "205-xona",
        "work_time": "09:00 - 15:00",
        "keywords": [
            "yurak",
            "bosim",
            "qon bosim",
            "ko'krak",
            "kokrak",
            "yuragim",
            "taxikardiya",
            "puls",
            "chap qo'l uvishdi",
        ],
    },
    {
        "doctor": "LOR shifokori",
        "room": "112-xona",
        "work_time": "09:00 - 16:00",
        "keywords": [
            "quloq",
            "burun",
            "tomoq",
            "angina",
            "eshitmayapman",
            "tumov",
            "burun bitdi",
            "tomoq og'riq",
        ],
    },
    {
        "doctor": "Nevropatolog",
        "room": "210-xona",
        "work_time": "10:00 - 17:00",
        "keywords": [
            "bosh aylanish",
            "asab",
            "uyushish",
            "qo'l uvishdi",
            "oyoq uvishdi",
            "migren",
            "bel og'riq",
            "umurtqa",
        ],
    },
    {
        "doctor": "Gastroenterolog",
        "room": "215-xona",
        "work_time": "09:30 - 16:30",
        "keywords": [
            "qorin",
            "oshqozon",
            "ich ketish",
            "ko'ngil aynish",
            "qusish",
            "jigar",
            "qabziyat",
            "hazm",
        ],
    },
    {
        "doctor": "Travmatolog",
        "room": "118-xona",
        "work_time": "09:00 - 18:00",
        "keywords": [
            "sinish",
            "chiqish",
            "lat yeyish",
            "oyog'im og'riyapti",
            "qo'lim og'riyapti",
            "jarohat",
            "suyak",
        ],
    },
    {
        "doctor": "Oftalmolog",
        "room": "120-xona",
        "work_time": "09:00 - 15:00",
        "keywords": [
            "ko'z",
            "koz",
            "ko'rish",
            "korish",
            "ko'zim",
            "qizarish",
            "ko'z yoshlanish",
            "ko'z og'riq",
        ],
    },
    {
        "doctor": "Dermatolog",
        "room": "122-xona",
        "work_time": "10:00 - 16:00",
        "keywords": [
            "teri",
            "toshma",
            "qichishish",
            "allergiya",
            "husnbuzar",
            "dog'",
            "qizarib ketdi",
        ],
    },
    {
        "doctor": "Stomatolog",
        "room": "130-xona",
        "work_time": "09:00 - 17:00",
        "keywords": [
            "tish",
            "milk",
            "og'iz",
            "ogiz",
            "tishim og'riyapti",
            "karies",
        ],
    },
]


def route_to_doctor(text: str) -> dict[str, str]:
    normalized = text.lower()

    for rule in DOCTOR_RULES:
        if any(keyword in normalized for keyword in rule["keywords"]):
            return {
                "doctor": rule["doctor"],
                "room": rule["room"],
                "work_time": rule["work_time"],
            }

    return {
        "doctor": "Terapevt",
        "room": "101-xona",
        "work_time": "09:00 - 17:00",
    }


class ChatRequest(BaseModel):
    session_id: str = "robot_1"
    message: str


class CameraRequest(BaseModel):
    session_id: str = "robot_1"
    camera_index: int = 0


@app.on_event("startup")
def startup():
    init_db()


@router.post("/camera/check")
def camera_check(data: CameraRequest):
    seen = detect_human_face(camera_index=data.camera_index, seconds=3)

    if not seen:
        return {
            "human_seen": False,
            "reply": "Hozircha inson aniqlanmadi.",
        }

    sessions[data.session_id] = {
        "state": "ask_full_name",
        "patient_id": None,
        "full_name": None,
    }

    return {
        "human_seen": True,
        "reply": (
            "Assalomu alaykum! Men UZMAX robot yordamchisiman. "
            "Iltimos, ism va familiyangizni ayting."
        ),
    }


@router.post("/chat")
def chat(data: ChatRequest):
    session_id = data.session_id
    message = data.message.strip()

    if session_id not in sessions:
        sessions[session_id] = {
            "state": "ask_full_name",
            "patient_id": None,
            "full_name": None,
        }
        return {
            "reply": "Assalomu alaykum! Iltimos, avval ism va familiyangizni ayting."
        }

    session = sessions[session_id]
    state = session["state"]

    if state == "ask_full_name":
        full_name = (
            message.replace("mening ismim", "")
            .replace("ismim", "")
            .strip()
        )

        if len(full_name.split()) < 2:
            return {
                "reply": "Iltimos, ism va familiyangizni to'liq ayting. Masalan: Ali Valiyev."
            }

        patient_id = register_patient(full_name)
        session["patient_id"] = patient_id
        session["full_name"] = full_name
        session["state"] = "ask_complaint"

        return {
            "reply": (
                f"Rahmat, {full_name}. Siz ro'yxatdan o'tdingiz. "
                "Endi shikoyatingizni ayting. Masalan: boshim og'riyapti, "
                "yo'tal bor, yuragim og'riyapti."
            )
        }

    if state == "ask_complaint":
        patient_id = session["patient_id"]
        full_name = session["full_name"]
        route = route_to_doctor(message)
        queue_no = create_visit(patient_id, message, route)
        session["state"] = "finished"

        return {
            "reply": (
                f"{full_name}, sizning murojaatingiz bo'yicha "
                f"{route['doctor']} qabuliga yo'naltirildingiz. "
                f"Xona: {route['room']}. "
                f"Ish vaqti: {route['work_time']}. "
                f"Sizning navbat raqamingiz: {queue_no}. "
                "Iltimos, navbatingizni kuting."
            ),
            "doctor": route["doctor"],
            "room": route["room"],
            "work_time": route["work_time"],
            "queue_no": queue_no,
        }

    if state == "finished":
        if "yangi" in message.lower() or "qayta" in message.lower():
            sessions[session_id] = {
                "state": "ask_full_name",
                "patient_id": None,
                "full_name": None,
            }
            return {
                "reply": "Yangi ro'yxatdan o'tish boshlandi. Iltimos, ism va familiyangizni ayting."
            }

        return {
            "reply": "Siz allaqachon navbatga yozildingiz. Yangi bemor bo'lsa, 'yangi' deb yozing."
        }

    sessions.pop(session_id, None)
    return {"reply": "Suhbat holati yangilandi. Iltimos, qaytadan boshlang."}


@router.get("/patients")
def get_patients():
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, full_name, created_at
        FROM patients
        ORDER BY id DESC
        LIMIT 50
        """
    )
    rows = cur.fetchall()
    con.close()

    return [
        {
            "id": row[0],
            "full_name": row[1],
            "created_at": row[2],
        }
        for row in rows
    ]


@router.get("/visits")
def get_visits():
    con = db_connect()
    cur = con.cursor()
    cur.execute(
        """
        SELECT visits.id, patients.full_name, visits.complaint, visits.doctor,
               visits.room, visits.work_time, visits.queue_no, visits.created_at
        FROM visits
        JOIN patients ON patients.id = visits.patient_id
        ORDER BY visits.id DESC
        LIMIT 50
        """
    )
    rows = cur.fetchall()
    con.close()

    return [
        {
            "id": row[0],
            "full_name": row[1],
            "complaint": row[2],
            "doctor": row[3],
            "room": row[4],
            "work_time": row[5],
            "queue_no": row[6],
            "created_at": row[7],
        }
        for row in rows
    ]


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hospital_robot:app", host="0.0.0.0", port=8000, reload=True)
