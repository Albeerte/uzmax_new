import base64
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uzmax_server"))
original_cwd = Path.cwd()
import main as server
os.chdir(original_cwd)


class OfflineChat:
    async def extract_person_name(self, text):
        raise RuntimeError("Cloud unavailable in regression test")

    async def get_response_stream(self, *args, **kwargs):
        raise RuntimeError("Cloud unavailable in regression test")
        yield ""


class SilentTTS:
    def __init__(self, synthesizer, speed, sample_rate, loop, audio_queue, voice):
        self.audio_queue = audio_queue

    def start(self):
        pass

    def feed(self, text):
        pass

    async def finish(self):
        await self.audio_queue.put(None)

    def cancel(self):
        self.audio_queue.put_nowait(None)


class PatientCheckInTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        faces = root / "register_faces"
        faces.mkdir()
        self.store = server.FaceVectorStore(str(root / "vectors"))
        self.addCleanup(self.store.client.close)
        patches = {
            "REGISTER_FACES_DIR": faces,
            "REGISTER_FACES_JSON": faces / "registry.json",
            "DOCTOR_QUEUE_FILE": root / "queue.json",
            "IMAGES_DIR": root / "images",
            "SIMPLE_FACE_MODE": True,
            "face_store": self.store,
            "OpenAIClient": OfflineChat,
            "YandexSpeechRecognizer": lambda **kwargs: None,
            "YandexStreamingSynthesizer": lambda **kwargs: None,
            "TtsStreamingSession": SilentTTS,
        }
        for name, value in patches.items():
            patcher = patch.object(server, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)
        self.person = self.store.register([1.0, 0.0, 0.0], "Xarilov", "Jamshid")

    def read_response(self, ws):
        events = []
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] == "ready_to_listen":
                break
        return next(event["text"] for event in events if event["type"] == "llm_done"), events

    def test_headache_is_not_a_greeting_or_english(self):
        text = "biroz boshim og\u02bbriyapti"
        self.assertFalse(server.is_patient_greeting(text))
        self.assertEqual(server.normalize_chat_lang(None, text), "uz-UZ")
        self.assertIsNone(server.local_direct_response(text, self.person, "uz-UZ"))
        self.assertNotIn("ism va familiyangizni", server.local_medical_fallback(text, self.person, "uz-UZ"))
        greeting = server.local_direct_response("salom", self.person, "uz-UZ")
        self.assertIn("Xarilov Jamshid", greeting)
        self.assertNotIn("ism va familiyangizni", greeting)

    def test_queue_request_and_reception_transfer(self):
        initial = server.check_in_recognized_patient(self.person, "uz-UZ")
        text = "biroz boshim og\u02bbriyapti meni ro\u02bbyxatga ol"
        self.assertTrue(server.patient_wants_queue(text))
        reply = server.local_direct_response(text, self.person, "uz-UZ")
        self.assertIn("Xarilov Jamshid", reply)
        self.assertIn("Navbat raqamingiz", reply)
        self.assertNotIn("ism va familiyangizni", reply)
        self.assertNotRegex(reply, r"\d{2}:\d{2}")
        queue = server.load_doctor_queue()
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["id"], initial["id"])
        self.assertEqual(queue[0]["doctor_id"], 5)
        self.assertFalse(queue[0]["auto_check_in"])
        returned = server.check_in_recognized_patient(self.person, "uz-UZ")
        self.assertEqual(returned["id"], initial["id"])
        self.assertEqual(returned["doctor_id"], 5)

    def test_called_patient_and_next_day_checkin(self):
        first = server.check_in_recognized_patient(self.person, "uz-UZ")
        queue = server.load_doctor_queue()
        queue[0]["status"] = "called"
        server.save_doctor_queue(queue)
        self.assertEqual(server.check_in_recognized_patient(self.person, "uz-UZ")["id"], first["id"])
        self.assertEqual(server.add_patient_to_doctor_queue(self.person, server.DOCTOR_DIRECTORY[0])["id"], first["id"])
        queue[0]["date"] = "2000-01-01"
        server.save_doctor_queue(queue)
        self.assertNotEqual(server.check_in_recognized_patient(self.person, "uz-UZ")["id"], first["id"])
        self.assertEqual(len(server.load_doctor_queue()), 2)

    def test_registration_handoff_stale_frame_and_returning_face(self):
        image = np.random.default_rng(7).integers(0, 255, (160, 160, 3), dtype=np.uint8)
        snapshot = server.REGISTER_FACES_DIR / "pending_test.jpg"
        cv2.imwrite(str(snapshot), image)
        sample = {"embedding": [0.0, 1.0, 0.0], "snapshot_path": str(snapshot), "quality": {"ok": True}}
        pending = {"type": "face_identity", "pending_registration": {**sample, "samples": [sample]}}
        with self.client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "set_settings", "live_mode": True})
            ws.send_json(pending)
            self.assertIn("ism va familiyangizni", self.read_response(ws)[0])
            ws.send_json({"type": "text_message", "text": "xarilov jamshid"})
            self.assertIn("Xarilov Jamshidmi", self.read_response(ws)[0])
            ws.send_json({"type": "text_message", "text": "ha"})
            text, events = self.read_response(ws)
            self.assertIn("ro'yxatdan o'tdingiz", text)
            registered = next(event for event in events if event["type"] == "patient_registered")
            person = registered["person"]
            self.assertEqual(registered["queue"]["person_id"], person["person_id"])
            ws.send_json(pending)
            ws.send_json({"type": "text_message", "text": "biroz boshim og\u02bbriyapti meni ro\u02bbyxatga ol"})
            self.assertIn("Navbat raqamingiz", self.read_response(ws)[0])
        self.assertEqual(len(server.read_registered_face_registry()), 1)
        self.assertEqual(len(server.load_doctor_queue()), 1)

        # Exercise the face endpoint against the saved registration, without a camera/model.
        encoded = base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("ascii")
        with patch.object(server, "get_face_encoder") as encoder:
            encoder.return_value.extract_embedding_from_base64.return_value = sample["embedding"]
            result = self.client.post("/api/faces/identify-local", json={"faces": [{
                "image": encoded, "selected_face": {"x": 0, "y": 0, "w": 160, "h": 160},
            }]}).json()["faces"][0]
        self.assertEqual(result["status"], "known")
        self.assertEqual(result["person"]["person_id"], person["person_id"])
        with self.client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"type": "set_settings", "live_mode": True})
            ws.send_json({"type": "face_identity", "person": result["person"]})
            text, _ = self.read_response(ws)
            self.assertIn("Xarilov Jamshid", text)
            self.assertIn("navbat raqamingiz", text)
            self.assertNotIn("ism va familiyangizni", text)
            ws.send_json({"type": "face_identity", "person": result["person"]})
            ws.send_json({"type": "text_message", "text": "salom"})
            self.assertNotIn("ism va familiyangizni", self.read_response(ws)[0])
        self.assertEqual(len(server.load_doctor_queue()), 1)


if __name__ == "__main__":
    unittest.main()
