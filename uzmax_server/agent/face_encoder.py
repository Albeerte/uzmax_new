import base64
import io
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("uzmax.face_encoder")

try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHTFACE = True
except ImportError:
    FaceAnalysis = None
    _HAS_INSIGHTFACE = False


class FaceEncoder:
    """API and local face encoder for RoboMed Face ID.

    Attempts to use InsightFace local ONNX model. If missing or failing,
    gracefully falls back to OpenCV Haar Cascade + Feature Vector API
    so that the server NEVER crashes on missing local model files.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("FACE_EMBED_MODEL", "buffalo_s")
        self.det_size = self._parse_det_size(os.getenv("FACE_EMBED_DET_SIZE", "640,640"))
        self.det_thresh = float(os.getenv("FACE_DET_THRESH", "0.5"))
        default_root = Path(__file__).resolve().parents[1] / "data" / "insightface"
        self.model_root = Path(os.getenv("FACE_EMBED_ROOT", str(default_root))).resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.app = None

        if _HAS_INSIGHTFACE:
            try:
                providers = ["CPUExecutionProvider"]
                app = FaceAnalysis(
                    name=self.model_name,
                    root=str(self.model_root),
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                app.prepare(ctx_id=-1, det_thresh=self.det_thresh, det_size=self.det_size)
                self.app = app
                logger.info("[FaceEncoder] InsightFace loaded successfully: %s", self.model_name)
            except Exception as exc:
                logger.warning("[FaceEncoder] InsightFace init failed (%s). Using OpenCV/API Fallback encoder.", exc)
                self.app = None
        else:
            logger.info("[FaceEncoder] InsightFace module not present. Using OpenCV/API Fallback encoder.")

        # OpenCV Haar Cascade Fallback Detector
        self._cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self._cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._cascade = None

    @staticmethod
    def _parse_det_size(value: str) -> tuple[int, int]:
        try:
            left, right = (value or "640,640").replace("x", ",").split(",", 1)
            return int(left.strip()), int(right.strip())
        except Exception:
            return 640, 640

    @staticmethod
    def _read_image(path: str) -> Image.Image:
        return Image.open(str(Path(path))).convert("RGB")

    @staticmethod
    def _decode_base64_image(image_data: str) -> Image.Image:
        if "," in image_data:
            _, image_data = image_data.split(",", 1)
        binary = base64.b64decode(image_data)
        return Image.open(io.BytesIO(binary)).convert("RGB")

    @staticmethod
    def _pil_to_bgr(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _largest_face(faces):
        if not faces:
            return None
        return max(
            faces,
            key=lambda face: max(0.0, float(face.bbox[2] - face.bbox[0]))
            * max(0.0, float(face.bbox[3] - face.bbox[1])),
        )

    def _fallback_embed_image(self, image: Image.Image) -> list[float]:
        """Fallback feature extractor using OpenCV face crop + normalized 512-dim descriptor."""
        frame = self._pil_to_bgr(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        crop = None
        if self._cascade is not None:
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                crop = gray[y:y+h, x:x+w]

        if crop is None or crop.size == 0:
            crop = gray

        # Resize crop to 16x32 = 512 features
        resized = cv2.resize(crop, (16, 32)).astype(np.float32).flatten()
        norm = np.linalg.norm(resized)
        if norm > 0:
            resized /= norm
        return resized.astype(float).tolist()

    @property
    def embedding_model(self) -> str:
        """Identifier stored with embeddings; vectors from different models are not comparable."""
        return f"insightface:{self.model_name}" if self.app is not None else "opencv_fallback"

    def detect_faces(self, frame: np.ndarray) -> list[dict] | None:
        """Detect faces with InsightFace SCRFD on a BGR frame.

        Returns boxes sorted by area (largest first), or None when InsightFace is
        unavailable so the caller can fall back to another detector.
        """
        if self.app is None or frame is None:
            return None
        bboxes, kpss = self.app.det_model.detect(frame, max_num=0, metric="default")
        height, width = frame.shape[:2]
        faces = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = (float(v) for v in bboxes[i, :5])
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            face = {
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "area": w * h,
                "score": round(score, 4),
                "source": "insightface_scrfd",
            }
            if kpss is not None:
                face["kps"] = [[round(float(px), 1), round(float(py), 1)] for px, py in kpss[i]]
            faces.append(face)
        return sorted(faces, key=lambda item: item["area"], reverse=True)

    def _embed_image(self, image: Image.Image) -> list[float]:
        if self.app is not None:
            try:
                frame = self._pil_to_bgr(image)
                faces = self.app.get(frame)
                face = self._largest_face(faces)
                if face is not None:
                    embedding = getattr(face, "normed_embedding", None)
                    if embedding is None:
                        embedding = getattr(face, "embedding", None)
                    if embedding is not None:
                        arr = np.asarray(embedding, dtype=np.float32)
                        norm = np.linalg.norm(arr)
                        if norm:
                            arr = arr / norm
                        return arr.astype(float).tolist()
            except Exception as exc:
                logger.warning("[FaceEncoder] InsightFace embed failed: %s", exc)
            # The fallback descriptor is also 512-dim but lives in a different space;
            # mixing it with ArcFace vectors would create false matches.
            return []

        return self._fallback_embed_image(image)

    def extract_embedding_from_path(self, path: str) -> list[float]:
        image = self._read_image(path)
        return self._embed_image(image)

    def extract_embedding_from_base64(self, image_data: str) -> list[float]:
        image = self._decode_base64_image(image_data)
        return self._embed_image(image)
