import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:
    raise RuntimeError(
        "insightface is not installed. Install insightface and onnxruntime before using Face ID."
    ) from exc


class FaceEncoder:
    """Local InsightFace/ArcFace encoder for Face ID.

    The default model pack is buffalo_s. It runs locally with ONNX Runtime and
    returns an identity embedding for the largest detected face.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("FACE_EMBED_MODEL", "buffalo_s")
        self.det_size = self._parse_det_size(os.getenv("FACE_EMBED_DET_SIZE", "640,640"))
        default_root = Path(__file__).resolve().parents[1] / "data" / "insightface"
        self.model_root = Path(os.getenv("FACE_EMBED_ROOT", str(default_root))).resolve()
        self.model_root.mkdir(parents=True, exist_ok=True)
        providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=self.model_name, root=str(self.model_root), providers=providers)
        self.app.prepare(ctx_id=-1, det_size=self.det_size)

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

    def _embed_image(self, image: Image.Image) -> list[float]:
        frame = self._pil_to_bgr(image)
        faces = self.app.get(frame)
        face = self._largest_face(faces)
        if face is None:
            raise ValueError("InsightFace did not detect a face")

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise ValueError("InsightFace did not return an embedding")

        arr = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm:
            arr = arr / norm
        return arr.astype(float).tolist()

    def extract_embedding_from_path(self, path: str) -> list[float]:
        image = self._read_image(path)
        return self._embed_image(image)

    def extract_embedding_from_base64(self, image_data: str) -> list[float]:
        image = self._decode_base64_image(image_data)
        return self._embed_image(image)
