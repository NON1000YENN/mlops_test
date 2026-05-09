from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from app.config import settings

_session: ort.InferenceSession | None = None
_input_name: str | None = None
_output_name: str | None = None


class ImageDecodeError(ValueError):
    """ใช้สำหรับบอกว่าไฟล์ที่ส่งมาไม่ใช่รูปภาพหรือรูปเสีย"""


def _get_session() -> ort.InferenceSession:
    """โหลด ONNX Runtime Session แบบ lazy-load ภายในแต่ละ process"""
    global _session, _input_name, _output_name

    if _session is None:
        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        _session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name
        _output_name = _session.get_outputs()[0].name

    return _session


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """แปลงรูปภาพเป็น numpy tensor รูปแบบ [1, 224, 224, 3]"""
    try:
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("Uploaded file is not a valid image or is corrupted.") from exc

    image = image.resize((settings.image_size, settings.image_size))
    arr = np.asarray(image).astype(np.float32)

    if settings.preprocess_mode == "mobilenet_v2":
        # MobileNetV2 preprocess_input: pixel range [0, 255] -> [-1, 1]
        arr = (arr / 127.5) - 1.0
    else:
        # ค่า default ที่ใช้บ่อยในงาน train CNN/ImageDataGenerator(rescale=1./255)
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0).astype(np.float32)


def predict_from_bytes(image_bytes: bytes) -> dict:
    """ฟังก์ชันนี้จะถูกรันใน ProcessPoolExecutor"""
    session = _get_session()
    input_tensor = preprocess_image(image_bytes)

    outputs = session.run([_output_name], {_input_name: input_tensor})[0]
    probs = np.asarray(outputs)[0].astype(float)

    # กรณี output เป็น logits ให้แปลงเป็น softmax กันพลาด
    if probs.min() < 0 or probs.max() > 1 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
        exp = np.exp(probs - np.max(probs))
        probs = exp / exp.sum()

    class_names = settings.class_names
    if len(class_names) != len(probs):
        class_names = [f"class_{i}" for i in range(len(probs))]

    pred_idx = int(np.argmax(probs))

    return {
        "label": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(probs))},
    }
