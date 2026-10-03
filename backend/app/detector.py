"""Eye-closure detection from JPEG frames using
MichalMlodawski/open-closed-eye-classification-mobilev2.

The detector classifies whether eyes in the frame are open or closed (0 = open, 1 = closed).
It combines MediaPipe Face Landmarker for precise eye region localization with
the fine-tuned MobileNetV2 ONNX model from Hugging Face:
`MichalMlodawski/open-closed-eye-classification-mobilev2`.
"""

from __future__ import annotations

import logging
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

EYE_MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/MichalMlodawski/open-closed-eye-classification-mobilev2/"
    "resolve/main/model.onnx"
)

FACE_LANDMARKER_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _download_file(url: str, dest_path: str) -> str:
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 100_000:
        return dest_path

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = f"{dest_path}.tmp"
    logger.info("Downloading %s from %s...", os.path.basename(dest_path), url)
    req = urllib.request.Request(url, headers={"User-Agent": "Act2Morse"})
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        out_file.write(resp.read())
    os.replace(temp_path, dest_path)
    logger.info("Successfully downloaded %s to %s", os.path.basename(dest_path), dest_path)
    return dest_path


def ensure_eye_model(eye_model_path: str) -> str:
    """Ensure MichalMlodawski/open-closed-eye-classification-mobilev2 ONNX exists on disk."""
    return _download_file(EYE_MODEL_DOWNLOAD_URL, eye_model_path)


def ensure_model(model_path: str) -> str:
    """Ensure the MediaPipe face_landmarker model exists on disk (backward compatibility)."""
    return _download_file(FACE_LANDMARKER_DOWNLOAD_URL, model_path)


class EyeDetector:
    """Extracts an eye-closure score (0 = open, 1 = closed) from a JPEG frame using
    MichalMlodawski/open-closed-eye-classification-mobilev2.
    """

    MODEL_REPO = "MichalMlodawski/open-closed-eye-classification-mobilev2"

    def __init__(
        self,
        model_path: str | None = None,
        face_landmarker_path: str | None = None,
    ) -> None:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_eye_path = os.path.join(
            backend_dir, "models", "eye_classification_mobilenetv2.onnx"
        )
        default_task_path = os.path.join(
            backend_dir, "models", "face_landmarker.task"
        )

        # Handle backward-compatible argument passing
        if model_path and model_path.endswith(".task"):
            face_landmarker_path = model_path
            eye_model_path = default_eye_path
        elif model_path:
            eye_model_path = model_path
        else:
            eye_model_path = default_eye_path

        if not face_landmarker_path:
            face_landmarker_path = default_task_path

        ensure_eye_model(eye_model_path)
        self._eye_net = cv2.dnn.readNetFromONNX(eye_model_path)

        self._landmarker = None
        if os.path.isfile(face_landmarker_path):
            try:
                options = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=face_landmarker_path),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    output_face_blendshapes=True,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                logger.warning("Could not initialize FaceLandmarker helper: %s", e)

        self._timestamp_us = 0

    def _classify_eye_crop(self, crop: np.ndarray) -> float:
        """Runs inference with MichalMlodawski/open-closed-eye-classification-mobilev2.

        Input: 224x224 RGB image, normalized with mean=0.5, std=0.5 (scale=1/127.5).
        Returns probability of Category 0 (Closed eye), where 0.0 = Open, 1.0 = Closed.
        """
        if crop is None or crop.size == 0:
            return 0.0

        blob = cv2.dnn.blobFromImage(
            cv2.resize(crop, (224, 224)),
            scalefactor=1.0 / 127.5,
            size=(224, 224),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self._eye_net.setInput(blob)
        out = self._eye_net.forward()
        logits = out[0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        # Class 0: Closed Eyes, Class 1: Open Eyes
        return float(probs[0])

    def blink_score(self, jpeg_bytes: bytes) -> float | None:
        """Run detection on one JPEG frame.

        Returns ``None`` when no face is found, otherwise the eye closure score
        (0.0 = completely open, 1.0 = completely closed).
        """
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None  # Unreadable frame: treat as no face

        self._timestamp_us += 33_333  # ~30 fps monotonic timestamps
        h, w = frame.shape[:2]

        if self._landmarker is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_us)
            if not result.face_landmarks:
                return None  # No face detected

            landmarks = result.face_landmarks[0]
            # Key eye landmarks in MediaPipe Face Mesh
            left_eye_indices = [33, 133, 159, 145, 158, 153, 144, 160]
            right_eye_indices = [362, 263, 386, 374, 385, 380, 373, 387]

            eye_scores: list[float] = []
            for eye_indices in (left_eye_indices, right_eye_indices):
                pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices])
                min_x, min_y = np.min(pts, axis=0)
                max_x, max_y = np.max(pts, axis=0)
                pad_x = int((max_x - min_x) * 0.35) + 6
                pad_y = int((max_y - min_y) * 0.35) + 6

                x1 = max(0, min_x - pad_x)
                y1 = max(0, min_y - pad_y)
                x2 = min(w, max_x + pad_x)
                y2 = min(h, max_y + pad_y)

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    eye_scores.append(self._classify_eye_crop(crop))

            # Blendshape reference score as complementary signal if available
            blendshape_score = None
            if result.face_blendshapes:
                shapes = result.face_blendshapes[0]
                b_scores = [
                    b.score for b in shapes if b.category_name in ("eyeBlinkLeft", "eyeBlinkRight")
                ]
                if len(b_scores) >= 2:
                    blendshape_score = float(sum(b_scores) / len(b_scores))

            if eye_scores:
                mobilenet_score = float(sum(eye_scores) / len(eye_scores))
                if blendshape_score is not None:
                    # Robust weighted blend: 60% MobileNetV2 classifier + 40% blendshape
                    return float(0.60 * mobilenet_score + 0.40 * blendshape_score)
                return mobilenet_score

            return blendshape_score

        # Fallback ROI if landmarker helper not active
        roi_x1 = int(w * 0.20)
        roi_y1 = int(h * 0.20)
        roi_x2 = int(w * 0.80)
        roi_y2 = int(h * 0.52)
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            return self._classify_eye_crop(roi_crop)

        return None
