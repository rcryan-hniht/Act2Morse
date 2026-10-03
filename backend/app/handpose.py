"""Hand pose estimation and gesture tracking using opencv/handpose_estimation_mediapipe.

Used in Fin2Morse mode to track 21 3D hand keypoints and translate finger micro-taps
into Morse code (dot / dash).
"""

from __future__ import annotations

import logging
import os
import urllib.request
from typing import Any

import cv2
import numpy as np

from app.mp_handpose import MPHandPose
from app.mp_palmdet import MPPalmDet

logger = logging.getLogger(__name__)

HANDPOSE_DOWNLOAD_URL = (
    "https://huggingface.co/opencv/handpose_estimation_mediapipe/resolve/main/"
    "handpose_estimation_mediapipe_2023feb.onnx"
)
PALMDET_DOWNLOAD_URL = (
    "https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/"
    "palm_detection_mediapipe_2023feb.onnx"
)


def _ensure_file(file_path: str, url: str) -> str:
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 100_000:
        return file_path

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    temp_path = f"{file_path}.tmp"
    logger.info("Downloading %s from %s...", os.path.basename(file_path), url)
    req = urllib.request.Request(url, headers={"User-Agent": "Act2Morse"})
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
        out_file.write(resp.read())
    os.replace(temp_path, file_path)
    logger.info("Successfully downloaded %s", os.path.basename(file_path))
    return file_path


def ensure_hand_models(hand_model_path: str, palm_model_path: str) -> tuple[str, str]:
    """Ensure opencv/handpose_estimation_mediapipe and palm detector models exist."""
    _ensure_file(hand_model_path, HANDPOSE_DOWNLOAD_URL)
    _ensure_file(palm_model_path, PALMDET_DOWNLOAD_URL)
    return hand_model_path, palm_model_path


class HandPoseDetector:
    """Detects hands and micro-tap pinch gestures for Fin2Morse mode.

    Model: opencv/handpose_estimation_mediapipe (plus opencv/palm_detection_mediapipe)
    """

    MODEL_REPO = "opencv/handpose_estimation_mediapipe"

    def __init__(
        self,
        hand_model_path: str,
        palm_model_path: str,
        conf_threshold: float = 0.70,
    ) -> None:
        ensure_hand_models(hand_model_path, palm_model_path)
        self.hand_model_path = hand_model_path
        self.palm_model_path = palm_model_path
        self.conf_threshold = conf_threshold

        self._palmdet = MPPalmDet(palm_model_path, scoreThreshold=0.5, nmsThreshold=0.3)
        self._handpose = MPHandPose(hand_model_path, confThreshold=conf_threshold)

        # Micro-tap state tracking
        self._is_tapping: bool = False
        self._tap_start_ms: int = 0
        self._last_symbol: str | None = None

    def process_frame(self, jpeg_bytes: bytes, now_ms: int) -> dict[str, Any]:
        """Detect hand landmarks and finger tap gestures from one JPEG frame.

        Returns a dictionary containing:
          - hand_detected: bool
          - confidence: float
          - is_finger_tapping: bool
          - pinch_distance: float
          - landmarks: list of [x, y] coordinates for 21 hand joints
          - symbol: '.' | '-' | None (emitted on tap release)
          - gesture: str
        """
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return self._empty_result(symbol=None)

        palms = self._palmdet.infer(frame)
        if palms is None or len(palms) == 0:
            # If tapping was active and hand disappears, evaluate release
            symbol = None
            if self._is_tapping:
                duration = now_ms - self._tap_start_ms
                self._is_tapping = False
                if 75 <= duration <= 1600:
                    symbol = "." if duration < 380 else "-"
            return self._empty_result(symbol=symbol)

        # Pick palm with highest score
        palm = palms[np.argmax(palms[:, -1])]
        hand_result = self._handpose.infer(frame, palm)
        if hand_result is None:
            return self._empty_result(symbol=None)

        conf = float(hand_result[-1])
        bbox = hand_result[0:4]
        landmarks_screen = hand_result[4:67].reshape(21, 3)

        # Landmark 4: Thumb Tip, Landmark 8: Index Tip
        thumb_tip = landmarks_screen[4, :2]
        index_tip = landmarks_screen[8, :2]
        pinch_dist_px = float(np.linalg.norm(thumb_tip - index_tip))

        # Palm scale (distance from wrist 0 to middle MCP 9)
        wrist = landmarks_screen[0, :2]
        middle_mcp = landmarks_screen[9, :2]
        palm_scale = float(np.linalg.norm(wrist - middle_mcp))
        norm_dist = pinch_dist_px / max(palm_scale, 20.0)

        # Pinch detection threshold
        is_pinching = norm_dist < 0.40

        symbol = None
        if is_pinching:
            if not self._is_tapping:
                self._is_tapping = True
                self._tap_start_ms = now_ms
        else:
            if self._is_tapping:
                duration = now_ms - self._tap_start_ms
                self._is_tapping = False
                if 75 <= duration <= 1600:
                    symbol = "." if duration < 380 else "-"

        # Convert landmarks to normalized coordinates [0.0, 1.0] relative to frame size
        h, w = frame.shape[:2]
        norm_landmarks = [
            [float(np.clip(pt[0] / w, 0.0, 1.0)), float(np.clip(pt[1] / h, 0.0, 1.0))]
            for pt in landmarks_screen[:, :2]
        ]

        return {
            "hand_detected": True,
            "confidence": round(conf, 2),
            "is_finger_tapping": self._is_tapping,
            "pinch_distance": round(norm_dist, 3),
            "landmarks": norm_landmarks,
            "symbol": symbol,
            "gesture": "Pinch Tap" if self._is_tapping else "Open Hand",
        }

    def _empty_result(self, symbol: str | None = None) -> dict[str, Any]:
        return {
            "hand_detected": False,
            "confidence": 0.0,
            "is_finger_tapping": False,
            "pinch_distance": 1.0,
            "landmarks": [],
            "symbol": symbol,
            "gesture": "",
        }
