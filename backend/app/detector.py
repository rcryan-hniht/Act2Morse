"""Eye-closure detection from JPEG frames using MediaPipe Face Landmarker.

The detector is instantiated once per WebSocket connection and is **not**
thread-safe, but each connection runs its frames through
``asyncio.to_thread`` (never in the event loop) per AGENTS.md.

Uses the ``eyeBlinkLeft`` / ``eyeBlinkRight`` blendshape scores, which run
0 = open, 1 = closed.
"""

from __future__ import annotations

import numpy as np

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class EyeDetector:
    """Extracts an eye-closure score (0 = open, 1 = closed) from a JPEG frame."""

    def __init__(self, model_path: str) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_us = 0

    def blink_score(self, jpeg_bytes: bytes) -> float | None:
        """Run detection on one JPEG frame.

        Returns ``None`` when no face was found, otherwise the mean of the
        ``eyeBlinkLeft`` and ``eyeBlinkRight`` blendshape scores.
        """
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None  # Not decodable as an image: treat as "no face".

        # VIDEO mode requires strictly increasing timestamps, in microseconds.
        self._timestamp_us += 33_333  # ~30 fps; order, not realism, matters.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_us)
        if not result.face_blendshapes:
            return None

        blendshapes = result.face_blendshapes[0]
        scores = [
            b.score
            for b in blendshapes
            if b.category_name in ("eyeBlinkLeft", "eyeBlinkRight")
        ]
        if len(scores) < 2:
            return None

        return float(sum(scores) / len(scores))
