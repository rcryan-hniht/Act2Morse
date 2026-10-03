"""Unit tests for Blink mode (MichalMlodawski/open-closed-eye-classification-mobilev2)
and Fin mode (opencv/handpose_estimation_mediapipe).
"""

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.config import load_settings
from app.detector import EyeDetector, ensure_eye_model
from app.handpose import HandPoseDetector, ensure_hand_models
from app.main import app
from app.session import MorseSession


@pytest.fixture
def settings():
    return load_settings()


class TestEyeDetectorBlinkMode:
    def test_ensure_eye_model(self, settings):
        path = ensure_eye_model(settings.eye_model_path)
        assert path == settings.eye_model_path

    def test_eye_detector_initialization(self, settings):
        detector = EyeDetector(
            model_path=settings.eye_model_path,
            face_landmarker_path=settings.model_path,
        )
        assert detector is not None
        assert detector.MODEL_REPO == "MichalMlodawski/open-closed-eye-classification-mobilev2"

    def test_blink_score_invalid_bytes(self, settings):
        detector = EyeDetector(model_path=settings.eye_model_path)
        # Corrupted / invalid image bytes should return None
        assert detector.blink_score(b"not_an_image") is None

    def test_blink_score_blank_frame(self, settings):
        detector = EyeDetector(
            model_path=settings.eye_model_path,
            face_landmarker_path=settings.model_path,
        )
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", blank)
        # No face detected in a completely blank frame
        score = detector.blink_score(jpeg.tobytes())
        assert score is None


class TestHandPoseDetectorFinMode:
    def test_ensure_hand_models(self, settings):
        h_path, p_path = ensure_hand_models(settings.hand_model_path, settings.palm_model_path)
        assert h_path == settings.hand_model_path
        assert p_path == settings.palm_model_path

    def test_handpose_detector_initialization(self, settings):
        detector = HandPoseDetector(
            hand_model_path=settings.hand_model_path,
            palm_model_path=settings.palm_model_path,
        )
        assert detector is not None
        assert detector.MODEL_REPO == "opencv/handpose_estimation_mediapipe"

    def test_handpose_blank_frame(self, settings):
        detector = HandPoseDetector(
            hand_model_path=settings.hand_model_path,
            palm_model_path=settings.palm_model_path,
        )
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", blank)
        result = detector.process_frame(jpeg.tobytes(), now_ms=1000)
        assert result["hand_detected"] is False
        assert result["is_finger_tapping"] is False
        assert result["symbol"] is None


class TestSessionDualMode:
    def test_session_mode_switch(self, settings):
        session = MorseSession(
            close_threshold=settings.close_threshold,
            open_threshold=settings.open_threshold,
            min_blink_ms=settings.min_blink_ms,
            dot_max_ms=settings.dot_max_ms,
            letter_gap_ms=settings.letter_gap_ms,
            word_gap_ms=settings.word_gap_ms,
        )
        assert session.mode == "Blink2Morse"

        session.set_mode("Fin2Morse")
        assert session.mode == "Fin2Morse"

        session.set_mode("blink")
        assert session.mode == "Blink2Morse"

        session.set_mode("fin")
        assert session.mode == "Fin2Morse"


class TestWebSocketDualMode:
    def test_health_check(self):
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["ok"] is True
        assert data["blink_model"] == "MichalMlodawski/open-closed-eye-classification-mobilev2"
        assert data["fin_model"] == "opencv/handpose_estimation_mediapipe"

    def test_websocket_mode_switching(self):
        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # Switch to Fin2Morse
            ws.send_json({"type": "set_mode", "mode": "Fin2Morse"})
            res1 = ws.receive_json()
            assert res1["type"] == "mode_changed"
            assert res1["mode"] == "Fin2Morse"
            assert res1["model"] == "opencv/handpose_estimation_mediapipe"

            # Switch to Blink2Morse
            ws.send_json({"type": "set_mode", "mode": "Blink2Morse"})
            res2 = ws.receive_json()
            assert res2["type"] == "mode_changed"
            assert res2["mode"] == "Blink2Morse"
            assert res2["model"] == "MichalMlodawski/open-closed-eye-classification-mobilev2"
