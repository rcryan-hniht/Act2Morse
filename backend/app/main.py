"""FastAPI app: CORS, /health, and the /ws WebSocket endpoint.

Dual-mode AI vision inference:
- Blink Mode: Uses MichalMlodawski/open-closed-eye-classification-mobilev2
- Fin Mode: Uses opencv/handpose_estimation_mediapipe

Frames are processed in separate threads using ``asyncio.to_thread`` with
backpressure control.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import load_settings
from app.detector import EyeDetector, ensure_eye_model, ensure_model
from app.handpose import HandPoseDetector, ensure_hand_models
from app.session import MorseSession

settings = load_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Ensure all AI models are ready on startup
    await asyncio.to_thread(ensure_eye_model, settings.eye_model_path)
    await asyncio.to_thread(ensure_hand_models, settings.hand_model_path, settings.palm_model_path)
    await asyncio.to_thread(ensure_model, settings.model_path)
    yield


app = FastAPI(title="Act2Morse backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Uptime check; also used by the frontend to wake a sleeping host."""
    return {
        "ok": True,
        "blink_model": "MichalMlodawski/open-closed-eye-classification-mobilev2",
        "fin_model": "opencv/handpose_estimation_mediapipe",
    }


async def _handle_blink_frame(
    detector: EyeDetector, session: MorseSession, data: bytes
) -> dict[str, object]:
    """Detect and decode one JPEG frame in Blink mode using
    MichalMlodawski/open-closed-eye-classification-mobilev2.
    """
    score = await asyncio.to_thread(detector.blink_score, data)
    now_ms = int(time.monotonic() * 1000)
    events = session.process(score, now_ms)

    return {
        "mode": "Blink2Morse",
        "model": "MichalMlodawski/open-closed-eye-classification-mobilev2",
        "face": score is not None,
        "score": score,
        "eyes_closed": bool(score is not None and session.eyes_closed),
        "events": events,
        "symbols": session.symbols,
        "text": session.text,
        "hand_detected": False,
        "hand_confidence": 0.0,
        "is_finger_tapping": False,
    }


async def _handle_fin_frame(
    detector: HandPoseDetector, session: MorseSession, data: bytes
) -> dict[str, object]:
    """Detect and decode one JPEG frame in Fin mode using
    opencv/handpose_estimation_mediapipe.
    """
    now_ms = int(time.monotonic() * 1000)
    hand_res = await asyncio.to_thread(detector.process_frame, data, now_ms)

    events: list[str] = []
    symbol = hand_res.get("symbol")
    if symbol in (".", "-"):
        session.decoder.add_symbol(symbol)
        events.append("dot" if symbol == "." else "dash")

    return {
        "mode": "Fin2Morse",
        "model": "opencv/handpose_estimation_mediapipe",
        "hand_detected": hand_res["hand_detected"],
        "hand_confidence": hand_res["confidence"],
        "is_finger_tapping": hand_res["is_finger_tapping"],
        "landmarks": hand_res.get("landmarks", []),
        "gesture": hand_res.get("gesture", ""),
        "events": events,
        "symbols": session.symbols,
        "text": session.text,
        "face": False,
        "score": None,
        "eyes_closed": False,
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Real-time Morse decoder supporting Blink mode and Fin mode."""
    await websocket.accept()
    eye_detector = EyeDetector(settings.eye_model_path, face_landmarker_path=settings.model_path)
    hand_detector = HandPoseDetector(settings.hand_model_path, settings.palm_model_path)
    session = MorseSession(
        close_threshold=settings.close_threshold,
        open_threshold=settings.open_threshold,
        min_blink_ms=settings.min_blink_ms,
        dot_max_ms=settings.dot_max_ms,
        letter_gap_ms=settings.letter_gap_ms,
        word_gap_ms=settings.word_gap_ms,
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if (data := message.get("bytes")) is not None:
                if session.mode == "Fin2Morse":
                    reply = await _handle_fin_frame(hand_detector, session, data)
                else:
                    reply = await _handle_blink_frame(eye_detector, session, data)
            elif (text_msg := message.get("text")) is not None:
                try:
                    parsed = json.loads(text_msg)
                except (json.JSONDecodeError, TypeError):
                    parsed = None

                if isinstance(parsed, dict) and parsed.get("type") in ("mode", "set_mode"):
                    mode_val = str(parsed.get("mode", ""))
                    session.set_mode(mode_val)
                    reply = {
                        "type": "mode_changed",
                        "mode": session.mode,
                        "model": (
                            "opencv/handpose_estimation_mediapipe"
                            if session.mode == "Fin2Morse"
                            else "MichalMlodawski/open-closed-eye-classification-mobilev2"
                        ),
                        "face": False,
                        "score": None,
                        "eyes_closed": False,
                        "hand_detected": False,
                        "hand_confidence": 0.0,
                        "is_finger_tapping": False,
                        "events": [],
                        "symbols": session.symbols,
                        "text": session.text,
                    }
                elif isinstance(parsed, dict) and parsed.get("type") == "reset":
                    session.reset()
                    reply = {
                        "mode": session.mode,
                        "face": False,
                        "score": None,
                        "eyes_closed": False,
                        "hand_detected": False,
                        "hand_confidence": 0.0,
                        "is_finger_tapping": False,
                        "events": [],
                        "symbols": session.symbols,
                        "text": session.text,
                    }
                elif isinstance(parsed, dict) and parsed.get("type") == "symbol":
                    sym = parsed.get("symbol")
                    events: list[str] = []
                    if sym in (".", "-"):
                        session.decoder.add_symbol(sym)
                        events.append("dot" if sym == "." else "dash")
                    reply = {
                        "mode": session.mode,
                        "face": True,
                        "score": None,
                        "eyes_closed": False,
                        "hand_detected": True,
                        "hand_confidence": 1.0,
                        "is_finger_tapping": False,
                        "events": events,
                        "symbols": session.symbols,
                        "text": session.text,
                    }
                elif isinstance(parsed, dict) and parsed.get("type") == "space":
                    session.decoder.finish_letter()
                    session.decoder.add_word_gap()
                    reply = {
                        "mode": session.mode,
                        "face": True,
                        "score": None,
                        "eyes_closed": False,
                        "events": ["letter_gap", "word_gap"],
                        "symbols": session.symbols,
                        "text": session.text,
                    }
                elif isinstance(parsed, dict) and parsed.get("type") in ("finalize", "letter_gap"):
                    char = session.decoder.finish_letter()
                    reply = {
                        "mode": session.mode,
                        "face": True,
                        "score": None,
                        "eyes_closed": False,
                        "events": ["letter_gap"] if char else [],
                        "symbols": session.symbols,
                        "text": session.text,
                    }
                elif isinstance(parsed, dict) and parsed.get("type") == "frame" and isinstance(parsed.get("image"), str):
                    img_str = parsed["image"]
                    if "," in img_str:
                        img_str = img_str.split(",", 1)[1]
                    try:
                        frame_bytes = base64.b64decode(img_str)
                        if session.mode == "Fin2Morse":
                            reply = await _handle_fin_frame(hand_detector, session, frame_bytes)
                        else:
                            reply = await _handle_blink_frame(eye_detector, session, frame_bytes)
                    except Exception:
                        reply = {
                            "mode": session.mode,
                            "face": False,
                            "score": None,
                            "eyes_closed": False,
                            "hand_detected": False,
                            "events": [],
                            "symbols": session.symbols,
                            "text": session.text,
                        }
                else:
                    reply = {
                        "mode": session.mode,
                        "face": False,
                        "score": None,
                        "eyes_closed": False,
                        "events": [],
                        "symbols": session.symbols,
                        "text": session.text,
                    }
            else:
                continue

            await websocket.send_json(reply)
    except WebSocketDisconnect:
        pass
