"""FastAPI app: CORS, /health, and the /ws WebSocket endpoint.

One EyeDetector + MorseSession per WebSocket connection. Frames are run
through ``asyncio.to_thread`` because MediaPipe is CPU-bound; the server
replies once per received message (backpressure: the client waits for the
reply before sending the next frame).
"""

from __future__ import annotations

import asyncio
import json
import time

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import load_settings
from app.detector import EyeDetector, ensure_model
from app.session import MorseSession

settings = load_settings()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Ensure model is available on startup
    await asyncio.to_thread(ensure_model, settings.model_path)
    yield


app = FastAPI(title="Blink2Morse backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, bool]:
    """Uptime check; also used by the frontend to wake a sleeping host."""
    return {"ok": True}


async def _handle_frame(
    detector: EyeDetector, session: MorseSession, data: bytes
) -> dict[str, object]:
    """Detect and decode one JPEG frame; builds the reply message."""
    score = await asyncio.to_thread(detector.blink_score, data)
    now_ms = int(time.monotonic() * 1000)
    events = session.process(score, now_ms)

    return {
        "face": score is not None,
        "score": score,
        "eyes_closed": bool(score is not None and session.eyes_closed),
        "events": events,
        "symbols": session.symbols,
        "text": session.text,
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Decode blinks to Morse for one connection."""
    await websocket.accept()
    detector = EyeDetector(settings.model_path)
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
                reply = await _handle_frame(detector, session, data)
            elif (text_msg := message.get("text")) is not None:
                # Only reset is expected; anything else is ignored but still
                # answered once so the client's backpressure doesn't stall.
                try:
                    parsed = json.loads(text_msg)
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, dict) and parsed.get("type") == "reset":
                    session.reset()
                reply = {
                    "face": False,
                    "score": None,
                    "eyes_closed": False,
                    "events": [],
                    "symbols": session.symbols,
                    "text": session.text,
                }
            else:
                continue  # Empty message: nothing to answer.

            await websocket.send_json(reply)
    except WebSocketDisconnect:
        pass
