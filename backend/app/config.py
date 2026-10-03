"""Settings for the Act2Morse backend.

Every threshold and timing can be overridden with an environment variable so
the decoding can be tuned per user and lighting conditions without touching
code. See AGENTS.md for the meaning of each value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    model_path: str
    allowed_origins: tuple[str, ...]
    # Hysteresis: eyes close above close_threshold, reopen below open_threshold.
    close_threshold: float
    open_threshold: float
    # Closure shorter than min_blink_ms is a natural blink and is ignored.
    min_blink_ms: int
    # Closure below dot_max_ms is a dot, otherwise a dash.
    dot_max_ms: int
    # Eyes open this long after a symbol ends the letter; this long also adds a space.
    letter_gap_ms: int
    word_gap_ms: int


def load_settings() -> Settings:
    """Build settings from the environment, with the defaults from AGENTS.md."""
    origins_raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173")
    return Settings(
        model_path=os.environ.get(
            "MODEL_PATH", os.path.join(_BACKEND_DIR, "models", "face_landmarker.task")
        ),
        allowed_origins=tuple(o.strip() for o in origins_raw.split(",") if o.strip()),
        close_threshold=_env_float("CLOSE_THRESHOLD", 0.5),   # was 0.6; lighter blinks reach ~0.5
        open_threshold=_env_float("OPEN_THRESHOLD", 0.25),    # was 0.4; ensures clean reopen signal
        min_blink_ms=_env_int("MIN_BLINK_MS", 80),            # was 200; 22fps = 45ms/frame, 80ms = ~2 frames
        dot_max_ms=_env_int("DOT_MAX_MS", 600),               # was 500; more room for dot at low fps
        letter_gap_ms=_env_int("LETTER_GAP_MS", 1200),        # was 1500
        word_gap_ms=_env_int("WORD_GAP_MS", 2500),            # was 3000
    )
