"""Tests for the session (app.session) and config defaults using fake scores."""

import os

import pytest

from app.session import MorseSession


@pytest.fixture
def session() -> MorseSession:
    return MorseSession(
        close_threshold=0.6,
        open_threshold=0.4,
        min_blink_ms=200,
        dot_max_ms=500,
        letter_gap_ms=1500,
        word_gap_ms=3000,
    )


class TestSession:
    def test_dot_dash_letter(self, session: MorseSession):
        # One dot then one dash -> ".." is "I"? no: ".-" is "A".
        session.process(0.9, 0)
        session.process(0.1, 300)  # dot
        assert session.symbols == "."
        session.process(0.9, 600)
        events = session.process(0.1, 1200)  # 600 ms closure -> dash
        assert events.count("dash") == 1
        assert session.symbols == ".-"
        session.process(0.1, 3000)  # letter gap at 1200+1500
        assert session.text == "A"

    def test_word_gap_adds_trailing_space(self, session: MorseSession):
        session.process(0.9, 0)
        session.process(0.1, 300)  # dot -> "E" after letter gap
        session.process(0.1, 2000)  # letter gap
        session.process(0.1, 3400)  # word gap
        assert session.text == "E "

    def test_sos(self, session: MorseSession):
        clock = {"t": 0}

        def blink(closed_ms: int) -> None:
            session.process(0.9, clock["t"])
            clock["t"] += closed_ms
            session.process(0.1, clock["t"])
            clock["t"] += 700  # inter-symbol open time (below letter gap)

        # S = ...
        for _ in range(3):
            blink(300)
        session.process(0.1, clock["t"] + 1500)  # letter gap
        # O = ---
        for _ in range(3):
            blink(700)
        session.process(0.1, clock["t"] + 1500)
        # S = ...
        for _ in range(3):
            blink(300)
        session.process(0.1, clock["t"] + 1500)

        assert session.text == "SOS"

    def test_no_face_frames_finalise_letter(self, session: MorseSession):
        session.process(0.9, 0)
        session.process(0.1, 300)  # dot
        assert session.process(None, 2000) == ["letter_gap"]
        assert session.text == "E"

    def test_reset_clears_symbols_and_text(self, session: MorseSession):
        session.process(0.9, 0)
        session.process(0.1, 300)  # dot
        session.process(0.1, 2000)  # letter gap -> "E"
        session.reset()
        assert session.symbols == ""
        assert session.text == ""


class TestConfigDefaults:
    def test_defaults_match_agents_md(self):
        from app.config import load_settings

        # Clear any overrides from the developer's environment.
        env_names = [
            "MODEL_PATH", "ALLOWED_ORIGINS", "CLOSE_THRESHOLD", "OPEN_THRESHOLD",
            "MIN_BLINK_MS", "DOT_MAX_MS", "LETTER_GAP_MS", "WORD_GAP_MS",
        ]
        saved = {n: os.environ.pop(n, None) for n in env_names}
        try:
            s = load_settings()
            assert s.close_threshold == 0.6
            assert s.open_threshold == 0.4
            assert s.min_blink_ms == 200
            assert s.dot_max_ms == 500
            assert s.letter_gap_ms == 1500
            assert s.word_gap_ms == 3000
            assert s.allowed_origins == ("http://localhost:5173",)
            assert s.model_path.endswith("models/face_landmarker.task")
        finally:
            for n, v in saved.items():
                if v is not None:
                    os.environ[n] = v

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch):
        from app.config import load_settings

        monkeypatch.setenv("DOT_MAX_MS", "600")
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
        s = load_settings()
        assert s.dot_max_ms == 600
        assert s.allowed_origins == ("https://a.example", "https://b.example")
