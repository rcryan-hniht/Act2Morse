"""Turns eye-closure scores over time into Morse dot/dash/gap events.

Pure module: no FastAPI, OpenCV or MediaPipe imports so it can be unit
tested with fake scores.

Timing rules (defaults in :mod:`app.config`, all configurable):

- Eyes are closed when ``score >= close_threshold`` and open again when
  ``score <= open_threshold``; two thresholds avoid flicker.
- Closure shorter than ``min_blink_ms`` is a natural blink and is ignored.
- Closure up to ``dot_max_ms`` is a dot, longer is a dash.
- Eyes open for ``letter_gap_ms`` after a symbol ends the letter; open for
  ``word_gap_ms`` also adds a space (word gap).
"""

from __future__ import annotations

from dataclasses import dataclass


DOT = "dot"
DASH = "dash"
LETTER_GAP = "letter_gap"
WORD_GAP = "word_gap"

_EYES_OPEN = "open"
_EYES_CLOSED = "closed"


@dataclass
class BlinkTracker:
    """State machine mapping eye-closure scores to Morse events.

    Call :meth:`update` once per frame with the closure score (0 = open,
    1 = closed) and the frame timestamp in milliseconds (server receive
    time); it returns the events triggered by that frame.
    """

    close_threshold: float
    open_threshold: float
    min_blink_ms: int
    dot_max_ms: int
    letter_gap_ms: int
    word_gap_ms: int

    def __post_init__(self) -> None:
        if self.open_threshold > self.close_threshold:
            raise ValueError("open_threshold must be <= close_threshold")
        if self.min_blink_ms >= self.dot_max_ms:
            raise ValueError("min_blink_ms must be < dot_max_ms")
        if self.letter_gap_ms >= self.word_gap_ms:
            raise ValueError("letter_gap_ms must be < word_gap_ms")
        self._eyes: str | None = None  # None until the first definite state
        self._closure_start_ms: int | None = None  # None until a closure starts
        self._last_symbol_end_ms: int | None = None  # None until first symbol
        self._letter_gap_sent = False
        self._word_gap_sent = False

    def update(self, score: float | None, now_ms: int) -> list[str]:
        """Process one frame; returns events triggered by this frame.

        ``score`` is ``None`` when no face was found: the eye-state machine
        is skipped but the gap timers keep running so a pending letter
        still finalises.
        """
        events: list[str] = []

        if score is not None:
            if self._eyes != _EYES_CLOSED and score >= self.close_threshold:
                # Crossing into a definite closure (or a first definitely-closed frame).
                self._eyes = _EYES_CLOSED
                self._closure_start_ms = now_ms
            elif self._eyes != _EYES_OPEN and score <= self.open_threshold:
                # Crossing into a definite opening (or a first definitely-open frame).
                self._eyes = _EYES_OPEN
                events.extend(self._end_closure(now_ms))

        # Gap logic runs from the last symbol's end regardless of eye state,
        # so holding the eyes closed (or no face) still finalises letters.
        if self._last_symbol_end_ms is not None:
            gap_ms = now_ms - self._last_symbol_end_ms
            if not self._letter_gap_sent and gap_ms >= self.letter_gap_ms:
                self._letter_gap_sent = True
                events.append(LETTER_GAP)
            if not self._word_gap_sent and gap_ms >= self.word_gap_ms:
                self._word_gap_sent = True
                events.append(WORD_GAP)

        return events

    def _end_closure(self, now_ms: int) -> list[str]:
        """Classify a finished closure; returns a dot/dash event or nothing."""
        if self._closure_start_ms is None:
            # No closure was in progress (e.g. the first frame was already
            # definitely open): there is nothing to classify.
            return []

        duration_ms = now_ms - self._closure_start_ms
        assert duration_ms >= 0  # frames are processed in receive order
        if duration_ms < self.min_blink_ms:
            return []  # Natural blink: ignored entirely.

        self._last_symbol_end_ms = now_ms
        self._letter_gap_sent = False
        self._word_gap_sent = False
        return [DOT if duration_ms <= self.dot_max_ms else DASH]

    @property
    def eyes_closed(self) -> bool | None:
        """Whether the eyes are currently closed (None before any closure)."""
        return self._eyes == _EYES_CLOSED

    def reset(self) -> None:
        """Clear all in-progress state (symbols are owned by the decoder)."""
        self._eyes = None
        self._closure_start_ms = None
        self._last_symbol_end_ms = None
        self._letter_gap_sent = False
        self._word_gap_sent = False
