"""One WebSocket connection's Morse session: tracker + decoder.

Pure module: no FastAPI, OpenCV or MediaPipe imports so it can be unit
tested with fake scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.blink import BlinkTracker, DASH, DOT, LETTER_GAP, WORD_GAP
from app.morse import MorseDecoder


@dataclass
class MorseSession:
    """Processes eye-closure frames for one connection.

    Call :meth:`process` once per frame with the closure score (0 = open,
    1 = closed, or ``None`` when no face was found) and the frame timestamp
    in milliseconds. Frames without a face only advance the gap timers.
    """

    close_threshold: float
    open_threshold: float
    min_blink_ms: int
    dot_max_ms: int
    letter_gap_ms: int
    word_gap_ms: int
    mode: str = "Blink2Morse"
    last_symbol_time_ms: int = 0

    def __post_init__(self) -> None:
        self.tracker = BlinkTracker(
            close_threshold=self.close_threshold,
            open_threshold=self.open_threshold,
            min_blink_ms=self.min_blink_ms,
            dot_max_ms=self.dot_max_ms,
            letter_gap_ms=self.letter_gap_ms,
            word_gap_ms=self.word_gap_ms,
        )
        self.decoder = MorseDecoder()

    def set_mode(self, mode: str) -> None:
        """Set active operation mode: 'Fin2Morse' or 'Blink2Morse'."""
        clean = mode.strip().lower()
        if "fin" in clean:
            self.mode = "Fin2Morse"
        else:
            self.mode = "Blink2Morse"

    def add_symbol_debounced(self, symbol: str, now_ms: int) -> bool:
        """Add a symbol with debouncing window to prevent double registration."""
        if symbol not in (".", "-"):
            return False
        if now_ms - self.last_symbol_time_ms < 280:
            return False
        self.decoder.add_symbol(symbol)
        self.last_symbol_time_ms = now_ms
        return True

    def process(self, score: float | None, now_ms: int) -> list[str]:
        """Process one frame; returns events triggered by this frame.

        The symbol buffer and decoded text are exposed via :attr:`symbols`
        and :attr:`text` for the server to include in every reply.
        """
        events: list[str] = []
        for event in self.tracker.update(score, now_ms):
            if event == DOT:
                self.decoder.add_symbol(".")
            elif event == DASH:
                self.decoder.add_symbol("-")
            elif event == LETTER_GAP:
                self.decoder.finish_letter()
            elif event == WORD_GAP:
                self.decoder.finish_letter()
                self.decoder.add_word_gap()
            events.append(event)
        return events

    @property
    def eyes_closed(self) -> bool | None:
        """Whether the eyes are currently closed (None when no face yet)."""
        return self.tracker.eyes_closed

    @property
    def symbols(self) -> str:
        """Morse symbols of the letter in progress."""
        return self.decoder.symbols

    @property
    def text(self) -> str:
        """Decoded text so far."""
        return self.decoder.text

    def reset(self) -> None:
        """Clear current symbols and decoded text."""
        self.tracker.reset()
        self.decoder.reset()
        self.last_symbol_time_ms = 0
