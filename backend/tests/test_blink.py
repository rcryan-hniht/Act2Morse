"""Tests for the blink tracker (app.blink) using fake scores.

Default settings: close 0.5 / open 0.25, min_blink 80 ms,
dot up to 600 ms, letter gap 1200 ms, word gap 2500 ms.
"""

import pytest

from app.blink import BlinkTracker, DASH, DOT, LETTER_GAP, WORD_GAP


@pytest.fixture
def tracker() -> BlinkTracker:
    return BlinkTracker(
        close_threshold=0.5,
        open_threshold=0.25,
        min_blink_ms=80,
        dot_max_ms=600,
        letter_gap_ms=1200,
        word_gap_ms=2500,
    )


class TestHysteresis:
    def test_closed_at_close_threshold(self, tracker: BlinkTracker):
        assert DOT not in tracker.update(0.5, 0)

    def test_flicker_between_thresholds_is_ignored(self, tracker: BlinkTracker):
        # Hover between open(0.25) and close(0.5): no definite state.
        for t in range(0, 2000, 100):
            tracker.update(0.38, t)
        assert tracker.update(0.38, 2000) == []

    def test_flicker_between_thresholds_then_dash(self, tracker: BlinkTracker):
        # No closure started yet; now a definite close and a long reopen.
        tracker.update(0.38, 0)
        tracker.update(0.9, 100)
        assert tracker.update(0.1, 800) == [DASH]

    def test_reopen_happens_only_below_open_threshold(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)  # closure starts
        # 0.38 is between thresholds: still closed
        assert tracker.update(0.38, 700) == []
        # 0.25 is at the open threshold: closure ends as a dash
        assert tracker.update(0.25, 700) == [DASH]


class TestSymbols:
    def test_short_closure_is_ignored(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.1, 50) == []  # < 80 ms: natural blink

    def test_closure_at_min_blink_is_dot(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.1, 80) == [DOT]

    def test_closure_just_under_dot_max_is_dot(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.1, 600) == [DOT]

    def test_closure_over_dot_max_is_dash(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.1, 601) == [DASH]

    def test_multiple_symbols_in_sequence(self, tracker: BlinkTracker):
        events: list[str] = []
        events += tracker.update(0.9, 0)
        events += tracker.update(0.1, 300)   # dot (300ms)
        events += tracker.update(0.9, 600)
        events += tracker.update(0.1, 900)   # dot (300ms)
        events += tracker.update(0.9, 1200)
        events += tracker.update(0.1, 2200)  # dash (1000ms)
        assert events == [DOT, DOT, DASH]

    def test_first_frame_open_then_blink(self, tracker: BlinkTracker):
        tracker.update(0.0, 0)  # first frame definitely open: no closure to end
        tracker.update(0.9, 100)
        assert tracker.update(0.0, 300) == [DOT]


class TestGaps:
    def test_letter_gap_at_threshold(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.1, 200)   # dot ends at 200
        assert LETTER_GAP in tracker.update(0.1, 1400)  # 1200ms after 200

    def test_letter_and_word_gap_together(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.1, 200)   # dot ends at 200
        assert tracker.update(0.1, 2800) == [LETTER_GAP, WORD_GAP]  # 2600ms after 200

    def test_gap_timer_resets_after_each_symbol(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.1, 200)   # dot at 200
        tracker.update(0.1, 1300)  # 1100ms after 200: no gap yet
        tracker.update(0.9, 1400)
        tracker.update(0.1, 1700)  # dot at 1700: timer restarts
        assert tracker.update(0.1, 1800) == []

    def test_gaps_fire_while_eyes_closed(self, tracker: BlinkTracker):
        # Letter gap must fire even if the user keeps blinking into the next
        # letter, or a symbol would never be finalised while eyes keep closing.
        tracker.update(0.9, 0)
        tracker.update(0.1, 200)   # dot at 200
        assert LETTER_GAP in tracker.update(0.9, 1500)  # closed, 1300ms after dot

    def test_no_gaps_before_any_symbol(self, tracker: BlinkTracker):
        assert tracker.update(0.1, 0) == []
        assert tracker.update(0.1, 10_000) == []


class TestNoFace:
    def test_none_score_skips_eye_state_but_advances_gaps(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.1, 200)  # dot at 200
        # Face lost mid-letter: the letter still finalises.
        assert tracker.update(None, 3000) == [LETTER_GAP, WORD_GAP]

    def test_none_score_then_closed_starts_new_closure(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.1, 200)  # dot at 200
        tracker.update(None, 3000)
        tracker.update(0.9, 3200)
        assert tracker.update(0.1, 4000) == [DASH]


class TestValidation:
    def test_thresholds_out_of_order_raise(self):
        with pytest.raises(ValueError):
            BlinkTracker(0.25, 0.5, 80, 600, 1200, 2500)  # open > close

    def test_min_blink_above_dot_max_raises(self):
        with pytest.raises(ValueError):
            BlinkTracker(0.5, 0.25, 700, 600, 1200, 2500)

    def test_letter_gap_above_word_gap_raises(self):
        with pytest.raises(ValueError):
            BlinkTracker(0.5, 0.25, 80, 600, 2500, 1200)


class TestReset:
    def test_reset_clears_state(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.1, 300)  # dot at 300
        tracker.reset()
        assert tracker.update(0.1, 10_000) == []
        tracker.update(0.9, 10_100)
        # A fresh 400 ms closure after reset is an ordinary dot.
        assert tracker.update(0.1, 10_500) == [DOT]
