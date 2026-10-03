"""Tests for the blink tracker (app.blink) using fake scores.

Default settings from AGENTS.md: close 0.6 / open 0.4, min_blink 200 ms,
dot up to 500 ms, letter gap 1500 ms, word gap 3000 ms.
"""

import pytest

from app.blink import BlinkTracker, DASH, DOT, LETTER_GAP, WORD_GAP


@pytest.fixture
def tracker() -> BlinkTracker:
    return BlinkTracker(
        close_threshold=0.6,
        open_threshold=0.4,
        min_blink_ms=200,
        dot_max_ms=500,
        letter_gap_ms=1500,
        word_gap_ms=3000,
    )


class TestHysteresis:
    def test_closed_at_close_threshold(self, tracker: BlinkTracker):
        assert DOT not in tracker.update(0.6, 0)

    def test_flicker_between_thresholds_is_ignored(self, tracker: BlinkTracker):
        # Hover between 0.4 and 0.6: no definite state, so a closure never
        # even starts.
        for t in range(0, 2000, 100):
            tracker.update(0.5, t)
        assert tracker.update(0.5, 2000) == []

    def test_flicker_between_thresholds_then_dash(self, tracker: BlinkTracker):
        # No closure started yet; now a definite close and a long reopen.
        tracker.update(0.5, 0)
        tracker.update(0.9, 100)
        assert tracker.update(0.3, 700) == [DASH]

    def test_reopen_happens_only_below_open_threshold(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)  # closure starts
        # 0.5 is between thresholds: still closed
        assert tracker.update(0.5, 600) == []
        # 0.4 is at the open threshold: closure ends as a dash
        assert tracker.update(0.4, 600) == [DASH]


class TestSymbols:
    def test_short_closure_is_ignored(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.2, 150) == []  # < 200 ms: natural blink

    def test_closure_at_min_blink_is_dot(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.2, 200) == [DOT]

    def test_closure_just_under_dot_max_is_dot(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.2, 500) == [DOT]

    def test_closure_over_dot_max_is_dash(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        assert tracker.update(0.2, 501) == [DASH]

    def test_multiple_symbols_in_sequence(self, tracker: BlinkTracker):
        events: list[str] = []
        events += tracker.update(0.9, 0)
        events += tracker.update(0.2, 300)  # dot
        events += tracker.update(0.9, 600)
        events += tracker.update(0.2, 900)  # dot
        events += tracker.update(0.9, 1200)
        events += tracker.update(0.2, 2200)  # dash
        assert events == [DOT, DOT, DASH]

    def test_first_frame_open_then_blink(self, tracker: BlinkTracker):
        tracker.update(0.0, 0)  # first frame definitely open: no closure to end
        tracker.update(0.9, 100)
        assert tracker.update(0.0, 400) == [DOT]


class TestGaps:
    def test_letter_gap_at_threshold(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.2, 300)  # dot at 300
        assert LETTER_GAP in tracker.update(0.2, 1800)  # 1500 after 300

    def test_letter_and_word_gap_together(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.2, 300)  # dot at 300
        assert tracker.update(0.2, 3400) == [LETTER_GAP, WORD_GAP]  # 3100 after 300

    def test_gap_timer_resets_after_each_symbol(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.2, 300)  # dot at 300
        tracker.update(0.2, 1600)  # 1300 after 300: no gap yet
        tracker.update(0.9, 1700)
        tracker.update(0.2, 2000)  # dot at 2000: timer restarts
        assert tracker.update(0.2, 2100) == []

    def test_gaps_fire_while_eyes_closed(self, tracker: BlinkTracker):
        # Letter gap must fire even if the user keeps blinking into the next
        # letter, or a symbol would never be finalised while eyes keep closing.
        tracker.update(0.9, 0)
        tracker.update(0.2, 300)  # dot at 300
        assert LETTER_GAP in tracker.update(0.9, 2000)  # closed, 1700 after dot

    def test_no_gaps_before_any_symbol(self, tracker: BlinkTracker):
        assert tracker.update(0.1, 0) == []
        assert tracker.update(0.1, 10_000) == []


class TestNoFace:
    def test_none_score_skips_eye_state_but_advances_gaps(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.2, 300)  # dot at 300
        # Face lost mid-letter: the letter still finalises.
        assert tracker.update(None, 4000) == [LETTER_GAP, WORD_GAP]

    def test_none_score_then_closed_starts_new_closure(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.2, 300)  # dot at 300
        tracker.update(None, 4000)
        tracker.update(0.9, 4200)
        assert tracker.update(0.2, 5000) == [DASH]


class TestValidation:
    def test_thresholds_out_of_order_raise(self):
        with pytest.raises(ValueError):
            BlinkTracker(0.4, 0.6, 200, 500, 1500, 3000)  # open > close

    def test_min_blink_above_dot_max_raises(self):
        with pytest.raises(ValueError):
            BlinkTracker(0.6, 0.4, 600, 500, 1500, 3000)

    def test_letter_gap_above_word_gap_raises(self):
        with pytest.raises(ValueError):
            BlinkTracker(0.6, 0.4, 200, 500, 3000, 1500)


class TestReset:
    def test_reset_clears_state(self, tracker: BlinkTracker):
        tracker.update(0.9, 0)
        tracker.update(0.2, 300)  # dot at 300
        tracker.reset()
        assert tracker.update(0.1, 10_000) == []
        tracker.update(0.9, 10_100)
        # A fresh 400 ms closure after reset is an ordinary dot.
        assert tracker.update(0.1, 10_500) == [DOT]
