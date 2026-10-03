"""Tests for the Morse table and decoder (app.morse)."""

from app.morse import MORSE_TABLE, MorseDecoder, decode_sequence


class TestDecodeSequence:
    def test_letters(self):
        assert decode_sequence(".-") == "A"
        assert decode_sequence("...") == "S"
        assert decode_sequence("-----") == "0"

    def test_empty_and_unknown(self):
        assert decode_sequence("") == ""
        assert decode_sequence("   ") == ""
        assert decode_sequence(".-.-.-.-") == "?"  # not a valid sequence

    def test_table_covers_alphabet_and_digits(self):
        assert all(MORSE_TABLE[c] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert all(MORSE_TABLE[c] for c in "0123456789")


class TestMorseDecoder:
    def test_single_letter(self):
        d = MorseDecoder()
        d.add_symbol(".")
        d.add_symbol(".")
        d.add_symbol(".")
        assert d.symbols == "..."
        assert d.finish_letter() == "S"
        assert d.text == "S"
        assert d.symbols == ""

    def test_finish_letter_empty_is_none(self):
        d = MorseDecoder()
        assert d.finish_letter() is None

    def test_unknown_sequence_appends_question_mark(self):
        d = MorseDecoder()
        for _ in range(8):
            d.add_symbol(".")
        assert d.finish_letter() == "?"
        assert d.text == "?"

    def test_word_gap_appends_space_once(self):
        d = MorseDecoder()
        d.add_symbol("-")
        d.finish_letter()
        d.add_word_gap()
        d.add_word_gap()
        assert d.text == "T "

    def test_word_gap_after_unknown_letter(self):
        d = MorseDecoder()
        for _ in range(7):  # 7 dots: not a valid Morse sequence
            d.add_symbol(".")
        d.finish_letter()
        d.add_word_gap()
        assert d.text == "? "

    def test_add_symbol_rejects_garbage(self):
        d = MorseDecoder()
        try:
            d.add_symbol("x")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    def test_reset(self):
        d = MorseDecoder()
        d.add_symbol(".")
        d.add_symbol("-")
        d.finish_letter()
        d.add_symbol("-")
        d.reset()
        assert d.symbols == ""
        assert d.text == ""

    def test_decodes_sos(self):
        d = MorseDecoder()
        for seq in ("...", "---", "..."):
            for s in seq:
                d.add_symbol(s)
            d.finish_letter()
        assert d.text == "SOS"

    def test_backspace(self):
        d = MorseDecoder()
        assert d.backspace() is None

        # Backspace on symbol in progress
        d.add_symbol(".")
        d.add_symbol("-")
        assert d.symbols == ".-"
        assert d.backspace() == "-"
        assert d.symbols == "."

        # Backspace to empty symbols
        assert d.backspace() == "."
        assert d.symbols == ""

        # Backspace on decoded text
        d.add_symbol(".")
        d.finish_letter()  # text = "E"
        d.add_symbol("-")
        d.finish_letter()  # text = "ET"
        assert d.text == "ET"
        assert d.backspace() == "T"
        assert d.text == "E"
        assert d.backspace() == "E"
        assert d.text == ""
        assert d.backspace() is None
