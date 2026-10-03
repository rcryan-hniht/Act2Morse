"""Morse code table and decoder.

Pure module: no FastAPI, OpenCV or MediaPipe imports so it can be unit
tested in isolation.
"""

from __future__ import annotations

# International Morse code. Shared sequences map to the first character,
# e.g. both the digit 0 and the plus sign are ``-----``.
MORSE_TABLE: dict[str, str] = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "'": ".----.",
    "@": ".--.-.",
}

# Reverse lookup: morse sequence -> character. Shared sequences resolve to
# the first entry in MORSE_TABLE (deterministic since Python 3.7 dicts).
REVERSE_MORSE: dict[str, str] = {code: char for char, code in MORSE_TABLE.items()}

UNKNOWN_CHAR = "?"


def decode_sequence(symbols: str) -> str:
    """Decode one letter's Morse symbols (e.g. ``"...-"`` -> ``"V"``).

    Unknown or empty sequences decode to ``"?"`` and ``""`` respectively.
    """
    symbols = symbols.strip()
    if not symbols:
        return ""
    return REVERSE_MORSE.get(symbols, UNKNOWN_CHAR)


class MorseDecoder:
    """Decodes Morse symbol sequences (one letter at a time) into text.

    Pure module companion to :mod:`app.blink`: consumes ``.``/``-`` symbols
    and letter/word gap events, keeps the symbol buffer of the letter in
    progress, and appends decoded characters to the running text.
    """

    def __init__(self) -> None:
        self.symbols: str = ""
        self.text: str = ""

    def add_symbol(self, symbol: str) -> None:
        """Append a ``.`` or ``-`` to the letter in progress."""
        if symbol not in (".", "-"):
            raise ValueError(f"invalid Morse symbol: {symbol!r}")
        self.symbols += symbol

    def finish_letter(self) -> str | None:
        """Close the current letter and append it to the text.

        Returns the decoded character, or ``None`` when there was no letter
        in progress. Unknown sequences append ``?`` to the text.
        """
        if not self.symbols:
            return None
        char = decode_sequence(self.symbols)
        self.text += char
        self.symbols = ""
        return char

    def add_word_gap(self) -> None:
        """Append a space after a word gap (unless one is already there)."""
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self) -> str | None:
        """Remove last Morse symbol in progress, or last decoded character from text.

        Returns the removed symbol/character, or None if both are empty.
        """
        if self.symbols:
            deleted = self.symbols[-1]
            self.symbols = self.symbols[:-1]
            return deleted
        if self.text:
            deleted = self.text[-1]
            self.text = self.text[:-1]
            return deleted
        return None

    def reset(self) -> None:
        """Clear symbols of the letter in progress and all decoded text."""
        self.symbols = ""
        self.text = ""
