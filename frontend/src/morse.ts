/**
 * Morse Code Dictionary and Translation Engine for Blink2Morse
 */

export const MORSE_TABLE: Record<string, string> = {
  // Letters
  'A': '.-',
  'B': '-...',
  'C': '-.-.',
  'D': '-..',
  'E': '.',
  'F': '..-.',
  'G': '--.',
  'H': '....',
  'I': '..',
  'J': '.---',
  'K': '-.-',
  'L': '.-..',
  'M': '--',
  'N': '-.',
  'O': '---',
  'P': '.--.',
  'Q': '--.-',
  'R': '.-.',
  'S': '...',
  'T': '-',
  'U': '..-',
  'V': '...-',
  'W': '.--',
  'X': '-..-',
  'Y': '-.--',
  'Z': '--..',

  // Numbers
  '0': '-----',
  '1': '.----',
  '2': '..---',
  '3': '...--',
  '4': '....-',
  '5': '.....',
  '6': '-....',
  '7': '--...',
  '8': '---..',
  '9': '----.',

  // Symbols
  '.': '.-.-.-',
  ',': '--..--',
  '?': '..--..',
  '!': '-.-.--',
  '/': '-..-.',
  '=': '-...-',
  '+': '.-.-.',
  '-': '-....-',
};

// Reverse lookup table: Morse -> Character
export const REVERSE_MORSE: Record<string, string> = Object.entries(MORSE_TABLE).reduce(
  (acc, [char, morse]) => {
    acc[morse] = char;
    return acc;
  },
  {} as Record<string, string>
);

export interface BlinkThresholds {
  minNoiseMs: number;     // Below this is considered noise/flutter (< 120ms)
  shortDotMaxMs: number;  // 120ms - 400ms is Dot (•)
  longDashMaxMs: number;  // 400ms - 1200ms is Dash (—)
  letterPauseMs: number;  // Open eye > 800ms completes a letter
  wordPauseMs: number;    // Open eye > 2000ms appends a word space
}

export const DEFAULT_THRESHOLDS: BlinkThresholds = {
  minNoiseMs: 100,
  shortDotMaxMs: 380,
  longDashMaxMs: 1200,
  letterPauseMs: 850,
  wordPauseMs: 2200,
};

export type BlinkSymbol = '.' | '-' | 'noise' | 'invalid';

/**
 * Classifies a blink duration into a Morse symbol.
 */
export function classifyBlink(
  durationMs: number,
  thresholds: BlinkThresholds = DEFAULT_THRESHOLDS
): { symbol: BlinkSymbol; display: string } {
  if (durationMs < thresholds.minNoiseMs) {
    return { symbol: 'noise', display: 'Noise' };
  }
  if (durationMs <= thresholds.shortDotMaxMs) {
    return { symbol: '.', display: '• Dot' };
  }
  if (durationMs <= thresholds.longDashMaxMs) {
    return { symbol: '-', display: '— Dash' };
  }
  return { symbol: 'invalid', display: 'Too Long' };
}

/**
 * Translates a single Morse sequence (e.g., '...') into a character (e.g., 'S').
 */
export function decodeMorseSequence(sequence: string): string {
  const clean = sequence.trim();
  return REVERSE_MORSE[clean] || (clean.length > 0 ? '?' : '');
}

/**
 * Decodes a full morse string with spaces between letters and '/' or double-space between words.
 * E.g., "... --- ... / .--. .-. ---" -> "SOS PRO"
 */
export function decodeFullMorse(fullMorse: string): string {
  const words = fullMorse.trim().split(/\s{2,}|\//);
  return words
    .map(word => {
      const letters = word.trim().split(/\s+/);
      return letters.map(code => REVERSE_MORSE[code] || (code ? '?' : '')).join('');
    })
    .join(' ');
}
