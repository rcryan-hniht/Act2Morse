import './style.css';
import {
  MORSE_TABLE,
  classifyBlink,
  decodeMorseSequence,
  DEFAULT_THRESHOLDS,
} from './morse.ts';
import { morseAudio } from './audio.ts';
import { cameraController, type CameraMetrics, type BlinkEvent } from './camera.ts';
import { wsBridge, type ConnectionStatus, type BackendResponse, type BackendMessage } from './ws.ts';

// State management
let currentMorseBuffer: string = '';
let decodedText: string = '';
let letterTimeoutId: number | null = null;
let wordTimeoutId: number | null = null;
let isAudioEnabled: boolean = true;
let isCameraActive: boolean = false;

// DOM Elements
const morseSymbolsDisplay = document.getElementById('morseSymbolsDisplay') as HTMLDivElement;
const decodedTextDisplay = document.getElementById('decodedTextDisplay') as HTMLSpanElement;

const btnSimDot = document.getElementById('btnSimDot') as HTMLButtonElement;
const btnSimDash = document.getElementById('btnSimDash') as HTMLButtonElement;
const btnSimSpace = document.getElementById('btnSimSpace') as HTMLButtonElement;
const btnSimClear = document.getElementById('btnSimClear') as HTMLButtonElement;
const btnTransmit = document.getElementById('btnTransmit') as HTMLButtonElement;
const btnSoundToggle = document.getElementById('btnSoundToggle') as HTMLButtonElement;


// Navbar DOM
const navBtnAction = document.getElementById('navBtnAction') as HTMLButtonElement | null;
const btnModeFinger = document.getElementById('btnModeFinger') as HTMLButtonElement | null;
const btnModeBlink = document.getElementById('btnModeBlink') as HTMLButtonElement | null;
const navLinkDocs = document.getElementById('navLinkDocs') as HTMLElement | null;
const morseFlipContainer = document.getElementById('morseFlipContainer') as HTMLElement | null;
const morseFlipInner = document.getElementById('morseFlipInner') as HTMLElement | null;
const btnFlipToDocs = document.getElementById('btnFlipToDocs') as HTMLButtonElement | null;
const btnFlipToChart = document.getElementById('btnFlipToChart') as HTMLButtonElement | null;

let currentMode: 'Fin2Morse' | 'Blink2Morse' = 'Fin2Morse';
let fingerTapStartTime = 0;

// Camera DOM
const btnToggleCamera = document.getElementById('btnToggleCamera') as HTMLButtonElement;
const btnToggleCamText = document.getElementById('btnToggleCamText') as HTMLSpanElement;
const webcamVideo = document.getElementById('webcamVideo') as HTMLVideoElement;
const webcamCanvas = document.getElementById('webcamCanvas') as HTMLCanvasElement;
const heroPortraitImg = document.getElementById('heroPortraitImg') as HTMLImageElement;
const camStatusDot = document.getElementById('camStatusDot') as HTMLSpanElement;
const camStatusText = document.getElementById('camStatusText') as HTMLSpanElement;

// Floating Bar DOM
const floatingBarStatus = document.getElementById('floatingBarStatus') as HTMLSpanElement;
const floatingBarSub = document.getElementById('floatingBarSub') as HTMLSpanElement;
const floatingIcon = document.getElementById('floatingIcon') as HTMLDivElement;

// Morse Chart DOM
const btnOpenAlphabet = document.getElementById('btnOpenAlphabet') as HTMLButtonElement;
const morseChartPanel = document.getElementById('morseChartPanel') as HTMLDivElement;
const morseChartGrid = document.getElementById('morseChartGrid') as HTMLDivElement;

// Toast DOM
const toastNotice = document.getElementById('toastNotice') as HTMLDivElement;
const toastMsg = document.getElementById('toastMsg') as HTMLSpanElement;

/**
 * Toast Helper
 */
function showToast(message: string, durationMs: number = 2400) {
  toastMsg.textContent = message;
  toastNotice.classList.add('show');
  setTimeout(() => {
    toastNotice.classList.remove('show');
  }, durationMs);
}

/**
 * Updates UI Displays for Morse symbols and Decoded letters
 */
function updateDisplay() {
  if (currentMorseBuffer.length === 0) {
    morseSymbolsDisplay.innerHTML = `<span class="morse-placeholder">Blink or tap buttons below to transmit...</span>`;
  } else {
    const formatted = currentMorseBuffer
      .split('')
      .map((sym) =>
        sym === '.'
          ? `<span class="morse-symbol-dot">•</span>`
          : `<span class="morse-symbol-dash">—</span>`
      )
      .join(' ');
    morseSymbolsDisplay.innerHTML = formatted;
  }

  decodedTextDisplay.textContent = decodedText.length > 0 ? decodedText : '—';
}

function pulseFloatingIcon() {
  floatingIcon.style.color = '#8B5CF6';
  setTimeout(() => {
    floatingIcon.style.color = '#111111';
  }, 220);
}

/**
 * Appends a Morse symbol (. or -) and resets auto-pause timer
 */
function appendSymbol(symbol: '.' | '-') {
  if (wsBridge.isConnected()) {
    wsBridge.sendSymbol(symbol);
    if (symbol === '.') {
      morseAudio.playDot();
      floatingBarStatus.textContent = `Blink: Dot (•)`;
    } else {
      morseAudio.playDash();
      floatingBarStatus.textContent = `Blink: Dash (—)`;
    }
    pulseFloatingIcon();
    return;
  }

  currentMorseBuffer += symbol;

  if (symbol === '.') {
    morseAudio.playDot();
    floatingBarStatus.textContent = `Blink: Dot (•)`;
    floatingBarSub.textContent = `Buffer: ${currentMorseBuffer} → Potential: ${decodeMorseSequence(currentMorseBuffer)}`;
  } else {
    morseAudio.playDash();
    floatingBarStatus.textContent = `Blink: Dash (—)`;
    floatingBarSub.textContent = `Buffer: ${currentMorseBuffer} → Potential: ${decodeMorseSequence(currentMorseBuffer)}`;
  }

  // Visual pulse on floating icon
  pulseFloatingIcon();

  updateDisplay();

  // Reset character finalize timer
  if (letterTimeoutId) clearTimeout(letterTimeoutId);
  if (wordTimeoutId) clearTimeout(wordTimeoutId);

  letterTimeoutId = window.setTimeout(() => {
    finalizeCharacter();
  }, DEFAULT_THRESHOLDS.letterPauseMs);
}

/**
 * Finalizes the current Morse sequence into a letter
 */
function finalizeCharacter() {
  if (currentMorseBuffer.length === 0) return;

  const char = decodeMorseSequence(currentMorseBuffer);
  if (char && char !== '?') {
    decodedText += char;
    morseAudio.playCharacterComplete();
    floatingBarStatus.textContent = `Decoded: "${char}"`;
    floatingBarSub.textContent = `Morse: ${currentMorseBuffer} → '${char}' registered`;
  } else {
    floatingBarStatus.textContent = `Unknown Morse: "${currentMorseBuffer}"`;
  }

  currentMorseBuffer = '';
  updateDisplay();

  // Set timer to add space between words if pause is long
  if (wordTimeoutId) clearTimeout(wordTimeoutId);
  wordTimeoutId = window.setTimeout(() => {
    if (decodedText.length > 0 && !decodedText.endsWith(' ')) {
      decodedText += ' ';
      morseAudio.playWordSpace();
      updateDisplay();
      floatingBarSub.textContent = `Word pause detected (Space added)`;
    }
  }, DEFAULT_THRESHOLDS.wordPauseMs);
}

/**
 * Camera Toggle Handler
 */
async function toggleCamera() {
  if (!isCameraActive) {
    btnToggleCamera.disabled = true;
    btnToggleCamText.textContent = 'Starting...';

    const success = await cameraController.start(webcamVideo, webcamCanvas);
    btnToggleCamera.disabled = false;

    if (success) {
      isCameraActive = true;
      heroPortraitImg.style.display = 'none';
      webcamVideo.style.display = 'block';
      webcamCanvas.style.display = 'block';

      btnToggleCamText.textContent = 'Stop Camera';
      if (navBtnAction) navBtnAction.textContent = 'Stop Camera';
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = 'LIVE TRACKING';
      showToast('Live eye camera activated');
    } else {
      btnToggleCamText.textContent = 'Launch Camera';
      if (navBtnAction) navBtnAction.textContent = 'Start Camera';
      showToast('Camera permission denied or camera unavailable');
    }
  } else {
    cameraController.stop();
    isCameraActive = false;
    webcamVideo.style.display = 'none';
    webcamCanvas.style.display = 'none';
    heroPortraitImg.style.display = 'block';

    btnToggleCamText.textContent = 'Launch Camera';
    if (navBtnAction) navBtnAction.textContent = 'Start Camera';
    camStatusDot.style.background = '#6B7280';
    camStatusText.textContent = 'WEBCAM OFF';
    showToast('Camera stopped');
  }
}

/**
 * Handle Blink events from camera
 */
function handleBlinkEvent(event: BlinkEvent) {
  const result = classifyBlink(event.durationMs);

  if (result.symbol === '.') {
    appendSymbol('.');
    flashCameraStatus('DOT');
  } else if (result.symbol === '-') {
    appendSymbol('-');
    flashCameraStatus('DASH');
  } else if (result.symbol === 'noise') {
    floatingBarSub.textContent = `Noise flutter (${event.durationMs}ms) ignored`;
  }
}

function flashCameraStatus(label: string) {
  camStatusDot.classList.add('blinking');
  camStatusText.textContent = `BLINK: ${label}`;
  setTimeout(() => {
    camStatusDot.classList.remove('blinking');
    if (isCameraActive) {
      camStatusText.textContent = 'LIVE TRACKING';
    }
  }, 350);
}

/**
 * Camera metrics updates
 */
cameraController.onMetrics((metrics: CameraMetrics) => {
  if (isCameraActive && !wsBridge.isConnected()) {
    camStatusText.textContent = `${metrics.isBlinking ? 'BLINK' : 'TRACKING'} • ${metrics.fps} FPS`;
  }
});

// Stream captured frames to AI backend via WebSocket with backpressure
cameraController.onFrame((canvas: HTMLCanvasElement) => {
  if (isCameraActive && wsBridge.isConnected()) {
    wsBridge.sendFrame(canvas);
  }
});

// Handle blinks detected by client vision fallback (when backend is offline)
cameraController.onBlink((event: BlinkEvent) => {
  if (!wsBridge.isConnected()) {
    handleBlinkEvent(event);
  }
});

/**
 * WebSocket backend listeners
 */
wsBridge.onStatusChange((status: ConnectionStatus) => {
  if (status === 'connected') {
    showToast('🟢 Connected to AI Backend (MediaPipe)');
    cameraController.setBackendState(true, null, false, false);
    if (isCameraActive) {
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = 'AI BACKEND READY';
    }
  } else {
    cameraController.setBackendState(false, null, false, false);
    if (isCameraActive) {
      camStatusDot.style.background = '#6B7280';
      camStatusText.textContent = 'LOCAL VISION (AI Offline)';
    }
  }
});

wsBridge.onResponse((res: BackendResponse) => {
  cameraController.setBackendState(true, res.score, res.face, res.eyes_closed);

  if (!isCameraActive) return;

  // Visual status indicators
  if (!res.face) {
    camStatusDot.style.background = '#F59E0B';
    camStatusText.textContent = 'LOOK AT CAMERA (NO FACE)';
    floatingBarSub.textContent = 'Position your face in front of the camera';
  } else if (res.eyes_closed) {
    camStatusDot.classList.add('blinking');
    camStatusDot.style.background = '#C9B8FF';
    camStatusText.textContent = `EYES CLOSED • ${(res.score ?? 0).toFixed(2)}`;
  } else {
    camStatusDot.classList.remove('blinking');
    camStatusDot.style.background = '#10B981';
    camStatusText.textContent = `AI TRACKING • ${(res.score ?? 0).toFixed(2)}`;
  }

  // Handle server-side Morse events (DO NOT RETURN EARLY ON !res.face)
  if (res.events && res.events.length > 0) {
    for (const evt of res.events) {
      if (evt === 'dot') {
        morseAudio.playDot();
        flashCameraStatus('DOT (•)');
        floatingBarStatus.textContent = 'Blink: Dot (•)';
        pulseFloatingIcon();
      } else if (evt === 'dash') {
        morseAudio.playDash();
        flashCameraStatus('DASH (—)');
        floatingBarStatus.textContent = 'Blink: Dash (—)';
        pulseFloatingIcon();
      } else if (evt === 'letter_gap') {
        morseAudio.playCharacterComplete();
        flashCameraStatus('LETTER DONE');
        floatingBarStatus.textContent = 'Letter completed';
      } else if (evt === 'word_gap') {
        morseAudio.playWordSpace();
        flashCameraStatus('WORD SPACE');
        floatingBarStatus.textContent = 'Word space added';
      }
    }
  }

  // Synchronize current symbols and decoded text from backend session
  currentMorseBuffer = res.symbols;
  decodedText = res.text;
  updateDisplay();

  if (currentMorseBuffer.length > 0) {
    floatingBarSub.textContent = `Buffer: ${currentMorseBuffer} → Potential: ${decodeMorseSequence(currentMorseBuffer)}`;
  } else if (decodedText.length > 0) {
    floatingBarSub.textContent = `Decoded: "${decodedText}"`;
  }
});

wsBridge.onMessage((msg: BackendMessage) => {
  if (msg.type === 'blink' && msg.duration) {
    cameraController.triggerSimulatedBlink(msg.duration);
  } else if (msg.type === 'morse' && msg.symbol) {
    appendSymbol(msg.symbol);
  }
});

// Try connecting to backend in background (gracefully fails if backend not yet running)
wsBridge.connect();

/**
 * Setup Event Listeners
 */
function setupEventListeners() {
  // Simulator buttons
  btnSimDot.addEventListener('click', () => appendSymbol('.'));
  btnSimDash.addEventListener('click', () => appendSymbol('-'));
  
  btnSimSpace.addEventListener('click', () => {
    if (wsBridge.isConnected()) {
      wsBridge.sendSpace();
      morseAudio.playWordSpace();
      showToast('Word space added');
      return;
    }
    finalizeCharacter();
    if (decodedText.length > 0 && !decodedText.endsWith(' ')) {
      decodedText += ' ';
      morseAudio.playWordSpace();
    }
    updateDisplay();
    showToast('Word space added');
  });

  btnSimClear.addEventListener('click', () => {
    currentMorseBuffer = '';
    decodedText = '';
    if (letterTimeoutId) clearTimeout(letterTimeoutId);
    if (wordTimeoutId) clearTimeout(wordTimeoutId);
    wsBridge.sendReset();
    updateDisplay();
    floatingBarStatus.textContent = 'Listening for blinks...';
    floatingBarSub.textContent = 'Short <380ms = Dot (•) | Long >380ms = Dash (—)';
    showToast('Buffer cleared');
  });

  // Up Arrow Transmit button
  btnTransmit.addEventListener('click', () => {
    finalizeCharacter();
    if (decodedText.trim().length > 0) {
      navigator.clipboard?.writeText(decodedText.trim()).catch(() => {});
      showToast(`Copied text: "${decodedText.trim()}"`);
    } else {
      showToast('No decoded text to transmit yet');
    }
  });

  // Sound toggle button
  btnSoundToggle.addEventListener('click', () => {
    isAudioEnabled = !isAudioEnabled;
    morseAudio.setEnabled(isAudioEnabled);
    btnSoundToggle.classList.toggle('active', isAudioEnabled);
    showToast(isAudioEnabled ? 'Morse audio sound ON' : 'Morse audio MUTED');
  });
  btnSoundToggle.classList.add('active');

  // Camera toggle buttons
  btnToggleCamera.addEventListener('click', toggleCamera);
  if (navBtnAction) {
    navBtnAction.addEventListener('click', toggleCamera);
  }

  // Mode Switcher: Fin2Morse vs Blink2Morse
  btnModeFinger?.addEventListener('click', () => {
    currentMode = 'Fin2Morse';
    btnModeFinger?.classList.add('active');
    btnModeBlink?.classList.remove('active');
    floatingBarStatus.textContent = 'Fin2Morse Active (Finger Tap)';
    floatingBarSub.textContent = 'Tap Spacebar or screen (Hold <380ms = • Dot, Hold >380ms = — Dash)';
    showToast('🖐️ Switched to Fin2Morse (Finger / Tactile Mode)');
  });

  btnModeBlink?.addEventListener('click', () => {
    currentMode = 'Blink2Morse';
    btnModeBlink.classList.add('active');
    btnModeFinger?.classList.remove('active');
    floatingBarStatus.textContent = 'Blink2Morse Active';
    floatingBarSub.textContent = 'Tracking eye movements (Short=Dot, Long=Dash)';
    showToast('👁️ Switched to Blink2Morse (Eye Tracking Mode)');
  });

  // 3D Flip Card: Docs (Fin & Blink mode cards) vs Morse Alphabet Chart
  const flipToDocs = () => {
    morseFlipInner?.classList.add('is-flipped');
    navLinkDocs?.classList.add('active');
    morseFlipContainer?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  const flipToChart = () => {
    morseFlipInner?.classList.remove('is-flipped');
    navLinkDocs?.classList.remove('active');
    morseFlipContainer?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  const toggleFlip = () => {
    if (morseFlipInner?.classList.contains('is-flipped')) {
      flipToChart();
    } else {
      flipToDocs();
    }
  };

  navLinkDocs?.addEventListener('click', (e) => {
    e.preventDefault();
    toggleFlip();
  });

  btnFlipToDocs?.addEventListener('click', flipToDocs);
  btnFlipToChart?.addEventListener('click', flipToChart);

  // Morse Chart highlight / focus from console button
  btnOpenAlphabet.addEventListener('click', () => {
    if (morseFlipInner?.classList.contains('is-flipped')) {
      flipToChart();
    }
    morseChartPanel?.classList.add('pulse-highlight');
    setTimeout(() => morseChartPanel?.classList.remove('pulse-highlight'), 600);
    showToast('Morse Chart is beside the camera');
  });

  // Keyboard accessibility and Fin2Morse tactile input
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && morseFlipInner?.classList.contains('is-flipped')) {
      flipToChart();
      return;
    }
    if (e.code === 'KeyD' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      appendSymbol('.');
    } else if (e.code === 'KeyF' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      appendSymbol('-');
    } else if (e.code === 'Space' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      e.preventDefault();
      if (currentMode === 'Fin2Morse') {
        if (!e.repeat && fingerTapStartTime === 0) {
          fingerTapStartTime = performance.now();
          floatingBarStatus.textContent = 'Fin2Morse: Pressing...';
          pulseFloatingIcon();
        }
      } else {
        finalizeCharacter();
      }
    }
  });

  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && currentMode === 'Fin2Morse' && fingerTapStartTime > 0) {
      e.preventDefault();
      const duration = performance.now() - fingerTapStartTime;
      fingerTapStartTime = 0;
      if (duration < 380) {
        appendSymbol('.');
        flashCameraStatus('DOT (•)');
      } else {
        appendSymbol('-');
        flashCameraStatus('DASH (—)');
      }
    }
  });
}

/**
 * Builds the Morse Code Chart Grid in modal
 */
function buildAlphabetGrid() {
  morseChartGrid.innerHTML = '';
  Object.entries(MORSE_TABLE).forEach(([char, code]) => {
    const item = document.createElement('div');
    item.className = 'morse-grid-item';
    item.innerHTML = `
      <span class="grid-char">${char}</span>
      <span class="grid-code">${code}</span>
    `;

    item.addEventListener('click', () => {
      // Play audio sequence for clicked letter
      playMorseSequence(code);
      showToast(`${char}: ${code}`);
    });

    morseChartGrid.appendChild(item);
  });
}

/**
 * Plays a sequence of dots and dashes for a letter
 */
async function playMorseSequence(sequence: string) {
  for (const sym of sequence) {
    if (sym === '.') {
      morseAudio.playDot();
      await new Promise((r) => setTimeout(r, 160));
    } else if (sym === '-') {
      morseAudio.playDash();
      await new Promise((r) => setTimeout(r, 340));
    }
  }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  buildAlphabetGrid();
  updateDisplay();
});
