import './style.css';
import {
  MORSE_TABLE,
  classifyBlink,
  decodeMorseSequence,
  DEFAULT_THRESHOLDS,
} from './morse.ts';
import { morseAudio } from './audio.ts';
import { cameraController, type CameraMetrics, type BlinkEvent } from './camera.ts';
import { wsBridge, type ConnectionStatus, type BackendMessage } from './ws.ts';

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

const tabDecoder = document.getElementById('tabDecoder') as HTMLButtonElement;
const tabAuth = document.getElementById('tabAuth') as HTMLButtonElement;
const viewDecoder = document.getElementById('viewDecoder') as HTMLDivElement;
const viewAuth = document.getElementById('viewAuth') as HTMLDivElement;

const btnGoogleAuth = document.getElementById('btnGoogleAuth') as HTMLButtonElement;
const btnEmailAuth = document.getElementById('btnEmailAuth') as HTMLButtonElement;

// Camera DOM
const btnToggleCamera = document.getElementById('btnToggleCamera') as HTMLButtonElement;
const btnToggleCamText = document.getElementById('btnToggleCamText') as HTMLSpanElement;
const navBtnAction = document.getElementById('navBtnAction') as HTMLButtonElement;
const webcamVideo = document.getElementById('webcamVideo') as HTMLVideoElement;
const webcamCanvas = document.getElementById('webcamCanvas') as HTMLCanvasElement;
const heroPortraitImg = document.getElementById('heroPortraitImg') as HTMLImageElement;
const camStatusDot = document.getElementById('camStatusDot') as HTMLSpanElement;
const camStatusText = document.getElementById('camStatusText') as HTMLSpanElement;

// Floating Bar DOM
const floatingBarStatus = document.getElementById('floatingBarStatus') as HTMLSpanElement;
const floatingBarSub = document.getElementById('floatingBarSub') as HTMLSpanElement;
const floatingIcon = document.getElementById('floatingIcon') as HTMLDivElement;

// Modal DOM
const alphabetModal = document.getElementById('alphabetModal') as HTMLDivElement;
const modalCloseBtn = document.getElementById('modalCloseBtn') as HTMLButtonElement;
const btnOpenAlphabet = document.getElementById('btnOpenAlphabet') as HTMLButtonElement;
const navLinkAlphabet = document.getElementById('navLinkAlphabet') as HTMLAnchorElement;
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

/**
 * Appends a Morse symbol (. or -) and resets auto-pause timer
 */
function appendSymbol(symbol: '.' | '-') {
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
  floatingIcon.style.color = '#8B5CF6';
  setTimeout(() => {
    floatingIcon.style.color = '#111111';
  }, 220);

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
      navBtnAction.textContent = 'Stop Camera';
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = 'LIVE TRACKING';
      showToast('Live eye camera activated');
    } else {
      btnToggleCamText.textContent = 'Launch Camera';
      showToast('Camera permission denied or camera unavailable');
    }
  } else {
    cameraController.stop();
    isCameraActive = false;
    webcamVideo.style.display = 'none';
    webcamCanvas.style.display = 'none';
    heroPortraitImg.style.display = 'block';

    btnToggleCamText.textContent = 'Launch Camera';
    navBtnAction.textContent = 'Start Camera';
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
  if (isCameraActive) {
    camStatusText.textContent = `${metrics.isBlinking ? 'BLINK' : 'TRACKING'} • ${metrics.fps} FPS`;
  }
});

cameraController.onBlink(handleBlinkEvent);

/**
 * WebSocket backend listeners
 */
wsBridge.onStatusChange((status: ConnectionStatus) => {
  if (status === 'connected') {
    showToast('Connected to Python backend (localhost:8000)');
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
  navBtnAction.addEventListener('click', toggleCamera);

  // Tab switching: Live Morse Console vs Auth
  tabDecoder.addEventListener('click', () => {
    tabDecoder.classList.add('active');
    tabAuth.classList.remove('active');
    viewDecoder.style.display = 'flex';
    viewAuth.style.display = 'none';
  });

  tabAuth.addEventListener('click', () => {
    tabAuth.classList.add('active');
    tabDecoder.classList.remove('active');
    viewDecoder.style.display = 'none';
    viewAuth.style.display = 'flex';
  });

  // Nav Login button toggles to Auth tab
  document.getElementById('navBtnLogin')?.addEventListener('click', () => {
    tabAuth.click();
    showToast('Switched to Sign In mode');
  });

  // Auth Button Actions
  btnGoogleAuth.addEventListener('click', () => {
    showToast('Google authentication initialized');
  });

  btnEmailAuth.addEventListener('click', () => {
    showToast('Email sign-in initialized');
  });

  // Alphabet modal
  const openAlphabetModal = () => {
    alphabetModal.classList.add('open');
  };
  const closeAlphabetModal = () => {
    alphabetModal.classList.remove('open');
  };

  btnOpenAlphabet.addEventListener('click', openAlphabetModal);
  navLinkAlphabet.addEventListener('click', (e) => {
    e.preventDefault();
    openAlphabetModal();
  });
  modalCloseBtn.addEventListener('click', closeAlphabetModal);
  alphabetModal.addEventListener('click', (e) => {
    if (e.target === alphabetModal) closeAlphabetModal();
  });

  // Keyboard accessibility shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && alphabetModal.classList.contains('open')) {
      closeAlphabetModal();
    } else if (e.code === 'KeyD' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      appendSymbol('.');
    } else if (e.code === 'KeyF' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      appendSymbol('-');
    } else if (e.code === 'Space' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      e.preventDefault();
      finalizeCharacter();
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
