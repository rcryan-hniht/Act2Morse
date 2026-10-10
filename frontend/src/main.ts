import './style.css';
import {
  MORSE_TABLE,
  classifyBlink,
  classifyFingerTap,
  decodeMorseSequence,
  DEFAULT_THRESHOLDS,
} from './morse.ts';
import { morseAudio } from './audio.ts';
import { cameraController, type CameraMetrics, type BlinkEvent, type FingerTapEvent, type GestureActionEvent } from './camera.ts';
import { wsBridge, type ConnectionStatus, type BackendResponse, type BackendMessage } from './ws.ts';

// State management
let currentMorseBuffer: string = '';
let decodedText: string = '';
let letterTimeoutId: number | null = null;
let wordTimeoutId: number | null = null;
let isCameraActive: boolean = false;
let isFingerHolding: boolean = false;
let holdAnimFrameId: number | null = null;
let currentMode: 'Fin2Morse' | 'Blink2Morse' = 'Fin2Morse';
let fingerTapStartTime = 0;

/**
 * Returns the pause duration required before auto-finalizing a character into text.
 * Fin2Morse uses a comfortable 1600ms gap so users can easily tap multi-dash numbers (e.g. '0' = '-----').
 */
function getLetterPauseMs(): number {
  return currentMode === 'Fin2Morse'
    ? DEFAULT_THRESHOLDS.finLetterPauseMs
    : DEFAULT_THRESHOLDS.letterPauseMs;
}

/**
 * Cancels pending letter/word finalization timers while user is actively typing or holding a tap.
 */
function cancelPendingFinalizeTimers() {
  if (letterTimeoutId !== null) {
    clearTimeout(letterTimeoutId);
    letterTimeoutId = null;
  }
  if (wordTimeoutId !== null) {
    clearTimeout(wordTimeoutId);
    wordTimeoutId = null;
  }
}

// DOM Elements
const decodedTextDisplay = document.getElementById('decodedTextDisplay') as HTMLSpanElement;

const btnTransmit = document.getElementById('btnTransmit') as HTMLButtonElement;

// Tactile Tap Pad DOM
const finTapZone = document.getElementById('finTapZone') as HTMLDivElement | null;
const finTapStatus = document.getElementById('finTapStatus') as HTMLSpanElement | null;
const finTapSub = document.getElementById('finTapSub') as HTMLSpanElement | null;
const finHoldProgress = document.getElementById('finHoldProgress') as HTMLDivElement | null;
const finTapIndicator = document.getElementById('finTapIndicator') as HTMLSpanElement | null;
const finTapBadge = document.getElementById('finTapBadge') as HTMLSpanElement | null;
const portraitBox = document.getElementById('portraitBox') as HTMLDivElement | null;

// Navbar DOM
const btnModeFinger = document.getElementById('btnModeFinger') as HTMLButtonElement | null;
const btnModeBlink = document.getElementById('btnModeBlink') as HTMLButtonElement | null;
const navLinkDocs = document.getElementById('navLinkDocs') as HTMLElement | null;
const morseFlipContainer = document.getElementById('morseFlipContainer') as HTMLElement | null;
const morseFlipInner = document.getElementById('morseFlipInner') as HTMLElement | null;
const btnFlipToDocs = document.getElementById('btnFlipToDocs') as HTMLButtonElement | null;
const btnFlipToChart = document.getElementById('btnFlipToChart') as HTMLButtonElement | null;

// Welcome Modal DOM (Bảng nổi giới thiệu)
const welcomeOverlay = document.getElementById('welcomeOverlay') as HTMLDivElement | null;
const btnWelcomeClose = document.getElementById('btnWelcomeClose') as HTMLButtonElement | null;
const btnWelcomeStart = document.getElementById('btnWelcomeStart') as HTMLButtonElement | null;
const btnWelcomeDocs = document.getElementById('btnWelcomeDocs') as HTMLButtonElement | null;
const welcomeBackdrop = document.getElementById('welcomeBackdrop') as HTMLDivElement | null;
const navLinkAbout = document.getElementById('navLinkAbout') as HTMLButtonElement | null;

// Camera DOM
const btnToggleCamera = document.getElementById('btnToggleCamera') as HTMLButtonElement;
const btnToggleCamText = document.getElementById('btnToggleCamText') as HTMLSpanElement;
const webcamVideo = document.getElementById('webcamVideo') as HTMLVideoElement;
const webcamCanvas = document.getElementById('webcamCanvas') as HTMLCanvasElement;
const heroPortraitImg = document.getElementById('heroPortraitImg') as HTMLElement | null;
const camStatusDot = document.getElementById('camStatusDot') as HTMLSpanElement;
const camStatusText = document.getElementById('camStatusText') as HTMLSpanElement;

// Floating Bar DOM
const floatingBarStatus = document.getElementById('floatingBarStatus') as HTMLSpanElement;
const floatingIcon = document.getElementById('floatingIcon') as HTMLDivElement;

// Morse Chart DOM
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
 * Updates the decoded line and briefly displays each newly completed character
 */
const cameraCharacter = document.getElementById('cameraCharacter') as HTMLDivElement;
let displayedText = '';
let characterTimeoutId: number | undefined;

function updateDisplay() {
  decodedTextDisplay.textContent = decodedText || 'Waiting for Morse...';
  decodedTextDisplay.scrollLeft = decodedTextDisplay.scrollWidth;
  if (decodedText.startsWith(displayedText) && decodedText.length > displayedText.length) {
    const char = decodedText.slice(-1);
    if (char.trim()) {
      window.clearTimeout(characterTimeoutId);
      cameraCharacter.textContent = char;
      cameraCharacter.hidden = false;
      characterTimeoutId = window.setTimeout(() => {
        cameraCharacter.hidden = true;
      }, 1000);
    }
  } else if (decodedText !== displayedText) {
    window.clearTimeout(characterTimeoutId);
    cameraCharacter.hidden = true;
  }
  displayedText = decodedText;
}

function pulseFloatingIcon() {
  floatingIcon.style.color = '#8B5CF6';
  setTimeout(() => {
    floatingIcon.style.color = '#111111';
  }, 220);
}

/**
 * Appends a Morse symbol (. or -) and resets auto-pause timer.
 * Maintains client-side buffer integrity for tactile finger tapping and mirrors to backend.
 */
function appendSymbol(symbol: '.' | '-') {
  // Cancel previous finalize timer so multi-symbol sequences (e.g. '0' = '-----') don't get cut off
  cancelPendingFinalizeTimers();

  currentMorseBuffer += symbol;

  if (symbol === '.') {
    morseAudio.playDot();
    floatingBarStatus.textContent = currentMode === 'Fin2Morse' ? 'Fin: Dot (•)' : 'Blink: Dot (•)';
  } else {
    morseAudio.playDash();
    floatingBarStatus.textContent = currentMode === 'Fin2Morse' ? 'Fin: Dash (—)' : 'Blink: Dash (—)';
  }


  // Visual pulse on floating icon
  pulseFloatingIcon();
  updateDisplay();

  // If backend is connected, mirror symbol for remote session logging
  if (wsBridge.isConnected()) {
    wsBridge.sendSymbol(symbol);
  }

  // Reset character finalize timer
  letterTimeoutId = window.setTimeout(() => {
    finalizeCharacter();
  }, getLetterPauseMs());
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
  } else {
    floatingBarStatus.textContent = `Unknown Morse: "${currentMorseBuffer}"`;
  }

  if (wsBridge.isConnected()) {
    wsBridge.sendFinalize();
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
      if (wsBridge.isConnected()) {
        wsBridge.sendSpace();
      }
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
      if (heroPortraitImg) heroPortraitImg.style.display = 'none';
      webcamVideo.style.display = 'block';
      webcamCanvas.style.display = 'block';

      btnToggleCamText.textContent = 'Stop Camera';
      btnToggleCamera.classList.add('is-active');

      if (currentMode === 'Fin2Morse') {
        cameraController.setMode('Fin2Morse');
        wsBridge.sendMode('Fin2Morse');
        camStatusDot.style.background = '#10B981';
        camStatusText.textContent = 'HAND TRACKING ACTIVE';
        showToast('🖐️ Fin camera activated (Hand & Finger Tracking)');
      } else {
        cameraController.setMode('Blink2Morse');
        wsBridge.sendMode('Blink2Morse');
        camStatusDot.style.background = '#10B981';
        camStatusText.textContent = 'EYE TRACKING [BETA] ACTIVE';
        showToast('👁️ Blink camera activated (Beta - Eye Tracking)');
      }
    } else {
      btnToggleCamText.textContent = 'Launch Camera';
      btnToggleCamera.classList.remove('is-active');
      if (heroPortraitImg) heroPortraitImg.style.display = 'flex';
      showToast('Camera permission denied or camera unavailable');
    }
  } else {
    cameraController.stop();
    isCameraActive = false;
    webcamVideo.style.display = 'none';
    webcamCanvas.style.display = 'none';
    if (heroPortraitImg) heroPortraitImg.style.display = 'flex';

    btnToggleCamText.textContent = 'Launch Camera';
    btnToggleCamera.classList.remove('is-active');
    camStatusDot.style.background = '#6B7280';
    camStatusText.textContent = 'WEBCAM OFF';
    showToast('Camera stopped');
  }
}

/**
 * Handle Blink events from camera
 */
function handleBlinkEvent(event: BlinkEvent) {
  if (currentMode !== 'Blink2Morse') return;
  const result = classifyBlink(event.durationMs);

  if (result.symbol === '.') {
    appendSymbol('.');
    flashCameraStatus('DOT');
  } else if (result.symbol === '-') {
    appendSymbol('-');
    flashCameraStatus('DASH');
  } else if (result.symbol === 'noise') {
  }
}

function flashCameraStatus(label: string) {
  camStatusDot.classList.add('blinking');
  camStatusText.textContent = `${label}`;
  setTimeout(() => {
    camStatusDot.classList.remove('blinking');
    if (isCameraActive) {
      camStatusText.textContent = currentMode === 'Fin2Morse' ? 'HAND TRACKING' : 'LIVE TRACKING';
    }
  }, 350);
}

/**
 * Camera metrics updates (Dual-mode: Hand in Fin2Morse, Eye in Blink2Morse)
 */
cameraController.onMetrics((metrics: CameraMetrics) => {
  if (!isCameraActive) return;

  if (currentMode === 'Fin2Morse') {
    if (metrics.isHandDetected) {
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = metrics.isFingerTapping
        ? 'FINGER TAP DETECTED'
        : `HAND ACTIVE • ${metrics.fps} FPS`;
    } else {
      camStatusDot.style.background = '#F59E0B';
      camStatusText.textContent = cameraController.isModelReady()
        ? 'SHOW HAND TO CAMERA'
        : (cameraController.isModelLoadingActive() ? 'LOADING VISION MODEL...' : 'SHOW HAND TO CAMERA');
    }
  } else {
    if (!wsBridge.isConnected()) {
      camStatusText.textContent = `${metrics.isBlinking ? 'BLINK [BETA]' : 'EYE TRACKING [BETA]'} • ${metrics.fps} FPS`;
    }
  }
});

// Stream captured frames to AI backend via WebSocket in both modes
cameraController.onFrame((canvas: HTMLCanvasElement) => {
  if (isCameraActive && wsBridge.isConnected()) {
    wsBridge.sendFrame(canvas);
  }
});

// Handle camera finger pinch state (holding vs releasing) in Fin2Morse mode
cameraController.onPinchState((isPinching: boolean, elapsedMs: number) => {
  if (currentMode !== 'Fin2Morse') return;

  if (isPinching) {
    cancelPendingFinalizeTimers();
    const dashThreshold = DEFAULT_THRESHOLDS.shortDotMaxMs;
    const isDash = elapsedMs >= dashThreshold;

    finTapZone?.classList.add('is-pressing');
    if (isDash) {
      finTapZone?.classList.add('is-dash-ready');
      if (finTapBadge) finTapBadge.textContent = '— DASH READY (Release to register)';
      floatingBarStatus.textContent = 'Fin: Dash Ready (—) • Release to register';
    } else {
      finTapZone?.classList.remove('is-dash-ready');
      if (finTapBadge) finTapBadge.textContent = `HOLD FOR DASH: ${Math.round(elapsedMs)}ms / ${dashThreshold}ms`;
      floatingBarStatus.textContent = `Fin: Holding (${Math.round(elapsedMs)}ms)...`;
    }

    if (finHoldProgress) {
      const pct = Math.min(100, (elapsedMs / dashThreshold) * 100);
      finHoldProgress.style.width = `${pct}%`;
      finHoldProgress.style.background = isDash ? '#8B5CF6' : '#00F0FF';
    }
  } else {
    stopHoldAnimation();
    finTapZone?.classList.remove('is-pressing', 'is-dash-ready');
  }
});

// Handle camera finger taps in Fin2Morse mode
cameraController.onFingerTap((event: FingerTapEvent) => {
  if (currentMode === 'Fin2Morse') {
    appendSymbol(event.symbol);
    flashCameraStatus(event.symbol === '.' ? 'FINGER DOT (•)' : 'FINGER DASH (—)');
  }
});

// Handle blinks detected by client vision fallback (in Blink2Morse mode)
cameraController.onBlink((event: BlinkEvent) => {
  if (!wsBridge.isConnected() && currentMode === 'Blink2Morse') {
    handleBlinkEvent(event);
  }
});

// Handle camera hand gestures (Closed Fist -> Space, Right Thumb Left -> Backspace)
cameraController.onGestureAction((event: GestureActionEvent) => {
  if (currentMode !== 'Fin2Morse') return;

  if (event.action === 'space') {
    finalizeCharacter();
    if (decodedText.length > 0 && !decodedText.endsWith(' ')) {
      decodedText += ' ';
      morseAudio.playWordSpace();
      updateDisplay();
      floatingBarStatus.textContent = 'Gesture: Space (✊ Closed Fist)';
      flashCameraStatus('SPACE (✊)');
      showToast('␣ Space added (✊ Closed Fist)');
      if (wsBridge.isConnected()) {
        wsBridge.sendSpace();
      }
    }
  } else if (event.action === 'backspace') {
    cancelPendingFinalizeTimers();
    let deletedLabel = '';

    if (currentMorseBuffer.length > 0) {
      const removed = currentMorseBuffer.slice(-1);
      currentMorseBuffer = currentMorseBuffer.slice(0, -1);
      deletedLabel = `Morse '${removed === '.' ? '•' : '—'}'`;
      morseAudio.playBackspace();
      flashCameraStatus('BACKSPACE (⌫)');
      floatingBarStatus.textContent = `Gesture: Deleted ${deletedLabel} (👈 Thumb Left)`;
      showToast(`⌫ Deleted ${deletedLabel} (👈 Thumb Left)`);

      // Restart letter finalize timer if buffer still contains symbols
      if (currentMorseBuffer.length > 0) {
        letterTimeoutId = window.setTimeout(() => {
          finalizeCharacter();
        }, getLetterPauseMs());
      }
    } else if (decodedText.length > 0) {
      const removed = decodedText.slice(-1);
      decodedText = decodedText.slice(0, -1);
      deletedLabel = removed === ' ' ? 'Space' : `'${removed}'`;
      morseAudio.playBackspace();
      flashCameraStatus('BACKSPACE (⌫)');
      floatingBarStatus.textContent = `Gesture: Deleted ${deletedLabel} (👈 Thumb Left)`;
      showToast(`⌫ Deleted ${deletedLabel} (👈 Thumb Left)`);
    } else {
      showToast('Buffer already empty');
    }

    updateDisplay();

    if (wsBridge.isConnected()) {
      wsBridge.sendBackspace();
    }
  }
});

/**
 * WebSocket backend listeners
 */
wsBridge.onStatusChange((status: ConnectionStatus) => {
  if (status === 'connected') {
    wsBridge.sendMode(currentMode);
    showToast('🟢 Connected to AI Backend (MichalMlodawski / OpenCV)');
    cameraController.setBackendState(true, null, false, false);
    if (isCameraActive) {
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = currentMode === 'Fin2Morse'
        ? 'AI FIN ACTIVE (opencv/handpose)'
        : 'AI BLINK READY (MichalMlodawski)';
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
  if (!isCameraActive) return;

  if (currentMode === 'Fin2Morse') {
    // In Fin2Morse mode: Hand tracking from backend (opencv/handpose_estimation_mediapipe)
    if (res.hand_detected) {
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = res.is_finger_tapping
        ? `AI: FINGER TAP DETECTED • ${cameraController.getFps()} FPS`
        : `AI HAND ACTIVE (opencv) • ${Math.round((res.hand_confidence ?? 0.95) * 100)}% • ${cameraController.getFps()} FPS`;
    } else {
      camStatusDot.style.background = '#F59E0B';
      camStatusText.textContent = `SHOW HAND TO CAMERA • ${cameraController.getFps()} FPS`;
    }
  } else {
    // In Blink2Morse mode: Eye tracking from backend (MichalMlodawski/open-closed-eye-classification-mobilev2)
    cameraController.setBackendState(true, res.score, res.face, res.eyes_closed);
    if (!res.face) {
      camStatusDot.style.background = '#F59E0B';
      camStatusText.textContent = 'LOOK AT CAMERA (NO FACE)';
    } else if (res.eyes_closed) {
      camStatusDot.classList.add('blinking');
      camStatusDot.style.background = '#C9B8FF';
      camStatusText.textContent = `EYES CLOSED (MichalMlodawski) • ${(res.score ?? 0).toFixed(2)}`;
    } else {
      camStatusDot.classList.remove('blinking');
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = `AI TRACKING (MichalMlodawski) • ${(res.score ?? 0).toFixed(2)}`;
    }
  }

  // Handle server-side Morse events (DO NOT RETURN EARLY ON !res.face)
  if (res.events && res.events.length > 0) {
    for (const evt of res.events) {
      if (evt === 'dot') {
        if (currentMode === 'Blink2Morse') {
          morseAudio.playDot();
          flashCameraStatus('DOT (•)');
          floatingBarStatus.textContent = 'Blink: Dot (•)';
          pulseFloatingIcon();
        }
      } else if (evt === 'dash') {
        if (currentMode === 'Blink2Morse') {
          morseAudio.playDash();
          flashCameraStatus('DASH (—)');
          floatingBarStatus.textContent = 'Blink: Dash (—)';
          pulseFloatingIcon();
        }
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

  // Synchronize vision symbols and decoded text from backend session only when camera is actively tracking
  if (isCameraActive) {
    currentMorseBuffer = res.symbols;
    decodedText = res.text;
    updateDisplay();

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
 * Tactile Hold Animation & Real-time Progress Tracking for Fin2Morse Tap Pad
 */
function startHoldAnimation() {
  if (holdAnimFrameId !== null) {
    cancelAnimationFrame(holdAnimFrameId);
    holdAnimFrameId = null;
  }

  const updateMeter = () => {
    if (!isFingerHolding || fingerTapStartTime === 0) return;
    const elapsed = performance.now() - fingerTapStartTime;
    const threshold = DEFAULT_THRESHOLDS.shortDotMaxMs; // 380ms
    const progress = Math.min(100, (elapsed / threshold) * 100);

    if (finHoldProgress) {
      finHoldProgress.style.width = `${progress}%`;
    }

    if (elapsed >= threshold) {
      finTapZone?.classList.add('is-dash-ready');
      if (finTapStatus) finTapStatus.textContent = 'Hold threshold reached: [ — Dash ] (Ready to release!)';
      if (finTapIndicator) finTapIndicator.style.background = '#111111';
      if (finTapBadge) finTapBadge.textContent = 'DASH (>380ms)';
    } else {
      finTapZone?.classList.remove('is-dash-ready');
      if (finTapStatus) finTapStatus.textContent = `Holding... [ • Dot ] (${Math.round(elapsed)}ms)`;
      if (finTapIndicator) finTapIndicator.style.background = '#8B5CF6';
      if (finTapBadge) finTapBadge.textContent = 'DOT (<380ms)';
    }

    holdAnimFrameId = requestAnimationFrame(updateMeter);
  };

  holdAnimFrameId = requestAnimationFrame(updateMeter);
}

function stopHoldAnimation() {
  if (holdAnimFrameId !== null) {
    cancelAnimationFrame(holdAnimFrameId);
    holdAnimFrameId = null;
  }
  if (finHoldProgress) {
    finHoldProgress.style.width = '0%';
  }
  if (finTapBadge) {
    finTapBadge.textContent = 'HOLD: <380ms • | >380ms —';
  }
}

function handleFingerTapDown() {
  cancelPendingFinalizeTimers();
  if (isFingerHolding) return;
  isFingerHolding = true;
  fingerTapStartTime = performance.now();

  finTapZone?.classList.add('is-pressing');
  finTapZone?.classList.remove('is-dash-ready');
  floatingBarStatus.textContent = 'Fin2Morse: Pressing...';
  pulseFloatingIcon();
  startHoldAnimation();
}

function handleFingerTapUp() {
  if (!isFingerHolding || fingerTapStartTime === 0) return;
  const duration = performance.now() - fingerTapStartTime;
  isFingerHolding = false;
  fingerTapStartTime = 0;
  stopHoldAnimation();

  finTapZone?.classList.remove('is-pressing', 'is-dash-ready');
  if (finTapIndicator) finTapIndicator.style.background = '#10B981';

  const symbol = classifyFingerTap(duration);
  appendSymbol(symbol);
  flashCameraStatus(symbol === '.' ? 'DOT (•)' : 'DASH (—)');

  if (finTapStatus) {
    finTapStatus.textContent = symbol === '.'
      ? `Registered: Dot (•) [${Math.round(duration)}ms]`
      : `Registered: Dash (—) [${Math.round(duration)}ms]`;
  }
  if (finTapSub) {
    const potential = decodeMorseSequence(currentMorseBuffer);
    finTapSub.textContent = `Buffer: ${currentMorseBuffer} (Potential: '${potential}') — Tap again or pause to finalize`;
  }
}

function handleFingerTapCancel() {
  if (isFingerHolding) {
    const duration = performance.now() - fingerTapStartTime;
    if (duration > 60) {
      handleFingerTapUp();
    } else {
      isFingerHolding = false;
      fingerTapStartTime = 0;
      stopHoldAnimation();
      finTapZone?.classList.remove('is-pressing', 'is-dash-ready');
      if (finTapIndicator) finTapIndicator.style.background = '#10B981';
      if (finTapStatus) finTapStatus.textContent = 'Touch or Click to input Morse';
    }
  }
}

/**
 * Setup Event Listeners
 */
function setupEventListeners() {
  // Fin2Morse Tactile Tap Pad (Pointer events for Touch, Mouse & Pen with zero latency)
  if (finTapZone) {
    finTapZone.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      handleFingerTapDown();
    });

    finTapZone.addEventListener('pointerup', (e) => {
      e.preventDefault();
      handleFingerTapUp();
    });

    finTapZone.addEventListener('pointercancel', (e) => {
      e.preventDefault();
      handleFingerTapCancel();
    });

    finTapZone.addEventListener('pointerleave', () => {
      if (isFingerHolding) {
        handleFingerTapUp();
      }
    });

    // Keyboard accessibility directly on the tap pad
    finTapZone.addEventListener('keydown', (e) => {
      if ((e.code === 'Space' || e.code === 'Enter') && !e.repeat) {
        e.preventDefault();
        handleFingerTapDown();
      }
    });

    finTapZone.addEventListener('keyup', (e) => {
      if (e.code === 'Space' || e.code === 'Enter') {
        e.preventDefault();
        handleFingerTapUp();
      }
    });
  }

  // Portrait box touch support when in Fin2Morse mode
  if (portraitBox) {
    portraitBox.addEventListener('pointerdown', (e) => {
      if (currentMode === 'Fin2Morse' && !isCameraActive) {
        if ((e.target as HTMLElement).closest('button')) return;
        e.preventDefault();
        handleFingerTapDown();
      }
    });

    portraitBox.addEventListener('pointerup', (e) => {
      if (currentMode === 'Fin2Morse' && !isCameraActive && isFingerHolding) {
        e.preventDefault();
        handleFingerTapUp();
      }
    });
  }

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

  // Camera toggle button
  btnToggleCamera.addEventListener('click', toggleCamera);

  // Mode Switcher: Fin2Morse vs Blink2Morse
  btnModeFinger?.addEventListener('click', () => {
    currentMode = 'Fin2Morse';
    cameraController.setMode('Fin2Morse');
    wsBridge.sendMode('Fin2Morse');
    btnModeFinger?.classList.add('active');
    btnModeBlink?.classList.remove('active');
    floatingBarStatus.textContent = 'Fin2Morse Active (Hand & Finger Tap)';
    if (isCameraActive) {
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = 'HAND TRACKING';
    }
    updateDisplay();
    showToast('🖐️ Switched to Fin2Morse (Pure Hand & Finger Tracking)');
  });

  btnModeBlink?.addEventListener('click', () => {
    currentMode = 'Blink2Morse';
    cameraController.setMode('Blink2Morse');
    wsBridge.sendMode('Blink2Morse');
    btnModeBlink.classList.add('active');
    btnModeFinger?.classList.remove('active');
    floatingBarStatus.textContent = 'Blink2Morse (Beta) Active';
    if (isCameraActive) {
      camStatusDot.style.background = '#10B981';
      camStatusText.textContent = 'EYE TRACKING [BETA]';
    }
    updateDisplay();
    showToast('👁️ Switched to Blink2Morse (Beta - Eye Tracking Mode)');
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



  // Floating Welcome Modal (Bảng nổi giới thiệu khi vào web)
  const openWelcomeModal = () => {
    if (welcomeOverlay) {
      welcomeOverlay.classList.add('is-open');
      document.body.style.overflow = 'hidden';
    }
  };

  const closeWelcomeModal = () => {
    if (welcomeOverlay) {
      welcomeOverlay.classList.remove('is-open');
      document.body.style.overflow = '';
    }
  };

  btnWelcomeClose?.addEventListener('click', closeWelcomeModal);
  welcomeBackdrop?.addEventListener('click', closeWelcomeModal);
  btnWelcomeStart?.addEventListener('click', closeWelcomeModal);
  btnWelcomeDocs?.addEventListener('click', () => {
    closeWelcomeModal();
    flipToDocs();
  });
  navLinkAbout?.addEventListener('click', (e) => {
    e.preventDefault();
    openWelcomeModal();
  });

  // Keyboard accessibility and Fin2Morse tactile input
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (welcomeOverlay?.classList.contains('is-open')) {
        closeWelcomeModal();
        return;
      }
      if (morseFlipInner?.classList.contains('is-flipped')) {
        flipToChart();
        return;
      }
    }
    if (e.code === 'KeyD' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      cancelPendingFinalizeTimers();
      appendSymbol('.');
    } else if (e.code === 'KeyF' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      cancelPendingFinalizeTimers();
      appendSymbol('-');
    } else if (e.code === 'Space' && !e.ctrlKey && !e.metaKey && document.activeElement?.tagName !== 'INPUT') {
      e.preventDefault();
      if (currentMode === 'Fin2Morse') {
        if (!e.repeat && !isFingerHolding) {
          handleFingerTapDown();
        }
      } else {
        finalizeCharacter();
      }
    }
  });

  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && currentMode === 'Fin2Morse' && isFingerHolding) {
      e.preventDefault();
      handleFingerTapUp();
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
  
  // Display floating welcome modal on first entry
  if (welcomeOverlay) {
    welcomeOverlay.classList.add('is-open');
    document.body.style.overflow = 'hidden';
  }
});
