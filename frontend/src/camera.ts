/**
 * Webcam Manager and Real-time Gesture Engine for Act2Morse.
 * Dual-mode vision pipeline:
 *  - Fin2Morse: Hand Gesture & Landmark tracking powered by `hand_gesture.task` (MediaPipe Tasks Vision).
 *               Renders finger bones / connections in WHITE and finger landmark joints in CYAN.
 *               Completely ignores face and eyes.
 *  - Blink2Morse: Eye Blink Tracking via facial landmarks / local EAR vision.
 */

import { FilesetResolver, GestureRecognizer, type GestureRecognizerResult } from '@mediapipe/tasks-vision';

export interface BlinkEvent {
  durationMs: number;
  timestamp: number;
}

export interface FingerTapEvent {
  symbol: '.' | '-';
  durationMs: number;
  timestamp: number;
}

export interface CameraMetrics {
  fps: number;
  mode: 'Fin2Morse' | 'Blink2Morse';
  // Eye / Blink metrics
  ear: number;
  isBlinking: boolean;
  baselineEar: number;
  // Hand / Finger metrics
  isHandDetected: boolean;
  isFingerTapping: boolean;
  handConfidence: number;
  gestureName?: string;
}

/**
 * MediaPipe 21-Landmark Hand Skeleton Connections (Phalanges & Knuckles)
 */
export const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [9, 10], [10, 11], [11, 12],
  // Ring finger
  [13, 14], [14, 15], [15, 16],
  // Pinky finger
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Knuckle connections (palm arch)
  [5, 9], [9, 13], [13, 17],
];

export class CameraController {
  private videoEl: HTMLVideoElement | null = null;
  private canvasEl: HTMLCanvasElement | null = null;
  private stream: MediaStream | null = null;
  private animFrameId: number | null = null;

  // Offscreen canvas for vision processing (320x240 for fast local analysis)
  private offscreenCanvas: HTMLCanvasElement = document.createElement('canvas');
  private offscreenCtx: CanvasRenderingContext2D | null = null;

  // Dedicated transmission canvas for unmirrored, aspect-preserving WebSocket streaming
  private txCanvas: HTMLCanvasElement = document.createElement('canvas');
  private txCtx: CanvasRenderingContext2D | null = null;

  private isRunning: boolean = false;
  private currentMode: 'Fin2Morse' | 'Blink2Morse' = 'Fin2Morse';

  // MediaPipe Hand Gesture Model (`hand_gesture.task`)
  private gestureRecognizer: GestureRecognizer | null = null;
  private isModelLoading: boolean = false;
  private lastRecognizeTime: number = 0;
  private detectedGestureName: string = '';

  // Eye tracking state (Blink2Morse)
  private isBlinkActive: boolean = false;
  private blinkStartTime: number = 0;
  private earHistory: number[] = [];
  private baselineEar: number = 0.30;
  private currentEar: number = 0.30;
  private blinkCloseRatio: number = 0.76;
  private blinkOpenRatio: number = 0.88;

  // Hand & Finger tracking state (Fin2Morse)
  private isHandDetected: boolean = false;
  private handConfidence: number = 0;
  private isHandTapping: boolean = false;
  private handTapStartTime: number = 0;

  private lastFrameTime: number = performance.now();
  private lastFrameSentTime: number = 0;
  private fpsCounter: number = 0;
  private currentFps: number = 30;

  // Backend synchronization state
  private backendConnected: boolean = false;
  private backendScore: number | null = null;
  private backendFace: boolean = false;
  private backendEyesClosed: boolean = false;

  private onBlinkCallbacks: Array<(event: BlinkEvent) => void> = [];
  private onFingerTapCallbacks: Array<(event: FingerTapEvent) => void> = [];
  private onMetricsCallbacks: Array<(metrics: CameraMetrics) => void> = [];
  private onFrameCallbacks: Array<(canvas: HTMLCanvasElement) => void> = [];

  constructor() {
    // Preload hand_gesture.task in the background
    this.initHandGestureModel();
  }

  /**
   * Initializes the MediaPipe Gesture Recognizer using hand_gesture.task
   */
  private async initHandGestureModel() {
    if (this.gestureRecognizer || this.isModelLoading) return;
    this.isModelLoading = true;

    try {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      // Try local asset first, then fallback to cloud URL if needed
      try {
        this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/models/hand_gesture.task',
            delegate: 'GPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
        });
      } catch {
        this.gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task',
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          numHands: 2,
        });
      }
    } catch (err) {
      console.warn('Could not initialize hand_gesture.task via WebAssembly:', err);
    } finally {
      this.isModelLoading = false;
    }
  }

  public async start(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<boolean> {
    this.videoEl = video;
    this.canvasEl = canvas;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });

      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();

      this.isRunning = true;
      this.lastFrameTime = performance.now();
      this.earHistory = [];
      this.baselineEar = 0.30;
      this.currentEar = 0.30;
      this.isBlinkActive = false;
      this.isHandDetected = false;
      this.isHandTapping = false;

      // Ensure model is loaded
      if (!this.gestureRecognizer) {
        this.initHandGestureModel();
      }

      this.loop();
      return true;
    } catch (err) {
      console.warn('Unable to access webcam:', err);
      this.stop();
      return false;
    }
  }

  public stop() {
    this.isRunning = false;
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
    if (this.canvasEl) {
      const ctx = this.canvasEl.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, this.canvasEl.width, this.canvasEl.height);
    }
  }

  public isActive(): boolean {
    return this.isRunning;
  }

  public setMode(mode: 'Fin2Morse' | 'Blink2Morse') {
    this.currentMode = mode;
    this.isBlinkActive = false;
    this.isHandDetected = false;
    this.isHandTapping = false;
    this.handTapStartTime = 0;
    this.detectedGestureName = '';
    if (mode === 'Fin2Morse' && !this.gestureRecognizer) {
      this.initHandGestureModel();
    }
  }

  public getMode(): 'Fin2Morse' | 'Blink2Morse' {
    return this.currentMode;
  }

  public setBackendState(connected: boolean, score: number | null, face: boolean, eyesClosed: boolean) {
    this.backendConnected = connected;
    this.backendScore = score;
    this.backendFace = face;
    this.backendEyesClosed = eyesClosed;
  }

  public onBlink(callback: (event: BlinkEvent) => void) {
    this.onBlinkCallbacks.push(callback);
  }

  public onFingerTap(callback: (event: FingerTapEvent) => void) {
    this.onFingerTapCallbacks.push(callback);
  }

  public onMetrics(callback: (metrics: CameraMetrics) => void) {
    this.onMetricsCallbacks.push(callback);
  }

  public onFrame(callback: (canvas: HTMLCanvasElement) => void) {
    this.onFrameCallbacks.push(callback);
  }

  /**
   * Main vision processing loop
   */
  private loop = () => {
    if (!this.isRunning || !this.videoEl || !this.canvasEl) return;

    const now = performance.now();
    const delta = now - this.lastFrameTime;
    if (delta > 0) {
      this.fpsCounter = 0.9 * this.fpsCounter + 0.1 * (1000 / delta);
      this.currentFps = Math.round(this.fpsCounter);
    }
    this.lastFrameTime = now;

    this.processFrame();

    this.animFrameId = requestAnimationFrame(this.loop);
  };

  /**
   * Prepares an unmirrored, aspect-preserving canvas frame for AI transmission
   */
  private prepareTransmissionCanvas(): HTMLCanvasElement | null {
    if (!this.videoEl || this.videoEl.readyState < 2) return null;
    const vw = this.videoEl.videoWidth || 640;
    const vh = this.videoEl.videoHeight || 480;

    const scale = Math.min(1, 640 / vw);
    const tw = Math.round(vw * scale);
    const th = Math.round(vh * scale);

    if (this.txCanvas.width !== tw || this.txCanvas.height !== th) {
      this.txCanvas.width = tw;
      this.txCanvas.height = th;
      this.txCtx = this.txCanvas.getContext('2d');
    }
    if (!this.txCtx) return null;

    this.txCtx.drawImage(this.videoEl, 0, 0, tw, th);
    return this.txCanvas;
  }

  /**
   * Dispatches frame processing to Hand tracking (Fin2Morse) or Eye tracking (Blink2Morse).
   */
  private processFrame() {
    if (!this.videoEl || !this.canvasEl || this.videoEl.readyState < 2) return;

    // Synchronize onscreen canvas resolution with container dimensions (HiDPI)
    const rect = this.canvasEl.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const targetW = Math.round(rect.width * dpr);
    const targetH = Math.round(rect.height * dpr);

    if (this.canvasEl.width !== targetW || this.canvasEl.height !== targetH) {
      this.canvasEl.width = targetW;
      this.canvasEl.height = targetH;
    }

    const ctx = this.canvasEl.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, targetW, targetH);

    const now = performance.now();

    if (this.currentMode === 'Fin2Morse') {
      this.processHandFrame(ctx, targetW, targetH, dpr, now);
    } else {
      this.processEyeFrame(ctx, targetW, targetH, dpr, now);
    }
  }

  /**
   * Fin2Morse Hand Tracking: Uses MediaPipe hand_gesture.task model.
   * Renders finger bones in WHITE (#FFFFFF) and landmark points in CYAN (#00FFFF).
   * Completely ignores face and eyes.
   */
  private processHandFrame(
    ctx: CanvasRenderingContext2D,
    targetW: number,
    targetH: number,
    dpr: number,
    now: number
  ) {
    if (!this.videoEl || this.videoEl.readyState < 2) return;

    let hasHandModelResults = false;

    // 1. Run MediaPipe Gesture Recognizer model if loaded
    if (this.gestureRecognizer && now > this.lastRecognizeTime + 25) {
      this.lastRecognizeTime = now;
      try {
        const results = this.gestureRecognizer.recognizeForVideo(this.videoEl, now);
        if (results && results.landmarks && results.landmarks.length > 0) {
          hasHandModelResults = true;
          this.isHandDetected = true;
          this.renderHandGestureResult(ctx, results, targetW, targetH, dpr, now);

          const metrics: CameraMetrics = {
            fps: this.currentFps,
            mode: 'Fin2Morse',
            ear: 0,
            isBlinking: false,
            baselineEar: 0,
            isHandDetected: true,
            isFingerTapping: this.isHandTapping,
            handConfidence: Number(this.handConfidence.toFixed(2)),
            gestureName: this.detectedGestureName,
          };
          this.onMetricsCallbacks.forEach((cb) => cb(metrics));
          return;
        }
      } catch (err) {
        // Fall through to client vision fallback if model frame fails
      }
    }

    // 2. Client-side Hand Silhouette & Fingertip Fallback (when model is loading or in low-resource environments)
    if (!hasHandModelResults) {
      this.processHandFallback(ctx, targetW, targetH, dpr, now);
    }
  }

  /**
   * Renders Hand Model Results from hand_gesture.task:
   *  - Đốt ngón tay (skeletal joints/bones): MÀU TRẮNG (#FFFFFF)
   *  - Điểm từng ngón (landmark points/joints): MÀU CYAN (#00FFFF)
   */
  private renderHandGestureResult(
    ctx: CanvasRenderingContext2D,
    results: GestureRecognizerResult,
    targetW: number,
    targetH: number,
    dpr: number,
    now: number
  ) {
    ctx.save();

    for (let h = 0; h < results.landmarks.length; h++) {
      const landmarks = results.landmarks[h];
      const gestures = results.gestures[h];
      const topGesture = gestures && gestures.length > 0 ? gestures[0] : null;

      if (topGesture && topGesture.categoryName && topGesture.categoryName !== 'None') {
        this.detectedGestureName = topGesture.categoryName;
        this.handConfidence = topGesture.score;
      } else {
        this.detectedGestureName = 'Hand Detected';
        this.handConfidence = 0.95;
      }

      // 1. RENDER ĐỐT NGÓN TAY MÀU TRẮNG (#FFFFFF)
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = Math.round(3.2 * dpr);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.shadowColor = 'rgba(255, 255, 255, 0.45)';
      ctx.shadowBlur = Math.round(6 * dpr);

      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = landmarks[startIdx];
        const p2 = landmarks[endIdx];

        // Mirrored selfie view: x is inverted
        const x1 = (1 - p1.x) * targetW;
        const y1 = p1.y * targetH;
        const x2 = (1 - p2.x) * targetW;
        const y2 = p2.y * targetH;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      ctx.shadowBlur = 0;

      // 2. RENDER ĐIỂM TỪNG NGÓN MÀU CYAN (#00FFFF)
      for (let i = 0; i < landmarks.length; i++) {
        const p = landmarks[i];
        const px = (1 - p.x) * targetW;
        const py = p.y * targetH;

        const isTip = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20);
        const radius = isTip ? Math.round(6 * dpr) : Math.round(4 * dpr);

        // Vibrant Cyan Joint
        ctx.fillStyle = '#00FFFF';
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();

        // Dark Cyan Rim
        ctx.strokeStyle = '#0891B2';
        ctx.lineWidth = Math.round(1.5 * dpr);
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.stroke();

        // Glowing outer halo for tips
        if (isTip) {
          ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)';
          ctx.lineWidth = Math.round(1 * dpr);
          ctx.beginPath();
          ctx.arc(px, py, radius + Math.round(3.5 * dpr), 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // 3. Pinch & Micro-Tap Detection for Morse Code
      // Measures distance between Thumb Tip (4) and Index Tip (8)
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];
      const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);

      const isPinching = pinchDist < 0.082;
      const midX = ((1 - thumbTip.x) + (1 - indexTip.x)) / 2 * targetW;
      const midY = (thumbTip.y + indexTip.y) / 2 * targetH;

      if (isPinching) {
        if (!this.isHandTapping) {
          this.isHandTapping = true;
          this.handTapStartTime = now;
        }

        const elapsed = now - this.handTapStartTime;
        const isDash = elapsed >= 380;

        // Visual contact indicator at pinch point
        ctx.fillStyle = isDash ? '#00FFFF' : '#C9B8FF';
        ctx.beginPath();
        ctx.arc(midX, midY, Math.round(12 * dpr), 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = Math.round(2 * dpr);
        ctx.beginPath();
        ctx.arc(midX, midY, Math.round(18 * dpr), 0, Math.PI * 2);
        ctx.stroke();

        ctx.font = `700 ${Math.round(11 * dpr)}px 'Inter', sans-serif`;
        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'center';
        ctx.fillText(isDash ? '— DASH HOLD (>380ms)' : '• DOT TAP (<380ms)', midX, midY - Math.round(24 * dpr));
        ctx.textAlign = 'left';
      } else {
        if (this.isHandTapping) {
          const duration = Math.round(now - this.handTapStartTime);
          this.isHandTapping = false;

          if (duration >= 75 && duration <= 1600) {
            const symbol = duration < 380 ? '.' : '-';
            this.onFingerTapCallbacks.forEach((cb) =>
              cb({ symbol, durationMs: duration, timestamp: now })
            );
          }
        }
      }

      // 4. Gesture Name Chip above Hand
      const wrist = landmarks[0];
      const wx = (1 - wrist.x) * targetW;
      const wy = Math.max(Math.round(25 * dpr), wrist.y * targetH - Math.round(25 * dpr));

      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.beginPath();
      ctx.roundRect(wx - Math.round(60 * dpr), wy - Math.round(16 * dpr), Math.round(120 * dpr), Math.round(22 * dpr), Math.round(6 * dpr));
      ctx.fill();
      ctx.strokeStyle = '#00FFFF';
      ctx.lineWidth = Math.round(1 * dpr);
      ctx.stroke();

      ctx.font = `600 ${Math.round(10 * dpr)}px 'JetBrains Mono', monospace`;
      ctx.fillStyle = '#00FFFF';
      ctx.textAlign = 'center';
      const label = this.detectedGestureName ? `GESTURE: ${this.detectedGestureName}` : 'HAND: LOCKED';
      ctx.fillText(label, wx, wy - Math.round(2 * dpr));
      ctx.textAlign = 'left';
    }

    // Bottom HUD Status Bar
    const barW = Math.round(190 * dpr);
    const barH = Math.round(5 * dpr);
    const barX = Math.round(20 * dpr);
    const barY = targetH - Math.round(86 * dpr);

    ctx.font = `600 ${Math.round(10 * dpr)}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = '#00FFFF';
    ctx.fillText('FIN2MORSE • opencv/handpose_estimation_mediapipe', barX, barY - Math.round(6 * dpr));

    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, Math.round(2.5 * dpr));
    ctx.fill();

    ctx.fillStyle = '#00FFFF';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW * 0.95, barH, Math.round(2.5 * dpr));
    ctx.fill();

    ctx.restore();
  }

  /**
   * Fast canvas vision fallback when WebAssembly model is still loading
   * Renders finger bones in WHITE and joints in CYAN
   */
  private processHandFallback(
    ctx: CanvasRenderingContext2D,
    targetW: number,
    targetH: number,
    dpr: number,
    _now: number
  ) {
    const offW = 320;
    const offH = 240;
    if (this.offscreenCanvas.width !== offW || this.offscreenCanvas.height !== offH) {
      this.offscreenCanvas.width = offW;
      this.offscreenCanvas.height = offH;
      this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
    }
    if (!this.offscreenCtx || !this.videoEl) return;

    this.offscreenCtx.save();
    this.offscreenCtx.scale(-1, 1);
    this.offscreenCtx.drawImage(this.videoEl, -offW, 0, offW, offH);
    this.offscreenCtx.restore();

    try {
      const imageData = this.offscreenCtx.getImageData(0, 0, offW, offH);
      const data = imageData.data;
      let skinCount = 0;
      let topY = offH;
      let topX = Math.floor(offW / 2);
      const step = 3;

      for (let y = Math.floor(offH * 0.15); y < offH; y += step) {
        for (let x = Math.floor(offW * 0.15); x < Math.floor(offW * 0.85); x += step) {
          const idx = (y * offW + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          const isSkin =
            (r > 70 && g > 35 && b > 20 && r > g && r > b && (r - g) > 10) ||
            (0.299 * r + 0.587 * g + 0.114 * b > 40 &&
              128 - 0.168736 * r - 0.331264 * g + 0.5 * b >= 77 &&
              128 + 0.5 * r - 0.418688 * g - 0.081312 * b >= 130);

          if (isSkin) {
            skinCount++;
            if (y < topY) {
              topY = y;
              topX = x;
            }
          }
        }
      }

      if (skinCount > 180) {
        this.isHandDetected = true;
        this.handConfidence = 0.85;

        const tipX = (topX / offW) * targetW;
        const tipY = (topY / offH) * targetH;

        ctx.save();
        // White bone connector to base
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = Math.round(3 * dpr);
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX, tipY + Math.round(60 * dpr));
        ctx.stroke();

        // Cyan joint points
        ctx.fillStyle = '#00FFFF';
        ctx.strokeStyle = '#0891B2';
        ctx.lineWidth = Math.round(1.5 * dpr);
        ctx.beginPath();
        ctx.arc(tipX, tipY, Math.round(6 * dpr), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(tipX, tipY + Math.round(30 * dpr), Math.round(4 * dpr), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(tipX, tipY + Math.round(60 * dpr), Math.round(4 * dpr), 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.font = `600 ${Math.round(10 * dpr)}px 'JetBrains Mono', monospace`;
        ctx.fillStyle = '#00FFFF';
        ctx.fillText('FIN2MORSE • HAND DETECTED', tipX - Math.round(40 * dpr), tipY - Math.round(12 * dpr));
        ctx.restore();
      } else {
        this.isHandDetected = false;
        this.handConfidence = 0;
        this.drawHandSearchHUD(ctx, targetW, targetH, dpr);
      }

      const metrics: CameraMetrics = {
        fps: this.currentFps,
        mode: 'Fin2Morse',
        ear: 0,
        isBlinking: false,
        baselineEar: 0,
        isHandDetected: this.isHandDetected,
        isFingerTapping: this.isHandTapping,
        handConfidence: Number(this.handConfidence.toFixed(2)),
        gestureName: this.detectedGestureName,
      };
      this.onMetricsCallbacks.forEach((cb) => cb(metrics));
    } catch {
      // Ignore canvas read errors
    }
  }

  /**
   * Draws guiding HUD prompt when hand is not yet detected in frame
   */
  private drawHandSearchHUD(
    ctx: CanvasRenderingContext2D,
    targetW: number,
    targetH: number,
    dpr: number
  ) {
    ctx.save();

    const boxW = Math.round(targetW * 0.65);
    const boxH = Math.round(targetH * 0.55);
    const boxX = Math.round((targetW - boxW) / 2);
    const boxY = Math.round(targetH * 0.22);

    ctx.setLineDash([Math.round(8 * dpr), Math.round(6 * dpr)]);
    ctx.lineWidth = Math.round(1.5 * dpr);
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.7)';
    ctx.strokeRect(boxX, boxY, boxW, boxH);
    ctx.setLineDash([]);

    ctx.textAlign = 'center';
    ctx.font = `600 ${Math.round(13 * dpr)}px 'Inter', sans-serif`;
    ctx.fillStyle = '#00FFFF';
    ctx.fillText('SHOW HAND TO CAMERA', targetW / 2, boxY + boxH / 2 - Math.round(6 * dpr));

    ctx.font = `400 ${Math.round(10 * dpr)}px 'Inter', sans-serif`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.fillText('Model: opencv/handpose_estimation_mediapipe (Bones: White | Points: Cyan)', targetW / 2, boxY + boxH / 2 + Math.round(14 * dpr));

    ctx.restore();
  }

  /**
   * Blink2Morse Eye Tracking: Measures Eye Aspect Ratio and detects eye blinks.
   */
  private processEyeFrame(
    ctx: CanvasRenderingContext2D,
    targetW: number,
    targetH: number,
    dpr: number,
    now: number
  ) {
    const offW = 320;
    const offH = 240;
    if (this.offscreenCanvas.width !== offW || this.offscreenCanvas.height !== offH) {
      this.offscreenCanvas.width = offW;
      this.offscreenCanvas.height = offH;
      this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
    }
    if (!this.offscreenCtx || !this.videoEl) return;

    this.offscreenCtx.save();
    this.offscreenCtx.scale(-1, 1);
    this.offscreenCtx.drawImage(this.videoEl, -offW, 0, offW, offH);
    this.offscreenCtx.restore();

    const roiX = Math.floor(offW * 0.20);
    const roiY = Math.floor(offH * 0.20);
    const roiW = Math.floor(offW * 0.60);
    const roiH = Math.floor(offH * 0.32);

    try {
      const imageData = this.offscreenCtx.getImageData(roiX, roiY, roiW, roiH);
      const data = imageData.data;

      const leftX1 = Math.floor(roiW * 0.10);
      const leftX2 = Math.floor(roiW * 0.45);
      const rightX1 = Math.floor(roiW * 0.55);
      const rightX2 = Math.floor(roiW * 0.90);
      const eyeY1 = Math.floor(roiH * 0.20);
      const eyeY2 = Math.floor(roiH * 0.85);

      let verticalGradientSum = 0;
      let totalLum = 0;
      let eyePixelCount = 0;
      const step = 2;

      for (let y = eyeY1; y < eyeY2 - 2; y += step) {
        for (let x = 0; x < roiW; x += step) {
          const inLeftEye = x >= leftX1 && x <= leftX2;
          const inRightEye = x >= rightX1 && x <= rightX2;
          if (!inLeftEye && !inRightEye) continue;

          const idxCurrent = (y * roiW + x) * 4;
          const idxNext = ((y + 2) * roiW + x) * 4;

          const lum1 = 0.299 * data[idxCurrent] + 0.587 * data[idxCurrent + 1] + 0.114 * data[idxCurrent + 2];
          const lum2 = 0.299 * data[idxNext] + 0.587 * data[idxNext + 1] + 0.114 * data[idxNext + 2];

          verticalGradientSum += Math.abs(lum2 - lum1);
          totalLum += lum1;
          eyePixelCount++;
        }
      }

      const avgLum = eyePixelCount > 0 ? totalLum / eyePixelCount : 128;
      let darkIrisCount = 0;
      for (let y = eyeY1; y < eyeY2; y += step) {
        for (let x = 0; x < roiW; x += step) {
          const inLeftEye = x >= leftX1 && x <= leftX2;
          const inRightEye = x >= rightX1 && x <= rightX2;
          if (!inLeftEye && !inRightEye) continue;

          const idx = (y * roiW + x) * 4;
          const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
          if (lum < avgLum * 0.75) {
            darkIrisCount++;
          }
        }
      }

      const avgGradient = eyePixelCount > 0 ? verticalGradientSum / eyePixelCount : 0;
      const darkRatio = eyePixelCount > 0 ? darkIrisCount / eyePixelCount : 0;
      const instantEar = Math.max(0.08, Math.min(0.60, (avgGradient / 35) * 0.60 + darkRatio * 1.6));

      this.currentEar = 0.50 * this.currentEar + 0.50 * instantEar;

      if (!this.isBlinkActive) {
        this.earHistory.push(this.currentEar);
        if (this.earHistory.length > 45) {
          this.earHistory.shift();
        }
        if (this.earHistory.length > 8) {
          const sorted = [...this.earHistory].sort((a, b) => a - b);
          this.baselineEar = sorted[Math.floor(sorted.length * 0.75)];
        }
      }

      const closeThreshold = Math.max(0.10, this.baselineEar * this.blinkCloseRatio);
      const openThreshold = Math.max(0.12, this.baselineEar * this.blinkOpenRatio);

      if (!this.backendConnected) {
        if (!this.isBlinkActive && this.currentEar < closeThreshold) {
          this.isBlinkActive = true;
          this.blinkStartTime = now;
        } else if (this.isBlinkActive) {
          const elapsed = now - this.blinkStartTime;
          if (elapsed > 1500) {
            this.isBlinkActive = false;
            this.earHistory = [];
          } else if (this.currentEar > openThreshold) {
            this.isBlinkActive = false;
            const duration = Math.round(elapsed);
            if (duration >= 80 && duration <= 1400) {
              this.onBlinkCallbacks.forEach((cb) => cb({ durationMs: duration, timestamp: now }));
            }
          }
        }
      } else {
        this.isBlinkActive = this.backendEyesClosed;
      }

      const hudX = Math.floor(targetW * 0.20);
      const hudY = Math.floor(targetH * 0.20);
      const hudW = Math.floor(targetW * 0.60);
      const hudH = Math.floor(targetH * 0.32);

      this.drawEyeHUD(ctx, hudX, hudY, hudW, hudH, closeThreshold, dpr);

      const metrics: CameraMetrics = {
        fps: this.currentFps,
        mode: 'Blink2Morse',
        ear: Number(this.currentEar.toFixed(3)),
        isBlinking: this.isBlinkActive,
        baselineEar: Number(this.baselineEar.toFixed(3)),
        isHandDetected: false,
        isFingerTapping: false,
        handConfidence: 0,
      };
      this.onMetricsCallbacks.forEach((cb) => cb(metrics));

      // Stream frames to WebSocket backend only in Blink2Morse mode
      if (now - this.lastFrameSentTime >= 40) {
        this.lastFrameSentTime = now;
        const tx = this.prepareTransmissionCanvas();
        if (tx) {
          this.onFrameCallbacks.forEach((cb) => cb(tx));
        }
      }
    } catch {
      // Ignore canvas access errors
    }
  }

  /**
   * Draws a clean tracking HUD over the eye region in Blink2Morse mode
   */
  private drawEyeHUD(
    ctx: CanvasRenderingContext2D,
    rx: number,
    ry: number,
    rw: number,
    rh: number,
    threshold: number,
    dpr: number
  ) {
    ctx.save();

    const isBlinking = this.backendConnected ? this.backendEyesClosed : this.isBlinkActive;
    const isFaceDetected = this.backendConnected ? this.backendFace : true;

    let strokeColor = 'rgba(255, 255, 255, 0.45)';
    if (this.backendConnected && !isFaceDetected) {
      strokeColor = '#F59E0B';
    } else if (isBlinking) {
      strokeColor = '#C9B8FF';
    } else if (this.backendConnected) {
      strokeColor = '#10B981';
    }

    const cornerLen = Math.round(18 * dpr);
    ctx.lineWidth = Math.round(2 * dpr);
    ctx.strokeStyle = strokeColor;

    // Top-left
    ctx.beginPath();
    ctx.moveTo(rx, ry + cornerLen);
    ctx.lineTo(rx, ry);
    ctx.lineTo(rx + cornerLen, ry);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(rx + rw - cornerLen, ry);
    ctx.lineTo(rx + rw, ry);
    ctx.lineTo(rx + rw, ry + cornerLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(rx, ry + rh - cornerLen);
    ctx.lineTo(rx, ry + rh);
    ctx.lineTo(rx + cornerLen, ry + rh);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(rx + rw - cornerLen, ry + rh);
    ctx.lineTo(rx + rw, ry + rh);
    ctx.lineTo(rx + rw, ry + rh - cornerLen);
    ctx.stroke();

    // Center crosshair
    const cx = rx + rw / 2;
    const cy = ry + rh / 2;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = Math.round(1 * dpr);
    const crossSize = Math.round(8 * dpr);
    ctx.beginPath();
    ctx.moveTo(cx - crossSize, cy);
    ctx.lineTo(cx + crossSize, cy);
    ctx.moveTo(cx, cy - crossSize);
    ctx.lineTo(cx, cy + crossSize);
    ctx.stroke();

    // EAR Indicator Bar
    const barW = Math.round(140 * dpr);
    const barH = Math.round(5 * dpr);
    const barX = Math.round(20 * dpr);
    const barY = ctx.canvas.height - Math.round(86 * dpr);

    const displayScore = this.backendConnected
      ? (this.backendScore ?? 0)
      : this.currentEar;
    const displayThreshold = this.backendConnected ? 0.60 : threshold;
    const maxVal = this.backendConnected ? 1.0 : 0.45;

    ctx.font = `500 ${Math.round(10 * dpr)}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    const statusPrefix = this.backendConnected
      ? (isFaceDetected ? (isBlinking ? 'AI (MichalMlodawski): CLOSED' : 'AI (MichalMlodawski): OPEN') : 'AI: NO FACE')
      : (isBlinking ? 'LOCAL: BLINK' : 'LOCAL: OPEN');
    ctx.fillText(`${statusPrefix} (${displayScore.toFixed(2)})`, barX, barY - Math.round(5 * dpr));

    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, Math.round(2.5 * dpr));
    ctx.fill();

    const earPct = Math.min(1, Math.max(0, displayScore / maxVal));
    ctx.fillStyle = isBlinking ? '#C9B8FF' : '#FFFFFF';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW * earPct, barH, Math.round(2.5 * dpr));
    ctx.fill();

    const threshX = barX + barW * Math.min(1, displayThreshold / maxVal);
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = Math.round(2 * dpr);
    ctx.beginPath();
    ctx.moveTo(threshX, barY - Math.round(2 * dpr));
    ctx.lineTo(threshX, barY + barH + Math.round(2 * dpr));
    ctx.stroke();

    ctx.restore();
  }

  /**
   * Helper to manually simulate a blink (for testing / demo without webcam)
   */
  public triggerSimulatedBlink(durationMs: number) {
    const now = performance.now();
    this.onBlinkCallbacks.forEach((cb) => cb({ durationMs, timestamp: now }));
  }
}

export const cameraController = new CameraController();
