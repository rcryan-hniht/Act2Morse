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

export type HandGestureAction = 'space' | 'backspace';

export interface GestureActionEvent {
  action: HandGestureAction;
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
  private lastVideoTime: number = -1;
  private latestHandResults: GestureRecognizerResult | null = null;
  private lastHandResultTime: number = 0;

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
  private fpsCounter: number = 60;
  private currentFps: number = 60;

  // Backend synchronization state
  private backendConnected: boolean = false;
  private backendScore: number | null = null;
  private backendFace: boolean = false;
  private backendEyesClosed: boolean = false;

  // Gesture action tracking (fist -> space, right thumb pointing left -> backspace)
  private fistHoldStartTime: number = 0;
  private fistTriggered: boolean = false;
  private thumbLeftHoldStartTime: number = 0;
  private thumbLeftTriggered: boolean = false;

  private onBlinkCallbacks: Array<(event: BlinkEvent) => void> = [];
  private onFingerTapCallbacks: Array<(event: FingerTapEvent) => void> = [];
  private onPinchStateCallbacks: Array<(isPinching: boolean, elapsedMs: number) => void> = [];
  private onGestureActionCallbacks: Array<(event: GestureActionEvent) => void> = [];
  private onMetricsCallbacks: Array<(metrics: CameraMetrics) => void> = [];
  private onFrameCallbacks: Array<(canvas: HTMLCanvasElement) => void> = [];

  constructor() {
    // Preload hand_gesture.task in the background
    this.initHandGestureModel();
  }

  public isModelReady(): boolean {
    return this.gestureRecognizer !== null;
  }

  public isModelLoadingActive(): boolean {
    return this.isModelLoading;
  }

  /**
   * Fetches the hand_gesture.task model bundle.
   * Checks local file and verifies ZIP magic header (PK\x03\x04).
   * Automatically falls back to Google's official MediaPipe CDN if local file is missing,
   * corrupted, or replaced by an SPA HTML rewrite (e.g. on Vercel).
   */
  private async fetchModelBuffer(): Promise<Uint8Array> {
    const candidates = [
      '/models/hand_gesture.task',
      './models/hand_gesture.task',
    ];

    for (const url of candidates) {
      try {
        const res = await fetch(url);
        const ctype = (res.headers.get('content-type') || '').toLowerCase();
        if (res.ok && !ctype.includes('text/html')) {
          const buf = await res.arrayBuffer();
          const u8 = new Uint8Array(buf);
          // Check for valid zip magic bytes PK\x03\x04
          if (u8.length > 50000 && u8[0] === 0x50 && u8[1] === 0x4b) {
            console.info('[Fin2Morse] Loaded local hand_gesture.task model bundle');
            return u8;
          }
        }
      } catch {
        // Continue to fallback
      }
    }

    console.info('[Fin2Morse] Local model unavailable or invalid; downloading from official MediaPipe Google CDN...');
    const cdnUrl = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';
    const cdnRes = await fetch(cdnUrl);
    if (!cdnRes.ok) {
      throw new Error(`Failed to load gesture recognizer model from CDN: ${cdnRes.statusText}`);
    }
    const cdnBuf = await cdnRes.arrayBuffer();
    console.info('[Fin2Morse] Successfully downloaded gesture_recognizer.task from Google CDN');
    return new Uint8Array(cdnBuf);
  }

  /**
   * Creates GestureRecognizer with GPU delegate, automatically falling back to CPU.
   */
  private async createRecognizer(vision: any, modelBuffer: Uint8Array): Promise<GestureRecognizer> {
    const options = {
      runningMode: 'VIDEO' as const,
      numHands: 1,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35,
    };

    try {
      return await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: modelBuffer,
          delegate: 'GPU',
        },
        ...options,
      });
    } catch (gpuErr) {
      console.warn('[Fin2Morse] WebGL/GPU delegate unavailable, using CPU delegate:', gpuErr);
      return await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: modelBuffer,
          delegate: 'CPU',
        },
        ...options,
      });
    }
  }

  /**
   * Initializes the MediaPipe Gesture Recognizer using hand_gesture.task.
   * Multi-tiered fallback for both WASM runtime and model buffer.
   */
  private async initHandGestureModel() {
    if (this.gestureRecognizer || this.isModelLoading) return;
    this.isModelLoading = true;

    try {
      const modelBuffer = await this.fetchModelBuffer();

      try {
        const vision = await FilesetResolver.forVisionTasks('/wasm');
        this.gestureRecognizer = await this.createRecognizer(vision, modelBuffer);
      } catch (localErr) {
        console.warn('[Fin2Morse] Local WASM init failed, attempting jsDelivr CDN WASM:', localErr);
        const cdnVision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        this.gestureRecognizer = await this.createRecognizer(cdnVision, modelBuffer);
      }
      console.info('[Fin2Morse] Hand tracking model loaded successfully');
    } catch (err) {
      console.error('[Fin2Morse] Could not initialize hand_gesture.task:', err);
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
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 60 },
          facingMode: 'user',
        },
        audio: false,
      });

      // Attempt to negotiate 60 FPS hardware capture if supported by webcam
      const track = this.stream.getVideoTracks()[0];
      if (track) {
        try {
          const caps = (track as any).getCapabilities?.();
          if (caps && caps.frameRate && caps.frameRate.max && caps.frameRate.max >= 60) {
            await track.applyConstraints({ frameRate: 60 });
          }
        } catch {
          // ignore constraint failure
        }
      }

      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();

      this.isRunning = true;
      this.lastFrameTime = performance.now();
      this.lastVideoTime = -1;
      this.lastHandResultTime = 0;
      this.latestHandResults = null;
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
    this.latestHandResults = null;
    this.lastVideoTime = -1;
    this.lastHandResultTime = 0;
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
    this.fistHoldStartTime = 0;
    this.fistTriggered = false;
    this.thumbLeftHoldStartTime = 0;
    this.thumbLeftTriggered = false;
    this.latestHandResults = null;
    this.lastVideoTime = -1;
    this.lastHandResultTime = 0;
    this.onPinchStateCallbacks.forEach((cb) => cb(false, 0));
    if (mode === 'Fin2Morse' && !this.gestureRecognizer) {
      this.initHandGestureModel();
    }
  }

  public getMode(): 'Fin2Morse' | 'Blink2Morse' {
    return this.currentMode;
  }

  public isHandVisible(): boolean {
    return this.isHandDetected;
  }

  public getFps(): number {
    return this.currentFps;
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

  public onPinchState(callback: (isPinching: boolean, elapsedMs: number) => void) {
    this.onPinchStateCallbacks.push(callback);
  }

  public onGestureAction(callback: (event: GestureActionEvent) => void) {
    this.onGestureActionCallbacks.push(callback);
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

    const scale = Math.min(1, 320 / vw);
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

    const dpr = Math.min(window.devicePixelRatio || 1, 1.25);
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
   * Uses video.currentTime to skip duplicate frames and avoid re-running inference on same image.
   */
  private processHandFrame(
    ctx: CanvasRenderingContext2D,
    targetW: number,
    targetH: number,
    dpr: number,
    now: number
  ) {
    if (!this.videoEl || this.videoEl.readyState < 2 || !this.videoEl.videoWidth || !this.videoEl.videoHeight) return;

    // 1. Run MediaPipe inference only when a genuinely new video frame is available
    //    video.currentTime advances when the decoder pushes a new decoded frame.
    const currentVideoTime = this.videoEl.currentTime;
    const isNewVideoFrame = (currentVideoTime !== this.lastVideoTime) || (now - this.lastRecognizeTime >= 33);

    if (this.gestureRecognizer && isNewVideoFrame) {
      this.lastVideoTime = currentVideoTime;
      const frameTime = Math.max(now, this.lastRecognizeTime + 1);
      this.lastRecognizeTime = frameTime;
      try {
        const results = this.gestureRecognizer.recognizeForVideo(this.videoEl, frameTime);
        if (results && results.landmarks && results.landmarks.length > 0) {
          this.latestHandResults = results;
          this.lastHandResultTime = now;
        }
      } catch (err) {
        console.warn('[Fin2Morse] recognizeForVideo error:', err);
      }
    }

    // 2. Render latest valid results if within 250ms (smoothing single-frame decoder dropouts)
    const hasValidHand = !!(
      this.latestHandResults &&
      this.latestHandResults.landmarks &&
      this.latestHandResults.landmarks.length > 0 &&
      this.lastHandResultTime > 0 &&
      (now - this.lastHandResultTime < 250)
    );

    if (hasValidHand && this.latestHandResults) {
      this.isHandDetected = true;
      this.renderHandGestureResult(ctx, this.latestHandResults, targetW, targetH, dpr, now);

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
    } else {
      // Clean clear when hand is lost or absent
      this.isHandDetected = false;
      this.handConfidence = 0;
      this.detectedGestureName = '';
      this.fistHoldStartTime = 0;
      this.fistTriggered = false;
      this.thumbLeftHoldStartTime = 0;
      this.thumbLeftTriggered = false;
      if (now - this.lastHandResultTime >= 250) {
        this.latestHandResults = null;
      }

      if (this.isHandTapping) {
        const duration = Math.round(now - this.handTapStartTime);
        this.isHandTapping = false;
        this.onPinchStateCallbacks.forEach((cb) => cb(false, duration));

        if (duration >= 75 && duration <= 2200) {
          const symbol = duration < 380 ? '.' : '-';
          this.onFingerTapCallbacks.forEach((cb) =>
            cb({ symbol, durationMs: duration, timestamp: now })
          );
        }
      }

      const metrics: CameraMetrics = {
        fps: this.currentFps,
        mode: 'Fin2Morse',
        ear: 0,
        isBlinking: false,
        baselineEar: 0,
        isHandDetected: false,
        isFingerTapping: false,
        handConfidence: 0,
        gestureName: '',
      };
      this.onMetricsCallbacks.forEach((cb) => cb(metrics));
    }

    // 3. Stream frames to WebSocket backend (every 66ms ~ 15fps)
    if (now - this.lastFrameSentTime >= 66) {
      this.lastFrameSentTime = now;
      const tx = this.prepareTransmissionCanvas();
      if (tx) {
        this.onFrameCallbacks.forEach((cb) => cb(tx));
      }
    }
  }

  /**
   * Transforms normalized MediaPipe video coordinates [0..1, 0..1]
   * to canvas coordinates matching CSS `object-fit: cover; object-position: center; transform: scaleX(-1)`.
   */
  public videoToCanvasCoord(
    normX: number,
    normY: number,
    targetW: number,
    targetH: number
  ): { x: number; y: number } {
    const vw = this.videoEl?.videoWidth || 640;
    const vh = this.videoEl?.videoHeight || 480;

    // Scale factor matching CSS `object-fit: cover`
    const scale = Math.max(targetW / vw, targetH / vh);
    const renderedW = vw * scale;
    const renderedH = vh * scale;

    // Centered offsets (negative if dimension overflow is cropped by container)
    const offsetX = (targetW - renderedW) / 2;
    const offsetY = (targetH - renderedH) / 2;

    // Mirrored selfie view (matching video CSS `transform: scaleX(-1)`)
    const mirroredX = 1 - normX;
    const x = offsetX + mirroredX * renderedW;
    const y = offsetY + normY * renderedH;

    return { x, y };
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

      // Precompute all 21 landmark canvas coordinates once
      const pts = landmarks.map((p) => this.videoToCanvasCoord(p.x, p.y, targetW, targetH));

      // 1. RENDER ĐỐT NGÓN TAY MÀU TRẮNG (#FFFFFF) - Batched single-path draw (no shadow blur overhead)
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = Math.round(3.2 * dpr);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const pt1 = pts[startIdx];
        const pt2 = pts[endIdx];
        ctx.moveTo(pt1.x, pt1.y);
        ctx.lineTo(pt2.x, pt2.y);
      }
      ctx.stroke();

      // 2. RENDER ĐIỂM TỪNG NGÓN MÀU CYAN (#00FFFF)
      for (let i = 0; i < pts.length; i++) {
        const pt = pts[i];
        const isTip = (i === 4 || i === 8 || i === 12 || i === 16 || i === 20);
        const radius = isTip ? Math.round(6.5 * dpr) : Math.round(4 * dpr);

        // Vibrant Cyan Joint
        ctx.fillStyle = '#00FFFF';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Dark Cyan Rim
        ctx.strokeStyle = '#0891B2';
        ctx.lineWidth = Math.round(1.5 * dpr);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
        ctx.stroke();

        // Glowing outer halo for tips
        if (isTip) {
          ctx.strokeStyle = 'rgba(0, 255, 255, 0.55)';
          ctx.lineWidth = Math.round(1.2 * dpr);
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, radius + Math.round(3.5 * dpr), 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // 3. Special Gestures Detection
      // Key Landmark coordinates in canvas space
      const pWrist = pts[0];
      const pThumbTip = pts[4];
      const pThumbMcp = pts[2];

      const pIndexMcp = pts[5];
      const pIndexPip = pts[6];
      const pIndexTip = pts[8];

      const pMiddleMcp = pts[9];
      const pMiddlePip = pts[10];
      const pMiddleTip = pts[12];

      const pRingPip = pts[14];
      const pRingTip = pts[16];

      const pPinkyPip = pts[18];
      const pPinkyTip = pts[20];

      const palmScale = Math.hypot(pWrist.x - pMiddleMcp.x, pWrist.y - pMiddleMcp.y) || 60;

      // Finger curl checks
      const isIndexCurled = Math.hypot(pIndexTip.x - pWrist.x, pIndexTip.y - pWrist.y) < Math.hypot(pIndexPip.x - pWrist.x, pIndexPip.y - pWrist.y) * 1.15;
      const isMiddleCurled = Math.hypot(pMiddleTip.x - pWrist.x, pMiddleTip.y - pWrist.y) < Math.hypot(pMiddlePip.x - pWrist.x, pMiddlePip.y - pWrist.y) * 1.15;
      const isRingCurled = Math.hypot(pRingTip.x - pWrist.x, pRingTip.y - pWrist.y) < Math.hypot(pRingPip.x - pWrist.x, pRingPip.y - pWrist.y) * 1.15;
      const isPinkyCurled = Math.hypot(pPinkyTip.x - pWrist.x, pPinkyTip.y - pWrist.y) < Math.hypot(pPinkyPip.x - pWrist.x, pPinkyPip.y - pWrist.y) * 1.15;
      const areFingersCurled = isIndexCurled && isMiddleCurled && isRingCurled && isPinkyCurled;

      // GESTURE 1: Closed Fist (Nắm tay lại) → Dấu cách (Space)
      const distThumbToIndex = Math.hypot(pThumbTip.x - pIndexMcp.x, pThumbTip.y - pIndexMcp.y);
      const isThumbTucked = distThumbToIndex < palmScale * 0.85;
      const isModelFist = topGesture?.categoryName === 'Closed_Fist';
      const isFist = isModelFist || (areFingersCurled && isThumbTucked);

      // GESTURE 2: Right Thumb Pointing Left (Ngón cái tay phải chỉ qua trái) → Trừ 1 ký tự (Backspace)
      const handedness = results.handedness && results.handedness[h] && results.handedness[h].length > 0 ? results.handedness[h][0] : null;
      const thumbDx = pThumbTip.x - pThumbMcp.x;
      const thumbDy = pThumbTip.y - pThumbMcp.y;
      const thumbLength = Math.hypot(thumbDx, thumbDy);
      const isRightHand = !handedness || handedness.categoryName.toLowerCase() === 'right' || handedness.score < 0.65;

      const isThumbPointingLeft =
        isRightHand &&
        !isFist &&
        thumbDx < -0.30 * palmScale &&
        Math.abs(thumbDx) > Math.abs(thumbDy) * 0.80 &&
        thumbLength > 0.40 * palmScale &&
        pThumbTip.x < pIndexMcp.x - 0.10 * palmScale &&
        isIndexCurled &&
        isMiddleCurled;

      if (isFist) {
        if (this.fistHoldStartTime === 0) {
          this.fistHoldStartTime = now;
        }
        const fistElapsed = now - this.fistHoldStartTime;
        if (fistElapsed >= 260 && !this.fistTriggered) {
          this.fistTriggered = true;
          this.onGestureActionCallbacks.forEach((cb) => cb({ action: 'space', timestamp: now }));
        }

        ctx.save();
        ctx.fillStyle = this.fistTriggered ? 'rgba(16, 185, 129, 0.28)' : 'rgba(0, 255, 255, 0.20)';
        ctx.beginPath();
        ctx.arc(pMiddleMcp.x, pMiddleMcp.y, Math.round(35 * dpr), 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = this.fistTriggered ? '#10B981' : '#00FFFF';
        ctx.lineWidth = Math.round(2 * dpr);
        ctx.beginPath();
        ctx.arc(pMiddleMcp.x, pMiddleMcp.y, Math.round(35 * dpr), 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = this.fistTriggered ? '#10B981' : '#00FFFF';
        ctx.font = `bold ${Math.round(11 * dpr)}px 'Inter', sans-serif`;
        ctx.textAlign = 'center';
        const fistText = this.fistTriggered ? '✊ CLOSED FIST → SPACE' : `✊ FIST HOLD (${Math.round(fistElapsed)}ms)`;
        ctx.fillText(fistText, pMiddleMcp.x, pMiddleMcp.y - Math.round(42 * dpr));
        ctx.restore();
      } else {
        this.fistHoldStartTime = 0;
        this.fistTriggered = false;
      }

      if (isThumbPointingLeft) {
        if (this.thumbLeftHoldStartTime === 0) {
          this.thumbLeftHoldStartTime = now;
        }
        const thumbElapsed = now - this.thumbLeftHoldStartTime;
        if (thumbElapsed >= 260 && !this.thumbLeftTriggered) {
          this.thumbLeftTriggered = true;
          this.onGestureActionCallbacks.forEach((cb) => cb({ action: 'backspace', timestamp: now }));
        }

        ctx.save();
        ctx.fillStyle = this.thumbLeftTriggered ? 'rgba(239, 68, 68, 0.28)' : 'rgba(245, 158, 11, 0.20)';
        ctx.beginPath();
        ctx.arc(pThumbTip.x, pThumbTip.y, Math.round(22 * dpr), 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = this.thumbLeftTriggered ? '#EF4444' : '#F59E0B';
        ctx.lineWidth = Math.round(2 * dpr);
        ctx.beginPath();
        ctx.arc(pThumbTip.x, pThumbTip.y, Math.round(22 * dpr), 0, Math.PI * 2);
        ctx.stroke();

        // Left Arrow at thumb tip (👈 Backspace)
        ctx.strokeStyle = this.thumbLeftTriggered ? '#EF4444' : '#F59E0B';
        ctx.lineWidth = Math.round(2.5 * dpr);
        ctx.beginPath();
        ctx.moveTo(pThumbTip.x - Math.round(10 * dpr), pThumbTip.y);
        ctx.lineTo(pThumbTip.x + Math.round(8 * dpr), pThumbTip.y);
        ctx.moveTo(pThumbTip.x - Math.round(10 * dpr), pThumbTip.y);
        ctx.lineTo(pThumbTip.x - Math.round(4 * dpr), pThumbTip.y - Math.round(5 * dpr));
        ctx.moveTo(pThumbTip.x - Math.round(10 * dpr), pThumbTip.y);
        ctx.lineTo(pThumbTip.x - Math.round(4 * dpr), pThumbTip.y + Math.round(5 * dpr));
        ctx.stroke();

        ctx.fillStyle = this.thumbLeftTriggered ? '#EF4444' : '#F59E0B';
        ctx.font = `bold ${Math.round(11 * dpr)}px 'Inter', sans-serif`;
        ctx.textAlign = 'right';
        const thumbText = this.thumbLeftTriggered ? '⌫ THUMB LEFT → BACKSPACE' : `⌫ HOLD THUMB LEFT (${Math.round(thumbElapsed)}ms)`;
        ctx.fillText(thumbText, pThumbTip.x - Math.round(16 * dpr), pThumbTip.y - Math.round(12 * dpr));
        ctx.restore();
      } else {
        this.thumbLeftHoldStartTime = 0;
        this.thumbLeftTriggered = false;
      }

      // 4. Pinch & Micro-Tap Detection for Morse Code
      // Measures distance between Thumb Tip (4) and Index Tip (8)
      const thumbTip = landmarks[4];
      const indexTip = landmarks[8];
      const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);

      const isPinching = (!isFist && !isThumbPointingLeft) && pinchDist < 0.082;
      const ptThumb = pts[4];
      const ptIndex = pts[8];
      const midX = (ptThumb.x + ptIndex.x) / 2;
      const midY = (ptThumb.y + ptIndex.y) / 2;

      if (isPinching) {
        if (!this.isHandTapping) {
          this.isHandTapping = true;
          this.handTapStartTime = now;
        }

        const elapsed = now - this.handTapStartTime;
        const isDash = elapsed >= 380;
        this.onPinchStateCallbacks.forEach((cb) => cb(true, elapsed));

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
          this.onPinchStateCallbacks.forEach((cb) => cb(false, duration));

          if (duration >= 75 && duration <= 2200) {
            const symbol = duration < 380 ? '.' : '-';
            this.onFingerTapCallbacks.forEach((cb) =>
              cb({ symbol, durationMs: duration, timestamp: now })
            );
          }
        }
      }

      // 4. Gesture Name Chip above Hand
      const ptWrist = pts[0];
      const wx = ptWrist.x;
      const wy = Math.max(Math.round(25 * dpr), ptWrist.y - Math.round(25 * dpr));

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
