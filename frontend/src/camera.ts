/**
 * Webcam Manager and Real-time Gesture Engine for Act2Morse.
 * Supports dual-mode vision processing:
 *  - Fin2Morse: Pure Hand & Finger Micro-Tap Tracking (completely ignores face & eyes).
 *  - Blink2Morse: Eye Blink Tracking via facial landmarks / local EAR vision.
 * Uses hardware-accelerated video rendering and an offscreen canvas pipeline for retina-sharp HUD overlay.
 */

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
}

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
  private handBox: { x: number; y: number; w: number; h: number } | null = null;
  private fingertipPoint: { x: number; y: number } | null = null;
  private smoothFingertipY: number = 0;
  private handBaselineY: number = 0;
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

  constructor() {}

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
      this.smoothFingertipY = 0;
      this.handBaselineY = 0;

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
    this.smoothFingertipY = 0;
    this.handBaselineY = 0;
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

    // Process frame according to active mode
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

    // 1. Synchronize the onscreen canvas resolution with container dimensions (HiDPI)
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

    // Clear onscreen canvas so natural <video> shines through
    ctx.clearRect(0, 0, targetW, targetH);

    // 2. Offscreen canvas for computer vision processing
    const offW = 320;
    const offH = 240;
    if (this.offscreenCanvas.width !== offW || this.offscreenCanvas.height !== offH) {
      this.offscreenCanvas.width = offW;
      this.offscreenCanvas.height = offH;
      this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });
    }
    if (!this.offscreenCtx) return;

    // Draw video to offscreen canvas (mirrored for natural webcam preview)
    this.offscreenCtx.save();
    this.offscreenCtx.scale(-1, 1);
    this.offscreenCtx.drawImage(this.videoEl, -offW, 0, offW, offH);
    this.offscreenCtx.restore();

    // 3. Compute visible portion of the video in the container (matching CSS object-fit: cover)
    const videoW = this.videoEl.videoWidth || 640;
    const videoH = this.videoEl.videoHeight || 480;
    const videoRatio = videoW / videoH;
    const containerRatio = rect.width / rect.height;

    let visibleRatio = 1.0;
    let cropOffsetRatio = 0.0;

    if (videoRatio > containerRatio) {
      visibleRatio = containerRatio / videoRatio;
      cropOffsetRatio = (1 - visibleRatio) / 2;
    }

    const visibleOffscreenW = offW * visibleRatio;
    const startX = offW * cropOffsetRatio;
    const now = performance.now();

    // 4. Branch based on Mode: Fin2Morse tracks HAND ONLY, Blink2Morse tracks EYES ONLY
    if (this.currentMode === 'Fin2Morse') {
      this.processHandFrame(ctx, targetW, targetH, dpr, now);
    } else {
      this.processEyeFrame(ctx, targetW, targetH, dpr, now, startX, visibleOffscreenW);
    }
  }

  /**
   * Fin2Morse Hand Tracking: Analyzes video frame for Hand presence and Finger micro-taps.
   * Completely ignores face and eyes.
   */
  private processHandFrame(
    ctx: CanvasRenderingContext2D,
    targetW: number,
    targetH: number,
    dpr: number,
    now: number
  ) {
    if (!this.offscreenCtx) return;

    const offW = 320;
    const offH = 240;

    // Hand Region of Interest: center and lower portion where hands and fingers naturally gesture
    const roiX = Math.floor(offW * 0.12);
    const roiY = Math.floor(offH * 0.12);
    const roiW = Math.floor(offW * 0.76);
    const roiH = Math.floor(offH * 0.80);

    try {
      const imageData = this.offscreenCtx.getImageData(roiX, roiY, roiW, roiH);
      const data = imageData.data;

      let skinCount = 0;
      let minX = roiW, maxX = 0, minY = roiH, maxY = 0;
      let topY = roiH;
      let topX = Math.floor(roiW / 2);
      const step = 2;

      for (let y = 0; y < roiH; y += step) {
        for (let x = 0; x < roiW; x += step) {
          const idx = (y * roiW + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];

          // Lighting-tolerant human skin chrominance test (YCbCr + RGB)
          const yLum = 0.299 * r + 0.587 * g + 0.114 * b;
          const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
          const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;

          const isSkin =
            (yLum > 35 && cb >= 77 && cb <= 132 && cr >= 130 && cr <= 175) ||
            (r > 70 && g > 35 && b > 20 && r > g && r > b && (r - g) > 10);

          if (isSkin) {
            skinCount++;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;

            // Track highest skin pixel cluster (fingertip)
            if (y < topY) {
              topY = y;
              topX = x;
            }
          }
        }
      }

      const totalSamples = (roiW / step) * (roiH / step);
      const skinRatio = skinCount / totalSamples;

      // Detection threshold: at least ~220 skin pixels in hand zone
      if (skinCount >= 220 && maxX > minX && maxY > minY) {
        this.isHandDetected = true;
        this.handConfidence = Math.min(1.0, skinRatio * 4.5);

        // Map coordinates back to canvas dimensions
        const boxX = ((roiX + minX) / offW) * targetW;
        const boxY = ((roiY + minY) / offH) * targetH;
        const boxW = ((maxX - minX) / offW) * targetW;
        const boxH = ((maxY - minY) / offH) * targetH;
        this.handBox = { x: boxX, y: boxY, w: boxW, h: boxH };

        const tipCanvasX = ((roiX + topX) / offW) * targetW;
        const tipCanvasY = ((roiY + topY) / offH) * targetH;
        this.fingertipPoint = { x: tipCanvasX, y: tipCanvasY };

        // Smooth fingertip position
        if (this.smoothFingertipY === 0) {
          this.smoothFingertipY = topY;
          this.handBaselineY = topY;
        } else {
          this.smoothFingertipY = 0.65 * this.smoothFingertipY + 0.35 * topY;
        }

        // Tap tracking: detect downward micro-tap motion of fingertip
        if (!this.isHandTapping) {
          this.handBaselineY = 0.96 * this.handBaselineY + 0.04 * this.smoothFingertipY;
          const downwardDip = this.smoothFingertipY - this.handBaselineY;

          // Downward motion threshold of fingertip
          if (downwardDip > 7.5) {
            this.isHandTapping = true;
            this.handTapStartTime = now;
          }
        } else {
          const elapsed = now - this.handTapStartTime;
          const upwardReturn = this.handBaselineY - this.smoothFingertipY;

          if (elapsed > 1600) {
            // Auto-recover if held too long
            this.isHandTapping = false;
            this.handBaselineY = this.smoothFingertipY;
          } else if (upwardReturn > -3.5) {
            // Fingertip released back up
            this.isHandTapping = false;
            this.handBaselineY = this.smoothFingertipY;
            const duration = Math.round(elapsed);
            if (duration >= 75 && duration <= 1500) {
              const symbol = duration < 380 ? '.' : '-';
              this.onFingerTapCallbacks.forEach((cb) =>
                cb({ symbol, durationMs: duration, timestamp: now })
              );
            }
          }
        }

        this.drawHandHUD(ctx, targetW, targetH, dpr, now);
      } else {
        this.isHandDetected = false;
        this.isHandTapping = false;
        this.handConfidence = 0;
        this.handBox = null;
        this.fingertipPoint = null;
        this.drawHandSearchHUD(ctx, targetW, targetH, dpr);
      }

      // Emit metrics
      const metrics: CameraMetrics = {
        fps: this.currentFps,
        mode: 'Fin2Morse',
        ear: 0,
        isBlinking: false,
        baselineEar: 0,
        isHandDetected: this.isHandDetected,
        isFingerTapping: this.isHandTapping,
        handConfidence: Number(this.handConfidence.toFixed(2)),
      };
      this.onMetricsCallbacks.forEach((cb) => cb(metrics));
    } catch {
      // Ignore canvas read errors
    }
  }

  /**
   * Draws hand tracking HUD with fingertip target and tap duration indicator
   */
  private drawHandHUD(
    ctx: CanvasRenderingContext2D,
    _targetW: number,
    targetH: number,
    dpr: number,
    now: number
  ) {
    if (!this.handBox) return;
    ctx.save();

    const { x, y, w, h } = this.handBox;
    const isTapping = this.isHandTapping;
    const elapsed = isTapping ? now - this.handTapStartTime : 0;
    const isDash = isTapping && elapsed >= 380;

    // Corner brackets color
    const strokeColor = isTapping ? (isDash ? '#111111' : '#8B5CF6') : '#10B981';
    const cornerLen = Math.round(20 * dpr);
    ctx.lineWidth = Math.round(2.5 * dpr);
    ctx.strokeStyle = strokeColor;

    // Top-left
    ctx.beginPath();
    ctx.moveTo(x, y + cornerLen);
    ctx.lineTo(x, y);
    ctx.lineTo(x + cornerLen, y);
    ctx.stroke();

    // Top-right
    ctx.beginPath();
    ctx.moveTo(x + w - cornerLen, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + cornerLen);
    ctx.stroke();

    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(x, y + h - cornerLen);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + cornerLen, y + h);
    ctx.stroke();

    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(x + w - cornerLen, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y + h - cornerLen);
    ctx.stroke();

    // Fingertip tracker circle
    if (this.fingertipPoint) {
      const fx = this.fingertipPoint.x;
      const fy = this.fingertipPoint.y;
      const radius = isTapping ? Math.round(14 * dpr) : Math.round(9 * dpr);

      ctx.fillStyle = isTapping ? (isDash ? '#111111' : '#C9B8FF') : 'rgba(201, 184, 255, 0.4)';
      ctx.beginPath();
      ctx.arc(fx, fy, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = isTapping ? '#FFFFFF' : '#8B5CF6';
      ctx.lineWidth = Math.round(2 * dpr);
      ctx.beginPath();
      ctx.arc(fx, fy, radius + Math.round(4 * dpr), 0, Math.PI * 2);
      ctx.stroke();

      // Tap status label above fingertip
      ctx.font = `600 ${Math.round(11 * dpr)}px 'Inter', sans-serif`;
      ctx.fillStyle = '#FFFFFF';
      ctx.textAlign = 'center';
      if (isTapping) {
        const tapLabel = isDash ? '— DASH (>380ms)' : '• DOT (<380ms)';
        ctx.fillText(tapLabel, fx, fy - radius - Math.round(8 * dpr));
      } else {
        ctx.fillText('FINGERTIP', fx, fy - radius - Math.round(6 * dpr));
      }
      ctx.textAlign = 'left';
    }

    // Bottom Status Badge on Video Feed
    const barW = Math.round(160 * dpr);
    const barH = Math.round(5 * dpr);
    const barX = Math.round(20 * dpr);
    const barY = targetH - Math.round(86 * dpr);

    ctx.font = `600 ${Math.round(10 * dpr)}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    const statusText = isTapping
      ? (isDash ? 'FIN: DASH HOLD (—)' : 'FIN: DOT TAP (•)')
      : 'FIN: HAND LOCKED (READY)';
    ctx.fillText(statusText, barX, barY - Math.round(5 * dpr));

    // Progress bar
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, Math.round(2.5 * dpr));
    ctx.fill();

    const progressPct = isTapping ? Math.min(1.0, elapsed / 380) : this.handConfidence;
    ctx.fillStyle = isTapping ? (isDash ? '#111111' : '#C9B8FF') : '#10B981';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW * progressPct, barH, Math.round(2.5 * dpr));
    ctx.fill();

    ctx.restore();
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

    // Subtle dashed amber frame
    ctx.setLineDash([Math.round(8 * dpr), Math.round(6 * dpr)]);
    ctx.lineWidth = Math.round(1.5 * dpr);
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
    ctx.strokeRect(boxX, boxY, boxW, boxH);
    ctx.setLineDash([]);

    // Prompt Text
    ctx.textAlign = 'center';
    ctx.font = `600 ${Math.round(13 * dpr)}px 'Inter', sans-serif`;
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText('SHOW HAND TO CAMERA', targetW / 2, boxY + boxH / 2 - Math.round(6 * dpr));

    ctx.font = `400 ${Math.round(10 * dpr)}px 'Inter', sans-serif`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.fillText('Hold up your hand / finger to tap Morse', targetW / 2, boxY + boxH / 2 + Math.round(14 * dpr));

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
    now: number,
    startX: number,
    visibleOffscreenW: number
  ) {
    if (!this.offscreenCtx) return;
    const offH = 240;

    // Region of Interest (ROI) for eye detection centered on visible face
    const roiX = Math.floor(startX + visibleOffscreenW * 0.20);
    const roiY = Math.floor(offH * 0.20);
    const roiW = Math.floor(visibleOffscreenW * 0.60);
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
   * Draws a clean, minimalist tracking HUD over the eye/face region in Blink2Morse mode
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

    // Center subtle crosshair
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
      ? (isFaceDetected ? (isBlinking ? 'AI: CLOSED' : 'AI: OPEN') : 'AI: NO FACE')
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
