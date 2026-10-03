/**
 * Webcam Manager and Real-time Eye Blink Detection Engine
 * Uses hardware-accelerated video rendering with zero aspect distortion
 * and an offscreen canvas pipeline for retina-sharp HUD overlay and eye tracking.
 */

export interface BlinkEvent {
  durationMs: number;
  timestamp: number;
}

export interface CameraMetrics {
  fps: number;
  ear: number;
  isBlinking: boolean;
  baselineEar: number;
}

export class CameraController {
  private videoEl: HTMLVideoElement | null = null;
  private canvasEl: HTMLCanvasElement | null = null;
  private stream: MediaStream | null = null;
  private animFrameId: number | null = null;

  // Offscreen canvas for vision processing (320x240 for fast MediaPipe & local analysis)
  private offscreenCanvas: HTMLCanvasElement = document.createElement('canvas');
  private offscreenCtx: CanvasRenderingContext2D | null = null;

  private isRunning: boolean = false;
  private isBlinkActive: boolean = false;
  private blinkStartTime: number = 0;

  // Adaptive threshold calibration for local vision fallback
  private earHistory: number[] = [];
  private baselineEar: number = 0.30;
  private currentEar: number = 0.30;
  private blinkCloseRatio: number = 0.78; // Closes below 78% of baseline
  private blinkOpenRatio: number = 0.88;  // Reopens above 88% of baseline

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

  public setBackendState(connected: boolean, score: number | null, face: boolean, eyesClosed: boolean) {
    this.backendConnected = connected;
    this.backendScore = score;
    this.backendFace = face;
    this.backendEyesClosed = eyesClosed;
  }

  public onBlink(callback: (event: BlinkEvent) => void) {
    this.onBlinkCallbacks.push(callback);
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

    // Process frame
    this.processFrame();

    this.animFrameId = requestAnimationFrame(this.loop);
  };

  /**
   * Computes Eye Aspect Ratio estimation and streams frames for AI processing.
   * Keeps video proportions 100% natural and renders crisp retina HUD on top.
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

    // Clear onscreen canvas so hardware-accelerated natural <video> shines through
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

    // 3. Compute the visible portion of the video in the container (matching CSS object-fit: cover)
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

    // Region of Interest (ROI) for eye detection centered on visible face
    const roiX = Math.floor(startX + visibleOffscreenW * 0.22);
    const roiY = Math.floor(offH * 0.24);
    const roiW = Math.floor(visibleOffscreenW * 0.56);
    const roiH = Math.floor(offH * 0.28);

    const now = performance.now();

    try {
      const imageData = this.offscreenCtx.getImageData(roiX, roiY, roiW, roiH);
      const data = imageData.data;

      // Local vision analysis: measure left eye & right eye zones separately
      // Left eye band: 12% - 44% of face ROI; Right eye band: 56% - 88% of face ROI
      const leftX1 = Math.floor(roiW * 0.12);
      const leftX2 = Math.floor(roiW * 0.44);
      const rightX1 = Math.floor(roiW * 0.56);
      const rightX2 = Math.floor(roiW * 0.88);
      const eyeY1 = Math.floor(roiH * 0.25);
      const eyeY2 = Math.floor(roiH * 0.80);

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

      // Dark pupil / iris contrast count
      let darkIrisCount = 0;
      for (let y = eyeY1; y < eyeY2; y += step) {
        for (let x = 0; x < roiW; x += step) {
          const inLeftEye = x >= leftX1 && x <= leftX2;
          const inRightEye = x >= rightX1 && x <= rightX2;
          if (!inLeftEye && !inRightEye) continue;

          const idx = (y * roiW + x) * 4;
          const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
          if (lum < avgLum * 0.78) {
            darkIrisCount++;
          }
        }
      }

      const avgGradient = eyePixelCount > 0 ? verticalGradientSum / eyePixelCount : 0;
      const darkRatio = eyePixelCount > 0 ? darkIrisCount / eyePixelCount : 0;

      // Instantaneous EAR heuristic: combined eye vertical gradient & dark pupil presence
      const instantEar = Math.max(0.08, Math.min(0.60, (avgGradient / 45) * 0.65 + darkRatio * 1.8));

      // Fast responsiveness with smooth baseline tracking
      this.currentEar = 0.55 * this.currentEar + 0.45 * instantEar;

      // Update baseline when not blinking
      if (!this.isBlinkActive) {
        this.earHistory.push(this.currentEar);
        if (this.earHistory.length > 50) {
          this.earHistory.shift();
        }
        if (this.earHistory.length > 10) {
          const sorted = [...this.earHistory].sort((a, b) => a - b);
          this.baselineEar = sorted[Math.floor(sorted.length * 0.70)];
        }
      }

      const closeThreshold = Math.max(0.12, this.baselineEar * this.blinkCloseRatio);
      const openThreshold = Math.max(0.15, this.baselineEar * this.blinkOpenRatio);

      // Client-side blink state machine (active when backend is offline)
      if (!this.backendConnected) {
        if (!this.isBlinkActive && this.currentEar < closeThreshold) {
          this.isBlinkActive = true;
          this.blinkStartTime = now;
        } else if (this.isBlinkActive && this.currentEar > openThreshold) {
          this.isBlinkActive = false;
          const duration = Math.round(now - this.blinkStartTime);
          if (duration >= 130 && duration <= 1600) {
            this.onBlinkCallbacks.forEach((cb) => cb({ durationMs: duration, timestamp: now }));
          }
        }
      } else {
        // Backend connected: synchronize isBlinkActive with backend
        this.isBlinkActive = this.backendEyesClosed;
      }

      // 4. Draw HUD targeting brackets on the crisp onscreen canvas
      const hudX = Math.floor(targetW * 0.22);
      const hudY = Math.floor(targetH * 0.24);
      const hudW = Math.floor(targetW * 0.56);
      const hudH = Math.floor(targetH * 0.28);

      this.drawHUD(ctx, hudX, hudY, hudW, hudH, closeThreshold, dpr);

      // Emit metrics
      const metrics: CameraMetrics = {
        fps: this.currentFps,
        ear: Number(this.currentEar.toFixed(3)),
        isBlinking: this.isBlinkActive,
        baselineEar: Number(this.baselineEar.toFixed(3)),
      };
      this.onMetricsCallbacks.forEach((cb) => cb(metrics));

      // 5. Emit canvas frame for WebSocket streaming (~22 FPS throttle to match CPU inference)
      if (now - this.lastFrameSentTime >= 45) {
        this.lastFrameSentTime = now;
        this.onFrameCallbacks.forEach((cb) => cb(this.offscreenCanvas));
      }
    } catch {
      // Ignore canvas access errors if any
    }
  }

  /**
   * Draws a clean, minimalist tracking HUD over the natural video feed
   */
  private drawHUD(
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

    // Corner bracket colors
    let strokeColor = 'rgba(255, 255, 255, 0.45)';
    if (this.backendConnected && !isFaceDetected) {
      strokeColor = '#F59E0B'; // Amber: searching face
    } else if (isBlinking) {
      strokeColor = '#C9B8FF'; // Accent: eyes closed
    } else if (this.backendConnected) {
      strokeColor = '#10B981'; // Green: locked with MediaPipe AI
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

    // EAR Indicator Bar positioned cleanly above the floating bar
    const barW = Math.round(140 * dpr);
    const barH = Math.round(5 * dpr);
    const barX = Math.round(20 * dpr);
    const barY = ctx.canvas.height - Math.round(86 * dpr);

    // Score computation
    const displayScore = this.backendConnected
      ? (this.backendScore ?? 0)
      : this.currentEar;
    const displayThreshold = this.backendConnected ? 0.60 : threshold;
    const maxVal = this.backendConnected ? 1.0 : 0.45;

    // Label above EAR bar
    ctx.font = `500 ${Math.round(10 * dpr)}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    const statusPrefix = this.backendConnected
      ? (isFaceDetected ? (isBlinking ? 'AI: CLOSED' : 'AI: OPEN') : 'AI: NO FACE')
      : (isBlinking ? 'LOCAL: BLINK' : 'LOCAL: OPEN');
    ctx.fillText(`${statusPrefix} (${displayScore.toFixed(2)})`, barX, barY - Math.round(5 * dpr));

    // Background track
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, Math.round(2.5 * dpr));
    ctx.fill();

    // Progress
    const earPct = Math.min(1, Math.max(0, displayScore / maxVal));
    ctx.fillStyle = isBlinking ? '#C9B8FF' : '#FFFFFF';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW * earPct, barH, Math.round(2.5 * dpr));
    ctx.fill();

    // Threshold indicator line
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
