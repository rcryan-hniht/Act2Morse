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

  // Offscreen canvas for vision processing (never stretched in DOM)
  private offscreenCanvas: HTMLCanvasElement = document.createElement('canvas');
  private offscreenCtx: CanvasRenderingContext2D | null = null;

  private isRunning: boolean = false;
  private isBlinkActive: boolean = false;
  private blinkStartTime: number = 0;

  // Adaptive threshold calibration
  private earHistory: number[] = [];
  private baselineEar: number = 0.32;
  private currentEar: number = 0.32;
  private blinkThresholdRatio: number = 0.72; // Blink triggers when EAR < 72% of baseline

  private lastFrameTime: number = performance.now();
  private fpsCounter: number = 0;
  private currentFps: number = 30;

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
   * Computes Eye Aspect Ratio estimation using contrast and vertical gradient in eye zone.
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

    // CLEAR onscreen canvas so hardware-accelerated natural <video> shines through
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

    // Draw video to offscreen canvas (mirrored)
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
      // Video is wider than container: horizontal edges are cropped
      visibleRatio = containerRatio / videoRatio;
      cropOffsetRatio = (1 - visibleRatio) / 2;
    }

    const visibleOffscreenW = offW * visibleRatio;
    const startX = offW * cropOffsetRatio;

    // Region of Interest (ROI) for eye detection centered on visible face
    const roiX = Math.floor(startX + visibleOffscreenW * 0.24);
    const roiY = Math.floor(offH * 0.28);
    const roiW = Math.floor(visibleOffscreenW * 0.52);
    const roiH = Math.floor(offH * 0.24);

    try {
      const imageData = this.offscreenCtx.getImageData(roiX, roiY, roiW, roiH);
      const data = imageData.data;

      // Calculate vertical edge gradient & contrast in eye ROI
      let verticalGradientSum = 0;
      let pixelCount = 0;
      const step = 2;

      for (let y = 0; y < roiH - 2; y += step) {
        for (let x = 0; x < roiW; x += step) {
          const idxCurrent = (y * roiW + x) * 4;
          const idxNext = ((y + 2) * roiW + x) * 4;

          const lum1 = 0.299 * data[idxCurrent] + 0.587 * data[idxCurrent + 1] + 0.114 * data[idxCurrent + 2];
          const lum2 = 0.299 * data[idxNext] + 0.587 * data[idxNext + 1] + 0.114 * data[idxNext + 2];

          verticalGradientSum += Math.abs(lum2 - lum1);
          pixelCount++;
        }
      }

      const avgGradient = pixelCount > 0 ? verticalGradientSum / pixelCount : 0;
      const instantEar = Math.max(0.1, Math.min(0.5, avgGradient / 75));

      // Moving average for baseline calibration
      this.earHistory.push(instantEar);
      if (this.earHistory.length > 60) {
        this.earHistory.shift();
      }

      this.currentEar = 0.7 * this.currentEar + 0.3 * instantEar;

      const sortedHistory = [...this.earHistory].sort((a, b) => a - b);
      if (sortedHistory.length > 15) {
        this.baselineEar = sortedHistory[Math.floor(sortedHistory.length * 0.75)];
      }

      const threshold = Math.max(0.18, this.baselineEar * this.blinkThresholdRatio);
      const now = performance.now();

      // State machine for blink detection
      if (this.currentEar < threshold) {
        if (!this.isBlinkActive) {
          this.isBlinkActive = true;
          this.blinkStartTime = now;
        }
      } else {
        if (this.isBlinkActive) {
          this.isBlinkActive = false;
          const duration = Math.round(now - this.blinkStartTime);
          if (duration >= 80 && duration <= 1800) {
            this.onBlinkCallbacks.forEach((cb) => cb({ durationMs: duration, timestamp: now }));
          }
        }
      }

      // 4. Draw HUD targeting brackets on the crisp onscreen canvas
      const hudX = Math.floor(targetW * 0.24);
      const hudY = Math.floor(targetH * 0.28);
      const hudW = Math.floor(targetW * 0.52);
      const hudH = Math.floor(targetH * 0.24);

      this.drawHUD(ctx, hudX, hudY, hudW, hudH, threshold, dpr);

      // Emit metrics
      const metrics: CameraMetrics = {
        fps: this.currentFps,
        ear: Number(this.currentEar.toFixed(3)),
        isBlinking: this.isBlinkActive,
        baselineEar: Number(this.baselineEar.toFixed(3)),
      };
      this.onMetricsCallbacks.forEach((cb) => cb(metrics));

      // Emit canvas frame for WebSocket streaming if needed
      this.onFrameCallbacks.forEach((cb) => cb(this.offscreenCanvas));
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

    // Eye ROI Box with subtle elegant corners
    const cornerLen = Math.round(18 * dpr);
    ctx.lineWidth = Math.round(2 * dpr);
    ctx.strokeStyle = this.isBlinkActive ? '#C9B8FF' : 'rgba(255, 255, 255, 0.45)';

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
    const barW = Math.round(130 * dpr);
    const barH = Math.round(5 * dpr);
    const barX = Math.round(20 * dpr);
    const barY = ctx.canvas.height - Math.round(86 * dpr);

    // Label above EAR bar
    ctx.font = `500 ${Math.round(10 * dpr)}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.fillText(`EAR: ${this.currentEar.toFixed(2)}`, barX, barY - Math.round(5 * dpr));

    // Background track
    ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, Math.round(2.5 * dpr));
    ctx.fill();

    // Progress
    const earPct = Math.min(1, Math.max(0, this.currentEar / 0.45));
    ctx.fillStyle = this.isBlinkActive ? '#C9B8FF' : '#FFFFFF';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW * earPct, barH, Math.round(2.5 * dpr));
    ctx.fill();

    // Threshold indicator line
    const threshX = barX + barW * Math.min(1, threshold / 0.45);
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
