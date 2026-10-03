/**
 * Webcam Manager and Real-time Eye Blink Detection Engine
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
          width: { ideal: 640 },
          height: { ideal: 480 },
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
   * Computes Eye Aspect Ratio estimation using contrast and vertical gradient in eye zone
   */
  private processFrame() {
    if (!this.videoEl || !this.canvasEl) return;
    const ctx = this.canvasEl.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const width = this.canvasEl.width;
    const height = this.canvasEl.height;

    // Draw video to canvas (mirrored for intuitive webcam feel)
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(this.videoEl, -width, 0, width, height);
    ctx.restore();

    // Define ROI (Region Of Interest) centered around where human eyes typically sit
    // Usually top 35%-55% vertically, middle 25%-75% horizontally
    const roiX = Math.floor(width * 0.28);
    const roiY = Math.floor(height * 0.32);
    const roiW = Math.floor(width * 0.44);
    const roiH = Math.floor(height * 0.22);

    try {
      const imageData = ctx.getImageData(roiX, roiY, roiW, roiH);
      const data = imageData.data;

      // Calculate vertical edge gradient & contrast in eye ROI
      // Open eye = high edge contrast between pupil, iris, eyelashes and sclera
      // Closed eye = eyelid skin with low gradient
      let verticalGradientSum = 0;
      let pixelCount = 0;
      const step = 2; // Step by 2 pixels for fast performance

      for (let y = 0; y < roiH - 2; y += step) {
        for (let x = 0; x < roiW; x += step) {
          const idxCurrent = (y * roiW + x) * 4;
          const idxNext = ((y + 2) * roiW + x) * 4;

          // Grayscale luminosity
          const lum1 = 0.299 * data[idxCurrent] + 0.587 * data[idxCurrent + 1] + 0.114 * data[idxCurrent + 2];
          const lum2 = 0.299 * data[idxNext] + 0.587 * data[idxNext + 1] + 0.114 * data[idxNext + 2];

          verticalGradientSum += Math.abs(lum2 - lum1);
          pixelCount++;
        }
      }

      const avgGradient = pixelCount > 0 ? verticalGradientSum / pixelCount : 0;

      // Map average gradient to a normalized EAR value (around 0.15 - 0.40)
      const instantEar = Math.max(0.1, Math.min(0.5, avgGradient / 75));

      // Moving average for baseline calibration
      this.earHistory.push(instantEar);
      if (this.earHistory.length > 60) {
        this.earHistory.shift();
      }

      // Smooth current EAR to reduce noise
      this.currentEar = 0.7 * this.currentEar + 0.3 * instantEar;

      // Update baseline from the 80th percentile of recent history (eyes are open most of the time)
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
          // Trigger blink event
          if (duration >= 80 && duration <= 1800) {
            this.onBlinkCallbacks.forEach((cb) => cb({ durationMs: duration, timestamp: now }));
          }
        }
      }

      // Draw HUD graphics onto canvas
      this.drawHUD(ctx, roiX, roiY, roiW, roiH, threshold);

      // Emit metrics
      const metrics: CameraMetrics = {
        fps: this.currentFps,
        ear: Number(this.currentEar.toFixed(3)),
        isBlinking: this.isBlinkActive,
        baselineEar: Number(this.baselineEar.toFixed(3)),
      };
      this.onMetricsCallbacks.forEach((cb) => cb(metrics));

      // Emit canvas frame for WebSocket streaming if needed
      this.onFrameCallbacks.forEach((cb) => cb(this.canvasEl!));
    } catch {
      // Ignore canvas access errors if any
    }
  }

  /**
   * Draws a clean, minimalist tracking HUD over the video feed
   */
  private drawHUD(
    ctx: CanvasRenderingContext2D,
    rx: number,
    ry: number,
    rw: number,
    rh: number,
    threshold: number
  ) {
    ctx.save();

    // Eye ROI Box with subtle corners
    const cornerLen = 14;
    ctx.lineWidth = 2;
    ctx.strokeStyle = this.isBlinkActive ? '#C9B8FF' : 'rgba(255, 255, 255, 0.4)';

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
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy);
    ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8);
    ctx.lineTo(cx, cy + 8);
    ctx.stroke();

    // Pill badge indicating status at top of video
    const badgeText = this.isBlinkActive ? 'BLINK DETECTED' : 'EYE TRACKING';
    ctx.font = '600 11px system-ui, -apple-system, sans-serif';
    const textWidth = ctx.measureText(badgeText).width;
    const badgeW = textWidth + 24;
    const badgeH = 22;
    const badgeX = 14;
    const badgeY = 14;

    ctx.fillStyle = this.isBlinkActive ? 'rgba(201, 184, 255, 0.9)' : 'rgba(0, 0, 0, 0.65)';
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 11);
    ctx.fill();

    // Small status dot
    ctx.fillStyle = this.isBlinkActive ? '#111111' : '#10B981';
    ctx.beginPath();
    ctx.arc(badgeX + 11, badgeY + 11, 4, 0, Math.PI * 2);
    ctx.fill();

    // Text inside pill
    ctx.fillStyle = this.isBlinkActive ? '#111111' : '#FFFFFF';
    ctx.fillText(badgeText, badgeX + 20, badgeY + 15);

    // EAR Bar at bottom of HUD
    const barW = 120;
    const barH = 5;
    const barX = 14;
    const barY = ctx.canvas.height - 18;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW, barH, 2.5);
    ctx.fill();

    // Progress of EAR
    const earPct = Math.min(1, Math.max(0, this.currentEar / 0.45));
    ctx.fillStyle = this.isBlinkActive ? '#C9B8FF' : '#FFFFFF';
    ctx.beginPath();
    ctx.roundRect(barX, barY, barW * earPct, barH, 2.5);
    ctx.fill();

    // Threshold indicator line
    const threshX = barX + barW * Math.min(1, threshold / 0.45);
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(threshX, barY - 2);
    ctx.lineTo(threshX, barY + barH + 2);
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
