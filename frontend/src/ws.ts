/**
 * WebSocket client for connecting to the Act2Morse Python backend.
 * Provides binary frame streaming with backpressure control and handles
 * real-time MediaPipe eye-blink detection & Morse decoding events.
 */

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export type MorseEvent = 'dot' | 'dash' | 'letter_gap' | 'word_gap';

export interface BackendResponse {
  face: boolean;
  score: number | null;
  eyes_closed: boolean;
  events: MorseEvent[];
  symbols: string;
  text: string;
}

// Backward-compatible message interface
export interface BackendMessage {
  type?: 'morse' | 'blink' | 'metrics' | 'pong' | 'status' | 'frame' | 'reset';
  symbol?: '.' | '-';
  duration?: number;
  letter?: string;
  word?: string;
  fps?: number;
  ear?: number;
  message?: string;
}

function resolveDefaultWsUrl(): string {
  const envUrl = import.meta.env.VITE_WS_URL;
  if (typeof envUrl === 'string' && envUrl.trim().length > 0) {
    return envUrl.trim();
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname || 'localhost';
  return `${protocol}//${host}:8000/ws`;
}

export class BlinkWebSocketBridge {
  private primaryUrl: string;
  private currentUrl: string;
  private fallbackUrl: string | null = null;
  private failedAttempts: number = 0;
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 3500;
  private shouldReconnect: boolean = true;
  private statusListeners: Array<(status: ConnectionStatus) => void> = [];
  private responseListeners: Array<(res: BackendResponse) => void> = [];
  private messageListeners: Array<(msg: BackendMessage) => void> = [];
  private currentStatus: ConnectionStatus = 'disconnected';

  // Backpressure: only send next frame once previous reply is received or timed out
  private isAwaitingResponse: boolean = false;
  private responseTimeoutId: number | null = null;

  constructor(url?: string) {
    this.primaryUrl = url || resolveDefaultWsUrl();
    this.currentUrl = this.primaryUrl;
    if (this.primaryUrl.includes('act2morse.onrender.com')) {
      this.fallbackUrl = this.primaryUrl.replace('act2morse.onrender.com', 'blink2morse.onrender.com');
    }
  }

  public isConnected(): boolean {
    return this.currentStatus === 'connected' && this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  public connect() {
    this.shouldReconnect = true;
    this.setStatus('connecting');

    try {
      this.ws = new WebSocket(this.currentUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        this.setStatus('connected');
        this.isAwaitingResponse = false;
        this.failedAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        this.isAwaitingResponse = false;
        if (this.responseTimeoutId !== null) {
          clearTimeout(this.responseTimeoutId);
          this.responseTimeoutId = null;
        }

        try {
          if (typeof event.data === 'string') {
            const parsed = JSON.parse(event.data);
            if ('face' in parsed && 'events' in parsed) {
              const res = parsed as BackendResponse;
              this.responseListeners.forEach((cb) => cb(res));
            }
            this.messageListeners.forEach((cb) => cb(parsed as BackendMessage));
          }
        } catch {
          // Non-JSON message received
        }
      };

      this.ws.onclose = () => {
        this.setStatus('disconnected');
        this.isAwaitingResponse = false;
        this.failedAttempts++;

        // If primary url fails to connect 2 times, fallback to legacy URL until primary comes alive
        if (this.failedAttempts >= 2 && this.fallbackUrl && this.currentUrl === this.primaryUrl) {
          this.currentUrl = this.fallbackUrl;
        } else if (this.failedAttempts >= 6 && this.currentUrl === this.fallbackUrl) {
          // Periodically re-try primary
          this.currentUrl = this.primaryUrl;
          this.failedAttempts = 0;
        }

        if (this.shouldReconnect) {
          setTimeout(() => this.connect(), this.reconnectInterval);
        }
      };

      this.ws.onerror = () => {
        this.setStatus('disconnected');
        this.isAwaitingResponse = false;
      };
    } catch {
      this.setStatus('disconnected');
      this.isAwaitingResponse = false;
    }
  }

  public disconnect() {
    this.shouldReconnect = false;
    this.isAwaitingResponse = false;
    if (this.responseTimeoutId !== null) {
      clearTimeout(this.responseTimeoutId);
      this.responseTimeoutId = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }

  public send(data: Record<string, unknown>) {
    if (this.isConnected()) {
      try {
        this.ws?.send(JSON.stringify(data));
      } catch {
        // Ignored
      }
    }
  }

  public sendReset() {
    this.send({ type: 'reset' });
  }

  public sendSymbol(symbol: '.' | '-') {
    this.send({ type: 'symbol', symbol });
  }

  public sendSpace() {
    this.send({ type: 'space' });
  }

  /**
   * Stream a video frame canvas to backend as binary JPEG with backpressure.
   * Drops frame if previous inference is still in-flight to prevent lag.
   */
  public sendFrame(canvas: HTMLCanvasElement): boolean {
    if (!this.isConnected() || this.isAwaitingResponse) {
      return false;
    }

    this.isAwaitingResponse = true;

    // Safety timeout in case server drops connection or message
    if (this.responseTimeoutId !== null) {
      clearTimeout(this.responseTimeoutId);
    }
    this.responseTimeoutId = window.setTimeout(() => {
      this.isAwaitingResponse = false;
      this.responseTimeoutId = null;
    }, 1200);

    canvas.toBlob(
      (blob) => {
        if (!blob || !this.isConnected()) {
          this.isAwaitingResponse = false;
          return;
        }
        try {
          this.ws?.send(blob);
        } catch {
          this.isAwaitingResponse = false;
        }
      },
      'image/jpeg',
      0.65
    );

    return true;
  }

  public onStatusChange(callback: (status: ConnectionStatus) => void) {
    this.statusListeners.push(callback);
    callback(this.currentStatus);
  }

  public onResponse(callback: (res: BackendResponse) => void) {
    this.responseListeners.push(callback);
  }

  public onMessage(callback: (msg: BackendMessage) => void) {
    this.messageListeners.push(callback);
  }

  private setStatus(status: ConnectionStatus) {
    this.currentStatus = status;
    this.statusListeners.forEach((cb) => cb(status));
  }

  public getStatus(): ConnectionStatus {
    return this.currentStatus;
  }
}

export const wsBridge = new BlinkWebSocketBridge();
