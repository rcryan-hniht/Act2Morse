/**
 * WebSocket client for connecting to the Blink2Morse Python backend.
 * Provides fallback to client-side detection when backend is offline.
 */

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface BackendMessage {
  type: 'morse' | 'blink' | 'metrics' | 'pong' | 'status';
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
  return 'ws://localhost:8000/ws';
}

export class BlinkWebSocketBridge {
  private url: string;
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 4000;
  private shouldReconnect: boolean = true;
  private statusListeners: Array<(status: ConnectionStatus) => void> = [];
  private messageListeners: Array<(msg: BackendMessage) => void> = [];
  private currentStatus: ConnectionStatus = 'disconnected';

  constructor(url?: string) {
    this.url = url || resolveDefaultWsUrl();
  }

  public connect() {
    this.shouldReconnect = true;
    this.setStatus('connecting');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.setStatus('connected');
        this.send({ type: 'status', message: 'frontend_ready' });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as BackendMessage;
          this.messageListeners.forEach((cb) => cb(data));
        } catch {
          // Non-JSON message received
        }
      };

      this.ws.onclose = () => {
        this.setStatus('disconnected');
        if (this.shouldReconnect) {
          setTimeout(() => this.connect(), this.reconnectInterval);
        }
      };

      this.ws.onerror = () => {
        this.setStatus('disconnected');
      };
    } catch {
      this.setStatus('disconnected');
    }
  }

  public disconnect() {
    this.shouldReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.setStatus('disconnected');
  }

  public send(data: Record<string, unknown>) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  public sendFrame(canvas: HTMLCanvasElement) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
      this.send({ type: 'frame', image: dataUrl });
    }
  }

  public onStatusChange(callback: (status: ConnectionStatus) => void) {
    this.statusListeners.push(callback);
    callback(this.currentStatus);
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
