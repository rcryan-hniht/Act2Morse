/**
 * Web Audio API synthesizer for authentic Morse Code audio beeps.
 */

class MorseSoundEngine {
  private ctx: AudioContext | null = null;
  private isEnabled: boolean = true;
  private frequency: number = 750; // Classic 750Hz telegraph tone

  constructor() {
    // AudioContext will be initialized on first user interaction
  }

  private initContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  public setEnabled(enabled: boolean) {
    this.isEnabled = enabled;
  }

  public getEnabled(): boolean {
    return this.isEnabled;
  }

  public playTone(durationMs: number, freq: number = this.frequency) {
    if (!this.isEnabled) return;
    try {
      this.initContext();
      if (!this.ctx) return;

      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      const startTime = this.ctx.currentTime;
      const endTime = startTime + durationMs / 1000;

      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, startTime);

      // Smooth attack & decay to prevent clicking sounds
      gain.gain.setValueAtTime(0, startTime);
      gain.gain.linearRampToValueAtTime(0.2, startTime + 0.008);
      gain.gain.setValueAtTime(0.2, endTime - 0.008);
      gain.gain.linearRampToValueAtTime(0, endTime);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(startTime);
      osc.stop(endTime);
    } catch {
      // Audio playback might be prevented before user gesture
    }
  }

  public playDot() {
    this.playTone(85, 780);
  }

  public playDash() {
    this.playTone(260, 780);
  }

  public playCharacterComplete() {
    this.playTone(45, 520);
  }

  public playWordSpace() {
    this.playTone(60, 440);
  }

  public playBackspace() {
    this.playTone(70, 340);
  }
}

export const morseAudio = new MorseSoundEngine();
