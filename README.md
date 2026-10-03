# Act2Morse

> **Real-time Assistive Communication Platform for Non-Verbal, Deaf-Mute, and Motor-Impaired Individuals**  
> _Transforming Micro Physical Actions into Living Words via Morse Code_

---

## 🌟 Mission & Overview

Communication is a fundamental human right. However, millions of people worldwide living with **mutism, deafness, ALS (Amyotrophic Lateral Sclerosis), locked-in syndrome, cerebral palsy, or severe motor disabilities** face immense challenges in expressing their daily thoughts, needs, and feelings to the world.

**Act2Morse** was created as an open, accessible, privacy-first assistive platform that bridges this gap. By utilizing standard webcams and modern computer vision, Act2Morse translates minimal physical gestures into **International Morse Code**, which is then automatically decoded in real time into clear alphanumeric text and natural audio:

1. **Blink2Morse (Ocular Mode)**: For non-verbal individuals with limited or no limb mobility. Micro eye blinks are captured via webcam and classified into Morse dots and dashes.
2. **Fin2Morse (Tactile / Fine-Motor Mode)**: For deaf, mute, or motor-impaired individuals who retain partial finger dexterity. Subtle taps or assistive switch presses are translated into Morse sequences.

---

## ✨ Key Features

- **Hands-Free Eye Tracking (Blink2Morse)**:
  - Powered by **MediaPipe Face Landmarker** running at 25–30 FPS with high-precision blendshape extraction (`eyeBlinkLeft`, `eyeBlinkRight`).
  - Adaptive thresholding distinguishing natural involuntary blinks from intentional Morse signals.
  - **Dot (`•`)**: Short deliberate blink (80ms – 600ms).
  - **Dash (`—`)**: Long deliberate closure (> 600ms).
  - **Automatic Letter Gap**: Pausing with eyes open for ~1.2s automatically finishes the current letter.
  - **Automatic Word Space**: Pausing with eyes open for ~2.5s inserts a word boundary space.

- **Tactile Micro-Tap Mode (Fin2Morse)**:
  - Designed for tactile communication using assistive switches, touchscreens, or keyboard keys.
  - Press duration timing: quick tap (< 380ms) for Dot, sustained press (> 380ms) for Dash.

- **Dual-Engine Architecture (Cloud/Server AI + Local Fallback)**:
  - **AI Backend Mode**: Streams video frames over a low-latency WebSocket to a Python FastAPI backend running MediaPipe for sub-millimeter landmark precision.
  - **Client-Side Vision Fallback**: If the server is offline, an in-browser computer vision heuristic monitors eye contrast and pupil gradient locally without requiring an active server connection.

- **Auditory & Visual Feedback**:
  - Web Audio API tone synthesizer providing distinct acoustic beeps for dots (high pitch), dashes, and character completion.
  - Live Retinal HUD with real-time Eye Aspect Ratio (EAR) metric bar, face-locking targeting reticle, and blink status indicators.

- **Interactive Side Morse Chart**:
  - Full international alphanumeric reference chart (A–Z, 0–9) placed directly beside the live camera feed for immediate practice and reference.
  - Tap any symbol card to hear its acoustic Morse sequence.

- **100% Privacy-Preserving**:
  - Processing happens strictly on-device or across your private local network. Video feeds are analyzed in-memory and discarded frame-by-frame; no footage or biometric data is ever stored or transmitted to external third parties.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Act2Morse Web App                    │
│                 (Vite + TypeScript)                    │
│                                                        │
│   Webcam Feed ──► Canvas Capture ──► Web Audio Beeps   │
│         │                                    ▲         │
│         ▼ (Binary JPEG Blob via WebSocket)   │         │
└─────────┼────────────────────────────────────┼─────────┘
          │                                    │
          ▼                                    │
┌──────────────────────────────────────────────┴─────────┐
│                 Act2Morse Backend                      │
│                (FastAPI + Python)                      │
│                                                        │
│   EyeDetector: MediaPipe Face Landmarker (VIDEO mode)  │
│   BlinkTracker: Timing state machine (Dot/Dash/Gaps)   │
│   MorseDecoder: Alphanumeric symbol buffer             │
└────────────────────────────────────────────────────────┘
```

---

#  Installation

### Prerequisites

```bash
git clone https://github.com/rcryan-hniht/Act2Morse
```

- **Node.js** (v18+) & **pnpm** (or npm)
- **Python** (v3.11+) & **uv** (or pip)
- A working webcam

---

### 1. Backend Setup (FastAPI & MediaPipe)

```bash
# Navigate to the backend directory
cd backend

# Install dependencies and sync environment with uv
uv sync

# Start the WebSocket inference server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend server will download the MediaPipe `face_landmarker.task` model automatically on first launch and serve the WebSocket endpoint at:
`ws://localhost:8000/ws`

---

### 2. Frontend Setup (Vite & TypeScript)

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
pnpm install

# Start the development server
pnpm dev
```

Open your browser at **`http://localhost:5173`**.

---

## 📖 How to Communicate with Act2Morse

1. Click **Start Camera** (or **Launch Camera**) on the navbar or video overlay and grant camera permission.
2. The HUD overlay will lock onto your face, displaying **`AI TRACKING • <score>`** in green.
3. Select your preferred mode from the top navigation bar:
   - **Blink2Morse**: Eye blink communication.
   - **Fin2Morse**: Finger / Spacebar tactile tap communication.
4. **Blinking rules (Blink2Morse)**:
   - **Dot (`•`)**: Blink quickly and deliberately (~150ms – 400ms) and open your eyes. You will hear a short beep.
   - **Dash (`—`)**: Close your eyes for ~0.8s – 1.0s and open them. You will hear a longer beep.
   - **Letter Completion**: Keep your eyes open for **1.2 seconds**. The dots and dashes will assemble into a letter (e.g., `• —` becomes `A`).
   - **Word Space**: Keep your eyes open for **2.5 seconds** to append a space between words.
5. Click the **Transmit (↑)** button to copy the decoded sentence to your clipboard.

---

## 🎛️ Configuration & Tuning

Backend detection parameters can be customized via environment variables:

| Variable          | Default | Description                                                      |
| :---------------- | :------ | :--------------------------------------------------------------- |
| `CLOSE_THRESHOLD` | `0.5`   | Blendshape closure score to trigger eye-closed state (0.0 – 1.0) |
| `OPEN_THRESHOLD`  | `0.25`  | Blendshape score to trigger eye-open state (hysteresis gap)      |
| `MIN_BLINK_MS`    | `80`    | Minimum blink duration in milliseconds (filters micro-glitches)  |
| `DOT_MAX_MS`      | `600`   | Maximum duration for a Dot; closures exceeding this are Dashes   |
| `LETTER_GAP_MS`   | `1200`  | Open-eye pause duration to finalize a character                  |
| `WORD_GAP_MS`     | `2500`  | Open-eye pause duration to insert a word space                   |

---

## 📜 International Morse Code Reference

| Char  | Morse  | Char  | Morse  | Char  | Morse  | Digit | Morse   |
| :---: | :----- | :---: | :----- | :---: | :----- | :---: | :------ |
| **A** | `.-`   | **J** | `.---` | **S** | `...`  | **1** | `.----` |
| **B** | `-...` | **K** | `-.-`  | **T** | `-`    | **2** | `..---` |
| **C** | `-.-.` | **L** | `.-..` | **U** | `..-`  | **3** | `...--` |
| **D** | `-..`  | **M** | `--`   | **V** | `...-` | **4** | `....-` |
| **E** | `.`    | **N** | `-.`   | **W** | `.--`  | **5** | `.....` |
| **F** | `..-.` | **O** | `---`  | **X** | `-..-` | **6** | `-....` |
| **G** | `--.`  | **P** | `.--.` | **Y** | `-.--` | **7** | `--...` |
| **H** | `....` | **Q** | `--.-` | **Z** | `--..` | **8** | `---..` |
| **I** | `..`   | **R** | `.-.`  |       |        | **9** | `----.` |
|       |        |       |        |       |        | **0** | `-----` |

---

## 🤝 Open Source & Accessibility

Act2Morse is built with the belief that open technology can tear down barriers to human connection. If you are an accessibility researcher, occupational therapist, or software engineer interested in contributing, pull requests and issues are warmly welcomed.
