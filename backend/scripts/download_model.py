#!/usr/bin/env python3
"""Download the required machine learning models for Act2Morse.

- Blink Mode: MichalMlodawski/open-closed-eye-classification-mobilev2
- Fin Mode: opencv/handpose_estimation_mediapipe (+ opencv/palm_detection_mediapipe)
- Helper: MediaPipe face_landmarker.task
"""

import argparse
import os
import sys

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.config import load_settings
from app.detector import ensure_eye_model, ensure_model
from app.handpose import ensure_hand_models


def main():
    parser = argparse.ArgumentParser(description="Download Act2Morse AI models.")
    parser.add_argument(
        "--mode",
        choices=["all", "blink", "fin"],
        default="all",
        help="Specify which mode's models to download: 'blink', 'fin', or 'all' (default: all)",
    )
    args = parser.parse_args()
    settings = load_settings()

    if args.mode in ("all", "blink"):
        print(f"Checking Blink mode model (MichalMlodawski/open-closed-eye-classification-mobilev2) at {settings.eye_model_path}...")
        ensure_eye_model(settings.eye_model_path)
        print("Blink mode eye classification model is ready.")

    if args.mode in ("all", "fin"):
        print(f"Checking Fin mode models (opencv/handpose_estimation_mediapipe) at {settings.hand_model_path} & {settings.palm_model_path}...")
        ensure_hand_models(settings.hand_model_path, settings.palm_model_path)
        print("Fin mode handpose estimation models are ready.")

    if args.mode == "all":
        print(f"Checking FaceLandmarker helper at {settings.model_path}...")
        ensure_model(settings.model_path)
        print("All Act2Morse models are ready.")


if __name__ == "__main__":
    main()
