#!/usr/bin/env python3
"""Download the MediaPipe Face Landmarker model asset if not already present."""

import os
import sys

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.detector import ensure_model

if __name__ == "__main__":
    target = os.environ.get(
        "MODEL_PATH",
        os.path.join(backend_dir, "models", "face_landmarker.task"),
    )
    print(f"Checking MediaPipe model at {target}...")
    ensure_model(target)
    print("MediaPipe model is ready.")
