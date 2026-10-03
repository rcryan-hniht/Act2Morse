FROM python:3.11-slim

# Prevent python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies needed for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml uv.lock* ./

# Install project dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Pre-download face_landmarker model into models/ so it's baked into the image
RUN mkdir -p models && \
    curl -fsSL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
    -o models/face_landmarker.task

# Copy application source code
COPY app ./app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MODEL_PATH=/app/models/face_landmarker.task
ENV ALLOWED_ORIGINS=*

EXPOSE 8000

# Start Uvicorn bound to 0.0.0.0 and dynamic $PORT for Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
