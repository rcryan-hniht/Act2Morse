FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app/backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY backend/app ./app
COPY --from=frontend /frontend/dist /app/frontend
ENV PATH="/app/backend/.venv/bin:$PATH"
EXPOSE 5173 8000
# Stop both services when either exits or Docker stops the container.
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & backend=$!; python -m http.server 5173 --bind 0.0.0.0 --directory /app/frontend & frontend=$!; trap 'kill \"$backend\" \"$frontend\" 2>/dev/null || true' EXIT; trap 'exit 143' TERM; trap 'exit 130' INT; wait -n"]
