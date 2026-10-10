#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
docker build -t act2morse:local .
echo 'Act2Morse: http://localhost:5173 (Ctrl+C to stop)'
exec docker run --rm --init \
    -p 127.0.0.1:5173:5173 \
    -p 127.0.0.1:8000:8000 \
    -v act2morse-models:/app/backend/models \
    act2morse:local
