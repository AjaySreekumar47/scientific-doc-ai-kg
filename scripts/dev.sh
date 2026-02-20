#!/usr/bin/env bash
set -e
PORT="${1:-8000}"
echo "Starting FastAPI on http://127.0.0.1:${PORT}"
uvicorn app.main:app --reload --host 127.0.0.1 --port "${PORT}"