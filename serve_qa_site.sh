#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HOST="${LIMO4SI_QA_HOST:-0.0.0.0}"
PORT="${LIMO4SI_QA_PORT:-8000}"
echo "Serving LIMO4SI QA site at http://${HOST}:${PORT}/"
echo "Open this in your browser: http://<server-ip>:${PORT}/"
echo "Serving directory: site/qa_benchmark"
python -m http.server "$PORT" --bind "$HOST" --directory site/qa_benchmark
