#!/bin/bash
set -e
# Serve the FastAPI app. A single worker keeps the resident model + embedding
# matrix in one process (min_instances=1 keeps it warm); --timeout-keep-alive
# is generous for slow video/presign round trips.
exec uvicorn web_server:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1 \
  --timeout-keep-alive 75
