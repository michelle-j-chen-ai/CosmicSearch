#!/bin/bash
set -e
# One worker: the atlas is loaded once into process memory at startup and
# min_instances=1 keeps it warm. Extra workers would each hold their own copy.
exec uvicorn app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1 \
  --timeout-keep-alive 75
