# syntax=docker/dockerfile:1
FROM python:3.13-slim

WORKDIR /app

# CPU-only torch stack. torchvision is required by the Cosmos-Embed remote
# code even for text-only encoding. Pin to the versions validated for the
# fine-tuned inference pipeline.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 torchvision==0.23.0

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
# lilypad_py (the Lilypad submit client, nls_launcher.py) lives on the AUTHENTICATED Applied
# (ursa) index, not public PyPI, and its declared deps include ray/wandb/oci (heavy, unused
# by the submit path). So it is VENDORED and installed WITHOUT deps -- the light
# import-closure it actually needs is already covered by requirements.txt. Its bundled
# `adp.services.lilypad` protos merge with the DORA `adp` vendored below (namespace pkgs).
COPY vendor ./vendor
RUN pip install --no-cache-dir --no-deps ./vendor/lilypad_py-*.whl

# DORA SDK proto stubs: data-explorer-py's `adp` package, VENDORED LOCALLY so the
# image builds on a plain Cloud Build with NO internal pip index / BuildKit
# secret. Like tag_backfill_dashboard's `ursa/` stubs, `adp/` is .gitignored (not
# committed) but present in the working tree and uploaded at deploy. Populate it
# once with:  cp -r <venv>/lib/python*/site-packages/adp ./adp
# dora_client uses only the proto stubs + raw grpc -- it does NOT import the heavy
# `adp.public.strada.dora` helper (which pulls job_metadata -> simian -> ...), so
# the public deps in requirements.txt (grpcio, protobuf, googleapis-common-protos)
# suffice. /app is on sys.path, so `import adp` resolves to this vendored copy.
COPY adp/ ./adp/

# interval_core.py: dependency-light (numpy) interval-projection + arrow helpers,
# shared with the offline Spark scan workflow; search_engine imports it.
COPY config.py oci_s3.py local_cache.py interval_core.py search_engine.py analytics.py app.py ./
# int8 PCA corpus backend (very large embedding sets); dispatched by search_engine.
COPY gpu_corpus.py ./
# FastAPI app (now the served frontend) + its modules and static assets.
COPY web_server.py dora_client.py db.py machine_auth.py nls_launcher.py ./
COPY web ./web

# Cosmos-Embed loads custom remote code via trust_remote_code; allow the HF
# download at startup and keep the cache on the writable layer.
ENV HF_HOME=/tmp/hf_cache
ENV PORT=8080
EXPOSE 8080

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
