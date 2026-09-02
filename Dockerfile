# syntax=docker/dockerfile:1
# `deps` = Python + torch + requirements.lock (CI-safe; no gitignored vendor/adp).
# Default final stage remains the full runtime image for deploy.
FROM python:3.13-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# CPU-only torch stack. torchvision is required by the Cosmos-Embed remote
# code even for text-only encoding. Pin to the versions validated for the
# fine-tuned inference pipeline.
RUN uv pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.8.0 torchvision==0.23.0

COPY requirements.lock .
RUN uv pip install -r requirements.lock

FROM deps AS runtime
# interval_core.py: dependency-light (numpy) interval-projection + arrow helpers,
# shared with the offline Spark scan workflow; search_engine imports it.
COPY config.py oci_s3.py local_cache.py interval_core.py search_engine.py analytics.py ./
# int8 PCA corpus backend (very large embedding sets); dispatched by search_engine.
# Full-corpus search: full_corpus.py holds the consolidated corpus's int8/PCA
# screen resident and ranks it; threshold_search/lance_writer/eps_bound are the
# exact-threshold retrieval path it shares (PCA metadata reader + error bound).
# search_engine imports threshold_search lazily, so a missing file here fails at
# call time rather than at build -- hence copying them explicitly.
COPY full_corpus.py lance_writer.py eps_bound.py ./
# FastAPI app (now the served frontend) + its modules and static assets.
# DORA SDK proto stubs: data-explorer-py's `adp` package, VENDORED LOCALLY so the
# image builds on a plain Cloud Build with NO internal pip index. `adp/` is
# .gitignored (not committed) but present in the working tree and uploaded at
# deploy. Populate it once with:  cp -r <venv>/lib/python*/site-packages/adp ./adp
# dora_client uses only the proto stubs + raw grpc, so the public deps already in
# requirements.lock (grpcio, protobuf, googleapis-common-protos, pyroaring)
# suffice. /app is on sys.path, so `import adp` resolves to this vendored copy.
COPY adp/ ./adp/

COPY web_server.py api_v1.py catalog.py deployment.py db.py dora_client.py machine_auth.py ./
COPY web ./web

# Cosmos-Embed loads custom remote code via trust_remote_code; allow the HF
# download at startup and keep the cache on the writable layer.
ENV HF_HOME=/tmp/hf_cache
ENV PORT=8080
EXPOSE 8080

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
