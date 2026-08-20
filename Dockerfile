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
# lilypad_py (the Lilypad submit client, nls_launcher.py) lives on the AUTHENTICATED Applied
# (ursa) index, not public PyPI, and its declared deps include ray/wandb/oci (heavy, unused
# by the submit path). So it is VENDORED and installed WITHOUT deps -- the light
# import-closure it actually needs is already covered by requirements.lock. Its bundled
# `adp.services.lilypad` protos merge with the DORA `adp` vendored below (namespace pkgs).
COPY vendor ./vendor
# The vendored wheel is tagged cp310-cp310-linux_x86_64, so an installer on this
# 3.13 base rejects it on the tag alone. It ships NO compiled extensions -- every
# entry sits under `.data/purelib` -- so it is pure Python and imports unmodified
# on 3.13. Retag it to py3-none-any and install that; the payload is copied byte
# for byte, only the WHEEL tag line differs.
RUN python - <<'RETAG' && uv pip install --no-deps ./vendor/lilypad_py-2.26.0-py3-none-any.whl
import glob, zipfile
src = glob.glob("vendor/lilypad_py-*-cp*.whl")[0]
zin = zipfile.ZipFile(src)
with zipfile.ZipFile("vendor/lilypad_py-2.26.0-py3-none-any.whl", "w",
                     zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        data = zin.read(info.filename)
        if info.filename.endswith(".dist-info/WHEEL"):
            data = (b"Wheel-Version: 1.0\nGenerator: retag\n"
                    b"Root-Is-Purelib: false\nTag: py3-none-any\n")
        zout.writestr(info, data)
RETAG

# DORA SDK proto stubs: data-explorer-py's `adp` package, VENDORED LOCALLY so the
# image builds on a plain Cloud Build with NO internal pip index / BuildKit
# secret. Like tag_backfill_dashboard's `ursa/` stubs, `adp/` is .gitignored (not
# committed) but present in the working tree and uploaded at deploy. Populate it
# once with:  cp -r <venv>/lib/python*/site-packages/adp ./adp
# dora_client uses only the proto stubs + raw grpc -- it does NOT import the heavy
# `adp.public.strada.dora` helper (which pulls job_metadata -> simian -> ...), so
# the public deps in requirements.lock (grpcio, protobuf, googleapis-common-protos)
# suffice. /app is on sys.path, so `import adp` resolves to this vendored copy.
COPY adp/ ./adp/

# interval_core.py: dependency-light (numpy) interval-projection + arrow helpers,
# shared with the offline Spark scan workflow; search_engine imports it.
COPY config.py oci_s3.py local_cache.py interval_core.py search_engine.py analytics.py app.py ./
# int8 PCA corpus backend (very large embedding sets); dispatched by search_engine.
COPY gpu_corpus.py ./
# Full-corpus search: full_corpus.py holds the consolidated corpus's int8/PCA
# screen resident and ranks it; threshold_search/lance_writer/eps_bound are the
# exact-threshold retrieval path it shares (PCA metadata reader + error bound).
# search_engine imports threshold_search lazily, so a missing file here fails at
# call time rather than at build -- hence copying them explicitly.
COPY full_corpus.py threshold_search.py lance_writer.py eps_bound.py ./
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
