# VLM Natural-Language Video Search

A web app for retrieving driving video clips by natural-language prompt. A user
types a description; the app encodes it with the fine-tuned Cosmos-Embed1 model,
ranks a corpus of pre-computed video embeddings by cosine similarity, and plays
back the top matches streamed directly from OCI.

This consumes the Lance embedding tables produced by
`../finetuning/fine_tuned_embed_inference.py`. It does **not** generate
embeddings or train models.

## Refining a search (relevance feedback)

A natural-language query lands in the joint text/video space, but that space is
only as well-aligned as the fine-tuning made it. A loose alignment shows up as a
recall problem: a prompt like "left turn" surfaces *some* true matches near the
top and scatters visually-identical ones further down, because the text vector
doesn't sit exactly where those videos cluster.

The app lets you fix this from the results, no re-prompting required. Check the
results that are genuinely what you want and press **Refine with N selected**.
The app averages the *video* embeddings of the rows you checked into a single
unit centroid (`search_engine.centroid_query`) and re-ranks the whole corpus by
cosine similarity to it. Because the centroid lives in video space among the
real matches, it side-steps the text-side misalignment and pulls up the rest of
the cluster the text query missed. This is nearest-centroid (Rocchio-style)
relevance feedback, and it is iterative -- refine, review, check more, refine
again. "Clear selection" drops the checked set; running a new text query resets
back to text search.

## Why this shape (CPU-only, single Cloud Run service)

The hard scaling worry was the encoder's GPU footprint. It turned out not to
matter: the ~10GB VRAM is for the *video* tower. Encoding a *query string* goes
through the text tower only, which runs on CPU in **~35-45ms**. Measured on this
codebase's model:

| Step | Cost |
| --- | --- |
| Query text encode (CPU, fp32) | ~35-45 ms |
| Rank 1M x 768 corpus (numpy BLAS gemv, fp32) | ~58 ms |
| Presign OCI URL | <5 ms (local signing) |
| **Total per query** | **~100 ms** |

So the entire app is one CPU Cloud Run service. No GPU, no separate encoder
microservice. The two memory consumers are the model (~5.3GB fp32) and the
embedding matrix (1M x 768 fp32 = ~3GB), which is why the service is sized at
16Gi.

### Brute force, not ANN

At 1M rows a flat matrix-vector product is ~58ms and exact. An ANN index (e.g.
LanceDB IVF_PQ) only pays off past ~10M rows, and the inference pipeline writes
the `vector` column as a plain list rather than a `FixedSizeList<float32>`, so a
native Lance index wouldn't apply without re-writing the table. See "Scaling"
below.

### fp32, not fp16

numpy has no BLAS kernel for fp16: the same 1M matmul is ~58ms in fp32 but
~1.2s in fp16. fp32 doubles the matrix RAM (3GB vs 1.5GB) but is 20x faster, so
it is the default. Set `NLS_MATRIX_DTYPE=float16` only on a RAM-constrained
deploy.

## Architecture

```
Browser (search, refine, threshold sweep, save, export)        Integrations
        |  /ui/* (IAP only)                                     |  /api/v1/* (API key)
        v                                                       v
Cloud Run service (CPU, 32Gi, min_instances=1)  --  web_server.py + api_v1.py
   |- loaded once, held resident:
   |     model   (search_engine.load_model)              ~5GB fp32
   |     corpora (full_corpus.load, one per project)     int8/PCA-256 screen, whole table
   |- per request:
   |     encode_query   ~40ms  (search_engine)
   |     select         ~190ms (full_corpus: int8 sweep -> eps band -> fp32 refine)
   |     exact_scores          (full_corpus: 768-d cosine for exports)
   |     presign_get           (oci_s3 -> browser streams MP4 from OCI)
   |- catalog.py: tags, versions, per-project thresholds, exports (Cloud SQL)
```

The public API is the seven endpoints under `/api/v1` (`POST/GET /tags`,
`GET/PUT/DELETE /tags/{tag}`, `GET /video`, `GET /health`); see the design page
"Cosmic Search API Design v2". The browser uses those plus four `/ui` routes
(live search over an unsaved query, threshold calibration, the Data Explorer
set picker, clip redirects) that the API gateway does not expose.

Files:
- `config.py` -- env-driven config.
- `deployment.py` -- the project registry: table, clip prefix and Data Explorer host per project.
- `oci_s3.py` -- OCI S3-compat client, Lance storage options, model download, presign.
- `local_cache.py` -- disk cache for model snapshots and fetched segment sets.
- `search_engine.py` -- model load, text/frame encoding, threshold fitting.
- `full_corpus.py` -- the resident corpus and the search cascade.
- `catalog.py` -- the tag store.
- `api_v1.py` -- the public API.
- `web_server.py` -- corpus lifecycle, `/ui` routes, pages.
- `smoke_test.py` -- offline checks for the search core (no model/network).

## Caching: the corpus and model are downloaded once

Each project's corpus table is read straight from OCI at load; the model
snapshot and fetched Data Explorer segment sets are cached on disk, guarded by a
file lock so concurrent requests download at most once. Two cache layers:

- **Disk cache** (`local_cache`): keyed by URI, with a `.nls_download_complete`
  marker (partial downloads are re-fetched, never served). Cache root resolves
  to `NLS_CACHE_ROOT`, else `/gcs/nls_cache` when the GCS-fuse mount exists
  (**shared across instances and persistent -- the corpus is downloaded once
  ever, not per cold start**), else `/tmp/nls_cache` for local dev.
- **In-memory cache** (`@st.cache_resource`): the loaded matrix and the model
  are held resident and shared across all user sessions in a process. A second
  user searching the same corpus pays neither download nor load.

The text-encoder model snapshot is cached the same way. Measured: a cold
download+load of a 5k-row corpus took ~21s; the second load was ~0s (cache hit).

## Data prerequisite: the consolidated corpus

### One image, one service per fleet

neuron and frontier deploy as two Cloud Run services off the same image,
differing only in `NLS_PROJECTS`. Two rather than one serving both, for memory:
each resident int8 corpus is >13GB against a 32Gi ceiling and one process cannot
hold both alongside the ~5GB encoder. The split also gives each fleet its own
Postgres schema -- `db.py` derives it from `K_SERVICE` -- so tags, marks and
exports never mix between fleets, and its own API-gateway keys. See
`project.toml` and `project-trucking.toml`.

Each project serves one Lance table, named in `deployment.py` (neuron:
`s3://neuron-prod-data-intelligence-exploratory/vlm/corpus/video_embeddings.lance`,
frontier: `s3://frontier-perception-datasets/vlm/corpus/video_embeddings.lance`).
The embedding pipeline in core-stack upserts into it; the app only reads. A
request cannot point a search at another table: a query scored against a
different model's embeddings returns a confident ranked list with nothing to
signal it.

The video objects referenced by `source_media_uri` must be readable with the
same credentials the app uses, laid out as
`<clip prefix>/dt=YYYY-MM-DD/<run_uuid>_t<chunk_start_unix>.mp4`.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt

export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export AWS_ENDPOINT_URL_S3=https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com
export AWS_REGION=us-phoenix-1
export NLS_MODEL_ARTIFACT_URI=s3://.../models/<session>/   # empty -> base model

uvicorn web_server:app --port 8080
```

Offline core test (no model, no network): `python smoke_test.py`.

## Deploy (Apps Platform V2)

```bash
./deploy.sh                        # cars
./deploy.sh project-trucking.toml  # trucks

# AWS creds for OCI access, as secrets:
apps-platform app secret set AWS_ACCESS_KEY_ID <key>    --service vlm-nls-search
apps-platform app secret set AWS_SECRET_ACCESS_KEY <key> --service vlm-nls-search
apps-platform app secret set AWS_ENDPOINT_URL_S3 https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com --service vlm-nls-search
apps-platform app secret set AWS_REGION us-phoenix-1     --service vlm-nls-search
apps-platform app secret set NLS_MODEL_ARTIFACT_URI s3://.../models/<session>/ --service vlm-nls-search
```

Cold start downloads the model (base model from HF, or the merged snapshot from
`NLS_MODEL_ARTIFACT_URI`) and loads the corpus into memory -- a few minutes.
`min_instances = 1` keeps it warm so users never pay that cost.

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `NLS_MODEL_ARTIFACT_URI` | `""` | Merged fine-tuned model S3 URI; empty = base model |
| `NLS_CACHE_ROOT` | `/gcs/nls_cache` or `/tmp/nls_cache` | Disk-cache root for downloaded corpora + models |
| `NLS_DEVICE` | `cpu` | torch device for encoding |
| `NLS_MATRIX_DTYPE` | `float32` | corpus matrix dtype (`float16` to halve RAM, 20x slower matmul) |
| `NLS_PRESIGN_TTL_S` | `3600` | presigned MP4 URL lifetime |
| `NLS_OWNER_EMAIL` | `""` | Comma-separated maintainer emails allowed to view the usage-analytics sidebar; empty = nobody (fail closed) |
| `AWS_*` | -- | OCI S3-compat credentials + endpoint |
| `NLS_PROJECTS` | `neuron` | Comma-separated projects this instance loads and serves (`neuron`, `frontier`). Every `/api/` call takes `project` (query param or body field); one that omits it is served from the **first** project in this list |
| `NLS_<PROJECT>_CORPUS_TABLE_URI` | per project, see `deployment.py` | Override a project's Lance table |
| `NLS_<PROJECT>_MP4_PREFIX` | per project | Override where a project's clips are stored |
| `NLS_<PROJECT>_DORA_HOSTNAME` | per project | Override a project's Data Explorer gRPC host |
| `NLS_MP4_PREFIX` | neuron clip prefix | Shared default clip prefix; `neuron` inherits it |
| `NLS_CORPUS_REFRESH_UTC` | `off` | Daily in-place corpus reload, `HH:MM` UTC. Off by default: a refresh drops the resident corpus and rebuilds it (32Gi cannot hold two copies), so the instance serves no full-corpus search meanwhile. Instances are replaced often enough on their own that a fresh process picks up the current table anyway |
| `NLS_ALERT_WEBHOOK` | `""` | Slack-style incoming webhook posted to when a corpus cannot load. Unset = log only. A corpus that will not load takes every search down and every instance fails identically, so nothing recovers on its own |

## OCI presigned-URL gotchas

Streaming MP4s to the browser via presigned URLs requires three non-default
boto3 client settings (all set in `oci_s3.s3_client`), or the URL fails:

- `signature_version="s3v4"` -- without it boto3 presigns legacy SigV2 URLs
  (`AWSAccessKeyId`/`Signature`/`Expires`) that OCI rejects with **404**.
- `region_name` passed explicitly (from `AWS_REGION`) -- the SigV4 signature
  embeds the region; a default (e.g. `us-west-2`) yields **400 "authorization
  header is malformed; the region ... is wrong"**.
- `s3={"addressing_style": "path"}` -- OCI S3-compat is path-style only.

With all three, a presigned GET returns 200 and a ranged GET returns 206 with
`Accept-Ranges: bytes`, so the browser `<video>` tag streams via range requests.

## The atlas: the corpus as a map

The **Explore** tab links to [vlm-embedding-atlas](https://vlm-embedding-atlas.experimental.apps.applied.dev/),
a separate Cloud Run service that renders the same corpus as a 2D projection
instead of a ranked list. Source is in [`embedding_atlas/`](embedding_atlas/) --
self-contained, with its own `Dockerfile` and `project.toml`.

It stays a separate service rather than a page in this app, for memory. This
process already holds a 13.86GB int8 screen plus a ~5GB encoder against a 32GiB
ceiling, and the screen grew 8.82GB -> 13.86GB in a week as the corpus went
34.4M -> 54.2M rows. The atlas runs in 2GiB. Spending the search service's
shrinking headroom on it -- and risking an OOM that Cloud Run answers by killing
the whole instance, search included -- buys nothing the link does not.

Read the map for local structure, not global distance: UMAP preserves
neighbourhoods, not the size of the gaps between them. Measured on this corpus,
PCA-50 kNN recovers 0.79 of the true 768-d top-10, neighbourhoods are not
drive-driven (1.3% same-drive), and hour-of-day and month each explain ~3% of
variance -- so a scene-organized map reflects the encoder's similarity, not the
projection.

## Scaling beyond 1M

- **~1-3M rows**: flat search still fine (~0.1-0.3s/query); bump memory.
- **>5M rows**: the matrix stops fitting comfortably in RAM. Either re-write the
  Lance `vector` column as `FixedSizeList<float32>` and build an IVF_PQ index for
  on-disk ANN, or shard the corpus across instances and merge top-k. Both keep
  the same encode-then-search structure.
- **Encoder throughput**: a single CPU encode is ~40ms, so one warm instance
  handles ~25 queries/s/core. Raise `max_instances` for more concurrent users;
  the model is stateless across requests.
