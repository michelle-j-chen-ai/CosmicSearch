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
Browser (prompt box + <video> grid)
        |
        v
Cloud Run service (CPU, 16Gi, min_instances=1)   app.py (Streamlit)
   |- load once @st.cache_resource:
   |     model  (search_engine.load_model)   ~5.3GB
   |     corpus (search_engine.load_corpus)  ~3GB, from NLS_EMBEDDINGS_URI
   |- per query:
   |     encode_query    ~40ms   (search_engine)  [text query]
   |     centroid_query          (search_engine)  [refine: mean of selected]
   |     rank_top_k      ~58ms   (search_engine, numpy)
   |     presign_get             (oci_s3 -> browser streams MP4 from OCI)
```

Files:
- `config.py` -- env-driven config.
- `oci_s3.py` -- OCI S3-compat client, Lance storage options, model download, presign.
- `local_cache.py` -- disk cache for downloaded Lance corpora and model snapshots.
- `search_engine.py` -- model load, query encode, corpus load, ranking, centroid refinement.
- `app.py` -- Streamlit UI (text search + relevance-feedback refinement).
- `smoke_test.py` -- offline checks for the search core (no model/network).

## Caching: the corpus and model are downloaded once

The Lance corpus URI is a runtime input (sidebar), not baked in. On first use
of a URI, `local_cache` downloads its `rank=NNNNN/` shards to a cache directory
keyed by the URI, guarded by a file lock so concurrent requests download it at
most once. Two cache layers:

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

## Data prerequisite: build the 1M-sample embedding corpus

The app needs an embeddings URI containing `rank=NNNNN/` Lance shards. Reuse the
existing inference pipeline:

1. Build a parquet of 1M random mini-segment chunks (a `chunk_id` column plus
   the MP4 source columns), the same schema `fine_tuned_embed_inference.py`
   consumes. Generate the MP4s with `../finetuning/generate_chunks_mp4_workflow.py`
   if they don't exist yet.
2. Run the Cosmos-Embed inference workload (`fine_tuned_embed_inference_lilypad_config.yaml`)
   with `output_uri` set to your corpus location, e.g.
   `s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/embeddings/corpus_1m/`.
3. Point the app at that location via `NLS_EMBEDDINGS_URI`.

The video objects referenced by `source_media_uri` in the Lance rows must be
readable with the same credentials the app uses (they live under
`vlm/chunks_mp4/dt=YYYY-MM-DD/<run_uuid>_t<chunk_start_unix>.mp4`).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt

export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export AWS_ENDPOINT_URL_S3=https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com
export AWS_REGION=us-phoenix-1
export NLS_EMBEDDINGS_URI=s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/embeddings/corpus_1m/
export NLS_MODEL_ARTIFACT_URI=s3://.../models/<session>/   # empty -> base model

python -m streamlit run app.py
```

Offline core test (no model, no network): `python smoke_test.py`.

## Deploy (Apps Platform V2)

```bash
apps-platform app deploy        # builds Dockerfile, deploys to Cloud Run

# AWS creds for OCI access, as secrets:
apps-platform app secret set AWS_ACCESS_KEY_ID <key>    --service vlm-nls-search
apps-platform app secret set AWS_SECRET_ACCESS_KEY <key> --service vlm-nls-search
apps-platform app secret set AWS_ENDPOINT_URL_S3 https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com --service vlm-nls-search
apps-platform app secret set AWS_REGION us-phoenix-1     --service vlm-nls-search
apps-platform app secret set NLS_EMBEDDINGS_URI s3://.../corpus_1m/ --service vlm-nls-search
apps-platform app secret set NLS_MODEL_ARTIFACT_URI s3://.../models/<session>/ --service vlm-nls-search
```

Cold start downloads the model (base model from HF, or the merged snapshot from
`NLS_MODEL_ARTIFACT_URI`) and loads the corpus into memory -- a few minutes.
`min_instances = 1` keeps it warm so users never pay that cost.

## Configuration

| Env var | Default | Meaning |
| --- | --- | --- |
| `NLS_EMBEDDINGS_URI` | `""` | Optional default Lance URI to prefill the search box (any URI can be typed at runtime) |
| `NLS_MODEL_ARTIFACT_URI` | `""` | Merged fine-tuned model S3 URI; empty = base model |
| `NLS_CACHE_ROOT` | `/gcs/nls_cache` or `/tmp/nls_cache` | Disk-cache root for downloaded corpora + models |
| `NLS_DEVICE` | `cpu` | torch device for encoding |
| `NLS_MATRIX_DTYPE` | `float32` | corpus matrix dtype (`float16` to halve RAM, 20x slower matmul) |
| `NLS_PRESIGN_TTL_S` | `3600` | presigned MP4 URL lifetime |
| `NLS_OWNER_EMAIL` | `""` | Comma-separated maintainer emails allowed to view the usage-analytics sidebar; empty = nobody (fail closed) |
| `AWS_*` | -- | OCI S3-compat credentials + endpoint |

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

## Scaling beyond 1M

- **~1-3M rows**: flat search still fine (~0.1-0.3s/query); bump memory.
- **>5M rows**: the matrix stops fitting comfortably in RAM. Either re-write the
  Lance `vector` column as `FixedSizeList<float32>` and build an IVF_PQ index for
  on-disk ANN, or shard the corpus across instances and merge top-k. Both keep
  the same encode-then-search structure.
- **Encoder throughput**: a single CPU encode is ~40ms, so one warm instance
  handles ~25 queries/s/core. Raise `max_instances` for more concurrent users;
  the model is stateless across requests.
