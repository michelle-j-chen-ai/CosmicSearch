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

## Companion tool: the embedding atlas

A separate Cloud Run service, **[vlm-embedding-atlas](https://vlm-embedding-atlas.experimental.apps.applied.dev/)**,
renders the same black-dwarf corpus as a 2D map instead of a ranked list. Source
lives in core-stack at
`onroad/tools/offboard/common/auto_labeling/vlm/embedding_atlas/`.

Where NLS answers "which clips match this query", the atlas answers "what is in
this corpus and how is it organized". Click a point to play the clip and see its
nearest neighbours; lasso a region to sample it.

### Shape

All projection work is offline (`precompute_atlas.py`); the served app never
opens the Lance table. The source is 34.4M x 768 fp32 (~104GB), reduced to a
63MB artifact:

| Stage | Result |
| --- | --- |
| Uniform sample, 250k of 34.4M rows (p=0.726%) | 21,699 drives |
| PCA 768 -> 50 (GPU) | 94% of sampled variance retained |
| UMAP 50 -> 2 (CPU, 124s) | `atlas.parquet` + `projection.pkl` |

`projection.pkl` holds the PCA basis and the fitted reducer. Coordinates from
two independent UMAP fits are not comparable, so placing anything new on *this*
map -- another checkpoint's embeddings, a text query -- requires the saved
basis; re-fitting produces an unrelated picture.

### Reading it honestly

UMAP preserves local neighbourhoods and distorts everything else. Cluster area,
inter-cluster distance, and cluster count (a function of `n_neighbors` /
`min_dist`) carry no information. Neighbour lookup in the app therefore runs in
PCA space, never on the 2D coordinates.

Measured caveats, all from the shipped artifact:

| Measurement | Value |
| --- | --- |
| PCA-50 kNN recall@10 vs true 768-d | **0.79** (median 0.80) |
| Trustworthiness@15, 2D / 3D | 0.973 / 0.985 |
| Neighbours from the same drive | 1.3% (168x chance; 90.6% of clips have none) |
| Variance explained by hour-of-day / month | 3.1% / 3.3% |

Two consequences worth carrying over to NLS work. First, roughly two of every
ten clips in the neighbour panel are not among the true top-10 -- fine for
browsing, not for anything quantitative; storing the sample as int8 768-d
(192MB) rather than fp32 PCA-50 would make it near-exact, which is the one place
in this pipeline quantization earns its keep. Second, local structure survives
*both* reductions well, so if the map looks scene-organized that is a property
of the encoder's similarity, not an artifact of the projection.

### Known gaps

- **No labels are joined**, so no cluster has been verified as semantic. This is
  the largest gap and it is a precompute-time join.
- **The rare tail is invisible by arithmetic.** At p=0.726%, a scenario with
  1,000 chunks corpus-wide appears as ~7 points. Uniform sampling faithfully
  reproduces corpus imbalance, which is exactly wrong for rare-class inspection;
  that needs label-aware over-sampling with inclusion weights.
- **One UMAP seed, no stability pass.** No claim about cluster shape is yet
  trustworthy.
- **Planned:** semantic contrast axes -- project clips onto
  `normalize(emb("A") - emb("B"))` directions so the coordinates are the same
  cosines retrieval is scored on. The query bank is encoded offline (~0.6MB for
  200 axes), so this needs no text tower at serving time. Running UMAP *within*
  that space would organize the map by maneuver rather than appearance.

## Scaling beyond 1M

- **~1-3M rows**: flat search still fine (~0.1-0.3s/query); bump memory.
- **>5M rows**: the matrix stops fitting comfortably in RAM. Either re-write the
  Lance `vector` column as `FixedSizeList<float32>` and build an IVF_PQ index for
  on-disk ANN, or shard the corpus across instances and merge top-k. Both keep
  the same encode-then-search structure.
- **Encoder throughput**: a single CPU encode is ~40ms, so one warm instance
  handles ~25 queries/s/core. Raise `max_instances` for more concurrent users;
  the model is stateless across requests.

## End-to-end master-vs-threshold benchmark

`bench_e2e.py` compares two retrieval paths over the same query, filters, and
threshold (`tau`):

- **master path** — score the resident 768-d model-space matrix
  (`score_corpus`), apply the real-app filter masks (vehicle / run / date
  window), cut at `tau`.
- **PR3 path** — `ThresholdCorpus.threshold_search` (prefilter + screen +
  re-rank) in the shipped `fast_curation` default.

Two hard-fail correctness gates (the script exits non-zero on violation):

- **membership** — the master path's PCA-256-space reference set and the PR3
  path's result set must be equal at the same `tau`.
- **eps bound** — every `bounded_approx` hit's screening score must satisfy
  `|fast_score - exact_score| <= score_error_bound + CROSS_SPACE_TOL`, where
  `exact_score` is the PR3 path's own exact re-rank score.

```bash
# synthetic mode (default): generates two legacy shards locally, converts to
# both corpora, runs the full sweep — no model, no credentials, no network.
python bench_e2e.py --source synthetic --rows 20000

# pre-built corpora (local dirs or s3:// URIs)
python bench_e2e.py --master-uri <dir|uri> --threshold-uri <dir|uri> [--repeats N]
```

The sweep covers filter cells: none; vehicle; date-window narrow (1 week) and
medium (4 weeks); run_uuids (one drive). A cell whose corpus lacks values for
its filter is skipped gracefully.
