# VLM Embedding Atlas

Interactive 2D map of a Cosmos-Embed video embedding table, deployed on Apps
Platform. Click any point to play the source MP4 chunk and see its nearest
neighbours; lasso a region to sample the clips inside it.

## Why it is split into two pieces

The source table (`black-dwarf`) is 34.4M rows x 768 fp32, about 104 GB. Two
constraints shape everything:

- **Sampling is not a compromise.** A 1000x1000 canvas resolves ~1e6 points;
  34M points is ~34 per pixel, so everything saturates to solid ink. Meaningful
  density tops out in the low hundreds of thousands. The artifact is 250k rows.
- **PCA before UMAP improves the result, not just the runtime.** High-dimensional
  distances concentrate, so a kNN graph on raw 768-d encodes more noise than one
  on the top-50 principal directions. On this corpus PCA-50 retains 94% of
  sampled variance and the effective rank at 95% variance is 55 of 768 dims.

So `precompute_atlas.py` does the expensive work offline and the served app only
holds the result. The app never reads the Lance table; after startup its only S3
traffic is presigning MP4 URLs.

## Precompute

```bash
python3 precompute_atlas.py \
  --embeddings-uri s3://neuron-prod-data-intelligence-exploratory/michelle/nls_search/black-dwarf/table/video_embeddings.lance \
  --output-uri s3://neuron-prod-data-intelligence-exploratory/michelle/nls_search/black-dwarf/atlas
```

Runs in ~4 minutes on one GPU box (77s to fetch 275k rows from OCI, ~2s for PCA
on the GPU, ~124s for UMAP on CPU). Writes three objects:

| object | contents |
|---|---|
| `atlas.parquet` | `x`, `y`, `chunk_id`, `run_uuid`, `chunk_start_unix`, `dt`, `source_media_uri`, `pca` (50-d) |
| `projection.pkl` | PCA mean + components, and the fitted UMAP reducer |
| `spectrum.json` | singular values, explained variance, effective rank |

`projection.pkl` is what makes the map extensible. Coordinates from two
independent UMAP fits are not comparable, so placing new points (another
checkpoint's embeddings, or a text query) on **this** map requires the saved
basis and reducer — re-fitting produces an unrelated picture.

Sampling is uniform-over-rows then capped at `--per-run-cap` rows per
`run_uuid`. Reading `run_uuid` for all 34M rows to build exact strata would cost
a full column scan, and the cap achieves the thing that matters: stopping one
long drive from dominating.

On this corpus the cap does not bind at all. The shipped 250k artifact spans
21,699 drives with a median of 9 and a maximum of 57 rows per drive, against a
cap of 400 — so it is effectively a pure uniform sample, and the cap is
insurance against a future corpus with long single-drive runs rather than an
active constraint. Worth re-checking with the log line above if the source table
changes shape.

## What the map can and cannot tell you

UMAP preserves *local* neighbourhoods and freely distorts everything else.
Points that land together really were neighbours in the projected space, but the
distance between two clusters, the relative sizes of clusters, and the axes
themselves carry no meaning.

This is why "nearest neighbours" in the UI is an **exact cosine kNN in PCA
space**, never a 2D-coordinate lookup: the map is a navigation surface, and the
answers come from the embedding space.

Exact in PCA space is not exact in the original space. Measured against true
768-d cosine on a 20k pool, 500 queries: **recall@10 = 0.79** (median 0.80, p10
0.69); only 7% of queries agree perfectly. So roughly two of every ten clips
shown are not among the true top-10. Fine for browsing, not for anything
quantitative. Storing the 250k sample as **int8 768-d (192 MB)** instead of
fp32 50-d (50 MB) would make neighbours near-exact and is the one place in this
pipeline where quantization would earn its keep.

## Colorings

- `density` — quantile-bucketed local point count. Overplotting hides density, so
  a sparse tail and a saturated core otherwise look identical.
- `dt` — drive date, showing dataset composition over time.
- `run_uuid` — whether a region is a single drive, which is the tell for a map
  keyed on nuisance variables rather than semantics.

Colouring by cosine similarity to a free-text query is the highest-value
addition and is deliberately **not** here: it needs the ~5 GB Cosmos-Embed text
tower resident, which changes the image size, memory footprint and cold-start
profile. The `pca` column and `projection.pkl` are stored so it can be added
without re-reading the source table.

## Deploy

Config is read from the environment. Set it as **secrets**, not plain env:
`apps-platform app deploy` re-sets only the platform-managed plain vars and
wipes the rest, but it preserves secret-mounted values (this bit the
`vlm-nls-search` services repeatedly).

```bash
# once, per key
printf '%s' "s3://.../black-dwarf/atlas/atlas.parquet" \
  | apps-platform app secret set ATLAS_URI --data-file=-
```

Required: `ATLAS_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_ENDPOINT_URL_S3`, `AWS_REGION` (`us-phoenix-1` — SigV4 embeds the region
and OCI rejects a mismatch).

```bash
docker build -t vlm-embedding-atlas:local .
apps-platform app deploy --image vlm-embedding-atlas:local
```

Two things the deploy does **not** do, both of which must be re-applied after
every `apps-platform app deploy`:

```bash
# 1. Mount the secrets as env. `app secret set` only writes them to Secret
#    Manager; nothing attaches them to the revision.
gcloud run services update vlm-embedding-atlas --region us-west1 \
  --project experimental-apps-v2 \
  --update-secrets ATLAS_URI=vlm-embedding-atlas-atlas-uri:latest,...

# 2. Turn off request-time CPU throttling. REQUIRED -- see project.toml.
gcloud run services update vlm-embedding-atlas --region us-west1 \
  --project experimental-apps-v2 --no-cpu-throttling --cpu-boost
```

Bootstrapping a brand-new service is circular: secrets cannot be created until
the service exists, and the first deploy therefore boots unconfigured. That is
why the atlas loads on a background thread — the container stays up and reports
the missing key via `/healthz` instead of crash-looping.

Logs: `gcloud run services logs read vlm-embedding-atlas --region us-west1 --project experimental-apps-v2 --limit 50`

### One service per fleet

Each fleet has its own map, from its own artifact:

| service | config | corpus | artifact |
|---|---|---|---|
| `vlm-embedding-atlas` | `project.toml` | neuron | `s3://neuron-prod-data-intelligence-exploratory/michelle/nls_search/black-dwarf/atlas/` |
| `vlm-embed-atlas-trucking` | `project-trucking.toml` | frontier | `s3://frontier-perception-datasets/vlm/atlas/` |

Not one service holding both: coordinates from two independent UMAP fits are not
comparable, so no single map can hold both corpora. A point's position only means
something relative to the other points on the same map.

Corpus tables name their embedding column per model (`vector_black_dwarf`), while
the older standalone atlas table calls it `vector` — `--vector-column` selects it,
and the artifact schema says `vector` either way.

## Endpoints

| route | returns |
|---|---|
| `GET /api/meta` | point count, PCA dim, available colorings, revision |
| `GET /api/points` | interleaved xy `Float32Array` (~2 MB) |
| `GET /api/coloring/{field}` | per-point `Uint8Array` category index; legend in the `X-Atlas-Legend` header |
| `GET /api/point/{i}` | metadata + presigned MP4 URL |
| `GET /api/neighbors/{i}?k=` | exact cosine kNN in PCA space |
| `POST /api/lasso` | points inside a polygon, nearest-to-centroid first |
| `GET /healthz` | 503 until the atlas is resident |

Coordinates and colour indices are raw little-endian binary, not JSON: 250k
points is ~2 MB as a `Float32Array` versus ~12 MB of JSON that also has to be
parsed on the main thread.
