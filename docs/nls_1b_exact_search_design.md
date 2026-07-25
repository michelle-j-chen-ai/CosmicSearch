# Design: exact threshold retrieval over 100M-1B embeddings in <10s (Lance, distributed on k8s)

Status: proposed design, research-verified 2026-07-03. The single-node
screen/eps/re-rank core (`eps_bound.py`, `lance_writer.py`,
`threshold_search.py`) is implemented and tested; the distributed fleet
(aggregator, fragment striping, coverage protocol, k8s manifests) described
below is not.
Companion doc: [nls_1b_exact_search_findings.md](nls_1b_exact_search_findings.md)
(repo analysis, verified hardware numbers, prior-art survey backing every
claim here).

## Goal

Return ALL segments with cosine similarity >= tau to a query embedding —
zero false negatives — over a corpus growing from 100M to 1B+ embeddings,
<10s per query, serving both the interactive app and the batch auto-labeling
scan from ONE Lance dataset. Deployment constraint: horizontally scalable on
Kubernetes using mainstream cloud machine types with good IO — no bare
metal, no storage-optimized monsters, no GPUs. Scale = add workers.

This zero-false-negative guarantee is against the true pre-quantization
cosine only when the writer's artifact supplies an independent fp32-256
projection; today's production artifact pipeline does not, so the shipped
`vector_fp` is a deterministic function of the int8 corpus (see the Lance
dataset spec's `vector_fp` entry and `eps_bound.py`'s module docstring for
the precise, narrower guarantee that configuration delivers).

## Design in one paragraph

Shard the corpus by Lance fragments across N scan workers on k8s. Each
worker holds its shard of the int8 PCA-256 screen column RESIDENT in RAM
(the proven gpu_corpus.py pattern — 12.2GB resident at 47.8M rows today; a
1B/8-way shard is 32GB) and answers a query by scanning its shard at memory
bandwidth with the existing numba int8 kernel, keeping everything
>= tau - eps (hard Cauchy-Schwarz quantization bound). Survivor bands are
re-ranked exactly against the fp32-256 column via Lance take() (1 IOP/row
from format 2.1 on) from local NVMe cache or object store. An aggregator unions
per-worker results (threshold union — no top-k merge) and verifies fragment
coverage is complete before declaring the result exact. Warm per-query
latency ~1-3s at any corpus size that keeps per-shard <= RAM; growth is
replicas+=1. Same topology as Milvus-FLAT / ClickHouse-distributed /
LanceDB's published 10B architecture — with a determinism guarantee none of
them provide.

## Why exact scan and not ANN

The workload is threshold retrieval for training-data curation: a missed
boundary vector silently drops a training example. ANN (IVF_PQ, HNSW,
Milvus range search) and the industrial int8-rerank cascades all trade
completeness for speed (~95-97% recall) — disqualified. The prior-art
survey found NO published system doing provably-complete cosine-threshold
retrieval on dense vectors at >=100M; the only exact-complete primitive is
the exhaustive scan. So the design makes the exhaustive scan cheap:
12x fewer bytes/row (768-d fp32 -> PCA-256 int8, score-lossless basis +
hard error bound), a Lance layout where a scan moves only those bytes, and
horizontal bandwidth. Pattern = GEMINI / lower-bounding-property (screen to
a provable superset, refine the boundary band exactly) — the same structure
as 2024-25 exact-search SoTA (SOFA, Panorama).

## Verified constants the design stands on

| Fact | Value | Source class |
|---|---|---|
| Screen column @1B | 256 GB (FSL<int8,256>) | arithmetic |
| PCA-768->256 score-lossless; int8 corr vs fp32-768 | 0.99995 | repo-measured @48M (gpu_corpus.py) |
| Repo numba int8 kernel | ~49 GB/s on 8 vCPU (12.2GB in 0.25s) | repo-measured |
| DDR5 socket sustained read | ~200-307 GB/s | measured (STREAM/McCalpin) |
| Object-store read per VM | ~5 GB/s (OCI VM 40Gbps NIC), 10-13 GB/s (100Gbps+) | official spec / measured |
| Local NVMe, mainstream shapes | GCP C3/C3D-lssd up to ~13 GB/s; OCI VM.DenseIO.E5.Flex 1-6 drives | official spec |
| Lance 2.1 fixed-width >=128B/value | full-zip encoding, 1 IOP random take() | official paper/blog |
| Exact-scan precedent | ClickHouse 12-80 GB/s fp32 cosine scans | measured |
| Encoder | ~40ms/query, CPU text tower, stateless | repo-measured |

## Cluster architecture (k8s)

```
                       ┌─ web_server (Cloud Run, us-west1; existing app tier)
   user / batch ETL ──▶│    encodes query (text tower, ~40ms)
                       ▼  ─────────── cross-cloud (KB request) ──────────
                 aggregator Deployment (OKE, us-phoenix-1; stateless, HPA)
                 │  fan-out: (q_pca, tau, eps, filter, dataset_version)
                 │  union rows >= tau; verify fragment coverage == 100%
                 ▼
      headless Service → scan-worker StatefulSet (N replicas, e.g. 8)
      worker r owns fragments {i : i mod N == r} of dataset version V
      ├─ RAM: shard of embedding_i8 resident (~256GB/N), numba kernel
      ├─ local NVMe (emptyDir/local-PV): fragment cache + vector_fp for
      │   take(); survives pod restart on same node
      └─ object store (OCI S3-compat, us-phoenix-1): source of truth, cold
          hydration — in-region to the workers
```

- **Shard assignment**: round-robin striping of fragment ids (not
  contiguous ranges). Rows are date-sorted, so striping spreads a
  date-filtered query across ALL workers evenly; contiguous ranges would
  concentrate it on 1-2 workers. Fragment min/max pruning applies per
  fragment wherever it lives.
- **Workers are scan-only**: no torch, no encoder — tiny image, fast
  startup. Encoding stays in web_server (CPU text tower, as today).
- **Coverage protocol (exactness guard)**: every worker response carries
  (dataset_version, fragments_scanned). The aggregator asserts the union
  equals the version's full fragment set; otherwise the result is marked
  incomplete and retried/failed — never silently partial. Batch/export
  requires complete; interactive may render partial WITH a visible
  incomplete flag.
- **Availability vs exactness**: replication factor 2 per shard for the
  interactive cluster (RAM is cheap at these sizes); RF1 + retry is
  acceptable for batch-only clusters.
- **Refresh loop**: workers poll for new committed dataset versions,
  hydrate only NEW fragment ids they own (append-only until compaction),
  atomic-swap the resident shard; readiness gate = "resident shard matches
  version V". Post-compaction fragment ids change: re-stripe and rehydrate a
  full per-worker shard (256GB/N = 32GB @N=8; ~7s at 5GB/s NIC, matching the
  Node sizing section's cold-hydration figure below).
- **Scaling**: corpus growth => raise N (shards shrink); QPS growth => add
  a replica set. Aggregator is stateless (HPA). Fragments are the unit,
  object store is the truth — scale is linear with no redesign point.
- **Batch scan**: same workers serve batch scans (per-worker outputs stream
  to the per-scan segments.lance as today), or an ephemeral job fleet reads
  the same dataset with the same striping. One storage layout serves both.

## Deployment topology (grounded in the repo's actual config)

Current deploy (project.toml, entrypoint.sh, Dockerfile, config.py,
oci_s3.py, nls_launcher.py, local_cache.py):

- App tier is Cloud Run, project `experimental-apps-v2`, region `us-west1`,
  32Gi / 8 vCPU, `min_instances=1`, served as `uvicorn web_server:app
  --workers 1` (entrypoint.sh). One warm instance holds the resident
  model + corpus matrix.
- Corpus lives on OCI object storage in `us-phoenix-1`
  (`_OCI_ENDPOINT`, `nls_launcher.py:43`; default `AWS_REGION` `us-phoenix-1`,
  `oci_s3.py:35`). Today's batch scan workers are pinned to
  `us-chicago-1` (`_ALLOWED_REGIONS`, `nls_launcher.py:45`) — every corpus
  byte already crosses OCI regions.
- Model + Lance corpus cache is a GCS-fuse mount (`/mnt/data` then `/gcs`,
  `local_cache.py:41`; `enable_gcs_fuse=true`, project.toml), shared across
  instances so the corpus downloads once.
- Stack is pylance + lancedb (requirements.txt:7-8); Lance vector search is
  never invoked — the loader fully materializes a numpy matrix
  (search_engine.py:260, 510).

Why the app tier cannot also host the scan workers: Cloud Run caps a service
at 32Gi / 8 vCPU, has no local NVMe (`/tmp` is RAM tmpfs), and gives no
StatefulSet semantics — it cannot hold a 32GB resident shard plus model plus
scratch, and cannot pin a worker to a node-local fragment cache. The scan
fleet therefore leaves Cloud Run regardless; the app tier stays.

Recommended topology:

- **App tier — unchanged.** Cloud Run `us-west1`, owns the text encoder
  (~40ms/query, CPU) and the Streamlit/FastAPI UI; keeps the GCS-fuse model
  cache. `web_server` gains the distributed-corpus backend beside the
  existing `load_corpus` dispatch and routes the batch scan API at the
  aggregator.
- **Scan-worker fleet — OKE, `us-phoenix-1`, same cloud and region as the
  corpus bucket.** This is the load-bearing call: workers hydrate and scan
  the corpus, so co-locating them with the bucket removes the per-byte
  cross-region/cross-cloud tax that the current `us-chicago-1` pin and any
  GKE alternative would pay. OKE over GKE because the corpus is on OCI;
  GKE would only win after a one-time corpus mirror to GCS.
- **Aggregator — co-located on the OKE cluster**, stateless (Deployment +
  HPA). Putting it in-cluster makes the fan-out and coverage union
  in-cluster RPC; the only cross-cloud hop is app (Cloud Run `us-west1`)
  to aggregator (OKE `us-phoenix-1`), and that call is a KB request / MB
  response. (Keeping the aggregator on Cloud Run is viable but adds a
  cross-cloud hop on every fan-out — not recommended.)

## Node sizing (mainstream shapes, N=8 example @1B)

Per worker: 32GB resident shard + headroom => 64GB-RAM nodes comfortable;
16-32 vCPU (kernel is memory-bandwidth-bound; repo hit 49GB/s on 8 vCPU).

| OKE node pool (us-phoenix-1) | Shape | Notes | ~List/mo/node |
|---|---|---|---|
| NVMe variant (default) | VM.DenseIO.E5.Flex 16 OCPU / 192GB / 2x6.8TB NVMe | shard in RAM + full vector_fp on local NVMe | ~$0.8-1.2k |
| RAM-only variant | VM.Standard.E5.Flex 16-24 OCPU / 128-192GB | take() goes to object store | ~$0.7-1k |

Fleet @N=8 RF2: ~$11-19k/mo list (RF1 batch-only: half). Every unit is
small, replaceable, elastic — no single machine above ~$1.5k/mo. Pick the
NVMe vs RAM-only variant from the Day-1 take() benchmark: if object-store
take() on the eps band is inside budget, RAM-only; else NVMe.

Latency budget (N=8, warm, SELECTIVE query): encode 40ms + fan-out ~10ms +
per-worker scan 32GB at 20-40GB/s effective = 0.8-1.6s + take() (1 IOP/row;
local NVMe ~ms, object store 0.5-2s) + union 50-200ms => **~1-3s**, inside
10s even with a straggler retry. Cold pod hydration: 32GB at ~5GB/s NIC ≈ 7s
once per pod lifetime, not per query. @100M the same fleet runs at N=2-4.

take() volume is the full RESULT set (ABOVE union BAND, both re-ranked; see
Lance dataset spec's take()-only note), not just the BAND width: BAND ~1e-4
selectivity (~12.5k rows/worker @1B N=8) is the mandatory-re-rank sliver
around tau, but ABOVE (score provably >= tau, no BAND ambiguity) can be
far larger for a broad/low-tau query and is re-ranked unconditionally for a
consistent exact score (Correctness spec). The ~1-3s figure above holds when
the WHOLE result is ~1e-4 selective (the realistic regime for a tight
threshold query); a broad query's take() volume — and therefore its
object-store cost and the NVMe-vs-RAM node choice — scales with the result
size, not the band alone. Size the node variant and per-query deadline from
the expected result-set selectivity of the actual workload, not the band
selectivity in isolation.

## Rejected alternatives

- **Single big node** (OCI BM.DenseIO ~40GB/s measured / GCP Z3-176
  37.7GB/s spec): meets <10s but is a vertical dead end on rare
  bare-metal-class shapes — excluded by deployment constraint. Useful as a
  P0 benchmark reference only.
- **GPU fleet**: no library support for exact range retrieval (FAISS GPU
  has no range_search on any index; cuVS brute_force is fp16/fp32-only, no
  out-of-core; the only exact GPU range primitive is L2-only). Custom
  kernels possible (gpu_corpus precedent) at $23-68k/mo — revisit only at
  QPS far beyond CPU-fleet economics.
- **ANN**: approximate — violates zero-false-negatives. Optional side
  index for interactive browsing only, never the curation/export path.
- **Direct object-store scan per query** (no residency): 256GB at 5-13GB/s
  per node makes IO the bottleneck; residency decouples query latency from
  object-store bandwidth.

## Lance dataset spec

One dataset (replaces the rank=NNNNN/ sharded lancedb tables):

- Write with data_storage_version="2.2"; readers accept anything >= 2.1,
  the version where fixed-width >=128B/value first gets full-zip encoding
  => 1-IOP take(). Never write "stable": it resolves to the running pylance's
  default, which is 2.0 on pylance 4.x and 2.1 on 9.x, so it can silently
  produce a pre-full-zip dataset. Verify ds.data_storage_version at runtime.
- Defaults max_rows_per_file=1,048,576 -> ~954 fragments @1B (fine;
  enable_v2_manifest_paths default on). compact_files(target_rows_per_
  fragment=1M) after appends; skips already-large fragments.
- Rows physically sorted by (chunk_start date, vehicle) at write. Lance
  prunes whole fragments via column min/max stats; there is NO sub-fragment
  contiguity guarantee, so write-time sorting is what makes date/vehicle
  prefilters skip fragments wholesale. Fragments are then STRIPED
  round-robin across workers.
- Columns:

  | column | type | bytes/row @1B | role |
  |---|---|---|---|
  | embedding_i8 | FSL<int8,256> | 256 GB | screen scan (the only column scanned) |
  | vector_fp | FSL<float32,256> | 1 TB | re-rank, take()-only |
  | tail_norm_f16 | float16 | 2 GB | reserved for optional progressive pruning |
  | run_uuid, chunk_start_unix, chunk_end_unix, segment_id, vehicle | scalars | ~50-80 GB | metadata + filters |

  Dropped: chunk_id, source_media_uri — derivable row-exact from
  run_uuid + chunk_start (proven in gpu_corpus.py); ~185GB saved @1B.
  PCA basis + per-dim quant scales live in dataset schema metadata.
  `vector_fp` is the true pre-quantization PCA-256 projection when the
  writer's artifact provides one (`lance_writer.PRE_QUANT_FP32_FILE`);
  today's production artifact pipeline does not, so it falls back to
  `int8 * scale / 127` (a deterministic function of `embedding_i8`, carrying
  no information beyond it) -- see `lance_writer.py`'s module docstring. The
  eps-bound re-rank's guarantee is against whichever of these `vector_fp`
  actually holds.
- Scalar indices: BTREE(chunk_start_unix), BTREE(segment_id),
  BITMAP(vehicle) (equality only — BITMAP range queries are documented
  slow; date ranges go to the BTREE).
- Stable row ids NOT used (experimental); compact with
  defer_index_remap=True (Fragment Reuse Index) so compaction doesn't
  force index rebuilds.
- Scan tuning: batch_size 8192 (2MB batches at 256B/row), scan_in_order=
  False, late_materialization=["vector_fp"] where the scanner drives
  re-rank. Hydration path: LANCE_IO_THREADS 64-256 with io_buffer_size
  raised in step (~32MB per IO thread); raise lance_aimd_max_rate if
  hydration saturates the default limiter.
- Ops loop: daily inference appends in batches (never per-row) ->
  compact_files -> optimize_indices() -> workers hydrate the new version
  incrementally.

## Correctness spec (the eps bound)

Per-dim symmetric int8 quantization error |e_d| <= s_d/(2*127). For a unit
query q projected to the PCA basis:

- **Hard bound (Cauchy-Schwarz)**: |q . e| <= ||q||_2 * ||s||_2/254
  = ||s||_2/254. This eps makes tau-eps a provable superset (GEMINI/LBP).
  Hoelder fallback: sum_d |q_d| s_d / 254. This bound is against whichever
  fp32 value the re-rank column holds; see the Lance dataset spec's
  `vector_fp` entry and `eps_bound.py`'s module docstring for when that is
  the true pre-quantization score vs a dequantized-int8 fallback.
- int8 score >= tau + eps => guaranteed member. The implementation
  unconditionally re-ranks these rows too (not "only if exact export scores
  are needed" as an earlier draft of this spec said): re-ranking is cheap
  relative to ABOVE's own take() cost, and it makes "every returned score
  >= tau" hold outright rather than resting on the (empirically large but
  formally unbounded) margin between the fastmath-fp32 screening kernel and
  the fp64 re-rank. [tau-eps, tau+eps) band => mandatory fp32 re-rank.
  Below tau-eps => provably excluded.
- **Never prune on a partial inner-product sum** — IP is not monotone in
  dimensions. Progressive pruning (if ever adopted) must use the per-row
  tail bound partial_IP + ||q_tail|| * tail_norm_row.
- CI check per corpus build (not implemented in this PR — see Implementation
  plan): measured band population vs eps on a 1M sample; alert if band
  selectivity drifts above the take() budget. Note this bounds BAND
  selectivity, not the take() volume that actually drives cost (ABOVE union
  BAND -- see Node sizing's latency budget).

## Optional accelerator (benchmark-gated, exactness-preserving)

Progressive head/tail split (Panorama/PDX-style): store PCA dims 0-63
(energy-ordered by construction) as a separate head column (64GB @1B);
scan head, prune rows where partial_IP + ||q_tail|| * tail_norm <
tau - eps (hard CS bound); fetch tail only for survivors. Cuts
steady-state bytes ~3-4x (PDX SIGMOD'25: vertical layout +40% for
dim-pruned SIMD scans). Adopt only if benchmarks show the margin is
needed — it complicates the writer and kernel.

## Implementation plan (one week to 100M live)

"Complete" means the deterministic scan fleet serving both the interactive
app and the batch scan from the single Lance dataset at the current/100M
corpus at N=2-4. Reaching 1B is `N += 1` as the corpus grows (see below),
not further engineering. One week is achievable because nearly every piece
reuses code the repo already runs in production — the score-lossless
SVD + per-dim int8 quant and the numba kernel (gpu_corpus.py: measured
0.99995 correlation, ~49 GB/s), the resident-shard load pattern, and the
Lilypad/Ray distributed-scan substrate; only the fleet wiring (worker
service, aggregator, k8s manifests) is net-new. Research grounds the
FEASIBILITY of each day (findings §2, §5, §7); the day estimates assume the
Day-1 gate passes, no cross-region blocker, and focused effort.

- **Day 1 — Pipeline core + benchmark gate.** Write the Ray/Lilypad rewrite
  job (findings §5, lance-ray) that reads the per-rank lancedb shards,
  applies the existing build_gpu_corpus SVD+int8 math, and emits a 100M
  slice as one sorted 2.2 dataset with FSL columns. On that slice measure
  the open evidence gaps (§8): resident-shard scan GB/s (16 vs 32 vCPU),
  cold hydration from object store (LANCE_IO_THREADS 64->256 sweep), take()
  latency for the ~10k-row eps band (local NVMe vs object store), and a
  2-worker fan-out with the coverage check. **Go/no-go:** warm query <3s,
  hydration <60s/pod, extrapolated 1B fleet math holds. A low scan number
  raises N (cost), not the deadline — N is the free variable; a slow
  object-store take() selects the local-NVMe node variant (Node sizing), a
  node-type choice, not new code. Either way the design absorbs the miss
  without a redesign.
- **Day 2 — Full pipeline + writer.** Build the dataset over the full
  current corpus: compact_files(target 1M rows), BTREE(chunk_start_unix),
  BTREE(segment_id), BITMAP(vehicle); verify score correlation on a 100M
  sample against the fp32 path. Land the FSL-column writer change in
  ../finetuning so future inference batches append already in the target
  layout. Scaffold the scan-worker service.
- **Day 3 — Worker + aggregator.** Scan-worker: resident int8 shard
  (gpu_corpus load pattern), numba kernel screening at tau-eps (hard
  Cauchy-Schwarz bound), eps-band fp32 re-rank via take(), and
  (dataset_version, fragments_scanned) on every response. Aggregator:
  stateless fan-out, threshold union (no top-k merge), fragment-coverage
  assert with an explicit incomplete flag. Fragment ids striped
  round-robin (§5, ShardedFragmentSampler).
- **Day 4 — k8s deploy (OKE, us-phoenix-1).** Node pool in the corpus
  bucket's region (Deployment topology): StatefulSet (worker r owns
  fragments {i mod N == r}) + headless Service + aggregator Deployment +
  HPA; emptyDir/local-PV NVMe cache; readiness = shard@version resident.
  web_server (Cloud Run, us-west1, unchanged) gains the distributed-corpus
  backend beside the existing load_corpus dispatch, superseding the
  download-everything path for this corpus; the batch scan API routes
  through the aggregator.
- **Day 5 — 100M live + turn-up.** Bring the 100M corpus up at N=2-4;
  end-to-end correctness (coverage assert green, score-correlation vs the
  current path, batch-scan parity against segments.lance). Enable RF2 for
  the interactive cluster (RAM is cheap at this size). Capture the scaling
  knob and refresh loop as a runbook.

**Beyond the week (operational, corpus-gated).** As the corpus grows toward
1B, raise N so each shard stays <= worker RAM and hydrate new fragment ids
incrementally; the architecture is unchanged (Cluster architecture, Node
sizing). This is a config/turn-up step, not new code — it cannot be "done
in a week" only because the 1B corpus does not exist yet; the path that
serves it ships on Day 5.

**Out of scope this week.** The optional progressive head/tail accelerator
and any IVF_PQ browse-only side index (Optional accelerator section) —
revisit only if the Day-1 gate or production margins demand. Also out of
scope in the current single-node `threshold_search` module: date/segment/
vehicle prefiltering against the scalar indices this spec builds (needs a
`take()`-addressable filtered scan, which needs stable row ids or Lance's
internal row-address API — see `threshold_search.py`'s module docstring) and
the distributed fleet itself (aggregator, striping, coverage protocol).

## Risks

| Risk | Mitigation |
|---|---|
| Worker down => incomplete results | RF2 interactive; coverage protocol makes incompleteness explicit, never silent; batch retries |
| Straggler blows the budget | striped shards equalize load; hedged retry to replica; per-worker deadline < budget |
| Commodity-node scan slower than repo's 49GB/s | P0 measures the actual shape; N is the free variable |
| Object-store take() latency (RAM-only nodes) | take() volume = ABOVE + BAND (the full result set, not just band ~1e-4 selectivity — see Node sizing); Lance parallel take at 64-256 IO threads; NVMe variants make it ~ms; size the node variant from the workload's expected result selectivity |
| Post-compaction reshuffle churn | ~1 min/worker rehydration; compact off-peak; defer_index_remap |
| Cross-cloud hop (app Cloud Run us-west1, workers OKE us-phoenix-1) | aggregator co-located on OKE; only cross-cloud call is app to aggregator (KB request / MB response); workers stay in-region to the corpus bucket |
| eps band drift with future models | CI band-population check per corpus build |

## Relationship to README "Scaling beyond 1M"

That section's FixedSizeList rewrite is absorbed (P1); its shard-and-merge
option won (upgraded to threshold-union + coverage protocol); its IVF_PQ
recommendation is superseded — ANN violates the zero-false-negative
requirement clarified since it was written; its encoder-throughput notes
remain valid unchanged. Update that section when this design lands.
