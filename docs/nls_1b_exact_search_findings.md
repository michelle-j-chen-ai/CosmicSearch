# Findings: exact threshold retrieval at 100M-1B — repo analysis + verified research

Compiled 2026-07-03. Evidence base for
[nls_1b_exact_search_design.md](nls_1b_exact_search_design.md): repo
analysis (file:line references), validation of prior internal research
(2026-06-14), and three verified research sweeps (Lance best practices,
cloud/hardware bandwidth, exact-search prior art). Source-class labels:
repo-measured / official spec / measured (third-party benchmark) / paper.

## 1. Requirement and workload (confirmed in code)

- Target: similarity matching over 100M -> 1B Cosmos-Embed1 768-d
  embeddings, <10s per query.
- The workload is THRESHOLD/RANGE retrieval with a zero-false-negative
  requirement, not top-k: the batch scan applies per-tag cosine cutoffs and
  returns ALL segments above them (nls_launcher.py:180-190,
  docs/vlm_scan_api.md), and the UI pages/exports by threshold — "the
  app's threshold paging and exports need every row's score"
  (gpu_corpus.py:310-313). A missed boundary vector silently drops a
  training example. This single fact disqualifies ANN for the retrieval
  path.
- Stack: pylance 4.0.0 + lancedb 0.30.2 (requirements.txt:7-8); corpus on
  OCI S3-compat object storage; app on GCP Cloud Run (32Gi/8vCPU cap,
  project*.toml).

## 2. Current state: Lance is a transport format, not a search engine

- load_corpus (search_engine.py:260) downloads the WHOLE corpus to disk
  cache, then fully materializes a numpy matrix; scoring is a brute-force
  numpy gemv in rank_top_k (search_engine.py:510). Lance-native vector
  search is never invoked anywhere in the repo.
- Four load paths (search_engine.py:270-281): GPU int8-PCA artifact (npy),
  embeddings.npy fast path, direct .lance dataset via
  lance.dataset().to_table() (full materialization), and rank=NNNNN/
  sharded lancedb tables concatenated shard by shard
  (search_engine.py:430).
- Two writer-side layout defects (also flagged in README.md:52-56):
  1. The inference pipeline writes `vector` as a plain list<float32>, not
     FixedSizeList — Lance vector indexing and zero-copy reads are
     impossible without a rewrite.
  2. The corpus is fragmented into per-rank lancedb directories — no
     single dataset to index or scan-plan against.
- 48M-row precedent (gpu_corpus.py): offline artifact with uncentered
  truncated SVD 768->256 (score-lossless; projected dot product equals
  original cosine) + per-dim symmetric int8 quantization; measured score
  correlation vs fp32-768 = 0.99995. 47.8M x 256 int8 = 12.2GB resident.
  CPU numba kernel: 12.2GB scanned in ~0.25s (~49GB/s, memory-bandwidth-
  bound) on the 8 vCPU instance. Metadata tricks: source_media_uri and
  chunk_id reconstructed row-exact from run_uuid + chunk_start (saves
  ~9GB resident at 48M; ~185GB on-disk at 1B).
- Region mismatch: scan workers pinned to us-chicago-1
  (nls_launcher.py:45) while the bucket endpoint is us-phoenix-1 — every
  corpus byte crosses regions today.
- Existing distribution substrates: Lilypad/Ray batch scans (16 CPU nodes,
  num_blocks=64, LANCE_IO_THREADS=16) and an offline Spark scan workflow
  (interval_core.py is importable by Spark executors); web_server dedups
  per-executor duplicate launches (idempotency key + pg advisory lock).

## 3. Quantified costs at scale (768-d)

| Resident format | @100M | @1B | Verdict |
|---|---|---|---|
| fp32 matrix | 307 GB | 3.07 TB | impossible in RAM tiers used |
| fp16 matrix | 154 GB | 1.54 TB | impossible |
| int8 PCA-256 | 25.6 GB | 256 GB | 100M: >32Gi Cloud Run cap; 1B: shard 256GB/N across workers |
| IVF_PQ codes (96 sub-vec) | ~1 GB | ~9.7 GB | ANN only — disqualified for retrieval path |

Compute is NOT the wall (numba kernel extrapolates to ~0.5s @100M,
~5s @1B single-node — and shards linearly). The walls are (a) residency,
(b) cold-download of the full corpus, (c) object-store bandwidth per node.
The transient double-residency of the current loader (Arrow table + numpy
concat coexist, search_engine.py:443-476) doubles peak RAM during load.

## 4. Validation of prior internal research (2026-06-14)

Confirmed: threshold workload + zero-false-negative requirement (see §1);
the two-phase design (scalar prefilter -> int8 coarse screen at tau-eps ->
fp32 re-rank of survivors via random access); the bytes-moved /
bandwidth sizing model (matches repo-measured kernel numbers).

Corrections (source code / primary sources as truth):
- "~1M in memory, ~10ms": actual is ~58ms matmul + ~40ms encode
  (README.md:38-43), and production already runs 47.8M rows via int8
  PCA-256 — the research missed the repo's strongest asset (score-lossless
  PCA), which is what makes the screen column 256B/row.
- "614GB RAM @100M": steady-state fp32 is 307GB; 614GB matches only the
  loader's transient double-residency (or an fp64 assumption).
- "_take_rows()": the public API is Dataset.take(indices, columns).
- "Milvus range = top-k only": outdated (Milvus 2.3+ has range search) —
  but the disqualification stands: it inherits ANN recall, no completeness
  guarantee.
- "2.2s @100M w/ 10% prefilter on 7GB/s NVMe": at int8-768 the design
  MISSES budget without prefilter help (76.8GB -> ~11s); with PCA-256 the
  full no-prefilter scan is 25.6GB -> ~3.7s — the guarantee stops
  depending on prefilter selectivity.
- Deployment gap: Cloud Run has no local NVMe (/tmp is RAM tmpfs;
  gcs-fuse reads measured pathological, search_engine.py:347-355) — the
  design implies leaving Cloud Run regardless.

## 5. Lance best practices at billion scale (verified against official sources)

- Fragment-parallel scanning is first-class: scanner(fragments=
  [LanceFragment...]) is in the public API; fragments serialize via
  to_json()/FragmentMetadata.from_json for cross-worker distribution;
  ShardedFragmentSampler(rank, world_size) is the reference sharding
  pattern; lance-ray provides distributed read/write/compaction.
- Topology validation at 10x target: LanceDB's published 10B-vector
  architecture (blog, 2026-04-29) = coordinator + per-segment plan
  executors, 10 segments x 1B rows, quantized scan + full-precision
  rerank, p99 21ms (ANN). Same scatter-gather + rerank shape as this
  design; ours differs by keeping a deterministic guarantee.
- Production 1B+ on Lance + S3 exists (AWS blog 2025-09-22: 3.5B vectors,
  12.9TB, IVF-PQ).
- Sizing defaults: max_rows_per_file=1,048,576 / max_bytes_per_file=90GB
  -> ~954 fragments @1B; compact_files(target_rows_per_fragment=1M) skips
  already-large fragments; enable_v2_manifest_paths default on.
- Format 2.1 (stable since Lance 0.38, opt-in via data_storage_version;
  pylance 4.x also has 2.2/2.3): adaptive structural encodings — values
  >=128B use FULL-ZIP = 1 IOP random access for fixed-width columns. Both
  design columns qualify (int8-256 = 256B, fp32-256 = 1024B). arXiv
  2504.15247: 2.1 scans meet/exceed Parquet; tuned-Parquet take ~350k
  rows/s, Lance 2.1 ties or beats.
- IO tuning (official performance guide): LANCE_IO_THREADS default 8
  local / 64 cloud, "may need 128-256 to saturate network";
  io_buffer_size default 2GB (~32MB per IO thread — raise together);
  batch_size default 8192; AIMD rate limiter defaults (max 5000 req/s,
  lance_aimd_* storage options).
- late_materialization=["<col>"]: fetches named columns via take() AFTER
  the filter (heuristic assumes ~0.1% selectivity) — purpose-built for the
  fp32 re-rank column.
- Scalar indexes: BTREE for high-cardinality (search scales with matching
  rows); BITMAP for <1k uniques but documented "extremely slow" for large
  ranges — date ranges belong on BTREE. Fragment pruning via column
  min/max statistics is documented; NO sub-fragment contiguity guarantee —
  physically sort rows at write time. Unindexed appended tail = flat scan
  until optimize_indices().
- Stable row ids are EXPERIMENTAL (stable across compaction, not updates)
  — avoided; compact_files(defer_index_remap=True) (Fragment Reuse Index)
  decouples compaction from index remaps instead.
- Exact search support: nearest={"use_index": False} is the documented
  brute-force path; batch queries supported. No GPU brute-force in Lance.
- EVIDENCE GAPS (must benchmark ourselves): no official Lance S3 or NVMe
  scan GB/s benchmarks anywhere; FSL int8 compression default
  undocumented; no numeric fragment-count cliff; no official
  append->compact cadence.

## 6. Verified hardware/cloud bandwidth (2026-07-03; GCP quotes MiB/s)

- Local NVMe, single node CAN exceed 26GB/s — but only on high-end shapes:
  GCP z3-highmem-176 36,000 MiB/s (~37.7GB/s) official; OCI
  BM.DenseIO.E4.128 ~40GB/s measured (Oracle FIO, 2025-09). Excluded from
  the design by deployment constraint; useful as benchmark references.
- Mainstream shapes (the design's building blocks): GCP C3/C3D-lssd up to
  ~13.1GB/s local NVMe; OCI VM.DenseIO.E5.Flex 8-48 OCPU / 96-576GB / 1-6
  NVMe drives / <=40Gbps NIC; OCI VM.Standard.E5.Flex up to ~1TB RAM.
- Network-attached storage cannot reach 26GB/s per VM: GCP Hyperdisk
  per-VM ceiling 25,000 MiB/s (M4N-224 only, read+write shared); object
  store per-VM is NIC-bound — GCS ~13GB/s demonstrated max (c4-highcpu-144
  + 150Gbps Tier1 + 12 buckets), OCI ~10-12.5GB/s (100Gbps bare metal),
  OCI VMs cap at 40Gbps = 5GB/s.
- RAM: 1TB-RAM VMs are inexpensive relative to storage-optimized (GCP
  m3-ultramem-32 976GB ~$4.4k/mo list); measured STREAM: Genoa dual-socket
  ~780GB/s, SPR ~220-260GB/s/socket. Residency is the cheap bandwidth.
- GPU: H100 SXM 3.35TB/s HBM (8x = 640GB / 26.8TB/s aggregate) — roofline
  scans 256GB int8 in ~10-80ms, but list cost $58-68k/mo (8xH100),
  $23-30k/mo (8xA100), and no library supports exact range on GPU (§7).
- Cost ladder (list/mo): 1TB-RAM VM $3-9k < 8-NVMe storage VM $5-8k <
  Z3-176 ~$16k < 8xA100 $23-30k < 8xH100 $58-68k. Commodity worker nodes
  for the design: ~$0.7-1.5k/mo each.

## 7. Prior art: exact threshold retrieval (verified survey)

- Pattern validation: "cheap lower-bound filter -> superset -> exact
  refine" is the textbook GEMINI / Lower-Bounding-Property method;
  2024-25 exact-search SoTA builds exactly this way (SOFA arXiv
  2411.17483; Panorama arXiv 2510.00566 — progressive Cauchy-Schwarz
  bounds, 2-40x over IVF at 100M).
- Gap confirmed: NO published system does provably-complete
  cosine-threshold retrieval on dense vectors at >=100M. Everything at
  that scale drops completeness: Milvus range search inherits ANN recall;
  DiskJoin (SIGMOD'25, 100M-1.4B) is explicitly approximate; SemDeDup
  (LAION-440M) compares only within clusters; Elastic/HF int8 cascades
  target ~95-97% recall. Provably-complete threshold methods (L2AP
  ICDE'14, Bayardo WWW'07, ICDT'19 cosine-threshold) are sparse/low-dim
  only. The design's assembly (int8 hard-eps screen + band re-rank, dense,
  100M+) is sound components in an unpublished combination.
- eps math (correctness-critical): per-dim symmetric quant error
  |e_d| <= s_d/(2*127). Deterministic dot-product error bounds: Hoelder
  sum|q_d|s_d/254; tighter Cauchy-Schwarz ||q||_2*||s||_2/254 = ||s||_2/254
  for unit queries. PARTIAL-sum inner-product pruning is NOT sound (IP is
  not monotone in dimensions, unlike partial squared-L2) — progressive
  pruning requires a per-row tail bound (partial_IP + ||q_tail|| *
  tail_norm_row). Full-column scan with the quant-eps bound is unaffected.
- GPU exact range is a library dead end: FAISS GPU has NO range_search on
  any index class (incl. the 2025 cuVS integration; kNN k-select cap
  2048); cuVS brute_force is fp16/fp32 only, dataset must fit VRAM (no
  out-of-core; NVIDIA's sanctioned overflow path is lossy IVF-PQ); RAFT's
  only exact range primitive (RBC eps-NN) is L2/Haversine, not cosine.
  FAISS range_search is exact+complete only on IndexFlat (CPU exhaustive);
  IVF variants are complete only at nprobe=nlist. For IP/cosine, radius is
  a similarity LOWER bound — matching tau semantics.
- Throughput anchors validating the bytes/bandwidth model: ClickHouse
  measured exact cosine scans 12.0GB/s @100Mx768 fp32 (60 cores) and
  80.8GB/s on a larger cluster; Milvus FLAT and ClickHouse-distributed are
  shipping examples of the shard-and-scan-exactly topology; int8 GEMV is
  memory-bandwidth-bound (single-core DRAM ~14-34GB/s; AMX irrelevant —
  GEMM-only), so ~26-50GB/s per multi-core node is a conservative 16-25%
  of one DDR5 socket.
- Optional exact-preserving accelerator: Panorama-style progressive
  energy-ordered bounds on a PDX-style vertical layout (SIGMOD'25: +40%
  for dim-pruned SIMD scans) — PCA output is already energy-ordered;
  head-block scan + hard CS tail bound can cut steady-state bytes ~3-4x.

## 8. Consolidated evidence gaps (benchmark before committing hardware)

1. Lance scan GB/s from local NVMe and object store — no official numbers
   exist; measure on the target node shape (design P0).
2. OCI E5-family NVMe read GB/s — unpublished; E4's measured 40GB/s is the
   only floor proxy.
3. Which version data_storage_version="stable" resolves to in pylance 4.x
   — check ds.data_storage_version at runtime.
4. take() latency at 10k scattered rows on 2.1 full-zip from NVMe vs
   object store — measure in P0.
5. eps band population on the real corpus (drives take() volume) — CI
   check per corpus build.

## 9. Primary sources (load-bearing subset)

- Repo: search_engine.py, gpu_corpus.py, nls_launcher.py, local_cache.py,
  oci_s3.py, interval_core.py, web_server.py, README.md, project*.toml.
- Lance: pylance source (dataset.py — scanner/write_dataset/compact
  signatures); lance.org guides (performance, object_store,
  distributed_write, read_and_write); docs.lancedb.com (scalar-index,
  performance); arXiv 2504.15247 (format 2.1 paper, Apr 2025); LanceDB
  blogs — "Lance file 2.1 stable" (2025-10-03), "How LanceDB accelerates
  vector search at 10-billion scale" (2026-04-29); AWS architecture blog
  1B-vector case study (2025-09-22); DuckDB "Test-driving Lance"
  (2026-05-21).
- Hardware: GCP local-SSD / storage-optimized / Hyperdisk / network
  bandwidth docs + live pricing tables (2026-07-03); Oracle compute-shapes
  + performance docs; Oracle Hammerspace Tier-0 FIO blog (2025-09-12);
  zettalane GCS single-VM benchmark (2026-03); NVIDIA A100/H100/H200
  datasheets; McCalpin SPR bandwidth (ISC'23); Phoronix STREAM runs.
- Prior art: FAISS wiki/source + issues #565/#3684/#3650; cuVS/RAFT docs;
  GEMINI (Faloutsos 1994); SOFA arXiv 2411.17483; Panorama arXiv
  2510.00566; PDX arXiv 2503.04422 (SIGMOD'25); L2AP (ICDE'14); Bayardo
  (WWW'07); ICDT'19 cosine-threshold; RaBitQ arXiv 2405.12497 (SIGMOD'24);
  ADSampling arXiv 2303.09855; DiskJoin arXiv 2508.18494 (SIGMOD'25);
  SemDeDup arXiv 2303.09540; Milvus range-search docs; ClickHouse vector
  search blogs (P1/P2, LAION-5B, QBit); Meta Faiss-cuVS blog (2025-05-08).
