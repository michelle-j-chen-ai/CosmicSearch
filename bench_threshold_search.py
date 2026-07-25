"""Manual PoC benchmark for the exact cosine-threshold search primitive.

Proves the design's goal on a real sample corpus staged on S3: `threshold_search`
returns ALL rows with cosine >= tau (zero false negatives), and does it in a
measured, reportable time. Standalone -- no model/torch, no pytest; runnable in
any container/VM with the OCI S3-compat credentials `oci_s3` already reads from
the environment (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL* /
AWS_REGION).

Pipeline (subcommands):

  build   fp32 embeddings.npy (+ metadata.parquet) on S3
            -> download, sample N rows (seeded)
            -> uncentered TruncatedSVD 768->256 (score-lossless; needs scikit-learn)
            -> per-dim symmetric int8 quant (exact gpu_corpus convention)
            -> artifact {pca_components, quant_scales, corpus_int8, metadata,
                         pca_projection_fp32}
            -> lance_writer.build_dataset -> exact-threshold .lance (local)
            -> oci_s3.upload_directory -> s3://.../sample.lance

  bench   exact-threshold .lance on S3 (or --dataset-local a local path)
            -> ThresholdCorpus(ds)              [COLD hydration: resident int8, timed]
            -> threshold_search(q, tau) x R     [WARM per-query: timed, min/p50/p95]
            -> oracle = vector_fp @ q_pca over all rows >= tau
            -> assert zero false negatives; report timings, GB/s, counts, eps

  all     build then bench, in one process, against the just-built local dataset
          (skips the S3 round-trip for the bench half unless --keep-uploaded).

Why this is a real proof and not a self-consistent tautology: `build` writes
`pca_projection_fp32.npy` -- the SVD's own pre-quantization fp32 projection --
so `lance_writer` stores a GENUINE fp32 `vector_fp` (not `dequant(int8)`), and
the int8 screen and the fp32 re-rank/oracle then differ by real quantization
error that the eps bound must absorb. The query is synthetic but built
in-subspace (`q_768 = P^T q_256`, P orthonormal), so the score-lossless-PCA
precondition holds exactly and `vector_fp @ q_pca` is the true cosine, making
the zero-false-negative check a crisp assertion rather than an approximation.

Example:
    # one-shot local proof (no S3), tiny synthetic-free real slice already local:
    python bench_threshold_search.py bench --dataset-local /path/to/sample.lance

    # full pipeline from the real corpus on S3:
    python bench_threshold_search.py build \\
        --embeddings-uri s3://bucket/prefix/embeddings.npy \\
        --sample-rows 1000000 --out-uri s3://bucket/prefix/nls_sample.lance
    python bench_threshold_search.py bench --dataset-uri s3://bucket/prefix/nls_sample.lance
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import lance
import lance_writer
import threshold_search as ts

_N_COMPONENTS = 256
_DEFAULT_REPEATS = 5
_DEFAULT_TAU_PERCENTILE = 99.0
_DEFAULT_SEED = 0


# --------------------------------------------------------------------------
# build: fp32 embeddings -> gpu_corpus int8 artifact (+ true fp32 projection)
# --------------------------------------------------------------------------


def _quantize_per_dim(projected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim symmetric int8 quant: scale_d = max|v[:, d]|, matching gpu_corpus.

    Byte-identical to gpu_corpus.py / lance_writer.py / eps_bound.py's convention
    (dequant = int8 * scale / 127).
    """
    scale = np.abs(projected).max(axis=0).astype("float32")
    scale = np.where(scale == 0, 1.0, scale)  # guard degenerate all-zero dims
    corpus_i8 = np.round(projected * (127.0 / scale)).clip(-127, 127).astype("int8")
    return corpus_i8, scale


def build_int8_artifact(
    embeddings: np.ndarray,
    metadata: pa.Table,
    out_dir: Path,
    *,
    sample_rows: int | None = None,
    seed: int = _DEFAULT_SEED,
) -> Path:
    """Convert raw fp32 embeddings into a gpu_corpus-style int8 PCA artifact.

    Writes the four files `lance_writer.build_dataset` consumes, PLUS
    `pca_projection_fp32.npy` (the pre-quantization PCA-256 projection) so the
    built dataset gets a genuine fp32 `vector_fp` rather than a dequant(int8)
    fallback -- see this module's docstring and `lance_writer.py`'s.

    `embeddings` is (N, 768) fp32, L2-unit-norm rows; `metadata` is a row-aligned
    Arrow table (needs run_uuid, chunk_start_unix, and a per-row id usable as
    segment_id -- `chunk_id` if present, else `segment_id`).
    """
    from sklearn.decomposition import TruncatedSVD  # lazy: only `build` needs it

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = embeddings.shape[0]
    if metadata.num_rows != n:
        raise ValueError(
            f"metadata row count ({metadata.num_rows}) != embeddings row count ({n})"
        )

    if sample_rows is not None and sample_rows < n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=sample_rows, replace=False))
        embeddings = np.ascontiguousarray(embeddings[idx])
        metadata = metadata.take(pa.array(idx))
        n = sample_rows

    embeddings = embeddings.astype("float32", copy=False)
    svd = TruncatedSVD(n_components=_N_COMPONENTS, algorithm="randomized", random_state=seed)
    projected = svd.fit_transform(embeddings).astype("float32")  # (N, 256) true fp32
    pca = svd.components_.astype("float32")  # (256, 768), rows orthonormal
    corpus_i8, scale = _quantize_per_dim(projected)

    np.save(out_dir / lance_writer.PCA_FILE, pca)
    np.save(out_dir / lance_writer.SCALE_FILE, scale)
    np.save(out_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    np.save(out_dir / lance_writer.PRE_QUANT_FP32_FILE, projected)

    cols = set(metadata.column_names)
    seg_source = "chunk_id" if "chunk_id" in cols else "segment_id"
    if seg_source not in cols or "run_uuid" not in cols or "chunk_start_unix" not in cols:
        raise ValueError(
            "metadata must have run_uuid, chunk_start_unix, and one of "
            f"chunk_id/segment_id; got columns {sorted(cols)}"
        )
    out_meta = pa.table(
        {
            "run_uuid": metadata.column("run_uuid"),
            "chunk_start_unix": metadata.column("chunk_start_unix").cast(pa.int64()),
            "segment_id": metadata.column(seg_source),
        }
    )
    pq.write_table(out_meta, out_dir / lance_writer.METADATA_FILE)
    return out_dir


# --------------------------------------------------------------------------
# bench: time + prove zero false negatives against an in-subspace query
# --------------------------------------------------------------------------


@dataclasses.dataclass
class BenchResult:
    corpus_rows: int
    dim: int
    tau: float
    tau_percentile: float
    matches: int
    above: int
    band: int
    band_selectivity: float
    eps: float
    max_int8_gap: float
    eps_covers_int8_gap: bool
    false_negatives: int
    extras: int
    zero_false_negatives: bool
    cold_hydrate_s: float
    warm_query_s_min: float
    warm_query_s_p50: float
    warm_query_s_p95: float
    screen_throughput_gbps: float
    e2e_s: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _in_subspace_query(pca: np.ndarray, seed: int) -> np.ndarray:
    """A unit-norm 768-d query lying exactly in the PCA row space.

    `q_768 = P^T q_256` with `q_256` unit-norm and P's rows orthonormal, so
    `||q_768|| == 1` and `P q_768 == q_256`: projecting the query back recovers
    it exactly, and the score-lossless-PCA precondition (Design "Correctness
    spec") holds, making `vector_fp @ (P q_768)` the true cosine.
    """
    rng = np.random.default_rng(seed)
    q_256 = rng.standard_normal(pca.shape[0]).astype(np.float64)
    q_256 /= np.linalg.norm(q_256)
    q_768 = pca.astype(np.float64).T @ q_256
    return q_768.astype("float32")


def _oracle_scores(ds: lance.LanceDataset, query_pca64: np.ndarray) -> np.ndarray:
    """Exact score of every row against `query_pca64`, read from vector_fp.

    Deliberately the slow full-column scan the fast path avoids -- the
    independent reference. With a genuine fp32 vector_fp (built via
    pca_projection_fp32.npy) this is the TRUE pre-quantization score, not a
    dequant(int8) copy of the screen.
    """
    table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN], scan_in_order=True)
    fp = ts._fixed_size_list_matrix(table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    return fp @ query_pca64


def run_benchmark(
    ds: lance.LanceDataset,
    *,
    repeats: int = _DEFAULT_REPEATS,
    tau_percentile: float = _DEFAULT_TAU_PERCENTILE,
    seed: int = _DEFAULT_SEED,
) -> BenchResult:
    """Time cold hydration + warm queries and prove zero false negatives.

    Takes an already-open exact-threshold dataset (local or S3-backed) so this is
    unit-testable without any S3 dependency.
    """
    if not lance_writer.is_exact_threshold_dataset(ds):
        raise ValueError("run_benchmark requires an exact-threshold Lance dataset")

    pca, scale = lance_writer.read_pca_metadata(ds)
    query = _in_subspace_query(pca, seed)
    query_pca = ts._project_query(query, pca)
    query_pca64 = query_pca.astype(np.float64)

    # Oracle + tau. tau at a percentile of the true score distribution so the
    # match set is a meaningful, non-trivial fraction of the corpus.
    oracle = _oracle_scores(ds, query_pca64)
    tau = float(np.percentile(oracle, tau_percentile))
    oracle_ids = set(np.nonzero(oracle >= tau)[0].tolist())

    e2e_t0 = time.perf_counter()

    # COLD: resident int8 hydration (once per pod in the design).
    cold_t0 = time.perf_counter()
    corpus = ts.ThresholdCorpus(ds)
    cold_hydrate_s = time.perf_counter() - cold_t0

    # Warm up once (numba JIT compiles on first kernel call; excluded from timing).
    corpus.threshold_search(query, tau)

    warm_times: list[float] = []
    hits = []
    for _ in range(max(1, repeats)):
        q_t0 = time.perf_counter()
        hits = corpus.threshold_search(query, tau)
        warm_times.append(time.perf_counter() - q_t0)

    e2e_s = time.perf_counter() - e2e_t0

    hit_ids = {h.row_id for h in hits}
    false_negatives = len(oracle_ids - hit_ids)
    extras = len(hit_ids - oracle_ids)

    # Screen-level diagnostics: how much did int8 quantization move scores vs
    # the eps window? Confirms eps is a real, non-vacuous bound on this corpus.
    n = corpus.num_rows
    eps = float(ts.eps_bound.eps_cauchy_schwarz(scale))
    screen = np.empty(n, dtype=np.float32)
    w = (query_pca * (scale.astype(np.float32) / np.float32(127.0))).astype(np.float32)
    ts.gpu_corpus._cpu_score_kernel()(
        np.ascontiguousarray(corpus._resident.corpus_i8), w, screen
    )
    above, band, _below = ts.eps_bound.classify(screen, tau, eps)
    max_int8_gap = float(np.abs(screen.astype(np.float64) - oracle).max())

    warm = sorted(warm_times)
    screen_bytes = n * pca.shape[0]  # int8 -> 1 byte/dim
    screen_gbps = (
        screen_bytes / (warm[len(warm) // 2] * 1e9) if warm[len(warm) // 2] > 0 else 0.0
    )

    return BenchResult(
        corpus_rows=n,
        dim=int(pca.shape[0]),
        tau=tau,
        tau_percentile=tau_percentile,
        matches=len(oracle_ids),
        above=int(above.sum()),
        band=int(band.sum()),
        band_selectivity=float(band.sum()) / n if n else 0.0,
        eps=eps,
        max_int8_gap=max_int8_gap,
        eps_covers_int8_gap=max_int8_gap <= eps,
        false_negatives=false_negatives,
        extras=extras,
        zero_false_negatives=false_negatives == 0,
        cold_hydrate_s=cold_hydrate_s,
        warm_query_s_min=warm[0],
        warm_query_s_p50=warm[len(warm) // 2],
        warm_query_s_p95=warm[min(len(warm) - 1, int(len(warm) * 0.95))],
        screen_throughput_gbps=screen_gbps,
        e2e_s=e2e_s,
    )


def _print_report(result: BenchResult, *, dataset_ref: str) -> None:
    r = result
    verdict = "PASS" if r.zero_false_negatives else "FAIL"
    print(f"\n=== threshold_search PoC benchmark: {dataset_ref} ===")
    print(f"corpus rows        : {r.corpus_rows:,} x {r.dim}d (int8 screen = "
          f"{r.corpus_rows * r.dim / 1e9:.3f} GB)")
    print(f"tau                : {r.tau:.6f}  (p{r.tau_percentile} of true score)")
    print(f"matches (>= tau)   : {r.matches:,}  ({r.matches / r.corpus_rows:.4%} of corpus)")
    print(f"  ABOVE / BAND     : {r.above:,} / {r.band:,}  "
          f"(band selectivity {r.band_selectivity:.2e})")
    print(f"eps (CS bound)     : {r.eps:.6f}")
    print(f"max |int8 - true|  : {r.max_int8_gap:.6f}  "
          f"(eps covers it: {r.eps_covers_int8_gap})")
    print("-- timing --")
    print(f"cold hydration     : {r.cold_hydrate_s:.3f} s  (resident int8, once)")
    print(f"warm query (p50)   : {r.warm_query_s_p50:.3f} s  "
          f"(min {r.warm_query_s_min:.3f}, p95 {r.warm_query_s_p95:.3f})")
    print(f"int8 screen tput   : {r.screen_throughput_gbps:.1f} GB/s")
    print(f"e2e (hydrate+{_DEFAULT_REPEATS}q)  : {r.e2e_s:.3f} s")
    print("-- exactness --")
    print(f"false negatives    : {r.false_negatives}   extras: {r.extras}")
    print(f"ZERO FALSE NEG     : {verdict}")


# --------------------------------------------------------------------------
# S3 glue + CLI
# --------------------------------------------------------------------------


def _s3():
    import oci_s3

    return oci_s3, oci_s3.s3_client()


def _download_embeddings(embeddings_uri: str, dest: Path) -> tuple[np.ndarray, pa.Table]:
    oci_s3, client = _s3()
    base = embeddings_uri.rsplit("/", 1)[0]
    emb_path = dest / "embeddings.npy"
    meta_path = dest / "metadata.parquet"
    print(f"downloading {embeddings_uri} ...")
    oci_s3.download_object(embeddings_uri, emb_path, client)
    oci_s3.download_object(base + "/metadata.parquet", meta_path, client)
    return np.load(emb_path), pq.read_table(meta_path)


def cmd_build(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        embeddings, metadata = _download_embeddings(args.embeddings_uri, tmp_path)
        print(f"loaded embeddings {embeddings.shape}; building int8 artifact "
              f"(sample_rows={args.sample_rows}) ...")
        artifact_dir = build_int8_artifact(
            embeddings, metadata, tmp_path / "artifact",
            sample_rows=args.sample_rows, seed=args.seed,
        )
        local_lance = tmp_path / "out.lance"
        print("building exact-threshold Lance dataset ...")
        lance_writer.build_dataset(artifact_dir, str(local_lance))
        oci_s3, client = _s3()
        print(f"uploading dataset -> {args.out_uri} ...")
        count = oci_s3.upload_directory(local_lance, args.out_uri, client)
        print(f"uploaded {count} objects to {args.out_uri}")
    return 0


def _open_dataset(args: argparse.Namespace) -> tuple[lance.LanceDataset, str]:
    if args.dataset_local:
        return lance.dataset(args.dataset_local), args.dataset_local
    import oci_s3

    return (
        lance.dataset(args.dataset_uri, storage_options=oci_s3.lance_storage_options()),
        args.dataset_uri,
    )


def cmd_bench(args: argparse.Namespace) -> int:
    ds, ref = _open_dataset(args)
    result = run_benchmark(
        ds, repeats=args.repeats, tau_percentile=args.tau_percentile, seed=args.seed
    )
    _print_report(result, dataset_ref=ref)
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0 if result.zero_false_negatives else 1


def cmd_all(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        embeddings, metadata = _download_embeddings(args.embeddings_uri, tmp_path)
        artifact_dir = build_int8_artifact(
            embeddings, metadata, tmp_path / "artifact",
            sample_rows=args.sample_rows, seed=args.seed,
        )
        local_lance = tmp_path / "out.lance"
        lance_writer.build_dataset(artifact_dir, str(local_lance))
        if args.out_uri:
            oci_s3, client = _s3()
            oci_s3.upload_directory(local_lance, args.out_uri, client)
            print(f"uploaded to {args.out_uri}")
        result = run_benchmark(
            lance.dataset(str(local_lance)),
            repeats=args.repeats, tau_percentile=args.tau_percentile, seed=args.seed,
        )
        _print_report(result, dataset_ref=str(local_lance))
        if args.json:
            Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
    return 0 if result.zero_false_negatives else 1


def _add_bench_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS,
                   help="warm queries to time (after a JIT warmup)")
    p.add_argument("--tau-percentile", type=float, default=_DEFAULT_TAU_PERCENTILE,
                   help="tau set at this percentile of the true score distribution")
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    p.add_argument("--json", type=str, default=None, help="also write the result as JSON here")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="fp32 embeddings on S3 -> exact-threshold .lance on S3")
    b.add_argument("--embeddings-uri", required=True,
                   help="s3://.../embeddings.npy (metadata.parquet expected alongside)")
    b.add_argument("--out-uri", required=True, help="s3://.../<name>.lance destination prefix")
    b.add_argument("--sample-rows", type=int, default=None,
                   help="downsample to this many rows (seeded); omit for the full corpus")
    b.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    b.set_defaults(func=cmd_build)

    n = sub.add_parser("bench", help="benchmark + prove zero-FN on an exact-threshold dataset")
    src = n.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset-uri", help="s3://.../<name>.lance (lance-native S3 read)")
    src.add_argument("--dataset-local", help="local path to an exact-threshold .lance dataset")
    _add_bench_args(n)
    n.set_defaults(func=cmd_bench)

    a = sub.add_parser("all", help="build then bench in one process (local bench)")
    a.add_argument("--embeddings-uri", required=True)
    a.add_argument("--out-uri", default=None, help="optional: also upload the built dataset here")
    a.add_argument("--sample-rows", type=int, default=None)
    _add_bench_args(a)
    a.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
