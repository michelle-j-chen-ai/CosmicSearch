"""Standalone benchmark: brute-force vs fast_curation vs exact_scores.

Builds a corpus into an exact-threshold Lance dataset and runs the same query
three ways:

  brute-force     the whole fp32 model-space matrix resident, one numpy gemv
                  per query; the ground-truth membership and score reference
                  is computed in the PCA-256 space the Lance paths score in
  fast_curation   the shipped default: only the int8 PCA column resident,
                  screened at `tau - eps`; ABOVE rows accepted with their
                  bounded screening score (no `take()`), BAND rows re-ranked
                  from `vector_fp` via `take()` (see `threshold_search.py`)
  exact_scores    internal reference: like fast_curation but re-ranks ABOVE
                  too, so every returned score is exact -- used to measure
                  the screening-score deviation fast_curation introduces

All three must agree on membership (no false dismissals, no extras); the run
exits non-zero otherwise. Score deviation is reported per-path, split by
score kind (exact vs bounded_approx), and the eps bound is checked for every
bounded row.

`--tau-percentile` matters as much as `--rows`: every BAND row costs a
`take()`, so a broad threshold benchmarks the re-rank path while a tight one
(the curation workload this is built for) benchmarks the screen.

Corpus source (`--source`), none of which require credentials by default:

  synthetic  rows generated in-process at any `--rows` (the default) -- the
             only source that scales far enough for the latency comparison
             to mean anything
  fixture    the committed 10k-row sample of a real corpus (see
             `FIXTURE_PATH`); a real score distribution with no network, but
             too small for its timings to be anything but overhead
  <path|uri> a real `embeddings.npy` plus its sibling `metadata.parquet`,
             local or `s3://`, reading only the leading `--rows` rows

    python bench_threshold_search.py                        # 100k synthetic
    python bench_threshold_search.py --rows 1000000
    python bench_threshold_search.py --tau-percentile 99.0  # broader match set
    python bench_threshold_search.py --source fixture       # real, offline
    python bench_threshold_search.py --rows 200000 --source \\
        s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/\\
embeddings/v3_lr_5e5-ckpt-6549_npy/embeddings.npy
"""

from __future__ import annotations

import argparse
import ast
import io
import shutil
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import lance
import lance_writer
import threshold_search as ts
from bench_common import MODEL_DIM, pca_basis

D = 256
_REPEATS = 5
# A 10,000-row seeded random sample of the production corpus, stored as its
# PCA-256 projection plus the basis (derived from all 902,827 rows) rather
# than raw 768-d vectors: the corpus is rank-256, so reconstruction is exact
# to ~2e-4 in score -- 60x below the eps bound -- at a third of the bytes.
# Regenerate with tools/make_real_fixture.py.
FIXTURE_PATH = Path(__file__).parent / "tests" / "fixtures" / "real_corpus_sample.npz"
# Per-dim energy decay of the synthetic latent. Real Cosmos-Embed corpora are
# strongly concentrated in their leading dimensions; measured against
# v3_lr_5e5-ckpt-6549 (902,827 rows), the top 64 of 256 components hold 0.9798
# of the energy and the top 128 hold 0.9953. d**-0.95 reproduces that closely
# (0.9822 / 0.9938) and lands eps within ~4% of the real corpus's. Flat
# (uniform-energy) latents do not: they leave the top 64 at ~0.29 and shrink
# the per-dim quantization-scale spread from ~146x to ~1.4x, which is what the
# eps bound is sized against.
_SPECTRUM_DECAY = 0.95


def synthetic_embeddings(n: int, seed: int = 0) -> tuple[np.ndarray, pa.Table]:
    """`n` unit-norm model-space rows with a realistic energy spectrum.

    Rows live IN a 256-d subspace, so the PCA-256 projection is exactly
    score-lossless -- a property the real corpus has too (its top 256 of 768
    components hold 1.00000 of the energy), and the precondition that makes
    the model-space and PCA-space scores comparable at all.
    """
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((MODEL_DIM, D)))
    pca = np.ascontiguousarray(basis.T.astype("float32"))  # (256, 768), energy-ordered

    decay = (np.arange(1, D + 1) ** -_SPECTRUM_DECAY).astype("float32")
    latent = (rng.standard_normal((n, D)) * decay).astype("float32")
    latent /= np.linalg.norm(latent, axis=1, keepdims=True)
    embeddings = (latent @ pca).astype("float32")

    return embeddings, _metadata_for(n)


def _metadata_for(n: int) -> pa.Table:
    """Synthetic scalar metadata: it takes no part in scoring, only in the
    writer's sort/index and the fields a hit carries back."""
    return pa.table(
        {
            "run_uuid": pa.array([f"run-{i % 100}" for i in range(n)]),
            "chunk_start_unix": pa.array(np.arange(n, dtype="int64")),
            "segment_id": pa.array([f"seg-{i}" for i in range(n)]),
        }
    )


def fixture_embeddings(rows: int | None = None) -> tuple[np.ndarray, pa.Table, np.ndarray]:
    """The committed real-corpus sample, reconstructed to model space.

    Returns `(embeddings, metadata, pca)` -- the basis comes with the fixture
    because it was derived from the full corpus, which a 10k sample cannot
    reproduce on its own.
    """
    with np.load(FIXTURE_PATH) as data:
        latent, pca = data["latent"], data["pca"]
    if rows:
        latent = latent[:rows]
    embeddings = np.ascontiguousarray(latent @ pca, dtype="float32")
    return embeddings, _metadata_for(embeddings.shape[0]), pca


def real_embeddings(source: str, rows: int) -> tuple[np.ndarray, pa.Table]:
    """Leading `rows` of a real `embeddings.npy` + its sibling metadata.

    `source` is a local path or an `s3://` URI; only the bytes for those rows
    are fetched, so pointing at a multi-GB corpus does not download it.
    """
    meta_source = source.rsplit("/", 1)[0] + "/metadata.parquet"
    if source.startswith("s3://"):
        import oci_s3

        client = oci_s3.s3_client()

        def _get(uri: str, byte_range: str | None = None) -> bytes:
            bucket, key = uri[len("s3://"):].split("/", 1)
            kwargs = {"Range": byte_range} if byte_range else {}
            return client.get_object(Bucket=bucket, Key=key, **kwargs)["Body"].read()

        head = _get(source, "bytes=0-255")
        preamble = 10 + struct.unpack("<H", head[8:10])[0]
        header = ast.literal_eval(head[10:preamble].decode().strip())
        n_total, dim = header["shape"]
        rows = min(rows, n_total)
        raw = _get(source, f"bytes={preamble}-{preamble + rows * dim * 4 - 1}")
        embeddings = np.frombuffer(raw, dtype=header["descr"]).reshape(rows, dim)
        metadata = pq.read_table(io.BytesIO(_get(meta_source)))[:rows]
    else:
        embeddings = np.array(np.load(source, mmap_mode="r")[:rows])
        metadata = pq.read_table(meta_source)[:rows]
        rows = embeddings.shape[0]

    columns = set(metadata.column_names)
    if "segment_id" not in columns:
        # The production artifact carries chunk_id, not segment_id; the writer
        # requires one per-row id it can index.
        metadata = metadata.append_column("segment_id", metadata.column("chunk_id"))
    return np.ascontiguousarray(embeddings, dtype="float32"), metadata.select(
        ["run_uuid", "chunk_start_unix", "segment_id"]
    )


def build_corpus(
    embeddings: np.ndarray, metadata: pa.Table, out_dir: Path, pca: np.ndarray
) -> str:
    """Quantize + write the exact-threshold dataset; returns its URI."""
    artifact_dir = out_dir / "artifact"
    artifact_dir.mkdir(parents=True)

    projected = (embeddings @ pca.T).astype("float32")
    scale = np.abs(projected).max(axis=0).astype("float32")
    corpus_i8 = np.clip(np.round(projected * 127.0 / scale), -127, 127).astype(np.int8)

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    # A genuine pre-quantization projection, so vector_fp is a real fp32
    # reference rather than dequant(int8) -- otherwise the exactness check
    # would only re-derive the screen in higher precision.
    np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, projected)
    pq.write_table(metadata, artifact_dir / lance_writer.METADATA_FILE)

    uri = str(out_dir / "corpus.lance")
    lance_writer.build_dataset(artifact_dir, uri)
    return uri


def _median_time(fn, repeats: int = _REPEATS):
    """(median seconds, last result) over `repeats` calls."""
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2], result


def run(n: int, tau_percentile: float, seed: int = 0, source: str = "synthetic") -> dict:
    """Build, time all three paths, verify membership. Returns the report row."""
    tmp = Path(tempfile.mkdtemp(prefix="nls_bench_"))
    try:
        t0 = time.perf_counter()
        if source == "fixture":
            embeddings, metadata, pca = fixture_embeddings(n)
        else:
            if source == "synthetic":
                embeddings, metadata = synthetic_embeddings(n, seed)
            else:
                embeddings, metadata = real_embeddings(source, n)
            pca = pca_basis(embeddings)
        n = embeddings.shape[0]
        uri = build_corpus(embeddings, metadata, tmp, pca)
        build_s = time.perf_counter() - t0
        disk_mb = sum(f.stat().st_size for f in Path(uri).rglob("*") if f.is_file()) / 1e6

        # A query in the PCA row space, so projecting it back is exact and the
        # two paths score in the same space.
        rng = np.random.default_rng(seed + 1)
        latent = rng.standard_normal(D)
        latent /= np.linalg.norm(latent)
        query = (pca.astype(np.float64).T @ latent).astype("float32")
        query /= np.linalg.norm(query)

        tau = float(np.percentile(embeddings @ query, tau_percentile))

        mem_s, mem_scores = _median_time(lambda: embeddings @ query)

        ds = lance.dataset(uri)
        t0 = time.perf_counter()
        corpus = ts.ThresholdCorpus(ds)
        hydrate_s = time.perf_counter() - t0

        # Brute-force ground truth in the vector_fp (PCA-256) space: the exact
        # reference the Lance paths score against. `projected` is the
        # pre-quantization PCA projection the corpus was built from (same
        # content as `vector_fp`), and `query_pca` is the query projected into
        # that same space.
        projected = (embeddings @ pca.T).astype("float32")
        query_pca = pca.astype(np.float64) @ query
        bf_scores = projected.astype(np.float64) @ query_pca
        bf_set = set(np.nonzero(bf_scores >= tau)[0].tolist())

        # Map row_id (dataset scan position) -> generation index (brute-force
        # order) by joining on the segment_id VALUE, not by parsing it. The
        # `metadata` table in scope here is generation-ordered (row i =
        # generation index i), so its segment_id column gives the lookup map.
        gen_segment_ids = metadata.column("segment_id").to_pylist()
        seg_to_gen = {sid: i for i, sid in enumerate(gen_segment_ids)}
        seg_table = ds.to_table(columns=["segment_id"], scan_in_order=True)
        seg_by_row = seg_table.column("segment_id").to_pylist()
        row_to_gen = {row_id: seg_to_gen[seg_by_row[row_id]] for row_id in range(len(seg_by_row))}

        def _hit_gen_ids(hits):
            return {row_to_gen[h.row_id] for h in hits}

        def _check(path_hits):
            gen_ids = _hit_gen_ids(path_hits)
            missing = bf_set - gen_ids
            extra = gen_ids - bf_set
            return gen_ids, missing, extra

        # numba JIT warmup, excluded from timing.
        corpus.threshold_search(query, tau, fast_curation=False)

        exact_s, exact_hits = _median_time(
            lambda: corpus.threshold_search(query, tau, fast_curation=False))
        _exact_ids, exact_missing, exact_extra = _check(exact_hits)

        fast_s, fast_hits = _median_time(
            lambda: corpus.threshold_search(query, tau, fast_curation=True))
        _fast_ids, fast_missing, fast_extra = _check(fast_hits)

        # Hard-fail on any false dismissal or false positive (both paths).
        if fast_missing or fast_extra:
            raise SystemExit(
                f"FAIL: fast_curation missing={len(fast_missing)} extra={len(fast_extra)} vs ground truth"
            )
        if exact_missing or exact_extra:
            raise SystemExit(
                f"FAIL: exact path missing={len(exact_missing)} extra={len(exact_extra)} vs ground truth"
            )

        # Score deviation, split by score kind.
        bf_by_gen = {i: float(bf_scores[i]) for i in range(n)}
        fast_devs_exact, fast_devs_bounded, bound_violations = [], [], 0
        for h in fast_hits:
            true = bf_by_gen[row_to_gen[h.row_id]]
            dev = abs(h.score - true)
            if h.score_kind == "bounded_approx":
                fast_devs_bounded.append(dev)
                if dev > h.score_error_bound + 1e-6:
                    bound_violations += 1
            else:
                fast_devs_exact.append(dev)
        if bound_violations:
            raise SystemExit(
                f"FAIL: {bound_violations} fast_curation rows exceeded their eps bound"
            )
        exact_devs = [abs(h.score - bf_by_gen[row_to_gen[h.row_id]]) for h in exact_hits]

        fast_above = sum(1 for h in fast_hits if h.score_kind == "bounded_approx")
        fast_band = sum(1 for h in fast_hits if h.score_kind == "exact")

        return {
            "rows": n,
            "source": source,
            "build_s": build_s,
            "disk_mb": disk_mb,
            "tau": tau,
            "tau_percentile": tau_percentile,
            "matches": len(fast_hits),
            "mem_resident_mb": embeddings.nbytes / 1e6,
            "mem_ms": mem_s * 1e3,
            "lance_resident_mb": n * D / 1e6,
            "lance_hydrate_s": hydrate_s,
            "fast_ms": fast_s * 1e3,
            "exact_ms": exact_s * 1e3,
            "fast_above": fast_above,
            "fast_band": fast_band,
            "membership_missing_fast": len(fast_missing),
            "membership_extra_fast": len(fast_extra),
            "membership_missing_exact": len(exact_missing),
            "membership_extra_exact": len(exact_extra),
            "score_dev_fast_max": max(fast_devs_exact + fast_devs_bounded) if (fast_devs_exact or fast_devs_bounded) else 0.0,
            "score_dev_fast_mean": (sum(fast_devs_exact + fast_devs_bounded) / len(fast_devs_exact + fast_devs_bounded)) if (fast_devs_exact or fast_devs_bounded) else 0.0,
            "score_dev_exact_max": max(exact_devs) if exact_devs else 0.0,
            "bound_violations_fast": bound_violations,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rows", type=int, default=100_000, help="corpus rows (default 100k)")
    p.add_argument(
        "--tau-percentile",
        type=float,
        default=99.99,
        help="threshold as a percentile of the score distribution; high = tight "
        "match set, the curation regime (default 99.99)",
    )
    p.add_argument(
        "--source",
        default="synthetic",
        metavar="SOURCE",
        help="'synthetic' (default), 'fixture' for the committed real-corpus "
        "sample, or a path / s3:// URI of a real embeddings.npy",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    r = run(args.rows, args.tau_percentile, args.seed, args.source)
    print(
        f"\n=== {r['rows']:,} rows x {D}d from {r['source']} "
        f"(prepared in {r['build_s']:.1f}s, {r['disk_mb']:.0f} MB on disk) ==="
    )
    print(
        f"tau p{r['tau_percentile']} = {r['tau']:.6f} -> {r['matches']:,} matches "
        f"({r['matches'] / r['rows']:.4%} of corpus)\n"
    )
    print(f"{'path':<16}{'resident':>14}{'per query':>14}{'startup':>12}")
    print(f"{'in-memory':<16}{r['mem_resident_mb']:>11.1f} MB{r['mem_ms']:>11.1f} ms{'-':>12}")
    print(
        f"{'fast_curation':<16}{r['lance_resident_mb']:>11.1f} MB{r['fast_ms']:>11.1f} ms"
        f"{r['lance_hydrate_s']:>9.2f} s"
    )
    print(
        f"{'exact_scores':<16}{r['lance_resident_mb']:>11.1f} MB{r['exact_ms']:>11.1f} ms"
        f"{r['lance_hydrate_s']:>9.2f} s"
    )
    print(
        f"\nfast_curation breakdown: {r['fast_above']:,} ABOVE (bounded, no take()) + "
        f"{r['fast_band']:,} BAND (exact, take())"
    )
    print(
        f"membership        : PASS (fast missing={r['membership_missing_fast']} extra={r['membership_extra_fast']}; "
        f"exact missing={r['membership_missing_exact']} extra={r['membership_extra_exact']})"
    )
    print(
        f"score deviation   : fast max={r['score_dev_fast_max']:.2e} mean={r['score_dev_fast_mean']:.2e} "
        f"| exact max={r['score_dev_exact_max']:.2e}"
    )
    print(
        f"eps bound         : {'PASS' if r['bound_violations_fast'] == 0 else 'FAIL'} "
        f"({r['bound_violations_fast']} violations)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
