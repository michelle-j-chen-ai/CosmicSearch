"""Standalone benchmark: resident fp32 brute force vs the Lance threshold scan.

Builds a corpus into an exact-threshold Lance dataset and runs the same query
both ways:

  in-memory   the whole fp32 model-space matrix resident, one numpy gemv per
              query -- what `search_engine.rank_top_k` does today
  lance       `ThresholdCorpus`: only the int8 PCA column resident, screened
              at `tau - eps`, boundary rows re-ranked from `vector_fp` via
              `take()` (see `threshold_search.py`)

Both must return the same number of rows; the run exits non-zero otherwise,
so the exactness guarantee is checked every time.

Reading the result: at any corpus that fits in RAM the brute force wins on
per-query latency, and that is expected -- the Lance path trades per-query
work for residency. The columns to watch are `resident` (what the process
must hold to answer a query at all) and how each scales with `--rows`. At
100M the fp32 matrix is ~307GB and the in-memory column has no answer at all.

`--tau-percentile` matters as much as `--rows`: every returned row costs a
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

D = 256
MODEL_DIM = 768
_REPEATS = 5
_GRAM_BLOCK_ROWS = 100_000
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


def pca_basis(embeddings: np.ndarray) -> np.ndarray:
    """Energy-ordered orthonormal (D, MODEL_DIM) basis: uncentered SVD via Gram.

    The 768x768 Gram matrix makes this exact and dependency-free -- its top-D
    eigenvectors are the right singular vectors a truncated SVD would return.
    Accumulated in row blocks so the float64 working copy stays bounded rather
    than scaling with the corpus (a whole-matrix upcast is ~8 bytes/value,
    which at tens of millions of rows exceeds RAM).
    """
    gram = np.zeros((embeddings.shape[1], embeddings.shape[1]), dtype=np.float64)
    for start in range(0, embeddings.shape[0], _GRAM_BLOCK_ROWS):
        block = embeddings[start : start + _GRAM_BLOCK_ROWS].astype(np.float64)
        gram += block.T @ block
    _eigenvalues, eigenvectors = np.linalg.eigh(gram)
    return np.ascontiguousarray(eigenvectors[:, ::-1][:, :D].T.astype("float32"))


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
    """Build, time both paths, verify they agree. Returns the report row."""
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
        mem_above = np.sort(np.asarray(mem_scores)[np.asarray(mem_scores) >= tau])[::-1]

        ds = lance.dataset(uri)
        t0 = time.perf_counter()
        corpus = ts.ThresholdCorpus(ds)
        hydrate_s = time.perf_counter() - t0
        corpus.threshold_search(query, tau, fast_curation=False)  # numba JIT warmup, excluded from timing
        lance_s, hits = _median_time(lambda: corpus.threshold_search(query, tau, fast_curation=False))

        # Row ids address the dataset's sorted order and the in-memory matrix
        # its generation order, so the two match sets cannot be compared by id.
        # Compare the sorted SCORES instead: same count and same values means
        # the same rows, up to the ~1e-4 gap between scoring in model space and
        # in the PCA-256 space (identical only because the corpus is rank-256).
        assert all(h.score >= tau for h in hits), "returned a row scoring below tau"
        lance_above = np.array([h.score for h in hits])
        if lance_above.size != mem_above.size:
            raise SystemExit(
                f"FAIL: brute force found {mem_above.size} rows >= tau, Lance path "
                f"returned {lance_above.size} -- the exact-threshold guarantee is broken"
            )
        if lance_above.size and not np.allclose(lance_above, mem_above, rtol=0, atol=1e-3):
            worst = float(np.abs(lance_above - mem_above).max())
            raise SystemExit(
                f"FAIL: the two paths returned the same number of rows but "
                f"different scores (max difference {worst:.2e}) -- they are not "
                f"returning the same rows"
            )

        return {
            "rows": n,
            "source": source,
            "build_s": build_s,
            "disk_mb": disk_mb,
            "tau": tau,
            "tau_percentile": tau_percentile,
            "matches": len(hits),
            "mem_resident_mb": embeddings.nbytes / 1e6,
            "mem_ms": mem_s * 1e3,
            "lance_resident_mb": n * D / 1e6,
            "lance_hydrate_s": hydrate_s,
            "lance_ms": lance_s * 1e3,
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
    print(f"{'path':<12}{'resident':>14}{'per query':>14}{'startup':>12}")
    print(f"{'in-memory':<12}{r['mem_resident_mb']:>11.1f} MB{r['mem_ms']:>11.1f} ms{'-':>12}")
    print(
        f"{'lance':<12}{r['lance_resident_mb']:>11.1f} MB{r['lance_ms']:>11.1f} ms"
        f"{r['lance_hydrate_s']:>9.2f} s"
    )
    print(
        f"\nresident ratio     : {r['mem_resident_mb'] / r['lance_resident_mb']:.1f}x less "
        f"held by the Lance path"
    )
    print(f"exactness          : PASS (both paths return the same {r['matches']:,} "
          f"rows, scores agreeing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
