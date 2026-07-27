"""Shared fixtures + synthetic-corpus builders for tests/*.

Run the whole suite from the repo root:
    python -m pytest tests/

The tests import repo-root modules directly (e.g. `import lance_writer`,
`import search_engine`) rather than as an installed package, so the repo
root must be on `sys.path`. `pythonpath = .` in `pytest.ini` (repo root)
already does this for normal `python -m pytest` invocations; the explicit
insert below is a fallback for runners that don't honor that ini option.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import lance  # noqa: E402
import lance_writer  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import threshold_search as ts  # noqa: E402

D = 256
MODEL_DIM = 768


def write_artifact(
    artifact_dir: Path,
    n: int,
    *,
    seed: int = 0,
    orthonormal_pca: bool = False,
    pre_quant_fp32: bool = False,
) -> Path:
    """Write a gpu_corpus-style int8 PCA artifact -- `build_dataset`'s input.

    The int8 corpus is quantized from a generated fp32 array using the
    production convention (`dequant = int8 * scale / 127`). With
    `pre_quant_fp32`, that pre-quantization array is also saved, so the built
    dataset gets a GENUINE fp32 `vector_fp` instead of the dequant(int8)
    fallback -- the only configuration in which an oracle is independent of
    the int8 screen (see `lance_writer.py`'s module docstring).

    `orthonormal_pca` makes the basis rows orthonormal, so a unit-norm query
    in its row space stays unit-norm after projection -- the precondition
    `eps_bound.eps_cauchy_schwarz` assumes, and what a real SVD basis gives.
    """
    rng = np.random.default_rng(seed)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if orthonormal_pca:
        basis, _ = np.linalg.qr(rng.standard_normal((MODEL_DIM, D)))
        pca = np.ascontiguousarray(basis.T.astype("float32"))
    else:
        pca = rng.standard_normal((D, MODEL_DIM)).astype("float32")

    true_fp32 = (
        rng.standard_normal((n, D)) * rng.uniform(0.5, 3.0, size=D)
    ).astype("float32")
    scale = np.abs(true_fp32).max(axis=0).astype("float32")
    corpus_i8 = np.clip(np.round(true_fp32 * 127.0 / scale), -127, 127).astype(np.int8)

    np.save(artifact_dir / lance_writer.PCA_FILE, pca)
    np.save(artifact_dir / lance_writer.SCALE_FILE, scale)
    np.save(artifact_dir / lance_writer.CORPUS_INT8_FILE, corpus_i8)
    if pre_quant_fp32:
        np.save(artifact_dir / lance_writer.PRE_QUANT_FP32_FILE, true_fp32)

    # Deliberately unsorted so the writer's (chunk_start_unix, vehicle) sort
    # is exercised rather than accidentally satisfied by the input.
    chunk_start = rng.integers(1_700_000_000, 1_700_100_000, size=n).astype("int64")
    pq.write_table(
        pa.table(
            {
                "run_uuid": [f"run-{i % 5}" for i in range(n)],
                "chunk_start_unix": chunk_start,
                "chunk_end_unix": chunk_start + 5,
                "segment_id": [f"seg-{i}" for i in range(n)],
                "vehicle": rng.choice(["veh_a", "veh_b", "veh_c"], size=n),
            }
        ),
        artifact_dir / lance_writer.METADATA_FILE,
    )
    return artifact_dir


def build_corpus(tmp_dir: Path, n: int, **kwargs) -> lance.LanceDataset:
    """`write_artifact` + `build_dataset` under `tmp_dir`."""
    artifact_dir = write_artifact(tmp_dir / "artifact", n, **kwargs)
    return lance_writer.build_dataset(artifact_dir, str(tmp_dir / "out.lance"))


def unit_query(seed: int = 1, pca: np.ndarray | None = None) -> np.ndarray:
    """A unit-norm model-space query; in `pca`'s row space when given.

    Passing an orthonormal `pca` yields `q = P.T @ q_256`, which projects back
    to exactly `q_256` -- the score-lossless-PCA precondition.
    """
    rng = np.random.default_rng(seed)
    if pca is None:
        q = rng.standard_normal(MODEL_DIM)
        return (q / np.linalg.norm(q)).astype("float32")
    latent = rng.standard_normal(pca.shape[0])
    latent /= np.linalg.norm(latent)
    return (pca.astype(np.float64).T @ latent).astype("float32")


def exact_scores(ds: lance.LanceDataset, query: np.ndarray) -> np.ndarray:
    """Every row's exact score, read from `vector_fp` in canonical row order.

    The naive full-column read `threshold_search` exists to avoid -- used as
    the independent reference implementation.
    """
    pca, _scale = lance_writer.read_pca_metadata(ds)
    query_pca = ts._project_query(query, pca).astype(np.float64)
    table = ds.to_table(columns=[lance_writer.VECTOR_FP_COLUMN], scan_in_order=True)
    fp = ts._fixed_size_list_matrix(table, lance_writer.VECTOR_FP_COLUMN, np.float64)
    return fp @ query_pca
