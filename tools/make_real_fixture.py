"""Regenerate the committed real-corpus test fixture from the production corpus.

Run only when the corpus behind `--source` changes; the output is committed,
so day-to-day test and benchmark runs never touch object storage.

The fixture stores a seeded random sample as its PCA-256 projection plus the
basis, not as raw 768-d vectors. The corpus is rank-256 (its top 256 of 768
components hold 1.00000 of the energy), so `latent @ pca` reconstructs the
rows to ~2e-4 in score -- far below the eps bound -- at a third of the bytes.
The basis is derived from EVERY row, which a 10k sample could not reproduce
on its own. No identifiers are stored: scalar metadata takes no part in
scoring and is synthesized at load time.

    aws --profile oci.phx s3 cp \\
        s3://neuron-prod-data-intelligence-exploratory/sibogeng/nls_search/\\
embeddings/v3_lr_5e5-ckpt-6549_npy/embeddings.npy /tmp/embeddings.npy
    python tools/make_real_fixture.py --source /tmp/embeddings.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

D = 256
_GRAM_BLOCK_ROWS = 100_000


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="local path to the corpus embeddings.npy")
    p.add_argument("--rows", type=int, default=10_000, help="sampled rows (default 10k)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "fixtures" / "real_corpus_sample.npz",
    )
    args = p.parse_args()

    full = np.load(args.source, mmap_mode="r")
    n_total, model_dim = full.shape
    print(f"source: {full.shape} {full.dtype}")

    # Uncentered SVD basis over the whole corpus, accumulated blockwise so the
    # multi-GB array is never fully resident.
    gram = np.zeros((model_dim, model_dim), dtype=np.float64)
    for start in range(0, n_total, _GRAM_BLOCK_ROWS):
        block = np.asarray(full[start : start + _GRAM_BLOCK_ROWS], dtype=np.float64)
        gram += block.T @ block
    _eigenvalues, eigenvectors = np.linalg.eigh(gram)
    pca = np.ascontiguousarray(eigenvectors[:, ::-1][:, :D].T.astype("float32"))

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_total, size=args.rows, replace=False))
    sample = np.asarray(full[idx], dtype=np.float32)
    latent = (sample.astype(np.float64) @ pca.T.astype(np.float64)).astype("float32")

    recon = latent.astype(np.float64) @ pca.astype(np.float64)
    query = rng.standard_normal(model_dim)
    query /= np.linalg.norm(query)
    print(
        f"reconstruction: max score err "
        f"{np.abs(sample @ query - recon @ query).max():.3e}, row norm err "
        f"{np.abs(np.linalg.norm(recon, axis=1) - 1).max():.3e}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, latent=latent, pca=pca)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
