"""How much of the corpus carries the DX identifiers, and which one to filter on.

`segment_id` is the Data Explorer external id, and is what a segment-set filter
matches on today. `dx_internal_id` is DORA's global internal counter -- the space
its roaring bitmaps are built over -- and is what would make a segment-set filter
a single bitmap call instead of paginating external ids. Filtering on a sparsely
populated column silently drops nearly every row, so measure before relying on it.

Reads two columns' null counts by pushdown; it does not read the vectors.

    python scripts/dx_coverage.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Run directly (``python scripts/dx_coverage.py``) and Python puts scripts/ on
# the path, not the repo root -- so the app modules below would not resolve.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import lance  # noqa: E402

import full_corpus  # noqa: E402
import oci_s3  # noqa: E402


def main() -> None:
    so = oci_s3.lance_storage_options()
    if not so.get("aws_endpoint"):
        raise SystemExit(
            "no S3 endpoint configured -- set AWS_ENDPOINT_URL_S3 (and the OCI "
            "key pair) before running; see the app's secrets"
        )
    uri = full_corpus.DEFAULT_CORPUS_TABLE_URI
    t = time.perf_counter()
    ds = lance.dataset(uri, storage_options=so)
    total = ds.count_rows()
    print(f"{uri}\nversion {ds.version}, {total:,} rows ({time.perf_counter() - t:.1f}s)\n")
    for col in ("segment_id", "dx_internal_id", "vehicle", "chunk_end_unix"):
        if col not in ds.schema.names:
            print(f"  {col:16s} column absent")
            continue
        t = time.perf_counter()
        n = ds.count_rows(filter=f"{col} IS NOT NULL")
        pct = 100.0 * n / total if total else 0.0
        print(f"  {col:16s} {n:>13,} / {total:,}  {pct:6.2f}%   [{time.perf_counter() - t:.1f}s]")


if __name__ == "__main__":
    main()
