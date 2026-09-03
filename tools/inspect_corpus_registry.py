"""Read each registered project's corpus metadata straight off its Lance table.

`deployment.py` is the only place a corpus URI comes from, and the basis that
projects those vectors travels in the table's own FIELD metadata. So everything
needed to decide whether a project can be served is readable from the registry
plus the table -- no side table, no separate manifest.

Run before enabling a project, or after the embedding pipeline rewrites a table:

    python tools/inspect_corpus_registry.py               # every known project
    python tools/inspect_corpus_registry.py frontier      # just one

Exits non-zero if any inspected project would fail to load.
"""

from __future__ import annotations

import pathlib
import sys

# Run from anywhere: the app modules live one level up, not on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import deployment
import full_corpus
import oci_s3

REQUIRED_COLUMNS = full_corpus.REQUIRED_COLUMNS


def _fmt(n: int) -> str:
    return f"{n:,}"


def _null_rate(ds, column: str, limit: int = 200_000) -> "str":
    """Null fraction of `column` over the first `limit` rows, or why not."""
    if column not in ds.schema.names:
        return "column absent"
    try:
        tbl = ds.head(limit) if hasattr(ds, "head") else ds.to_table(columns=[column])
        col = tbl.column(column)
        return f"{col.null_count / max(len(col), 1):.1%} null (first {_fmt(len(col))} rows)"
    except Exception as exc:  # noqa: BLE001 -- diagnostic tool
        return f"unreadable: {type(exc).__name__}"


def inspect(name: str) -> bool:
    import lance

    spec = deployment.get(name)
    print(f"\n=== {name}  ({spec.label})")
    print(f"  table      {spec.corpus_table_uri}")
    print(f"  clips      {spec.mp4_prefix}")
    print(f"  dora       {spec.dora_hostname or '(unset)'}")

    try:
        ds = lance.dataset(spec.corpus_table_uri,
                           storage_options=oci_s3.lance_storage_options())
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL       cannot open: {type(exc).__name__}: {str(exc)[:200]}")
        return False

    rows = ds.count_rows()
    print(f"  version    {getattr(ds, 'version', '?')}   storage {getattr(ds, 'data_storage_version', '?')}")
    print(f"  rows       {_fmt(rows)}")

    ok = True

    # Exactly the check `load()` runs, so this cannot pass a table the app rejects.
    try:
        model_id = full_corpus.validate(ds, full_corpus.CORPUS_MODEL)
        pca, scales, _ = full_corpus._read_field_pca(ds, full_corpus.CORPUS_MODEL)
        print(f"  contract   OK  model_id {model_id}, pca {pca.shape} {pca.dtype}, "
              f"scales {scales.shape}")
        # Resident int8 footprint is what decides whether the service fits.
        gib = rows * pca.shape[0] / (1024 ** 3)
        print(f"  resident   {gib:.2f} GiB int8 screen ({pca.shape[0]}-d) + ~5 GiB encoder")
        if gib > 20:
            print("  WARN       screen + encoder leaves little headroom under a 32Gi ceiling")
    except full_corpus.CorpusContractError as exc:
        print(f"  FAIL       contract: {exc}")
        ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL       basis unreadable: {type(exc).__name__}: {str(exc)[:200]}")
        ok = False

    # Reported for comparison only: the app reads the FIELD metadata above.
    schema_meta = sorted((ds.schema.metadata or {}).keys())
    print(f"  schema meta {[k.decode() for k in schema_meta] or '(none)'}")

    for col in ("segment_id", "dx_internal_id", "created_at_unix_s"):
        print(f"  {col:<18} {_null_rate(ds, col)}")

    # One clip path, resolved the way the app resolves it.
    try:
        head = ds.head(1) if hasattr(ds, "head") else ds.to_table(columns=list(REQUIRED_COLUMNS))
        d = head.to_pylist()[0]
        import datetime as dt
        day = dt.datetime.fromtimestamp(d["chunk_start_unix"], tz=dt.timezone.utc).strftime("%Y-%m-%d")
        uri = full_corpus.media_uri(spec.mp4_prefix, day, d["run_uuid"], d["chunk_start_unix"])
        bucket, key = oci_s3.parse_s3_uri(uri)
        client = oci_s3.s3_client(fast_fail=True)
        try:
            obj = client.head_object(Bucket=bucket, Key=key)
            print(f"  clip       OK {_fmt(obj['ContentLength'])} bytes  {uri}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL       clip not readable: {type(exc).__name__}  {uri}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN       could not build a clip path: {type(exc).__name__}")

    print(f"  {'PASS' if ok else 'FAIL'}       {name} would {'' if ok else 'NOT '}load")
    return ok


def main(argv: list[str]) -> int:
    names = argv[1:] or deployment.names()
    bad = [n for n in names if not deployment.is_known(n)]
    if bad:
        print(f"unknown project(s) {bad}; known: {deployment.names()}")
        return 2
    results = {n: inspect(n) for n in names}
    print("\n" + "  ".join(f"{n}={'PASS' if v else 'FAIL'}" for n, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
