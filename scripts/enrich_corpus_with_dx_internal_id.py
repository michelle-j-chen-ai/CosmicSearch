"""Offline: add a ``dx_internal_id`` column to an NLS corpus for fast segment-set
downsampling.

The NLS app can filter the corpus to a Data Explorer segment set in a single
``DescribeDataSet(include_bitmap=true)`` call IF every corpus row carries DORA's
**global internal segment counter** (``dx_internal_id``) -- the integer space the
roaring bitmaps are built over (stable across all datasets). This script computes
that column once, offline, and writes an enriched copy of the corpus.

Pipeline
--------
1. Read the corpus lance and collect distinct non-null ``segment_id`` (the neuron
   external_id, e.g. ``rog144_20250915_221102-...-...``). ``segment_id`` itself is
   produced upstream by ``generate_mini_segment_chunks_workflow.py`` from
   ``CustomEntity.segment_info.segment_id`` -- this script does NOT recompute it.
2. Resolve ``external_id -> dx_internal_id``. DORA exposes the counter ONLY inside
   a dataset's roaring bitmap, so we need the segments in *some* decodable dataset:
     --source-dataset-uuid UUID  : decode an EXISTING dataset that already contains
                                    the corpus segments (read-only, preferred).
     (default)                   : register the corpus ids as a temporary dataset,
                                    decode it, then deprecate it (one prod write).
   The recipe: ``DescribeDataSet(EXTERNAL_IDS_ONLY)`` returns segments in
   internal-counter order, so ``zip(enumerated_external_ids, sorted(bitmap))``
   recovers each external_id's counter. (See dora_client.fetch_segment_bitmap for
   the runtime side that decodes a set's bitmap for the AND.)
3. Optionally also add ``dx_segment_id`` (the segment UUID) via BatchDescribeSegments.
4. Write the enriched lance (rows with no segment_id get null ids).

Usage
-----
    python enrich_corpus_with_dx_internal_id.py \
        --in  s3://.../chunks.lance  (or a local path) \
        --out ./chunks_with_dx.lance \
        [--source-dataset-uuid UUID] [--with-segment-uuid] [--drop-unmapped]

Auth: set DATA_EXPLORER_SDK_GRPC_HOSTNAME + machine creds (deployed) or
DATA_EXPLORER_SDK_GRPC_AUTH_TOKEN (local). Requires lance, pyarrow, pyroaring,
and dora_client on the path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import dora_client
import lance
import pyarrow as pa
from adp.public.proto import pagination_pb2
from adp.services.dora_event_management.proto import (
    dora_event_management_service_pb2 as pb,
)
from pyroaring import BitMap

LOGGER = logging.getLogger(__name__)
_TMP_SET_NAME = "nls_internal_id_enrich_tmp"


def _decode_dataset(stub, uuid: str) -> dict[str, int]:
    """external_id -> dx_internal_id for a dataset, via one enumeration + one bitmap.

    Enumeration order == internal-counter order, so zipping the ordered
    external_ids with the sorted bitmap recovers each id's global counter.
    """
    ext: list[str] = []
    cursor = ""
    while True:
        resp = stub.DescribeDataSet(
            pb.DescribeDataSetRequest(
                uuid=uuid,
                segments_page_request=pagination_pb2.PageRequest(
                    page_size=100000, after_cursor=cursor
                ),
                mode=pb.DESCRIBE_DATA_SET_MODE_EXTERNAL_IDS_ONLY,
            )
        )
        ext += [s.external_id for s in resp.dataset.segments]
        if not resp.segments_page_info or not resp.segments_page_info.end_cursor:
            break
        cursor = resp.segments_page_info.end_cursor
    rb = stub.DescribeDataSet(
        pb.DescribeDataSetRequest(
            uuid=uuid,
            segments_page_request=pagination_pb2.PageRequest(page_size=1),
            include_bitmap=True,
        )
    )
    bitmap = sorted(BitMap.deserialize(bytes(rb.dataset.roaring_bitmap)))
    if len(ext) != len(bitmap):
        raise ValueError(
            f"enumeration ({len(ext)}) != bitmap cardinality ({len(bitmap)}) for {uuid}"
        )
    return dict(zip(ext, bitmap))


def _register_and_decode(stub, external_ids: list[str]) -> dict[str, int]:
    """Register external_ids as a temporary dataset, decode it, then deprecate it."""
    cr = stub.CreateDataSet(pb.CreateDataSetRequest(name=_TMP_SET_NAME))
    uuid = cr.dataset_uuid
    LOGGER.info("created temp dataset %s for %d ids", uuid, len(external_ids))
    try:
        for i in range(0, len(external_ids), 1000):
            stub.AddSegmentsToDataSet(
                pb.AddSegmentsToDataSetRequest(
                    dataset_uuid=uuid, external_ids=external_ids[i : i + 1000]
                )
            )
        return _decode_dataset(stub, uuid)
    finally:
        stub.UpdateDataSet(
            pb.UpdateDataSetRequest(dataset_uuid=uuid, set_deprecated=True)
        )
        LOGGER.info("deprecated temp dataset %s", uuid)


def _enrich_table(stub, table, source_dataset_uuid, with_segment_uuid, drop_unmapped):
    """Append dx_internal_id (+ optional dx_segment_id) to an arrow table that has
    a ``segment_id`` column. Shared by the lance and npy code paths."""
    seg = table.column("segment_id").to_pylist()
    distinct = sorted({x for x in seg if x})
    LOGGER.info(
        "corpus rows=%d distinct non-null segment_ids=%d", len(seg), len(distinct)
    )

    if source_dataset_uuid:
        mapping = _decode_dataset(stub, source_dataset_uuid)
    else:
        mapping = _register_and_decode(stub, distinct)
    resolved = len(set(distinct) & mapping.keys())
    LOGGER.info(
        "resolved %d/%d (%.1f%%)",
        resolved,
        len(distinct),
        100 * resolved / max(1, len(distinct)),
    )

    table = table.append_column(
        "dx_internal_id", pa.array([mapping.get(x) for x in seg], type=pa.int64())
    )

    if with_segment_uuid:
        uuid_by_ext: dict[str, str] = {}
        for i in range(0, len(distinct), 1000):
            resp = stub.BatchDescribeSegments(
                pb.BatchDescribeSegmentsRequest(external_ids=distinct[i : i + 1000])
            )
            for s in resp.segments:
                if s.external_id and s.segment_uuid:
                    uuid_by_ext[s.external_id] = s.segment_uuid
        table = table.append_column(
            "dx_segment_id",
            pa.array([uuid_by_ext.get(x) for x in seg], type=pa.string()),
        )

    if drop_unmapped:
        import pyarrow.compute as pc

        table = table.filter(pc.is_valid(table.column("dx_internal_id")))
        LOGGER.info("kept %d rows after dropping unmapped", table.num_rows)

    return table


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="src",
        required=True,
        help="input corpus: a .lance dir/URI, OR an npy-format dir containing "
        "metadata.parquet (+ embeddings.npy). Both formats are local paths or "
        "(for lance) S3 URIs; download an S3 npy corpus locally first.",
    )
    ap.add_argument(
        "--out",
        dest="dst",
        required=True,
        help="output path (lance dir for a lance input; a dir to receive the "
        "enriched metadata.parquet + copied embeddings.npy for an npy input)",
    )
    ap.add_argument(
        "--source-dataset-uuid",
        default=None,
        help="decode this EXISTING DX dataset (read-only) instead of registering one",
    )
    ap.add_argument(
        "--with-segment-uuid",
        action="store_true",
        help="also add dx_segment_id (segment UUID) via BatchDescribeSegments",
    )
    ap.add_argument(
        "--drop-unmapped",
        action="store_true",
        help="keep only rows whose segment_id resolved (drops null/unmapped rows)",
    )
    args = ap.parse_args()

    stub = dora_client._get_stub()

    # npy format: a directory holding metadata.parquet (+ embeddings.npy). The
    # segment_id lives in metadata.parquet and the app's _load_corpus_npy reads
    # dx_internal_id straight from it, so we only enrich + rewrite that parquet
    # and copy the (unchanged) embeddings.npy alongside it.
    src = Path(args.src)
    if src.is_dir() and (src / "metadata.parquet").exists():
        import shutil

        import pyarrow.parquet as pq

        meta = pq.read_table(src / "metadata.parquet")
        table = _enrich_table(
            stub,
            meta,
            args.source_dataset_uuid,
            args.with_segment_uuid,
            args.drop_unmapped,
        )
        out = Path(args.dst)
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / "metadata.parquet")
        emb = src / "embeddings.npy"
        if emb.exists() and out.resolve() != src.resolve():
            shutil.copy2(emb, out / "embeddings.npy")
        LOGGER.info(
            "wrote npy corpus %s: %d rows, metadata columns=%s",
            out,
            table.num_rows,
            table.schema.names,
        )
        return

    ds = lance.dataset(args.src)
    table = _enrich_table(
        stub,
        ds.to_table(),
        args.source_dataset_uuid,
        args.with_segment_uuid,
        args.drop_unmapped,
    )
    lance.write_dataset(table, args.dst, mode="overwrite")
    LOGGER.info(
        "wrote %s: %d rows, columns=%s", args.dst, table.num_rows, table.schema.names
    )


if __name__ == "__main__":
    main()
