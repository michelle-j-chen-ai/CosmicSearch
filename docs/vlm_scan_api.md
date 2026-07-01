# VLM Auto-Labeling Scan API

How to drive the offline VLM interval-labeling workflow from a scenario ETL (e.g. a
`VLMLabeler` `IntervalExtractorGroup`). One HTTP call takes a **scenario Lance dataset**
(the downsample) plus **VLM tag specs** (tag name + similarity threshold), launches the
NLS segment-scan Lilypad workload, and writes a **Lance keyed by scenario/segment id with
one interval column per tag**. Identical parallel calls are **de-duplicated server-side**
to a single workload.

> TL;DR — `POST /api/launch_segment_scan` is the `michelles_api(...)` in the pseudocode.
> It returns a workload id + the output Lance URI; poll `GET /api/scan_status` until
> `SUCCEEDED`, then read the Lance.

---

## 1. Endpoints

| Fleet | Base URL |
|-------|----------|
| Cars   | `https://vlm-nls-search.experimental.apps.applied.dev` |
| Trucks | `https://vlm-nls-search-trucking.experimental.apps.applied.dev` |

The service is IAP-gated and uses ADP machine auth. Call it with the same machine
credentials the rest of the offboard tooling uses (the app injects its own OCI / Lilypad
creds — callers do not pass any).

| Method & path | Purpose |
|---|---|
| `POST /api/launch_segment_scan` | Launch (or dedup-join) a scan. Returns `{workload_id, lance_uri, deduplicated, ...}`. |
| `GET  /api/scan_status?execution=<workload_id>` | Poll a scan: `{done, phase, error}` (`phase` ∈ `RUNNING` / `SUCCEEDED` / `FAILED` …). |
| `GET  /api/scans` | List recent launches (id / status / when) for debugging. |

---

## 2. Mapping the pseudocode to the API

```text
michelles_api(
    input_dataset_uri = <Scenario Dataloader Lance dataset>,   ──▶  filter_lance_uri
    vlm_tag_specs = [{ tag_name, similarity_score_threshold }] ──▶  tags + thresholds
    output_lance_uri = <Output Lance dataset>,                 ──▶  returned as lance_uri
)
```

| Pseudocode field | Request field | Notes |
|---|---|---|
| `input_dataset_uri` | `filter_lance_uri` | **Downsample.** The scan output is restricted to the rows of this dataset. The app reads its id column (priority `segment_id` → `scenario_id` → `run_uuid`) and passes the id set to the worker inline. Cap: 50,000 ids. |
| `vlm_tag_specs[].tag_name` | `tags: [str]` | Each tag's 768-d search vector is reused from the DB (scoped to the active model) or encoded + persisted on first use. |
| `vlm_tag_specs[].similarity_score_threshold` | `thresholds: {tag: float}` (+ `default_threshold`) | Per-tag cosine cutoff. A tag absent from `thresholds` falls back to `default_threshold`. |
| `output_lance_uri` | **returned** `lance_uri` | Server-determined and **deterministic**: `s3://<scan-output>/<scan_id>/segments.lance`, where `scan_id` is derived from the dedup key. Read it from the response. |

---

## 3. The output Lance

One row per scenario/segment, one column per tag; each cell is a list of
`[start_ns, end_ns]` intervals (unix nanoseconds).

```
segment_id        | lane_change_left                      | lane_change_right
------------------+---------------------------------------+--------------------------
scenario_name_1   | []                                    | [[100, 200]]
scenario_name_2   | [[100, 200], [300, 400]]              | [[100, 200], [300, 400]]
...
```

- **Key column:** `segment_id` — this is the scenario id (the same id space as
  `input_dataset_uri`).
- **Tag columns:** `list<list<int64>>`, lists of `[start_ns, end_ns]`.
- **`merge_intervals`** (request flag, default `true`):
  - `true` → merged spans (contiguous hot clips fused into one interval).
  - `false` → one best (highest-scoring) clip per segment.
- A `manifest.json` is written next to `segments.lance` with provenance
  (`tags`, `thresholds`, `filter_lance_uri`, `num_downsample_ids`, `merge_intervals`, …).

---

## 4. Parallelization / dedup contract (the Spark restrictions)

This is what makes the call safe to invoke from a parallelized scenario ETL.

- **Blocking is fine.** The launch itself is a fast gRPC submit; you then block by polling
  `scan_status`. There is no requirement to launch asynchronously.
- **Duplicate concurrent requests are coalesced to ONE workload.** Two layers:
  1. *In-process single-flight* — the up-to-80 concurrent requests on one app instance
     collapse to one launch; the rest wait and share the result.
  2. *Postgres advisory lock* (`pg_advisory_xact_lock` on the dedup key) — the
     cross-instance authority. If a workload already exists for the key, its id is
     returned instead of launching again. The lock auto-releases on commit / crash, so
     there is no stale-lock cleanup.
- **"Block & return cached value."** A duplicate request gets
  `{"deduplicated": true, "workload_id": <existing>}` — the **same** id as the original.
  All callers then poll the same workload and read the same output Lance. Nothing is
  recomputed.
- **Dedup key.** Either:
  - a content hash of the output-determining fields (`tags`, `thresholds`,
    `default_threshold`, model, scan corpus, `filter_lance_uri`, dates, `vehicle`,
    `drive_id`, `merge_intervals`, `create_segment_set`), **or**
  - a client-supplied **`Idempotency-Key` header** — use this so a job's *retries across
    stages* also coalesce (not just simultaneous calls). Recommended: hash
    `input_dataset_uri` + the sorted tag specs.

---

## 5. Reference integration (blocking, dedup-safe)

```python
import hashlib, json, time, requests

_BASE = "https://vlm-nls-search.experimental.apps.applied.dev"


def michelles_api(
    *,
    input_dataset_uri: str,
    vlm_tag_specs: list[dict],          # [{"tag_name": ..., "similarity_score_threshold": ...}]
    merge_intervals: bool = True,
    session: requests.Session,          # carries ADP machine auth
    poll_s: float = 10.0,
    timeout_s: float = 3600.0,
) -> str:
    """Launch (or dedup-join) the VLM segment scan and BLOCK until it finishes.
    Returns the output Lance URI. Safe to call concurrently with identical args:
    duplicate calls coalesce to one workload and all return the same Lance."""
    tags = [s["tag_name"] for s in vlm_tag_specs]
    thresholds = {
        s["tag_name"]: float(s["similarity_score_threshold"])
        for s in vlm_tag_specs
        if s.get("similarity_score_threshold") is not None
    }
    # Stable idempotency key so retries across ETL stages also coalesce.
    idem = hashlib.sha256(
        json.dumps(
            {"in": input_dataset_uri, "tags": sorted(tags),
             "th": {k: thresholds[k] for k in sorted(thresholds)},
             "merge": merge_intervals},
            sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()

    r = session.post(
        f"{_BASE}/api/launch_segment_scan",
        headers={"Idempotency-Key": idem},
        json={
            "tags": tags,
            "thresholds": thresholds,
            "filter_lance_uri": input_dataset_uri,   # the downsample
            "merge_intervals": merge_intervals,
        },
        timeout=60,
    )
    r.raise_for_status()
    launched = r.json()
    workload_id = launched["workload_id"]
    lance_uri = launched["lance_uri"]                 # deterministic output path

    # Block until the (possibly shared) workload finishes.
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = session.get(
            f"{_BASE}/api/scan_status", params={"execution": workload_id}, timeout=30
        ).json()
        if st.get("done"):
            if st.get("phase") != "SUCCEEDED":
                raise RuntimeError(f"scan {workload_id} {st.get('phase')}: {st.get('error')}")
            return lance_uri
        time.sleep(poll_s)
    raise TimeoutError(f"scan {workload_id} did not finish within {timeout_s}s")
```

### In the `VLMLabeler`

```python
class VLMLabeler(IntervalExtractorGroup[Scores]):
    def __init__(self, *, input_dataset_uri, vlm_tag_specs, session) -> None:
        # OK to block here. Duplicate constructions (scenario-ETL parallelization)
        # dedup to one workload server-side; this returns the cached Lance URI.
        self._tags = [s["tag_name"] for s in vlm_tag_specs]
        self._lance_uri = michelles_api(
            input_dataset_uri=input_dataset_uri,
            vlm_tag_specs=vlm_tag_specs,
            session=session,
        )
        self._table = None  # lazy: load the Lance once, on first label()

    def _rows(self):
        if self._table is None:
            import lance
            self._table = lance.dataset(self._lance_uri).to_table().to_pydict()
            self._index = {sid: i for i, sid in enumerate(self._table["segment_id"])}
        return self._table, self._index

    def label(self, scenario, routes) -> dict[str, list[DataInterval[Scores]]]:
        table, index = self._rows()
        i = index.get(scenario.id)                # scenario id == segment_id key
        out: dict[str, list[DataInterval[Scores]]] = {}
        if i is None:
            return out                            # scenario not in the downsample → no labels
        for tag in self._tags:
            spans = table[tag][i] or []           # list[[start_ns, end_ns]]
            out[tag] = [
                DataInterval(start_ns=int(a), end_ns=int(b), value=Scores(...))
                for a, b in spans
            ]
        return out
```

---

## 6. Request reference

`POST /api/launch_segment_scan` — body (`SegmentScanRequest`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `tags` | `list[str]` | — (required) | VLM tags to label. |
| `thresholds` | `dict[str,float]` | `{}` | Per-tag cosine cutoff. |
| `default_threshold` | `float` | `0.3` | Fallback cutoff for tags missing from `thresholds`. |
| `filter_lance_uri` | `str` | `""` | **Downsample** dataset (the scenario Lance). |
| `merge_intervals` | `bool` | `true` | Merge spans vs. one best clip per segment. |
| `from_date` / `to_date` | `str` | server default / open | Optional drive-date window (`YYYY-MM-DD`). |
| `segment_set_uuid` | `str` | `""` | Optional DORA segment-set filter. |
| `vehicle` / `drive_id` | `str` | `""` | Optional vehicle / run_uuid filters. |
| `create_segment_set` | `bool` | `false` | Also register the output segments as a DORA set. |

Optional header: **`Idempotency-Key: <stable-hash>`** — coalesces this call with any other
carrying the same key.

### Response

```json
{
  "workload_id":   "nls-segment-scan-71a2c3f2-uh7inu",
  "execution_id":  "nls-segment-scan-71a2c3f2-uh7inu",
  "lance_uri":     "s3://.../nls_scans/<scan_id>/segments.lance",
  "deduplicated":  false,
  "tags":          ["lane_change_left"],
  "thresholds":    {"lane_change_left": 0.31},
  "encoded":       []
}
```

`deduplicated: true` means you joined an in-flight / completed workload — `workload_id`
and `lance_uri` point at the original.

---

## 7. Notes & gotchas

- **Output path is server-chosen** (`lance_uri` in the response), derived deterministically
  from the dedup key. Read it from the response rather than constructing it.
- **Coverage:** if a scenario id in `input_dataset_uri` was never embedded in the active
  corpus, it simply won't appear in the output Lance (no row) — `label()` returns no
  labels for it. There is no error for missing coverage.
- **Worker permissions:** the worker has no S3 LIST; the app resolves the downsample id
  set and passes it inline. This is why the 50,000-id cap exists — narrow the downsample
  if you exceed it.
- **Status polling** refreshes the stored job and (if `create_segment_set`) registers the
  DORA set once on success — so always poll to completion rather than assuming.
```
