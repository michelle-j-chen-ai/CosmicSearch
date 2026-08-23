# CosmicSearch API

Retrieve driving video clips by natural-language description, by example clip,
or by relevance feedback — then export the matches as CSV and parquet.

Everything is scored against one pinned corpus of **34,453,199** pre-embedded
8-second clips. A query is encoded into the same joint space as the video
embeddings, and the whole corpus is ranked in roughly 190 ms.

**Base URL**

```
https://vlm-nls-search.experimental.apps.applied.dev
```

---

## Getting started

### 1. Authenticate

The service sits behind Google IAP. Browser traffic authenticates through your
Applied SSO session; IAP injects the caller identity as
`x-goog-authenticated-user-email`, which the service uses to attribute saved
vectors and exports.

For programmatic access, create an API-gateway key:

```bash
apps-platform app apikey create --service vlm-nls-search
```

### 2. Check the corpus is loaded

The corpus is held in memory and takes ~2.5 minutes to load after a cold start.
Retrieval returns `503` until it is resident, so poll first:

```bash
curl https://vlm-nls-search.experimental.apps.applied.dev/api/full_corpus_status
```

```json
{
  "status": "ready",
  "num_rows": 34453199,
  "corpus_uri": "s3://…/vlm/corpus/video_embeddings.lance",
  "model": "black_dwarf",
  "elapsed_s": 0
}
```

`status` is one of `idle`, `loading`, `ready`, `error`. On `idle`, `POST
/api/full_corpus_load` starts the load and returns immediately.

### 3. Run your first search

```bash
curl -X POST https://vlm-nls-search.experimental.apps.applied.dev/api/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query": "ego makes an unprotected left turn", "k": 50}'
```

---

## Retrieve

`POST /api/retrieve` is the only retrieval endpoint. Search and export are the
same operation — rank the corpus, cut it, return the clips — and differ only in
how the result is serialized.

### Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | `""` | Natural-language description. Required unless `vector` or `refine_from_marks` is given. |
| `vector` | float[] | `null` | A pre-computed 768-d query vector, e.g. one returned by a previous export or `/api/search_by_upload`. |
| `k` | integer | `null` | Return the best `k` clips. **Exactly one of `k` or `threshold` is required.** |
| `threshold` | float | `null` | Return every clip scoring at or above this cutoff. Unbounded, so see `max_rows`. |
| `refine` | string | `"auto"` | `auto` resolves ambiguous clips exactly; `never` answers from the in-memory screen alone; `always` fails rather than returning approximate membership. |
| `exact` | boolean | `false` | Replace scores with true 768-d cosine. Costs one S3 read; refused if projected past 5 minutes. |
| `output` | string | `"hits"` | `hits` returns JSON. `csv` returns a file attachment and writes a parquet. |
| `page`, `limit` | integer | `0`, `50` | Paging for `output=hits`. Paging stops at 5,000 results. |
| `max_rows` | integer | `0` | Ceiling for `threshold` mode. `0` uses the service default (2,000,000). |
| `from_date`, `to_date` | string | `null` | ISO dates, e.g. `2026-06-01`. |
| `vehicle` | string | `null` | Comma- or space-separated vehicle ids. Multiple are honoured. |
| `drive_id` | string | `null` | Comma- or space-separated `run_uuid`s. |
| `filter_lance_uri` | string | `null` | A Lance or parquet dataset whose `segment_id` or `run_uuid` column restricts the search. |
| `marks` | Mark[] | `[]` | Relevance feedback. Downvoted clips are excluded from results. |
| `refine_from_marks` | boolean | `false` | Build the query direction from the marks (Rocchio) instead of the text. |
| `negative_weight` | float | `0.5` | How hard to push away from downvoted clips. |
| `text_weight` | float | `0.3` | How much of the original text query to blend back in. |
| `tag`, `interval`, `dedupe_segment` | — | — | `output=csv` only. See [Export](#export). |

A **Mark** is `{"chunk_id": str, "mark": "up" | "down", "row": int}`. The `row`
is the `row` value from a previous response — it addresses the clip in the
corpus, and a mark without one is ignored.

### Response — `output: "hits"`

```json
{
  "hits": [
    {
      "rank": 1,
      "score": 0.5265,
      "score_error_bound": 0.0184,
      "chunk_id": "019b46b3-…#t1766334630",
      "run_uuid": "019b46b3-faf3-72a9-b904-2bbb0825e74b",
      "segment_id": "…",
      "start_timestamp_ns": 1766334630000000000,
      "end_timestamp_ns": 1766334638000000000,
      "start_utc": "2026-08-21 18:30:30 UTC",
      "end_utc": "2026-08-21 18:30:38 UTC",
      "source_media_uri": "s3://…/dt=2026-08-21/019b46b3…_t1766334630.mp4",
      "vehicle": "mce113",
      "row": 18220417
    }
  ],
  "total": 50,
  "candidates": 34453199,
  "num_rows_searched": 34453199,
  "score_kind": "bounded_approx",
  "score_error_bound": 0.0184,
  "band_rows": 13,
  "refined": true,
  "elapsed_ms": 192.8,
  "corpus_uri": "s3://…/video_embeddings.lance",
  "corpus_version": 41,
  "corpus_loaded_utc": "2026-08-22 01:25:32 UTC",
  "filters_applied": { "vehicle": ["mce113"], "from_date": null, "to_date": null }
}
```

Fields worth understanding:

- **`row`** addresses the clip in the corpus. Pass it back in `marks`, or to
  `/api/full_rescore`. It is only valid against the `corpus_version` that
  returned it.
- **`score_kind`** is `bounded_approx` (score is within `±score_error_bound` of
  the true value), `pca_exact` (membership resolved exactly), or `exact` (true
  768-d cosine). **A threshold calibrated on one kind does not transfer to
  another.**
- **`candidates`** is how many clips passed your filters, not how many were
  returned. `total` is what paging can reach.
- **`band_rows`** is how many clips were close enough to the cutoff that the
  service had to read them from storage to decide. Low is good.

### Response — `output: "csv"`

The body is a CSV attachment. The facts about the export are in headers:

| Header | Meaning |
|---|---|
| `X-NLS-Rows` | rows written |
| `X-NLS-Candidates` | rows that matched before any cap |
| `X-NLS-Truncated` | `1` if capped — the export is a **partial** answer |
| `X-NLS-Cutoff` | the score cutoff used |
| `X-NLS-Score-Kind` | `bounded_approx` \| `pca_exact` \| `exact` |
| `X-NLS-Parquet` | `s3://…` of the parquet copy, or empty if the write failed |
| `X-NLS-Export-Name` | the shared stem of the CSV and parquet |
| `X-NLS-Elapsed-Ms` | wall time |

Always check `X-NLS-Truncated`. A `1` means clips matched that are not in your file.

---

## Export

Set `output: "csv"`. Additional parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `tag` | string | `""` | Names the CSV, the parquet, and the history entry. |
| `interval` | boolean | `false` | Merge adjacent matching clips into variable-length spans per drive, and export those instead of clips. |
| `dedupe_segment` | boolean | `false` | Keep only the best clip per `segment_id`. |

```bash
curl -X POST …/api/retrieve -H 'Content-Type: application/json' -o left_turns.csv -D headers.txt \
  -d '{"query":"unprotected left turn","threshold":0.42,"output":"csv",
       "tag":"left_turn","exact":true,"max_rows":200000}'
```

Exports are serialized: a second concurrent export returns `429`. Interval
exports have **no maximum span length** — a long run of matching clips becomes
one long interval, so a multi-minute result usually means the cutoff is below
the ambient similarity for that drive.

---

## Sharpen and calibrate

### `POST /api/full_rescore`

True 768-d cosine for clips you already have. Up to 1,000 rows per call.

```json
{ "query": "unprotected left turn", "rows": [18220417, 902355] }
```

```json
{ "took_ms": 657.8, "score_kind": "exact",
  "scores": [ {"row": 18220417, "score": 0.4913} ] }
```

Returns `409` if the corpus has been reloaded since those rows were issued — the
positions would point at different clips.

### `POST /api/full_threshold`

Fit a cutoff from labelled examples, and get the next clips worth labelling.

```json
{ "query": "unprotected left turn",
  "marks": [ {"chunk_id":"…","mark":"up","row":18220417} ],
  "objective": "f1" }
```

The response carries `threshold`, `suggested_threshold` (label-free), a
precision/recall `fit`, a score `histogram`, a boundary-biased `sample` to label
next, and **`selected_at_threshold`** — how many clips that cutoff would export.
Check that before exporting; a plausible-looking cutoff can select a tenth of
the corpus.

### `POST /api/score_distribution`

Score histogram for a query across the whole corpus, with the cutoff a top-k
export would use. Useful for judging whether a query separates anything at all.

---

## Search by example

### `POST /api/search_by_upload`

Encode an image or short video into a query vector.

```json
{ "image_b64": "…", "filename": "example.jpg" }
```

Returns `{vector, dim, label, n_frames}`. Pass `vector` to `/api/retrieve`.

### `POST /api/search_by_window`

Query by a clip already in the corpus: name a drive or segment and an optional
time window, and its embeddings are pooled into the query.

```json
{ "run_uuid": "019b46b3-faf3-72a9-b904-2bbb0825e74b",
  "start_ns": 1766334630000000000, "end_ns": 1766334750000000000 }
```

Same response as `/api/retrieve`, plus `query_clips` (a filmstrip of what was
pooled) and `query_clip_count`.

---

## Playback and artifacts

| Endpoint | Purpose |
|---|---|
| `GET /api/video?uri=…` | 307 redirect to a presigned MP4 URL. Pass `source_media_uri` from a hit. |
| `GET /api/export_file?uri=…` | 307 redirect to a presigned parquet. Restricted to the export prefix. |
| `GET /api/scans?limit=50` | Past exports and scans, with their artifact URIs. |
| `POST /api/save_vector` | Persist a query vector under a tag, so it can be resumed later. |
| `GET /api/search_session/{session_id}` | The stored query, vector and filters for a past export. |

---

## Batch and curation

These predate `/api/retrieve` and keep their own artifact writers. They work, but
`/api/retrieve` is the supported path for new integrations.

| Endpoint | Purpose |
|---|---|
| `POST /api/export_config` | Run several queries at once, each with its own `k` or `threshold`, and concatenate into one CSV + parquet. |
| `POST /api/curate_preview` | The search half of the above, as JSON, for review before committing. |
| `POST /api/curate_export` | Export an explicit, hand-picked set of rows. |

## Service metadata

| Endpoint | Purpose |
|---|---|
| `GET /api/platform` | Which deployment this is; both variants run the same image. |
| `GET /api/tags_catalog` | Saved searches as JSON, keyed on tag. |
| `GET /api/search_history` | The same as an HTML fragment. |
| `POST /api/refit_policy` | Re-fit the learned threshold policy from accumulated labelling episodes. |

---

## Corpus and model

| Endpoint | Purpose |
|---|---|
| `GET /api/full_corpus_status` | `status`, `num_rows`, `corpus_uri`, `elapsed_s`. |
| `POST /api/full_corpus_load` | Start the load; returns immediately. |
| `POST /api/full_corpus_refresh` | Drop and reload, picking up clips added since. Retrieval is unavailable during the reload. |
| `GET /api/corpus` | Row count, dimensions, date span, active encoder. |
| `GET /api/model` | The active encoder. |
| `GET /healthz` | Liveness. `ready` flips when the model is up. |

The corpus URI is fixed. `GET /api/corpus?uri=…` with anything else returns
`400` rather than silently ranking against a table the queries were not encoded
for.

---

## Errors

| Status | Meaning |
|---|---|
| `400` | Invalid parameters — no cutoff, both cutoffs, an unusable `filter_lance_uri`, a vector of the wrong dimensionality. |
| `404` | Nothing matched the query and filters. |
| `409` | Row positions no longer resolve to the clips they were issued for; re-run the search. |
| `422` | Malformed body. |
| `429` | Another export is running. |
| `503` | Corpus or model still loading, or exact scoring exceeded its time budget mid-fetch. Retry. |

Errors return `{"detail": "…"}` and the message says what to change.

---

## Notes on results

**Scores are not probabilities.** They are cosine similarities in a learned
joint space. What matters is the *separation* between the top matches and the
bulk, not the absolute value. If the top 200 scores span less than
`score_error_bound`, the model considers those clips interchangeable and their
order is close to arbitrary.

**Clips are 8 seconds at 448p, pooled into one vector.** Fine, high-frequency
detail — raindrops on a windshield, small distant text — is largely averaged
away. Scene-level descriptions retrieve far better than close-up ones.

**Row positions are snapshot-scoped.** The corpus grows daily. A `row` from
before a refresh may address a different clip afterwards; `corpus_version` on
every response tells you which snapshot you are holding.
