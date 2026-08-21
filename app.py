"""Streamlit app: natural-language search over VLM video embeddings.

The user supplies a Lance embeddings URI and a prompt. We download the Lance
corpus to a disk cache (once per URI, shared across users and instances),
encode the prompt with Cosmos-Embed on CPU (~33ms), rank the resident matrix by
cosine similarity (~58ms at 1M), and render the top matches as inline videos
streamed straight from OCI via presigned URLs.

When the text-to-video alignment is loose -- a query surfaces some true matches
but misses others that look identical -- the user marks results 👍/👎 and
"Refine": the app builds a new search direction toward the 👍 and away from the
👎 (a prototype / Rocchio direction, deliberately not a max-margin classifier --
see search_engine.refine_query) and re-ranks by it. The original text query can
optionally be blended back in to anchor iterative refinement.

Run locally:
    NLS_MODEL_ARTIFACT_URI=s3://.../models/<session>/ python -m streamlit run app.py
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import io
import logging
import time
import uuid

import analytics
import botocore.exceptions
import dora_client
import local_cache
import oci_s3
import search_engine
import streamlit as st

from config import AppConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

st.set_page_config(page_title="VLM Video Search", layout="wide")

_CORPUS_ERRORS = (
    ValueError,
    FileNotFoundError,
    OSError,
    botocore.exceptions.BotoCoreError,
    botocore.exceptions.ClientError,
)


@st.cache_resource(show_spinner="Loading text encoder (one-time)...")
def get_model(model_artifact_uri: str, device: str) -> tuple[object, object]:
    """Load + cache the encoder once per process, keyed by model URI."""
    return search_engine.load_model(model_artifact_uri, device)


@st.cache_resource(show_spinner=False)
def get_corpus(embeddings_uri: str, matrix_dtype: str) -> search_engine.Corpus:
    """Download (cached) + load a corpus once per process, keyed by its URI.

    Across user sessions in one instance this is an in-memory hit; the first
    load also populates the on-disk download cache shared across instances.
    """
    return search_engine.load_corpus(embeddings_uri, matrix_dtype)


@st.cache_data(ttl=300, show_spinner="Listing Data Explorer segment sets...")
def get_segment_sets(name_filter: str) -> list[dora_client.SegmentSet]:
    """List DORA segment sets (cached briefly so reruns don't re-hit gRPC)."""
    return dora_client.list_segment_sets(name_filter)


@st.cache_data(show_spinner="Fetching segment ids...")
def get_segment_ids(dataset_uuid: str) -> frozenset[str]:
    """Fetch a dataset's segment external_ids (cached per dataset UUID)."""
    return dora_client.fetch_segment_ids(dataset_uuid)


def _fmt_start(chunk_start_unix: int) -> str:
    return dt.datetime.fromtimestamp(chunk_start_unix, tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


# Relevance marks accumulate across refine rounds in this session_state dict:
# chunk_id -> {"index": int, "mark": _MARK_POS | _MARK_NEG}. It is a plain (non-
# widget) key, so Streamlit never garbage-collects it when a marked result
# scrolls out of the current top-k -- labels persist and accumulate until the
# user runs a new text query, switches corpus, or clears them. Each per-result
# radio (keyed _MARK_PREFIX+chunk_id) syncs its value into this dict on render.
_MARKS_KEY = "nls_marks"
_MARK_PREFIX = "mark::"
_MARK_SKIP = "skip"
_MARK_POS = "relevant"
_MARK_NEG = "not_relevant"
_MARK_OPTIONS = [_MARK_SKIP, _MARK_POS, _MARK_NEG]
_MARK_LABELS = {_MARK_SKIP: "–", _MARK_POS: "👍", _MARK_NEG: "👎"}


def _marks() -> dict[str, dict]:
    return st.session_state.setdefault(_MARKS_KEY, {})


def _clear_marks() -> None:
    st.session_state[_MARKS_KEY] = {}
    for key in [k for k in st.session_state if k.startswith(_MARK_PREFIX)]:
        del st.session_state[key]


def _sync_mark(chunk_id: str, index: int, mark: str) -> None:
    """Fold one result's current radio value into the accumulator."""
    marks = _marks()
    if mark == _MARK_SKIP:
        marks.pop(chunk_id, None)
    else:
        marks[chunk_id] = {"index": index, "mark": mark}


def _marked_indices() -> tuple[list[int], list[int]]:
    """(positive, negative) corpus row indices accumulated across refine rounds."""
    positives = [m["index"] for m in _marks().values() if m["mark"] == _MARK_POS]
    negatives = [m["index"] for m in _marks().values() if m["mark"] == _MARK_NEG]
    return positives, negatives


# Result paging. The page can start at an explicit rank or at a similarity-score
# threshold; whichever input the user last touched wins (tracked in _BROWSE_MODE).
# Both inputs persist by widget key so paging survives reruns.
_BROWSE_MODE_KEY = "nls_browse_mode"
_START_RANK_KEY = "nls_start_rank"
_START_SCORE_KEY = "nls_start_score"
_EXPORT_TAG_KEY = "nls_export_tag"
_EXPORT_READY_KEY = "nls_export_ready"


def _set_browse_mode(mode: str):
    """Callback factory: mark which start-input the user just edited."""

    def _cb() -> None:
        st.session_state[_BROWSE_MODE_KEY] = mode

    return _cb


def _reset_browse() -> None:
    """Send paging back to the top (rank 1). Called on a new query / refine."""
    st.session_state[_BROWSE_MODE_KEY] = "rank"
    st.session_state[_START_RANK_KEY] = 1
    st.session_state.pop(_START_SCORE_KEY, None)


def _day_range_to_unix(from_date: dt.date, to_date: dt.date) -> tuple[int, int]:
    """[start, end) unix seconds: from_date 00:00 UTC through end of to_date."""
    start = dt.datetime.combine(from_date, dt.time.min, tzinfo=dt.timezone.utc)
    end = dt.datetime.combine(
        to_date, dt.time.min, tzinfo=dt.timezone.utc
    ) + dt.timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def _results_csv(hits: list[search_engine.RankedHit], tag: str) -> str:
    """Serialize ranked hits to CSV: rank, score, segment_id, and per-clip ids.

    `segment_id` is the original 30s source segment (empty if the corpus
    metadata lacks it); `chunk_id` (= "<run_uuid>#t<chunk_start_unix>") is the
    8s clip within it. `tag` is a user-supplied label written into a rightmost
    `tag` column, identical for every row, so a search can be exported as a
    labeled dataset.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "rank",
            "score",
            "segment_id",
            "chunk_id",
            "run_uuid",
            "chunk_start_unix",
            "source_media_uri",
            "tag",
        ]
    )
    for h in hits:
        writer.writerow(
            [
                h.rank,
                f"{h.score:.6f}",
                h.segment_id,
                h.chunk_id,
                h.run_uuid,
                h.chunk_start_unix,
                h.source_media_uri,
                tag,
            ]
        )
    return buf.getvalue()


def _current_user() -> str:
    """Authenticated user email from the IAP-injected request header.

    Behind Apps Platform's IAP, requests carry
    ``X-Goog-Authenticated-User-Email: accounts.google.com:<email>``. Falls back
    to the user-id header, then "anonymous" (e.g. running locally without IAP).
    """
    try:
        headers = st.context.headers or {}
    except Exception:  # noqa: BLE001 -- st.context absent on very old Streamlit
        return "anonymous"
    for key in ("X-Goog-Authenticated-User-Email", "X-Goog-Authenticated-User-Id"):
        raw = headers.get(key) or headers.get(key.lower())
        if raw:
            return raw.split(":", 1)[-1]  # strip the "accounts.google.com:" prefix
    return "anonymous"


def _record_visit_once() -> None:
    """Record one visit per Streamlit session (first script run of the session)."""
    if st.session_state.get("nls_visit_recorded"):
        return
    session_id = st.session_state.get("nls_session_id") or uuid.uuid4().hex
    st.session_state["nls_session_id"] = session_id
    analytics.record_visit(_current_user(), session_id, time.time())
    st.session_state["nls_visit_recorded"] = True


@st.cache_data(ttl=30, show_spinner=False)
def _visit_summary() -> dict:
    """Aggregate persisted visits (cached briefly so reruns don't re-glob)."""
    visits = analytics.load_visits()
    by_user = collections.Counter(v.get("user", "?") for v in visits)
    return {"total": len(visits), "users": by_user.most_common(), "recent": visits[:20]}


def _render_analytics_sidebar(
    current_user: str, maintainer_emails: frozenset[str]
) -> None:
    # Visits are recorded for everyone, but only configured maintainers may view
    # them. Fail closed: an empty maintainer set shows the view to no one.
    if current_user.strip().lower() not in maintainer_emails:
        return
    with st.sidebar:
        with st.expander("Usage analytics", expanded=False):
            s = _visit_summary()
            c1, c2 = st.columns(2)
            c1.metric("Total visits", s["total"])
            c2.metric("Unique users", len(s["users"]))
            if s["users"]:
                st.caption("Visits by user")
                st.dataframe(
                    {
                        "user": [u for u, _ in s["users"]],
                        "visits": [n for _, n in s["users"]],
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            if s["recent"]:
                st.caption("Recent visits")
                st.dataframe(
                    {
                        "when (UTC)": [
                            _fmt_start(int(v["ts_unix"])) for v in s["recent"]
                        ],
                        "user": [v.get("user", "?") for v in s["recent"]],
                    },
                    hide_index=True,
                    use_container_width=True,
                )


_CHROME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="st-"], .stApp, p, div, span, label, button, input, textarea {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', sans-serif; letter-spacing: -0.01em; }
code, pre, .stCode, [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }

.stApp { background:
    radial-gradient(1200px 600px at 80% -10%, rgba(200,255,77,0.06), transparent 60%),
    #0d0f13; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }

/* Hero */
.hero { margin: 0 0 1.1rem 0; padding: 0 0 0.9rem 0;
        border-bottom: 1px solid rgba(232,234,240,0.10); }
.hero-kicker { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.22em;
               text-transform:uppercase; color:#c8ff4d; margin-bottom:0.35rem; }
.hero-title { font-size:2.5rem; font-weight:700; line-height:1.02; margin:0;
              color:#f3f5fa; }
.hero-sub { color:#9aa0ad; font-size:0.98rem; max-width:62ch; margin-top:0.5rem; }

/* Sidebar section headers: compact, accented, uppercase */
section[data-testid="stSidebar"] h2 {
    font-size:0.78rem !important; letter-spacing:0.16em; text-transform:uppercase;
    color:#c8ff4d !important; margin:0.4rem 0 0.2rem 0;
    border-top:1px solid rgba(232,234,240,0.08); padding-top:0.9rem; }
section[data-testid="stSidebar"] .block-container { padding-top:1rem; }

/* Filter summary bar */
.filterbar { display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center;
             margin:0.2rem 0 0.9rem 0; }
.chip { font-family:'IBM Plex Mono',monospace; font-size:0.76rem;
        padding:0.22rem 0.6rem; border-radius:999px;
        background:rgba(200,255,77,0.10); border:1px solid rgba(200,255,77,0.35);
        color:#dff7a6; white-space:nowrap; }
.chip-muted { background:rgba(232,234,240,0.06); border-color:rgba(232,234,240,0.16);
              color:#aab0bd; }
.chip-count { background:transparent; border:none; color:#f3f5fa; font-weight:600;
              padding-left:0; font-size:0.95rem; }

/* Make the primary search box read as the hero action */
.stForm { border:1px solid rgba(200,255,77,0.22) !important; border-radius:14px;
          background:rgba(23,26,33,0.6); padding:0.4rem 0.9rem 0.2rem; }
.stTextInput input { font-size:1.05rem; }
</style>
"""


_EXAMPLE_QUERIES = [
    "unprotected left turn across oncoming traffic",
    "pedestrian crossing at a crosswalk",
    "vehicle cutting in from the right lane",
    "construction cones narrowing the lane",
]


def _run_query(query_text: str, processor, model, device: str) -> None:
    """Encode a text query into the active search vector and reset the view."""
    t0 = time.time()
    vector = search_engine.encode_query(query_text, processor, model, device)
    st.session_state["nls_vector"] = vector
    st.session_state["nls_text_vector"] = vector
    st.session_state["nls_label"] = f'text query: "{query_text.strip()}"'
    st.session_state["nls_encode_ms"] = (time.time() - t0) * 1000
    _clear_marks()
    _reset_browse()


def _inject_chrome() -> None:
    st.markdown(_CHROME_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero">'
        '<div class="hero-kicker">Cosmos-Embed &middot; video retrieval</div>'
        '<div class="hero-title">Natural-Language Video Search</div>'
        '<div class="hero-sub">Describe a moment in plain English &mdash; the corpus '
        "is ranked by visual similarity. Narrow the population with the filters in "
        "the sidebar (date range, or a Data Explorer segment set), then export the "
        "matches.</div></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_chrome()
    config = AppConfig.from_env()
    _record_visit_once()

    try:
        processor, model = get_model(config.model_artifact_uri, config.device)
    except _CORPUS_ERRORS as exc:
        st.error(f"Failed to load text encoder: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Corpus")
        st.caption("The pool of video clips you search over.")
        # Not a text input any more: the corpus is pinned. Pointing a run at a
        # different table silently returns a plausible ranked list scored against
        # another model's embeddings, so there is nothing safe to type here.
        embeddings_uri = full_corpus.DEFAULT_CORPUS_TABLE_URI
        st.code(embeddings_uri, language=None)
        st.caption(f"Cache root: `{local_cache.cache_root()}`")

        st.header("Display")
        page_size = st.slider(
            "Results per page",
            min_value=4,
            max_value=100,
            value=24,
            step=4,
            help="How many ranked clips to show per page.",
        )
        cols_per_row = st.slider(
            "Grid columns",
            min_value=2,
            max_value=6,
            value=3,
            help="Clips per row in the results grid.",
        )

        with st.expander("Advanced · refine by 👍 / 👎 examples", expanded=False):
            st.caption(
                "After a search, mark results 👍 / 👎, then press **Refine** to "
                "re-rank toward your 👍 and away from your 👎."
            )
            negative_weight = st.slider(
                "Avoid-👎 strength",
                0.0,
                2.0,
                0.5,
                0.1,
                help="How hard to push the search away from your 👎 examples. "
                "0 ignores them; higher avoids harder. Moderate by default -- with "
                "few marks, over-avoiding chases noise.",
            )
            blend_text = st.checkbox(
                "Keep the original text query in the mix",
                value=False,
                help="Blend your typed query back into the refined direction so "
                "iterative refinement stays anchored to your prompt.",
            )
            text_weight = (
                st.slider("Text-query weight", 0.0, 1.0, 0.3, 0.05)
                if blend_text
                else 0.0
            )

    _render_analytics_sidebar(_current_user(), config.maintainer_emails)

    st.caption(
        f"Encoder: {config.model_artifact_uri or 'base Cosmos-Embed1-448p'} "
        f"| device: {config.device}"
    )

    if not embeddings_uri.strip():
        st.info("Enter a Lance embeddings URI in the sidebar to begin.")
        return

    try:
        corpus = get_corpus(embeddings_uri.strip(), config.matrix_dtype)
    except _CORPUS_ERRORS as exc:
        st.error(f"Failed to load corpus `{embeddings_uri}`: {exc}")
        st.stop()
    st.caption(
        f"Corpus: {corpus.num_rows:,} chunks x {corpus.dim} dim ({corpus.matrix.dtype})"
    )

    # The active query vector (text-encoded, or a refinement direction) persists
    # in session_state so mark clicks and slider tweaks don't re-encode or reset
    # the search. The original text-query vector is kept separately so a refine
    # can optionally blend it back in. Switching corpus invalidates both (the
    # row indices, and thus any marks, no longer refer to the same matrix).
    if st.session_state.get("nls_corpus_uri") != embeddings_uri.strip():
        st.session_state.pop("nls_vector", None)
        st.session_state.pop("nls_text_vector", None)
        st.session_state.pop("nls_label", None)
        _clear_marks()
        _reset_browse()
        st.session_state["nls_corpus_uri"] = embeddings_uri.strip()

    # Date-range filter on each clip's start time. Defaults to the full corpus
    # span (no narrowing until the user changes it). Bounds come from the corpus,
    # so this section is rendered after the corpus loads.
    span_lo, span_hi = corpus.time_span()
    lo_date = dt.datetime.fromtimestamp(span_lo, tz=dt.timezone.utc).date()
    hi_date = dt.datetime.fromtimestamp(span_hi, tz=dt.timezone.utc).date()
    with st.sidebar:
        st.header("Filters")
        st.caption("Narrow the pool of clips that search ranks, shows, and exports.")
        date_range = st.date_input(
            "Date range (UTC)",
            value=(lo_date, hi_date),
            min_value=lo_date,
            max_value=hi_date,
            help="Keep only clips whose start time falls in this UTC range. "
            "Defaults to the full corpus span.",
        )
    # st.date_input returns a single date mid-edit (before the second click).
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        from_date, to_date = date_range
    else:
        single = date_range[0] if isinstance(date_range, (tuple, list)) else date_range
        from_date = to_date = single
    range_start_unix, range_end_unix = _day_range_to_unix(from_date, to_date)
    # Leave bounds open when the picker still spans the whole corpus, so the
    # default path skips the date mask entirely.
    start_unix = None if from_date <= lo_date else range_start_unix
    end_unix = None if to_date >= hi_date else range_end_unix

    # Optional downsample to a Data Explorer (DORA) segment set: the chosen set
    # is the superset, and only mini-segments whose segment_id is in it are
    # displayed/exported. Matching needs the corpus to carry segment_id.
    corpus_has_segment_id = bool((corpus.segment_id_array() != "").any())
    allowed_segment_ids = None
    segset_chip = None
    with st.sidebar:
        st.markdown("**Data Explorer segment set**")
        downsample = st.checkbox(
            "Restrict to a segment set",
            value=False,
            help="The chosen set becomes the superset: only mini-segments whose "
            "segment_id is in it are searched, displayed, and exported.",
        )
        if not corpus_has_segment_id:
            st.caption(
                ":warning: This corpus has no `segment_id`, so a set can't match "
                "it -- selecting one would return nothing."
            )
        if downsample:
            name_filter = st.text_input(
                "Segment-set name filter",
                value="",
                help="Name substring to find a Data Explorer segment set "
                "(server-side filter over ~12k datasets).",
            )
            sets: list[dora_client.SegmentSet] = []
            if not name_filter.strip():
                st.caption("Type a name filter to list segment sets.")
            else:
                try:
                    sets = get_segment_sets(name_filter.strip())
                except dora_client.DoraUnavailable as exc:
                    st.error(str(exc))
            if sets:
                choice = st.selectbox(
                    "Segment set",
                    options=sets,
                    format_func=lambda s: s.label(),
                    key="nls_segset_choice",
                )
                try:
                    allowed_segment_ids = get_segment_ids(choice.dataset_uuid)
                    segset_chip = (
                        f"{choice.name} v{choice.version} "
                        f"({len(allowed_segment_ids):,} segs)"
                    )
                except dora_client.DoraUnavailable as exc:
                    st.error(str(exc))
            elif name_filter.strip():
                st.caption("No segment sets match that filter.")

    with st.form("search_form"):
        query = st.text_input(
            "What are you looking for?",
            placeholder="Describe a moment, e.g. ego makes an unprotected left turn "
            "across oncoming traffic",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Search", type="primary")
    if submitted and query.strip():
        _run_query(query, processor, model, config.device)

    # One-click example queries to make the search affordance obvious.
    st.caption("Try an example:")
    ex_cols = st.columns(len(_EXAMPLE_QUERIES))
    for col, example in zip(ex_cols, _EXAMPLE_QUERIES):
        if col.button(example, key=f"ex_{example}", use_container_width=True):
            _run_query(example, processor, model, config.device)
            st.rerun()

    query_vector = st.session_state.get("nls_vector")
    if query_vector is None:
        st.info(
            "Type a natural-language prompt above (or tap an example) to rank the "
            "corpus by visual similarity."
        )
        return
    if st.session_state.get("nls_encode_ms") is not None:
        st.caption(f"Encoded query in {st.session_state['nls_encode_ms']:.0f}ms")

    # Score the whole corpus once, then build the (date- and/or segment-set
    # filtered) descending-score order. Both the on-screen page and the CSV
    # export are windows into `order`, so they follow the same filtering.
    t1 = time.time()
    scores = search_engine.score_corpus(query_vector, corpus)
    order = search_engine.ranked_order(
        scores,
        corpus,
        start_unix=start_unix,
        end_unix=end_unix,
        allowed_segment_ids=allowed_segment_ids,
    )
    rank_ms = (time.time() - t1) * 1000
    total = int(order.size)

    # Active-filters bar: always shows what's currently shaping the results, so
    # the query + every applied filter are visible in one place above the grid.
    chips = [f'<span class="chip chip-count">{total:,} clips</span>']
    chips.append(f'<span class="chip">&#128269; {st.session_state["nls_label"]}</span>')
    if start_unix is not None or end_unix is not None:
        chips.append(
            f'<span class="chip">&#128197; {from_date} &rarr; {to_date}</span>'
        )
    if segset_chip is not None:
        chips.append(f'<span class="chip">&#127919; {segset_chip}</span>')
    else:
        chips.append('<span class="chip chip-muted">all segments</span>')
    st.markdown(
        f'<div class="filterbar">{"".join(chips)}</div>', unsafe_allow_html=True
    )

    if total == 0:
        st.warning(
            "No clips match the current filters"
            + (
                " (try widening the date range or segment set)."
                if (allowed_segment_ids or start_unix or end_unix)
                else "."
            )
        )
        return

    # Browse controls: page from any rank or from a similarity-score threshold.
    # Whichever input the user last edited decides the page start.
    score_hi = float(scores[order[0]])
    score_lo = float(scores[order[-1]])
    # Clamp any persisted values into the current (corpus/date-dependent) bounds,
    # so a shrinking match set or a re-scored corpus can't trip Streamlit's
    # min/max validation on the number inputs.
    if _START_RANK_KEY in st.session_state:
        st.session_state[_START_RANK_KEY] = min(
            max(int(st.session_state[_START_RANK_KEY]), 1), total
        )
    if _START_SCORE_KEY in st.session_state:
        st.session_state[_START_SCORE_KEY] = max(
            score_lo, min(score_hi, float(st.session_state[_START_SCORE_KEY]))
        )

    st.markdown("###### Browse the ranking")
    rank_col, score_col, prev_col, next_col = st.columns([3, 3, 1, 1])
    with rank_col:
        start_rank = st.number_input(
            "Jump to rank #",
            min_value=1,
            max_value=total,
            value=1,
            step=page_size,
            key=_START_RANK_KEY,
            on_change=_set_browse_mode("rank"),
            help=f"Start the page at this rank (1 = best match, of {total:,}).",
        )
    with score_col:
        start_score = st.number_input(
            "…or jump to similarity ≤",
            min_value=score_lo,
            max_value=score_hi,
            value=score_hi,
            step=0.01,
            format="%.3f",
            key=_START_SCORE_KEY,
            on_change=_set_browse_mode("score"),
            help="Start the page at the first clip scoring at or below this "
            "similarity (1.0 = identical, lower = less similar).",
        )
    if st.session_state.get(_BROWSE_MODE_KEY) == "score":
        start = search_engine.start_index_for_score(scores, order, start_score)
    else:
        start = int(start_rank) - 1
    start = max(0, min(start, total - 1))
    with prev_col:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
        if st.button("< Prev", disabled=start <= 0, use_container_width=True):
            st.session_state[_START_RANK_KEY] = max(1, start + 1 - page_size)
            st.session_state[_BROWSE_MODE_KEY] = "rank"
            st.rerun()
    with next_col:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
        if st.button(
            "Next >", disabled=start + page_size >= total, use_container_width=True
        ):
            st.session_state[_START_RANK_KEY] = start + 1 + page_size
            st.session_state[_BROWSE_MODE_KEY] = "rank"
            st.rerun()

    hits = search_engine.hits_from_order(corpus, scores, order, start, page_size)
    if not hits:
        st.warning("No results.")
        return
    end_rank = hits[-1].rank
    st.caption(
        f"Showing ranks {start + 1:,}-{end_rank:,} of {total:,} for "
        f"{st.session_state['nls_label']} -- scores "
        f"{hits[0].score:.3f}..{hits[-1].score:.3f} "
        f"(ranked {corpus.num_rows:,} chunks in {rank_ms:.0f}ms)"
    )

    # Relevance-feedback controls. Marks are read from the persisted accumulator
    # (synced from the radios on the previous rerun), so they cover every result
    # marked across refine rounds, not just the ones currently on screen.
    positives, negatives = _marked_indices()
    refine_col, clear_col, _ = st.columns([3, 2, 4])
    with refine_col:
        refine = st.button(
            f"Refine ({len(positives)} 👍 / {len(negatives)} 👎)",
            type="primary",
            disabled=not positives,
            help="Re-rank the whole corpus by a direction toward your 👍 and "
            "away from your 👎. Iterate to tighten the result set.",
        )
    with clear_col:
        if st.button("Clear marks", disabled=not (positives or negatives)):
            _clear_marks()
            st.rerun()
    if refine:
        text_vector = st.session_state.get("nls_text_vector") if blend_text else None
        try:
            direction = search_engine.refine_query(
                corpus,
                positives,
                negative_indices=negatives,
                text_vector=text_vector,
                negative_weight=negative_weight,
                text_weight=text_weight,
            )
        except ValueError as exc:
            st.error(f"Could not refine: {exc}")
        else:
            st.session_state["nls_vector"] = direction
            label = f"{len(positives)} 👍"
            if negatives:
                label += f" - {negative_weight:.1f}x{len(negatives)} 👎"
            if text_vector is not None:
                label += f" + text x{text_weight:.2f}"
            st.session_state["nls_label"] = label
            _reset_browse()
            # Keep the marks -- they accumulate across refine rounds; only a new
            # text query (or Clear marks) resets them.
            st.rerun()

    # Export the metadata of the top-N results for the current ranking (text
    # query or refinement direction), within the active date range. Always starts
    # from rank 1 regardless of which page is on screen.
    export_col, _ = st.columns([3, 6])
    with export_col:
        export_n = st.number_input(
            "Rows to export",
            min_value=1,
            max_value=total,
            value=int(min(max(page_size, 100), total)),
            step=10,
            help="How many top-ranked results (within the current date range) to "
            "include in the metadata dump.",
        )
        st.text_input(
            "Tag",
            key=_EXPORT_TAG_KEY,
            help="Optional label written into a rightmost 'tag' column, identical "
            "for every exported row. Use it to turn a search into a labeled "
            "dataset.",
        )
        export_hits = search_engine.hits_from_order(
            corpus, scores, order, 0, int(export_n)
        )
        # st.download_button bakes its bytes when the script runs, so a tag typed
        # but not yet applied (no Enter) would export blank. Gate the download
        # behind an explicit "Prepare" click: that click commits the pending tag
        # edit in the same rerun, so the CSV is always built from the tag the user
        # sees in the caption below -- no Enter-then-click ordering to get wrong.
        if st.button("Prepare CSV export"):
            st.session_state[_EXPORT_READY_KEY] = True
        if st.session_state.get(_EXPORT_READY_KEY):
            tag = st.session_state.get(_EXPORT_TAG_KEY, "")
            st.download_button(
                f"Download top-{len(export_hits)} metadata (CSV)",
                data=_results_csv(export_hits, tag),
                file_name="nls_results.csv",
                mime="text/csv",
                help="rank, similarity score, per-clip identifiers, and your tag "
                "for the current ranking.",
            )
            if tag:
                st.caption(f"`tag` column = `{tag}`")
            else:
                st.caption("`tag` column empty (no tag entered).")

    client = oci_s3.s3_client()
    for row_start in range(0, len(hits), cols_per_row):
        row_hits = hits[row_start : row_start + cols_per_row]
        for col, hit in zip(st.columns(len(row_hits)), row_hits):
            with col:
                st.markdown(f"**#{hit.rank:,}** &nbsp; score `{hit.score:.3f}`")
                try:
                    url = oci_s3.presign_get(
                        hit.source_media_uri, client, config.presign_ttl_s
                    )
                    st.video(url)
                except (ValueError, botocore.exceptions.ClientError) as exc:
                    st.error(f"video unavailable: {exc}")
                mark_key = _MARK_PREFIX + hit.chunk_id
                prior = _marks().get(hit.chunk_id)
                # Restore a prior label if this result re-appears after a refine
                # (its widget key may have been GC'd while it was off-screen).
                if mark_key not in st.session_state and prior is not None:
                    st.session_state[mark_key] = prior["mark"]
                mark = st.radio(
                    "mark",
                    _MARK_OPTIONS,
                    key=mark_key,
                    horizontal=True,
                    label_visibility="collapsed",
                    format_func=_MARK_LABELS.get,
                )
                _sync_mark(hit.chunk_id, hit.index, mark)
                st.caption(f"`{hit.chunk_id}`")
                st.caption(_fmt_start(hit.chunk_start_unix))


if __name__ == "__main__":
    main()
