"use strict";

const EXAMPLES = [];

const state = {
  query: "",
  mode: "search", // "search" | "refine" | "resume"
  page: 0,
  pageSize: 24,
  total: 0,
  corpus: null,
  segUuid: null,
  segLabel: null,
  segName: null,
  embeddingsUri: null,
  marks: {}, // chunk_id -> {mark:"up"|"down", segment_id, rank, score}
  resumeVec: null, // stored search vector when resuming a saved search
  resumeLabel: "", // the NL query that vector came from (label only)
  pendingResume: null, // saved-session id to resume once the corpus is ready
  offlineScan: true, // offered by default; cleared on deployments with the full corpus resident
};

const $ = (id) => document.getElementById(id);

function fmtInt(n) { return n.toLocaleString("en-US"); }

// ---------- bootstrap ----------
async function init() {
  renderExamples();
  wireEvents();
  wireTabs();
  loadPlatformTag();
  state.pageSize = parseInt($("page-size").value, 10);
  // Deep link from Search history: /?resume=<id> resumes that saved search once
  // the corpus is ready.
  const r = new URLSearchParams(location.search).get("resume");
  if (r) state.pendingResume = r;
  loadDefaultCorpusWhenReady();
}

// Tag the top-left with the active platform (cars vs trucking). Derived from the
// deployment's runtime env server-side, since both run the same image.
async function loadPlatformTag() {
  try {
    const p = await fetch("/api/platform").then((r) => r.json());
    const el = $("platform-tag");
    if (el && p && p.label) {
      el.textContent = p.label;
      el.classList.add(p.name === "trucks" ? "trucks" : "cars");
    }
    // Offline (Lilypad) scan is hidden where the full dataset is resident in CPU memory (e.g.
    // frontier) -- Download CSV over the in-app corpus already covers the whole dataset there.
    state.offlineScan = !p || p.offline_scan !== false;
    document.body.classList.toggle("no-offline-scan", !state.offlineScan);
  } catch (e) { /* non-fatal cosmetic tag */ }
}

// ---------- client-side tab switching (no full reload) ----------
function _setActiveTab(href) {
  document.querySelectorAll(".maintabs .tab").forEach((a) =>
    a.classList.toggle("active", a.getAttribute("href") === href));
}

function _hideAllViews() {
  $("search-view").classList.add("hidden");
  $("history-view").classList.add("hidden");
  $("curate-view").classList.add("hidden");
}

function showSearchView() {
  _hideAllViews();
  undockFilters();
  $("search-view").classList.remove("hidden");
  _setActiveTab("/");
  document.title = "VLM Video Search";
}

function showCurateView() {
  _hideAllViews();
  $("curate-view").classList.remove("hidden");
  _setActiveTab("/curate");
  document.title = "Export";
  // Populate the history picker once per view entry if not already loaded.
  if (!state.exportHistory) loadExportHistory();
  // Refresh the launched-scans panel (status may have advanced) -- only where the offline
  // scan exists; it's hidden on deployments with the full dataset resident (e.g. frontier).
  if (state.offlineScan !== false) loadScanJobs();
  // Dock the shared filter controls inline so the scan's scope is visible + editable in place.
  dockFiltersInline();
}

async function showHistoryView() {
  const hv = $("history-view");
  _setActiveTab("/tags");
  _hideAllViews();
  undockFilters();
  hv.classList.remove("hidden");
  hv.innerHTML = '<p class="sub" style="color:#9aa0a6">loading search history…</p>';
  document.title = "Search history";
  try {
    hv.innerHTML = await fetch("/api/search_history").then((r) => r.text());
  } catch (e) {
    hv.innerHTML = '<p class="sub" style="color:#ffcf5c">could not load search history.</p>';
  }
}

function _routeTo(path) {
  // Search history + Export are one page now: /tags is kept as an alias of /curate so
  // old links/bookmarks still land on the merged view.
  if (path === "/curate" || path === "/tags") showCurateView();
  else showSearchView();
}

// Intercept the header tabs so switching swaps views in place (keeping the
// search DOM + state alive) instead of doing a full page reload + re-init.
function wireTabs() {
  document.querySelectorAll(".maintabs .tab").forEach((a) => {
    a.addEventListener("click", (e) => {
      const href = a.getAttribute("href");
      if (href !== "/" && href !== "/tags" && href !== "/curate") return;
      e.preventDefault();
      history.pushState({ view: href }, "", href);
      _routeTo(href);
    });
  });
  window.addEventListener("popstate", () => _routeTo(location.pathname));
  // Honor a deep link / refresh that lands here at /curate or /tags.
  _routeTo(location.pathname);
}

// The model + default corpus take minutes to warm on a cold start, during which
// /api/corpus returns 503. Poll until it's ready, then auto-apply the default
// corpus -- so the page populates itself without a manual reload.
async function loadDefaultCorpusWhenReady() {
  const pill = $("corpus-pill");
  const MAX_ATTEMPTS = 240; // ~12 min at 3s
  for (let attempt = 0; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const r = await fetch("/api/corpus");
      if (r.ok) {
        const c = await r.json();
        if (c && c.num_rows != null) {
          applyCorpus(c);
          setStatus("Type a query above to begin.");
          if (state.pendingResume) {
            const id = state.pendingResume; state.pendingResume = null;
            resumeSession(id);
          }
          return;
        }
      }
      // 503 (still warming) or an unexpected body -> keep waiting.
      pill.textContent = "warming up — loading model + corpus…";
    } catch (e) {
      pill.textContent = "warming up — loading model + corpus…";
    }
    await new Promise((res) => setTimeout(res, 3000));
  }
  pill.textContent = "corpus unavailable — reload to retry";
}

function applyCorpus(c) {
  state.corpus = c;
  state.embeddingsUri = c.embeddings_uri;
  $("embeddings-uri").value = c.embeddings_uri || "";
  if (c.model_uri !== undefined) $("model-uri").value = c.model_uri || "";
  $("corpus-pill").textContent =
    `${fmtInt(c.num_rows)} clips · ${c.dim}d · ${c.model}`;
  $("from-date").value = c.span_lo_date;
  $("to-date").value = c.span_hi_date;
  $("from-date").min = c.span_lo_date; $("from-date").max = c.span_hi_date;
  $("to-date").min = c.span_lo_date; $("to-date").max = c.span_hi_date;
  // segment-set availability depends on whether THIS corpus carries segment_id
  const hasSeg = !!c.has_segment_id;
  clearSegSet(); // reset the combobox + state for the new corpus
  state.filterLanceUri = null; // a dataset's segment_ids are corpus-specific
  $("lance-filter").value = "";
  $("lance-clear").classList.add("hidden");
  $("lance-note").textContent = "keeps only chunks whose segment_id is in this dataset";
  $("lance-note").classList.remove("warn");
  $("segset-filter").disabled = !hasSeg;
  if (!hasSeg) {
    $("segset-note").textContent = "⚠ This corpus has no segment_id — a set can't match it.";
    $("segset-note").classList.add("warn");
  } else {
    // Default to NO segment set: the user types a keyword into the single combo
    // box, then picks a match from the dropdown.
    $("segset-note").textContent = "type a keyword to search Data Explorer sets";
    $("segset-note").classList.remove("warn");
  }
}

async function loadCorpus() {
  const uri = $("embeddings-uri").value.trim();
  if (!uri) return;
  $("corpus-note").textContent = "loading corpus (downloading + into memory)…";
  $("corpus-note").classList.remove("warn");
  $("corpus-pill").textContent = "loading corpus…";
  try {
    const c = await fetch("/api/corpus?uri=" + encodeURIComponent(uri))
      .then((r) => { if (!r.ok) return r.json().then((j) => { throw new Error(j.detail || r.status); }); return r.json(); });
    applyCorpus(c);
    // reset the active search + marks for the new corpus
    state.query = ""; state.marks = {}; updateMarkCount();
    $("grid").innerHTML = ""; $("pager").classList.add("hidden");
    $("filterbar").classList.add("hidden");
    $("corpus-note").textContent =
      `Loaded ${fmtInt(c.num_rows)} clips · segment_id ${c.has_segment_id ? "present ✓" : "absent"}`;
    setStatus("Corpus loaded — run a search.");
  } catch (e) {
    $("corpus-note").textContent = "failed to load corpus: " + e.message;
    $("corpus-note").classList.add("warn");
    $("corpus-pill").textContent = "corpus unavailable";
  }
}

// Swap the resident text encoder at runtime to test embedding search across
// models. The model MUST match the corpus's embedding space, so this pairs with
// loading the corpus that was embedded with the same checkpoint.
async function loadModel() {
  const uri = $("model-uri").value.trim();
  $("model-note").textContent = "loading model (downloading + into memory, ~minutes)…";
  $("model-note").classList.remove("warn");
  try {
    const m = await fetch("/api/model?uri=" + encodeURIComponent(uri))
      .then((r) => { if (!r.ok) return r.json().then((j) => { throw new Error(j.detail || r.status); }); return r.json(); });
    $("model-note").textContent = "model: " + m.label;
    // Reflect the new encoder in the pill (keep the corpus counts).
    if (state.corpus) {
      $("corpus-pill").textContent =
        `${fmtInt(state.corpus.num_rows)} clips · ${state.corpus.dim}d · ${m.label}`;
      state.corpus.model = m.label;
      state.corpus.model_uri = m.model_uri;
    }
    // The joint space changed; re-run the active query so results match.
    if (state.query) reload({ page: 0 });
    else setStatus("Model loaded — run a search.");
  } catch (e) {
    $("model-note").textContent = "failed to load model: " + e.message;
    $("model-note").classList.add("warn");
  }
}

function renderExamples() {
  const box = $("examples");
  EXAMPLES.forEach((ex) => {
    const b = document.createElement("button");
    b.className = "example";
    b.type = "button";
    b.textContent = ex;
    b.onclick = () => { $("query").value = ex; runSearch({ page: 0 }); };
    box.appendChild(b);
  });
}

function wireEvents() {
  $("search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch({ page: 0 }); });
  $("page-size").addEventListener("change", () => {
    state.pageSize = parseInt($("page-size").value, 10);
    if (state.query) reload({ page: 0 });
  });
  $("from-date").addEventListener("change", () => { if (state.query) reload({ page: 0 }); });
  $("to-date").addEventListener("change", () => { if (state.query) reload({ page: 0 }); });
  $("prev").addEventListener("click", () => reload({ page: state.page - 1 }));
  $("next").addEventListener("click", () => reload({ page: state.page + 1 }));
  $("jump-rank-btn").addEventListener("click", () => {
    const r = parseInt($("jump-rank").value, 10);
    if (r >= 1) reload({ start_rank: r });
  });
  $("jump-score-btn").addEventListener("click", () => {
    const s = parseFloat($("jump-score").value);
    if (!Number.isNaN(s)) reload({ start_score: s });
  });
  $("refine-btn").addEventListener("click", () => runRefine({ page: 0 }));
  $("clear-marks").addEventListener("click", clearMarks);
  $("blend-text").addEventListener("change", (e) => {
    $("text-weight").disabled = !e.target.checked;
  });
  $("dist-btn").addEventListener("click", loadScoreDistribution);
  $("save-vec-btn").addEventListener("click", saveVector);
  $("search-export-btn").addEventListener("click", exportSearchCsv);
  // Export view: history picker + Download CSV (resident corpus) + launch offline scan.
  $("curate-csv-btn").addEventListener("click", downloadCsv);
  $("curate-scan-btn").addEventListener("click", launchCurateScan);
  $("export-hist-reload").addEventListener("click", loadExportHistory);
  $("export-hist-all").addEventListener("click", () => exportHistSelectAll(true));
  $("export-hist-none").addEventListener("click", () => exportHistSelectAll(false));
  $("export-hist-filter").addEventListener("input", filterExportHistory);
  // Per-tag "open ↗": reload that tag's saved search into the Search view to iterate.
  $("export-history").addEventListener("click", (e) => {
    const b = e.target.closest("button.exp-resume");
    if (b && b.dataset.id) resumeSession(b.dataset.id);
  });
  $("scan-jobs-reload").addEventListener("click", loadScanJobs);
  $("load-corpus").addEventListener("click", loadCorpus);
  $("load-model").addEventListener("click", loadModel);

  // Resume buttons live inside the (server-rendered) Search-history fragment, so
  // delegate: intercept the click and resume in place instead of a full reload.
  $("history-view").addEventListener("click", (e) => {
    const a = e.target.closest("a.resume");
    if (!a || !a.dataset.id) return;
    e.preventDefault();
    resumeSession(a.dataset.id);
  });

  wireSegCombo();
}

// All wiring for the single segment-set combobox: type -> search -> dropdown.
function wireSegCombo() {
  let t;
  $("segset-filter").addEventListener("input", (e) => {
    clearTimeout(t);
    // Editing the box abandons any locked-in set (until a new one is picked).
    if ($("segset-combo").classList.contains("chosen")) {
      $("segset-combo").classList.remove("chosen");
      $("segset-clear").classList.add("hidden");
      state.segUuid = null; state.segName = null; state.segLabel = null;
    }
    const v = e.target.value;
    // Require >=2 chars: a 1-char filter matches almost everything and forces a
    // huge DORA fetch for no useful result.
    if (v.trim().length < 2) {
      hideSegMenu();
      $("segset-note").textContent = v.trim()
        ? "keep typing… (min 2 characters)"
        : "type a keyword to search Data Explorer sets";
      return;
    }
    t = setTimeout(() => loadSegmentSets(v), 300);
  });
  $("segset-filter").addEventListener("focus", () => {
    if (_segMenuItems.length && $("segset-filter").value.trim().length >= 2 &&
        !$("segset-combo").classList.contains("chosen")) showSegMenu();
  });
  $("segset-filter").addEventListener("keydown", (e) => {
    if ($("segset-menu").classList.contains("hidden")) return;
    if (e.key === "ArrowDown") { e.preventDefault(); moveSegActive(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveSegActive(-1); }
    else if (e.key === "Enter") {
      if (_segActiveIdx >= 0 && _segMenuItems[_segActiveIdx]) {
        e.preventDefault(); chooseSet(_segMenuItems[_segActiveIdx]);
      }
    } else if (e.key === "Escape") { hideSegMenu(); }
  });
  // Delay the close so a menu item's click/mousedown can register first.
  $("segset-filter").addEventListener("blur", () => setTimeout(hideSegMenu, 120));
  $("segset-clear").addEventListener("click", () => {
    clearSegSet();
    $("segset-note").textContent = "type a keyword to search Data Explorer sets";
    $("segset-note").classList.remove("warn");
    if (state.query) reload({ page: 0 });
  });

  // Lance/parquet downsample dataset: apply on Enter or blur; clear via the ×.
  $("lance-filter").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); applyLanceFilter(); }
  });
  $("lance-filter").addEventListener("blur", applyLanceFilter);
  $("lance-clear").addEventListener("click", clearLanceFilter);

  // Vehicle id filter: apply on Enter/blur, clear via the ×. The value is read
  // live by _vehicleValue(), so applying just toggles the × and re-runs.
  $("vehicle-filter").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); applyVehicleFilter(); }
  });
  $("vehicle-filter").addEventListener("blur", applyVehicleFilter);
  $("vehicle-clear").addEventListener("click", clearVehicleFilter);

  // Drive-id (run_uuid) filter: same apply-on-Enter/blur + × pattern as vehicle.
  $("drive-filter").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); applyDriveFilter(); }
  });
  $("drive-filter").addEventListener("blur", applyDriveFilter);
  $("drive-clear").addEventListener("click", clearDriveFilter);
}

// Drive-id filter is read live from the input by _driveValue(); applying just
// shows/hides the × and re-runs the search when the value actually changed.
function applyDriveFilter() {
  const v = $("drive-filter").value.trim();
  $("drive-clear").classList.toggle("hidden", !v);
  if (v === (state._driveApplied || "")) return;
  state._driveApplied = v;
  if (state.query) reload({ page: 0 });
}

function clearDriveFilter() {
  const had = !!$("drive-filter").value.trim();
  $("drive-filter").value = "";
  $("drive-clear").classList.add("hidden");
  state._driveApplied = "";
  if (had && state.query) reload({ page: 0 });
}

// Vehicle filter is read live from the input by _vehicleValue(); applying just
// shows/hides the × and re-runs the search when the value actually changed.
function applyVehicleFilter() {
  const v = $("vehicle-filter").value.trim();
  $("vehicle-clear").classList.toggle("hidden", !v);
  if (v === (state._vehicleApplied || "")) return;
  state._vehicleApplied = v;
  if (state.query) reload({ page: 0 });
}

function clearVehicleFilter() {
  const had = !!$("vehicle-filter").value.trim();
  $("vehicle-filter").value = "";
  $("vehicle-clear").classList.add("hidden");
  state._vehicleApplied = "";
  if (had && state.query) reload({ page: 0 });
}

// Read the lance/parquet path box; if it changed, re-run so the new downsample
// applies. The status (segments in dataset / chunks left / errors) comes back
// on the search response and is rendered by renderLanceNote().
function applyLanceFilter() {
  const v = $("lance-filter").value.trim();
  if (v === (state.filterLanceUri || "")) return;
  state.filterLanceUri = v || null;
  $("lance-clear").classList.toggle("hidden", !v);
  $("lance-note").textContent = v
    ? "applying — keeps only chunks whose segment_id is in this dataset…"
    : "keeps only chunks whose segment_id is in this dataset";
  $("lance-note").classList.remove("warn");
  if (state.query) reload({ page: 0 });
}

function clearLanceFilter() {
  $("lance-filter").value = "";
  $("lance-clear").classList.add("hidden");
  $("lance-note").textContent = "keeps only chunks whose segment_id is in this dataset";
  $("lance-note").classList.remove("warn");
  const had = !!state.filterLanceUri;
  state.filterLanceUri = null;
  if (had && state.query) reload({ page: 0 });
}

// Reflect the server's downsample status for the lance dataset on the note line.
function renderLanceNote(data) {
  if (!state.filterLanceUri) return;
  if (data.filter_lance_error) {
    $("lance-note").textContent = "dataset error: " + data.filter_lance_error;
    $("lance-note").classList.add("warn");
    return;
  }
  $("lance-note").classList.remove("warn");
  const unit = data.filter_lance_key === "run_uuid" ? "runs" : "segments";
  $("lance-note").textContent =
    `dataset: ${fmtInt(data.filter_lance_count || 0)} ${unit} · ${fmtInt(data.total)} chunks left`;
}

// ---------- segment sets (single combobox: type -> dropdown -> pick) ----------
const _segSetCache = {}; // filter -> sets[]; avoids re-hitting DORA while typing
let _segMenuItems = [];  // sets currently rendered in the dropdown
let _segActiveIdx = -1;  // keyboard-highlighted option (-1 = none)

function showSegMenu() {
  $("segset-menu").classList.remove("hidden");
  $("segset-filter").setAttribute("aria-expanded", "true");
}
function hideSegMenu() {
  $("segset-menu").classList.add("hidden");
  $("segset-filter").setAttribute("aria-expanded", "false");
  _segActiveIdx = -1;
}

// Reset the combobox to its empty/no-set state (no re-search; callers decide).
function clearSegSet() {
  state.segUuid = null; state.segName = null; state.segLabel = null;
  $("segset-combo").classList.remove("chosen");
  $("segset-clear").classList.add("hidden");
  $("segset-filter").value = "";
  _segMenuItems = [];
  hideSegMenu();
}

// Lock a picked set into the box and apply it (prefetch ids + re-search).
function chooseSet(set) {
  state.segUuid = set.uuid;
  state.segName = set.name;
  state.segLabel = `${set.name} v${set.version} (${fmtInt(set.num_segments)} segs)`;
  $("segset-filter").value = state.segLabel;
  $("segset-combo").classList.add("chosen");
  $("segset-clear").classList.remove("hidden");
  hideSegMenu();
  prefetchSegmentSet(set.uuid);
  if (state.query) reload({ page: 0 });
}

function moveSegActive(delta) {
  const opts = $("segset-menu").querySelectorAll(".combo-opt");
  if (!opts.length) return;
  _segActiveIdx = (_segActiveIdx + delta + opts.length) % opts.length;
  opts.forEach((o, i) => o.classList.toggle("active", i === _segActiveIdx));
  opts[_segActiveIdx].scrollIntoView({ block: "nearest" });
}

function renderSegMenu(sets) {
  const menu = $("segset-menu");
  _segMenuItems = sets;
  _segActiveIdx = -1;
  if (!sets.length) {
    menu.innerHTML = '<div class="combo-msg">no sets match that name</div>';
    showSegMenu();
    return;
  }
  menu.innerHTML = "";
  sets.forEach((s, i) => {
    const o = document.createElement("div");
    o.className = "combo-opt";
    o.setAttribute("role", "option");
    o.dataset.idx = String(i);
    o.innerHTML =
      `<span class="opt-name">${escapeHtml(s.name)}</span>` +
      `<span class="opt-meta">v${escapeHtml(String(s.version))} · ${fmtInt(s.num_segments)} segments</span>`;
    // mousedown (not click) so it fires before the input's blur closes the menu.
    o.addEventListener("mousedown", (e) => { e.preventDefault(); chooseSet(s); });
    menu.appendChild(o);
  });
  showSegMenu();
}

async function loadSegmentSets(filter) {
  const key = filter.trim().toLowerCase();
  $("segset-note").textContent = "loading segment sets…";
  $("segset-note").classList.remove("warn");
  try {
    const sets = _segSetCache[key] || await fetch(
      "/api/segment_sets?name_filter=" + encodeURIComponent(filter))
      .then((r) => { if (!r.ok) throw new Error("DORA " + r.status); return r.json(); });
    _segSetCache[key] = sets;
    renderSegMenu(sets);
    $("segset-note").textContent = sets.length
      ? `${sets.length} set${sets.length > 1 ? "s" : ""} — pick one to downsample`
      : "no sets match that name";
  } catch (e) {
    $("segset-note").textContent = "couldn't reach Data Explorer: " + e.message;
    $("segset-note").classList.add("warn");
    hideSegMenu();
  }
}

// ---------- search / refine ----------
function _vehicleValue() {
  return ($("vehicle-filter").value || "").trim() || null;
}

function _driveValue() {
  return ($("drive-filter").value || "").trim() || null;
}

function _filterBody() {
  return {
    page_size: state.pageSize,
    from_date: $("from-date").value || null,
    to_date: $("to-date").value || null,
    segment_set_uuid: state.segUuid,
    filter_lance_uri: state.filterLanceUri || null,
    vehicle: _vehicleValue(),
    drive_id: _driveValue(),
    embeddings_uri: state.embeddingsUri,
  };
}

// The active filter set (date / segment set / lance / vehicle / drive) in persistence shape,
// read live from the shared Filters flyout. Saved with a vector, sent with the offline scan,
// and rendered into the launch-point summary — one source of truth for all three.
function _activeFilters() {
  return {
    from_date: $("from-date").value || null,
    to_date: $("to-date").value || null,
    segment_set_uuid: state.segUuid || null,
    segment_set_name: state.segName || null,
    filter_lance_uri: state.filterLanceUri || null,
    vehicle: _vehicleValue(),
    drive_id: _driveValue(),
  };
}

// Dock the shared filter controls (.filter-body) inline into the curate view so the scan's
// filter scope is visible + editable in place, instead of the right-edge hover flyout. The
// SAME DOM node carries all wiring + current values, so this relocates (not duplicates) it —
// the inline inputs ARE the active filters, pre-loaded by default and freely edited/cleared.
function dockFiltersInline() {
  const host = $("curate-filter-host");
  const body = document.querySelector(".filter-body");
  if (!host || !body) return;
  if (body.parentElement !== host) {
    body.classList.add("docked");
    host.appendChild(body);
  }
  // The flyout tab would now be an empty dead hover zone — hide the whole panel shell.
  $("filter-panel").classList.add("hidden");
}

// Restore the filter controls to the right-edge flyout (for the search/history views).
function undockFilters() {
  const panel = $("filter-panel");
  const body = document.querySelector(".filter-body");
  if (panel && body && body.parentElement !== panel) {
    body.classList.remove("docked");
    panel.appendChild(body);
  }
  if (panel) panel.classList.remove("hidden");
}

// A new text query (form submit / example) -> search mode.
async function runSearch(startOpts) {
  const q = $("query").value.trim();
  if (!q) return;
  state.query = q;
  state.mode = "search";
  await _issue("/api/search", { query: q, ...(startOpts || {}), ..._filterBody() }, "search");
}

// Re-rank by the 👍/👎 marks -> refine mode.
async function runRefine(startOpts) {
  if (!state.query) { setStatus("Run a search first, then mark 👍/👎 and refine.", true); return; }
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({
    chunk_id, segment_id: m.segment_id, mark: m.mark, index: m.index, rank: m.rank, score: m.score,
  }));
  if (!marks.some((m) => m.mark === "up")) { setStatus("Mark at least one 👍 to refine.", true); return; }
  state.mode = "refine";
  await _issue("/api/refine", {
    query: state.query,
    marks,
    negative_weight: parseFloat($("neg-weight").value),
    text_weight: $("blend-text").checked ? parseFloat($("text-weight").value) : 0.0,
    ...(startOpts || {}),
    ..._filterBody(),
  }, "refine");
}

// Paging / jumps / filter changes preserve the current mode.
function reload(startOpts) {
  if (state.mode === "refine") return runRefine(startOpts);
  if (state.mode === "resume") return runVectorSearch(startOpts);
  return runSearch(startOpts);
}

// ---------- resume a saved search (from Search history) ----------
// Rank by the stored search vector instead of re-encoding text -> "resume" mode.
async function runVectorSearch(startOpts) {
  if (!state.resumeVec) { setStatus("Nothing to resume.", true); return; }
  state.mode = "resume";
  await _issue("/api/search_by_vector", {
    vector: state.resumeVec,
    query: state.resumeLabel || state.query || "",
    ...(startOpts || {}),
    ..._filterBody(),
  }, "search");
}

// Fetch a saved session by id, restore its query + filters + vector on the
// search page, and run the vector search. Switches the corpus first if the
// saved search used a different one.
async function resumeSession(id) {
  showSearchView();
  setStatus("Loading saved search…");
  let s;
  try {
    s = await fetch("/api/search_session/" + encodeURIComponent(id))
      .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
  } catch (e) {
    setStatus("Could not load saved search: " + e.message, true);
    return;
  }
  if (!s.vector || !s.vector.length) {
    setStatus("That saved search has no stored vector to resume.", true);
    return;
  }
  // Switch corpus if the saved search ran on a different one (loadCorpus resets
  // dates/segset, so restore those AFTER it).
  if (s.embeddings_uri && s.embeddings_uri !== state.embeddingsUri) {
    $("embeddings-uri").value = s.embeddings_uri;
    await loadCorpus();
  }
  $("query").value = s.query || "";
  state.query = s.query || "";
  // Restore the tag so a re-Save updates THIS saved vector (upsert by tag) instead
  // of creating a new one.
  $("save-vec-tag").value = s.tag || "";
  // Restore the 👍/👎 marks saved with this search, so the user sees them and
  // can refine further. renderGrid re-applies the on-state per visible card by
  // chunk_id, and marks that carry an index feed straight into a follow-up Refine.
  state.marks = {};
  (s.thumbs_up || []).forEach((m) => {
    if (m && m.chunk_id) state.marks[m.chunk_id] = {
      mark: "up", segment_id: m.segment_id, index: m.index, rank: m.rank, score: m.score,
    };
  });
  (s.thumbs_down || []).forEach((m) => {
    if (m && m.chunk_id) state.marks[m.chunk_id] = {
      mark: "down", segment_id: m.segment_id, index: m.index, rank: m.rank, score: m.score,
    };
  });
  updateMarkCount();
  if (s.from_date) $("from-date").value = s.from_date;
  if (s.to_date) $("to-date").value = s.to_date;
  // Restore the segment set (display chip + background id load) if there was one.
  if (s.segment_set_uuid) {
    state.segUuid = s.segment_set_uuid;
    state.segName = s.segment_set_name || s.segment_set_uuid;
    state.segLabel = s.segment_set_name || ("id " + s.segment_set_uuid.slice(0, 12) + "…");
    $("segset-filter").value = state.segLabel;
    $("segset-combo").classList.add("chosen");
    $("segset-clear").classList.remove("hidden");
    prefetchSegmentSet(s.segment_set_uuid);
  } else {
    clearSegSet();
  }
  // Restore the lance-downsample + vehicle filters. Set state + boxes directly
  // (not via apply*/clear*, which would each fire their own reload) — the single
  // runVectorSearch at the end of resume applies the whole restored filter set.
  state.filterLanceUri = s.filter_lance_uri || null;
  $("lance-filter").value = s.filter_lance_uri || "";
  $("lance-clear").classList.toggle("hidden", !s.filter_lance_uri);
  $("vehicle-filter").value = s.vehicle || "";
  $("vehicle-clear").classList.toggle("hidden", !s.vehicle);
  state._vehicleApplied = s.vehicle || "";
  $("drive-filter").value = s.drive_id || "";
  $("drive-clear").classList.toggle("hidden", !s.drive_id);
  state._driveApplied = s.drive_id || "";
  state.resumeVec = s.vector;
  state.resumeLabel = s.query || "";
  state.mode = "resume";
  await runVectorSearch({ page: 0 });
}

async function _issue(endpoint, body, mode) {
  setStatus(mode === "refine" ? "Refining…" : "Searching…");
  $("grid").innerHTML = "";
  showDistribution(null);  // a new ranking invalidates any shown distribution
  let data;
  try {
    data = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => {
      if (!r.ok) return r.json().then((j) => {
        console.error("API error", endpoint, r.status, j);
        let msg = j.detail;
        if (Array.isArray(msg)) {
          // FastAPI 422: render each validation error as "field: message".
          msg = msg
            .map((e) => `${(e.loc || []).join(".")}: ${e.msg}`)
            .join("; ");
        }
        throw new Error(msg || ("HTTP " + r.status));
      });
      return r.json();
    });
  } catch (e) {
    setStatus((mode === "refine" ? "Refine" : "Search") + " failed: " + e.message, true);
    return;
  }
  state.total = data.total;
  state.page = data.page;
  state.label = data.label;
  renderFilterBar(data);
  renderLanceNote(data);
  $("refinebar").classList.remove("hidden");
  if (data.total === 0) {
    setStatus("No clips match the current query + filters. Try widening the date range or segment set.");
    $("pager").classList.add("hidden");
    return;
  }
  if (data.segment_set_pending) {
    // Big segment set still downloading in the background — these results are
    // the UNFILTERED ranking; poll and auto-re-run once the set is ready.
    pollSegmentSet();
  } else if (_segPoll) {
    clearInterval(_segPoll); _segPoll = null;
  }
  if (!data.segment_set_pending) {
    setStatus(
      `Ranked ${fmtInt(state.corpus.num_rows)} clips in ${data.elapsed_ms} ms · ` +
      `similarity ${data.score_lo}–${data.score_hi}`
    );
  }
  renderGrid(data.hits);
  renderPager(data);
}

let _segPoll = null;
// Kick off the server-side id-load the moment a set is chosen, so the (cached)
// DORA pull overlaps the user typing a query -- by search time the ids are
// usually already resident and the first filtered search needs no re-run.
let _prefetchPoll = null;
function prefetchSegmentSet(uuid) {
  if (!uuid) return;
  if (_prefetchPoll) { clearInterval(_prefetchPoll); _prefetchPoll = null; }
  $("segset-note").textContent = "pre-loading segment ids…";
  $("segset-note").classList.remove("warn");
  // Kick off the background load, then poll fast (~0.8s) and flip the note to
  // "ready" the moment the ids land -- otherwise the message looks stuck even
  // though a typical set loads in well under a second.
  fetch("/api/segment_set_prefetch?uuid=" + encodeURIComponent(uuid)).catch(() => {});
  const tick = async () => {
    if (state.segUuid !== uuid) { clearInterval(_prefetchPoll); _prefetchPoll = null; return; }
    try {
      const st = await fetch("/api/segment_set_status?uuid=" + encodeURIComponent(uuid))
        .then((r) => r.json());
      if (st.ready) {
        clearInterval(_prefetchPoll); _prefetchPoll = null;
        $("segset-note").textContent = `segment set ready (${fmtInt(st.count)} segments)`;
        $("segset-note").classList.remove("warn");
      } else if (st.status === "error") {
        clearInterval(_prefetchPoll); _prefetchPoll = null;
        $("segset-note").textContent = "segment set load failed: " + (st.error || "unknown");
        $("segset-note").classList.add("warn");
      } else {
        $("segset-note").textContent = st.count
          ? `pre-loading segment ids… ${fmtInt(st.count)} so far`
          : "pre-loading segment ids…";
      }
    } catch (e) { /* transient; keep polling */ }
  };
  tick();
  _prefetchPoll = setInterval(tick, 800);
}

function pollSegmentSet() {
  if (_segPoll || !state.segUuid) return;
  const uuid = state.segUuid;
  setStatus("⏳ Loading segment set in the background — results are NOT downsampled yet; will auto-apply when ready.");
  _segPoll = setInterval(async () => {
    if (state.segUuid !== uuid) { clearInterval(_segPoll); _segPoll = null; return; }
    try {
      const st = await fetch("/api/segment_set_status?uuid=" + encodeURIComponent(uuid)).then((r) => r.json());
      if (st.ready) {
        clearInterval(_segPoll); _segPoll = null;
        setStatus(`Segment set ready (${fmtInt(st.count)} segments) — applying…`);
        reload({ page: 0 });
      } else if (st.status === "error") {
        clearInterval(_segPoll); _segPoll = null;
        setStatus("Segment set load failed: " + (st.error || "unknown"), true);
      } else {
        setStatus(`⏳ Loading segment set… ${fmtInt(st.count)} ids fetched (results not yet downsampled).`);
      }
    } catch (e) { /* transient; keep polling */ }
  }, 3000);
}

function renderFilterBar(data) {
  const bar = $("filterbar");
  const f = data.funnel || {};
  const total = f.corpus_total;
  const chips = [`<span class="chip count">${fmtInt(data.total)} clips</span>`];
  chips.push(`<span class="chip">${escapeHtml(data.label || state.query)}</span>`);
  // Corpus size, so every "X of Y" below has an anchor.
  if (total != null) chips.push(`<span class="chip muted">of ${fmtInt(total)} corpus</span>`);
  const full = $("from-date").value === state.corpus.span_lo_date &&
               $("to-date").value === state.corpus.span_hi_date;
  if (!full) {
    const d = f.date_filtered != null
      ? ` · ${fmtInt(f.in_date_range)} of ${fmtInt(total)}` : "";
    chips.push(`<span class="chip">📅 ${$("from-date").value} → ${$("to-date").value}${d}</span>`);
  }
  if (state.segLabel && data.segment_set_pending) {
    chips.push(`<span class="chip muted">⏳ ${escapeHtml(state.segLabel)} loading — NOT applied</span>`);
  } else if (state.segLabel) {
    const s = (f.in_segment_set != null)
      ? ` · ${fmtInt(f.in_segment_set)} of ${fmtInt(total)} in corpus` : "";
    chips.push(`<span class="chip">🎯 ${escapeHtml(state.segLabel)}${s}</span>`);
  } else {
    chips.push('<span class="chip muted">all segments</span>');
  }
  if (state.filterLanceUri) {
    if (data.filter_lance_error) {
      chips.push(`<span class="chip warn">📂 dataset error — NOT applied</span>`);
    } else {
      const u = data.filter_lance_key === "run_uuid" ? "runs" : "segs";
      chips.push(`<span class="chip">📂 dataset ${fmtInt(data.filter_lance_count || 0)} ${u}</span>`);
    }
  }
  const veh = _vehicleValue();
  if (veh) chips.push(`<span class="chip">vehicle ${escapeHtml(veh)}</span>`);
  const drv = _driveValue();
  if (drv) {
    const n = drv.split(/[,\s]+/).filter(Boolean).length;
    chips.push(`<span class="chip">drive ${n > 1 ? n + " ids" : escapeHtml(drv)}</span>`);
  }
  bar.innerHTML = chips.join("");
  bar.classList.remove("hidden");
}

// Build one result card. `opts.controls` is HTML for the bottom row (marks for the
// search grid; a keep/remove checkbox for curate). `opts.badge` shows e.g. the
// matching query. preload="metadata" + #t=0.1 paints the first frame as a thumbnail
// without downloading the whole clip.
function buildHitCard(h, opts = {}) {
  const card = document.createElement("div");
  card.className = "card" + (opts.cardClass ? " " + opts.cardClass : "");
  const vsrc = "/api/video?uri=" + encodeURIComponent(h.source_media_uri);
  const badge = opts.badge
    ? `<div class="card-badge" title="matched query">${escapeHtml(opts.badge)}</div>`
    : "";
  card.innerHTML = `
      <video controls preload="metadata" playsinline src="${vsrc}#t=0.1"></video>
      <div class="card-body">
        <div class="card-top">
          <span class="rank">#${fmtInt(h.rank)}</span>
          <span class="score">${h.score.toFixed(3)}</span>
        </div>
        ${badge}
        <div class="card-meta">
          <div><span class="k">id</span> ${escapeHtml(h.chunk_id)}</div>
          ${h.segment_id ? `<div><span class="k">seg</span> ${escapeHtml(h.segment_id)}</div>` : ""}
          <div><span class="k">utc</span> ${h.start_utc}${h.end_utc ? ` → ${h.end_utc}` : ""}</div>
        </div>
        ${opts.controls || ""}
      </div>`;
  return card;
}

function renderGrid(hits) {
  const grid = $("grid");
  grid.innerHTML = "";
  hits.forEach((h, i) => {
    const m = state.marks[h.chunk_id];
    const card = buildHitCard(h, {
      controls: `
        <div class="marks">
          <button class="mark up ${m && m.mark === "up" ? "on" : ""}" title="Relevant">👍</button>
          <button class="mark down ${m && m.mark === "down" ? "on" : ""}" title="Not relevant">👎</button>
        </div>`,
    });
    card.style.animationDelay = (i * 18) + "ms";
    const [upBtn, downBtn] = card.querySelectorAll(".mark");
    upBtn.onclick = () => toggleMark(h, "up", card);
    downBtn.onclick = () => toggleMark(h, "down", card);
    grid.appendChild(card);
  });
  updateMarkCount();
}

function toggleMark(hit, kind, card) {
  const cur = state.marks[hit.chunk_id];
  if (cur && cur.mark === kind) {
    delete state.marks[hit.chunk_id]; // toggle off
  } else {
    state.marks[hit.chunk_id] = {
      mark: kind, segment_id: hit.segment_id, index: hit.index,
      rank: hit.rank, score: hit.score,
    };
  }
  const m = state.marks[hit.chunk_id];
  const [upBtn, downBtn] = card.querySelectorAll(".mark");
  upBtn.classList.toggle("on", !!m && m.mark === "up");
  downBtn.classList.toggle("on", !!m && m.mark === "down");
  updateMarkCount();
}

function updateMarkCount() {
  const vals = Object.values(state.marks);
  const up = vals.filter((m) => m.mark === "up").length;
  const down = vals.filter((m) => m.mark === "down").length;
  const el = $("mark-count");
  if (el) el.textContent = up || down ? `${up} 👍 / ${down} 👎 marked` : "";
  const rb = $("refine-btn");
  if (rb) {
    rb.disabled = up === 0;
    rb.textContent = up ? `Refine (${up} 👍 / ${down} 👎)` : "Refine (mark 👍 first)";
  }
}

function clearMarks() {
  state.marks = {};
  document.querySelectorAll(".mark.on").forEach((b) => b.classList.remove("on"));
  updateMarkCount();
}

function renderPager(data) {
  const pages = Math.ceil(data.total / data.page_size);
  const lo = data.page * data.page_size + 1;
  const hi = Math.min(data.total, (data.page + 1) * data.page_size);
  $("page-label").textContent = `${fmtInt(lo)}–${fmtInt(hi)} of ${fmtInt(data.total)} · page ${data.page + 1}/${fmtInt(pages)}`;
  $("prev").disabled = data.page <= 0;
  $("next").disabled = (data.page + 1) >= pages;
  $("pager").classList.remove("hidden");
}

// Name the download from the server's X-NLS-Export-Name (so the CSV, the saved
// parquet, and the DX segment set all share one name); fall back if absent.
function exportFilename(resp, fallback) {
  const n = resp.headers.get("X-NLS-Export-Name") || "";
  return n ? n + ".csv" : fallback;
}

// Direct top-k CSV export of the CURRENT search from the search page -- the original
// search-page Download CSV, shown where there's no offline workflow (e.g. trucking). Reuses
// /api/export (re-scores the query under the active filters) and the shared filter set.
async function exportSearchCsv() {
  if (!state.query) { setStatus("Run a search first to export its results.", true); return; }
  if (!state.embeddingsUri) { setStatus("Load a corpus first.", true); return; }
  const k = parseInt($("search-export-k").value, 10) || 100;
  const btn = $("search-export-btn");
  btn.disabled = true;
  setStatus(`Exporting top-${k} for "${state.query}"…`);
  try {
    const resp = await fetch("/api/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: state.query, k,
        dedupe_segment: $("search-export-dedup").checked,
        ..._activeFilters(),
        embeddings_uri: state.embeddingsUri,
      }),
    });
    if (!resp.ok) {
      let d; try { d = (await resp.json()).detail; } catch (_e) { /* non-JSON */ }
      throw new Error(d || ("export " + resp.status));
    }
    const parquet = resp.headers.get("X-NLS-Parquet") || "";
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = exportFilename(resp, "search_export.csv");
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    setStatus(`Downloaded top-${k} CSV` + (parquet ? ` · parquet → ${parquet}` : " · ⚠ parquet not written"), !parquet);
  } catch (e) {
    setStatus("Export failed: " + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

// Status suffix reporting the DX segment-set result, if one was requested.
function segsetNote(resp) {
  const ok = resp.headers.get("X-NLS-Segset") || "";
  const err = resp.headers.get("X-NLS-Segset-Error") || "";
  if (ok) return ` · segment set ✓ ${ok}`;
  if (err) return ` · ⚠ segment set failed: ${err}`;
  return "";
}

// Save the current (refined) search vector under a tag, for reuse on the Curate tab.
// The server reads the cached refined vector, so a search/refine must have run first.
async function saveVector() {
  const tag = ($("save-vec-tag").value || "").trim();
  const noteId = "save-vec-note";
  if (!tag) { scanNote(noteId, "Enter a tag name.", true); return; }
  if (!state.query) { scanNote(noteId, "Run a search or refine first to define a vector.", true); return; }
  const btn = $("save-vec-btn");
  btn.disabled = true;
  scanNote(noteId, "Saving…");
  try {
    const r = await fetch("/api/save_vector", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tag, query: state.query,
        // Persist the active filter set with the vector so Resume restores it exactly.
        ..._activeFilters(),
      }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
    const { dim } = await r.json();
    scanNote(noteId, `saved <strong>${escapeHtml(tag)}</strong> (${dim}-d) ✓ — pick it on the Export tab`);
  } catch (e) {
    scanNote(noteId, "Save failed: " + escapeHtml(e.message), true);
  } finally {
    btn.disabled = false;
  }
}

// Launch the per-segment Lilypad scan over the Curate page's assembled queries (arbitrary
// N). The server reuses each query's saved vector from exp-db (encoding + persisting any
// new ones), scores the FULL corpus, and writes a per-segment Lance (one column per query).
async function launchCurateScan() {
  const noteId = "curate-scan-note";
  const set = collectExportQueries();
  const tags = [...new Set(set.map((x) => x.query))].filter(Boolean);
  if (!tags.length) { scanNote(noteId, "Tick at least one saved search (or add an ad-hoc line).", true); return; }
  // Per-tag cosine threshold (no single global cutoff).
  const thresholds = {};
  set.forEach((x) => { thresholds[x.query] = x.threshold; });
  const btn = $("curate-scan-btn");
  btn.disabled = true;
  scanNote(noteId, `Launching per-segment scan over ${tags.length} tag(s)…`);
  try {
    const r = await fetch("/api/launch_segment_scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tags, thresholds, default_threshold: parseFloat($("curate-threshold").value) || 0.3,
        create_segment_set: $("curate-scan-segset").checked,
        // Full active filter set (forwarded into the workflow + persisted with the scan;
        // date applied by the worker today). Same set shown in the Scan filters summary.
        ..._activeFilters(),
      }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
    const { execution_id, url, encoded } = await r.json();
    const enc = encoded && encoded.length ? ` · encoded ${encoded.length} new` : "";
    scanNote(noteId, `launched <a href="${url}" target="_blank" rel="noopener">${execution_id}</a>${enc} · polling…`);
    loadScanJobs();
    pollScan(execution_id, url, noteId, "per-segment scan", "segments.lance");
  } catch (e) {
    scanNote(noteId, "Launch failed: " + escapeHtml(e.message), true);
  } finally {
    btn.disabled = false;
  }
}

// ---------- interval preview (visualize merged intervals before download) ----------

// Feedback line for the similarity-distribution loader — routed to the main status line
// (the Export flyout that used to host its own note was removed).
function previewNote(html, isErr) {
  setStatus(html, isErr);
}

// Similarity-score distribution on its own — works regardless of the interval
// toggle. Reuses the preview area, showing just the whole-corpus histogram
// (with the threshold the current mode/k/cutoff would produce).
async function loadScoreDistribution() {
  if (!state.query) {
    previewNote("Run a search first to compute a distribution.", true);
    setStatus("Run a search first to compute a distribution.", true);
    return;
  }
  const body = {
    query: state.query,
    k: 2000,
    from_date: $("from-date").value || null,
    to_date: $("to-date").value || null,
    segment_set_uuid: state.segUuid,
    filter_lance_uri: state.filterLanceUri || null,
    vehicle: _vehicleValue(),
    drive_id: _driveValue(),
    embeddings_uri: state.embeddingsUri,
    interval_mode: "k",
    interval_score: null,
  };
  previewNote("Computing similarity distribution…");
  setStatus("Computing similarity distribution…");
  try {
    const dist = await fetch("/api/score_distribution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
    showDistribution(dist);
    $("score-dist-view").scrollIntoView({ behavior: "smooth", block: "start" });
    const tnote = dist.tau != null ? ` · τ ${dist.tau.toFixed(3)}` : "";
    previewNote(`distribution over ${fmtInt(dist.total)} clips${tnote} — scrolled below ↓`);
    setStatus(`Similarity distribution · ${fmtInt(dist.total)} clips${tnote}`);
  } catch (e) {
    previewNote("Distribution failed: " + escapeHtml(e.message), true);
    setStatus("Distribution failed: " + e.message, true);
  }
}

// ---------- Lilypad workload status (shared note + poller) ----------
function scanNote(noteId, html, isErr) {
  const n = $(noteId);
  n.innerHTML = html;
  n.classList.toggle("warn", !!isErr);
}

// Poll a launched Lilypad workload's phase every 20s (bounded); the workload runs
// regardless of the tab, and the console link stays clickable if the user navigates away.
async function pollScan(execId, url, noteId, what, artifact) {
  const link = `<a href="${url}" target="_blank" rel="noopener">${execId}</a>`;
  for (let i = 0; i < 90; i++) {
    await new Promise((res) => setTimeout(res, 20000));
    let s;
    try {
      s = await fetch("/api/scan_status?execution=" + encodeURIComponent(execId)).then((r) => r.json());
    } catch (e) { continue; }
    if (s.done) {
      if (s.phase === "SUCCEEDED") {
        scanNote(noteId, `${what} ${link} ✓ ${s.phase} — ${artifact} written under the scan output prefix`);
      } else {
        scanNote(noteId, `${what} ${link} · ${escapeHtml(s.phase)}${s.error ? " — " + escapeHtml(s.error.slice(0, 200)) : ""}`, true);
      }
      return;
    }
    scanNote(noteId, `${what} ${link} · ${escapeHtml(s.phase || "running")}…`);
  }
  scanNote(noteId, `${what} ${link} · still running — check the console link.`);
}

// Render (or clear) the similarity histogram in the top-of-search section.
function showDistribution(dist) {
  const t = $("score-dist-view");
  t.innerHTML = "";
  if (!dist) { t.classList.add("hidden"); return; }
  t.appendChild(buildDistribution(dist));
  t.classList.remove("hidden");
}

// Similarity-score histogram across the ENTIRE corpus for the current query.
// X axis = similarity-score bins; Y axis = percentage of clips per bin. The
// threshold tau is drawn as a DRAGGABLE vertical line (also click-to-jump):
// moving it sets the interval score-cutoff live, so it's an interactive picker.
function buildDistribution(dist) {
  const wrap = document.createElement("div");
  wrap.className = "score-dist";
  const W = 720, H = 220, mL = 46, mR = 12, mT = 14, mB = 34;
  const plotW = W - mL - mR, plotH = H - mT - mB;
  const counts = dist.counts, edges = dist.edges, total = dist.total || 1;
  const nb = counts.length;
  const pcts = counts.map((c) => 100 * c / total);
  const maxPct = Math.max(0.001, ...pcts);
  const lo = edges[0], hi = edges[edges.length - 1];
  const span = (hi - lo) || 1;
  const X = (s) => mL + ((s - lo) / span) * plotW;
  const Y = (p) => mT + (1 - p / maxPct) * plotH;
  const bw = plotW / nb;
  const bars = pcts.map((p, i) => {
    const x = mL + i * bw, y = Y(p);
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" `
      + `width="${Math.max(0.5, bw - 0.6).toFixed(1)}" height="${(mT + plotH - y).toFixed(1)}" class="sd-bar" />`;
  }).join("");
  const yTicks = [0, maxPct / 2, maxPct].map((p) => {
    const y = Y(p).toFixed(1);
    return `<line x1="${mL}" y1="${y}" x2="${W - mR}" y2="${y}" class="sd-grid" />`
      + `<text x="${mL - 6}" y="${(Y(p) + 3).toFixed(1)}" class="sd-ytick">${p.toFixed(1)}%</text>`;
  }).join("");
  const NX = 7;
  const xTicks = Array.from({ length: NX + 1 }, (_, i) => {
    const s = lo + (span * i) / NX, x = X(s).toFixed(1);
    return `<line x1="${x}" y1="${mT + plotH}" x2="${x}" y2="${mT + plotH + 4}" class="sd-grid" />`
      + `<text x="${x}" y="${H - mB + 16}" class="sd-xtick">${s.toFixed(2)}</text>`;
  }).join("");
  const axis = `<line x1="${mL}" y1="${mT + plotH}" x2="${W - mR}" y2="${mT + plotH}" class="sd-axis" />`
    + `<line x1="${mL}" y1="${mT}" x2="${mL}" y2="${mT + plotH}" class="sd-axis" />`;
  const initTau = (dist.tau != null) ? dist.tau : dist.mean;
  const tx = X(initTau).toFixed(1);
  wrap.innerHTML = `
    <div class="interval-head">
      <h3>Similarity distribution</h3>
      <span class="sub">${fmtInt(dist.total)} clips · entire corpus · min ${dist.min.toFixed(3)}
        · mean ${dist.mean.toFixed(3)} · max ${dist.max.toFixed(3)}</span>
    </div>
    <svg class="sd-spark" viewBox="0 0 ${W} ${H}" role="img"
        aria-label="similarity distribution: x = similarity, y = percent of clips; drag the line to set the threshold">
      ${yTicks}
      ${bars}
      ${axis}
      ${xTicks}
      <line class="sd-thresh" x1="${tx}" y1="${mT}" x2="${tx}" y2="${mT + plotH}" />
      <line class="sd-hit" x1="${tx}" y1="${mT}" x2="${tx}" y2="${mT + plotH}" />
      <path class="sd-grip" d="M ${tx} ${mT} m -5 -1 l 10 0 l -5 7 z" />
      <text class="sd-taulbl" x="${(+tx + 7).toFixed(1)}" y="${mT + 9}">τ ${initTau.toFixed(3)}</text>
      <text x="${(mL + plotW / 2).toFixed(1)}" y="${H - 2}" class="sd-axlbl">similarity score</text>
    </svg>
    <div class="sub sd-readout"></div>`;

  // --- interactivity: drag / click the threshold line ---------------------
  const svg = wrap.querySelector("svg.sd-spark");
  const line = wrap.querySelector(".sd-thresh");
  const hit = wrap.querySelector(".sd-hit");
  const grip = wrap.querySelector(".sd-grip");
  const label = wrap.querySelector(".sd-taulbl");
  const readout = wrap.querySelector(".sd-readout");
  let curTau = initTau;

  // Clips at/above v, interpolated within partial bins (smooth live count).
  const clipsAtOrAbove = (v) => {
    let c = 0;
    for (let i = 0; i < nb; i++) {
      const e0 = edges[i], e1 = edges[i + 1];
      if (e1 <= v) continue;
      if (e0 >= v) { c += counts[i]; continue; }
      c += counts[i] * (e1 - v) / (e1 - e0);
    }
    return Math.round(c);
  };
  const setTau = (v) => {
    curTau = Math.max(lo, Math.min(hi, v));
    const x = X(curTau);
    line.setAttribute("x1", x); line.setAttribute("x2", x);
    hit.setAttribute("x1", x); hit.setAttribute("x2", x);
    grip.setAttribute("d", `M ${x} ${mT} m -5 -1 l 10 0 l -5 7 z`);
    label.setAttribute("x", x + 7); label.textContent = `τ ${curTau.toFixed(3)}`;
    const above = clipsAtOrAbove(curTau);
    const pct = total ? 100 * above / total : 0;
    readout.innerHTML = `threshold τ = <b>${curTau.toFixed(3)}</b> · ${fmtInt(above)} clips `
      + `(${pct.toFixed(2)}%) at or above τ · <span class="sd-hint">drag the line to adjust</span>`;
  };
  // Pointer x (client) -> similarity value (uniform scale: viewBox W fills width).
  const valueAt = (clientX) => {
    const r = svg.getBoundingClientRect();
    const ux = (clientX - r.left) / r.width * W;
    return lo + ((ux - mL) / plotW) * span;
  };
  let dragging = false;
  const onDown = (e) => {
    dragging = true;
    svg.setPointerCapture(e.pointerId);
    setTau(valueAt(e.clientX));
    e.preventDefault();
  };
  const onMove = (e) => { if (dragging) setTau(valueAt(e.clientX)); };
  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    // Informational only: the histogram shows where a threshold lands. Use the value
    // as the "scan threshold" on the Curate tab when launching a per-segment scan.
    readout.innerHTML = `threshold τ = <b>${curTau.toFixed(3)}</b> · `
      + `${fmtInt(clipsAtOrAbove(curTau))} clips at or above · `
      + `<span class="sd-hint">use as the Curate scan threshold</span>`;
  };
  svg.addEventListener("pointerdown", onDown);
  svg.addEventListener("pointermove", onMove);
  svg.addEventListener("pointerup", onUp);
  svg.addEventListener("pointercancel", onUp);
  setTau(curTau);  // initial readout
  return wrap;
}

// ---------- curate from config (preview → select → export) ----------
const CURATE_CAP = 60; // cards rendered in the selection grid before "show all"

// Parse a `query, k, threshold` textarea. Trailing comma-separated numbers are read
// right-to-left: a float in (0,1] is the per-tag threshold, an integer is k. Both are
// optional; missing values fall back to the default-k / default-threshold inputs.
function parseConfigQueries(queriesId, defKId, defThreshId) {
  const defK = parseInt($(defKId).value || "50", 10) || 50;
  const defT = defThreshId ? (parseFloat($(defThreshId).value) || 0.3) : 0.3;
  return $(queriesId).value
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(",").map((p) => p.trim());
      let query = parts[0];
      let k = defK, threshold = defT;
      for (const tok of parts.slice(1)) {
        if (!tok) continue;
        const n = Number(tok);
        if (Number.isNaN(n)) { query += ", " + tok; continue; } // comma inside the query text
        if (Number.isInteger(n) && n > 1) k = n;
        else if (n > 0 && n <= 1) threshold = n;
      }
      return { query: query.replace(/,+$/, "").trim(), k, threshold };
    })
    .filter((x) => x.query);
}

function curateNote(msg, isError) {
  const el = $("curate-note");
  el.textContent = msg;
  el.classList.toggle("error", !!isError);
}

// ---------- export history picker (select prior searches, set per-row k) ----------

// Short, readable label for a corpus/model uri (last non-empty path segment).
function _uriShort(uri) {
  const parts = String(uri || "").replace(/\/+$/, "").split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "";
}

// Fetch the deduped search history and render the selectable table.
async function loadExportHistory() {
  const wrap = $("export-history");
  wrap.innerHTML = '<p class="note">loading history…</p>';
  try {
    const data = await fetch("/api/tags_catalog").then((r) => r.json());
    state.exportHistory = data.entries || [];
  } catch (e) {
    state.exportHistory = [];
    wrap.innerHTML = '<p class="note error">could not load history.</p>';
    return;
  }
  renderExportHistory();
}

// Compact one-line summary of the filter set saved with a vector (date / segment set /
// lance / vehicle / drive). Returns "" when no filters were captured so the cell shows "—".
function _fmtFilters(f) {
  if (!f) return "";
  const parts = [];
  const from = (f.from_date || "").trim();
  const to = (f.to_date || "").trim();
  if (from || to) parts.push(`${from || "…"}→${to || "latest"}`);
  const seg = (f.segment_set_name || f.segment_set_uuid || "").trim();
  if (seg) parts.push("seg: " + seg);
  if ((f.vehicle || "").trim()) parts.push("veh: " + f.vehicle.trim());
  if ((f.drive_id || "").trim()) parts.push("drive: " + f.drive_id.trim());
  if ((f.filter_lance_uri || "").trim()) parts.push("lance: " + _uriShort(f.filter_lance_uri));
  return parts.join(" · ");
}

function renderExportHistory() {
  const wrap = $("export-history");
  const rows = state.exportHistory || [];
  if (!rows.length) {
    wrap.innerHTML = '<p class="note">No saved searches yet — run a search and Save vector, or download a CSV, and it shows up here.</p>';
    return;
  }
  const defK = parseInt($("curate-k").value || "50", 10) || 50;
  const defT = parseFloat($("curate-threshold").value) || 0.3;
  const body = rows
    .map((e, i) => {
      // Global friendly model name (server-provided via _MODEL_LABELS, e.g. white-dwarf).
      const model = e.model_label || _uriShort(e.model_uri) || "base";
      const dim = e.vec_dim ? `${e.vec_dim}-d` : "no vec";
      // Show the query only when it differs from the tag (the tag is the key + the
      // Lance column name; the query is just its human description).
      const desc = e.query && e.query !== e.tag ? e.query : "";
      const openBtn = e.id != null
        ? `<button type="button" class="ghost exp-resume" data-id="${e.id}" title="Resume this tag's saved search in Search to iterate on it">resume ↗</button>`
        : "";
      // Filters saved with the vector (date/segment set/lance/vehicle/drive) — the same set
      // the scan is launched with when this tag is resumed.
      const filt = _fmtFilters(e.filters);
      return `<tr data-i="${i}" data-tag="${escapeHtml(e.tag)}">
        <td class="exp-c"><input type="checkbox" class="exp-pick" /></td>
        <td class="exp-q" title="${escapeHtml(e.tag)}">${escapeHtml(e.tag)}</td>
        <td class="exp-tag muted" title="${escapeHtml(e.query)}">${escapeHtml(desc)}</td>
        <td class="exp-meta muted" title="${escapeHtml(e.model_uri || "base model")}">${escapeHtml(model)} · ${dim}</td>
        <td class="exp-filters muted" title="${escapeHtml(filt || "no filters saved")}">${filt ? escapeHtml(filt) : "—"}</td>
        <td class="exp-kc"><input type="number" class="exp-k" min="1" value="${e.k || defK}" /></td>
        <td class="exp-tc requires-offline-scan"><input type="number" class="exp-t" step="0.01" min="0" max="1" value="${e.threshold > 0 ? e.threshold : defT}" title="per-tag cosine threshold for the scan (saved from the last launch)" /></td>
        <td class="exp-open-c">${openBtn}</td>
      </tr>`;
    })
    .join("");
  wrap.innerHTML = `<table class="exp-hist-table">
      <thead><tr><th></th><th>tag</th><th>query</th><th>corpus model · vec</th><th>filters</th><th>k</th><th class="requires-offline-scan">thresh</th><th></th></tr></thead>
      <tbody>${body}</tbody></table>`;
  filterExportHistory();
}

// Show only rows whose query or tag matches the filter box (case-insensitive substring).
function filterExportHistory() {
  const q = ($("export-hist-filter").value || "").trim().toLowerCase();
  $("export-history").querySelectorAll("tbody tr").forEach((tr) => {
    const hay = (tr.querySelector(".exp-q").textContent + " " + tr.querySelector(".exp-tag").textContent).toLowerCase();
    tr.style.display = !q || hay.includes(q) ? "" : "none";
  });
}

// Tick/untick every currently-visible history row.
function exportHistSelectAll(on) {
  $("export-history").querySelectorAll("tbody tr").forEach((tr) => {
    if (tr.style.display === "none") return;
    const cb = tr.querySelector(".exp-pick");
    if (cb) cb.checked = on;
  });
}

// The assembled working set for preview / download / scan: ticked history rows (each
// with its inline k + per-tag threshold) merged with the optional ad-hoc textarea
// lines, deduped by query (case-insensitive; first occurrence — a ticked row — wins).
function collectExportQueries() {
  const out = [];
  const seen = new Set();
  const defT = parseFloat($("curate-threshold").value) || 0.3;
  const push = (query, k, threshold) => {
    const qq = (query || "").trim().replace(/,+$/, "");
    if (!qq) return;
    const key = qq.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    out.push({
      query: qq,
      k: parseInt(k, 10) || 50,
      threshold: Number(threshold) > 0 ? Number(threshold) : defT,
    });
  };
  // History rows are keyed by TAG (the DB primary key + the scan's Lance column name).
  $("export-history").querySelectorAll("tbody tr").forEach((tr) => {
    const cb = tr.querySelector(".exp-pick");
    if (cb && cb.checked) {
      push(tr.dataset.tag, tr.querySelector(".exp-k").value, tr.querySelector(".exp-t").value);
    }
  });
  parseConfigQueries("curate-queries", "curate-k", "curate-threshold").forEach((x) =>
    push(x.query, x.k, x.threshold));
  return out;
}

// ---------- recent scans panel (durable launched-workload list) ----------

const SCAN_TERMINAL = /SUCCEEDED|COMPLETED|FAILED|STOPPED/i;
let _scanPollTimer = null;

function _scanBadge(s) {
  const t = (s || "").toUpperCase();
  let cls = "running";
  if (/SUCCEEDED|COMPLETED/.test(t)) cls = "ok";
  else if (/FAILED|STOPPED/.test(t)) cls = "err";
  return `<span class="scan-badge ${cls}">${escapeHtml(s || "—")}</span>`;
}

async function loadScanJobs() {
  const wrap = $("scan-jobs");
  try {
    const data = await fetch("/api/scans").then((r) => r.json());
    state.scanJobs = data.jobs || [];
  } catch (e) {
    wrap.innerHTML = '<p class="note error">could not load scans.</p>';
    return;
  }
  renderScanJobs();
  scheduleScanJobsPoll();
}

function renderScanJobs() {
  const wrap = $("scan-jobs");
  const jobs = state.scanJobs || [];
  if (!jobs.length) { wrap.innerHTML = '<p class="note">No scans launched yet.</p>'; return; }
  const rows = jobs
    .map((j) => {
      const id = j.console_url
        ? `<a href="${j.console_url}" target="_blank" rel="noopener">${escapeHtml(j.execution_id)}</a>`
        : escapeHtml(j.execution_id);
      const tagList = (j.tags || [])
        .map((t) => escapeHtml(t) + (j.thresholds && j.thresholds[t] != null ? ` @${j.thresholds[t]}` : ""))
        .join(", ");
      const err = j.error ? ` <span class="muted">${escapeHtml(j.error.slice(0, 120))}</span>` : "";
      const out = j.lance_uri
        ? `<code title="${escapeHtml(j.lance_uri)}">${escapeHtml(j.lance_uri)}</code>`
        : '<span class="muted">—</span>';
      // Data Explorer cell: the registered DORA segment set (uuid + name), or pending /
      // a dash when registration wasn't requested.
      let dx = '<span class="muted">—</span>';
      if (j.segset_label) dx = `<code title="${escapeHtml(j.segset_label)}">${escapeHtml(j.segset_label)}</code>`;
      else if (j.register_segset) dx = '<span class="muted">pending…</span>';
      // Filters the scan was launched with (date/segment-set/lance/vehicle/drive).
      const filt = _fmtFilters(j.filters);
      return `<tr><td class="sj-when muted">${escapeHtml(j.created_at)}</td>
        <td class="sj-id">${id}</td>
        <td class="sj-tags" title="${tagList}">${(j.tags || []).length} tag(s): ${tagList}</td>
        <td class="sj-filters muted" title="${escapeHtml(filt || "no filters")}">${filt ? escapeHtml(filt) : "—"}</td>
        <td class="sj-status">${_scanBadge(j.status)}${err}</td>
        <td class="sj-out">${out}</td>
        <td class="sj-dx">${dx}</td></tr>`;
    })
    .join("");
  wrap.innerHTML = `<table class="scan-jobs-table">
      <thead><tr><th>launched (UTC)</th><th>workload</th><th>tags (@threshold)</th><th>filters</th><th>status</th><th>output (segments.lance)</th><th>Data Explorer (segment set)</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
}

// While there are non-terminal scans, re-poll their status every 20s (the endpoint
// also refreshes the stored status) and re-render in place. One timer at a time.
function scheduleScanJobsPoll() {
  if (_scanPollTimer) return;
  const tick = async () => {
    _scanPollTimer = null;
    const pending = (state.scanJobs || []).filter((j) => !SCAN_TERMINAL.test(j.status || ""));
    if (!pending.length) return;
    await Promise.all(
      pending.map(async (j) => {
        try {
          const s = await fetch("/api/scan_status?execution=" + encodeURIComponent(j.execution_id)).then((r) => r.json());
          if (s.phase) j.status = s.phase;
          if (s.error) j.error = s.error;
          // On completion the status call may register + return the DORA segment set.
          if (s.segset_label) j.segset_label = s.segset_label;
          if (s.segset_uuid) j.segset_uuid = s.segset_uuid;
        } catch (e) { /* transient; retry next tick */ }
      }),
    );
    renderScanJobs();
    _scanPollTimer = setTimeout(tick, 20000);
  };
  _scanPollTimer = setTimeout(tick, 20000);
}

async function loadCuratePreview() {
  const queries = collectExportQueries();
  if (!queries.length) { curateNote("Tick at least one saved search (or add an ad-hoc line).", true); return; }
  if (!state.embeddingsUri) { curateNote("Load a corpus first.", true); return; }
  curateNote(`Running ${queries.length} queries…`);
  $("curate-load").disabled = true;
  try {
    const body = {
      queries,
      from_date: $("from-date").value || null,
      to_date: $("to-date").value || null,
      segment_set_uuid: state.segUuid,
      segment_set_name: state.segName,
      filter_lance_uri: state.filterLanceUri || null,
      vehicle: _vehicleValue(),
      embeddings_uri: state.embeddingsUri,
    };
    const resp = await fetch("/api/curate_preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      let d; try { d = (await resp.json()).detail; } catch (_e) { /* non-JSON */ }
      throw new Error(d || ("preview " + resp.status));
    }
    const data = await resp.json();
    state.curate = { perQuery: data.per_query || [], selected: new Set(), rows: [], showAll: false };
    const total = state.curate.perQuery.reduce((s, g) => s + g.num_hits, 0);
    curateNote(`Loaded ${state.curate.perQuery.length} queries · ${fmtInt(total)} matches. Select what to keep, then Export.`);
    $("curate-toolbar").classList.remove("hidden");
    renderCurate(true);
  } catch (e) {
    curateNote("Preview failed: " + e.message, true);
  } finally {
    $("curate-load").disabled = false;
  }
}

// Direct local export: each assembled tag's top-k from the LOADED (resident) corpus,
// concatenated into one CSV (+ parquet to S3). No preview/selection needed -- this is
// the "download CSV" for the small corpus. (The toolbar's "Download selected" exports a
// hand-picked subset after Load preview.)
async function downloadCsv() {
  const set = collectExportQueries();
  if (!set.length) { curateNote("Tick at least one saved search (or add an ad-hoc line).", true); return; }
  if (!state.embeddingsUri) { curateNote("Load a corpus first.", true); return; }
  const queries = set.map((x) => ({ query: x.query, k: x.k }));
  curateNote(`Exporting top-k for ${queries.length} tag(s) from the loaded corpus…`);
  $("curate-csv-btn").disabled = true;
  try {
    const resp = await fetch("/api/export_config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        queries, dedupe: true,
        dedupe_segment: $("curate-csv-dedup").checked,
        from_date: $("from-date").value || null,
        to_date: $("to-date").value || null,
        segment_set_uuid: state.segUuid,
        segment_set_name: state.segName,
        filter_lance_uri: state.filterLanceUri || null,
        vehicle: _vehicleValue(),
        embeddings_uri: state.embeddingsUri,
      }),
    });
    if (!resp.ok) {
      let d; try { d = (await resp.json()).detail; } catch (_e) { /* non-JSON */ }
      throw new Error(d || ("export " + resp.status));
    }
    const parquet = resp.headers.get("X-NLS-Parquet") || "";
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = exportFilename(resp, "config_export.csv");
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    curateNote(`Downloaded CSV for ${queries.length} tag(s)` + (parquet ? ` · parquet → ${parquet}` : " · ⚠ parquet not written"), !parquet);
  } catch (e) {
    curateNote("Download failed: " + e.message, true);
  } finally {
    $("curate-csv-btn").disabled = false;
  }
}

// Compute the rows to display from the per-query preview, honoring the inline
// dedupe toggle. Dedupe collapses a chunk that surfaced in multiple queries to its
// highest-scoring occurrence (recording every query it matched). Each row carries a
// stable `key` for the selection set.
function curateDisplayRows(dedupe) {
  const pq = (state.curate && state.curate.perQuery) || [];
  if (dedupe) {
    const byChunk = new Map();
    for (const g of pq) {
      for (const h of g.hits) {
        const e = byChunk.get(h.chunk_id);
        if (!e) byChunk.set(h.chunk_id, { h, query: g.query, matched: new Set([g.query]) });
        else { e.matched.add(g.query); if (h.score > e.h.score) { e.h = h; e.query = g.query; } }
      }
    }
    return [...byChunk.values()]
      .sort((a, b) => b.h.score - a.h.score)
      .map((e) => ({ h: e.h, query: e.query, matched: [...e.matched], key: e.h.chunk_id }));
  }
  const rows = [];
  for (const g of pq) {
    for (const h of g.hits) {
      rows.push({ h, query: g.query, matched: [g.query], key: g.query + "\u0000" + h.chunk_id });
    }
  }
  return rows.sort((a, b) => b.h.score - a.h.score);
}

function renderCurate(resetSelection) {
  const dedupe = $("curate-dedupe").checked;
  const rows = curateDisplayRows(dedupe);
  state.curate.rows = rows;
  if (resetSelection) state.curate.selected = new Set(rows.map((r) => r.key));

  const wrap = $("curate-results");
  wrap.innerHTML = "";

  // --- Selection grid (the export set): selectable cards, capped with "show all". ---
  const selHead = document.createElement("div");
  selHead.className = "curate-section-head";
  selHead.textContent = "Selection (what will be exported)";
  wrap.appendChild(selHead);

  const grid = document.createElement("div");
  grid.className = "grid curate-grid";
  const shown = state.curate.showAll ? rows : rows.slice(0, CURATE_CAP);
  shown.forEach((r) => grid.appendChild(curateCard(r, dedupe)));
  wrap.appendChild(grid);

  if (!state.curate.showAll && rows.length > CURATE_CAP) {
    const more = document.createElement("button");
    more.className = "ghost curate-more";
    more.textContent = `Show all ${fmtInt(rows.length)} cards`;
    more.onclick = () => { state.curate.showAll = true; renderCurate(false); };
    wrap.appendChild(more);
  }

  // --- Per-query groups (which segments matched which query): compact, collapsed. ---
  const byHead = document.createElement("div");
  byHead.className = "curate-section-head";
  byHead.textContent = "Matches by query";
  wrap.appendChild(byHead);
  for (const g of state.curate.perQuery) {
    const det = document.createElement("details");
    det.className = "curate-group";
    const rowsHtml = g.hits
      .map((h) => `<li><span class="score">${h.score.toFixed(3)}</span>
        <span class="seg">${escapeHtml(h.segment_id || h.chunk_id)}</span>
        <a href="/api/video?uri=${encodeURIComponent(h.source_media_uri)}" target="_blank" rel="noopener">play</a></li>`)
      .join("");
    det.innerHTML = `<summary>${escapeHtml(g.query)} <span class="muted">· ${fmtInt(g.num_hits)} matches</span></summary>
      <ul class="curate-group-list">${rowsHtml}</ul>`;
    wrap.appendChild(det);
  }

  curateUpdateCounts();
}

function curateCard(r, dedupe) {
  const h = r.h;
  const checked = state.curate.selected.has(r.key);
  const badge = dedupe && r.matched.length > 1
    ? `${r.query}  +${r.matched.length - 1}`
    : r.query;
  const card = buildHitCard(h, {
    cardClass: "curate-card" + (checked ? "" : " unpicked"),
    badge,
    controls: `<label class="curate-keep"><input type="checkbox" ${checked ? "checked" : ""}/> keep</label>`,
  });
  const cb = card.querySelector(".curate-keep input");
  cb.onchange = () => {
    if (cb.checked) state.curate.selected.add(r.key);
    else state.curate.selected.delete(r.key);
    card.classList.toggle("unpicked", !cb.checked);
    curateUpdateCounts();
  };
  return card;
}

function curateUpdateCounts() {
  const rows = (state.curate && state.curate.rows) || [];
  const sel = state.curate.selected;
  const kept = rows.filter((r) => sel.has(r.key));
  const segs = new Set(kept.map((r) => r.h.segment_id).filter(Boolean));
  $("curate-counts").textContent =
    `${fmtInt(rows.length)} matches · ${fmtInt(kept.length)} selected · ${fmtInt(segs.size)} segments`;
}

function curateSelectAll(on) {
  if (!state.curate) return;
  state.curate.selected = on ? new Set(state.curate.rows.map((r) => r.key)) : new Set();
  // Update checkboxes + card state in place (avoid reloading every video).
  $("curate-results").querySelectorAll(".curate-card").forEach((card) => {
    const cb = card.querySelector(".curate-keep input");
    if (cb) { cb.checked = on; card.classList.toggle("unpicked", !on); }
  });
  curateUpdateCounts();
}

async function curateExport() {
  if (!state.curate || !state.curate.rows.length) { curateNote("Load a preview first.", true); return; }
  const sel = state.curate.selected;
  const kept = state.curate.rows.filter((r) => sel.has(r.key));
  if (!kept.length) { curateNote("Select at least one segment to export.", true); return; }
  const rows = kept.map((r) => ({
    query: r.query,
    rank: r.h.rank,
    score: r.h.score,
    segment_id: r.h.segment_id,
    chunk_id: r.h.chunk_id,
    run_uuid: r.h.run_uuid,
    start_timestamp_ns: r.h.start_timestamp_ns,
    end_timestamp_ns: r.h.end_timestamp_ns,
    source_media_uri: r.h.source_media_uri,
  }));
  curateNote(`Exporting ${fmtInt(rows.length)} rows…`);
  $("curate-export-btn").disabled = true;
  try {
    const resp = await fetch("/api/curate_export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        rows,
        create_segment_set: $("curate-segset-toggle").checked,
        embeddings_uri: state.embeddingsUri,
        segment_set_uuid: state.segUuid,
        filter_lance_uri: state.filterLanceUri || null,
        from_date: $("from-date").value || null,
        to_date: $("to-date").value || null,
      }),
    });
    if (!resp.ok) {
      let d; try { d = (await resp.json()).detail; } catch (_e) { /* non-JSON */ }
      throw new Error(d || ("export " + resp.status));
    }
    const parquet = resp.headers.get("X-NLS-Parquet") || "";
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = exportFilename(resp, "curate_export.csv");
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    const segs = new Set(kept.map((r) => r.h.segment_id).filter(Boolean));
    curateNote(
      `Exported ${fmtInt(rows.length)} rows · ${fmtInt(segs.size)} segments` +
        (parquet ? ` · parquet → ${parquet}` : " · ⚠ parquet not written") +
        segsetNote(resp),
      !parquet,
    );
  } catch (e) {
    curateNote("Export failed: " + e.message, true);
  } finally {
    $("curate-export-btn").disabled = false;
  }
}

function setStatus(msg, isError) {
  const s = $("status");
  s.textContent = msg;
  s.classList.toggle("error", !!isError);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

init();
