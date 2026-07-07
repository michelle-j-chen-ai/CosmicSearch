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
// Monotonic stamp for in-flight rankings; _issue drops superseded responses so
// rapid live auto-refines can't clobber each other out of order.
let _issueSeq = 0;

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
  renderExportPanel();
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
  $("video-search-toggle").addEventListener("click", () => {
    const panel = $("video-search-panel");
    const open = panel.classList.toggle("hidden") === false;
    $("video-search-toggle").setAttribute("aria-expanded", String(open));
    $("video-search-toggle").textContent = (open ? "▼" : "▶") + " Search by video clip";
  });
  $("vs-search-btn").addEventListener("click", () => runWindowSearch({ page: 0 }));
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
  $("clear-marks").addEventListener("click", clearMarks);
  // Changing the avoid-👎 strength re-ranks immediately if there are 👍 marks.
  $("avoid-neg").addEventListener("change", () => scheduleAutoRefine(0));
  $("dist-btn").addEventListener("click", loadScoreDistribution);
  $("threshold-btn").addEventListener("click", () => loadThresholdSearch());
  $("save-vec-btn").addEventListener("click", saveVector);
  $("search-export-btn").addEventListener("click", exportSearchCsv);
  // Saved-searches page: history picker + a SINGLE Export action that adapts to the
  // deployment (instant CSV over the resident corpus, or an offline per-segment scan).
  $("export-btn").addEventListener("click", doExport);
  $("export-hist-reload").addEventListener("click", loadExportHistory);
  $("export-hist-all").addEventListener("click", () => { exportHistSelectAll(true); renderExportPanel(); });
  $("export-hist-none").addEventListener("click", () => { exportHistSelectAll(false); renderExportPanel(); });
  $("export-hist-filter").addEventListener("input", filterExportHistory);
  // Per-tag "resume ↗" + selection-count refresh when a row is ticked.
  $("export-history").addEventListener("click", (e) => {
    const b = e.target.closest("button.exp-resume");
    if (b && b.dataset.id) resumeSession(b.dataset.id);
  });
  $("export-history").addEventListener("change", (e) => {
    if (e.target.classList && e.target.classList.contains("exp-pick")) renderExportPanel();
  });

  // ⚙ corpus/model settings popover.
  $("settings-gear").addEventListener("click", toggleSettings);
  // Guided-rail Step 3 -> Save drawer; overlay / × close it.
  $("save-open-btn").addEventListener("click", openSaveDrawer);
  $("drawer-close").addEventListener("click", closeSaveDrawer);
  $("drawer-overlay").addEventListener("click", closeSaveDrawer);
  // Single-export panel: cutoff (τ / top-k) + sample (interval / segment) selectors.
  $("cutoff-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]"); if (b) setCutoffMode(b.dataset.mode);
  });
  $("sample-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]"); if (b) setSampleMode(b.dataset.mode);
  });
  // Keep the (hidden) scan register flag in sync with the single visible toggle.
  $("curate-csv-segset").addEventListener("change", () => {
    $("curate-scan-segset").checked = $("curate-csv-segset").checked;
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
  const rs = $("refine-status"); if (rs) rs.textContent = "refining…";
  await _issue("/api/refine", {
    query: state.query,
    marks,
    // avoid-👎 preset -> negative_weight; text query stays blended at a fixed weight
    // to anchor the refinement (classic Rocchio alpha*q0), no separate control.
    negative_weight: parseFloat($("avoid-neg").value),
    text_weight: 0.3,
    ...(startOpts || {}),
    ..._filterBody(),
  }, "refine");
}

// Debounced live refine: as the user marks clips, re-rank in place (no button).
// Coalesces rapid multi-marking into one call. Only in the normal results view —
// in the Tune-threshold dedicated view marks feed the fit, and the grid is hidden.
let _refineTimer = null;
function scheduleAutoRefine(delay = 450) {
  const sv = $("search-view");
  if (!sv || sv.classList.contains("threshold-active") || !state.query) return;
  clearTimeout(_refineTimer);
  _refineTimer = setTimeout(() => {
    const up = Object.values(state.marks).filter((m) => m.mark === "up").length;
    if (up > 0) {
      runRefine({ page: 0 });
    } else if (state.mode === "refine") {
      // Last 👍 removed -> fall back to the plain text ranking.
      state.mode = "search";
      runSearch({ page: 0 });
    }
  }, delay);
}

// Paging / jumps / filter changes preserve the current mode.
function reload(startOpts) {
  if (state.mode === "refine") return runRefine(startOpts);
  if (state.mode === "resume") return runVectorSearch(startOpts);
  if (state.mode === "window") return runWindowSearch(startOpts);
  return runSearch(startOpts);
}

// ---------- search by video clip (query-by-example over the corpus) ----------
// Accept a unix timestamp typed as seconds or nanoseconds; normalize to ns
// (the API divides back to seconds). Blank -> 0 (that side of the window open).
function _toNs(raw) {
  const v = (raw || "").trim().replace(/[_,\s]/g, "");
  if (!v) return 0;
  const n = Number(v);
  if (!isFinite(n) || n <= 0) return 0;
  // < 1e12 is implausible as ns (would be < year 2001), so treat it as seconds.
  return n < 1e12 ? Math.round(n * 1e9) : Math.round(n);
}

async function runWindowSearch(startOpts) {
  // Paging/filter reloads reuse the window captured on the first run.
  if (startOpts && startOpts.page === 0 || !state.windowReq) {
    const run_uuid = $("vs-run-uuid").value.trim();
    const segment_id = $("vs-segment-id").value.trim();
    if (!run_uuid && !segment_id) {
      setStatus("Enter a drive (run_uuid) or a segment id to search by clip.", true);
      return;
    }
    const key = run_uuid || segment_id;
    state.windowReq = {
      run_uuid, segment_id,
      start_ns: _toNs($("vs-start").value),
      end_ns: _toNs($("vs-end").value),
      query: "video clip: " + key,
    };
  }
  state.mode = "window";
  state.query = state.windowReq.query;
  await _issue("/api/search_by_window", {
    ...state.windowReq,
    ...(startOpts || {}),
    ..._filterBody(),
  }, "search");
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
  // Race guard: live auto-refine can fire several requests in quick succession;
  // stamp each and drop any response that a newer request has superseded, so a
  // slow earlier ranking can't clobber the latest one.
  const seq = ++_issueSeq;
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
    if (seq === _issueSeq) setStatus((mode === "refine" ? "Refine" : "Search") + " failed: " + e.message, true);
    if (seq === _issueSeq) { const rs = $("refine-status"); if (rs) rs.textContent = ""; }
    return;
  }
  if (seq !== _issueSeq) return;  // a newer request superseded this one — drop it
  const rs = $("refine-status"); if (rs) rs.textContent = "";
  state.total = data.total;
  state.page = data.page;
  state.label = data.label;
  renderFilterBar(data);
  renderLanceNote(data);
  $("refinebar").classList.remove("hidden");
  // Any fresh search/refine returns to the results view (leaves threshold mode).
  exitThresholdView();
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
  renderQueryStrip(data);
  renderGrid(data.hits);
  renderPager(data);
}

// The matched example chunks for a "search by video clip" query (a filmstrip
// above the results). Hidden for ordinary text/resume searches.
function renderQueryStrip(data) {
  const strip = $("query-strip");
  const clips = data.query_clips;
  if (!clips || !clips.length) {
    strip.classList.add("hidden");
    strip.innerHTML = "";
    return;
  }
  const span = data.query_span_seconds
    ? ` · ${fmtInt(data.query_span_seconds)}s` : "";
  const head = document.createElement("div");
  head.className = "query-strip-head";
  head.textContent =
    `Query clip — averaged ${fmtInt(data.query_chunk_count)} chunk(s)${span}` +
    (clips.length < data.query_chunk_count ? ` (showing ${clips.length})` : "");
  const row = document.createElement("div");
  row.className = "query-strip-row";
  clips.forEach((h) => row.appendChild(buildHitCard(h, {
    cardClass: "query-card", badge: "query",
  })));
  strip.innerHTML = "";
  strip.appendChild(head);
  strip.appendChild(row);
  strip.classList.remove("hidden");
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
  scheduleAutoRefine();  // live: marking re-ranks the results (debounced)
}

function updateMarkCount() {
  const vals = Object.values(state.marks);
  const up = vals.filter((m) => m.mark === "up").length;
  const down = vals.filter((m) => m.mark === "down").length;
  const el = $("mark-count");
  if (el) {
    el.textContent = up
      ? `auto-refining from ${up} 👍 / ${down} 👎`
      : (down ? `${down} 👎 marked (mark a 👍 to re-rank)` : "");
  }
  // Rail Step 1 summary + progress bars (target ~3 of each for a reliable sweep).
  const sum = $("rail-refine-summary");
  if (sum) sum.textContent = `${up} 👍 · ${down} 👎`;
  const ub = $("rail-up-bar"); if (ub) ub.style.width = Math.min(up / 3 * 100, 100) + "%";
  const db = $("rail-down-bar"); if (db) db.style.width = Math.min(down / 3 * 100, 100) + "%";
  const rn = $("step-refine-num"); if (rn) rn.classList.toggle("done", up > 0 || down > 0);
}

function clearMarks() {
  state.marks = {};
  document.querySelectorAll(".mark.on").forEach((b) => b.classList.remove("on"));
  updateMarkCount();
  // If we were showing a refined ranking, revert to the plain query immediately.
  if (state.mode === "refine") { state.mode = "search"; runSearch({ page: 0 }); }
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
        // Remember the chosen export defaults (k + cosine threshold; blank threshold = top-k).
        k: parseInt($("save-vec-k").value, 10) || 0,
        threshold: parseFloat($("save-vec-threshold").value) || 0,
        // Persist the active filter set with the vector so Resume restores it exactly.
        ..._activeFilters(),
      }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
    const { dim } = await r.json();
    scanNote(noteId, `saved <strong>${escapeHtml(tag)}</strong> (${dim}-d) ✓`);
    // Drawer flow: land on Saved searches with this tag ready to export.
    closeSaveDrawer();
    showToast(`Saved "${tag}" — configure export below`);
    showCurateView();
    loadExportHistory();
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
  // Per-tag cosine threshold (no single global cutoff). A blank/0 row falls back to the
  // default-threshold input (a scan always needs a positive cutoff, unlike top-k CSV export).
  const defThr = parseFloat($("curate-threshold").value) || 0.3;
  const thresholds = {};
  set.forEach((x) => { thresholds[x.query] = x.threshold > 0 ? x.threshold : defThr; });
  const btn = $("export-btn");
  btn.disabled = true;
  scanNote(noteId, `Launching per-segment scan over ${tags.length} tag(s)…`);
  try {
    const r = await fetch("/api/launch_segment_scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tags, thresholds, default_threshold: parseFloat($("curate-threshold").value) || 0.3,
        create_segment_set: $("curate-scan-segset").checked,
        merge_intervals: $("curate-scan-merge").checked,
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

// ---------- threshold search (fit a cutoff from 👍/👎 + active labeling) ----
// state.marks -> the endpoint's Mark shape. Only the corpus row index + up/down
// verdict matter for fitting; chunk_id is carried for completeness.
function _marksArray() {
  return Object.entries(state.marks).map(([chunk_id, m]) => ({
    chunk_id, mark: m.mark, index: m.index, segment_id: m.segment_id || "",
  }));
}

// Fit a threshold from the current 👍/👎 marks and fetch the next batch of
// boundary clips to label. Re-runnable: each round folds in the newest marks.
async function loadThresholdSearch() {
  if (!state.query) {
    previewNote("Run a search first, then mark a few 👍/👎 to fit a threshold.", true);
    setStatus("Run a search first to fit a threshold.", true);
    return;
  }
  const objSel = $("thr-objective");
  const objective = objSel ? objSel.value : "f1";
  const minP = $("thr-minp");
  const body = Object.assign(_filterBody(), {
    query: state.query,
    marks: _marksArray(),
    objective,
    min_precision: minP ? (parseFloat(minP.value) || 0.9) : 0.9,
    val_fraction: 0.0,
    sample_size: 12,
  });
  previewNote("Fitting threshold from your labels…");
  setStatus("Fitting threshold…");
  try {
    const data = await fetch("/api/threshold_search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
    showThreshold(data);
    updateRailThresh(data.threshold != null ? data.threshold : data.suggested_threshold);
    $("threshold-view").scrollIntoView({ behavior: "smooth", block: "start" });
    const tn = data.threshold != null ? ` · τ ${data.threshold.toFixed(3)}` : "";
    previewNote(`threshold search — ${data.num_up} 👍 / ${data.num_down} 👎${tn} — scrolled below ↓`);
    setStatus(`Threshold search · ${data.num_up} 👍 / ${data.num_down} 👎${tn}`);
  } catch (e) {
    previewNote("Threshold search failed: " + escapeHtml(e.message), true);
    setStatus("Threshold search failed: " + e.message, true);
  }
}

function showThreshold(data) {
  const t = $("threshold-view");
  t.innerHTML = "";
  if (!data) { exitThresholdView(); return; }
  t.appendChild(buildThreshold(data));
  t.classList.remove("hidden");
  // Dedicated view: hide the video results grid (+ pager/chips/dist) so the
  // threshold workspace is separate from the search ordering. CSS on the parent
  // does the hiding; Back / any new search removes the class.
  $("search-view").classList.add("threshold-active");
  t.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Leave the dedicated threshold view and restore the results grid. Idempotent.
function exitThresholdView() {
  const sv = $("search-view");
  if (sv) sv.classList.remove("threshold-active");
  const t = $("threshold-view");
  if (t) { t.classList.add("hidden"); t.innerHTML = ""; }
  state.thrLastTau = null;  // reset τ-stability tracking for the next concept
}

// A small precision-recall curve SVG for the labeled set.
function _prCurveSvg(curve) {
  if (!curve || !curve.recall || curve.recall.length < 2) return "";
  const W = 240, H = 200, m = 30;
  const pw = W - m - 8, ph = H - m - 20;
  const X = (r) => m + r * pw;              // recall 0..1
  const Y = (p) => 8 + (1 - p) * ph;        // precision 0..1
  const pts = curve.recall.map((r, i) => `${X(r).toFixed(1)},${Y(curve.precision[i]).toFixed(1)}`).join(" ");
  const grid = [0, 0.5, 1].map((v) =>
    `<line x1="${m}" y1="${Y(v).toFixed(1)}" x2="${W - 8}" y2="${Y(v).toFixed(1)}" class="sd-grid" />`
    + `<text x="${m - 4}" y="${(Y(v) + 3).toFixed(1)}" class="sd-ytick">${v}</text>`).join("");
  return `<svg class="sd-spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="precision-recall curve">
      ${grid}
      <line x1="${m}" y1="${8 + ph}" x2="${W - 8}" y2="${8 + ph}" class="sd-axis" />
      <line x1="${m}" y1="8" x2="${m}" y2="${8 + ph}" class="sd-axis" />
      <polyline points="${pts}" fill="none" class="sd-prline" />
      <text x="${(m + pw / 2).toFixed(1)}" y="${H - 2}" class="sd-axlbl">recall → (y: precision)</text>
    </svg>`;
}

// Overlaid 👍 (green) / 👎 (red) labeled-score strip above the corpus histogram,
// with the fitted τ marked. The threshold is where the two clouds separate.
function _labeledStripSvg(data) {
  const hist = data.histogram;
  if (!hist) return "";
  const W = 720, H = 70, mL = 46, mR = 12;
  const pw = W - mL - mR;
  const lo = hist.edges[0], hi = hist.edges[hist.edges.length - 1];
  const span = (hi - lo) || 1;
  const X = (s) => mL + ((s - lo) / span) * pw;
  const tick = (s, cls) => `<line x1="${X(s).toFixed(1)}" y1="14" x2="${X(s).toFixed(1)}" y2="46" class="${cls}" />`;
  const ups = (data.up_scores || []).map((s) => tick(s, "thr-up")).join("");
  const downs = (data.down_scores || []).map((s) => tick(s, "thr-down")).join("");
  const tau = data.threshold;
  const tauLine = tau != null
    ? `<line x1="${X(tau).toFixed(1)}" y1="6" x2="${X(tau).toFixed(1)}" y2="54" class="sd-thresh" />`
    + `<text x="${(X(tau) + 5).toFixed(1)}" y="14" class="sd-taulbl">τ ${tau.toFixed(3)}</text>`
    : "";
  return `<svg class="sd-spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="labeled positive/negative scores">
      <text x="${mL - 40}" y="26" class="sd-ytick">👍</text>
      <text x="${mL - 40}" y="44" class="sd-ytick">👎</text>
      ${ups}${downs}${tauLine}
    </svg>`;
}

// Slug a query into a default tag name (lowercase, underscore-joined, capped).
function _querySlug(q) {
  return (q || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40);
}

// Persist the fitted τ to a tag via the same /api/save_vector path the sidebar
// "Save vector" uses — writing export_log.threshold, which Export CSV and the
// offline scan read. The server saves the cached query vector (the one τ was fit
// against), so vector + τ stay consistent.
async function saveThresholdToTag(tau) {
  const noteEl = $("thr-save-note");
  const tag = ($("thr-tag").value || "").trim();
  if (!tag) { noteEl.classList.add("warn"); noteEl.textContent = "Enter a tag name."; return; }
  if (!state.query) { noteEl.classList.add("warn"); noteEl.textContent = "Run a search first."; return; }
  const btn = $("thr-save");
  btn.disabled = true;
  noteEl.classList.remove("warn");
  noteEl.textContent = "Saving…";
  try {
    const r = await fetch("/api/save_vector", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tag, query: state.query,
        k: parseInt(($("save-vec-k") || {}).value, 10) || 50,
        threshold: tau,
        ..._activeFilters(),
      }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
    await r.json();
    noteEl.innerHTML = `Saved <b>${escapeHtml(tag)}</b> · τ=${tau.toFixed(3)} — now used by `
      + `Download CSV + the offline scan. Find it on the <b>Export</b> tab.`;
    // Mirror into the sidebar inputs so the two stay consistent.
    if ($("save-vec-tag")) $("save-vec-tag").value = tag;
    if ($("save-vec-threshold")) $("save-vec-threshold").value = tau.toFixed(3);
  } catch (e) {
    noteEl.classList.add("warn");
    noteEl.textContent = "Save failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// Objective codes -> plain-language labels for the picker.
const THR_OBJECTIVES = [
  ["f1", "Balanced (F1)"],
  ["precision", "High precision (few false hits)"],
  ["youden", "Best separation"],
];

// The Tune-threshold panel: a guided 3-step flow — Teach (label) -> See (fit) ->
// Save (persist to tag). The active step is emphasized by state; steps 2/3 stay
// disabled until there's a fit.
function buildThreshold(data) {
  const wrap = document.createElement("div");
  wrap.className = "score-dist thr-panel";
  const fit = data.fit;
  const hasFit = !!fit;
  const tau = data.threshold;
  // First-pass policy: a label-free suggested τ from the query's score distribution.
  // effTau is what Save uses — the labeled fit when present, else the suggestion.
  const suggested = data.suggested_threshold;
  const effTau = hasFit ? tau : suggested;
  const up = data.num_up || 0, down = data.num_down || 0;
  const enough = up >= 3 && down >= 3;
  const pct = (x) => (x == null ? "—" : (100 * x).toFixed(1) + "%");

  // Progress hint: tell the user the single next thing to do.
  let hint;
  if (!hasFit) {
    hint = up === 0 && down === 0
      ? `Suggested τ ${effTau != null ? effTau.toFixed(3) : "—"} is set from the score distribution — save it as-is, or mark clips 👍/👎 to refine.`
      : `Need at least one 👍 and one 👎 to refine — you have ${up} 👍 / ${down} 👎 (suggested τ still available below).`;
  } else if (!enough) {
    hint = `Cutoff from few labels (${up} 👍 / ${down} 👎). Label more + Sharpen for a reliable τ.`;
  } else {
    hint = "Looking solid. Sharpen a round or two if τ still moves, then save it to a tag.";
  }

  // τ-stability across re-fits.
  let stab = "";
  if (hasFit && state.thrLastTau != null) {
    const d = Math.abs(tau - state.thrLastTau);
    stab = d < 0.005 ? "τ stable ✓" : `τ moved ${d.toFixed(3)} since last round`;
  }
  if (hasFit) state.thrLastTau = tau;

  const objOptions = THR_OBJECTIVES
    .map(([v, label]) => `<option value="${v}">${label}</option>`).join("");
  const metrics = hasFit
    ? `<span class="sub">τ <b>${tau.toFixed(3)}</b> · precision ${pct(fit.precision)}
         · recall ${pct(fit.recall)} · F1 ${pct(fit.f1)} · AP ${pct(fit.average_precision)}
         · ${fit.n_pos} 👍 / ${fit.n_neg} 👎${fit.held_out ? " · held-out" : ""}
         ${stab ? ` · <b>${stab}</b>` : ""}</span>`
    : "";
  const note = data.note ? `<div class="sub sd-hint">${escapeHtml(data.note)}</div>` : "";

  wrap.innerHTML = `
    <div class="thr-topbar">
      <button id="thr-back" type="button" class="ghost">← Back to results</button>
      <span class="thr-context">${escapeHtml(state.query || "")}</span>
    </div>
    <div class="interval-head"><h3>Tune threshold</h3></div>
    <div class="thr-progress">${escapeHtml(hint)}</div>
    ${note}

    <div class="thr-steps">
      <section class="thr-step ${hasFit ? "done" : "active"}">
        <div class="thr-step-head"><span class="thr-step-num">1</span> Teach it
          <span class="thr-counts">${up} 👍 / ${down} 👎</span></div>
        <div class="sub">Mark clips 👍 (a match) / 👎 (not) — aim for at least 3 of each.
          These are picked near the current cutoff, where your labels matter most.</div>
        <div id="thr-grid" class="grid thr-grid"></div>
        <button id="thr-more" type="button" class="ghost">More clips to label</button>
      </section>

      <section class="thr-step active">
        <div class="thr-step-head"><span class="thr-step-num">2</span> See the cutoff</div>
        ${hasFit
          ? `${metrics}
          <div class="thr-controls">
            <label>optimize for
              <select id="thr-objective">${objOptions}</select>
            </label>
            <label class="thr-minp-wrap">min precision
              <input id="thr-minp" type="number" step="0.05" min="0" max="1" value="0.90" />
            </label>
            <button id="thr-sharpen" type="button" class="ghost">Sharpen (more labels + re-fit)</button>
          </div>`
          : `<span class="sub">Suggested τ <b>${effTau != null ? effTau.toFixed(3) : "—"}</b>
               — from this query's score distribution (label clips above to refine).</span>`}
        <div class="thr-charts">
          <div class="thr-strip">${_labeledStripSvg(data)}<div id="thr-hist"></div></div>
          ${hasFit ? `<div class="thr-pr">${_prCurveSvg(fit.curve)}</div>` : ""}
        </div>
      </section>

      <section class="thr-step active">
        <div class="thr-step-head"><span class="thr-step-num">3</span> Save it</div>
        <div class="thr-save-row">
          <label>tag <input id="thr-tag" type="text" placeholder="tag name" value="${escapeHtml(_querySlug(state.query))}" /></label>
          <button id="thr-save" type="button" class="primary" ${effTau != null ? "" : "disabled"}>Save threshold to tag</button>
        </div>
        <div id="thr-save-note" class="note"></div>
        <div class="sub sd-hint">Saves τ + this query to the tag — the value Download CSV and the offline scan use.</div>
      </section>
    </div>`;

  // Corpus histogram (reuse the distribution chart). The endpoint marks it at the
  // fitted τ when labeled, else the suggested τ — so it's shown in both cases.
  if (data.histogram) {
    const h = Object.assign({}, data.histogram, { mode: "score" });
    wrap.querySelector("#thr-hist").appendChild(buildDistribution(h));
  }

  wrap.querySelector("#thr-back").onclick = () => exitThresholdView();
  wrap.querySelector("#thr-more").onclick = () => loadThresholdSearch();
  // Save works with or without labels — persists the fit τ if present, else the
  // suggested τ, so a fresh tag can be saved with zero labeling.
  if (effTau != null) {
    wrap.querySelector("#thr-save").onclick = () => saveThresholdToTag(effTau);
  }

  if (hasFit) {
    const objSel = wrap.querySelector("#thr-objective");
    objSel.value = fit.objective;
    const minpWrap = wrap.querySelector(".thr-minp-wrap");
    const syncMinp = () => { minpWrap.style.display = objSel.value === "precision" ? "" : "none"; };
    syncMinp();
    // Changing the objective (or the precision floor) re-fits immediately.
    objSel.addEventListener("change", () => { syncMinp(); loadThresholdSearch(); });
    wrap.querySelector("#thr-minp").addEventListener("change", () => loadThresholdSearch());
    wrap.querySelector("#thr-sharpen").onclick = () => loadThresholdSearch();
  }

  // Active-labeling grid: standard cards + mark buttons; marking then Sharpen/More
  // folds the new labels into the next fit (marks live in the shared state.marks).
  const grid = wrap.querySelector("#thr-grid");
  (data.sample || []).forEach((h, i) => {
    const m = state.marks[h.chunk_id];
    const card = buildHitCard(h, {
      cardClass: "thr-card",
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
  return wrap;
}

// ---------- export config helpers (assemble tags -> CSV / scan) ----------

// Parse a `query, k, threshold` textarea. Trailing comma-separated numbers are read
// right-to-left: a float in (0,1] is the per-tag threshold, an integer is k. Both are
// optional; missing values fall back to the default-k / default-threshold inputs.
function parseConfigQueries(queriesId, defKId, defThreshId) {
  const defK = parseInt($(defKId).value || "50", 10) || 50;
  // Default threshold 0 = top-k (the historical Download CSV default); a positive
  // default-threshold input switches unspecified lines to similarity-cutoff mode.
  const defT = defThreshId ? (parseFloat($(defThreshId).value) || 0) : 0;
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
        <td class="exp-tc"><input type="number" class="exp-t" step="0.01" min="0" max="1" placeholder="top-k" value="${e.threshold > 0 ? e.threshold : ''}" title="per-tag cosine threshold: Download CSV keeps clips >= this (capped at k); blank = top-k. Also used by the offline scan (which falls back to the default)." /></td>
        <td class="exp-open-c">${openBtn}</td>
      </tr>`;
    })
    .join("");
  wrap.innerHTML = `<table class="exp-hist-table">
      <thead><tr><th></th><th>tag</th><th>query</th><th>corpus model · vec</th><th>filters</th><th>k</th><th>thresh</th><th></th></tr></thead>
      <tbody>${body}</tbody></table>`;
  filterExportHistory();
  const cnt = $("saved-count");
  if (cnt) cnt.textContent = rows.length ? `(${rows.length})` : "";
  renderExportPanel();
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
  const push = (query, k, threshold) => {
    const qq = (query || "").trim().replace(/,+$/, "");
    if (!qq) return;
    const key = qq.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    // Raw per-row threshold: >0 => similarity-cutoff mode; 0/blank => pure top-k. Each
    // consumer applies its own default (Download CSV keeps 0 = top-k; the scan substitutes
    // its default_threshold for 0 since a scan always needs a positive cutoff).
    out.push({
      query: qq,
      k: parseInt(k, 10) || 50,
      threshold: Number(threshold) > 0 ? Number(threshold) : 0,
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

// Direct local export: each assembled tag's top-k from the LOADED (resident) corpus,
// concatenated into one CSV (+ parquet to S3). No preview/selection needed -- this is
// the "download CSV" for the small corpus. (The toolbar's "Download selected" exports a
// hand-picked subset after Load preview.)
async function downloadCsv() {
  const set = collectExportQueries();
  if (!set.length) { curateNote("Tick at least one saved search (or add an ad-hoc line).", true); return; }
  if (!state.embeddingsUri) { curateNote("Load a corpus first.", true); return; }
  // Cutoff selector: Top-K forces pure top-k (threshold 0); Threshold keeps each row's τ.
  const queries = set.map((x) => ({
    query: x.query, k: x.k,
    threshold: _cutoffMode === "topk" ? 0 : (x.threshold || 0),
  }));
  curateNote(`Exporting ${queries.length} tag(s) from the loaded corpus…`);
  $("export-btn").disabled = true;
  try {
    const resp = await fetch("/api/export_config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        queries, dedupe: true,
        dedupe_segment: $("curate-csv-dedup").checked,
        // Also register the exported segments as a DORA / Data Explorer segment set.
        create_segment_set: $("curate-csv-segset").checked,
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
    curateNote(`Downloaded CSV for ${queries.length} tag(s)` +
      (parquet ? ` · parquet → ${parquet}` : " · ⚠ parquet not written") + segsetNote(resp), !parquet);
  } catch (e) {
    curateNote("Download failed: " + e.message, true);
  } finally {
    $("export-btn").disabled = false;
  }
}

// ---------- new shell: settings gear, save drawer, single-export panel ----------

// ⚙ corpus/model settings popover (relocated header controls).
function toggleSettings() {
  const pop = $("settings-pop");
  const open = pop.classList.toggle("hidden") === false;
  $("settings-gear").setAttribute("aria-expanded", String(open));
}

// Save drawer (guided-rail Step 3). Reuses the save-vec-* inputs + saveVector().
function openSaveDrawer() {
  if (!state.query) { setStatus("Run a search or refine first to define a vector.", true); return; }
  const vals = Object.values(state.marks);
  const up = vals.filter((m) => m.mark === "up").length;
  const down = vals.filter((m) => m.mark === "down").length;
  const modeLbl = state.mode === "refine" ? "refined" : (state.mode === "resume" ? "resumed" : "text");
  $("save-vec-summary").innerHTML =
    `query <b>${escapeHtml(state.query)}</b><br>vector <b>${modeLbl}</b> · votes <b>${up} 👍 / ${down} 👎</b>`;
  if (!($("save-vec-tag").value || "").trim()) $("save-vec-tag").value = _querySlug(state.query);
  $("drawer-overlay").classList.add("open");
  $("save-drawer").classList.add("open");
}
function closeSaveDrawer() {
  $("drawer-overlay").classList.remove("open");
  $("save-drawer").classList.remove("open");
}

// Reflect the fitted/suggested τ in the rail's Threshold step.
function updateRailThresh(tau) {
  const el = $("rail-thresh-val");
  if (el) el.textContent = (tau == null) ? "—" : `τ ${Number(tau).toFixed(3)}`;
  const num = $("step-thresh-num");
  if (num) num.classList.toggle("done", tau != null);
}

// ---------- single-export panel (Saved searches page) ----------
// Cutoff = Top-K sends each row's k (pure top-k); Cutoff = Threshold sends each row's
// cosine threshold (kept >= tau, capped at k) — the real /api/export_config semantics,
// no fictional conversion.
let _cutoffMode = "threshold";
// Interval (merge contiguous above-threshold clips) vs Segment (one best clip per
// segment). Offline-scan deployments only (merge_intervals lives on the scan path).
let _sampleMode = "interval";

function setCutoffMode(mode) {
  _cutoffMode = mode === "topk" ? "topk" : "threshold";
  renderExportPanel();
}
function setSampleMode(mode) {
  _sampleMode = mode === "segment" ? "segment" : "interval";
  // The scan reads merge_intervals from this checkbox.
  const m = $("curate-scan-merge");
  if (m) m.checked = _sampleMode === "interval";
  renderExportPanel();
}

function _selectedTagCount() {
  return $("export-history").querySelectorAll("tbody tr .exp-pick:checked").length;
}

function renderExportPanel() {
  const offline = state.offlineScan !== false;
  const badge = $("mech-badge");
  if (badge) {
    badge.textContent = offline ? "async" : "instant";
    badge.className = "mech-badge " + (offline ? "async" : "instant");
  }
  // Active states on the segmented selectors.
  $("cutoff-seg").querySelectorAll("button[data-mode]").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === _cutoffMode));
  $("sample-seg").querySelectorAll("button[data-mode]").forEach((b) =>
    b.classList.toggle("active", b.dataset.mode === _sampleMode));
  const cd = $("cutoff-desc");
  if (cd) cd.textContent = _cutoffMode === "topk"
    ? "Each tag exports its top-K clips (pure ranking; ignores threshold)."
    : "Each tag keeps clips at or above its saved cosine threshold τ (capped at k).";
  const sd = $("sample-desc");
  if (sd) sd.textContent = _sampleMode === "segment"
    ? "One best (highest-scoring) clip per segment — no interval merge."
    : "Merge contiguous above-threshold clips per segment into time intervals.";

  const n = _selectedTagCount();
  const note = $("selection-note");
  if (note) {
    note.classList.toggle("empty", n === 0);
    note.textContent = n === 0
      ? "No searches selected — tick one or more rows above to export."
      : `Exporting ${n} search${n === 1 ? "" : "es"}.`;
  }
  const btn = $("export-btn");
  if (btn) {
    btn.textContent = offline ? "🚀 Export (offline scan)" : "⬇ Export CSV";
    btn.className = "btn-export " + (offline ? "async" : "instant");
    btn.disabled = n === 0;
  }
}

// One button, deployment-aware: instant CSV (resident corpus) or async per-segment scan.
function doExport() {
  if (state.offlineScan !== false) launchCurateScan();
  else downloadCsv();
}

let _toastTimer = null;
function showToast(msg) {
  const t = $("toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove("show"), 3000);
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
