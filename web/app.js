"use strict";

/* CosmicSearch front-end. The UI is the reference mockup (search -> refine ->
   threshold sweep -> save -> saved-searches table -> export); every control is
   wired to the real backend. Capabilities the mockup doesn't surface (corpus/model
   switch behind the gear, video-clip search, DORA segment-set resolution, threshold
   objective/active-learning) live server-side and are driven with sensible defaults. */

const $ = (id) => document.getElementById(id);
const fmtInt = (n) => Number(n || 0).toLocaleString("en-US");
const escapeHtml = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  platform: null,
  offlineScan: true,
  corpus: null,
  embeddingsUri: null,
  query: "",
  label: "",
  mode: "search",            // search | refine | window | resume
  page: 0,
  pageSize: 24,
  total: 0,
  scoreHi: 0.4,
  hits: [],                  // current page hits
  marks: {},                 // chunk_id -> {mark, segment_id, index, rank, score}
  windowReq: null,
  resumeVec: null,
  resumeLabel: "",
  // filters resolved from the search-page panel + the saved-searches (export) panel
  searchSegUuid: null, searchSegName: null,
  exportSegUuid: null, exportSegName: null,
  // threshold
  tempTau: null, confirmedTau: null, suggestedTau: null,
  tauUserSet: false,         // true once the user drags/confirms τ (then labels stop moving it)
  sweep: null,               // {edges,counts,total,min,max,mean, up[], down[]}
  sweepActive: false,        // grid shows the stratified boundary sample while sweeping
  sweepSample: [],           // active-learning batch from /api/threshold_search
  rendered: [],              // clips currently in the grid (for vote lookup)
  // saved searches
  savedRows: [], selected: new Set(),
  scanJobs: [],
  cutoffMode: "threshold",
  perTagK: {},               // tag -> top-K, for per-search K in a multi-tag Top-K export
  sampleMode: "interval",
};
let _issueSeq = 0;

/* ===================== bootstrap ===================== */
async function init() {
  wireEvents();
  await loadPlatform();
  loadDefaultCorpusWhenReady();
  loadSaved();
}

async function loadPlatform() {
  try {
    const p = await fetch("/api/platform").then((r) => r.json());
    state.platform = p;
    state.offlineScan = !p || p.offline_scan !== false;
    const tag = $("brandTag");
    if (p && p.label) { tag.textContent = p.label; tag.className = "tag " + (p.name === "trucks" ? "trucking" : "cars"); }
  } catch (e) { /* cosmetic */ }
  $("scansSection").style.display = state.offlineScan ? "block" : "none";
  renderExportPanel();
}

function applyCorpus(c) {
  state.corpus = c;
  state.embeddingsUri = c.embeddings_uri;
  state.scoreHi = 0.4;
  $("corpusPill").innerHTML = `🗂 <b>${fmtInt(c.num_rows)} clips</b> · ${escapeHtml(c.model || "model")}`;
  // The pill reported the resident browse corpus even while searches ran against
  // the full one, so it read as though the app only held ~2M clips. Correct it
  // whenever full-corpus mode is the active path.
  refreshCorpusPill();
  if ($("embeddings-uri")) $("embeddings-uri").value = c.embeddings_uri || "";
  if ($("model-uri") && c.model_uri !== undefined) $("model-uri").value = c.model_uri || "";
  // Search filters track the LOADED (in-app browse) corpus -- you're searching it, so
  // default its date range to that corpus's span.
  { const f = $("sf-dateFrom"); if (f) { f.value = c.span_lo_date; f.min = c.span_lo_date; f.max = c.span_hi_date; }
    const t = $("sf-dateTo"); if (t) { t.value = c.span_hi_date; t.min = c.span_lo_date; t.max = c.span_hi_date; } }
  // Export/offline-scan runs over the FULL main embedding space (the scan corpus, which
  // spans far wider than the browse corpus), so its dates default to UNBOUNDED (blank =
  // whole corpus). Don't pin them to the loaded span or the scan is silently narrowed
  // (and don't constrain min/max, so earlier dates than the browse corpus stay pickable).
  ["ex-dateFrom", "ex-dateTo"].forEach((id) => { const el = $(id); if (el) { el.value = ""; el.removeAttribute("min"); el.removeAttribute("max"); } });
}

async function loadDefaultCorpusWhenReady() {
  for (let i = 0; i <= 240; i++) {
    try {
      const r = await fetch("/api/corpus");
      if (r.ok) { const c = await r.json(); if (c && c.num_rows != null) { applyCorpus(c); return; } }
      $("corpusPill").textContent = "warming up — loading model + corpus…";
    } catch (e) { $("corpusPill").textContent = "warming up — loading model + corpus…"; }
    await new Promise((res) => setTimeout(res, 3000));
  }
  $("corpusPill").textContent = "corpus unavailable — reload to retry";
}

/* ===================== event wiring ===================== */
function wireEvents() {
  $("navSearch").onclick = () => setPage("search");
  $("navSaved").onclick = () => setPage("saved");
  $("settingsGear").onclick = () => $("settingsPop").classList.toggle("hidden");
  $("load-corpus").onclick = loadCorpus;
  $("load-model").onclick = loadModel;

  const go = () => runSearch();
  $("searchBtn").onclick = go;
  $("searchBtn2").onclick = go;
  $("searchInput").onkeydown = (e) => { if (e.key === "Enter") go(); };
  $("searchInput2").onkeydown = (e) => { if (e.key === "Enter") go(); };
  $("clipToggle").onclick = () => $("clipPanel").classList.toggle("hidden");
  wireUploadSearch();
  $("vs-search-btn").onclick = () => runWindowSearch({ page: 0 });

  $("filtersChip").onclick = toggleSearchFilters;
  $("applyFiltersBtn").onclick = applySearchFilters;
  wireDxCombo();
  wireFullCorpusToggles();

  $("pagePrev").onclick = () => reload({ page: state.page - 1 });
  $("pageNext").onclick = () => reload({ page: state.page + 1 });
  $("jumpRankBtn").onclick = () => { const r = parseInt($("jumpRank").value, 10); if (r >= 1) reload({ start_rank: r }); };
  $("jumpSimBtn").onclick = () => { const s = parseFloat($("jumpSim").value); if (!Number.isNaN(s)) reload({ start_score: s }); };
  $("quickDedup").onchange = () => renderGrid();
  $("perPage").onchange = () => { state.pageSize = parseInt($("perPage").value, 10) || 24; if (state.query) reload({ page: 0 }); };

  $("refineBtn").onclick = () => runRefine({ page: 0 });
  $("sweepBtn").onclick = openSweep;
  $("sweepClose").onclick = closeSweep;
  $("useThresholdBtn").onclick = confirmThreshold;
  document.querySelectorAll(".rail-header").forEach((h) => h.onclick = () => setActiveStep(h.dataset.step));

  $("saveOpenBtn").onclick = openSaveDrawer;
  $("drawerClose").onclick = closeDrawer;
  $("overlay").onclick = closeDrawer;
  $("finalSaveBtn").onclick = finalSave;

  $("savedReload").onclick = loadSaved;
  $("newSearchBtn").onclick = () => setPage("search");
  $("savedFilter").oninput = renderTable;
  $("modelFilter").onchange = renderTable;
  $("scansReload").onclick = () => loadScanJobs(true);
  $("tbody").onclick = (e) => {
    const b = e.target.closest("button.resume-link");
    if (b && b.dataset.id) resumeSession(b.dataset.id);
  };
  $("tbody").onchange = (e) => {
    if (e.target.matches("input[type=checkbox]")) {
      const i = parseInt(e.target.dataset.i, 10);
      if (e.target.checked) state.selected.add(i); else state.selected.delete(i);
      const tr = e.target.closest("tr"); if (tr) tr.classList.toggle("checked", e.target.checked);
      renderExportPanel();
    }
  };

  $("cutoffSeg").onclick = (e) => { const b = e.target.closest("button[data-mode]"); if (b) setCutoffMode(b.dataset.mode); };
  $("sampleSeg").onclick = (e) => { const b = e.target.closest("button[data-mode]"); if (b) setSampleMode(b.dataset.mode); };
  $("exportBtn").onclick = doExport;
}

/* ===================== page + settings ===================== */
function setPage(which) {
  $("navSearch").classList.toggle("active", which === "search");
  $("navSaved").classList.toggle("active", which === "saved");
  $("page-search").classList.toggle("active", which === "search");
  $("page-saved").classList.toggle("active", which === "saved");
  if (which === "saved") {
    // Each render is isolated so a failure in one never blocks the others -- in
    // particular, a render error must not prevent the Recent scans list from loading.
    try { renderTable(); } catch (e) { console.error("renderTable failed", e); }
    try { renderExportPanel(); } catch (e) { console.error("renderExportPanel failed", e); }
    if (state.offlineScan) loadScanJobs();
  }
}

async function loadCorpus() {
  const uri = $("embeddings-uri").value.trim();
  if (!uri) return;
  setNote("corpus-note", "loading corpus (downloading + into memory)…");
  $("corpusPill").textContent = "loading corpus…";
  try {
    const c = await fetch("/api/corpus?uri=" + encodeURIComponent(uri)).then((r) => { if (!r.ok) return r.json().then((j) => { throw new Error(j.detail || r.status); }); return r.json(); });
    applyCorpus(c);
    state.query = ""; state.marks = {}; $("resultsGrid").innerHTML = "";
    setNote("corpus-note", `Loaded ${fmtInt(c.num_rows)} clips · segment_id ${c.has_segment_id ? "present ✓" : "absent"}`);
  } catch (e) { setNote("corpus-note", "failed to load corpus: " + e.message, true); $("corpusPill").textContent = "corpus unavailable"; }
}
async function loadModel() {
  const uri = $("model-uri").value.trim();
  setNote("model-note", "loading model (into memory, ~minutes)…");
  try {
    const m = await fetch("/api/model?uri=" + encodeURIComponent(uri)).then((r) => { if (!r.ok) return r.json().then((j) => { throw new Error(j.detail || r.status); }); return r.json(); });
    setNote("model-note", "model: " + m.label);
    if (state.corpus) { state.corpus.model = m.label; state.corpus.model_uri = m.model_uri; applyCorpus(state.corpus); }
    if (state.query) reload({ page: 0 });
  } catch (e) { setNote("model-note", "failed to load model: " + e.message, true); }
}

/* ===================== search-page filters ===================== */
function toggleSearchFilters() {
  const p = $("searchFiltersPanel");
  p.style.display = p.style.display === "none" ? "block" : "none";
}
// Data Explorer segment-set combobox: type -> live DORA search -> dropdown -> pick.
// A factory so both the Search filter and the Saved-searches (export) filter get
// the same type-ahead dropdown, each writing to its own state.
function _makeDxCombo(cfg) {
  const inp = $(cfg.inputId), combo = $(cfg.comboId), menu = $(cfg.menuId), note = $(cfg.noteId);
  if (!inp) return;
  let timer = null, items = [];
  const show = () => menu.classList.remove("hidden");
  const hide = () => menu.classList.add("hidden");
  inp.addEventListener("input", () => {
    clearTimeout(timer);
    combo.classList.remove("chosen");
    cfg.onClear();
    const v = inp.value.trim();
    if (v.length < 2) { hide(); if (note) note.textContent = v ? "keep typing… (min 2 characters)" : ""; return; }
    timer = setTimeout(async () => {
      if (note) note.textContent = "loading segment sets…";
      try {
        items = (await fetch("/api/segment_sets?name_filter=" + encodeURIComponent(v)).then((r) => { if (!r.ok) throw new Error("DORA " + r.status); return r.json(); })) || [];
        if (!items.length) { menu.innerHTML = '<div class="combo-msg">no sets match that name</div>'; show(); if (note) note.textContent = "no sets match that name"; return; }
        menu.innerHTML = items.map((s, i) => `<div class="combo-opt" data-i="${i}" role="option"><span class="opt-name">${escapeHtml(s.name)}</span><span class="opt-meta">v${escapeHtml(String(s.version))} · ${fmtInt(s.num_segments)} segments</span></div>`).join("");
        menu.querySelectorAll(".combo-opt").forEach((o) => o.addEventListener("mousedown", (e) => {
          e.preventDefault();
          const s = items[parseInt(o.dataset.i, 10)];
          inp.value = `${s.name} v${s.version}`;
          combo.classList.add("chosen");
          if (note) note.textContent = `using ${s.name} v${s.version} (${fmtInt(s.num_segments)} segs)`;
          hide();
          fetch("/api/segment_set_prefetch?uuid=" + encodeURIComponent(s.uuid)).catch(() => {});
          cfg.onChoose(s);
        }));
        show();
        if (note) note.textContent = `${items.length} set${items.length > 1 ? "s" : ""} — pick one to downsample`;
      } catch (e) { if (note) note.textContent = "couldn't reach Data Explorer: " + e.message; hide(); }
    }, 300);
  });
  inp.addEventListener("focus", () => { if (items.length && inp.value.trim().length >= 2 && !combo.classList.contains("chosen")) show(); });
  inp.addEventListener("blur", () => setTimeout(hide, 150));
}
function wireDxCombo() {
  _makeDxCombo({
    inputId: "sf-dxset", comboId: "sf-dxset-combo", menuId: "sf-dxset-menu", noteId: "sf-dxset-note",
    onClear: () => { state.searchSegUuid = null; state.searchSegName = null; },
    onChoose: (s) => { state.searchSegUuid = s.uuid; state.searchSegName = `${s.name} v${s.version}`; if (state.query) reload({ page: 0 }); },
  });
  _makeDxCombo({
    inputId: "ex-dxset", comboId: "ex-dxset-combo", menuId: "ex-dxset-menu", noteId: "ex-dxset-note",
    onClear: () => { state.exportSegUuid = null; state.exportSegName = null; },
    onChoose: (s) => { state.exportSegUuid = s.uuid; state.exportSegName = `${s.name} v${s.version}`; },
  });
}

function applySearchFilters() {
  $("filtersChip").classList.add("active");
  $("filtersChip").textContent = "⚙ filters & view (active)";
  if (state.query) reload({ page: 0 });
  else showToast("Filters set — run a search");
}
function _splitList(v) { return (v || "").trim() || null; }
function _searchFilters() {
  return {
    page_size: state.pageSize,
    from_date: $("sf-dateFrom").value || null,
    to_date: $("sf-dateTo").value || null,
    segment_set_uuid: state.searchSegUuid,
    filter_lance_uri: _splitList($("sf-lance").value),
    vehicle: _splitList($("sf-vehicle").value),
    drive_id: _splitList($("sf-drive").value),
    embeddings_uri: state.embeddingsUri,
  };
}

/* ===================== search / refine / window / resume ===================== */
async function runSearch(startOpts) {
  const q = ($("resultsState").style.display !== "none" ? $("searchInput2").value : $("searchInput").value).trim()
    || $("searchInput").value.trim() || $("searchInput2").value.trim();
  if (!q) { showToast("Type a query first"); return; }
  $("searchInput2").value = q;
  state.query = q; state.mode = "search";
  // Full-corpus mode ranks all ~34M clips instead of the loaded corpus. It is a
  // different endpoint because that corpus is loaded on demand and has no paging
  // or refine; the response envelope is identical, so the grid is unchanged.
  if (_fullCorpusOn()) {
    const page = (startOpts && startOpts.page) || 0;
    // Same query and filters as the buffer we already hold -> page locally.
    if (state.fullBuf && state.fullBuf.key === _fullKey(q)) { _renderFullPage(page); return; }
    await _issueFullCorpus(q, page);
    return;
  }
  await _issue("/api/search", { query: q, ...(startOpts || {}), ..._searchFilters() });
}
async function runRefine(startOpts) {
  if (!state.query) return;
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, segment_id: m.segment_id, mark: m.mark, index: m.index, rank: m.rank, score: m.score }));
  if (!marks.some((m) => m.mark === "up")) { showToast("Mark at least one 👍 to re-rank"); return; }
  state.mode = "refine";
  showToast(`Re-ranking with ${marks.filter((m) => m.mark === "up").length} 👍 / ${marks.filter((m) => m.mark === "down").length} 👎…`);
  await _issue("/api/refine", { query: state.query, marks, negative_weight: 0.5, text_weight: 0.3, ...(startOpts || {}), ..._searchFilters() });
}
function _toNs(raw) {
  const v = (raw || "").trim().replace(/[_,\s]/g, ""); if (!v) return 0;
  const n = Number(v); if (!isFinite(n) || n <= 0) return 0;
  return n < 1e12 ? Math.round(n * 1e9) : Math.round(n);
}
async function runWindowSearch(startOpts) {
  if ((startOpts && startOpts.page === 0) || !state.windowReq) {
    const run_uuid = $("vs-run-uuid").value.trim(), segment_id = $("vs-segment-id").value.trim();
    if (!run_uuid && !segment_id) { showToast("Enter a drive or segment id"); return; }
    state.windowReq = { run_uuid, segment_id, start_ns: _toNs($("vs-start").value), end_ns: _toNs($("vs-end").value), query: "video clip: " + (run_uuid || segment_id) };
  }
  state.mode = "window"; state.query = state.windowReq.query;
  await _issue("/api/search_by_window", { ...state.windowReq, ...(startOpts || {}), ..._searchFilters() });
}
/* ===================== search by uploaded image (drag & drop) ===================== */
function wireUploadSearch() {
  const toggle = $("uploadToggle"), panel = $("uploadPanel"), dz = $("dropzone"), inp = $("uploadFile");
  if (!toggle || !dz) return;
  toggle.onclick = () => panel.classList.toggle("hidden");
  dz.onclick = () => inp.click();
  dz.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inp.click(); } };
  inp.onchange = () => { if (inp.files && inp.files[0]) handleUpload(inp.files[0]); };
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleUpload(f);
  });
}
function _uploadNote(msg, isErr) { const n = $("uploadNote"); if (n) { n.textContent = msg || ""; n.classList.toggle("err", !!isErr); } }
// Corpus clips are ~8-frame windows of a few seconds; sample an uploaded video over
// a window near that length so the query stays in-distribution (matching is by the
// 8 frames the model sees, not by fps -- see design notes).
const _UPLOAD_NUM_FRAMES = 8;
const _UPLOAD_SAMPLE_WINDOW_S = 8;   // widest span we sample 8 frames across
const _UPLOAD_MAX_DURATION_S = 600;  // hard reject absurdly long files
function _dataUrl(file) { return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(file); }); }
function _seek(v, t) { return new Promise((res) => { const on = () => { v.removeEventListener("seeked", on); res(); }; v.addEventListener("seeked", on); v.currentTime = t; }); }
async function _extractVideoFrames(file) {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.muted = true; v.playsInline = true; v.preload = "auto"; v.src = url;
  try {
    await new Promise((res, rej) => { v.onloadedmetadata = () => res(); v.onerror = () => rej(new Error("cannot read this video")); });
    const dur = (v.duration && isFinite(v.duration)) ? v.duration : 0;
    if (!dur) throw new Error("unknown video duration");
    if (dur > _UPLOAD_MAX_DURATION_S) throw new Error(`video too long (${Math.round(dur)}s > ${_UPLOAD_MAX_DURATION_S}s) — trim it first`);
    // Sample 8 frames across a centered window capped at _UPLOAD_SAMPLE_WINDOW_S.
    const span = Math.min(dur, _UPLOAD_SAMPLE_WINDOW_S);
    const start = Math.max(0, (dur - span) / 2);
    const canvas = document.createElement("canvas");
    const frames = [];
    for (let i = 0; i < _UPLOAD_NUM_FRAMES; i++) {
      const t = Math.min(start + span * (i + 0.5) / _UPLOAD_NUM_FRAMES, Math.max(0, dur - 0.03));
      await _seek(v, t);
      if (!canvas.width) { canvas.width = v.videoWidth || 448; canvas.height = v.videoHeight || 448; }
      canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
      frames.push(canvas.toDataURL("image/jpeg", 0.85).split(",", 2)[1]);
    }
    return { frames, dur, span };
  } finally { URL.revokeObjectURL(url); }
}
async function handleUpload(file) {
  const isImage = (file.type || "").startsWith("image/");
  const isVideo = (file.type || "").startsWith("video/");
  if (!isImage && !isVideo) { _uploadNote("Please drop an image or a video.", true); return; }
  if (file.size > 500 * 1024 * 1024) { _uploadNote("File too large (max 500MB).", true); return; }
  const imgEl = $("uploadPreview"), vidEl = $("uploadPreviewVid");
  imgEl.classList.add("hidden"); vidEl.classList.add("hidden");
  try {
    let frames_b64, noteAfter;
    if (isImage) {
      const dataUrl = await _dataUrl(file);
      imgEl.src = dataUrl; imgEl.classList.remove("hidden"); $("dzInner").classList.add("hidden");
      frames_b64 = [String(dataUrl).split(",", 2)[1] || ""];
      _uploadNote("Encoding image…");
      noteAfter = "";
    } else {
      vidEl.src = URL.createObjectURL(file); vidEl.classList.remove("hidden"); $("dzInner").classList.add("hidden");
      _uploadNote("Reading video frames…");
      const { frames, dur, span } = await _extractVideoFrames(file);
      frames_b64 = frames;
      _uploadNote(`Encoding ${frames.length} frames sampled from ${span.toFixed(1)}s${dur > span ? ` (of ${dur.toFixed(0)}s)` : ""}…`);
      noteAfter = dur > _UPLOAD_SAMPLE_WINDOW_S ? `Sampled a ${span.toFixed(0)}s window — clips match best at ~2-4s.` : "";
    }
    const enc = await fetch("/api/search_by_upload", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames_b64, filename: file.name, content_type: file.type }),
    }).then((r) => r.ok ? r.json() : r.json().then((j) => { throw new Error(j.detail || ("HTTP " + r.status)); }));
    // The uploaded example is now just a query vector -> reuse the resume path so
    // paging, refine, sweep, save, and offline-scan export all work unchanged.
    state.resumeVec = enc.vector;
    state.resumeLabel = (isVideo ? "🎬 " : "🖼️ ") + (enc.label || "uploaded example");
    state.query = state.resumeLabel;
    state.mode = "resume";
    state.marks = {};
    _uploadNote(noteAfter);
    await runVectorSearch({ page: 0 });
  } catch (e) { _uploadNote("Could not search by upload: " + e.message, true); }
}
async function runVectorSearch(startOpts) {
  if (!state.resumeVec) return;
  state.mode = "resume";
  await _issue("/api/search_by_vector", { vector: state.resumeVec, query: state.resumeLabel || state.query || "", ...(startOpts || {}), ..._searchFilters() });
}
function reload(startOpts) {
  if (state.mode === "refine") return runRefine(startOpts);
  if (state.mode === "resume") return runVectorSearch(startOpts);
  if (state.mode === "window") return runWindowSearch(startOpts);
  return runSearch(startOpts);
}

// Show the corpus the next search will actually use, not the resident one.
async function refreshCorpusPill() {
  let st = null;
  try { st = await fetch("/api/full_corpus_status").then((r) => r.json()); } catch (e) { return; }
  if (!st) return;
  const model = (state.corpus && state.corpus.model) || st.model || "model";
  if (st.status === "ready" && st.num_rows) {
    $("corpusPill").innerHTML = `🗂 <b>${fmtInt(st.num_rows)} clips</b> · full corpus · ${escapeHtml(model)}`;
  } else if (st.status === "loading") {
    $("corpusPill").innerHTML = `🗂 <b>full corpus loading…</b> ${Math.round(st.elapsed_s || 0)}s · ${escapeHtml(model)}`;
    setTimeout(refreshCorpusPill, 10000);
  } else if (st.status === "error") {
    $("corpusPill").innerHTML = `🗂 <b>full corpus failed</b> · ${escapeHtml(model)}`;
  }
}

// Search always covers the whole corpus. There is no toggle: offering the ~2M
// resident subset as a choice meant users could silently search 6% of the data
// and read the result as complete.
function _fullCorpusOn() { return true; }
function wireFullCorpusToggles() { refreshCorpusPill(); }
async function _issueFullCorpus(q, page) {
  // The corpus is read and decoded on first use (minutes, ~12GB), so the server
  // answers 503 until it is resident. Kick the load, tell the user where it is
  // up to, and poll rather than leaving the grid on "Searching...".
  $("emptyState").style.display = "none";
  $("resultsState").style.display = "block";
  const status = await fetch("/api/full_corpus_status").then((r) => r.json()).catch(() => null);
  // Paging hits this same path; when the corpus is already resident the poll
  // below is skipped and the request goes straight out.
  if (!status || status.status !== "ready") {
    await fetch("/api/full_corpus_load", { method: "POST" }).catch(() => {});
    $("gridStatus").textContent =
      "Loading the full corpus (first use, a few minutes) — this search will start automatically.";
    for (let i = 0; i < 60; i++) {
      await new Promise((res) => setTimeout(res, 10000));
      const s = await fetch("/api/full_corpus_status").then((r) => r.json()).catch(() => null);
      if (!s) continue;
      if (s.status === "error") { $("gridStatus").textContent = "Full corpus failed to load: " + s.error; return; }
      if (s.status === "ready") break;
      $("gridStatus").textContent = `Loading the full corpus… ${Math.round(s.elapsed_s || 0)}s`;
    }
  }
  // Fetch one batch and page through it locally. A page is a slice of scores
  // that are already in memory server-side, so re-requesting per page would be
  // a fresh 34M-row scan for results we already hold. One batch also means the
  // exact-score round trip can cover every page in a single fetch.
  const body = {
    query: q,
    page: 0,
    limit: FULL_BATCH,
    from_date: $("sf-dateFrom").value || null,
    to_date: $("sf-dateTo").value || null,
    vehicle: _splitList($("sf-vehicle").value),
    drive_id: _splitList($("sf-drive").value),
    segment_set_uuid: state.searchSegUuid,
    filter_lance_uri: _splitList($("sf-lance") ? $("sf-lance").value : ""),
  };
  $("gridStatus").textContent = "Searching all clips…";
  let data;
  try {
    data = await fetch("/api/full_search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => {
      if (!r.ok) return r.json().then((j) => { throw new Error(j.detail || ("HTTP " + r.status)); });
      return r.json();
    });
  } catch (e) {
    $("gridStatus").textContent = "Search failed: " + e.message;
    return;
  }
  state.fullBuf = { key: _fullKey(q), data, hits: data.hits || [] };
  _renderFullPage(page || 0);
}

// One request covers this many results; the pager slices them client-side.
const FULL_BATCH = 200;

function _fullKey(q) {
  return [q, $("sf-dateFrom").value, $("sf-dateTo").value,
          $("sf-vehicle").value, $("sf-drive").value,
          state.searchSegUuid || "",
          $("sf-lance") ? $("sf-lance").value : ""].join("|");
}

function _renderFullPage(page) {
  const buf = state.fullBuf;
  const ps = state.pageSize || 24;
  const pages = Math.max(1, Math.ceil(buf.hits.length / ps));
  state.page = Math.min(Math.max(0, page), pages - 1);
  state.hits = buf.hits.slice(state.page * ps, (state.page + 1) * ps);
  state.total = buf.hits.length;
  state.label = buf.data.label || state.query;
  state.scoreHi = buf.data.score_hi || 0.4;
  $("vectorChip").textContent = `vector: text "${state.query}"`;
  // State plainly what was searched and what the filters did. "full corpus" is
  // otherwise an unverifiable claim, and a filter that silently matched nothing
  // is indistinguishable from a filter that was never applied -- which is the
  // failure that went unnoticed for a week.
  const d = buf.data;
  const f = d.filters_applied || {};
  const active = Object.entries(f).filter(([, v]) => v && v.length).map(([k, v]) => `${k}=${v}`);
  const filtLine = active.length
    ? ` · filters ${active.join(", ")} narrowed to ${fmtInt(d.candidates_after_filters)}`
    : " · no filters";
  $("resultCountText").textContent = buf.hits.length
    ? `searched ALL ${fmtInt(d.num_rows_searched)} clips in ${d.elapsed_ms} ms${filtLine}`
      + ` · showing top ${fmtInt(buf.hits.length)}`
      + ` · similarity ${d.score_lo}–${d.score_hi} ±${d.score_error_bound} (approximate)`
    : `searched ALL ${fmtInt(d.num_rows_searched)} clips${filtLine} · nothing matched`;
  const prov = d.corpus_loaded_utc
    ? `corpus ${String(d.corpus_uri || "").split("/").slice(-2).join("/")}`
      + (d.corpus_version != null ? ` v${d.corpus_version}` : "")
      + ` · ${fmtInt(d.num_rows_searched)} rows · loaded ${d.corpus_loaded_utc}`
    : "";
  $("gridStatus").textContent = buf.hits.length ? prov : "Try widening the date range or filters. " + prov;
  renderQueryStrip(buf.data);
  renderGrid();
  renderPager();
  rescoreVisible();
}

// The page renders from quantized scores, which are bounded but not comparable
// to thresholds calibrated on the float corpus. Fetch the real 768-d cosine for
// the rows on screen and swap them in. Fired after render, never awaited: it is
// one S3 round trip with a ~1.65s floor, so blocking on it would undo the 190ms
// search for a number the user has not looked at yet.
async function rescoreVisible() {
  const rows = (state.hits || []).map((h) => h.row).filter((r) => r != null && r >= 0);
  if (!rows.length || !state.query) return;
  let data;
  try {
    data = await fetch("/api/full_rescore", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: state.query, rows }),
    }).then((r) => (r.ok ? r.json() : null));
  } catch (e) { return; }
  if (!data || !data.scores) return;
  const byRow = new Map(data.scores.map((s) => [s.row, s.score]));
  let changed = 0;
  for (const h of state.hits) {
    const exact = byRow.get(h.row);
    if (exact != null && exact !== h.score) { h.score = exact; changed++; }
    if (exact != null) h.score_kind = "exact";
  }
  if (!changed) return;
  // Exact scores can reorder rows that sat within the error bound of each other.
  state.hits.sort((a, b) => b.score - a.score);
  renderGrid();
  const el = $("resultCountText");
  if (el) el.textContent = el.textContent.replace(/\(approximate\)/, "(exact)");
}
async function _issue(endpoint, body) {
  const seq = ++_issueSeq;
  $("emptyState").style.display = "none";
  $("resultsState").style.display = "block";
  $("gridStatus").textContent = state.mode === "refine" ? "Re-ranking…" : "Searching…";
  let data;
  try {
    data = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then((r) => { if (!r.ok) return r.json().then((j) => { let m = j.detail; if (Array.isArray(m)) m = m.map((e) => `${(e.loc || []).join(".")}: ${e.msg}`).join("; "); throw new Error(m || ("HTTP " + r.status)); }); return r.json(); });
  } catch (e) {
    if (seq === _issueSeq) $("gridStatus").textContent = "Search failed: " + e.message;
    return;
  }
  if (seq !== _issueSeq) return;
  state.total = data.total; state.page = data.page; state.label = data.label || state.query;
  state.hits = data.hits || []; state.scoreHi = data.score_hi || 0.4;
  $("vectorChip").textContent = "vector: " + (state.mode === "refine" ? `text "${state.query}" + feedback` : (state.mode === "window" ? state.query : `text "${state.query}"`));
  const filt = _funnelText(data);
  $("resultCountText").textContent = data.total
    ? `${fmtInt(data.total)} clips ranked in ${data.elapsed_ms} ms · similarity ${data.score_lo}–${data.score_hi}${filt}`
    : "no clips match the current query + filters";
  $("gridStatus").textContent = data.total ? "" : "Try widening the date range or segment set.";
  renderQueryStrip(data);
  renderGrid();
  renderPager();
  if (state.sweep && $("sweepPanel").classList.contains("open")) refreshSweep();
}
function _funnelText(data) {
  const f = data.funnel || {}; const bits = [];
  if (f.corpus_total != null) bits.push(`of ${fmtInt(f.corpus_total)} corpus`);
  if (data.filter_lance_count != null && state) { /* lance count shown in filters note */ }
  return bits.length ? " · " + bits.join(" · ") : "";
}

/* ===================== grid ===================== */
function scoreColor(s) { if (s > 0.32) return "#c9f31d"; if (s > 0.24) return "#a9d92a"; if (s > 0.16) return "#8a8f4a"; return "#6b6b70"; }
function _dedupHits(hits) {
  if (!$("quickDedup").checked) return hits;
  const seen = new Set(); const out = [];
  for (const h of hits) { const k = h.segment_id || h.chunk_id; if (seen.has(k)) continue; seen.add(k); out.push(h); }
  return out;
}
function renderGrid() {
  const grid = $("resultsGrid");
  const compact = grid.classList.contains("compact");
  // While sweeping, the grid shows the stratified boundary sample (clips near the
  // cutoff, where labels matter most) — the active-learning loop.
  const hits = state.sweepActive ? (state.sweepSample || []) : _dedupHits(state.hits);
  state.rendered = hits;
  const maxS = Math.max(state.scoreHi, 0.001);
  grid.innerHTML = hits.map((h) => {
    const m = state.marks[h.chunk_id];
    const vc = m && m.mark === "up" ? "voted-up" : (m && m.mark === "down" ? "voted-down" : "");
    const col = scoreColor(h.score);
    const vsrc = "/api/video?uri=" + encodeURIComponent(h.source_media_uri || "");
    return `<div class="card ${vc}" data-chunk="${escapeHtml(h.chunk_id)}">
      <div class="thumb">
        <span class="rank-badge">#${fmtInt(h.rank)}</span>
        <span class="score-badge" style="background:${col}">${h.score.toFixed(3)}</span>
        <video controls preload="metadata" playsinline src="${vsrc}#t=0.1"></video>
      </div>
      <div class="card-body">
        <div class="confidence-bar"><div class="confidence-fill" style="width:${Math.max(h.score, 0) / maxS * 100}%; background:${col}"></div></div>
        <div class="card-id">${escapeHtml(h.segment_id || h.chunk_id)}</div>
        <div class="card-actions">
          <div class="vote-btn up ${m && m.mark === "up" ? "active" : ""}" data-v="up">👍</div>
          <div class="vote-btn down ${m && m.mark === "down" ? "active" : ""}" data-v="down">👎</div>
        </div>
        <span class="expand-toggle">show details ⌄</span>
        <div class="detail-meta">id ${escapeHtml(h.chunk_id)}<br>utc ${escapeHtml(h.start_utc || "")}${h.end_utc ? " → " + escapeHtml(h.end_utc) : ""}</div>
      </div></div>`;
  }).join("");
  grid.querySelectorAll(".card").forEach((card) => {
    const cid = card.dataset.chunk;
    card.querySelector(".vote-btn.up").onclick = () => vote(cid, "up");
    card.querySelector(".vote-btn.down").onclick = () => vote(cid, "down");
    card.querySelector(".expand-toggle").onclick = (e) => e.target.nextElementSibling.classList.toggle("open");
  });
  if (!compact) grid.classList.remove("compact");
  if (state.sweepActive) $("gridStatus").textContent = "Labeling clips near the cutoff (stratified sample) — vote to refine τ";
}
function renderQueryStrip(data) {
  const strip = $("queryStrip");
  const clips = data.query_clips;
  if (!clips || !clips.length) { strip.classList.add("hidden"); strip.innerHTML = ""; return; }
  const cards = clips.map((h) => `<div class="card"><div class="thumb">
      <span class="score-badge" style="background:${scoreColor(h.score)}">${h.score.toFixed(3)}</span>
      <video controls preload="metadata" playsinline src="/api/video?uri=${encodeURIComponent(h.source_media_uri || "")}#t=0.1"></video>
      <span class="card-badge">query</span></div></div>`).join("");
  strip.innerHTML = `<div class="query-strip-head">Query clip — averaged ${fmtInt(data.query_chunk_count)} chunk(s)${data.query_span_seconds ? " · " + fmtInt(data.query_span_seconds) + "s" : ""}</div><div class="query-strip-row">${cards}</div>`;
  strip.classList.remove("hidden");
}
function renderPager() {
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  $("pageLabel").textContent = `page ${fmtInt(state.page + 1)} / ${fmtInt(pages)}`;
  $("pagePrev").disabled = state.page <= 0;
  $("pageNext").disabled = (state.page + 1) >= pages;
}

/* ===================== votes / rail ===================== */
function vote(chunkId, dir) {
  const h = (state.rendered || []).find((x) => x.chunk_id === chunkId) || state.hits.find((x) => x.chunk_id === chunkId); if (!h) return;
  const cur = state.marks[chunkId];
  if (cur && cur.mark === dir) delete state.marks[chunkId];
  else state.marks[chunkId] = { mark: dir, segment_id: h.segment_id, index: h.index, row: h.row, rank: h.rank, score: h.score };
  // Toggle ONLY the clicked card in place — never re-render the whole grid (that
  // would reload every video on every click).
  const m = state.marks[chunkId];
  const card = $("resultsGrid").querySelector(`.card[data-chunk="${(window.CSS && CSS.escape) ? CSS.escape(chunkId) : chunkId}"]`);
  if (card) {
    card.classList.toggle("voted-up", !!m && m.mark === "up");
    card.classList.toggle("voted-down", !!m && m.mark === "down");
    card.querySelector(".vote-btn.up").classList.toggle("active", !!m && m.mark === "up");
    card.querySelector(".vote-btn.down").classList.toggle("active", !!m && m.mark === "down");
  }
  updateRail();
  // While sweeping: plot the label immediately (local), then (debounced) re-fit the
  // threshold from the backend so τ moves with the labels — WITHOUT replacing the
  // stratified grid (no video reload). Each fit is logged to threshold_episodes.
  if (state.sweepActive && state.sweep) { drawSweep(); scheduleFit(); }
}
let _fitTimer = null;
function scheduleFit() { clearTimeout(_fitTimer); _fitTimer = setTimeout(fitOnly, 500); }
async function fitOnly() {
  if (!state.sweepActive) return;
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, mark: m.mark, index: m.index, row: m.row, segment_id: m.segment_id || "" }));
  const f = _searchFilters();
  try {
    const thr = await fetch(_thresholdEndpoint(), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, marks, objective: "f1", min_precision: 0.9, val_fraction: 0.0, sample_size: 12, ...f }) }).then((r) => r.ok ? r.json() : null);
    if (!thr) return;
    state.suggestedTau = thr.suggested_threshold;
    const fitTau = thr.threshold;
    // Move τ to the fresh fit/suggestion unless the user has manually set it.
    if (!state.tauUserSet && (fitTau != null || state.suggestedTau != null)) state.tempTau = (fitTau != null) ? fitTau : state.suggestedTau;
    const fit = thr.fit;
    if (fit && fitTau != null) $("threshSub").textContent = `fit τ ${fitTau.toFixed(3)} · P ${(100 * fit.precision).toFixed(0)}% · R ${(100 * fit.recall).toFixed(0)}% · F1 ${(100 * fit.f1).toFixed(0)}% (${fit.n_pos}👍/${fit.n_neg}👎)`;
    else if (state.suggestedTau != null) $("threshSub").textContent = `suggested τ ${state.suggestedTau.toFixed(3)} — label 👍/👎 to fit`;
    drawSweep();
  } catch (e) { /* transient */ }
}
function _markScores() {
  const up = [], down = [];
  Object.values(state.marks).forEach((m) => { if (typeof m.score === "number") (m.mark === "up" ? up : down).push(m.score); });
  return { up, down };
}
function _counts() {
  const v = Object.values(state.marks);
  return { up: v.filter((m) => m.mark === "up").length, down: v.filter((m) => m.mark === "down").length };
}
function updateRail() {
  const { up, down } = _counts();
  const s = `${up} 👍 · ${down} 👎`;
  $("refineSummary").textContent = s; $("voteLabel").textContent = s;
  $("upBar").style.width = Math.min(up / 3 * 100, 100) + "%";
  $("downBar").style.width = Math.min(down / 3 * 100, 100) + "%";
  $("refineBtn").disabled = !(up >= 1 && down >= 1);
  $("threshSub").textContent = (up >= 3 && down >= 3) ? "Ready for a full sweep"
    : `Label ${Math.max(0, 3 - up)} more 👍 and ${Math.max(0, 3 - down)} more 👎 for a reliable fit (or sweep now with the suggested τ)`;
  $("stepRefineNum").classList.toggle("done", up > 0 || down > 0);
  $("stepCard-refine").classList.toggle("done", up > 0 || down > 0);
}
function setActiveStep(step) {
  ["refine", "thresh", "save"].forEach((s) => $("stepCard-" + s).classList.toggle("open", s === step));
}

/* ===================== threshold sweep (real data) ===================== */
async function openSweep() {
  if (!state.query) { showToast("Run a search first"); return; }
  state.sweepActive = true;
  // A confirmed τ stays put; otherwise let labels drive it.
  if (state.confirmedTau != null) { state.tempTau = state.confirmedTau; state.tauUserSet = true; }
  else { state.tempTau = null; state.tauUserSet = false; }
  $("sweepPanel").classList.add("open");
  setActiveStep("thresh");
  await refreshSweep();   // fetches the stratified sample + renders it in the grid
  $("sweepPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}
function closeSweep() {
  state.sweepActive = false;
  $("sweepPanel").classList.remove("open");
  $("resultsGrid").classList.remove("compact");
  renderGrid();
}

async function refreshSweep() {
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, mark: m.mark, index: m.index, row: m.row, segment_id: m.segment_id || "" }));
  const f = _searchFilters();
  try {
    const [dist, thr] = await Promise.all([
      fetch("/api/score_distribution", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, k: 2000, ...f, interval_mode: "k", interval_score: null }) }).then((r) => r.ok ? r.json() : null),
      fetch(_thresholdEndpoint(), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, marks, objective: "f1", min_precision: 0.9, val_fraction: 0.0, sample_size: 12, ...f }) }).then((r) => r.ok ? r.json() : null),
    ]);
    const hist = (dist && dist.edges) ? dist : (thr && thr.histogram) ? thr.histogram : null;
    if (!hist) { showToast("Could not compute distribution"); return; }
    state.suggestedTau = thr ? thr.suggested_threshold : null;
    const fitTau = thr ? thr.threshold : null;
    state.sweep = { edges: hist.edges, counts: hist.counts, total: hist.total, min: hist.min, max: hist.max, mean: hist.mean, up: (thr && thr.up_scores) || [], down: (thr && thr.down_scores) || [] };
    // Stratified boundary batch to label next (active learning). Fall back to the
    // current results if the backend returned none (e.g. before any labels).
    state.sweepSample = (thr && thr.sample && thr.sample.length) ? thr.sample : _dedupHits(state.hits);
    if (state.tempTau == null) state.tempTau = state.confirmedTau != null ? state.confirmedTau : (fitTau != null ? fitTau : (state.suggestedTau != null ? state.suggestedTau : hist.mean));
    if (state.sweepActive) renderGrid();
    drawSweep();
  } catch (e) { showToast("Sweep failed: " + e.message); }
}

function drawSweep() {
  const sw = state.sweep; if (!sw) return;
  // Ticks track the live labels (real per-clip scores from state.marks).
  const ms = _markScores(); sw.up = ms.up; sw.down = ms.down;
  const svg = $("sweepSvg");
  const W = 1000, H = 180, padL = 10, padR = 10, padB = 28;
  const lo = sw.edges[0], hi = sw.edges[sw.edges.length - 1], span = (hi - lo) || 1;
  const x = (s) => padL + ((s - lo) / span) * (W - padL - padR);
  const maxC = Math.max(1, ...sw.counts);
  const plotH = H - padB - 8;
  let bars = "";
  for (let i = 0; i < sw.counts.length; i++) {
    const h = (sw.counts[i] / maxC) * plotH, x0 = x(sw.edges[i]), x1 = x(sw.edges[i + 1]);
    bars += `<rect x="${x0.toFixed(1)}" y="${(H - padB - h).toFixed(1)}" width="${Math.max(0.5, x1 - x0 - 1).toFixed(1)}" height="${h.toFixed(1)}" fill="#26262e"/>`;
  }
  let ticks = "";
  sw.up.forEach((s) => { const cx = x(s); ticks += `<line x1="${cx}" y1="${H - padB}" x2="${cx}" y2="${H - padB - 118}" stroke="#c9f31d" stroke-width="2"/><circle cx="${cx}" cy="${H - padB - 118}" r="4" fill="#c9f31d"/>`; });
  sw.down.forEach((s) => { const cx = x(s); ticks += `<line x1="${cx}" y1="${H - padB}" x2="${cx}" y2="${H - padB - 96}" stroke="#ff6b5c" stroke-width="2"/><circle cx="${cx}" cy="${H - padB - 96}" r="4" fill="#ff6b5c"/>`; });
  const tauX = x(state.tempTau);
  const line = `<line x1="${tauX}" y1="${H - padB + 6}" x2="${tauX}" y2="6" stroke="#ececf0" stroke-width="2" stroke-dasharray="4 3"/><polygon points="${tauX - 6},2 ${tauX + 6},2 ${tauX},12" fill="#ececf0"/>`;
  const tk = (v) => x(v).toFixed(1);
  const axis = `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="#3a3a44" stroke-width="1"/>
    <text x="${tk(lo)}" y="${H - 8}" fill="#55555e" font-size="11" font-family="IBM Plex Mono">${lo.toFixed(2)}</text>
    <text x="${tk((lo + hi) / 2)}" y="${H - 8}" fill="#55555e" font-size="11" font-family="IBM Plex Mono" text-anchor="middle">${((lo + hi) / 2).toFixed(2)}</text>
    <text x="${tk(hi)}" y="${H - 8}" fill="#55555e" font-size="11" font-family="IBM Plex Mono" text-anchor="end">${hi.toFixed(2)}</text>`;
  svg.innerHTML = bars + axis + ticks + line;
  // Bind drag handlers ONCE on the <svg> (survives innerHTML redraws). Drag state +
  // value mapping read live from state.sweep, so the τ line tracks the pointer.
  if (!svg._sweepBound) {
    svg._sweepBound = true;
    svg.addEventListener("pointerdown", (e) => { _sweepDragging = true; try { svg.setPointerCapture(e.pointerId); } catch (_e) { } _sweepDragTo(e.clientX); e.preventDefault(); });
    svg.addEventListener("pointermove", (e) => { if (_sweepDragging) _sweepDragTo(e.clientX); });
    window.addEventListener("pointerup", () => { _sweepDragging = false; });
  }
  updateSweepStats();
}
let _sweepDragging = false;
function _sweepDragTo(clientX) {
  const sw = state.sweep; if (!sw) return;
  const svg = $("sweepSvg"), W = 1000, padL = 10, padR = 10;
  const lo = sw.edges[0], hi = sw.edges[sw.edges.length - 1], span = (hi - lo) || 1;
  const r = svg.getBoundingClientRect();
  const ux = (clientX - r.left) / r.width * W;
  state.tempTau = Math.max(lo, Math.min(hi, lo + ((ux - padL) / (W - padL - padR)) * span));
  state.tauUserSet = true;   // manual drag pins τ; labels no longer move it
  updateSweepStats();
  drawSweep();
}
function _clipsAtOrAbove(v) {
  const sw = state.sweep; let c = 0;
  for (let i = 0; i < sw.counts.length; i++) {
    const e0 = sw.edges[i], e1 = sw.edges[i + 1];
    if (e1 <= v) continue;
    if (e0 >= v) { c += sw.counts[i]; continue; }
    c += sw.counts[i] * (e1 - v) / (e1 - e0);
  }
  return Math.round(c);
}
function updateSweepStats() {
  const sw = state.sweep, t = state.tempTau;
  $("statCapture").textContent = `${sw.up.filter((s) => s >= t).length} / ${sw.up.length} 👍`;
  $("statExclude").textContent = `${sw.down.filter((s) => s < t).length} / ${sw.down.length} 👎`;
  // Fraction of the corpus score-density at or above τ. A percentage (not an absolute
  // count) because the offline scan's final number diverges from any clip estimate --
  // it dedups to segments, merges intervals, and applies downsample filters. The
  // percentage is the honest, scan-invariant signal of how selective this τ is.
  const above = _clipsAtOrAbove(t);
  const pct = sw.total ? (above / sw.total) * 100 : 0;
  const shown = pct === 0 ? "0%" : pct < 0.1 ? "<0.1%" : pct.toFixed(pct < 10 ? 1 : 0) + "%";
  $("statCorpus").textContent = shown;
  $("tauReadout").textContent = t.toFixed(3);
}
function confirmThreshold() {
  if (state.tempTau == null) return;
  state.confirmedTau = state.tempTau;
  state.tauUserSet = true;
  $("threshVal").textContent = state.confirmedTau.toFixed(3);
  $("threshSummary").textContent = `τ ${state.confirmedTau.toFixed(3)}`;
  $("threshSub").textContent = "Set from sweep";
  $("stepThreshNum").classList.add("done");
  $("stepCard-thresh").classList.add("done");
  closeSweep();
  setActiveStep("save");
  showToast(`Threshold set at τ = ${state.confirmedTau.toFixed(3)}`);
}

/* ===================== save drawer ===================== */
function openSaveDrawer() {
  if (!state.query) { showToast("Run a search first"); return; }
  const { up, down } = _counts();
  const tau = state.confirmedTau != null ? state.confirmedTau.toFixed(3) : (state.suggestedTau != null ? state.suggestedTau.toFixed(3) + " (suggested)" : "—");
  $("vecSummaryBox").innerHTML = `query <b>"${escapeHtml(state.query)}"</b><br>vector <b>${state.mode === "refine" ? "refined" : (state.mode === "resume" ? "resumed" : "text")}</b> · votes <b>${up}👍 / ${down}👎</b><br>threshold <b>τ ${tau}</b>`;
  if (!$("tagInput").value.trim()) $("tagInput").value = _querySlug(state.query);
  setNote("saveNote", "");
  $("overlay").classList.add("open"); $("drawer").classList.add("open");
}
function closeDrawer() { $("overlay").classList.remove("open"); $("drawer").classList.remove("open"); }
function _querySlug(q) { return (q || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 40); }

async function finalSave() {
  const tag = $("tagInput").value.trim();
  if (!tag) { setNote("saveNote", "Enter a tag name.", true); return; }
  const tau = state.confirmedTau != null ? state.confirmedTau : (state.suggestedTau != null ? state.suggestedTau : 0);
  const btn = $("finalSaveBtn"); btn.disabled = true; setNote("saveNote", "Saving…");
  try {
    // Persist the 👍/👎 marks with the saved search so Resume can restore them
    // (session-restore reads these back into state.marks). Shape matches the resume
    // reader: {chunk_id, segment_id, index, rank, score} per mark.
    const _mark = (chunk_id, m) => ({ chunk_id, segment_id: m.segment_id, index: m.index, rank: m.rank, score: m.score });
    const thumbs_up = Object.entries(state.marks).filter(([, m]) => m.mark === "up").map(([c, m]) => _mark(c, m));
    const thumbs_down = Object.entries(state.marks).filter(([, m]) => m.mark === "down").map(([c, m]) => _mark(c, m));
    const r = await fetch("/api/save_vector", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      tag, query: state.query, k: 50, threshold: tau,
      from_date: $("sf-dateFrom").value || null, to_date: $("sf-dateTo").value || null,
      segment_set_uuid: state.searchSegUuid, segment_set_name: state.searchSegName,
      filter_lance_uri: _splitList($("sf-lance").value), vehicle: _splitList($("sf-vehicle").value), drive_id: _splitList($("sf-drive").value),
      thumbs_up, thumbs_down,
    }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
    await r.json();
    closeDrawer();
    showToast(`Saved "${tag}" — configure export below`);
    await loadSaved();
    setPage("saved");
    const idx = state.savedRows.findIndex((e) => e.tag === tag);
    if (idx >= 0) { state.selected = new Set([idx]); renderTable(); renderExportPanel(); const row = $("tbody").querySelector(`tr[data-i="${idx}"]`); if (row) row.classList.add("new-row"); }
  } catch (e) { setNote("saveNote", "Save failed: " + e.message, true); }
  finally { btn.disabled = false; }
}

/* ===================== saved searches table ===================== */
function _uriShort(u) { const p = String(u || "").replace(/\/+$/, "").split("/").filter(Boolean); return p.length ? p[p.length - 1] : ""; }
function _fmtDates(f) { const from = (f && f.from_date) || "", to = (f && f.to_date) || ""; return (from || to) ? `${from || "…"} → ${to || "latest"}` : "—"; }

async function loadSaved() {
  try { const d = await fetch("/api/tags_catalog").then((r) => r.json()); state.savedRows = d.entries || []; }
  catch (e) { state.savedRows = []; }
  renderTable();
  renderExportPanel();
}
function _rowModel(e) { return e.model_label || _uriShort(e.model_uri) || "base"; }
// Populate the model dropdown: loaded model first (default selection), then the
// other models present in history, then "All models". Rebuilt only when the set
// changes, preserving the user's current pick.
function _syncModelFilter(all) {
  const sel = $("modelFilter");
  const loaded = state.corpus ? state.corpus.model : null;
  const models = [...new Set(all.map(_rowModel))].filter(Boolean).sort();
  const sig = (loaded || "") + "::" + models.join("|");
  if (sel._sig === sig) return;
  sel._sig = sig;
  const seen = new Set(); const opts = [];
  if (loaded) { opts.push(`<option value="${escapeHtml(loaded)}">${escapeHtml(loaded)} (loaded)</option>`); seen.add(loaded); }
  models.forEach((m) => { if (!seen.has(m)) { opts.push(`<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`); seen.add(m); } });
  opts.push(`<option value="__all__">All models</option>`);
  const prev = sel.value;
  sel.innerHTML = opts.join("");
  if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;   // keep user's pick
  else sel.value = (loaded && seen.has(loaded)) ? loaded : "__all__";             // default: loaded model
}
function renderTable() {
  const all = state.savedRows || [];
  const q = ($("savedFilter").value || "").trim().toLowerCase();
  _syncModelFilter(all);
  const modelSel = $("modelFilter").value;
  const matched = all.map((e, i) => ({ e, i }))
    .filter(({ e }) => modelSel === "__all__" || _rowModel(e) === modelSel);
  const hidden = all.length - matched.length;
  $("savedCount").textContent = matched.length ? `(${matched.length})` : "";
  const body = matched.map(({ e, i }) => {
    const hay = (e.tag + " " + (e.query || "")).toLowerCase();
    if (q && !hay.includes(q)) return "";
    const model = e.model_label || _uriShort(e.model_uri) || "base";
    const dim = e.vec_dim ? `${e.vec_dim}-d` : "no vec";
    const tau = (e.threshold > 0) ? e.threshold.toFixed(3) : "top-k";
    return `<tr data-i="${i}" class="${state.selected.has(i) ? "checked" : ""}">
      <td><input type="checkbox" data-i="${i}" ${state.selected.has(i) ? "checked" : ""}></td>
      <td><span class="tag-pill">${escapeHtml(e.tag)}</span></td>
      <td><div class="query-text">${escapeHtml(e.query && e.query !== e.tag ? e.query : "")}</div></td>
      <td class="model-cell"><b>${escapeHtml(model)}</b><br>${dim}</td>
      <td class="date-cell">${escapeHtml(_fmtDates(e.filters))}</td>
      <td><span class="kt-pill numeric">τ ${tau}</span></td>
      <td>${e.id != null ? `<button class="resume-link" data-id="${e.id}">resume ↗</button>` : ""}</td>
    </tr>`;
  }).join("");
  let html = body;
  if (!body) html = `<tr><td colspan="7" style="color:var(--muted-2)">${all.length ? "No saved searches for this model / filter — pick another model above." : "No saved searches yet — run a search and Save."}</td></tr>`;
  if (hidden > 0 && modelSel !== "__all__") html += `<tr><td colspan="7" style="color:var(--muted-2);font-size:12px;">${hidden} search${hidden === 1 ? "" : "es"} from other models hidden — choose “All models” above to see them.</td></tr>`;
  $("tbody").innerHTML = html;
}
async function resumeSession(id) {
  setPage("search"); showToast("Loading saved search…");
  let s;
  try { s = await fetch("/api/search_session/" + encodeURIComponent(id)).then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }); }
  catch (e) { showToast("Could not load: " + e.message); return; }
  if (!s.vector || !s.vector.length) { showToast("That saved search has no stored vector"); return; }
  if (s.embeddings_uri && s.embeddings_uri !== state.embeddingsUri) { $("embeddings-uri").value = s.embeddings_uri; await loadCorpus(); }
  $("searchInput2").value = s.query || ""; state.query = s.query || "";
  $("tagInput").value = s.tag || "";
  state.marks = {};
  (s.thumbs_up || []).forEach((m) => { if (m && m.chunk_id) state.marks[m.chunk_id] = { mark: "up", segment_id: m.segment_id, index: m.index, rank: m.rank, score: m.score }; });
  (s.thumbs_down || []).forEach((m) => { if (m && m.chunk_id) state.marks[m.chunk_id] = { mark: "down", segment_id: m.segment_id, index: m.index, rank: m.rank, score: m.score }; });
  if (s.from_date) $("sf-dateFrom").value = s.from_date;
  if (s.to_date) $("sf-dateTo").value = s.to_date;
  $("sf-lance").value = s.filter_lance_uri || ""; $("sf-vehicle").value = s.vehicle || ""; $("sf-drive").value = s.drive_id || "";
  state.searchSegUuid = s.segment_set_uuid || null; state.searchSegName = s.segment_set_name || null;
  $("sf-dxset").value = s.segment_set_name || "";
  if (s.segment_set_uuid) fetch("/api/segment_set_prefetch?uuid=" + encodeURIComponent(s.segment_set_uuid)).catch(() => {});
  state.resumeVec = s.vector; state.resumeLabel = s.query || ""; state.mode = "resume";
  updateRail();
  await runVectorSearch({ page: 0 });
  showToast(`Resumed "${s.tag || "search"}"`);
}

/* ===================== export panel ===================== */
function estimateTauFromK(k) {
  const n = (state.corpus && state.corpus.num_rows) || 2260540;
  const ratio = Math.max(1e-6, Math.min(0.399, k / (n * 0.4)));
  return Math.max(0, Math.min(0.42, 0.05 + 0.25 * Math.sqrt(-Math.log(ratio))));
}
function setCutoffMode(mode) {
  state.cutoffMode = mode;
  $("cutoffSeg").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  renderCutoffField();
}
function renderCutoffField() {
  const box = $("cutoffField");
  if (state.cutoffMode === "threshold") {
    box.innerHTML = `<span style="font-size:11.5px;color:var(--muted-2);">Exports each selected search at its own saved τ (capped at k).</span>`;
    return;
  }
  const rows = [...state.selected].map((i) => state.savedRows[i]).filter(Boolean);
  if (rows.length > 1) {
    // Per-search Top-K: one K input per selected tag (backend takes a per-query k).
    const items = rows.map((e) => {
      const k = state.perTagK[e.tag] != null ? state.perTagK[e.tag] : (e.k || 50);
      return `<div class="pertag-k-row"><span class="pertag-k-tag" title="${escapeHtml(e.tag)}">${escapeHtml(e.tag)}</span>`
        + `<input class="pertag-k-input" data-tag="${escapeHtml(e.tag)}" value="${escapeHtml(String(k))}" inputmode="numeric"></div>`;
    }).join("");
    box.innerHTML = `<div class="pertag-k-head"><label>K per search</label>`
      + `<span class="pertag-k-all">set all <input id="kAllInput" placeholder="K" inputmode="numeric"></span></div>`
      + `<div class="pertag-k-list">${items}</div>`;
    box.querySelectorAll(".pertag-k-input").forEach((inp) => {
      inp.oninput = () => { state.perTagK[inp.dataset.tag] = inp.value; };
    });
    const all = $("kAllInput");
    if (all) all.oninput = () => {
      box.querySelectorAll(".pertag-k-input").forEach((inp) => {
        inp.value = all.value; state.perTagK[inp.dataset.tag] = all.value;
      });
    };
  } else {
    box.innerHTML = `<label>K =</label><input id="kInput" value="50"><span id="kConversion" style="font-size:11.5px;color:var(--muted-2);font-family:var(--mono);"></span>`;
    $("kInput").oninput = updateKConversion; updateKConversion();
  }
}
// Top-K for one selected saved-search row. With >1 tag selected the per-tag
// input wins (default = the tag's saved k, matching what the input shows);
// with a single tag it's the global K input. Falls back to 50.
function kForRow(e) {
  if (state.selected.size > 1) {
    const per = parseInt(state.perTagK[e.tag], 10);
    if (Number.isFinite(per) && per > 0) return per;
    const saved = parseInt(e.k, 10);
    return Number.isFinite(saved) && saved > 0 ? saved : 50;
  }
  const g = parseInt(($("kInput") || {}).value, 10);
  return Number.isFinite(g) && g > 0 ? g : 50;
}
function updateKConversion() {
  const k = parseInt($("kInput").value, 10) || 0;
  $("kConversion").innerHTML = `≈ τ <span style="color:var(--accent);font-weight:600;">${estimateTauFromK(k).toFixed(3)}</span> <span style="color:var(--muted-2);">(exports by rank K)</span>`;
}
function setSampleMode(mode) {
  state.sampleMode = mode;
  $("sampleSeg").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  renderSampleMode();
}
function renderSampleMode() {
  const seg = state.sampleMode === "segment";
  $("sampleDesc").textContent = seg
    ? "Fixed 30s, non-overlapping — one row per segment by definition."
    : "Uses the underlying 8s granularity to derive variable-length ranges — adjacent/overlapping matches merge into one interval.";
  $("dedupSub").textContent = seg
    ? "Not applicable — segment mode already produces one row per segment."
    : "Merged intervals can still span segments — dedup keeps one row per segment.";
  $("dedupSub").classList.toggle("disabled-note", seg);
  $("dedupSwitch").classList.toggle("locked", seg);
  $("dedupInput").disabled = seg;
  if (seg) $("dedupInput").checked = true;
}
function renderExportPanel() {
  const offline = state.offlineScan;
  const badge = $("mechBadge");
  badge.textContent = offline ? "async" : "instant";
  badge.className = "mech-badge " + (offline ? "async" : "instant");
  const n = state.selected.size;
  const note = $("selectionNote");
  if (n === 0) { note.className = "selection-note empty"; note.textContent = "No searches selected — check one or more rows above to export."; }
  else {
    const tags = [...state.selected].map((i) => state.savedRows[i] && state.savedRows[i].tag).filter(Boolean);
    note.className = "selection-note";
    note.innerHTML = `Exporting <b>${n}</b> search${n === 1 ? "" : "es"}: ${tags.map((t) => `<b>${escapeHtml(t)}</b>`).join(", ")}`;
  }
  const btn = $("exportBtn");
  btn.disabled = n === 0;
  if (offline) { btn.textContent = "🚀 Export"; btn.className = "btn-export async"; $("exportNote").textContent = "Launches an offline scan — you'll get a link when it's ready."; }
  else { btn.textContent = "⬇ Export"; btn.className = "btn-export instant"; $("exportNote").textContent = "Downloads a CSV immediately."; }
  $("jobStatus").classList.remove("show");
  renderCutoffField();
  renderSampleMode();
}
function _exportFilters() {
  return {
    from_date: $("ex-dateFrom").value || null, to_date: $("ex-dateTo").value || null,
    filter_lance_uri: _splitList($("ex-lance").value), vehicle: _splitList($("ex-vehicle").value), drive_id: _splitList($("ex-drive").value),
    segment_set_uuid: state.exportSegUuid, segment_set_name: state.exportSegName,
  };
}
function _exportFilename(resp, fb) { const n = resp.headers.get("X-NLS-Export-Name") || ""; return n ? n + ".csv" : fb; }

function doExport() {
  if (state.selected.size === 0) return;
  if (_fullCorpusOn()) fullExport(); else if (state.offlineScan) launchScan(); else downloadCsv();
}
// Threshold fitting follows whichever corpus the results came from. The two
// endpoints address marks differently -- the resident one by `index`, the
// full-corpus one by `row` -- and sending full-corpus marks to the resident
// endpoint silently drops every label, which reads as "no labels yet".
function _thresholdEndpoint() {
  return _fullCorpusOn() ? "/api/full_threshold" : "/api/threshold_search";
}

// Export straight from the resident 34M-row corpus. This is the online
// replacement for the Lilypad scan: the ranking pass is the same one a search
// runs, so the only extra cost is materializing the rows.
async function fullExport() {
  const rows = [...state.selected].map((i) => state.savedRows[i]).filter(Boolean);
  if (!rows.length) return;
  const topk = state.cutoffMode === "topk";
  const btn = $("exportBtn"); btn.disabled = true;
  const note = $("exportNote");
  const filters = _exportFilters();
  let done = 0;
  try {
    for (const e of rows) {
      const tag = e.tag || e.query;
      note.textContent = `Exporting ${tag} from all 34M clips… (${done + 1}/${rows.length})`;
      const body = {
        query: e.query || e.tag, tag,
        interval: state.sampleMode === "interval",
        dedupe_segment: $("dedupInput").checked,
        create_segment_set: $("segsetInput").checked,
        exact: !!$("exactInput") && $("exactInput").checked,
        from_date: filters.from_date, to_date: filters.to_date,
        vehicle: filters.vehicle, drive_id: filters.drive_id,
        segment_set_uuid: filters.segment_set_uuid,
        filter_lance_uri: filters.filter_lance_uri,
      };
      if (topk) body.k = kForRow(e) || 50;
      else body.threshold = e.threshold > 0 ? e.threshold : 0.3;
      const resp = await fetch("/api/full_export", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        let d; try { d = (await resp.json()).detail; } catch (_e) { }
        throw new Error(`${tag}: ${d || "HTTP " + resp.status}`);
      }
      const h = (k) => resp.headers.get(k) || "";
      const blob = await resp.blob(); const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url;
      a.download = _exportFilename(resp, tag + ".csv");
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
      done++;
      const bits = [`${fmtInt(+h("X-NLS-Rows"))} rows`, `${h("X-NLS-Elapsed-Ms")} ms`];
      if (h("X-NLS-Parquet")) bits.push(`parquet → ${h("X-NLS-Parquet")}`);
      else bits.push("⚠ parquet not written");
      if (h("X-NLS-Segset")) bits.push(`segment set ${h("X-NLS-Segset")}`);
      if (h("X-NLS-Segset-Error")) bits.push(`⚠ ${h("X-NLS-Segset-Error")}`);
      // A capped threshold export is a partial answer; say so on the same line
      // as the row count rather than letting it read as the complete set.
      if (h("X-NLS-Truncated") === "1") {
        bits.push(`⚠ capped — ${fmtInt(+h("X-NLS-Candidates"))} matched`);
      }
      if (h("X-NLS-Score-Kind") === "bounded_approx") {
        bits.push(`scores ±${h("X-NLS-Score-Error-Bound")}`);
      }
      note.textContent = `${tag}: ` + bits.join(" · ");
    }
    showToast(`Exported ${done} tag(s) from the full corpus`);
  } catch (err) {
    note.textContent = "Export failed: " + err.message;
  } finally { btn.disabled = false; }
}

async function downloadCsv() {
  const rows = [...state.selected].map((i) => state.savedRows[i]).filter(Boolean);
  const topk = state.cutoffMode === "topk";
  // Top-K uses a per-search K (kForRow: the tag's own input when >1 selected,
  // else the single global K); threshold mode keeps each tag's saved k cap.
  const queries = rows.map((e) => ({ query: e.query || e.tag, k: topk ? kForRow(e) : (e.k || 50), threshold: topk ? 0 : (e.threshold || 0) }));
  const btn = $("exportBtn"); btn.disabled = true; $("exportNote").textContent = `Exporting ${queries.length} tag(s) from the loaded corpus…`;
  try {
    const resp = await fetch("/api/export_config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      queries, dedupe: true, dedupe_segment: $("dedupInput").checked, create_segment_set: $("segsetInput").checked, embeddings_uri: state.embeddingsUri, ..._exportFilters(),
    }) });
    if (!resp.ok) { let d; try { d = (await resp.json()).detail; } catch (_e) { } throw new Error(d || ("export " + resp.status)); }
    const parquet = resp.headers.get("X-NLS-Parquet") || "";
    const blob = await resp.blob(); const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = _exportFilename(resp, "export.csv");
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    $("exportNote").textContent = `Downloaded CSV for ${queries.length} tag(s)` + (parquet ? ` · parquet → ${parquet}` : " · ⚠ parquet not written");
    showToast("CSV downloaded");
  } catch (e) { $("exportNote").textContent = "Export failed: " + e.message; }
  finally { btn.disabled = false; }
}
async function launchScan() {
  const rows = [...state.selected].map((i) => state.savedRows[i]).filter(Boolean);
  const tags = rows.map((e) => e.tag);
  const defThr = 0.3;
  const thresholds = {}; rows.forEach((e) => { thresholds[e.tag] = e.threshold > 0 ? e.threshold : defThr; });
  // Top-K offline scan: only when a segment set / lance downsample scopes the scan (the
  // worker ranks within that set). Otherwise Top-K is meaningless offline -- guide the user.
  const topk = state.cutoffMode === "topk";
  const hasScope = !!(state.exportSegUuid || ($("ex-lance") && ($("ex-lance").value || "").trim()));
  if (topk && !hasScope) {
    const jsg = $("jobStatus"); jsg.classList.add("show");
    jsg.innerHTML = `<span style="color:var(--neg)">Top-K needs a segment set (or lance downsample) to rank within — pick one under Filters, or switch Cutoff to Threshold.</span>`;
    return;
  }
  const topK = topk ? (parseInt($("kInput").value, 10) || 50) : null;
  const btn = $("exportBtn"); btn.disabled = true; btn.textContent = "⏳ Job queued…";
  const js = $("jobStatus"); js.classList.add("show"); js.innerHTML = `<span>launching per-segment scan over ${tags.length} tag(s)…</span>`;
  try {
    const r = await fetch("/api/launch_segment_scan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      tags, thresholds, default_threshold: defThr, create_segment_set: $("segsetInput").checked, merge_intervals: state.sampleMode === "interval", ..._exportFilters(),
      ...(topK ? { top_k: topK } : {}),
    }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || ("HTTP " + r.status));
    const { execution_id, url } = await r.json();
    js.innerHTML = `<span>job <b><a href="${url}" target="_blank" rel="noopener">${escapeHtml(execution_id)}</a></b> launched</span><span>polling…</span>`;
    showToast("Scan launched — see Recent scans");
    loadScanJobs();
    pollScan(execution_id, url);
  } catch (e) { js.innerHTML = `<span style="color:var(--neg)">launch failed: ${escapeHtml(e.message)}</span>`; }
  finally { btn.disabled = false; renderExportPanel(); }
}
async function pollScan(execId, url) {
  for (let i = 0; i < 90; i++) {
    await new Promise((res) => setTimeout(res, 20000));
    let s; try { s = await fetch("/api/scan_status?execution=" + encodeURIComponent(execId)).then((r) => r.json()); } catch (e) { continue; }
    loadScanJobs();
    if (s.done) return;
  }
}

/* ===================== recent scans ===================== */
function _scanStatusClass(st) { const t = (st || "").toUpperCase(); if (/SUCCEEDED|COMPLETED/.test(t)) return "succeeded"; if (/FAILED|STOPPED/.test(t)) return "failed"; return "queued"; }
let _scansRepollTimer = null;
async function loadScanJobs(live) {
  if (!state.offlineScan) return;
  clearTimeout(_scansRepollTimer);
  const body = $("scansBody");
  if (body && !(state.scanJobs && state.scanJobs.length)) {
    body.innerHTML = `<tr><td colspan="8" style="color:var(--muted-2)">Loading recent scans…</td></tr>`;
  }
  // The list is a pure DB read server-side (never blocks on Lilypad/DORA/OCI); live=1
  // (Reload) forces the server's background refresher to kick. The timeout is a belt --
  // if anything is ever slow, the panel shows an error instead of an endless spinner.
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000);
  let refreshing = false;
  try {
    const r = await fetch("/api/scans?limit=100" + (live ? "&live=1" : ""), { signal: ctrl.signal });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    state.scanJobs = d.jobs || [];
    refreshing = !!d.refreshing;
  } catch (e) {
    console.error("loadScanJobs failed", e);
    const msg = e.name === "AbortError" ? "timed out" : e.message;
    if (body) body.innerHTML = `<tr><td colspan="8" style="color:var(--neg)">Failed to load recent scans: ${escapeHtml(msg)}</td></tr>`;
    return;
  } finally { clearTimeout(timer); }
  renderScans();
  // Converge to live truth: while the server is refreshing rows in the background (or
  // jobs are still in flight), re-poll the cheap list until everything is settled.
  const active = (state.scanJobs || []).some((j) => !/SUCCEEDED|COMPLETED|FAILED|ABORTED/.test((j.status || "").toUpperCase()));
  if ((refreshing || active) && $("page-saved").classList.contains("active")) {
    _scansRepollTimer = setTimeout(() => loadScanJobs(false), 7000);
  }
}
function renderScans() {
  const jobs = state.scanJobs || [];
  $("scansBody").innerHTML = jobs.length ? jobs.map((j) => {
    const id = j.console_url ? `<a class="scan-name" href="${j.console_url}" target="_blank" rel="noopener">${escapeHtml(j.execution_id)}</a>` : `<span class="scan-name">${escapeHtml(j.execution_id)}</span>`;
    const topK = j.filters && j.filters.top_k;
    // Top-K mode ranks within the scope and IGNORES the per-tag thresholds, so don't
    // print "@tau" for those scans -- show a top-K=N badge instead (else it reads as a
    // threshold scan). Threshold scans keep the per-tag "@tau" suffix.
    const tagList = (j.tags || []).map((t) => escapeHtml(t) + (!topK && j.thresholds && j.thresholds[t] != null ? ` @${j.thresholds[t]}` : "")).join(", ");
    const modeBadge = topK ? `<span class="scan-mode-topk" title="Top-K ranking within the downsampled scope (per-tag thresholds ignored)">top-K=${topK}</span> ` : "";
    const filt = _fmtScanFilters(j.filters);
    const out = j.lance_uri
      ? `<div class="scan-output"><span class="scan-uri" title="${escapeHtml(j.lance_uri)}">${escapeHtml(j.lance_uri)}</span><button class="scan-copy" data-uri="${escapeHtml(j.lance_uri)}" title="Copy full path">copy</button></div>`
      : "—";
    let dx = `<span class="scan-dx none">—</span>`;
    if (j.segset_label) dx = `<span class="scan-dx">${escapeHtml(j.segset_label)}</span>`;
    else if (j.register_segset) dx = `<span class="scan-dx none">pending…</span>`;
    return `<tr>
      <td class="scan-time">${escapeHtml(j.created_at || "")}</td>
      <td>${id}</td>
      <td class="scan-tags">${(j.tags || []).length} tag(s): ${modeBadge}${tagList}</td>
      <td class="scan-filters">${escapeHtml(filt)}</td>
      <td><span class="status-pill ${_scanStatusClass(j.status)}">${escapeHtml(j.status || "—")}</span></td>
      <td class="scan-counts">${_fmtScanCounts(j.counts, j.status)}</td>
      <td>${out}</td>
      <td>${dx}</td></tr>`;
  }).join("") : `<tr><td colspan="8" style="color:var(--muted-2)">No scans launched yet.</td></tr>`;
}
// Result counts (total segments + per-tag breakdown), mirroring the Data Explorer view.
function _fmtScanCounts(c, status) {
  if (!c) return /SUCCEEDED|COMPLETED/.test((status || "").toUpperCase()) ? `<span class="scan-dx none">…</span>` : `<span class="scan-dx none">—</span>`;
  const total = c.num_segments != null ? c.num_segments.toLocaleString() : "?";
  const unit = c.per_tag_is_segments ? "segments" : "intervals";
  const perTag = c.per_tag || {};
  const rows = Object.keys(perTag).sort((a, b) => (perTag[b] || 0) - (perTag[a] || 0))
    .map((t) => `<div class="ct-row"><span class="ct-tag">${escapeHtml(t)}</span><span class="ct-n">${(perTag[t] || 0).toLocaleString()}</span></div>`).join("");
  const clips = c.num_clips_scanned != null ? ` <span class="ct-sub">of ${c.num_clips_scanned.toLocaleString()} clips</span>` : "";
  return `<div class="scan-counts-box"><div class="ct-total"><b>${total}</b> segments${clips}</div>`
    + (rows ? `<div class="ct-list" title="${unit} per tag">${rows}</div>` : "") + `</div>`;
}
// Copy a scan's full output Lance URI (delegated so it survives re-renders).
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".scan-copy");
  if (!btn) return;
  const uri = btn.dataset.uri || "";
  navigator.clipboard.writeText(uri).then(
    () => showToast("Copied output path"),
    () => showToast("Copy failed — select the path manually"),
  );
});
function _fmtScanFilters(f) {
  if (!f) return "—"; const p = [];
  if (f.from_date || f.to_date) p.push(`${f.from_date || "…"}→${f.to_date || "latest"}`);
  if (f.segment_set_name || f.segment_set_uuid) p.push("seg: " + (f.segment_set_name || f.segment_set_uuid));
  if (f.vehicle) p.push("veh: " + f.vehicle);
  if (f.drive_id) p.push("drive: " + f.drive_id);
  if (f.filter_lance_uri) p.push("lance: " + _uriShort(f.filter_lance_uri));
  return p.join(" · ") || "—";
}

/* ===================== misc ===================== */
function setNote(id, html, isErr) { const n = $(id); if (!n) return; n.innerHTML = html; n.classList.toggle("warn", !!isErr); }
let _toastTimer = null;
function showToast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(_toastTimer); _toastTimer = setTimeout(() => t.classList.remove("show"), 3000); }

init();
