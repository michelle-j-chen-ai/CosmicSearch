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
  health: null,
  project: localStorage.getItem("nls.project") || null,
  corpus: null,
  resumeTag: null, resumeQuery: "",
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
  // threshold
  tempTau: null, confirmedTau: null, suggestedTau: null,
  tauUserSet: false,         // true once the user drags/confirms τ (then labels stop moving it)
  sweep: null,               // {edges,counts,total,min,max,mean, up[], down[]}
  sweepActive: false,        // grid shows the stratified boundary sample while sweeping
  sweepSample: [],           // active-learning batch from /api/threshold_search
  rendered: [],              // clips currently in the grid (for vote lookup)
  // saved searches
  searchSegUuid: null, searchSegName: null,
  exportSegUuid: null, exportSegName: null,
  savedRows: [], selected: new Set(),
  scanJobs: [],
  cutoffMode: "threshold",
  perTagK: {},               // tag -> top-K, for per-search K in a multi-tag Top-K export
  sampleMode: "interval",
};
let _issueSeq = 0;

// Every /api call carries the project it addresses: as a query parameter, and
// in the JSON body when there is one, so GET and POST routes read it the same way.
function apiFetch(url, opts) {
  const project = state.project;
  if (project && typeof url === "string" && (url.startsWith("/api/") || url.startsWith("/ui/")) && !/[?&]project=/.test(url)) {
    url += (url.includes("?") ? "&" : "?") + "project=" + encodeURIComponent(project);
  }
  if (project && opts && typeof opts.body === "string") {
    try {
      const b = JSON.parse(opts.body);
      if (b && typeof b === "object" && !Array.isArray(b) && b.project == null) {
        b.project = project;
        opts = { ...opts, body: JSON.stringify(b) };
      }
    } catch (e) { /* not a JSON body */ }
  }
  return fetch(url, opts);
}

/* ===================== bootstrap ===================== */
async function init() {
  wireEvents();
  await loadPlatform();
  loadDefaultCorpusWhenReady();
  loadSaved();
}

async function loadPlatform() {
  try {
    const h = await fetch("/api/v1/health").then((r) => r.json());
    state.health = h;
    const names = Object.keys(h.projects || {});
    const def = names.includes("neuron") ? "neuron" : names[0];
    if (!state.project || !names.includes(state.project)) state.project = def;
    localStorage.setItem("nls.project", state.project);
    const sel = $("projectSelect");
    sel.innerHTML = names.map((n) =>
      `<option value="${escapeHtml(n)}"${n === state.project ? " selected" : ""}>${escapeHtml(n.toUpperCase())}</option>`).join("");
    sel.className = "tag " + state.project;
    sel.disabled = names.length < 2;
    // Every view is scoped to one corpus, so switching restarts the app on the other.
    sel.onchange = () => { localStorage.setItem("nls.project", sel.value); location.reload(); };
  } catch (e) { /* cosmetic */ }
  $("scansSection").style.display = "block";
  renderExportPanel();
}

// /api/v1/health is the one status call: which projects exist, whether this
// one's corpus is resident, and what it covers.
async function _health() {
  const r = await fetch("/api/v1/health");
  const h = await r.json().catch(() => null);
  if (h) state.health = h;
  return h;
}
function _projectHealth(h) { return (h && h.projects && h.projects[state.project]) || null; }
function _corpusFromHealth(p) {
  return {
    num_rows: p.rows, corpus_version: p.corpus_version, embeddings_uri: p.corpus_table_uri || "",
    span_lo_date: p.date_span ? p.date_span[0] : "", span_hi_date: p.date_span ? p.date_span[1] : "",
    model: (state.health && state.health.model && state.health.model.id) || "black_dwarf", has_segment_id: true,
  };
}
function applyCorpus(c) {
  state.corpus = c;
  state.embeddingsUri = c.embeddings_uri;
  state.scoreHi = 0.4;
  $("corpusPill").innerHTML = `🗂 <b>${fmtInt(c.num_rows)} clips</b> · ${escapeHtml(c.model || "model")}`;
  // The pill reported the resident browse corpus even while searches ran against
  // the full one, so it read as though the app only held ~2M clips. Correct it
  // whenever full-corpus mode is the active path.
  wireDxCombo();
  refreshCorpusPill();
  if ($("embeddings-uri")) $("embeddings-uri").value = c.embeddings_uri || "";
  if ($("model-uri") && c.model_uri !== undefined) $("model-uri").value = c.model_uri || "";
  // Search filters track the LOADED (in-app browse) corpus -- you're searching it, so
  // default its date range to that corpus's span.
  { const f = $("sf-dateFrom"); if (f) { f.value = c.span_lo_date; f.min = c.span_lo_date; f.max = c.span_hi_date; }
    const t = $("sf-dateTo"); if (t) { t.value = c.span_hi_date; t.min = c.span_lo_date; t.max = c.span_hi_date; } }
  // Export runs over the full corpus, so its dates default to UNBOUNDED (blank =
  // whole corpus); don't pin them to the loaded span or the export is silently
  // narrowed.
  ["ex-dateFrom", "ex-dateTo"].forEach((id) => { const el = $(id); if (el) { el.value = ""; el.removeAttribute("min"); el.removeAttribute("max"); } });
}

async function loadDefaultCorpusWhenReady() { await refreshCorpusPill(); }
/* ===================== event wiring ===================== */
function wireEvents() {
  $("navSearch").onclick = () => setPage("search");
  $("navSaved").onclick = () => setPage("saved");
  $("settingsGear").onclick = () => $("settingsPop").classList.toggle("hidden");

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
  $("scansReload").onclick = () => loadScanJobs(true);
  $("tbody").onclick = (e) => {
    const b = e.target.closest("button.resume-link");
    if (b && b.dataset.tag) resumeTag(b.dataset.tag);
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
    loadScanJobs();
  }
}

// Reports the pinned corpus; there is nothing to switch to. The server rejects
// a `uri` that is not the pinned one, so offering the choice here would only
// produce an error the user cannot act on.
/* ===================== search-page filters ===================== */
function toggleSearchFilters() {
  const p = $("searchFiltersPanel");
  p.style.display = p.style.display === "none" ? "block" : "none";
}
// Data Explorer segment-set combobox: type -> live DORA search -> dropdown -> pick.
// A factory so both the Search filter and the Saved-searches (export) filter get
// the same type-ahead dropdown, each writing to its own state.
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
        items = (await apiFetch("/ui/segment_sets?name_filter=" + encodeURIComponent(v)).then((r) => { if (!r.ok) throw new Error("DORA " + r.status); return r.json(); })) || [];
        if (!items.length) { menu.innerHTML = '<div class="combo-msg">no sets match that name</div>'; show(); if (note) note.textContent = "no sets match that name"; return; }
        menu.innerHTML = items.map((s, i) => `<div class="combo-opt" data-i="${i}" role="option"><span class="opt-name">${escapeHtml(s.name)}</span><span class="opt-meta">v${escapeHtml(String(s.version))} · ${fmtInt(s.num_segments)} segments</span></div>`).join("");
        menu.querySelectorAll(".combo-opt").forEach((o) => o.addEventListener("mousedown", (e) => {
          e.preventDefault();
          const s = items[parseInt(o.dataset.i, 10)];
          inp.value = `${s.name} v${s.version}`;
          combo.classList.add("chosen");
          if (note) note.textContent = `using ${s.name} v${s.version} (${fmtInt(s.num_segments)} segs)`;
          hide();
          apiFetch("/ui/segment_sets?prefetch=" + encodeURIComponent(s.uuid)).catch(() => {});
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
    filter_lance_uri: _splitList($("sf-lance").value),
    vehicle: _splitList($("sf-vehicle").value),
    drive_id: _splitList($("sf-drive").value),
    segment_set_uuid: state.searchSegUuid,
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
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, segment_id: m.segment_id, mark: m.mark, index: m.index, row: m.row, rank: m.rank, score: m.score }));
  if (!marks.some((m) => m.mark === "up")) { showToast("Mark at least one 👍 to re-rank"); return; }
  state.mode = "refine";
  showToast(`Re-ranking with ${marks.filter((m) => m.mark === "up").length} 👍 / ${marks.filter((m) => m.mark === "down").length} 👎…`);
  if (_fullCorpusOn()) {
    await _issueFullVector("/ui/search", {
      query: state.query, marks, k: FULL_BATCH, output: "hits",
      negative_weight: 0.5, text_weight: 0.3, refine_from_marks: true,
    }, (startOpts && startOpts.page) || 0);
    return;
  }
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
  await _issueFullVector("/ui/search", {
    window: state.windowReq, k: FULL_BATCH, output: "hits",
  }, (startOpts && startOpts.page) || 0);
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
    const enc = await apiFetch("/ui/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames_b64, output: "vector" }),
    }).then((r) => r.ok ? r.json() : r.json().then((j) => { throw new Error(j.detail || ("HTTP " + r.status)); }));
    // The uploaded example is now just a query vector -> reuse the resume path so
    // paging, refine, sweep, save, and offline-scan export all work unchanged.
    state.resumeVec = enc.vector; state.resumeTag = null;
    state.resumeLabel = (isVideo ? "🎬 " : "🖼️ ") + (enc.label || "uploaded example");
    state.query = state.resumeLabel;
    state.mode = "resume";
    state.marks = {};
    _uploadNote(noteAfter);
    await runVectorSearch({ page: 0 });
  } catch (e) { _uploadNote("Could not search by upload: " + e.message, true); }
}
async function runVectorSearch(startOpts) {
  if (!state.resumeVec && !state.resumeTag) return;
  state.mode = "resume";
  const extra = state.resumeTag
    ? { tag: state.resumeTag, query: state.resumeLabel || state.query || "", k: FULL_BATCH, output: "hits" }
    : { vector: state.resumeVec, query: state.resumeLabel || state.query || "", k: FULL_BATCH, output: "hits" };
  await _issueFullVector("/ui/search", extra, (startOpts && startOpts.page) || 0);
}
function reload(startOpts) {
  if (state.mode === "refine") return runRefine(startOpts);
  if (state.mode === "resume") return runVectorSearch(startOpts);
  if (state.mode === "window") return runWindowSearch(startOpts);
  return runSearch(startOpts);
}

// Show the corpus the next search will actually use, not the resident one.
async function refreshCorpusPill() {
  const h = await _health().catch(() => null);
  const p = _projectHealth(h);
  if (!p) return;
  const model = (h.model && h.model.id) || "model";
  if (p.ready) {
    $("corpusPill").innerHTML = `🗂 <b>${fmtInt(p.rows)} clips</b> · corpus v${p.corpus_version} · ${escapeHtml(model)}`;
    if (!state.corpus) applyCorpus(_corpusFromHealth(p));
    $("embeddings-uri").value = p.corpus_table_uri || "";
    setNote("corpus-note", `${fmtInt(p.rows)} clips · ${p.date_span ? p.date_span.join(" → ") : ""}`);
  } else if (p.status === "error") {
    $("corpusPill").innerHTML = `🗂 <b>corpus failed</b> · ${escapeHtml(p.error || "")}`;
  } else {
    $("corpusPill").innerHTML = `🗂 <b>corpus loading…</b> ${Math.round(p.elapsed_s || 0)}s · ${escapeHtml(model)}`;
    setTimeout(refreshCorpusPill, 10000);
  }
}
// Search always covers the whole corpus. There is no toggle: offering the ~2M
// resident subset as a choice meant users could silently search 6% of the data
// and read the result as complete.
function _fullCorpusOn() { return true; }
function wireFullCorpusToggles() { refreshCorpusPill(); }
// The corpus is read and decoded on first use (minutes, ~12GB), so the server
// answers 503 until it is resident. Kick the load, show progress, and poll
// rather than leaving the grid on "Searching...". Returns false if it failed.
async function _ensureFullCorpus() {
  $("emptyState").style.display = "none";
  $("resultsState").style.display = "block";
  for (let i = 0; i < 60; i++) {
    const h = await _health().catch(() => null);
    const p = _projectHealth(h);
    if (p && p.ready) { if (!state.corpus) applyCorpus(_corpusFromHealth(p)); return true; }
    if (p && p.status === "error") { $("gridStatus").textContent = "Corpus failed to load: " + (p.error || ""); return false; }
    $("gridStatus").textContent = `Loading the ${state.project} corpus (a few minutes on a cold start)… ${Math.round((p && p.elapsed_s) || 0)}s`;
    await new Promise((res) => setTimeout(res, 10000));
  }
  $("gridStatus").textContent = "Corpus did not finish loading — try again.";
  return false;
}
async function _issueFullCorpus(q, page) {
  if (!(await _ensureFullCorpus())) return;
  await _fullFetch("/ui/search", { query: q, k: FULL_BATCH, output: "hits" }, page, _fullKey(q));
}

// The filter half of every full-corpus request body, so text search, resume,
// upload and refine cannot drift apart on which filters they apply.
function _fullFilterBody() {
  return {
    page: 0,
    limit: FULL_BATCH,
    from_date: $("sf-dateFrom").value || null,
    to_date: $("sf-dateTo").value || null,
    vehicle: _splitList($("sf-vehicle").value),
    drive_id: _splitList($("sf-drive").value),
    segment_set_uuid: state.searchSegUuid,
    filter_lance_uri: _splitList($("sf-lance") ? $("sf-lance").value : ""),
  };
}

// Fetch one batch and page through it locally. A page is a slice of scores that
// are already in memory server-side, so re-requesting per page would be a fresh
// 34M-row scan for results we already hold. One batch also means the exact-score
// round trip can cover every page in a single fetch.
async function _fullFetch(endpoint, extra, page, key) {
  $("gridStatus").textContent = "Searching all clips…";
  let data;
  try {
    data = await apiFetch(endpoint, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ..._fullFilterBody(), ...extra }),
    }).then((r) => {
      if (!r.ok) return r.json().then((j) => { throw new Error(j.detail || ("HTTP " + r.status)); });
      return r.json();
    });
  } catch (e) {
    $("gridStatus").textContent = "Search failed: " + e.message;
    return;
  }
  // Carry the ranking's vector so a rescore addresses the same direction.
  state.fullBuf = { key, data, hits: data.hits || [], vector: data.vector || null };
  _renderFullPage(page || 0);
}

// Resume / upload / refine: same corpus, same filters, same envelope as a text
// search. Keyed so a different vector or a different mark set never reuses the
// previous buffer.
async function _issueFullVector(endpoint, extra, page) {
  if (!(await _ensureFullCorpus())) return;
  const key = [endpoint, _fullKey(state.query || ""), JSON.stringify(extra.marks || extra.vector || extra.tag || "").length,
               JSON.stringify(extra.marks || ""), extra.tag || ""].join("|");
  await _fullFetch(endpoint, extra, page, key);
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
      // Exact scores carry no error bound; printing one next to them would be
      // stating an uncertainty that no longer applies.
      + (d.score_kind === "exact"
          ? ` · similarity ${d.score_lo}–${d.score_hi} (exact)`
          : ` · similarity ${d.score_lo}–${d.score_hi} ±${d.score_error_bound} (approximate)`)
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
  const buf = state.fullBuf;
  if (!buf || !buf.hits || !buf.hits.length || !state.query) return;
  // Already sharpened this result set -- paging must not refetch or re-sort.
  if (buf.exact) return;
  // Rescore the WHOLE buffer, not the visible page. Two reasons: the ranking has
  // to be global or page 2 can hold a clip that outranks page 1 and never moves,
  // and the round trip has a ~1.65s floor regardless of row count, so doing it
  // per page pays that cost again for every page.
  const rows = buf.hits.map((h) => h.row).filter((r) => r != null && r >= 0);
  if (!rows.length) return;
  let data;
  try {
    data = await apiFetch("/ui/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      // The vector the ranking used, NOT the query text. A refine, window or
      // upload has no text that reproduces its direction, and sending `query`
      // made the server re-encode a UI label and rescore against that.
      body: JSON.stringify(
        buf.vector ? { vector: buf.vector, rows, output: "scores" }
                   : { query: state.query, rows, output: "scores" },
      ),
    }).then((r) => (r.ok ? r.json() : null));
  } catch (e) { return; }
  if (!data || !data.scores) return;
  const byRow = new Map(data.scores.map((s) => [s.row, s.score]));
  let changed = 0;
  for (const h of buf.hits) {
    const exact = byRow.get(h.row);
    if (exact != null && exact !== h.score) { h.score = exact; changed++; }
    if (exact != null) h.score_kind = "exact";
  }
  buf.exact = true;
  if (!changed) return;
  // Exact scores can reorder rows that sat within the error bound of each other.
  buf.hits.sort((a, b) => b.score - a.score);
  // Renumber. `rank` came from the screening pass; leaving it alone after a
  // re-sort is what showed ranks 1, 19, 5, 2 down the grid -- correct ordering
  // labelled with the ordering it replaced.
  buf.hits.forEach((h, i) => { h.rank = i + 1; });
  const d = buf.data;
  if (d) {
    const sc = buf.hits.map((h) => h.score);
    d.score_lo = Math.round(Math.min(...sc) * 1e4) / 1e4;
    d.score_hi = Math.round(Math.max(...sc) * 1e4) / 1e4;
    d.score_kind = "exact";
  }
  _renderFullPage(state.page || 0);
}
async function _issue(endpoint, body) {
  const seq = ++_issueSeq;
  $("emptyState").style.display = "none";
  $("resultsState").style.display = "block";
  $("gridStatus").textContent = state.mode === "refine" ? "Re-ranking…" : "Searching…";
  let data;
  try {
    data = await apiFetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
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
    const vsrc = "/ui/video?uri=" + encodeURIComponent(h.source_media_uri || "");
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
      <video controls preload="metadata" playsinline src="/ui/video?uri=${encodeURIComponent(h.source_media_uri || "")}#t=0.1"></video>
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
  // stratified grid (no video reload).
  if (state.sweepActive && state.sweep) { drawSweep(); scheduleFit(); }
}
let _fitTimer = null;
function scheduleFit() { clearTimeout(_fitTimer); _fitTimer = setTimeout(fitOnly, 500); }
async function fitOnly() {
  if (!state.sweepActive) return;
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, mark: m.mark, index: m.index, row: m.row, segment_id: m.segment_id || "" }));
  const f = _searchFilters();
  try {
    const thr = await apiFetch(_thresholdEndpoint(), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, marks, objective: "f1", min_precision: 0.9, val_fraction: 0.0, sample_size: 12, ...f }) }).then((r) => r.ok ? r.json() : null);
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
      // The histogram comes back from /api/calibrate itself; a second call to a
      // separate distribution endpoint scored the same 34.4M rows for the same
      // answer.
      Promise.resolve(null),
      apiFetch(_thresholdEndpoint(), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, marks, objective: "f1", min_precision: 0.9, val_fraction: 0.0, sample_size: 12, ...f }) }).then((r) => r.ok ? r.json() : null),
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
  const tag = $("tagInput").value.trim().toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 64);
  if (!tag) { setNote("saveNote", "Enter a tag name (letters, digits, underscores).", true); return; }
  if (state.mode === "resume" && !state.resumeTag) {
    setNote("saveNote", "An uploaded image or video can't be saved as a tag — search by text or by a corpus clip instead.", true);
    return;
  }
  const btn = $("finalSaveBtn"); btn.disabled = true; setNote("saveNote", "Saving…");
  try {
    const input = (state.mode === "window" && state.windowReq)
      ? { type: "video", run_uuid: state.windowReq.run_uuid || "", segment_id: state.windowReq.segment_id || "",
          start_ns: state.windowReq.start_ns || 0, end_ns: state.windowReq.end_ns || 0, pooling: "mean" }
      : { type: "text", text: state.resumeTag ? (state.resumeQuery || state.query) : state.query };
    const body = { tag, project: state.project, input, description: ($("tagDesc") && $("tagDesc").value.trim()) || "" };
    if (state.confirmedTau != null) { body.threshold_mode = "explicit"; body.threshold = +state.confirmedTau.toFixed(4); }
    else body.threshold_mode = "suggested";
    let r = await apiFetch("/api/v1/tags", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (r.status === 409) setNote("saveNote", "Tag already exists on this project — refining it with your feedback…");
    else if (!r.ok) throw new Error(await _detail(r));
    const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, mark: m.mark }));
    if (marks.some((m) => m.mark === "up")) {
      r = await apiFetch(`/api/v1/tags/${encodeURIComponent(tag)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: state.project, marks }),
      });
      if (!r.ok) throw new Error(await _detail(r));
    }
    closeDrawer();
    showToast(`Saved "${tag}" — configure export below`);
    await loadSaved();
    setPage("saved");
    const idx = state.savedRows.findIndex((e) => e.tag === tag);
    if (idx >= 0) { state.selected = new Set([idx]); renderTable(); renderExportPanel(); const row = $("tbody").querySelector(`tr[data-i="${idx}"]`); if (row) row.classList.add("new-row"); }
  } catch (e) { setNote("saveNote", "Save failed: " + e.message, true); }
  finally { btn.disabled = false; }
}
// The API's error body: {detail: {code, message, status}} or a plain string.
function _detail(r) {
  return r.json().then((j) => {
    const d = j && j.detail;
    if (d && d.message) return d.message;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((e) => `${(e.loc || []).join(".")}: ${e.msg}`).join("; ");
    return "HTTP " + r.status;
  }).catch(() => "HTTP " + r.status);
}
/* ===================== saved searches table ===================== */
function _uriShort(u) { const p = String(u || "").replace(/\/+$/, "").split("/").filter(Boolean); return p.length ? p[p.length - 1] : ""; }
function _fmtDates(f) { const from = (f && f.from_date) || "", to = (f && f.to_date) || ""; return (from || to) ? `${from || "…"} → ${to || "latest"}` : "—"; }

async function loadSaved() {
  try { const d = await apiFetch("/api/v1/tags?page_size=500").then((r) => r.json()); state.savedRows = d.tags || []; }
  catch (e) { state.savedRows = []; }
  renderTable();
  renderExportPanel();
}
// Populate the model dropdown: loaded model first (default selection), then the
// other models present in history, then "All models". Rebuilt only when the set
// changes, preserving the user's current pick.
function renderTable() {
  const all = state.savedRows || [];
  const q = ($("savedFilter").value || "").trim().toLowerCase();
  $("savedCount").textContent = all.length ? `(${all.length})` : "";
  const body = all.map((e, i) => {
    const hay = (e.tag + " " + (e.description || "")).toLowerCase();
    if (q && !hay.includes(q)) return "";
    const th = (e.thresholds || {})[state.project];
    const tau = th
      ? `τ ${Number(th.value).toFixed(3)} <span style="color:var(--muted-2)">${escapeHtml(th.mode)}${th.stale ? " · stale" : ""}</span>`
      : `<span style="color:var(--muted-2)">no ${escapeHtml(state.project)} threshold</span>`;
    const others = Object.keys(e.thresholds || {}).filter((p) => p !== state.project);
    return `<tr data-i="${i}" class="${state.selected.has(i) ? "checked" : ""}">
      <td><input type="checkbox" data-i="${i}" ${state.selected.has(i) ? "checked" : ""}></td>
      <td><span class="tag-pill">${escapeHtml(e.tag)}</span></td>
      <td><div class="query-text">${escapeHtml(e.description || "")}</div></td>
      <td class="model-cell">v${e.version}${e.pinned_version != null ? ` <span style="color:var(--muted-2)">pinned v${e.pinned_version}</span>` : ""}${others.length ? `<br><span style="color:var(--muted-2)">also on ${escapeHtml(others.join(", "))}</span>` : ""}</td>
      <td class="date-cell">${escapeHtml(String(e.updated_at || "").slice(0, 10))}</td>
      <td><span class="kt-pill numeric">${tau}</span></td>
      <td><button class="resume-link" data-tag="${escapeHtml(e.tag)}">resume ↗</button></td>
    </tr>`;
  }).join("");
  $("tbody").innerHTML = body || `<tr><td colspan="7" style="color:var(--muted-2)">${all.length ? "No saved searches match that filter." : "No saved searches yet — run a search and Save."}</td></tr>`;
}
async function resumeTag(tag) {
  setPage("search"); showToast("Loading saved search…");
  let rec;
  try { rec = await apiFetch(`/api/v1/tags/${encodeURIComponent(tag)}`).then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); }); }
  catch (e) { showToast("Could not load: " + e.message); return; }
  const latest = rec.versions[rec.versions.length - 1] || {};
  const src = latest.source || {};
  const label = src.type === "text" ? src.text : `${tag} (${src.type})`;
  $("searchInput2").value = label || ""; state.query = label || tag;
  $("tagInput").value = tag;
  state.marks = {}; state.resumeVec = null;
  state.resumeTag = tag; state.resumeQuery = src.type === "text" ? src.text : "";
  state.resumeLabel = label || tag; state.mode = "resume";
  const th = (latest.thresholds || {})[state.project];
  state.confirmedTau = th ? Number(th.value) : null;
  if (state.confirmedTau != null) { $("threshVal").textContent = state.confirmedTau.toFixed(3); $("threshSummary").textContent = `τ ${state.confirmedTau.toFixed(3)}`; }
  updateRail();
  await runVectorSearch({ page: 0 });
  showToast(`Resumed "${tag}" v${latest.version || ""}`);
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
  const badge = $("mechBadge");
  badge.textContent = "instant";
  badge.className = "mech-badge instant";
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
  btn.textContent = "⬇ Export"; btn.className = "btn-export instant"; $("exportNote").textContent = "Downloads a CSV immediately.";
  $("jobStatus").classList.remove("show");
  renderCutoffField();
  renderSampleMode();
}
function _exportFilters() {
  return {
    from_date: $("ex-dateFrom").value || null, to_date: $("ex-dateTo").value || null,
    filter_lance_uri: _splitList($("ex-lance").value), vehicle: _splitList($("ex-vehicle").value),
    segment_set_uuid: state.exportSegUuid,
  };
}
function doExport() {
  if (state.selected.size === 0) return;
  fullExport();
}
// Threshold fitting follows whichever corpus the results came from. The two
// endpoints address marks differently -- the resident one by `index`, the
// full-corpus one by `row` -- and sending full-corpus marks to the resident
// endpoint silently drops every label, which reads as "no labels yet".
function _thresholdEndpoint() { return "/ui/calibrate"; }
// Export straight from the resident corpus: the ranking pass is the same one a
// search runs, so the only extra cost is materializing the rows.
async function fullExport() {
  const rows = [...state.selected].map((i) => state.savedRows[i]).filter(Boolean);
  if (!rows.length) return;
  const topk = state.cutoffMode === "topk";
  const fmt = ($("exportFormat") && $("exportFormat").value) || "csv";
  const btn = $("exportBtn"); btn.disabled = true;
  const note = $("exportNote");
  const f = _exportFilters();
  let done = 0;
  try {
    for (const e of rows) {
      const q = new URLSearchParams({
        output: fmt, interval: String(state.sampleMode === "interval"),
        segment_mode: String($("dedupInput").checked),
        create_segment_set: String(!!$("segsetInput") && $("segsetInput").checked),
      });
      if (topk) q.set("k", String(kForRow(e) || 50));
      for (const [k, v] of Object.entries(f)) if (v) q.set(k, v);
      const url = `/api/v1/tags/${encodeURIComponent(e.tag)}?${q}`;
      note.textContent = `Exporting ${e.tag}… (${done + 1}/${rows.length})`;
      // The same URL starts the export and reports on it; poll until it is ready.
      let body = null;
      for (let attempt = 0; attempt < 360; attempt++) {
        const r = await apiFetch(url);
        if (!r.ok) throw new Error(`${e.tag}: ${await _detail(r)}`);
        body = await r.json();
        const st = body.export && body.export.status;
        if (r.status !== 202 && st !== "running" && st !== "pending") break;
        note.textContent = `Exporting ${e.tag}… ${st} (${attempt * 5}s)`;
        await new Promise((res) => setTimeout(res, 5000));
      }
      const ex = (body && body.export) || {};
      if (ex.status !== "ready") throw new Error(`${e.tag}: ${ex.error || ex.status || "did not finish"}`);
      if (ex.download_url) window.open(ex.download_url, "_blank", "noopener");
      done++;
      const bits = [`${fmtInt(ex.num_rows)} rows`, `corpus v${body.corpus_version}`];
      if (ex.segment_set_uuid) bits.push(`segment set ${ex.segment_set_uuid}`);
      note.textContent = `${e.tag}: ` + bits.join(" · ");
    }
    showToast(`Exported ${done} tag(s)`);
    loadScanJobs(true);
  } catch (err) {
    note.textContent = "Export failed: " + err.message;
  } finally { btn.disabled = false; }
}
/* ===================== recent exports ===================== */
function _scanStatusClass(st) { if (st === "ready") return "succeeded"; if (st === "error") return "failed"; return "queued"; }
let _scansRepollTimer = null;
async function loadScanJobs(live) {
  clearTimeout(_scansRepollTimer);
  const body = $("scansBody");
  const tags = (state.selected.size ? [...state.selected].map((i) => state.savedRows[i]) : (state.savedRows || []).slice(0, 25))
    .filter(Boolean).map((e) => e.tag);
  if (!tags.length) { body.innerHTML = `<tr><td colspan="7" style="color:var(--muted-2)">No exports yet — save a search and export it.</td></tr>`; return; }
  if (!(state.scanJobs && state.scanJobs.length)) body.innerHTML = `<tr><td colspan="7" style="color:var(--muted-2)">Loading recent exports…</td></tr>`;
  try {
    const recs = await Promise.all(tags.map((t) => apiFetch(`/api/v1/tags/${encodeURIComponent(t)}`).then((r) => (r.ok ? r.json() : null))));
    state.scanJobs = recs.filter(Boolean).flatMap((rec) => (rec.exports || []).map((x) => ({ ...x, tag: rec.tag })));
    state.scanJobs.sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
  } catch (e) {
    body.innerHTML = `<tr><td colspan="7" style="color:var(--neg)">Failed to load recent exports: ${escapeHtml(e.message)}</td></tr>`;
    return;
  }
  renderScans();
  const active = state.scanJobs.some((j) => j.status === "running" || j.status === "pending");
  if (active && $("page-saved").classList.contains("active")) _scansRepollTimer = setTimeout(() => loadScanJobs(false), 7000);
}
function renderScans() {
  try { _renderScans(); }
  catch (e) {
    // One malformed row must not blank the whole archive -- this panel is the
    // only place some artifacts are discoverable.
    console.error("renderScans failed", e);
    $("scansBody").innerHTML = `<tr><td colspan="8" style="color:var(--neg)">Could not render the export list: ${escapeHtml(e.message)}</td></tr>`;
  }
}
function _renderScans() {
  const jobs = state.scanJobs || [];
  $("scansBody").innerHTML = jobs.length ? jobs.map((j) => {
    const p = j.params || {};
    const what = [p.interval ? "intervals" : "clips", p.segment_mode ? "1 per segment" : "", p.k ? `top-${p.k}` : "", _fmtScanFilters(p)]
      .filter((x) => x && x !== "—").join(" · ");
    const out = j.uri
      ? `<div class="scan-output"><span class="scan-uri" title="${escapeHtml(j.uri)}">${escapeHtml(_uriShort(j.uri))}</span>`
        + `<button class="scan-copy" data-uri="${escapeHtml(j.uri)}" title="Copy full path">copy</button>`
        + (j.download_url ? `<a class="scan-copy" href="${escapeHtml(j.download_url)}" target="_blank" rel="noopener">download</a>` : "") + `</div>`
      : (j.error ? `<span style="color:var(--neg)">${escapeHtml(j.error)}</span>` : "—");
    const dx = j.segment_set_uuid ? `<span class="scan-dx">${escapeHtml(j.segment_set_uuid)}</span>` : `<span class="scan-dx none">—</span>`;
    return `<tr>
      <td class="scan-time">${escapeHtml(j.created_at || "")}</td>
      <td><span class="tag-pill">${escapeHtml(j.tag)}</span> v${j.version} · ${escapeHtml(j.project)}</td>
      <td class="scan-filters">${escapeHtml(what)}</td>
      <td><span class="status-pill ${_scanStatusClass(j.status)}">${escapeHtml(j.status || "—")}</span></td>
      <td class="scan-counts">${j.num_rows != null ? fmtInt(j.num_rows) : "—"}</td>
      <td>${out}</td>
      <td>${dx}</td></tr>`;
  }).join("") : `<tr><td colspan="7" style="color:var(--muted-2)">No exports yet.</td></tr>`;
}
// Result counts (total segments + per-tag breakdown), mirroring the Data Explorer view.
// Copy an output path, or open a preview of it (delegated so both survive
// re-renders of the scans table).
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".scan-copy");
  if (!btn) return;
  const uri = btn.dataset.uri || "";
  if (!uri) return;
  navigator.clipboard.writeText(uri).then(
    () => showToast("Copied output path"),
    () => showToast("Copy failed — select the path manually"),
  );
});


function _fmtScanFilters(f) {
  if (!f) return "—"; const p = [];
  if (f.from_date || f.to_date) p.push(`${f.from_date || "…"}→${f.to_date || "latest"}`);
  if (f.segment_set_uuid) p.push("seg: " + f.segment_set_uuid);
  if (f.vehicle) p.push("veh: " + f.vehicle);
  if (f.filter_lance_uri) p.push("lance: " + _uriShort(f.filter_lance_uri));
  return p.join(" · ") || "—";
}

/* ===================== misc ===================== */
function setNote(id, html, isErr) { const n = $(id); if (!n) return; n.innerHTML = html; n.classList.toggle("warn", !!isErr); }
let _toastTimer = null;
function showToast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(_toastTimer); _toastTimer = setTimeout(() => t.classList.remove("show"), 3000); }

init();
