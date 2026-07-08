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
  if ($("embeddings-uri")) $("embeddings-uri").value = c.embeddings_uri || "";
  if ($("model-uri") && c.model_uri !== undefined) $("model-uri").value = c.model_uri || "";
  ["sf-dateFrom", "ex-dateFrom"].forEach((id) => { const el = $(id); if (el) { el.value = c.span_lo_date; el.min = c.span_lo_date; el.max = c.span_hi_date; } });
  ["sf-dateTo", "ex-dateTo"].forEach((id) => { const el = $(id); if (el) { el.value = c.span_hi_date; el.min = c.span_lo_date; el.max = c.span_hi_date; } });
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
  $("vs-search-btn").onclick = () => runWindowSearch({ page: 0 });

  $("filtersChip").onclick = toggleSearchFilters;
  $("applyFiltersBtn").onclick = applySearchFilters;
  wireDxCombo();

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
  $("scansReload").onclick = loadScanJobs;
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
  if (which === "saved") { renderTable(); renderExportPanel(); if (state.offlineScan) loadScanJobs(); }
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
  else state.marks[chunkId] = { mark: dir, segment_id: h.segment_id, index: h.index, rank: h.rank, score: h.score };
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
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, mark: m.mark, index: m.index, segment_id: m.segment_id || "" }));
  const f = _searchFilters();
  try {
    const thr = await fetch("/api/threshold_search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, marks, objective: "f1", min_precision: 0.9, val_fraction: 0.0, sample_size: 12, ...f }) }).then((r) => r.ok ? r.json() : null);
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
  const marks = Object.entries(state.marks).map(([chunk_id, m]) => ({ chunk_id, mark: m.mark, index: m.index, segment_id: m.segment_id || "" }));
  const f = _searchFilters();
  try {
    const [dist, thr] = await Promise.all([
      fetch("/api/score_distribution", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, k: 2000, ...f, interval_mode: "k", interval_score: null }) }).then((r) => r.ok ? r.json() : null),
      fetch("/api/threshold_search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: state.query, marks, objective: "f1", min_precision: 0.9, val_fraction: 0.0, sample_size: 12, ...f }) }).then((r) => r.ok ? r.json() : null),
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
  const above = _clipsAtOrAbove(t);
  const scale = (state.corpus && sw.total) ? state.corpus.num_rows / sw.total : 1;
  $("statCorpus").textContent = `~${fmtInt(Math.round(above * scale))}`;
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
    const r = await fetch("/api/save_vector", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      tag, query: state.query, k: 50, threshold: tau,
      from_date: $("sf-dateFrom").value || null, to_date: $("sf-dateTo").value || null,
      segment_set_uuid: state.searchSegUuid, segment_set_name: state.searchSegName,
      filter_lance_uri: _splitList($("sf-lance").value), vehicle: _splitList($("sf-vehicle").value), drive_id: _splitList($("sf-drive").value),
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
  } else {
    box.innerHTML = `<label>K =</label><input id="kInput" value="50"><span id="kConversion" style="font-size:11.5px;color:var(--muted-2);font-family:var(--mono);"></span>`;
    $("kInput").oninput = updateKConversion; updateKConversion();
  }
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
  if (state.offlineScan) launchScan(); else downloadCsv();
}
async function downloadCsv() {
  const rows = [...state.selected].map((i) => state.savedRows[i]).filter(Boolean);
  const topk = state.cutoffMode === "topk";
  const kGlobal = parseInt(($("kInput") || {}).value, 10) || 50;
  const queries = rows.map((e) => ({ query: e.query || e.tag, k: topk ? kGlobal : (e.k || 50), threshold: topk ? 0 : (e.threshold || 0) }));
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
  const btn = $("exportBtn"); btn.disabled = true; btn.textContent = "⏳ Job queued…";
  const js = $("jobStatus"); js.classList.add("show"); js.innerHTML = `<span>launching per-segment scan over ${tags.length} tag(s)…</span>`;
  try {
    const r = await fetch("/api/launch_segment_scan", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      tags, thresholds, default_threshold: defThr, create_segment_set: $("segsetInput").checked, merge_intervals: state.sampleMode === "interval", ..._exportFilters(),
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
async function loadScanJobs() {
  if (!state.offlineScan) return;
  try { const d = await fetch("/api/scans").then((r) => r.json()); state.scanJobs = d.jobs || []; }
  catch (e) { return; }
  renderScans();
}
function renderScans() {
  const jobs = state.scanJobs || [];
  $("scansBody").innerHTML = jobs.length ? jobs.map((j) => {
    const id = j.console_url ? `<a class="scan-name" href="${j.console_url}" target="_blank" rel="noopener">${escapeHtml(j.execution_id)}</a>` : `<span class="scan-name">${escapeHtml(j.execution_id)}</span>`;
    const tagList = (j.tags || []).map((t) => escapeHtml(t) + (j.thresholds && j.thresholds[t] != null ? ` @${j.thresholds[t]}` : "")).join(", ");
    const filt = _fmtScanFilters(j.filters);
    const out = j.lance_uri ? `<div class="scan-output">${escapeHtml(j.lance_uri.split("/").slice(-2).join("/"))}<div class="full-path">${escapeHtml(j.lance_uri)}</div></div>` : "—";
    let dx = `<span class="scan-dx none">—</span>`;
    if (j.segset_label) dx = `<span class="scan-dx">${escapeHtml(j.segset_label)}</span>`;
    else if (j.register_segset) dx = `<span class="scan-dx none">pending…</span>`;
    return `<tr>
      <td class="scan-time">${escapeHtml(j.created_at || "")}</td>
      <td>${id}</td>
      <td class="scan-tags">${(j.tags || []).length} tag(s): ${tagList}</td>
      <td class="scan-filters">${escapeHtml(filt)}</td>
      <td><span class="status-pill ${_scanStatusClass(j.status)}">${escapeHtml(j.status || "—")}</span></td>
      <td>${out}</td>
      <td>${dx}</td></tr>`;
  }).join("") : `<tr><td colspan="7" style="color:var(--muted-2)">No scans launched yet.</td></tr>`;
}
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
