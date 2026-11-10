"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const CITE = /\[([^\[\],]+?),\s*Pages?:?\s*([\d\s,\-–]+)\]/gi;
const state = { docs: [], examples: [], selected: new Set(), mode: "hybrid", pages: [], active: 0, example: null, res: null };
const short = c => (c || "").replace(/\s+(Limited|Ltd\.?)$/i, "");
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) { } } };

function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(toast.h); toast.h = setTimeout(() => t.hidden = true, 4200); }
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) { } throw new Error(m); }
  return r.json();
}

/* ---------- views and theme ---------- */
function show(view) {
  if (!["ask", "compare", "results"].includes(view)) view = "ask";
  $$(".view").forEach(v => v.hidden = v.dataset.view !== view);
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.view === view));
  if (view === "results" && !$("#results").dataset.ready) renderResults();
  if (location.hash.slice(1) !== view) history.replaceState(null, "", "#" + view);
}
$$(".tabs button").forEach(b => b.onclick = () => show(b.dataset.view));
window.addEventListener("hashchange", () => show(location.hash.slice(1)));
$("#theme").onclick = () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "light" : "dark"; store.set("pd-theme", document.documentElement.dataset.theme);
};
if (store.get("pd-theme")) document.documentElement.dataset.theme = store.get("pd-theme");

/* ---------- filings rail ---------- */
function renderDocs() {
  $("#doc-list").innerHTML = state.docs.map(d => `<label class="doc"><input type="checkbox" value="${esc(d.id)}" ${state.selected.has(d.id) ? "checked" : ""}><span class="nm" title="${esc(d.company)}">${esc(short(d.company))}</span><span class="ty">${esc(d.type === "PROSPECTUS" ? "RHP" : d.type)}</span></label>`).join("");
  $$("#doc-list input").forEach(i => i.onchange = () => { i.checked ? state.selected.add(i.value) : state.selected.delete(i.value); });
}
$("#all-docs").onclick = () => { state.selected.clear(); renderDocs(); };
function setDocs(ids) { state.selected = new Set(ids); renderDocs(); }

/* ---------- examples ---------- */
function renderChips(el, onPick) {
  el.innerHTML = state.examples.map((e, i) => `<button type="button" class="chip" data-i="${i}" title="${esc(e.question)}"><b>${esc(e.label)}</b><span>${esc(e.question)}</span></button>`).join("");
  $$(".chip", el).forEach(b => b.onclick = () => onPick(state.examples[+b.dataset.i]));
}
function pickExample(e) { $("#q").value = e.question; setDocs(e.docs); state.example = e; submit(); }

/* ---------- answer rendering ---------- */
function answerNode(res) {
  const wrap = document.createElement("div");
  wrap.className = "answer" + (res.abstained ? " none" : "");
  let last = 0, i = 0; const text = res.answer || "";
  for (const m of text.matchAll(CITE)) {
    wrap.append(text.slice(last, m.index));
    const c = res.citations[i++], b = document.createElement("button");
    b.type = "button"; b.className = "cite"; b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 3h9l4 4v14H6z"/></svg>${esc(c && c.doc ? short(res.contexts.find(x => x.doc === c.doc)?.company || c.doc) : m[1])} p. ${esc(m[2].trim())}`;
    b.onclick = () => c && c.doc ? openPage(c.doc, c.pages[0], figures(res.answer), true) : toast("That citation does not match a loaded filing.");
    wrap.append(b); last = m.index + m[0].length;
  }
  wrap.append(text.slice(last)); return wrap;
}
const figures = a => [...new Set((a.replace(CITE, "").match(/\d[\d,]*(?:\.\d+)?/g) || []).filter(x => x.replace(/\D/g, "").length >= 3 && !/^(19|20)\d\d$/.test(x)))].slice(0, 12);
const FAIL = {
  no_key: ["Answer writing is off on this server", "It can search the filings but has no model key, so it cannot write an answer. The best passages are below. The saved example questions still show full answers."],
  rate_limit: ["The answer service is busy", "The model provider is limiting requests. Wait a minute and ask again. The passages below are still valid."],
  timeout: ["The answer service took too long", "The search worked. Ask again, or narrow the question to fewer filings."],
  bad_key: ["The model key was rejected", "Check the key in NVIDIA_API_KEYS and restart the server. The passages below are still valid."],
  empty: ["The model returned an empty answer", "Ask again. The passages below are still valid."],
  other: ["The model did not return an answer", "The search worked and its passages are below."],
};
function failNote(res) {
  const f = FAIL[res.error_kind] || FAIL.other, top = res.stages.reranked[0];
  const best = top ? `<div class="best"><b>Closest passage, ${esc(pageLabel(top))}</b><p>${esc(tidy(top.text).slice(0, 320))}${top.text.length > 320 ? "…" : ""}</p><button type="button" class="link" data-open="${esc(top.doc)}|${top.page_start}">Open page</button></div>` : "";
  return `<div class="note-box ${res.error_kind === "no_key" ? "info" : "warn"}"><b>${esc(f[0])}</b><p>${esc(f[1])}</p>${res.error_kind === "other" && res.error ? `<small>${esc(res.error)}</small>` : ""}</div>${best}`;
}
function badge(kind, icon, text) {
  const ic = { ok: '<path d="M5 12l4 4 10-10"/>', warn: '<path d="M12 8v5M12 17h.01"/>', info: '<circle cx="12" cy="12" r="8"/><path d="M12 11v5M12 8h.01"/>' }[icon];
  return `<span class="badge ${kind}"><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ic}</svg>${esc(text)}</span>`;
}
function badges(res) {
  const b = [], a = res.audit;
  if (res.abstained) b.push(badge("warn", "warn", "Not found in the selected filings"));
  else if (a) {
    if (a.unsupported.length) b.push(badge("bad", "warn", `Not in the retrieved pages: ${a.unsupported.slice(0, 4).join(", ")}`));
    else if (a.numbers_checked) b.push(badge("good", "ok", `All ${a.numbers_checked} figures appear in the retrieved pages`));
    else b.push(badge("", "info", "No figures to check"));
    if (a.bad_citations.length) b.push(badge("warn", "warn", `Cited page was not retrieved: ${a.bad_citations.join(", ")}`));
    else if (res.citations.length) b.push(badge("good", "ok", "Every citation points to a retrieved page"));
  }
  if (res.verification_action === "regenerated_ok") b.push(badge("good", "ok", "Rewritten after the figure check failed"));
  if (res.verification_action === "abstained") b.push(badge("warn", "warn", "Withheld: figures could not be verified"));
  if (res.timing?.generation_s) b.push(badge("", "info", `Search ${res.timing.retrieval_s} s, answer ${res.timing.generation_s} s`));
  if (res.cached) b.push(badge("", "info", "Replayed from a saved run"));
  return `<div class="badges">${b.join("")}</div>`;
}

/* ---------- viewer ---------- */
function openPage(doc, page, hl, cited) {
  let i = state.pages.findIndex(p => p.doc === doc && p.page === page);
  if (i < 0) { state.pages.push({ doc, page, hl: hl || [], cited: !!cited }); i = state.pages.length - 1; }
  state.active = i; renderViewer();
  $$(".cite").forEach(c => c.setAttribute("aria-pressed", "false"));
}
function renderViewer() {
  const v = $("#viewer"); if (!v) return;
  const name = d => short(state.docs.find(x => x.id === d)?.company || d), gold = p => state.example && state.example.gold_pages.includes(p.page) && state.example.docs.includes(p.doc);
  v.innerHTML = `<div class="v-tabs" role="tablist">${state.pages.map((p, i) => `<button class="v-tab" role="tab" aria-selected="${i === state.active}" data-i="${i}">${esc(name(p.doc))} p. ${p.page}${p.cited ? '<small>cited</small>' : ""}${gold(p) ? '<small>reference page</small>' : ""}</button>`).join("")}</div>
  <div class="v-body" id="vbody"></div>
  <div class="v-meta"><span id="vcap"></span><span class="vnav"><button id="vprev" type="button">Previous page</button><button id="vnext" type="button">Next page</button></span></div>`;
  $$(".v-tab", v).forEach(t => t.onclick = () => { state.active = +t.dataset.i; renderViewer(); });
  const p = state.pages[state.active];
  if (!p) { $("#vbody").innerHTML = `<div class="empty">Open a citation or a retrieved passage to read the source page here.</div>`; return; }
  const img = new Image(); img.alt = `${name(p.doc)}, page ${p.page}`; img.src = `/api/page/${encodeURIComponent(p.doc)}/${p.page}?hl=${encodeURIComponent((p.hl || []).join("|"))}`;
  $("#vbody").append(img); img.onclick = () => lightbox(img.src);
  $("#vcap").textContent = `${name(p.doc)}, page ${p.page}${p.hl?.length ? ". Figures from the answer are highlighted." : ""}`;
  $("#vprev").disabled = p.page <= 1;
  $("#vprev").onclick = () => { p.page--; p.cited = false; renderViewer(); }; $("#vnext").onclick = () => { p.page++; p.cited = false; renderViewer(); };
}
function lightbox(src) {
  const l = document.createElement("div"); l.className = "lightbox"; l.innerHTML = `<img src="${esc(src)}" alt="Source page, enlarged">`;
  const close = () => { l.remove(); removeEventListener("keydown", k); }, k = e => e.key === "Escape" && close();
  l.onclick = close; addEventListener("keydown", k); document.body.append(l);
}

/* ---------- how the system found it ---------- */
const STOP = new Set("the a an of to in on for and or is are was were what who which how much many does do did by with as at from that this it its be been has have had their there into than then about per".split(" "));
const termsOf = q => [...new Set((String(q).toLowerCase().match(/[a-z0-9][a-z0-9.,\-]*[a-z0-9]|[a-z0-9]/g) || []).filter(w => w.length > 2 && !STOP.has(w)))];
function mark(text, ts) {
  if (!ts.length) return esc(text);
  const re = new RegExp("(" + ts.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") + ")", "gi");
  return String(text).split(re).map((p, i) => i % 2 ? `<mark>${esc(p)}</mark>` : esc(p)).join("");
}
const ord = n => { const v = n % 100, s = ["th", "st", "nd", "rd"]; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
const tidy = t => { let x = String(t).trim(); if (/^[a-z]/.test(x)) { const i = x.indexOf(" "); x = "… " + (i > 0 ? x.slice(i + 1) : x); } return x; };
const pageLabel = c => `${short(c.company)} p. ${c.page_start}${c.page_end > c.page_start ? "-" + c.page_end : ""}`;
const MODE = { hybrid: "Meaning search and keyword search, top 10 from each", dense: "Meaning search only, top 10", bm25: "Keyword search only, top 10" };

function traceHTML(res, ex) {
  const s = res.stages, t = res.timing, mode = res.settings.mode, terms = termsOf(res.question);
  const useD = mode !== "bm25", useB = mode !== "dense";
  const ids = l => new Set(l.map(c => c.cid)), bothN = [...ids(s.dense)].filter(x => ids(s.bm25).has(x)).length;
  const finalIds = ids(s.reranked), ctxChars = res.contexts.reduce((a, c) => a + c.chars, 0), widened = res.contexts.filter(c => c.expanded).length;
  const gold = c => ex && ex.docs.includes(c.doc) && ex.gold_pages.some(p => p >= c.page_start && p <= c.page_end);
  const gen = t.generation_s, tot = Math.max(t.total_s, 0.01);
  const steps = [
    ["Search", `${(res.corpus?.chunks ?? 0).toLocaleString()} passages in ${res.corpus?.filings ?? "?"} ${res.corpus?.filings === 1 ? "filing" : "filings"}`, MODE[mode]],
    ["Combine", mode === "hybrid" ? `${s.fused.length} candidates` : `${s.fused.length} candidates`, mode === "hybrid" ? `${bothN} of ${s.dense.length} found by both searches` : "Kept in the order the search returned them"],
    ["Rerank", `Best ${s.reranked.length} kept`, "A second model reads each candidate with the question and reorders them"],
    ["Read", `${res.contexts.length} blocks, ${ctxChars.toLocaleString()} characters`, widened ? `${widened} widened to the whole section around the match` : "Sent to the model as found"],
    ["Answer", gen != null ? `${gen} s` : (res.error_kind === "no_key" ? "Off" : "Not written"), res.error_kind === "no_key" ? "No model key on this server" : res.failed && !res.answer ? "The answer service did not reply" : res.cached ? "Replayed from a saved run" : "Written by the model, citing pages"],
  ];
  const flow = `<ol class="flow">${steps.map((x, i) => `<li><i class="dot">${i + 1}</i><div><span class="f-name">${esc(x[0])}</span><span class="f-main">${esc(x[1])}</span><div class="f-sub">${esc(x[2])}</div></div></li>`).join("")}</ol>`;
  const bar = `<div class="timebar" role="img" aria-label="Search ${t.retrieval_s} seconds, answer ${gen ?? 0} seconds"><i class="tb-s" style="flex:${Math.max(t.retrieval_s, .05)}"></i>${gen ? `<i class="tb-a" style="flex:${gen}"></i>` : ""}</div>
    <div class="timelab"><span>Search <b>${t.retrieval_s} s</b></span><span>Answer <b>${gen ?? "n/a"}${gen != null ? " s" : ""}</b></span><span>Total <b>${t.total_s} s</b></span></div>`;
  const cards = s.reranked.map((c, i) => {
    const d = useD ? c.from_dense : null, b = useB ? c.from_bm25 : null, f = c.from_fused;
    const by = d && b ? ["both", "Found by both searches"] : d ? ["dense", "Found by meaning search"] : b ? ["bm25", "Found by keyword search"] : ["fused", "Found in the combined list"];
    const moved = f && f !== i + 1 ? (f > i + 1 ? `Reranker moved it up from ${ord(f)}` : `Reranker moved it down from ${ord(f)}`) : "Stayed in place after reranking";
    return `<article class="ev ${gold(c) ? "gold" : ""}" data-doc="${esc(c.doc)}" data-page="${c.page_start}">
      <div class="ev-n" aria-label="Rank ${i + 1}">${i + 1}</div>
      <div class="ev-main">
        <div class="ev-h"><b>${esc(pageLabel(c))}</b>${c.chapter ? `<span class="sec">${esc(c.chapter)}</span>` : (c.type === "table" ? '<span class="sec">Table</span>' : "")}<span class="tag ${by[0]}">${by[1]}</span>${gold(c) ? '<span class="tag goldtag">Reference evidence</span>' : ""}</div>
        <p class="ev-t clamp">${mark(tidy(c.text), terms)}</p>
        <div class="ev-f"><span class="how">${[d ? `Meaning search: ${ord(d)}` : "", b ? `Keyword search: ${ord(b)}` : "", moved].filter(Boolean).map(x => `<i>${esc(x)}</i>`).join("")}</span>
          <span class="ev-a"><button type="button" class="link more">Show more</button><button type="button" class="link open">Open page</button></span></div>
      </div></article>`;
  }).join("");
  const list = (title, sub, l, on) => `<div class="sx-col"><h4>${esc(title)}</h4><div class="sx-sub">${esc(on ? sub : "Not used in this setting")}</div>${l.map((c, i) => {
    const other = (l === s.dense ? s.bm25 : s.dense).some(x => x.cid === c.cid);
    return `<button type="button" class="sx-row ${on ? "" : "off"} ${finalIds.has(c.cid) ? "used" : ""}" data-doc="${esc(c.doc)}" data-page="${c.page_start}"><span class="n">${i + 1}</span><span class="pg">${esc(pageLabel(c))}</span><span class="sc">${esc(c.chapter || (c.type === "table" ? "Table" : ""))}</span>${other ? '<span class="both">in both</span>' : ""}</button>`; }).join("")}</div>`;
  return `${flow}${bar}
  <h4 class="sub-h">The ${s.reranked.length} passages used to answer</h4>
  <p class="sub-p">Highlighted words come from your question. Select a passage to read it, or open its page in the viewer.</p>
  <div class="evs">${cards}</div>
  <details class="sx"><summary>Compare what each search returned</summary>
    <p class="sub-p">Each search ranks every passage on its own. Filled markers show the ones that were finally used.</p>
    <div class="sx-grid">${list("Meaning search", "Finds passages with similar meaning", s.dense, useD)}${list("Keyword search", "Finds passages with the same words", s.bm25, useB)}</div></details>
  ${ex ? '<div class="legend"><span><i class="lg-gold"></i>Marked as evidence in the reference set</span></div>' : ""}`;
}
function wireTrace(root) {
  $$(".ev", root).forEach(c => {
    const t = $(".ev-t", c), more = $(".more", c);
    more.onclick = () => { const open = t.classList.toggle("clamp"); more.textContent = open ? "Show more" : "Show less"; };
    $(".open", c).onclick = () => openPage(c.dataset.doc, +c.dataset.page, [], false);
  });
  $$(".sx-row", root).forEach(r => r.onclick = () => openPage(r.dataset.doc, +r.dataset.page, [], false));
}

/* ---------- ask ---------- */
function payload(extra = {}) { return { question: $("#q").value.trim(), docs: [...state.selected], mode: state.mode, verify: $("#verify").checked ? "on" : "off", hierarchy: "parent", ...extra }; }
$$(".seg button").forEach(b => b.onclick = () => { state.mode = b.dataset.mode; $$(".seg button").forEach(x => x.setAttribute("aria-pressed", x === b)); });
$("#ask-form").onsubmit = e => { e.preventDefault(); if (state.example && state.example.question !== $("#q").value.trim()) state.example = null; submit(); };
$("#q").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); } });

let seq = 0;
async function submit() {
  const my = ++seq, p = payload(); if (!p.question) { toast("Type a question first."); $("#q").focus(); return; }
  const out = $("#result"), btn = $("#go"); btn.disabled = true; btn.textContent = "Asking";
  out.innerHTML = `<div class="res"><div class="col"><div class="card"><div class="steps" id="steps"><i class="on">Searching the filings</i><i>Reranking passages</i><i>Writing the answer</i><i class="el" id="elapsed">0 s</i></div>${$("#status").dataset.s === "wait" ? '<p class="warm">The search models are still loading. This happens once after the server starts and takes about 20 seconds.</p>' : ""}<div class="loading"><div class="sk" style="width:92%"></div><div class="sk" style="width:78%"></div><div class="sk" style="width:55%"></div></div></div></div><div class="col"><div class="card viewer"><div class="empty">The source page appears here.</div></div></div></div>`;
  const t0 = Date.now(), timers = [2500, 6000].map((ms, i) => setTimeout(() => $$("#steps i").forEach((el, j) => el.classList.toggle("on", j === i + 1)), ms));
  const tick = setInterval(() => { const e = $("#elapsed"); if (e) e.textContent = ((Date.now() - t0) / 1000).toFixed(0) + " s"; }, 500); timers.push(tick);
  try { const res = await api("/api/ask", p); if (my !== seq) return; state.res = res; render(res); }
  catch (e) { if (my !== seq) return; out.innerHTML = `<div class="card" style="margin-top:16px"><div class="err">${esc(e.message)}</div></div>`; }
  finally { timers.forEach(x => { clearTimeout(x); clearInterval(x); }); if (my === seq) { btn.disabled = false; btn.textContent = "Ask"; } }
}

function render(res) {
  const ex = state.example, out = $("#result");
  const cited = res.citations.filter(c => c.doc).flatMap(c => c.pages.slice(0, 3).map(p => ({ doc: c.doc, page: p, hl: figures(res.answer), cited: true })));
  const seen = new Set(); state.pages = cited.filter(p => !seen.has(p.doc + p.page) && seen.add(p.doc + p.page));
  if (!state.pages.length) state.pages = res.stages.reranked.slice(0, 3).map(c => ({ doc: c.doc, page: c.page_start, hl: [], cited: false }));
  state.active = 0;
  const answer = res.failed && !res.answer ? failNote(res) : "";
  out.innerHTML = `<div class="res"><div class="col">
    <div class="card"><h3>Answer</h3><div id="ans"></div>${answer}${res.failed && !res.answer ? "" : badges(res)}
      ${ex ? `<div class="ref"><b>Reference answer.</b> ${esc(ex.reference)}<br>Evidence pages in the reference set: ${ex.gold_pages.slice(0, 10).join(", ")}${ex.gold_pages.length > 10 ? "…" : ""}${ex.answerable ? "" : ". This question cannot be answered from the document, so the right reply is to say so."}</div>` : ""}</div>
    <div class="card"><h3>How the system found it</h3>${traceHTML(res, ex)}</div></div>
    <div class="col"><div class="card viewer" id="viewer"></div></div></div>`;
  if (!res.failed || res.answer) $("#ans").append(answerNode(res));
  wireTrace(out); $$("[data-open]", out).forEach(b => b.onclick = () => { const [d, p] = b.dataset.open.split("|"); openPage(d, +p, [], false); }); renderViewer();
}

/* ---------- compare ---------- */
const PRESETS = [
  { id: "mode", title: "Hybrid or meaning-only search", desc: "Does keyword search add anything?", a: { label: "Hybrid", mode: "hybrid" }, b: { label: "Meaning only", mode: "dense" } },
  { id: "verify", title: "Figure check on or off", desc: "Re-asks when a figure is not in the pages.", a: { label: "Check off", verify: "off" }, b: { label: "Check on", verify: "on" } },
  { id: "parent", title: "With or without surrounding text", desc: "Whole section around the match, or only the match.", a: { label: "Section context", hierarchy: "parent" }, b: { label: "Match only", hierarchy: "flat" } },
];
let preset = PRESETS[0];
function renderPresets() {
  $("#presets").innerHTML = PRESETS.map(p => `<button class="preset" type="button" data-id="${p.id}" aria-pressed="${p === preset}"><b>${esc(p.title)}</b><span>${esc(p.desc)}</span></button>`).join("");
  $$(".preset").forEach(b => b.onclick = () => { preset = PRESETS.find(p => p.id === b.dataset.id); renderPresets(); });
}
const pageKeys = r => [...new Set(r.stages.reranked.map(c => c.doc + ":" + c.page_start))];
const label = k => { const [d, p] = k.split(":"); return `${short(state.docs.find(x => x.id === d)?.company || d)} p. ${p}`; };
function side(tag, cfg, res, other, ex) {
  const mine = pageKeys(res), theirs = new Set(pageKeys(other));
  const chips = mine.map(k => { const [d, p] = k.split(":"); return `<button type="button" class="pgchip ${theirs.has(k) ? "" : "only"} ${ex && ex.docs.includes(d) && ex.gold_pages.includes(+p) ? "gold" : ""}" data-k="${esc(k)}">${esc(label(k))}</button>`; }).join("");
  return `<div class="card"><div class="side-h"><b>${esc(cfg.label)}</b><span>Setting ${tag}</span></div><div class="ans-slot" data-tag="${tag}"></div>${res.failed && !res.answer ? failNote(res) : badges(res)}
    <div class="pgset" aria-label="Pages used">${chips}</div>
    <div class="delta">Search ${res.timing.retrieval_s} s, answer ${res.timing.generation_s ?? "n/a"} s. Highlighted pages were retrieved by this setting only.</div></div>`;
}
let cseq = 0;
async function compare() {
  const cmy = ++cseq, q = $("#cq").value.trim(); if (!q) { toast("Type a question or choose an example."); return; }
  const base = { question: q, docs: state.cmpDocs || [], mode: "hybrid", verify: "off", hierarchy: "parent" };
  const out = $("#cmp-out"), btn = $("#cmp-go"); btn.disabled = true; btn.textContent = "Comparing";
  out.innerHTML = [0, 1].map(() => `<div class="card"><div class="loading"><div class="sk" style="width:90%"></div><div class="sk" style="width:70%"></div><div class="sk" style="width:50%"></div></div></div>`).join("");
  try {
    const [a, b] = await Promise.all([api("/api/ask", { ...base, ...preset.a, label: undefined }), api("/api/ask", { ...base, ...preset.b, label: undefined })]);
    if (cmy !== cseq) return;
    const ex = state.examples.find(e => e.question === q);
    out.innerHTML = side("A", preset.a, a, b, ex) + side("B", preset.b, b, a, ex);
    [a, b].forEach((r, i) => { const s = $(`.ans-slot[data-tag="${"AB"[i]}"]`); if (r.answer) s.append(answerNode(r)); });
    $$(".pgchip", out).forEach(c => c.onclick = () => { const [d, p] = c.dataset.k.split(":"); lightbox(`/api/page/${encodeURIComponent(d)}/${p}`); });
    $$(".cite", out).forEach(c => c.onclick = () => toast("Open the same question in Ask to read the cited page."));
    $$("[data-open]", out).forEach(b => b.onclick = () => { const [d, p] = b.dataset.open.split("|"); lightbox(`/api/page/${encodeURIComponent(d)}/${p}`); });
  } catch (e) { if (cmy === cseq) out.innerHTML = `<div class="card" style="grid-column:1/-1"><div class="err">${esc(e.message)}</div></div>`; }
  finally { if (cmy === cseq) { btn.disabled = false; btn.textContent = "Compare"; } }
}
$("#cmp-go").onclick = compare; $("#cq").addEventListener("keydown", e => e.key === "Enter" && compare());

/* ---------- results ---------- */
const pct = v => v.toFixed(2);
function bars(rows, keys, max = 1) { return rows.map(r => `<div class="bar-grp"><div class="lbl"><b>${esc(r.label)}</b></div>${keys.map(([k, name, cls]) => `<div class="bar-row"><span>${name}</span><div class="track"><div class="fill ${r.base || cls === "a" ? "a" : ""}" style="width:${(r[k] / max) * 100}%"></div></div><span class="val">${pct(r[k])}</span></div>`).join("")}</div>`).join(""); }
function forest(rows) {
  const W = 640, L = 250, R = 20, T = 14, rowH = 30, H = T + rows.length * rowH + 34, lo = -0.22, hi = 0.14, x = v => L + (v - lo) / (hi - lo) * (W - L - R);
  let s = `<svg class="forest" viewBox="0 0 ${W} ${H}" role="img" aria-label="Change in recall at 5 versus the baseline for each setting, with 95 percent intervals"><line x1="${x(0)}" x2="${x(0)}" y1="${T - 6}" y2="${H - 30}" stroke="var(--line-2)" stroke-width="1.5"/>`;
  [-0.2, -0.1, 0, 0.1].forEach(t => s += `<text x="${x(t)}" y="${H - 12}" text-anchor="middle">${t > 0 ? "+" : ""}${t.toFixed(1)}</text>`);
  rows.forEach((r, i) => { const y = T + i * rowH + 14, col = r.sig ? "var(--accent)" : "var(--muted)";
    s += `<text x="${L - 12}" y="${y + 4}" text-anchor="end"><tspan class="st">${esc(r.study)}</tspan> ${esc(r.level)}</text><line x1="${x(r.lo)}" x2="${x(r.hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2"/><circle cx="${x(r.delta)}" cy="${y}" r="4.5" fill="${col}"/>`; });
  return s + "</svg>";
}
async function renderResults() {
  const d = await api("/api/results"), el = $("#results"); el.dataset.ready = 1;
  const h = d.headline;
  el.innerHTML = `
  <h1 class="h-serif">What the experiments found</h1>
  <p class="sub">121 questions on 12 Indian IPO filings. Every change was tested alone against the same baseline.</p>
  <div class="find"><h2>Hybrid search is the one change that clearly helped.</h2><p>Mixing keyword and meaning search finds the right page more often than meaning search alone (${h[0].v}, ${h[0].s}).</p></div>
  <div class="grid2">
    <div class="card chart-card"><h2>${esc(d.retrieval_mode.title)}</h2><div class="m">Share of required pages found in the top 5 (${esc(d.retrieval_mode.metric)})</div>
      <div class="bars">${bars(d.retrieval_mode.rows, [["dev", "Dev", ""], ["test", "Test", ""], ["pooled", "Both", "a"]])}</div><p class="note">${esc(d.retrieval_mode.note)}</p></div>
    <div class="card chart-card"><h2>${esc(d.ceiling.title)}</h2><div class="m">Judge score (${esc(d.ceiling.metric)}) for two generators</div>
      <div class="bars">${bars(d.ceiling.rows, [["nano", "Nano", "a"], ["bolt", "Bolt", ""]])}</div><p class="note">${esc(d.ceiling.note)}</p></div>
  </div>
  <div class="sect"><h2>Most other choices made no measurable difference.</h2><p>Each line is the change in recall at 5 against the baseline, with a 95% interval. Only the line that stays clear of zero is a real effect.</p></div>
  <div class="card chart-card"><div style="overflow-x:auto">${forest(d.ablations.rows)}</div></div>
  <div class="sect"><h2>Retrieval, not writing, is where answers are lost.</h2><p>Final test run on 68 questions. Table questions are solved; questions that span two filings are not.</p></div>
  <div class="grid2">
    <div class="card chart-card"><h2>By question type</h2><div class="m">Pages found (recall at 5) and answer quality (judge score)</div>
      <table class="tbl"><thead><tr><th>Type</th><th class="num">n</th><th>Recall</th><th>Judge</th></tr></thead><tbody>${d.by_type.rows.map(r => `<tr><td>${esc(r.type)}</td><td class="num">${r.n}</td><td><span class="mini"><i style="width:${r.recall * 100}%"></i></span> <span class="num">${pct(r.recall)}</span></td><td><span class="mini"><i style="width:${r.judge * 100}%"></i></span> <span class="num">${pct(r.judge)}</span></td></tr>`).join("")}</tbody></table></div>
    <div class="card chart-card"><h2>${esc(d.taxonomy.title)}</h2><div class="m">Every answer sorted by what went right or wrong</div>
      <div class="stack" role="img" aria-label="Outcome shares">${d.taxonomy.rows.map(r => `<i class="t-${r.tone}" style="width:${r.n / 68 * 100}%" title="${esc(r.label)}: ${r.n}"></i>`).join("")}</div>
      <ul class="tax">${d.taxonomy.rows.map(r => `<li><i class="t-${r.tone}"></i>${esc(r.label)}<span class="n">${r.n}</span></li>`).join("")}</ul></div>
  </div>
  <div class="sect"><h2>Checking figures costs little and catches rare errors.</h2><p>With the check on, answers that contain a figure missing from the retrieved text fell from ${(d.verification.rows[0].off * 100).toFixed(1)}% to ${(d.verification.rows[0].on * 100).toFixed(1)}%. The overall judge score did not change.</p></div>`;
}

/* ---------- server status ---------- */
async function pollStatus() {
  const el = $("#status"); let st;
  try { st = await api("/api/status"); } catch (e) { el.dataset.s = "down"; el.textContent = "Server not reachable"; return; }
  if (!st.ready) { el.dataset.s = "wait"; el.textContent = "Loading search models"; setTimeout(pollStatus, 1500); return; }
  el.dataset.s = st.answers_live ? "live" : "off";
  el.textContent = st.answers_live ? "Answers live" : "Saved answers only";
  el.title = st.answers_live ? `Answers are written by ${st.model}` : `No model key on this server. Search works, and ${st.saved_answers} saved answers replay. Start the server with NVIDIA_API_KEYS set to write new answers.`;
}

/* ---------- boot ---------- */
(async function init() {
  show(location.hash.slice(1) || "ask"); pollStatus();
  try { [state.docs, state.examples] = await Promise.all([api("/api/docs"), api("/api/examples")]); } catch (e) { toast("Could not reach the server: " + e.message); return; }
  renderDocs(); renderPresets();
  renderChips($("#chips"), pickExample);
  renderChips($("#cmp-chips"), e => { $("#cq").value = e.question; state.cmpDocs = e.docs; compare(); });
  if (!$("#result").innerHTML) pickExample(state.examples.find(e => e.id === "t-tab-05") || state.examples[0]);
})();
