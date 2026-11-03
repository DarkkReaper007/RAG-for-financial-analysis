"""Gold-set verification UI.   python review_app/app.py   ->  http://127.0.0.1:7861
Shows each gold item with its source PDF page(s). Decisions (accept / edit / reject) are appended to eval/gold/review_decisions.jsonl;
the frozen gold_v1.json is never modified (a later gold_v2 is built from the decisions)."""
import html
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import fitz
import gradio as gr
import gradio_client.utils as _gcu

# Gradio 4.44 + newer Pydantic: API-schema generation crashes on boolean JSON-schema values ("argument of type 'bool' is not iterable").
_orig_schema = _gcu._json_schema_to_python_type
_gcu._json_schema_to_python_type = lambda schema, defs=None: "Any" if isinstance(schema, bool) else _orig_schema(schema, defs)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ragbench import gold  # noqa: E402

ITEMS, GOLD_HASH = gold.load()
BY_ID = {i["id"]: i for i in ITEMS}
MAN = {d["id"]: d for d in json.load(open(ROOT / "data" / "manifest.json"))["docs"]}
DEC = ROOT / "eval" / "gold" / "review_decisions.jsonl"
CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)
_pdfs = {}
AUDIT = set(random.Random(42).sample([i["id"] for i in ITEMS], math.ceil(0.15 * len(ITEMS))))   # fixed 15% audit sample
REASONS = ["ambiguous question", "wrong answer", "wrong or missing evidence page", "trivial / low value", "not self-contained", "duplicate", "other"]


def decisions():
    out = {}
    if DEC.exists():
        for l in open(DEC):
            r = json.loads(l)
            out[r["id"]] = r
    return out


def current(item_id):
    """Item with the latest edits applied."""
    it = dict(BY_ID[item_id])
    e = (decisions().get(item_id) or {}).get("edits") or {}
    for k in ("question", "reference_answer", "evidence_groups"):
        if k in e:
            it[k] = e[k]
    return it


def pdf(doc):
    if doc not in _pdfs:
        _pdfs[doc] = fitz.open(ROOT / "data" / "prospectuses" / MAN[doc]["file"])
    return _pdfs[doc]


def page_image(doc, page):
    p = CACHE / f"{doc}_{page}.png"
    if not p.exists():
        pdf(doc)[page - 1].get_pixmap(dpi=85).save(str(p))
    return str(p)


def page_html(doc, page, item):
    text = html.escape(pdf(doc)[page - 1].get_text())
    marks = []
    prov = item["provenance"]
    if prov.get("quote") and prov.get("source_page") == page:
        marks.append(prov["quote"])
    marks += [q for q in prov.get("quotes", [])]
    marks += [str(f) for f in item.get("key_facts", []) if len(str(f)) >= 3]
    for m in marks:
        pat = r"\s+".join(re.escape(html.escape(w)) for w in m.split())
        text = re.sub(pat, lambda mo: f"<mark>{mo.group(0)}</mark>", text, count=3)
    return f"<div style='max-height:430px;overflow:auto;white-space:pre-wrap;font-size:13px;border:1px solid #ccc;padding:8px'>{text}</div>"


def status_of(i, dec):
    d = dec.get(i)
    return d["decision"] if d else "unreviewed"


def filtered(split, typ, status, audit_only):
    dec = decisions()
    ids = [i["id"] for i in ITEMS if (split == "all" or i["split"] == split) and (typ == "all" or i["type"] == typ)
           and (status == "all" or status_of(i["id"], dec) == status) and (not audit_only or i["id"] in AUDIT)]
    return ids


def pages_of(item):
    out = []
    for gi, g in enumerate(item["evidence_groups"]):
        doc = item["docs"][gi] if item["type"] == "multi_document" else item["docs"][0]
        out += [(f"{doc} p.{p}  (evidence group {gi + 1})", f"{doc}|{p}") for p in g]
    return out


def view(ids, idx, page_sel=None):
    dec = decisions()
    reviewed = sum(1 for i in BY_ID if i in dec)
    acc = sum(1 for d in dec.values() if d["decision"] == "accept")
    if not ids:
        return ids, idx, f"No items match. Reviewed {reviewed}/{len(BY_ID)}", "", "", "", "", "", gr.update(choices=[], value=None), None, "", ""
    idx = max(0, min(idx, len(ids) - 1))
    it = current(ids[idx])
    st = status_of(it["id"], dec)
    pg = pages_of(it)
    sel = page_sel if page_sel in [v for _, v in pg] else (pg[0][1] if pg else None)
    img = html_ = None
    if sel:
        doc, p = sel.split("|")
        img, html_ = page_image(doc, int(p)), page_html(doc, int(p), it)
    prog = f"**Item {idx + 1} / {len(ids)}**  |  overall reviewed {reviewed}/{len(BY_ID)} (accepted {acc})  |  gold hash `{GOLD_HASH[:10]}`"
    meta = (f"**{it['id']}**  ·  type `{it['type']}`  ·  split `{it['split']}`  ·  docs {it['docs']}  ·  topic {it['topic'] or '-'}  ·  "
            f"status **{st}**" + ("  ·  🔎 *in 15% audit sample*" if it["id"] in AUDIT else ""))
    if it["answerable"]:
        ev = "\n".join(f"- group {gi + 1}: pages {g}" + (f" (doc `{it['docs'][gi]}`)" if it["type"] == "multi_document" else "") for gi, g in enumerate(it["evidence_groups"]))
        ev += "\n\n*Any ONE page in a group is enough; every group must be satisfied.*"
    else:
        ev = "*Unanswerable: the correct behaviour is to say the documents don't contain it. Check the document really doesn't.*"
    prov = it["provenance"]
    pv = f"method `{prov['method']}`; verified by {', '.join(prov['verification'])}" + (f"; LLM ratings {prov['ratings']}" if prov.get("ratings") else "")
    return (ids, idx, prog, meta, it["question"], it["reference_answer"], json.dumps(it["evidence_groups"]), ev + f"\n\n*Key facts:* {it['key_facts']}\n\n*Provenance:* {pv}",
            gr.update(choices=pg, value=sel), img, html_, "")


def save(ids, idx, kind, reviewer, q, ref, ev, reason, note):
    if not ids:
        return view(ids, idx)
    if not reviewer.strip():
        return view(ids, idx)[:-1] + ("⚠️ enter your reviewer name first",)
    it = BY_ID[ids[idx]]
    rec = {"id": it["id"], "decision": kind, "reviewer": reviewer.strip(), "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": reason if kind == "reject" else None, "note": note}
    if kind == "edit":
        try:
            groups = json.loads(ev)
            assert isinstance(groups, list) and all(isinstance(g, list) for g in groups)
        except Exception:
            return view(ids, idx)[:-1] + ("⚠️ evidence must be JSON like [[19, 287]] (one list per group)",)
        rec["edits"] = {"question": q.strip(), "reference_answer": ref.strip(), "evidence_groups": groups}
    with open(DEC, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out = view(ids, idx + 1 if idx + 1 < len(ids) else idx)
    return out[:-1] + (f"✅ saved {kind} for {it['id']}",)


with gr.Blocks(title="Gold set verification") as app:
    gr.Markdown("## Gold set verification\nCheck each item against the source page: is the question clear, is the reference answer correct, are the evidence pages right (and are there other pages that also answer it)?")
    with gr.Row():
        reviewer = gr.Textbox(label="Reviewer name", scale=2)
        split = gr.Dropdown(["all", "dev", "test"], value="all", label="Split", scale=1)
        typ = gr.Dropdown(["all"] + sorted({i["type"] for i in ITEMS}), value="all", label="Type", scale=1)
        status = gr.Dropdown(["unreviewed", "all", "accept", "edit", "reject"], value="unreviewed", label="Status", scale=1)
        audit = gr.Checkbox(label="15% audit sample only", scale=1)
    ids, idx = gr.State([]), gr.State(0)
    prog, meta = gr.Markdown(), gr.Markdown()
    with gr.Row():
        with gr.Column(scale=5):
            q = gr.Textbox(label="Question (editable)", lines=2)
            ref = gr.Textbox(label="Reference answer (editable)", lines=3)
            ev_json = gr.Textbox(label="Evidence groups (editable JSON)", lines=1)
            ev_md = gr.Markdown()
            reason = gr.Dropdown(REASONS, value=REASONS[0], label="Reject reason")
            note = gr.Textbox(label="Note (optional)")
            with gr.Row():
                b_acc, b_edit, b_rej = gr.Button("✅ Accept as is", variant="primary"), gr.Button("✏️ Save edits"), gr.Button("❌ Reject", variant="stop")
            with gr.Row():
                b_prev, b_next = gr.Button("← Previous"), gr.Button("Next →")
            msg = gr.Markdown()
        with gr.Column(scale=6):
            page_dd = gr.Dropdown(label="Evidence page", choices=[])
            with gr.Row():
                extra_doc = gr.Dropdown(sorted(MAN), label="Check another page: doc", scale=2)
                extra_pg = gr.Number(label="page", precision=0, scale=1)
                b_extra = gr.Button("View page", scale=1)
            img = gr.Image(label="PDF page", type="filepath", height=520)
            txt = gr.HTML()
    outs = [ids, idx, prog, meta, q, ref, ev_json, ev_md, page_dd, img, txt, msg]

    def reset(s, t, st, a):
        return view(filtered(s, t, st, a), 0)
    for c in (split, typ, status, audit):
        c.change(reset, [split, typ, status, audit], outs)
    app.load(reset, [split, typ, status, audit], outs)
    b_next.click(lambda i, x: view(i, x + 1), [ids, idx], outs)
    b_prev.click(lambda i, x: view(i, x - 1), [ids, idx], outs)
    for btn, kind in ((b_acc, "accept"), (b_edit, "edit"), (b_rej, "reject")):
        btn.click(lambda i, x, r, qq, rr, ee, re_, nn, k=kind: save(i, x, k, r, qq, rr, ee, re_, nn), [ids, idx, reviewer, q, ref, ev_json, reason, note], outs)
    page_dd.change(lambda i, x, p: view(i, x, p)[8:11], [ids, idx, page_dd], [page_dd, img, txt])
    def extra(i, x, d, p):
        if not d or not p:
            return gr.update(), None, ""
        it = current(i[x]) if i else {"provenance": {}, "key_facts": []}
        return gr.update(), page_image(d, int(p)), page_html(d, int(p), it)
    b_extra.click(extra, [ids, idx, extra_doc, extra_pg], [page_dd, img, txt])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7861, show_error=True)
