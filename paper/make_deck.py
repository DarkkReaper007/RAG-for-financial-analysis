"""Builds paper/slides.pptx (16:9): the project talk. Same figures as the report.
    .venv/bin/python paper/make_deck.py            # writes the deck and runs the text-fit check
    .venv/bin/python paper/make_deck.py --preview  # also writes PNG previews to paper/_preview/
The companion deck that explains only the pipeline is paper/make_pipeline_deck.py."""
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from deckkit import (INK, LINE, MARK, MUTED, PAPER, SLATE, SLATE_T, TEAL, TEAL_T, WARN, WARN_T, WHITE, C, H, HEAD, MONO, W,
                     arrow, badge, box, card_text, line, measure, run_checks, text, write)

HERE = Path(__file__).parent
FIG = HERE / "figs"
DARK_CHIP = C("22343F")
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


def pic(slide, name, x, y, w, h):
    iw, ih = Image.open(FIG / name).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(str(FIG / name), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def new(title, notes, section=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
    text(s, 0.6, 0.4, 12.1, 1.0, title, size=28, bold=True, font=HEAD)
    if section:
        text(s, 0.6, 7.05, 6, 0.3, section, size=11, color=MUTED)
    text(s, 11.7, 7.05, 1.0, 0.3, str(len(prs.slides)), size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(slide, x, y, w, h, items, size=17, gap=10):
    paras = []
    for it in items:
        head, _, rest = it.partition("||")
        paras.append([(head, {"bold": True}), (rest, {"color": MUTED})] if rest else [head])
    return text(slide, x, y, w, h, paras, size=size, gap=gap)


def chip(slide, x, y, label, filled=False, size=14, h=0.36):
    w = measure(label, "Calibri", filled, size) / 72 + 0.36
    b = box(slide, x, y, w, h, fill=TEAL if filled else WHITE, line=TEAL if filled else LINE, lw=1.25, radius=h / 2)
    write(b, label, size=size, bold=filled, color=WHITE if filled else INK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return w


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = INK
text(s, 0.9, 1.3, 11.6, 2.3, "Which design choices actually matter in RAG over Indian IPO documents?", size=44, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 3.75, 11.6, 0.6, "A reproducible ablation study with an evaluation framework", size=24, color=C("C9D6DF"))
for i, lab in enumerate(["12 IPO filings, 6,718 pages", "121 questions with page evidence", "5 ablation studies"]):
    b = box(s, 0.9 + i * 3.95, 4.85, 3.7, 0.7, fill=DARK_CHIP, radius=0.1)
    write(b, lab, size=16, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.9, 6.5, 11, 0.4, "Sarvesh   |   Minor specialization, Big Data Analytics", size=16, color=C("8FA3B2"))
s.notes_slide.notes_text_frame.text = ("One-line pitch: we built a RAG system for SEBI offer documents, but the main result is a set of measurements that say which of its design choices matter. "
                                      "Scope in one line: 12 filings, 121 questions with page-level evidence, 5 single-factor studies. About 15 minutes plus questions.")

# ---------------------------------------------------------------- 2 motivation
s = new("Offer documents are long, full of tables and sometimes incomplete",
        "Why this is hard. DRHPs and RHPs run 400-700 pages and the same figure appears in the summary, the financial statements and the discussion. Investors need the exact figure and the page it came from. "
        "In a draft the final price is printed as [●], so a useful system must also say 'not in the document'. On the right is a real test question the pipeline answers correctly, with the page it cites.", "Motivation")
rows = [("400-700", "pages per filing", "The same figure appears in the summary, the statements and the discussion.", HEAD),
        ("Tables", "carry the answers", "Profit, borrowings, share capital and pay sit in tables.", HEAD),
        ("[●]", "marks undecided values", "A draft has no final price, so a good system must say so.", "Calibri")]
for i, (big, head_, sub, fnt) in enumerate(rows):
    y = 1.75 + i * 1.65
    b = box(s, 0.6, y, 2.1, 1.4, fill=TEAL_T, radius=0.08)
    write(b, big, size=28 if fnt == "Calibri" else 26, bold=True, font=fnt, color=TEAL, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 2.95, y + 0.1, 4.0, 1.3, [[(head_, {"bold": True, "size": 19})], [(sub, {"color": MUTED})]], size=16, gap=3)
text(s, 7.5, 1.7, 5.2, 0.35, "A question the pipeline answers correctly", size=16, bold=True, color=MUTED)
card_text(s, 7.5, 2.15, 5.2, 1.5, [[("Question", {"bold": True, "color": TEAL, "size": 14})], [("What was the profit for the year achieved by Hyundai in Fiscal 2022 compared to Fiscal 2021?", {"font": HEAD, "size": 18})]], fill=PAPER, pad=0.2, size=18, gap=4)
arrow(s, 10.1, 3.7, 10.1, 4.05, color=MUTED)
a = card_text(s, 7.5, 4.1, 5.2, 2.3, [[("Answer", {"bold": True, "color": TEAL, "size": 14})], [("₹29,015.91 million (up from ₹18,811.56 million in Fiscal 2021)", {"size": 21})]], fill=TEAL_T, pad=0.2, size=21, gap=4)
cb = box(s, 7.7, 5.7, 3.3, 0.38, fill=TEAL, radius=0.19)
write(cb, "Hyundai Motor India, page 314-315", size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 7.5, 6.5, 5.2, 0.4, "Every figure is found on the cited pages.", size=14, color=MUTED)

# ---------------------------------------------------------------- 3 research questions
s = new("Two research questions guide the whole study",
        "RQ1 asks which of the many design choices in a RAG pipeline are worth it. Most teams pick them by habit and each has a cost. We answer it with five one-factor-at-a-time ablations against a single baseline. "
        "RQ2 asks where the bottleneck is. If retrieval is the problem, a bigger model is a waste; if generation is the problem, better search is a waste. We answer it with a ceiling test that gives the model the correct pages.", "Research questions")
cards = [("RQ1", "Which pipeline components improve retrieval and answer quality?",
          ["Pipelines have many choices: how to cut text, how to write tables, which search, which embedding model, whether to check figures.",
           "Most are chosen by habit, and each costs time or compute: a larger embedding model is about 8 times slower to index.",
           "On Indian offer documents nobody has measured which choices pay off."],
          "Five studies, each changing one choice against one baseline."),
         ("RQ2", "Where is the bottleneck, in retrieval or in generation?",
          ["It decides where to spend effort: better search, or a bigger model.",
           "A bigger generator costs more: the 550B model is about twice as slow as the 30B one.",
           "If the right page is never found, no model can answer correctly."],
          "A ceiling test: give the model the correct pages and see how far the score rises.")]
for i, (tag, q, why, how) in enumerate(cards):
    x = 0.6 + i * 6.25
    box(s, x, 1.6, 5.85, 5.25, fill=PAPER, radius=0.08)
    p = box(s, x + 0.3, 1.85, 0.95, 0.42, fill=TEAL, radius=0.21)
    write(p, tag, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, 2.4, 5.25, 0.95, q, size=21, bold=True, font=HEAD)
    text(s, x + 0.3, 3.4, 5.25, 2.5, [[("Why it matters", {"bold": True, "color": TEAL, "size": 15})]] + [[w] for w in why], size=15, gap=4)
    text(s, x + 0.3, 6.0, 5.25, 0.8, [[("How we answer it  ", {"bold": True, "color": TEAL}), (how, {"color": INK})]], size=15)

# ---------------------------------------------------------------- 4 related work
s = new("Related work",
        "Three strands. Pipelines: hybrid search, reranking and context ordering are well studied, but not measured on long financial filings. Evaluation: RAGAS, ARES and RAGChecker score answers, mostly with an LLM judge; "
        "none of them checks retrieval against annotated evidence pages. Financial QA: FinanceBench covers US filings and shows that searching across many companies is far harder than searching one. "
        "Our gap: no benchmark or ablation study for Indian IPO documents with page-level evidence. We use annotated evidence pages and add rule-based checks that do not rely on a judge.", "Related work")
rw = [("RAG pipelines", "Hybrid search with rank fusion, cross-encoder reranking and context ordering are well studied.", "Which of these choices matter on long financial filings has not been measured."),
      ("RAG evaluation", "RAGAS, ARES and RAGChecker score answers, mostly with an LLM judge.", "None checks retrieval against annotated evidence pages."),
      ("Financial QA", "FinanceBench tests questions on US filings and finds that searching across many companies is much harder than searching one.", "It covers US filings only, not Indian offer documents.")]
for i, (h_, does, miss) in enumerate(rw):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.9, 3.55, fill=PAPER, radius=0.08)
    text(s, x + 0.25, 1.8, 3.4, 0.5, h_, size=21, bold=True, font=HEAD)
    text(s, x + 0.25, 2.4, 3.4, 1.4, does, size=16, color=MUTED)
    text(s, x + 0.25, 3.8, 3.4, 1.3, [[("What is missing", {"bold": True, "color": WARN, "size": 14})], [(miss, {})]], size=16, gap=3)
g = box(s, 0.6, 5.4, 12.1, 1.45, fill=TEAL, radius=0.08)
write(g, [[("Our gap", {"bold": True, "size": 22, "font": HEAD})], [("No benchmark and no ablation study exists for Indian IPO offer documents, and existing evaluations do not check which page the evidence came from.", {"size": 18})]], color=WHITE, margin=0.3, anchor=MSO_ANCHOR.MIDDLE, gap=4)

# ---------------------------------------------------------------- 5 pipeline diagram
s = new("The pipeline: eight stages, each one a setting we can switch",
        "Walk left to right. The slate stages run once per filing and are saved to disk as an index. The teal stages run for every question: two searches are merged, a second model reranks the shortlist, "
        "each hit is widened to its section, the language model answers with page citations, and an optional check verifies the figures. Search runs over all 12 filings at once, so the system must also pick the right company.", "System")
text(s, 0.6, 1.5, 8, 0.35, [[("Once per filing: build the index", {"bold": True, "color": SLATE})]], size=17)
top = [("12 PDF filings", "the input", None), ("Parse", "text, chapters, tables", 1), ("Chunk", "pieces and sections", 2), ("Embed", "384 numbers each", 3), ("Saved index", "searched later", None)]
tw, tg = 2.0, 0.525
tcx = []
for i, (nm, cap, n) in enumerate(top):
    x = 0.6 + i * (tw + tg)
    tcx.append(x + tw / 2)
    if i == 4:
        sh = box(s, x, 1.95, tw, 1.3, fill=SLATE, shape=MSO_SHAPE.CAN)
        write(sh, [[(nm, {"bold": True, "size": 17, "color": WHITE})], [(cap, {"size": 12, "color": C("C9D6DF")})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.05)
    else:
        sh = box(s, x, 1.95, tw, 1.3, fill=PAPER if n is None else SLATE_T, line=LINE if n is None else None, radius=0.08)
        write(sh, [[(nm, {"bold": True, "size": 17})], [(cap, {"size": 12, "color": MUTED})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.08)
        if n:
            badge(s, x + 0.08, 2.0, 0.3, str(n), SLATE, size=11)
    if i:
        arrow(s, x - tg + 0.04, 2.6, x - 0.04, 2.6, color=MUTED)
bot = [("Question", "typed by the user", None), ("Retrieve", "meaning + keyword", 4), ("Rerank", "best 5 kept", 5), ("Assemble", "widen, order", 6), ("Generate", "cites pages", 7), ("Verify", "figures, pages", 8), ("Answer", "with page citations", None)]
bw, bg = 1.45, 0.325
bx = []
for i, (nm, cap, n) in enumerate(bot):
    x = 0.6 + i * (bw + bg)
    bx.append(x + bw / 2)
    dark = n is None
    fill = INK if i == 0 else (TEAL if i == 6 else TEAL_T)
    sh = box(s, x, 4.5, bw, 1.5, fill=fill, radius=0.08)
    col = WHITE if dark else INK
    write(sh, [[(nm, {"bold": True, "size": 17, "color": col})], [(cap, {"size": 12, "color": C("C9D6DF") if dark else MUTED})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.06)
    if n:
        badge(s, x + 0.06, 4.55, 0.3, str(n), TEAL, size=11)
    if i:
        arrow(s, x - bg + 0.03, 5.25, x - 0.03, 5.25, color=MUTED)
# the index feeds Retrieve
line(s, tcx[4], 3.25, tcx[4], 3.8, color=SLATE, width=1.75)
line(s, tcx[4], 3.8, bx[1], 3.8, color=SLATE, width=1.75)
arrow(s, bx[1], 3.8, bx[1], 4.5, color=SLATE, width=1.75)
text(s, 4.6, 3.42, 4.5, 0.32, "the index is searched for every question", size=12, color=SLATE, align=PP_ALIGN.CENTER)
text(s, 0.6, 4.0, 2.4, 0.35, [[("For every question", {"bold": True, "color": TEAL})]], size=17)
text(s, 0.6, 6.4, 12.1, 0.4, "Slate stages run once per filing. Teal stages run for every question. Search covers all 12 filings at once.", size=14, color=MUTED)

# ---------------------------------------------------------------- 6 gold set
s = new("How the 121-question gold set was built",
        "A gold set is the answer key. We sampled 366 pages across the 12 filings and had an LLM draft one question per page; 227 passed automatic checks (the quote is on the page, the question is self-contained, the company is named, no copying, a second LLM agrees). "
        "102 were chosen to balance question types. The review by hand removed 15 and 10 were refilled from the next-best candidates, so the net change is 5 fewer, giving 97 (5 short because too few acceptable candidates remained for some filings and types). We then looked for every other page that supports each answer: 478 pages were judged and the average question ended up with 2.83 evidence pages. "
        "Finally 12 two-filing and 12 unanswerable questions were added, giving 121. Evidence is stored in groups: any page in a group counts, and every group must be hit. "
        "The set was audited by the author only, and the split between dev and test is by filing. The caveats are on the limitations slide.", "Evaluation framework")
flow = [("Sample", "366", "pages from the 12 filings"), ("Draft and filter", "227", "LLM questions that pass the automatic checks"), ("Select", "102", "chosen to balance question types"),
        ("Review", "97", "net 5 fewer after review by hand"), ("Expand evidence", "2.8", "evidence pages per question on average"), ("Add special cases", "121", "with 12 two-filing and 12 unanswerable")]
fw, fg = 1.85, 0.2
for i, (st, big, lab) in enumerate(flow):
    x = 0.6 + i * (fw + fg)
    last = i == len(flow) - 1
    box(s, x, 1.5, fw, 2.05, fill=TEAL if last else TEAL_T, radius=0.08)
    text(s, x + 0.12, 1.6, fw - 0.24, 0.35, st, size=14, bold=True, color=WHITE if last else TEAL)
    text(s, x + 0.12, 1.95, fw - 0.24, 0.65, big, size=32, bold=True, font=HEAD, color=WHITE if last else INK)
    text(s, x + 0.12, 2.65, fw - 0.24, 0.85, lab, size=14, color=C("E6F3F0") if last else MUTED)
    if i:
        arrow(s, x - fg + 0.02, 2.5, x - 0.02, 2.5, color=MUTED, width=1.5)
# artefact A: one gold item
box(s, 0.6, 3.85, 4.1, 3.0, fill=PAPER, radius=0.08)
text(s, 0.8, 3.95, 3.7, 0.3, "One gold item", size=16, bold=True, font=HEAD)
text(s, 0.8, 4.3, 3.7, 1.55, [[("Question  ", {"bold": True, "color": TEAL}), ("What was the total expenses amount for Zomato in Fiscal 2020 compared to Fiscal 2019?", {})],
                              [("Answer  ", {"bold": True, "color": TEAL}), ("₹50,063.11 million, up from ₹36,081.51 million", {})]], size=14, gap=4)
text(s, 0.8, 5.95, 3.7, 0.3, "Evidence pages (any one counts)", size=13, bold=True, color=MUTED)
cx = 0.8
for p in ("76", "224", "319", "322"):
    cx += chip(s, cx, 6.3, "p. " + p, filled=True, size=13, h=0.34) + 0.1
# artefact B: evidence groups
box(s, 4.85, 3.85, 3.85, 3.0, fill=PAPER, radius=0.08)
text(s, 5.05, 3.95, 3.5, 0.3, "Evidence groups", size=16, bold=True, font=HEAD)
text(s, 5.05, 4.3, 3.5, 0.55, "A two-filing question has one group per filing. A page from each group is needed.", size=13, color=MUTED)
for k, (lab, pg) in enumerate((("Zomato", "p. 83"), ("Delhivery", "p. 85"))):
    gx = 5.05 + k * 1.75
    g_ = box(s, gx, 5.05, 1.6, 1.35, fill=WHITE, line=TEAL, lw=1.5, radius=0.1)
    text(s, gx + 0.1, 5.1, 1.4, 0.3, lab, size=13, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    chip(s, gx + 0.3, 5.5, pg, filled=True, size=13, h=0.34)
    text(s, gx + 0.1, 5.95, 1.4, 0.3, "group " + str(k + 1), size=12, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 5.05, 6.5, 3.5, 0.3, "Unanswerable questions have no group.", size=12, color=MUTED)
# artefact C: split
box(s, 8.85, 3.85, 3.85, 3.0, fill=PAPER, radius=0.08)
text(s, 9.05, 3.95, 3.5, 0.3, "Dev and test split", size=16, bold=True, font=HEAD)
wd, wt = 3.45 * 53 / 121, 3.45 * 68 / 121
d_ = box(s, 9.05, 4.4, wd, 0.7, fill=TEAL, radius=0.06)
write(d_, "dev 53", size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
t_ = box(s, 9.05 + wd + 0.03, 4.4, wt - 0.03, 0.7, fill=SLATE, radius=0.06)
write(t_, "test 68", size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 9.05, 5.2, 3.5, 0.6, "Split by filing: 5 filings for dev, 7 for test.", size=14, color=MUTED)
text(s, 9.05, 5.85, 3.5, 0.9, "7 question types: table, fact, number, list, definition, two-filing, unanswerable.", size=14, color=MUTED)

# ---------------------------------------------------------------- 7 question types
s = new("Seven question types test different skills",
        "The 121 questions fall into seven types. Factoid, numeric, table and list questions each have about 21 to 24 items; definition questions are few by design (6). Two-filing questions need one lookup in each of two filings. "
        "Unanswerable questions have no answer in the documents (for example a share price after listing, which a draft does not contain): the right behaviour is to decline. "
        "Figures are checked against key facts, lists and descriptions by the judge, and unanswerable questions by whether the system declines. Counts are dev / test: 53 and 68 in total.", "Evaluation framework")
types = [("Factoid", "What was the revenue from operations for Zomato in Fiscal 2021?", "one fact from the text", "10 / 13", TEAL_T),
         ("Numeric", "What is the minimum bid amount for Retail Individual Bidders in Zomato's offer?", "an exact figure with its unit", "10 / 13", TEAL_T),
         ("Table", "What was the total expenses amount for Zomato in Fiscal 2020 compared to Fiscal 2019?", "values read from a table", "10 / 14", TEAL_T),
         ("List", "What are the specific qualitative strengths supporting Zomato's offer price?", "several items, all needed", "10 / 11", TEAL_T),
         ("Definition", "What does the abbreviation MCA signify?", "a term defined in the document", "3 / 3", TEAL_T),
         ("Two filings", "Who are the statutory auditors of Zomato and of Delhivery?", "one lookup in each of two filings", "5 / 7", SLATE_T),
         ("Unanswerable", "What was the closing price of Zomato shares on their first day of trading?", "declining when the answer is not there", "5 / 7", WARN_T)]
text(s, 0.6, 1.45, 2.0, 0.3, "Type", size=13, bold=True, color=MUTED)
text(s, 2.75, 1.45, 5.5, 0.3, "Example question", size=13, bold=True, color=MUTED)
text(s, 8.55, 1.45, 3.0, 0.3, "What it tests", size=13, bold=True, color=MUTED)
text(s, 11.6, 1.45, 1.1, 0.3, "Dev / test", size=13, bold=True, color=MUTED, align=PP_ALIGN.RIGHT)
for i, (nm, ex, what, cnt, fill) in enumerate(types):
    y = 1.8 + i * 0.7
    tcol = WARN if fill == WARN_T else (SLATE if fill == SLATE_T else TEAL)
    b = box(s, 0.6, y, 1.95, 0.6, fill=fill, radius=0.08)
    write(b, nm, size=15, bold=True, color=tcol, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 2.75, y, 5.6, 0.6, ex, size=14, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.55, y, 3.0, 0.6, what, size=14, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 11.6, y, 1.1, 0.6, cnt, size=15, bold=True, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    line(s, 0.6, y + 0.65, 12.7, y + 0.65, color=LINE, width=0.75)
text(s, 0.6, 6.75, 12.1, 0.3, "Total 53 dev and 68 test questions. Figures are checked against key facts, lists by the judge, and unanswerable questions by whether the system declines.", size=12, color=MUTED)

# ---------------------------------------------------------------- 7b scoring
s = new("How each answer is scored",
        "Three checks, each comparing two things. Retrieval: do the pages the system retrieved include the evidence pages from the gold set? Grounding: are the figures in the answer in the retrieved text, are the cited pages real, "
        "and did the system decline when it should (and only then)? The judge: an LLM compares the answer with the gold answer and scores coverage, faithfulness and redundancy. "
        "Retrieval and grounding follow fixed rules, so they give the same score every time; only the judge uses an LLM. The judge agrees with a rule-based key-fact check on 84% of 83 answers.", "Evaluation framework")
lay = [("Retrieval", "fixed rules", "Pages the system retrieved", "Evidence pages in the gold set", "Did the right page come back?", "Recall, nDCG and MRR at the page level", TEAL_T, TEAL),
       ("Grounding", "fixed rules", "The answer", "The retrieved text", "Are the figures and citations real?", "Also: did it decline when it should, and only then", TEAL_T, TEAL),
       ("Judge", "an LLM (gpt-oss-120b) compares", "The answer", "The gold answer", "Is it complete, accurate and concise?", "Scores coverage, faithfulness and redundancy", WARN_T, WARN)]
for i, (nm, how, a_, b_, q, sub, fill, col) in enumerate(lay):
    y = 1.6 + i * 1.6
    lb = box(s, 0.6, y, 2.3, 1.4, fill=fill, radius=0.08)
    write(lb, [[(nm, {"bold": True, "size": 21, "font": HEAD, "color": col})], [(how, {"size": 14, "color": MUTED})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.1, gap=2)
    box(s, 3.2, y, 2.6, 1.4, fill=PAPER, line=LINE, radius=0.08)
    text(s, 3.3, y + 0.1, 2.4, 1.2, a_, size=16, bold=True, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    vs = box(s, 5.95, y + 0.45, 0.5, 0.5, fill=WHITE, line=LINE, shape=MSO_SHAPE.OVAL)
    write(vs, "vs", size=13, bold=True, color=MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 6.6, y, 2.6, 1.4, fill=PAPER, line=LINE, radius=0.08)
    text(s, 6.7, y + 0.1, 2.4, 1.2, b_, size=16, bold=True, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    arrow(s, 9.3, y + 0.7, 9.7, y + 0.7, color=MUTED)
    text(s, 9.8, y + 0.1, 2.9, 1.3, [[(q, {"bold": True, "size": 18})], [(sub, {"color": MUTED})]], size=14, gap=3, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 6.45, 12.1, 0.45, "Retrieval and grounding follow fixed rules, so they score the same every time. Only the judge uses an LLM.", size=15, color=MUTED)

# ---------------------------------------------------------------- 7c the judge
s = new("The LLM judge: what it does and how we checked it",
        "The judge is gpt-oss-120b, accessed through Groq, a different model family from the Nemotron generators and from the model that drafted the questions. It sees only the question, the gold answer and the predicted answer, never the retrieved text. "
        "It splits the gold answer into items and grades coverage, faithfulness and redundancy on five levels (excellent 1.0, good 0.75, acceptable 0.5, bad 0.25, extremely bad 0); the overall score is their mean. "
        "An abstention on an answerable question scores 0 without a call. To check the judge we compared its coverage with a rule-based key-fact check on 83 dev answers: they agree on 84% (60 + 10 of 83), with a moderate correlation of 0.56. "
        "The judge has not been compared with human graders, which is on the limitations slide.", "Evaluation framework")
text(s, 0.6, 1.5, 6.4, 0.35, "What the judge does", size=18, bold=True, font=HEAD)
for k, lab in enumerate(("Question", "Gold answer", "Predicted answer")):
    b = box(s, 0.6, 2.0 + k * 0.75, 2.2, 0.6, fill=PAPER, line=LINE, radius=0.08)
    write(b, lab, size=15, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 2.85, 2.3 + k * 0.75, 3.35, 3.0 + (k - 1) * 0.1, color=MUTED, width=1.5)
jb_ = box(s, 3.4, 2.1, 1.65, 2.05, fill=WARN_T, line=WARN, lw=1.25, radius=0.1)
write(jb_, [[("LLM judge", {"bold": True, "size": 16, "color": WARN})], [("gpt-oss-120b", {"size": 14, "color": INK})], [("different model family", {"size": 12, "color": MUTED})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.08, gap=3)
for k, (nm, q) in enumerate((("Coverage", "all gold items present?"), ("Faithfulness", "anything wrong or invented?"), ("Redundancy", "anything repeated?"))):
    y = 2.0 + k * 0.75
    arrow(s, 5.1, 3.1, 5.55, y + 0.3, color=MUTED, width=1.5)
    b = box(s, 5.6, y, 1.9, 0.6, fill=TEAL_T, radius=0.08)
    write(b, [[(nm, {"bold": True, "size": 14, "color": TEAL})], [(q, {"size": 11, "color": MUTED})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.04)
text(s, 0.6, 4.4, 6.9, 1.9, [[("Each axis has five levels, from excellent (1.0) to extremely bad (0). The overall score is the mean of the three.", {})],
                             [("The judge never sees the retrieved text, only the three answers above.", {"color": MUTED})],
                             [("An abstention on an answerable question scores 0 without a call.", {"color": MUTED})]], size=15, gap=6)
text(s, 8.0, 1.5, 4.7, 0.35, "Is the judge reliable?", size=18, bold=True, font=HEAD)
text(s, 8.0, 1.9, 4.7, 0.55, "Its coverage score against a rule-based key-fact check, on 83 dev answers", size=13, color=MUTED)
cells = [(60, "agree", TEAL_T, TEAL), (9, "disagree", WARN_T, WARN), (4, "disagree", WARN_T, WARN), (10, "agree", TEAL_T, TEAL)]
text(s, 9.3, 2.5, 1.7, 0.4, "all key facts found", size=12, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 11.1, 2.5, 1.6, 0.4, "a key fact missing", size=12, color=MUTED, align=PP_ALIGN.CENTER)
text(s, 8.0, 3.2, 1.3, 0.8, "judge: complete", size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
text(s, 8.0, 4.5, 1.3, 0.8, "judge: parts missing", size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
for k, (n_, lab, fill, col) in enumerate(cells):
    cx_, cy_ = 9.3 + (k % 2) * 1.8, 2.95 + (k // 2) * 1.3
    b = box(s, cx_, cy_, 1.7, 1.2, fill=fill, radius=0.08)
    write(b, [[(str(n_), {"bold": True, "size": 28, "font": HEAD, "color": col})], [(lab, {"size": 13, "color": MUTED})]], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.04)
text(s, 8.0, 5.5, 4.7, 0.5, [[("They agree on 84% of answers.", {"bold": True})]], size=16)
card_text(s, 8.0, 6.0, 4.7, 0.8, [[("Not compared with human graders.", {"color": WARN, "bold": True})]], fill=WARN_T, pad=0.15, size=14, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 8 ablations
s = new("Five single-factor ablations against one baseline",
        "One factor varies at a time; everything else stays at the baseline (the filled chip). Chunking: fixed-size windows, cutting by chapter, adding the parent section the model reads, and adding the chapter name to each chunk (the 'header'). "
        "Table format: inline leaves tables as ordinary page text; flat makes one chunk per table with rows written as 'a | b | c'; markdown adds the caption and a header row. Retrieval: meaning only (dense), keyword only (BM25) or both (hybrid). "
        "Verification checks figures and cited pages. Embedder: bge-small, bge-large or e5-large. Held fixed: equal-weight merge, MiniLM reranker, context ordering, parent size cap, no query rewriting.", "Experiments")
text(s, 0.6, 1.4, 12.1, 0.35, "Each study changes one factor. Everything else stays at the baseline (filled chip).", size=15, color=MUTED)
abl = [("Chunking", "How text is cut, and what the model reads around a match", [("Fixed-size windows", 0), ("By chapter", 0), ("By chapter + parent section", 1), ("Also chapter name in each chunk", 0)]),
       ("Table format", "How a table is turned into text", [("Left in page text", 0), ("Own chunk, rows a | b | c", 0), ("Own chunk, markdown + caption", 1)]),
       ("Retrieval mode", "How passages are found", [("Meaning only", 0), ("Keyword only", 0), ("Hybrid", 1)]),
       ("Verification", "Check figures and pages in the answer", [("Off", 1), ("On", 0)]),
       ("Embedding model", "The model that turns text into numbers", [("bge-small", 1), ("bge-large", 0), ("e5-large", 0)])]
y = 1.95
for nm, what, chips in abl:
    cx_, cy_, rows_ = 4.3, y, 1
    for lab, base in chips:
        wch = measure(lab, "Calibri", bool(base), 14) / 72 + 0.36
        if cx_ + wch > 12.7:
            cx_, cy_, rows_ = 4.3, cy_ + 0.44, rows_ + 1
        chip(s, cx_, cy_ + 0.12, lab, filled=bool(base))
        cx_ += wch + 0.12
    rh = max(0.8, 0.44 * rows_ + 0.35)
    text(s, 0.6, y + 0.05, 3.5, rh, [[(nm, {"bold": True, "size": 18})], [(what, {"color": MUTED, "size": 13})]], size=18, gap=1)
    line(s, 0.6, y + rh - 0.04, 12.7, y + rh - 0.04, color=LINE, width=1)
    y += rh
text(s, 0.6, 6.45, 12.1, 0.4, "Generators compared: bolt (small, 4B class), nano (30B reasoning) and ultra (550B). The answer judge is gpt-oss-120b.", size=14, color=MUTED)

# ---------------------------------------------------------------- 9 retrieval mode
s = new("Hybrid search beats dense-only search; keyword search alone is just as good",
        "Pooled over 109 questions, dense is 0.11 below hybrid, McNemar p = 0.015. BM25 alone matches hybrid pooled, so we can say hybrid is better than dense, not that it is better than keyword search. This is the only significant effect in the study.", "Results")
pic(s, "retrieval_mode.png", 0.6, 1.6, 7.6, 5.0)
card_text(s, 8.6, 1.7, 4.1, 1.1, [[("What recall@5 means", {"bold": True, "size": 15})], [("The share of questions where the top 5 pages include the evidence needed.", {"color": MUTED})]], fill=PAPER, pad=0.2, size=15, gap=3)
bullets(s, 8.6, 3.05, 4.1, 3.4, ["0.578 vs 0.468||  hybrid vs dense-only, pooled over dev and test", "The only clear effect||  found in the whole study", "Keyword search alone||  scores 0.578 too: lower on dev, higher on test"], size=17, gap=14)

# ---------------------------------------------------------------- 10 within noise
s = new("The other four choices made no reliable difference",
        "Each row is the change in recall at 5 versus the baseline with a 95% interval. Only dense retrieval excludes zero. These are 'no effect large enough to detect with 109 questions', not proof of no effect. Larger embedders cost about eight times more to index.", "Results")
pic(s, "forest.png", 0.6, 1.6, 7.9, 5.2)
card_text(s, 8.8, 1.7, 3.9, 2.45, [[("How to read the chart", {"bold": True, "size": 15})],
                                   [("Each row changes one thing against the baseline. The dot is the change in recall@5. The bar is the 95% range for the true change. A bar that crosses zero could be chance. Only dense-only search is clearly below zero.", {"color": MUTED})]], fill=PAPER, pad=0.2, size=14, gap=3)
bullets(s, 8.8, 4.4, 3.9, 2.4, ["Parent section||  helps the small generator: 0.50 to 0.59 (bolt)", "Verification||  unsupported figures fall from 5.8% to 1.9%; the score is unchanged"], size=15, gap=10)

# ---------------------------------------------------------------- 10b generators
s = new("Three generators compared: nano is the best choice",
        "We compared three language models from NVIDIA's Nemotron family as the answer writer: bolt (about 4B parameters, small and fast), nano (30B, a reasoning model) and ultra (550B). The choice rule was fixed before the final run: the highest judge score on dev among generators that fail on at most 10% of questions. "
        "That picked nano (0.594 against 0.580 for bolt and 0.566 for ultra). On test the order is ultra 0.650, nano 0.643, bolt 0.597, but the gap between ultra and nano is within noise and ultra takes about twice as long, so nano is the practical best choice. "
        "We kept the pre-specified choice and report the test ordering as it came out.", "Results")
gens = [("bolt", "about 4B parameters, small and fast", 0.580, 0.597, "1.9 s", False),
        ("nano", "30B reasoning model", 0.594, 0.643, "3.4 s", True),
        ("ultra", "550B, the largest", 0.566, 0.650, "6.5 s", False)]
for i, (nm, sub, dev, tst, tm, best) in enumerate(gens):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.9, 3.85, fill=TEAL_T if best else PAPER, line=TEAL if best else None, lw=2, radius=0.08)
    text(s, x + 0.25, 1.75, 2.2, 0.55, nm, size=26, bold=True, font=HEAD, color=TEAL if best else INK)
    if best:
        cb_ = box(s, x + 2.55, 1.85, 1.15, 0.38, fill=TEAL, radius=0.19)
        write(cb_, "chosen", size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.25, 2.35, 3.4, 0.35, sub, size=14, color=MUTED)
    for j, (lab, v) in enumerate((("Judge score on dev", dev), ("Judge score on test", tst))):
        yy = 2.95 + j * 0.85
        text(s, x + 0.25, yy, 2.6, 0.3, lab, size=13, color=MUTED)
        text(s, x + 2.75, yy, 0.9, 0.3, f"{v:.3f}", size=14, bold=True, align=PP_ALIGN.RIGHT)
        box(s, x + 0.25, yy + 0.33, 3.4, 0.2, fill=WHITE, line=LINE, lw=0.75, radius=0.05)
        box(s, x + 0.25, yy + 0.33, 3.4 * v, 0.2, fill=TEAL if best else C("8FA3B2"), radius=0.05)
    text(s, x + 0.25, 4.7, 3.4, 0.5, [[("Median time per answer, test  ", {"color": MUTED}), (tm, {"bold": True})]], size=14)
card_text(s, 0.6, 5.7, 12.1, 1.15, [[("Why nano  ", {"bold": True, "color": TEAL}),
                                     ("It scored highest on dev under a rule fixed in advance. On test ultra is 0.007 higher (0.650 against 0.643), within noise, and takes about twice as long. bolt is fastest but lower on dev.", {})]],
          fill=PAPER, pad=0.25, size=16, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 11 ceiling
s = new("Giving the generator the right pages lifts the score from 0.59 to 0.84",
        "Ceiling experiment on dev: the model is handed the gold evidence pages instead of the retrieved ones. The retrieval loss is +0.27 with a 95% interval of 0.10 to 0.43. Largest losses on two-filing and numeric questions. "
        "Generator choice: nano by the dev rule fixed before the final run; on test ultra is nominally higher but within noise and twice as slow.", "Results")
pic(s, "ceiling.png", 0.6, 1.7, 6.2, 4.9)
card_text(s, 7.2, 1.7, 5.5, 1.2, [[("The ceiling test", {"bold": True, "size": 15})], [("Hand the model the correct pages and skip retrieval. The gap to the normal score is what retrieval loses.", {"color": MUTED})]], fill=PAPER, pad=0.2, size=15, gap=3)
bullets(s, 7.2, 3.15, 5.5, 3.5, ["Retrieval is the bottleneck||  the gap to 0.84 is mostly pages that were never found", "Biggest losses||  two-filing questions (nano 0.20 to 0.60) and numeric questions (0.56 to 0.93)", "Same pattern for bolt||  0.58 rises to 0.79 with the right pages"], size=17, gap=14)

# ---------------------------------------------------------------- 12 final test
s = new("On the held-out test, tables are solved and two-filing questions fail",
        "Final one-shot run, 68 questions, nano. 34 good answers and 7 correct abstentions. 20 failures start with evidence not retrieved. Only 7 fail with the evidence in hand. Oracle retrieval (right filing only) gives 0.787 recall.", "Results")
pic(s, "by_type.png", 0.5, 1.6, 6.3, 3.4)
pic(s, "taxonomy.png", 6.9, 1.6, 6.0, 3.4)
bullets(s, 0.7, 5.3, 12, 1.6, ["recall@5 0.656||  judge score 0.651, faithfulness 0.824, correct refusal on 7 of 7 unanswerable questions", "Most failures start in retrieval||  20 of 27 failures had no evidence retrieved; searching only the right filing lifts recall to 0.787"], size=18, gap=8)

# ---------------------------------------------------------------- 13 demo
s = new("Prospectus Desk: read the answer, then check it on the source page",
        "Live demo: run python -m demo.server. Show Ask with the Hyundai profit question, click the citation to open page 314 with the figures highlighted, open the retrieval trace, then Compare hybrid vs dense on the two-filing question, then Results. Examples are cached so the demo works offline.", "Demo")
pic(s, "demo_page.png", 0.6, 1.5, 4.6, 5.4)
bullets(s, 5.6, 1.8, 7.2, 5.0, ["Ask||  answer with citation buttons; figures highlighted on the source page", "How the system found it||  the five steps, and the passages used with which search found each", "Compare||  hybrid vs dense, figure check on or off, with and without surrounding text", "Results||  the charts from this talk", "Honest examples||  six test questions, including a wrong figure and a missed two-filing question"], size=18, gap=12)

# ---------------------------------------------------------------- 14 limitations
s = new("Limitations",
        "Three limits to state before the questions. First, the gold set: LLM-drafted, checked by one person, small, so small effects cannot be ruled out, and the drafting model is also a tested generator. "
        "Second, one judge: the answer-quality score comes from one LLM that agrees with a rule-based check only moderately and was never compared with human graders. Third, scope: 12 filings from one regulator and no comparison with other published systems.", "Limitations")
lim = [("The gold set", "LLM-drafted and checked by the author only; no second annotator. It is small, so a null result means no effect large enough to detect. The drafting model is also one of the three generators tested."),
       ("One judge", "Answer quality comes from a single LLM (gpt-oss-120b). It agrees only moderately with a rule-based check and was never compared with human graders."),
       ("One corpus", "12 filings from one regulator, and no comparison with another published RAG system, so the findings are for this setting.")]
for i, (h_, body) in enumerate(lim):
    x = 0.6 + i * 4.1
    box(s, x, 1.7, 3.9, 4.4, fill=WARN_T, radius=0.08)
    text(s, x + 0.3, 1.95, 3.3, 0.6, h_, size=22, bold=True, font=HEAD, color=WARN)
    text(s, x + 0.3, 2.65, 3.3, 3.3, body, size=19)

# ---------------------------------------------------------------- 15 future work
s = new("Future work", "Order by value: check the gold set and the judge against people first, then table retrieval.", "Future work")
fut = [("Human checks", "Compare a random sample of the gold set and of the judge's scores with human annotators, with agreement figures."),
       ("Table-aware retrieval", "Keep row and column headers with each figure, rewrite computed questions, and train a reranker on financial text."),
       ("Questions across filings", "Add multi-hop retrieval and an explicit document router for the weakest category."),
       ("Broader evaluation", "More filings, final prospectuses with the offer price, other published RAG systems and an outside benchmark such as FinanceBench.")]
for i, (h_, body) in enumerate(fut):
    x, y = 0.6 + (i % 2) * 6.1, 1.7 + (i // 2) * 2.6
    box(s, x, y, 5.95, 2.4, fill=TEAL_T if (i + i // 2) % 2 == 0 else SLATE_T, radius=0.08)
    text(s, x + 0.3, y + 0.25, 5.35, 0.5, h_, size=21, bold=True, font=HEAD)
    text(s, x + 0.3, y + 0.85, 5.35, 1.45, body, size=17, color=MUTED)

# ---------------------------------------------------------------- 16 conclusions
s = new("Conclusions",
        "Two takeaways. Hybrid search is the only one of the five choices with a measurable effect, and it beats dense-only search but not keyword search. Retrieval, not generation, limits answer quality: with the right pages the score rises from 0.59 to 0.84, "
        "and questions that span two filings almost always fail.", "Conclusions")
concl = [("Hybrid search beat dense-only search", "Recall@5 is 0.58 against 0.47 for dense-only search. The other four choices made no reliable difference."),
         ("Retrieval limits quality", "With the right pages the judge score rises from 0.59 to 0.84, and questions that span two filings almost always fail.")]
for i, (h_, body) in enumerate(concl):
    y = 1.8 + i * 2.35
    b = box(s, 0.6, y, 4.6, 2.05, fill=TEAL_T, radius=0.08)
    write(b, h_, size=24, bold=True, font=HEAD, color=TEAL, margin=0.3, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 5.5, y, 7.2, 2.05, body, size=20, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 17 backup
s = new("Backup: generator comparison", "Reference table for questions about generator choice.", "Backup")
data = [("", "Dev overall", "Dev fail", "Dev median s", "Test overall", "Test fail", "Test median s"), ("bolt", "0.580", "0%", "1.05", "0.597", "0%", "1.87"),
        ("nano (chosen)", "0.594", "4%", "4.83", "0.643", "1%", "3.36"), ("ultra", "0.566", "0%", "8.54", "0.650", "0%", "6.48")]
tbl = s.shapes.add_table(4, 7, Inches(0.7), Inches(1.9), Inches(11.9), Inches(2.4)).table
for r, row in enumerate(data):
    tbl.rows[r].height = Inches(0.6)
    for c_, v in enumerate(row):
        cell = tbl.cell(r, c_)
        cell.fill.solid(); cell.fill.fore_color.rgb = INK if r == 0 else (WHITE if r % 2 else PAPER)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        run = cell.text_frame.paragraphs[0].add_run(); run.text = v
        run.font.size = Pt(17); run.font.bold = r == 0 or c_ == 0; run.font.name = "Calibri"
        run.font.color.rgb = WHITE if r == 0 else INK
text(s, 0.7, 4.7, 11.9, 0.9, "Judge: gpt-oss-120b, comparing the predicted answer with the gold answer. Selection rule, fixed before the final run: highest dev judge score among generators with a failure rate of at most 10%.", size=15, color=MUTED)

prs.save(HERE / "slides.pptx")
print("slides:", len(prs.slides))
run_checks(prs, HERE / "_preview" / "slides" if "--preview" in sys.argv else None)
