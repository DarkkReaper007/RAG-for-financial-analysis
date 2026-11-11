"""Builds paper/pipeline_slides.pptx: how the IPO RAG pipeline works, stage by stage (no results).
    .venv/bin/python paper/make_pipeline_deck.py            # writes the deck and runs the text-fit check
    .venv/bin/python paper/make_pipeline_deck.py --preview  # also writes PNG previews to paper/_preview/ (approximate renderer, no PowerPoint needed)
Fonts: Cambria (headings) and Calibri (text), both shipped with Office. Layout is checked against the real font metrics when the Office font files are present."""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from deckkit import (BODY, C, HEAD, INK, LINE, MARK, MONO, MUTED, PAPER, SLATE, SLATE_T, TEAL, TEAL_T, W, H, WARN, WARN_T, WHITE,
                     arrow, badge, box, card_text, line, run_checks, text, write)

HERE = Path(__file__).parent
OUT = HERE / "pipeline_slides.pptx"
STAGES = ["Parse", "Chunk", "Embed", "Retrieve", "Rerank", "Assemble", "Generate", "Verify"]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]


def stage_color(i):
    return SLATE if i < 3 else TEAL


def tracker(slide, current):
    n, gap, w = len(STAGES), 0.12, 1.4
    x0 = 0.6
    for i, nm in enumerate(STAGES):
        x = x0 + i * (w + (12.1 - n * w) / (n - 1))
        on = i == current
        sh = box(slide, x, 1.17, w, 0.38, fill=stage_color(i) if on else (SLATE_T if i < 3 else TEAL_T), shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.19, name=f"tracker {nm}")
        write(sh, f"{i + 1}  {nm}", size=13, bold=on, color=WHITE if on else MUTED, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def new(title, notes, stage=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
    text(s, 0.6, 0.38, 12.1, 0.7, title, size=32, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    if stage is not None:
        tracker(s, stage)
    text(s, 0.6, 7.05, 6, 0.3, "How the IPO RAG pipeline works", size=11, color=MUTED)
    text(s, 11.7, 7.05, 1.0, 0.3, str(len(prs.slides)), size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(slide, x, y, w, h, items, size=16, gap=10):
    """items: 'Head||rest' -> bold head then muted rest."""
    paras = []
    for it in items:
        head, _, rest = it.partition("||")
        paras.append([(head, {"bold": True}), (rest, {"color": MUTED})] if rest else [head])
    return text(slide, x, y, w, h, paras, size=size, gap=gap)


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = INK
text(s, 0.9, 1.7, 11.5, 1.2, "How the pipeline works", size=54, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 3.0, 11.5, 0.9, "From a 600-page offer document to an answer with page citations", size=26, color=C("C9D6DF"))
text(s, 0.9, 4.0, 11.5, 0.5, "Retrieval-augmented question answering over Indian IPO filings (DRHP and RHP)", size=18, color=C("8FA3B2"))
n = len(STAGES)
x0, x1 = 1.3, 12.0
for i, nm in enumerate(STAGES):
    cx = x0 + i * (x1 - x0) / (n - 1)
    if i:
        line(s, cx - (x1 - x0) / (n - 1) + 0.19, 6.0, cx - 0.19, 6.0, color=C("4A6070"), width=1.5)
    badge(s, cx - 0.19, 5.81, 0.38, str(i + 1), C("5E7C91") if i < 3 else C("19A58F"), size=13)
    text(s, cx - 0.7, 6.3, 1.4, 0.35, nm, size=14, color=C("C9D6DF"), align=PP_ALIGN.CENTER)
text(s, 0.9, 5.15, 6, 0.4, "Eight stages: three build the index, five answer a question", size=15, color=C("8FA3B2"))
s.notes_slide.notes_text_frame.text = ("This deck explains the pipeline only, not the results. Eight stages in two phases: the first three run once per filing to build an index, "
                                      "the last five run for every question. Each stage is a setting we can switch, which is what the ablation study varies. About 20 minutes.")

# ---------------------------------------------------------------- 2 overview
s = new("The pipeline has two phases",
        "Phase one runs once per filing and is saved to disk. Phase two runs per question and takes under a second for search plus a few seconds for the language model. "
        "Everything the model says is supposed to come from the passages that phase two hands it.")
text(s, 0.6, 1.4, 12.1, 0.4, [[("Once per filing: build the index", {"bold": True, "color": SLATE})]], size=18)
offline = [("Parse", "Read each page's text, label its chapter and find the tables"),
           ("Chunk", "Cut each chapter into 1,000-character pieces, grouped into 3,000-character sections"),
           ("Embed and index", "Turn every piece into 384 numbers; store them with a keyword index")]
for i, (nm, d) in enumerate(offline):
    x = 0.6 + i * 4.1
    card_text(s, x, 1.9, 3.9, 1.85, [[(nm, {"bold": True, "size": 22})], [(d, {"color": MUTED})]], fill=SLATE_T, pad=0.2, size=17, gap=6)
    badge(s, x + 3.4, 2.02, 0.36, str(i + 1), SLATE, size=13)
text(s, 0.6, 4.05, 12.1, 0.4, [[("For every question: find and answer", {"bold": True, "color": TEAL})]], size=18)
online = [("Retrieve", "Two searches, by meaning and by keyword, each return their top 10; merge them"),
          ("Rerank", "A second model reads question and passage together; keep the best 5"),
          ("Assemble", "Widen each hit to its section; put the best at the start and end"),
          ("Generate", "The language model answers from these passages only, citing pages"),
          ("Verify", "Optional: check every figure and page; rewrite once or decline")]
for i, (nm, d) in enumerate(online):
    x = 0.6 + i * 2.45
    card_text(s, x, 4.55, 2.3, 2.3, [[(nm, {"bold": True, "size": 20})], [(d, {"color": MUTED})]], fill=TEAL_T, pad=0.18, size=15, gap=6)
    badge(s, x + 1.84, 4.66, 0.36, str(i + 4), TEAL, size=13)

# ---------------------------------------------------------------- 3 parse
s = new("Parse: turn the PDF into pages, chapters and tables",
        "PyMuPDF reads each page. Paragraphs are kept as separate blocks in reading order. A page that opens a known SEBI chapter heading sets the chapter label, "
        "and later pages inherit it until the next heading. Tables are found with PyMuPDF's table finder, which is slow, so the result is cached per file.", stage=0)
bullets(s, 0.6, 1.95, 6.6, 4.9, [
    "Text, page by page||  extracted in reading order; page numbers and stray fragments are dropped",
    "Chapters||  35 known SEBI chapter names (Risk Factors, Objects of the Offer, Management's Discussion and Analysis ...). A page that opens one sets the label; later pages inherit it",
    "Tables||  found with PyMuPDF's table finder, kept with their page and caption. Detection takes minutes per filing, so it is cached",
    "Output||  for every page: its text blocks, its chapter and its tables"], size=19, gap=18)
px, py, pw, ph = 7.7, 1.95, 5.0, 4.85
box(s, px, py, pw, ph, fill=WHITE, line=LINE, lw=1.5, radius=0.05)
t1 = box(s, px + 0.25, py + 0.22, 1.0, 0.34, fill=TEAL, radius=0.17); write(t1, "page 312", size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
t2 = box(s, px + 1.4, py + 0.22, 3.35, 0.34, fill=SLATE_T, radius=0.17); write(t2, "chapter: Management's Discussion and Analysis", size=11, color=SLATE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
card_text(s, px + 0.25, py + 0.8, 4.5, 1.15, [[("text block", {"size": 11, "color": MUTED})], "Our total income increased by 15.85% to ₹479,660.48 million in Fiscal 2022 from ₹414,046.50 million in Fiscal 2021"], fill=PAPER, pad=0.12, size=13, gap=2)
card_text(s, px + 0.25, py + 2.1, 4.5, 0.85, [[("text block", {"size": 11, "color": MUTED})], "The increase in our revenue from operations was largely due to ..."], fill=PAPER, pad=0.12, size=13, gap=2)
card_text(s, px + 0.25, py + 3.05, 4.5, 1.55, [[("table (with its page and caption)", {"size": 11, "color": MUTED})], ""], fill=TEAL_T, pad=0.12, size=13, gap=2)
for r in range(3):
    for c in range(3):
        cx, cy = px + 0.45 + c * 1.4, py + 3.55 + r * 0.33
        box(s, cx, cy, 1.32, 0.27, fill=WHITE, line=LINE, lw=0.75, radius=0.02)

# ---------------------------------------------------------------- 4 chunk
s = new("Chunk: small pieces to search, bigger sections to read",
        "Search works best on small, focused pieces. Reading works best with the whole surrounding section. So we index about 1,000-character child chunks and remember which 3,000-character parent each belongs to. "
        "Chunks never cross a chapter. Every detected table is its own chunk. Across the 12 filings this gives 39,429 chunks, between 2,593 and 4,675 per filing.", stage=1)
bar = box(s, 0.6, 1.85, 10.34, 0.42, fill=SLATE_T, radius=0.05)
write(bar, "One chapter of a filing, in reading order", size=14, bold=True, color=SLATE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 2.4, 6, 0.3, "Child chunks: what the searches look at", size=14, bold=True)
cw, step = 1.38, 1.28
xs = [0.6 + i * step for i in range(8)]
for i, x in enumerate(xs):
    b = box(s, x, 2.75, cw, 0.7, fill=TEAL_T, line=TEAL, lw=1.0, radius=0.05)
    write(b, f"chunk {i + 1}", size=13, color=TEAL, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 3.52, 6, 0.3, "Neighbouring chunks share 150 characters", size=12, color=MUTED)
text(s, 0.6, 3.9, 6, 0.3, "Parent sections: what the model reads", size=14, bold=True)
for a, z, nm in ((0, 2, "parent 1"), (3, 5, "parent 2"), (6, 7, "parent 3")):
    x, x2 = xs[a] + 0.02, xs[z] + cw - 0.02
    b = box(s, x, 4.25, x2 - x, 0.5, fill=SLATE_T, line=SLATE, lw=1.0, radius=0.05)
    write(b, nm, size=13, color=SLATE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
tb_ = box(s, 11.15, 2.75, 1.55, 0.7, fill=WARN_T, line=WARN, lw=1.0, radius=0.05)
write(tb_, "table chunk", size=13, bold=True, color=WARN, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 11.15, 2.4, 1.6, 0.3, "Tables", size=14, bold=True)
cols = [("Child chunk", "About 1,000 characters, packed paragraph by paragraph. A chunk never crosses a chapter."),
        ("Parent section", "About 3,000 characters of neighbouring chunks from one chapter. Used later, to give the model more context."),
        ("Table chunk", "Each detected table becomes its own chunk, written as a text grid, with its page.")]
for i, (h_, d) in enumerate(cols):
    card_text(s, 0.6 + i * 4.1, 5.1, 3.9, 1.3, [[(h_, {"bold": True, "size": 16})], [(d, {"color": MUTED})]], fill=PAPER, pad=0.16, size=14, gap=3)
text(s, 0.6, 6.5, 12.1, 0.4, [[("Every chunk remembers its filing, page range, chapter and parent. ", {"color": MUTED}), ("39,429 chunks across the 12 filings", {"bold": True}), (" (2,593 to 4,675 per filing).", {"color": MUTED})]], size=14)

# ---------------------------------------------------------------- 5 embed and index
s = new("Embed and index: every chunk becomes a row of numbers",
        "An embedding model maps text to a list of 384 numbers so that passages with similar meaning end up close together. We store one row per chunk in a single numpy file, in the same order as the chunk list. "
        "Vectors are normalised, so a dot product is a cosine similarity. We also build a BM25 keyword index over the same chunks. There is no vector database: at this size an exact matrix multiplication takes milliseconds.", stage=2)
bullets(s, 0.6, 1.95, 6.3, 4.9, [
    "Embedding model||  bge-small turns each chunk into 384 numbers; chunks with similar meaning get similar numbers",
    "Normalised||  every vector has length 1, so a dot product between two of them is their cosine similarity",
    "One matrix||  rows are chunks, in the same order as the chunk file; saved to disk once per filing",
    "Keyword index||  BM25 over the same chunks, built when the index is loaded",
    "No vector database||  39,429 rows is about 60 MB; exact search over it takes milliseconds"], size=19, gap=16)
gx, gy, cell = 7.9, 2.35, 0.3
text(s, gx, 1.95, 4.8, 0.3, "384 numbers per chunk", size=14, bold=True, color=SLATE)
for r in range(10):
    for c in range(12):
        hot = r == 4
        box(s, gx + c * (cell + 0.02), gy + r * (cell + 0.02), cell, cell, fill=TEAL if hot else SLATE_T, radius=0.02)
text(s, gx, gy + 10 * (cell + 0.02) + 0.08, 4.8, 0.55, [[("39,429 chunks (rows). ", {"bold": True}), ("One highlighted row is one chunk's vector.", {"color": MUTED})]], size=13)

# ---------------------------------------------------------------- 6 retrieve
s = new("Retrieve: two searches, then merge the lists",
        "The question is searched twice. Meaning search embeds the question and multiplies it with the matrix to get a score for every chunk. Keyword search scores chunks by shared words, weighting rare words more. "
        "Each returns its top 10. Reciprocal rank fusion merges the two ranked lists: a chunk's score is the sum of 1 over 60 plus its rank in each list. Chunks that both searches like rise to the top. "
        "Search runs over all selected filings together, so there is no document router.", stage=3)
q = card_text(s, 0.6, 3.3, 1.7, 1.05, ["Question"], fill=INK, pad=0.12, size=16, bold=True, color=WHITE, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
card_text(s, 2.8, 1.95, 3.5, 1.7, [[("Meaning search", {"bold": True, "size": 17, "color": TEAL})], [("Embed the question, multiply by the matrix: one score per chunk. Keep the top 10.", {"color": MUTED})]], fill=TEAL_T, pad=0.16, size=14, gap=4)
card_text(s, 2.8, 4.05, 3.5, 1.7, [[("Keyword search (BM25)", {"bold": True, "size": 17, "color": TEAL})], [("Score chunks by shared words, rare words count more. Keep the top 10.", {"color": MUTED})]], fill=TEAL_T, pad=0.16, size=14, gap=4)
card_text(s, 6.75, 3.1, 1.5, 1.45, [[("Merge", {"bold": True, "size": 17})], [("10 candidates", {"color": MUTED})]], fill=PAPER, line=LINE, pad=0.14, size=14, gap=3, anchor=MSO_ANCHOR.MIDDLE)
arrow(s, 2.3, 3.6, 2.8, 2.8); arrow(s, 2.3, 4.05, 2.8, 4.9); arrow(s, 6.3, 2.8, 6.75, 3.5); arrow(s, 6.3, 4.9, 6.75, 4.2)
text(s, 0.6, 6.0, 7.7, 0.9, "Search runs over every selected filing at once (all 12 by default). There is no document router, so the system also has to find the right company.", size=14, color=MUTED)
card_text(s, 8.65, 1.95, 4.05, 4.85, [
    [("Reciprocal rank fusion", {"bold": True, "size": 17})],
    [("score = sum of 1 / (60 + rank)", {"font": MONO, "size": 14, "color": TEAL})],
    [("A passage that both searches like rises:", {"color": MUTED})],
    [("Passage A: 2nd by meaning, 5th by keyword", {"bold": True})],
    [("1/62 + 1/65 = 0.0315", {"font": MONO, "size": 14})],
    [("Passage B: 1st by meaning, not in keyword top 10", {"bold": True})],
    [("1/61 = 0.0164", {"font": MONO, "size": 14})],
    [("A ranks above B. Names, terms and figures match as words; paraphrases match by meaning.", {"color": MUTED})]], fill=PAPER, pad=0.22, size=16, gap=11)

# ---------------------------------------------------------------- 7 rerank
s = new("Rerank: a second model re-orders the shortlist",
        "The first stage compares the question with each passage separately, which is fast but coarse. A cross-encoder reads the question and the passage together and scores how well they match. "
        "It is too slow for 39,429 passages, so it only sees the 10 candidates. The picture is a real example, the Hyundai profit question: the passage on page 312 was third in the merged list and first after reranking, "
        "while the page 314-315 passage dropped from first to fifth.", stage=4)
text(s, 0.6, 1.8, 3.0, 0.3, "Merged list (10)", size=14, bold=True, color=SLATE)
text(s, 5.2, 1.8, 3.0, 0.3, "After reranking (5)", size=14, bold=True, color=TEAL)
known = {1: "p. 314-315", 3: "p. 312", 4: "p. 312-313", 5: "p. 310", 10: "p. 309"}
rowh, y0 = 0.43, 2.2
dest = {3: 1, 4: 2, 10: 3, 5: 4, 1: 5}
for r in range(1, 11):
    y = y0 + (r - 1) * rowh
    hot = r in known
    b = box(s, 0.6, y, 2.7, rowh - 0.07, fill=SLATE_T if hot else PAPER, line=SLATE if hot else None, lw=1.0, radius=0.04)
    write(b, [[(f"{r}   ", {"bold": True, "color": SLATE}), (known.get(r, "..."), {"color": INK if hot else MUTED})]], size=13, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)
ry = {k: 2.2 + (k - 1) * 0.86 for k in range(1, 6)}
for src, k in dest.items():
    y = ry[k]
    b = box(s, 5.2, y, 2.7, 0.55, fill=TEAL_T, line=TEAL, lw=1.0, radius=0.04)
    write(b, [[(f"{k}   ", {"bold": True, "color": TEAL}), (known[src], {})]], size=13, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)
    arrow(s, 3.3, y0 + (src - 1) * rowh + (rowh - 0.07) / 2, 5.2, y + 0.275, color=TEAL if k < src else WARN if k > src else MUTED, width=1.5)
text(s, 0.6, 6.6, 7.6, 0.35, "Real example, the Hyundai profit question. Teal arrows moved up, the amber arrow moved down.", size=12, color=MUTED)
card_text(s, 8.6, 1.95, 4.1, 4.85, [
    [("Why a second model", {"bold": True, "size": 17})],
    [("The first searches score the question and each passage separately. A cross-encoder reads them together, so it judges relevance more carefully.", {"color": MUTED})],
    [("Why only 10 candidates", {"bold": True, "size": 17})],
    [("It is accurate but slow, far too slow to run on all 39,429 passages.", {"color": MUTED})],
    [("Model", {"bold": True, "size": 17})],
    [("ms-marco-MiniLM-L-6-v2. Keep the best 5.", {"color": MUTED})]], fill=PAPER, pad=0.22, size=16, gap=9)

# ---------------------------------------------------------------- 8 assemble
s = new("Assemble: build the context the model will read",
        "Each of the five hits is replaced by its parent section when that is no longer than 4,000 characters, otherwise the small chunk is kept; two hits in the same parent are merged. "
        "Language models use the middle of a long context less well, so the order puts the best passages at the start and the end. Each block is labelled with company, section and page so that the model can cite.", stage=5)
steps = [("Widen", "Replace each hit by its parent section (up to 4,000 characters); if the section is longer, keep the small chunk. Two hits in one section count once."),
         ("Reorder", "Put the best passages at the start and the end: order 1, 3, 5, 4, 2. Models read the middle of a long context least well."),
         ("Label", "Head each block with company, section and page, so the model can cite the page it used.")]
for i, (h_, d) in enumerate(steps):
    y = 1.8 + i * 1.2
    badge(s, 0.6, y + 0.05, 0.42, str(i + 1), TEAL, size=14)
    text(s, 1.2, y, 5.7, 1.18, [[(h_, {"bold": True, "size": 18})], [(d, {"color": MUTED})]], size=15, gap=2)
cx = card_text(s, 7.3, 1.95, 5.4, 2.15, [[("[Context 1] Company: Hyundai Motor India Limited | Section: Management's Discussion And Analysis | Page: 312", {"font": MONO, "size": 12, "bold": True, "color": TEAL})],
                                         [("... Our total income increased by 15.85% to ₹479,660.48 million in Fiscal 2022 from ₹414,046.50 million in Fiscal 2021 ...", {"color": MUTED})]], fill=PAPER, line=LINE, pad=0.18, size=14, gap=6)
text(s, 7.3, 4.2, 5.4, 0.3, "A block as the model sees it. Five of them are sent, in the order below.", size=13, color=MUTED)
text(s, 0.6, 5.25, 8, 0.35, "Order of the five blocks in the prompt", size=14, bold=True)
order = [1, 3, 5, 4, 2]
for i, k in enumerate(order):
    b = box(s, 0.6 + i * 2.45, 5.65, 2.3, 0.75, fill=TEAL if k in (1, 2) else (TEAL_T if k in (3, 4) else PAPER), line=TEAL if k > 2 else None, lw=1.0, radius=0.05)
    write(b, f"best passage #{k}", size=14, bold=True, color=WHITE if k in (1, 2) else TEAL, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 6.5, 3, 0.3, "start", size=12, color=MUTED)
text(s, 5.2, 6.5, 3, 0.3, "middle", size=12, color=MUTED)
text(s, 9.7, 6.5, 3.0, 0.3, "end", size=12, color=MUTED, align=PP_ALIGN.RIGHT)

# ---------------------------------------------------------------- 9 generate
s = new("Generate: answer from the passages only",
        "The language model gets six rules and the labelled passages. Temperature is zero. The generator is Nemotron nano on NVIDIA's hosted API. It is a reasoning model, so it thinks before it writes, which is why answers take several seconds. "
        "Timeouts and empty replies are marked as failed and are never scored or shown as answers.", stage=6)
card_text(s, 0.6, 1.95, 6.5, 4.85, [
    [("The instructions the model receives", {"bold": True, "size": 17})],
    [("1  ", {"bold": True, "color": TEAL}), "Use only the numbered passages; no outside knowledge"],
    [("2  ", {"bold": True, "color": TEAL}), "Cite every fact as [Company, Page: N]"],
    [("3  ", {"bold": True, "color": TEAL}), "Quote figures exactly, with their unit; do not round or convert"],
    [("4  ", {"bold": True, "color": TEAL}), "A value shown as [●] is not decided in a draft: say so"],
    [("5  ", {"bold": True, "color": TEAL}), "If the answer is not in the passages, reply with a fixed sentence: \"The provided documents do not contain this information.\""],
    [("6  ", {"bold": True, "color": TEAL}), "Be concise"]], fill=PAPER, pad=0.25, size=17, gap=13)
card_text(s, 7.5, 1.95, 5.2, 1.95, [[("Example answer", {"bold": True, "size": 15, "color": MUTED})],
                                    [("₹29,015.91 million (up from ₹18,811.56 million in Fiscal 2021) ", {"size": 18}), ("[Hyundai Motor India, Page: 314-315]", {"font": MONO, "size": 12, "color": TEAL, "bold": True})]], fill=TEAL_T, pad=0.2, size=18, gap=6)
bullets(s, 7.5, 4.2, 5.2, 2.6, [
    "Model||  Nemotron nano (a reasoning model) on NVIDIA's hosted API",
    "Settings||  temperature 0, up to 4,096 tokens",
    "Failures are returned||  a timeout or empty reply is marked failed, never shown as an answer"], size=16, gap=12)

# ---------------------------------------------------------------- 10 verify
s = new("Verify: check the figures before showing the answer",
        "Verification is optional and off in the baseline configuration; the demo has a Strict figure check switch. It extracts every number from the answer and checks that it appears in the retrieved text, and that every cited page was among the retrieved pages. "
        "If a check fails the model is asked once to rewrite using only supported figures. If figures are still unsupported the system declines. It checks support, not whether the figure answers the question.", stage=7)
def node(x, y, w, h, title, sub, fill, tcolor=INK):
    return card_text(s, x, y, w, h, [[(title, {"bold": True, "size": 17, "color": tcolor})], [(sub, {"color": MUTED if fill != INK else C("C9D6DF")})]], fill=fill, pad=0.16, size=14, gap=3, anchor=MSO_ANCHOR.MIDDLE)
node(0.6, 2.0, 2.4, 1.3, "Answer", "from the model", PAPER)
node(3.5, 1.85, 4.2, 1.6, "Two checks", "Every number appears in the retrieved text. Every cited page was retrieved.", SLATE_T, SLATE)
node(8.3, 2.0, 2.2, 1.3, "Pass", "show the answer", TEAL_T, TEAL)
arrow(s, 3.0, 2.65, 3.5, 2.65); arrow(s, 7.7, 2.65, 8.3, 2.65)
text(s, 5.8, 3.62, 3.0, 0.3, "if a check fails", size=14, bold=True, color=WARN)
arrow(s, 5.6, 3.45, 5.6, 4.15, color=WARN)
node(3.5, 4.2, 4.2, 1.3, "Rewrite once", "Ask the model to use only supported figures", WARN_T, WARN)
arrow(s, 7.7, 4.85, 8.3, 4.35, color=WARN); arrow(s, 7.7, 4.85, 8.3, 5.55, color=WARN)
node(8.3, 3.7, 4.4, 1.3, "Check again: pass", "show the rewritten answer", TEAL_T, TEAL)
node(8.3, 5.2, 4.4, 1.3, "Still unsupported", "decline: \"not available (unverifiable figures ...)\"", WARN_T, WARN)
text(s, 0.6, 5.9, 6.4, 0.9, "Off in the baseline; the demo has a Strict figure check switch. It checks that figures are in the retrieved text, not that they answer the question.", size=14, color=MUTED)

# ---------------------------------------------------------------- 11 example
s = new("One question, all the way through",
        "A real run from the demo. The question is searched over the Hyundai filing's 2,593 passages. Each search keeps 10, three of which are in both lists. The merged list has 10 candidates, the reranker keeps 5, "
        "and the five are widened to their sections, 12,970 characters in all. Search takes 0.4 seconds, the model 7.1 seconds. The check finds all three figures in the retrieved pages.")
card_text(s, 0.6, 1.4, 12.1, 0.95, [[("Question  ", {"bold": True, "color": TEAL, "size": 15}), ("What was the profit for the year achieved by Hyundai in Fiscal 2022 compared to Fiscal 2021?", {"font": HEAD, "size": 18})]], fill=PAPER, pad=0.2, anchor=MSO_ANCHOR.MIDDLE, size=18)
facts = [("2,593", "passages in the Hyundai filing", SLATE_T, SLATE), ("10 + 10", "hits from the two searches, 3 in both lists", SLATE_T, SLATE),
         ("10", "merged candidates", TEAL_T, TEAL), ("5", "passages kept by the reranker", TEAL_T, TEAL), ("12,970", "characters sent to the model", TEAL_T, TEAL)]
for i, (big, lab, fill, col) in enumerate(facts):
    x = 0.6 + i * 2.45
    card_text(s, x, 2.7, 2.3, 1.75, [[(big, {"bold": True, "size": 34, "font": HEAD, "color": col})], [(lab, {"color": MUTED})]], fill=fill, pad=0.16, size=14, gap=4, anchor=MSO_ANCHOR.MIDDLE)
    if i:
        arrow(s, x - 0.15, 3.55, x, 3.55, color=MUTED, width=1.5)
card_text(s, 0.6, 4.8, 7.6, 1.8, [[("Answer", {"bold": True, "color": TEAL, "size": 16})],
                                  [("₹29,015.91 million (up from ₹18,811.56 million in Fiscal 2021) ", {"size": 22}), ("[Hyundai Motor India, Page: 314-315]", {"font": MONO, "size": 13, "color": TEAL, "bold": True})]], fill=TEAL_T, pad=0.22, size=22, gap=8, anchor=MSO_ANCHOR.MIDDLE)
card_text(s, 8.5, 4.8, 4.2, 1.8, [[("Timing and checks", {"bold": True, "size": 17})],
                                  [("Search 0.4 s, answer 7.1 s", {"color": MUTED})], [("All 3 figures appear in the retrieved pages", {"color": MUTED})], [("The citation points to a retrieved page", {"color": MUTED})]], fill=PAPER, pad=0.2, size=15, gap=5, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 12 settings
s = new("Every stage is a setting you can switch",
        "Each row is a field in the pipeline configuration. The five ablation studies change one row at a time and keep everything else at the default. The bottom box lists what is held fixed throughout.")
rows = [("Parse", "Table format", "markdown", "flat, inline"), ("Chunk", "Chunk boundaries", "by chapter", "fixed windows"),
        ("Chunk", "Parent section", "kept", "chunks only"), ("Chunk", "Chapter name in the text", "off", "on"),
        ("Embed", "Embedding model", "bge-small", "bge-large, e5-large"), ("Retrieve", "Search type", "hybrid", "meaning only, keyword only"),
        ("Verify", "Figure and citation check", "off", "on")]
tx, ty, tw = 0.6, 1.45, 12.1
gf = s.shapes.add_table(len(rows) + 1, 4, Inches(tx), Inches(ty), Inches(tw), Inches(0.46 * (len(rows) + 1)))
tbl = gf.table
for j, wd in enumerate((1.7, 4.0, 2.5, 3.9)):
    tbl.columns[j].width = Inches(wd)
for i in range(len(rows) + 1):
    tbl.rows[i].height = Inches(0.46)
heads = ["Stage", "Setting", "Default", "Other options"]
for i, row in enumerate([heads] + rows):
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK if i == 0 else (WHITE if i % 2 else PAPER)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.14)
        tf = cell.text_frame
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = val
        r.font.size = Pt(15); r.font.name = BODY
        r.font.bold = i == 0 or j == 2
        r.font.color.rgb = WHITE if i == 0 else (TEAL if j == 2 else INK)
card_text(s, 0.6, 5.55, 12.1, 1.3, [[("Held fixed in every experiment  ", {"bold": True, "size": 15}),
                                    ("5 passages kept, 10 candidates per search, equal-weight merge (k = 60), MiniLM reranker, best-at-the-ends ordering, 4,000-character section cap, no query rewriting.", {"color": MUTED})]],
          fill=PAPER, pad=0.2, size=15, anchor=MSO_ANCHOR.MIDDLE)

# ---------------------------------------------------------------- 13 design choices
s = new("Why it is built this way",
        "Each choice has a reason tied to these documents. Exact search keeps the experiments clean. Chapter-aware chunks keep topics apart. Search small and read big combines precision with context. Hybrid search covers exact terms and paraphrases. "
        "The reranker is accurate but slow, so it only sees a shortlist. And because draft filings have blanks, the system is built to decline rather than guess.")
cards = [("No vector database","39,429 vectors fit in memory. Exact results mean nothing approximate can distort a comparison."),
         ("Cut along chapters", "A chunk never mixes two chapters. Each keeps its chapter label and page for citation."),
         ("Search small, read big", "Small chunks match precisely. The model then reads the whole section around the match."),
         ("Two kinds of search", "Names, terms and figures match as words. Paraphrases match by meaning. Merging covers both."),
         ("Rerank a short list", "The accurate model is slow, so it only sees 10 candidates, not 39,429 passages."),
         ("Allowed to say no", "Drafts have blanks. The model must decline when the answer is missing, and failures are returned, not hidden.")]
for i, (h_, d) in enumerate(cards):
    x, y = 0.6 + (i % 3) * 4.1, 1.5 + (i // 3) * 2.8
    card_text(s, x, y, 3.9, 2.55, [[(h_, {"bold": True, "size": 21})], [(d, {"color": MUTED})]], fill=TEAL_T if i % 2 == 0 else SLATE_T, pad=0.25, size=17, gap=10)

# ---------------------------------------------------------------- 14 scope and code
s = new("What it does not do, and where the code lives",
        "These are deliberate limits of the version we evaluated. The right-hand table maps each stage to its module so that a teammate can find the code.")
text(s, 0.6, 1.45, 5.3, 0.4, "Not part of this pipeline", size=18, bold=True)
bullets(s, 0.6, 1.95, 5.3, 4.9, [
    "No query rewriting||  the question is searched as typed",
    "No document router||  every selected filing is searched together",
    "No multi-step retrieval||  one search round per question",
    "Text PDFs only||  no OCR for scanned pages",
    "One question at a time||  no conversation memory"], size=19, gap=18)
code = [("Parse", "ipo_rag/parsing.py, structure.py"), ("Chunk", "ipo_rag/chunking.py"), ("Embed and index", "ipo_rag/indexing.py"),
        ("Retrieve, rerank, assemble", "ipo_rag/retrieval.py"), ("Generate", "ipo_rag/generation.py"), ("Verify", "ipo_rag/verification.py"),
        ("Ties it together", "ipo_rag/pipeline.py, config.py"), ("Web demo", "demo/server.py")]
gf = s.shapes.add_table(len(code) + 1, 2, Inches(6.4), Inches(1.5), Inches(6.3), Inches(0.5 * (len(code) + 1)))
t2 = gf.table
t2.columns[0].width = Inches(2.5); t2.columns[1].width = Inches(3.8)
for i, row in enumerate([("Stage", "Module")] + code):
    t2.rows[i].height = Inches(0.5)
    for j, val in enumerate(row):
        cell = t2.cell(i, j)
        cell.fill.solid(); cell.fill.fore_color.rgb = INK if i == 0 else (WHITE if i % 2 else PAPER)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE; cell.margin_left = Inches(0.14)
        r = cell.text_frame.paragraphs[0].add_run(); r.text = val
        r.font.size = Pt(14); r.font.bold = i == 0 or j == 0
        r.font.name = MONO if (j == 1 and i > 0) else BODY
        if j == 1 and i > 0:
            r.font.size = Pt(13)
        r.font.color.rgb = WHITE if i == 0 else INK

prs.save(OUT)
print("wrote", OUT, len(prs.slides), "slides")

run_checks(prs, HERE / "_preview" / "pipeline" if "--preview" in sys.argv else None)
