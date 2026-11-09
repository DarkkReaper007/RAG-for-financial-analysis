"""Builds paper/slides.pptx (16:9) from the same figures as the report. python paper/make_deck.py"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
FIG = HERE / "figs"
INK, MUTED, TEAL, PAPER, LINE = RGBColor(0x14, 0x20, 0x2B), RGBColor(0x5A, 0x68, 0x76), RGBColor(0x0B, 0x6E, 0x5F), RGBColor(0xF4, 0xF6, 0xF8), RGBColor(0xDB, 0xE1, 0xE7)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5


def tb(slide, x, y, w, h, text, size=18, bold=False, color=INK, font="Calibri", align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color; r.font.name = font
    return box


def bullets(slide, x, y, w, h, items, size=18, gap=8):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        head, _, rest = it.partition("||")
        p.space_after = Pt(gap)
        if rest:
            r = p.add_run(); r.text = head; r.font.bold = True; r.font.size = Pt(size); r.font.color.rgb = INK; r.font.name = "Calibri"
            r = p.add_run(); r.text = rest; r.font.size = Pt(size); r.font.color.rgb = MUTED; r.font.name = "Calibri"
        else:
            r = p.add_run(); r.text = head; r.font.size = Pt(size); r.font.color.rgb = INK; r.font.name = "Calibri"
    return box


def pic(slide, name, x, y, w, h):
    path = FIG / name
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def new(title, notes="", section=None):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    bar = s.shapes.add_shape(1, 0, 0, Inches(0.18), Inches(H)); bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background()
    tb(s, 0.6, 0.4, 12.1, 1.2, title, size=30, color=INK, font="Georgia", anchor=MSO_ANCHOR.TOP)
    if section:
        tb(s, 0.6, 7.0, 6, 0.3, section, size=11, color=MUTED)
    tb(s, 11.8, 7.0, 1.0, 0.3, str(len(prs.slides)), size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


# 1 title
s = prs.slides.add_slide(BLANK)
bg = s.background.fill; bg.solid(); bg.fore_color.rgb = PAPER
band = s.shapes.add_shape(1, 0, 0, Inches(0.35), Inches(H)); band.fill.solid(); band.fill.fore_color.rgb = TEAL; band.line.fill.background()
tb(s, 0.9, 2.0, 11.5, 2.2, "Which design choices actually matter in RAG over Indian IPO documents?", size=40, font="Georgia")
tb(s, 0.9, 4.4, 11, 0.6, "A reproducible ablation study with an evaluation framework", size=22, color=MUTED)
tb(s, 0.9, 6.2, 11, 0.5, "Sarvesh   |   Minor specialization, Big Data Analytics", size=16, color=MUTED)
s.notes_slide.notes_text_frame.text = "One-line pitch: we built a RAG system for SEBI offer documents, but the main result is the evaluation that says which of its design choices are supported by measurements. About 15 minutes plus questions."

s = new("Offer documents are long, repetitive and full of tables",
        "Motivation. DRHPs and RHPs run 400-700 pages. The same figure appears in the summary, the financial statements and the MD&A. Investors need an exact figure and the page it came from. For draft filings the offer price is blank, so a system must also be able to say 'not in the document'.", "Motivation")
bullets(s, 0.7, 1.7, 7.0, 4.8, ["400 to 700 pages||  per filing; 12 filings, 6,718 pages in this study", "Tables carry the answers||  profit, borrowings, share capital, remuneration",
                                  "Drafts have blanks||  the final price is printed as a placeholder, so some questions have no answer", "A useful system must||  find the page, quote figures exactly, cite it, and decline when the answer is missing"], size=20, gap=14)
tb(s, 8.2, 2.2, 4.5, 3.5, "Who is the registrar to the offer in the IPOs of Hyundai Motor India and LG Electronics India?", size=22, font="Georgia", color=TEAL)
tb(s, 8.2, 4.4, 4.5, 1.2, "A real test question. It needs two filings. The system fails it (recall 0.14 on this type).", size=15, color=MUTED)

s = new("Two research questions",
        "RQ1 is the ablation study. RQ2 uses a ceiling experiment that hands the generator the correct pages.", "Research questions")
bullets(s, 0.7, 1.8, 11.8, 4.5, ["RQ1||  Which pipeline components improve retrieval and answer quality?", "RQ2||  Where is the bottleneck, in retrieval or in generation?"], size=28, gap=30)

s = new("Existing work evaluates RAG without checking the right page was found",
        "RAGAS, ARES and RAGChecker are mostly reference-free. FinanceBench covers US filings and is not used here. No benchmark exists for Indian IPO documents. We score retrieval against annotated evidence pages and add deterministic checks that do not rely on a judge.", "Related work")
bullets(s, 0.7, 1.8, 11.8, 4.8, ["RAG pipelines||  hybrid search with rank fusion (BM25 + dense), cross-encoder rerankers, context ordering", "RAG evaluation||  RAGAS, ARES, RAGChecker: reference-free, strong on faithfulness, silent on which page", "Financial QA||  FinanceBench (US filings) reports that a shared index across companies is far harder than the right document", "Gap||  no benchmark or ablation study for Indian IPO offer documents"], size=21, gap=16)

s = new("The pipeline: every stage is a setting that can be switched",
        "Walk left to right. Exact cosine search over stored embeddings, no vector database, so there is no approximate-search error to confound ablations. Retrieval is over all 12 filings at once, so the system must also pick the right company.", "System")
pic(s, "pipeline.png", 0.6, 1.7, 12.1, 3.0)
bullets(s, 0.7, 4.9, 12, 2.0, ["Chunks||  about 1,000 characters inside a chapter; the parent section (3,000 characters) is returned as context", "Retrieval||  bge-small embeddings + BM25, fused by reciprocal rank fusion, reranked by a cross-encoder", "Verification||  every figure in the answer must be in the retrieved text; one rewrite, else abstain"], size=17, gap=8)

s = new("A gold set built with LLM help and scored at the page level",
        "121 questions. Questions drafted by an LLM from sampled pages, filtered by rules, verified by a second LLM, then evidence expanded to every page that supports the answer. Evidence is stored as groups: any one page in a group counts, every group must be hit. Audited by the author. Document-level dev/test split.", "Evaluation framework")
bullets(s, 0.7, 1.8, 6.4, 5, ["366 pages sampled||  one LLM-drafted question per page", "227 pass the filters||  quote on page, self-contained, company named, no copying, second-LLM check", "102 selected, 15 removed by hand||  97 single-page questions", "Evidence expanded||  478 more pages judged; 2.83 evidence pages per question", "+ 12 two-filing and 12 unanswerable||  121 questions in all"], size=17, gap=10)
bullets(s, 7.5, 1.8, 5.2, 5, ["Evidence groups||  any page in a group counts; a question needs every group (one per filing for two-filing questions)", "Split by document||  5 filings dev (53 questions), 7 test (68)", "Audited||  by the author only; no second annotator", "Caveats||  drafter also an evaluated generator; test questions seen by the authors; keyword overlap 0.35"], size=17, gap=12)

s = new("Three layers of measurement, plus paired statistics",
        "Retrieval metrics need no LLM. Grounding metrics are deterministic. The judge compares the gold answer to the predicted answer only. Decisions follow rules fixed before the test split was used.", "Evaluation framework")
bullets(s, 0.7, 1.8, 6.0, 5, ["Retrieval||  recall, nDCG, MRR at the page level, before and after reranking", "Grounding||  figures in the answer present in the retrieved text, valid citations, correct and false abstention", "Judge||  gold vs predicted answer: coverage, faithfulness, redundancy, five levels"], size=18, gap=14)
bullets(s, 7.1, 1.8, 5.6, 5, ["Statistics||  paired bootstrap intervals, exact McNemar, Holm correction", "Pre-registered rules||  a change replaces the baseline only if the gain is supported; verification adopted only if it cuts unsupported figures without losing score; generator chosen on dev", "Judge check||  84% agreement with a deterministic key-fact check (83 answers)"], size=18, gap=14)

s = new("Five single-factor ablations against one baseline",
        "One factor varies at a time. Held fixed: equal RRF weights, MiniLM reranker, context ordering, parent cap, no query rewriting.", "Experiments")
rows = [("Chunking and hierarchy", "fixed, section, section+parent*, +header"), ("Table representation", "inline, flat, markdown*"), ("Retrieval mode", "dense, BM25, hybrid*"), ("Verification", "off*, on"), ("Embedding model", "bge-small*, bge-large, e5-large")]
for i, (a, b) in enumerate(rows):
    y = 1.9 + i * 0.85
    tb(s, 0.8, y, 4.6, 0.6, a, size=21, bold=True)
    tb(s, 5.5, y, 7.2, 0.6, b, size=21, color=MUTED)
    ln = s.shapes.add_connector(1, Inches(0.8), Inches(y + 0.7), Inches(12.6), Inches(y + 0.7)); ln.line.color.rgb = LINE
tb(s, 0.8, 6.3, 11.8, 0.5, "* baseline. Generators compared: bolt (small), nano (30B reasoning), ultra (550B). Judge: gpt-oss-120b.", size=15, color=MUTED)

s = new("Hybrid search beats dense-only search; keyword search alone is just as good",
        "Pooled over 109 questions, dense is 0.11 below hybrid, McNemar p = 0.015. BM25 alone matches hybrid pooled, so we can say hybrid is better than dense, not that it is better than keyword search. This is the only significant effect in the study.", "Results")
pic(s, "retrieval_mode.png", 0.6, 1.6, 7.6, 4.9)
bullets(s, 8.5, 2.0, 4.3, 4.5, ["0.578 vs 0.468||  hybrid vs dense, recall@5, pooled", "p = 0.015||  the only clear effect", "BM25 = hybrid||  0.578 pooled; lower on dev, higher on test"], size=19, gap=16)

s = new("Chunking, tables, embedder size and verification are within noise",
        "Each row is the change in recall at 5 versus the baseline with a 95% interval. Only dense retrieval excludes zero. These are 'no effect large enough to detect with 109 questions', not proof of no effect. Larger embedders cost about eight times more to index.", "Results")
pic(s, "forest.png", 0.6, 1.6, 7.9, 5.2)
bullets(s, 8.8, 2.0, 4.1, 4.5, ["Within +-0.05||  on both splits", "Parent section||  helps the small generator: judge 0.50 to 0.59 (bolt)", "Verification||  unsupported figures 5.8% to 1.9%, score unchanged"], size=18, gap=14)

s = new("Giving the generator the right pages lifts the score from 0.59 to 0.84",
        "Ceiling experiment on dev. The retrieval loss is +0.27 with a 95% interval of 0.10 to 0.43. Largest losses on two-filing and numeric questions. Generator choice: nano by the dev rule; on test ultra is nominally higher but within noise and twice as slow.", "Results")
pic(s, "ceiling.png", 0.6, 1.7, 6.2, 4.6)
bullets(s, 7.2, 2.0, 5.6, 4.6, ["Retrieval is the bottleneck||  the remaining gap to 0.84 is mostly pages not found", "Generators||  bolt 0.58, nano 0.59, ultra 0.57 on dev: statistically tied", "Chosen||  nano by the pre-registered rule; on test ultra 0.650 vs nano 0.643"], size=19, gap=16)

s = new("On the held-out test, tables are solved and two-filing questions fail",
        "Final one-shot run, 68 questions, nano. 34 good answers and 7 correct abstentions. 20 failures start with evidence not retrieved. Only 7 fail with the evidence in hand. Oracle retrieval (right filing only) gives 0.787 recall.", "Results")
pic(s, "by_type.png", 0.5, 1.6, 6.3, 3.4)
pic(s, "taxonomy.png", 6.9, 1.6, 6.0, 3.4)
bullets(s, 0.7, 5.3, 12, 1.6, ["recall@5 0.656||  judge overall 0.651, faithfulness 0.824, correct abstention 7 of 7", "Most failures start in retrieval||  20 of 27 failures had no evidence retrieved; oracle filing choice lifts recall to 0.787"], size=18, gap=8)

s = new("Prospectus Desk: read the answer, then check it on the source page",
        "Live demo: run python -m demo.server. Show Ask with the Hyundai profit question, click the citation to open page 314 with the figures highlighted, open the retrieval trace, then Compare hybrid vs dense on the two-filing question, then Results. Examples are cached so the demo works offline.", "Demo")
pic(s, "demo_page.png", 0.6, 1.5, 4.6, 5.4)
bullets(s, 5.6, 1.8, 7.2, 5.0, ["Ask||  answer with citation buttons; figures highlighted on the source page", "How it found it||  dense, keyword, fused and reranked lists, with reference evidence marked", "Compare||  hybrid vs dense, figure check on/off, with and without surrounding text", "Results||  the charts from this talk", "Honest examples||  six test questions, including a wrong figure and a missed two-filing question"], size=18, gap=12)

s = new("Limitations", "Say these first before the questions come.", "Limitations")
bullets(s, 0.7, 1.7, 12, 5.2, ["Gold set||  LLM-built, audited by the author only; small (48 and 61 answerable questions), intervals about +-0.12; the drafter is also a tested generator", "Single judge||  checked against a deterministic metric, not against human grading", "Null results||  mean no effect large enough to detect, not no effect", "Generator choice||  nano vs ultra ordering differs between dev and test, both within noise", "Scope||  12 filings from one regulator; router and query rewriting held fixed", "Baselines||  no comparison with another published RAG system or a long-context model"], size=19, gap=12)

s = new("Future work", "Order by value: the audit first, then table retrieval.", "Future work")
bullets(s, 0.7, 1.7, 12, 5.2, ["Human annotation||  of a random sample of the gold set and of the judge, with agreement statistics", "Table-aware retrieval||  keep row and column headers with each figure; rewrite computed questions; a reranker trained on financial text", "Questions across filings||  multi-hop or agentic retrieval and an explicit document router", "More evaluation||  RAGAS as a supplementary row; published systems and an external benchmark such as FinanceBench", "Larger corpus||  more filings, final prospectuses with the offer price"], size=19, gap=14)

s = new("Conclusions",
        "Two takeaways. The framework is the main contribution: it found two bugs in our own pipeline, including hybrid retrieval that was secretly dense.", "Conclusions")
bullets(s, 0.7, 1.8, 12, 4.8, ["Only hybrid retrieval clearly helped||  recall@5 0.58 vs 0.47 for dense (p = 0.015); other choices are within noise", "Retrieval limits quality||  gold pages lift the judge score from 0.59 to 0.84; two-filing questions fail", "Scope||  one corpus and an author-audited gold set: evidence for this setting, not proof for others", "The evaluation is the contribution||  it caught two bugs in our own pipeline and says where to invest next"], size=21, gap=18)

s = new("Backup: generator comparison", "Reference table for questions about generator choice.", "Backup")
data = [("", "Dev overall", "Dev fail", "Dev p50 s", "Test overall", "Test fail", "Test p50 s"), ("bolt", "0.580", "0%", "1.05", "0.597", "0%", "1.87"), ("nano (chosen)", "0.594", "4%", "4.83", "0.643", "1%", "3.36"), ("ultra", "0.566", "0%", "8.54", "0.650", "0%", "6.48")]
tbl = s.shapes.add_table(4, 7, Inches(0.7), Inches(2.0), Inches(11.9), Inches(2.6)).table
for r, row in enumerate(data):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c); cell.text = v
        for p in cell.text_frame.paragraphs:
            for run in p.runs:
                run.font.size = Pt(18); run.font.bold = (r == 0); run.font.name = "Calibri"
tb(s, 0.7, 5.0, 11.9, 0.8, "Judge: gpt-oss-120b on gold-vs-prediction. Selection rule: highest dev judge score among generators with failure rate at most 10%.", size=15, color=MUTED)

prs.save(HERE / "slides.pptx")
print("slides:", len(prs.slides))
