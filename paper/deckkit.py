"""Shared helpers for the two decks (slides.pptx, pipeline_slides.pptx): palette, shape helpers, and a layout checker / PNG previewer.
The checker measures text with the real Office font files (when installed), so it reports overflow without needing PowerPoint."""
import sys
from pathlib import Path

from lxml import etree
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

W, H = 13.333, 7.5
HEAD, BODY, MONO = "Cambria", "Calibri", "Consolas"
C = lambda h: RGBColor.from_string(h)
INK, MUTED, WHITE = C("14202B"), C("5A6876"), C("FFFFFF")
SLATE, SLATE_T = C("2F4858"), C("E4EAEF")
TEAL, TEAL_T = C("0B6E5F"), C("DDF0EB")
LINE, PAPER, MARK = C("D5DCE3"), C("F5F7F9"), C("FFE08A")
WARN, WARN_T = C("9A5B00"), C("FBEFD5")


# ---------------------------------------------------------------- helpers
def _noshadow(sh):
    try:
        sh.shadow.inherit = False
    except Exception:
        pass


def box(slide, x, y, w, h, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06, name=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = min(0.5, radius / min(w, h) * 1.0) if min(w, h) > 0 else 0.1
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(lw)
    _noshadow(sh)
    if name:
        sh.name = name
    return sh


def write(sh, paras, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, gap=0, margin=0.0, italic=False):
    """paras: list of paragraphs; a paragraph is a str or a list of runs; a run is a str or (str, dict(bold,color,size,font,italic))."""
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin * 0.6)
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(gap)
        p.space_before = Pt(0)
        runs = [para] if isinstance(para, (str, tuple)) else para
        for r in runs:
            txt, o = (r, {}) if isinstance(r, str) else r
            run = p.add_run()
            run.text = txt
            f = run.font
            f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
            f.name = o.get("font", font); f.color.rgb = o.get("color", color)
    return sh


def text(slide, x, y, w, h, paras, **kw):
    sh = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    return write(sh, paras, **kw)


def card_text(slide, x, y, w, h, paras, fill=PAPER, line=None, pad=0.18, anchor=MSO_ANCHOR.TOP, **kw):
    sh = box(slide, x, y, w, h, fill=fill, line=line)
    return write(sh, paras, margin=pad, anchor=anchor, **kw)


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75):
    cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    cn.line.color.rgb = color; cn.line.width = Pt(width)
    ln = cn.line._get_or_add_ln()
    tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
    tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return cn


def line(slide, x1, y1, x2, y2, color=LINE, width=1.0):
    cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    cn.line.color.rgb = color; cn.line.width = Pt(width)
    return cn


def badge(slide, x, y, d, label, fill, size=14, color=WHITE):
    sh = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    return write(sh, label, size=size, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ---------------------------------------------------------------- checks and preview
OFFICE = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")
FONTS = {("Georgia", False): "/System/Library/Fonts/Supplemental/Georgia.ttf", ("Georgia", True): "/System/Library/Fonts/Supplemental/Georgia Bold.ttf", ("Calibri", False): "Calibri.ttf", ("Calibri", True): "Calibrib.ttf", ("Cambria", False): "Cambria.ttc", ("Cambria", True): "Cambriab.ttf",
         ("Consolas", False): "Consola.ttf", ("Consolas", True): "Consolab.ttf"}
_fc = {}


def font_at(name, bold, size):
    from PIL import ImageFont
    key = (name, bold, size)
    if key not in _fc:
        f = FONTS.get((name, bool(bold)), "Calibri.ttf")
        f = Path(f) if str(f).startswith("/") else OFFICE / f
        _fc[key] = ImageFont.truetype(str(f), max(1, int(round(size * 10))))
    return _fc[key]


def measure(txt, name, bold, size):
    return font_at(name, bold, size).getlength(txt) / 10.0       # points


def layout(tf, width_pt):
    """-> list of lines (each a list of (word, name, bold, size, rgb)), and the total height in points."""
    lines_out, total = [], 0.0
    for p in tf.paragraphs:
        words = []
        for r in p.runs:
            f = r.font
            size = f.size.pt if f.size else 18
            try:
                rgb = f.color.rgb
            except Exception:
                rgb = None
            for k, tok in enumerate(r.text.split(" ")):
                words.append((tok, f.name or BODY, bool(f.bold), size, rgb, k > 0 or (r.text.startswith(" ") and False)))
        if not words:
            lines_out.append(([], 14.0)); total += 14 * 1.2; continue
        cur, curw = [], 0.0
        plines = []
        for tok, nm, b, sz, rgb, _ in words:
            wpt = measure(tok + " ", nm, b, sz)
            if cur and curw + measure(tok, nm, b, sz) > width_pt:
                plines.append(cur); cur, curw = [], 0.0
            cur.append((tok, nm, b, sz, rgb)); curw += wpt
        if cur:
            plines.append(cur)
        for ln in plines:
            lh = max(w[3] for w in ln) * 1.22
            lines_out.append((ln, lh)); total += lh
        total += (p.space_after.pt if p.space_after is not None else 0)
    return lines_out, total


def font_px(nm, bd, sz, S):
    from PIL import ImageFont
    f = FONTS.get((nm, bool(bd)), "Calibri.ttf")
    f = Path(f) if str(f).startswith("/") else OFFICE / f
    return ImageFont.truetype(str(f), max(6, int(sz / 72 * S)))


def inset(sh):
    tf = sh.text_frame
    return (tf.margin_left or 0) / 12700, (tf.margin_right or 0) / 12700, (tf.margin_top or 0) / 12700, (tf.margin_bottom or 0) / 12700


def check(prs):
    bad = 0
    for n, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            x, y, w, h = sh.left / 914400, sh.top / 914400, sh.width / 914400, sh.height / 914400
            if sh.shape_type != 9 and (x < -0.01 or y < -0.01 or x + w > W + 0.01 or y + h > H + 0.01):
                print(f"slide {n}: {sh.name} outside the slide ({x:.2f},{y:.2f},{w:.2f},{h:.2f})"); bad += 1
            if getattr(sh, "has_text_frame", False) and sh.has_text_frame and sh.text_frame.text.strip():
                l, r, t, b = inset(sh)
                _, tot = layout(sh.text_frame, w * 72 - l - r)
                if tot > h * 72 - t - b + 1.5:
                    print(f"slide {n}: text overflows '{sh.text_frame.text[:50]}' needs {tot / 72:.2f} in, has {h - (t + b) / 72:.2f} in"); bad += 1
            if sh.has_table if hasattr(sh, "has_table") else False:
                tb = sh.table
                for i, row in enumerate(tb.rows):
                    for j, cell in enumerate(row.cells):
                        cw = tb.columns[j].width / 12700 - 14 - 7.2
                        _, tot = layout(cell.text_frame, cw)
                        if tot > row.height / 12700 - 7.2:
                            print(f"slide {n}: table cell overflows '{cell.text_frame.text[:40]}'"); bad += 1
    print("text-fit check:", "ok" if not bad else f"{bad} problem(s)")
    return bad


def preview(prs, outdir):
    from PIL import Image, ImageDraw
    outdir.mkdir(parents=True, exist_ok=True)
    S = 110                                                          # px per inch
    for n, slide in enumerate(prs.slides, 1):
        bg = slide.background.fill
        img = Image.new("RGB", (int(W * S), int(H * S)), (255, 255, 255))
        try:
            c = bg.fore_color.rgb
            img.paste((c[0], c[1], c[2]), [0, 0, img.width, img.height])
        except Exception:
            pass
        d = ImageDraw.Draw(img)
        for sh in slide.shapes:
            x, y, w, h = [v / 914400 * S for v in (sh.left, sh.top, sh.width, sh.height)]
            if sh.shape_type == 9 or sh.name.startswith("Connector") or sh.name.startswith("Straight"):
                pass
            try:
                kind = str(sh.auto_shape_type)
            except Exception:
                kind = ""
            if kind:
                try:
                    fill = sh.fill.fore_color.rgb if sh.fill.type == 1 else None
                except Exception:
                    fill = None
                try:
                    ln = sh.line.color.rgb if sh.line.fill.type == 1 else None
                except Exception:
                    ln = None
                oval = "OVAL" in kind
                rad = min(w, h) / 2 if oval else min(sh.adjustments[0] * min(w, h) if len(sh.adjustments) else 0, min(w, h) / 2) if "ROUNDED" in kind else 0
                if fill is not None or ln is not None:
                    d.rounded_rectangle([x, y, x + w, y + h], radius=rad, fill=tuple(fill) if fill is not None else None, outline=tuple(ln) if ln is not None else None, width=1)
            elif sh.shape_type == 9 or "Connector" in sh.name:
                pass
            if sh.shape_type == 13:                                       # picture
                try:
                    import io
                    pic = Image.open(io.BytesIO(sh.image.blob)).convert("RGB").resize((max(1, int(w)), max(1, int(h))))
                    img.paste(pic, (int(x), int(y)))
                except Exception:
                    pass
                continue
            if sh.shape_type == 9 or "Connector" in type(sh).__name__:
                try:
                    col = tuple(sh.line.color.rgb)
                except Exception:
                    col = (120, 130, 140)
                x1, y1, x2, y2 = [v / 914400 * S for v in (sh.begin_x, sh.begin_y, sh.end_x, sh.end_y)]
                d.line([x1, y1, x2, y2], fill=col, width=2)
                import math
                a = math.atan2(y2 - y1, x2 - x1)
                if sh.line._get_or_add_ln().find("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd") is not None:
                    d.polygon([(x2, y2), (x2 - 9 * math.cos(a - 0.4), y2 - 9 * math.sin(a - 0.4)), (x2 - 9 * math.cos(a + 0.4), y2 - 9 * math.sin(a + 0.4))], fill=col)
                continue
            if getattr(sh, "has_text_frame", False) and sh.has_text_frame and sh.text_frame.text.strip():
                l, r, t, b = inset(sh)
                lines, tot = layout(sh.text_frame, sh.width / 12700 - l - r)
                avail = sh.height / 12700 - t - b
                anchor = sh.text_frame.vertical_anchor
                cy = y + t / 72 * S + ((avail - tot) / 2 / 72 * S if anchor == MSO_ANCHOR.MIDDLE else 0)
                paras = sh.text_frame.paragraphs
                li = 0
                for p in paras:
                    pass
                for ln, lh in lines:
                    lw = sum(measure(tok + " ", nm, bd, sz) for tok, nm, bd, sz, _ in ln)
                    align = paras[0].alignment
                    cx = x + l / 72 * S
                    if align == PP_ALIGN.CENTER:
                        cx = x + (w - lw / 72 * S) / 2
                    elif align == PP_ALIGN.RIGHT:
                        cx = x + w - r / 72 * S - lw / 72 * S
                    for tok, nm, bd, sz, rgb in ln:
                        from PIL import ImageFont
                        f = font_px(nm, bd, sz, S)
                        d.text((cx, cy), tok, font=f, fill=tuple(rgb) if rgb is not None else (20, 32, 43))
                        cx += measure(tok + " ", nm, bd, sz) / 72 * S
                    cy += lh / 72 * S
            if getattr(sh, "has_table", False) and sh.has_table:
                tb = sh.table
                yy = y
                for row in tb.rows:
                    xx = x
                    rh = row.height / 914400 * S
                    for j, cell in enumerate(row.cells):
                        cw = tb.columns[j].width / 914400 * S
                        try:
                            fc = tuple(cell.fill.fore_color.rgb)
                        except Exception:
                            fc = (255, 255, 255)
                        d.rectangle([xx, yy, xx + cw, yy + rh], fill=fc, outline=(213, 220, 227))
                        lines, tot = layout(cell.text_frame, tb.columns[j].width / 12700 - 14 - 7.2)
                        cy = yy + (rh - tot / 72 * S) / 2
                        for ln, lh in lines:
                            cx = xx + 0.14 * S
                            for tok, nm, bd, sz, rgb in ln:
                                from PIL import ImageFont
                                f = font_px(nm, bd, sz, S)
                                d.text((cx, cy), tok, font=f, fill=tuple(rgb) if rgb is not None else (20, 32, 43))
                                cx += measure(tok + " ", nm, bd, sz) / 72 * S
                            cy += lh / 72 * S
                        xx += cw
                    yy += rh
        img.save(outdir / f"slide-{n:02d}.png")
    print("previews in", outdir)




def run_checks(prs, preview_dir=None):
    """Prints text-fit problems; with preview_dir also writes slide-NN.png previews. Returns the number of problems."""
    if not OFFICE.exists():
        print("Office font files not found; skipped the text-fit check")
        return 0
    bad = check(prs)
    if preview_dir:
        preview(prs, Path(preview_dir))
    return bad
