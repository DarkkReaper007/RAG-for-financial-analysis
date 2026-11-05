"""Chapter detection for SEBI-layout offer documents (shared by the pipeline chunker and the gold-set tools)."""
import re

CHAPTERS = ["DEFINITIONS AND ABBREVIATIONS", "FORWARD-LOOKING STATEMENTS", "SUMMARY OF THE OFFER DOCUMENT", "SUMMARY OF THE DRAFT RED HERRING PROSPECTUS",
            "SUMMARY OF THE RED HERRING PROSPECTUS", "THE OFFER", "GENERAL INFORMATION", "CAPITAL STRUCTURE", "OBJECTS OF THE OFFER",
            "BASIS FOR OFFER PRICE", "STATEMENT OF SPECIAL TAX BENEFITS", "RISK FACTORS", "INDUSTRY OVERVIEW", "OUR BUSINESS",
            "KEY REGULATIONS AND POLICIES", "HISTORY AND CERTAIN CORPORATE MATTERS", "OUR MANAGEMENT", "OUR PROMOTERS AND PROMOTER GROUP",
            "OUR GROUP COMPANIES", "DIVIDEND POLICY", "FINANCIAL INFORMATION", "FINANCIAL STATEMENTS", "OTHER FINANCIAL INFORMATION",
            "CAPITALISATION STATEMENT", "FINANCIAL INDEBTEDNESS", "MANAGEMENT'S DISCUSSION AND ANALYSIS", "OUTSTANDING LITIGATION",
            "GOVERNMENT AND OTHER APPROVALS", "OTHER REGULATORY AND STATUTORY DISCLOSURES", "OFFER STRUCTURE", "TERMS OF THE OFFER",
            "OFFER PROCEDURE", "MAIN PROVISIONS OF THE ARTICLES OF ASSOCIATION", "MATERIAL CONTRACTS AND DOCUMENTS FOR INSPECTION", "DECLARATION"]


def chapter_of(text):
    """Return the chapter this page opens, or None. Skips the leading page number and 'SECTION II:' prefixes."""
    lines = []
    for l in text.splitlines():
        l = re.sub(r"\s+", " ", l).strip().upper().replace("’", "'")
        if not l or re.fullmatch(r"[\dIVXLC]{1,5}", l):
            continue
        l = re.sub(r"^SECTION [IVX]+\s*[:\-–]\s*", "", l)
        lines.append(l)
        if len(lines) == 4:
            break
    hits = {c for c in CHAPTERS if any(l.startswith(c) and len(l) < len(c) + 40 for l in lines)}
    if "FINANCIAL INFORMATION" not in hits and any(l.startswith(("RESTATED CONSOLIDATED FINANCIAL INFORMATION", "RESTATED FINANCIAL")) for l in lines):
        hits.add("FINANCIAL INFORMATION")
    return next(iter(hits)) if len(hits) == 1 else None  # >1 hit = contents page, not an opener


def label_pages(page_texts, first_content_page=9):
    """Forward-fill chapter labels over a list of page texts (index 0 = page 1)."""
    out, cur = [], None
    for i, t in enumerate(page_texts, 1):
        c = chapter_of(t)
        if c and (i >= first_content_page or c == "DEFINITIONS AND ABBREVIATIONS"):
            cur = c
        out.append(cur)
    return out


# --- US SEC filings (10-K / 10-Q / earnings release): "Item N." headings and the primary financial statements ---
_ITEM = re.compile(r"^(?:PART\s+[IV]+\s*[,.\-–—]?\s*)?ITEM\s+(\d{1,2}[AB]?)\s*[.:\-–—]?\s*(.{0,140})$")
_STATEMENTS = [("CONSOLIDATED STATEMENTS OF COMPREHENSIVE", "STATEMENT OF COMPREHENSIVE INCOME"),
               ("CONSOLIDATED STATEMENTS OF INCOME", "INCOME STATEMENT"), ("CONSOLIDATED STATEMENT OF INCOME", "INCOME STATEMENT"),
               ("CONSOLIDATED STATEMENTS OF OPERATIONS", "INCOME STATEMENT"), ("CONSOLIDATED STATEMENT OF OPERATIONS", "INCOME STATEMENT"),
               ("CONSOLIDATED STATEMENTS OF EARNINGS", "INCOME STATEMENT"), ("CONSOLIDATED STATEMENT OF EARNINGS", "INCOME STATEMENT"),
               ("CONSOLIDATED BALANCE SHEET", "BALANCE SHEET"), ("CONSOLIDATED STATEMENTS OF FINANCIAL POSITION", "BALANCE SHEET"),
               ("CONSOLIDATED STATEMENTS OF CASH FLOWS", "CASH FLOW STATEMENT"), ("CONSOLIDATED STATEMENT OF CASH FLOWS", "CASH FLOW STATEMENT"),
               ("CONSOLIDATED STATEMENTS OF SHAREHOLDERS", "STATEMENT OF EQUITY"), ("CONSOLIDATED STATEMENTS OF STOCKHOLDERS", "STATEMENT OF EQUITY"),
               ("CONSOLIDATED STATEMENTS OF CHANGES IN", "STATEMENT OF EQUITY"), ("CONSOLIDATED STATEMENTS OF EQUITY", "STATEMENT OF EQUITY"),
               ("NOTES TO CONSOLIDATED FINANCIAL STATEMENTS", "NOTES TO FINANCIAL STATEMENTS"), ("NOTES TO THE CONSOLIDATED FINANCIAL", "NOTES TO FINANCIAL STATEMENTS"),
               ("NOTES TO CONDENSED CONSOLIDATED", "NOTES TO FINANCIAL STATEMENTS"), ("CONDENSED CONSOLIDATED BALANCE", "BALANCE SHEET"),
               ("CONDENSED CONSOLIDATED STATEMENTS OF CASH", "CASH FLOW STATEMENT"), ("CONDENSED CONSOLIDATED STATEMENTS OF", "INCOME STATEMENT"),
               ("MANAGEMENT'S DISCUSSION AND ANALYSIS", "ITEM 7 MD&A"), ("RISK FACTORS", "ITEM 1A RISK FACTORS")]
_ITEM_NAMES = {"1": "ITEM 1 BUSINESS", "1A": "ITEM 1A RISK FACTORS", "1B": "ITEM 1B UNRESOLVED STAFF COMMENTS", "2": "ITEM 2 PROPERTIES / MD&A (10-Q)",
               "3": "ITEM 3 LEGAL PROCEEDINGS", "4": "ITEM 4 MINE SAFETY", "5": "ITEM 5 MARKET FOR EQUITY", "6": "ITEM 6 RESERVED", "7": "ITEM 7 MD&A",
               "7A": "ITEM 7A MARKET RISK", "8": "ITEM 8 FINANCIAL STATEMENTS", "9": "ITEM 9 CHANGES IN ACCOUNTANTS", "9A": "ITEM 9A CONTROLS AND PROCEDURES",
               "10": "ITEM 10 DIRECTORS", "11": "ITEM 11 EXECUTIVE COMPENSATION", "12": "ITEM 12 SECURITY OWNERSHIP", "13": "ITEM 13 RELATED TRANSACTIONS",
               "14": "ITEM 14 ACCOUNTANT FEES", "15": "ITEM 15 EXHIBITS"}


_ORDER = ["1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15", "16"]


def sec_section_of(text, cur_item=None):
    """-> (label, item) for the section a US filing page opens, or (None, None). Item headings are searched on the whole page (they often start mid-page), statement titles only in the first lines; a table-of-contents page (several
    Item lines) opens nothing. An Item heading must be a short line and may only move forward a few steps in filing order (a cross-reference such
    as 'see Item 7' at the top of a page is not a heading)."""
    lines = []
    for l in text.splitlines():
        l = re.sub(r"\s+", " ", l).strip().upper().replace("’", "'")
        if l and not re.fullmatch(r"[\dIVXLC\-– ]{1,8}", l):
            lines.append(l)
    items = [m.group(1) for l in lines if len(l) < 160 and (m := _ITEM.match(l))]
    if len(set(items)) >= 3:
        return None, None
    pos = _ORDER.index(cur_item) if cur_item in _ORDER else -1
    for it in items:
        if it in _ORDER and pos <= _ORDER.index(it) <= pos + 5:
            return _ITEM_NAMES.get(it, f"ITEM {it}"), it
    for l in lines[:8]:
        for prefix, name in _STATEMENTS:
            if l.startswith(prefix) and len(l) < len(prefix) + 60:
                return name, None
    return None, None


def label_pages_sec(page_texts):
    out, cur, item = [], None, None
    for t in page_texts:
        lab, it = sec_section_of(t, item)
        if lab:
            cur = lab
            item = it or item
        out.append(cur)
    return out
