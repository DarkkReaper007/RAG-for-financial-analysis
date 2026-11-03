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
