import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# RAG_CORPUS=ipo (default: the 12 SEBI filings) | financebench (external benchmark subset)
_NAME = os.environ.get("RAG_CORPUS", "ipo")
_BASE = ROOT / "data" if _NAME == "ipo" else ROOT / "data" / _NAME
_PDFS = _BASE / "prospectuses" if _NAME == "ipo" else _BASE / "pdfs"


def manifest():
    return {d["id"]: d for d in json.load(open(_BASE / "manifest.json"))["docs"]}


def pdf_path(doc):
    return _PDFS / doc["file"]
