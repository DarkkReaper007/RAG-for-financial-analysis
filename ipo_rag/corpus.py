import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def manifest():
    return {d["id"]: d for d in json.load(open(ROOT / "data" / "manifest.json"))["docs"]}


def pdf_path(doc):
    return ROOT / "data" / "prospectuses" / doc["file"]
