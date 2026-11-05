"""Frozen gold set loader. Refuses to run if the file hash differs from the recorded one."""
import hashlib
import json
import os
from pathlib import Path

GOLD_DIR = Path(__file__).resolve().parent.parent / "eval" / "gold"


def load(split=None, types=None, version=None):
    version = version or os.environ.get("RAG_GOLD", "v1")
    body = (GOLD_DIR / f"gold_{version}.json").read_text()
    want = (GOLD_DIR / f"gold_{version}.sha256").read_text().strip()
    got = hashlib.sha256(body.encode()).hexdigest()
    if got != want:
        raise RuntimeError(f"gold_{version}.json hash mismatch (expected {want[:12]}, got {got[:12]}): gold set was modified")
    items = json.loads(body)["items"]
    if split:
        items = [i for i in items if i["split"] == split]
    if types:
        items = [i for i in items if i["type"] in types]
    return items, got


def groups_of(item):
    """Evidence groups as sets of (doc, page). Multi-document items: group i belongs to docs[i]; otherwise all to docs[0]."""
    return [{(item["docs"][gi] if item["type"] == "multi_document" else item["docs"][0], p) for p in g}
            for gi, g in enumerate(item["evidence_groups"])]
