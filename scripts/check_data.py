"""Check that the filing PDFs are in place and unchanged.  python scripts/check_data.py
Reads data/manifest.json (IPO filings) and, if present, data/financebench/manifest.json. Prints what is missing or different."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sets = [("IPO filings", ROOT / "data" / "manifest.json", ROOT / "data" / "prospectuses"),
        ("FinanceBench filings (optional)", ROOT / "data" / "financebench" / "manifest.json", ROOT / "data" / "financebench" / "pdfs")]
bad = 0
for name, manifest, folder in sets:
    if not manifest.exists():
        continue
    docs = json.load(open(manifest))["docs"]
    print(f"\n{name}: {len(docs)} expected in {folder.relative_to(ROOT)}/")
    for d in docs:
        f = folder / d["file"]
        if not f.exists():
            print(f"  MISSING  {d['id']:28s} {d['file']}")
            bad += 1
            continue
        ok = hashlib.sha256(f.read_bytes()).hexdigest() == d["sha256"]
        print(f"  {'ok     ' if ok else 'CHANGED'}  {d['id']:28s} {d['file']}")
        bad += not ok
print("\nAll good." if not bad else f"\n{bad} file(s) need attention. Get the PDFs from the team's shared folder (see README).")
sys.exit(1 if bad else 0)
