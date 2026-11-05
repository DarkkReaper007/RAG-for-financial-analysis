"""Build data/manifest.json: one record per filing, with a FIXED dev/test split by document."""
import fitz, hashlib, json, os

DOCS = [  # file, id, company, doc_type, sector, year, split
 ("1626944616560.pdf", "zomato", "ZOMATO LIMITED", "prospectus", "food delivery / tech", 2021, "dev"),
 ("1636529320240.pdf", "nykaa", "FSN E-COMMERCE VENTURES LIMITED", "prospectus", "e-commerce / retail", 2021, "test"),
 ("1731315962150.pdf", "swiggy", "SWIGGY LIMITED", "prospectus", "food delivery / tech", 2024, "test"),
 ("1644761029168.pdf", "lic", "LIFE INSURANCE CORPORATION OF INDIA", "drhp", "insurance", 2022, "dev"),
 ("1703580725954.PDF", "ola_electric", "OLA ELECTRIC MOBILITY LIMITED", "drhp", "electric vehicles", 2023, "dev"),
 ("1718692993829.PDF", "hyundai_india", "HYUNDAI MOTOR INDIA LIMITED", "drhp", "automobiles", 2024, "test"),
 ("1730791806554_562.pdf", "hdb_financial", "HDB FINANCIAL SERVICES LIMITED", "drhp", "NBFC", 2024, "test"),
 ("1733743806550_598.pdf", "lg_india", "LG ELECTRONICS INDIA LIMITED", "drhp", "consumer durables", 2024, "test"),
 ("Delhivery Limited - Draft Red Herring Prospectus (November 1, 2021)_20211102142410.pdf", "delhivery", "DELHIVERY LIMITED", "drhp", "logistics", 2021, "dev"),
 ("1775547017596_1210.pdf", "manipal_health", "MANIPAL HEALTH ENTERPRISES LIMITED", "drhp", "healthcare", 2026, "dev"),
 ("1781870482419.PDF", "jio_platforms", "JIO PLATFORMS LIMITED", "drhp", "telecom / digital", 2026, "test"),
 ("NSEDRHP_20260617214858.pdf", "nse", "NATIONAL STOCK EXCHANGE OF INDIA LIMITED", "drhp", "exchange", 2026, "test"),
]
out = []
for f, i, c, t, s, y, sp in DOCS:
    p = f"data/prospectuses/{f}"
    out.append({"id": i, "file": f, "company": c, "doc_type": t, "sector": s, "year": y, "split": sp,
                "pages": len(fitz.open(p)), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()})
json.dump({"version": 1, "docs": out}, open("data/manifest.json", "w"), indent=2)
print({sp: [d["id"] for d in out if d["split"] == sp] for sp in ("dev", "test")}, sum(d["pages"] for d in out), "pages")
