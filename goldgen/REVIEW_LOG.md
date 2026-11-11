# Review log - gold set v1

**Verification level: LLM-verified + spot-checked by Claude; the project authors have since declared the set audited (no per-item decision log is committed).** State this precisely in any report. Recommend a human
re-check of at least a random 15% before treating the set as final (see "Recommended human checks").

## Pipeline funnel
| Stage | Count |
|---|---|
| Pages sampled (30 per filing + 6 definition pages) | 366 |
| Candidates generated (`nemotron-3.5-bolt-e4b`) | 366 |
| Passed automatic filters (quote on page, no placeholder, self-contained, company named, no 6-gram copy, verifier `nemotron-3.5-lightning-30b-a3b`) | 227 |
| Filter failures | 139: company_not_named 58, quote_not_on_page 34, copies_6gram 28, verifier_rejected 28, verifier_no_reply 8, not_self_contained 4, placeholder 1 |
| Selected by quota (24 each factoid/numeric/table/list + 6 definition; rated natural/value 1-5 by the verifier model) | 102 |
| After my review exclusions (15) and reselection | 97 (shortfalls: hyundai factoid 1, list 2; hdb numeric 1, list 1) |
| + multi-document (12) + unanswerable (12) | 121 |

## What I (Claude) reviewed
- All 102 initially selected single-page items: question, answer, and where doubtful the source quote.
- All 24 multi-document / unanswerable items: question, answer, and source quotes for the doubtful ones.
- Not independently verified: numeric correctness beyond (a) the quote being verbatim on the cited page and (b) the verifier model agreeing. I
  spot-read a handful of figures against what I know of the filings (e.g. Zomato FY2021 revenue, registrars, auditors) and found them consistent.

## Findings
- **15 of 102 (15%) auto-selected items were removed on review**, even after LLM filtering and rating. Reasons: ambiguous reference (which
  cost / which order / which subsidiary), unverifiable from a flattened table (Swiggy ESOP "85.63 years"), cross-reference rather than a fact,
  vacuous answer, trivia (awards, director age by DIN, minor administrative detail). IDs and reasons: `review_exclusions.json`.
  => unreviewed LLM-generated questions in this corpus are noticeably noisy; the filters alone are not enough.
- **Multi-document items:** fixed a wrong item before release (Delhivery face value "₹1 or ₹10" - the page says ₹1 "prior to filing" something;
  replaced the pair by a registrar question) and reworded a "city" question whose answer was a locality (Hyundai, Irungattukottai).
- Company-name casing was mangled by title-casing ("Fsn", "Hdb", "Lg"); fixed with explicit display names.
- Known weaknesses that remain: many questions end with "in X's offer document" (awkward but harmless); several numeric answers carry no
  unit in the reference (graders should compare numbers, not units); definition questions are low-value by design (6 only).
- Items rated value<=1 by the verifier were dropped automatically unless they were definitions.

## Recommended human checks
1. Random 15% of the 97 single-page items: confirm answer and evidence page(s) against the PDF.
2. All 12 multi-document items (each is two lookups).
3. Evidence groups: confirm the extra pages added by expansion actually state the answer.
4. The test split is generated but should stay unseen during pipeline tuning. Note: the reviewer (Claude) has read the test questions while also
   building the pipeline; tune only on dev.

## Final build (gold_v1)
- **121 items** (dev 53 / test 68): factoid 23, numeric 23, table 24, list 21, definition 6, multi-document 12, unanswerable 12. 12 filings, 9-13 items each.
- **Evidence expansion:** 478 candidate pages judged, 62 of 97 single-page items gained extra pages (mean 1.94 extra, max 6; large groups are
  repeated financial figures such as revenue, borrowings, EPS, dividends). Mean evidence pages per answerable item: 2.83.
- **Verifier mix (disclose):** 463 of 478 evidence judgments by `nvidia/nemotron-3.5-lightning-30b-a3b`, 15 (3%) plus 2 quality ratings by
  `groq:openai/gpt-oss-120b` after NVIDIA rate-limited. Per-page judge is stored in `evidence_judged_by`.
- **Rate-limit bug fixed:** an earlier evidence run treated rate-limited (empty) replies as "page does not answer", which would have dropped valid
  evidence; unchecked pages are now retried and reported. Final run: 0 unchecked.
- **Validator:** `goldgen/validate_gold.py` -> 0 problems (quotes on cited pages, page bounds, split consistency, evidence/answerable consistency,
  hash). It caught one question without a "?" (fixed in `build_gold.py`).
- **Shortfalls vs plan:** 97 not 102 single-page items (hyundai factoid 1, list 2; hdb numeric 1, list 1 - too few acceptable candidates).
- **Key facts:** present for 90 of 109 answerable items (numeric answers and short answers); list/long answers rely on the judge instead.
- File: `eval/gold/gold_v1.json`, hash in `gold_v1.sha256`. Tune only on the dev split.
