"""Run: python eval/test_metrics.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from metrics import *

assert hit_at_k([5, 9, 12], [9], 1) == 0 and hit_at_k([5, 9, 12], [9], 2) == 1
assert recall_at_k([5, 9, 12], [9, 40], 3) == 0.5
assert precision_at_k([5, 9, 12, 9], [9], 4) == 0.5
assert mrr([5, 9, 12], [9]) == 0.5 and mrr([1, 2], [9]) == 0
assert abs(ndcg_at_k([9, 5], [9], 2) - 1.0) < 1e-9 and ndcg_at_k([5, 9], [9], 2) < 1
assert hit_at_k([10], [9], 1, tol=1) == 1 and hit_at_k([10], [9], 1, tol=0) == 0
assert extract_numbers("Price band ₹ 72 to ₹ 76 [Source: Z, Page: 87]; issue size 9,375.00 crore") == {"9375.00"}
assert numeric_support("Offer price is 76.00 and size 9,375.00", ["offer 76.00 ... 9,375.00"]) == 1.0
assert numeric_support("size 9,999.00", ["size 9,375.00"]) == 0.0
assert numeric_support("no numbers here", ["x"]) is None
assert citation_accuracy("x [Source: Z, Section: A, Page: 87] y [Page: 99]", [87, 12]) == 0.5
assert citation_accuracy("no cites", [1]) is None
assert is_abstention("Based on the available context, I cannot find sufficient information") and not is_abstention("The price is 76")
assert fact_recall("Price band is Rs. 72 to 76", ["72", "₹76"]) == 1.0 and fact_recall("x", []) is None
assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([], 50) is None
m, lo, hi = mean_ci([1, 0, 1, 1, 0, 1]); assert lo <= m <= hi
assert citation_accuracy("Context 1, Section: BLW | Page: 357, says", [357]) == 1.0
print("metrics OK")
