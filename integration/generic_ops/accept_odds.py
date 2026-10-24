"""How often a fake reduction (a repeated subtraction, exact only for quotients up to its count) passes acceptance
(GENERIC_OPS.md §2.1): spcore's Acceptor tests 60 rows per verify size, and a row's quotient o is another row's x digits
at N's top W places (spcore's Draw), so a fake with L subtractions passes width W with chance P(o <= L)^60 over H1's
training rows. The s3i ablation's case (integration/ablation_s3/reduce_check.py): B[P[...]] does 10 x 2W x 10
subtractions, 600 at W = 3 and 800 at W = 4; the thousand-subtraction fakes of §2.1 pass when o <= 999.
Local: standard library, seconds, under 1 GB.

    cd cpu_first && python3 integration/generic_ops/accept_odds.py
"""
import json
import os

DATA = os.path.expanduser("~/github/one-layer/one-layer-deeper/data/generated/"
                          "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2/train.jsonl")
N_PROBES = 60                   # the Acceptor's rows per verify size

rows = [json.loads(line) for line in open(DATA)]
for W, limits in ((3, (600, 999)), (4, (800, 999, 8000))):
    q = [r["x"] // 10 ** max(0, len(str(r["modulus"])) - W) for r in rows]
    for limit in limits:
        p = sum(v <= limit for v in q) / len(q)
        print(f"W = {W}: o <= {limit} on {p:.3f} of {len(q):,} rows; all {N_PROBES} probes: {p ** N_PROBES:.2e}")
