"""CONSOLIDATE.md §9: this study's cost, from the billing report's hours for each of its apps (as
bar3/scale/digits/watch_spend.py reads them). An hour still under way shows what is billed so far.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/consolidate/cost.py
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from modal.billing import workspace_billing_report

APPS = {"ap-q2HJiDVMcZ9MXJJPy62qhQ": "cs1: binary x mod N (11:40 PDT)",
        "ap-a6jBPpqnObbGT5aYaKTv4g": "cs1: x² mod N, 2x + 1, x² + 1, x² − 1",
        "ap-XYw0KdQYAP3Zz7WGP1m48P": "cs1: x² + 3, Fibonacci, x³, 10·x²",
        "ap-YJfpvvAVucUOWhMIfw59DQ": "cs1: x² + x, (x + 1)², 3·x², x⁴",
        "ap-n7Lri7KzvuSypXsxIq2vH0": "cs1: two sources, triangular and 10·x² + 1",
        "ap-SJbkXmmepvuYMCcFshXBNr": "cs1: 2^x, x³ + x, Lucas",
        "ap-k1k8SG3ZQEkmp9hlTE9R4j": "cs1: Tribonacci, floor(x²/N), triangular",
        "ap-C0RlLC8AzSO3BhSbIq3gBI": "ab1: x² mod N, 4 arms (12:13 PDT)",
        "ap-TSFIHUpkMqg3OiO8lPDSlm": "ab1: x² + 1, 6 arms",
        "ap-5XSSzaCsw92fVag3biKtA6": "ab1: x² − 1, 6 arms",
        "ap-HlcznuyOiKfz53oDMwBSUw": "ab1: x² + 3, 4 arms",
        "ap-4AUyecqU68wXStc9IlOqvn": "ab1: Fibonacci, 6 arms",
        "ap-5D27AsVnCcF66zhpJ0fojy": "ab1: x³, 4 arms",
        "ap-Y3IOBAfwdR2RDmjzss062C": "ab1: 10·x², 4 arms",
        "ap-wpUURnokF6MJQM9r8OkO1v": "ab1: 2^x, 6 arms",
        "ap-czpBzsHC0QdUKlDniZXklq": "ab1: triangular (two sources), 6 arms",
        "ap-QHXHRQ2mGCaUsiqTflcTPu": "ab1: 10·x² + 1 (two sources), 4 arms",
        "ap-I5PARhhrTDKuU3gmNiZ50O": "ab1: binary x mod N, 6 arms",
        "ap-j9rJwQjCGLaiTx5TOHlZ9s": "ab1: (x + 1)², 4 arms",
        "ap-fIO87HXNxi5aNbNN42ctDj": "ab1: 3·x², 4 arms",
        "ap-Hh8wvNpE2XwrfDL7tjSr74": "ab1: x⁴, 4 arms",
        "ap-JVtQxMZgDhykk89lykyn5p": "ab1: x² + x, 6 arms",
        "ap-JVEgm7n7iLYOjJsUBCfsx9": "ab2: x² mod N, 3 arms (12:44 PDT)",
        "ap-2i0RK6Shr0aOaPMu1ojSnp": "ab2: 2x + 1, 3 arms",
        "ap-T79OuKDFn6HqHHtkcM0hL4": "ab2: x² + 3, 3 arms",
        "ap-RL3x7bB9beGQ7HBy6Bt3Lx": "ab2: x³, 3 arms",
        "ap-mlkgsJCRcEFyvEzzaMLvsr": "ab2: 10·x², 3 arms",
        "ap-Y7v3X7iMs6SU6pZAbN17yw": "ab2: (x + 1)², 3 arms",
        "ap-GJFZXiUM2glNpTVOyO74SR": "ab2: 3·x², 3 arms",
        "ap-20g0BnIfoLXltdnMZN98oY": "ab2: x⁴, 3 arms",
        "ap-eJtnETMcB41aYes8aVXwyi": "ab2: x² + 1, 2 arms",
        "ap-phiXFyLKv0zEzVEwwwfuZR": "ab2: x² − 1, 2 arms",
        "ap-EDfKrE8Ljfh9VC0XYlFZbJ": "ab2: Fibonacci, 2 arms",
        "ap-CseIwePZnjtxgdwdzSzQNZ": "ab2: x² + x, 2 arms",
        "ap-n8HKdEbmDFSzihSccyZwU8": "ab2: 2^x, 2 arms",
        "ap-WND1QLIqfuEUd6vbJEq8wL": "ab2: triangular (two sources), 2 arms",
        "ap-CWCCjct0WaXnLLCZbS26gk": "ab2: 10·x² + 1 (two sources), 2 arms",
        "ap-22pqpJBVndrqI0KpS22l9s": "ab2: binary x mod N, 2 arms"}
START = datetime(2026, 10, 26, 18, tzinfo=timezone.utc)      # the hour the first app was created in (18:40 UTC)


def main():
    hours = defaultdict(lambda: defaultdict(float))
    for r in workspace_billing_report(start=START, resolution="h"):
        if r.get("object_id") in APPS:
            hours[r["object_id"]][f"{r['interval_start']:%H:%M}"] += float(r["cost"])
    total = 0.0
    for app, name in APPS.items():
        h = hours.get(app, {})
        cost = sum(h.values())
        total += cost
        print(f"{app} {name}: ${cost:.2f}  " + ", ".join(f"{k} UTC ${v:.2f}" for k, v in sorted(h.items())))
    print(f"all: ${total:.2f}")


if __name__ == "__main__":
    main()
