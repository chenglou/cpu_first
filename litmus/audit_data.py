"""Data facts of a generated dataset that decide what the H1 machine's label-free lessons can pin down (LITMUS.md).

facts(root) reads root/train.jsonl and root/dataset_config.json (the prompt's layout is found from the tokens):
  * moduli: how many distinct N the training rows use, and their digit counts;
  * the reader's code (IO.md section 5): under the true reading, a training row's decisive place is the highest place
    where x's and N's digit strings differ; when x has a digit there, x < N says that digit's code is below N's digit's.
    The code is pinned to the identity only if every adjacent pair (k, k + 1) is decisive in some row (no digit lies
    between them to chain through), so `adjacent_pairs` counts the k with (k, k + 1) decisive, of 9, and `pairs` the
    ordered pairs the decisive ones imply transitively, of 45;
  * the reading's direction: rows where x < N fails once N's digits are read the other way round (rev N), or x's
    (rev x); with none, x < N cannot tell the two directions apart (IO.md's reading lesson);
  * T: the training T values, the rows the I/O lesson uses answers from (T = 1 or 2, REAL_MAX_T), and the rows whose
    T reads differently from its two ends (the only rows T's direction lesson can use, IO.md).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

NAMES = {2: "N", 3: "X", 4: "T", 5: "Y"}


def digits(v: int) -> list[int]:
    return [int(c) for c in str(v)][::-1]                                    # lowest first


def facts(root: str) -> dict:
    cfg = json.loads((Path(root) / "dataset_config.json").read_text())
    moduli, tvals, decisive = Counter(), Counter(), set()
    t_two_ways = answers = rev_n = rev_x = 0
    layout = n_reversed = None
    with open(Path(root) / "train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            n, x, t = int(r["modulus"]), int(r["x"]), int(r["time_steps"])
            moduli[n] += 1
            tvals[t] += 1
            answers += 1 <= t <= 2
            ds = digits(t)
            t_two_ways += len(ds) > 1 and ds != ds[::-1]
            nd, xd = digits(n), digits(x)
            rev_n += x >= int(str(n)[::-1])
            rev_x += int(str(x)[::-1]) >= n
            for p in reversed(range(len(nd))):
                a = xd[p] if p < len(xd) else None
                if a != nd[p]:
                    if a is not None:
                        decisive.add((a, nd[p]))
                    break
            if layout is None:
                ids = r["input_ids"]
                layout = "".join(NAMES.get(i, "") for i in ids)
                at = ids.index(2) + 1
                run = []
                while at < len(ids) and ids[at] >= 7:
                    run.append(ids[at] - 7)
                    at += 1
                n_reversed = run == nd
    closure = set(decisive)
    changed = True
    while changed:
        new = {(a, d) for a, b in closure for c, d in closure if b == c} - closure
        closure |= new
        changed = bool(new)
    ndig = Counter(len(str(n)) for n in moduli)
    return {
        "train_rows": sum(tvals.values()), "split_counts": cfg.get("split_counts"), "max_seq_len": cfg.get("max_seq_len"),
        "layout": layout, "n_reversed": n_reversed, "moduli": len(moduli), "modulus_digits": dict(sorted(ndig.items())),
        "n_digit_values": sorted({d for n in moduli for d in digits(n)}),
        "adjacent_pairs": sum((k, k + 1) in closure for k in range(9)), "pairs": len(closure),
        "unpinned_adjacent": [f"{k}<{k + 1}" for k in range(9) if (k, k + 1) not in closure],
        "rows_rejecting_rev_n": rev_n, "rows_rejecting_rev_x": rev_x,
        "train_T": dict(sorted(tvals.items())), "answer_rows": answers, "t_two_way_rows": t_two_ways,
    }


if __name__ == "__main__":
    import sys
    for root in sys.argv[1:]:
        print(root, json.dumps(facts(root)))


SPLITS = ("test", "ood_t", "ood_n_t", *(f"depth_{p}t_{t}" for p in ("", "ood_n_") for t in (1, 2, 4, 8, 16, 32, 64)))


def h1_agreement(root: str, splits=SPLITS) -> dict:
    """Per split, the share of rows whose answer H1's x^(2^T) mod N matches as the evaluator scores it: only the answer's
    own digit positions are compared (labels are tail-aligned; the model's digits before them are not checked), so this
    is (x^(2^T) mod N) mod 10^len(answer) == answer. What an exact H1 machine would score on a litmus task (LITMUS.md
    section 4); `<split>=` is plain equality."""
    out = {}
    for split in splits:
        path = Path(root) / f"{split}.jsonl"
        if not path.exists():
            continue
        same = equal = total = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                n, x, t = int(r["modulus"]), int(r["x"]), int(r["time_steps"])
                y, h = int(r["result"]), pow(x, 2 ** t, n)
                same += h % 10 ** len(str(y)) == y
                equal += h == y
                total += 1
        out[split] = [same, total]
        out[split + "="] = [equal, total]
    return out
