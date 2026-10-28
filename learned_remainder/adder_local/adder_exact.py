"""How exact the learned addition is on its own, from the snapped table a run printed (TABLE lines of its record).

    python adder_exact.py NAME [NAME ...]     # records in cpu_first/history/runs/learned_remainder/<name>.json
    python adder_exact.py --exact             # the true subtraction table (the reference)

For each table: its rows against true subtraction r - m - c (computed here, outside the submission), then the learned
adder of submission_adder.py, A + B = A - (0 - B) with both subtractions the snapped table, against Python integers:
  * every pair at widths 1, 2 and 3 (10^6 pairs at width 3: every (digit, digit, carry in) case at every place);
  * 200,000 random pairs at widths 9 and 18, and 20,000 at widths 36 and 72 (digits and the final carry).
Small: well under 1 GB, a few seconds.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "chain_local", "stub"), os.path.dirname(HERE)]
import torch
import submission_adder as sa

RUNS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "history", "runs", "learned_remainder")


def table_from(name):
    rec = json.load(open(os.path.join(RUNS, name + ".json")))
    lines = rec["stdout"].splitlines()
    dig = next(l.split("=", 1)[1] for l in lines if l.startswith("TABLE digit="))
    bor = next(l.split("=", 1)[1] for l in lines if l.startswith("TABLE borrow="))
    return torch.tensor([int(c) for c in dig]), torch.tensor([int(c) for c in bor])


def value(d):
    return sum(int(v) * 10 ** i for i, v in enumerate(d))


def check(Td, Tb):
    tdt, tbt = sa.exact_tables("cpu")
    out = {"rows_right": int(((Td == tdt) & (Tb == tbt)).sum())}
    for width in (1, 2, 3):
        n = 10 ** width
        a = torch.arange(n).repeat_interleave(n)
        b = torch.arange(n).repeat(n)
        S, carry = sa.add_learned(sa.to_digits(a, width), sa.to_digits(b, width), Td, Tb)
        out[f"all_pairs_w{width}"] = f"{int((sa.from_digits(S) + carry * n == a + b).sum())}/{n * n}"
    g = torch.Generator().manual_seed(0)
    for width, count in ((9, 200_000), (18, 200_000), (36, 20_000), (72, 20_000)):
        A = torch.randint(0, 10, (count, width), generator=g)
        B = torch.randint(0, 10, (count, width), generator=g)
        S, carry = sa.add_learned(A, B, Td, Tb)
        if width <= 18:
            ok = int((sa.from_digits(S) + carry * 10 ** width == sa.from_digits(A) + sa.from_digits(B)).sum())
        else:
            ok = sum(value(S[i].tolist()) + int(carry[i]) * 10 ** width == value(A[i].tolist()) + value(B[i].tolist())
                     for i in range(count))
        out[f"random_w{width}"] = f"{ok}/{count}"
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["--exact"]:
        print("exact table:", check(*sa.exact_tables("cpu")))
    for name in sys.argv[1:]:
        if name != "--exact":
            print(name + ":", check(*table_from(name)))
