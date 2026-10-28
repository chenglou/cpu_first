"""Where does a trained trace model fail? For the first wrong token of each failing trace, report its line type and,
for comparison/quotient lines, how close the decisive pair was: the number of leading digits that E and k*N share."""

import argparse
import json

import numpy as np
import torch

import relax_common as rc
import relax_trace as R


def lead_agree(a, b, n):
    da, db = rc.digits(np.array([a]), n)[0][::-1], rc.digits(np.array([b]), n)[0][::-1]
    k = 0
    while k < n and da[k] == db[k]:
        k += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bits", type=int, default=12)
    ap.add_argument("--fmt", default="v3")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    W = rc.width(a.bits)
    lay = R.Layout(W, a.fmt)
    dev = torch.device(a.device)
    m = R.TraceGPT(lay, a.d, 8, a.layers).to(dev)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev))
    m.eval()
    d = rc.h1like(a.bits, 0)
    rng = np.random.default_rng(31)
    N = d["pool"][rng.integers(0, len(d["pool"]), a.n)]
    s = (rng.random(a.n) * N).astype(np.int64)
    toks, y = lay.build(N, s)
    gen = []
    for i in range(0, a.n, 2048):
        gen.append(m.generate(torch.as_tensor(toks[i:i + 2048, :lay.prompt_len], device=dev)).cpu().numpy())
    g = np.concatenate(gen)
    bad = np.where((g != toks).any(1))[0]
    # all comparisons in the reference traces, for the base rate of "close calls"
    stats = {"n": a.n, "failing": int(len(bad)), "first_err_line": {}, "fail_close": [], "base_close": []}
    for row in range(a.n):
        P = int(s[row]) ** 2
        Rr = P // 10 ** W
        pd = rc.digits(np.array([P]), 2 * W)[0]
        for i in range(W - 1, -1, -1):
            E = 10 * Rr + int(pd[i])
            q = E // int(N[row])
            for k in (q, q + 1):  # the decisive comparisons: q*N <= E < (q+1)*N
                if 1 <= k <= 9:
                    stats["base_close"].append(lead_agree(E, k * int(N[row]), W + 1))
            Rr = E - q * int(N[row])
        if row > 3000:
            break
    for row in bad:
        pos = int(np.argmax(g[row] != toks[row]))
        name = lay.segs[lay.line[pos]][0]
        stats["first_err_line"][name] = stats["first_err_line"].get(name, 0) + 1
        if name in ("G", "Q"):
            # which division step: find the E line just before pos
            li = lay.line[pos]
            e_start = lay.start[li - 1] if name == "G" else lay.start[li - (2 if a.fmt == "v3" else 1)]
            E = int(rc.undigits(toks[row:row + 1, e_start + 1:e_start + 1 + W + 1])[0])
            q = E // int(N[row])
            k = q + 1 if (name == "G" and g[row, pos] == 1) or (name == "Q" and g[row, pos] > q) else q
            k = min(max(k, 1), 9)
            stats["fail_close"].append(lead_agree(E, k * int(N[row]), W + 1))
    for key in ("fail_close", "base_close"):
        v = np.array(stats[key])
        stats[key] = {"count": int(len(v)), "hist_leading_digits_shared": np.bincount(v, minlength=W + 2).tolist()}
    print(json.dumps(stats, indent=1))
    if a.out:
        json.dump(stats, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
