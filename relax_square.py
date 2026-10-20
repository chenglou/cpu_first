"""The user's question: does supervising the value before the remainder (the raw square s^2, up to ~14 digits) give a
usable gradient? Periodic track: regress the pre-activation v on s^2 (no wrap ambiguity), and let the designed
N-periodic unit + snapping do the reduction. No curriculum, random small init.

  data fresh  s uniform in [0, N) at b bits          data h1  H1's own 1,000 T=1 rows, with x^2 attached as a label
  opt  lm     Levenberg-Marquardt on (v - s^2)/N     opt adam  Adam on the same residual, then the LM polish
Kinds as in relax_periodic.py: mono (6 weights on 1,s,N,s^2,sN,N^2), bilinear (4 GLU-style units on (1,s,N)),
bidig (4 units on (1, 8 digits of s, N): the place values 10^i must be learned).
"""
import argparse
import json
import math
import time

import numpy as np
import torch

import relax_common as rc
import relax_periodic as P

torch.set_default_dtype(torch.float64)


def lm_square(model, N, s, sq, iters=50, damp=1e-6):
    S, Nn, Q = P.tt(s), P.tt(N), P.tt(sq)

    def resid():
        v, J = model.v_and_jac(S, Nn)
        return (v - Q) / Nn, J / Nn[:, None]

    with torch.no_grad():
        r, J = resid()
        cost = float((r * r).sum())
        for _ in range(iters * 4):
            col = J.norm(dim=0).clamp_min(1e-300)
            Jn = J / col
            A = Jn.T @ Jn
            delta = torch.linalg.solve(A + damp * torch.eye(A.shape[0]), Jn.T @ r) / col
            model.add_flat(-delta)
            r2, J2 = resid()
            c2 = float((r2 * r2).sum())
            if c2 <= cost:
                r, J, cost, damp = r2, J2, c2, max(damp / 10, 1e-12)
            else:
                model.add_flat(delta)
                damp *= 10
    return cost


def adam_square(model, N, s, sq, steps=3000, lr=1e-2, batch=1024, seed=0):
    """First-order: Adam on the normalized regression residual (parameters rescaled per feature, like LSUV)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for t in range(steps):
        i = rng.integers(0, len(N), batch)
        Nn, S, Q = P.tt(N[i]), P.tt(s[i]), P.tt(sq[i])
        loss = (((model.v(S, Nn) - Q) / Nn) ** 2).mean()
        for g in opt.param_groups:
            g["lr"] = lr * (0.5 * (1 + math.cos(math.pi * t / steps)) * 0.999 + 0.001)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["mono", "bilinear", "bidig"])
    ap.add_argument("--bits_list", type=int, nargs="+", default=[6, 12, 16, 20, 24])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--opt", default="lm", choices=["lm", "adam"])
    ap.add_argument("--data", default="fresh", choices=["fresh", "h1"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = {}
    if a.data == "h1":
        import h1
        tr = h1.load("train")
        t1 = tr["T"] == 1
        Nh, Xh, Yh = tr["N"][t1], tr["X"][t1], tr["Y"][t1]
    for kind in a.kinds:
        for bits in ([20] if a.data == "h1" else a.bits_list):
            succ, cert_ok, times, exs = 0, 0, [], []
            for seed in range(a.seeds):
                t0 = time.time()
                m = P.PeriodicStep(kind, K=4, seed=seed)
                m.phase_small_init(bits if a.data == "fresh" else 20, 1e-3, seed)
                if a.data == "fresh":
                    N, s, _ = rc.fresh_step(np.random.default_rng(seed), bits, 8192)
                else:
                    N, s = Nh, Xh
                sq = s * s
                if a.opt == "adam":
                    adam_square(m, N, s, sq, seed=seed)
                    pre = P.exact_frac(m, 24 if a.data == "h1" else bits)
                lm_square(m, N, s, sq)
                times.append(time.time() - t0)
                ex = P.exact_frac(m, 24 if a.data == "h1" else bits, n=50000)
                exs.append(ex if a.opt == "lm" else (pre, ex))
                succ += ex == 1.0
                if a.data == "h1" or bits == 24:
                    import h1

                    def predict(Nn, Xx, Tt):
                        st = Xx % Nn
                        for tv in np.unique(Tt):
                            k = Tt == tv
                            c = st[k].copy()
                            for _ in range(int(tv)):
                                c = P.step_fn(m)(Nn[k], c)
                            st[k] = c
                        return h1.digits(st)

                    cert = [h1.certify(predict, p, verbose=False)["max_t"] for p in ("depth_t", "depth_ood_n_t")]
                    cert_ok += cert == [64, 64]
            key = f"{kind}_{a.opt}_{a.data}_b{bits}"
            out[key] = {"success": succ, "seeds": a.seeds, "exact": exs, "time_per_seed": float(np.mean(times)),
                        "h1_certified_T64_both": cert_ok if (a.data == "h1" or bits == 24) else None}
            print(key, json.dumps(out[key]), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
