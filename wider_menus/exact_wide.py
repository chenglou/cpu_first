"""Exact counts over the wide menus with every table at the truth (MENUS.md section 3): which wirings pass each lesson,
and which of those are correct (they compute x^2 mod N on every certification row) or impostors (they pass the
lessons on the training rows but fail elsewhere).

  chain   every combination of the chain block, the I/O at the truth: the division ladder (one step from x + q N
          gives back x) on `rows` training rows x 10 rungs; the passing ones run the whole machine
  io      every combination of the I/O block, the chain at the truth (and at trial multiples): the laws x mod N = x,
          N mod N = 0, x + Q N mod N = x and the answers, each on `rows` training rows; the passing ones run the whole
          machine
  joint   (division level <= 1) every combination of the chain and the I/O blocks together under the laws alone
  mul     every pair of column rules (lo, hi): the routed ladder (every rung of every row j); every number rule:
          distributivity

    ~/github/sotaku/venv/bin/python -m wider_menus.exact_wide W=3 div=3 mul=2 part=chain,io,mul
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

try:
    from . import wide as G
    from .tiny_wide import Data, classify
except ImportError:
    import wide as G
    from tiny_wide import Data, classify

CFG = dict(W=3, widths=(2,), div=3, mul=2, part="chain,io,mul", rows=256, cert_rows=3000, seed=0, chunk=400000,
           out="")


def decoded(menu, names, cmb):
    return {n: str(menu[n][int(cmb[i])]) for i, n in enumerate(names)}


def run(cfg):
    W = cfg["W"]
    P, C = W + 1, 2 * W + 2
    dm, mm = G.div_menu(cfg["div"], W), G.mul_menu(cfg["mul"], W)
    data = Data.train(W, cfg["widths"], 20000, cfg["seed"], 0.0)
    cert = Data.cert(W, cfg["cert_rows"], cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    pick = torch.from_numpy(rng.choice(data.n, cfg["rows"], replace=False))
    Nd, xd = data.Nd[pick], data.xd[pick]
    yd = data.yd[pick]
    n = Nd.shape[0]
    truth = G.true_index(dm, G.DIV_TRUE)
    exact_mult = None                                     # multiples by exact arithmetic (the true layout)
    out = {"cfg": cfg, "sizes": {"div": G.size(dm), "chain": G.size(dm, G.CHAIN), "io": G.size(dm, G.IO),
                                 "mul": G.size(mm)}}

    def whole(ix_over, names, cmb, d):
        """Certification of the whole machine (true multiply) for the combinations cmb of `names`, others at the
        truth: the share of d's rows right per combination."""
        K = cmb.shape[0]
        res = []
        for k in range(K):
            ix = {nm: torch.full((d.n,), (int(cmb[k, names.index(nm)]) if nm in names else ix_over.get(nm, truth[nm])))
                  for nm in dm}
            V = G.digits(torch.from_numpy(d.x * d.x), 2 * W)
            o = G.divide_val(V, d.Nd, G.decode(dm, ix, W), exact_mult)
            res.append(float((o == d.yd).all(1).double().mean()))
        return res

    def verdicts(names, cmb, fn, base=None):
        """fn(ix, reps) -> [rows * reps] bool for per-row wirings; evaluated in chunks. Returns [K, rows] bool."""
        K = cmb.shape[0]
        per = max(1, cfg["chunk"] // n)
        outs = []
        for a in range(0, K, per):
            c = cmb[a:a + per]
            k = c.shape[0]
            ix = {nm: torch.full((k * n,), (base or truth)[nm], dtype=torch.long) for nm in dm}
            for i, nm in enumerate(names):
                ix[nm] = c[:, i].repeat_interleave(n)
            outs.append(fn(ix, k).view(k, n))
        return torch.cat(outs)

    z = torch.zeros_like(xd)
    Q = torch.from_numpy(rng.integers(0, 10, (n, W)))
    Qv = (G.value(Q) * G.value(Nd) + G.value(xd)) % 10 ** (2 * W)
    Qd = G.digits(Qv, 2 * W)
    V = G.digits(G.value(xd) ** 2, 2 * W)
    laws = {
        "x": lambda ix, k: (G.divide_val(torch.cat([xd, z], 1).repeat(k, 1), Nd.repeat(k, 1), G.decode(dm, ix, W))
                            == xd.repeat(k, 1)).all(1),
        "n": lambda ix, k: (G.divide_val(torch.cat([Nd, z], 1).repeat(k, 1), Nd.repeat(k, 1), G.decode(dm, ix, W))
                            == 0).all(1),
        "q": lambda ix, k: (G.divide_val(Qd.repeat(k, 1), Nd.repeat(k, 1), G.decode(dm, ix, W))
                            == xd.repeat(k, 1)).all(1),
        "a": lambda ix, k: (G.divide_val(V.repeat(k, 1), Nd.repeat(k, 1), G.decode(dm, ix, W))
                            == yd.repeat(k, 1)).all(1),
    }

    def summarize(name, names, cmb, vs, base_names=None):
        """Counts of combinations passing every row of each law set; the passing ones' certification."""
        res = {"combinations": int(cmb.shape[0])}
        for key, v in vs.items():
            full = v.all(1)
            res[key] = {"pass_all_rows": int(full.sum()),
                        "pass_99": int((v.double().mean(1) >= 0.99).sum()),
                        "pass_90": int((v.double().mean(1) >= 0.9).sum())}
        last = list(vs)[-1]
        good = torch.nonzero(vs[last].all(1)).flatten()
        if good.numel() <= 400:
            certs = whole(base_names or {}, list(names), cmb[good], cert)
            res["passing"] = [dict(decoded(dm, names, cmb[g]), cert=c) for g, c in zip(good.tolist(), certs)]
            res["passing_correct"] = sum(c == 1.0 for c in certs)
        out[name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "passing"}), flush=True)

    t0 = time.time()
    parts = cfg["part"].split(",")
    if "chain" in parts:
        cmb = G.combos(dm, G.CHAIN)
        q = torch.arange(10).repeat(n)
        xs, Ns = xd.repeat_interleave(10, 0), Nd.repeat_interleave(10, 0)
        reg = (G.value(xs) + q * G.value(Ns)) % 10 ** P
        K = cmb.shape[0]
        ix = {nm: torch.full((K * 10 * n,), truth[nm], dtype=torch.long) for nm in dm}
        for i, nm in enumerate(G.CHAIN):
            ix[nm] = cmb[:, i].repeat_interleave(10 * n)
        w = G.decode(dm, ix, W)
        Nr = torch.where(w["nread"][:, None] == 1, Ns.repeat(K, 1).flip(-1), Ns.repeat(K, 1))
        Mv = G.value(Nr)
        Mq = (torch.arange(1, G.S)[None] * Mv[:, None]) % 10 ** P
        v = (G.chain_val(reg.repeat(K), Mv, Mq, w, 10 ** P) == G.value(xs).repeat(K)).view(K, 10 * n)
        summarize("chain", G.CHAIN, cmb, {"ladder": v})
    if "io" in parts:
        cmb = G.combos(dm, G.IO)
        for base_name, over in (("truth", {}), ("trial", {"a": "Rp", "b": "Mq"})):
            base = dict(truth)
            for k_, v_ in over.items():
                base[k_] = dm[k_].index(v_) if v_ in dm[k_] else base[k_]
            if any(v_ not in dm[k_] for k_, v_ in over.items()):
                continue
            vs = {}
            acc = None
            for key in ("x", "n", "q", "a"):
                v = verdicts(G.IO, cmb, laws[key], base)
                acc = v if acc is None else acc & v
                vs[{"x": "x", "n": "x+n", "q": "x+n+q", "a": "x+n+q+a"}[key]] = acc
            vs["a_only"] = verdicts(G.IO, cmb, laws["a"], base)
            vs["x+n+a"] = vs["x+n"] & vs["a_only"]
            vs = {k: vs[k] for k in ("x", "x+n", "a_only", "x+n+a", "x+n+q", "x+n+q+a")}
            summarize(f"io_{base_name}", G.IO, cmb, vs, {k_: base[k_] for k_ in G.CHAIN})
    if "joint" in parts and cfg["div"] <= 1:
        names = G.CHAIN + G.IO
        cmb = G.combos(dm, names)
        vs, acc = {}, None
        for key in ("x", "n", "q", "a"):
            v = verdicts(names, cmb, laws[key])
            acc = v if acc is None else acc & v
            vs[{"x": "x", "n": "x+n", "q": "x+n+q", "a": "x+n+q+a"}[key]] = acc
        vs["x+n+a"] = vs["x+n"] & verdicts(names, cmb, laws["a"])
        summarize("joint", names, cmb, {k: vs[k] for k in ("x", "x+n", "x+n+a", "x+n+q", "x+n+q+a")})
    if "mul" in parts:
        X = torch.cat([xd, Nd])
        rules = mm["lo"]
        R = len(rules)
        # each rule's contribution of the low digits (lo) and the high digits (hi) to every rung of every row j
        b = torch.arange(10)
        prod = X[:, None, :] * b[None, :, None]                              # [n, b, i]
        dig = [prod % 10, prod // 10]
        target = G.value(X)[:, None] * b[None]                               # [n, b]
        parts_ = []
        for kind in range(2):
            pr = torch.zeros(R, X.shape[0], 10, W, dtype=torch.long)
            for r, rule in enumerate(rules):
                cr = G.claims(rule, rule, (0, 1), W, rows_only=True)[kind]  # [i, j]
                wgt = torch.where(cr >= 0, 10 ** cr.clamp(min=0), torch.zeros_like(cr))
                pr[r] = torch.einsum("nbi,ij->nbj", dig[kind], wgt)
            parts_.append(pr.reshape(R, -1))
        tg = (target[:, :, None] * 10 ** torch.arange(W)[None, None]).reshape(-1)
        lo_need = tg[None] - parts_[0]                                       # what the hi part must add
        keys = {}
        for r in range(R):
            keys.setdefault(tuple(parts_[1][r].tolist()), []).append(r)
        pairs = []
        for r in range(R):
            for r2 in keys.get(tuple(lo_need[r].tolist()), []):
                pairs.append((rules[r], rules[r2]))
        res = {"rules_per_kind": R, "layouts": R * R, "ladder_pass": len(pairs), "passing": [str(p) for p in pairs]}
        ref = sum(G.row_val(X, X[:, j], j, G.claims(G.MUL_TRUE["lo"], G.MUL_TRUE["hi"], (0, 1), W, True))
                  for j in range(W)) % 10 ** C
        nums = []
        for num in mm["num"]:
            ok = bool((G.mul_val(X, X, G.claims(G.MUL_TRUE["lo"], G.MUL_TRUE["hi"], num, W)) == ref).all())
            nums.append({"num": str(num), "distributive": ok})
        res["number_rules"] = nums
        out["mul"] = res
        print("mul", json.dumps(res), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    if cfg["out"]:
        with open(cfg["out"], "w") as f:
            json.dump(out, f, indent=1, default=str)
    return out


def main(argv):
    cfg = dict(CFG)
    for a in argv:
        k, v = a.split("=", 1)
        cfg[k] = tuple(int(t) for t in v.split(",")) if isinstance(CFG[k], tuple) else type(CFG[k])(v)
    run(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
