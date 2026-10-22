"""Which chain wirings of the H1 file's free-width menu pass every training lesson, and are they right on the unseen
8-digit moduli? (TOURNAMENT.md section 5). Every table at the truth (MODE "designed"); the chain's lessons as the file
runs them, but on many rows at once: the link law over all 29,584 link combinations, then every pair of a passing link
combination, a gate combination (66,564) and a gate table (4) on the division ladder's registers, without and with
the moved-up rows (CHAIN_NORM). The pairs that pass are then run on the ladder registers of 8-digit moduli (the
depth_ood_n splits' N, x below N): "right" means the step gives back x on every one.

    ~/github/sotaku/venv/bin/python -m modal run tournament/h1_census.py       # on Modal (one CPU container)
    ~/github/sotaku/venv/bin/python tournament/h1_census.py ROWS                # locally (several GB at ROWS = 256)

Writes tournament/records/h1_census.json.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CPU = HERE.parent
DATA = "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"
image = (modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
         .add_local_dir(CPU / "learned_remainder" / "chain_local" / "stub", "/root/stub")
         .add_local_file(CPU / "wider_menus" / "local" / "local_wide.py", "/root/local_wide.py")
         .add_local_file(HERE / "submission_tournament.py", "/root/submission_tournament.py")
         .add_local_file(HERE / "h1_census.py", "/root/h1_census.py")
         .add_local_dir(CPU.parent / "one-layer-deeper" / "data" / "generated" / DATA, f"/root/data/{DATA}"))
app = modal.App("tournament-census", image=image)


def census(rows: int = 256, data: str | None = None, sub: str | None = None) -> dict:
    sys.path[:0] = [p for p in (os.path.dirname(sub or ""), "/root", "/root/stub",
                                str(CPU / "wider_menus" / "local")) if p]
    if data:
        os.environ["H1_DATA"] = data
    import torch
    import local_wide as LW
    import torch.nn.functional as F
    t0 = time.time()
    sc = LW.sc = LW.load_module(sub or str(HERE / "submission_tournament.py"), ["MODE=\"designed\""])
    torch.manual_seed(0)
    m = sc.build_model(LW.spec())
    tabs = m.tables()
    Td, Tb = tabs[0], tabs[1]
    P, W, S = sc.P, sc.W, sc.S

    def numbers(split, n):
        """x and N digits [n, W] of the first n rows of a split (the reader at the truth)."""
        ids, am, lab, tpos = LW.batch(LW.load(split, n))
        e, end = m.ext(ids, am)
        k = m.selected(e, end)
        code = m.ranks(k)
        return m.read(1, e, end, code)[0], m.read(0, e, end, code)[0]

    def ladder_regs(Xs, Ns, norm):
        """The division ladder's registers x + q N (q = 0..9) [n * 10, P], their N [n * 10, W] and x [n * 10, P]; with
        norm also every row moved up until N fills W places."""
        q = torch.arange(10)
        lad = sc.row_hard(Ns[:, None, :].expand(len(Ns), 10, W), q[None].expand(len(Ns), 10),
                          torch.zeros(len(Ns), 10, dtype=torch.long), tabs)              # N * q [n, 10, C]
        reg = sc.add_learned(F.pad(Xs, (0, sc.C - W))[:, None].expand(len(Xs), 10, sc.C), lad, Td, Tb)[0][..., :P]
        reg, Nq, xq = reg.reshape(-1, P), Ns.repeat_interleave(10, 0), F.pad(Xs, (0, P - W)).repeat_interleave(10, 0)
        if norm:
            lead = (Ns.flip(-1).cumsum(-1) == 0).sum(-1).repeat_interleave(10)
            reg, Nq, xq = (torch.cat([reg, sc.move_digits(reg, lead)]), torch.cat([Nq, sc.move_digits(Nq, lead)]),
                           torch.cat([xq, sc.move_digits(xq, lead)]))
        mq = (sc.multiples(Nq, tabs), sc.multiples(Nq.flip(-1), tabs))
        return {"reg": reg, "Nq": Nq, "xq": xq, "mq": mq}

    def link_pass(Xs, Ns):
        """The link law on rows (x, N) for every link combination (sc's own law, through link_hard)."""
        n = len(Xs)
        r1 = torch.stack([F.pad(Xs, (0, P - W)), F.pad(Ns, (0, P - W))], 1).reshape(2 * n, P)
        N1 = Ns.repeat_interleave(2, 0)
        lt1 = torch.tensor([1, 0]).repeat(n)
        isN = lt1 == 0
        m1 = (sc.multiples(N1, tabs)[:, 0], sc.multiples(N1.flip(-1), tabs)[:, 0])
        combos = sc.block_decode("link", torch.arange(m.blk["link"].shape[0]))
        wbase = m.wiring(2 * n, "cpu")

        def law(idx, ri):
            k, r = idx.numel(), ri.numel()
            wk = sc.codes(sc.combo_wiring({kk: v[ri] for kk, v in wbase.items()}, sc.BLOCKS["link"], combos[idx], r))
            R = r1[ri].repeat(k, 1)
            Mr = sc.n_register(N1[ri].repeat(k, 1), wk["nread"])
            rev = (wk["nread"] == 1)[:, None]
            Mqs = torch.where(rev, m1[1][ri].repeat(k, 1), m1[0][ri].repeat(k, 1))
            _, E, _, fb = sc.link_hard(1, R, R, Mr, Mqs, F.pad(R[:, 1:], (0, 1)), F.pad(R[:, :1], (0, P - 1)), wk,
                                       Td, Tb)
            return ((fb == lt1[ri].repeat(k)) & (~isN[ri].repeat(k) | (E == 0).all(1))).view(k, r)
        return torch.nonzero(sc.staged(law, combos.shape[0], 2 * n, "cpu", stages=(2, 8, 64))).flatten()

    def pairs_pass(cand, regs, loop):
        """Every (link in cand, gate combination, gate table) pair on the ladder registers regs; with loop only the
        gate table that passes the loop law (commit on no borrow, hold on borrow)."""
        c1 = sc.block_decode("link", cand)
        c2 = sc.block_decode("gate", torch.arange(m.blk["gate"].shape[0]))
        g = (torch.tensor([1]), torch.tensor([0])) if loop else (torch.arange(2), torch.arange(2))
        pc = torch.cartesian_prod(torch.arange(len(cand)), torch.arange(len(c2)), *g)
        nreg = regs["reg"].shape[0]
        g = torch.Generator().manual_seed(0)
        order = torch.randperm(nreg, generator=g)
        wq = m.wiring(nreg, "cpu")

        def ladder(idx, ri):
            k, r = idx.numel(), ri.numel()
            q = pc[idx]
            comb = torch.cat([c1[q[:, 0]], c2[q[:, 1]], q[:, 2:]], 1)
            rr = order[ri]
            wk = sc.codes(sc.combo_wiring({kk: v[rr] for kk, v in wq.items()}, sc.CHAIN_SW, comb, r, gate=True))
            sub_ = {"reg": regs["reg"][rr], "Nq": regs["Nq"][rr], "xq": regs["xq"][rr],
                    "mq": (regs["mq"][0][rr], regs["mq"][1][rr])}
            out = m.chain_run(sub_["reg"].repeat(k, 1), wk, sub_, k, tabs)
            return (out == sub_["xq"].repeat(k, 1)).all(1).view(k, r)
        ok = sc.staged(ladder, pc.shape[0], nreg, "cpu", stages=(1, 4, 16, 64, 256))
        q = pc[ok]
        return torch.cat([c1[q[:, 0]], c2[q[:, 1]], q[:, 2:]], 1)

    def name(comb):
        names = {k: sc.DIV_MENU[k][int(v)] for k, v in zip(sc.CHAIN_SW, comb[:-2])}
        names.update({k: sc.DIV_MENU[k][0] for k in sc.DIV_MENU if k not in names})
        names["num"] = (0, 1)
        lab = sc.classify(names)["chain"]
        return (lab, " ".join(f"{k}={names[k]}".replace(" ", "") for k in sc.CHAIN_SW) +
                f" gate={'C' if comb[-2] else 'H'}{'C' if comb[-1] else 'H'}")

    Xt, Nt = numbers("train", rows)
    Xo, No = [], []
    for split in ("depth_ood_n_t_1", "depth_ood_n_t_2", "depth_ood_n_t_4", "depth_ood_n_t_8"):
        x, N = numbers(split, None)
        big = (N[:, W - 1] > 0)                                                          # 8-digit moduli
        Xo.append(x[big])
        No.append(N[big])
    Xo, No = torch.cat(Xo), torch.cat(No)
    rec = {"rows": rows, "ood_rows_8digit": len(Xo), "link_combinations": int(m.blk["link"].shape[0]),
           "gate_combinations": int(m.blk["gate"].shape[0])}
    cand = link_pass(Xt, Nt)
    rec["link_pass"] = int(cand.numel())
    ood = ladder_regs(Xo, No, False)
    for norm, loop in ((False, False), (True, False), (False, True), (True, True)):
        ok = pairs_pass(cand, ladder_regs(Xt, Nt, norm), loop)
        right = pairs_pass_on(m, sc, ok, ood, tabs)
        tally = collections.Counter()
        for comb, r in zip(ok.tolist(), right.tolist()):
            lab, nm = name(comb)
            tally[(lab, bool(r))] += 1
        key = {(False, False): "plain", (True, False): "with_norm", (False, True): "with_loop",
               (True, True): "with_norm_loop"}[(norm, loop)]
        rec[key] = {"pass_training": int(len(ok)), "right_on_8digit": int(right.sum()),
                    "by_kind": {f"{lab}|{'right' if r else 'wrong on 8-digit N'}": v for (lab, r), v in tally.items()},
                    "examples_wrong": [name(c)[1] for c, r in zip(ok.tolist(), right.tolist()) if not r][:12],
                    "examples_right": [name(c)[1] for c, r in zip(ok.tolist(), right.tolist()) if r][:12]}
        print(key, json.dumps(rec[key], indent=1), f"{time.time() - t0:.0f}s", flush=True)
    rec["seconds"] = time.time() - t0
    return rec


def pairs_pass_on(m, sc, combs, regs, tabs):
    """Whether each chain combination (rows of combs: CHAIN_SW option indices + the gate table's two rows) gives back x
    from every register of regs."""
    import torch
    out = []
    nreg = regs["reg"].shape[0]
    for comb in combs:
        wk = sc.codes(sc.combo_wiring(m.wiring(nreg, "cpu"), sc.CHAIN_SW, comb[None], nreg, gate=True))
        r = m.chain_run(regs["reg"], wk, regs, 1, tabs)
        out.append(bool((r == regs["xq"]).all()))
    return torch.tensor(out, dtype=torch.bool)


@app.function(cpu=8, memory=32768, timeout=4 * 3600)
def run(rows: int) -> dict:
    import torch
    torch.set_num_threads(8)
    return census(rows, data=f"/root/data/{DATA}", sub="/root/submission_tournament.py")


@app.local_entrypoint()
def main(rows: int = 256) -> None:
    rec = run.remote(rows)
    out = HERE / "records" / "h1_census.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items()}, indent=1), f"-> {out}")


if __name__ == "__main__":
    r = census(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
    (HERE / "records" / "h1_census_local.json").write_text(json.dumps(r, indent=1))
