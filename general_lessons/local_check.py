"""Local plumbing checks of submission_general.py (each well under a minute and 1 GB; litmus/local_check.py adapted to
the lesson sets and the round menu). learned_remainder/chain_local/stub stands in for the evaluator's API.

    python local_check.py DATA [N=8] [SPLIT ...] [K=V ...]        # both prediction paths, every table and switch at
                                                                  # the truth (MODE = "designed")
    python local_check.py DATA lessons [N=64] [K=V ...]           # every lesson set's hard checks at the truth
    python local_check.py DATA grad [N=16] [SPLIT ...] [K=V ...]  # the gradient test (rule 8)

K=V sets the constant K to the Python source V, e.g. DESIGNED_SIGMA='"lowzero"' or
DESIGNED_ROUND='{"rmul": "ten", "radd": 1, "rcube": "none"}' (the selected set and the round's options in designed
mode). `lessons` runs, for each lesson set, the hard versions (train=False) of its division ladder (with csub), its step
law and its whole-reduction laws on the first N training rows (the reader at the truth; the identity instances x and T),
and the selected machine on the rows with T <= REAL_MAX_T against their answers.
"""
import json
import os
import re
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "learned_remainder", "chain_local", "stub")]
import torch  # noqa: E402
from benchmark import ModelSpec  # noqa: E402

FILE = os.path.join(HERE, "submission_general.py")


def load_module(sets=()):
    src = open(FILE).read()
    for k, v in sets:
        src, count = re.subn(rf"^{k} = .*$", f"{k} = {v}", src, count=1, flags=re.M)
        if count != 1:
            raise KeyError(k)
    mod = types.ModuleType("sub")
    sys.modules["sub"] = mod
    exec(compile(src, FILE, "exec"), mod.__dict__)
    return mod


def rows(data, split, n):
    out = []
    with open(os.path.join(data, split + ".jsonl")) as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def batch(rs):
    L, T = max(len(r["input_ids"]) for r in rs), max(len(r["labels"]) for r in rs)
    ids = torch.zeros(len(rs), L, dtype=torch.long)
    lab = torch.full((len(rs), T), -100, dtype=torch.long)
    tpos = torch.full((len(rs), T), -1, dtype=torch.long)
    for i, r in enumerate(rs):
        n, m = len(r["input_ids"]), len(r["labels"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        lab[i, :m] = torch.tensor(r["labels"])
        tpos[i, :m] = torch.arange(n - m, n)
    return ids, ids != 0, lab, tpos


def designed_model(data, sets):
    sc = load_module(sets + [("MODE", '"designed"')])
    cfg = json.load(open(os.path.join(data, "dataset_config.json")))
    torch.manual_seed(0)
    m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
    print(f"set={sc.PREDICATES[m.sigma()]} wiring: {m.wiring_names()} round: {m.round_names()}", flush=True)
    return sc, m


def lessons(data, n, sets):
    sc, m = designed_model(data, sets)
    ids, am, lab, tpos = batch(rows(data, "train", n))
    with torch.no_grad():
        e, end = m.ext(ids, am)
        k = m.selected(e, end)
        code = m.ranks(k)
        Nd, _ = m.read(0, e, end, code)
        Xd, _ = m.read(1, e, end, code)
        Cd, tcount = m.read(2, e, end, code)
        small = (tcount == 1) & (Cd[:, 0] >= 1) & (Cd[:, 0] <= sc.REAL_MAX_T)
        tabs = m.tables()
        X = Xd.shape[0]
        _, lad_ok, verified, lad_rows = m.ladder(torch.cat([Xd, Nd]), tabs, False)
        for c, pred in enumerate(sc.PREDICATES):
            _, div_ok, regs = m.chain_lesson(c, Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, False)
            out = {"ladder": lad_ok, "division ladder + csub": div_ok}
            for name, names in sc.IO_BLOCKS:
                if name == "step":
                    out["step law"] = m.step_lesson(c, regs, tabs, False, names)[1]
                else:
                    out["whole-reduction laws"] = m.io_lesson(c, Xd, Nd, Cd, small, tabs, False, names, ["x", "T"])[1]
            print(f"  set {pred}: " + "  ".join(f"{kk}: {v:.3f}" for kk, v in out.items()), flush=True)
        sel = torch.nonzero(small).flatten()
        if sel.numel():
            yd = m.read_answer(m.label_seq(e, lab, lab != -100, tpos), e, end, code)
            s_ = sc.run_hard(Xd[sel], Cd[sel], Nd[sel], tabs, m.wcodes(sel.numel(), sel.device), cap=sc.REAL_MAX_T + 1)[0]
            print(f"  selected machine, answers of T <= {sc.REAL_MAX_T} ({sel.numel()} rows): "
                  f"{float((s_ == yd[sel]).all(1).float().mean()):.3f}", flush=True)


def grad(data, n, splits, sets):
    sc, m = designed_model(data, sets)
    rs = sum((rows(data, sp, n // 2) for sp in (splits or ["depth_t_8", "depth_ood_n_t_4"])), [])
    ids, am, lab, tpos = batch(rs)
    sc.PATH = "ste"
    m.ctl.used = None
    m.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits, _ = m(ids, am)
        tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)]
        valid = lab != -100
        ok = ((tl.argmax(-1) == lab) | ~valid).all(1)
        loss = torch.nn.functional.cross_entropy(tl[valid] / 20.0, lab[valid])
        loss.backward()
    bad = [name for name, p in m.named_parameters()
           if p.grad is None or not bool(torch.isfinite(p.grad).all()) or float(p.grad.abs().max()) == 0]
    print(f"GRADTEST rows={len(lab)} exact={float(ok.float().mean()):.3f} every_parameter_gets_gradient={not bad} "
          f"without={bad}", flush=True)


def main(argv):
    data, rest = argv[0], argv[1:]
    n = int(next((a.split("=")[1] for a in rest if a.startswith("N=")), 8))
    sets = [tuple(a.split("=", 1)) for a in rest if "=" in a and not a.startswith("N=")]
    splits = [a for a in rest if "=" not in a] or ["depth_t_1", "depth_t_8", "depth_ood_n_t_2", "depth_ood_n_t_64"]
    if splits[:1] == ["grad"]:
        return grad(data, n if any(a.startswith("N=") for a in rest) else 16, splits[1:], sets)
    if splits == ["lessons"]:
        return lessons(data, n if any(a.startswith("N=") for a in rest) else 64, sets)
    sc, m = designed_model(data, sets)
    for split in splits:
        ids, am, lab, tpos = batch(rows(data, split, n))
        res = {}
        for p in ("index", "ste"):
            sc.PATH = p
            m.ctl.used = None
            t0 = time.time()
            with torch.no_grad():
                res[p] = (m(ids, am)[0], time.time() - t0)
        tl = {p: res[p][0][torch.arange(len(lab))[:, None], tpos.clamp_min(0)].argmax(-1) for p in res}
        ex = {p: float(((tl[p] == lab) | (lab == -100)).all(1).float().mean()) for p in res}
        print(f"{split} n={len(lab)} exact index={ex['index']:.3f} ste={ex['ste']:.3f} "
              f"identical_logits={bool(torch.equal(res['index'][0], res['ste'][0]))} "
              f"t={res['index'][1]:.1f}s/{res['ste'][1]:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
