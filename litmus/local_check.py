"""Local plumbing checks of a litmus variant (each well under a minute and 1 GB): the submission with every table and
switch at the truth (MODE = "designed", the H1 machine), optionally some switches moved to another option by hand, run
on the first rows of some splits of a dataset; both prediction paths ("index" and "ste") must give the same logits, and
the share of exactly right rows is printed. learned_remainder/chain_local/stub stands in for the evaluator's API.

    python local_check.py FILE DATA [N=8] [SPLIT ...] [switch=option ...] [K=V ...]
    python local_check.py FILE DATA lessons [N=64] [switch=option ...] [K=V ...]   # the division lessons' hard checks
    python local_check.py FILE DATA grad [N=16] [SPLIT ...] [switch=option ...]    # the gradient test (rule 8)

`grad` is wider_menus/local/local_wide.py's gradient test: the model in eval mode with autograd on, a cross-entropy on
the final logits at the answers' positions divided by 20, one backward pass; every parameter tensor must get a finite
gradient that is not all zero (default splits: depth_t_8 and depth_ood_n_t_4). `lessons` runs the hard versions (train=False) of the division ladder, the step law and the whole-division laws on the
first N training rows (the reader at the truth), and the machine on the rows with T <= REAL_MAX_T against their answers.

FILE is relative to cpu_first/; DATA a generated dataset folder; `switch=option` (a division switch, e.g.
plus1=one, by its option's text in DIV_MENU) moves that switch; K=V sets the constant K to the Python source V.
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
import torch
from benchmark import ModelSpec


def load_module(path, sets=()):
    src = open(os.path.join(ROOT, path)).read()
    for k, v in sets:
        src, count = re.subn(rf"^{k} = .*$", f"{k} = {v}", src, count=1, flags=re.M)
        if count != 1:
            raise KeyError(k)
    mod = types.ModuleType("sub")
    sys.modules["sub"] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
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


def designed_model(path, data, sets, moves):
    sc = load_module(path, sets + [("MODE", '"designed"')])
    cfg = json.load(open(os.path.join(data, "dataset_config.json")))
    torch.manual_seed(0)
    m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
    with torch.no_grad():
        for k, v in moves:
            opts = sc.DIV_MENU[k]
            i = [str(o).replace(" ", "") for o in opts].index(v.replace(" ", ""))
            m.sw[k].copy_(10.0 * torch.nn.functional.one_hot(torch.tensor(i), len(opts)).float() - 5.0)
    print("wiring:", m.wiring_names(), flush=True)
    return sc, m


def lessons(path, data, n, sets, moves):
    sc, m = designed_model(path, data, sets, moves)
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
        _, div_ok, regs = m.chain_lesson(Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, False)
        out = {"ladder": lad_ok, "division ladder": div_ok}
        for name, names in sc.IO_BLOCKS:
            if name == "step":
                out["step law"] = m.step_lesson(regs, tabs, False, names)[1]
            else:
                out["whole-division laws"] = m.io_lesson(Xd, Nd, Cd, small, tabs, False, names)[1]
        sel = torch.nonzero(small).flatten()
        if sel.numel():
            yd = m.read_answer(m.label_seq(e, lab, lab != -100, tpos), e, end, code)
            s_ = sc.run_hard(Xd[sel], Cd[sel], Nd[sel], tabs, m.wcodes(sel.numel(), sel.device), cap=sc.REAL_MAX_T + 1)[0]
            out[f"answers of T <= {sc.REAL_MAX_T} ({sel.numel()} rows)"] = float((s_ == yd[sel]).all(1).float().mean())
    print("  ".join(f"{k}: {v:.3f}" for k, v in out.items()), flush=True)


def grad(path, data, n, splits, sets, moves):
    sc, m = designed_model(path, data, sets, moves)
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
    sw = {k: float(p.grad.abs().max()) if p.grad is not None else 0.0 for k, p in m.sw.items()}
    print(f"GRADTEST rows={len(lab)} exact={float(ok.float().mean()):.3f} every_parameter_gets_gradient={not bad} "
          f"without={bad} switch_grads={ {k: f'{v:.1e}' for k, v in sw.items()} }", flush=True)


def main(argv):
    path, data, rest = argv[0], argv[1], argv[2:]
    n = int(next((a.split("=")[1] for a in rest if a.startswith("N=")), 8))
    sets = [tuple(a.split("=", 1)) for a in rest if "=" in a and a.split("=")[0].isupper() and not a.startswith("N=")]
    moves = [tuple(a.split("=", 1)) for a in rest if "=" in a and not a.split("=")[0].isupper()]
    splits = [a for a in rest if "=" not in a] or ["depth_t_1", "depth_t_8", "depth_ood_n_t_2", "depth_ood_n_t_64"]
    if splits[:1] == ["grad"]:
        return grad(path, data, n if any(a.startswith("N=") for a in rest) else 16, splits[1:], sets, moves)
    if splits == ["lessons"]:
        return lessons(path, data, n if any(a.startswith("N=") for a in rest) else 64, sets, moves)
    sc, m = designed_model(path, data, sets, moves)
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
