"""Local checks of a strict file against its certified file (STRICT.md §3), on the CPU, in well under a minute:
  keep     the buffers' whole-batch memory against the certified file's lists, on random batch sizes;
  ops      SteMachine.run_ops against SteMachine.run on the round programs a certified run found: every register,
           flag and pointer, and the tables' gradients, equal;
  buffers  set_round's buffers read back: the instructions equal trace's, the text equal the programs;
  forward  the designed reader and cells (MODE "designed") with those programs: the certified file (ctl.lib,
           ctl.progs) and the strict file (set_round) give equal logits on H1's rows;
  grad     rule 8's gradient test on the strict file (INTEGRATION.md §5): which parameter tensors the final logits'
           loss reaches, with the right tokens' logits and with cross-entropy as the objective.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/strict/checks/strict_local.py [KIND=sp2|s3i] [N=2]
    [LIB=integration/strict/libs/sp2_official_lib.json] [SPLITS=depth_t_1,depth_t_4] [TESTS=keep,ops,buffers,forward,grad]
"""
import json
import os
import random
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
STRICT = os.path.dirname(HERE)
INTEG = os.path.dirname(STRICT)
ROOT = os.path.dirname(INTEG)
sys.path[:0] = [os.path.join(ROOT, "learned_remainder", "chain_local", "stub")]
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
KIND = args.get("KIND", "sp2")
N = int(args.get("N", "2"))
CERT = {"sp2": os.path.join(INTEG, "self_practice", "submission_stage3_sp2.py"),
        "s3i": os.path.join(INTEG, "submission_stage3_s3i.py")}[KIND]
NEW = os.path.join(STRICT, f"submission_stage3_{KIND}_strict.py")
LIB = os.path.join(ROOT, args.get("LIB", f"integration/strict/libs/{KIND}_official_lib.json"))
SPLITS = args.get("SPLITS", "depth_t_1,depth_t_4").split(",")
TESTS = args.get("TESTS", "keep,adam,ops,buffers,forward,statedict,grad").split(",")
DATA = os.environ.get("H1_DATA", os.path.join(os.path.dirname(ROOT), "one-layer-deeper", "data", "generated",
                                              "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


def load(path, name, sets=()):
    import re
    src = open(path).read()
    for k, v in sets:
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v}", src, count=1, flags=re.M)
        assert n == 1, k
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def rows(split, n):
    out = []
    with open(os.path.join(DATA, split + ".jsonl")) as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    L, T = max(len(r["input_ids"]) for r in out), max(len(r["labels"]) for r in out)
    ids = torch.zeros(len(out), L, dtype=torch.long)
    lab = torch.full((len(out), T), -100, dtype=torch.long)
    tpos = torch.full((len(out), T), -1, dtype=torch.long)
    for i, r in enumerate(out):
        a, b = len(r["input_ids"]), len(r["labels"])
        ids[i, :a] = torch.tensor(r["input_ids"])
        lab[i, :b] = torch.tensor(r["labels"])
        tpos[i, :b] = torch.arange(a - b, a)
    return ids, ids != 0, lab, tpos


def model(mod):
    from benchmark import ModelSpec
    cfg = json.load(open(os.path.join(DATA, "dataset_config.json")))
    torch.manual_seed(0)
    return mod.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()


old = load(CERT, "cert", [("MODE", '"designed"')])
new = load(NEW, "strict", [("MODE", '"designed"')])
found = json.load(open(LIB))
lib = {k: new.parse(v) for k, v in found["lib"].items()}
progs = [new.parse(p) for p in found["round"]]
ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)


if "keep" in TESTS:                 # the whole-batch memory: the certified file's list against the buffers
    rnd = random.Random(1)
    good = True
    for cap in (8, 64, 4096):
        rows_, sizes = torch.zeros(cap, 2, 3, dtype=torch.long), torch.zeros(cap, dtype=torch.long)
        ref = []
        for step in range(300):
            n = rnd.choice([0, 1, 3, 7, cap // 2, cap, cap + 1]) if cap < 4096 else rnd.choice([512, 48, 47, 0])
            batch = torch.randint(0, 10, (n, 2, 3))
            if n:                                           # (the certified file appends only nonempty batches here)
                ref.append(batch)
                while sum(b.shape[0] for b in ref) > cap:
                    ref.pop(0)
            new.LibraryMachine.keep(rows_, sizes, batch)
            got = new.LibraryMachine.kept(rows_, sizes)
            want = torch.cat(ref) if ref else None
            if (got is None) != (want is None) or (got is not None and not torch.equal(got, want)):
                good = False
                print("keep differs", cap, step, n, None if got is None else got.shape,
                      None if want is None else want.shape)
                break
    report("keep", good, "(caps 8, 64, 4096; 300 batches each)")

if "adam" in TESTS:                 # the reset in the optimizer against the certified file's in the scheduler
    res = []
    for mod in (old, new):
        torch.manual_seed(5)
        w = torch.nn.Parameter(torch.randn(7, 3))
        ctl = mod.Control()
        opt = (torch.optim.Adam([w], lr=0.05) if mod is old else mod.Adam([w], ctl, lr=0.05))
        sched = mod.Schedule(opt, ctl)
        g = torch.Generator().manual_seed(9)
        for step in range(40):
            target = torch.randn(7, 3, generator=g)
            loss = ((w - target) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            if step in (11, 23):                    # the forward of this step switched to phase B
                ctl.reset = True
            opt.step()
            sched.step()
        res.append(w.detach().clone())
    report("adam", torch.equal(res[0], res[1]), "(two resets in 40 steps: the same parameters)")

if "ops" in TESTS:                  # run_ops against run, values and gradients
    torch.manual_seed(3)
    c = new.exact_cells("cpu")
    logit = lambda t, n: (10.0 * F.one_hot(t, n).float() - 5.0 + 0.3 * torch.randn(*t.shape, n)).requires_grad_()
    raw = [logit(c.sbb_d, 10), logit(c.sbb_c, 2), logit(c.adc_d, 10), logit(c.adc_c, 2), logit(c.mul_lo, 10),
           logit(c.mul_hi, 10)]
    hot = lambda p: new.ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.softmax(-1))
    st = new.round_settings(None, new.PR)
    X, P = 3, new.PR
    regs = [F.one_hot(torch.randint(0, 10, (X, P)), 10).float() for _ in range(5)]
    c2 = F.one_hot(torch.randint(0, 2, (X,)), 2).float()
    h10, k10 = (F.one_hot(torch.randint(0, 10, (X,)), 10).float() for _ in range(2))
    for prog in progs:
        outs = []
        for way in ("run", "ops"):
            for p in raw:
                p.grad = None
            SBB = torch.cat([hot(raw[0]), hot(raw[1])], -1).reshape(200, 12)
            ADC = torch.cat([hot(raw[2]), hot(raw[3])], -1).reshape(200, 12)
            MUL = torch.cat([hot(raw[4]), hot(raw[5])], -1).reshape(10 ** 4, 20)
            m = new.SteMachine(SBB, ADC, MUL, P, new.damp)
            t0 = time.time()
            if way == "run":
                out = m.run(regs, c2, h10, k10, prog, lib)
            else:
                out = m.run_ops(regs, c2, h10, k10, new._trace_seq(st, prog, lib, 0, new._traces(st, lib))[0])
            loss = sum((r * torch.arange(10.0)).sum() for r in out[0]) + out[1][:, 1].sum() + out[3].sum()
            loss.backward()
            outs.append(([r.detach() for r in out[0]], out[1].detach(), out[2].detach(), out[3].detach(), out[4],
                         [None if p.grad is None else p.grad.clone() for p in raw], time.time() - t0))
        a, b = outs
        same = (all(torch.equal(x, y) for x, y in zip(a[0], b[0])) and torch.equal(a[1], b[1])
                and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3]) and a[4] == b[4]
                and all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
                        for x, y in zip(a[5], b[5])))
        nz = sum(int(g is not None and bool((g != 0).any())) for g in b[5])
        report("ops", same, f"{new.show(prog)}: values and gradients equal; {nz}/6 tables with a gradient; "
                            f"run {a[6]:.1f}s, run_ops {b[6]:.1f}s")

if {"buffers", "forward", "statedict", "grad"} & set(TESTS):
    mo, mn = model(old), model(new)
    mo.ctl.lib, mo.ctl.progs = dict(lib), [tuple(p) for p in progs]
    mn.set_round(dict(lib), progs)
    if "buffers" in TESTS:
        st = new.round_settings(None, new.PR)
        tr = [new._trace_seq(st, tuple(p), lib, 0, new._traces(st, lib))[0] for p in progs]
        lib_b, progs_b = mn.round_text()
        used = new.lib_names(lib, [it for p in progs for it in p])
        same = (mn.n_progs() == len(progs) and all(mn.round_trace(i) == tr[i] for i in range(len(progs)))
                and progs_b == [tuple(p) for p in progs] and lib_b == {k: lib[k] for k in lib if k in used})
        n_state = sum(v.numel() for _, v in mn.state_dict().items())
        report("buffers", same, f"{mn.n_progs()} programs, {[len(t) for t in tr]} instructions, {len(lib_b)} of "
                                f"{len(lib)} definitions in round_src; state dict {n_state:,} numbers "
                                f"(buffers {n_state - sum(p.numel() for p in mn.parameters()):,})")
    if "forward" in TESTS:
        for split in SPLITS:
            ids, am, lab, tpos = rows(split, N)
            with torch.no_grad():
                t0 = time.time()
                lo = mo(ids, am)[0]
                t1 = time.time()
                ln = mn(ids, am)[0]
                t2 = time.time()
            tl = ln[torch.arange(len(lab))[:, None], tpos.clamp_min(0)].argmax(-1)
            ex = float(((tl == lab) | (lab == -100)).all(1).float().mean())
            report("forward", torch.equal(lo, ln), f"{split} n={len(lab)}: logits equal; exact {ex:.3f} "
                                                   f"(certified {t1 - t0:.1f}s, strict {t2 - t1:.1f}s)")
    if "statedict" in TESTS:        # the model is its state dict: a fresh model given only mn's answers as mn does
        ids, am, lab, tpos = rows(SPLITS[-1], N)
        res = []
        for mod, m in ((old, mo), (new, mn)):
            torch.manual_seed(1)                     # another random start
            from benchmark import ModelSpec
            cfg = json.load(open(os.path.join(DATA, "dataset_config.json")))
            fresh = mod.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
            sd = m.state_dict()
            for k, v in sd.items():                  # (the round's buffers grow when it is found: size them first)
                mod_, _, name = k.rpartition(".")
                owner = fresh.get_submodule(mod_) if mod_ else fresh
                if name in owner._buffers and owner._buffers[name].shape != v.shape:
                    owner._buffers[name] = torch.empty_like(v)
            fresh.load_state_dict(sd)
            with torch.no_grad():
                res.append(torch.equal(fresh(ids, am)[0], m(ids, am)[0]))
        report("statedict", res[1] and not res[0], f"{SPLITS[-1]} n={N}: a fresh strict model loaded with the state dict "
                                                   f"answers as the trained one: {res[1]}; a fresh certified one: {res[0]} "
                                                   f"(its programs are not in the state dict)")
    if "grad" in TESTS:
        for obj in ("logit", "ce"):
            res = []
            for m in (mo, mn):
                m.zero_grad(set_to_none=True)
                ids, am, lab, tpos = rows(SPLITS[0], N)
                logits = m(ids, am)[0]
                tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)]
                valid = lab != -100
                loss = (-tl[valid].float().gather(1, lab[valid][:, None]).sum() if obj == "logit"
                        else F.cross_entropy(tl[valid].float(), lab[valid]))
                loss.backward()
                got = {name: (None if p.grad is None else float(p.grad.abs().max())) for name, p in m.named_parameters()}
                res.append(got)
            same = res[0] == res[1]
            got = res[1]
            with_g = [k for k, v in got.items() if v]
            zero = [k for k, v in got.items() if v == 0.0]
            none = [k for k, v in got.items() if v is None]
            report(f"grad {obj}", same, f"{SPLITS[0]} n={N}: {len(with_g)}/{len(got)} tensors with a gradient, "
                                        f"all-zero {zero}, none {none}; equal to the certified file's")
            for k in got:
                print(f"    {k}: max |grad| {got[k]}")
print("ALL PASS" if ok_all else "SOME FAILED", flush=True)
