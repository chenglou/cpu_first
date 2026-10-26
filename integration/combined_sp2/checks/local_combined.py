"""Each study's local checks, run on the combined file (integration/combined_sp2/COMBINED.md §3). Every part loads the
built file as a module (the evaluator's `benchmark` from learned_remainder/chain_local/stub) and reuses the study's own
check code where it takes a module; the parts that needed changing say how. Exact cells; CPU; local.

  policy    the contest's static source policy (one-layer-deeper/submission_validation.py) and the 256 KiB limit
  machines  learned_loop/checks/test_count.py on this file's machines: run_items, apply_items, the outer machine and
            the one-hot machine, both from the items and from the buffers' instructions (answer_trace), on count-loop
            programs, on programs that mix count loops and C blocks (ll x g3), the count's error, and H1's answer
            program C.S[T.round1 STA.X] at full size with g3's official run's library, T = 1 to 16
  isa       generic_ops/test_isa.py's checks on this file's instruction set (if1,k), its round machine replaced by the
            outer machine (round_machine is gone with the countdown) and the one-hot machine also run from the buffers
  designed  learned_loop/checks/designed_ll.py: the prediction path with the reader and cells at the truth, the
            answer program in the buffers (set_round), on H1's splits up to T = 64; and run_outer, the same rows
  buffers   strict/checks/strict_local.py's keep / buffers / statedict, and the count rows' buffers against the
            learned loop's dict (the same rows, in the same order, at every step of random batches)
  replay    learned_loop/checks/smoke_replay.py: a finished climb with g3's official rounds replayed, then training
            steps: the count rows fill, the answer program is found and put in the buffers, phase D, the freeze
  climb     generic_ops/smoke_g.py: phase C from the climb's start with toy budgets (the climb runs inside training,
            its work counted); and the budgets bind when tiny (strict's check)
  fix       ablation_s3/fix/check_fix_variants.py's check_sp2 with this file as the fixed copy and g3 as the original
            (g3's official run's library: its bindings and instruction set are this file's)
  book      the generated book's code is book_gen_fix.py's, line for line; and sp_ablation/check_books.py's part 2 (every
            target the book gives is the law's value by Python's integers) on this file

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/combined_sp2/checks/local_combined.py PART ...
"""
import json
import os
import random
import re
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
COMB = os.path.dirname(HERE)
INTEG = os.path.dirname(COMB)
CPU = os.path.dirname(INTEG)
sys.path[:0] = [os.path.join(CPU, "learned_remainder", "chain_local", "stub"), os.path.join(CPU, "wider_menus", "local"),
                CPU, INTEG]
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FILE = os.environ.get("COMBINED_FILE", os.path.join(COMB, "submission_stage3_combined.py"))
DATA = os.environ.get("H1_DATA", os.path.join(os.path.dirname(CPU), "one-layer-deeper", "data", "generated",
                                              "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
G3_LIB = os.path.join(HERE, "g3_official_lib.json")      # strict/checks/lib_from_record.py of g3's official record
dev = "cpu"
t00 = time.time()


def load(sets=()):
    """The combined file as a module (sets: (NAME, source) pairs, each replacing a constant's line)."""
    src = open(FILE).read()
    for k, v in sets:
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v}", src, count=1, flags=re.M)
        assert n == 1, k
    mod = types.ModuleType("combined")
    sys.modules["combined"] = mod
    exec(compile(src, FILE, "exec"), mod.__dict__)
    return mod


def g3_lib(M, round_i=None):
    """g3's official run's library (if1,k, its bindings) with its rounds as T.round<i> (from 1)."""
    d = json.load(open(G3_LIB))
    lib = {k: M.parse(v) for k, v in d["lib"].items()}
    for i, r in enumerate(d["round"]):
        lib[f"T.round{i + 1}"] = M.parse(r)
    return lib, [M.parse(r) for r in d["round"]]


def say(*a):
    print(*a, f"({time.time() - t00:.0f}s)", flush=True)


# ================================================================================================== policy
def part_policy():
    sys.path.insert(0, os.path.join(os.path.dirname(CPU), "one-layer-deeper"))
    from submission_validation import validate_submission_source
    src = open(FILE).read()
    validate_submission_source("submission.py", src, 256 * 1024)
    say(f"policy: the contest's static source policy passes; {len(src.encode()):,} bytes of {256 * 1024:,}")


# ================================================================================================== machines
def machines_for(M, cells):
    def start(st, ins, n, seed):
        return M.fresh(st, {r: M.pad(v, st.P) for r, v in ins.items()}, M.gen_on(dev, seed), n, dev)

    def by_ref(st, prog, lib, ins, n, seed):
        b = start(st, ins, n, seed)
        M.run_items(st, b, prog, lib)
        return b

    def by_apply(st, prog, lib, ins, n, seed):
        b = start(st, ins, n, seed).take(torch.zeros(3, dtype=torch.long))
        for it in prog:
            b = M.apply_items(st, b, torch.zeros(3, dtype=torch.long), [it], lib)
        assert all(bool((b.regs[0] == b.regs[i]).all()) for i in range(3))
        return b.take(torch.zeros(1, dtype=torch.long))

    def by_outer(st, prog, lib, ins, n, seed, from_ops=False):
        m = (M.outer_machine_ops(st, M.answer_trace(st, prog, lib), n, dev) if from_ops
             else M.outer_machine(st, prog, lib, n, dev))
        return m.run(ins["X"], ins["N"], ins["S"], M.gen_on(dev, seed)).clone()

    def by_ste(st, prog, lib, ins, n, seed, from_ops=False):
        b = start(st, ins, n, seed)
        hot = lambda t, k: F.one_hot(t.long(), k).float()
        regs = [hot(b.regs[0, r].T, 10) for r in range(5)]
        SBB = torch.cat([hot(cells.sbb_d, 10), hot(cells.sbb_c, 2)], -1).reshape(200, 12)
        ADC = torch.cat([hot(cells.adc_d, 10), hot(cells.adc_c, 2)], -1).reshape(200, 12)
        MUL = torch.cat([hot(cells.mul_lo, 10), hot(cells.mul_hi, 10)], -1).reshape(10 ** 4, 20)
        m = M.SteMachine(SBB, ADC, MUL, st.P, lambda x: x)
        args = (regs, hot(b.c[0], 2), hot(b.h[0], 10), hot(b.k[0], 10))
        out = (m.run_ops(*args, M.answer_trace(st, prog, lib)) if from_ops else m.run(*args, prog, lib, p=0))
        return out[0][0].argmax(-1)

    def check(st, prog, lib, ins, n, seed=3, ste=True):
        ref = by_ref(st, prog, lib, ins, n, seed)
        app = by_apply(st, prog, lib, ins, n, seed)
        same = bool((ref.regs == app.regs).all() and (ref.c == app.c).all() and (ref.h == app.h).all()
                    and (ref.k == app.k).all() and (ref.p == app.p).all())
        A = ref.regs[0, M.A_].T.long()
        outer = all(bool((by_outer(st, prog, lib, ins, n, seed, o) == A).all()) for o in (False, True))
        oh = all(bool((by_ste(st, prog, lib, ins, n, seed, o) == A).all()) for o in (False, True)) if ste else None
        return same, outer, oh, A
    return check, by_ref


def part_machines():
    M = load()
    cells = M.Cells(10, dev)
    check, by_ref = machines_for(M, cells)
    rng = random.Random(0)
    st = M.Setting(10, 2, cells, 4)
    n = 16
    g = torch.Generator().manual_seed(0)
    S = torch.randint(0, 13, (n, 4), generator=g) * 0
    S[:, 0] = torch.randint(0, 10, (n,), generator=g)
    S[:4, 1] = 1                                                       # counts 10-12 ...
    S[4, :3] = torch.tensor([0, 5, 1])                                 # ... and 150: above the cap
    ins = {"X": torch.randint(0, 10, (n, 4), generator=g), "N": torch.randint(0, 10, (n, 4), generator=g), "S": S}
    prims = [p for p in M.PRIMS if p != "INC"] + ["INC"] * 3
    fails = 0
    progs = [M.parse("C.S[INC ADC.X]"), M.parse("C.S[ADC.X STA.X]"), M.parse("LDA.X C.S[STA.Y ADC.Y INC] SHL.A"),
             M.parse("C.X[INC]"), M.parse("C.S[]"), M.parse("C.S[SBB.N] C.S[ADC.N STA.Y]")]
    for _ in range(40):                                               # test_count.py's random programs
        body = tuple(rng.choice(prims) for _ in range(rng.randint(1, 4)))
        pre = tuple(rng.choice(prims) for _ in range(rng.randint(0, 2)))
        post = tuple(rng.choice(prims) for _ in range(rng.randint(0, 2)))
        kind = rng.choice(M.COUNT_KINDS[1:] if rng.random() < 0.2 else ("C.S",))
        progs.append(pre + (("REP", kind, body),) + post)
    n_ll = len(progs)
    cblock = lambda: ("REP", "C", (rng.choice(prims),))                # g3's if block (if1: one item)
    progs += [M.parse("C.S[SBB.Y C[ADC.Y] INC]"), M.parse("SBB.X C[STA.Y] C.S[C[LDA.Y] ADC.X]"),
              M.parse("C.S[SBB.N C[C[ADC.N] STA.X] SHL.A]")]
    for _ in range(30):                                               # count loops and C blocks together (ll x g3)
        body = tuple(cblock() if rng.random() < 0.4 else rng.choice(prims) for _ in range(rng.randint(1, 4)))
        pre = tuple(cblock() if rng.random() < 0.4 else rng.choice(prims) for _ in range(rng.randint(0, 2)))
        post = tuple(cblock() if rng.random() < 0.4 else rng.choice(prims) for _ in range(rng.randint(0, 2)))
        kind = rng.choice(M.COUNT_KINDS[1:] if rng.random() < 0.2 else ("C.S",))
        progs.append(pre + (("REP", kind, body),) + post)
    for prog in progs:
        same, ok_o, ok_s, _ = check(st, prog, {}, ins, n)
        if not (same and ok_o and ok_s):
            fails += 1
            print("FAIL", M.show(prog), same, ok_o, ok_s)
    say(f"machines: {n_ll} of test_count.py's programs and {len(progs) - n_ll} with C blocks at P = 4, every machine "
        f"from items and from instructions: {fails} failing")
    b = by_ref(st, M.parse("CLA CLC C.S[ADC.X CLC]"), {}, ins, n, 3)
    assert int(b.regs[0, :, :, 4].sum()) == 0 and int(b.c[0, 4]) == 0, "the error clears the machine"
    want = (S[:, 0] + 10 * S[:, 1]) * ins["X"][:, 0] % 10
    keep = torch.arange(n) != 4
    assert bool((b.regs[0, M.A_, 0].long()[keep] == want[keep]).all())
    say("machines: counts 0-12 run that many times; 150 clears the probe")
    from integration.self_practice.core import digits, load_rows
    lib, rounds = g3_lib(M)
    st = M.round_settings(cells, 16)
    prog = M.parse("C.S[T.round1 STA.X]")
    for T in (1, 2, 4, 8, 16):
        rows = load_rows(6, seed=T, T=T)
        Nd, Xd, Yd = (digits(rows, k) for k in ("modulus", "x", "result"))
        ins = {"X": Xd, "N": Nd, "S": digits(rows, "time_steps", 2)}
        same, ok_o, ok_s, A = check(st, prog, lib, ins, len(rows), ste=T <= 2)
        right = bool((A[:, :8] == Yd).all())
        say(f"machines: {M.show(prog)} (T.round1 = {M.show(rounds[0])}) T={T}: machines agree {same}, outer {ok_o}, "
            f"one-hot {ok_s}; the answer right {right}")
        fails += not (same and ok_o and ok_s is not False and right)
    print("MACHINES OK" if not fails else f"MACHINES: {fails} FAILURES", flush=True)
    return fails


# ================================================================================================== isa
def part_isa():
    from integration.generic_ops import test_isa as TI
    M = load()
    isa = f"{M.COND},{M.SHIFT}" + (",brg" if M.BRING else "")
    say(f"isa: {isa}, {len(M.PRIMS)} primitives {M.PRIMS}; loops {M.LOOP_KINDS}")
    assert M.WHOLE and all(t in M.PRIMS for t in M.WHOLE)
    gen = M.gen_on(dev, 3)
    cells = M.Cells(10, dev)
    sets = [M.Setting(10, 1, cells), M.Setting(10, 2, cells), M.Setting(10, 8, cells, 16)]
    lib = TI.lib_for(M, isa)
    for st in sets:
        TI.check_lockstep(M, f"primitives {st} P={st.P}", st, list(M.PRIMS), {}, gen)
    voc = [c.rep for c in M.enumerate_items(list(M.PRIMS), {}, sets[:2], dev, max_body=2)]
    say(f"isa: vocabulary to body cost 2: {len(voc)} items, {sum(1 for v in voc if M.is_loop(v) and v[1] == 'C')} C "
        f"blocks")
    for st in sets[:2]:
        TI.check_lockstep(M, f"vocabulary {st}", st, voc, {}, gen)
    nested = [M.parse(s) for s in ("C[T.link]", "B[C[T.lt]]", "C[C[LDA.S] STA.X]", "P[T.link INC]", "C[T.mac SHL.Y]",
                                   "T.chain SHL.X SHL.A", "C[SHR.X] SHL.Y", "W[C[ADC.X] INC]")]
    nested = [p[0] if len(p) == 1 else ("REP", M.ONE, p) for p in nested]
    comp = [c.rep for c in M.enumerate_items(list(M.PRIMS) + sorted(lib), lib, sets[:2], dev, max_body=1)] + nested
    for st in sets:
        TI.check_lockstep(M, f"routines and nests {st}", st, comp, lib, gen, rows=2000)
    from integration.self_practice.core import digits, load_rows
    st = M.round_settings(cells, 16)
    rows = load_rows(64, seed=1, T=1)
    Nd, Xd, Yd = (digits(rows, k).to(dev) for k in ("modulus", "x", "result"))
    progs = [tuple(M.parse("T.mul STA.X LDA.N T.red"))]
    for _ in range(40):
        progs.append(tuple(comp[int(i)] for i in torch.randint(0, len(comp), (5,), generator=gen, device=dev)))
    bad = 0
    for prog in progs:                       # the outer machine (from its instructions) against run_prog at 16 places
        zS = torch.zeros(Xd.shape[0], 16, dtype=torch.long)          # (the outer machine starts with the count in S)
        ref, _ = M.run_prog(st, prog, lib, {"X": M.pad(Xd, 16), "N": M.pad(Nd, 16), "S": zS}, M.gen_on(dev, 5), dev)
        m = M.outer_machine_ops(st, M.answer_trace(st, prog, lib), Xd.shape[0], dev)
        got = m.run(Xd, Nd, torch.zeros(Xd.shape[0], 2, dtype=torch.long), M.gen_on(dev, 5))
        bad += not bool(torch.equal(ref, got.long()))
    say(f"isa: the outer machine vs run_prog, {len(progs)} programs at 16 places: {bad} differ")
    assert bad == 0
    oh = lambda t, n: F.one_hot(t.long().flatten(), n).float()
    SBB = torch.cat([oh(cells.sbb_d, 10), oh(cells.sbb_c, 2)], 1)
    ADC = torch.cat([oh(cells.adc_d, 10), oh(cells.adc_c, 2)], 1)
    MUL = torch.cat([oh(cells.mul_lo, 10), oh(cells.mul_hi, 10)], 1)
    ste = M.SteMachine(SBB, ADC, MUL, 16, lambda x: x)
    bad = 0
    for prog in progs[:25]:                  # the one-hot machine, from the items and from the instructions
        prog = M.unwrap(prog)
        n = Xd.shape[0]
        b = M.Batch(torch.zeros(1, 5, 16, n, dtype=torch.uint8), torch.zeros(1, n, dtype=torch.uint8),
                    torch.zeros(1, n, dtype=torch.uint8), torch.zeros(1, n, dtype=torch.uint8),
                    torch.zeros(1, dtype=torch.long))
        b.regs[0, M.X_] = M.pad(Xd, 16).T.to(torch.uint8)
        b.regs[0, M.N_] = M.pad(Nd, 16).T.to(torch.uint8)
        M.run_items(st, b, prog, lib)
        z = torch.zeros(n, 16, 10)
        z[..., 0] = 1
        regs = [z, oh(M.pad(Xd, 16), 10).view(n, 16, 10), z, oh(M.pad(Nd, 16), 10).view(n, 16, 10), z]
        z2, z10 = torch.zeros(n, 2), torch.zeros(n, 10)
        z2[:, 0] = z10[:, 0] = 1
        for out in (ste.run(regs, z2, z10, z10, prog, lib), ste.run_ops(regs, z2, z10, z10, M.answer_trace(st, prog, lib))):
            got = torch.stack([r.argmax(-1) for r in out[0]], 0)
            exact = all(bool(((r == 0) | (r == 1)).all()) for r in out[0])
            same_ = (bool(torch.equal(got.permute(0, 2, 1), b.regs[0].long())) and torch.equal(out[1].argmax(-1), b.c[0].long())
                     and torch.equal(out[3].argmax(-1), b.k[0].long()) and torch.equal(out[2].argmax(-1), b.h[0].long()))
            bad += not (same_ and exact)
    say(f"isa: the one-hot machine (items and instructions) vs run_items, 25 programs at 16 places: {bad} differ")
    assert bad == 0
    rnd = M.parse("LDA.X STA.Y T.mul STA.X LDA.N T.red")
    rows = load_rows(400, seed=1, T=1)
    ans = tuple(digits(rows, k).to(dev) for k in ("modulus", "x", "result"))
    ok = M.round_ok(rnd, lib, M.Cells(10, dev), ans, 16, dev)
    say(f"isa: the hand-written round {M.show(rnd)}, T.link = {M.show(lib['T.link'])}, T.red = {M.show(lib['T.red'])}: "
        f"round_ok on 400 T = 1 rows {ok}")
    assert ok
    print("ISA OK", flush=True)


# ================================================================================================== designed
def designed_model(M, prog_text="C.S[T.round1 STA.X]"):
    from benchmark import ModelSpec
    cfg = json.load(open(os.path.join(DATA, "dataset_config.json")))
    m = M.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
    lib, _ = g3_lib(M)
    m.set_round(lib, [M.parse(prog_text)])
    return m


def part_designed(n=8):
    import local_check as lc
    M = load([("MODE", '"designed"')])
    m = designed_model(M)
    lib, progs = m.round_text()
    say(f"designed: the buffers hold {[M.show(p) for p in progs]} and {len(lib)} definitions; "
        f"round_ops {tuple(m.round_ops.shape)}; parameters {sum(p.numel() for p in m.parameters()):,}")
    fails = 0
    for split in ("depth_t_1", "depth_t_8", "depth_t_16", "depth_ood_n_t_2", "depth_ood_n_t_64"):
        rows = lc.glc.rows(DATA, split, n)
        ids, am, lab, tpos = lc.glc.batch(rows)
        t0 = time.time()
        with torch.no_grad():
            logits = m(ids, am)[0]
        tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)].argmax(-1)
        ex = float(((tl == lab) | (lab == -100)).all(1).float().mean())
        dg = lambda key, L=8: torch.tensor([[int(c) for c in reversed(str(r[key]))][:L] + [0] * (L - min(L, len(str(r[key]))))
                                            for r in rows])
        with torch.no_grad():
            s = m.run_outer(dg("x"), dg("time_steps", M.K), dg("modulus"), 0)
        hard = float((s == dg("result")).all(1).float().mean())
        fails += ex < 1 or hard < 1
        say(f"designed: {split} n={len(lab)}: prediction path exact {ex:.3f}, run_outer exact {hard:.3f} "
            f"({time.time() - t0:.1f}s)")
    print("DESIGNED ALL EXACT" if not fails else f"DESIGNED: {fails} splits not exact", flush=True)
    return fails


# ================================================================================================== buffers
def part_buffers():
    import local_check as lc
    M = load([("MODE", '"designed"')])
    fails = 0
    # keep: the count rows' buffers against the learned loop's dict (its keep_counts and outer_unit's rows)
    m = designed_model(M)
    m.ctl.phase, m.ctl.io = "C", 5
    pool, gen = {}, torch.Generator().manual_seed(5)
    K, W = M.K, M.W
    for step in range(300):
        nb = int(torch.randint(1, 200, (1,), generator=gen))
        T = torch.tensor([1, 2, 4, 8, 11, 3, 9, 0, 22])[torch.randint(0, 9 if step > 100 else 4, (nb,), generator=gen)]
        Cd = torch.stack([T % 10, T // 10], 1)
        Nd, Xd, yd = (torch.randint(0, 10, (nb, W), generator=gen) for _ in range(3))
        aux = {"Nd": Nd, "Xd": Xd, "Cd": Cd}
        m.t_readings = lambda aux_, Cd=Cd: (torch.ones(Cd.shape[0], dtype=torch.bool), None, None, None, None, Cd, Cd)
        m.keep_counts(aux, yd)
        sel = torch.arange(nb)                                         # the learned loop's dict, as it kept them
        for key in {tuple(r) for r in Cd.tolist()}:
            idx = sel[(Cd == torch.tensor(key)).all(1)]
            new = (Nd[idx], Xd[idx], Cd[idx], yd[idx])
            old = pool.get(key)
            pool[key] = new if old is None else tuple(torch.cat([a, b]) for a, b in zip(old, new))
        share = min(M.COUNT_EACH, max(1, M.COUNT_POOL // len(pool)))
        for key, v in pool.items():
            pool[key] = tuple(t[-share:] for t in v)
        keys = torch.nonzero(m.count_n).flatten().tolist()
        R = torch.cat([m.count_rows[s, :int(m.count_n[s])] for s in keys])
        got = (R[:, 0], R[:, 1], R[:, 2, :K], R[:, 3])
        want = tuple(torch.cat([pool[k][i] for k in sorted(pool)]) for i in range(4))
        fails += not all(torch.equal(a, b) for a, b in zip(got, want))
    say(f"buffers: the count rows' buffers equal the learned loop's dict after each of 300 random batches: "
        f"{'yes' if not fails else 'NO'}; kept {m.counts_kept()}")
    # buffers: the instructions are answer_trace's, the text reads back
    m = designed_model(M, "C.S[T.round1 STA.X]")
    lib, progs = m.round_text()
    full, _ = g3_lib(M)
    st = M.round_settings(None, M.PR)
    ok = (progs == [M.parse("C.S[T.round1 STA.X]")] and all(lib[k] == full[k] for k in lib)
          and list(m.round_trace(0)) == list(M.answer_trace(st, progs[0], full))
          and M.lib_names(full, progs[0]) == set(lib))
    say(f"buffers: round_ops is answer_trace's ({int(m.round_len[0])} instructions), round_src reads back the program "
        f"and the {len(lib)} definitions it uses: {ok}")
    fails += not ok
    # statedict: a fresh model from another random start, loaded with this one's state dict, gives its logits
    from benchmark import ModelSpec
    cfg = json.load(open(os.path.join(DATA, "dataset_config.json")))
    torch.manual_seed(123)
    m2 = M.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
    sd = m.state_dict()
    for k in ("round_ops", "round_src"):                              # (they grow when the round is found)
        setattr(m2, k, torch.zeros_like(sd[k]))
    m2.load_state_dict(sd)
    rows = lc.glc.rows(DATA, "depth_t_4", 6)
    ids, am, lab, tpos = lc.glc.batch(rows)
    with torch.no_grad():
        a, b = m(ids, am)[0], m2(ids, am)[0]
    same = bool(torch.equal(a, b))
    say(f"buffers: a fresh model loaded with the state dict gives the same logits: {same}; state "
        f"{sum(p.numel() for p in m.parameters()):,} parameters, {sum(b.numel() for b in m.buffers()):,} buffer numbers")
    fails += not same
    print("BUFFERS OK" if not fails else f"BUFFERS: {fails} FAILURES", flush=True)
    return fails


# ================================================================================================== replay
def part_replay(steps=80, B=64):
    import local_wide as LW
    from benchmark import OptimizerSpec, TokenLossBatch
    base = [("MODE", '"designed"'), ("LOG_EVERY", "10"), ("FREEZE_WIN", "20"), ("T_AFTER", "3"),
            ("OUTER_STATES", "300000")]
    M = LW.sc = load(base)
    torch.manual_seed(0)
    m = M.build_model(LW.spec())
    m.ctl.phase, m.ctl.io, m.ctl.gated_on = "C", 4, -1000
    m.ctl.cells_on = True
    m.train()
    bundle = M.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    full = {k: M.parse(v) for k, v in json.load(open(G3_LIB))["lib"].items()}
    rounds = [M.parse(r) for r in json.load(open(G3_LIB))["round"]]
    keep = M.lib_names(full, [it for r in rounds for it in r])
    m.ctl.cells = m.snapped_cells()
    c = m.ctl.climb = M.SelfClimb(m.ctl.cells, "cpu", M.PR, chunk=1 << 18, **M.SP)
    m.ctl.climb_work0 = M.WORK[0]
    c.lib = {k: v for k, v in full.items() if k in keep}
    c.rounds, c.done = rounds, True
    say(f"replay: a finished climb with g3's official rounds and the {len(c.lib)} definitions they use")
    rows = LW.load("train")
    g = torch.Generator().manual_seed(0)
    for step in range(1, steps + 1):
        t0 = time.time()
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab, tpos = LW.batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = M.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        bundle.optimizer.step()
        bundle.scheduler.step()
        ctl = m.ctl
        print(f"STEP {step} loss {float(loss):.4f} phase {ctl.phase} io {ctl.io} counts {m.counts_kept()} ans_ok "
              f"{ctl.ans_ok} streak {ctl.streak} progs {m.prog_names()} {time.time() - t0:.2f}s", flush=True)
        if ctl.phase == "frozen":
            break
    progs = m.round_text()[1]
    ok = m.ctl.phase == "frozen" and progs == [M.parse(f"C.S[T.round{i + 1} STA.X]") for i in range(len(rounds))]
    print(f"REPLAY FROZEN, the answer programs {[M.show(p) for p in progs]} in the buffers" if ok else
          f"REPLAY NOT FROZEN {m.ctl.history}", flush=True)
    return not ok


# ================================================================================================== climb
def part_climb(steps=30, B=64, tiny=False):
    import local_wide as LW
    from benchmark import OptimizerSpec, TokenLossBatch
    base = [("MODE", '"designed"'), ("LOG_EVERY", "10"), ("SP_PROBE", "30000"), ("ROUND_STATES", "20000"),
            ("SP_LEVELS", "2"), ("ANS_MIN", "8"), ("SP_DEEP", "30000"), ("SP_FALLBACK", "30000")]
    if tiny:                                         # strict's binding check: the budgets end the searches
        base += [("SEARCH_WORK", "200_000"), ("CLIMB_WORK", "2_000_000")]
    M = LW.sc = load(base)
    torch.manual_seed(0)
    m = M.build_model(LW.spec())
    m.ctl.phase, m.ctl.io, m.ctl.gated_on = "C", 5, -1000
    m.ctl.cells_on = True
    m.train()
    bundle = M.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    rows = LW.load("train")
    g = torch.Generator().manual_seed(0)
    for step in range(1, steps + 1):
        t0 = time.time()
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab, tpos = LW.batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = M.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        bundle.optimizer.step()
        bundle.scheduler.step()
        print(f"STEP {step} loss {float(loss):.4f} phase {m.ctl.phase} answers t={int(m.ans_t)} "
              f"{int(m.ans_sizes.sum())} pool {int(m.pool_sizes.sum())} work {M.WORK[0]:.4g} {time.time() - t0:.2f}s",
              flush=True)
    c = m.ctl.climb
    say(f"climb{' (tiny budgets)' if tiny else ''}: level {c.level if c else None}, {len(c.lib) if c else 0} "
        f"definitions, work {M.WORK[0]:.4g}; climb lines {len(m.ctl.climb_log)}")
    return 0


# ================================================================================================== fix
def part_fix():
    src = open(os.path.join(INTEG, "ablation_s3", "fix", "check_fix_variants.py")).read()
    head = src[:src.index("\nfor name in names:\n")]
    ns = {"__name__": "check_fix_variants", "__file__": os.path.join(INTEG, "ablation_s3", "fix", "x.py")}
    argv, sys.argv = sys.argv, ["check_fix_variants.py", "g3"]
    try:
        exec(compile(head, "check_fix_variants.py", "exec"), ns)
    finally:
        sys.argv = argv
    rel = os.path.relpath(FILE, CPU)
    say(f"fix: check_fix_variants.check_sp2 with {rel} as the fixed copy, g3 as the original, g3's official library")
    fix = ns["check_sp2"]("combined", "integration/generic_ops/submission_stage3_g3.py", rel,
                          "integration/generic_ops/records/runner/submission_stage3_g3_official.json")
    sizes = [w for _, w in fix.SP_VERIFY]
    say(f"fix: above probes: {ns['above_ok'](fix, 'sp2', sizes)}")


# ================================================================================================== book
def part_book():
    comb = open(FILE).read()
    gen = open(os.path.join(INTEG, "sp_ablation", "fix", "book_gen_fix.py")).read()
    a, b = "# ===================================================================================================== the law book\n", \
        "# ===================================================================================================== many goals\n"
    same = comb[comb.index(a):comb.index(b)] == gen[gen.index(a):gen.index(b)]
    say(f"book: the law book's code (PATTERNS to the end of law_target) is book_gen_fix.py's, line for line: {same}")
    import check_books_part2 as P2
    return (not same) + P2.run(load())


# ================================================================================================== grad
def part_grad(n=2, split="depth_t_2"):
    """learned_loop/checks/gradtest_ll.py: the reader and cells at the truth, two answer programs in the tournament
    (the count loop over two of g3's rounds: with one, the tournament's softmax is constant), the right tokens' logits
    on n rows of one split; which parameter tensors get a gradient, through the buffers' instructions."""
    import local_check as lc
    M = load([("MODE", '"designed"')])
    m = designed_model(M)
    lib, _ = g3_lib(M)
    m.set_round(lib, [M.parse("C.S[T.round1 STA.X]"), M.parse("C.S[T.round3 STA.X]")])
    ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(DATA, split, n))
    t0 = time.time()
    logits = m(ids, am)[0]
    tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)]
    valid = lab != -100
    loss = -tl[valid].float().gather(1, lab[valid][:, None]).sum()
    say(f"grad: {split} n={n} exact={float(((tl.argmax(-1) == lab) | ~valid).all(1).float().mean()):.3f} "
        f"({time.time() - t0:.1f}s forward)")
    loss.backward()
    none, zero, got = [], [], []
    for name, p in m.named_parameters():
        if p.grad is None:
            none.append(name)
        elif float(p.grad.abs().max()) == 0.0:
            zero.append(name)
        else:
            got.append((name, float(p.grad.abs().max())))
    say(f"grad: {len(list(m.parameters()))} tensors; with a gradient: {len(got)} | none: {none} | all zero: {zero}")
    for name, g in got:
        print(f"   {name}: max |grad| {g:.3e}", flush=True)


PARTS = {"policy": part_policy, "grad": part_grad, "machines": part_machines, "isa": part_isa, "designed": part_designed,
         "buffers": part_buffers, "replay": part_replay, "climb": part_climb,
         "climb_tiny": lambda: part_climb(steps=12, tiny=True), "fix": part_fix, "book": part_book}

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    for p in sys.argv[1:] or ["policy", "machines", "isa", "designed", "buffers"]:
        PARTS[p]()
