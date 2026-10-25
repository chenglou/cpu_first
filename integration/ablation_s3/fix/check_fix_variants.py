"""Checks the eight fixed copies (make_fix_variants.py, ABLATION_S3.md §8) locally, each against its certified original,
with exact cells and the library its official run found (its round_found line):
1. the edits are §8's: every line a copy adds or removes is one that s3i's or sp2's fix adds or removes (the same
   code), apart from the anchors' own lines;
2. the new probes: with N < X < 2N and X - N the answer at every verify size (above_probes), and the reduction's at
   full size, X = x + q N exactly (by Python's integers), built by doubling with the file's own learned addition;
3. the old sizes' probes are the original's (inputs, targets and the machines' random starts), so a climb whose
   programs were right replays exactly;
4. the verdicts: the file's own conditional subtraction and reduction pass the fixed checks; the alias "x if x < N,
   else 0" and a loop of conditional subtractions (1,000 of them; 10,000 for generic_ops' g3, whose acceptance
   reaches four digits) pass the original's checks and fail the fixed ones. s3i's variants: the reduction's loop
   B[P[T.modB~ T.modB]] on a pool that lets it through (check_fix.py's part 2).
Local: CPU, about a minute, under 1 GB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/fix/check_fix_variants.py [NAME ...]
"""

import ast
import difflib
import json
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CPU_FIRST = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(CPU_FIRST, "wider_menus", "local"))
import local_wide as LW  # noqa: E402

DATA = os.path.expanduser("~/github/one-layer/one-layer-deeper/data/generated/"
                          "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2/train.jsonl")
S3I, SP2 = "integration/submission_stage3_s3i.py", "integration/self_practice/submission_stage3_sp2.py"
REF = {"s3i": (S3I, "integration/ablation_s3/fix/submission_stage3_s3i_fix.py"),
       "sp2": (SP2, "integration/ablation_s3/fix/submission_stage3_sp2_fix.py")}
VARIANTS = {   # name: (original, fixed copy, the original's official record, family)
    "s3i_strict": ("integration/strict/submission_stage3_s3i_strict.py",
                   "integration/strict/fix/submission_stage3_s3i_strict_fix.py",
                   "integration/strict/records/s3is_official.json", "s3i"),
    "sp2_strict": ("integration/strict/submission_stage3_sp2_strict.py",
                   "integration/strict/fix/submission_stage3_sp2_strict_fix.py",
                   "integration/strict/records/sp2s_official.json", "sp2"),
    "ll": ("integration/learned_loop/submission_stage3_ll.py", "integration/learned_loop/fix/submission_stage3_ll_fix.py",
           "integration/learned_loop/records/ll2_official.json", "sp2"),
    "bridge": ("bar3/bridge/submission_stage3_bridge.py", "bar3/bridge/fix/submission_stage3_bridge_fix.py",
               "bar3/bridge/records/runner/submission_stage3_bridge_official2.json", "s3i"),
    "spb_g3": ("integration/sp_breadth/submission_spb_g3.py", "integration/sp_breadth/fix/submission_spb_g3_fix.py",
               "integration/sp_breadth/records/g3_h1_h100.json", "sp2"),
    "book_min": ("integration/sp_ablation/variants/book_min.py", "integration/sp_ablation/fix/book_min_fix.py",
                 "integration/sp_ablation/records/book_min_h100_seed74.json", "sp2"),
    "book_gen": ("integration/sp_ablation/variants/book_gen.py", "integration/sp_ablation/fix/book_gen_fix.py",
                 "integration/sp_ablation/records/book_gen_h100_seed74.json", "sp2"),
    "g3": ("integration/generic_ops/submission_stage3_g3.py", "integration/generic_ops/fix/submission_stage3_g3_fix.py",
           "integration/generic_ops/records/runner/submission_stage3_g3_official.json", "sp2"),
}
N_PROBES, POOL = 60, 4096

names = sys.argv[1:] or list(VARIANTS)
t0 = time.time()
rows = [json.loads(line) for line in open(DATA)]
Nv = torch.tensor([r["modulus"] for r in rows])
xv = torch.tensor([r["x"] for r in rows])
digits = lambda v, L: torch.stack([(v // 10 ** j) % 10 for j in range(L)], 1)
ND, XD = digits(Nv, 8), digits(xv, 8)                                     # the read rows, LSB first
val = lambda D: sum(D[:, j].long() * 10 ** j for j in range(D.shape[1]))  # digits [n, P] -> values (exact: P <= 16)
gen = torch.Generator().manual_seed(1)


def changes(a, b):
    """The lines b adds to a and removes from a, b's first line (the fixed copy's header) left out."""
    la, lb = open(os.path.join(CPU_FIRST, a)).read().splitlines(), open(os.path.join(CPU_FIRST, b)).read().splitlines()
    d = [l for l in difflib.unified_diff(la, lb[1:], n=0, lineterm="") if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    return {l for l in d if l[0] == "+"}, {l for l in d if l[0] == "-"}


def library(record):
    """The library of a record's round_found line, and the routines its round programs call."""
    for l in json.load(open(os.path.join(CPU_FIRST, record)))["stdout"].splitlines():
        if l.startswith("LR round_found"):
            calls = [t for t in l[:l.index("library: ")].replace("'", " ").split() if t.startswith("T.")]
            return ast.literal_eval(l[l.index("library: ") + 9:].strip()), calls
    raise SystemExit(f"{record}: no round_found line")


def above_ok(mod, family, sizes):
    """Part 2's first half: the above probes at the verify sizes."""
    right = drawn = total = 0
    for W in sizes:
        st = mod.Setting(10, W, mod.exact_cells("cpu"))
        p = torch.randint(0, len(rows), (N_PROBES,), generator=gen)
        P = st.P if family == "s3i" else st.P - 1                         # modx's width; red on pair's ("carry")
        Nt, _ = mod.top_parts(ND[p], XD[p], P)
        X, A, ok = mod.above_probes(mod.pad(Nt, st.P), st, mod.gen_on("cpu", 11), "cpu")
        n, x, a = val(mod.pad(Nt, st.P)), val(X), val(A)
        right += int(((n < x) & (x < 2 * n) & (x - n == a))[ok].sum())
        drawn += int(ok.sum())
        total += ok.numel()
    return f"{drawn}/{total} drawn at W = {', '.join(map(str, sizes))}; N < X < 2N and X - N the answer on {right}/{drawn}"


def sp2_parse():
    """sp2's program parser (the same items in every sp2 variant)."""
    if not hasattr(sp2_parse, "fn"):
        sp2_parse.fn = LW.load_module(os.path.join(CPU_FIRST, SP2)).parse
    return sp2_parse.fn


# ---------------------------------------------------------------------------------------------------- s3i's variants
def check_s3i(name, orig_path, fix_path, record):
    orig, fix = (LW.load_module(os.path.join(CPU_FIRST, f)) for f in (orig_path, fix_path))
    lib_txt, _ = library(record)
    modx = lib_txt["T.modx"]
    alias = "T.lt CLA P[SELC.X INC]" if "SELC" in modx else "T.lt LDA.X STA.S CLA SEL.S"   # "x if x < N, else 0"
    real, loop = "B3.1 P[BRG.X STA.Y T.modB~]", "B[P[T.modB~ T.modB]]"
    helpers_txt = {"add": lib_txt["T.add"], "modB": lib_txt["T.modB"]}

    def setup(mod):
        cells = mod.exact_cells("cpu")
        return cells, {k: mod.parse(v) for k, v in lib_txt.items()}, {k: mod.parse(v) for k, v in helpers_txt.items()}
    pool = torch.randint(0, len(rows), (POOL,), generator=gen)

    def machines(mod, Ws, n, seed):
        cells, lib, helpers = setup(mod)
        prob = mod.LawProblem("reduce", [mod.Setting(10, W, cells) for W in Ws], n, seed, (ND[pool], XD[pool]),
                              helpers, lib, "cpu")
        return [b.features() for b in prob.roots] + [t["A"] for t in prob.targets]
    same = all(torch.equal(a, b) for a, b in zip(machines(orig, (1, 2), 8, 0), machines(fix, (1, 2), 8, 0)))
    for W in (1, 2, 3):
        same &= all(torch.equal(a, b) for a, b in zip(machines(orig, (W,), N_PROBES, 11),
                                                     machines(fix, (W,), N_PROBES, 11)))
    print(f"  3. the reduction's machines at the search's and the verify sizes equal the original's: {same}")
    # full size: X = x + q N, q another row's x, by doubling with the learned addition
    cells, lib, helpers = setup(fix)
    prob = fix.LawProblem("reduce", [fix.Setting(10, fix.W, cells)], N_PROBES, 11, (ND[pool], XD[pool]), helpers, lib,
                          "cpu")
    b = prob.roots[0]
    gen11 = fix.gen_on("cpu", 11)
    pick = torch.randint(0, POOL, (4 * N_PROBES,), generator=gen11)
    Nt, xt = fix.top_parts(ND[pool][pick], XD[pool][pick], fix.W)
    q = val(xt[N_PROBES:2 * N_PROBES])
    want = val(xt[:N_PROBES]) + q * val(Nt[:N_PROBES])
    got = val(b.regs[0, fix.RI["X"]].T.long())
    print(f"  2. full size: X = x + q N on {int((got == want).sum())}/{N_PROBES} probes (q up to {int(q.max()):,})")
    # verdicts: a pool that lets the loop through the original's three-digit check
    pick3 = torch.randint(0, POOL, (4 * N_PROBES,), generator=orig.gen_on("cpu", 11))[N_PROBES:2 * N_PROBES]
    shift = (torch.floor(torch.log10(Nv.double())).long() + 1 - 3).clamp(min=0)
    q3 = xv // 10 ** shift
    for _ in range(20000):
        p = torch.randint(0, len(rows), (POOL,), generator=gen)
        if int(q3[p[pick3]].max()) <= 600:
            break
    v = {}
    for tag, mod in (("orig", orig), ("fix", fix)):
        cells, lib, helpers = setup(mod)
        climb = mod.Climb(cells, "cpu", mod.PR, chunk=1 << 13, **dict(mod.CLIMB, budget=20_000, fallback_budget=20_000))
        vf = climb.verify_for if mod is fix else (lambda task: climb.verify)
        red = mod.Acceptor("reduce", lib, helpers, (ND[p], XD[p]), vf("reduce"), "cpu")
        mx = mod.Acceptor("modx", lib, {}, (ND[p], XD[p]), vf("modx"), "cpu")
        v[tag] = (red(mod.parse(loop)), red(mod.parse(real)), mx(mod.parse(alias)), mx(mod.parse(modx)))
    print(f"  4. on a pool whose 60 three-digit probes have q <= 600: the original accepts the loop {v['orig'][0]}, "
          f"the reduction {v['orig'][1]}; the fix {v['fix'][0]}, {v['fix'][1]}")
    print(f"     modx: the original accepts the alias {alias!r} {v['orig'][2]}, its modx {modx!r} {v['orig'][3]}; "
          f"the fix {v['fix'][2]}, {v['fix'][3]}")
    return fix


# ---------------------------------------------------------------------------------------------------- sp2's variants
def check_sp2(name, orig_path, fix_path, record):
    orig, fix = (LW.load_module(os.path.join(CPU_FIRST, f)) for f in (orig_path, fix_path))
    lib_txt, rounds = library(record)
    verify = getattr(fix, "SP_VERIFY", ((10, 1), (10, 2), (10, 3)))
    pair = next(k for k in lib_txt if k.startswith("T.red.pair."))         # the first conditional subtraction
    bd = tuple(pair.split(".")[-1])
    if "SEL.S" in fix.PRIMS:
        alias = f"STA.S T.lt.pair.{bd[0]}{bd[1]} CLA SEL.S" if bd[0] == "A" else None
    else:                                                                  # the if block: A := r0 where the borrow is set
        alias = f"LDA.{bd[0]} T.lt.pair.A{bd[1]} CLA C[LDA.{bd[0]}]"
    reds = sorted({t.rstrip("]") for t in rounds if t.startswith("T.red.x+")})
    loops = 10 ** 4 if len(verify) > 3 else 10 ** 3                       # past every quotient of the verify sizes

    def setup(mod):
        cells = mod.exact_cells("cpu")
        parse = getattr(mod, "parse", None) or sp2_parse()      # (sp_breadth's build drops it: unused)
        lib = {k: parse(v) for k, v in lib_txt.items()}
        ops = mod.Ops()
        for k in lib_txt:                                                  # the routines, in the order practised
            if k.startswith("T.") and k.count(".") >= 3:
                parts = k[2:].split(".")
                ops.add(".".join(parts[:-2]), parts[-2], k, tuple(parts[-1]))
        climb = mod.SelfClimb(cells, "cpu", mod.PR, verify=verify)
        return cells, lib, ops, climb

    def settings(mod, climb, law, pattern):
        if mod is orig:
            return climb.verify
        return (climb.verify + (climb.full if (law, pattern) in mod.FULL_CHECK else [])
                + (climb.above if (law, pattern) in mod.ABOVE_CHECK else []))
    p = torch.randint(0, len(rows), (POOL,), generator=gen)
    tests = [("red", "pair", bd, [lib_txt[pair]] + ([alias] if alias else []))]
    for r in reds:
        pat, rb = r.split(".")[2], tuple(r.split(".")[3])
        sub = (f"STA.Y LDA.X {'B[' * 3}T.red.pair.AY{']' * 3}" if rb == ("X", "A") else
               f"LDA.{rb[0]} {'B[' * 2}T.red.x+dn.A{rb[1]}{']' * 2}")       # 1,000 conditional subtractions
        if loops == 10 ** 4:
            sub = sub.replace("B[", "B[B[", 1).replace("]", "]]", 1)
        tests.append(("red", pat, rb, [lib_txt[r], sub]))
    res, probs = {}, {}
    for tag, mod in (("orig", orig), ("fix", fix)):
        cells, lib, ops, climb = setup(mod)
        for law, pat, b, progs in tests:
            acc = mod.Acceptor(law, types.SimpleNamespace(pattern=pat, binding=b),
                               settings(mod, climb, law, pat), (ND[p], XD[p]), ops, lib, "cpu")
            res[tag, pat] = acc.many([(getattr(mod, "parse", None) or sp2_parse())(s) for s in progs], lib)
            probs[tag, pat] = acc.probs
    same = True
    for law, pat, b, _ in tests:
        for (st0, b0, g0), (st1, b1, g1) in zip(probs["orig", pat], probs["fix", pat][:len(verify)]):
            same &= torch.equal(b0.regs, b1.regs) and all(torch.equal(g0.targets[0][k], g1.targets[0][k])
                                                          for k in g0.targets[0] if torch.is_tensor(g0.targets[0][k]))
    print(f"  3. the acceptance machines at the verify sizes {[w for _, w in verify]} equal the original's: {same}")
    for law, pat, b, progs in tests:
        if pat != "pair":                                                  # the full-size probes, exactly
            cells, lib, ops, climb = setup(fix)
            r = fix.Root(pat, b, climb.full, N_PROBES, (ND[p], XD[p]), ops, lib, "cpu", 11)
            d = r.draws[0]
            q = fix.count_of(d.o if pat == "x+on" else d.x)                 # x + o N, or book_gen's x + x N
            want = val(d.x) + q * val(d.N)
            got = val(r.roots[0].regs[0, fix.RI[b[0]]].T.long())
            print(f"  2. {law} on {pat} at W = {fix.W}: X = x + q N on {int((got == want).sum())}/{d.n} probes "
                  f"(q up to {int(q.max()):,})")
        what = ["its routine", "the alias" if pat == "pair" else f"a loop of {loops:,} subtractions"]
        print(f"  4. {law} on {pat} in {''.join(b)}: " + "; ".join(
            f"{w} {progs[i]!r}: original {res['orig', pat][i]}, fix {res['fix', pat][i]}"
            for i, w in enumerate(what[:len(progs)])))
    return fix


for name in names:
    orig_path, fix_path, record, family = VARIANTS[name]
    print(f"{name}: {fix_path}")
    ref_add, ref_rm = changes(*REF[family])
    add, rm = changes(orig_path, fix_path)
    print(f"  1. lines added: {len(add)}, {len(add - ref_add)} not in {family}'s fix {sorted(add - ref_add)}; "
          f"removed: {len(rm)}, {len(rm - ref_rm)} not in {family}'s fix {sorted(rm - ref_rm)}")
    fix = (check_s3i if family == "s3i" else check_sp2)(name, orig_path, fix_path, record)
    sizes = [w for _, w in getattr(fix, "SP_VERIFY", ((10, 1), (10, 2), (10, 3)))]
    print(f"  2. above probes: {above_ok(fix, family, sizes)}")
print(f"({time.time() - t0:.0f} s)")
