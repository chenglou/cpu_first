"""The fix of two weak checks (ABLATION_S3.md §8), in s3i and likewise in sp2. Writes fix/<name>.py from the original by
a few text edits, each asserted to occur exactly once, and prints the md5s; reads the originals and never writes them.

1. The reduction. Its practice problem draws x + q N with q another row's top W digits, and its acceptance checks at
   W = 1, 2 and 3 digits only. A loop of conditional subtractions (B[P[T.modB~ T.modB]], 600 of them at 3 digits)
   passes when no 3-digit probe's quotient is above 600, about 3% of the time on H1's rows, and is wrong at full size
   (reduce_check.py). The fix checks the reduction at W = 8 as well, where q reaches 10^8 and no loop within the
   search's reach performs that many subtractions. Building those probes by q single additions would take up to 10^8
   of them, so at that size x + q N is built by doubling with the learner's own addition (the same sum: about 40
   additions).
2. The conditional subtraction (modx; sp2's red on pair). Its probes are x < N (keep x) and x = N (0) only; a program
   that keeps everything but N, or gives 0 from N up, passes (the bridge session's climb accepted one, and so did
   occam_steps_off's). The fix also checks it with N < X < 2N at the verify sizes: X is N with one digit raised by d
   without a carry, that digit read from the learned addition table (modx comes before the adder, so no adder is
   needed or available), and the answer is d at that place.

Both checks only add acceptance probes, on settings of their own: the search's probes and the original acceptance
probes are built exactly as before, drawing the same random machine starts, so the climb is the original's wherever
the original's checks were right.

Also writes fix/no_step_fix.py, the no_step arm (make_variants.py's edits) on the fixed s3i: the run the weak
reduction check broke (no_step on seed 74), again with the fix.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/fix/make_fix.py [--diff NAME]
"""

from __future__ import annotations

import difflib
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTEG = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import make_variants as MV  # noqa: E402  (the ablation arms' edits)

# the conditional subtraction's extra probes, inserted before build() (s3i) and Draw (sp2)
ABOVE_PROBES = '''def above_probes(Nt, st, gen, device):
    """Probes with N < X < 2N for the conditional subtraction's check (fix): X is N with its digit at a place j
    raised by d, read from a single entry of the learned addition table with no carry (it comes before any adder, so
    none is needed), and the answer X - N is d at place j. ok marks the probes where such a d was drawn (one try
    each)."""
    X, A = Nt.clone(), torch.zeros_like(Nt)
    ok = torch.zeros(Nt.shape[0], dtype=torch.bool, device=Nt.device)
    for i in range(Nt.shape[0]):
        nz = torch.nonzero(Nt[i]).flatten()
        if nz.numel() == 0:
            continue
        top = int(nz.max())
        j = int(torch.randint(0, top + 1, (1,), generator=gen, device=device))
        d = int(torch.randint(1, st.B, (1,), generator=gen, device=device))
        a = int(Nt[i, j])
        if int(st.cells.adc_c[a, d, 0]) or (j == top and d >= a):      # a carry, or X would reach 2N
            continue
        X[i, j], A[i, j] = st.cells.adc_d[a, d, 0], d
        ok[i] = True
    return X, A, ok


'''
ABOVE_S3I = ABOVE_PROBES

S3I_EDITS = [
    ("ROUND_RETRY = 10       # once the climb is done with no round",
     "FULL_CHECK = (\"reduce\",)  # fix: tasks also checked at the round's size (W digits): there the probes' quotients\n"
     "                          # reach the round's, so no loop of subtractions within the search's reach passes\n"
     "ABOVE_CHECK = (\"modx\",)   # fix: tasks also checked with N < X < 2N (above_probes): the conditional "
     "subtraction's\n"
     "                          # own probes are x < N and x = N only\n"
     "ROUND_RETRY = 10       # once the climb is done with no round"),
    ("def build(task, st, n, gen, rows, helpers, lib, device):\n",
     ABOVE_S3I + "def build(task, st, n, gen, rows, helpers, lib, device):\n"),
    ('    if base == "modx":\n        kind = torch.arange(n, device=device) % 3\n',
     '    if base == "modx":\n'
     '        if getattr(st, "above", False):                      # fix: N < X < 2N (above_probes)\n'
     '            Xa, Aa, ok = above_probes(Nt, st, gen, device)\n'
     '            return ({reg("X"): torch.where(ok[:, None], Xa, xt), "N": Nt},\n'
     '                    {"A": torch.where(ok[:, None], Aa, torch.where(eq[:, None], zero, xt))}, None)\n'
     '        kind = torch.arange(n, device=device) % 3\n'),
    ('''                D[more] = run("add", {"X": D[more], "N": Nv[more]})[0]
        return D
    if base == "lt":''',
     '''                D[more] = run("add", {"X": D[more], "N": Nv[more]})[0]
        return D

    def times_add(D, Nv, times):              # fix: the same sum by doubling (quotients up to 10^W at full size)
        D, M, t = D.clone(), Nv.clone(), times.clone()
        while bool((t > 0).any()):
            odd = (t % 2) == 1
            if bool(odd.any()):
                D[odd] = run("add", {"X": D[odd], "N": M[odd]})[0]
            t = t // 2
            more = t > 0
            if bool(more.any()):
                M[more] = run("add", {"X": M[more], "N": M[more]})[0]
        return D
    if base == "lt":'''),
    ('        return ({reg("X"): repeat_add(xt, Nt, count_of(other)), "N": Nt},',
     '        add_q = times_add if st.W == W else repeat_add       # fix: at the round\'s size by doubling (q up to 10^W)\n'
     '        return ({reg("X"): add_q(xt, Nt, count_of(other)), "N": Nt},'),
    ("        self.verify = [Setting(B, W, cells) for B, W in verify]\n",
     "        self.verify = [Setting(B, W, cells) for B, W in verify]\n"
     "        self.full = [Setting(10, W, cells)]                     # fix: the round's size, for FULL_CHECK\n"
     "        self.above = [Setting(B, W, cells) for B, W in verify]  # fix: the verify sizes with N < X < 2N, for\n"
     "        for st in self.above:                                   # ABOVE_CHECK\n"
     "            st.above = True\n"),
    ("            ok = Acceptor(task, self.lib, helpers, rows, self.verify, self.device)",
     "            ok = Acceptor(task, self.lib, helpers, rows, self.verify_for(task), self.device)"),
    ("                    cache[k] = accept(task, p, self.lib, helpers, rows, self.verify, self.device)",
     "                    cache[k] = accept(task, p, self.lib, helpers, rows, self.verify_for(task), self.device)"),
    ("    def routines(self):\n        return tuple(sorted(k for k in self.lib if k.startswith(\"T.\")))\n",
     "    def routines(self):\n        return tuple(sorted(k for k in self.lib if k.startswith(\"T.\")))\n\n"
     "    def verify_for(self, task):\n"
     "        \"\"\"A task's acceptance sizes (fix): the verify settings, the round's own size for FULL_CHECK, and the verify\n"
     "        sizes again with N < X < 2N for ABOVE_CHECK.\"\"\"\n"
     "        base = task.rstrip(\"~\")\n"
     "        return (self.verify + (self.full if base in FULL_CHECK else [])\n"
     "                + (self.above if base in ABOVE_CHECK else []))\n"),
]

# sp2 (integration/self_practice/): the reduction is the law red on the pattern x+on (red(a + Q N) = a, Q another row's
# x), posed in every binding by SelfClimb; its round calls T.red.x+on. The same two changes, on that law and pattern.
SP2_EDITS = [
    ("ROUND_RETRY = 10       # once the climb is done with no round, the round is searched again every this many steps\n",
     "ROUND_RETRY = 10       # once the climb is done with no round, the round is searched again every this many steps\n"
     "FULL_CHECK = ((\"red\", \"x+on\"),)  # fix: laws on patterns whose acceptance also runs at the round's size (W "
     "digits):\n"
     "                                 # their quotients reach the round's, so no loop of subtractions in reach passes\n"
     "ABOVE_CHECK = ((\"red\", \"pair\"),)  # fix: laws on patterns also checked with N < a < 2N (above_probes): red's\n"
     "                                 # own probes on pair are a < N and a = N only\n"),
    ("class Draw:\n", ABOVE_PROBES + "class Draw:\n"),
    ("        self.st, self.gen, self.device, self.pattern = st, gen, device, pattern\n        self.cache = {}\n",
     "        self.st, self.gen, self.device, self.pattern = st, gen, device, pattern\n        self.cache = {}\n"
     "        self.above = getattr(st, \"above\", False) and pattern == \"pair\"     # fix: N < a < 2N (above_probes)\n"
     "        if self.above:\n"
     "            self.above_x, self.above_a, self.above_ok = above_probes(self.N, st, gen, device)\n"),
    ("    def operands(self, ops, lib):\n        \"\"\"The operands' digits [n, P] per operand position, each probe of its kind.\"\"\"\n",
     "    def operands(self, ops, lib):\n        \"\"\"The operands' digits [n, P] per operand position, each probe of its kind.\"\"\"\n"
     "        if self.above:                                          # fix: (N + d B^j, N)\n"
     "            return [self.above_x, self.N]\n"),
    ("    the law says nothing on this draw (or its answers do not vary).\"\"\"\n",
     "    the law says nothing on this draw (or its answers do not vary).\"\"\"\n"
     "    if d.above:                                                 # fix: red(N + d B^j) = d B^j, where one was drawn\n"
     "        return {\"A\": d.above_a, \"Am\": d.above_ok} if law == \"red\" else None\n"),
    ('''            D[more] = ops.run("add", "pair", st, lib, [D[more], v[more]], gen, device)[0]
    return D
''',
     '''            D[more] = ops.run("add", "pair", st, lib, [D[more], v[more]], gen, device)[0]
    return D


def times_add(ops, st, lib, D, v, times, gen, device):
    """The same sum as repeat_add by doubling, O(log times) additions (fix: quotients up to 10^W at full size)."""
    D, M, t = D.clone(), v.clone(), times.clone()
    while bool((t > 0).any()):
        odd = torch.nonzero(t % 2 == 1).flatten()
        if odd.numel():
            D[odd] = ops.run("add", "pair", st, lib, [D[odd], M[odd]], gen, device)[0]
        t = t // 2
        more = torch.nonzero(t > 0).flatten()
        if more.numel():
            M[more] = ops.run("add", "pair", st, lib, [M[more], M[more]], gen, device)[0]
    return D
'''),
    ("            v = repeat_add(ops, st, lib, self.x, self.N, count_of(self.o), gen, dev)",
     "            add_q = times_add if st.W == W else repeat_add      # fix: at the round's size by doubling\n"
     "            v = add_q(ops, st, lib, self.x, self.N, count_of(self.o), gen, dev)"),
    ("        self.chunk, self.fuse, self.search_s, self.seed = chunk, fuse, search_s, seed\n",
     "        self.chunk, self.fuse, self.search_s, self.seed = chunk, fuse, search_s, seed\n"
     "        self.full = [Setting(10, W, cells)]                     # fix: the round's size, for FULL_CHECK\n"
     "        self.above = [Setting(B, W, cells) for B, W in verify]  # fix: the verify sizes with N < a < 2N, for\n"
     "        for st in self.above:                                   # ABOVE_CHECK\n"
     "            st.above = True\n"),
    ("Acceptor(g.name, root, self.verify, rows, self.ops, self.lib, self.device))",
     "Acceptor(g.name, root,\n"
     "                             self.verify + (self.full if (g.name, root.pattern) in FULL_CHECK else [])\n"
     "                             + (self.above if (g.name, root.pattern) in ABOVE_CHECK else []),\n"
     "                             rows, self.ops, self.lib, self.device))"),
]

FIXES = {   # name: (original, its md5, edits, the header's description)
    "submission_stage3_s3i_fix": (INTEG / "submission_stage3_s3i.py", "0e8131440e1842546c67e7378caf472d", S3I_EDITS,
                                  "the reduction's acceptance also checked at the round's size (8 digits), its "
                                  "probes built by doubling, and the conditional subtraction's also with "
                                  "N < x < 2N"),
    "submission_stage3_sp2_fix": (INTEG / "self_practice" / "submission_stage3_sp2.py",
                                  "afc70fe91069a85c416eba18b48ee30e", SP2_EDITS,
                                  "the reduction law's (red on x+on) acceptance also checked at the round's size "
                                  "(8 digits), its probes built by doubling, and the conditional subtraction's "
                                  "(red on pair) also with N < a < 2N"),
    "no_step_fix": (HERE / "submission_stage3_s3i_fix.py", "09a8263cf09321fef9ba3d4ffbc2aea6", MV.ARMS["no_step"][1],
                    f"no_step's edits (make_variants.py): {MV.ARMS['no_step'][0]}"),     # built after the fixed s3i
}


def build(name: str) -> tuple[str, str]:
    path, md5, edits, what = FIXES[name]
    src = path.read_text()
    got = hashlib.md5(src.encode()).hexdigest()
    if got != md5:
        raise SystemExit(f"{path} md5 {got} != {md5}: the original changed")
    out = f"# {name} (integration/ABLATION_S3.md §8): {path.name} (md5 {md5[:8]}) with {what}.\n" + src
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{name}: an edit's old text occurs {n} times, not once: {old[:80]!r}")
        out = out.replace(old, new)
    compile(out, f"{name}.py", "exec")
    return src, out


def main(argv):
    if argv[:1] == ["--diff"]:
        src, out = build(argv[1])
        sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                                   FIXES[argv[1]][0].name, f"fix/{argv[1]}.py", n=1))
        return
    for name in FIXES:
        _, out = build(name)
        path = HERE / f"{name}.py"
        if path.exists() and path.read_text() != out and "--force" not in argv:
            raise SystemExit(f"{path} differs from this build (pass --force to replace it)")
        path.write_text(out)
        print(f"{name}  {hashlib.md5(out.encode()).hexdigest()}  {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
