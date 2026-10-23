"""Builds integration/submission_stage1.py (INTEGRATION.md, stage 1: the general lessons + the tournament) from three
certified files, which it never modifies:

    tournament/submission_tournament.py      (ours: md5 7ae547d9...; the MLP cells, the chain at free attention's
                                              width, the division's wiring as tournaments, the link and loop laws,
                                              the moved-up rows)
    general_lessons/submission_general.py    (theirs: md5 ddfdc065...; the law library, the miner, the answer-chosen
                                              lesson sets, the shared round menu)
    wider_menus/submission_wide_anchored.py  (their common ancestor: md5 c82b11fa...)

Step 1 is `git merge-file` of the two over their ancestor (deterministic for these inputs: 14 conflicts). Step 2
resolves each conflict (RESOLVE, in order: "A" ours, "B" theirs, or new text). Step 3 replaces whole methods where the
two designs meet (the per-set tournament: chain_lesson, step_lesson, io_lesson) and patches a few exact lines (each
must occur once). So the file's diff from either parent is this script.

    ~/github/sotaku/venv/bin/python integration/make_stage1.py
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OURS = ROOT / "tournament" / "submission_tournament.py"
THEIRS = ROOT / "general_lessons" / "submission_general.py"
BASE = ROOT / "wider_menus" / "submission_wide_anchored.py"
OUT = HERE / "submission_stage1.py"
MD5 = {OURS: "7ae547d9799f24eee81bc5862d2411cc", THEIRS: "ddfdc065329a9059acac48160f59a65f",
       BASE: "c82b11fa8a93b09c0724c3931cf8de98"}


def sub1(src: str, old: str, new: str) -> str:
    n = src.count(old)
    if n != 1:
        raise ValueError(f"expected one occurrence, found {n}: {old[:100]!r}")
    return src.replace(old, new)


def replace_def(src: str, name: str, new: str, indent: str = "    ") -> str:
    """Replace the whole function `name` (at `indent`) up to the next definition at the same level."""
    m = re.search(rf"^{indent}def {name}\(", src, flags=re.M)
    if not m:
        raise ValueError(name)
    if len(re.findall(rf"^{indent}def {name}\(", src, flags=re.M)) != 1:
        raise ValueError(f"{name} defined more than once")
    nxt = re.search(rf"^{indent}(def |@|class )|^(def |class |[A-Z_]+ = )", src[m.end():], flags=re.M)
    end = m.end() + nxt.start() if nxt else len(src)
    return src[:m.start()] + new.rstrip("\n") + "\n\n" + ("\n" if indent == "" else "") + src[end:]


def merge3() -> str:
    for f, h in MD5.items():
        assert hashlib.md5(f.read_bytes()).hexdigest() == h, f"a certified file changed: {f}"
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for name, f in (("ours.py", OURS), ("base.py", BASE), ("theirs.py", THEIRS)):
            p = Path(td) / name
            p.write_bytes(f.read_bytes())
            paths.append(str(p))
        r = subprocess.run(["git", "merge-file", "-p", *paths], capture_output=True, text=True)
    assert r.returncode == 14, f"expected 14 conflicts, git merge-file says {r.returncode}"
    return r.stdout


def resolve(text: str, resolutions: list) -> str:
    """Replace the conflicts, in order, by "A" (ours), "B" (theirs) or a given text."""
    pat = re.compile(r"^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n", re.S | re.M)
    out, pos, i = [], 0, 0
    for m in pat.finditer(text):
        out.append(text[pos:m.start()])
        r = resolutions[i]
        out.append(m.group(1) if r == "A" else m.group(2) if r == "B" else r(m.group(1), m.group(2)))
        pos, i = m.end(), i + 1
    assert i == len(resolutions), (i, len(resolutions))
    out.append(text[pos:])
    return "".join(out)


DOC = '''"""Integration, stage 1 (cpu_first/integration/INTEGRATION.md): the general lessons and the tournament in one file.
It is tournament/submission_tournament.py (TOURNAMENT.md: the MLP cells, the chain at free attention's width, the
division's wiring learned as tournaments) with general_lessons/submission_general.py's lessons (LESSONS.md: the law
library, the mined facts, the answer-chosen lesson sets and the shared round menu) in place of its hand-picked ones:
  * every lesson set (PREDICATES: "below", long division's ladder; "lowzero", Montgomery's) owns a whole copy of the
    division's tournament blocks ("link", "gate", "step", "read"), taught only by its own lessons. The set's link law
    and its ladder come from the set's predicate (red.link, red.step): below, one link from x signals "below N" and
    takes N to 0, and from x + N q the chain gives x; lowzero, one link from 10 x and from N adds N, and from
    10 x + N q the chain gives 10 x or 10 (x + N). The pairs take only the gate table's settings that pass the loop
    law and the gated subtraction's law (csub), both laws of the library now; the moved-up rows (CHAIN_NORM) are
    below's only (a moved N is not coprime to 10, so it is no instance of lowzero's predicate);
  * every block of every set is a tournament: its lessons run only its active combinations (within PRUNE nats of
    the block's best; the others count as failing), the step and reading blocks included, which keeps the second
    set's lessons affordable;
  * the selector over the sets and the round menu are trained by the answers alone, as in the general file; the
    prediction path is straight-through over the sets, each set's switches its blocks' argmax combinations;
  * the table's lessons are the general file's (mined order and nonzero facts, the laws on every pair).
The two parents' docstrings follow.

'''


def doc(a: str, b: str) -> str:
    return DOC + a.lstrip('"') + "\n" + b.lstrip('"')


CONSTS = '''                       # two final subtractions | 5 (TOURNAMENT.md) level 4 with the chain at free attention's width:
                       # each operand any source at any place offset -3..3, the subtraction's carry route, the gate's
                       # flag the borrow out of any place, and what a link writes on commit and on hold any of the link's
                       # values at any offset
IO_MODE = "local"      # how the I/O block is taught: "local" (the step's switches bring and post by the step law on the
                       # division ladder's registers; the reading's switches by the whole-division laws and the answers;
                       # the only mode of this file)
LR_WIRE = 0.5          # Adam on the tournament blocks' logits (one logit per whole combination of a block)
PRUNE = 12.0           # a block's combination is active (run, scored) while its logit is within PRUNE nats of the
                       # block's best; the others count as failing (each carries less than e^-PRUNE of the best's weight)
PAIR_CAP = 64          # the pairs' lesson takes at most this many link combinations (the best of those passing the link
                       # law)
STAGE_REGS = 4         # a combination's verdict: on this many of the lesson's registers first, then on all of them for
                       # the combinations that pass those (the same verdict, less work)
CHUNK = 1 << 19        # combinations x registers run at once
LOOP_LAW = True        # the chain's pairs only with gate settings that pass the loop law (chain_lesson): a count of at
                       # least 1 runs its round, a count of 0 stops the loop
CHAIN_NORM = True      # the chain's lessons (the link law, the division ladder, and through its registers the step law)
                       # also run on each row's x and N moved up until N's top digit is at the N register's top place
                       # (the below set's; a moved N is no instance of lowzero's predicate): no training modulus has 8
                       # digits, so without them no register ever reaches the top place, and free-width wirings that
                       # differ only there (an operand moved up one place and back, the borrow out of place 7 read as
                       # the final one) pass every lesson and fail on the unseen 8-digit moduli
'''


CLASSIFY = '''    key = tuple(names[k] for k in ("a", "b", "op", "cin", "flag", "keep", "cv", "hv"))
    plain, plus = ("sub", "j-1"), ("add", "j-1")
    wr = (("E", 0), ("D", 0))
    chain = {(("D", 0), ("M", 0), *plain, "fin", "gated", *wr): "repeated subtraction",
             (("D", 0), ("M", 0), *plain, "lt", "gated", *wr): "repeated subtraction",   # D < N: D - N's borrow
             (("D", 0), ("M", 0), *plain, "fin", "gated", ("E", 0), ("A", 0)): "repeated subtraction",  # A is D
             (("D", 0), ("M", 0), *plain, "lt", "gated", ("E", 0), ("A", 0)): "repeated subtraction",
             (("Rp", 0), ("Mq", 0), *plain, "fin", "gated", *wr): "trial multiples",
             (("D", 0), ("M", 0), *plus, "zD", "gated", *wr): "montgomery, repeated addition",
             (("M", 0), ("D", 0), *plus, "zD", "gated", *wr): "montgomery, repeated addition",
             (("Rp", 0), ("Mq", 0), *plus, "zE", "gated", *wr): "montgomery, trial multiples (the sum's digit tested)",
             (("Rp", 0), ("Mq", 0), *plus, "zD", "gated", *wr):
                 "montgomery, trial multiples (the register's digit tested)",
             (("Mq", 0), ("Rp", 0), *plus, "zE", "gated", *wr): "montgomery, trial multiples (the sum's digit tested)",
             (("Mq", 0), ("Rp", 0), *plus, "zD", "gated", *wr):
                 "montgomery, trial multiples (the register's digit tested)"}.get(key, "other")
'''


PARAMS = '''        self.blks = nn.ModuleList([nn.ParameterDict({n: nn.Parameter(INIT * torch.randn(menu_size(DIV_MENU, b)))
                                                     for n, b in BLOCKS.items()}) for _ in PREDICATES])
                                                                            # per lesson set: the division's tournament
                                                                            # blocks, one logit per whole combination
        self.lsel = nn.Parameter(INIT * torch.randn(len(PREDICATES)))       # the lesson sets' selector
        self.rnd = nn.ParameterDict({k: nn.Parameter(INIT * torch.randn(len(ROUND_MENU[k])))
                                     for k in ROUND_SW})                    # the shared round menu
'''


DESIGNED = '''        for c, pred in enumerate(PREDICATES):
            for n, b in BLOCKS.items():
                true = torch.tensor([[DIV_MENU[k].index(SET_TRUE[pred][k]) for k in b]])
                self.blks[c][n].copy_(hot(block_encode(n, true)[0], self.blks[c][n].shape[0]))
        self.lsel.copy_(hot(torch.tensor(PREDICATES.index(DESIGNED_SIGMA)), len(PREDICATES)))
        for k in ROUND_SW:
            self.rnd[k].copy_(hot(torch.tensor(ROUND_MENU[k].index(DESIGNED_ROUND[k])), len(ROUND_MENU[k])))
'''


CHOICE = '''    def sigma(self) -> int:
        """The selected lesson set: the selector's argmax."""
        return int(self.lsel.argmax())

    def bd(self, c: int | None = None) -> nn.ParameterDict:
        """Lesson set c's tournament blocks (default: the selected set's)."""
        return self.blks[self.sigma() if c is None else c]

    def choice(self, k: str, c: int | None = None) -> int:
        """The argmax option of lesson set c's division switch k (0: a single fixed option): its option in its block's
        argmax combination."""
        if k not in SWITCHED:
            return 0
        n = BLOCK_OF[k]
        return int(block_decode(n, self.bd(c)[n].argmax()[None])[0, BLOCKS[n].index(k)])

    def active(self, n: str, c: int | None = None) -> Tensor:
        """Lesson set c's block n's active combinations (flat indices): within PRUNE nats of its best logit."""
        th = self.bd(c)[n].detach()
        return torch.nonzero(th >= th.max() - PRUNE).flatten()

    def block_logw(self, n: str, c: int | None = None) -> Tensor:
        return self.bd(c)[n].float().log_softmax(-1)
'''


STE = '''        for n, b in BLOCKS.items():
            # the block's argmax combination as a straight-through one-hot over every combination (its value: exactly
            # that combination; its gradient: the block's softmax), seen by each switch through its marginal
            pw = self.bd(c)[n].float().softmax(-1)
            if not torch.is_grad_enabled():
                for k in b:
                    out[k] = F.one_hot(torch.tensor(self.choice(k, c), device=pw.device), len(DIV_MENU[k])).float()
                continue
            st = ste(F.one_hot(pw.argmax(), pw.shape[0]).float(), pw)
            combos = block_decode(n, torch.arange(pw.shape[0], device=pw.device))
            for i, k in enumerate(b):
                out[k] = torch.zeros(len(DIV_MENU[k]), device=pw.device).index_add(0, combos[:, i], st)
        out.update({k: hot(v) for k, v in self.rnd.items()})
        return out

    def wiring(self, X: int, device, c: int | None = None) -> dict:
        """Lesson set c's argmax wiring (default: the selected set's), per row [X]: every switch's option index, the
        gate's two rows (commit on flag 0 / 1) and the round's options."""
        w = {k: torch.full((X,), self.choice(k, c), dtype=torch.long, device=device) for k in SWITCHED}
        w.update({k: torch.full((X,), int(v.argmax()), dtype=torch.long, device=device) for k, v in self.rnd.items()})
'''


COMBO = '''    def combo_logw(self, names: tuple, combos: Tensor, gate: bool, c: int | None = None) -> Tensor:
        """log of the mixture's weight of every combination combos [K, len(names) (+ 2 gate rows)] of lesson set c's
        switches: names is one or more whole blocks; a combination's weight is the product of its blocks' combinations'
        weights (each block one logit per whole combination) and of the gate's rows."""
        lw = 0
        for n, b in BLOCKS.items():
            if b[0] in names:
                cols = [names.index(k) for k in b]
                lw = lw + self.block_logw(n, c)[block_encode(n, combos[:, cols])]
'''


LESSONS_DIV = '''        # the division's wiring, per lesson set (kind 3): the chain's tournament blocks and the gate by the set's link
        # law and ladder (phase C), then its step law and its reading's whole-reduction laws (and the answers)
        admissible = [PREDICATES[c] != "lowzero" or (usable and ctl.miner.holds("N mod 2 != 0")
                                                     and ctl.miner.holds("N mod 5 != 0")) for c in range(len(PREDICATES))]
        chain_on = ctl.phase == "C"
        per, io_parts = [], []
        for c in range(len(PREDICATES)):
            if not admissible[c]:
                per.append(None)
                continue
            div_loss, div_ok, regs = self.chain_lesson(c, Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, chain_on, Cd)
            if chain_on and div_loss is not None:
                loss = loss + div_loss
            law_ok, step_ok = 1.0, 1.0
            for name, names in IO_BLOCKS:
                if name == "step":
                    step_loss, step_ok = self.step_lesson(c, regs, tabs, ctl.wire_ios[c], names)
                    if ctl.wire_ios[c] and step_loss is not None:
                        loss = loss + step_loss
                else:
                    io, law_ok = self.io_lesson(c, Xd, Nd, Cd, small, tabs, ctl.wire_ios[c], names, ident)
                    if ctl.wire_ios[c]:
                        io_parts.append(io)
            per.append({"div_ok": div_ok, "law_ok": law_ok, "step_ok": step_ok})
        if io_parts:
'''


LOGLINE = '''                  f"wiring: {self.wiring_names()} {ctl.t_log} | sets={','.join(PREDICATES)} "
                  f"selector={','.join(f'{float(p):.3f}' for p in self.lsel.float().softmax(-1).tolist())} "
                  f"fit={','.join(f'{v:.3f}' for v in ctl.sel_fit)} round: {self.round_names()} "
                  f"per_set={[None if p is None else {k: round(v, 3) for k, v in p.items()} for p in aux.get('per') or []]} "
                  f"tournament: {ctl.chain_info} active="
                  f"{'/'.join(','.join(f'{n}={int(self.active(n, c).numel())}' for n in BLOCKS) for c in range(len(PREDICATES)))} "
                  f"facts={ctl.miner.describe()}",
'''


PROBS = '''        print("WIRING probabilities " + " ".join(f"{PREDICATES[c]}.{n}={float(v.float().softmax(-1).max()):.4f}"
                                                 f"/active={int(self.active(n, c).numel())}"
                                                 for c in range(len(PREDICATES)) for n, v in self.blks[c].items())
'''


RESOLVE = [
    doc,                                   # 1 the docstring
    lambda a, b: CONSTS,                   # 2 the tournament's constants (IO_MODE "local" only)
    lambda a, b: CLASSIFY,                 # 3 classify: level 5's keys, with the Montgomery forms
    lambda a, b: PARAMS,                   # 4 the parameters: per set, the blocks; the selector; the round
    lambda a, b: DESIGNED,                 # 5 designed(): per set, its blocks at the set's truth
    lambda a, b: CHOICE,                   # 6 choice, active, block_logw: per set
    lambda a, b: STE,                      # 7 ste_switches (per set's blocks) and wiring
    lambda a, b: COMBO,                    # 8 combo_logw: per set's blocks
    "B",                                   # 9  chain_lesson (replaced whole below)
    "B",                                   # 10
    "B",                                   # 11
    lambda a, b: LESSONS_DIV,              # 12 lessons(): per set, with the counts for the loop law
    lambda a, b: LOGLINE,                  # 13 the log line: both
    lambda a, b: PROBS,                    # 14 the report: per set's blocks
]


CHAIN_LESSON = '''    def chain_lesson(self, c: int, Xd: Tensor, Nd: Tensor, rowsN: Tensor, verified: Tensor, tabs, train: bool,
                     Cd: Tensor | None = None):
        """The chain's two tournament blocks of lesson set c (TOURNAMENT.md), on the batch's first CHAIN_ROWS rows whose
        N rungs are all verified. The registers and targets come from the set's predicate (red.link, red.step):
          below    the link law: one link from x (x < N) signals that the register is below N (its final borrow, or
                   carry), and from N it gives 0 without one; the ladder: from x + N q (q = 0..9) the chain moves the
                   register down by multiples of N until it is below N: it gives x;
          lowzero  the link law: one link from 10 x and from N gives the register plus N (the learned adder) without a
                   carry out; the ladder: from 10 x + N q the chain moves the register up by multiples of N until its
                   lowest digit is 0: it gives 10 x (q = 0) or 10 (x + N);
        the ladder's registers built from the machine's own row N q and its adder.
          * the link law (set c's block "link"): every active combination runs it on the hard machine;
          * the ladder (the pairs): every pair of a link combination that passes the link law (the PAIR_CAP best) and
            an active combination of set c's block "gate", with each setting of the gate table that passes the gate's
            laws, runs one step's chain on the hard machine;
          * the gate table's laws, on the batch's own numbers with the snapped table: the loop law (LOOP_LAW: every
            row's count is at least 1, so its first round must run, the gate committing on the borrow of C - 1; a count
            of 0 must stop the loop, the gate holding on the borrow of 0 - 1) and csub (GATE_FINAL: the gated
            subtraction of N takes x + N to x and holds x). Each pins the gate table, which every set and the
            countdown share, to "commit on no borrow".
        With CHAIN_NORM the below set's lessons also run on the rows moved up until N's top digit is at the N register's
        top place (x 10^l and N 10^l are an instance of below's ladder, so the registers reach the top places, which no
        training modulus does). The lowzero set gets no moved rows: N 10^l is not coprime to 10, so it is not an
        instance of the lowzero predicate (the admissibility rule the miner applies to the data); its ladder keeps the
        top-place gap TOURNAMENT.md describes, which only matters where the answers choose lowzero. A combination passes when it passes on every register; each lesson's
        loss is the mixture's (tourney_nll; when `train`). The sources R and d are the register's upper places and its
        lowest digit; the multiples of N come from the machine's own rows. Returns the loss, whether the argmax chain
        (and gate) passes, and the ladder's registers (for the step law)."""
        Td, Tb = tabs[0], tabs[1]
        dev = Xd.device
        n = min(CHAIN_ROWS, Xd.shape[0])
        pred = PREDICATES[c]
        with torch.no_grad():
            keep = verified[:n].all(1)
            Xs, Ns, rows = Xd[:n][keep], Nd[:n][keep], rowsN[:n][keep]        # rows [m, 10, C]: N * q (row 0)
            m = Xs.shape[0]
            if m == 0:
                return None, 0.0, None
            x0, Mx = F.pad(Xs, (0, P - W)), F.pad(Ns, (0, P - W))
            xN = add_learned(x0, Mx, Td, Tb)[0]                               # x + N
            if pred == "lowzero":
                reg = add_learned(F.pad(Xs, (1, C - W - 1))[:, None].expand(m, 10, C), rows, Td, Tb)[0][..., :P]
                q0 = (torch.arange(10, device=dev) == 0)[None, :, None]
                xq = torch.where(q0, F.pad(Xs, (1, P - W - 1))[:, None], F.pad(xN, (1, 0))[:, None, :P]).reshape(-1, P)
                sq = torch.where(q0, x0[:, None], xN[:, None]).reshape(-1, P)
                r0 = F.pad(Xs, (1, P - W - 1))                                # the link law's first register: 10 x
            else:
                reg = add_learned(F.pad(Xs, (0, C - W))[:, None].expand(m, 10, C), rows, Td, Tb)[0][..., :P]
                xq = sq = x0.repeat_interleave(10, 0)
                r0 = x0                                                       # ... x
            reg = reg.reshape(m * 10, P)
            Nq = Ns.repeat_interleave(10, 0)
            csub = []                                                         # (register, its csub result) pairs
            for R in (xN, x0):
                E, cb, _ = sub_hard(R, Mx, Td, Tb)
                csub.append((R, E, cb))
            norm = CHAIN_NORM and pred != "lowzero"                          # (above)
            if norm:                                                         # + every row moved up
                lead = (Ns.flip(-1).cumsum(-1) == 0).sum(-1)                 # N's leading zero digits [m]
                l10 = lead.repeat_interleave(10)
                r0, Ns = torch.cat([r0, move_digits(r0, lead)]), torch.cat([Ns, move_digits(Ns, lead)])
                reg, Nq, xq, sq = (torch.cat([t, move_digits(t, l10)]) for t in (reg, Nq, xq, sq))
                m = 2 * m
            mq = None
            if HAS_MQ:
                mq = (multiples(Nq, tabs), multiples(Nq.flip(-1), tabs))      # N's multiples, as read and reversed
            regs = {"reg": reg, "Nq": Nq, "xq": xq, "sq": sq, "mq": mq, "pred": pred}
            ca = self.wcodes(m * 10, dev, c)
            Ra = self.chain_run(reg, ca, regs, 1, tabs, self.ctl.gvisits)
            ok_a = bool((Ra == xq).all())
            # the gate table's laws, every setting (g0, g1): commits on flag 0, on flag 1
            gset = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=dev)
            gate_ok = torch.ones(4, dtype=torch.bool, device=dev)
            if LOOP_LAW and Cd is not None:
                one = F.one_hot(torch.zeros_like(Cd[:, 0]), Cd.shape[1])       # the digits of 1
                b_run = sub_hard(Cd, one, Td, Tb)[1]                           # a count of at least 1, minus 1
                b_stop = sub_hard(torch.zeros_like(Cd), one, Td, Tb)[1]        # a count of 0, minus 1
                runs = torch.where(b_run[None] == 0, gset[:, :1], gset[:, 1:]) == 1          # [4, X]
                stops = torch.where(b_stop[None] == 0, gset[:, :1], gset[:, 1:]) == 0
                gate_ok = gate_ok & runs.all(1) & stops.all(1)
            if GATE_FINAL:
                for R, E, cb in csub:
                    cm = torch.where(cb[None] == 0, gset[:, :1], gset[:, 1:]) == 1           # [4, m]
                    gate_ok = gate_ok & (torch.where(cm[..., None], E[None], R[None]) == x0[None]).all(-1).all(1)
            garg = self.g.argmax(-1)
            ok_g = bool(gate_ok[int(garg[0]) * 2 + int(garg[1])])
            div_ok = float(ok_a and ok_g)
            if not train:
                return None, div_ok, regs
            # the link law: registers r0_0, N_0, r0_1, N_1, ... (so that the first stages see both kinds; with
            # CHAIN_NORM each row followed by its moved-up copy)
            perm = (torch.stack([torch.arange(m // 2), torch.arange(m // 2) + m // 2], 1).flatten().to(dev)
                    if norm else torch.arange(m, device=dev))
            r1 = torch.stack([r0[perm], F.pad(Ns[perm], (0, P - W))], 1).reshape(2 * m, P)
            N1 = Ns[perm].repeat_interleave(2, 0)
            if pred == "lowzero":
                t1 = add_learned(r1, F.pad(N1, (0, P - W)), Td, Tb)[0]         # the register plus N
            lt1 = torch.tensor([1, 0], device=dev).repeat(m)                 # [r < N] (below)
            isN = lt1 == 0
            mq1 = None if mq is None else tuple(t[::10, 0][perm].repeat_interleave(2, 0) for t in mq)   # 1 N
            act1 = self.active("link", c)
            combos1 = block_decode("link", act1)
            wbase = self.wiring(2 * m, dev, c)

            def law(idx, ri):
                k, r = idx.numel(), ri.numel()
                wk = codes(combo_wiring({kk: v[ri] for kk, v in wbase.items()}, BLOCKS["link"], combos1[idx], r))
                R = r1[ri].repeat(k, 1)
                Mr = n_register(N1[ri].repeat(k, 1), wk["nread"])
                Mqs = torch.zeros_like(R)
                if mq1 is not None:
                    rev = (wk["nread"] == 1)[:, None]
                    Mqs = torch.where(rev, mq1[1][ri].repeat(k, 1), mq1[0][ri].repeat(k, 1))
                _, E, _, fb = link_hard(1, R, R, Mr, Mqs, F.pad(R[:, 1:], (0, 1)), F.pad(R[:, :1], (0, P - 1)), wk,
                                        Td, Tb)
                if pred == "lowzero":
                    ok = (fb == 0) & (E == t1[ri].repeat(k, 1)).all(1)
                else:
                    ok = (fb == lt1[ri].repeat(k)) & (~isN[ri].repeat(k) | (E == 0).all(1))
                return ok.view(k, r)
            v1 = staged(law, act1.numel(), 2 * m, dev, stages=(2, STAGE_REGS))
            pass1 = act1[v1]
            cand = pass1
            if cand.numel() > PAIR_CAP:
                cand = cand[self.blks[c]["link"].detach()[cand].topk(PAIR_CAP).indices]
            # the pairs, on the ladder's registers (first q = 0 and 9 of the first row, and of its moved-up copy)
            act2 = self.active("gate", c)
            gok = torch.nonzero(gate_ok).flatten()                             # the gate settings left
            K = cand.numel() * act2.numel() * gok.numel()
            first = [0, m * 5 + 9, 9, m * 5] if norm else [0, 9, 1, 5]
            order = torch.cat([torch.tensor(first, device=dev), torch.tensor(
                [i for i in range(m * 10) if i not in first], dtype=torch.long, device=dev)])
            pc = None
            if K:
                pc = torch.cartesian_prod(torch.arange(cand.numel(), device=dev), torch.arange(act2.numel(), device=dev),
                                          gok)
                pc = torch.cat([pc[:, :2], gset[pc[:, 2]]], 1)                 # [K, (link, gate, g0, g1)]
            info = {"link_active": act1.numel(), "link_pass": pass1.numel(), "gate_active": act2.numel(), "pairs": K,
                    "gates": gok.numel()}
            if K:
                c1 = block_decode("link", cand)
                c2 = block_decode("gate", act2)
                wq = self.wiring(m * 10, dev, c)

                def ladder(idx, ri):
                    k, r = idx.numel(), ri.numel()
                    q = pc[idx]
                    comb = torch.cat([c1[q[:, 0]], c2[q[:, 1]], q[:, 2:]], 1)
                    rr = order[ri]
                    wk = codes(combo_wiring({kk: v[rr] for kk, v in wq.items()}, CHAIN_SW, comb, r, gate=True))
                    sub = {"reg": reg[rr], "Nq": Nq[rr], "xq": xq[rr],
                           "mq": None if mq is None else (mq[0][rr], mq[1][rr])}
                    out = self.chain_run(sub["reg"].repeat(k, 1), wk, sub, k, tabs)
                    return (out == sub["xq"].repeat(k, 1)).all(1).view(k, r)
                v2 = staged(ladder, K, m * 10, dev)
                info["pairs_pass"] = int(v2.sum())
            self.ctl.chain_info[pred] = info
        loss = None
        t1_ = tourney_nll(self.block_logw("link", c)[act1], v1)
        if t1_ is not None:
            loss = t1_
        if K:
            lg = self.g.float().log_softmax(-1)
            lw = (self.block_logw("link", c)[cand][pc[:, 0]] + self.block_logw("gate", c)[act2][pc[:, 1]]
                  + lg[0][pc[:, 2]] + lg[1][pc[:, 3]])
            t2 = tourney_nll(lw, v2)
            if t2 is not None:
                loss = t2 if loss is None else loss + t2
        return loss, div_ok, regs
'''


STEP_LESSON = '''    def step_lesson(self, c: int, regs: dict | None, tabs, train: bool, names: tuple):
        """The step law of lesson set c: the division ladder's register split as the set's predicate says (chain_run);
        one whole step (the bring-in, the argmax chain, the post move) must give the set's step target: x (below), x
        or x + N (lowzero: the chain's 10 x or 10 (x + N) moved down one place). Every active combination of set c's
        step block runs on the hard machine (the others count as failing); the loss is the mixture's per register
        (when `train`). Returns the loss and whether the argmax step passes."""
        if regs is None:
            return None, 0.0
        reg, xq = regs["reg"], regs["sq"]
        dev = reg.device
        n = reg.shape[0]
        with torch.no_grad():
            ok = float((self.chain_run(reg, self.wcodes(n, dev, c), regs, 1, tabs, bring=True) == xq).all(1)
                       .float().mean())
            if not train:
                return None, ok
            combos = block_decode("step", self.active("step", c))
            Kc = combos.shape[0]
            wk = codes(combo_wiring(self.wiring(n, dev, c), names, combos, n))
            verdict = (self.chain_run(reg.repeat(Kc, 1), wk, regs, Kc, tabs, bring=True) == xq.repeat(Kc, 1)).all(1)
            verdict = verdict.view(Kc, n)
        return mixture_nll(self.combo_logw(names, combos, gate=False, c=c), verdict).mean(), ok
'''


IO_LESSON = '''    def io_lesson(self, c: int, Xd: Tensor, Nd: Tensor, Cd: Tensor, small: Tensor, tabs, train: bool, names: tuple,
                  ident: list):
        """The reading's switches `names` of lesson set c: the whole-reduction laws at the set's alignment s (0 for
        below, W for lowzero), red.identity red(a 10^s) = a for every mined a < N in `ident`, red.zero red(N 10^s) = 0,
        red.invariant red(x 10^s + Q N) = x (Q another row's x, Q N by the machine's own rows; only with x in ident),
        through the whole division, and the rows with T <= REAL_MAX_T through the whole machine (the round at its
        argmax; their outputs are compared with the answers in token_training_loss). Every active combination of set
        c's reading block runs on the hard machine (the other switches at their argmax; the inactive combinations count
        as failing). Returns the laws' verdicts [K, n], the answer rows' outputs [K, m, W] and their indices, and
        whether the argmax wiring passes the laws."""
        Td, Tb = tabs[0], tabs[1]
        dev = Xd.device
        combos = block_decode("read", self.active("read", c))
        Kc = combos.shape[0]
        n = min(IO_ROWS if Kc <= 4096 else IO_ROWS_ENUM, Xd.shape[0])
        up = (W, 0) if PREDICATES[c] == "lowzero" else (0, W)               # the dividend's alignment
        with torch.no_grad():
            Xs, Ns = Xd[:n], Nd[:n]
            wa = self.wiring(n, dev, c)
            ca = codes(wa)
            mults = None
            if bool(((ca["a"] == 3) | (ca["b"] == 3)).any()):
                mults = multiples(torch.where(ca["nread"][:, None] == 1, Ns.flip(-1), Ns), tabs)
            vals = {"x": Xs, "T": F.pad(Cd[:n], (0, W - K))}
            laws = [(F.pad(vals[a], up), vals[a]) for a in ident]             # red.identity
            if "n" in IO_LAWS:
                laws.append((F.pad(Ns, up), torch.zeros_like(Ns)))            # red.zero
            if "q" in IO_LAWS and "x" in ident:                               # red.invariant: x 10^s + Q N
                Q = Xd.roll(1, 0)[:n]
                rows = row_hard(Ns[:, None, :].expand(n, W, W), Q, torch.arange(W, device=dev)[None].expand(n, W), tabs)
                Dq = F.pad(Xs, (up[0], C - W - up[0]))
                for j in range(W):
                    Dq = add_learned(Dq, rows[:, j], Td, Tb)[0]
                laws.append((Dq[:, :2 * W], Xs))
            ok = torch.ones(n, dtype=torch.bool, device=dev)
            for Dv, r in laws:
                ok = ok & (divide_hard(Dv, Ns, ca, Td, Tb, tabs=tabs, mults=mults) == r).all(1)
            law_ok = float(ok.float().mean()) if laws else 0.0
            if not train:
                return None, law_ok
            wk = codes(combo_wiring(wa, names, combos, n))
            mk = None
            if mults is not None or HAS_MQ:                                 # every row's own nread
                both = [multiples(Ns, tabs), multiples(Ns.flip(-1), tabs)]
                mk = torch.where((wk["nread"] == 1)[:, None, None], both[1].repeat(Kc, 1, 1), both[0].repeat(Kc, 1, 1))
            verdict = torch.ones(Kc * n, dtype=torch.bool, device=dev)
            for Dv, r in laws:
                verdict = verdict & (divide_hard(Dv.repeat(Kc, 1), Ns.repeat(Kc, 1), wk, Td, Tb, tabs=tabs, mults=mk)
                                     == r.repeat(Kc, 1)).all(1)
            verdict = verdict.view(Kc, n)
            sel_rows = torch.nonzero(small).flatten()
            outs = None
            if sel_rows.numel() and Kc * sel_rows.numel() <= 65536:
                mr = sel_rows.numel()
                wm = codes(combo_wiring(self.wiring(mr, dev, c), names, combos, mr))
                outs = run_hard(Xd[sel_rows].repeat(Kc, 1), Cd[sel_rows].repeat(Kc, 1), Nd[sel_rows].repeat(Kc, 1),
                                tabs, wm, cap=answer_cap(Cd[sel_rows]))[0].view(Kc, mr, W)
        return {"verdict": verdict if laws else None, "outs": outs, "sel": sel_rows, "combos": combos, "names": names,
                "gate": False, "c": c}, law_ok
'''


ANSWER_FNS = '''

def answer_t(Cs: Tensor) -> int:
    """The count T of the digits Cs [n, K] (LSB first) of the first row, as a Python int (the hard machine's round
    bound only)."""
    return sum(int(d) * 10 ** j for j, d in enumerate(Cs[0].tolist())) if Cs.shape[0] else 0


def answer_rows(Cd: Tensor, tcount: Tensor | None = None) -> Tensor:
    """The rows whose answers the lessons read (the selector, the round, the reading's answers, the freeze's check):
    the rows whose T (as read, digits Cd LSB first) is the batch's smallest nonzero T, compared the reader's way (the
    highest place where two counts differ decides), at most ANS_WORK // (T + 1) of them. Nothing about the data's T is
    assumed: on H1's batches that is T = 1 (about 5 rows of 512), on data whose every T is 8 it is T = 8 (general
    lessons v1 read the one-digit T <= 2 only, and so never saw an answer on such data). tcount (T's digit count) is
    not used; it is there for general_lessons/local_check.py's call."""
    keep = (Cd != 0).any(1)
    if not bool(keep.any()):
        return keep
    for j in reversed(range(Cd.shape[1])):                                  # the least digit at each place, top down
        keep = keep & (Cd[:, j] == Cd[keep, j].min())
    t = answer_t(Cd[keep][:1])
    keep[torch.nonzero(keep).flatten()[max(1, ANS_WORK // (t + 1)):]] = False
    return keep


def answer_cap(Cs: Tensor) -> int:
    """The rounds the hard machine needs on the answer rows (their T's digits Cs; they share one T): T + 1."""
    return answer_t(Cs[:1]) + 1
'''


def build() -> str:
    src = resolve(merge3(), RESOLVE)
    for name, code in (("chain_lesson", CHAIN_LESSON), ("step_lesson", STEP_LESSON), ("io_lesson", IO_LESSON)):
        src = replace_def(src, name, code)
    # the Montgomery truth at level 5 (designed mode and the reports)
    src = sub1(src, '''DIV_MONT = {"a": "D", "b": "M", "op": "add", "flag": "zD", "keep": "gated", "nread": "given", "vread": "given",
            "start": "zero", "order": "up", "bring": (0, 0, "add"), "post": -1, "final": "addhigh", "read": (0, "given")}''',
               '''DIV_MONT = {"a": ("D", 0), "b": ("M", 0), "op": "add", "cin": "j-1", "flag": "zD", "keep": "gated", "cv": ("E", 0),
            "hv": ("D", 0), "nread": "given", "vread": "given", "start": "zero", "order": "up", "bring": (0, 0, "add"),
            "post": -1, "final": "addhigh", "read": (0, "given")}''')
    # the laws the tournament added, in the library
    src = sub1(src, '''    "csub": ("the gated subtraction of N keeps a < N and takes a + N to a", "x, x + N"),
}''', '''    "csub": ("the gated subtraction of N keeps a < N and takes a + N to a", "x, x + N"),
    "red.link": ("one link moves its register by N in the set's direction: below, from a < N it signals below (its "
                 "final borrow) and takes N to 0; lowzero, it takes a to a + N without a carry out",
                 "x (below) or 10 x (lowzero), and N; and the rows moved up (CHAIN_NORM)"),
    "loop.count": ("a count of at least 1 runs its round (the gate commits on the borrow of C - 1); a count of 0 stops "
                   "the loop (the gate holds on the borrow of 0 - 1)", "the rows' counts T"),
}''')
    # the schedule's per-set tournament records
    src = sub1(src, "        self.chain_info: dict = {}            # the chain's tournament, this step: active and passing combinations",
               "        self.chain_info: dict = {}            # per lesson set: its chain's tournament, this step (active and\n"
               "                                              # passing combinations)")
    # the reports: per set, the blocks' probabilities
    src = sub1(src, '''                  + " probabilities=" + ",".join(f"{k}:{float(v.float().softmax(-1).max()):.4f}"
                                                 for k, v in self.sws[c].items()), flush=True)''',
               '''                  + " probabilities=" + ",".join(f"{k}:{float(v.float().softmax(-1).max()):.4f}"
                                                 for k, v in self.blks[c].items()), flush=True)''')
    # the optimizer: every set's blocks at LR_WIRE
    src = sub1(src, '''    wire = [p for n, p in model.named_parameters() if n.startswith("blk.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("sub_cell.", "prod_cell.", "blk."))]''',
               '''    wire = [p for n, p in model.named_parameters() if n.startswith("blks.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("sub_cell.", "prod_cell.", "blks."))]''')
    # the answer rows: the batch's smallest nonzero T, read off the data (no H1-shaped "T <= 2")
    src = sub1(src, "REAL_MAX_T = 2         # the I/O lesson uses the answers of the batch's rows with T <= REAL_MAX_T (one digit)",
               "ANS_WORK = 96          # the answers used: the rows with the batch's smallest nonzero T (answer_rows), at most\n"
               "                       # ANS_WORK // (T + 1) of them (rows x rounds of the hard machine's work)")
    src = sub1(src, "def menu_size(menu: dict, names=None) -> int:",
               ANSWER_FNS.lstrip("\n") + "\n\ndef menu_size(menu: dict, names=None) -> int:")
    src = sub1(src, "            small = (tcount == 1) & (Cd[:, 0] >= 1) & (Cd[:, 0] <= REAL_MAX_T)  # the rows whose answers are used",
               "            small = answer_rows(Cd)                                         # the rows whose answers are used")
    src = sub1(src, 'aux["tabs"], codes(wk), cap=REAL_MAX_T + 1)[0]', 'aux["tabs"], codes(wk), cap=answer_cap(aux["Cd"][sel]))[0]')
    src = sub1(src, "model.wcodes(sel.numel(), sel.device), cap=REAL_MAX_T + 1)[0]",
               'model.wcodes(sel.numel(), sel.device), cap=answer_cap(aux["Cd"][sel]))[0]')
    src = sub1(src, "        self.ans_ok = 0.0                     # the argmax machine on the rows' answers (T <= REAL_MAX_T), this step",
               "        self.ans_ok = 0.0                     # the argmax machine on the answer rows' answers, this step")
    for a, b in (("the rows with T <= REAL_MAX_T", "the answer rows (answer_rows)"),
                 ("the rows with\n        T <= REAL_MAX_T", "the answer\n        rows (answer_rows)"),
                 ("with T <= REAL_MAX_T", "of the answer rows (answer_rows)"),
                 ("T <= REAL_MAX_T", "the answer rows")):
        src = src.replace(a, b)
    for old in ("self.sw[", "self.sw.", "self.sws", "self.blk[", "self.blk.", "self.swd(", "CHAIN_COMBOS", "ctl.wire_io ",
                "REAL_MAX_T"):
        assert old not in src, f"an old use is left: {old}"
    return src


def main() -> None:
    src = build()
    OUT.write_text(src)
    print(f"wrote {OUT} ({len(src.splitlines())} lines, {len(src.encode())} bytes, "
          f"md5 {hashlib.md5(src.encode()).hexdigest()})")
    assert len(src.encode()) <= 256 * 1024, "over the 256 KiB limit"


if __name__ == "__main__":
    sys.exit(main())
