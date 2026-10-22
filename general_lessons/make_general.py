"""Builds general_lessons/submission_general.py from the certified wider_menus/submission_wide_anchored.py (md5
c82b11fa...), which it never modifies. Every change is one replacement below (an exact text that must occur once, or a
whole function replaced by name), so the diff from the best model is this file.

    ~/github/sotaku/venv/bin/python general_lessons/make_general.py
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "wider_menus" / "submission_wide_anchored.py"
OUT = HERE / "submission_general.py"
BASE_MD5 = "c82b11fa8a93b09c0724c3931cf8de98"


def sub1(src: str, old: str, new: str) -> str:
    n = src.count(old)
    if n != 1:
        raise ValueError(f"expected one occurrence, found {n}: {old[:80]!r}")
    return src.replace(old, new)


def replace_def(src: str, name: str, new: str, indent: str = "    ") -> str:
    """Replace the whole function `name` (at `indent`) up to the next definition at the same level."""
    m = re.search(rf"^{indent}def {name}\(", src, flags=re.M)
    if not m:
        raise ValueError(name)
    nxt = re.search(rf"^{indent}(def |@|class )|^(def |class |[A-Z_]+ = )", src[m.end():], flags=re.M)
    end = m.end() + nxt.start() if nxt else len(src)
    return src[:m.start()] + new.rstrip("\n") + "\n\n" + ("\n" if indent == "" else "") + src[end:]


DOC = '''"""GeneralLessons (cpu_first/general_lessons/LESSONS.md): submission_wide_anchored.py with its lessons made general
instead of hand-picked, so that the same file, unchanged, can be run on other tasks of the family:

  * kind 1, a law library (LAWS below): every law is stated once about an operation of the machine (the subtraction
    table, the adder a + b = a - (0 - b), the product cell and the multiply, the reduction, the gated subtraction) and
    applied to the numbers each row gives and the machine computes from them; the table's laws are also applied to a
    register and another row's register (ALL_PAIRS), not only to a register and N;
  * kind 2, mined facts: inside training the model keeps the relations among the fields it reads (N, x, T; and the
    answer y) that hold on every row of the last MINE_WIN batches, from a small grammar: order (a < b), equality
    (a = b, a constant), range (a < 10^k), nonzero, divisibility by 2, 3, 5. Order facts drive the borrow lessons (a < b
    exactly when a - b borrows, on every top part) and the reduction's identity instances (red(a) = a for every mined
    a < N); nonzero facts give "0 - a borrows"; "N mod 2 != 0 and N mod 5 != 0" makes the low-zero reduction
    admissible. Range, equality and the other divisibility facts are reported, never used: they can hold by accident
    (N < 10^7 holds on every H1 training row and fails on its 8-digit moduli);
  * kind 3, answer-chosen lesson sets: the division's step lessons and its whole-reduction laws come from one template,
    instantiated for each predicate of PREDICATES: "below" (the chain moves its register down by multiples of N until
    it is below N: long division's ladder, its step law, red(a) = a, red(N) = 0, red(x + Q N) = x) and "lowzero" (the
    chain moves it up by multiples of N until its lowest digit is 0: from 10 x + q N it gives 10 x or 10 (x + N), the
    step law with the digit added in place, and the same laws with the dividend at alignment W: red(a 10^W) = a,
    red(N 10^W) = 0, red(a 10^W + Q N) = a). Each lesson set has its own division switches, taught by its own lessons
    only. A selector over the sets is trained by the answers alone (a factorized mixture over lesson set and round
    options of "this set's argmax machine reproduces the rows with T <= REAL_MAX_T"), so a wrong lesson set loses
    instead of forcing its machine, and no set can win by having no lessons. Both sets share the law of the gated
    subtraction (csub: it keeps a < N and takes a + N to a), which pins the commit gate's meaning for the countdown too;
  * a shared round menu (ROUND_MENU), the same for every task: the square times 10 or not, plus a one-digit constant,
    and after the reduction times the round's own input and reduced again or not. Only the answers teach it.
The prediction path uses the selected lesson set's switches (straight-through over the sets, so the final logits'
gradient reaches the selector and both sets' switches) and the round's switches. Everything else is
submission_wide_anchored.py's.

WideMenus: the learned-wiring H1 machine (learned_wiring/submission_io3.py, WIRING.md) with its wiring chosen from'''


CONSTANTS = '''FREEZE_WIN = 200       # phase C: freeze after this many consecutive steps with every check exact
# the general lessons (LESSONS.md)
PREDICATES = ("lowzero", "below")   # the reduction lesson sets: the chain moves its register by multiples of N until
                       # its lowest digit is 0 (adding) | until it is below N (subtracting); each set has its own switches
MINE_WIN = 10          # a mined relation is in force while it has held on every row of this many batches in a row
GATE_FINAL = True      # the chain lessons also check the gate in the gated subtraction's role (csub: a < N held,
                       # a + N taken to a)
ALL_PAIRS = True       # the table's laws also on a register and another row's register, not only against N
ROUND_MENU = {"rmul": ["ten", "one"], "radd": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], "rcube": ["times_x", "none"]}
ROUND_SW = ("rmul", "radd", "rcube")   # the shared round menu: x 10 | + a one-digit constant | x the input, reduced
DESIGNED_SIGMA = "below"               # MODE = "designed" (the plumbing check): the selected lesson set and ...
DESIGNED_ROUND = {"rmul": "one", "radd": 0, "rcube": "none"}   # ... the round's options, set by hand'''


LAWS_BLOCK = '''

# ------------------------------------------------------------------------------------------------ the law library
# Each law is about an operation of the machine, never about a task; the lessons below apply it to operands from the
# pool of numbers a row gives (its fields as read) and the machine computes from them (the P-place registers Rg: x and
# the windows of the machine's own square of x; each register minus N again and again, D; another row's registers).
# Where a law needs a premise (a < b), the premise is a mined fact (Miner), never assumed.
LAWS = {
    "sub.identity": ("a - 0 = a, no borrow", "the registers"),
    "sub.self": ("a - a = 0, no borrow", "the registers"),
    "sub.inverse": ("(a - b) + b = a, the final borrow equal to the carry", "D against N; a register against another "
                    "row's (ALL_PAIRS)"),
    "sub.bijective": ("every row and every column of the cell uses each digit once", "the table"),
    "sub.order": ("a < b exactly when a - b borrows, on every top part; subtracting the same digit keeps the order, "
                  "subtracting more gives less, unless exactly one wraps, which then comes out larger", "every mined "
                  "order fact; the entries the batch's subtractions read"),
    "sub.wrap": ("for a < b below 10^W, (a - b) - b does not borrow in W + 1 places", "every mined order fact"),
    "sub.zero_minus": ("0 - a borrows when a != 0", "every mined nonzero fact"),
    "add.identity": ("0 + a = a", "N"),
    "add.commute": ("a + b = b + a", "a register and N; a register and another row's (ALL_PAIRS)"),
    "mul.zero": ("a 0 = 0", "the product cell; every row of the multiply"),
    "mul.ladder": ("a (b + 1) = a b + a", "the product cell; the rows of X b for X the rows' x and N"),
    "mul.distribute": ("X Y = sum_j X Y_j 10^j", "X X for X the rows' x and N, under every number rule"),
    "red.identity": ("red(a 10^s) = a for a < N", "every mined a < N; s = 0 (below) or W (lowzero)"),
    "red.zero": ("red(N 10^s) = 0", "N"),
    "red.invariant": ("red(a 10^s + Q N) = a for a < N", "x, Q another row's x"),
    "red.step": ("one step of the reduction moves its register by multiples of N until the set's predicate holds",
                 "x + N q (below), 10 x + N q (lowzero), q = 0..9"),
    "csub": ("the gated subtraction of N keeps a < N and takes a + N to a", "x, x + N"),
}
ORDER_TERMS = ("N", "x", "T")


def compare(A: Tensor, B: Tensor) -> Tensor:
    """The reader's comparison of two read numbers (digit labels [X, L], LSB first): -1, 0, 1 per row, decided at the
    highest place where they differ (as the reader's ranking decides x < N)."""
    L = A.shape[1]
    top = torch.where(A != B, torch.arange(L, device=A.device), -1).max(-1).values
    a = A.gather(1, top.clamp(min=0)[:, None])[:, 0]
    b = B.gather(1, top.clamp(min=0)[:, None])[:, 0]
    return torch.where(top < 0, 0, torch.where(a < b, -1, 1))


def relations(terms: dict, pairs: bool = True) -> dict:
    """The miner's candidate relations among read numbers (name -> digits [X, L] of the same width), each a bool per
    row: order, equality, constant across rows, nonzero, range (a < 10^k) and divisibility by 2, 3, 5 (from the digit
    labels: the lowest digit for 2 and 5, the digit sum for 3; the miner's bookkeeping, never a lesson's target)."""
    out = {}
    names = list(terms)
    if pairs:
        for a, b in itertools.permutations(names, 2):
            out[f"{a} < {b}"] = compare(terms[a], terms[b]) < 0
        for a, b in itertools.combinations(names, 2):
            out[f"{a} = {b}"] = compare(terms[a], terms[b]) == 0
    for a in names:
        A = terms[a]
        out[f"{a} constant"] = (A == A[:1]).all(1)
        out[f"{a} != 0"] = (A != 0).any(1)
        for k in range(1, A.shape[1]):
            out[f"{a} < 10^{k}"] = (A[:, k:] == 0).all(1)
        for d, v in ((2, A[:, 0] % 2), (5, A[:, 0] % 5), (3, A.sum(1) % 3)):
            out[f"{a} mod {d} != 0"] = v != 0
            for r in range(d):
                out[f"{a} mod {d} = {r}"] = v == r
    return out


class Miner:
    """Relations kept while they hold on every row of the last MINE_WIN batches (a batch where one fails pauses it);
    `fails` counts the batches where each failed since it was first evaluated."""

    def __init__(self) -> None:
        self.streak: dict = {}
        self.fails: dict = {}
        self.first: dict = {}
        self.announced: set = set()

    def update(self, rels: dict, step: int) -> list:
        new = []
        for k, ok in rels.items():
            good = bool(ok.all())
            self.streak[k] = self.streak.get(k, 0) + 1 if good else 0
            self.fails[k] = self.fails.get(k, 0) + (not good)
            if self.streak[k] >= MINE_WIN and k not in self.announced:
                self.announced.add(k)
                self.first[k] = step
                new.append(k)
        return new

    def holds(self, k: str) -> bool:
        return self.streak.get(k, 0) >= MINE_WIN

    def facts(self) -> list:
        """The relations in force, without those a stronger one implies (a < 10^k when a < 10^(k-1) holds; a mod d != 0
        when a mod d = r holds)."""
        f = {k for k in self.streak if self.holds(k)}
        out = []
        for k in sorted(f):
            if " < 10^" in k:
                a, e = k.split(" < 10^")
                if f"{a} < 10^{int(e) - 1}" in f:
                    continue
            if k.endswith(" != 0") and " mod " in k:
                a, d = k.split(" mod ")[0], k.split(" mod ")[1].split()[0]
                if any(f"{a} mod {d} = {r}" in f for r in range(1, 5)):
                    continue
            out.append(k)
        return out

    def describe(self) -> str:
        return ";".join(f"{k}(since {self.first.get(k, '?')}, fails {self.fails.get(k, 0)})" for k in self.facts())'''


ROUND_FNS = '''

def round_hard(s: Tensor, Nd: Tensor, w: dict, tabs, visits: Tensor | None = None, gvisits: Tensor | None = None,
               pvisits: Tensor | None = None, mults: Tensor | None = None) -> Tensor:
    """One round on digits with the shared round menu (per-row codes rmul, radd, rcube in w): the square of s, times
    10 (moved up one place), plus a one-digit constant (the learned adder), reduced; with rcube the result times s (the
    multiply's second operand), reduced again."""
    Td, Tb = tabs[0], tabs[1]
    V = square_hard(s, tabs, visits, pvisits)
    if bool((w["rmul"] == 1).any()):
        V = torch.where((w["rmul"] == 1)[:, None], move_digits(V, torch.ones_like(V[:, 0])), V)
    if bool((w["radd"] > 0).any()):
        cst = F.one_hot(torch.zeros_like(V[:, 0]), 2 * W) * w["radd"][:, None]
        V = torch.where((w["radd"] > 0)[:, None], add_learned(V, cst, Td, Tb, visits)[0], V)
    R = divide_hard(V, Nd, w, Td, Tb, visits, gvisits, tabs=tabs, mults=mults)
    if bool((w["rcube"] == 1).any()):
        R3 = divide_hard(square_hard(R, tabs, visits, pvisits, Sb=s), Nd, w, Td, Tb, visits, gvisits, tabs=tabs,
                         mults=mults)
        R = torch.where((w["rcube"] == 1)[:, None], R3, R)
    return R

'''


ROUND_STE = '''

def round_ste(s: Tensor, M: Tensor, Mq, TDC: Tensor, PROD: Tensor, G: Tensor, SW: dict, MAPS: tuple,
              used: Tensor | None = None, pused: Tensor | None = None, gused: Tensor | None = None) -> Tensor:
    """round_hard on one-hot digits with the straight-through round switches (SW's rmul, radd, rcube)."""
    V = square_ste(s, TDC, PROD, MAPS[1], used, pused)
    V = pickl(SW["rmul"], [{"one": lambda: V, "ten": lambda: move_soft(V, 1)}[v] for v in ROUND_MENU["rmul"]])

    def plus(cst: int) -> Tensor:
        if cst == 0:
            return V
        c = onehot0(V.shape[0], V.shape[1], V.device, V.dtype)
        c[:, 0] = F.one_hot(torch.tensor(cst, device=V.device), 10).to(V.dtype)
        return damp(add_soft_learned(V, c, TDC, used)[0])
    V = pickl(SW["radd"], [lambda c=c: plus(c) for c in ROUND_MENU["radd"]])
    R = divide_ste(V, M, Mq, TDC, G, SW, used, gused)
    return pickl(SW["rcube"], [{"none": lambda: R, "times_x": lambda: divide_ste(
        square_ste(R, TDC, PROD, MAPS[1], used, pused, Sb=s), M, Mq, TDC, G, SW, used, gused)}[v]
        for v in ROUND_MENU["rcube"]])

'''


CHAIN_LESSON = '''    def chain_lesson(self, c: int, Xd: Tensor, Nd: Tensor, rowsN: Tensor, verified: Tensor, tabs, train: bool):
        """The division ladder of lesson set c (one division step at a time). Its registers and targets come from the
        set's predicate (red.step):
          below    from x + N q (q = 0..9) the chain moves the register down by multiples of N until it is below N:
                   it gives x;
          lowzero  from 10 x + N q it moves the register up by multiples of N until its lowest digit is 0: it gives
                   10 x (q = 0) or 10 (x + N);
        built from the machine's own row N q and its adder. Rows whose N rungs are not all verified are left out. With
        GATE_FINAL the verdicts also ask the gate to do the gated subtraction's job on the row's own numbers (csub: from
        x + N it subtracts N, from x it holds). Every combination of the set's chain switches and the gate's rows runs
        on the hard machine; the loss is the factorized mixture's (when `train`). Returns the loss, whether the argmax
        chain (and gate) passes, and the registers (for the step law)."""
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
            else:
                reg = add_learned(F.pad(Xs, (0, C - W))[:, None].expand(m, 10, C), rows, Td, Tb)[0][..., :P]
                xq = sq = x0.repeat_interleave(10, 0)
            reg = reg.reshape(m * 10, P)
            Nq = Ns.repeat_interleave(10, 0)
            mq = None
            if "Mq" in DIV_MENU["a"] + DIV_MENU["b"]:
                mq = (multiples(Nq, tabs), multiples(Nq.flip(-1), tabs))      # N's multiples, as read and reversed
            regs = {"reg": reg, "Nq": Nq, "xq": xq, "sq": sq, "mq": mq, "pred": pred}
            ca = self.wcodes(m * 10, dev, c)
            Ra = self.chain_run(reg, ca, regs, 1, tabs, self.ctl.gvisits)
            ok_a = (Ra == xq).all(1)
            csub = []                                                         # (register, its csub result) pairs
            for R in (xN, x0):
                E, cb, _ = sub_hard(R, Mx, Td, Tb)
                csub.append((R, E, cb))
            if GATE_FINAL:
                gc = self.g.argmax(-1) == 1
                for R, E, cb in csub:
                    cm = torch.where(cb == 0, gc[0], gc[1])
                    ok_a = ok_a & (torch.where(cm[:, None], E, R) == x0).all(1).repeat_interleave(10, 0)
            div_ok = float(ok_a.float().mean())
            if not train:
                return None, div_ok, regs
            combos = torch.tensor(CHAIN_COMBOS, device=dev)
            Kc = combos.shape[0]
            wk = codes(combo_wiring(self.wiring(m * 10, dev, c), CHAIN_SW, combos, m * 10, gate=True))
            verdict = (self.chain_run(reg.repeat(Kc, 1), wk, regs, Kc, tabs) == xq.repeat(Kc, 1)).all(1)
            verdict = verdict.view(Kc, m * 10)
            if GATE_FINAL:
                g0, g1 = combos[:, -2] == 1, combos[:, -1] == 1                # commit on flag 0, on flag 1
                okg = torch.ones(Kc, m, dtype=torch.bool, device=dev)
                for R, E, cb in csub:
                    cm = torch.where(cb[None] == 0, g0[:, None], g1[:, None])  # [Kc, m]
                    okg = okg & (torch.where(cm[..., None], E[None], R[None]) == x0[None]).all(-1)
                verdict = verdict & okg.repeat_interleave(10, 1)
        lw = self.combo_logw(CHAIN_SW, combos, gate=True, c=c)
        return mixture_nll(lw, verdict).mean(), div_ok, regs
'''


CHAIN_RUN = '''    def chain_run(self, reg: Tensor, w: dict, regs: dict, Kc: int, tabs, gvisits: Tensor | None = None,
                  bring: bool = False) -> Tensor:
        """One step on the division ladder's registers (Kc copies, copy-major) with per-row codes w: the chain alone
        from the register reg, or (bring) the whole step with a digit coming in (the bring-in, the chain, the post
        move). How the register is split for the step law follows the lesson set's predicate: below, its upper places
        (moved down) and its lowest digit; lowzero, the register with part e of its lowest digit left in place and the
        rest coming in (e: the register's next digit where it is not larger, else 0; d - e by the learned table), so
        that writing the digit over the place, which loses e, fails where adding it there passes."""
        Td, Tb = tabs[0], tabs[1]
        Nq = regs["Nq"].repeat(Kc, 1)
        R0 = F.pad(reg[:, 1:], (0, 1))                                      # the register's upper places
        d = reg[:, 0]
        if bring and regs.get("pred") == "lowzero":
            e = reg[:, 1]
            de, be, _ = sub_hard(d[:, None], e[:, None], Td, Tb)
            e = torch.where(be == 1, torch.zeros_like(e), e)
            R0 = torch.cat([e[:, None], reg[:, 1:]], 1)
            d = torch.where(be == 1, d, de[:, 0])
        mq = None
        if regs["mq"] is not None:
            rev = (w["nread"] == 1)[:, None, None]
            mq = lambda s: torch.where(rev[:, 0], regs["mq"][1][:, s - 1].repeat(Kc, 1),
                                       regs["mq"][0][:, s - 1].repeat(Kc, 1))
        Rp = bring_in(R0, d, w, Td, Tb) if bring else reg
        D = chain_hard(Rp, n_register(Nq, w["nread"]), w, Td, Tb, None, gvisits, mq, R0, F.pad(d[:, None], (0, P - 1)))
        return move_digits(D, w["post"]) if bring else D
'''


STEP_LESSON = '''    def step_lesson(self, c: int, regs: dict | None, tabs, train: bool, names: tuple):
        """The step law of lesson set c: the division ladder's register split as the set's predicate says (chain_run);
        one whole step (the bring-in, the argmax chain, the post move) must give the set's step target: x (below), x
        or x + N (lowzero: the chain's 10 x or 10 (x + N) moved down one place). Every combination of the step block's
        switches runs on the hard machine; the loss is the factorized mixture's (when `train`). Returns the loss and
        whether the argmax step passes."""
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
            combos = torch.tensor(list(itertools.product(*[range(len(DIV_MENU[k])) for k in names])), device=dev)
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
        argmax; their outputs are compared with the answers in token_training_loss). Every combination runs on the hard
        machine (the other switches at their argmax). Returns the laws' verdicts [K, n], the answer rows' outputs
        [K, m, W] and their indices, and whether the argmax wiring passes the laws."""
        Td, Tb = tabs[0], tabs[1]
        dev = Xd.device
        combos = torch.tensor(list(itertools.product(*[range(len(DIV_MENU[k])) for k in names])), device=dev)
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
            if mults is not None or "Mq" in DIV_MENU["a"] + DIV_MENU["b"]:          # every row's own nread
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
                                tabs, wm, cap=REAL_MAX_T + 1)[0].view(Kc, mr, W)
        return {"verdict": verdict if laws else None, "outs": outs, "sel": sel_rows, "combos": combos, "names": names,
                "gate": False, "c": c}, law_ok
'''


LESSONS = '''    def mine(self, terms: dict) -> None:
        """The miner's step on the fields as read (terms: name -> digits [X, P]): every candidate relation of
        `relations`; newly mined facts are logged."""
        ctl = self.ctl
        for k in ctl.miner.update(relations(terms), ctl.step):
            print(f"LR mined step={ctl.step} fact={k.replace(' ', '')}", flush=True)

    def lessons(self, Nd, Xd, Cd, small, width, tabs, e, end, k, code) -> dict:
        """The label-free lessons (the label-dependent ones run in token_training_loss on what this returns): the
        module docstring, LAWS and WIRING.md."""
        ctl = self.ctl
        ctl.step += 1
        dev = Xd.device
        Td, Tb, Plo, Phi, Gc = tabs[:5]
        M = F.pad(Nd, (0, P - W))                                            # N's digits as read, P places
        if ctl.visits is None:
            ctl.visits = torch.zeros(200, dtype=torch.long, device=dev)
            ctl.pvisits = torch.zeros(100, dtype=torch.long, device=dev)
            ctl.gvisits = torch.zeros(2, dtype=torch.long, device=dev)
        common = {"ctl": ctl, "model": self, "e": e, "end": end, "k": k, "code": code, "Xd": Xd, "Nd": Nd, "M": M,
                  "Cd": Cd, "small": small, "width": width, "tabs": tabs, "io": None, "exact": 0.0,
                  "table_exact": 0.0, "per": [None] * len(PREDICATES)}
        if ctl.phase == "frozen":
            return {"loss": self.zero(), **common}
        TDC = self.soft_table()
        X, NL = Xd.shape[0], W * (S - 1)
        hot = lambda v: F.one_hot(v, 10).float()
        log = lambda p: p.clamp_min(1e-12).log()
        digits_nll = lambda Y, target: -log(Y).gather(-1, target[..., None]).squeeze(-1).sum(-1)
        terms = {"N": M, "x": F.pad(Xd, (0, P - W)), "T": F.pad(Cd, (0, P - K))}
        if ctl.io >= 3:                                                      # the reader ranks its readings: mine
            with torch.no_grad():
                self.mine(terms)
        usable = ctl.phase in ("B", "C") or ctl.io >= 4                      # mined facts teach once the reading is chosen
        orders = [(a, b) for a in ORDER_TERMS for b in ORDER_TERMS if a != b and usable and ctl.miner.holds(f"{a} < {b}")]
        nonzero = [a for a in ORDER_TERMS if usable and ctl.miner.holds(f"{a} != 0")]
        ident = [a for a in ("x", "T") if (a, "N") in orders]               # red.identity's instances
        with torch.no_grad():
            V, sq_add = square_hard(Xd, tabs, check=True)                    # the machine's own square of x
            if CHECK_REGS == "own":                                          # the argmax division's step registers
                regs = []
                divide_hard(V, Nd, self.wcodes(X, dev), Td, Tb, regs=regs, tabs=tabs)
            else:                                                            # x and the square's P-place windows
                regs = [F.pad(Xd, (0, P - W))] + [V[:, t:t + P] for t in range(W - 1)]
            Rg = torch.stack(regs, 1).reshape(X * W, P)
            MW = M.repeat_interleave(W, 0)
            links = [Rg]
            for _ in range(S - 2):                                           # each register minus N again and again
                links.append(sub_hard(links[-1], MW, Td, Tb)[0])
            D = torch.stack(links, 1).reshape(X * NL, P)
            Mr = M.repeat_interleave(NL, 0)
            Z = torch.zeros_like(M)
            R2 = Rg.view(X, W, P).roll(1, 0).reshape(X * W, P)              # another row's registers (ALL_PAIRS)
        facts = bool(orders)
        digits_on = ctl.gated_on is not None
        pieces = digits_on and ctl.step - ctl.gated_on >= LATE_AFTER         # the pieces' own table lessons
        if pieces and ctl.step - ctl.gated_on == LATE_AFTER:
            ctl.history.append((ctl.step, "pieces"))
            print(f"LR pieces_join step={ctl.step}", flush=True)
        sc = STE_CIN == "all"
        sa = STE_CIN in ("all", "anchors")
        zero = torch.zeros_like(Rg)
        loss = self.zero()
        # sub.order on every mined order fact a < b: a[j:] - b[j:] borrows iff the digit strings differ there,
        # b[j:] - a[j:] never borrows
        for a_, b_ in orders:
            A_, B_ = terms[a_], terms[b_]
            for j in range(P):
                differ = (A_[:, j:] != B_[:, j:]).any(1)
                p1 = link_soft(hot(A_[:, j:]), B_[:, j:], TDC)[1][:, -1, 1]
                p2 = link_soft(hot(B_[:, j:]), A_[:, j:], TDC)[1][:, -1, 1]
                loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
        if facts:
            with torch.no_grad():
                okf = torch.ones(X, dtype=torch.bool, device=dev)
                for a_, b_ in orders:
                    okf = okf & (sub_hard(terms[a_], terms[b_], Td, Tb)[1] == 1) & \\
                        (sub_hard(terms[b_], terms[a_], Td, Tb)[1] == 0)
                ctl.facts_streak = ctl.facts_streak + 1 if bool(okf.all()) else 0
        if digits_on:
            # sub.order (digits) and sub.bijective on the entries the batch's subtractions read
            with torch.no_grad():
                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0
                if ALL_PAIRS:
                    E2, _, i2 = sub_hard(Rg, R2, Td, Tb)
                    C2, _, i3 = sub_hard(zero, R2, Td, Tb)
                    i4 = sub_hard(E2, C2, Td, Tb)[2]
                    for ii in (i2, i3, i4):
                        used = used | (torch.bincount(ii.flatten(), minlength=200) > 0)
            pa = self.a.float().softmax(-1)
            pd = pa.reshape(200, 10)
            for lo, hi in order_pairs(Tb, used):
                loss = loss - log((pd[lo] * (1 - pd[hi].cumsum(-1))).sum(-1)).mean()
            loss = loss - log(10 * pa.mean(0)).mean() - log(10 * pa.mean(1)).mean()
        if facts and not digits_on and ctl.facts_streak >= GATE_WIN:
            ctl.gated_on = ctl.step                                          # the borrows referee from the next step
            ctl.history.append((ctl.step, "order_digits"))
            print(f"LR order_digits step={ctl.step}", flush=True)
        weight = torch.ones(P, device=dev)
        weight[0] = ID_PLACE0                                               # the lowest place tells the swapped code

        def inverse_soft(Dv: Tensor, Bv: Tensor, Cs: Tensor, bc: Tensor) -> Tensor:
            """sub.inverse, complete: (D - B) - (0 - B) = D with the final borrows be = bc - bf (Cs, bc: 0 - B)."""
            Es, be = link_soft(hot(Dv), Bv, TDC, sc)
            back, bf = sub_soft(Es, Cs, TDC, None, sc)
            pe, pf, pc = be[:, -1, 1], bf[:, -1, 1], bc[:, -1, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
            nll = (-log(back).gather(-1, Dv[..., None]).squeeze(-1) * weight).sum(-1) - ID_BORROW * log(agree)
            return nll.mean() / (weight.sum() + ID_BORROW)
        if pieces:
            Cs, bc = link_soft(hot(Z), M, TDC, sc)                           # 0 - N, once per row
            loss = loss + inverse_soft(D, Mr, Cs.repeat_interleave(NL, 0), bc.repeat_interleave(NL, 0))
            if ALL_PAIRS:                                                    # a register against another row's
                C2s, bc2 = link_soft(hot(zero), R2, TDC, sc)
                loss = loss + inverse_soft(Rg, R2, C2s, bc2)
            bc = bc[:, -1]
        # sub.identity, sub.self on the registers
        checks = []                                                         # hard versions: (A, B, target, borrow)
        for wgt, A, B, target in ((ANCHOR_ZERO, Rg, zero, Rg), (ANCHOR_SELF, Rg, Rg, zero)):
            if wgt > 0:                                                     # R - 0 = R, R - R = 0, no final borrow
                if pieces:
                    Y, b = link_soft(hot(A), B, TDC, sa)
                    loss = loss + wgt * (digits_nll(Y, target) - log(b[:, -1, 0])).mean() / (P + 1)
                checks.append((A, B, target, 0))
        if pieces and ANCHOR_NEG2 > 0:                                      # add.identity: 0 + N = N, complete
            Y2, b2 = sub_soft(hot(Z), Cs, TDC, None, sc)
            agree2 = bc[:, 1] * b2[:, -1, 1] + bc[:, 0] * b2[:, -1, 0]
            loss = loss + ANCHOR_NEG2 * (digits_nll(Y2, M) - log(agree2)).mean() / (P + 1)

        def commute_soft(A: Tensor, B: Tensor, CB: Tensor, bcb: Tensor) -> Tensor:
            """add.commute: A + B = B + A, i.e. A - (0 - B) = B - (0 - A) with equal carries (CB, bcb: 0 - B)."""
            A1, b1 = sub_soft(hot(A), CB, TDC, None, sc)
            CA, bca = link_soft(hot(torch.zeros_like(A)), A, TDC, sc)
            A2, b2 = sub_soft(hot(B), CA, TDC, None, sc)
            c1 = bcb * b1[:, -1, 0]                                          # P(carry = 1)
            c2 = bca[:, -1, 1] * b2[:, -1, 0]
            agree = log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))
            return -agree.mean() / (P + 1)
        if pieces and ANCHOR_COMM > 0:
            loss = loss + ANCHOR_COMM * commute_soft(Rg, MW, Cs.repeat_interleave(W, 0), bc[:, 1].repeat_interleave(W, 0))
            if ALL_PAIRS:
                loss = loss + ANCHOR_COMM * commute_soft(Rg, R2, C2s, bc2[:, -1, 1])
        # the mined facts' own laws: sub.order's final borrows, sub.wrap, sub.zero_minus
        for a_, b_ in orders:
            A_, B_ = terms[a_], terms[b_]
            if pieces:
                loss = loss - ANCHOR_LESS * log(link_soft(hot(A_), B_, TDC, sc)[1][:, -1, 1]).mean()
                loss = loss - ANCHOR_MORE * log(link_soft(hot(B_), A_, TDC, sc)[1][:, -1, 0]).mean()
                if ANCHOR_WRAP > 0:
                    Ew = link_soft(hot(A_), B_, TDC, sc)[0]
                    loss = loss - ANCHOR_WRAP * log(link_soft(Ew, B_, TDC, sc)[1][:, -1, 0]).mean()
            checks.append((A_, B_, None, 1))
            checks.append((B_, A_, None, 0))
        for a_ in nonzero:
            if pieces and ANCHOR_NEG > 0:
                loss = loss - ANCHOR_NEG * log(link_soft(hot(torch.zeros_like(terms[a_])), terms[a_], TDC, sc)[1][:, -1, 1]
                                               ).mean()
            checks.append((torch.zeros_like(terms[a_]), terms[a_], None, 1))
        # the product cell (SQUARE.md), once the table can be trusted
        lad_loss, lad_ok, verified, lad_rows = self.ladder(torch.cat([Xd, Nd]), tabs, ctl.ladder)
        if ctl.ladder:
            loss = loss + LADDER_W * lad_loss
        num_loss, num_ok = self.num_lesson(torch.cat([Xd, Nd]), tabs, ctl.ladder)   # the number rule
        if ctl.ladder and num_loss is not None:
            loss = loss + num_loss
        with torch.no_grad():                                               # the same checks on the hard machine
            Eh, beh, idx = sub_hard(D, Mr, Td, Tb)
            tally(ctl.visits, idx, 200)
            Ch, bch, idc = sub_hard(Z, M, Td, Tb)
            backh, bfh, idf = sub_hard(Eh, Ch.repeat_interleave(NL, 0), Td, Tb)
            tally(ctl.visits, idc, 200)
            tally(ctl.visits, idf, 200)
            ok = (backh == D).all(-1) & (beh == bch.repeat_interleave(NL, 0) - bfh)
            table = float(ok.float().mean())
            if ALL_PAIRS:
                E2h, be2, i2 = sub_hard(Rg, R2, Td, Tb)
                C2h, bc2h, i3 = sub_hard(zero, R2, Td, Tb)
                back2, bf2, i4 = sub_hard(E2h, C2h, Td, Tb)
                for ii in (i2, i3, i4):
                    tally(ctl.visits, ii, 200)
                table = min(table, float(((back2 == Rg).all(-1) & (be2 == bc2h - bf2)).float().mean()))
                A1h, c1h = add_learned(Rg, R2, Td, Tb, ctl.visits)
                A2h, c2h = add_learned(R2, Rg, Td, Tb, ctl.visits)
                table = min(table, float(((A1h == A2h).all(-1) & (c1h == c2h)).float().mean()))
            if ANCHOR_NEG2 > 0:
                Y2h, b2h, id2 = sub_hard(Z, Ch, Td, Tb)
                tally(ctl.visits, id2, 200)
                table = min(table, float(((Y2h == M).all(-1) & (b2h == bch)).float().mean()))
            if ANCHOR_COMM > 0:
                A1h, c1h = add_learned(Rg, MW, Td, Tb, ctl.visits)
                A2h, c2h = add_learned(MW, Rg, Td, Tb, ctl.visits)
                table = min(table, float(((A1h == A2h).all(-1) & (c1h == c2h)).float().mean()))
            for A, B, target, borrow in checks:
                Yh, bh, ida = sub_hard(A, B, Td, Tb)
                tally(ctl.visits, ida, 200)
                okc = bh == borrow
                if target is not None:
                    okc = okc & (Yh == target).all(-1)
                table = min(table, float(okc.float().mean()))
            if ANCHOR_WRAP > 0:
                for a_, b_ in orders:
                    Ewh, _, idw = sub_hard(terms[a_], terms[b_], Td, Tb)
                    tally(ctl.visits, idw, 200)
                    table = min(table, float((sub_hard(Ewh, terms[b_], Td, Tb)[1] == 0).float().mean()))
            top_ = Xd.max(1).values[:, None, None].expand(-1, W, 1)
            sq = verified[:X].gather(2, top_)[..., 0].all(1) & sq_add       # the squares of x check out
            exact = min(table, lad_ok, float(sq.float().mean()), float(num_ok))
        # the division's wiring, per lesson set (kind 3): the chain's switches and the gate by the set's division
        # ladder (phase C), then its step law and its reading's whole-reduction laws (and the answers)
        admissible = [PREDICATES[c] != "lowzero" or (usable and ctl.miner.holds("N mod 2 != 0")
                                                     and ctl.miner.holds("N mod 5 != 0")) for c in range(len(PREDICATES))]
        chain_on = ctl.phase == "C"
        per, io_parts = [], []
        for c in range(len(PREDICATES)):
            if not admissible[c]:
                per.append(None)
                continue
            div_loss, div_ok, regs = self.chain_lesson(c, Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, chain_on)
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
            common["io"] = io_parts
        cs = self.sigma()
        sel = per[cs] or {"div_ok": 0.0, "law_ok": 0.0, "step_ok": 0.0}
        exact = min(exact, sel["law_ok"], sel["step_ok"], sel["div_ok"])
        if MODE == "learned":
            loss = loss + self.io_lessons(e, end, k, code, V[:, :W])
        common.update(exact=exact, table_exact=table, lad_ok=lad_ok, div_ok=sel["div_ok"], law_ok=sel["law_ok"],
                      step_ok=sel["step_ok"], num_ok=float(num_ok), per=per)
        return {"loss": loss, **common}
'''


ROUND_LESSON = '''    def round_logw(self, combos: Tensor) -> Tensor:
        """log of the factorized mixture's weight of every combination of the round's options [K, 3]."""
        return sum(self.rnd[k].float().log_softmax(-1)[combos[:, i]] for i, k in enumerate(ROUND_SW))

    def round_lesson(self, aux: dict, yd: Tensor) -> Tensor:
        """The lesson sets' selector and the shared round menu, by the answers alone: every lesson set whose I/O
        lessons have joined runs its argmax machine under every combination of the round's options on the rows with
        T <= REAL_MAX_T; the loss is the factorized mixture's over (lesson set, round options). Nothing else trains the
        selector, so a lesson set cannot win by its own lessons' loss, and a set whose machine fits no answer only
        loses weight."""
        ctl = self.ctl
        sel = torch.nonzero(aux["small"]).flatten()
        if sel.numel() == 0 or not any(ctl.wire_ios):
            return self.lsel.sum() * 0.0
        dev = sel.device
        combos = torch.tensor(ROUND_COMBOS, device=dev)
        K, m = combos.shape[0], sel.numel()
        with torch.no_grad():
            vs = []
            for c in range(len(PREDICATES)):
                if not ctl.wire_ios[c]:
                    vs.append(torch.zeros(K, m, dtype=torch.bool, device=dev))
                    continue
                wk = repeat_wiring(self.wiring(m, dev, c), K)
                for i, k in enumerate(ROUND_SW):
                    wk[k] = combos[:, i].repeat_interleave(m)
                out = run_hard(aux["Xd"][sel].repeat(K, 1), aux["Cd"][sel].repeat(K, 1), aux["Nd"][sel].repeat(K, 1),
                               aux["tabs"], codes(wk), cap=REAL_MAX_T + 1)[0]
                vs.append((out.view(K, m, W) == yd[sel][None]).all(-1))
            verdict = torch.cat(vs)                                           # [sets x K, m]
            ctl.sel_fit = [float(v.any(0).float().mean()) for v in vs]       # share of answer rows some combination fits
        lw = (self.lsel.float().log_softmax(-1)[:, None] + self.round_logw(combos)[None]).reshape(-1)
        return mixture_nll(lw, verdict).mean()

    def mine_answers(self, aux: dict, yd: Tensor) -> None:
        """The miner on the answers (reported only): the relations between the answer y and N, x, T."""
        ctl = self.ctl
        if ctl.io < 3:
            return
        with torch.no_grad():
            terms = {"N": aux["M"], "x": F.pad(aux["Xd"], (0, P - W)), "T": F.pad(aux["Cd"], (0, P - K)),
                     "y": F.pad(yd, (0, P - W))}
            rels = {k: v for k, v in relations(terms).items() if "y" in k.split()}
            for k in ctl.miner.update(rels, ctl.step):
                print(f"LR mined step={ctl.step} fact={k.replace(' ', '')}", flush=True)
'''


def build() -> str:
    src = BASE.read_text()
    assert hashlib.md5(src.encode()).hexdigest() == BASE_MD5, "the certified base file changed"
    src = sub1(src, '"""WideMenus: the learned-wiring H1 machine (learned_wiring/submission_io3.py, WIRING.md) with its '
                    'wiring chosen from', DOC)
    src = sub1(src, "FREEZE_WIN = 200       # phase C: freeze after this many consecutive steps with every check exact",
               CONSTANTS)
    # the menus: the Montgomery truth (designed mode) and the round menu's combinations
    src = sub1(src, 'MUL_TRUE = {"lo": (1, 1, 0), "hi": (1, 1, 1), "num": (0, 1)}   # columns i + j and i + j + 1, numbers by j',
               'MUL_TRUE = {"lo": (1, 1, 0), "hi": (1, 1, 1), "num": (0, 1)}   # columns i + j and i + j + 1, numbers by j\n'
               'DIV_MONT = {"a": "D", "b": "M", "op": "add", "flag": "zD", "keep": "gated", "nread": "given", '
               '"vread": "given",\n            "start": "zero", "order": "up", "bring": (0, 0, "add"), "post": -1, '
               '"final": "addhigh", "read": (0, "given")}\n'
               'SET_TRUE = {"below": DIV_TRUE, "lowzero": DIV_MONT}   # designed mode and the reports only\n'
               'ROUND_COMBOS = list(itertools.product(*[range(len(ROUND_MENU[k])) for k in ROUND_SW]))')
    src = sub1(src, 'IO_MODE = "local"      #', 'IO_MODE = "local"      # (the general file: "local" only)')
    src = sub1(src, "def menu_size(menu: dict, names=None) -> int:", LAWS_BLOCK.lstrip("\n") + "\n\n\ndef menu_size(menu: dict, "
               "names=None) -> int:")
    # codes(): the round's options
    src = sub1(src, '''    out["rdir"] = lut([int(v[1] == "reversed") for v in DIV_MENU["read"]])[ix["read"]]
    return out''', '''    out["rdir"] = lut([int(v[1] == "reversed") for v in DIV_MENU["read"]])[ix["read"]]
    rvals = {"rmul": [int(v == "ten") for v in ROUND_MENU["rmul"]], "radd": ROUND_MENU["radd"],
             "rcube": [int(v == "times_x") for v in ROUND_MENU["rcube"]]}
    for k in ROUND_SW:                                                      # the round's options (absent: the first)
        out[k] = lut(rvals[k])[idx[k] if k in idx else torch.zeros(X, dtype=torch.long, device=dev)]
    return out''')
    # the multiply's second operand (the round option "times the input")
    src = sub1(src, '''def square_hard(Sd: Tensor, tabs, visits: Tensor | None = None, pvisits: Tensor | None = None, check: bool = False,
                full: Tensor | None = None):''', '''def square_hard(Sd: Tensor, tabs, visits: Tensor | None = None, pvisits: Tensor | None = None, check: bool = False,
                full: Tensor | None = None, Sb: Tensor | None = None):''')
    src = sub1(src, "    idx = (Sd[:, :, None] * 10 + Sd[:, None, :]).reshape(X, W * W)          # pair (i, j): cell (s_i, s_j)",
               "    Sb = Sd if Sb is None else Sb                                           # the product Sd * Sb\n"
               "    idx = (Sd[:, :, None] * 10 + Sb[:, None, :]).reshape(X, W * W)          # pair (i, j): cell (s_i, s_j)")
    src = sub1(src, '''def square_ste(Sd: Tensor, TDC: Tensor, PROD: Tensor, FULL: Tensor, used: Tensor | None = None,
               pused: Tensor | None = None):''', '''def square_ste(Sd: Tensor, TDC: Tensor, PROD: Tensor, FULL: Tensor, used: Tensor | None = None,
               pused: Tensor | None = None, Sb: Tensor | None = None):''')
    src = sub1(src, "    cell = (Sd[:, :, None, :, None] * Sd[:, None, :, None, :]).reshape(X, W * W, 100)   # pair (i, j): (s_i, s_j)",
               "    Sb = Sd if Sb is None else Sb                                           # the product Sd * Sb\n"
               "    cell = (Sd[:, :, None, :, None] * Sb[:, None, :, None, :]).reshape(X, W * W, 100)   # pair (i, j): (s_i, s_j)")
    # the round, on both paths
    src = sub1(src, "def run_hard(Sd: Tensor, C: Tensor, Nd: Tensor, tabs, w: dict,", ROUND_FNS.lstrip("\n") +
               "\ndef run_hard(Sd: Tensor, C: Tensor, Nd: Tensor, tabs, w: dict,")
    src = sub1(src, "        R = divide_hard(square_hard(s, tabs, visits, pvisits), Nd, w, Td, Tb, visits, gvisits, tabs=tabs, mults=mults)",
               "        R = round_hard(s, Nd, w, tabs, visits, gvisits, pvisits, mults)")
    src = sub1(src, "def run_ste(Sd: Tensor, Cnt: Tensor, N1: Tensor,", ROUND_STE.lstrip("\n") +
               "\ndef run_ste(Sd: Tensor, Cnt: Tensor, N1: Tensor,")
    src = sub1(src, "        R = divide_ste(square_ste(s, TDC, PROD, MAPS[1], used, pused), M, Mq, TDC, G, SW, used, gused)",
               "        R = round_ste(s, M, Mq, TDC, PROD, G, SW, MAPS, used, pused, gused)")
    # the schedule's state: per lesson set
    src = sub1(src, "        self.chain_streak = 0                 # steps in a row with the argmax chain exact on the division ladder",
               "        self.chain_streak = 0                 # steps in a row with the argmax chain exact on the division ladder\n"
               "        self.wire_ios = [designed] * len(PREDICATES)   # per lesson set: its I/O lessons have joined\n"
               "        self.chain_streaks = [0] * len(PREDICATES)     # per lesson set: steps with its argmax chain exact\n"
               "        self.miner = Miner()                  # the mined facts (kind 2)\n"
               "        self.sel_fit = [0.0] * len(PREDICATES)  # per lesson set: answer rows some round combination fits")
    # classify: the Montgomery forms
    src = sub1(src, '''             ("Rp", "Mq", "sub", "borrow", "gated"): "trial multiples"}.get(key, "other")''',
               '''             ("Rp", "Mq", "sub", "borrow", "gated"): "trial multiples",
             ("D", "M", "add", "zD", "gated"): "montgomery, repeated addition",
             ("M", "D", "add", "zD", "gated"): "montgomery, repeated addition",
             ("Rp", "Mq", "add", "zE", "gated"): "montgomery, trial multiples (the sum's digit tested)",
             ("Rp", "Mq", "add", "zD", "gated"): "montgomery, trial multiples (the register's digit tested)",
             ("Mq", "Rp", "add", "zE", "gated"): "montgomery, trial multiples (the sum's digit tested)",
             ("Mq", "Rp", "add", "zD", "gated"): "montgomery, trial multiples (the register's digit tested)"
             }.get(key, "other")''')
    # the model's parameters: one switch set per lesson set, the selector, the round's switches
    src = sub1(src, "        self.sw = nn.ParameterDict({k: nn.Parameter(INIT * torch.randn(len(DIV_MENU[k]))) for k in SWITCHED})",
               "        self.sws = nn.ModuleList([nn.ParameterDict({k: nn.Parameter(INIT * torch.randn(len(DIV_MENU[k])))\n"
               "                                                    for k in SWITCHED}) for _ in PREDICATES])   # per set\n"
               "        self.lsel = nn.Parameter(INIT * torch.randn(len(PREDICATES)))       # the lesson sets' selector\n"
               "        self.rnd = nn.ParameterDict({k: nn.Parameter(INIT * torch.randn(len(ROUND_MENU[k])))\n"
               "                                     for k in ROUND_SW})                    # the shared round menu")
    src = sub1(src, '''        for k in SWITCHED:
            self.sw[k].copy_(hot(torch.tensor(DIV_MENU[k].index(DIV_TRUE[k])), len(DIV_MENU[k])))''',
               '''        for c, pred in enumerate(PREDICATES):
            for k in SWITCHED:
                self.sws[c][k].copy_(hot(torch.tensor(DIV_MENU[k].index(SET_TRUE[pred][k])), len(DIV_MENU[k])))
        self.lsel.copy_(hot(torch.tensor(PREDICATES.index(DESIGNED_SIGMA)), len(PREDICATES)))
        for k in ROUND_SW:
            self.rnd[k].copy_(hot(torch.tensor(ROUND_MENU[k].index(DESIGNED_ROUND[k] if k != "rmul" else
                                                                   DESIGNED_ROUND[k])), len(ROUND_MENU[k])))''')
    src = sub1(src, '''    def choice(self, k: str) -> int:
        """The argmax option of the division's switch k (0: a single fixed option)."""
        return int(self.sw[k].argmax()) if k in SWITCHED else 0''',
               '''    def sigma(self) -> int:
        """The selected lesson set: the selector's argmax."""
        return int(self.lsel.argmax())

    def swd(self, c: int | None = None) -> nn.ParameterDict:
        """Lesson set c's division switches (default: the selected set's)."""
        return self.sws[self.sigma() if c is None else c]

    def choice(self, k: str, c: int | None = None) -> int:
        """The argmax option of lesson set c's division switch k (0: a single fixed option)."""
        return int(self.swd(c)[k].argmax()) if k in SWITCHED else 0''')
    src = sub1(src, '''    def ste_switches(self) -> dict:
        """The division's switches on the prediction path: straight-through one-hots [n] (values exactly 0 or 1); a
        single fixed option is the constant [1]."""
        out = {k: torch.ones(1, device=self.a.device) for k in DIV_MENU}
        out.update({k: ste(F.one_hot(v.argmax(), v.shape[0]).float(), v.float().softmax(-1)) for k, v in self.sw.items()})
        return out''', '''    def ste_switches(self, c: int | None = None) -> dict:
        """Lesson set c's division switches and the round's on the prediction path: straight-through one-hots [n]
        (values exactly 0 or 1); a single fixed option is the constant [1]."""
        hot = lambda v: ste(F.one_hot(v.argmax(), v.shape[0]).float(), v.float().softmax(-1))
        out = {k: torch.ones(1, device=self.a.device) for k in DIV_MENU}
        out.update({k: hot(v) for k, v in self.swd(c).items()})
        out.update({k: hot(v) for k, v in self.rnd.items()})
        return out''')
    src = sub1(src, '''    def wiring(self, X: int, device) -> dict:
        """The argmax wiring, per row [X]: every switch's option index and the gate's two rows (commit on flag 0 / 1)."""
        w = {k: torch.full((X,), int(v.argmax()), dtype=torch.long, device=device) for k, v in self.sw.items()}''',
               '''    def wiring(self, X: int, device, c: int | None = None) -> dict:
        """Lesson set c's argmax wiring (default: the selected set's), per row [X]: every switch's option index, the
        gate's two rows (commit on flag 0 / 1) and the round's options."""
        w = {k: torch.full((X,), int(v.argmax()), dtype=torch.long, device=device) for k, v in self.swd(c).items()}
        w.update({k: torch.full((X,), int(v.argmax()), dtype=torch.long, device=device) for k, v in self.rnd.items()})''')
    src = sub1(src, '''    def wcodes(self, X: int, device) -> dict:
        return codes(self.wiring(X, device))

    def names(self) -> dict:
        out = {k: DIV_MENU[k][self.choice(k)] for k in DIV_MENU}''', '''    def wcodes(self, X: int, device, c: int | None = None) -> dict:
        return codes(self.wiring(X, device, c))

    def round_names(self) -> str:
        return " ".join(f"{k}={ROUND_MENU[k][int(self.rnd[k].argmax())]}" for k in ROUND_SW)

    def names(self, c: int | None = None) -> dict:
        out = {k: DIV_MENU[k][self.choice(k, c)] for k in DIV_MENU}''')
    src = sub1(src, '''    def wiring_true(self) -> bool:
        return all(self.choice(k) == DIV_MENU[k].index(DIV_TRUE[k]) for k in DIV_MENU)

    def wiring_names(self) -> str:
        gate = "".join("C" if int(c) == 1 else "H" for c in self.g.argmax(-1))
        return (" ".join(f"{k}={DIV_MENU[k][self.choice(k)]}".replace(" ", "") for k in DIV_MENU if k in SWITCHED)
                + f" gate={gate}")

    def combo_logw(self, names: tuple, combos: Tensor, gate: bool) -> Tensor:
        """log of the factorized mixture's weight of every combination combos [K, len(names) (+ 2 gate rows)]: the
        product of the switches' probabilities (and of the gate's rows)."""
        lw = 0
        for i, n in enumerate(names):
            lw = lw + self.sw[n].float().log_softmax(-1)[combos[:, i]]''', '''    def wiring_true(self, c: int | None = None) -> bool:
        pred = PREDICATES[self.sigma() if c is None else c]
        return all(self.choice(k, c) == DIV_MENU[k].index(SET_TRUE[pred][k]) for k in DIV_MENU)

    def wiring_names(self, c: int | None = None) -> str:
        gate = "".join("C" if int(g) == 1 else "H" for g in self.g.argmax(-1))
        return (" ".join(f"{k}={DIV_MENU[k][self.choice(k, c)]}".replace(" ", "") for k in DIV_MENU if k in SWITCHED)
                + f" gate={gate}")

    def combo_logw(self, names: tuple, combos: Tensor, gate: bool, c: int | None = None) -> Tensor:
        """log of the factorized mixture's weight of every combination combos [K, len(names) (+ 2 gate rows)] of lesson
        set c's switches: the product of the switches' probabilities (and of the gate's rows)."""
        lw = 0
        for i, n in enumerate(names):
            lw = lw + self.swd(c)[n].float().log_softmax(-1)[combos[:, i]]''')
    # the prediction path: straight-through over the lesson sets
    src = sub1(src, '''        TDC, PROD, G = self.ste_tables()
        s = run_ste(field(1), field(2), field(0), TDC, PROD, G, self.ste_switches(), self.ste_sources(), ctl.used,
                    ctl.used_count, ctl.pused, ctl.used_gate)''', '''        TDC, PROD, G = self.ste_tables()
        Sx, Sc, Sn, MAPS = field(1), field(2), field(0), self.ste_sources()
        SEL = hot(self.lsel.float())                                         # the selected lesson set [sets]
        s = pickl(SEL, [lambda c=c: run_ste(Sx, Sc, Sn, TDC, PROD, G, self.ste_switches(c), MAPS, ctl.used,
                                            ctl.used_count, ctl.pused, ctl.used_gate) for c in range(len(PREDICATES))])''')
    # the lessons
    for name, code in (("chain_lesson", CHAIN_LESSON), ("chain_run", CHAIN_RUN), ("step_lesson", STEP_LESSON),
                       ("io_lesson", IO_LESSON)):
        src = replace_def(src, name, code)
    src = replace_def(src, "lessons", LESSONS)
    src = sub1(src, "    def io_lessons(self, e: Tensor, end: Tensor, k: Tensor, code: Tensor, vals: Tensor) -> Tensor:",
               ROUND_LESSON + "\n    def io_lessons(self, e: Tensor, end: Tensor, k: Tensor, code: Tensor, vals: Tensor) -> Tensor:")
    src = sub1(src, '''        for io in aux["io"]:
            lw = self.combo_logw(io["names"], io["combos"], gate=io["gate"])
            loss = loss + mixture_nll(lw, io["verdict"]).mean()''', '''        for io in aux["io"]:
            lw = self.combo_logw(io["names"], io["combos"], gate=io["gate"], c=io["c"])
            if io["verdict"] is not None:
                loss = loss + mixture_nll(lw, io["verdict"]).mean()''')
    # the schedule
    src = sub1(src, '''            print(f"LR step={ctl.step} phase={ctl.phase} io={ctl.io} ladder={int(ctl.ladder)} wire_io={int(ctl.wire_io)} "''',
               '''            print(f"LR step={ctl.step} phase={ctl.phase} io={ctl.io} ladder={int(ctl.ladder)} "
                  f"wire_io={''.join(str(int(v)) for v in ctl.wire_ios)} "''')
    src = sub1(src, '''                  f"wiring: {self.wiring_names()} {ctl.t_log}",
                  flush=True)''', '''                  f"wiring: {self.wiring_names()} {ctl.t_log} | sets={','.join(PREDICATES)} "
                  f"selector={','.join(f'{float(p):.3f}' for p in self.lsel.float().softmax(-1).tolist())} "
                  f"fit={','.join(f'{v:.3f}' for v in ctl.sel_fit)} round: {self.round_names()} "
                  f"per_set={[None if p is None else {k: round(v, 3) for k, v in p.items()} for p in aux.get('per') or []]} "
                  f"facts={ctl.miner.describe()}",
                  flush=True)''')
    src = sub1(src, '''            if not ctl.wire_io:
                ctl.chain_streak = ctl.chain_streak + 1 if aux.get("div_ok", 0.0) == 1.0 else 0
                if ctl.chain_streak >= WIRE_WIN or IO_MODE == "joint":
                    ctl.wire_io = True
                    ctl.history.append((ctl.step, "wire_io", ctl.in_phase))
                    print(f"LR io_wiring_lesson_joins step={ctl.step} wiring: {self.wiring_names()}", flush=True)
            ok = exact == 1.0 and ctl.io_exact and ctl.wire_io and ctl.ans_ok == 1.0''', '''            for c, pc in enumerate(aux.get("per") or []):
                if pc is None or ctl.wire_ios[c]:
                    continue
                ctl.chain_streaks[c] = ctl.chain_streaks[c] + 1 if pc["div_ok"] == 1.0 else 0
                if ctl.chain_streaks[c] >= WIRE_WIN:
                    ctl.wire_ios[c] = True
                    ctl.history.append((ctl.step, f"wire_io_{PREDICATES[c]}", ctl.in_phase))
                    print(f"LR io_wiring_lesson_joins set={PREDICATES[c]} step={ctl.step} wiring: "
                          f"{self.wiring_names(c)}", flush=True)
            ok = exact == 1.0 and ctl.io_exact and ctl.wire_ios[self.sigma()] and ctl.ans_ok == 1.0''')
    src = sub1(src, '''                      f"{self.layout_names()} {classify(self.names())}", flush=True)''',
               '''                      f"{self.layout_names()} {classify(self.names())} set={PREDICATES[self.sigma()]} "
                      f"round: {self.round_names()}", flush=True)''')
    # the label lessons: the selector and the round menu; the miner on the answers
    src = sub1(src, '''        if aux["io"] is not None:                                           # the division's I/O lesson
            loss = loss + model.wire_answers(aux, yd)''', '''        if aux["io"] is not None:                                           # the division's I/O lesson
            loss = loss + model.wire_answers(aux, yd)
        if any(ctl.wire_ios):                                               # the lesson sets' selector, the round
            loss = loss + model.round_lesson(aux, yd)
        model.mine_answers(aux, yd)''')
    # the reports
    src = sub1(src, '''        names = self.names()
        print("WIRING " + self.wiring_names() + " true=" + str(self.wiring_true())
              + " chain=" + classify(names)["chain"].replace(" ", "_"), flush=True)''', '''        names = self.names()
        for c, pred in enumerate(PREDICATES):
            print(f"WIRING set={pred} " + self.wiring_names(c) + " true=" + str(self.wiring_true(c))
                  + " chain=" + classify(self.names(c))["chain"].replace(" ", "_")
                  + " probabilities=" + ",".join(f"{k}:{float(v.float().softmax(-1).max()):.4f}"
                                                 for k, v in self.sws[c].items()), flush=True)
        print(f"LESSONS selected={PREDICATES[self.sigma()]} selector="
              + ",".join(f"{p}:{float(v):.4f}" for p, v in zip(PREDICATES, self.lsel.float().softmax(-1).tolist()))
              + f" wire_io={ctl.wire_ios} round: {self.round_names()} round_probabilities="
              + ",".join(f"{k}:{float(self.rnd[k].float().softmax(-1).max()):.4f}" for k in ROUND_SW), flush=True)
        print(f"MINED {ctl.miner.describe()}", flush=True)
        print("LAWS " + "; ".join(f"{k}: {v[0]} [{v[1]}]" for k, v in LAWS.items()), flush=True)''')
    src = sub1(src, '''        print("WIRING probabilities " + " ".join(f"{k}={float(v.float().softmax(-1).max()):.4f}"
                                                 for k, v in self.sw.items())
              + f" lo={float(self.lay[0].float().softmax(-1).max()):.4f}"''', '''        print("WIRING probabilities"
              + f" lo={float(self.lay[0].float().softmax(-1).max()):.4f}"''')
    assert "self.sw[" not in src and "self.sw." not in src and "ctl.wire_io " not in src, "an old use is left"
    return src


READER_DOC = '''"""GENERAL-LESSONS DIAGNOSTIC (cpu_first/general_lessons/LESSONS.md), not a submission: submission_general.py with the
reader and the writer set to the truth and held there (litmus/variants/submission_reader_frozen.py's change, made to the
general file): the digit tokens' code, the pointers (READER_GIVEN: N read from its right end, the Easy/Medium layout,
or its left, H1's), T's direction, the I/O stages done, and the reader's label-free lessons off. It asks whether the
general lessons let the rest of the machine (above all the subtraction table, through the laws on every pair of
numbers) learn from one fixed modulus once the prompt is read right. Setting the reader by hand breaks rule 6.

'''


def build_reader_frozen(src: str) -> str:
    src = sub1(src, '"""GeneralLessons (cpu_first/general_lessons/LESSONS.md):', READER_DOC.rstrip("\n") + "\n" +
               "GeneralLessons (cpu_first/general_lessons/LESSONS.md):")
    src = sub1(src, "MINE_WIN = 10          #", "READER_FROZEN = True   # diagnostic: no label-free reader lessons\n"
               'READER_GIVEN = "right" # diagnostic: N read from the right end of its run ("right") or the left ("left")\n'
               "MINE_WIN = 10          #")
    src = sub1(src, '''        if MODE == "designed":
            self.designed()
''', '''        if MODE == "designed":
            self.designed()
        else:
            self.reader_given()

    @torch.no_grad()
    def reader_given(self) -> None:
        """diagnostic: the reader and writer at the truth (designed()'s I/O part, N's direction from READER_GIVEN),
        and the I/O stages done."""
        DIGIT, MARK_N, MARK_X, MARK_T = 7, 2, 3, 4
        self.dg.fill_(-5.0)
        self.dg[DIGIT:DIGIT + 10] = 5.0
        self.u.copy_(torch.arange(self.V, dtype=torch.float)[None].expand_as(self.u))
        self.wc.fill_(-5.0)
        self.wc[torch.arange(10), DIGIT + torch.arange(10)] = 5.0
        self.anc.fill_(-5.0)
        self.strd.fill_(-5.0)
        self.shape.copy_(10.0 * F.one_hot(torch.tensor(0), len(SHAPES)).float() - 5.0)   # the step shape
        n_field = (MARK_N, 0) if READER_GIVEN == "left" else (MARK_X, 1)
        for f, (tok, step) in enumerate([n_field, (MARK_T, 1), (self.END, 1), (self.END, 1)]):
            self.anc[f, tok] = 5.0
            self.strd[f, step] = 5.0
        self.ctl.io = 5
''')
    src = sub1(src, '''        if MODE == "learned":
            loss = loss + self.io_lessons(e, end, k, code, V[:, :W])''', '''        if MODE == "learned" and not READER_FROZEN:                         # diagnostic: the reader keeps its code
            loss = loss + self.io_lessons(e, end, k, code, V[:, :W])''')
    return src


def main() -> None:
    src = build()
    OUT.write_text(src)
    print(f"wrote {OUT} ({len(src.splitlines())} lines, md5 {hashlib.md5(src.encode()).hexdigest()})")
    (HERE / "variants").mkdir(exist_ok=True)
    diag = build_reader_frozen(src)
    (HERE / "variants" / "submission_general_reader_frozen.py").write_text(diag)
    print(f"wrote variants/submission_general_reader_frozen.py (md5 {hashlib.md5(diag.encode()).hexdigest()})")


if __name__ == "__main__":
    main()
