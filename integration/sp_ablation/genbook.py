"""The generated law book (integration/sp_ablation/SP_ABLATION.md §5): the text make_books.py puts into a copy of sp2 in
place of its hand-written book (PATTERNS, LAW_BOOK, Draw, law_target). Not importable on its own: it runs inside the
submission, after libcore and spcore's helpers (Ops, repeat_add, count_of, digit_string, top_parts, pad, width_for,
NotPosable). Two blocks, between the marker lines.

What is generated, and what stays designed:
  * numbers: the division law's form r + q b, on the two pairs the data says are ordered, r < b: x' < N' (the mined
    order) and a digit d < B (a digit's range); q one, a digit, another row's x (o), or x itself;
  * patterns: the comparison pair (x and N in both orders, with itself, with zero: sp2's), each number with N, x with
    each count, and o, x with each count;
  * operations: the machine's binary arithmetic (lt, eq, min, max, red, div, add, sub, mul) and every composition
    f(a, g(b, c)) of its two value operations (+, x);
  * every operation is posed on every pattern of its arity. A law abstains wherever no fact (x' <= N', equal digit
    strings), identity (the division law, sub.inverse, sub.self, sub.identity, red.zero, add.identity, mul.zero) or
    practised routine of the learner (its addition, repeated; its chain for the base form) gives its value, and where
    its largest value would not fit the registers.
"""

# ---8<--- book
# The book, generated (integration/sp_ablation/SP_ABLATION.md §5). A pattern: the instance kinds (one data term per
# operand; probe i is of kind i mod len), how many digits its numbers need beyond the operands' width w ("carry":
# w + 1; "double": 2 w, wherever a number multiplies: the widest w that fits the registers), and a row filter ("ne":
# only rows with x' != N', where the base form's bound needs x' < N').
QS = ("1", "d", "o", "x")                 # the counts q of r + q b: one, a digit, another row's x, x itself
NFORM = {"x+n": "1", "x+dn": "d", "x+on": "o", "x+xn": "x"}       # r + q N with r = x (x' <= N')
BFORM = {"Bx+d": "x", "Bo+d": "o"}                                # r + q B with r = d (d < B)
PATTERNS = {"pair": ((("x", "n"), ("n", "x"), ("x", "x"), ("x", "z")), "carry", "")}
PATTERNS.update({t: (((t, "n"),), "carry" if q in ("1", "d") else "double", "") for t, q in NFORM.items()})
PATTERNS.update({t: (((t, "n"),), "carry", "ne" if q == "x" else "") for t, q in BFORM.items()})
PATTERNS.update({f"x,{q}": ((("x", q),), "carry" if q == "d" else "double", "") for q in QS[1:]})
PATTERNS.update({f"o,x,{q}": ((("o", "x", q),), "carry" if q == "d" else "double", "") for q in QS[1:]})
# The operations: what each answers ("c": the flag, "A": a number), each posed on every pattern of its arity
OPS2 = {"lt": "c", "eq": "c", "min": "A", "max": "A", "red": "A", "div": "A", "add": "A", "sub": "A", "mul": "A"}
OPS3 = {f"{f}.{g}": "A" for f in ("add", "mul") for g in ("add", "mul")}          # f(a, g(b, c))
LAW_BOOK = {law: (ans, tuple(p for p in PATTERNS if len(PATTERNS[p][0][0]) == ar))
            for ar, ops_ in ((2, OPS2), (3, OPS3)) for law, ans in ops_.items()}
GROW = {"same": 0, "carry": 1, "double": 2}
# ---8<--- end book

# ---8<--- draw
class Draw:
    """One draw of the data's numbers for a pattern at a setting: n rows' top parts at the pattern's width (x' and N',
    so x' <= N'), another row's x' (o), the digits 0..B-1 over the probes (d), zero; the composite terms are built on
    demand by the learner's routines (NotPosable when one is missing)."""

    def __init__(self, pattern, st, n, rows, gen, device):
        kinds, grow, filt = PATTERNS[pattern]
        P, B = st.P, st.B
        Nd, Xd = rows
        w = width_for(grow, st)
        pick = torch.randint(0, Nd.shape[0], (4 * n,), generator=gen, device=device)
        Nt, xt = top_parts(Nd[pick], Xd[pick], w)
        eq = (Nt == xt).all(1)
        idx = torch.arange(n, device=device)
        if filt == "ne":
            idx = torch.nonzero(~eq[:3 * n]).flatten()[:n]
        self.n = idx.numel()
        self.x, self.N = pad(xt[idx], P), pad(Nt[idx], P)
        self.o = pad(xt[3 * n:3 * n + self.n], P)
        self.eq = eq[idx]
        self.xz = (self.x == 0).all(1)
        self.dv = torch.arange(self.n, device=device) % B
        self.d = digit_string(self.dv, P)
        self.z = torch.zeros_like(self.x)
        self.kind = torch.arange(self.n, device=device) % len(kinds)
        self.st, self.gen, self.device, self.pattern, self.w = st, gen, device, pattern, w
        self.cache = {}

    def count(self, q):
        """The count q per probe: one, the digit, another row's x, or x (how many times an addition repeats)."""
        if q == "1":
            return torch.ones(self.n, dtype=torch.long, device=self.device)
        return self.dv if q == "d" else count_of(self.o if q == "o" else self.x)

    def term(self, name, ops, lib):
        if name in self.cache:
            return self.cache[name]
        st, gen, dev = self.st, self.gen, self.device
        if name in ("x", "z", "o", "d"):
            v = getattr(self, name)
        elif name == "n":
            v = self.N
        elif name in NFORM:                    # x + q N: q additions of N to x (the learner's addition)
            v = repeat_add(ops, st, lib, self.x, self.N, self.count(NFORM[name]), gen, dev)
        elif name in BFORM:                    # d + q B: q's digits moved up one place, d below
            q = self.x if BFORM[name] == "x" else self.o
            v = torch.cat([self.dv[:, None], q[:, :st.P - 1]], 1)
        else:
            raise ValueError(name)
        self.cache[name] = v
        return v

    def operands(self, ops, lib):
        """The operands' digits [n, P] per operand position, each probe of its kind."""
        kinds = PATTERNS[self.pattern][0]
        out = []
        for j in range(len(kinds[0])):
            v = torch.zeros_like(self.x)
            for k, kd in enumerate(kinds):
                m = self.kind == k
                if bool(m.any()):
                    v[m] = self.term(kd[j], ops, lib)[m]
            out.append(v)
        return out

    def nform(self, name):
        """A term as r + q N on every probe, by the facts (r: x or zero; x' = N' makes x into N, x' = 0 into zero):
        (r is x, q, the digits of the quotient by N, where those are read off), or None for a term of no such form
        (o, a digit, a base form)."""
        zero = torch.zeros(self.n, dtype=torch.long, device=self.device)
        one, every = digit_string(zero + 1, self.st.P), torch.ones(self.n, dtype=torch.bool, device=self.device)
        if name == "z":
            return zero.bool(), zero, self.z, every
        if name == "n":
            return zero.bool(), zero + 1, one, every
        if name == "x":                        # x div N: 0, or 1 where x' = N'
            return ~self.xz & ~self.eq, self.eq.long(), torch.where(self.eq[:, None], one, self.z), every
        if name in NFORM:                      # (x + q N) div N: q's digits, but q + 1 where x' = N' (not read off)
            q = self.count(NFORM[name])
            qd = {"1": one, "d": self.d, "o": self.o, "x": self.x}[NFORM[name]]
            return ~self.xz & ~self.eq, q + self.eq.long(), qd, ~self.eq
        return None

    def top(self, name):
        """The largest value a term takes at this draw's width (bookkeeping for the registers' room, as width_for)."""
        big = 10 ** self.w - 1
        if name in ("x", "o", "n"):
            return big
        if name == "d":
            return self.st.B - 1
        if name == "z":
            return 0
        if name in NFORM:
            return big + {"1": 1, "d": self.st.B - 1, "o": big, "x": big}[NFORM[name]] * big
        if name in BFORM:
            return (self.st.B - 1) + self.st.B * big
        raise ValueError(name)


def law_target(law, d, ops, lib):
    """The law's target on a draw: {"A": digits [n, P], "Am": mask [n]} and/or {"c": [n], "cm": mask} and/or a "check"
    (a function of the candidates' A digits [S, n, P] -> bool [S]), from facts, identities and the learner's routines,
    the same rules for every law on every pattern; None when the law says nothing on this draw (or its answers do not
    vary)."""
    kinds = PATTERNS[d.pattern][0]
    k, P, dev, st, gen = d.kind, d.st.P, d.device, d.st, d.gen
    room = d.st.B ** P                          # a value fits the registers when below this
    A = torch.zeros(d.n, P, dtype=torch.long, device=dev)
    Am = torch.zeros(d.n, dtype=torch.bool, device=dev)
    c = torch.zeros(d.n, dtype=torch.long, device=dev)
    cm = torch.zeros(d.n, dtype=torch.bool, device=dev)
    check_rows = None
    val = lambda name: d.term(name, ops, lib)
    for ki, terms in enumerate(kinds):
        sel = k == ki
        if not bool(sel.any()):
            continue
        got = None                              # (A or c, where it speaks), or the check's rows
        if len(terms) == 3:                     # f(a, g(b, c)): the learner's addition, repeated for a count
            a, b, q = terms
            f, g = law.split(".")
            if g == "mul" and q not in ("d", "o", "x"):
                continue
            inner = (d.top(b) + d.top(q)) if g == "add" else d.top(b) * d.top(q)
            top = (d.top(a) + inner) if f == "add" else d.top(a) * inner
            if top >= room:
                continue
            if g == "add":
                v = ops.run("add", "pair", st, lib, [val(b), val(q)], gen, dev)[0]
            else:
                v = repeat_add(ops, st, lib, d.z, val(b), d.count(q), gen, dev)
            if f == "add":
                v = ops.run("add", "pair", st, lib, [val(a), v], gen, dev)[0]
            else:
                v = repeat_add(ops, st, lib, d.z, val(a), count_of(v), gen, dev)
            got = ("A", v, sel)
        else:
            a, b = terms
            na, nb = d.nform(a), d.nform(b)
            if law in ("lt", "eq", "min", "max") and na and nb:       # the order of r + q N forms, by the facts
                (ra, qa, _, _), (rb, qb, _, _) = na, nb
                lt = (qa < qb) | ((qa == qb) & ~ra & rb)
                same = (qa == qb) & (ra == rb)
                if law in ("lt", "eq"):
                    got = ("c", (lt if law == "lt" else same).long(), sel)
                else:
                    small = lt | same
                    pick = small if law == "min" else ~small
                    got = ("A", torch.where(pick[:, None], val(a), val(b)), sel)
            elif law in ("red", "div") and a == b:                   # red.zero; a div a = 1 (a != 0)
                if law == "red":
                    got = ("A", d.z, sel)
                elif a in ("x", "o"):
                    got = ("A", digit_string(torch.ones(d.n, dtype=torch.long, device=dev), P),
                           sel & ~(val(a) == 0).all(1))
            elif law == "red" and b == "n" and na:                    # the division law: red(r + q N) = r
                got = ("A", torch.where(na[0][:, None], d.x, d.z), sel)
            elif law == "div" and b == "n" and na:                    # ... and its quotient q (where it is read off)
                got = ("A", na[2], sel & na[3])
            elif law == "red" and b == "n" and BFORM.get(a) == "x":   # d + x B < B N for x' < N': the chain's range
                src = "x+dn" if ops.has("red", "x+dn") else "x+on"
                got = ("A", ops.run("red", src, st, lib, [val(a), d.N], gen, dev)[0], sel)
            elif law == "add" and b == "z":                           # add.identity
                got = ("A", val(a), sel)
            elif law == "add" and (a, b) == ("x", "n"):               # sub.inverse, by the learner's red and lt
                if not (ops.has("red", "pair") and ops.has("lt", "pair")):
                    raise NotPosable("add needs red and lt")
                check_rows = sel & ~d.eq
                got = ("A", d.x, sel & False)
            elif law == "sub" and a == b:                             # sub.self
                got = ("A", d.z, sel)
            elif law == "sub" and b == "z":                           # sub.identity
                got = ("A", val(a), sel)
            elif law == "sub" and b == "n" and a in NFORM:            # (x + q N) - N = x + (q - 1) N, q >= 1
                q = d.count(NFORM[a])
                got = ("A", repeat_add(ops, st, lib, d.x, d.N, (q - 1).clamp(min=0), gen, dev), sel & (q >= 1))
            elif law == "mul" and b == "z":                           # mul.zero
                got = ("A", d.z, sel)
            elif law == "mul" and b in ("d", "o", "x") and a in ("x", "o", "n") and d.top(a) * d.top(b) < room:
                got = ("A", repeat_add(ops, st, lib, d.z, val(a), d.count(b), gen, dev), sel)   # mul.distribute
        if got is None:
            continue
        kind, v, m = got
        if kind == "c":
            c, cm = torch.where(m, v, c), cm | m
        else:
            A, Am = torch.where(m[:, None], v, A), Am | m
    t = {}
    if bool(Am.any()) or check_rows is not None:
        t.update(A=A, Am=Am)
    if bool(cm.any()):
        t.update(c=c, cm=cm)
    if check_rows is not None:
        sel = torch.nonzero(check_rows).flatten()
        x, N = d.x, d.N

        def check(A_out, lib):                 # sub.inverse: red(a + b, b) = a and a + b >= b, for a < b
            S = A_out.shape[0]
            if sel.numel() == 0:
                return torch.ones(S, dtype=torch.bool, device=dev)
            flat = A_out[:, sel].reshape(-1, P).long()
            Ns = N[sel].repeat(S, 1)
            ok = torch.zeros(S, dtype=torch.bool, device=dev)
            for i in range(ops.count("red", "pair")):     # any practised red that confirms it
                r = ops.run("red", "pair", st, lib, [flat, Ns], gen, dev, i)[0].view(S, -1, P)
                ok |= (r == x[sel][None]).all(-1).all(-1)
            cc = ops.run("lt", "pair", st, lib, [flat, Ns], gen, dev)[1].view(S, -1)
            return ok & (cc == 0).all(-1)
        t["check"] = check
    if not t:
        return None
    if "A" in t:                               # the answers must vary where the law speaks (else nothing to learn)
        m = t["Am"]
        if int(m.sum()) < 2 and "check" not in t:
            return None
        if "check" not in t and bool((t["A"][m] == t["A"][m][:1]).all()):
            return None
    if "c" in t:
        cm = t["cm"]
        if bool((t["c"][cm] == t["c"][cm][:1]).all()):
            return None
    return t
# ---8<--- end draw
