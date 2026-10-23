"""Builds general_lessons/submission_general_v3.py from submission_general_v2.py (md5 30d15786...), which it never
modifies. Every change is one replacement below (an exact text that must occur once), so the diff from v2 is this
file (FIXED_MODULUS.md).

    ~/github/sotaku/venv/bin/python general_lessons/fixed/make_v3.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "submission_general_v2.py"
OUT = HERE.parent / "submission_general_v3.py"
BASE_MD5 = "30d15786764bcece1ccdd11957a1bb7a"


def sub1(src: str, old: str, new: str) -> str:
    n = src.count(old)
    if n != 1:
        raise ValueError(f"expected one occurrence, found {n}: {old[:80]!r}")
    return src.replace(old, new)


V3_DOC = '''"""GeneralLessons v3 (cpu_first/general_lessons/fixed/FIXED_MODULUS.md): submission_general_v2.py with whole digit
codes chosen by the answers where the mined facts cannot choose them. v2 learned which digit token means which value,
and which way N and x are written, from x < N across many moduli: the reader's ranking (IO.md) orders two digit tokens
when they meet at the highest place where x and N differ. With one fixed modulus that fact orders only digits against
N's own, so thousands to millions of whole (reading, code) candidates pass it, the reader could not choose, and the
subtraction table, whose borrows came from the same fact, never became exact (LESSONS.md section 4). v3 changes three
things, and only where the facts leave more than one whole candidate (on H1 they leave one after the first batch, and
v3 is v2 there step for step):

  * the reader's digit-order facts (digit_facts): from io stage 3 on, for every reading hypothesis, the token pairs the
    x < N lesson orders at the decisive place, the tokens met where x has ended (not coded 0), and whether the
    hypothesis failed a row whatever the code. They give the readings the facts leave, the codes each leaves
    (code_count, a count of the orders' linear extensions) and whether more than one whole (reading, code) passes
    (ambiguous). Where it does, the reading mixture credits only the readings the facts leave, and the reading is
    chosen once the mixture's posterior sits on them, not on one of them (the answers choose among those later);
  * the reader's ranking ties the table's borrows: where the facts leave the code ambiguous, sub.order also runs on
    each read number and another row's (x against the next row's x, and N, T likewise), the premise "a < b" being the
    reader's own comparison of the two label strings (compare, the miner's), on every top part and in the facts the
    digit lessons wait for. Every reader's label is a rank among the digit tokens, so this teaches the standard
    subtraction on labels whichever code the reader holds: the machine becomes exact before the code is right
    (fixed/tiny_tie.py), and whole codes can then be scored through it;
  * where every row of the batch has the same N, the multiply's routed ladder also runs on the machine's own square of
    x (its low W digits): with one modulus half the ladder's numbers are one number, and on three-digit moduli the
    rules that pile every digit of the multiplicand into one column lost too little to lose (E10: the high digits'
    rule stuck at column j + 1 with the ladder's checks at 40%);
  * the code tournament (code_lesson): once every admissible lesson set's machine passes its label-free checks, every
    whole (reading, code) the facts leave (census: all 10! codes filtered by the facts) is a contestant, with every
    lesson set and every combination of the round menu. Each step, every surviving (contestant, set, round) triple runs
    the machine on the batch's rows with one-digit T, read through the contestant's code, against the answer read
    through it; a triple that fails a row is out (the machine's verdicts are exact by then, so a failure is final).
    The answers' mixture over the survivors' sets and round options trains the selector and the round menu, as v2's
    round lesson does under the reader's code. Rows go one at a time, those with the fewest distinct digit tokens
    first, and the machine runs once per distinct input (the same verdicts as running every triple on every row).
    When the survivors share one code (and one reading, or the reader's own is among theirs), or stop changing for
    MINE_WIN batches, that whole code is the answers' choice (ctl.leader); if nothing survives, the census is made
    again WIRE_WIN steps later. The chosen code's order of the digits becomes a Bradley-Terry lesson for that
    reading's ranking, and the reading lesson becomes that reading's own log-probability, field by field (the
    mixture's pull toward a reading is proportional to the weight the pointers still give it, which after hundreds
    of steps on a provisional reading is too small to move them), with the pointers' shared shape held, as v2's T
    lesson holds it (unheld, the pull switched every field to the offsets shape, whose writer slots past a short
    answer were never trained and wrote over it), so the reader learns both by gradient like any other fact, and
    the prediction path is v2's. While the facts leave the code open, the writer's pointer lesson covers every slot,
    continuing the answer's LSB-first order past its end, not only the answer's slots and the one past it. Until then the writer's code has no
    lesson (its pointer has): a code learned for the reader's provisional ranking would only have to be unlearned.

v2's description follows.

'''

HELPERS = '''

def code_count(below: list, nonmin: int) -> int:
    """How many codes (bijections of the ten digit tokens to the labels 0..9) keep the reader's digit-order facts:
    below[t] the bitmask of the tokens that must be coded lower than token t, nonmin the bitmask of the tokens that
    must not be coded 0. The linear extensions, counted by a DP over subsets of tokens placed from label 0 up (0: the
    facts contradict each other)."""
    f = [0] * 1024
    f[0] = 1
    for S in range(1024):
        if not f[S]:
            continue
        for t in range(10):
            if S >> t & 1 or below[t] & ~S or (S == 0 and nonmin >> t & 1):
                continue
            f[S | 1 << t] += f[S]
    return f[1023]


_ALL_CODES = None


def all_codes() -> Tensor:
    """Every code: the labels of the ten digit tokens (in token order) [10!, 10] int8, on the CPU, made once."""
    global _ALL_CODES
    if _ALL_CODES is None:
        order = torch.zeros(1, 0, dtype=torch.int8)                       # the tokens listed from label 0 up
        for t in range(10):
            n, L = order.shape
            new = torch.empty(n, L + 1, L + 1, dtype=torch.int8)
            for p in range(L + 1):
                new[:, p] = torch.cat([order[:, :p], torch.full((n, 1), t, dtype=torch.int8), order[:, p:]], 1)
            order = new.reshape(n * (L + 1), L + 1)
        code = torch.empty_like(order)
        code.scatter_(1, order.long(), torch.arange(10, dtype=torch.int8).expand_as(order).contiguous())
        _ALL_CODES = code
    return _ALL_CODES


def label_number(D: Tensor) -> Tensor:
    """Label digits [n, L] (LSB first) as one number per row [n] (L <= 9)."""
    return (D.long() * 10 ** torch.arange(D.shape[1], device=D.device)).sum(1)


def menu_size'''

CONTROL = '''        self.io_parts = ""
        # v3 (FIXED_MODULUS.md): the reader's digit-order facts and the code tournament
        self.dtok: Tensor | None = None       # the digit tokens the facts were gathered with
        self.dpairs: Tensor | None = None     # per reading hypothesis: token a is coded lower than token b [H, V, V]
        self.dnonmin: Tensor | None = None    # ... token b met where x has ended: not coded 0 [H, V]
        self.dfail: Tensor | None = None      # ... the hypothesis failed a row whatever the code [H]
        self.dseen = 0                        # batches the facts were gathered on
        self.dcount: list = []                # per hypothesis: the codes its facts leave
        self.dsig: list = []                  # ... and the facts they were counted from
        self.readings: list = []              # the hypotheses the facts leave (some code passes)
        self.ambiguous = False                # more than one whole (reading, code) passes the facts
        self.pinned = False                   # the facts left exactly one (reported once)
        self.cands = None                     # the code tournament's contestants: (hypothesis [C], code [C, 10])
        self.alive: dict | None = None        # per reading: its surviving (contestant, set, combination) triples
                                              # [n, 3], or None while none of its rows has been scored
        self.code_runs = 0                    # machine runs the code tournament made
        self.code_wait = 0                    # no census before this step (after one that fit nothing)
        self.alive_same = 0                   # batches in a row the survivors did not change
        self.leader = None                    # the answers' whole code: (hypothesis, labels per token [V])
        self.adopted = False                  # the reader reads with it
        self.code_hist: list = []             # (step, event, details) for the report'''

IO_FACTS = '''        ctl.best = int(passes.argmax())
        if FIXED_CODES and ctl.io >= 3:                                    # v3: the reader's digit-order facts
            with torch.no_grad():
                self.digit_facts(ta, tb, both, padx, tfits, ok)
        if FIXED_CODES and ctl.leader is not None:                         # v3: the answers' whole code, a lesson
            h0, lab = ctl.leader                                           # for that reading's ranking
            dt = torch.nonzero(isd).flatten()
            la = lab[dt]
            lo_, hi_ = torch.nonzero(la[:, None] < la[None, :], as_tuple=True)
            loss = loss - F.logsigmoid((u[h0, dt[hi_]] - u[h0, dt[lo_]]) / BT_TAU).mean()'''

IO_LK = '''            Lk = hp.float() + EPS
            if FIXED_CODES and ctl.ambiguous and ctl.leader is None:       # v3: only readings the facts leave
                keep_h = torch.zeros(len(HYP), dtype=torch.bool, device=dev)
                keep_h[ctl.readings] = True
                Lk = torch.where(keep_h[None], Lk, EPS)'''

IO_CHOSEN_OLD = '''            ctl.chosen = float((post / norm.detach()[:, None].clamp_min(1e-30)).max(1).values[ok].mean())'''
IO_CHOSEN_NEW = '''            pn = post / norm.detach()[:, None].clamp_min(1e-30)
            if FIXED_CODES and ctl.ambiguous and ctl.leader is None:
                ctl.chosen = float(pn[:, ctl.readings].sum(1)[ok].mean())   # v3: the readings the facts leave
            else:
                ctl.chosen = float(pn.max(1).values[ok].mean())'''

TIE_SOFT_OLD = '''                loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
'''
TIE_SOFT_NEW = '''                loss = loss - (torch.where(differ, log(p1), log(1 - p1)).mean() + log(1 - p2).mean()) / P
        # v3: where the facts leave the code ambiguous, the reader's ranking ties the borrows: sub.order on each read
        # number and another row's, the premise "a < b" being the reader's comparison of the two label strings
        ties = []
        if FIXED_CODES and ctl.ambiguous and usable:
            for f_ in ORDER_TERMS:
                A_ = terms[f_]
                B_ = A_.roll(1, 0)
                ties += [(A_, B_), (B_, A_)]
            for A_, B_ in ties:
                for j in range(P):
                    lt = compare(A_[:, j:], B_[:, j:]) < 0
                    pb = link_soft(hot(A_[:, j:]), B_[:, j:], TDC)[1][:, -1, 1]
                    loss = loss - torch.where(lt, log(pb), log(1 - pb)).mean() / P
'''

TIE_STREAK_OLD = '''                ctl.facts_streak = ctl.facts_streak + 1 if bool(okf.all()) else 0'''
TIE_STREAK_NEW = '''                for A_, B_ in ties:
                    okf = okf & (sub_hard(A_, B_, Td, Tb)[1] == (compare(A_, B_) < 0).long())
                ctl.facts_streak = ctl.facts_streak + 1 if bool(okf.all()) else 0'''

TIE_USED_OLD = '''                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0'''
TIE_USED_NEW = '''                used = torch.bincount(sub_hard(D, Mr, Td, Tb)[2].flatten(), minlength=200) > 0
                for A_, B_ in ties:
                    used = used | (torch.bincount(sub_hard(A_, B_, Td, Tb)[2].flatten(), minlength=200) > 0)'''

TIE_CHECK_OLD = '''            if ANCHOR_WRAP > 0:
                for a_, b_ in orders:
                    Ewh, _, idw = sub_hard(terms[a_], terms[b_], Td, Tb)'''
TIE_CHECK_NEW = '''            for A_, B_ in ties:                                              # v3: the tie's verdicts
                _, btie, idt = sub_hard(A_, B_, Td, Tb)
                tally(ctl.visits, idt, 200)
                table = min(table, float((btie == (compare(A_, B_) < 0).long()).float().mean()))
            if ANCHOR_WRAP > 0:
                for a_, b_ in orders:
                    Ewh, _, idw = sub_hard(terms[a_], terms[b_], Td, Tb)'''

CALL_OLD = '''            loss = loss + model.round_lesson(aux, yd)'''
CALL_NEW = '''            loss = loss + model.round_lesson(aux, yd)
        if FIXED_CODES and ctl.ambiguous and ctl.leader is None and MODE == "learned":   # v3: whole codes
            loss = loss + model.code_lesson(aux, labels, valid, tpos)'''

REPORT_OLD = '''        print(f"MINED {ctl.miner.describe()}", flush=True)'''
REPORT_NEW = '''        print(f"MINED {ctl.miner.describe()}", flush=True)
        if FIXED_CODES:
            counts = {"".join(map(str, HYP[h])): ctl.dcount[h] for h in ctl.readings}
            lead = None if ctl.leader is None else ("".join(map(str, HYP[ctl.leader[0]])),
                                                    ctl.leader[1][ctl.dtok].tolist())
            print(f"CODES ambiguous={ctl.ambiguous} readings={counts} leader={lead} adopted={ctl.adopted} "
                  f"history={ctl.code_hist}", flush=True)'''

METHODS = '''    # ------------------------------------------------------------------ v3: whole digit codes (FIXED_MODULUS.md)
    def digit_facts(self, ta: Tensor, tb: Tensor, both: Tensor, padx: Tensor, tfits: Tensor, ok: Tensor) -> None:
        """The reader's digit-order facts, gathered per reading hypothesis on the rows read as three runs (io_lessons'
        decisive places: ta, tb the tokens of x and N there [X, H], -1 where a run has ended): the pairs (ta, tb) where
        both are digits (ta is coded lower), tb where x has ended (not coded 0), and the hypotheses that fail a row
        whatever the code (x equal to N, N ending first, T longer than its slots). Then the codes each surviving
        hypothesis leaves (counted when its facts change), the readings left, and ambiguous: more than one whole
        (reading, code) once MINE_WIN batches have been seen."""
        ctl = self.ctl
        dev, V, H = ta.device, self.V, len(HYP)
        isd = (self.dg > 0).detach()
        if ctl.dtok is None or not torch.equal(ctl.dtok, isd):              # (re)start with these digit tokens
            ctl.dtok = isd.clone()
            ctl.dpairs = torch.zeros(H, V, V, dtype=torch.bool, device=dev)
            ctl.dnonmin = torch.zeros(H, V, dtype=torch.bool, device=dev)
            ctl.dfail = torch.zeros(H, dtype=torch.bool, device=dev)
            ctl.dseen, ctl.dcount, ctl.dsig = 0, [None] * H, [None] * H
            ctl.cands, ctl.alive = None, None                                # a census of other tokens is void
        hh = torch.arange(H, device=dev)[None].expand_as(ta)
        m = both & ok[:, None]
        ctl.dpairs[hh[m], ta[m], tb[m]] = True
        m = padx & ok[:, None]
        ctl.dnonmin[hh[m], tb[m]] = True
        ctl.dfail |= ((~(both | padx) | ~tfits) & ok[:, None]).any(0)
        ctl.dseen += 1
        dt = torch.nonzero(isd).flatten()
        if dt.numel() != 10:
            ctl.readings, ctl.ambiguous = [], False
            return
        pairs = ctl.dpairs[:, dt][:, :, dt].cpu()
        nonmin = ctl.dnonmin[:, dt].cpu()
        fail = ctl.dfail.cpu()
        for h in range(H):
            if bool(fail[h]) or ctl.dcount[h] == 0:                          # more facts cannot undo a contradiction
                ctl.dcount[h] = 0
                continue
            sig = (pairs[h].numpy().tobytes(), nonmin[h].numpy().tobytes())
            if sig != ctl.dsig[h]:
                ctl.dsig[h] = sig
                below = [sum(1 << a for a in range(10) if bool(pairs[h, a, b])) for b in range(10)]
                ctl.dcount[h] = code_count(below, sum(1 << b for b in range(10) if bool(nonmin[h, b])))
        ctl.readings = [h for h in range(H) if ctl.dcount[h]]
        total = sum(ctl.dcount[h] for h in ctl.readings)
        amb = ctl.dseen >= MINE_WIN and total > 1
        if amb != ctl.ambiguous:
            counts = {"".join(map(str, HYP[h])): ctl.dcount[h] for h in ctl.readings}
            print(f"LR digit_codes ambiguous={amb} step={ctl.step} readings={counts}", flush=True)
            ctl.code_hist.append((ctl.step, "ambiguous" if amb else "pinned_again", counts))
        ctl.ambiguous = amb
        if ctl.dseen >= MINE_WIN and total == 1 and not ctl.pinned:
            ctl.pinned = True
            print(f"LR digit_codes pinned_by_facts step={ctl.step} reading={HYP[ctl.readings[0]]}", flush=True)

    def census(self, dev) -> int:
        """The code tournament's contestants: every reading the facts leave with every code its facts leave (all 10!
        codes, filtered). Returns their number."""
        ctl = self.ctl
        dt = torch.nonzero(ctl.dtok).flatten()
        perms = all_codes()
        hs, cs = [], []
        for h in ctl.readings:
            keep = torch.ones(perms.shape[0], dtype=torch.bool)
            for a, b in torch.nonzero(ctl.dpairs[h][dt][:, dt].cpu()).tolist():
                keep &= perms[:, a] < perms[:, b]
            for b in torch.nonzero(ctl.dnonmin[h][dt].cpu()).flatten().tolist():
                keep &= perms[:, b] != 0
            c = perms[keep]
            hs.append(torch.full((c.shape[0],), h, dtype=torch.long))
            cs.append(c)
        ctl.cands = (torch.cat(hs).to(dev), torch.cat(cs).to(dev))
        ctl.alive, ctl.alive_same = None, 0
        n = int(ctl.cands[0].numel())
        counts = {"".join(map(str, HYP[h])): int(c.shape[0]) for h, c in zip(ctl.readings, cs)}
        print(f"LR code_census step={ctl.step} contestants={n} readings={counts}", flush=True)
        ctl.code_hist.append((ctl.step, "census", counts))
        return n

    def code_machine(self, xs: Tensor, Ns: Tensor, Ts: Tensor, sets: Tensor, combos: Tensor, tabs) -> Tensor:
        """The hard machine on label digits (x, N [n, W], T [n, K]) with each row's lesson set and round combination
        (its index into ROUND_COMBOS): the result's label digits [n, W], in chunks of 65,536 rows."""
        dev = xs.device
        out = torch.zeros_like(xs)
        rc = torch.tensor(ROUND_COMBOS, device=dev)
        for s in sets.unique().tolist():
            idx = torch.nonzero(sets == s).flatten()
            for a in range(0, idx.numel(), 65536):
                ii = idx[a:a + 65536]
                w = self.wiring(ii.numel(), dev, s)
                for i, k in enumerate(ROUND_SW):
                    w[k] = rc[combos[ii], i]
                out[ii] = run_hard(xs[ii], Ts[ii], Ns[ii], tabs, codes(w), cap=int(Ts[ii, 0].max()) + 1)[0]
        return out

    def code_rows(self, aux: dict, labels: Tensor, valid: Tensor, tpos: Tensor) -> dict:
        """The batch's tokens as the code tournament reads them: every run in both directions (digit indices 0..9 of
        the reader's digit tokens, -1 past the run's end) [X, run, dir, W], the answer at the writer's slots [X, W]
        (LSB first), the rows read as three runs [X] and the runs' lengths [X, 3]."""
        ctl = self.ctl
        e, end = aux["e"], aux["end"]
        dev = e.device
        X, Lp = e.shape
        ok, si, ti, length = self.runs(e, end)
        j = torch.arange(W, device=dev)
        tokp = e.gather(1, (si[..., None] + j).clamp(max=Lp - 1).reshape(X, -1)).view(X, 3, W)
        tokm = e.gather(1, (ti[..., None] - j).clamp(min=0).reshape(X, -1)).view(X, 3, W)
        liv = j[None, None] < length[..., None]
        Rt = torch.stack([torch.where(liv, tokp, -1), torch.where(liv, tokm, -1)], 2)
        at = self.label_seq(e, labels, valid, tpos).gather(1, self.hard_pos(3, e, end))
        At = torch.where(self.isdig(at).long().cumprod(-1).bool(), at, -1)
        t2i = torch.full((self.OUT + 1,), -1, dtype=torch.long, device=dev)
        t2i[torch.nonzero(ctl.dtok).flatten()] = torch.arange(10, device=dev)
        ix = lambda t: torch.where(t >= 0, t2i[t.clamp(min=0)], -1)
        return {"R": ix(Rt), "A": ix(At), "ok": ok, "length": length}

    def code_score(self, h: int, tri: Tensor | None, cidx: Tensor, rows: Tensor, b: dict, ready: list, tabs) -> Tensor:
        """The surviving triples of reading h after this batch's rows `rows` (fewest distinct digit tokens first).
        tri [n, 3] (contestant, set, combination), or None for every contestant cidx with every ready set and
        combination: the first row then runs once per distinct (x, N, T) under every set and combination. Then one
        row at a time while more than CODE_ROWS triples survive, then the next CODE_BATCH rows at once; the machine runs
        once per distinct input (x, N, T, set, combination)."""
        rn, dn, rx, dx = HYP[h]
        rt = 3 - rn - rx
        R, A = b["R"], b["A"]
        Ni, xi, Ti = R[:, rn, dn], R[:, rx, dx], R[:, rt, 0, :K]
        code_c = self.ctl.cands[1]
        dev = R.device
        K_ = len(ROUND_COMBOS)
        lab = lambda cc, D: torch.where(D[None] >= 0, cc[:, D.clamp(min=0)], torch.zeros((), dtype=cc.dtype,
                                                                                          device=dev)).long()
        r_at = 0
        if tri is None:
            r = int(rows[0])
            cc = code_c[cidx]
            xs, Ns_, Ts_, yv = lab(cc, xi[r]), lab(cc, Ni[r]), lab(cc, Ti[r]), label_number(lab(cc, A[r]))
            key = torch.stack([label_number(xs), label_number(Ns_), label_number(Ts_)], 1)
            uk, inv = key.unique(dim=0, return_inverse=True)
            first = torch.zeros(uk.shape[0], dtype=torch.long, device=dev).scatter_(
                0, inv, torch.arange(inv.numel(), device=dev))
            G, S_ = uk.shape[0], len(ready)
            gg = torch.arange(G, device=dev).repeat(S_ * K_)
            ss = torch.tensor(ready, device=dev).repeat_interleave(K_ * G)
            kk = torch.arange(K_, device=dev).repeat_interleave(G).repeat(S_)
            on = label_number(self.code_machine(xs[first][gg], Ns_[first][gg], Ts_[first][gg], ss, kk, tabs))
            on = on.view(S_ * K_, G)
            found = []
            for q in range(S_ * K_):
                hit = torch.nonzero(on[q][inv] == yv).flatten()
                if hit.numel():
                    found.append(torch.stack([cidx[hit], torch.full_like(hit, ready[q // K_]),
                                              torch.full_like(hit, q % K_)], 1))
            tri = torch.cat(found) if found else torch.zeros(0, 3, dtype=torch.long, device=dev)
            self.ctl.code_runs += S_ * K_ * G
            r_at = 1
        while tri.shape[0] and r_at < rows.numel():
            take = rows[r_at:r_at + (1 if tri.shape[0] > CODE_ROWS else CODE_BATCH)]
            r_at += take.numel() if tri.shape[0] > CODE_ROWS else rows.numel()
            m, n = tri.shape[0], take.numel()
            cc = code_c[tri[:, 0]]
            xs = lab(cc, xi[take].reshape(-1)).view(m, n, -1).reshape(m * n, -1)
            Ns_ = lab(cc, Ni[take].reshape(-1)).view(m, n, -1).reshape(m * n, -1)
            Ts_ = lab(cc, Ti[take].reshape(-1)).view(m, n, -1).reshape(m * n, -1)
            ys = label_number(lab(cc, A[take].reshape(-1)).view(m, n, -1).reshape(m * n, -1))
            st, ct = tri[:, 1].repeat_interleave(n), tri[:, 2].repeat_interleave(n)
            key = torch.stack([label_number(xs), label_number(Ns_), label_number(Ts_), st, ct], 1)
            uk, inv = key.unique(dim=0, return_inverse=True)
            first = torch.zeros(uk.shape[0], dtype=torch.long, device=dev).scatter_(
                0, inv, torch.arange(inv.numel(), device=dev))
            out = label_number(self.code_machine(xs[first], Ns_[first], Ts_[first], st[first], ct[first], tabs))
            self.ctl.code_runs += uk.shape[0]
            tri = tri[(out[inv] == ys).view(m, n).all(1)]
        return tri

    def code_lesson(self, aux: dict, labels: Tensor, valid: Tensor, tpos: Tensor) -> Tensor:
        """The code tournament (the module docstring): the census once every admissible lesson set's machine passes
        its label-free checks (and the table and the multiply theirs), then each step the surviving (contestant, set,
        round) triples of every reading on the batch's rows with one-digit T (code_score). Returns the answers'
        mixture loss over the survivors' lesson sets and round options (the selector's and the round menu's)."""
        ctl = self.ctl
        zero = self.lsel.sum() * 0.0
        per = aux.get("per") or []
        sets = [c for c, p in enumerate(per) if p is not None]
        ready = [c for c in sets if ctl.wire_ios[c] and min(per[c].values()) == 1.0]
        if (not sets or len(ready) < len(sets) or aux.get("table_exact", 0.0) < 1.0 or aux.get("lad_ok", 0.0) < 1.0
                or ctl.dtok is None or int(ctl.dtok.sum()) != 10):
            return zero
        dev = aux["e"].device
        if ctl.step < ctl.code_wait:
            return zero
        with torch.no_grad():
            if ctl.cands is None:
                if self.census(dev) == 0:
                    ctl.code_wait = ctl.step + WIRE_WIN
                    return zero
                ctl.alive = {h: None for h in ctl.cands[0].unique().tolist()}
            hyp_c, code_c = ctl.cands
            b = self.code_rows(aux, labels, valid, tpos)
            before = {h: (None if t is None else t.shape[0]) for h, t in ctl.alive.items()}
            for h, tri in list(ctl.alive.items()):
                rt = 3 - HYP[h][0] - HYP[h][2]
                use = b["ok"] & (b["length"][:, rt] == 1) & (b["A"][:, 0] >= 0)
                rows = torch.nonzero(use).flatten()
                if rows.numel() == 0:
                    continue
                pres = torch.zeros(rows.numel(), 10, dtype=torch.bool, device=dev)
                rn, dn, rx, dx = HYP[h]
                for D in (b["R"][rows, rn, dn], b["R"][rows, rx, dx], b["R"][rows, rt, 0, :K]):
                    pres |= (F.one_hot(D.clamp(min=0), 10).bool() & (D >= 0)[..., None]).any(1)
                rows = rows[pres.sum(1).argsort(stable=True)]                 # fewest distinct digit tokens first
                ctl.alive[h] = self.code_score(h, tri, torch.nonzero(hyp_c == h).flatten(), rows, b, ready,
                                               aux["tabs"])
            ctl.alive = {h: t for h, t in ctl.alive.items() if t is None or t.shape[0]}
            if not ctl.alive:                                                  # nothing fits: count again next time
                print(f"LR code_none_fit step={ctl.step}", flush=True)
                ctl.code_hist.append((ctl.step, "none_fit", 0))
                ctl.cands, ctl.alive, ctl.code_wait = None, None, ctl.step + WIRE_WIN
                return zero
            after = {h: (None if t is None else t.shape[0]) for h, t in ctl.alive.items()}
            ctl.alive_same = ctl.alive_same + 1 if after == before else 0
            if any(t is None for t in ctl.alive.values()):
                return zero
            alive = torch.cat(list(ctl.alive.values()))
            conts = alive[:, 0].unique()
            lab_c = code_c[conts]
            hs = hyp_c[conts]
            kmode = int(torch.mode(aux["k"]).values)
            if after != before:
                print(f"LR code_survivors step={ctl.step} triples={alive.shape[0]} contestants={conts.numel()} "
                      f"readings={sorted(set(''.join(map(str, HYP[h])) for h in hs.tolist()))} "
                      f"machine_runs={ctl.code_runs}", flush=True)
            one_code = bool((lab_c == lab_c[:1]).all())
            if (one_code and (hs.unique().numel() == 1 or bool((hs == kmode).any()))) or ctl.alive_same >= MINE_WIN:
                pick = conts[hs == kmode][0] if bool((hs == kmode).any()) else conts[0]
                labv = torch.zeros(self.V, dtype=torch.long, device=dev)
                labv[torch.nonzero(ctl.dtok).flatten()] = code_c[pick].long()
                ctl.leader = (int(hyp_c[pick]), labv)
                tie = "none" if conts.numel() == 1 else ("readings" if one_code else "codes")
                print(f"LR code_chosen step={ctl.step} reading={HYP[ctl.leader[0]]} code={code_c[pick].tolist()} "
                      f"survivors={conts.numel()} tie={tie} machine_runs={ctl.code_runs}", flush=True)
                ctl.code_hist.append((ctl.step, "chosen", "".join(map(str, HYP[ctl.leader[0]])),
                                      code_c[pick].tolist(), int(conts.numel()), tie, ctl.code_runs))
                ctl.history.append((ctl.step, "code_chosen", int(conts.numel())))
            pairs = torch.unique(alive[:, 1:], dim=0)
        lw = self.lsel.float().log_softmax(-1)[pairs[:, 0]] + self.round_logw(torch.tensor(ROUND_COMBOS,
                                                                                          device=dev)[pairs[:, 1]])
        return -torch.logsumexp(lw, 0)

    def mine(self, terms: dict) -> None:'''

ADOPT_OLD = '''        if ctl.io != before:
            print(f"LR io_stage={ctl.io} step={ctl.step} history={ctl.history[-1]}", flush=True)'''
ADOPT_NEW = '''        if ctl.io != before:
            print(f"LR io_stage={ctl.io} step={ctl.step} history={ctl.history[-1]}", flush=True)
        if FIXED_CODES and ctl.leader is not None and not ctl.adopted:     # v3: does the reader read with it yet?
            h0, lab = ctl.leader
            isd = self.dg > 0
            ranks = self.ranks(torch.tensor([h0], device=lab.device))[0]
            kk = aux["k"][aux["k"] >= 0]
            if kk.numel() and bool((kk == h0).all()) and bool((ranks[isd] == lab[isd]).all()):
                ctl.adopted = True
                ctl.history.append((ctl.step, "code_adopted"))
                ctl.code_hist.append((ctl.step, "adopted"))
                print(f"LR code_adopted step={ctl.step}", flush=True)'''

CONSTS_OLD = '''DESIGNED_ROUND = {"rmul": "one", "radd": 0, "rcube": "none"}   # ... the round's options, set by hand'''
CONSTS_NEW = '''DESIGNED_ROUND = {"rmul": "one", "radd": 0, "rcube": "none"}   # ... the round's options, set by hand
FIXED_CODES = True     # v3: where the reader's digit-order facts leave more than one whole (reading, code), its ranking
                       # ties the table's borrows and the answers choose the whole code (the code tournament)
CODE_ROWS = 256        # the code tournament scores one row at a time while more triples than this survive, ...
CODE_BATCH = 64        # ... then this many rows of the batch at once'''

WRITER1_OLD = '''            rv = self.wsm()[vals, inv.gather(1, vals)]
            loss = loss - rv.clamp_min(EPS).log().mean()'''
WRITER1_NEW = '''            if not (FIXED_CODES and ctl.ambiguous and ctl.leader is None):   # v3: no code to unlearn later
                rv = self.wsm()[vals, inv.gather(1, vals)]
                loss = loss - rv.clamp_min(EPS).log().mean()'''
WRITER2_OLD = '''        loss = loss - wsm[code[rows_l, tl], tl].clamp_min(EPS).log().mean()   # read an answer token, write it back'''
WRITER2_NEW = '''        if not (FIXED_CODES and ctl.ambiguous and ctl.leader is None):       # v3: no code to unlearn later
            loss = loss - wsm[code[rows_l, tl], tl].clamp_min(EPS).log().mean()   # read an answer token, write it back'''

READ_OLD = '''            loss = loss - ((lik.clamp_min(1e-30).log() - norm.clamp_min(1e-30).log()) * okf).sum() / okf.sum()'''
READ_NEW = '''            if FIXED_CODES and ctl.leader is not None:                     # v3: the answers chose the reading too:
                rn0, dn0, rx0, dx0 = HYP[ctl.leader[0]]                    # its own log-probability, field by field,
                rh = self.shape_rho((0, 1, 2), e, end, si, ti, length, hold_shape=True)   # the shared shape held
                lp = (rh[:, 0, rn0, dn0].clamp_min(1e-38).log() + rh[:, 1, rx0, dx0].clamp_min(1e-38).log()
                      + rh[:, 2, 3 - rn0 - rx0].sum(-1).clamp_min(1e-38).log())
                loss = loss - (lp * okf).sum() / okf.sum()
            else:
                loss = loss - ((lik.clamp_min(1e-30).log() - norm.clamp_min(1e-30).log()) * okf).sum() / okf.sum()'''
TILE_OLD = '''            if ctl.io >= 4:                                                # whole runs, distinct fields'''
TILE_NEW = '''            if ctl.io >= 4 and not (FIXED_CODES and ctl.leader is not None):   # whole runs, distinct fields'''

LADDER_OLD = '''        lad_loss, lad_ok, verified, lad_rows = self.ladder(torch.cat([Xd, Nd]), tabs, ctl.ladder)'''
LADDER_NEW = '''        lad_in = [Xd, Nd]
        if FIXED_CODES and bool((Nd == Nd[:1]).all()):                    # v3: one N in every row repeats one example:
            lad_in.append(V[:, :W])                                         # the machine's own square of x joins
        lad_loss, lad_ok, verified, lad_rows = self.ladder(torch.cat(lad_in), tabs, ctl.ladder)'''
CHAINN_OLD = '''            div_loss, div_ok, regs = self.chain_lesson(c, Xd, Nd, lad_rows[X:, 0], verified[X:, 0], tabs, chain_on)'''
CHAINN_NEW = '''            div_loss, div_ok, regs = self.chain_lesson(c, Xd, Nd, lad_rows[X:2 * X, 0], verified[X:2 * X, 0], tabs,
                                                       chain_on)'''

WSLOT_OLD = '''        pw = (self.shape_probs(3, e, end, apos[:, None], m[:, None])[:, 0] * self.shape_logp().exp()).sum(-1)'''
WSLOT_NEW = '''        mw = torch.full_like(m, W - 1) if FIXED_CODES and ctl.ambiguous else m   # v3: every slot, past the answer too
        pw = (self.shape_probs(3, e, end, apos[:, None], mw[:, None])[:, 0] * self.shape_logp().exp()).sum(-1)'''

LAYOUT_METHOD = '''    def layout_lesson(self, Xn: Tensor, tabs) -> Tensor:
        """v3, where every row has the same N: the multiply's column rules as whole options, scored exactly. Once the
        product cell's own ladder passes (the table already does: the ladder has joined), every surviving low-digit
        rule runs the routed ladder's rung 1 on the batch's ladder numbers (X * 1: every high digit is 0, so the low rule
        alone decides it), then every surviving high-digit rule runs rungs 1..9 with the most probable surviving low
        rule, on the hard machine (the learned adder and cell). A rule that fails a rung of a row is out; the survivors
        are scored again on every later batch. The lesson is -log of each kind's total weight on its survivors: the
        ladder's blend of rules stalls on short numbers, where the rules that pile every digit of the multiplicand into
        one column lose too little (E10: the high digits' rule stuck at column j + 1)."""
        ctl = self.ctl
        zero = self.lay.sum() * 0.0
        with torch.no_grad():
            if ctl.lay_pass is None:
                if not self.cell_ladder(tabs)[1]:
                    return zero
                ctl.lay_pass = torch.ones(2, N_RULES, dtype=torch.bool, device=Xn.device)
                ctl.lay_census = ctl.step
            Td, Tb, Plo, Phi = tabs[:4]
            X = Xn[:LAYOUT_CENSUS]
            n = X.shape[0]
            maps = self.rule_maps()
            Xj = torch.stack([F.pad(X, (j, C - W - j)) for j in range(W)], 0)   # X 10^j [W, n, C]

            def rows(b: int, lo_r: Tensor, hi_r: Tensor) -> Tensor:          # rules [k] -> rows j of X * b [k, W, n, C]
                idx = X * 10 + b
                z = torch.zeros_like(X[:, :1])
                k = lo_r.numel()
                lo = torch.cat([Plo[idx], z], 1)[None, None].expand(k, W, n, W + 1)
                hi = torch.cat([Phi[idx], z], 1)[None, None].expand(k, W, n, W + 1)
                lo = lo.gather(-1, maps[lo_r][:, :, None, :].expand(k, W, n, C))
                hi = hi.gather(-1, maps[hi_r][:, :, None, :].expand(k, W, n, C))
                return add_learned(lo.reshape(-1, C), hi.reshape(-1, C), Td, Tb)[0].view(k, W, n, C)

            for kind in range(2):
                cand = torch.nonzero(ctl.lay_pass[kind]).flatten()
                if kind == 1:
                    los = torch.nonzero(ctl.lay_pass[0]).flatten()
                    if los.numel() == 0:
                        break
                    lo_best = los[self.lay[0][los].argmax()]
                for a in range(0, cand.numel(), 128):
                    rr = cand[a:a + 128]
                    if kind == 0:                                              # rung 1: the low rule alone
                        ok = (rows(1, rr, rr) == Xj[None]).flatten(1).all(1)
                    else:                                                      # rungs 1..9 with the best low rule
                        ok = torch.ones(rr.numel(), dtype=torch.bool, device=X.device)
                        prev = torch.zeros(rr.numel(), W, n, C, dtype=X.dtype, device=X.device)
                        for b in range(1, 10):
                            cur = rows(b, lo_best.expand(rr.numel()), rr)
                            tgt = add_learned(prev.reshape(-1, C), Xj[None].expand_as(prev).reshape(-1, C),
                                              Td, Tb)[0].view_as(prev)
                            ok &= (cur == tgt).flatten(1).all(1)
                            prev = cur
                    ctl.lay_pass[kind, rr] = ok
            counts = [int(v) for v in ctl.lay_pass.sum(1)]
            if counts != ctl.lay_counts:
                ctl.lay_counts = counts
                rules = MUL_MENU["lo"]
                show = lambda k: ",".join(str(rules[int(i)]).replace(" ", "") for i in torch.nonzero(ctl.lay_pass[k]).flatten()[:4])
                print(f"LR layout_survivors step={ctl.step} lo={counts[0]} ({show(0)}) hi={counts[1]} ({show(1)})",
                      flush=True)
            if min(counts) == 0:                                              # nothing fits: start again next step
                ctl.lay_pass, ctl.lay_counts = None, None
                return zero
        lw = self.lay.float().log_softmax(-1)
        return -(torch.logsumexp(lw[0][ctl.lay_pass[0]], 0) + torch.logsumexp(lw[1][ctl.lay_pass[1]], 0))

    def ladder(self, Xn: Tensor, tabs, train: bool):'''
LAYCALL_OLD = '''        lad_loss, lad_ok, verified, lad_rows = self.ladder(torch.cat(lad_in), tabs, ctl.ladder)
        if ctl.ladder:
            loss = loss + LADDER_W * lad_loss'''
LAYCALL_NEW = '''        lad_loss, lad_ok, verified, lad_rows = self.ladder(torch.cat(lad_in), tabs, ctl.ladder)
        if ctl.ladder:
            loss = loss + LADDER_W * lad_loss
            if len(lad_in) > 2:                                              # v3: one N: the layout's rules, whole
                loss = loss + self.layout_lesson(torch.cat([Xd[:31], Nd[:1], V[:32, :W]]), tabs)'''
LAYCTL_OLD = '''        self.code_wait = 0                    # no census before this step (after one that fit nothing)'''
LAYCTL_NEW = '''        self.code_wait = 0                    # no census before this step (after one that fit nothing)
        self.lay_pass: Tensor | None = None   # the multiply's surviving column rules [2 (lo, hi), N_RULES]
        self.lay_counts = None
        self.lay_census = None                # the step the layout's rules were first scored whole'''
LAYCONST_OLD = '''CODE_BATCH = 64        # ... then this many rows of the batch at once'''
LAYCONST_NEW = '''CODE_BATCH = 64        # ... then this many rows of the batch at once
LAYOUT_CENSUS = 64     # where every row has the same N: the ladder numbers the multiply's column rules are scored on'''


def build(src: str) -> str:
    src = V3_DOC + src[3:]
    src = sub1(src, CONSTS_OLD, CONSTS_NEW)
    src = sub1(src, "\n\ndef menu_size", HELPERS)
    src = sub1(src, '        self.io_parts = ""', CONTROL)
    src = sub1(src, "        ctl.best = int(passes.argmax())", IO_FACTS)
    src = sub1(src, "            Lk = hp.float() + EPS", IO_LK)
    src = sub1(src, IO_CHOSEN_OLD, IO_CHOSEN_NEW)
    src = sub1(src, TIE_SOFT_OLD, TIE_SOFT_NEW)
    src = sub1(src, TIE_STREAK_OLD, TIE_STREAK_NEW)
    src = sub1(src, TIE_USED_OLD, TIE_USED_NEW)
    src = sub1(src, TIE_CHECK_OLD, TIE_CHECK_NEW)
    src = sub1(src, CALL_OLD, CALL_NEW)
    src = sub1(src, REPORT_OLD, REPORT_NEW)
    src = sub1(src, "    def mine(self, terms: dict) -> None:", METHODS)
    src = sub1(src, ADOPT_OLD, ADOPT_NEW)
    src = sub1(src, WRITER1_OLD, WRITER1_NEW)
    src = sub1(src, WRITER2_OLD, WRITER2_NEW)
    src = sub1(src, READ_OLD, READ_NEW)
    src = sub1(src, TILE_OLD, TILE_NEW)
    src = sub1(src, LADDER_OLD, LADDER_NEW)
    src = sub1(src, CHAINN_OLD, CHAINN_NEW)
    src = sub1(src, WSLOT_OLD, WSLOT_NEW)
    src = sub1(src, "    def ladder(self, Xn: Tensor, tabs, train: bool):", LAYOUT_METHOD)
    src = sub1(src, LAYCALL_OLD, LAYCALL_NEW)
    src = sub1(src, LAYCTL_OLD, LAYCTL_NEW)
    src = sub1(src, LAYCONST_OLD, LAYCONST_NEW)
    return src


def main() -> None:
    base = BASE.read_bytes()
    md5 = hashlib.md5(base).hexdigest()
    if md5 != BASE_MD5:
        raise SystemExit(f"{BASE} md5 {md5} != {BASE_MD5}")
    out = build(base.decode())
    OUT.write_text(out)
    print(f"wrote {OUT} md5 {hashlib.md5(out.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
