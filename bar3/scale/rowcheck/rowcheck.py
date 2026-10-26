"""The duplicate check on a problem's own rows (ROWCHECK.md §2).

The frozen learner calls a solved problem an alias when its best program has the signature of a primitive or a routine,
and a duplicate when it has the signature of an earlier learnable problem's best program. A signature is what a
program does to 24 random machine states (bar3/learner.Signatures); every program that dies on all of them shares one.

With cfg["rowcheck"] the signature only proposes: the problem is an alias when a primitive or routine with that
signature is right on the problem's own rows, a duplicate when an earlier learnable problem's best program with that
signature is. Its own rows are the instances it is accepted on and its check instances, its registers set as a job
sets them. This is observational equivalence on the problem's examples, as in Escher's observational equivalence
reduction (Albarghouthi, Gulwani and Kincaid, CAV 2013, §3.3). A problem that no candidate solves stays learnable and
joins its signature's list.
"""

from __future__ import annotations

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import machine_for
from bar3.scale.jobs import run_on


def rows_of(c):
    """[(xs, Ns, init, want, size)]: the instances c is accepted on (its `every`, else its search instances), then its
    check instances."""
    init = getattr(c, "init", {}) or {}
    ev = getattr(c, "every", ()) or (c.xs, c.Ns, init, c.want)
    return [(ev[0], ev[1], ev[2], ev[3], "k"), (c.cx, c.cN, getattr(c, "cinit", {}) or {}, c.cw, "k2")]


def solves(items, pl, c, k, k2):
    """Is the program (parsed items, library pl) right on all of c's own rows, without running out of steps?"""
    outs = list(getattr(c, "outs", ("A",)))
    oi = [brm.RI[r] for r in outs]
    for xs, Ns, init, want, size in rows_of(c):
        regs, dead = run_on(tuple(items), pl, xs, Ns, init, machine_for(k if size == "k" else k2))
        if dead or not bool((regs[:, oi] == np.asarray(want).reshape(len(xs), len(oi))).all()):
            return False
    return True


def tokens_by_sig(sigs, lib):
    """{signature: [primitive or routine token]}, as the frozen learner's `have` set, with the tokens kept."""
    out = {}
    for t in list(brm.PRIMS) + list(lib):
        out.setdefault(sigs.of((t,), lib), []).append(t)
    return out


def classify(s, c, lib, sig_tok, learn_rc, status, k, k2):
    """(status, of) for a problem c whose best program has signature s: ("alias", token) when a primitive or routine
    with that signature solves c on its own rows; ("duplicate", name) when an earlier learnable problem's best program
    with that signature does; else ("learnable", None), and c joins the signature's list."""
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    for t in sig_tok.get(s, []):
        if solves((t,), pl, c, k, k2):
            return "alias", t
    for n in learn_rc.get(s, []):
        if solves(dsl.parse(status[n]["best"]), pl, c, k, k2):
            return "duplicate", n
    learn_rc.setdefault(s, []).append(c.name)
    return "learnable", None
