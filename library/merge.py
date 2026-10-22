"""Merging recurring pieces of working programs into new single instructions (LIBRARY.md §3).

The corpus is {task: [program, ...]}: every minimal program found for each solved task, in the current vocabulary.
Three ways to grow the vocabulary:

- BPE (`bpe`): the most frequent adjacent pair becomes one new token, again and again. A pair is two neighbouring
  tokens in one item sequence (top level or a loop's body), or a loop whose body is a single token (its header and
  that token). Each task's programs share a weight of 1, so a pair counts once per task that always uses it. Merges
  continue while the best pair's weight is at least `min_count` tasks.
- MDL (`mdl`): any contiguous run of items (tokens and whole loops) can become one token. Each round adopts the run
  that most shortens the description: the shortest program of every task plus the library's definitions, at
  log2(vocabulary size) bits a token. It stops when no run shortens it.
- MDL with holes (`mdl(holes=True)`): a run may also leave one token out as an argument, which makes a new loop-like
  construct: a header that takes a one-token body, e.g. SCAN[x] = P[x INC]. An occurrence costs 2 (header and
  argument), and the search can fill the hole with any token.

A macro is a token whose definition is an item sequence; an abstraction (a hole) is a new loop kind whose body is
substituted for its hole. Both live in `lib`: name -> items, with the hole written "_".
"""

from __future__ import annotations

import math
from collections import Counter

from library import dsl

HOLE = "_"


def is_loop(it):
    return isinstance(it, tuple)


def seqs(items):
    """Every item sequence in a program: the top level and every loop body, outermost first."""
    yield items
    for it in items:
        if is_loop(it):
            yield from seqs(it[2])


def rewrite(items, fn):
    """Apply fn to every item sequence, innermost first; fn returns the new sequence."""
    out = []
    for it in items:
        if is_loop(it):
            out.append(("REP", it[1], rewrite(it[2], fn)))
        else:
            out.append(it)
    return tuple(fn(tuple(out)))


def lib_cost(lib):
    return sum(dsl.cost(v) + 1 for v in lib.values())


def n_symbols(lib, n_prims=len(dsl.PRIMS), n_kinds=len(dsl.LOOP_KINDS)):
    return n_prims + n_kinds + len(lib)


def description_length(corpus, lib, lib_weight=1.0):
    """Bits: the shortest program of each task plus the library, at log2(#symbols) bits per token. lib_weight < 1
    makes definitions cheaper (DreamCoder's structure penalty is the same kind of knob)."""
    toks = sum(min(dsl.cost(p) for p in progs) for progs in corpus.values() if progs) + lib_weight * lib_cost(lib)
    return toks * math.log2(n_symbols(lib))


def _name(lib, prefix):
    i = 1
    while f"{prefix}{i}" in lib:
        i += 1
    return f"{prefix}{i}"


# ---------------------------------------------------------------------------------------------------------------
# BPE

def _pairs(seq, overlapping=True):
    """Adjacent token pairs of one sequence, and its loops whose body is a single token. Standard BPE counts every
    adjacent pair (overlapping=True); record r2 paired left to right without overlaps."""
    out = []
    i = 0
    while i < len(seq):
        a = seq[i]
        if is_loop(a):
            if len(a[2]) == 1 and not is_loop(a[2][0]):
                out.append(("LOOP", a[1], a[2][0]))
            i += 1
            continue
        if i + 1 < len(seq) and not is_loop(seq[i + 1]):
            out.append(("PAIR", a, seq[i + 1]))
            if not overlapping or (a == seq[i + 1]):     # "a a a" holds one mergeable "a a" per two tokens
                i += 2
                continue
        i += 1
    return out


def pair_counts(corpus, weighting="max", overlapping=True):
    """weighting "max": a task counts a pair as often as its best program uses it (any minimal program will do,
    so a task backs every pair one of its programs needs); "mean": the average over its programs (record r1)."""
    cnt = Counter()
    for progs in corpus.values():
        if not progs:
            continue
        per = []
        for p in progs:
            c = Counter()
            for s in seqs(p):
                c.update(_pairs(s, overlapping))
            per.append(c)
        keys = set().union(*per)
        for pr in keys:
            vals = [c[pr] for c in per]
            cnt[pr] += max(vals) if weighting == "max" else sum(vals) / len(vals)
    return cnt


def _apply_pair(pr, name):
    kind = pr[0]

    def fn(seq):
        out = []
        i = 0
        while i < len(seq):
            it = seq[i]
            if kind == "LOOP" and is_loop(it) and it[1] == pr[1] and it[2] == (pr[2],):
                out.append(name)
                i += 1
            elif kind == "PAIR" and i + 1 < len(seq) and it == pr[1] and seq[i + 1] == pr[2]:
                out.append(name)
                i += 2
            else:
                out.append(it)
                i += 1
        return out

    return fn


def _pair_size(pr, lib):
    """How many primitives a merged pair stands for."""
    items = (("REP", pr[1], (pr[2],)),) if pr[0] == "LOOP" else (pr[1], pr[2])
    return dsl.cost(dsl.expand(items, lib))


def bpe(corpus, lib, max_merges=12, min_count=2.0, prefix="B", log=None, weighting="max", overlapping=True,
        tiebreak="spelling"):
    """Greedy pair merges. Returns (corpus rewritten, lib with the new macros, list of new names).
    tiebreak: among pairs of equal weight, "spelling" takes the largest spelling (reproducible, arbitrary);
    "largest" takes the pair standing for the most primitives (the biggest saving), then spelling."""
    lib = dict(lib)
    new = []
    for _ in range(max_merges):
        cnt = pair_counts(corpus, weighting, overlapping)
        if not cnt:
            break
        if tiebreak == "largest":
            pr, w = max(cnt.items(), key=lambda kv: (round(kv[1], 9), _pair_size(kv[0], lib), str(kv[0])))
        else:
            pr, w = max(cnt.items(), key=lambda kv: (round(kv[1], 9), str(kv[0])))
        if w < min_count:
            break
        name = _name(lib, prefix)
        lib[name] = ((("REP", pr[1], (pr[2],)),) if pr[0] == "LOOP" else (pr[1], pr[2]))
        fn = _apply_pair(pr, name)
        corpus = {t: [rewrite(p, fn) for p in progs] for t, progs in corpus.items()}
        new.append(name)
        if log is not None:
            log.append({"name": name, "def": dsl.show(lib[name]), "weight": round(w, 3),
                        "expands_to": dsl.show(dsl.expand(lib[name], lib))})
    return corpus, lib, new


# ---------------------------------------------------------------------------------------------------------------
# MDL

def _match(pat, seq, i):
    """Does pattern pat (items, may contain one HOLE token at any depth) match seq at i? Returns the filler or
    True (no hole) or None."""
    filler = []

    def m(p, s):
        if p == HOLE:
            if is_loop(s):
                return False
            filler.append(s)
            return True
        if is_loop(p):
            return is_loop(s) and p[1] == s[1] and len(p[2]) == len(s[2]) and all(m(a, b) for a, b in zip(p[2], s[2]))
        return p == s

    if i + len(pat) > len(seq):
        return None
    if all(m(a, b) for a, b in zip(pat, seq[i:i + len(pat)])):
        return filler[0] if filler else True
    return None


def _apply_pattern(pat, name, holed):
    def fn(seq):
        out = []
        i = 0
        while i < len(seq):
            f = _match(pat, seq, i)
            if f is not None:
                out.append(("REP", name, (f,)) if holed else name)
                i += len(pat)
            else:
                out.append(seq[i])
                i += 1
        return out

    return fn


def _holes(items):
    """Every way to replace one token (at any depth) of an item sequence by HOLE."""
    for j, it in enumerate(items):
        if is_loop(it):
            for b in _holes(it[2]):
                yield items[:j] + (("REP", it[1], b),) + items[j + 1:]
        else:
            yield items[:j] + (HOLE,) + items[j + 1:]


def candidates(corpus, holes=False, max_len=6):
    """Contiguous runs of items of total cost >= 2 (and, with holes, their one-hole versions)."""
    out = set()
    for progs in corpus.values():
        for p in progs:
            for s in seqs(p):
                for i in range(len(s)):
                    for j in range(i + 1, min(len(s), i + max_len) + 1):
                        run = s[i:j]
                        if dsl.cost(run) < 2:
                            continue
                        out.add(run)
                        if holes:
                            for h in _holes(run):
                                if dsl.cost(h) >= 3:
                                    out.add(h)
    return out


def _has_hole(items):
    return any((it == HOLE) or (is_loop(it) and _has_hole(it[2])) for it in items)


def mdl(corpus, lib, holes=False, max_rounds=30, prefix="M", log=None, kinds=None, lib_weight=1.0):
    """Greedy MDL. kinds: the list of loop kinds, extended in place with every new abstraction's name."""
    lib = dict(lib)
    new = []
    for _ in range(max_rounds):
        base = description_length(corpus, lib, lib_weight)
        best = None
        for pat in candidates(corpus, holes):
            holed = _has_hole(pat)
            name = _name(lib, "H" if holed else prefix)
            trial_lib = dict(lib)
            trial_lib[name] = pat
            fn = _apply_pattern(pat, name, holed)
            trial = {t: [rewrite(p, fn) for p in progs] for t, progs in corpus.items()}
            dl = description_length(trial, trial_lib, lib_weight)
            if dl < base - 1e-9 and (best is None or dl < best[0] - 1e-9
                                     or (abs(dl - best[0]) < 1e-9 and str(pat) < str(best[1]))):
                best = (dl, pat, name, trial, holed)
        if best is None:
            break
        dl, pat, name, corpus, holed = best
        lib[name] = pat
        new.append(name)
        if holed and kinds is not None:
            kinds.append(name)
        if log is not None:
            log.append({"name": name, "def": dsl.show(pat), "bits_after": round(dl, 1), "bits_before": round(base, 1),
                        "holed": holed})
    return corpus, lib, new
