"""The search of tdsl.py with many items applied at once (INTEGRATION.md, stage 2): rows are (item, state) pairs; a
token runs once on all the rows that apply it, and a loop runs its iterations on all its rows together, its body's
elements grouped position by position (rows sorted by the element there, one run per distinct element). The search
then costs a few hundred batched runs per chunk of rows instead of one small run per item, which is what makes loop
bodies of cost 3 (the multiply's: a row, two moves; LIBRARY.md §5.3) affordable on a GPU. Same machine, same items,
same breadth-first search as tdsl.bfs (and library/search.py); a linear 64-bit hash instead of tdsl's mixing loop.
"""

from __future__ import annotations

import time

import torch

from integration.lib import tdsl
from integration.lib.tdsl import Batch, ItemClass, Result, run_items, show

ONE = "ONE"           # a pseudo loop kind run once: a straight-line body (for the body signatures)


def count(st, kind):
    return 1 if kind == ONE else st.count(kind)


# ---------------------------------------------------------------------------------------------------------------
# hashing: a random linear combination of the row's 64-bit words (mod 2^64), then a mix

_GEN = torch.Generator().manual_seed(12345)
_MULT = (torch.randint(-(1 << 62), 1 << 62, (4096,), generator=_GEN, dtype=torch.int64) * 2 + 1)


def hash_rows(F: torch.Tensor) -> torch.Tensor:
    S_, n = F.shape
    pad = (-n) % 8
    if pad:
        F = torch.cat([F, torch.zeros(S_, pad, dtype=torch.uint8, device=F.device)], 1)
    Wd = F.contiguous().view(torch.int64)
    h = (Wd * _MULT[:Wd.shape[1]].to(F.device)[None]).sum(1)
    h = h ^ tdsl._lsr(h, 31)
    h = h * tdsl._M2
    return h ^ tdsl._lsr(h, 29)


def keys_of(batches):
    return hash_rows(torch.cat([b.features() for b in batches], 1))


# ---------------------------------------------------------------------------------------------------------------
# heterogeneous application

def _put(out: Batch, idx: torch.Tensor, sub: Batch):
    out.regs[idx] = sub.regs
    out.c[idx] = sub.c
    out.h[idx] = sub.h
    out.k[idx] = sub.k
    out.p[idx] = sub.p


def _groups(ids: torch.Tensor):
    """Rows grouped by id: [(id, row indices)] (ids >= 0 only)."""
    keep = torch.nonzero(ids >= 0).flatten()
    if keep.numel() == 0:
        return []
    order = keep[torch.sort(ids[keep], stable=True).indices]
    sid = ids[order]
    uniq, counts = torch.unique_consecutive(sid, return_counts=True)
    out, start = [], 0
    for u, c in zip(uniq.tolist(), counts.tolist()):
        out.append((u, order[start:start + c]))
        start += c
    return out


def apply_items(st, b: Batch, item_of_row: torch.Tensor, items: list, lib) -> Batch:
    """A new batch: row r of b with items[item_of_row[r]] applied (tokens, loops over bodies of any items)."""
    out = b.copy()
    dev = b.regs.device
    loop = [isinstance(it, tuple) and it[1] not in lib for it in items]
    is_loop = torch.tensor(loop, dtype=torch.bool, device=dev)[item_of_row]
    # tokens (and anything that is not a plain loop) run per distinct item
    whole = torch.where(~is_loop, item_of_row, torch.full_like(item_of_row, -1))
    for u, idx in _groups(whole):
        sub = out.take(idx)
        run_items(st, sub, (items[u],), lib)
        _put(out, idx, sub)
    if not any(loop):
        return out
    # loops: the iterations of every loop row together, the body's elements grouped position by position
    elems, eid = {}, []
    L = max((len(it[2]) for it, lp in zip(items, loop) if lp), default=0)
    cnt = torch.zeros(len(items), dtype=torch.long)
    table = torch.full((len(items), L), -1, dtype=torch.long)
    for i, (it, lp) in enumerate(zip(items, loop)):
        if not lp:
            continue
        cnt[i] = count(st, it[1])
        for j, e in enumerate(it[2]):
            if e not in elems:
                elems[e] = len(elems)
                eid.append(e)
            table[i, j] = elems[e]
    cnt, table = cnt.to(dev), table.to(dev)
    rcnt = torch.where(is_loop, cnt[item_of_row], torch.zeros_like(item_of_row))
    rtab = table[item_of_row]
    for itn in range(int(rcnt.max())):
        live = rcnt > itn
        for j in range(L):
            ids = torch.where(live, rtab[:, j], torch.full_like(item_of_row, -1))
            for u, idx in _groups(ids):
                sub = out.take(idx)
                run_items(st, sub, (eid[u],), lib)
                _put(out, idx, sub)
    return out


# ---------------------------------------------------------------------------------------------------------------
# items

def _signatures(settings, tests, items, lib, chunk=1 << 21):
    """One signature per item: its outputs on the test machines of every setting, hashed."""
    S_ = tests[0].S
    out = []
    step = max(1, chunk // S_)
    for i0 in range(0, len(items), step):
        part = items[i0:i0 + step]
        k = len(part)
        dev = tests[0].regs.device
        rows_item = torch.arange(k, device=dev).repeat_interleave(S_)
        rows_state = torch.arange(S_, device=dev).repeat(k)
        outs = [apply_items(st, t.take(rows_state), rows_item, part, lib) for st, t in zip(settings, tests)]
        keys = keys_of(outs).view(k, S_)
        out.append(hash_rows(keys.contiguous().view(torch.uint8).view(k, S_ * 8)))
    return torch.cat(out).tolist() if out else []


def enumerate_items(tokens, lib, settings, kinds=tdsl.LOOP_KINDS, max_body=2, max_arity=3, seed=7, S_test=24,
                    NP_test=6, log=None):
    """tdsl.enumerate_items with batched signatures: every distinct item (tokens, then loops over bodies of cost up to
    max_body built from the distinct items found so far), the cheapest spelling first."""
    t0 = time.time()
    gen = torch.Generator().manual_seed(seed)
    tests = [tdsl.random_states(st, S_test, NP_test, gen) for st in settings]
    ident = _signatures(settings, tests, [("REP", ONE, ())], lib)[0]
    classes, by_sig = [], {ident: None}

    def add_all(cands, c):
        for item, s in zip(cands, _signatures(settings, tests, cands, lib)):
            if s in by_sig:
                cl = by_sig[s]
                if cl is not None and cl.cost == c:
                    cl.members.append(item)
                continue
            cl = ItemClass(item, c, [item])
            by_sig[s] = cl
            classes.append(cl)

    add_all(list(tokens), 1)
    body_sig = {ident}
    for bc in range(1, max_body + 1):
        reps = list(classes)
        cands = []

        def extend(prefix, pc, n):
            if pc == bc:
                cands.append(prefix)
                return
            if n == max_arity:
                return
            for cl in reps:
                if pc + cl.cost <= bc:
                    extend(prefix + (cl.rep,), pc + cl.cost, n + 1)
        extend((), 0, 0)
        found = []
        for body, s in zip(cands, _signatures(settings, tests, [("REP", ONE, b) for b in cands], lib)):
            if s not in body_sig:
                body_sig.add(s)
                found.append(body)
        loops = [("REP", kind, body) for body in found for kind in kinds
                 if not (kind in lib and (len(body) != 1 or isinstance(body[0], tuple)))]
        add_all(loops, bc + 1)
        if log:
            log(f"    items: bodies of cost {bc}: {len(cands)} candidates, {len(found)} distinct, {len(loops)} loops; "
                f"{len(classes)} classes, {time.time() - t0:.1f}s")
    return classes


# ---------------------------------------------------------------------------------------------------------------
# the search

def bfs(prob, classes, lib, max_cost=8, max_states=400_000, accept=None, limit=200, deadline=None, chunk=1 << 21,
        verbose=False):
    """tdsl.bfs with every item of a cost applied to a slice of its source level at once."""
    t0 = time.time()
    settings = prob.settings
    dev = prob.roots[0].regs.device
    levels = [prob.roots]
    seen = torch.sort(keys_of(prob.roots)).values
    parents = [None]
    res = Result(prob.task)
    res.level_sizes.append(1)
    if bool(prob.goal(prob.roots)[0]):
        res.found_cost, res.goals, res.parents = 0, [0], parents
        return res
    by_cost = {}
    for ci, cl in enumerate(classes):
        by_cost.setdefault(cl.cost, []).append(ci)
    by_cost = {c: torch.tensor(v, dtype=torch.long, device=dev) for c, v in by_cost.items()}
    for d in range(1, max_cost + 1):
        last = d == max_cost
        ck, cc, cs, ci_, chunks, gk = [], [], [], [], [], []
        n_cand = n_fresh = 0
        for c, cis in by_cost.items():
            sl = d - c
            if sl < 0 or levels[sl] is None:
                continue
            S_src = levels[sl][0].S
            n_it = cis.numel()
            s_step = max(1, min(S_src, chunk))
            for s0 in range(0, S_src, s_step):
                srcs = [b.take(slice(s0, s0 + s_step)) for b in levels[sl]]
                ns = srcs[0].S
                i_step = max(1, chunk // ns)
                for i0 in range(0, n_it, i_step):
                    ii = cis[i0:i0 + i_step]
                    k = ii.numel()
                    rows_item = torch.arange(k, device=dev).repeat_interleave(ns)
                    rows_state = torch.arange(ns, device=dev).repeat(k)
                    part = [classes[i].rep for i in ii.tolist()]
                    outs = [apply_items(st, b.take(rows_state), rows_item, part, lib) for st, b in zip(settings, srcs)]
                    res.expanded += k * ns
                    keys = keys_of(outs)
                    m = tdsl._fresh(seen, keys)
                    if not bool(m.any()):
                        continue
                    idx = torch.nonzero(m).flatten()
                    n_fresh += idx.numel()
                    sub = [b.take(idx) for b in outs]
                    del outs
                    g = prob.goal(sub)
                    if bool(g.any()):
                        gk.append(keys[idx][g])
                    if last:
                        if not bool(g.any()):
                            continue
                        gi = torch.nonzero(g).flatten()
                        idx, sub = idx[gi], [b.take(gi) for b in sub]
                    ck.append(keys[idx])
                    cc.append(ii[rows_item[idx]].to(torch.int32))
                    cs.append(torch.full((idx.numel(),), sl, dtype=torch.int16, device=dev))
                    ci_.append(rows_state[idx] + s0)
                    chunks.append(sub)
                    n_cand += idx.numel()
                    if not last and n_cand > 1.5 * max_states:
                        last = True
                        res.stop = f"budget: level {d} streamed (over {max_states} states)"
                        new = [[], [], [], [], []]
                        for j in range(len(chunks)):
                            gm = torch.nonzero(prob.goal(chunks[j])).flatten()
                            if gm.numel():
                                new[0].append(ck[j][gm])
                                new[1].append(cc[j][gm])
                                new[2].append(cs[j][gm])
                                new[3].append(ci_[j][gm])
                                new[4].append([b.take(gm) for b in chunks[j]])
                        ck, cc, cs, ci_, chunks = new
            if deadline is not None and time.time() > deadline:
                res.stop = f"deadline at level {d}"
                break
        if ck:
            allk = torch.cat(ck)
            uk, inv = torch.unique(allk, sorted=True, return_inverse=True)
            pos = torch.arange(allk.numel(), device=dev)
            first = torch.full((uk.numel(),), allk.numel(), dtype=torch.long, device=dev).scatter_reduce(
                0, inv, pos, reduce="amin")
            sfirst = torch.sort(first).values
            rank = torch.searchsorted(sfirst, first)
            child = rank[inv]
            order = torch.sort(child, stable=True).indices
            parents.append(tuple(t.cpu().numpy() for t in (child[order], torch.cat(cc)[order], torch.cat(cs)[order],
                                                            torch.cat(ci_)[order])))
            del allk, inv, child, order, ck, cc, cs, ci_
            if not last and not res.stop:
                parts, off = [[] for _ in settings], 0
                for j in range(len(chunks)):
                    ch = chunks[j]
                    n = ch[0].S
                    lo_, hi_ = torch.searchsorted(sfirst, off), torch.searchsorted(sfirst, off + n)
                    rows = sfirst[int(lo_):int(hi_)] - off
                    for si in range(len(settings)):
                        parts[si].append(ch[si].take(rows))
                    chunks[j] = None
                    off += n
                levels.append([Batch.concat(p) for p in parts])
                del parts
            else:
                levels.append(None)
            chunks = None
        else:
            uk = torch.zeros(0, dtype=torch.int64, device=dev)
            rank = torch.zeros(0, dtype=torch.long, device=dev)
            parents.append(None)
            levels.append(None)
        res.level_sizes.append(int(uk.numel()) if not last else int(n_fresh))
        if uk.numel() > max_states and not res.stop:
            res.stop = f"budget: level {d} has {uk.numel()} > {max_states} states"
        if verbose:
            print(f"  level {d}: {uk.numel()} states, {time.time() - t0:.1f}s", flush=True)
        if gk:
            g = torch.unique(torch.cat(gk))
            res.found_cost = d
            res.goals = rank[torch.searchsorted(uk, g)].tolist()
            if accept is None:
                break
            res.parents = parents
            progs = tdsl.programs(res, classes, limit=limit)
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                break
            res.aliases.append((d, [show(p) for p in progs[:20]]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        seen = torch.sort(torch.cat([seen, uk])).values
    res.seconds = time.time() - t0
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {max_cost}"
    res.parents = parents
    return res
