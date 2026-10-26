"""bar3/scale/engine.py's search on the GPU (GPU_SEARCH.md), exact: the same states, the same hash keys, the same order,
so every record field but the seconds comes out the same. The API is engine.py's (bfs, bfs_multi, Codes, and the item
enumeration, which stays on the CPU), so bar3/scale/jobs.py runs on it unchanged once `use()` swaps its engine.

A level is built as engine.bfs builds it, in the same order: every item in class order, each on every state of its
source level in order. Here the (item, state) pairs of a level go through the kernel in big chunks (kernel.py), and
the bookkeeping runs on the GPU too: the seen test (a sorted array of every earlier level's keys), the fits and goal
tests, and at the level's end the first occurrence of each new key (a stable sort), the parents, sorted by child as
np.argsort(kind="stable") sorts them, and the new seen array. Keys are int64 with the top bit flipped, so their order
is the uint64 order numpy sorts by (the goals' order, and so the programs', follows it).

What can differ from the CPU search: only where a search stops on its time limit (the GPU checks the clock after each
chunk, the CPU after each class), and the seconds. A level that passes the budget is streamed as on the CPU: whether
the budget is passed at one chunk or another, the outcome is the same (every fresh state is counted, every goal
state kept), since the CPU's test is on the running count of fresh states.

The parents and held levels stay on the GPU; programs() and near_misses() read them through lists that copy an entry
to numpy when it is first read.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from library import dsl
from library.inventing import brm
from library.search import hash_rows
from bar3.search import Problem, Result, programs, programs_at, near_misses  # noqa: F401  (re-exported)
from bar3.scale import engine as E
from bar3.scale.engine import ItemClass, compile_items, enumerate_items, sig  # noqa: F401  (re-exported for jobs.py)
from bar3.gpu_search import kernel as K

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_BYTES = 1 << 30        # a chunk's output rows (the kernel's buffer)


class Prof:
    """Seconds by phase of the level loop, when PROF is set (each tick waits for the GPU; off by default)."""

    def __init__(self):
        self.t = {}
        self.last = time.time()

    def tick(self, name):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        now = time.time()
        self.t[name] = self.t.get(name, 0.0) + now - self.last
        self.last = now


PROF = None


def _tick(name):
    if PROF is not None:
        PROF.tick(name)


class Codes(E.Codes):
    """engine.Codes, plus the packed bytecode on the GPU (built on first use)."""

    def __init__(self, classes, lib):
        super().__init__(classes, lib)
        self._dev = {}

    def gpu(self, device):
        if device not in self._dev:
            self._dev[device] = K.GCodes(self, device)
        return self._dev[device]


class _Dev:
    """A GPU entry of a lazy list."""

    def __init__(self, x):
        self.x = x


class _Lazy(list):
    """A list whose GPU entries are copied to numpy (by `conv`) when first read."""

    def __init__(self, conv):
        super().__init__()
        self._conv = conv

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        v = list.__getitem__(self, i)
        if isinstance(v, _Dev):
            v = self._conv(v.x)
            list.__setitem__(self, i, v)
        return v

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def dev(self, i):
        v = list.__getitem__(self, i)
        return v.x if isinstance(v, _Dev) else v


def _parents_np(t):
    child, cc, cs, ci = t
    return (child.cpu().numpy().astype(np.int64), cc.cpu().numpy().astype(np.int32),
            cs.cpu().numpy().astype(np.int16), ci.cpu().numpy().astype(np.int64))


class _Spec:
    """A problem's goal on the GPU: the answer registers and the wanted values per probe."""

    def __init__(self, prob, NP, device):
        self.out_regs = list(getattr(prob, "outs", [brm.A]))
        want = np.asarray(prob.want).reshape(NP, len(self.out_regs))
        self.outs = torch.tensor(self.out_regs, dtype=torch.long, device=device)
        self.want = torch.from_numpy(want.astype(np.int32)).to(device)
        self.n = len(prob.want)

    def fits(self, rows, NP):
        """Probes right, per row (rows [n, ROW] int16, contiguous): the answer registers read in place."""
        n, ROW = rows.shape
        v = rows.as_strided((n, NP, 5), (ROW, 5, 1), rows.storage_offset())
        if len(self.out_regs) == 1:
            return (v[:, :, self.out_regs[0]] == self.want[None, :, 0]).sum(dim=1)
        return (v.index_select(2, self.outs) == self.want[None]).all(dim=2).sum(dim=1)


class _Space:
    """One search's levels, seen keys and parents on the GPU."""

    def __init__(self, root, classes, codes, m, device=None):
        self.dev = device or DEVICE
        self.NP = root.regs.shape[1]
        self.ROW = K.row_len(self.NP)
        assert m.cap < 32768
        self.m, self.codes, self.gc = m, codes, codes.gpu(self.dev)
        self.cost = np.array([cl.cost for cl in classes], np.int64)
        row = np.zeros((1, self.ROW), np.int16)
        row[0, :self.NP * 5] = root.regs.reshape(-1).astype(np.int16)
        self.rows = [torch.from_numpy(row).to(self.dev)]       # held levels' rows (None when not held)
        self.root = root
        k = (hash_rows(root.features()) ^ np.uint64(K.FLIP)).view(np.int64)
        self.seen = torch.from_numpy(k.copy()).to(self.dev)
        self.levels = _Lazy(self._level_np)
        self.levels.append(root)
        self.parents = _Lazy(_parents_np)
        self.parents.append(None)
        self.expanded = 0
        self.CH = max(1, CHUNK_BYTES // (2 * self.ROW))

    def _level_np(self, rows):
        S = rows.shape[0]
        regs = rows[:, :self.NP * 5].cpu().numpy().astype(np.int32).reshape(S, self.NP, 5)
        return brm.Batch(regs, np.zeros(S, bool))

    def segments(self, d):
        """bfs's (class, source level) order for level d, as runs of consecutive classes on one source level."""
        out, cur = [], None
        for ci, c in enumerate(self.cost.tolist()):
            sl = d - c
            if sl < 0 or self.rows[sl] is None:
                cur = None
                continue
            if cur is not None and cur[2] == sl and cur[1] == ci:
                cur[1] = ci + 1
            else:
                cur = [ci, ci + 1, sl]
                out.append(cur)
        return out

    def fresh(self, keys):
        pos = torch.searchsorted(self.seen, keys).clamp_(max=self.seen.numel() - 1)
        return self.seen[pos] != keys

    def level(self, d, last, specs, max_states, t0, time_limit):
        """Level d as engine.bfs builds it, for the goals `specs` (the live problems). Returns a dict: last, stop,
        n_fresh, best (per spec), gk (per spec: goal keys or None), uk, rank."""
        NP, m = self.NP, self.m
        ck, cc, cs, ci_, rows, gm = [], [], [], [], [], []
        gk = [[] for _ in specs]
        n_goal = [0] * len(specs)
        best = [0] * len(specs)
        n_cand = n_fresh = 0
        stop = ""
        for c0, c1, sl in self.segments(d):
            src = self.rows[sl]
            S = src.shape[0]
            total = (c1 - c0) * S
            for lo in range(0, total, self.CH):
                n = min(self.CH, total - lo)
                _tick("other")
                out, keys, drop = K.apply_chunk(self.gc, self.codes, c0, S, lo, n, src, NP, m.cap, m.limit)
                _tick("kernel")
                self.expanded += n
                ok = ~drop
                ok &= self.fresh(keys)
                if last:
                    # streamed: only the count, the best fits and the goal states are kept; no row is gathered
                    nf = int(ok.sum())
                    _tick("fresh")
                    if nf:
                        n_fresh += nf
                        gany = None
                        for j, sp in enumerate(specs):
                            f = torch.where(ok, sp.fits(out, NP), -1)
                            best[j] = max(best[j], int(f.max()))
                            g = f == sp.n
                            ng = int(g.sum())
                            if ng:
                                gk[j].append(keys[g])
                                n_goal[j] += ng
                            gany = g if gany is None else gany | g
                        idx = torch.nonzero(gany).squeeze(1)
                        if idx.numel():
                            g_ = lo + idx
                            ck.append(keys[idx])
                            cc.append((c0 + g_ // S).to(torch.int32))
                            cs.append(torch.full((idx.numel(),), sl, dtype=torch.int16, device=self.dev))
                            ci_.append(g_ % S)
                            rows.append(None)
                            gm.append(torch.ones(idx.numel(), dtype=torch.bool, device=self.dev))
                            n_cand += idx.numel()
                        _tick("fits_keep")
                    if time.time() - t0 > time_limit:
                        stop = f"time: {time_limit}s at level {d}"
                        break
                    continue
                idx = torch.nonzero(ok).squeeze(1)
                nf = idx.numel()
                _tick("fresh")
                if nf:
                    n_fresh += nf
                    sub = out.index_select(0, idx)
                    k = keys.index_select(0, idx)
                    gany = None
                    for j, sp in enumerate(specs):
                        f = sp.fits(sub, NP)
                        best[j] = max(best[j], int(f.max()))
                        g = f == sp.n
                        ng = int(g.sum())
                        if ng:
                            gk[j].append(k[g])
                            n_goal[j] += ng
                        gany = g if gany is None else gany | g
                    g_ = lo + idx
                    ck.append(k)
                    cc.append((c0 + g_ // S).to(torch.int32))
                    cs.append(torch.full((idx.numel(),), sl, dtype=torch.int16, device=self.dev))
                    ci_.append(g_ % S)
                    rows.append(sub)
                    gm.append(gany)
                    n_cand += idx.numel()
                    if n_cand > 1.5 * max_states:      # from the next chunk on, the level is streamed
                        last = True
                        stop = f"budget: level {d} streamed (over {max_states} states)"
                        keep = [(j, torch.nonzero(gm[j]).squeeze(1)) for j in range(len(ck))]
                        keep = [(j, q) for j, q in keep if q.numel()]
                        ck = [ck[j][q] for j, q in keep]
                        cc = [cc[j][q] for j, q in keep]
                        cs = [cs[j][q] for j, q in keep]
                        ci_ = [ci_[j][q] for j, q in keep]
                        gm = [gm[j][q] for j, q in keep]
                        rows = [None] * len(ck)
                    _tick("fits_keep")
                if time.time() - t0 > time_limit:
                    stop = f"time: {time_limit}s at level {d}"
                    break
            if stop.startswith("time"):
                break
        out = dict(last=last, stop=stop, n_fresh=n_fresh, best=best, gk=[None] * len(specs))
        if ck:
            allk = torch.cat(ck)
            sk, perm = torch.sort(allk, stable=True)
            new = torch.ones_like(sk, dtype=torch.bool)
            new[1:] = sk[1:] != sk[:-1]
            gid = torch.cumsum(new, 0) - 1
            uk = sk[new]
            first = perm[new]
            inv = torch.empty_like(gid)
            inv[perm] = gid
            sfirst, of = torch.sort(first)
            rank = torch.empty_like(of)
            rank[of] = torch.arange(of.numel(), device=self.dev)
            child = rank[inv]
            order = torch.sort(child, stable=True).indices
            self.parents.append(_Dev((child[order], torch.cat(cc)[order], torch.cat(cs)[order],
                                      torch.cat(ci_)[order])))
            if not last and not stop:
                parts, off = [], 0
                for r in rows:
                    n = r.shape[0]
                    a, b = torch.searchsorted(sfirst, torch.tensor([off, off + n], device=self.dev)).tolist()
                    parts.append(r.index_select(0, sfirst[a:b] - off))
                    off += n
                lv = torch.cat(parts)
                self.rows.append(lv)
                self.levels.append(_Dev(lv))
            else:
                self.rows.append(None)
                self.levels.append(None)
        else:
            uk = torch.zeros(0, dtype=torch.int64, device=self.dev)
            rank = torch.zeros(0, dtype=torch.int64, device=self.dev)
            self.parents.append(None)
            self.rows.append(None)
            self.levels.append(None)
        for j in range(len(specs)):
            if n_goal[j]:
                g = torch.unique(torch.cat(gk[j]))
                out["gk"][j] = rank[torch.searchsorted(uk, g)].tolist()
        out["uk"] = uk
        out["size"] = int(uk.numel()) if not last else int(n_fresh)
        _tick("level_end")
        return out

    def add_seen(self, uk):
        self.seen = torch.sort(torch.cat([self.seen, uk])).values


def bfs(prob: Problem, classes, lib, m, max_cost=10, max_states=1_000_000, time_limit=600.0, accept=None, limit=200,
        codes=None):
    """engine.bfs on the GPU."""
    t0 = time.time()
    codes = codes or Codes(classes, lib)
    sp = _Space(prob.root, classes, codes, m)
    levels, parents = sp.levels, sp.parents
    res = Result()
    res.level_sizes.append(1)
    res.best_fit.append(int(prob.fits(prob.root)[0]))
    if prob.goal(prob.root)[0]:
        if accept is None or any(accept([()])):
            res.found_cost, res.goals, res.parents, res.levels = 0, [0], parents, levels
            return res
        res.aliases.append((0, [""]))
    spec = [_Spec(prob, sp.NP, sp.dev)]
    res.level_seconds = []                  # the clock at each level's end (timing.py; not a record field)
    for d in range(1, max_cost + 1):
        lv = sp.level(d, d == max_cost, spec, max_states, t0, time_limit)
        if lv["stop"]:
            res.stop = lv["stop"]
        res.level_sizes.append(lv["size"])
        res.best_fit.append(lv["best"][0])
        res.level_seconds.append(round(time.time() - t0, 4))
        if lv["gk"][0] is not None:
            res.found_cost = d
            res.goals = lv["gk"][0]
            res.parents = parents
            if accept is None:
                break
            progs = programs(res, classes, limit=limit)
            ok = accept(progs)
            if any(ok):
                res.accepted = [p for p, o in zip(progs, ok) if o]
                break
            res.aliases.append((d, [dsl.show(p) for p in progs[:20]]))
            res.found_cost, res.goals = None, None
        if res.stop:
            break
        sp.add_seen(lv["uk"])
    res.seconds = time.time() - t0
    res.expanded = sp.expanded
    if res.found_cost is None and not res.stop:
        res.stop = f"exhausted cost {max_cost}"
    res.parents = parents
    res.levels = levels
    return res


def bfs_multi(probs, classes, lib, m, max_cost=10, max_states=1_000_000, time_limit=600.0, accepts=None, limit=200,
              codes=None):
    """engine.bfs_multi on the GPU: one search for problems posed on the same instances, each with its own Result."""
    t0 = time.time()
    codes = codes or Codes(classes, lib)
    root = probs[0].root
    sp = _Space(root, classes, codes, m)
    levels, parents = sp.levels, sp.parents
    out = [Result() for _ in probs]
    live = []
    for j, (pr, res) in enumerate(zip(probs, out)):
        res.level_sizes.append(1)
        res.best_fit.append(int(pr.fits(root)[0]))
        if pr.goal(root)[0] and (accepts is None or any(accepts[j]([()]))):
            res.found_cost, res.goals, res.parents, res.levels = 0, [0], parents, levels
        else:
            if pr.goal(root)[0]:
                res.aliases.append((0, [""]))
            live.append(j)
    specs = [_Spec(pr, sp.NP, sp.dev) for pr in probs]
    stop = ""
    for d in range(1, max_cost + 1):
        if not live:
            break
        lv = sp.level(d, d == max_cost, [specs[j] for j in live], max_states, t0, time_limit)
        stop = lv["stop"]
        still = []
        for q, j in enumerate(live):
            res = out[j]
            res.level_sizes.append(lv["size"])
            res.best_fit.append(lv["best"][q])
            res.stop = stop
            done = False
            if lv["gk"][q] is not None:
                res.found_cost = d
                res.goals = lv["gk"][q]
                res.parents = parents
                if accepts is None:
                    done = True
                else:
                    progs = programs(res, classes, limit=limit)
                    ok = accepts[j](progs)
                    if any(ok):
                        res.accepted = [p for p, o in zip(progs, ok) if o]
                        done = True
                    else:
                        res.aliases.append((d, [dsl.show(p) for p in progs[:20]]))
                        res.found_cost, res.goals = None, None
            if not done and not stop:
                still.append(j)
        live = still
        if stop:
            break
        sp.add_seen(lv["uk"])
    for res in out:
        res.seconds = time.time() - t0
        res.expanded = sp.expanded
        if res.found_cost is None and not res.stop:
            res.stop = f"exhausted cost {max_cost}"
        res.parents = parents
        res.levels = levels
    return out


def use(on=True):
    """The flag: bar3/scale/jobs.py's searches run here (on) or on the compiled CPU engine (off), in this process."""
    import sys
    import bar3.scale.jobs as J
    J.engine = sys.modules[__name__] if on else E
    J._CLASSES.clear()
