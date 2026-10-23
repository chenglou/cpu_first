"""Diagnostic (INTEGRATION.md §3, s3h): the in-run round search of s3g seed 5 at level 6, offline, with its exact answer
pool (lib/s3g5_pools.json, dumped by a replay of that run) and its library (lib/s3g5_sleeps.txt): does it miss here
too, under the current state hash and a mixed one; and where the program LDA.X STA.Y T.mul STA.X T.reduce's states
go in the search (each prefix's state: its level, fresh or not, and any other state with the same key). Not a lesson.

    env: STEP (619), HASH (weak | mixed | both)
"""
import json
import os
import time

import torch

from integration.lib import libcore as L
from integration.lib.round_hash_diag import lib_of, mixed, weak

dev = "cuda" if torch.cuda.is_available() else "cpu"
here = os.path.dirname(os.path.abspath(__file__))
step = os.environ.get("STEP", "619")
rows = json.load(open(os.path.join(here, "s3g5_pools.json")))["pools"][step]
dig = lambda v: [int(ch) for ch in reversed(str(v))][:8] + [0] * (8 - min(8, len(str(v))))
ans = tuple(torch.tensor([dig(r[i]) for r in rows], device=dev) for i in range(3))
lib = lib_of(os.environ.get("LIB", "s3g5"))
cells = L.Cells(10, dev)
good = ["LDA.X", "STA.Y", "T.mul", "STA.X", "T.reduce"]
print("pool step", step, len(rows), "rows; the program's fit:", L.round_fit(L.parse(" ".join(good)), lib, cells, ans, 16, dev),
      flush=True)
for hname in (["weak", "mixed"] if os.environ.get("HASH", "both") == "both" else [os.environ["HASH"]]):
    L.hash_rows = {"weak": weak, "mixed": mixed}[hname]
    prob = L.round_problem(cells, ans, 8, 16, dev)
    st = prob.settings[0]
    feats, keys = [], []
    for k in range(len(good) + 1):                                   # the program's prefix states and their keys
        b = prob.roots[0].copy()
        L.run_items(st, b, L.parse(" ".join(good[:k])), lib)
        feats.append(b.features()[0])
        keys.append(int(L.keys_of([b])[0]))
    print(hname, "prefix keys", keys, "goal reached by the program:", bool(prob.goal([b])[0]), flush=True)
    seen_log = []
    orig_fresh, orig_keys = L._fresh, L.keys_of

    def fresh(seen, ks):
        m = orig_fresh(seen, ks)
        for j, kk in enumerate(keys):
            hit = torch.nonzero(ks == kk).flatten()
            if hit.numel():
                seen_log.append((j, int(hit.numel()), int(m[hit].sum())))
        return m

    def keys_of(batches):
        ks = orig_keys(batches)
        if len(batches) == 1:
            F = batches[0].features()
            for j, kk in enumerate(keys):
                hit = torch.nonzero(ks == kk).flatten()
                for i in hit.tolist():
                    if not bool((F[i] == feats[j]).all()):
                        print(f"   COLLISION: a state with prefix {j}'s key differs from it in "
                              f"{int((F[i] != feats[j]).sum())} bytes", flush=True)
        return ks
    L._fresh, L.keys_of = fresh, keys_of
    c = L.Climb(cells, dev, 16)
    c.lib = dict(lib)
    t0 = time.time()
    progs, rec = c.search_round(ans)
    L._fresh, L.keys_of = orig_fresh, orig_keys
    agg = {}
    for j, nhit, nfresh in seen_log:
        a = agg.setdefault(j, [0, 0])
        a[0] += nhit
        a[1] += nfresh
    print(hname, "ROUND", rec["found_cost"], rec["level_sizes"], rec["stop"], rec["programs"][:4],
          f"{time.time() - t0:.1f}s", flush=True)
    print(hname, "prefix states met as candidates (prefix: [times, of them fresh]):", agg, flush=True)
