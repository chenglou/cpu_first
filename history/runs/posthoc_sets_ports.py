"""Reproduce the post-hoc label-free chain (posthoc_edge.py of the previous session) on the ports run's 45,216
training-consistent machines and save, per machine, which checks it survives. Uses a frozen copy of spec_grammar.py.
argv: spec_dir data_dir device out_json"""
import collections, json, random, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
import spec_grammar as sg
sg.DATA = sys.argv[2]
dev = torch.device(sys.argv[3])
CF = "/Users/chenglou/Library/Application Support/Claude/scratch-workspaces/2037fc6c-747d-4cf1-a5f5-8a4697a4e390/5ef58abe-646a-4ac9-9fae-d0ea15914ce2/scratch-2026-10-19-536f87/cpu_first"
audit = json.load(open(CF + "/runs/h100_search_law_ports.json"))["out"]["audit"]


def to_idx(f, v):
    if f == "gate":
        v = int(v, 16)
    elif isinstance(v, list):
        v = tuple(v)
    return sg.GRAMMAR[f].index(v)


combos = [dict(sg.REF_IDX, **{f: to_idx(f, v) for f, v in a["machine"].items()}) for a in audit]
cert = np.array([a["seen_maxT"] == 64 and a["ood_maxT"] == 64 for a in audit])
fam_of = np.array([c["mult"] for c in combos])
rng = random.Random(7)
ps = [p for p in range(4095, 3800, -1) if p % 4 == 3 and sg.is_prime(p)]
Ns, Xs = [], []
for i in range(32):
    p, q = rng.sample(ps[:24], 2)
    N = p * q
    Ns.append(N)
    Xs.append(N - 1 - rng.randrange(0, 64) if i % 2 else rng.randrange(N // 2, N))
edge = sg.LawRows(np.array(Ns), np.array(Xs), dev)


class EdgeRows:
    def __init__(self, N, X):
        self.N, self.X = N, X
        self.Nd = torch.as_tensor(sg.digits_of(N, sg.ND), device=dev)
        self.Xd = torch.as_tensor(sg.digits_of(X, sg.ND), device=dev)
        self.n = len(N)

    def sub(self, sl):
        return EdgeRows(self.N[sl], self.X[sl])


erows = EdgeRows(np.array(Ns), np.array(Xs))
syn = sg.LawRows(*sg.blum_moduli(24, 16), dev)
tr = sg.load("train")
prompts = sg.Rows(tr, np.arange(8), dev)
alive = np.ones(len(combos), bool)
t0 = time.time()
checks = [(f"{law}@24bit", lambda L, r, law=law: sg.law_sides(L, r, law), syn) for law in ["id", "comm", "assoc"]]
checks.append(("semigroup 50+49 (prompts)", lambda L, r: sg.semigroup_sides(L, r, 50, 49), prompts))
checks += [(f"{law}@edge", lambda L, r, law=law: sg.law_sides(L, r, law), edge) for law in ["assoc", "mirror2"]]
checks.append(("semigroup 32+32 (edge)", lambda L, r: sg.semigroup_sides(L, r, 32, 32), erows))
after = {}
for name, fn, rows in checks:
    for fam in np.unique(fam_of):
        idx = np.where(alive & (fam_of == fam))[0]
        if len(idx) == 0:
            continue
        sp = sg.Space.of([combos[i] for i in idx])
        surv = sg.screen(sp, fn, None, [rows.sub(slice(0, 1)), rows.sub(slice(1, rows.n))], dev, 1 << 20, sg.Stats())
        keep = np.zeros(len(idx), bool)
        keep[surv.cpu().numpy()] = True
        alive[idx[~keep]] = False
    after[name] = alive.copy()
    by = collections.Counter(sg.GRAMMAR["mult"][fam_of[i]] for i in np.where(alive)[0])
    print(f"after {name}: {alive.sum()} machines {dict(by)}; certify among them: {(alive & cert).sum()}; "
          f"not certifying: {(alive & ~cert).sum()}; certifying removed: {(~alive & cert).sum()} ({time.time() - t0:.0f}s)",
          flush=True)
json.dump({"cert": cert.tolist(), "after": {k: v.tolist() for k, v in after.items()}}, open(sys.argv[4], "w"))
