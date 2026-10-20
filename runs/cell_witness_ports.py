"""Every machine that survives the current label-free checks (4,480), on declared-range witness rows: 24-bit Blum N
near 2^24 with random x (T = 64), plus the rows whose exact chain reaches an operand >= 1.5 2^24 (exhaustive over x
for the largest moduli). Cross-tabulated with the per-cell laws' verdicts.
argv: spec_dir data_dir device sets_json cell_audit_json chain_json n_random out_json"""
import collections, json, random, sys, time
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
import spec_grammar as sg

sg.DATA = sys.argv[2]
dev = torch.device(sys.argv[3])
sets, cells = json.load(open(sys.argv[4])), json.load(open(sys.argv[5]))["machines"]
chain, n_random = json.load(open(sys.argv[6])), int(sys.argv[7])
CF = sys.argv[1]
combos, audit = sg.combos_from_audit(CF + "/runs/h100_search_law_ports.json")
alive = np.array(sets["after"][list(sets["after"])[-1]])
idx = np.nonzero(alive)[0]

rng = random.Random(1)
ps = [p for p in range(4095, 3000, -1) if p % 4 == 3 and sg.is_prime(p)]
Ns, Xs, Ts, src = [], [], [], []
while len(Ns) < n_random:
    p, q = rng.sample(ps[:40], 2)
    if (p * q).bit_length() == 24:
        Ns.append(p * q), Xs.append(rng.randrange(1, p * q)), Ts.append(64), src.append("random")
for h in chain:
    for t in sorted({min(h["t"] + 1, 64), 64}):
        Ns.append(h["N"]), Xs.append(h["x"]), Ts.append(t), src.append("chain")
d = {"N": np.array(Ns), "X": np.array(Xs), "T": np.array(Ts),
     "Y": np.array([pow(int(x), 1 << int(t), int(N)) for x, N, t in zip(Xs, Ns, Ts)])}
src = np.array(src)
print(f"{len(idx)} machines x {len(Ns)} rows ({(src == 'random').sum()} random near 2^24, {(src == 'chain').sum()} "
      f"from the chain search)", flush=True)
t0 = time.time()
wrong = {}
for name in ("random", "chain"):
    rows = np.nonzero(src == name)[0]
    cnt = sg.evaluate_lanes([combos[i] for i in idx], sg.Rows(d, rows, dev), dev, batch=4096)
    wrong[name] = len(rows) - cnt
print(f"evaluated in {time.time() - t0:.0f}s", flush=True)
witness = (wrong["random"] > 0) | (wrong["chain"] > 0)
cert = np.array([audit[i]["seen_maxT"] == 64 and audit[i]["ood_maxT"] == 64 for i in idx])
vlaw = np.array([cells[i]["value_law"] == "" for i in idx])
wlaw = np.array([cells[i]["width_law"] == "" for i in idx])
tab = collections.Counter()
for c, v, w, x in zip(cert, vlaw, wlaw, witness):
    tab[("certify" if c else "not certify", "value ok" if v else "value FAIL", "width ok" if w else "width FAIL",
         "WITNESS" if x else "no witness")] += 1
for k, n in sorted(tab.items()):
    print(f"  {k}: {n}", flush=True)
json.dump({"rows": {"random": int((src == 'random').sum()), "chain": int((src == 'chain').sum())},
           "table": {" | ".join(k): n for k, n in sorted(tab.items())},
           "machines": [{"i": int(i), "certify": bool(c), "value_ok": bool(v), "width_ok": bool(w),
                         "wrong_random": int(a), "wrong_chain": int(b)}
                        for i, c, v, w, a, b in zip(idx, cert, vlaw, wlaw, wrong["random"], wrong["chain"])]},
          open(sys.argv[8], "w"), indent=1)
