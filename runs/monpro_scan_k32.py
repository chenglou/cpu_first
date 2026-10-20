"""The machines the width law rejects without a witness (h100_search_witness_k32: certify, width FAIL, no witness), on
MonPro(s, s) directly: operands s just below the closure bound s* (low 4 bits all ones, the iteration-1 worst case) and
uniform s < s*, for the largest 24-bit Blum moduli, against exact Montgomery with R' = 2^(Km - 2). Positive control:
the same wirings at R = 2^26 (W = 26), which the witness rows break.
argv: spec_dir device rows_per_modulus out_json"""
import collections, json, math, sys, time
import numpy as np, torch

sys.path.insert(0, sys.argv[1])
import spec_grammar as sg

dev, n_rows, CF = torch.device(sys.argv[2]), int(sys.argv[3]), sys.argv[1]
combos, audit = sg.combos_from_audit(CF + "/runs/h100_search_nowidth_k32.json")
ms = json.load(open(CF + "/runs/h100_search_witness_k32.json"))["out"]["machines"]
odd = [c for c, m in zip(combos, ms) if m["certify"] and m["width_law"] and not any(m["wrong_rows"].values())]
key = lambda c: tuple(c[f] for f in sg.FAMILY["r4"] + ["ppdir"])
shapes = {key(c) for c in odd}
ctrl = [c for c, m in zip(combos, ms) if m["width_law"] and any(m["wrong_rows"].values()) and key(c) in shapes
        and sg.GRAMMAR["Ke"][c["Ke"]] == 26 and sg.GRAMMAR["W"][c["W"]] == 26 and m["digit_scale"] == 2]
print(f"{len(odd)} machines without a witness ({len(shapes)} wirings); {len(ctrl)} same-wiring controls at R = 2^26")
ps = [p for p in range(4095, 3000, -1) if p % 4 == 3 and sg.is_prime(p)]
big = sorted({p * q for p in ps for q in ps if p < q and (p * q).bit_length() == 24}, reverse=True)[:4]
rng = np.random.default_rng(0)
out = {}
for name, group in (("no witness", odd), ("control R=2^26", ctrl)):
    for km in sorted({sg.GRAMMAR["Km"][c["Km"]] for c in group}):
        part = [c for c in group if sg.GRAMMAR["Km"][c["Km"]] == km]
        Rp = 1 << (km - 2)
        space = sg.Space.of(part)
        bad, t0, rows = np.zeros(len(part), bool), time.time(), 0
        for N in big:
            smax = (Rp - math.isqrt(Rp * Rp - 4 * Rp * N)) // 2  # s* for this N: s < s* stays below s*
            top = smax - 1 - 16 * np.arange(n_rows // 2)
            top = top - ((top - 15) % 16)  # low 4 bits all ones
            s = np.concatenate([top[top > 0], rng.integers(1, smax, n_rows // 2)])
            mask = Rp - 1
            ninv = (-pow(N, -1, Rp)) & mask
            m = (((s * s) & mask) * ninv) & mask
            want = (s * s + m * N) >> (km - 2)
            for b in range(0, len(s), 8192):
                sb, wb = s[b:b + 8192], want[b:b + 8192]
                L = sg.Lanes(space, torch.arange(len(part), device=dev), id_range=(0, len(part) - 1))
                S, Nw = sg.bin_words(sb, dev), sg.bin_words(np.full(len(sb), N), dev)
                got = sg.monpro(L, S, S, Nw, L.wmask)
                bad |= ~sg.alive_equal(L, got, sg.bin_words(wb, dev)).cpu().numpy()
                rows += len(sb)
        out[f"{name}, Km={km}"] = {"machines": len(part), "rows": rows, "machines_with_a_wrong_product": int(bad.sum()),
                                   "seconds": time.time() - t0}
        print(f"{name}, Km = {km} (R' = 2^{km - 2}): {len(part)} machines x {rows} products: "
              f"{int(bad.sum())} machines with a wrong product ({time.time() - t0:.0f}s)", flush=True)
json.dump(out, open(sys.argv[4], "w"), indent=1)
