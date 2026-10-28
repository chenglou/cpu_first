"""Relaxation 3 end to end: train the periodic step on fresh 24-bit data (small-s curriculum, Levenberg-Marquardt,
fp64), then loop it with integer snapping: own T-ladder at 24 bits and the real H1 certification profiles."""
import json
import sys
import time

import numpy as np
import torch

import h1
import relax_common as rc
import relax_periodic as P

torch.set_default_dtype(torch.float64)
out = {}
for kind in ("mono", "bilinear"):
    for seed in range(3):
        t0 = time.time()
        m = P.PeriodicStep(kind, K=4, seed=seed)
        m.phase_small_init(24, 1e-3, seed)
        r = P.train_stagewise(m, 24, curriculum="smax", stages=48, seed=seed, opt="gn")
        train_t = time.time() - t0
        big = P.exact_frac(m, 24, n=200000, seed=11)
        lad = rc.ladder(P.step_fn(m), 24, verbose=False)

        def predict(N, X, T):
            s = X % N
            for tv in np.unique(T):
                k = T == tv
                c = s[k].copy()
                for _ in range(int(tv)):
                    c = P.step_fn(m)(N[k], c)
                s[k] = c
            return h1.digits(s)

        cert = {p: h1.certify(predict, p, verbose=False) for p in ("depth_t", "depth_ood_n_t")}
        key = f"{kind}_seed{seed}"
        out[key] = {"train_time": train_t, "single_step_24b_200k": big, "ladder_24b": lad, **cert,
                    "wall": time.time() - t0,
                    "params": {k: v.detach().numpy().tolist() for k, v in m.named_parameters()}}
        print(key, json.dumps({k: v for k, v in out[key].items() if k != "params"}), flush=True)
        if kind == "mono":
            print("   learned w (1, s, N, s^2, sN, N^2) =", m.w.detach().numpy().tolist(), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
