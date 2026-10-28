"""fp64 'analog' exact arithmetic with learned encoders: how close must learned place weights be?

Model: v_x = sum_i w_i x_i and v_N = sum_i u_i N_i (learned place weights over LSB-first decimal digits), optionally
snapped to the nearest integer right after the encoder (straight-through gradient), then exact blocks in floating
point: p = v_x * v_x, r = p - v_N floor(p / v_N) with a one-step correction. The answer is round(r); the true
weights are 10^i. Only the encoders are learned, from final answers (loss on r against y).

    python -m microlab.ext_exact_analog basin --bits 8 13 16 20 24
    python -m microlab.ext_exact_analog base       # 1-parameter version v = sum b^i d_i: loss and basin in b
    python -m microlab.ext_exact_analog fp32       # is the exact forward even representable?
    python -m microlab.ext_exact_analog random     # random init, with and without a curriculum over sizes
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from . import ext_exact as E
from .tasks import TASKS

torch.set_default_dtype(torch.float64)   # parameters too: a float32 base b is quantised to ~1e-6 steps near 10


def digit_rows(n, bits, rng):
    b = TASKS["sqmod"].sample(n, bits, rng)
    w = b.inputs["N"].shape[1]
    place = 10 ** np.arange(w, dtype=np.int64)
    return {"Nd": b.inputs["N"], "xd": b.inputs["x"], "y": (b.target * place).sum(1),
            "N": (b.inputs["N"] * place).sum(1), "x": (b.inputs["x"] * place).sum(1), "w": w}


def forward(wx, wN, xd, Nd, snap, dtype):
    vx = xd @ wx
    vN = Nd @ wN
    if snap:
        vx = vx + (torch.round(vx) - vx).detach()
        vN = vN + (torch.round(vN) - vN).detach()
    p = vx * vx
    q = torch.floor(p / vN)
    r = p - q * vN
    r = torch.where(r < 0, r + vN, r)
    r = torch.where(r >= vN, r - vN, r)
    return r, vx, vN


def loss_of(r, y, N, kind):
    if kind == "circ":
        return (1 - torch.cos(2 * math.pi * (r - y) / N)).mean()
    return torch.log1p(((r - y) / N) ** 2).mean()


def predict(r, vN):
    """The fixed read-out after the remainder block: round, then wrap into [0, round(v_N)) (an r just below v_N is
    the residue 0 seen from below)."""
    yh, n = torch.round(r), torch.round(vN)
    yh = torch.where(yh >= n, yh - n, yh)
    return torch.where(yh < 0, yh + n, yh)


def exact_rate(wx, wN, rows, snap, dtype):
    t = lambda a: torch.as_tensor(a, dtype=dtype)
    with torch.no_grad():
        r, _, vN = forward(t(wx), t(wN), t(rows["xd"]), t(rows["Nd"]), snap, dtype)
    return float((predict(r, vN).to(torch.float64).numpy() == rows["y"]).mean())


def train(w0x, w0N, rows, snap, steps=1500, lr0=None, loss="circ", dtype=torch.float64, lr_floor=1e-7):
    t = lambda a: torch.as_tensor(a, dtype=dtype)
    wx = torch.nn.Parameter(t(w0x).clone())
    wN = torch.nn.Parameter(t(w0N).clone())
    opt = torch.optim.Adam([wx, wN], lr=lr0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_floor ** (s / max(steps - 1, 1)))
    xd, Nd, y, N = t(rows["xd"]), t(rows["Nd"]), t(rows["y"]), t(rows["N"])
    for _ in range(steps):
        r, _, _ = forward(wx, wN, xd, Nd, snap, dtype)
        l = loss_of(r, y, N, loss)
        opt.zero_grad()
        l.backward()
        for p in (wx, wN):
            p.grad.nan_to_num_(0.0, 0.0, 0.0)
        opt.step()
        sched.step()
    return wx.detach().to(torch.float64).numpy(), wN.detach().to(torch.float64).numpy()


def cmd_basin(a):
    """Init = true weights + sigma * gaussian (absolute, every weight). Train; success = exact on held-out rows."""
    out = []
    for bits in a.bits:
        train_rows = digit_rows(1000, bits, np.random.default_rng(1))
        held = digit_rows(2000, bits, np.random.default_rng(2))
        w = train_rows["w"]
        true = 10.0 ** np.arange(w)
        for snap in [bool(s) for s in a.snaps]:
            for sigma in a.sigmas:
                init_ok = wins = train_ok = 0
                dv = []
                for seed in range(a.seeds):
                    g = np.random.default_rng(100 + seed)
                    w0x = true + sigma * g.standard_normal(w)
                    w0N = true + sigma * g.standard_normal(w)
                    dv.append(np.sqrt(np.mean((train_rows["xd"] @ (w0x - true)) ** 2)))
                    init_ok += exact_rate(w0x, w0N, held, snap, torch.float64) == 1.0
                    ok = tr = False
                    for mult in a.lr_mults:      # best of a few step sizes, scaled to the perturbation
                        wx, wN = train(w0x, w0N, train_rows, snap, steps=a.steps, lr0=mult * sigma, loss=a.loss)
                        ok = exact_rate(wx, wN, held, snap, torch.float64) == 1.0
                        tr = tr or ok or exact_rate(wx, wN, train_rows, snap, torch.float64) == 1.0
                        if ok:
                            break
                    wins += ok
                    train_ok += tr
                rec = {"bits": bits, "snap": snap, "sigma": sigma, "rms_dv": float(np.mean(dv)),
                       "init_exact": init_ok, "trained_exact": wins, "trained_exact_on_train_rows": train_ok,
                       "seeds": a.seeds}
                out.append(rec)
                print(f"  bits {bits:2d} snap {int(snap)} sigma {sigma:.0e} (rms error of v_x {np.mean(dv):.2e}, "
                      f"relative {np.mean(dv) / 2 ** bits:.1e}): exact at init {init_ok}/{a.seeds}, "
                      f"after training {wins}/{a.seeds} (exact on the 1000 training rows: {train_ok}/{a.seeds})", flush=True)
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "analog_basin", "loss": a.loss, "steps": a.steps, "lr_mults": a.lr_mults,
                             "out": out}) + "\n")


def cmd_base(a):
    """One shared learned base b: v = sum_i b^i d_i for x and N. Loss along b = 10 + delta; basin of Adam in b."""
    out = []
    for bits in a.bits:
        rows = digit_rows(1000, bits, np.random.default_rng(1))
        held = digit_rows(2000, bits, np.random.default_rng(2))
        w = rows["w"]
        xd, Nd = torch.as_tensor(rows["xd"], dtype=torch.float64), torch.as_tensor(rows["Nd"], dtype=torch.float64)
        y, N = torch.as_tensor(rows["y"], dtype=torch.float64), torch.as_tensor(rows["N"], dtype=torch.float64)
        line = []
        for snap in (False, True):
            for k in range(1, 14):
                delta = 10.0 ** -k
                b = torch.tensor(10.0 + delta, dtype=torch.float64)
                pw = b ** torch.arange(w, dtype=torch.float64)
                with torch.no_grad():
                    r, _, _ = forward(pw, pw, xd, Nd, snap, torch.float64)
                    l = float(loss_of(r, y, N, "circ"))
                    ex = float((predict(r, Nd @ pw).numpy() == rows["y"]).mean())
                line.append({"snap": snap, "delta": delta, "loss": l, "exact": ex})
        # gradient descent on b from 10 + delta
        gd = []
        for delta in [10.0 ** -k for k in range(1, 12)]:
            for snap in (False, True):
                wins, finals = 0, []
                for sign in (1.0, -1.0):
                    for mult in (0.3, 0.03):     # never exactly the distance (Adam's first step is +-lr)
                        b = torch.nn.Parameter(torch.tensor(10.0 + sign * delta, dtype=torch.float64))
                        opt = torch.optim.Adam([b], lr=mult * delta)
                        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1e-7 ** (s / 1499))
                        for _ in range(1500):
                            pw = b ** torch.arange(w, dtype=torch.float64)
                            r, _, _ = forward(pw, pw, xd, Nd, snap, torch.float64)
                            l = loss_of(r, y, N, "circ")
                            opt.zero_grad(); l.backward(); b.grad.nan_to_num_(0.0, 0.0, 0.0); opt.step()
                            sched.step()
                        pw = (b.detach() ** torch.arange(w, dtype=torch.float64)).numpy()
                        if exact_rate(pw, pw, held, snap, torch.float64) == 1.0:
                            wins += 1
                            finals.append(float(b.detach()) - 10)
                            break
                gd.append({"delta": delta, "snap": snap, "wins_of_2_starts": wins, "final_b_minus_10": finals,
                           "exact": wins / 2})
        out.append({"bits": bits, "scan": line, "gd": gd})
        print(f"  bits {bits}: loss/exact along b = 10 + delta (no snap): " +
              " ".join(f"{d['delta']:.0e}:{d['loss']:.2f}/{d['exact']:.2f}" for d in line if not d["snap"]), flush=True)
        print(f"           with snap: " +
              " ".join(f"{d['delta']:.0e}:{d['loss']:.2f}/{d['exact']:.2f}" for d in line if d["snap"]), flush=True)
        print(f"           Adam from b = 10 + delta, exact after: " +
              " ".join(f"{g['delta']:.0e}{'s' if g['snap'] else ''}:{g['exact']:.2f}" for g in gd), flush=True)
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "analog_base", "out": out}) + "\n")


def cmd_fp32(a):
    """Exactness of the forward pass at the true weights, by float width and modulus size."""
    out = []
    for bits in a.bits:
        rows = digit_rows(2000, bits, np.random.default_rng(2))
        true = 10.0 ** np.arange(rows["w"])
        res = {"bits": bits}
        for name, dt in (("fp32", torch.float32), ("fp64", torch.float64)):
            for snap in (False, True):
                res[f"{name}_snap{int(snap)}"] = exact_rate(true, true, rows, snap, dt)
        out.append(res)
        print("  " + "  ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "analog_fp32", "out": out}) + "\n")


def cmd_random(a):
    """Random init (w_i = 10^i * U(0.5, 1.5)); train at the target size, or through a curriculum of sizes."""
    out = []
    for mode in ("direct", "curriculum"):
        for snap in (False, True):
            wins = 0
            for seed in range(a.seeds):
                g = np.random.default_rng(500 + seed)
                bits_seq = [24] if mode == "direct" else [4, 7, 10, 13, 16, 20, 24]
                w = 8
                wx = 10.0 ** np.arange(w) * g.uniform(0.5, 1.5, w)
                wN = 10.0 ** np.arange(w) * g.uniform(0.5, 1.5, w)
                for bits in bits_seq:
                    rows = digit_rows(1000, bits, np.random.default_rng(seed + bits))
                    pad = lambda d: np.pad(d, ((0, 0), (0, w - d.shape[1])))
                    rows = {**rows, "xd": pad(rows["xd"]), "Nd": pad(rows["Nd"])}
                    wx, wN = train(wx, wN, rows, snap, steps=a.steps, lr0=a.lr, loss="circ", lr_floor=1e-4)
                held = digit_rows(2000, 24, np.random.default_rng(2))
                ok = exact_rate(wx, wN, held, snap, torch.float64) == 1.0
                wins += ok
                rel = np.abs(wx / 10.0 ** np.arange(w) - 1)
                print(f"  {mode} snap {int(snap)} seed {seed}: exact {ok}; |w_i/10^i - 1| = "
                      f"{' '.join(f'{v:.1e}' for v in rel)}", flush=True)
            out.append({"mode": mode, "snap": snap, "wins": wins, "seeds": a.seeds})
            print(f"  => {mode} snap {int(snap)}: {wins}/{a.seeds}", flush=True)
    with open(E.LOG, "a") as fh:
        fh.write(json.dumps({"exp": "analog_random", "steps": a.steps, "lr": a.lr, "out": out}) + "\n")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("basin")
    s.add_argument("--bits", type=int, nargs="+", default=[8, 13, 16, 20, 24])
    s.add_argument("--sigmas", type=float, nargs="+",
                   default=[1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0])
    s.add_argument("--seeds", type=int, default=6)
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--loss", default="circ", choices=["circ", "mse"])
    s.add_argument("--lr_mults", type=float, nargs="+", default=[1.0, 0.1])
    s.add_argument("--snaps", type=int, nargs="+", default=[0, 1])
    s = sub.add_parser("base")
    s.add_argument("--bits", type=int, nargs="+", default=[8, 13, 16, 20, 24])
    s = sub.add_parser("fp32")
    s.add_argument("--bits", type=int, nargs="+", default=[8, 10, 11, 12, 13, 16, 20, 24, 26, 27, 28])
    s = sub.add_parser("random")
    s.add_argument("--seeds", type=int, default=5)
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--lr", type=float, default=0.05)
    for sp in sub.choices.values():
        sp.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    {"basin": cmd_basin, "base": cmd_base, "fp32": cmd_fp32, "random": cmd_random}[a.cmd](a)


if __name__ == "__main__":
    main()
