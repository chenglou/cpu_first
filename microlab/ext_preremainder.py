"""Learning before the remainder: does seeing x^2 (the value before mod N) help a network learn x^2 mod N?

Registers into microlab's TASKS / MODELS:
- task  `sqb`   : x -> x^2, with x drawn exactly as in sqmod at b bits (the multiplication part alone).
- task  `modsq` : (N, v = x^2) -> x^2 mod N, aux q = x^2 // N (the remainder part alone, on sqmod's own values).
- task  `quot`  : (N, v = x^2) -> x^2 // N (the division inside the remainder).
- task  `subq`  : (N, v = x^2, q = x^2 // N) -> x^2 - qN (the remainder once the quotient is known).
- task  `divstep`: (N, R, d) -> (10R + d) mod N, one long-division step; `chain` = that net chained over x^2's digits.
- task  `sqmodx`: sqmod with x < frac * N; `--x_sched a:f0` grows the x range from f0 * N to N (a size curriculum).
- model `ht`    : half-transparent. An xf encoder writes the 2w digits of x^2 as an internal layer ("pre"); a fixed,
                  exact, non-learned module turns them into the answer (digits -> integer -> mod N -> digits).
                  How the final answer's loss reaches the internal digits (`est`):
                    st    : straight-through with the quotient frozen. Answer digit j = internal digit j shifted by the
                            module's own offset (so its argmax is the exact answer digit); CE on the answer reaches
                            internal digits 0..w-1 only, since moving a high digit changes the quotient.
                    local : exact one-digit lookahead. The answer error of every single-digit change of the internal
                            value, weighted by that digit's softmax (a local-expectation gradient).
                    best  : the same lookahead, but log-likelihood of the best single-digit fix(es), only at digits
                            whose value changes the answer (a hindsight relabel through the exact module).
                    rf    : REINFORCE through the exact module (K samples per row, leave-one-out baseline).
                  `--aux pre` supervises the internal digits directly (`pre_hi` / `pre_lo`: only the high / low half);
                  `--final_weight 0` makes that the only loss.

`train()` mirrors microlab.run.run (same init, AdamW, OneCycle, clipping, data stream and evaluation examples, so a
plain run gives the same numbers) and adds: accuracy of every aux head; for ht, whether the internal value P is x^2,
(N - x)^2 or another representative x^2 + kN; held-out moduli (`--holdout 8` trains on 7/8 of the moduli and scores
the rest; tasks without a modulus hold out x values); exact match by quotient size; loss-weight schedules for
curricula; and a learning curve. Results go to microlab/results_preremainder.jsonl;
`python -m microlab.ext_preremainder table [tag]` summarizes. Findings: FINDINGS_pre_remainder.md.

    python -m microlab.ext_preremainder sqmod --size 8 --model xf --aux pre --holdout 8 --threads 2
    python -m microlab.ext_preremainder sqmod --size 8 --model ht --est best --holdout 8 --steps 6000 --threads 2
    python -m microlab.ext_preremainder sqmod --size 8 --model xf --aux pre --aux_sched lin:0.6     # curriculum
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .models import MODELS, Transformer, model
from .run import _tensors
from .tasks import TASKS, Batch, Task, _odd_moduli, digits, ndig

RESULTS = os.path.join(os.path.dirname(__file__), "results_preremainder.jsonl")
P10 = [10 ** i for i in range(18)]


class SqB(Task):
    def __init__(self):
        super().__init__("sqb", "bits", "x -> x^2, x drawn as in sqmod (x < N, N a b-bit odd modulus)")

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"x": w}, 2 * w, {}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        x = (rng.random(n) * N).astype(np.int64)
        w = ndig((1 << b) - 1)
        return Batch({"x": digits(x, w)}, digits(x * x, 2 * w))


class ModSq(Task):
    def __init__(self):
        super().__init__("modsq", "bits", "(N, v = x^2) -> x^2 mod N (aux q = x^2 // N)")

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "v": 2 * w}, w, {"q": w}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        x = (rng.random(n) * N).astype(np.int64)
        w = ndig((1 << b) - 1)
        p = x * x
        return Batch({"N": digits(N, w), "v": digits(p, 2 * w)}, digits(p % N, w), {"q": digits(p // N, w)})


class Quot(Task):
    def __init__(self):
        super().__init__("quot", "bits", "(N, v = x^2) -> x^2 // N (the division inside the remainder)")

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "v": 2 * w}, w, {}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        x = (rng.random(n) * N).astype(np.int64)
        w = ndig((1 << b) - 1)
        p = x * x
        return Batch({"N": digits(N, w), "v": digits(p, 2 * w)}, digits(p // N, w), {"q": digits(p // N, w)})


class SubQ(Task):
    def __init__(self):
        super().__init__("subq", "bits", "(N, v = x^2, q = x^2 // N) -> v - qN (the remainder, quotient known)")

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "v": 2 * w, "q": w}, w, {}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        x = (rng.random(n) * N).astype(np.int64)
        w = ndig((1 << b) - 1)
        p = x * x
        return Batch({"N": digits(N, w), "v": digits(p, 2 * w), "q": digits(p // N, w)}, digits(p % N, w),
                     {"q": digits(p // N, w)})


class DivStep(Task):
    """One step of schoolbook long division: bring down digit d, R <- (10R + d) mod N. Chaining it over the digits of
    x^2 (most significant first, R starting at 0) gives x^2 mod N; train() reports that chain as `chain`."""

    def __init__(self):
        super().__init__("divstep", "bits", "(N, R < N, d) -> (10R + d) mod N (aux q = the quotient digit)")

    def widths(self, b):
        w = ndig((1 << b) - 1)
        return {"N": w, "R": w, "d": 1}, w, {"q": 1}

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        R = (rng.random(n) * N).astype(np.int64)
        d = rng.integers(0, 10, n)
        w = ndig((1 << b) - 1)
        v = 10 * R + d
        return Batch({"N": digits(N, w), "R": digits(R, w), "d": d[:, None]}, digits(v % N, w),
                     {"q": (v // N)[:, None]})


class SqModX(Task):
    """sqmod with x < frac * N, for a size curriculum: train() sets frac from `x_sched`; evaluation uses frac = 1."""

    def __init__(self):
        super().__init__("sqmodx", "bits", "x^2 mod N with x < frac * N (aux pre, q as in sqmod)")
        self.frac = 1.0

    def widths(self, b):
        return TASKS["sqmod"].widths(b)

    def sample(self, n, b, rng):
        N = _odd_moduli(n, b, rng)
        x = (rng.random(n) * N * self.frac).astype(np.int64)
        w = ndig((1 << b) - 1)
        p = x * x
        return Batch({"N": digits(N, w), "x": digits(x, w)}, digits(p % N, w),
                     {"pre": digits(p, 2 * w), "q": digits(p // N, w)})


for _t in (SqB(), ModSq(), Quot(), SubQ(), DivStep(), SqModX()):
    TASKS[_t.name] = _t


@model("ht")
class HalfTransparent(Transformer):
    def __init__(self, in_widths, out_width, aux_widths, est="st", rf_samples=4, **kw):
        assert "N" in in_widths, "ht needs the modulus as an input"
        # The encoder's "target" head is the internal x^2 (2w digits); the answer comes from the fixed module.
        super().__init__(in_widths, 2 * out_width, {k: v for k, v in aux_widths.items() if not k.startswith("pre")},
                         **kw)
        self.ans_width, self.est, self.rf_samples = out_width, est, rf_samples
        self.register_buffer("p10", torch.tensor(P10), persistent=False)

    def value(self, d):                       # LSB-first digits [..., L] -> int64 [...]
        return (d * self.p10[: d.shape[-1]]).sum(-1)

    def digits(self, v, w):                   # int64 [...] -> LSB-first digits [..., w]
        return torch.div(v.unsqueeze(-1), self.p10[:w], rounding_mode="floor") % 10

    def forward(self, inputs):
        o = super().forward(inputs)
        pre, w = o.pop("target"), self.ans_width
        with torch.no_grad():
            d = pre.argmax(-1)
            r = self.value(d) % self.value(inputs["N"])
            c = (d[:, :w] - self.digits(r, w)) % 10
        o["target"] = pre[:, :w].gather(-1, (torch.arange(10, device=pre.device) + c.unsqueeze(-1)) % 10)
        o["pre"] = pre
        return o

    def final_loss(self, inputs, out, y):
        if self.est == "st":
            return F.cross_entropy(out["target"].flatten(0, 1), y.flatten())
        pre, w = out["pre"], self.ans_width
        N = self.value(inputs["N"])
        if self.est in ("local", "best"):
            with torch.no_grad():
                d = pre.argmax(-1)                                                    # [B, L]
                L = d.shape[1]
                alt = self.value(d)[:, None, None] + (torch.arange(10, device=d.device) - d.unsqueeze(-1)) * \
                    self.p10[:L, None]                                                # [B, L, 10] one-digit changes
                cost = (self.digits(alt % N[:, None, None], w) != y[:, None, None]).float().mean(-1)
            if self.est == "local":
                return (pre.softmax(-1) * cost).sum((1, 2)).mean()
            # best: log-likelihood of the best single-digit fix(es), only at digits whose value changes the answer
            lo, hi = cost.min(-1, keepdim=True).values, cost.max(-1).values
            lp = pre.log_softmax(-1).masked_fill(cost > lo, float("-inf")).logsumexp(-1)
            return -(lp * (hi > lo.squeeze(-1))).sum(1).mean()
        if self.est == "rf":
            K = self.rf_samples
            with torch.no_grad():
                s = torch.distributions.Categorical(logits=pre).sample((K,))          # [K, B, L]
                R = -(self.digits(self.value(s) % N, w) != y).float().mean(-1)        # [K, B]
                base = (R.sum(0, keepdim=True) - R) / (K - 1)
            lp = pre.log_softmax(-1).unsqueeze(0).expand(K, -1, -1, -1).gather(-1, s.unsqueeze(-1)).sum((2, 3))
            return -((R - base) * lp).mean()
        raise ValueError(self.est)


# ---------------------------------------------------------------------------------------------------- training


def _subset(b: Batch, idx) -> Batch:
    return Batch({k: v[idx] for k, v in b.inputs.items()}, b.target[idx], {k: v[idx] for k, v in b.aux.items()})


def _held(b: Batch, holdout: int) -> np.ndarray:
    d = b.inputs["N"] if "N" in b.inputs else next(iter(b.inputs.values()))    # the modulus, else the first input
    v = (d * (10 ** np.arange(d.shape[1]))).sum(1)
    return (v - 1) // 2 % holdout == 3 % holdout


def sample(task, n, size, rng, holdout=0, held=False) -> Batch:
    """task.sample, restricted to the training moduli (held=False) or the held-out ones when holdout > 0
    (tasks without a modulus hold out values of their first input instead)."""
    b = task.sample(n, size, rng)
    if not holdout:
        return b
    parts, have = [], 0
    while True:
        keep = np.nonzero(_held(b, holdout) == held)[0]
        parts.append(_subset(b, keep))
        have += len(keep)
        if have >= n:
            break
        b = task.sample(n, size, rng)
    cat = Batch({k: np.concatenate([p.inputs[k] for p in parts]) for k in parts[0].inputs},
                np.concatenate([p.target for p in parts]),
                {k: np.concatenate([p.aux[k] for p in parts]) for k in parts[0].aux})
    return _subset(cat, np.arange(n))


def weight(spec: str, t: float) -> float:
    """Loss-weight schedule at training progress t in [0, 1): const | off | down:a | up:a | lin:a (1 -> 0 by a)."""
    if spec in ("", "const"):
        return 1.0
    if spec == "off":
        return 0.0
    kind, _, at = spec.partition(":")
    at = float(at or 0.5)
    if kind == "down":
        return 1.0 if t < at else 0.0
    if kind == "up":
        return 0.0 if t < at else 1.0
    if kind == "lin":
        return max(0.0, 1.0 - t / at)
    raise ValueError(spec)


def x_range(spec: str, t: float) -> float:
    """x-range fraction at progress t for `x_sched` "a:f0": from f0 up to 1, linearly, reached at progress a."""
    at, _, f0 = spec.partition(":")
    f0 = float(f0 or 0.1)
    return f0 + (1 - f0) * min(1.0, t / float(at))


@torch.no_grad()
def evaluate(net, task, size, n=2000, seed=10_000, holdout=0, held=False):
    if hasattr(task, "frac"):
        task.frac = 1.0                       # always score the full range; train() resets it every step
    b = sample(task, n, size, np.random.default_rng(seed), holdout, held)
    dev = next(net.parameters()).device
    x, y, a = _tensors(b, dev)
    out = net(x)
    ok = (out["target"].argmax(-1) == y).cpu()
    res = {"exact": ok.all(1).float().mean().item(), "per_digit": ok.float().mean(0).tolist()}
    if "q" in b.aux:                          # exact match by quotient x^2 // N: is it only the no-reduction rows?
        q = (b.aux["q"] * (10 ** np.arange(b.aux["q"].shape[1]))).sum(1)
        ex = ok.all(1).numpy()
        res["by_q"] = {name: [float(ex[m].mean()) if m.any() else None, float(m.mean())] for name, m in
                       (("0", q == 0), ("1", q == 1), ("2-3", (q >= 2) & (q <= 3)), ("4+", q >= 4))}
    for k, v in out.items():
        if k != "target" and k in a:
            okk = (v.argmax(-1) == a[k]).cpu()
            res[k] = {"exact": okk.all(1).float().mean().item(), "per_digit": okk.float().mean(0).tolist()}
    if isinstance(net, HalfTransparent):
        P, N, X2 = net.value(out["pre"].argmax(-1)), net.value(x["N"]), net.value(a["pre"])
        res["pre"]["congruent"] = ((P - X2) % N == 0).float().mean().item()   # a valid representative of x^2 mod N
        res["pre"]["below_N"] = (P < N).float().mean().item()                # the module is a no-op
        res["pre"]["mirror"] = (P == (N - net.value(x["x"])) ** 2).float().mean().item()  # P = (N - x)^2
        k = torch.div(P - X2, N, rounding_mode="floor")[(P - X2) % N == 0]   # which representative: P = x^2 + kN
        vals, counts = k.unique(return_counts=True)
        top = counts.argsort(descending=True)[:6]
        res["pre"]["k"] = {str(vals[i].item()): counts[i].item() / len(P) for i in top}
    return res


def _r(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return [_r(u) for u in v]
    if isinstance(v, dict):
        return {k: _r(u) for k, u in v.items()}
    return v


def train(task_name, size, model="xf", aux=(), aux_weight=1.0, aux_sched="const", final_weight=1.0,
          final_sched="const", steps=3000, batch=256, lr=None, seeds=1, seed0=0, pool=0, holdout=0, ood=1,
          eval_every=0, x_sched="", device="cpu", tag="", verbose=True, log=RESULTS, **model_kw):
    task = TASKS[task_name]
    in_w, out_w, aux_w = task.widths(size)
    use_aux = {k: aux_w[k.partition("_")[0]] for k in aux}     # pre_hi / pre_lo: half of ht's internal x^2
    lr = lr or (3e-3 if model == "mlp" else 5e-4)
    results = []
    for seed in range(seed0, seed0 + seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = MODELS[model](in_w, out_w, use_aux, **model_kw).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
        fixed = sample(task, pool, size, rng, holdout) if pool else None
        t0, curve = time.time(), []
        for step in range(steps):
            if x_sched:
                task.frac = x_range(x_sched, step / steps)
            b = _subset(fixed, rng.integers(0, pool, batch)) if fixed is not None else \
                sample(task, batch, size, rng, holdout)
            x, y, a = _tensors(b, device)
            out = net(x)
            fw, aw = final_weight * weight(final_sched, step / steps), aux_weight * weight(aux_sched, step / steps)
            loss = 0.0
            if fw:
                loss = fw * (net.final_loss(x, out, y) if hasattr(net, "final_loss") else
                             F.cross_entropy(out["target"].flatten(0, 1), y.flatten()))
            if aw:
                for k in use_aux:
                    name, _, half = k.partition("_")
                    o, t = out[name], a[name]
                    if half:
                        h = t.shape[1] // 2
                        o, t = (o[:, h:], t[:, h:]) if half == "hi" else (o[:, :h], t[:, :h])
                    loss = loss + aw * F.cross_entropy(o.flatten(0, 1), t.flatten())
            if torch.is_tensor(loss):
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            if eval_every and (step + 1) % eval_every == 0 and step + 1 < steps:
                e = evaluate(net, task, size, n=500, seed=20_000 + step, holdout=holdout)
                curve.append([step + 1, round(e["exact"], 3)] + ([round(e["pre"]["exact"], 3)] if "pre" in e else []))
        r = {"task": task_name, "size": size, "model": model, "aux": list(aux), "aux_weight": aux_weight,
             "aux_sched": aux_sched, "final_weight": final_weight, "final_sched": final_sched, "steps": steps,
             "pool": pool, "holdout": holdout, "seed": seed, **_r(evaluate(net, task, size, holdout=holdout))}
        if x_sched:
            r["x_sched"] = x_sched
        if holdout:
            ho = evaluate(net, task, size, seed=10_001, holdout=holdout, held=True)
            r["ho"] = _r({k: v for k, v in ho.items() if k in ("exact", "per_digit", "pre", "by_q")})
        if fixed is not None:
            r["train_exact"] = round(evaluate_pool(net, fixed, device), 4)
        if task_name == "divstep":
            r["chain"] = {"exact": round(chain(net, size, holdout=holdout), 4)}
            if holdout:
                r["chain"]["ho"] = round(chain(net, size, seed=10_001, holdout=holdout, held=True), 4)
        r["ood_size"] = r["ood_exact"] = None
        if ood and task.kind == "bits" and size + ood <= 24 and task.widths(size + ood) == task.widths(size):
            r["ood_size"], r["ood_exact"] = size + ood, round(evaluate(net, task, size + ood)["exact"], 4)
        r.update(curve=curve, seconds=round(time.time() - t0, 1), tag=tag, model_kw=model_kw)
        results.append(r)
        if log:
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
        if verbose:
            print(line(r), flush=True)
    return results


@torch.no_grad()
def chain(net, size, n=2000, seed=10_000, holdout=0, held=False):
    """x^2 mod N on sqmod's evaluation rows by chaining a trained divstep net over the true digits of x^2."""
    b = sample(TASKS["sqmod"], n, size, np.random.default_rng(seed), holdout, held)
    dev = next(net.parameters()).device
    x, y, a = _tensors(b, dev)
    R = torch.zeros_like(y)
    for i in reversed(range(a["pre"].shape[1])):
        R = net({"N": x["N"], "R": R, "d": a["pre"][:, i:i + 1]})["target"].argmax(-1)
    return (R == y).all(1).float().mean().item()


@torch.no_grad()
def evaluate_pool(net, fixed, device):
    x, y, _ = _tensors(fixed, device)
    return (net(x)["target"].argmax(-1) == y).all(1).float().mean().item()


def _name(r):
    s = r["model"] + (f"[{r['model_kw']['est']}]" if "est" in r.get("model_kw", {}) else "")
    if r["aux"]:
        s += "+" + ",".join(r["aux"]) + (f"x{r['aux_weight']:g}" if r["aux_weight"] != 1 else "") + \
             (f"@{r['aux_sched']}" if r["aux_sched"] != "const" else "")
    if r["final_weight"] != 1 or r["final_sched"] != "const":
        s += f" final{r['final_weight']:g}" + (f"@{r['final_sched']}" if r["final_sched"] != "const" else "")
    s += f" x@{r['x_sched']}" if r.get("x_sched") else ""
    return s + (f" pool{r['pool']}" if r["pool"] else "")


def line(r):
    f = lambda v: "[" + " ".join(f"{u:.2f}" for u in v) + "]"
    s = f"{r['task']}@{r['size']} {_name(r)} seed{r['seed']}: exact {r['exact']:.3f}"
    s += f"  ho {r['ho']['exact']:.3f}" if "ho" in r else ""
    s += f"  train {r['train_exact']:.3f}" if "train_exact" in r else ""
    s += f"  ood@{r['ood_size']} {r['ood_exact']:.3f}" if r["ood_exact"] is not None else ""
    s += f"  digits(lo->hi) {f(r['per_digit'])}"
    s += f"  chain {r['chain']['exact']:.3f}" + (f" ho {r['chain']['ho']:.3f}" if "ho" in r["chain"] else "") \
        if "chain" in r else ""
    for where, d in (("", r), ("ho ", r.get("ho", {}))):
        if "by_q" in d:
            s += f"  {where}by_q " + " ".join(f"{k}:" + ("-" if v[0] is None else f"{v[0]:.2f}")
                                              for k, v in d["by_q"].items())
    for k in ("pre", "q"):
        if k in r:
            s += f"  {k} {r[k]['exact']:.3f} {f(r[k]['per_digit'])}"
            if "congruent" in r[k]:
                s += f" cong {r[k]['congruent']:.3f} <N {r[k]['below_N']:.3f}"
    if r["curve"]:
        s += "  curve " + " ".join(f"{c[1]:.2f}" + (f"/{c[2]:.2f}" if len(c) > 2 else "") for c in r["curve"])
    return s + f"  {r['seconds']}s"


def load(tag=None, path=RESULTS):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    return [r for r in rows if tag is None or r["tag"] == tag]


def table(rows):
    """Seed-averaged: exact on training moduli, on held-out moduli (ho), on the pool (train), per digit, aux heads."""
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["size"], _name(r), r["steps"]), []).append(r)
    f = lambda v: "[" + " ".join(f"{u:.2f}" for u in v) + "]"
    m = lambda rs, get: np.mean([get(r) for r in rs], 0)
    for (task, size, name, steps), rs in sorted(groups.items()):
        s = f"{task}@{size:<2} {name:40} {steps:>5} steps n={len(rs)}  exact {m(rs, lambda r: r['exact']):.3f}"
        s += f"  ho {m(rs, lambda r: r['ho']['exact']):.3f}" if all("ho" in r for r in rs) else ""
        s += f"  train {m(rs, lambda r: r['train_exact']):.3f}" if all("train_exact" in r for r in rs) else ""
        s += f"  digits {f(m(rs, lambda r: r['per_digit']))}"
        for k in ("pre", "q"):
            if all(k in r for r in rs):
                s += f"  {k} {m(rs, lambda r: r[k]['exact']):.3f} {f(m(rs, lambda r: r[k]['per_digit']))}"
                if "congruent" in rs[0][k]:
                    s += f" cong {m(rs, lambda r: r[k]['congruent']):.3f} <N {m(rs, lambda r: r[k]['below_N']):.3f}"
        print(s)


def main(argv=None, log=RESULTS):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["table"]:                       # python -m microlab.ext_preremainder table [tag]
        return table(load(argv[1] if len(argv) > 1 else None))
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--model", default="xf", choices=sorted(MODELS))
    p.add_argument("--aux", default="", help="comma list of auxiliary targets, e.g. pre,q")
    p.add_argument("--aux_weight", type=float, default=1.0)
    p.add_argument("--aux_sched", default="const", help="const | off | down:a | up:a | lin:a")
    p.add_argument("--final_weight", type=float, default=1.0)
    p.add_argument("--final_sched", default="const")
    p.add_argument("--x_sched", default="", help="sqmodx only: a:f0, x range grows from f0*N to N by progress a")
    p.add_argument("--est", default=None, help="ht only: st | local | best | rf")
    p.add_argument("--rf_samples", type=int, default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--holdout", type=int, default=0, help="hold out 1/h of the moduli and score them separately")
    p.add_argument("--ood", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=0)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    kw = {k: v for k, v in (("depth", a.depth), ("d", a.d), ("est", a.est), ("rf_samples", a.rf_samples))
          if v is not None}
    return train(a.task, a.size, model=a.model, aux=tuple(x for x in a.aux.split(",") if x),
                 aux_weight=a.aux_weight, aux_sched=a.aux_sched, final_weight=a.final_weight,
                 final_sched=a.final_sched, steps=a.steps, batch=a.batch, lr=a.lr, seeds=a.seeds, seed0=a.seed0,
                 pool=a.pool, holdout=a.holdout, ood=a.ood, eval_every=a.eval_every, x_sched=a.x_sched, tag=a.tag,
                 device=a.device, log=log, **kw)


if __name__ == "__main__":
    main()
