"""Relaxation 2: "show your work". A generic GPT is supervised on the full worked trace of ONE squaring
(long multiplication, then long division), generates the trace itself at test time, and is looped T times with the
answer hard-snapped (parsed digits) between steps.

Trace of s^2 mod N, every number LSB-first, fixed width W:
  [N] n  [S] s
  for j < W:      [P] s*s_j (W+1)        [A] running sum s*(s mod 10^(j+1)) (2W)
  for k = 1..9:   [M] k*N (W+1)                       (the divisor's multiples table)
  R = top W digits of s^2 (< N); for i = W-1..0:
                  [E] 10R + digit_i(s^2) (W+1)  [Q] q_i = E // N  [R] E - q_i N (W)
  [END]           the last R line is the answer
Model: pre-LN decoder, rotary positions over the sequence + a learned "place in its number" embedding (Abacus-style),
greedy decoding with a KV cache. Curricula are loss masks over trace lines (teacher forcing throughout):
  none     every line supervised from the start
  forward  normal curriculum: supervise the first lines first, add later lines over the first 60% of training
  reverse  start with most lines given (in context, unsupervised), remove the help over the first 60% of training
Data: --data h1like = per-step traces along an H1-shaped finite set at b bits (relaxation 2 alone); --data fresh =
unlimited fresh (N, s) (relaxations 2 + 4).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import relax_common as rc

TOK = {"N": 10, "S": 11, "P": 12, "A": 13, "M": 14, "E": 15, "Q": 16, "R": 17, "END": 18, "C": 19, "G": 20, "D": 21}
V = 22


class Layout:
    """Static token layout for width W: marker/digit positions, place indices, line ids.
    fmt v1: as in the module docstring. fmt v2: each multiples line also states its multiplier k ([M] k kN), and each
    division step copies the chosen multiple next to the subtraction ([E] e [Q] q [C] qN [R] e-qN), so the lookup is
    by content and the subtraction is between adjacent lines. fmt v3: v2 plus the nine comparisons written out before
    the quotient digit ([G] g_1..g_9, g_k = 1 if kN <= e; q is the number of ones). fmt v4: no comparison at all: each
    division step writes the nine trial subtractions ([D] k (e - kN mod 10^(W+1)) borrow), each a local LSB-first
    subtraction whose last digit is the borrow-out (1 if e < kN); then [Q] q (the borrow-free lines) and [R] r, the
    low digits of line q."""

    def __init__(self, W: int, fmt: str = "v1"):
        self.W, self.fmt = W, fmt
        segs = [("N", W), ("S", W)]
        for _ in range(W):
            segs += [("P", W + 1), ("A", 2 * W)]
        for _ in range(9):
            segs += [("M", W + 1)]
        for _ in range(W):
            if fmt == "v4":
                segs += [("E", W + 1)] + [("D", W + 2)] * 9 + [("Q", 1), ("R", W)]
                continue
            segs += [("E", W + 1)] + ([("G", 9)] if fmt == "v3" else []) + [("Q", 1)] + \
                ([("C", W + 1)] if fmt in ("v2", "v3") else []) + [("R", W)]
        segs += [("END", 0)]
        self.segs = segs
        place, line, start = [], [], []
        pos = 0
        for li, (name, n) in enumerate(segs):
            start.append(pos)
            place.append(2 * W + 1)  # marker
            line.append(li)
            pos += 1
            if (name == "M" and fmt in ("v2", "v3", "v4")) or name == "D":
                place.append(2 * W)  # the multiplier digit k
                line.append(li)
                pos += 1
            place += list(range(2 * W + 2, 2 * W + 2 + n)) if name == "G" else list(range(n))
            line += [li] * n
            pos += n
        self.L = pos
        self.place = np.array(place)
        self.n_place = int(self.place.max()) + 1
        self.line = np.array(line)
        self.start = start
        self.prompt_len = 2 * (W + 1)  # [N] n [S] s
        self.n_lines = len(segs)  # line 0,1 = prompt; 2.. = trace (incl. END)
        self.ans_start = start[len(segs) - 2] + 1  # digits of the last R line

    def build(self, N, s):
        W = self.W
        B = len(N)
        out = np.empty((B, self.L), dtype=np.int64)
        col = 0

        def put(name, vals, n, k=None):
            nonlocal col
            out[:, col] = TOK[name]
            col += 1
            if k is not None:
                out[:, col] = k
                col += 1
            if n:
                out[:, col:col + n] = rc.digits(vals, n)
                col += n

        put("N", N, W)
        put("S", s, W)
        sd = rc.digits(s, W)
        acc = np.zeros(B, dtype=np.int64)
        for j in range(W):
            p = s * sd[:, j]
            acc = acc + p * 10 ** j
            put("P", p, W + 1)
            put("A", acc, 2 * W)
        for k in range(1, 10):
            put("M", k * N, W + 1, k if self.fmt in ("v2", "v3", "v4") else None)
        P = acc
        R = P // 10 ** W
        pd = rc.digits(P, 2 * W)
        for i in range(W - 1, -1, -1):
            E = 10 * R + pd[:, i]
            q = E // N
            R = E - q * N
            put("E", E, W + 1)
            if self.fmt == "v4":
                for k in range(1, 10):
                    out[:, col] = TOK["D"]
                    out[:, col + 1] = k
                    out[:, col + 2:col + 3 + W] = rc.digits((E - k * N) % 10 ** (W + 1), W + 1)
                    out[:, col + 3 + W] = (E < k * N).astype(np.int64)
                    col += W + 4
                put("Q", q, 1)
                put("R", R, W)
                continue
            if self.fmt == "v3":
                out[:, col] = TOK["G"]
                out[:, col + 1:col + 10] = (np.arange(1, 10)[None, :] * N[:, None] <= E[:, None]).astype(np.int64)
                col += 10
            put("Q", q, 1)
            if self.fmt in ("v2", "v3"):
                put("C", q * N, W + 1)
            put("R", R, W)
        put("END", None, 0)
        assert col == self.L
        return out, R


def rope_cache(L, D, device, base=10000.0):
    inv = base ** (-torch.arange(0, D, 2, device=device, dtype=torch.float32) / D)
    ang = torch.arange(L, device=device, dtype=torch.float32)[:, None] * inv[None]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cos, sin, cache=None, t0=0):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            K, Vc = cache
            K[:, :, t0:t0 + L] = k
            Vc[:, :, t0:t0 + L] = v
            a = F.scaled_dot_product_attention(q, K[:, :, :t0 + L], Vc[:, :, :t0 + L],
                                               is_causal=(L > 1 and t0 == 0))
        x = x + self.o(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class TraceGPT(nn.Module):
    def __init__(self, lay: Layout, d=256, h=8, layers=6):
        super().__init__()
        self.lay, self.d, self.h, self.nl = lay, d, h, layers
        self.tok = nn.Embedding(V, d)
        self.plc = nn.Embedding(lay.n_place, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        cos, sin = rope_cache(lay.L + 1, d // h, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("place", torch.as_tensor(lay.place), persistent=False)

    def forward(self, x, t0=0, caches=None):
        L = x.shape[1]
        e = self.tok(x) + self.plc(self.place[t0:t0 + L])[None]
        cos, sin = self.cos[t0:t0 + L], self.sin[t0:t0 + L]
        for i, blk in enumerate(self.blocks):
            e = blk(e, cos, sin, None if caches is None else caches[i], t0)
        return self.head(self.ln(e))

    @torch.no_grad()
    def generate(self, prompt):
        """Greedy decoding of the whole trace from the prompt; returns all tokens [B, L]."""
        B, P = prompt.shape
        dev = prompt.device
        dh = self.d // self.h
        caches = [(torch.zeros(B, self.h, self.lay.L, dh, device=dev), torch.zeros(B, self.h, self.lay.L, dh, device=dev))
                  for _ in range(self.nl)]
        out = torch.empty(B, self.lay.L, dtype=torch.long, device=dev)
        out[:, :P] = prompt
        logits = self.forward(prompt, 0, caches)[:, -1]
        for t in range(P, self.lay.L):
            nxt = logits.argmax(-1)
            out[:, t] = nxt
            if t + 1 < self.lay.L:
                logits = self.forward(nxt[:, None], t, caches)[:, -1]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--data", default="h1like", choices=["h1like", "fresh", "h1"])
    ap.add_argument("--curriculum", default="none", choices=["none", "forward", "reverse"])
    ap.add_argument("--fmt", default="v1", choices=["v1", "v2", "v3", "v4"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_n", type=int, default=1024)
    ap.add_argument("--final_n", type=int, default=20000)
    ap.add_argument("--resume", default="", help="state_dict .pt to continue training from")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ladder_only_if_exact", action="store_true",
                    help="skip the T-ladder when single-step accuracy is below 100% (it cannot certify)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    W = 8 if a.data == "h1" else rc.width(a.bits)
    lay = Layout(W, a.fmt)
    model = TraceGPT(lay, a.d, 8, a.layers).to(dev)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location=dev))
        print(f"resumed from {a.resume}", flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = lambda s: min(1.0, s / max(1, a.warmup)) * (0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)) * 0.97 + 0.03)
    line_t = torch.as_tensor(lay.line, device=dev)
    n_tr = lay.n_lines - 2  # trace lines incl END
    print(f"W={W} L={lay.L} trace lines={n_tr} params={nparam} data={a.data} curriculum={a.curriculum} fmt={a.fmt}",
          flush=True)

    pool = unseen = None
    if a.data == "h1like":
        d = rc.h1like(a.bits, a.seed)
        PN, PS, PY = rc.step_pairs(d)
        pool, unseen = d["pool"], d["unseen"]
        print(f"h1like: {len(PN)} step traces, {len(pool)} pool moduli, {len(unseen)} unseen", flush=True)
    elif a.data == "h1":
        import h1
        tr = h1.load("train")
        rows = [(int(n), int(x), int(t)) for n, x, t in zip(tr["N"], tr["X"], tr["T"])]
        PN, PS = [], []
        for n, x, t in rows:
            s = x % n
            for _ in range(t):
                PN.append(n), PS.append(s)
                s = s * s % n
        PN, PS = np.array(PN, dtype=np.int64), np.array(PS, dtype=np.int64)
        pool, unseen = np.unique(tr["N"]), np.array([], dtype=np.int64)
        print(f"h1 real: {len(PN)} step traces from {len(rows)} rows, {len(pool)} moduli (13/16/20-bit), W=8", flush=True)

    def batch(B, r=rng):
        if a.data == "fresh":
            N, s, _ = rc.fresh_step(r, a.bits, B)
        else:
            i = r.integers(0, len(PN), B)
            N, s = PN[i], PS[i]
        return N, s

    def loss_mask(frac, B):
        """[B, L-1] mask over predicted positions (target token index 1..L-1)."""
        tl = line_t[1:] - 2  # trace line index of each target token (-2,-1 = prompt)
        base = tl >= 0
        if a.curriculum == "none" or frac >= 0.6:
            return base[None].expand(B, -1)
        g = frac / 0.6
        jit = rng.integers(0, 3, B)
        if a.curriculum == "forward":
            # supervise trace lines < H: H grows from 1 line to all of them
            h = torch.as_tensor((np.round(1 + g * (n_tr - 1)) + jit).astype(np.int64), device=dev)
            return base[None] & (tl[None] < h[:, None])
        # reverse: the first G trace lines are given (in context, unsupervised); G shrinks from all-but-1 to 0
        given = torch.as_tensor(np.maximum(0, np.round((1 - g) * (n_tr - 1)) - jit).astype(np.int64), device=dev)
        return base[None] & (tl[None] >= given[:, None])

    @torch.no_grad()
    def gen_answer(N, s, bs=2048):
        model.eval()
        outs = []
        for i in range(0, len(N), bs):
            toks, _ = lay.build(N[i:i + bs], s[i:i + bs])
            g = model.generate(torch.as_tensor(toks[:, :lay.prompt_len], device=dev)).cpu().numpy()
            outs.append(g)
        model.train()
        g = np.concatenate(outs)
        return g

    def answer_of(g):
        ans = g[:, lay.ans_start:lay.ans_start + W]
        bad = (ans > 9).any(1)
        v = rc.undigits(np.minimum(ans, 9))
        v[bad] = -1
        return v

    def eval_exact(N, s):
        toks, y = lay.build(N, s)
        g = gen_answer(N, s)
        ok = answer_of(g) == y
        tr_ok = (g == toks).all(1)
        # first wrong line type among failures
        first = {}
        for row in np.where(~tr_ok)[0][:200]:
            pos = int(np.argmax(g[row] != toks[row]))
            nm = lay.segs[lay.line[pos]][0]
            first[nm] = first.get(nm, 0) + 1
        return float(ok.mean()), float(tr_ok.mean()), first

    erng = np.random.default_rng(4242)
    ev = {}
    if a.data == "fresh":
        ev["fresh"] = rc.fresh_step(erng, a.bits, a.eval_n)[:2]
    elif a.data == "h1":
        Ne = pool[erng.integers(0, len(pool), a.eval_n)]
        ev["pool"] = (Ne, (erng.random(a.eval_n) * Ne).astype(np.int64))
    else:
        ev["pool"] = rc.fresh_step(erng, a.bits, a.eval_n, pool)[:2]
        if len(unseen):
            ev["unseen"] = rc.fresh_step(erng, a.bits, a.eval_n, unseen)[:2]

    t0 = time.time()
    log = []
    for step in range(1, a.steps + 1):
        frac = step / a.steps
        for gq in opt.param_groups:
            gq["lr"] = a.lr * sched(step)
        N, s = batch(a.batch)
        toks, _ = lay.build(N, s)
        x = torch.as_tensor(toks, device=dev)
        lo = model(x[:, :-1])
        m = loss_mask(frac, a.batch)
        ce = F.cross_entropy(lo.reshape(-1, V), x[:, 1:].reshape(-1), reduction="none").view(a.batch, -1)
        loss = (ce * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            with torch.no_grad():
                tf_acc = float(((lo.argmax(-1) == x[:, 1:]) | ~m).all(1).float().mean())
            res = {k: eval_exact(*v) for k, v in ev.items()}
            el = time.time() - t0
            log.append({"step": step, "loss": float(loss.detach()), "tf_seq_acc": tf_acc, "time": el,
                        **{k: v[0] for k, v in res.items()}})
            print(f"step {step} loss {float(loss.detach()):.4f} tf_seq {tf_acc:.3f} " +
                  " ".join(f"{k}: ans={v[0]:.4f} trace={v[1]:.4f} firsterr={v[2]}" for k, v in res.items()) +
                  f" t={el:.0f}s", flush=True)
    train_time = time.time() - t0
    if a.out:
        torch.save(model.state_dict(), a.out.replace(".json", ".pt"))

    out = {"args": vars(a), "params": nparam, "L": lay.L, "train_time": train_time, "log": log}
    frng = np.random.default_rng(999)

    def step_fn(N, s):
        return answer_of(gen_answer(N, s))

    if a.data == "h1":
        import h1
        Nf = pool[frng.integers(0, len(pool), a.final_n)]
        sf = (frng.random(a.final_n) * Nf).astype(np.int64)
        out["single_step_pool"] = eval_exact(Nf, sf)
        print("final single-step (H1 pool moduli):", out["single_step_pool"], flush=True)

        def predict_h1(N, X, T):
            s = X % N
            for tv in np.unique(T):
                k = T == tv
                c = s[k].copy()
                for _ in range(int(tv)):
                    c = step_fn(N[k], c)
                s[k] = c
            return h1.digits(s)

        for prof in ("depth_t", "depth_ood_n_t"):
            out[prof] = h1.certify(predict_h1, prof)
    elif a.data == "fresh":
        N, s, _ = rc.fresh_step(frng, a.bits, a.final_n)
        out["single_step_fresh"] = eval_exact(N, s)
        print("final single-step (fresh):", out["single_step_fresh"], flush=True)
        out["ladder_fresh"] = rc.ladder(step_fn, a.bits)
    else:
        N, s, _ = rc.fresh_step(frng, a.bits, a.final_n, pool)
        out["single_step_pool"] = eval_exact(N, s)
        print("final single-step (pool moduli):", out["single_step_pool"], flush=True)
        if not a.ladder_only_if_exact or out["single_step_pool"][0] == 1.0:
            out["ladder_pool"] = rc.ladder(step_fn, a.bits, moduli=pool)
        if len(unseen):
            N, s, _ = rc.fresh_step(frng, a.bits, a.final_n, unseen)
            out["single_step_unseen"] = eval_exact(N, s)
            print("final single-step (unseen moduli):", out["single_step_unseen"], flush=True)
            if not a.ladder_only_if_exact or out["single_step_unseen"][0] == 1.0:
                out["ladder_unseen"] = rc.ladder(step_fn, a.bits, moduli=unseen)
    out["total_time"] = time.time() - t0
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print("done", json.dumps({k: v for k, v in out.items() if k not in ("log", "args")})[:3000], flush=True)


if __name__ == "__main__":
    main()
