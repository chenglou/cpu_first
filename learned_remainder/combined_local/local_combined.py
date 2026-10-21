"""Checks of submission_combined.py outside the evaluator. Needs the H1 jsonl files (H1_DATA) and torch;
chain_local/stub stands in for the evaluator's `benchmark` API. Locally only the small commands (under a minute, well
under 1 GB); the rest runs on Modal (modal_diag_combined.py).

  python local_combined.py designed [N=64] [SPLIT ...]   # every table at the truth: both prediction paths must
                                                        # reproduce every answer and give the same logits
  python local_combined.py grad [N=32]                   # the gradient test at the truth (see gradtest)
  python local_combined.py train STEPS B [SEED] [K=V ...] # training as the evaluator runs it (zero_grad, backward,
                                                        # clip 1, Adam, scheduler), a log line every 25 steps; at the
                                                        # end the depth rungs on both paths and the gradient test

The gradient test: the model in eval mode with autograd on, a cross-entropy loss on the final logits at the answer's
target positions (as the evaluator scores them), one backward pass; every parameter tensor must get a finite gradient
that is not all zero, and the forward must still be exact. The logits are 0 at the written token and -20 elsewhere, so
the plain cross-entropy of a right answer is exactly 0 in fp32 and its gradient on the right token's logit is exactly
0 (the writer's pointer reaches the loss only through that logit): the test divides the logits by TAU = 20 first.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "chain_local", "stub"), os.path.dirname(HERE)]
import torch
import torch.nn.functional as F
import submission_combined as sc
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch

DATA = os.environ.get("H1_DATA", os.path.expanduser(
    "~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
RUNGS = [f"depth_{p}_{t}" for p in ("t", "ood_n_t") for t in (1, 2, 4, 8, 16, 32, 64)]


def load(split, n=None):
    rows = []
    with open(os.path.join(DATA, split + ".jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


def batch(rows):
    """The evaluator's collate for separate input/output rows: ids, mask, labels (-100 padded), target positions."""
    L = max(len(r["input_ids"]) for r in rows)
    T = max(len(r["labels"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    lab = torch.full((len(rows), T), -100, dtype=torch.long)
    tpos = torch.full((len(rows), T), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        n, m = len(r["input_ids"]), len(r["labels"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        lab[i, :m] = torch.tensor(r["labels"])
        tpos[i, :m] = torch.arange(n - m, n)
    return ids, ids != 0, lab, tpos


def spec():
    return ModelSpec(vocab_size=17, max_seq_len=21, maximum_model_state_elements=500_000_000)


def setconst(kv):
    k, v = kv.split("=", 1)
    old = getattr(sc, k)
    setattr(sc, k, v if isinstance(old, str) else type(old)(v) if not isinstance(old, bool) else v in ("1", "True"))


def scored(logits, lab, tpos):
    """The evaluator's scoring: per row, every answer token is the argmax at its target position."""
    tl = logits[torch.arange(logits.shape[0])[:, None], tpos.clamp_min(0)]
    valid = lab != -100
    return ((tl.argmax(-1) == lab) | ~valid).all(1), tl, valid


def eval_paths(m, splits, n):
    """Both prediction paths on the first n rows of each split: exact rows and whether the logits are identical."""
    out = {}
    for split in splits:
        ids, am, lab, tpos = batch(load(split, n))
        res = {}
        for path in ("index", "ste"):
            sc.PATH = path
            m.ctl.used = None
            t0 = time.time()
            with torch.no_grad():
                logits, _ = m(ids, am)
            res[path] = (logits, time.time() - t0)
        sc.PATH = "ste"
        ex = {p: float(scored(res[p][0], lab, tpos)[0].float().mean()) for p in res}
        same = bool(torch.equal(res["index"][0], res["ste"][0]))
        out[split] = (ex, same)
        print(f"{split} n={len(lab)} exact index={ex['index']:.4f} ste={ex['ste']:.4f} identical_logits={same} "
              f"t_index={res['index'][1]:.1f}s t_ste={res['ste'][1]:.1f}s", flush=True)
    return out


TAU = 20.0


def gradtest(m, n=32, splits=("depth_t_1", "depth_t_2"), required=True):
    """Backprop a cross-entropy on the final logits (eval mode, autograd on) and report each parameter's gradient."""
    rows = sum((load(s, n // len(splits)) for s in splits), [])
    ids, am, lab, tpos = batch(rows)
    m.eval()
    m.zero_grad(set_to_none=True)
    sc.PATH = "ste"
    m.ctl.used = None
    with torch.enable_grad():
        logits, _ = m(ids, am)
        ok, tl, valid = scored(logits, lab, tpos)
        loss = F.cross_entropy(tl[valid] / TAU, lab[valid])
        loss.backward()
    report, all_ok = [], True
    for name, p in m.named_parameters():
        g = p.grad
        good = g is not None and bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0
        nz = 0 if g is None else int((g.flatten(0, -2) if g.dim() > 1 else g[:, None]).abs().amax(-1).gt(0).sum())
        rows_total = p.numel() // (p.shape[-1] if p.dim() > 1 else 1)
        report.append(f"{name}: max|grad|={0 if g is None else float(g.abs().max()):.3e} rows_with_grad={nz}/{rows_total}")
        all_ok &= good
    fields = lambda p: "[" + ",".join(f"{v:.1e}" for v in p.grad.abs().amax(-1).tolist()) + "]"
    print(f"GRADTEST splits={','.join(splits)} rows={len(lab)} exact={float(ok.float().mean()):.4f} "
          f"loss={float(loss):.4e} every_parameter_gets_gradient={all_ok}{'' if required else ' (informational)'}",
          flush=True)
    for r in report:
        print("GRADTEST  " + r, flush=True)
    print(f"GRADTEST  per field (N, x, T, writer): anchor {fields(m.anc)} step {fields(m.strd)}; "
          f"gate rows (borrow 0, 1) {fields(m.g)}", flush=True)
    return all_ok


def designed(n, splits):
    sc.MODE = "designed"
    torch.manual_seed(0)
    m = sc.build_model(spec()).eval()
    eval_paths(m, splits or RUNGS, n)


def grad(n):
    sc.MODE = "designed"
    torch.manual_seed(0)
    m = sc.build_model(spec())
    gradtest(m, n)
    gradtest(m, n, ("depth_t_64", "depth_ood_n_t_64"), required=False)


def table_report(m):
    Td, Tb, Plo, Phi, Gc = m.tables()
    tdt, tbt = sc.exact_tables("cpu")
    plo, phi = sc.exact_products("cpu")
    wrong = [f"({i // 20},{(i // 2) % 10},{i % 2})" for i in range(200) if Td[i] != tdt[i] or Tb[i] != tbt[i]]
    pwrong = [f"({i // 10},{i % 10})" for i in range(100) if Plo[i] != plo[i] or Phi[i] != phi[i]]
    return wrong, pwrong, Gc.tolist()


def train(steps, B, seed=1, rungs_n=128):
    torch.manual_seed(seed)
    m = sc.build_model(spec()).train()
    bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    rows = load("train")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab, tpos = batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        bundle.scheduler.step()
        c = m.ctl
        if step % 25 == 0 or step == 1 or c.phase == "frozen":
            wr, pw, gate = table_report(m)
            print(f"step {step} loss {float(loss):.4f} phase {c.phase} io {c.io} ladder {int(c.ladder)} "
                  f"win {sum(c.win) / max(1, len(c.win)):.3f} rows_right {200 - len(wr)}/200 "
                  f"cells_right {100 - len(pw)}/100 gate {gate} io_exact {c.io_exact} chosen {c.chosen:.3f} "
                  f"wrong {wr[:6]} cells_wrong {pw[:6]} {c.t_log} {time.time() - t0:.1f}s", flush=True)
        if c.phase == "frozen":
            break
    print(f"history {m.ctl.history}", flush=True)
    m.eval()
    eval_paths(m, RUNGS, rungs_n)
    gradtest(m)
    gradtest(m, 32, ("depth_t_64", "depth_ood_n_t_64"), required=False)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if "=" in a and not a.startswith("N="):
            setconst(a)
    n = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("N=")), None)
    args = [a for a in sys.argv[1:] if "=" not in a]
    if args[0] == "designed":
        designed(n or 64, args[1:])
    elif args[0] == "grad":
        grad(n or 32)
    else:
        sc.LOG_EVERY = 25
        train(int(args[1]), int(args[2]), int(args[3]) if len(args) > 3 else 1)
