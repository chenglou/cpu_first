"""Local checks of submission_control.py (small: well under 1 GB, under a minute). Needs the H1 jsonl files (H1_DATA, as
in cpu_first/h1.py) and torch; chain_local/stub stands in for the evaluator's `benchmark` API.
  python local_control.py designed [SPLIT ...]        # exact table and gate: the machine must reproduce every answer
  python local_control.py train STEPS B [SEED] [K=V]  # a short training run at batch B (K=V: set a constant)
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "chain_local", "stub"), os.path.dirname(HERE)]
import torch
import torch.nn.functional as F
import submission_control as sc
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch

DATA = os.environ.get("H1_DATA", (
    "/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-"
    "4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-"
    "efb0abec6f53/scratchpad/data/h1"))
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
    L = max(len(r["input_ids"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    lab = torch.full((len(rows), 8), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r["input_ids"])] = torch.tensor(r["input_ids"])
        lab[i, :len(r["labels"])] = torch.tensor(r["labels"])
    return ids, ids != 0, lab


def spec():
    return ModelSpec(vocab_size=17, max_seq_len=21, maximum_model_state_elements=500_000_000)


def setconst(kv):
    k, v = kv.split("=", 1)
    old = getattr(sc, k)
    setattr(sc, k, type(old)(v) if not isinstance(old, bool) else v in ("1", "True", "true"))


def wrong_rows(m):
    Td, Tb, Gc = m.tables()
    tdt, tbt = sc.exact_tables("cpu")
    return [f"({i // 20},{(i // 2) % 10},{i % 2})->({int(Td[i])},{int(Tb[i])})" for i in range(200)
            if Td[i] != tdt[i] or Tb[i] != tbt[i]]


def designed(splits):
    sc.MODE = "designed"
    torch.manual_seed(0)
    m = sc.build_model(spec()).eval()
    for split in splits or RUNGS:
        rows = load(split)
        ids, am, lab = batch(rows)
        t0 = time.time()
        with torch.no_grad():
            Nd, Xd, Cd, end, width, twidth = sc.parse(ids, am)
            x = sc.from_digits(Xd)
            C = Cd[:, :int(twidth.max())]
            M = F.pad(Nd, (0, sc.P - sc.W))
            Td, Tb, Gc = m.tables()
            s, prev, n, held = sc.run_hard(x, C, M, Td, Tb, Gc)
        y = torch.tensor([r["result"] for r in rows])
        T = torch.tensor([r["time_steps"] for r in rows])
        print(split, "run==result", float((s == y).float().mean()), "rounds==T", bool((n == T).all()),
              "held", bool(held.all()), f"K={C.shape[1]}", f"{time.time() - t0:.2f}s", flush=True)


def train(steps, B, seed=1):
    torch.manual_seed(seed)
    m = sc.build_model(spec()).train()
    bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    rows = load("train")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab = batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, None, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        bundle.scheduler.step()
        if step % 25 == 0 or step == 1 or m.ctl.phase == "frozen":
            wr = wrong_rows(m)
            gate = m.tables()[2].tolist()
            print(f"step {step} loss {float(loss):.4f} phase {m.ctl.phase} win {sum(m.ctl.win)/max(1,len(m.ctl.win)):.3f} "
                  f"rows_right {200 - len(wr)}/200 gate {gate} p_commit "
                  f"{[round(p, 3) for p in m.gate_soft().tolist()]} wrong {wr[:4]} {time.time() - t0:.1f}s", flush=True)
        if m.ctl.phase == "frozen":
            break


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if "=" in a:
            setconst(a)
    args = [a for a in sys.argv[1:] if "=" not in a]
    if args[0] == "designed":
        designed(args[1:])
    else:
        sc.LOG_EVERY = 25
        train(int(args[1]), int(args[2]), int(args[3]) if len(args) > 3 else 1)
