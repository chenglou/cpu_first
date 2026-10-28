"""Local checks of submission_chain.py (small: well under 1 GB, under a minute). Needs the H1 jsonl files (H1_DATA, as
in cpu_first/history/h1.py) and torch; `stub/` stands in for the evaluator's `benchmark` API.
  python local_chain.py designed             # exact table: the chain must reproduce every answer of the depth rungs
  python local_chain.py train STEPS B [SEED] # a short training run at batch B with the file's defaults
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "stub"), os.path.dirname(HERE)]
import torch
import submission_chain as sc
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch

DATA = os.environ.get("H1_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "one-layer-deeper", "data", "generated",
    "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


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


def designed():
    sc.MODE = "designed"
    torch.manual_seed(0)
    m = sc.build_model(spec()).eval()
    for split in [f"depth_{p}_{t}" for p in ("t", "ood_n_t") for t in (1, 2, 4, 8, 16, 32, 64)]:
        rows = load(split)
        ids, am, lab = batch(rows)
        t0 = time.time()
        with torch.no_grad():
            Nd, Xd, Td_, end, width = sc.parse(ids, am)
            x, T = sc.from_digits(Xd), sc.from_digits(Td_[:, :2])
            M = torch.nn.functional.pad(Nd, (0, sc.P - sc.W))
            Td, Tb = m.tables()
            s = m.run(x, T, M, Td, Tb)
            logits, _ = m(ids, am)
        y = torch.tensor([r["result"] for r in rows])
        xs = torch.tensor([r["x"] for r in rows]); Ns = torch.tensor([r["modulus"] for r in rows])
        print(split, "run==result", float((s == y).float().mean()), "x<N", bool((xs < Ns).all()),
              "maxN", int(Ns.max()), f"{time.time() - t0:.2f}s", flush=True)


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
        if step % 25 == 0 or step == 1:
            Td, Tb = m.tables()
            tdt, tbt = sc.exact_tables("cpu")
            right = int(((Td == tdt) & (Tb == tbt)).sum())
            print(f"step {step} loss {float(loss):.4f} phase {m.ctl.phase} win {sum(m.ctl.win)/max(1,len(m.ctl.win)):.3f} "
                  f"rows_right {right}/200 {time.time() - t0:.1f}s", flush=True)
        if m.ctl.phase == "frozen":
            break


if __name__ == "__main__":
    if sys.argv[1] == "designed":
        designed()
    else:
        sc.LOG_EVERY = 25
        train(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
