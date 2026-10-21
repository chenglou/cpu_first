"""Local checks of submission_square.py (small: well under 1 GB, under a minute). Needs the H1 jsonl files (H1_DATA, as
in cpu_first/h1.py) and torch; `stub/` stands in for the evaluator's `benchmark` API.
  python local_square.py designed             # exact tables: squares of random 8-digit numbers, then every depth rung
  python local_square.py train STEPS B [SEED] # a short training run at batch B with the file's defaults
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "stub"), os.path.dirname(HERE)]
import torch
import submission_square as sq
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch

DATA = os.environ.get("H1_DATA", (
    "/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-"
    "4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-"
    "efb0abec6f53/scratchpad/data/h1"))


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
    sq.MODE = "designed"
    print("ADDER", sq.ADDER, flush=True)
    torch.manual_seed(0)
    m = sq.build_model(spec()).eval()
    tabs = m.tables()
    v = torch.cat([torch.randint(0, 10 ** 8, (20000,)), torch.tensor([0, 1, 9, 99999999, 16777215, 10 ** 7])])
    s2 = sq.from_digits(sq.square_hard(sq.to_digits(v, sq.W), tabs))
    print("square_hard == v*v on", v.numel(), "numbers below 10^8:", bool((s2 == v * v).all()), flush=True)
    for split in [f"depth_{p}_{t}" for p in ("t", "ood_n_t") for t in (1, 2, 4, 8, 16, 32, 64)]:
        rows = load(split)
        ids, am, lab = batch(rows)
        t0 = time.time()
        with torch.no_grad():
            Nd, Xd, Td_, end, width = sq.parse(ids, am)
            T = sq.from_digits(Td_[:, :2])
            M = torch.nn.functional.pad(Nd, (0, sq.P - sq.W))
            s = sq.from_digits(m.run(Xd, T, M, tabs))
            logits, _ = m(ids, am)
        y = torch.tensor([r["result"] for r in rows])
        tok = logits.argmax(-1)
        ok = [all(int(tok[i, len(r["input_ids"]) - len(r["labels"]) + k]) == r["labels"][k]
                  for k in range(len(r["labels"]))) for i, r in enumerate(rows)]
        print(split, "run==result", float((s == y).float().mean()), "logits exact", sum(ok) / len(ok),
              f"{time.time() - t0:.2f}s", flush=True)


def train(steps, B, seed=1):
    torch.manual_seed(seed)
    m = sq.build_model(spec()).train()
    bundle = sq.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    rows = load("train")
    g = torch.Generator().manual_seed(seed)
    tdt, tbt = sq.exact_tables("cpu")
    plt, pht = sq.exact_products("cpu")
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab = batch([rows[i] for i in idx])
        logits, aux = m(ids, am)
        loss = sq.token_training_loss(TokenLossBatch(logits, lab, lab != -100, None, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        bundle.scheduler.step()
        if step % 25 == 0 or step == 1:
            Td, Tb, Plo, Phi = m.tables()
            right = int(((Td == tdt) & (Tb == tbt)).sum())
            pright = int(((Plo == plt) & (Phi == pht)).sum())
            print(f"step {step} loss {float(loss):.4f} phase {m.ctl.phase} win {sum(m.ctl.win)/max(1,len(m.ctl.win)):.3f} "
                  f"sub_rows_right {right}/200 prod_cells_right {pright}/100 {time.time() - t0:.1f}s", flush=True)
        if m.ctl.phase == "frozen":
            break
    Td, Tb, Plo, Phi = m.tables()
    print("wrong subtractor rows (r, m, c):", [(i // 20, (i // 2) % 10, i % 2, int(Td[i]), int(Tb[i])) for i in range(200)
                                               if Td[i] != tdt[i] or Tb[i] != tbt[i]],
          "wrong product cells (a, b):", [(i // 10, i % 10) for i in range(100) if Plo[i] != plt[i] or Phi[i] != pht[i]])


if __name__ == "__main__":
    if os.environ.get("ADDER"):
        sq.ADDER = os.environ["ADDER"]
    if sys.argv[1] == "designed":
        designed()
    else:
        sq.LOG_EVERY = 25
        train(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
