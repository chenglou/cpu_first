"""Local checks of submission_io.py (small: well under 1 GB, under a minute). Needs torch and the H1 jsonl files (H1_DATA,
as in cpu_first/h1.py); `stub/` stands in for the evaluator's `benchmark` API. Batches are built like the evaluator's
(inputs padded with 0, labels left-aligned with -100, target positions right-aligned) and scored like it (argmax of
the logits at the target positions, exact per row).
  python local_io.py designed              # exact table + I/O set at the true reading: every depth rung's answers
  python local_io.py train STEPS B [SEED]  # a short training run at batch B with the file's defaults
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "stub"), os.path.dirname(HERE)]
import torch
import submission_io as sio
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch

DATA = os.environ.get("H1_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "one-layer-deeper", "data", "generated",
    "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


def load(split):
    """A split as compact tensors: inputs padded to 21 with 0, labels padded to 8 with -100, (N, x, T, y). With
    IO_CACHE set, the tensors are cached there (outside the repository)."""
    cache = os.environ.get("IO_CACHE")
    if cache and os.path.exists(os.path.join(cache, split + ".pt")):
        return torch.load(os.path.join(cache, split + ".pt"))
    ids, lab, num = [], [], []
    for line in open(os.path.join(DATA, split + ".jsonl")):
        r = json.loads(line)
        ids.append(r["input_ids"] + [0] * (21 - len(r["input_ids"])))
        lab.append(r["labels"] + [-100] * (8 - len(r["labels"])))
        num.append([r["modulus"], r["x"], r["time_steps"], r["result"]])
    out = torch.tensor(ids), torch.tensor(lab), torch.tensor(num)
    if cache:
        os.makedirs(cache, exist_ok=True)
        torch.save(out, os.path.join(cache, split + ".pt"))
    return out


def batch(ids, lab):
    L = int((ids != 0).sum(1).max())
    ids = ids[:, :L]
    mask = ids != 0
    cnt = (lab != -100).sum(1)
    Lt = int(cnt.max())
    lab = lab[:, :Lt]
    k = torch.arange(Lt)[None]
    tp = torch.where(k < cnt[:, None], mask.sum(1)[:, None] - cnt[:, None] + k, -1)
    return ids, mask, lab, tp


def exact_rows(logits, lab, tp):
    tok = logits[torch.arange(logits.shape[0])[:, None], tp.clamp(min=0)].argmax(-1)
    return ((tok == lab) | (lab == -100)).all(1)


def spec():
    return ModelSpec(vocab_size=17, max_seq_len=21, maximum_model_state_elements=500_000_000)


def designed():
    sio.MODE, sio.IO_MODE = "designed", "designed"
    torch.manual_seed(0)
    m = sio.build_model(spec()).eval()
    for split in [f"depth_{p}_{t}" for p in ("t", "ood_n_t") for t in (1, 2, 4, 8, 16, 32, 64)]:
        ids, lab, num = load(split)
        ids, mask, lab, tp = batch(ids, lab)
        t0 = time.time()
        with torch.no_grad():
            logits, _ = m(ids, mask)
        print(split, "exact", float(exact_rows(logits, lab, tp).float().mean()), "8-digit N rows",
              int((num[:, 0] >= 10 ** 7).sum()), f"{time.time() - t0:.2f}s", flush=True)


def train(steps, B, seed=1):
    torch.manual_seed(seed)
    m = sio.build_model(spec()).train()
    bundle = sio.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    ids_all, lab_all, num_all = load("train")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(ids_all), (B,), generator=g)
        ids, mask, lab, tp = batch(ids_all[idx], lab_all[idx])
        logits, aux = m(ids, mask)
        tok_logits = logits[torch.arange(B)[:, None], tp.clamp(min=0)]
        loss = sio.token_training_loss(TokenLossBatch(tok_logits, lab, lab != -100, tp, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        bundle.scheduler.step()
        if step % 25 == 0 or step == 1:
            Td, Tb = m.tables()
            tdt, tbt = sio.exact_tables("cpu")
            right = int(((Td == tdt) & (Tb == tbt)).sum())
            with torch.no_grad():
                num = num_all[idx]
                e, end = m.ext(ids, mask)
                k = m.selected(e, end)
                code = m.ranks(k)
                Nd, _ = m.read(0, e, end, code); Xd, _ = m.read(1, e, end, code); Td_, _ = m.read(2, e, end, code)
                tr = {n: round(float((sio.from_digits(d) == num[:, i]).float().mean()), 3)
                      for n, d, i in (("N", Nd, 0), ("x", Xd, 1), ("T", Td_, 2))}
            print(f"step {step} loss {float(loss):.4f} phase {m.ctl.phase} io {m.ctl.io} chosen {m.ctl.chosen:.3f} "
                  f"io_exact {m.ctl.io_exact} table {right}/200 read {tr} {m.ctl.t_log} {time.time() - t0:.1f}s", flush=True)
        if m.ctl.phase == "frozen":
            print("frozen at", step, m.ctl.history, flush=True)
            break
    m.eval()
    for split in [f"depth_{p}_{t}" for p in ("t", "ood_n_t") for t in (1, 2, 4, 8, 16, 32, 64)]:
        ids, lab, num = load(split)
        ids, mask, lab, tp = batch(ids, lab)
        with torch.no_grad():
            logits, _ = m(ids, mask)
        ok = exact_rows(logits, lab, tp)
        w8 = num[:, 0] >= 10 ** 7
        print(split, "exact", int(ok.sum()), "/", len(ok), "8-digit N rows exact", int(ok[w8].sum()), "/", int(w8.sum()), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "designed":
        designed()
    else:
        sio.LOG_EVERY = 25
        train(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
