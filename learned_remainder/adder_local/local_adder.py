"""Local checks of submission_adder.py (small: well under 1 GB, under a minute at batch <= 64). Needs the H1 jsonl files
(H1_DATA, as in cpu_first/h1.py) and torch; ../chain_local/stub stands in for the evaluator's `benchmark` API.
  python local_adder.py designed                        # exact table: the learned-adder check passes, a + b is exact
  python local_adder.py train STEPS B [SEED] [K=V ...]  # a short training run at batch B; K=V overrides a constant
Every 25 steps the train mode prints the snapped table against true subtraction and against the consistent wrong tables
of landscape.py (which one it is closest to), and the wrong rows.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "chain_local", "stub"), os.path.dirname(HERE), HERE]
import torch
import submission_adder as sa
from benchmark import ModelSpec, OptimizerSpec, TokenLossBatch
from landscape import candidates, DATA


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
    Td, Tb = sa.exact_tables("cpu")
    g = torch.Generator().manual_seed(0)
    for width in (2, 9, 18, 36):
        A = torch.randint(0, 10, (20000, width), generator=g)
        B = torch.randint(0, 10, (20000, width), generator=g)
        S, carry = sa.add_learned(A, B, Td, Tb)
        a, b, s = (sum(int(v) * 10 ** i for i, v in enumerate(row)) for row in (A[0], B[0], S[0]))
        ok = 0
        for i in range(A.shape[0]):
            a = sum(int(v) * 10 ** k for k, v in enumerate(A[i].tolist()))
            b = sum(int(v) * 10 ** k for k, v in enumerate(B[i].tolist()))
            s = sum(int(v) * 10 ** k for k, v in enumerate(S[i].tolist()))
            ok += (s + int(carry[i]) * 10 ** width) == a + b
        print(f"exact table: add_learned width {width}: {ok}/{A.shape[0]} exact (digits and carry)")
    TDC = torch.cat([torch.nn.functional.one_hot(Td.view(10, 10, 2), 10).float(),
                     torch.nn.functional.one_hot(Tb.view(10, 10, 2), 2).float()], -1).permute(0, 2, 1, 3).reshape(20, 120)
    A = torch.randint(0, 10, (2000, 9), generator=g)
    B = torch.randint(0, 10, (2000, 9), generator=g)
    Ys, cs = sa.add_learned_soft(torch.nn.functional.one_hot(A, 10).float(), torch.nn.functional.one_hot(B, 10).float(), TDC)
    Yh, ch = sa.add_learned(A, B, Td, Tb)
    print("exact table: add_learned_soft == add_learned:", bool((Ys.argmax(-1) == Yh).all() and ((cs > 0.5).long() == ch).all()))
    # the training check with the exact table must pass everywhere
    rows = load("train", 256)
    ids, am, lab = batch(rows)
    sa.MODE = "learned"
    torch.manual_seed(0)
    m = sa.build_model(spec()).train()
    with torch.no_grad():
        m.a.copy_(torch.nn.functional.one_hot(Td.view(10, 10, 2), 10).float() * 30)
        m.b.copy_(torch.nn.functional.one_hot(Tb.view(10, 10, 2), 2).float() * 30)
    sa.LOG_EVERY = 1
    m(ids, am)


def compare(Td, Tb):
    out = {}
    for name, (cd, cb) in candidates().items():
        out[name] = int(((Td == cd) & (Tb == cb)).sum())
    return out


def train(steps, B, seed=1):
    dev = os.environ.get("DEVICE", "cpu")
    torch.manual_seed(seed)
    m = sa.build_model(spec()).train().to(dev)
    bundle = sa.build_optimizer(m, OptimizerSpec(3600.0, dev))
    rows = load("train")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    tdt, tbt = sa.exact_tables("cpu")
    fix = os.environ.get("FIX", "")                     # diagnostic only: hold the borrow ("b") or digit ("a") table true
    def hold():
        with torch.no_grad():
            if "b" in fix:
                m.b.copy_(torch.nn.functional.one_hot(tbt.view(10, 10, 2), 2).float().to(m.b.device) * 8)
            if "a" in fix:
                m.a.copy_(torch.nn.functional.one_hot(tdt.view(10, 10, 2), 10).float().to(m.a.device) * 8)
    hold()
    frac = float(os.environ.get("TRUE_FRAC", "0"))     # diagnostic only: start this share of digit rows at the truth
    rows_expr = os.environ.get("TRUE_ROWS", "")       # diagnostic only: start the rows (r, m, c) matching this at the truth
    if frac > 0 or rows_expr:
        with torch.no_grad():
            pick = torch.rand(200, generator=torch.Generator().manual_seed(seed + 99)) < frac
            if rows_expr:
                pick = torch.tensor([bool(eval(rows_expr, {"r": i // 20, "m": (i // 2) % 10, "c": i % 2})) for i in range(200)])
            sharp = torch.nn.functional.one_hot(tdt, 10).float() * 4
            m.a.view(200, 10)[pick.to(m.a.device)] = sharp[pick].to(m.a.device)
    xs = (torch.arange(10)[:, None, None] + torch.arange(10)[None, :, None] + torch.arange(2)[None, None, :]).flatten()
    ad_t, ac_t = xs % 10, (xs >= 10).long()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        ids, am, lab = (t.to(dev) for t in batch([rows[i] for i in idx]))
        logits, aux = m(ids, am)
        loss = sa.token_training_loss(TokenLossBatch(logits, lab, lab != -100, None, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        bundle.optimizer.step()
        bundle.scheduler.step()
        hold()
        if step % 25 == 0 or step == 1 or m.ctl.phase == "frozen":
            Td, Tb = (t.cpu() for t in m.tables())
            cmp = compare(Td, Tb)
            best = max((k for k in cmp if k != "true"), key=lambda k: cmp[k])
            wrong = [(i // 20, (i // 2) % 10, i % 2, int(Td[i]), int(Tb[i])) for i in range(200)
                     if Td[i] != tdt[i] or Tb[i] != tbt[i]]
            add = ""
            if sa.ADDER == "table":
                Ad, Ac = (t.cpu() for t in m.adder_tables())
                add = f" adder_true {int(((Ad == ad_t) & (Ac == ac_t)).sum())}/200"
            print(f"step {step} loss {float(loss):.4f} phase {m.ctl.phase} win {sum(m.ctl.win)/max(1,len(m.ctl.win)):.3f} "
                  f"true {cmp['true']}/200{add} closest_wrong {best} {cmp[best]}/200 {time.time() - t0:.1f}s"
                  + (f" wrong(r,m,c->d,b) {wrong[:12]}" if len(wrong) <= 12 else ""), flush=True)
        if m.ctl.phase == "frozen":
            break
    Td, Tb = (t.cpu() for t in m.tables())
    torch.save({"Td": Td, "Tb": Tb, "a": m.a.detach(), "b": m.b.detach()}, os.environ.get("DUMP", "/dev/null"))
    for c in range(2):
        print(f"digit table, borrow in {c} (rows r, columns m) | borrow out")
        for r in range(10):
            print(" ".join(str(int(Td[r * 20 + mm * 2 + c])) for mm in range(10)), "|",
                  " ".join(str(int(Tb[r * 20 + mm * 2 + c])) for mm in range(10)))


if __name__ == "__main__":
    if sys.argv[1] == "designed":
        designed()
    else:
        sa.LOG_EVERY = 25
        for kv in sys.argv[5:]:
            k, v = kv.split("=", 1)
            setattr(sa, k, eval(v))
        train(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
