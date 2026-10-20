"""Which table rows are wrong during a short local run (snapped table vs true subtraction, printed every 25 steps), and
how often the digits-only check and "x mod N = x" pass on 500 probe rows.
  python diag_chain.py STEPS B SEED [ID_LINKS ID_BORROW [noself]]   # e.g. 700 64 6 all 0 (the first recipe's stall)"""
import os, sys, time
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from local_chain import *  # noqa
import torch.nn.functional as F


def wrong_rows(m):
    Td, Tb = m.tables()
    tdt, tbt = sc.exact_tables("cpu")
    out = []
    for i in range(200):
        if Td[i] != tdt[i] or Tb[i] != tbt[i]:
            out.append(f"({i // 20},{(i // 2) % 10},{i % 2})->({int(Td[i])},{int(Tb[i])}) truth ({int(tdt[i])},{int(tbt[i])})")
    return out


def checks(m, rows):
    ids, am, lab = batch(rows)
    with torch.no_grad():
        Nd, Xd, Td_, end, width = sc.parse(ids, am)
        x = sc.from_digits(Xd)
        M = F.pad(Nd, (0, sc.P - sc.W))
        Td, Tb = m.tables()
        D = sc.link_inputs(x * x, sc.W, M, Td, Tb)[0].reshape(-1, sc.P)
        Mr = M.repeat_interleave(sc.W * (sc.S - 1), 0)
        Dh, _, _ = sc.link_hard(D, Mr, Td, Tb)
        ident = float((sc.add_hard(Dh, Mr)[0] == D).all(-1).float().mean())
        selfx = float((sc.divide_hard(x, 1, M, Td, Tb) == sc.to_digits(x, sc.P)).all(1).float().mean())
    return ident, selfx


def main(steps, B, seed):
    torch.manual_seed(seed)
    sc.LOG_EVERY = 10 ** 9
    m = sc.build_model(spec()).train()
    bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    rows = load("train")
    probe = rows[:2000:4]
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
        if step % 25 == 0:
            ident, selfx = checks(m, probe)
            w = wrong_rows(m)
            print(f"step {step} {m.ctl.phase} ident {ident:.3f} xmodN {selfx:.3f} wrong {len(w)}: {' '.join(w[:6])} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if m.ctl.phase == "frozen":
            print("frozen", step)
            break


sc.ID_LINKS = sys.argv[4] if len(sys.argv) > 4 else "all"
sc.ID_BORROW = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
sc.SELF_LESSON = (sys.argv[6] != "noself") if len(sys.argv) > 6 else True
print("ID_LINKS", sc.ID_LINKS, "ID_BORROW", sc.ID_BORROW, "SELF_LESSON", sc.SELF_LESSON)
main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
