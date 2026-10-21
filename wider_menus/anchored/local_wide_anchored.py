"""Training diagnostics of a wide-menu submission (WIDE_ANCHORED.md: submission_wide_anchored.py, or submission_wide.py
for comparison) outside the evaluator: learned_wiring/anchored/local_anchored.py on wider_menus/local/local_wide.py's
training loop, with what the anchors' schedule needs watched:
  * the reader against the truth: the share of the batch's rows whose snapped reading of N, x and T equals the row's
    own modulus, x and T (the data's fields, read here outside the submission), of rows read as N = 0, and of rows
    whose read x is below their read N (what the borrow facts assume);
  * the subtraction table's borrows and digits against true subtraction (each out of 200), the borrow facts' streak
    and the step the digit lessons start;
  * the step's gradient norm before the evaluator-style clipping (all parameters), and the norm of T's pointer
    gradient (its anchor and step logits).
A compact line every 5 steps (the reader is checked before the step's lessons, on the batch they see) and
local_wide.py's line every 25; at the end the depth rungs on both prediction paths and local_wide.py's two gradient
tests. Needs the H1 jsonl files (H1_DATA) and torch; runs on Modal (modal_diag_wide_anchored.py).

  python local_wide_anchored.py FILE STEPS B [SEED] [K=V ...]   (K=V sets the constant K to the Python source V)
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "local"))
import local_wide as lw  # noqa: E402  (puts the evaluator stub on the path)
import torch  # noqa: E402
from benchmark import OptimizerSpec, TokenLossBatch  # noqa: E402


def truth_digits(vals, n):
    """The digits of vals in n places, LSB first [X, n]."""
    v = torch.tensor(vals, dtype=torch.long)[:, None]
    return (v // 10 ** torch.arange(n)) % 10


def reader_report(m, ids, am, rows) -> str:
    """The snapped reader on the batch against the rows' own numbers."""
    sc = lw.sc
    with torch.no_grad():
        e, end = m.ext(ids, am)
        k = m.selected(e, end)
        code = m.ranks(k)
        Nd, _ = m.read(0, e, end, code)
        Xd, _ = m.read(1, e, end, code)
        Cd, _ = m.read(2, e, end, code)
    okN = (Nd == truth_digits([r["modulus"] for r in rows], sc.W)).all(1)
    okX = (Xd == truth_digits([r["x"] for r in rows], sc.W)).all(1)
    okT = (Cd == truth_digits([r["time_steps"] for r in rows], sc.K)).all(1)
    pw = 10 ** torch.arange(sc.W)
    less = (Xd * pw).sum(1) < (Nd * pw).sum(1)
    f = lambda t: f"{float(t.float().mean()):.4f}"
    return (f"read N={f(okN)} x={f(okX)} T={f(okT)} N,x={f(okN & okX)} whole_runs={f(k >= 0)} "
            f"N_read_0={f((Nd == 0).all(1))} read_x<N={f(less)}")


def table_counts(m):
    """Borrows right and digits right out of 200 each."""
    Td, Tb = m.tables()[:2]
    tdt, tbt = lw.sc.exact_tables("cpu")
    return int((Tb == tbt).sum()), int((Td == tdt).sum())


def train(steps, B, seed=1, rungs_n=128):
    sc = lw.sc
    torch.manual_seed(seed)
    m = sc.build_model(lw.spec()).train()
    bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    rows = lw.load("train")
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows), (B,), generator=g)
        brows = [rows[i] for i in idx]
        ids, am, lab, tpos = lw.batch(brows)
        rd = reader_report(m, ids, am, brows) if step % 5 == 0 or step == 1 else None
        logits, aux = m(ids, am)
        loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        bundle.optimizer.zero_grad()
        loss.backward()
        gT = sum(float(p.grad[2].norm()) ** 2 for p in (m.anc, m.strd) if p.grad is not None) ** 0.5
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))   # the norm before clipping
        bundle.optimizer.step()
        bundle.scheduler.step()
        c = m.ctl
        if rd is not None:
            nb, nd = table_counts(m)
            print(f"s {step} phase {c.phase} io {c.io} ladder {int(c.ladder)} wire_io {int(c.wire_io)} "
                  f"borrows {nb}/200 digits {nd}/200 table_exact {aux.get('table_exact', 0.0):.4f} "
                  f"facts_streak {getattr(c, 'facts_streak', '-')} order_digits {getattr(c, 'gated_on', '-')} {rd} "
                  f"loss {float(loss.detach()):.4f} gradnorm {gn:.3e} T_pointer_grad {gT:.3e} "
                  f"{time.time() - t0:.1f}s", flush=True)
        if step % 25 == 0 or step == 1 or c.phase == "frozen":
            wr, pw, gate = lw.table_report(m)
            print(f"step {step} loss {float(loss.detach()):.4f} phase {c.phase} io {c.io} ladder {int(c.ladder)} "
                  f"rows_right {200 - len(wr)}/200 cells_right {100 - len(pw)}/100 gate {gate} io_exact {c.io_exact} "
                  f"chosen {c.chosen:.3f} wiring: {m.wiring_names()} layout: {m.layout_names()} "
                  f"wrong {wr[:6]} cells_wrong {pw[:6]} {c.t_log} {time.time() - t0:.1f}s", flush=True)
        if c.phase == "frozen":
            break
    print(f"history {m.ctl.history}", flush=True)
    m.eval()
    lw.eval_paths(m, lw.RUNGS, rungs_n)
    lw.gradtest(m)
    lw.gradtest(m, 32, ("depth_t_8", "depth_ood_n_t_4"))


if __name__ == "__main__":
    argv = sys.argv[2:]
    lw.sc = lw.load_module(sys.argv[1], [a for a in argv if "=" in a])
    args = [a for a in argv if "=" not in a]
    lw.sc.LOG_EVERY = 25
    train(int(args[0]), int(args[1]), int(args[2]) if len(args) > 2 else 1)
