"""At the first recipe's stall (local seed 6, batch 64: only row (0, 0, borrow 1) wrong, digit 0 instead of 9), slide that
row's digit from 0 to 9 (swap the two logits gradually) and measure the link check's soft loss on fixed link inputs:
the digits-only identity (the first recipe) and the complete one (digits + final borrow = adder's final carry)."""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from local_chain import *  # noqa
import torch.nn.functional as F

sc.ID_BORROW = 0.0
sc.LOG_EVERY = 10 ** 9
torch.manual_seed(6)
m = sc.build_model(spec()).train()
bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
rows = load("train")
g = torch.Generator().manual_seed(6)
tdt, tbt = sc.exact_tables("cpu")
for step in range(1, 701):
    idx = torch.randint(0, len(rows), (64,), generator=g)
    ids, am, lab = batch([rows[i] for i in idx])
    logits, aux = m(ids, am)
    loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, None, aux))
    bundle.optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    bundle.optimizer.step(); bundle.scheduler.step()
    Td, Tb = m.tables()
    wrong = [i for i in range(200) if Td[i] != tdt[i] or Tb[i] != tbt[i]]
    if step >= 550 and wrong == [1]:          # row (0, 0, 1) is flat index 1
        break
print("stopped at step", step, "wrong rows", wrong)
a, b = m.a.detach().clone(), m.b.detach().clone()
print("row (0,0,1) digit probs", [round(p, 3) for p in a[0, 0, 1].softmax(-1).tolist()],
      "borrow probs", [round(p, 3) for p in b[0, 0, 1].softmax(-1).tolist()])
probe = rows[:1024]
ids, am, lab = batch(probe)
with torch.no_grad():
    Nd, Xd, Td_, end, width = sc.parse(ids, am)
    x = sc.from_digits(Xd)
    M = F.pad(Nd, (0, sc.P - sc.W))
    D, live = sc.link_inputs(x * x, sc.W, M, Td, Tb)
    D, Mr = D.reshape(-1, sc.P), M.repeat_interleave(sc.W * (sc.S - 1), 0)
weight = torch.ones(sc.P); weight[0] = sc.ID_PLACE0
for lam in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
    a2 = a.clone()
    a2[0, 0, 1, 0] = (1 - lam) * a[0, 0, 1, 0] + lam * a[0, 0, 1, 9]
    a2[0, 0, 1, 9] = (1 - lam) * a[0, 0, 1, 9] + lam * a[0, 0, 1, 0]
    TDC = torch.cat([a2.softmax(-1), b.softmax(-1)], -1).permute(0, 2, 1, 3).reshape(20, 120)
    with torch.no_grad():
        Ds, bs = sc.link_soft(F.one_hot(D, 10).float(), Mr, TDC)
        back, carry = sc.add_soft(Ds, Mr)
        nll = -back.clamp_min(1e-12).log().gather(-1, D[..., None]).squeeze(-1)
        digits = float((nll * weight).sum(-1).mean() / weight.sum())
        agree = carry * bs[:, 1] + (1 - carry) * bs[:, 0]
        full = float(((nll * weight).sum(-1) - agree.clamp_min(1e-12).log()).mean() / (weight.sum() + 1))
    print(f"lambda {lam:.1f} P(9) {float(a2[0, 0, 1].softmax(-1)[9]):.3f} digits-only check {digits:.4f} "
          f"complete check {full:.4f}", flush=True)
