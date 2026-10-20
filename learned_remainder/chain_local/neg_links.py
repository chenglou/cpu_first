"""Share of the chain's links that go negative (final borrow 1) on the rows' own x^2 divisions: exact table vs the
stalled table (exact except row (0, 0, borrow 1) -> digit 0), and the share of digits-only checks that pass."""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
from local_chain import *  # noqa
import torch.nn.functional as F

rows = load("train", 1024)
ids, am, lab = batch(rows)
Nd, Xd, Td_, end, width = sc.parse(ids, am)
x = sc.from_digits(Xd)
M = F.pad(Nd, (0, sc.P - sc.W))
for name in ["exact", "stalled (0,0,1)->0"]:
    Td, Tb = sc.exact_tables("cpu")
    if name != "exact":
        Td = Td.clone(); Td[1] = 0
    D, live = sc.link_inputs(x * x, sc.W, M, Td, Tb)
    D, Mr = D.reshape(-1, sc.P), M.repeat_interleave(sc.W * (sc.S - 1), 0)
    Dh, bh, _ = sc.link_hard(D, Mr, Td, Tb)
    back, carry = sc.add_hard(Dh, Mr)
    Te, Tbe = sc.exact_tables("cpu")
    neg_true = float((sc.link_hard(D, Mr, Te, Tbe)[1] == 1).float().mean())
    print(f"{name}: links going negative {float((bh == 1).float().mean()):.3f} (truly negative inputs {neg_true:.3f}), "
          f"digits-only check passes {float((back == D).all(-1).float().mean()):.3f}, "
          f"complete check passes {float(((back == D).all(-1) & (carry == bh)).float().mean()):.3f}, "
          f"negative links per step {float((bh == 1).float().view(-1, sc.S - 1).sum(1).mean()):.2f}")
