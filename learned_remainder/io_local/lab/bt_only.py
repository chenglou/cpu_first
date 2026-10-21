"""IO.md §7, item 6. Rank code fit by Bradley-Terry on the decisive digit pairs of x < N (true reading fixed), all training rows.
Decisive pair: the highest place where x's and N's tokens differ (where x - N sets its final borrow); when x is shorter,
the pair is (padding, N's top digit) and the padding ranks with the lowest digit token."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packdata import pack
seed, steps, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(seed)
num = pack("train")["num"]
P10 = 10 ** torch.arange(9)
u = torch.nn.Parameter(0.1 * torch.randn(10))          # one score per digit token (tokens 7..16 = digits 0..9)
opt = torch.optim.Adam([u], lr=float(os.environ.get("LR", "0.05")))
g = torch.Generator().manual_seed(seed)
tau = float(os.environ.get("TAU", "1"))
for step in range(1, steps + 1):
    idx = torch.randint(0, len(num), (B,), generator=g)
    N, x = num[idx, 0], num[idx, 1]
    Nd, xd = (N[:, None] // P10) % 10, (x[:, None] // P10) % 10          # tokens' digit identities, LSB first
    wN = (N[:, None] >= P10).sum(1); wx = (x[:, None] >= P10).sum(1)
    pad = torch.arange(9)[None] >= wx[:, None]                            # x's dead slots
    diff = (Nd != xd) | (pad & (torch.arange(9)[None] < wN[:, None]))
    k = torch.where(diff, torch.arange(9)[None], -1).max(1).values        # decisive place
    a, b = xd.gather(1, k[:, None])[:, 0], Nd.gather(1, k[:, None])[:, 0]
    ispad = pad.gather(1, k[:, None])[:, 0]
    ua = torch.where(ispad, u.min().detach().expand(B) - 0 * u[a], u[a])  # padding: the lowest digit's score
    ua = torch.where(ispad, u[u.argmin()].expand(B), u[a])
    loss = -torch.nn.functional.logsigmoid((u[b] - ua) / tau).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        rank = u.argsort().argsort()
        print(step, round(float(loss), 4), "code(0..9)", "".join(str(int(r)) for r in rank), flush=True)
