"""IO.md §7, item 5 (free and rank codes taught by "x < N" through the soft chain). Code-only check on ALL training rows: with the true reading fixed, does "x < N" (exact subtraction chain as the
comparator) teach the digit code? Variants via env: CODE=free|rank, ST=1 (hard ranks forward), LIN=1, RT=<step>."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reader_lab5 as L
from packdata import pack
seed, steps, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(seed)
num = pack("train")["num"]
P10 = 10 ** torch.arange(L.P)
cv = torch.nn.Parameter(0.1 * torch.randn(17, 10))
opt = torch.optim.Adam([cv], lr=float(os.environ.get("LR", "0.05")))
TDC = L.exact_tdc()
g = torch.Generator().manual_seed(seed)
mode = os.environ.get("CODE", "free")
tau, sw = float(os.environ.get("TAU", "1")), float(os.environ.get("SW", "0.5"))
rt_from = int(os.environ.get("RT", "100000000"))
def code():
    if mode == "rank":
        u = cv[7:, 0]
        r = torch.sigmoid((u[:, None] - u[None, :]) / tau).sum(1) - 0.5
        soft = (-(torch.arange(10.)[None] - r[:, None]) ** 2 / (2 * sw * sw)).softmax(-1)
        if os.environ.get("ST"):
            soft = torch.nn.functional.one_hot(u.argsort().argsort(), 10).float() + soft - soft.detach()
        return soft
    return cv[7:].softmax(-1)
T4 = TDC.view(10, 2, 10, 12)
def borrow_tc(px, pN, same, TDC):
    """Final borrow of x - N, places whose tokens are equal share one value (the code is a function of the token)."""
    X = px.shape[0]
    c = torch.zeros(X, 2); c[:, 0] = 1
    for j in range(px.shape[1]):
        J = torch.where(same[:, j, None, None], torch.diag_embed(px[:, j]), px[:, j, :, None] * pN[:, j, None, :])
        c = torch.einsum("xrm,xc,rcmo->xo", J, c, T4[..., 10:])
    return c[:, 1]
for step in range(1, steps + 1):
    idx = torch.randint(0, len(num), (B,), generator=g)
    N, x = num[idx, 0], num[idx, 1]
    Nd, xd = (N[:, None] // P10) % 10, (x[:, None] // P10) % 10
    wN = torch.tensor([len(str(int(a))) for a in N]); wx = torch.tensor([len(str(int(a))) for a in x])
    lN = (torch.arange(L.P)[None] < wN[:, None]).float()[..., None]; lx = (torch.arange(L.P)[None] < wx[:, None]).float()[..., None]
    c = code()
    VN = lN * c[Nd] + (1 - lN) * L.ZERO
    Vx = lx * c[xd] + (1 - lx) * L.ZERO
    if os.environ.get("TC"):
        same = torch.cat([(Nd == xd) & (lN[..., 0] == lx[..., 0]).bool() | ((lN[..., 0] == 0) & (lx[..., 0] == 0)),
                          torch.zeros(B, 0, dtype=torch.bool)], 1)
        p = borrow_tc(Vx, VN, same, TDC)
    else:
        p = L.borrow_soft(Vx, VN, TDC)
    loss = (1 - p).mean() if os.environ.get("LIN") else -p.clamp_min(1e-6).log().mean()
    if step > rt_from:
        rt = (c * c / c.sum(0, keepdim=True)).sum(-1)
        loss = loss - rt.clamp_min(1e-6).log().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        with torch.no_grad():
            hard = code().argmax(-1)
            ok = ((x * 0 + (hard[xd] * P10).sum(1)) < (hard[Nd] * P10).sum(1)).float().mean()
        print(step, round(float(loss), 4), "code(0..9)", "".join(str(int(h)) for h in hard), "hard x<N", round(float(ok), 4), flush=True)
