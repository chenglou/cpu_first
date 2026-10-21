"""Which consistent subtraction tables pass which checks (tiny: 256 training rows, integer tables, a few seconds).

Each candidate is a full 200-row table (r, m, borrow in) -> (digit, borrow out). Every check runs on the candidate's
own numbers, as training would: the link inputs D of its own hard division of the rows' x^2, and the rows' N.
  python landscape.py [ROWS]
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "chain_local", "stub"), os.path.dirname(HERE)]
import torch
import submission_chain as sc

DATA = os.environ.get("H1_DATA", (
    "/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-"
    "4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-"
    "efb0abec6f53/scratchpad/data/h1"))


def candidates():
    r = torch.arange(10)[:, None, None].expand(10, 10, 2)
    m = torch.arange(10)[None, :, None].expand(10, 10, 2)
    c = torch.arange(2)[None, None, :].expand(10, 10, 2)
    out = {}
    x = r - m - c
    out["true"] = (x % 10, (x < 0).long())
    out["borrow_free"] = ((r - m - c) % 10, torch.zeros_like(r))
    out["copy_minuend"] = (r.clone(), torch.zeros_like(r))
    xs = r - m - (1 - c)
    out["swapped_code"] = (xs % 10, 1 - (xs < 0).long())
    xr = m - r - c
    out["reversed"] = (xr % 10, (xr < 0).long())
    out["negated_digits"] = ((r - m + c) % 10, (((-r) % 10) - ((-m) % 10) - c < 0).long())
    # digit d = (d mod 2, d mod 5); borrows only in one component (other abelian groups of order 10^P)
    crt = lambda a2, a5: (5 * a2 + 6 * a5) % 10
    x5 = r % 5 - m % 5 - c
    out["z5_borrows"] = (crt((r - m) % 2, x5 % 5), (x5 < 0).long())
    x2 = r % 2 - m % 2 - c
    out["z2_borrows"] = (crt(x2 % 2, (r - m) % 5), (x2 < 0).long())
    d, b = (x % 10).clone(), (x < 0).long()
    d[0, 0, 1] = 0
    out["stuck_001"] = (d, b)
    # what route A settled on locally (seed 1): copy the minuend, 0 on the diagonal, a borrow that sticks once raised
    lazy_d = torch.where((r == m) & (c == 0), 0, r)
    lazy_b = torch.where(c == 1, 1, ((m != 0) & (r != m)).long())
    out["lazy_sticky"] = (lazy_d, lazy_b)
    return {k: (v[0].flatten(), v[1].flatten()) for k, v in out.items()}


def load(n):
    with open(os.path.join(DATA, "train.jsonl")) as f:
        lines = f.readlines()
    g = torch.Generator().manual_seed(0)
    return [json.loads(lines[i]) for i in torch.randperm(len(lines), generator=g)[:n].tolist()]


def sub(A, B, Td, Tb):
    """A - B on integers, any width: digits, final borrow."""
    c = torch.zeros_like(A[:, 0])
    out = []
    for j in range(A.shape[1]):
        idx = (A[:, j] * 10 + B[:, j]) * 2 + c
        out.append(Td[idx])
        c = Tb[idx]
    return torch.stack(out, 1), c


def main(n):
    rows = load(n)
    N = torch.tensor([r["modulus"] for r in rows])
    x = torch.tensor([r["x"] for r in rows])
    T = torch.tensor([r["time_steps"] for r in rows])
    y = torch.tensor([r["result"] for r in rows])
    M = sc.to_digits(N, sc.P)
    Mx = M.repeat_interleave(sc.W * (sc.S - 1), 0)
    print(f"rows={n} T values={sorted(set(T.tolist()))}")
    hdr = ("table", "id_digits", "id_complete", "id_negborrow", "a-0=a", "a-a=0", "0-(0-N)=N", "D-(D-N)=N",
           "R+N=N+R", "x-N borrows", "N-x no borrow", "(x-N)-N no borrow", "x mod N=x", "answers T=1", "designed-adder id")
    print(" | ".join(hdr))
    for name, (Td, Tb) in candidates().items():
        D, _ = sc.link_inputs(x * x, sc.W, M, Td, Tb)
        D = D.reshape(-1, sc.P)
        E, be = sub(D, Mx, Td, Tb)                                   # the link: D - N
        Z = torch.zeros_like(Mx)
        C, bc = sub(Z, Mx, Td, Tb)                                   # 0 - N
        F, bf = sub(E, C, Td, Tb)                                    # (D - N) - (0 - N) = (D - N) + N
        dig = (F == D).all(1)
        comp = dig & (be == bc - bf)
        negb = dig & (be + bf == 1) & (bc == 1)
        a0 = (sub(D, torch.zeros_like(D), Td, Tb)[0] == D).all(1) & (sub(D, torch.zeros_like(D), Td, Tb)[1] == 0)
        Daa, baa = sub(D, D, Td, Tb)
        aa = (Daa == 0).all(1) & (baa == 0)
        nn_ = (sub(Z, C, Td, Tb)[0] == Mx).all(1)
        dd = (sub(D, E, Td, Tb)[0] == Mx).all(1)
        Rg = D.view(n, sc.W, sc.S - 1, sc.P)[:, :, 0].reshape(n * sc.W, sc.P)
        MW = M.repeat_interleave(sc.W, 0)
        CW, bcw = sub(torch.zeros_like(MW), MW, Td, Tb)
        A1, b1 = sub(Rg, CW, Td, Tb)                                  # R + N = R - (0 - N)
        CR, bcr = sub(torch.zeros_like(Rg), Rg, Td, Tb)
        A2, b2 = sub(MW, CR, Td, Tb)                                  # N + R = N - (0 - R)
        comm = (A1 == A2).all(1) & (bcw - b1 == bcr - b2)
        xd = sc.to_digits(x, sc.P)
        less = sub(xd, M, Td, Tb)[1] == 1
        more = sub(M, xd, Td, Tb)[1] == 0
        wrap = sub(sub(xd, M, Td, Tb)[0], M, Td, Tb)[1] == 0
        xm = (sc.divide_hard(x, 1, M, Td, Tb) == sc.to_digits(x, sc.P)).all(1)
        t1 = T == 1
        ans = (sc.from_digits(sc.divide_hard(x[t1] * x[t1], sc.W, M[t1], Td, Tb)) == y[t1])
        back, carry = sc.add_hard(E, Mx)
        des = (back == D).all(1) & (carry == be)
        f = lambda v: f"{float(v.float().mean()):.3f}"
        print(" | ".join([name, f(dig), f(comp), f(negb), f(a0), f(aa), f(nn_), f(dd), f(comm), f(less), f(more), f(wrap), f(xm), f(ans), f(des)]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)
