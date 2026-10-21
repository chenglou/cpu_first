"""The wide grammar's semantics, checked on exact arithmetic (MENUS.md section 1): the menu sizes, long division by
repeated subtraction and by trial multiples and their aliases exact, the Montgomery wiring computing V 10^-W mod N, and
which number rules square correctly.

    ~/github/sotaku/venv/bin/python wider_menus/check_grammar.py
"""
import sys, torch, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from wider_menus import wide as G
from wider_menus.tiny_wide import Data, coprime_moduli
for W in (3, 4):
    P = W + 1
    for lvl in range(4):
        dm = G.div_menu(lvl, W)
        print(W, lvl, "div", G.size(dm), "chain", G.size(dm, G.CHAIN), "io", G.size(dm, G.IO))
    for lvl in range(3):
        mm = G.mul_menu(lvl, W)
        print(W, lvl, "mul", G.size(mm), len(mm["lo"]))
    d = Data.cert(W, 3000, 0)
    dm = G.div_menu(3, W)
    def wiring(over):
        t = dict(G.DIV_TRUE); t.update(over)
        ix = {k: torch.full((d.n,), dm[k].index(t[k])) for k in dm}
        return G.decode(dm, ix, W)
    Vd = G.digits(torch.from_numpy(d.x * d.x), 2 * W)
    for name, over in [("truth", {}), ("trial", {"a": "Rp", "b": "Mq"}), ("bring add", {"bring": (1, 0, "add")}),
                       ("csub", {"final": "csub"}), ("addhigh", {"final": "addhigh"})]:
        out = G.divide_val(Vd, d.Nd, wiring(over))
        print(W, name, float((out == d.yd).all(1).double().mean()))
    mont = {"start": "zero", "order": "up", "bring": (0, 0, "add"), "op": "add", "flag": "zD", "post": -1, "final": "addhigh"}
    out = G.value(G.divide_val(Vd, d.Nd, wiring(mont)))
    Rinv = np.array([pow(10 ** W, -1, int(n)) for n in d.N])
    redc = (d.x * d.x % d.N) * Rinv % d.N
    print(W, "montgomery == V 10^-W mod N:", float((out.numpy() == redc).mean()), " == V mod N:", float((out.numpy() == d.y).mean()))
    # multiply layouts
    mm = G.mul_menu(2, W)
    xd = d.xd
    for name, num in [("by j", (0, 1)), ("by i", (1, 0)), ("diag i-j", (1, -1)), ("diag j-i", (-1, 1)), ("anti i+j", (1, 1)), ("const", (0, 0))]:
        cols = G.claims((1, 1, 0), (1, 1, 1), num, W)
        sq = G.mul_val(xd, xd, cols) % 10 ** (2 * W)
        print(W, "mul", name, float((sq.numpy() == d.x * d.x).mean()))
    cr = G.claims((1, 1, 0), (1, 1, 1), (0, 1), W, rows_only=True)
    ok = all(bool((G.row_val(xd, torch.full((d.n,), b), j, cr).numpy() == d.x * b * 10 ** j).all()) for b in range(10) for j in range(W))
    print(W, "rows exact", ok)
    # rule maps vs claims (rows_only) agree for every rule
    maps = G.rule_maps(mm["lo"], W)
    bad = 0
    for r, rule in enumerate(mm["lo"][:400]):
        cr = G.claims(rule, rule, (0, 1), W, rows_only=True)
        for j in range(W):
            for c in range(2 * W + 2):
                src = [i for i in range(W) if cr[0, i, j] == c]
                m = int(maps[r, j, c])
                if (src[0] if src else W) != m:
                    bad += 1
    print(W, "map mismatches", bad)
