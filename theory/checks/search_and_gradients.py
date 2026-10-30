# Search, networks and gradients on a tiny law reader. Backs theory/SEARCH_AND_GRADIENTS.md.
#
# Candidate laws f(x) mod N come from a small grammar (leaves x and 1, + and ×, at most 5 tokens), deduplicated by
# what they compute and ordered shortest first. A task is R rows (x, N, y). Three ways to pick a law from a task:
#   search     the first candidate that fits every row;
#   network    that search compiled into a ReLU network: weighted sums and max(0, ·), nothing else;
#   gradients  softmax weights over the candidates, trained by gradient descent on the labels' log-likelihood.
# Then a random ReLU network rewritten as if-else source, and a hard-wired law for comparison.
# Run: python3 theory/checks/search_and_gradients.py   (needs numpy; under a minute)
import math
import random

import numpy as np

NMAX = 7     # moduli 2..NMAX, so x and y lie in 0..NMAX-1
R = 6        # rows per task
SIZE = 5     # tokens per expression
PAIRS = [(x, n) for n in range(2, NMAX + 1) for x in range(n)]


# ---- the grammar and the search ----

def exprs(size, memo={}):
    """Expressions with exactly `size` tokens: ('x',), ('1',), ('+', a, b), ('*', a, b)."""
    if size not in memo:
        out = [("x",), ("1",)] if size == 1 else []
        for left in range(1, size - 1):
            for op in "+*":
                out += [(op, a, b) for a in exprs(left) for b in exprs(size - 1 - left)]
        memo[size] = out
    return memo[size]


def value(e, x):
    if len(e) == 1:
        return x if e[0] == "x" else 1
    a, b = value(e[1], x), value(e[2], x)
    return a + b if e[0] == "+" else a * b


def text(e):
    return e[0] if len(e) == 1 else f"({text(e[1])}{e[0]}{text(e[2])})"


CANDS, seen = [], set()
for size in range(1, SIZE + 1, 2):               # shortest first: the search's order
    for e in exprs(size):
        key = tuple(value(e, x) % n for x, n in PAIRS)
        if key not in seen:
            seen.add(key)
            CANDS.append(e)
K = len(CANDS)
NONE = K


def search(rows):
    """The index of the first candidate that fits every row, or NONE."""
    for i, e in enumerate(CANDS):
        if all(value(e, x) % n == y for x, n, y in rows):
            return i
    return NONE


def fitting(rows):
    return {i for i, e in enumerate(CANDS) if all(value(e, x) % n == y for x, n, y in rows)}


# ---- the search as a ReLU network ----

class Net:
    """A ReLU network written as a graph. Each node is a weighted sum of earlier nodes plus a constant, or max(0, ·) of
    one earlier node; nothing else. Integer bounds are tracked so the gadgets know how many units they need."""

    def __init__(self):
        self.nodes, self.lo, self.hi = [], [], []

    def _add(self, node, lo, hi):
        self.nodes.append(node)
        self.lo.append(lo)
        self.hi.append(hi)
        return len(self.nodes) - 1

    def input(self, col, lo, hi):
        return self._add(("in", col), lo, hi)

    def sum(self, terms, const=0, lo=None, hi=None):
        if lo is None:
            lo = const + sum(w * (self.lo[n] if w > 0 else self.hi[n]) for n, w in terms)
            hi = const + sum(w * (self.hi[n] if w > 0 else self.lo[n]) for n, w in terms)
        return self._add(("sum", terms, const), lo, hi)

    def relu(self, n):
        return self._add(("relu", n), max(0, self.lo[n]), max(0, self.hi[n]))

    def run(self, X, outs):
        last = {}
        for j, node in enumerate(self.nodes):
            for n in ([t[0] for t in node[1]] if node[0] == "sum" else [node[1]] if node[0] == "relu" else []):
                last[n] = j
        keep, vals = set(outs), [None] * len(self.nodes)
        for j, node in enumerate(self.nodes):
            if node[0] == "in":
                vals[j] = X[:, node[1]].astype(float)
                continue
            if node[0] == "sum":
                v = np.full(len(X), float(node[2]))
                for n, w in node[1]:
                    v += w * vals[n]
                reads = [t[0] for t in node[1]]
            else:
                v = np.maximum(vals[node[1]], 0.0)
                reads = [node[1]]
            vals[j] = v
            for n in reads:
                if last[n] == j and n not in keep:
                    vals[n] = None
        return np.stack([vals[o] for o in outs], 1)

    def size(self):
        relus = sum(node[0] == "relu" for node in self.nodes)
        weights = sum(len(node[1]) for node in self.nodes if node[0] == "sum")
        depth = [0] * len(self.nodes)
        for j, node in enumerate(self.nodes):
            if node[0] == "sum":
                depth[j] = max([depth[n] for n, _ in node[1]], default=0)
            elif node[0] == "relu":
                depth[j] = depth[node[1]] + 1
        return relus, weights, max(depth)


def step(net, z):                     # [z >= 1] for integer z
    return net.sum([(net.relu(z), 1), (net.relu(net.sum([(z, 1)], -1)), -1)], lo=0, hi=1)


def square(net, v):                   # v*v for integer v >= 0: the sum of 2k + 1 over k < v
    return net.sum([(step(net, net.sum([(v, 1)], -k)), 2 * k + 1) for k in range(int(net.hi[v]))],
                   lo=0, hi=net.hi[v] ** 2)


def absval(net, d):
    return net.sum([(net.relu(d), 1), (net.relu(net.sum([(d, -1)])), 1)], lo=0, hi=max(-net.lo[d], net.hi[d]))


def mul(net, a, b):                   # a*b for integers a, b >= 0: ((a + b)^2 - |a - b|^2) / 4
    s = square(net, net.sum([(a, 1), (b, 1)]))
    d = square(net, absval(net, net.sum([(a, 1), (b, -1)])))
    return net.sum([(s, 0.25), (d, -0.25)], lo=net.lo[a] * net.lo[b], hi=net.hi[a] * net.hi[b])


def mod(net, v, n):                   # v mod N for integers v >= 0 and 2 <= N <= NMAX
    terms = [(v, 1)]
    for q in range(1, int(net.hi[v]) // 2 + 1):
        ge = step(net, net.sum([(v, 1), (n, -q)], 1))                               # [v >= q*N]
        terms.append((net.relu(net.sum([(n, 1), (ge, NMAX)], -NMAX)), -1))          # N if so, else 0
    return net.sum(terms, lo=0, hi=NMAX - 1)


def equal(net, a, b):                 # [a == b] for integers: max(0, 1 - |a - b|)
    return net.relu(net.sum([(absval(net, net.sum([(a, 1), (b, -1)])), -1)], 1))


def build_search_net():
    net = Net()
    one = net.sum([], 1)
    fits = []
    rows = [(net.input(3 * r, 0, NMAX - 1), net.input(3 * r + 1, 2, NMAX), net.input(3 * r + 2, 0, NMAX - 1))
            for r in range(R)]
    made = [{} for _ in range(R)]     # shared subexpressions, per row

    def compile_expr(e, r):
        key = text(e)
        if key not in made[r]:
            if e == ("x",):
                made[r][key] = rows[r][0]
            elif e == ("1",):
                made[r][key] = one
            else:
                a, b = compile_expr(e[1], r), compile_expr(e[2], r)
                made[r][key] = net.sum([(a, 1), (b, 1)]) if e[0] == "+" else mul(net, a, b)
        return made[r][key]

    for e in CANDS:
        hits = [equal(net, mod(net, compile_expr(e, r), rows[r][1]), rows[r][2]) for r in range(R)]
        fits.append(net.relu(net.sum([(h, 1) for h in hits], 1 - R)))                     # every row fits
    outs = [net.relu(net.sum([(fits[i], 1)] + [(fits[j], -1) for j in range(i)])) for i in range(K)]
    outs.append(net.relu(net.sum([(f, -1) for f in fits], 1)))                             # none fits
    return net, outs


# ---- gradients over selection weights ----

def gradients(rows, steps=3000, lr=1.0, eps=1e-3):
    """Softmax weights over the candidates, trained on -sum log q_r, where q_r is the probability the mixture gives the
    label of row r (eps keeps it finite when no candidate explains a row). Returns the final weights p."""
    A = np.array([[value(e, x) % n == y for e in CANDS] for x, n, y in rows], float)
    u = np.array([1.0 / n for _, n, _ in rows])
    w = np.zeros(K)
    for _ in range(steps):
        p = np.exp(w - w.max())
        p /= p.sum()
        s = A @ p
        q = (1 - eps) * s + eps * u
        w += lr * (1 - eps) * p * ((A - s[:, None]) / q[:, None]).sum(0)    # minus the loss's gradient
    p = np.exp(w - w.max())
    return p / p.sum()


# ---- tasks ----

def task(rng, law=None, flips=0):
    rows = []
    for _ in range(R):
        x, n = rng.choice(PAIRS)
        rows.append((x, n, value(CANDS[law], x) % n if law is not None else rng.randrange(n)))
    for r in rng.sample(range(R), flips):
        x, n, y = rows[r]
        rows[r] = (x, n, (y + rng.randrange(1, n)) % n)
    return rows


def main():
    rng = random.Random(0)
    print(f"{K} candidate laws, shortest first:", ", ".join(text(e) for e in CANDS))

    # 1. The search compiled into a ReLU network gives the same pick on every task.
    net, outs = build_search_net()
    relus, weights, depth = net.size()
    print(f"\n1. The network: {relus:,} ReLU units, {weights:,} weights, {depth} ReLU layers deep, no other operation.")
    tasks = [task(rng, law) for law in range(K) for _ in range(300)]
    tasks += [task(rng, rng.randrange(K), flips=1) for _ in range(1000)]
    tasks += [task(rng) for _ in range(1000)]
    X = np.array([[v for row in t for v in row] for t in tasks])
    Y = net.run(X, outs)
    exact = bool(np.all((Y == 0) | (Y == 1)) and np.all(Y.sum(1) == 1))
    agree = sum(int(np.argmax(Y[i])) == search(t) for i, t in enumerate(tasks))
    picks = [search(t) for t in tasks]
    print(f"   {len(tasks):,} tasks ({K} x 300 from each law, 1,000 with one label flipped, 1,000 random):")
    print(f"   outputs exactly one-hot: {exact}; same pick as the search: {agree:,} of {len(tasks):,}"
          f" ({sum(p == NONE for p in picks):,} of them 'none fits').")

    # 2. Gradients keep exactly the candidates that fit every row.
    groups = {"one fits": [], "several fit": [], "none fits": []}
    for t in tasks[::5]:
        c = fitting(t)
        groups["one fits" if len(c) == 1 else "several fit" if c else "none fits"].append((t, c))
    print("\n2. Gradients (3,000 steps of descent on the softmax weights), on every 5th task above:")
    for name, items in groups.items():
        mass, top_in, spread, top, share = [], 0, [], [], []
        for t, c in items:
            p = gradients(t)
            mass.append(sum(p[i] for i in c))
            top_in += int(np.argmax(p)) in c or not c
            spread.append(int((p >= 0.05).sum()))
            top.append(float(p.max()))
            if c:
                share.append(p[search(t)] / mass[-1])
        if name == "none fits":
            print(f"   {name:>11}: {len(items):4} tasks; the search says none; gradients settle on a blend of"
                  f" {np.mean(spread):.1f} candidates at >= 5% each (median top weight {np.median(top):.2f})")
        else:
            print(f"   {name:>11}: {len(items):4} tasks; weight on the fitting candidates: min {min(mass):.4f},"
                  f" median {np.median(mass):.4f}; top weight among them: {top_in} of {len(items)}; the search's"
                  f" pick holds a median {np.median(share):.2f} of that weight")

    # 3. What is picked changes with the data; a hard-wired law does not.
    square_law = next(i for i, e in enumerate(CANDS) if text(e) == "(x*x)")
    rows_by_law = {}
    for law in range(K):
        for _ in range(2000):
            t = task(rng, law)
            if fitting(t) == {law}:
                rows_by_law[law] = t
                break
    laws = sorted(rows_by_law)
    Xs = np.array([[v for row in rows_by_law[k] for v in row] for k in laws])
    by_net = [int(np.argmax(y)) for y in net.run(Xs, outs)]
    by_search = [search(rows_by_law[k]) for k in laws]
    by_grad = [int(np.argmax(gradients(rows_by_law[k]))) for k in laws]
    print(f"\n3. One task per law that its rows pin down ({len(laws)} of {K} laws found such rows in 2,000 tries):")
    for name, picks in (("search", by_search), ("network", by_net), ("gradients", by_grad),
                        ("hard-wired x*x", [square_law] * len(laws))):
        right = sum(p == k for p, k in zip(picks, laws))
        print(f"   {name:>14}: {len(set(picks)):2} different picks, right on {right:2} of {len(laws)};"
              f" bits decided by the data: {math.log2(len(set(picks))):.2f}")

    # 4. A ReLU network is an if-else program: rewrite one, compare, count its regions.
    g = np.random.default_rng(0)
    W1, b1 = g.integers(-3, 4, (16, 2)), g.integers(-5, 6, 16)
    W2, b2 = g.integers(-3, 4, (16, 16)), g.integers(-5, 6, 16)
    W3, b3 = g.integers(-3, 4, 16), int(g.integers(-5, 6))
    lines = ["def program(x0, x1):"]
    for j in range(16):
        lines += [f"    a = {W1[j, 0]}*x0 + {W1[j, 1]}*x1 + {b1[j]}", f"    if a > 0: h{j} = a",
                  f"    else: h{j} = 0"]
    for j in range(16):
        lines += [f"    a = " + " + ".join(f"{W2[j, k]}*h{k}" for k in range(16)) + f" + {b2[j]}",
                  f"    if a > 0: g{j} = a", f"    else: g{j} = 0"]
    lines.append("    return " + " + ".join(f"{W3[k]}*g{k}" for k in range(16)) + f" + {b3}")
    source = "\n".join(lines)
    assert "max" not in source and "relu" not in source
    space = {}
    exec(source, space)
    grid = [(a, b) for a in range(-30, 31) for b in range(-30, 31)]
    same, patterns = 0, {}
    for a, b in grid:
        h = np.maximum(W1 @ np.array([a, b]) + b1, 0)
        z = np.maximum(W2 @ h + b2, 0)
        same += int(W3 @ z + b3) == space["program"](a, b)
        patterns.setdefault(tuple(h > 0) + tuple(z > 0), []).append((a, b))
    affine = 0
    for pattern, pts in patterns.items():     # within a region the network is one affine formula, read off the weights
        dh, dz = np.diag(pattern[:16]).astype(int), np.diag(pattern[16:]).astype(int)
        coef = W3 @ dz @ W2 @ dh @ W1
        const = W3 @ dz @ (W2 @ dh @ b1 + b2) + b3
        affine += all(space["program"](a, b) == coef[0] * a + coef[1] * b + const for a, b in pts)
    print(f"\n4. A random ReLU network (2 -> 16 -> 16 -> 1, integer weights) rewritten as {len(lines)} lines of"
          f" if-else with no max or ReLU:")
    print(f"   same output on {same:,} of {len(grid):,} grid points; {len(patterns)} regions on the grid, and the"
          f" network is one affine formula on {affine} of them")


if __name__ == "__main__":
    main()
