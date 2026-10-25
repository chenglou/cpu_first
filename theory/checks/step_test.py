# Is t(x*B + d) a function of (t(x), d, N)?  (and unary: t(x+1) a function of (t(x), N)?)
# Also: is it a function of (x mod N, d, N) [Horner state]?  Small N, x up to a bound.
from math import gcd

def fib_mod(n, N):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % N
    return a % N

def inv2(N):
    return pow(2, -1, N)

targets = {
    "x mod N":       (lambda x, N: x % N, False),
    "2x+1":          (lambda x, N: (2*x + 1) % N, False),
    "2^x":           (lambda x, N: pow(2, x, N), False),
    "x^2":           (lambda x, N: (x*x) % N, False),
    "x^2+1":         (lambda x, N: (x*x + 1) % N, False),
    "x^2-1":         (lambda x, N: (x*x - 1) % N, False),
    "x^2+3":         (lambda x, N: (x*x + 3) % N, False),
    "10x^2":         (lambda x, N: (10*x*x) % N, False),
    "10x^2+1":       (lambda x, N: (10*x*x + 1) % N, False),
    "x^3":           (lambda x, N: pow(x, 3, N), False),
    "mont x^2/2":    (lambda x, N: (x*x*inv2(N)) % N, True),   # N odd only
    "fib":           (lambda x, N: None, False),               # handled separately
}

def is_function(rows):
    seen = {}
    for key, val in rows:
        if key in seen and seen[key] != val:
            return False
        seen[key] = val
    return True

def check(name, f, odd_only, B, Nmax=7, X=2000):
    rows_ans, rows_res, fails = [], [], []
    for N in range(1, Nmax + 1):
        if odd_only and N % 2 == 0:
            continue
        vals = {}
        def t(x):
            if name == "fib":
                return fib_mod(x, N) if N > 1 else 0
            return f(x, N)
        for x in range(0, X):
            if B == 1:
                nxt = [(x + 1, 0)]
            else:
                nxt = [(x * B + d, d) for d in range(B)]
            for y, d in nxt:
                if y >= X * (B if B > 1 else 1) + B:
                    continue
                rows_ans.append(((t(x), d, N), t(y)))
                rows_res.append(((x % N, d, N), t(y)))
    return is_function(rows_ans), is_function(rows_res)

for name, (f, odd) in targets.items():
    out = []
    for B in (1, 2, 10):
        a, r = check(name, f, odd, B, X=600 if name == "fib" else 2000)
        out.append(f"B={B}: answer-state {'yes' if a else 'no '} / (x mod N)-state {'yes' if r else 'no '}")
    print(f"{name:12s} | " + " | ".join(out))

# Per-N detail for x^2 and x^3 and fib with B=10: which N break the answer-state test
def per_N(name, f, B):
    bad = []
    for N in range(1, 8):
        rows = []
        for x in range(0, 700):
            for d in range(B):
                y = x * B + d
                tx = fib_mod(x, N) if name == 'fib' else f(x, N)
                ty = fib_mod(y, N) if name == 'fib' else f(y, N)
                rows.append(((tx, d), ty))
        if not is_function(rows):
            bad.append(N)
    return bad
for name in ("x^2", "x^3", "10x^2", "x^2+1", "fib"):
    f = targets[name][0]
    print(name, "B=10 fails at N =", per_N(name, f, 10), "; B=2 fails at N =", per_N(name, f, 2))
