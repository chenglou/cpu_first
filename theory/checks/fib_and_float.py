import math, random


def fib_pair(n, N):
    a, b = 0, 1 % N
    for _ in range(n):
        a, b = b, (a + b) % N
    return (a % N, b % N)

def is_function(rows):
    seen = {}
    for k, v in rows:
        if k in seen and seen[k] != v:
            return False
        seen[k] = v
    return True

for B in (1, 2, 10):
    rows = []
    for N in range(1, 8):
        for x in range(0, 400):
            nxt = [(x + 1, 0)] if B == 1 else [(x * B + d, d) for d in range(B)]
            for y, d in nxt:
                rows.append(((fib_pair(x, N), d, N), fib_pair(y, N)))
    print("fib pair-state, B =", B, ":", is_function(rows))

# Float check: exp(2*log(x)) rounded vs x*x, for x near the top of H1's range (x < 2^24)
random.seed(0)
xs = [random.randrange(2**23, 2**24) for _ in range(200000)]
bad64 = sum(1 for x in xs if round(math.exp(2 * math.log(x))) != x * x)
import struct
def f32(v):
    return struct.unpack('f', struct.pack('f', v))[0]
bad32 = 0; direct32 = 0
for x in xs:
    lx = f32(math.log(f32(x)))
    sq = f32(math.exp(f32(2.0 * lx)))
    if round(sq) != x * x: bad32 += 1
    if f32(f32(x) * f32(x)) != x * x: direct32 += 1
print(f"x in [2^23, 2^24): fp64 exp(2 ln x) wrong after rounding: {bad64}/{len(xs)}")
print(f"fp32 exp(2 ln x) wrong: {bad32}/{len(xs)}; fp32 direct x*x wrong: {direct32}/{len(xs)}")
# fp64 at 8 decimal digits (x < 1e8), beyond H1 sizes
xs8 = [random.randrange(5 * 10**7, 10**8) for _ in range(200000)]
bad64_8 = sum(1 for x in xs8 if round(math.exp(2 * math.log(x))) != x * x)
direct64_8 = sum(1 for x in xs8 if float(x) * float(x) != x * x)
print(f"x in [5e7, 1e8): fp64 exp(2 ln x) wrong: {bad64_8}/{len(xs8)}; fp64 direct x*x inexact: {direct64_8}/{len(xs8)}")
print("2^24 =", 2**24, " 2^48 =", 2**48, " digits of 2^48:", len(str(2**48)))
