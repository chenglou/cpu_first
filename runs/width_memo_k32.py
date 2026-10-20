"""Is the memoized width law (spec_grammar.width_verdicts) the width law? CW's population, three ways.

1. The 53,440 machines of h100_search_nowidth_k32 (combos_from_audit): every machine that passes C0 in the requested
   search, which is what CW sees there. The memoized verdict (the same string) against width_law on every machine.
2. Statically: every factor width_law and _width_iteration read (G["..."] in their source) is outside WIDTH_IGNORES, and
   no factor is read through a computed name.
3. By perturbation: one machine per distinct key of (1), and random machines from the whole grammar (R > 4 N, so they
   get past the first check), each with every WIDTH_IGNORES factor redrawn at random a few times; the verdict must not
   change.

Run from cpu_first/: python runs/width_memo_k32.py  (CPU only, about half a minute)
"""
import ast
import collections
import inspect
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_grammar as sg  # noqa: E402

combos, _ = sg.combos_from_audit("runs/h100_search_nowidth_k32.json")
t0 = time.time()
plain = [sg.width_law(c, 24) for c in combos]
t_plain = time.time() - t0
t0 = time.time()
memo, settings = sg.width_verdicts(combos, 24)
t_memo = time.time() - t0
kept = sum(not w for w in plain)
print(f"1. {len(combos):,} machines: width_law {t_plain:.1f}s, width_verdicts {t_memo:.2f}s ({settings:,} distinct "
      f"settings); identical verdict strings: {sum(a == b for a, b in zip(plain, memo)):,} of {len(combos):,}; "
      f"kept {kept:,}, reasons {dict(collections.Counter(w.split(': ', 1)[-1] for w in plain if w))}")

reads, computed = set(), []
for fn in (sg.width_law, sg._width_iteration):
    for node in ast.walk(ast.parse(inspect.getsource(fn))):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "G":
            if isinstance(node.slice, ast.Constant):
                reads.add(node.slice.value)
            else:
                computed.append(ast.unparse(node))
print(f"2. width_law reads {len(reads)} factors: {sorted(reads)}; read and in WIDTH_IGNORES: "
      f"{sorted(reads & sg.WIDTH_IGNORES)}; read through a computed name: {computed}; neither read nor ignored: "
      f"{[f for f in sg.GRAMMAR if f not in reads | sg.WIDTH_IGNORES]}")

rng = random.Random(0)
redraw = lambda c: dict(c, **{f: rng.randrange(len(sg.GRAMMAR[f])) for f in sg.WIDTH_IGNORES})
keys = [f for f in sg.GRAMMAR if f not in sg.WIDTH_IGNORES]
one_per_key = list({tuple(c[f] for f in keys): c for c in combos}.values())
random_machines = []
while len(random_machines) < 8000:
    c = {f: rng.randrange(len(o)) for f, o in sg.GRAMMAR.items()}
    if 2 * sg.GRAMMAR["Ke"][c["Ke"]] - sg.GRAMMAR["Kx"][c["Kx"]] >= 26:
        random_machines.append(c)
for name, pop in (("one machine per distinct key of 1", one_per_key), ("random grammar machines, R > 4N", random_machines)):
    t0, same, n, reasons = time.time(), 0, 0, collections.Counter()
    for c in pop:
        w = sg.width_law(c, 24)
        reasons[w.split(": ", 1)[-1] if w else "kept"] += 1
        for _ in range(4):
            same += sg.width_law(redraw(c), 24) == w
            n += 1
    print(f"3. {name}: {len(pop):,} machines x 4 redraws of WIDTH_IGNORES: {same:,} of {n:,} verdicts unchanged "
          f"({time.time() - t0:.1f}s); verdicts {dict(reasons.most_common())}")
