"""Writes round 3's copies (COMPOSE3.md §2): the committed files with round 2's replacement lists
(bar3/scale/compose2/make_copies.py, module paths moved to this folder), then round 3's, each asserted to match
exactly once, the changes behind flags:
- TARGET_FN (hooks.TARGET_F): the oracle's answers on its draws, and the 400 answer rows' at full size, are that
  function of (x, N): x2 (the default, the data's own answers), x3, 10x2p1;
- TIEFOLLOW (hooks.follow): the tie-break finds the rest of the way from the practised subgoal by its value.
`hooks.py` is round 2's hooks.py with the same path moves and round 3's replacements; it keeps round 2's subgoal
score (bar3/scale/compose2/subgoal_c2.py). With every round-3 flag off the copies run as round 2's ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compose3.make_copies3
"""

from __future__ import annotations

from pathlib import Path

from bar3.scale.compose2 import make_copies as M2

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent


def move(s):
    """Round 2's module paths, moved to this folder (its subgoal score stays round 2's)."""
    for a, b in (("bar3.scale.compose2.", "bar3.scale.compose3."), ("bar3.scale.compose2 import", "bar3.scale.compose3 import"),
                 ("spcore_c2", "spcore_c3"), ("oracle_c2", "oracle_c3"), ("climb_c2", "climb_c3"), ("clean_c2", "clean_c3")):
        s = s.replace(a, b)
    return s.replace("bar3.scale.compose3 import subgoal_c2", "bar3.scale.compose2 import subgoal_c2")


R2 = lambda lst: [(old, move(new)) for old, new in lst]          # noqa: E731

SPCORE3 = [
    ("    rowcheck = tiebreak = subgoal_on = rest_deep = False         # compose2: ROWCHECK, TIEBREAK, SUBGOAL, REST_DEEP\n",
     "    rowcheck = tiebreak = subgoal_on = rest_deep = False         # compose2: ROWCHECK, TIEBREAK, SUBGOAL, REST_DEEP\n"
     "    tiefollow = False                                            # compose3: TIEFOLLOW\n"),
]
ORACLE3 = [
    ("        self.ys = [x * x % N for x, N in zip(self.xs, self.Ns)]\n",
     "        self.ys = [H.TARGET_F(x, N) for x, N in zip(self.xs, self.Ns)]      # compose3: TARGET_FN\n"),
]
CORE3 = [
    ("import torch\n\n", "import torch\n\nfrom bar3.scale.compose3 import hooks as H                   # compose3\n"),
    ('    ans = tuple(digits(arows, k).to(dev) for k in ("modulus", "x", "result"))\n',
     '    if env("TARGET_FN", "x2") != "x2":                          # compose3: TARGET_FN\'s answers, by Python\n'
     '        arows = [dict(r, result=H.TARGET_F(int(r["x"]), int(r["modulus"]))) for r in arows]\n'
     '    ans = tuple(digits(arows, k).to(dev) for k in ("modulus", "x", "result"))\n'),
    ("    cfg = dict(cfg_rec, round_too=c.round_too, **flags)\n",
     "    c.tiefollow = env(\"TIEFOLLOW\", \"0\") == \"1\"                    # compose3: TIEFOLLOW\n"
     "    if c.tiefollow:\n"
     "        flags[\"tiefollow\"] = True\n"
     "    if env(\"TARGET_FN\", \"x2\") != \"x2\":\n"
     "        flags[\"target_fn\"] = env(\"TARGET_FN\")\n"
     "    cfg = dict(cfg_rec, round_too=c.round_too, **flags)\n"),
]
HOOKS3 = [
    ("SUB_DEEP = 10_000_000                           # SUBGOAL.md's deeper probe\n",
     "SUB_DEEP = 10_000_000                           # SUBGOAL.md's deeper probe\n"
     "TARGET_FN = os.environ.get(\"TARGET_FN\", \"x2\")   # compose3: the target, a function of (x, N)\n"
     "TARGET_F = {\"x2\": lambda x, N: x * x % N, \"x3\": lambda x, N: pow(x, 3, N),\n"
     "            \"10x2p1\": lambda x, N: (10 * x * x + 1) % N}[TARGET_FN]\n"),
    ("from types import SimpleNamespace\n", "import os\nfrom types import SimpleNamespace\n"),
    ("        rr = next((x for x in climb._level_roots if x.key == f\"rest.{sub[4:]}@XN\"), None)\n"
     "        if rr is None:\n"
     "            continue\n",
     "        rr = next((x for x in climb._level_roots if x.key == f\"rest.{sub[4:]}@XN\"), None)\n"
     "        if rr is None and getattr(climb, \"tiefollow\", False):  # compose3: TIEFOLLOW, the rest found by its value\n"
     "            rr = follow(climb, sub)\n"
     "        if rr is None:\n"
     "            continue\n"),
    ("def ties(climb, status):\n",
     "def follow(climb, sub):\n"
     "    \"\"\"TIEFOLLOW: the rest root posed this level whose starting value, run on the target's probes, is the practised\n"
     "    subgoal's value there (a value is posed again under a new name each level), or None.\"\"\"\n"
     "    from bar3.scale.compose3.oracle_c3 import prefix_values\n"
     "    t = climb.target\n"
     "    tr = getattr(t, \"_root\", None)\n"
     "    if tr is None or sub not in t.values:\n"
     "        return None\n"
     "    run = lambda nm: [prefix_values(t.values[nm][0], st, rt, climb.lib)[t.values[nm][1]]   # noqa: E731\n"
     "                      for st, rt in zip(climb.settings, tr.roots)]\n"
     "    want = run(sub)\n"
     "    for x in climb._level_roots:\n"
     "        if getattr(x, \"kind\", None) != \"rest\":\n"
     "            continue\n"
     "        nm = \"val.\" + x.key.split(\"@\")[0][len(\"rest.\"):]\n"
     "        if nm != sub and nm in t.values and all(bool((a == b).all()) for a, b in zip(want, run(nm))):\n"
     "            return x\n"
     "    return None\n"
     "\n\n"
     "def ties(climb, status):\n"),
]

SP = CPU / "integration" / "self_practice"
R1 = CPU / "bar3" / "scale" / "compose"
JOBS = [(SP / "spcore.py", "spcore_c3.py", R2(M2.SPCORE) + SPCORE3), (SP / "oracle.py", "oracle_c3.py", R2(M2.ORACLE) + ORACLE3),
        (SP / "core.py", "core_c3.py", R2(M2.CORE) + CORE3), (R1 / "climb.py", "climb_c3.py", R2(M2.CLIMB)),
        (R1 / "clean.py", "clean_c3.py", R2(M2.CLEAN))]


def main():
    for src, dst, repl in JOBS:
        text = src.read_text()
        for old, new in repl:
            n = text.count(old)
            assert n == 1, (dst, n, old)
            text = text.replace(old, new)
        rel = src.relative_to(CPU)
        head = (f'"""COMPOSE3.md §2: written by bar3/scale/compose3/make_copies3.py from {rel}, {len(repl)} asserted '
                f'replacements\n(round 2\'s marked "compose2", round 3\'s "compose3"). The rest of this docstring is the '
                f'original\'s.\n\n')
        assert text.startswith('"""')
        (HERE / dst).write_text(head + text[3:])
        print(f"wrote {dst}: {len(repl)} replacements")
    src = CPU / "bar3" / "scale" / "compose2" / "hooks.py"
    text = move(src.read_text())
    for old, new in HOOKS3:
        n = text.count(old)
        assert n == 1, ("hooks.py", n, old)
        text = text.replace(old, new)
    head = ('"""COMPOSE3.md §2: written by bar3/scale/compose3/make_copies3.py from bar3/scale/compose2/hooks.py, its module\n'
            f'paths moved here and {len(HOOKS3)} asserted replacements (TARGET_FN, TIEFOLLOW). The rest of this docstring is\n'
            'the original\'s.\n\n')
    (HERE / "hooks.py").write_text(head + text[3:])
    print(f"wrote hooks.py: {len(HOOKS3)} replacements")


if __name__ == "__main__":
    main()
