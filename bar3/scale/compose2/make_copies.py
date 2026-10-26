"""Writes round 2's copies (COMPOSE2.md §2), each from a committed file with asserted replacements (each must match
exactly once), the changes behind flags whose hooks are in hooks.py:
- spcore_c2.py from integration/self_practice/spcore.py: SPELL (spellings tested a goal state, in bfs_roots), the
  tokens by signature, ROWCHECK in classify, the ties (SUBGOAL, TIEBREAK) in hardest and concepts, SUBGOAL's scores
  after each pose and its deeper probe before classify;
- oracle_c2.py from integration/self_practice/oracle.py: the copy's search, RESTRICT_ROWS, LAWNEAR;
- core_c2.py from integration/self_practice/core.py: round 1's changes (";" lists, ROUND_TOO, CLEAN_VALUES) and the
  flags, from the env;
- climb_c2.py and clean_c2.py from round 1's bar3/scale/compose/climb.py and clean.py, on the copies.
With every flag off (and SPELL 0) the copies run as round 1's driver ran.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/compose2/make_copies.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parent.parent.parent
SP = CPU / "integration" / "self_practice"
R1 = CPU / "bar3" / "scale" / "compose"

SPCORE = [
    ("import itertools\n", "import itertools\nimport os\n"),
    ("import torch\n\nfrom integration.lib.libcore import (",
     "import torch\n\nfrom bar3.scale.compose2 import hooks as H                       # compose2\n"
     "from integration.lib.libcore import ("),
    ("_SALT_ON = {}\n",
     "_SALT_ON = {}\n"
     "SPELL_K = int(os.environ.get(\"SPELL\", \"0\") or 0)              # compose2: SPELL, spellings tested a goal state\n"),
    ("                    f = programs(_OneGoal(res, g), classes, limit=1)\n"
     "                    if f:\n"
     "                        cand.append((len(f[0]), cost(expand(f[0], lib)), f[0], g))\n",
     "                    if SPELL_K and len(cand) >= max_goals:   # compose2: SPELL, at most max_goals programs a level\n"
     "                        break\n"
     "                    f = programs(_OneGoal(res, g), classes, limit=SPELL_K or 1)   # compose2: SPELL spellings\n"
     "                    for q in f[:SPELL_K or 1]:\n"
     "                        cand.append((len(q), cost(expand(q, lib)), q, g))\n"),
    ("            self.tokens_sig = (tuple(sorted(lib)), tests, set(sigs))\n",
     "            self.tokens_sig = (tuple(sorted(lib)), tests, set(sigs), {})\n"
     "            for t_, s_ in zip(toks, sigs):                             # compose2: the tokens by signature\n"
     "                self.tokens_sig[3].setdefault(s_, []).append(t_)\n"),
    ("        status, learn_sig = {}, {}\n", "        status, learn_sig, learn_rc = {}, {}, {}\n"),
    ("            if st[\"status\"] == \"learnable\":\n"
     "                s = self.signature(progs[0], self.lib)\n",
     "            if st[\"status\"] == \"learnable\" and self.rowcheck:          # compose2: ROWCHECK (ROWCHECK.md)\n"
     "                s = self.signature(progs[0], self.lib)\n"
     "                st[\"status\"], of = H.row_check(self, name, s, learn_rc, status)\n"
     "                if of is not None:\n"
     "                    st[\"of\"] = of\n"
     "            elif st[\"status\"] == \"learnable\":\n"
     "                s = self.signature(progs[0], self.lib)\n"),
    ("        return sorted(self.learnable(status), key=lambda n: -status[n][\"cost\"])[:k]\n",
     "        return sorted(self.learnable(status),                         # compose2: ties (SUBGOAL, TIEBREAK)\n"
     "                      key=lambda n: (-status[n][\"cost\"],) + H.tie_rank(self, status, [n]))[:k]\n"),
    ("        ranked = sorted(by.values(), key=lambda v: (-min(c for _, c, _, _ in v), min(i for _, _, i, _ in v)))\n",
     "        ranked = sorted(by.values(), key=lambda v: (-min(c for _, c, _, _ in v),)   # compose2: ties\n"
     "                        + H.tie_rank(self, status, [n for *_, n in v]) + (min(i for _, _, i, _ in v),))\n"),
    ("            self.pool, roots = self.pose(rows)\n",
     "            self.pool, roots = self.pose(rows)\n"
     "            H.subgoal_scores(self, rows, roots)                         # compose2: SUBGOAL\n"),
    ("        if self.phase == \"classify\":\n"
     "            self._status = self.classify()\n",
     "        if self.phase == \"classify\":\n"
     "            line = H.subgoal_deep(self, rows, deadline)               # compose2: SUBGOAL's deeper probe\n"
     "            if line is not None:\n"
     "                return line\n"
     "            self._status = self.classify()\n"),
    ("    def __init__(self, cells, device, P_round, rule=\"lp\",",
     "    rowcheck = tiebreak = subgoal_on = rest_deep = False         # compose2: ROWCHECK, TIEBREAK, SUBGOAL, REST_DEEP\n\n"
     "    def __init__(self, cells, device, P_round, rule=\"lp\","),
]

ORACLE = [
    ("from integration.self_practice.spcore import Goal, _Res, bfs_multi, width_for\n",
     "from bar3.scale.compose2.spcore_c2 import Goal, _Res, bfs_multi, width_for   # compose2: the copy\n"
     "from bar3.scale.compose2 import hooks as H\n"),
    ("        self.values = {}                            # value goal name -> (near-miss program, (item, register))\n",
     "        self.values = {}                            # value goal name -> (near-miss program, (item, register))\n"
     "        self.subgoal, self.subgoals_seen = None, []                 # compose2: SUBGOAL\n"
     "        self.lawnear = self.restrict_rows = False                   # compose2: LAWNEAR, RESTRICT_ROWS\n"),
    ("                    d0 = tr.draws[-1]\n",
     "                    d0 = H.share_rows(self, climb, rows) if self.restrict_rows else tr.draws[-1]  # RESTRICT_ROWS\n"),
    ("        rec = dict(task=\"target\", level=climb.level, found_cost=res[0][\"found_cost\"], level_sizes=sizes, fits=fits,\n",
     "        if near and self.lawnear:                                   # compose2: LAWNEAR\n"
     "            ln = H.law_near(self, climb, root, levels, parents, classes, goal)\n"
     "            out += [(f, d, p) for f, d, p, _ in ln]\n"
     "            climb.records.append(dict(task=\"lawnear\", level=climb.level,\n"
     "                                      near=[(f, d, show(p), r) for f, d, p, r in ln]))\n"
     "        rec = dict(task=\"target\", level=climb.level, found_cost=res[0][\"found_cost\"], level_sizes=sizes, fits=fits,\n"),
]

CORE = [
    ('               patterns=env("PATTERNS").split(",") if env("PATTERNS") else None,\n',
     '               patterns=env("PATTERNS").split(";") if env("PATTERNS") else None,      # compose: ";"\n'),
    ('               laws=env("LAWS").split(",") if env("LAWS") else None)\n',
     '               laws=env("LAWS").split(";") if env("LAWS") else None)                  # compose: ";"\n'),
    ("         PATTERNS / LAWS (comma lists: a subset of the law book, for smoke tests), OUT (the records folder),\n",
     "         PATTERNS / LAWS (\";\" lists: a subset of the law book), OUT (the records folder),\n"),
    ("from integration.self_practice import spcore as S\n",
     "from bar3.scale.compose2 import spcore_c2 as S                    # compose2: the copy\n"
     "from bar3.scale.compose2 import climb_c2 as CC\n"),
    ("        from integration.self_practice.oracle import SmallTarget\n",
     "        from bar3.scale.compose2.oracle_c2 import SmallTarget      # compose2: the copy\n"
     "        if env(\"CLEAN_VALUES\", \"0\") == \"1\":                    # compose: clean near-miss values only\n"
     "            from bar3.scale.compose2.clean_c2 import CleanTarget as SmallTarget\n"),
    ("        target = SmallTarget(sources=tuple(x for x in sources if x != \"law\"))\n",
     "        target = SmallTarget(sources=tuple(x for x in sources if x != \"law\"))\n"
     "        target.lawnear = env(\"LAWNEAR\", \"0\") == \"1\"              # compose2: LAWNEAR\n"
     "        target.restrict_rows = env(\"RESTRICT_ROWS\", \"0\") == \"1\"  # compose2: RESTRICT_ROWS\n"),
    ("    c = S.SelfClimb(L.Cells(10, dev), dev, 16,", "    c = CC.ComposeClimb(L.Cells(10, dev), dev, 16,"),
    ("    cfg = cfg_rec\n",
     "    c.round_too = env(\"ROUND_TOO\", \"0\") == \"1\"                    # compose: the round's search too\n"
     "    c.subgoal_on, c.tiebreak, c.rowcheck, c.rest_deep = (env(k, \"0\") == \"1\" for k in (\"SUBGOAL\", \"TIEBREAK\",\n"
     "                                                                                    \"ROWCHECK\", \"REST_DEEP\"))\n"
     "    flags = {k.lower(): True for k in (\"CLEAN_VALUES\", \"SUBGOAL\", \"LAWNEAR\", \"ROWCHECK\", \"TIEBREAK\",\n"
     "                                       \"RESTRICT_ROWS\", \"REST_DEEP\") if env(k, \"0\") == \"1\"}   # compose2: on flags only\n"
     "    if S.SPELL_K:\n"
     "        flags[\"spell\"] = S.SPELL_K\n"
     "    cfg = dict(cfg_rec, round_too=c.round_too, **flags)\n"),
]

CLIMB = [("from integration.self_practice.spcore import SelfClimb\n",
          "from bar3.scale.compose2.spcore_c2 import SelfClimb                    # compose2: the copy\n")]
CLEAN = [("from integration.self_practice.oracle import SmallTarget, prefix_values\n",
          "from bar3.scale.compose2.oracle_c2 import SmallTarget, prefix_values      # compose2: the copy\n")]

JOBS = [(SP / "spcore.py", "spcore_c2.py", SPCORE), (SP / "oracle.py", "oracle_c2.py", ORACLE),
        (SP / "core.py", "core_c2.py", CORE), (R1 / "climb.py", "climb_c2.py", CLIMB),
        (R1 / "clean.py", "clean_c2.py", CLEAN)]


def main():
    for src, dst, repl in JOBS:
        text = src.read_text()
        for old, new in repl:
            n = text.count(old)
            assert n == 1, (dst, n, old)
            text = text.replace(old, new)
        rel = src.relative_to(CPU)
        head = (f'"""COMPOSE2.md §2: written by bar3/scale/compose2/make_copies.py from {rel}, {len(repl)} asserted '
                f'replacements\n(marked "compose2"). The rest of this docstring is the original\'s.\n\n')
        assert text.startswith('"""')
        (HERE / dst).write_text(head + text[3:])
        print(f"wrote {dst}: {len(repl)} replacements")


if __name__ == "__main__":
    main()
