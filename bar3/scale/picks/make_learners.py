"""Writes learn_newton.py and learn_digits.py (PICKS.md §2): bar3/scale/subgoal/learn.py, the learner frozen at 97f30b9,
with the two-source study's replacements (bar3/scale/newton/make_learner.py) or the digits study's
(bar3/scale/digits/make_learner.py, and its DigitsTap), then the row check's (bar3/scale/rowcheck/make_learners.py),
then this study's three flags (picks.py): `tiescore`, `twin` and `bigx_score`. Every replacement is asserted to match
once, so no copy can drift from the original silently. With every flag off each copy is that study's learner
(check.py replays their records).

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/picks/make_learners.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = HERE.parent / "subgoal" / "learn.py"
sys.path.insert(0, str(ROOT))

from bar3.scale.digits import make_learner as MD  # noqa: E402  (their replacements, read, not run)
from bar3.scale.newton import make_learner as MN  # noqa: E402
from bar3.scale.rowcheck import make_learners as MR  # noqa: E402

PICKS = [
    ("from bar3.scale.rowcheck import rowcheck as RC\n",
     "from bar3.scale.rowcheck import rowcheck as RC\n"
     "from bar3.scale.picks import picks as PK\n"),
    ("        mdl0 = mdl(tp, n_sym(lib), k)\n",
     "        mdl0 = PK.mdl(cfg, tp, n_sym(lib), k)                                       # picks: bigx_score\n"),
    ("                    scores[c.name] = mdl0 - mdl(r, n_sym(lib) + 1, k)\n",
     "                    scores[c.name] = mdl0 - PK.mdl(cfg, r, n_sym(lib) + 1, k)       # picks: bigx_score\n"),
    ("                        pairs.append((mdl0 - mdl(r, n_sym(lib) + 2, k), learnable[a].name, learnable[b].name,\n",
     "                        pairs.append((mdl0 - PK.mdl(cfg, r, n_sym(lib) + 2, k), learnable[a].name,  # picks\n"
     "                                      learnable[b].name,\n"),
    ('        learnable = [c for c in pool if status.get(c.name, {}).get("status") == "learnable"]\n',
     '        learnable = [c for c in pool if status.get(c.name, {}).get("status") == "learnable"]\n'
     '        twin = PK.twins(subs, status, learnable) if cfg.get("twin") else set()       # picks: the twin\n'
     '        tie = PK.rest_scores(pool, chained, k, k2) if cfg.get("tiescore") and rule == "lp" else {}  # picks\n'
     '        subs, tsubs = subs | twin, tsubs | twin | ({tie["name"]} if tie.get("name") else set())\n'
     "        if twin or tie:\n"
     '            log(f"  picks: twin {sorted(twin)}, rest {tie}")\n'),
    ("            pick = pick[:K - 1] + [c for c in learnable if c.name in subs]   # subgoal: practised when learnable\n",
     "            pick = pick[:K - 1] + [c for c in learnable if c.name in subs]   # subgoal: practised when learnable\n"
     "        if twin and twin & {c.name for c in learnable} - {c.name for c in pick}:          # picks: the twin\n"
     "            pick = pick[:K - 1] + [c for c in learnable if c.name in twin]\n"),
    ("                  scores=scores, pairs=pairs[:200], picked=[c.name for c in pick],\n",
     "                  scores=scores, pairs=pairs[:200], picked=[c.name for c in pick],\n"
     "                  **PK.record(cfg, twin, tie),                                         # picks\n"),
]

NOTE = '''- with cfg["tiescore"], cfg["twin"] or cfg["bigx_score"], this study's three changes to the picks (bar3/scale/picks/
  picks.py), marked "# picks:" below; each round's record then also holds what they did (`picks`).
'''


def build(doc, repl, tail, out):
    text = SRC.read_text()
    body = text[text.index("from __future__"):]
    for old, new in repl + MR.ROWCHECK + PICKS:
        n = body.count(old)
        assert n == 1, (n, old)
        body = body.replace(old, new)
    marker = 'Everything else'
    assert doc.count(marker) == 1, "the doc's last line"
    doc = doc.replace(marker, MR.NOTE + NOTE + marker)
    doc = doc.replace('"""', '"""PICKS.md §2: written by bar3/scale/picks/make_learners.py, from the following.\n\n', 1)
    out.write_text(doc + body.rstrip() + "\n" + tail)
    print(f"wrote {out}: {len(repl)} + {len(MR.ROWCHECK)} + {len(PICKS)} replacements")


def main():
    build(MN.DOC, MN.REPL, "", HERE / "learn_newton.py")
    build(MD.DOC, MD.REPL, MD.TAIL, HERE / "learn_digits.py")


if __name__ == "__main__":
    main()
