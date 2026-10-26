"""Writes learn_newton.py and learn_digits.py (CONSOLIDATE.md §2): bar3/scale/reach's two learner copies (the learner
frozen at 97f30b9 with every earlier study's flags, each replacement asserted there), with the allocator (alloc.py)
behind cfg["alloc"]: it takes the place of the probes (the old first probe, its census, stays the same job) and of the
choosing step; with cfg["alloc_oldpick"] too (the ablation, PREREG_ADDENDUM1.md), the old choosing rule cfg["rule"] runs
on the allocator's classification instead. Each replacement here is asserted to match once. With cfg["alloc"] off each
copy is reach's.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.consolidate.make_learners
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REACH = HERE.parent / "reach"

ALLOC = [
    ("from bar3.scale.reach import reach as RH\n",
     "from bar3.scale.reach import reach as RH\n"
     "from bar3.scale.consolidate import alloc as AL\n"),
    ("        if rule in (\"random\", \"curated\"):\n"
     "            ra = rb = []\n",
     "        if cfg.get(\"alloc\"):                                                        # consolidate: the allocator\n"
     "            ra, rb, deep, status, al = AL.probe_and_choose(cfg, pool, lib, kinds, B, tgt, tp, k, k2, run_jobs, sigs,\n"
     "                                                           sub, mdl0, n_sym, how, log)\n"
     "        elif rule in (\"random\", \"curated\"):\n"
     "            ra = rb = []\n"),
    ("        if rule == \"curated\":\n"
     "            pick = list(pool)\n",
     "        if cfg.get(\"alloc\") and not cfg.get(\"alloc_oldpick\"):                     # consolidate\n"
     "            pick, scores = al[\"pick\"], al[\"scores\"]\n"
     "        elif rule == \"curated\":\n"
     "            pick = list(pool)\n"),
    ("**({\"reach\": dict(twin5=sorted(twin5))} if cfg.get(\"twin5\") else {}),   # reach\n",
     "**({\"reach\": dict(twin5=sorted(twin5))} if cfg.get(\"twin5\") else {}),   # reach\n"
     "                  **({\"alloc\": al[\"record\"]} if cfg.get(\"alloc\") else {}),           # consolidate\n"),
]


def main():
    for name in ("learn_newton", "learn_digits"):
        text = (REACH / f"{name}.py").read_text()
        for old, new in ALLOC:
            n = text.count(old)
            assert n == 1, (name, n, old)
            text = text.replace(old, new)
        head = (f'"""CONSOLIDATE.md §2: written by bar3/scale/consolidate/make_learners.py from bar3/scale/reach/{name}.py, '
                f'with {len(ALLOC)}\nasserted replacements (marked "consolidate"): the allocator behind cfg["alloc"]. '
                'The rest of this docstring is the\noriginal\'s.\n\n')
        assert text.startswith('"""')
        (HERE / f"{name}.py").write_text(head + text[3:])
        print(f"wrote {name}.py: {len(ALLOC)} replacements")


if __name__ == "__main__":
    main()
