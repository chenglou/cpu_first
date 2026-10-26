"""This study's configs (CONSOLIDATE.md §2): each earlier win's config with the allocator in place of its choosing
rule and the old spending rules (alloc.py), the row check on (a bug fix), and every old patch to the picks off
(the rest's tie-break, the twin, the twin as a fifth pick, the deeper look, the big-x score). The subgoal score stays,
as one of the allocator's values, and its deeper probe's budget is the allocator's third probe rung.
- `alloc_deep`: the held-out test's frozen config (`lp_clean_deep_sub`'s sources and budgets);
- `alloc_ns`: the two-source config (`lp_clean_deep_sub_ns`: the running sum and the offset);
- `alloc_b_src`: the digits config (`lp_clean_b_src`: base 2, the one-more-digit source).
LEARNER[config] names the learner copy.

The ablation's configs (PREREG_ADDENDUM1.md) add one old rule back, or two, to each: `<config>_lp` (hardest first's
four picks), `<config>_k4` (four picks by value), `<config>_probe` (the uniform second probe), `<config>_lp_probe`,
`<config>_ct2a2` (anchored pairs' picks: the old rule's own code on the allocator's classification), `<config>_ct2a2_probe`;
and addendum 2's (PREREG_ADDENDUM2.md): `<config>_ct2a2_probe_noties`, with the subgoal's ties off; `<config>_k4_probe`;
`<config>_kmax_probe` (four by max(measured gain, prior))."""

from __future__ import annotations

from bar3.scale.picks.configs import CONFIGS as PK
from bar3.scale.subgoal.configs import DEEP

OLD_PATCHES = ("tiescore", "twin", "twin5", "deeplook", "bigx_score")


def _alloc(base):
    c = dict(PK[base], rule="alloc", alloc=True, rowcheck=True, subgoal=dict(deep=DEEP))
    for f in OLD_PATCHES:
        c.pop(f, None)
    return c


CONFIGS = {"alloc_deep": _alloc("lp_clean_deep_sub"), "alloc_ns": _alloc("lp_clean_deep_sub_ns"),
           "alloc_b_src": _alloc("lp_clean_b_src")}
LEARNER = {"alloc_deep": "newton", "alloc_ns": "newton", "alloc_b_src": "digits"}
ABLATE = {"lp": dict(alloc_pick="lp"), "k4": dict(alloc_pick="k"), "probe": dict(alloc_probe_all=True),
          "lp_probe": dict(alloc_pick="lp", alloc_probe_all=True),
          "ct2a2": dict(alloc_oldpick=True, rule="ct2", anchor_cost=2),
          "ct2a2_probe": dict(alloc_oldpick=True, rule="ct2", anchor_cost=2, alloc_probe_all=True),
          "ct2a2_probe_noties": dict(alloc_oldpick=True, rule="ct2", anchor_cost=2, alloc_probe_all=True,
                                     subgoal=dict(deep=DEEP, ties=False)),
          "k4_probe": dict(alloc_pick="k", alloc_probe_all=True),
          "kmax_probe": dict(alloc_pick="kmax", alloc_probe_all=True)}
for _b in list(CONFIGS):
    for _a, _f in ABLATE.items():
        CONFIGS[f"{_b}_{_a}"], LEARNER[f"{_b}_{_a}"] = dict(CONFIGS[_b], **_f), LEARNER[_b]
