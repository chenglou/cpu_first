# Addendum 1: the ablation, pre-registered

Written 2026-10-26 12:20 PDT, after the pre-registered allocator's runs (tag cs1) and before any run below.
CONSOLIDATE.md records its hash.

## What cs1 showed (why these arms)

**Outcomes:**
- The allocator came for 2x + 1 (round 0) and for none of the other earlier wins.
- That covers x² mod N, x² ± 1, x² + 3, Fibonacci, x³, 10·x², (x + 1)², 3·x², x⁴, 2^x, the triangular numbers and 10·x² + 1 (two-source config), and binary x mod N (digits config).
- x² + x was still running at writing (round 3, not come).
- The non-wins did not come, as predicted.

**Mechanisms, measured over its 93 rounds** (logs/cs1_mechanisms.txt, logs/cs1_oldpicks.txt):
1. **59 rounds practised nothing.** No measured gain beat its library cost, so the library did not change, and each later round repeated the one before.
2. **Of 8,720 first looks, 6% showed any gain.** A first look is the target searched at 30k states with one routine. A null look replaces the routine's prior, so that routine is never practised.

   A look this shallow rarely finds the target. A routine's worth can also show only with another routine, or a round later: hardest first's round-0 picks for x³, practised with no measured gain, brought x³ in round 1.
3. **Round 0 searched a median 0.38 of hardest first's states.**
   - R counts nominal budgets, while a search overshoots its nominal budget by finishing its last level.
   - So the allocator probed at 1M about a quarter of the candidates the old rule probed.
   - Most of the old rules' round-0 picks were among those it left unprobed (x² mod N: 0 of hardest first's 4 were learnable to it).

## Arms

Each arm is the cs1 allocator (alloc_deep, alloc_ns or alloc_b_src) with one old rule added back, or two. Each rule sits behind its own flag, off in cs1.
- **`probe`:** the old uniform second probe. Every candidate the census left unsolved is probed at 1M, outside R. The picks are still the gain threshold's.
- **`lp`:** hardest first's picks in place of the threshold: the 4 costliest learnable candidates (ties: the subgoal's, then pool order). The probes are still by value.
- **`k4`:** the fixed count alone: the 4 learnable candidates of highest value (the measured gain, else the prior), with no threshold.
- **`lp_probe`:** `lp` and `probe` together. This is close to the old hardest first; the differences are the row check, and the third rung taken by value in place of the subgoal's deeper probe.
- **`ct2a2`:** anchored pairs' picks, run by the old rule's own code on the allocator's classification. That code looks at every learnable candidate, adds pair looks anchored on the 2 costliest, and picks four.
- **`ct2a2_probe`:** `ct2a2` with the uniform probe. This is close to the old anchored pairs.

**Formulas:**
- `probe`, `lp`, `k4` and `lp_probe` run on every earlier win the allocator lost:
  - x² mod N, x² + 1, x² − 1, x² + 3, Fibonacci, x³, 10·x², (x + 1)², 3·x², x⁴, x² + x and 2^x (alloc_deep);
  - the triangular numbers and 10·x² + 1 (alloc_ns);
  - binary x mod N (alloc_b_src).
- `ct2a2` and `ct2a2_probe` run where anchored pairs won alone or first: 2^x, the triangular numbers, x² − 1, x² + 1, x² + x, Fibonacci and binary x mod N.
- That makes 74 runs, in 15 groups of one formula each, read from the cs1 group's store.
- If cs1 brings x² + x after all, its arms count as a check, not as an ablation of a lost win.

## Predictions

| arm | lost wins that come | confidence |
|---|---|---|
| `probe` | at most 4 of 15: the threshold still blocks, since most looks stay null | medium |
| `lp` | 5 to 9 of hardest first's 12; where its picks needed the 1M probe, they differ | low-medium |
| `k4` | 4 to 8 of 15: no stall; an unlooked costly routine's prior outranks most measured gains, so it often picks as `lp` does | low |
| `lp_probe` | at least 9 of hardest first's 12 (x² mod N, x² ± 1, x² + 3, Fibonacci, x³, 10·x², (x + 1)², 3·x², x⁴, x² + x, binary x mod N); not 2^x, the triangular numbers or 10·x² + 1, which hardest first never brought without a patch | medium-high |
| `ct2a2_probe` | at least 5 of its 7 | medium |
| `ct2a2` | 1 to 4 of 7 | low |

## How the outcome is read

These rules are written before the runs:
- **If `lp_probe` and `ct2a2_probe` bring most of the lost wins and `probe` does not,** then what carries the wins is practising a fixed number of routines every round, ranked by cost or by looks. The threshold is what loses them.
- **If `lp` brings about as many as `lp_probe`,** hardest first does not need the uniform probe.
- **If `k4` brings about as many as `lp`,** the fixed count is the working part, not the ranking by cost.

**What counts as general:** a rule is general when it names no formula, problem or source, and has no constant set for a formula.
- Hardest first is general. Cost is a property of any routine.
- The fixed count of four is general, but a constant.
- The uniform probe is general. It is a budget rule.
- Anchored pairs are general.

## Money

- cs1 cost about $5. These 74 runs are estimated at $15 to $20.
- The limits stand: soft $40, cap $50 overall, `spend.py 4750` before each launch, and one watcher for all apps.
- If money runs short, `lp_probe` and `ct2a2_probe` come first (the old rules, rebuilt), then `probe`, `lp`, `k4` and `ct2a2`.
