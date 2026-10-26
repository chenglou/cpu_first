# Addendum 2: one rule for all the wins, pre-registered

Written 2026-10-26 12:45 PDT, while addendum 1's runs (tag ab1) were under way, before any run below. CONSOLIDATE.md
records its hash.

## What ab1 had shown at writing (20 of its 74 runs ended)

**The rebuilt old rules bring their old wins back:**
- `lp_probe` (hardest first with the uniform probe) had brought 7:
  - x² mod N, r0;
  - x² − 1, x² + 3, x³, x² + x, (x + 1)², 3·x², each r1.
- `ct2a2_probe` (anchored pairs with the uniform probe) had brought 4: x² + 1 r0, x² − 1 r0, x² + x r0, Fibonacci r1.
- In 5 of the 6 runs with a round 0 so far, `ct2a2_probe` made all four of the old anchored pairs' round-0 picks
  (x² − 1: 3 of 4). `lp_probe` made 3 or 4 of hardest first's, except on 10·x²: 1 of 4.
  - There the allocator's third rung, taken by value, solved three more cost-7 candidates that the old learner left
    unsolved (it gave 10M only to the subgoal).
  - Hardest first took those in place of the constant 10 in three registers.

**Either half alone brings little:**
- `probe` (the uniform probe, picks by the threshold) had brought 3: x² mod N, x² − 1, x² + 3.
- `lp` (hardest first without the uniform probe) had brought none.
- `k4` (four by value) and `ct2a2` (anchored pairs without the uniform probe) had brought only x² − 1.

## Questions

1. **Does one old choosing rule carry every earlier win?**
   - By the earlier records, anchored pairs alone brought 15 of the 16 wins, all but 10·x² + 1.
   - ab1 runs `ct2a2_probe` on 7 of them. Here it runs on the other 9.
2. **Does the subgoal's tie-break do work?**
   - It is the last of the coordinator's listed rules still inside the rebuilt rules.
   - `ct2a2_probe_noties` turns it off. The subgoal's deeper probe stays, as the allocator's third rung.
3. **Can one score stand for both choosing rules, given a fixed count and the uniform probe?**
   - `kmax_probe` practises the four learnable candidates of highest max(measured gain, prior). The prior is the
     allocator's: (cost − 1)·tok, the bits the routine saves per use, or the subgoal score's bits when more.
   - So a null look cannot erase a routine's prior. That is the optimistic estimate: a shallow look that finds
     something shows a gain, and one that finds nothing shows little.
   - It was designed after cs1 and the part of ab1 above.
   - `k4_probe` (four by the allocator's own value) is not run: under the uniform probe nearly every learnable
     candidate is looked at, so its ranking among null looks would fall to pool order. The money goes to `kmax_probe`.

## Arms and formulas (40 runs, in 16 groups of one formula each)

- **`ct2a2_probe`** on 9 formulas: x² mod N, 2x + 1, x² + 3, x³, 10·x², (x + 1)², 3·x², x⁴ (alloc_deep), and 10·x² + 1
  (alloc_ns).
- **`kmax_probe`** on all 16 earlier wins.
- **`ct2a2_probe_noties`** on the 15 earlier wins anchored pairs ever brought: all but 10·x² + 1.
- **Stores:** each group reads its ab1 group's store, or cs1's for 2x + 1.

## Predictions

| arm | earlier wins that come | confidence |
|---|---|---|
| `ct2a2_probe`, the 9 | at least 6 of anchored pairs' 8: x² mod N r0, 2x + 1 r0, x² + 3 r1, x³ r1, (x + 1)² r1, 3·x² r1, x⁴ r1, 10·x² (r4 before, so the least sure); not 10·x² + 1 | medium-high |
| `ct2a2_probe_noties` | the same as `ct2a2_probe`, but for at most 2 | medium |
| `kmax_probe` | 6 to 11 of 16. The subgoal score's bits outrank most measured gains and costs, so it practises the subgoal's candidates first | low |

## How the outcome is read

- **If `ct2a2_probe` brings all of anchored pairs' wins:** one old rule carries every earlier win but 10·x² + 1. That
  rule is anchored pairs, with the uniform 1M probe and four picks a round. Hardest first, the subgoal's separately
  budgeted deeper probe, the twin, the twin as a fifth pick and the deeper look can all go. 10·x² + 1 needs hardest
  first with the rest's tie-break.
- **If `ct2a2_probe_noties` brings the same:** the subgoal's tie-break can go too.
- **If `kmax_probe` brings about as many as `ct2a2_probe`:** one score can stand for both choosing rules, given a fixed
  count and the uniform probe.

## Money

- **So far:** cs1 cost about $5. ab1 is estimated at about $19, from its states: about $0.05 per billion states, by
  cs1. These 40 runs are estimated at about $12.
- **The limits stand:** soft $40, cap $50, `spend.py 4750` before each launch, one watcher for every live app.
- **If money runs short,** the 9 `ct2a2_probe` runs come first, then `kmax_probe`, then `ct2a2_probe_noties`.
