# Stage 3, one hand-made part at a time

*Written 2026-10-23/24 (PT) for a judge of the contest. Code, plans and records are in
[`integration/ablation_s3/`](ablation_s3/). The model is stage 3's final file,
[`submission_stage3_s3i.py`](submission_stage3_s3i.py) (s3i, md5 `0e8131440e1842546c67e7378caf472d`), which this
study never edits.*

## In plain words

<!-- PLAIN -->
**The question.** In one training run, s3i learns its digit tables, practises small problems on a general digit
machine, builds a library of programs from what it solves, and composes the round from that library. INTEGRATION.md
§4 lists what a person still chose. A judge will ask which of those choices the result depends on. So each part was
removed in turn, and the model was trained again on the contest's hardware and budget: one H100, 3,600 s, the
official manifest, on seed 74 (the official seed) and on seed 1. The unchanged file, run beside the arms, replays
s3i's own runs on both seeds exactly.

**What the result needs.** Without each of these parts, the run never finds the round, so it never certifies. Each
failed on both seeds, at the same level of the climb and missing the same piece:
1. **SEL.S**, the instruction that keeps or discards a result on the borrow. Without it, the conditional subtraction
   ("x mod N for x below 2N") is never found, and every later problem builds on that.
2. **BRG.X**, the instruction that brings X's next digit in. Without it, neither the long-division step nor the
   reduction is found.
3. **The two practice problems that define long division's step** (red.step's chain and step). Without them, no
   reduction is found.
4. **The multiply's row posed as "add x times a digit into a running sum"** (mac with A as its addend). Posed instead
   as "x times a digit, plus N", it is solved, but the multiply is never found.
5. **The register twins**, each problem also posed with X and Y swapped. Without them, no reduction is found. s3i's
   reduction runs the chain on Y, because X still holds the digits to be brought in, and only the chain's twin works
   on Y. Twins also give BPE the second use a merge needs; that is how two other arms built a reduction on A
   instead.
6. **Routines**: each solved problem becomes one callable instruction. Without them, no multiply is found: its loop
   repeats the accumulating row, and that row fits in a loop body only as one instruction.
7. **Occam's tie-break toward the program that runs fewer steps.** Without it, the comparison is kept in a slower
   spelling. The search's routines-first pass then makes the conditional subtraction "x if x < N, else 0". That
   passes its own practice problem but not the adder's check, so no adder is found and nothing after it.

**What it needs on one seed of two: the chain problem alone** (modB, red.step's ladder). Without it, s3i's reduction
can no longer call the chain on Y as one instruction, so the reduction has to be a loop of the long-division step.
That works only if the step keeps its result in Y. The step's two spellings, one on X and one on Y, cost the same,
and the tie goes to whichever the search lists first. On seed 1 that was Y: the reduction came a level early and the
run certified. On seed 74 it was X, and no reduction was found. With the chain problem, the reduction does not
depend on that tie.

**What only makes it faster.** Without each of these, the run still certifies on both seeds inside the official
budget, the freeze coming by 1,584 s at the latest:
- the curriculum's order (the multiply practised after its row, the reduction after its chain): +277 s and +321 s
  to the round (seeds 74 and 1);
- not searching a twin again once its original is solved: +364 s and +416 s;
- BPE merges: +282 s and +291 s;
- taking the round's answers from rows with T = 1. The data-derived rule (the batch's smallest T) found the round 82
  steps later on seed 74, where one batch had no T = 1 row, and at the same step on seed 1, where every batch had one.

The seconds are corrected for host speed, which varies by about 20% between H100s (§4); seed 74's +277 s is not, for
want of identical searches to measure the host by.

**What it doesn't need.**
- *The search's routines-first order*: the round comes at the same step on both seeds.
- *The late table lessons, the register-width wrap-around fact among them*: the climb is identical line for line on
  both seeds.
- *The step problem alone*: s3i's reduction never calls the step. On seed 1 the run certified three steps early and
  264 s faster, with fewer problems to search. Seed 74 failed by chance (next paragraph).
- *The multiply's row problem* (mrow; one seed): the round comes a step earlier. The row's programs only supplied a
  merge that s3i's reduction starts with; without it the reduction comes a level later in another form.
- *The round's cost limit above 5*: at 6 nothing changes (one seed), since the round's search stops after cost 5 at
  its state cap anyway. At 4 there is no round: every round program found costs 5.

**A weak check, found on the way.** Without the step problem, seed 74's reduction search ran two units earlier, on a
different pool of rows. Its check accepted a repeated subtraction that is right only while the quotient is small: up
to 600 at the check's largest size (3 digits), wrong on every full-size product. Nothing can be built on it, so the
run could not certify. s3i's check lets that program through when none of its 60 three-digit probes needs more than
600 subtractions: about 3% of the time on H1's rows. The same chance applies to s3i's own reduction search on any
seed; only that one run in this study drew it.

**What this says about Bar 2.** The parts the result needs are where INTEGRATION.md §6 said the decomposition lives:
- two task-shaped instructions;
- the practice problems that define long division's step;
- how the multiply's row and the registers are posed;
- two parts of the learning machinery (routines, and Occam's preference for faster programs).

The curriculum's order, the search's schedule, the late table lessons, the row problem and the step problem alone are
not what the result rests on. The order was the main budget concern for s3e. On s3i's faster search engine it costs
about 280-320 s, not the round.

**How firm this is.**
- *Two seeds per main part.* The fourteen main arms got the same verdict on both seeds, and on both every arm solved
  the same problems at the same levels.
- *The five later parts.* Of those, only the chain and the step problems ran on both seeds. The second seed changed
  both verdicts: the chain problem from needed to needed on one seed of two, and the step problem from needed to not
  needed.
- *The kind of failure.* Every failure of the seven needed parts is the same kind. The climb ran out of things to try,
  between 2 and 34 minutes into training: a level found nothing new, and every search for the missing piece had
  covered every program up to its cost limit without one that passes. So more training time cannot rescue these
  arms. Nor could a search one cost level wider: for the six whose failure is the search's reach, the missing piece's
  searches were widened to every program up to cost 6, up to 5.7 billion per search, and none found it (§7).
<!-- /PLAIN -->

## 1. What was removed, and how

s3i learns x^(2^T) mod N in one training run: it learns to read, learns three digit tables from laws, climbs a library
of programs on a general digit machine by practising small problems built from its own training rows, and composes
the round "copy x, multiply, reduce" from that library ([INTEGRATION.md](INTEGRATION.md) §3). INTEGRATION.md §4 lists
what a person still chose. Each arm below removes one of those parts and keeps everything else as it is.

- **By a constant.** When s3i already has a constant for the part, the arm runs s3i itself with that constant
  rewritten by the shared runner's `--set` (the `set` column).
- **By a variant file.** Otherwise the arm runs a copy of s3i made by
  [`ablation_s3/make_variants.py`](ablation_s3/make_variants.py): one comment line at the top, then a few text edits,
  each asserted to occur exactly once, and nothing else. `make_variants.py --diff ARM` prints an arm's edits. The md5s
  are in [`variants/MD5SUMS.json`](ablation_s3/variants/MD5SUMS.json).

[`check_variants.py`](ablation_s3/check_variants.py) checked every arm locally before any run (29 s, 474 MB). It loads
each arm as the runner does and checks the change as the climb sees it: the practice problems it will pose, the
instructions its searches may use, what the round's search composes, and the answer filter. Then it runs a few climb
units with toy budgets. For the two arms that rewrite a practice problem's target (`no_modB`, `no_mrow`), it checks
that the new target equals s3i's on the same 64 probes.

| arm | the hand-made part (INTEGRATION.md §4) | how it is removed | md5 of what ran |
|---|---|---|---|
| control | none: s3i as certified | the file, untouched | `0e813144` |
| prereq_off | the curriculum's order: the multiply is practised only after its row (mac), the reduction only after its chain (modB) | `--set PREREQ=False`: mul and reduce wait only for add, as in s3e | `67150b01` |
| mac_n | mac posed with the running sum as its addend (A + x d), designed so that the multiply's accumulating row is a problem of its own | `--set MAC="N"`: mac is posed as x d + N, libcore's default before variant C | `3ec724a4` |
| no_twins | every practice problem also posed with registers X and Y swapped (lt~, modx~, ..., reduce~) | variant: the climb's task list without twins | `0c3c62fe` |
| no_bpe | BPE: pairs that recur in two solved tasks merge into one new instruction | `--set MAX_MERGES=0` (routines stay) | `ce2dc2bb` |
| no_routines | routines: every solved task's program becomes one callable instruction T.<task> | variant: no routine is made. The round's search, which composes routines in s3i, composes the library's merges instead, and runs whenever the library changes | `0979021d` |
| comp_first_off | search rule: compositions of the routines are searched first | `--set COMP_FIRST=False` | `2b73b7cf` |
| occam_steps_off | search rule: Occam's ties go to the program that executes fewer steps | `--set OCCAM_STEPS=False`: ties go to the first found | `dfc699c2` |
| skip_twins_off | search rule: a twin whose original is solved and whose own search used up its cap is not searched again | `--set SKIP_TWINS=False` | `020b1e0c` |
| no_late | the late table lessons (sub.inverse, sub.identity, add.commute, the mined facts' borrows, the register-width wrap-around fact, sub.zero_minus), which join 100 steps after the digit lessons | `--set LATE_AFTER=10**9`: they never join. The table's checks, which gate the next phase, stay | `6acc6325` |
| no_redstep | the law library's two step definitions (red.step): the chain (modB, "x + q N comes back to x") and the long-division step (step) | variant: neither is posed; reduce waits for add only | `766e6653` |
| no_brg | the instruction BRG.X (bring X's top digit into A) | variant: out of every search's vocabulary; the machine still has it | `304eff00` |
| no_sel | the instruction SEL.S (keep or discard on the borrow) | variant: out of every search's vocabulary | `ec3f4d7d` |
| ans_smallest_t | the round's answers are the rows with T = 1 | variant: the rows with the batch's smallest T (the data-derived rule phase D already uses), whatever that T is | `ca9d80cf` |
| no_step | red.step's step problem alone (the chain stays) | variant: step is not posed; nothing waits for it | `7ee6e0c6` |
| no_modB | red.step's chain problem alone (the step stays) | variant: modB is not posed; step's target is computed by B conditional subtractions (B the digit base) instead of the chain, and step waits for modx | `29543773` |
| no_mrow | mul.ladder's row problem (mrow) | variant: mrow is not posed; mac's target computes the row by repeated addition, and mac waits for add only | `90d6c792` |
| round_cost4 | the round's search space: programs of cost 5 at most | `--set ROUND_COST=4` | `a8b5fc41` |
| round_cost6 | the same | `--set ROUND_COST=6` | `64e6e00d` |

The first fourteen ran on seeds 74 and 1, the last five on seed 74, and `no_step` and `no_modB` also on seed 1.

## 2. Results

Every run is on one H100 with the official manifest (`benchmark/manifests/h100_hard_h1.json`). Seed 74's runs use it
untouched, so a certifying arm's seed-74 run is an official run; seed 1's change only the seed. Every run that found
the round certified (T = 64 on both ladders), and every run that did not has no round program to run. Times are the
evaluator's elapsed seconds at the first logged step after the event, as INTEGRATION.md reports s3i: "round 671 /
1,047 s" is the round found at step 671, 1,047 s into training. In parentheses: the round's time corrected for host
speed, the climb's part divided by the run's host factor (§4). Records: `ablation_s3/records/<arm>_h100_seed<s>.json`;
the numbers and the per-search detail come from [`analyze.py`](ablation_s3/analyze.py).

<!-- RESULTS -->
| arm | seed 74 | seed 1 | never found | verdict |
|---|---|---|---|---|
| control (s3i) | round 671 / 1,047 s; frozen 870 / 1,092 s | round 613 / 1,096 s; frozen 812 / 1,128 s | - | replays s3i's runs exactly (§4) |
| comp_first_off | round 671 / 1,032 s; frozen 870 / 1,090 s | round 613 / 1,029 s; frozen 812 / 1,068 s | - | **not needed** |
| no_late | round 671 / 1,248 s (1,035); frozen 870 / 1,289 s; the climb identical | round 613 / 988 s (1,064); frozen 812 / 1,010 s; the climb identical | - | **not needed** |
| no_step | no round: a size-limited reduction accepted at level 4; out of time at level 6 | round 610 / 798 s (832); frozen 809 / 827 s | - | **not needed**: seed 74 failed by chance (§3) |
| no_mrow | round 670 / 1,209 s; frozen 869 / 1,248 s | not run | - | **not needed** (one seed) |
| round_cost6 | round 671 / 976 s (1,030); frozen 870 / 1,015 s | not run | - | **no effect** (one seed) |
| prereq_off | round 679 / 1,324 s; frozen 878 / 1,372 s | round 621 / 1,342 s (1,417); frozen 820 / 1,378 s | - | speed only: +277 s, +321 s |
| skip_twins_off | round 671 / 1,328 s (1,411); frozen 870 / 1,367 s | round 613 / 1,551 s (1,512); frozen 812 / 1,584 s | - | speed only: +364 s, +416 s |
| no_bpe | round 672 / 1,408 s (1,329); frozen 871 / 1,460 s | round 614 / 1,304 s (1,387); frozen 813 / 1,334 s | - | speed only: +282 s, +291 s |
| ans_smallest_t | round 753 / 1,473 s, on the 8th try; frozen 952 / 1,526 s | round 613 / 1,299 s (1,119); frozen 812 / 1,340 s; the climb identical | - | speed only: +82 steps on seed 74 |
| no_modB | no round: the step came on X; ended after level 6 (step 677, ~1,050 s) | round 611 / 1,570 s; frozen 810 / 1,611 s (the step came on Y) | reduce (seed 74) | **needed on one seed of two**: a tie (§3) |
| round_cost4 | no round: every piece solved; ended after level 6 (step 682, ~990 s) | not run | the round (it costs 5) | the limit cannot go below 5 |
| mac_n | no round: ended after level 5 (step 670, ~1,530 s) | ended after level 5 (step 612, ~1,680 s) | mul | **needed** |
| no_routines | no round: ended after level 5 (step 670, ~710 s) | ended after level 5 (step 612, ~650 s) | mul | **needed** |
| no_twins | no round: ended after level 6 (step 661, ~630 s) | ended after level 6 (step 603, ~640 s) | reduce | **needed** |
| no_redstep | no round: ended after level 6 (step 674, ~600 s) | ended after level 6 (step 616, ~470 s) | reduce | **needed** |
| no_brg | no round: ended after level 6 (step 682, ~1,750 s) | ended after level 6 (step 624, ~2,010 s) | step, reduce | **needed** |
| occam_steps_off | no round: ended after level 2 (step 651, ~190 s) | ended after level 2 (step 593, ~200 s) | add and every problem after it | **needed** |
| no_sel | no round: ended after level 1 (step 648, ~170 s) | ended after level 1 (step 590, ~150 s) | modx and every problem after it | **needed** |
<!-- /RESULTS -->

The failed runs all evaluate the same way: no round program, so nothing to run, and 5/10,002, 20/10,002 and 7/9,999
right on ood_n_t, ood_t and test (s3_official's numbers: what a model without a round gets right by accident). After
its climb ended, each failed run kept searching for the round every 10 steps on the answers it held, 162 to 1,600
times, until the deadline; none found one.

## 3. Arm by arm

The control's climb, for reference (s3i_official's too; all eight of s3i's H100 seeds have the same shape, the round
31 or 32 units after the climb starts):
- **levels 0-2:** lt, then modx = `LDA.X STA.S T.lt SEL.S` (the conditional subtraction), then add;
- **level 3:** modB = `B[T.modx STA.X]` (the chain) and mrow (the row, by the primitives fallback);
- **level 4:** mac = `LDH.Y CLK P[MUL.X INC]` (the accumulating row), step = `LDA.Y BRG.X STA.Y T.modB~`, and
  reduce = `B3.1 P[BRG.X STA.Y T.modB~]` (B3.1 = CLK CLA: clear, then for each place bring a digit in and run the
  chain on Y);
- **level 5:** mul = `CLA W[T.mac SHL.X SHR.Y]`;
- **level 6:** the round `LDA.X STA.Y T.mul STA.X T.reduce`.

**Seed 1.** Every one of the fourteen main arms solved the same problems at the same levels on seed 1 as on seed 74;
their library-building lines differ only where Occam's tie-break picked another spelling of the same cost (the step
on X instead of Y in prereq_off, a multiply ending `INC SHL.X` instead of `SHL.X SHR.Y` in no_twins). So each
mechanism below holds on both seeds; seed-1 specifics are given where they add something. Such a tie decides
no_modB's outcome (below).

**Needed.**
- **no_sel.** modx's searches at levels 0 and 1 (the second with T.lt and the merges) found no program that passes.
  Level 1 solved nothing, so the climb ended. With nothing to keep or discard on the borrow, no
  subtraction-if-it-fits is within reach, and add, the chain, the step and the reduction all wait on modx.
- **occam_steps_off.** lt has two cost-5 spellings, `CLC P[LDA.X SBB.N INC]` and `CLC LDA.X P[SBB.N INC]`. Both give
  the right borrow, but only the second leaves x - N in A. With ties going to the first found, T.lt is the first.
  - modx's routines-first pass composes T.lt, and the cheapest program there that passes is
    `T.lt LDA.X STA.S CLA SEL.S`, which means "x if x < N, else 0".
  - modx's practice problem tests only x < N and x = N (red.identity), so this passes. Because the routines pass found
    a program, the library pass never ran; it offered the right one, `LDA.X STA.S B0.3 SEL.S` (cost 4), through the
    merge B0.3.
  - The adder's check, "(x + N) reduced by modx gives x back", calls the learned modx on the adder's output, which is
    at least N, and this modx returns 0 there. So no adder passes, at any cost: add fails at level 2 and the climb
    ends.

  This is the failure INTEGRATION.md §3 met in s3c, which the tie-break was added to fix. So the tie-break is needed
  together with routines-first, and modx's thin practice problem is part of the story.
- **no_routines.** BPE alone carries the reduction half. The link on A (`STA.S CLC P[SBB.N INC] SEL.S`) and the chain
  over it become merges, and reduce comes at level 4 as `B3.2 P[BRG.X B3.1]`. That expands to
  `CLK CLA P[BRG.X B[STA.S CLC P[SBB.N INC] SEL.S]]`, the hand-written reduction LIBRARY.md's MDL run found, with a
  CLK in front. The multiply does not come. mac is solved, but without routines its program stays three items after
  merging (`LDH.Y CLK B4.5`), so a loop of it with two shifts needs a body of 5 against the search's limit of 3. mul's
  searches at level 5 reached cost 5 without a program, and the climb ended. This is INVENTING.md's "routines
  necessary" result, now at H1's size. Seed 1's library is identical, line for line.
- **mac_n.** mac is solved at level 4 as `T.mrow STA.X T.add`: the row, then add N. The multiply needs each row added
  into a running sum, which this mac does not provide, as the A/B found (INTEGRATION.md §2.1). mul and mul~ searched
  to cost 5 at level 5 (462 s and 328 s on seed 74) without a program, so the climb ended.
- **no_twins.** The step comes as `LDA.Y BRG.X STA.X T.modB`, the chain on X. In the reduction, X holds the digits
  still to be brought in, so a loop of these steps would overwrite them. reduce's searches at levels 4, 5 and 6
  reached cost 5 without a program, and it stalled. Twins do a second job too: BPE merges a pair only when two tasks
  use it, and a task and its twin are what usually make two. In comp_first_off and no_routines, that is how the link
  on A became a merge and the reduction came without the Y chain. Without twins, neither path exists.
- **no_redstep.** With no chain and no step problem, reduce was searched at levels 3 to 6 (the last two by
  compositions after it stalled), reaching cost 5 each time without a program. The multiply came at level 5 as usual.
  So long division's reduction is not found without the practice problems that define its step.
- **no_brg.** step and reduce were searched at levels 4, 5 and 6, all four stalled, and the climb ended at level 6.
  The multiply came at level 5. Nothing else brings one digit of X into A, so long division's "bring the next digit
  down" is not built from the rest.

**Speed only.**
- **prereq_off.** mul, mul~, reduce and reduce~ are searched at level 3, before their pieces exist (about 38 s each),
  and mul and mul~ again at level 4 (81 s and 70 s). reduce comes at level 4 as in the control. After two
  used-up searches, mul stalls: from then on only compositions, when new routines appear. At level 5 that finds
  `CLA W[T.mac~ SHL.Y SHR.X]` from the new T.mac. The round comes 8 steps later on both seeds (679, 621): 277 s later
  on seed 74 (uncorrected) and 321 s on seed 1 (corrected). s3e, the build without the order, missed the budget on the
  engine s3h and s3i made more than twice as fast.
- **skip_twins_off.** step~ and reduce~ are searched again at level 5 (330 s and 392 s for step~, found at cost 5;
  6-7 s for reduce~, found at cost 3). The round's search then has a larger library (43-47 s against 26 s): 364 s
  and 416 s later in all, at the same step.
- **no_bpe.** Without the CLK CLA merge, reduce is not found at level 4. At level 5 it comes as
  `P[MUL.Y SHL.Y] P[T.step]` (seed 1: `P[MUL.X SHL.Y] P[T.step]`): clear Y, then repeat the long-division step (the
  first in-run discovery's form, s3d). One step and 282-291 s later on both seeds (corrected; an earlier version of
  this page said about 545 s, from a host correction that §4 explains and no longer uses).
- **ans_smallest_t.** The training data has T = 1, 2, 4, 8 and 16 (1,000, 1,000 and 33,333 rows each), so nearly
  every batch of 512 has a T = 1 row. In seed 74's first 800 steps, one batch (step 647) had none, and no T = 2 row
  either. Its smallest T was 4, and 19 of its rows joined the pool. No one-round program reproduces a four-round
  answer, so the round's search failed at level 6 and on 7 retries. It accepted the round at step 753 (82 steps after
  the control's), when those rows had left the 512-row pool.
  [`batch_t.py`](ablation_s3/batch_t.py) replays the evaluator's batch order to confirm this. It reproduces the
  control's pool sizes at steps 650 and 700 (43 and 295) and ans_smallest_t's (62 and 314: 19 more). On seed 1 every
  batch from the climb's start (step 582) to step 1,041 has a T = 1 row, so the two rules keep the same answers and
  the climb is the control's, line for line.
  On H1 the T = 1 filter only saves those steps. On data with no T = 1 rows (M4, every T is 8), neither rule gives the
  round's search one-round answers.

**Not needed.**
- **comp_first_off.** Without the compositions-first pass, the library search finds the same pieces, a few as
  merges: modx as `LDA.X STA.S B0.3 SEL.S`, and reduce at level 4 as `B3.2 P[BRG.X B3.1]`, the hand-written
  reduction on A (as in no_routines). The round comes at the same step on both seeds, at 1,032 s and 1,029 s against
  the control's 1,047 s and 1,096 s: no slower. (These are uncorrected: its searches differ from the control's from
  level 1 on, too early for a host measurement.)
- **no_late.** The table passes its checks at the same steps (on seed 74: the switch at 277, cells at 297, the climb
  at 640). On both seeds the climb's log lines equal the control's once the seconds are removed. The late lessons, the
  wrap-around fact among them, add nothing. ANCHOR_WRAP and the other ANCHOR_* constants in s3i are never read: the
  wrap fact is one of the late lessons, which `LATE_AFTER` gates.
- **no_mrow** (seed 74). Without the row problem, mac waits only for add, so it is practised at level 3 and solved
  there with s3i's own program, `LDH.Y CLK P[MUL.X INC]`; the multiply follows at level 4, a level early. The
  reduction comes a level late. s3i's reduction starts with the merge CLK CLA (B3.1), which BPE makes from the two
  row programs (`LDH.Y CLK CLA P[MUL.X INC]` and its twin). Without them there is no such merge, so reduce's searches
  at level 4 reach cost 5 without a program. At level 5 it comes as `LDA.N STA.Y P[T.step]`: set Y to N, which is 0
  modulo N, then run the long-division step once for each place. The round comes at step 670, a step before the
  control's. Its 1,209 s cannot be host-corrected (its searches differ from the control's from level 3 on, leaving
  29 s of identical ones, under the 30 s the correction needs), and the 162 s difference is within the host spread.
- **round_cost6 and round_cost4** (seed 74). At ROUND_COST = 6 the round comes at the same step with the same
  programs, 1,031 s corrected. The round's search also stores at most 4M states per level (ROUND_STATES), and its
  cost-5 level holds 17-28M, so the search stops after cost 5 at either limit; only its stop reason changes. At
  ROUND_COST = 4 every piece is solved as in the control, but the round's search exhausts cost 4 at level 6 and on
  618 retries. The round s3i finds, `LDA.X STA.Y T.mul STA.X T.reduce`, has five parts (copy x into Y, multiply,
  store the product, reduce), and every round program stage 3 has found costs 5. So the limit sits at the round's
  size, and the state cap, not the cost limit, is what bounds the round's search.

**The chain and the step, one at a time.**
- **no_modB: needed on one seed of two.** Without the chain problem, step waits only for modx and is solved at level
  2, with the chain spelled out: `LDA.Y BRG.X B[STA.X T.modx]` (bring the digit in, then reduce on X) or
  `LDA.Y BRG.X B[STA.Y T.modx~]` (the same on Y). The two cost the same and execute the same number of steps, so
  Occam's tie goes to whichever the search lists first.
  - *Seed 74* listed X first. s3i's reduction calls the chain on Y as one instruction (T.modB~) inside its loop;
    spelled out, the loop body costs 5 against the search's limit of 3. And a loop of the step on X overwrites the
    digits X still holds, as in no_twins. reduce's searches reached cost 5 at levels 3 to 6 without a program
    (stalling at 5), and the climb ended after level 6.
  - *Seed 1* listed Y first. Then reduce came at level 3, a level before the control's, as
    `P[SHL.Y] MUL.Y P[T.step]`: clear Y, then run the step once for each place. The round came at step 611 and the
    run certified.
  - So the chain problem is what makes s3i's reduction independent of that tie: with it, the reduction uses the chain
    on Y directly, whichever register the step got (prereq_off's step on seed 1 went to X, and it certified).
  - no_modB's seconds are not comparable. Its searches that use the most GPU ran 2-5x slower than the control's
    identical ones on both seeds (the row problem's fallback: 134 s and 267 s against 47 s and 48 s), for a reason
    not found here, so its 1,570 s to the round on seed 1 is left uncorrected.
- **no_step: not needed.** s3i's reduction does not call T.step. On seed 1 the run certified three steps before the
  control and 264 s sooner (corrected): two fewer searches at level 4. Seed 74 failed, for a reason that has nothing
  to do with the step problem:
  - With no step problem, the level-4 searches are mac, mac~, reduce and reduce~ instead of mac, mac~, step, step~,
    reduce and reduce~. So reduce is searched two units earlier, on a pool two batches different.
  - Its routines pass then accepted `B[P[T.modB~ T.modB]]` at cost 4: ten times, for each place, the chain on Y and
    then the chain on X. That is repeated subtraction: 10 x 2W x 10 conditional subtractions of N from X.
  - Every search that can call T.reduce then ran slowly. The round's search took 278 s against s3i's 18 s, and the
    multiply's search at level 5 stopped at the 900 s per-search limit (mul~ came, at 831 s).
  - The round's search at levels 5 and 6 then exhausted cost 5 with the multiply in hand. No round can use a
    reduction that fails at full size, and reduce, being solved, is never searched again. The training budget ran
    out at level 6.

**A weak check.** [`reduce_check.py`](ablation_s3/reduce_check.py) runs `B[P[T.modB~ T.modB]]` with exact cells:
- It is right on every probe of 1 and 2 digits, right up to a quotient of 599 at 3 digits, and wrong on every
  full-size product.
- s3i's reduction check tests a program on 60 probes at each of 1, 2 and 3 digits. It lets this one through when
  none of the 60 three-digit probes has a quotient above 600. 5.7% of H1's training rows do, so a pool lets it
  through with chance 0.943^60 ≈ 3%.
- The search never offers the cheaper `B[B[T.modB]]` (1,000 subtractions, right at all three sizes). It ends in the
  same machine state on the search's own 1- and 2-digit probes as `B[T.modB]` (100 subtractions), which the check
  rejects, and the search tests one program per state. `B[P[T.modB~ T.modB]]` reaches a new state because it also
  changes Y.

The same 3% applies to s3i's own reduction search on any seed, since it meets the same program at cost 4 before
finding the real reduction at cost 5. A check probing larger quotients would close it.

## 4. How hard each arm was tried

- **Runs.** Every run is one H100 at the official budget:
  - the fourteen main arms on seeds 74 and 1 (runs `abl-s3-stage1-s74` and `abl-s3-stage1-s1`: 10.5 and 10.4
    H100-hours, $42.36 and $41.91 at list price);
  - the five later arms on seed 74 (`abl-s3-more-s74`: 3.8 H100-hours, $15.54);
  - no_step and no_modB on seed 1 (`abl-s3-more-s1`: 0.94 H100-hours, about $3.80).

  The runner's `--mode cost` gives each bill. Modal was at $1,760 of the workspace's $2,000 limit when the first
  runs started. The user raised the limit before the rest ($2,006 billed when they started).
- **The control is exact.** `tools/modal_runner.py --mode compare integration/records/s3i_official.json
  integration/ablation_s3/records/control_h100_seed74.json` finds every field equal: the training trace (201 points),
  the loss at every logged step, every rung and split, the freeze at 870. Only the climb log's seconds differ. Seed 1
  against `integration/records/s3i_h100_seed1.json` likewise: every field equal, the freeze at 812.
- **Host speed.** The climb runs mostly on the host CPU (INTEGRATION.md §3), and H100 hosts differ. no_late's climb is
  identical to the control's, yet on seed 74 each of its levels took 10-35% longer, and its round came 201 s later.
  So [`analyze.py`](ablation_s3/analyze.py) measures each run's host. It sums the seconds of the searches identical
  to the control's of the same seed (same level, task, vocabulary and state counts) and divides by the control's
  sum. Then it divides the climb's part of the round time by that factor. The factors were 0.93-1.22. no_late's
  corrected rounds are 1,035 s and 1,064 s against the control's 1,047 s and 1,096 s.
  - It leaves out one search: the row problem's primitives-alone fallback at level 3. That search streams 300-400M
    single instructions, which keeps the GPU busy rather than the host, and its time ranged from 0.5x to 5.5x the
    control's between runs whose other identical searches agreed within 30%. An earlier version of this page counted
    it, which put no_bpe's host 12-21% faster than the control's and BPE's cost at about 545 s. But no_bpe's levels 0-2
    took the control's seconds exactly; without the fallback its factors are 0.94 and 1.06, and the cost is 282-291 s.
  - A run gets no correction when under 30 s of its searches match the control's: comp_first_off (its searches differ
    from level 1 on), prereq_off on seed 74, ans_smallest_t on seed 74, no_mrow and no_modB (from level 2 or 3 on).
- **No needed part's failure ran out of time.** Each of the fourteen failed runs of the seven needed parts ended its
  climb by s3i's own rule, a level that solves and merges nothing, between 150 s and 2,010 s into training. It then
  searched for the round 162 to 1,600 more times until the deadline.
  - Every search for a missing piece stopped at its state cap: 5.8M stored states per level, the next level checked
    for goals as it streams. That covers every program up to cost 5 in the vocabulary of the moment, with loop bodies
    up to cost 3.
  - None stopped at the 900 s per-search limit; the longest took 494 s.
  - So a 3x training budget would end the same way, and the relaxed reruns the brief planned to separate discovery
    from budget were not needed.
  - no_step's run did run out of time, but its round was already out of reach (§3).
  - What might change an outcome is a wider search. §7 widened the six arms whose failure is a search's reach to
    every program up to cost 6, and none found its piece. "Needed" here means needed within s3i's search reach and
    one cost level beyond it.
- **Two seeds.** s3i's eight H100 seeds climb in the same shape (mac, step and reduce at level 4, mul at level 5, the
  round 31 or 32 units in). Seed 1 repeated seed 74 for every main arm: the same verdicts, the same problems solved
  at the same levels, and the speed-only costs within 52 s of each other. The second seed did change both later arms
  it was run for: no_modB turned on a tie, and no_step's seed-74 failure was a 3% chance.

## 5. What was not run, and why

- **More seeds.** Two seeds settle the main arms (every verdict and every mechanism agree). no_mrow, round_cost4 and
  round_cost6 ran on seed 74 only: their mechanisms (a merge that no longer forms; the round's size and its state
  cap) do not depend on the seed. no_modB's outcome turns on which of two equal programs the search lists first, so
  more seeds would only estimate how often that is Y.
- **Not built:**
  - *Dropping lt or modx.* The adder's practice problem checks its answer with both: x + N must reduce back to x, and
    must not be below N. Removing either means redesigning add's check, not leaving one part out.
  - *Dropping add.* Every later problem waits for it, and their targets are computed with it.
  - *Dropping mul or reduce.* The round's search composes straight-line programs of at most 5 items, with no loops.
    Without a multiply routine or a reduction routine there is no round, by construction.
  - *The reader and writer, the countdown, the cells' input shapes, the sizes and the search's caps.* None is a part
    one can leave out; changing them would be a different study. §7 widens the caps for the failed arms only.
- **The 3x-budget reruns** (B200s, as the brief planned): not needed (§4).

## 6. Reproduce

From `cpu_first/`, with the sotaku venv's Python (`PY=~/github/sotaku/venv/bin/python`). The local commands take
seconds and stay under 1 GB; the runs go through the shared runner (tools/README.md).

Build the variants (asserts s3i's md5; refuses to change a variant that differs on disk) and check every arm:

```bash
$PY integration/ablation_s3/make_variants.py
$PY integration/ablation_s3/check_variants.py
$PY integration/ablation_s3/make_variants.py --diff no_twins        # one arm's edits
```

Write the plans, check the bill, launch, fetch. The first stage launched nothing new at $1,900 or more of the
workspace's $2,000 limit (`spend.py 1900`); after the user raised it to $5,000, the guard is `spend.py 4750`.

```bash
MAIN=control,prereq_off,mac_n,no_twins,no_bpe,no_routines,comp_first_off,occam_steps_off,skip_twins_off,no_late,no_redstep,no_brg,no_sel,ans_smallest_t
$PY integration/ablation_s3/make_plan.py stage1_s74 --seeds 74 --hardware h100 --arms $MAIN
$PY integration/ablation_s3/make_plan.py stage1_s1 --seeds 1 --hardware h100 --arms $MAIN
$PY integration/ablation_s3/make_plan.py more_s74 --seeds 74 --hardware h100 \
  --arms no_step,no_modB,no_mrow,round_cost4,round_cost6
$PY integration/ablation_s3/make_plan.py more_s1 --seeds 1 --hardware h100 --arms no_step,no_modB
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach tools/modal_runner.py --mode plan --plan integration/ablation_s3/plans/stage1_s74.json \
  --out integration/ablation_s3/records --cap 14 --run abl-s3-stage1-s74
#   likewise stage1_s1 (--cap 14 --run abl-s3-stage1-s1), more_s74 (--cap 5 --run abl-s3-more-s74),
#   more_s1 (--cap 2 --run abl-s3-more-s1)
$PY tools/modal_runner.py --mode peek --run abl-s3-stage1-s74
$PY tools/modal_runner.py --mode fetch --run abl-s3-stage1-s74 --wait
$PY tools/modal_runner.py --mode cost --run abl-s3-stage1-s74
```

Read the records:

```bash
$PY integration/ablation_s3/analyze.py              # one line per run (host factor, corrected round, same climb)
$PY integration/ablation_s3/analyze.py --levels     # every search of every climb, per level
$PY integration/ablation_s3/analyze.py --table      # one markdown row per run
$PY tools/modal_runner.py --mode compare integration/records/s3i_official.json \
  integration/ablation_s3/records/control_h100_seed74.json
$PY tools/modal_runner.py --mode compare integration/records/s3i_h100_seed1.json \
  integration/ablation_s3/records/control_h100_seed1.json
$PY integration/ablation_s3/batch_t.py SEED=74 FROM=641 TO=800    # the batches without a T = 1 row
$PY integration/ablation_s3/batch_t.py SEED=1 FROM=1 TO=1400
$PY integration/ablation_s3/reduce_check.py         # no_step's reduction, and how often s3i's check accepts it
```

A run that Modal stops part way is relaunched with `make_plan.py ... --only-missing` and `--mode plan --force` under a
new run tag (the evaluator keeps no checkpoints, so a job starts again from the beginning).

## 7. A wider search for the needed parts

**The result.** For each of the six needed parts whose failure is a search's reach, the missing piece's search was
widened to cover every program up to cost 6, one level past s3i's reach. None found the piece. Together these searches
streamed 0.4 to 5.7 billion cost-6 programs each, and every climb ended where its arm's did. So within one more level
of the search, these parts are still needed. The seventh needed part, Occam's tie-break, is not a search's reach (below).

The seven needed parts fail on both seeds because the climb runs out of things to try within s3i's search reach. This
asks whether a wider search finds the missing piece anyway. Each failed search stopped at the state cap (5.8M stored
states per level), which streams level 5 and so covers every program up to cost 5. The cost limit (7) and the
per-search time limit (900 s) were not what stopped them.

**What was widened.** Each run repeats a failed arm on seed 74 with one change on top of the arm's own. The missing
piece's library searches (merges, routines and primitives together) may store up to 60M states per level (130M and
110M for the multiply), so level 5 is stored and level 6 streamed: every program up to cost 6. These searches also
have no time limit of their own. Everything else, including every other task's searches and the stall rule, is the
arm's. Two run settings change, neither inside the climb:
- *The training budget* is relaxed to 21,600 s in case the searches are slow; none needed more than 45 minutes.
- *Training stops at 1,500 steps* (`--set MAX_STEPS=1500`; s3i's is 20,000, and nothing else reads it). The climb ends
  by about step 700, and a round found then is frozen 200 steps later, so the cap only cuts the round's retries after
  a climb that found nothing. At the relaxed budget those retries would have run for hours. A first launch without the
  cap was stopped after 55 minutes for that reason; it wrote no records.

Widening only the missing piece keeps the rest of the climb as it was. Widening every search would also send the
twins' failing searches (step~ and reduce~ at level 4, in every arm) to cost 6. A stored state takes 655 bytes
(INTEGRATION.md §2.1): 60M take 39 GB, half an H100. The multiply's searches hold 97-117M states at cost 5, so those
two ran on H200s (141 GB), which replay H100 runs exactly. `check_variants.py` checks that the widened task's library
search gets the larger cap and no time limit, and that the arm's other tasks do not.

| run | widened | its library searches with the wider cap | cost-6 programs streamed | found | md5 of what ran |
|---|---|---|---|---|---|
| no_sel_wide (H100) | modx, modx~ | levels 0 and 1, four searches | 383M-846M each | nothing; the climb ended after level 1 | `243908fc` |
| no_twins_wide (H100) | reduce | levels 4 and 5, then stalled | 644M, 864M | nothing; ended after level 6 | `02fe4f76` |
| no_redstep_wide (H100) | reduce | levels 3 and 4, then stalled | 1.83B, 2.28B | nothing; ended after level 6 | `bb0e2370` |
| no_brg_wide (H100) | step | levels 4 and 5, then stalled | 1.56B, 1.86B | nothing; ended after level 6 | `a8dab894` |
| mac_n_wide (H200, 110M) | mul | level 5 | 4.51B | nothing; ended after level 5 | `803cce25` (with `MAC="N"`) |
| no_routines_wide130 (H200, 130M) | mul | level 5 | 5.69B | nothing; ended after level 5 | `676724cf` |

- *The stall rule* is the arm's: after two searches that used up their caps, a task is searched by compositions only,
  at s3i's cap. So the reduction (no_twins, no_redstep) and the step (no_brg) each got two cost-6 searches, at the
  levels where the original arms searched them first. modx and the multiply got cost-6 searches at every level that
  searched them before the climb ended.
- *no_routines at 110M*: its first wide run stopped after cost 5, because the multiply's cost-5 level held 116.9M
  distinct states, over the cap (`no_routines_wide_h200_r21600_seed74`, md5 `fe9848e4`). The 130M run is the one
  that covered cost 6.
- *What cost 6 cannot settle*: the program no_routines would need, mac's three items and two shifts in a loop, costs 7
  with a loop body of 5. Covering cost 7 would stream about 50 times more programs than these searches did.

**Not widened.** *occam_steps_off* is not a search's reach. The adder's check calls the learned modx, which returns 0
for every value at or above N, so no adder passes at any cost.

The variants are built by `make_variants.py` (`WIDE_ARMS`), and the runs cost about $42 in all, the stopped first
launch's $19.60 included. To repeat them:

```bash
$PY integration/ablation_s3/make_plan.py wide2_s74 --seeds 74 --hardware h100 --budget 21600 --set MAX_STEPS=1500 \
  --arms no_sel_wide,no_twins_wide,no_redstep_wide,no_brg_wide
$PY integration/ablation_s3/make_plan.py wide2_mul_s74 --seeds 74 --hardware h200 --budget 21600 --set MAX_STEPS=1500 \
  --arms mac_n_wide,no_routines_wide
$PY integration/ablation_s3/make_plan.py wide3_mul_s74 --seeds 74 --hardware h200 --budget 21600 --set MAX_STEPS=1500 \
  --arms no_routines_wide130
$PY -m modal run --detach tools/modal_runner.py --mode plan --plan integration/ablation_s3/plans/wide2_s74.json \
  --out integration/ablation_s3/records --cap 4 --run abl-s3-wide2-s74   # likewise wide2_mul_s74, wide3_mul_s74
```
