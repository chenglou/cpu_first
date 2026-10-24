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
removed in turn, and the model was trained again on the contest's hardware and budget (one H100, 3,600 s, the
untouched official manifest, seed 74, the official seed). The unchanged file, run beside the arms, replays s3i's
official run exactly.

**What the result needs.** Without each of these parts, the run never finds the round, so it never certifies (1 seed
each):
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

**What only makes it faster.** Without each of these, the run still certifies inside the official budget, the freeze
coming by 1,526 s at the latest:
- the curriculum's order (the multiply practised after its row, the reduction after its chain): +270 s to the round;
- not searching a twin again once its original is solved: +360 s;
- BPE merges: +550 s;
- taking the round's answers from rows with T = 1. The data-derived rule (the batch's smallest T) also works, 82 steps
  later.

The seconds are corrected for host speed, which varies by about 20% between H100s (§4).

**What it doesn't need.**
- *The search's routines-first order*: the same pieces come about as fast, a few as BPE merges instead of routines.
- *The late table lessons, the register-width wrap-around fact among them*: the tables pass every check at the same
  steps without them, and the climb is identical line for line.

**What this says about Bar 2.** The parts the result needs are where INTEGRATION.md §6 said the decomposition lives:
- two task-shaped instructions;
- the practice problems that define long division's step;
- how the multiply's row and the registers are posed;
- two parts of the learning machinery (routines, and Occam's preference for faster programs).

The curriculum's order, the search's schedule and the late table lessons are not what the result rests on. The
order was the main budget concern for s3e. On s3i's faster search engine it costs about 270 s, not the round.

**How firm this is.** It is one seed per arm, the spend approved so far; seed 1 and five more arms are built and wait
(§5). Every failure is the same kind. The climb ran out of things to try, between 3 and 30 minutes into training. A
level found nothing new, and every search for the missing piece had covered every program up to its cost limit
without one that passes. So more training time cannot rescue these arms. A wider search might; that was not tried.
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

Built and checked but not run yet (§5): `no_step` (`7ee6e0c6`), `no_modB` (`29543773`), `no_mrow` (`90d6c792`),
`round_cost4` and `round_cost6` (`--set ROUND_COST=4` / `6`).

## 2. Results

Every run is seed 74 on the untouched official manifest (`benchmark/manifests/h100_hard_h1.json`) on one H100: the
contest's own evaluation. A certifying arm's run is therefore an official run. Times are the evaluator's elapsed
seconds at the first logged step after the event, as INTEGRATION.md reports s3i. "Corrected" divides the climb's part
of that time by the run's host speed (§4). Records: `ablation_s3/records/<arm>_h100_seed74.json`; this table and the
per-search detail come from [`analyze.py`](ablation_s3/analyze.py).

<!-- RESULTS -->
| arm | certified (T = 64 on both ladders) | round found: step, s (corrected) | frozen: step, s | climb | never found | verdict |
|---|---|---|---|---|---|---|
| control (s3i) | yes | 671, 1,047 | 870, 1,092 | round at level 6 | - | replays s3i_official exactly (§4) |
| comp_first_off | yes | 671, 1,032 (906) | 870, 1,090 | round at level 6 | - | **not needed** |
| no_late | yes | 671, 1,248 (1,035) | 870, 1,289 | identical to the control's | - | **not needed** |
| prereq_off | yes | 679, 1,324 (1,317) | 878, 1,372 | round at level 6 | - | speed only: +270 s |
| skip_twins_off | yes | 671, 1,328 (1,411) | 870, 1,367 | round at level 6 | - | speed only: +364 s |
| no_bpe | yes | 672, 1,408 (1,593) | 871, 1,460 | round at level 6 | - | speed only: +546 s |
| ans_smallest_t | yes | 753, 1,473 | 952, 1,526 | round at level 6, accepted on the 8th try | - | speed only: +82 steps |
| mac_n | no | - | - | ended after level 5 (step 670, ~1,530 s) | mul | **needed** |
| no_routines | no | - | - | ended after level 5 (step 670, ~710 s) | mul | **needed** |
| no_twins | no | - | - | ended after level 6 (step 661, ~630 s) | reduce | **needed** |
| no_redstep | no | - | - | ended after level 6 (step 674, ~600 s) | reduce | **needed** |
| no_brg | no | - | - | ended after level 6 (step 682, ~1,750 s) | step, reduce | **needed** |
| occam_steps_off | no | - | - | ended after level 2 (step 651, ~190 s) | add and every problem after it | **needed** |
| no_sel | no | - | - | ended after level 1 (step 648, ~170 s) | modx and every problem after it | **needed** |
<!-- /RESULTS -->

The failed runs all evaluate the same way: no round program, so nothing to run, and 5/10,002, 20/10,002 and 7/9,999
right on ood_n_t, ood_t and test (s3_official's numbers: what a model without a round gets right by accident). After
its climb ended, each failed run kept searching for the round every 10 steps on the answers it held, 167 to 1,502
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
  - The adder's check, "(x + N) reduced by modx gives x back", needs a real reduction for values between N and 2N, so
    add fails at level 2 and the climb ends.

  This is the failure INTEGRATION.md §3 met in s3c, which the tie-break was added to fix. So the tie-break is needed
  together with routines-first, and modx's thin practice problem is part of the story.
- **no_routines.** BPE alone carries the reduction half. The link on A (`STA.S CLC P[SBB.N INC] SEL.S`) and the chain
  over it become merges, and reduce comes at level 4 as `B3.2 P[BRG.X B3.1]`. That expands to
  `CLK CLA P[BRG.X B[STA.S CLC P[SBB.N INC] SEL.S]]`, the hand-written reduction LIBRARY.md's MDL run found, with a
  CLK in front. The multiply does not come. mac is solved, but without routines its program stays three items after
  merging (`LDH.Y CLK B4.5`), so a loop of it with two shifts needs a body of 5 against the search's limit of 3. mul's
  searches at level 5 reached cost 5 without a program, and the climb ended. This is INVENTING.md's "routines
  necessary" result, now at H1's size.
- **mac_n.** mac is solved at level 4 as `T.mrow STA.X T.add`: the row, then add N. The multiply needs each row added
  into a running sum, which this mac does not provide, as the A/B found (INTEGRATION.md §2.1). mul and mul~ searched
  to cost 5 at level 5 (462 s and 328 s) without a program, so the climb ended.
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
  `CLA W[T.mac~ SHL.Y SHR.X]` from the new T.mac. The round comes 8 steps later (679) and 270 s later, corrected.
  s3e, the build without the order, missed the budget on the engine s3h and s3i made more than twice as fast.
- **skip_twins_off.** step~ and reduce~ are searched again at level 5 (330 s, finding step~ at cost 5; 6 s, finding
  reduce~ at cost 3). The round's search then has a larger library (47 s against 26 s): 364 s later in all.
- **no_bpe.** Without the CLK CLA merge, reduce is not found at level 4. At level 5 it comes as
  `P[MUL.Y SHL.Y] P[T.step]`: clear Y, then repeat the long-division step (the first in-run discovery's form, s3d).
  546 s later.
- **ans_smallest_t.** The training data has T = 1, 2, 4, 8 and 16 (1,000, 1,000 and 33,333 rows each), so nearly
  every batch of 512 has a T = 1 row. In seed 74's first 800 steps, one batch (step 647) had none, and no T = 2 row
  either. Its smallest T was 4, and 19 of its rows joined the pool. No one-round program reproduces a four-round
  answer, so the round's search failed at level 6 and on 7 retries. It accepted the round at step 753 (82 steps after
  the control's), when those rows had left the 512-row pool.
  [`batch_t.py`](ablation_s3/batch_t.py) replays the evaluator's batch order to confirm this. It reproduces the
  control's pool sizes at steps 650 and 700 (43 and 295) and ans_smallest_t's (62 and 314: 19 more).
  On H1 the T = 1 filter only saves those steps. On data with no T = 1 rows (M4, every T is 8), neither rule gives the
  round's search one-round answers.

**Not needed.**
- **comp_first_off.** Without the compositions-first pass, the library search finds the same pieces, a few as
  merges: modx as `LDA.X STA.S B0.3 SEL.S`, and reduce at level 4 as `B3.2 P[BRG.X B3.1]`, the hand-written
  reduction on A (as in no_routines). The round comes at the same step, 671. Levels 4 and 5 trade time (259 s and
  413 s against 500 s and 308 s), and the corrected time is 906 s against 1,047 s.
- **no_late.** The table passes its checks at the same steps (the switch at 277, cells at 297, the climb at 640). The
  climb's 31 log lines equal the control's once the seconds are removed. The late lessons, the wrap-around fact among
  them, add nothing on this seed. ANCHOR_WRAP and the other ANCHOR_* constants in s3i are never read: the wrap fact
  is one of the late lessons, which `LATE_AFTER` gates.

## 4. How hard each arm was tried

- **One run per arm**, seed 74, the untouched official manifest, one H100: 14 runs, 10.5 H100-hours, about $42 at
  list price (run `abl-s3-stage1-s74`; the runner's `--mode cost` gives the bill). That is what the spend approved
  so far covers. Modal was at $1,760 of the workspace's $2,000 limit when the runs started and $1,863 when they ended.
- **The control is exact.** `tools/modal_runner.py --mode compare integration/records/s3i_official.json
  integration/ablation_s3/records/control_h100_seed74.json` finds every field equal: the training trace (201 points),
  the loss at every logged step, every rung and split, the freeze at 870. Only the climb log's seconds differ.
- **Host speed.** The climb runs mostly on the host CPU (INTEGRATION.md §3), and H100 hosts differ. no_late's climb is
  identical to the control's, yet each of its levels took 10-35% longer, and its round came 201 s later. So
  [`analyze.py`](ablation_s3/analyze.py) measures each run's host. It sums the seconds of the searches identical to the
  control's (same level, task, vocabulary and state counts) and divides by the control's sum. Then it divides the
  climb's part of the round time by that factor. The factors were 0.88-1.22. no_late's corrected round is 1,035 s
  against the control's 1,047 s.
  ans_smallest_t gets no correction: its practice problems differ from level 3 on (its earlier round search shifts
  the pool by a step), leaving under 30 s of identical searches.
- **No failure ran out of time.** Each failed climb ended by s3i's own rule, a level that solves and merges nothing,
  between 170 s and 1,750 s into training. It then searched for the round 167 to 1,502 more times until the deadline.
  - Every search for a missing piece stopped at its state cap: 5.8M stored states per level, the next level checked
    for goals as it streams. That covers every program up to cost 5 in the vocabulary of the moment, with loop bodies
    up to cost 3.
  - None stopped at the 900 s per-search limit; the longest took 462 s.
  - So a 3x training budget would end the same way, and the relaxed reruns the brief planned to separate discovery
    from budget were not needed.
  - What might change an outcome is a wider search (larger caps, longer bodies or programs): not tried. "Needed"
    here means needed within s3i's search reach.
- **One seed.** s3i's eight H100 seeds climb in the same shape (mac, step and reduce at level 4, mul at level 5, the
  round 31 or 32 units in), so one seed already says much. But a second seed per arm is the obvious next check (§5).

## 5. What was not run, and why

- **Built and checked, waiting for the spend decision:**
  - seed 1 of the 14 arms (about $42);
  - `no_step` (only the step problem dropped; s3i's reduction uses the chain, not T.step);
  - `no_modB` (only the chain dropped; step's target computed by ten conditional subtractions instead);
  - `no_mrow` (the row problem dropped; mac's target computed by repeated addition);
  - `round_cost4` and `round_cost6` (the round's cost limit). Every round found in stage 3 costs 5, so cost 4 is
    expected to find none.
- **Not built:**
  - *Dropping lt or modx.* The adder's practice problem checks its answer with both: x + N must reduce back to x, and
    must not be below N. Removing either means redesigning add's check, not leaving one part out.
  - *Dropping add.* Every later problem waits for it, and their targets are computed with it.
  - *Dropping mul or reduce.* The round's search composes straight-line programs of at most 5 items, with no loops.
    Without a multiply routine or a reduction routine there is no round, by construction.
  - *The reader and writer, the countdown, the cells' input shapes, the sizes and the search's caps.* None is a part
    one can leave out; changing them would be a different study.
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

Write the plan, check the bill (no launch at $1,900 or more of the $2,000 limit), launch, fetch:

```bash
$PY integration/ablation_s3/make_plan.py stage1_s74 --seeds 74 --hardware h100 \
  --arms control,prereq_off,mac_n,no_twins,no_bpe,no_routines,comp_first_off,occam_steps_off,skip_twins_off,no_late,no_redstep,no_brg,no_sel,ans_smallest_t
$PY integration/ablation_s3/spend.py
$PY -m modal run --detach tools/modal_runner.py --mode plan --plan integration/ablation_s3/plans/stage1_s74.json \
  --out integration/ablation_s3/records --cap 14 --run abl-s3-stage1-s74
$PY tools/modal_runner.py --mode peek --run abl-s3-stage1-s74
$PY tools/modal_runner.py --mode fetch --run abl-s3-stage1-s74 --wait
```

Read the records:

```bash
$PY integration/ablation_s3/analyze.py              # one line per run (host factor, corrected round time)
$PY integration/ablation_s3/analyze.py --levels     # every search of every climb, per level
$PY integration/ablation_s3/analyze.py --table      # the markdown table
$PY tools/modal_runner.py --mode compare integration/records/s3i_official.json \
  integration/ablation_s3/records/control_h100_seed74.json
$PY integration/ablation_s3/batch_t.py SEED=74 FROM=641 TO=800    # the batches without a T = 1 row
```

The next stage, when the spend allows: the same with `--seeds 1` (plan `stage1_s1`, run `abl-s3-stage1-s1`), and the
five built arms with `--arms no_step,no_modB,no_mrow,round_cost4,round_cost6 --seeds 74`. A run that Modal stops part
way is relaunched with `make_plan.py ... --only-missing` and `--mode plan --force` under a new run tag (the evaluator
keeps no checkpoints, so a job starts again from the beginning).
