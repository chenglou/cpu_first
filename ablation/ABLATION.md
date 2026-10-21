# What holds the learned machine's training up

## In plain words

**What this is about.** The learned calculator for H1 squares x, divides by N, keeps the remainder, and repeats that
T times. All of its arithmetic is done by one learned table of 200 entries: "this digit minus that digit, minus a
borrow, gives this digit and maybe a new borrow". Getting that table exactly right took a training recipe that grew
piece by piece. Each time a run failed, someone added a lesson, a weight, a switch or a waiting rule to fix it. The
core recipe ([`submission_adder.py`](../learned_remainder/submission_adder.py): long division by repeated subtraction,
with even the additions done by the same table) ended up with about twenty such ingredients. This study asks which of
them still matter.

**How.** Each ingredient was switched off on its own and the recipe retrained from scratch on four seeds. Every run
went through the same unmodified official evaluator the pieces were certified with, on Modal CPUs. Then a bare recipe
was built back up an ingredient at a time, pairs that might stand in for each other were removed together, and two
small exact calculations looked at why. That made 262 training runs in all (260 on CPU, 2 on the H100), about 50
container-hours.

**What holds everything up.** Taken out one at a time, only two of the 21 ingredients break the recipe:
- **A short teacher.** For the first 70 steps the table checks its subtractions with a hand-written adder. Checks made
  only of the table itself are passed by many wrong tables: digits relabeled, borrows in the wrong places, "copy the
  first number". Without those 70 steps no run gets past 39 of 200 entries.
- **A fact about the real size of numbers.** "x − N goes below zero, wraps round to a huge number, and so taking N
  away once more does not go below zero." It fixes the one entry nothing else can see: "0 minus 0 with a borrow",
  which must give 9 (the run of 9s of a wrap-around). Without it, that entry's family of ten stays wrong.

**What was dead weight.** Every other ingredient, removed alone, changed nothing or only the speed:
- the learned adder's own self-check, the headline of the adder piece;
- the extra check on the final borrow, and the extra weight on the lowest digit;
- the extra facts "N − x does not go below zero" and "0 − N does";
- the adder's own rules "0 + N = N" and "R + N = N + R";
- "x mod N = x", and the training answers;
- training along the machine's own path, the two training phases, the optimizer reset, and the final freeze.

That is not the same as useless. Many of them stand in for one another: three lessons all say "x is smaller than N",
and three teach the digits after the warm start. Strip the recipe to its core and some of them matter again (§4–§5).

**What was worse than dead weight.** Two lessons that sound harmless, "R − 0 = R" and "R − R = 0", quietly taught the
wrong thing. Training computes with blurry borrows. So a lesson about "subtract 0 with no borrow coming in" also
teaches its neighbour, "subtract 0 with a borrow coming in", the same answer, and that answer is wrong for it. The two
lessons spread one hard entry's error to the two whole families of stuck entries that later ingredients (the
wrap-around fact, "x mod N = x") had been added to repair. Without them the full recipe trains faster (frozen at step
~280 instead of ~330). The root entry, "0 minus 0 with a borrow", still needs the wrap-around fact.

**The general rules that came out.**
1. A check made only of the learned table can hold a wrong table in place, but cannot correct it.
2. Meaning has to come from outside the table: a trusted reference for a short while (what the digits mean), and the
   true order and size of the row's numbers (what a borrow means).
3. A lesson must train only the entries it actually uses. Blurry lessons leak onto neighbouring entries, and lessons
   that say "no borrow" leak the wrong way. Making the two harmful lessons follow this rule (they read the machine's
   own borrows instead of blurry ones) turned them helpful: the full recipe trained fastest that way, and on 3 of 4
   seeds they even did the job of the wrap-around fact, the ingredient that depends on today's register width.
4. Several ingredients were patches for problems that other ingredients created.

**Did the simplified recipe certify? Yes.** [`submission_simple.py`](submission_simple.py) keeps the machine and
three lessons:
- the check, with the hand-written adder for 70 steps and then the learned one;
- "x − N goes below zero, N − x does not";
- the wrap-around fact.

Nothing else: no answers, no phases, no optimizer reset, no freeze. It never sees a training answer. On the
untouched official manifest on an H100 (`simple_official`, the file byte for byte) it got all 768 questions right at
every T up to 64, on seen and unseen moduli. Its table came out 200/200, and every entry used at evaluation was
trained. Training took 301 s (3,000 steps; every check passes from about step 250), evaluation 23 s. On CPU the same
training certified on 8 of 8 seeds, and so did the same three lessons with the old schedule kept (frozen at steps
260–305, where the full recipe took 327–343). The simplified recipe has no slack left, though: each of its lessons is
needed on some seed, and its warm start cannot drop much below 50 steps (§4, §10).

**Under learned wiring.** The next step learns the wiring too. The warm start's check and the "x < N" facts are
statements about the subtraction cell and about the row's numbers, so they should survive it. Two lessons of the
simplified recipe are tied to today's hand-made circuit:
- the wrap-around fact relies on the register being wider than N. A version on the machine's own chains works just as
  well (§8), but it still needs that width;
- the learned-adder check relies on the hand-wired adder, A − (0 − B).

Most of the dead weight was tied to the hand-made circuit too (§9).

**The combined model.** The decisive removals were repeated on the certified model that learns every part at once
(18 runs, §11). Some results carry over: R − 0 = R and R − R = 0, the final-borrow check and the lowest-place weight
can each go there too. Two things change where the pieces meet:
- the wrap-around fact is not needed there, because that model keeps the hand-written adder longer, until its reader
  has worked out the prompt;
- the learned-adder check becomes essential, and the core's simplified lessons fail on 1 of 4 seeds. The reader delays
  the facts about x and N, and until they arrive, the lessons the core could drop are what hold the table.

## 1. The ingredients

Every non-structural ingredient of the core recipe, with the write-up that added it. "Bare" is the check (hand-written
adder for 70 steps, then the learned one, digits only, every place alike) with the schedule and nothing else (§4).
"Tied" means the ingredient names a part of today's hand-made circuit (§9). Leave-one-out and add-one-in cells read:
seeds certified, the steps at which the certified runs froze, and the end state of the failures (entries right, and
the consistent wrong table closest to it).

<!-- TABLE:ingredients -->
| ingredient | constant | why it was added | leave one out (seeds 74, 1, 2, 3) | add one in (bare + it; seeds 74, 1) | wiring |
|---|---|---|---|---|---|
| warm start: the hand-written adder in the check for steps 1–70 | `ADDER_WARM = 70` | ADDER.md §5.2: from a random start the learned-adder check settles on the lazy table (46 runs, 0 certified) | 0/4; failed: 38/200 (lazy 77), 29/200 (lazy 74), 30/200 (lazy 66), 39/200 (mod-5 borrows 35) | in the bare minimum | general: checks the cell on the operands it is given |
| the learned-adder check (D − N) − (0 − N) = D after the warm start | `ID_AFTER_WARM` (new) | ADDER.md: the point of the piece, no hand-written arithmetic after the warm start | 4/4, frozen 394–479 | in the bare minimum | tied: the adder A − (0 − B) and the chain's links |
| complete identity: the check's final borrow = the adder's final carry | `ID_BORROW = 1` | CHAIN.md §5.3: the digits-only check has a local minimum at (0, 0, 1); seeds stalled 350–1,150 steps, one failed | 4/4, frozen 325–336 | 0/2; failed: 190/200 (borrow-free 91), 190/200 (borrow-free 91) | tied: the stop rule reads the final borrow |
| lowest place weighted 5× | `ID_PLACE0 = 5` | LEARNED_REMAINDER.md §3.4, §4.2: the swapped borrow code differs only at place 0; two first official runs fell there | 4/4, frozen 317–362 | 0/2, frozen 632; failed: 160/200 (borrow-free 81), 66/200 (swapped borrow code 46) | tied: place 0 is the ripple's start (the fixed borrow-in) |
| R − 0 = R on the registers | `ANCHOR_ZERO` | ADDER.md §1.2: the plan's lessons (pins the column m = 0) | 4/4, frozen 322–337 | 0/2; failed: 190/200 (borrow-free 91), 190/200 (borrow-free 91) | general (any numbers) |
| R − R = 0 on the registers | `ANCHOR_SELF` | ADDER.md §1.2: the plan's lessons (pins the diagonal) | 4/4, frozen 280–302 | 0/2; failed: 184/200 (borrow-free 90), 58/200 (swapped borrow code 52) | general (any numbers) |
| x − N borrows (x < N) | `ANCHOR_LESS` | ADDER.md §4: with N − x, teaches all 200 borrows; replaces the designed carry's comparison | 4/4, frozen 615–632 | 0/2; failed: 0/200 (swapped borrow code 120), 0/200 (swapped borrow code 100) | general fact (needs the reader's x and N) |
| N − x does not borrow (x < N) | `ANCHOR_MORE` | ADDER.md §4: as above | 4/4, frozen 334–345 | 0/2; failed: 190/200 (borrow-free 91), 45/200 (swapped borrow code 56) | general fact (needs the reader's x and N) |
| (x − N) − N does not borrow (a wrapped x − N is large) | `ANCHOR_WRAP` | ADDER.md §5.1: the (r, 0, 1) trap ten entries from the truth; without it the warm start stalls at 168–170 | 0/4; failed: 190/200 (borrow-free 91), 171/200 (borrow-free 83), 170/200 (borrow-free 83), 170/200 (borrow-free 82) | 1/2, frozen 367; failed: 190/200 (borrow-free 91) | tied: 9 register places for 8-digit moduli |
| 0 − N borrows (N > 0), after the warm start | `ANCHOR_NEG` | ADDER.md §2: makes the learned adder's carry 1 − b_f; the plan's lessons | 4/4, frozen 327–336 | 0/2; failed: 0/200 (swapped borrow code 120), 23/200 (swapped borrow code 74) | general fact |
| 0 + N = N with the learned adder, after the warm start | `ANCHOR_NEG2` | ADDER.md §4: knocks out the lazy table from a random start | 4/4, frozen 327–340 | 0/2; failed: 0/200 (swapped borrow code 138), 58/200 (swapped borrow code 63) | tied: the adder A − (0 − B) |
| R + N = N + R with the learned adder, after the warm start | `ANCHOR_COMM` | ADDER.md §4: knocks out the lazy table from a random start | 4/4, frozen 333–353 | 0/2; failed: 11/200 (swapped borrow code 124), 59/200 (swapped borrow code 51) | tied: the adder A − (0 − B) |
| x mod N = x (phase B, soft own-borrow choice over one division step) | `SELF_LESSON` | LEARNED_REMAINDER.md §4.2: fixes the identity's (r, r, 1) saddle; CHAIN.md §5.4: still helps | 4/4, frozen 327–343 | 0/2; failed: 0/200 (swapped borrow code 120), 23/200 (swapped borrow code 74) | tied: the chain and the stop rule (its content is x − N borrows) |
| the rows' answers, T ≤ 2 (phase B) | `REAL_MAX_T = 2` | LEARNED_REMAINDER.md §4.2: a direct lesson on stall rows such as (9, 9, 0) | 4/4, frozen 326–336 | 0/2; failed: 0/200 (swapped borrow code 120), 23/200 (swapped borrow code 74) | general (labels) |
| the answers along the machine's own hard path | `ANSWER_PATH = "hard"` | LEARNED_REMAINDER.md §3.3: soft choices teach biased lessons; 10× cheaper | 4/4, frozen 705–748 | only with the answers | general |
| phase switch: phase B's lessons and the freeze wait for 80% of the checks (or 300 steps) | `ID_SWITCH`, `STAGE_WIN`, `ID_STEPS` | LEARNED_REMAINDER.md §4.2: x mod N = x from the start biases every borrow toward 1 | 4/4, frozen 370–377 | in the bare minimum | general |
| Adam's state cleared at the switch | `ADAM_RESET` (new) | LEARNED_REMAINDER.md §3.3: phase A's large gradients in Adam's memory slowed the repairs | 4/4, frozen 326–338 | in the bare minimum | general |
| freeze: learning rate 0 after 200 exact steps | `FREEZE_WIN = 200` | LEARNED_REMAINDER.md §3.1 (stops training once the self-checks are exact) | 4/4, none froze | in the bare minimum | general |
| the check reads every link, also after the chain's first borrow | `ID_LINKS` (new) | CHAIN.md §4: still valid checks; §5.4: live links only did not fix the stall | 4/4, frozen 334–356 | in the bare minimum | tied: the chain |
| the remainder lessons score only the places below N's width | `CE_SCOPE` (new) | LEARNED_REMAINDER.md §3.2: over-subtracting trials write 9s above N's width | 4/4, frozen 327–341 | only with x mod N = x or the answers | tied: the layout |
| the freeze and the switch read every lesson's hard check | `HARD_ANCHORS` (new) | ADDER.md §2: each anchor's hard version joins the checks the schedule reads | 4/4, frozen 327–343 | only with anchors | general |
<!-- /TABLE -->

Not ingredients here: the datapath (reading, squaring, the chain, the stop rule, the T loop), the table's form, the
check's inputs (every link of the machine's own division of each row's x², label-free), the loss normalizations,
`INIT = 0.1` and Adam at 0.05, the one-step form of x mod N = x (`SELF_STEPS`; 8 steps wreck the table,
LEARNED_REMAINDER.md §4.2) and the float32 renormalization of the soft division (ADDER.md §8). Already gone from the
older recipes: the uniform hold, the estimator, the staging k = 1…8, the made-up dividends, the borrow-biased start.

## 2. How the ablation ran

**The recipe under test** is [`submission_adder.py`](../learned_remainder/submission_adder.py) (md5 `44e658ab…`,
certified as `adder_warm_official`): chain division with the learned 200-row subtraction table, learned addition
A − (0 − B) in the self-check, and a 70-step warm start with the hand-written adder. Its lessons, schedule and
datapath are described in ADDER.md §2 and CHAIN.md §3–4.

**The ablation copy.** [`submission_ablate.py`](submission_ablate.py) is that file with an off-switch for every
ingredient the original had no constant for, the general rules of §8, and a log line with the snapped table every 50
steps:

| constant | default (= the certified file) | off |
|---|---|---|
| `ID_AFTER_WARM` | 1: the learned-adder check after the warm start | 0 (with `ID_GATE = True` its hard version still gates the freeze) |
| `ID_LINKS` | `"all"`: every link the hard chain computes | `"live"`: up to the chain's first borrow |
| `CE_SCOPE` | `"width"`: the remainder lessons score the places below N's width | `"all"` |
| `ADAM_RESET` | clear Adam's state at the switch | keep it |
| `HARD_ANCHORS` | every lesson's hard version joins the checks the schedule reads | only the check and x mod N = x |
| `ANCHOR_ORDER`, `ANCHOR_CHAIN`, `HARD_CIN` | 0, 0, False (§8) | – |
| `TABLE_EVERY` | 0 | 50 in every ablation run (logging only) |

At its defaults the copy trains exactly as the certified file. [`local_equiv.py`](local_equiv.py) runs both side by
side and finds identical losses and parameters at every step: 90 steps across the warm start's end, and 60 steps
across the switch to phase B. Through the evaluator, `loo_base_s74` and `loo_base_s1` reproduce the certified file's
CPU records line for line (switch at 134 and 123, frozen at 343 and 331, the same table). Every variant is a
`CONSTANT = value` rewrite of the copy, as the pieces' `--set` did, and every record keeps its exact source.

**The evaluator path** is the pieces' own: `benchmark.runner` in the untouched competition image, on Modal CPU
containers (8 CPUs, 16 GB). The manifest is the official H1 one with only `device: cpu` and the seed changed, the
submission's batch is 512 and its step cap 3,000. [`modal_ablate.py`](modal_ablate.py) is `modal_chain.py`, deployed
once as the app `ablation` with at most 20 CPU containers; [`launch.py`](launch.py) spawns a plan,
[`fetch.py`](fetch.py) downloads its records into `runs/`, and [`analyze.py`](analyze.py) reads them. It checks
certification (T = 64 on both profiles, every rung 768/768) and the steps of the switch and the freeze. It also
compares the snapped table, outside the submission, with true subtraction and with the consistent wrong tables of
ADDER.md §3 (lazy, swapped borrow code, digits relabeled, mod-2 / mod-5 borrows, borrow-free, copy the minuend,
reversed).

**Seeds.** 74 (the official seed), 1, 2, 3 for leave one out; 74 and 1 for exploratory recipes; 74 and 1–7 for the
final recipe.

**What counts as a failure.** Not certified within the recipe's own 3,000 steps (the combined model: its own 20,000
steps or the 3,600 s training budget). In the full recipe every run that froze certified, and every failure never
froze; two stripped recipes froze on a wrong table (§4).

**Probes** (local, each under 2 s and 0.7 GB):
- [`probe_trap.py`](probe_trap.py) computes every lesson's loss exactly on fixed numbers (1,024 training rows and the
  machine's own link inputs), along a straight path from a wrong table to the truth;
- [`probe_grad.py`](probe_grad.py) takes the true table blurred to a given sharpness and computes which way each
  lesson's gradient pushes the entries of a trap family.

This is the tiny-core method of [TINYCORE.md](../tinycore/TINYCORE.md) applied to the real table: hold everything
fixed but the one thing in question, and compute it exactly.

## 3. Leave one out

Each row switches one ingredient off in the full recipe and keeps everything else. On the same seeds the full recipe
switched to phase B at steps 134, 123, 127, 120 and froze at 343, 331, 329, 327 (`adder_warm_cpu_seed*`; `base` is
the ablation copy at its defaults). The answers (`REAL_MAX_T = 0`) reuse the certified file's own runs that never see
an answer (`adder_warm_idonly_cpu_seed*`, the same source). "Rows right" is the snapped table at the end; a failure
cell names the consistent wrong table closest to it and the families of wrong entries.

<!-- TABLE:loo -->
| removed | variant | seeds | certified | switch | frozen at | rows right | failure: closest wrong table, wrong rows |
|---|---|---|---|---|---|---|---|
| nothing (the ablation copy at its defaults) | base | 74, 1 | 2/2 | 134, 123 | 343, 331 | 200, 200 |  |
| warm start | noWarm | 74, 1, 2, 3 | 0/4 | 300, 300, 300, 300 | –, –, –, – | 38, 29, 30, 39 | s74: 38/200, lazy 77, (r,0,1)x10 (r,r,1)x9 other x143; s1: 29/200, lazy 74, (r,0,1)x10 (r,r,1)x9 other x152; s2: 30/200, lazy 66, (r,0,1)x10 (r,r,1)x9 other x151; s3: 39/200, mod-5 borrows 35, (r,0,1)x10 (r,r,1)x6 other x145 |
| learned-adder check | noLearnedCheck | 74, 1, 2, 3 | 4/4 | 121, 100, 105, 96 | 479, 415, 394, 434 | 200, 200, 200, 200 |  |
| complete identity | noIdBorrow | 74, 1, 2, 3 | 4/4 | 136, 124, 128, 127 | 336, 325, 328, 327 | 200, 200, 200, 200 |  |
| lowest place ×5 | noPlace0 | 74, 1, 2, 3 | 4/4 | 144, 120, 121, 117 | 362, 321, 321, 317 | 200, 200, 200, 200 |  |
| R − 0 = R | noZero | 74, 1, 2, 3 | 4/4 | 111, 118, 123, 115 | 322, 337, 325, 330 | 200, 200, 200, 200 |  |
| R − R = 0 | noSelf | 74, 1, 2, 3 | 4/4 | 96, 81, 97, 80 | 296, 281, 302, 280 | 200, 200, 200, 200 |  |
| x − N borrows | noLess | 74, 1, 2, 3 | 4/4 | 300, 300, 300, 300 | 615, 631, 632, 621 | 200, 200, 200, 200 |  |
| N − x no borrow | noMore | 74, 1, 2, 3 | 4/4 | 145, 134, 141, 136 | 345, 334, 341, 336 | 200, 200, 200, 200 |  |
| (x − N) − N no borrow | noWrap | 74, 1, 2, 3 | 0/4 | 300, 300, 300, 300 | –, –, –, – | 190, 171, 170, 170 | s74: 190/200, borrow-free 91, (r,0,1)x10; s1: 171/200, borrow-free 83, (r,0,1)x10 (r,r,1)x2 other x17; s2: 170/200, borrow-free 83, (r,0,1)x10 (r,r,1)x2 other x18; s3: 170/200, borrow-free 82, (r,0,1)x10 (r,r,1)x2 other x18 |
| 0 − N borrows | noNeg | 74, 1, 2, 3 | 4/4 | 135, 124, 127, 120 | 336, 327, 329, 327 | 200, 200, 200, 200 |  |
| 0 + N = N | noNeg2 | 74, 1, 2, 3 | 4/4 | 129, 121, 122, 119 | 340, 327, 329, 328 | 200, 200, 200, 200 |  |
| R + N = N + R | noComm | 74, 1, 2, 3 | 4/4 | 128, 118, 123, 119 | 353, 335, 333, 339 | 200, 200, 200, 200 |  |
| x mod N = x | noXmod | 74, 1, 2, 3 | 4/4 | 134, 123, 127, 120 | 343, 331, 329, 327 | 200, 200, 200, 200 |  |
| own hard path for the answers | softAnswers | 74, 1, 2, 3 | 4/4 | 134, 123, 127, 120 | 725, 705, 748, 739 | 200, 200, 200, 200 |  |
| phase switch | noPhaseA | 74, 1, 2, 3 | 4/4 | 1, 1, 1, 1 | 370, 372, 375, 377 | 200, 200, 200, 200 |  |
| Adam reset | noAdamReset | 74, 1, 2, 3 | 4/4 | 134, 123, 127, 120 | 338, 328, 329, 326 | 200, 200, 200, 200 |  |
| freeze | noFreeze | 74, 1, 2, 3 | 4/4 | 134, 123, 127, 120 | –, –, –, – | 200, 200, 200, 200 |  |
| every link checked | liveLinks | 74, 1, 2, 3 | 4/4 | 156, 109, 116, 106 | 356, 334, 335, 336 | 200, 200, 200, 200 |  |
| score below N's width | ceAll | 74, 1, 2, 3 | 4/4 | 134, 123, 127, 120 | 341, 327, 329, 328 | 200, 200, 200, 200 |  |
| freeze reads every check | noHardAnchors | 74, 1, 2, 3 | 4/4 | 134, 123, 127, 120 | 343, 331, 329, 327 | 200, 200, 200, 200 |  |
| the answers (`adder_warm_idonly_cpu_seed*`) | noAnswers | 74, 1, 2, 3 | 4/4 | 134, 123, 127, 120 | 336, 327, 329, 326 | 200, 200, 200, 200 |  |
<!-- /TABLE -->

What it says:
- **Two ingredients hold the full recipe up on their own.**
  - Without the warm start the table ends with 29–39 of 200 entries right, nearest the lazy table of ADDER.md §3:
    every borrow right, the digits copying the first operand. ADDER.md found the same.
  - Without the wrap-around fact the (r, 0, borrow 1) family, "subtract 0 with a borrow coming in", ignores the
    borrow for 2,700 steps: 170–190 of 200 right.
- **"x − N borrows" matters for speed only.** Without it, phase A sits at about 181 of 200 until its 300-step cap: row
  (0, 0, 1) and the (r, r, 1) family are wrong, their borrows too. A phase-B lesson then repairs them, and the run
  freezes at 615–632 instead of ~330. §5 shows which phase-B lesson does it.
- **Everything else changes nothing, or only the speed.**
  - "x mod N = x" leaves even the freeze step unchanged on every seed, and so do the hard anchors. The Adam reset and
    the scoring scope move it by at most 5 steps.
  - Without R − R = 0 the run freezes 30–50 steps earlier (280–302).
  - Without the learned-adder check it freezes 60–140 steps later (394–479).
  - Without the phase switch (phase B from step 2) it freezes at 370–377.
  - Without the freeze, the table stays exact for the remaining 2,850 steps.
  - With the answers through the soft own-borrow choice instead of the machine's own hard path, a right entry,
    (9, 0, borrow 1), is pushed wrong for about 250 steps after the table was already exact (frozen at 705–748).

## 4. Add one in, and small recipes

**The bare minimum** keeps the check and the schedule and nothing else: the hand-written adder for 70 steps, then the
learned one, digits only, every place weighted the same; the phase switch, the Adam reset and the freeze. The add-one-in
column of §1 adds one ingredient back to it (seeds 74 and 1). Small recipes built from it:

<!-- TABLE:small -->
| recipe | certified, frozen at, failures |
|---|---|
| bare + x − N borrows + wrap | 2/4, frozen 315–354; failed: 198/200 (borrow-free 100), 198/200 (borrow-free 100) |
| bare + N − x no borrow + wrap | 1/4, frozen 442; failed: 190/200 (borrow-free 91), 190/200 (borrow-free 91), 190/200 (borrow-free 91) |
| bare + x − N + N − x (no wrap) | 0/4; failed: 190/200 (borrow-free 91), 98/200 (borrow-free 52), 190/200 (borrow-free 91), 24/200 (swapped borrow code 84) |
| **bare + x − N + N − x + wrap (the simplified recipe)** | 2/2, frozen 280–305 |
| the same, on 8 seeds | 8/8, frozen 260–305 |
| the same, no schedule (no phase switch, no Adam reset, no freeze): `submission_simple.py`'s training | 8/8, none froze |
| + x mod N = x | 2/2, frozen 280–305 |
| + R − 0 = R, R − R = 0 | 2/2, frozen 415–441 |
| x − N + N − x + wrap, no check lesson after the warm start (the learned check only gates the freeze) | 3/4, frozen 260–318; failed: 199/200 (borrow-free 99) |
| the same, no schedule | 5/8, none froze; failed: 199/200 (borrow-free 99), 199/200 (borrow-free 99), 199/200 (borrow-free 99) |
| the same + the answers | 1/2, frozen 287; failed: 127/200 (borrow-free 74) |
| the same + x mod N = x + the answers | 1/2, frozen 287; failed: 156/200 (borrow-free 88) |
| the same + R + N = N + R | 2/2, frozen 281–312 |
| x − N + N − x, no check lesson, no wrap | 0/4; failed: 198/200 (borrow-free 99), 13/200 (swapped borrow code 116), 199/200 (borrow-free 100), 10/200 (swapped borrow code 139) |
<!-- /TABLE -->

What it says:
- **No single ingredient makes the bare check work.** Added one at a time (§1's add-one-in column, seeds 74 and 1),
  only the wrap-around fact certifies a seed, and only one of two. The 25 failures fall into two groups:
  - **the swapped borrow code** (0–66 of 200 right, 12–118 of the 200 borrows): the bare check alone, and with
    x − N borrows, 0 − N borrows, 0 + N = N, R + N = N + R, x mod N = x or the answers, on both seeds. The bare check
    weights every place alike, so it cannot tell the swapped code from the truth (LEARNED_REMAINDER.md §3.4), and
    none of these lessons pushes hard enough at that end of the table. (x − N borrows does reject the full swapped
    code; the runs settle next to it, on a variant that passes x − N.)
  - **the (r, 0, borrow 1) family**, the run-of-9s trap (184–190 of 200, the borrows right): with the complete identity,
    R − 0 = R, R − R = 0 (seed 74), N − x (seed 74), the lowest-place weight (seed 74), or the wrap-around fact
    (seed 74). These lessons break the swapped code's symmetry, and then the trap is the next thing in the way.
  So the simplified recipe needs one lesson against each: the wrap-around fact against the trap (it also rejects the
  swapped code), and the two order facts to pin the borrows.
- **The freeze can lock in a wrong table once the recipe is stripped.** In the full recipe no run ever froze on a
  wrong table: the freeze waits for every check, facts included. Two stripped runs did, with every check they read
  exact for 200 steps: the bare check with the lowest-place weight (seed 74, 160 of 200 right) and the simplified
  recipe with a 30-step warm start (§10, 100 of 200 right). Both are Rule 1: a check made of the table can be passed
  by a wrong table. `submission_simple.py` has no freeze.
- **Three facts on top of the bare check are enough:** x − N borrows, N − x does not, and the wrap-around fact. 8/8
  seeds with the schedule (frozen at 260–305, sooner than the full recipe) and 8/8 without it. With R − 0 = R and
  R − R = 0 added back it slows to 415–441.
- **None of the three can go.** Without the wrap-around fact every seed fails (0/4, and 0/4 without the check too).
  On 5 of the 8 runs the root entry (0, 0, 1) stays wrong and drags its family along, as in the full recipe. The other
  3 land near the swapped borrow code (84–139 of its 200 entries): the bare check weights every place alike, and
  x − N and N − x alone did not keep that code out. The wrap-around fact rejects it even harder (6.0 nats per row at
  the swapped code, against 4.4 for x − N; `probe_trap.py whole:"swapped borrow code"`), so in the simplified
  recipe it also does the old job of the lowest-place weight.
- **Leave one out on the simplified recipe itself** (seeds 74, 1, 2, 3): every one of its lessons is needed on some
  seed. Without the wrap-around fact 0/4; without x − N borrows 1/4; without N − x 2/4; without the check after the
  warm start 3/4 with the schedule and 5/8 without; without the warm start 0/2 (§10). The full recipe's redundancy is
  what made most of its ingredients look like dead weight; the simplified recipe has none left.
- **In the small recipe, two of the dead-weight ingredients carry weight.** This is the covering that leave-one-out
  cannot see:
  - Without N − x, seed 74 ends with two compensating borrow errors, (1, 1, 1) → no borrow and (9, 8, 0) → borrow.
    The table then passes 99.6–100% of the checks, never freezes, and gets 67% of the test rows. In the full recipe the
    other "no borrow" lessons (R − 0, R − R, the complete identity) covered for it.
  - Without the check after the warm start (the facts alone), seed 74 ends with (0, 0, 0) → 9, with the learned check
    gating the freeze; with no schedule at all 3 of 8 seeds end the same way (5/8 certify). No fact reads that digit.
    In the full recipe R − 0 = R, 0 + N = N and the answers covered for it. Any lesson that reads every digit will do:
    with R + N = N + R in place of the check, both seeds certify (281–312).
- **Answers without a self-check can do harm.** With the facts and the answers but no check after the warm start, seed
  74 was at 199 of 200 at step 200. The answers, learned along the hard path of that slightly wrong machine, then pulled
  it down to 127 of 200 by step 2,500 (156 with x mod N = x as well). With a self-check beside them they never did.

## 5. Pairs that cover for each other

Leave-one-out calls an ingredient dead weight if the recipe certifies without it. Two ingredients that each do the
other's job both look like dead weight; removing them together tells.

<!-- TABLE:pairs -->
| removed together | certified, frozen at, failures |
|---|---|
| x − N borrows, x mod N = x | 4/4, frozen 607–615 |
| the learned adder's lessons after the warm start: its check, 0 + N = N, R + N = N + R | 4/4, frozen 645–1924 |
| R − 0 = R, R − R = 0 | 4/4, frozen 266–298 |
| R − 0 = R, R − R = 0, wrap | 0/4; failed: 187/200 (borrow-free 88), 187/200 (borrow-free 88), 187/200 (borrow-free 88), 186/200 (borrow-free 87) |
| R − 0 = R, R − R = 0, wrap, x mod N = x | 0/4; failed: 187/200 (borrow-free 88), 187/200 (borrow-free 88), 187/200 (borrow-free 88), 186/200 (borrow-free 87) |
| the learned check, wrap | 0/4; failed: 163/200 (mod-5 borrows 75), 122/200 (borrow-free 65), 118/200 (borrow-free 63), 123/200 (borrow-free 63) |
| the learned check, wrap, R + N = N + R | 0/4; failed: 173/200 (borrow-free 94), 178/200 (borrow-free 97), 172/200 (borrow-free 96), 176/200 (borrow-free 95) |
<!-- /TABLE -->

What it says:
- **"x < N" comes in three forms:** x − N borrows (from step 1), x mod N = x and the answers (both in phase B). Remove
  the first two and the answers still repair the borrows after the switch (frozen at 607–615). Remove the first alone
  and x mod N = x does it. Remove x mod N = x alone and nothing changes.
- **The learned adder's three lessons** (its check, 0 + N = N, R + N = N + R) can all go after the warm start, with the
  learned check's hard version still gating the freeze. The (r, r, 1) family planted by R − R = 0 then waits for
  phase B's lessons, frozen at 645–717, and on seed 74 at 1,924.
- **R − 0 = R and R − R = 0 together** are not a covering pair: without both, the full recipe is faster still (frozen
  at 266–298), and at step 50 neither trap family is there.
- **The wrap-around fact is needed even without the two anchors.** Without R − 0 = R, R − R = 0 and the wrap-around
  fact, every seed ends at 186–187 of 200, with x mod N = x kept and with it removed alike: the whole (r, 0, borrow 1)
  family ignoring the borrow, from step ~150 to 3,000. Its root is the one entry (0, 0, 1) (Rule 4 in §7).
- **The learned check and the wrap-around fact are not a covering pair, they are a lock and its key.** Without both,
  with R − 0 = R and R − R = 0 still planting the traps, every seed fails (118–178 of 200), with or without
  R + N = N + R. With the learned check gone but the wrap-around fact kept, the recipe certifies (§3); with the wrap
  fact gone but the check kept, it fails (§3). The check holds the trap and the wrap fact opens it.

## 6. Why: what each lesson does near the truth

### 6.1 Every near-truth trap is one table entry
Every failing or slow run of §3 ends with the same entries wrong, two families with a borrow coming in:
(r, 0, borrow 1), "subtract 0 with a borrow", which should give r − 1 (and 9 for r = 0), and (r, r, borrow 1),
which should give 9. [`probe_trap.py`](probe_trap.py) slides those entries from the trap's digit to the true one,
everything else at the truth (logit 6, so 98% sharp), and computes each lesson exactly on fixed numbers: 1,024
training rows and the trapped machine's own 73,728 link inputs.

| path (entries moved) | learned-adder check | designed-adder check | (x − N) − N no borrow | R + N = N + R | 0 + N = N | x − N borrows, N − x no borrow, 0 − N borrows |
|---|---|---|---|---|---|---|
| (r, 0, 1), all 10 | 0.207 → **0.353** (λ = 0.5) → 0.087: barrier | 0.30 → 0.03, downhill | 0.64 → 0.003, downhill | 0.85 → 0.10, downhill | 0.30 → 0.25 → **0.30** → 0.05: barrier | flat |
| (r, r, 1), all 10 | 0.330 → **0.368** → 0.090: barrier | 0.33 → 0.04, downhill | 0.85 → 0.003, downhill | 0.87 → 0.10, downhill | barrier | flat |
| (0, 0, 1) alone | 0.244 → **0.350** → 0.089: barrier | 0.23 → 0.04, downhill | 0.64 → 0.003, downhill | 0.86 → 0.10, downhill | barrier | flat |
| (r, 0, 1) without (0, 0, 1) | 0.506 → 0.090, downhill | downhill | flat | downhill | flat | flat |
| (r, r, 1) without (0, 0, 1) | 1.128 → 0.091, downhill | downhill | flat | downhill | flat | flat |

So the trap is the single entry (0, 0, borrow 1), the run of 9s of CHAIN.md §5.2: with it wrong, the learned adder that
should price the error is built from the same wrong entry, and the check has a barrier between the trap and the truth.
With it right, the learned check pulls the rest of both families in by itself. Facts that read only a final borrow
(x − N borrows, N − x does not, 0 − N borrows) are blind to these digits, because the trapped entries keep their
borrows. Two lessons see the trap from outside: the wrap-around fact (a wrapped x − N must be large) and
commutativity (R + N and N + R use the entry differently, so they disagree).

### 6.2 Where the trap comes from: two anchors leak through soft borrows
A lesson on soft digits mixes, at every place, the entry with borrow-in 0 and the one with borrow-in 1, weighted by
the soft borrow from the place below. R − 0 = R and R − R = 0 target the borrow-in-0 entries (r, 0, 0) → r and
(r, r, 0) → 0. Through the soft borrow they also teach the borrow-in-1 entries next to them the same digits:
(r, 0, 1) → r and (r, r, 1) → 0, exactly the two trap families. [`probe_grad.py`](probe_grad.py) measures it: at the
true table blurred to sharpness s, the gradient of each lesson (with its weight in the recipe) on the trap
families' digit logits, as a push toward the trap (×10⁴; per family (r, 0, 1) / (r, r, 1) / (0, 0, 1)):

| sharpness (P(true digit)) | designed-adder check | R − 0 = R | R − R = 0 | (x − N) − N no borrow | warm-start total |
|---|---|---|---|---|---|
| 2.0 (0.45) | −216 / −90 / −14 | +379 / +286 / +286 | +494 / +711 / +494 | −84 | **+573 / +823 / +682** |
| 2.5 (0.58) | −280 / −159 / −90 | +165 / +127 / +127 | +238 / +333 / +238 | −55 | **+68 / +246 / +220** |
| 3.0 (0.69) | −311 / −212 / −155 | +68 / +53 / +53 | +102 / +142 / +102 | −30 | −171 / −47 / −30 |
| 4.0 (0.86) | −229 / −177 / −148 | +10 / +8 / +8 | +16 / +22 / +16 | −6 | −209 / −153 / −130 |

While the table is still blurry, during the warm start, the two anchors push both families toward the trap harder than
the hand-written adder pulls them out. The runs show the same thing directly, at step 50 (inside the warm start):

| variant (seeds 74, 1, 2, 3) | (r, 0, 1) entries wrong at step 50 (of 10) | (r, r, 1) entries wrong, r ≥ 1 (of 9) |
|---|---|---|
| without N − x, 0 − N or x mod N = x (as the full recipe: they do not read these digits, or start later) | 10, 10, 10, 10 | 9, 9, 9, 9 |
| without R − 0 = R | **1, 1, 1, 1** | 9, 9, 9, 9 |
| without R − R = 0 | 10, 10, 9, 10 | **1, 0, 0, 0** |

The anchors were added to pin the column m = 0 and the diagonal; the wrap-around fact (ADDER.md §5.1) and "x mod N = x"
(LEARNED_REMAINDER.md §4.2) were added later to repair (r, 0, 1) and (r, r, 1).

## 7. The general rules

Grouped by what they do, the ingredients that ever mattered come down to four rules. Each special case below is one
instance of one of them.

**Rule 1. A check made only of the learned table can hold a table but not teach one.**
- The learned-adder identity, 0 + N = N, R + N = N + R, R − 0 = R and R − R = 0 are all passed by whole families of
  wrong tables: digits relabeled, borrows through only the mod-2 or mod-5 part of each digit, the swapped borrow code
  (for the adder identities), the lazy table (for the learned check) (ADDER.md §3; `probe_trap.py whole:…`).
- Near the truth, the learned-adder identity and 0 + N = N have a barrier at the run-of-9s entry (§6.1). The adder that
  should price the error is built from the same wrong entry.
- That is why the learned-adder check, the headline of ADDER.md, could be removed from the full recipe at no cost:
  next to the facts it taught nothing new, and it was one of the two lessons holding the trap.
- It still has a job in the simplified recipe: after the warm start it is the only lesson that reads every digit
  entry, and without it the entry (0, 0, 0) can stay wrong (§4). In the combined model it is the only lesson left
  that pulls the (r, r, 1) family in (§11).
- A wrong table that passes every label-free lesson of the simplified recipe exists: digit r − m − [m even], borrows
  true (§10). Only the warm start, R − 0 = R or the answers reject it.

**Rule 2. Meaning has to come from outside the table, once.** Two outside sources did all the teaching:
- **Digit meanings from a trusted reference for a short while.** That is the hand-written adder's 70 steps. ADDER.md §5
  found that the learned check finishes the table only from about half the digits right, and nothing label-free gets a
  random table there.
- **Borrow meanings from the true order of the row's numbers, x < N.**
  - "x − N borrows", "x mod N = x" and the answers carry it; each is dead weight next to the others (§5).
  - It also made the lowest-place weight dead weight in the full recipe. That weight was the defense against the
    swapped borrow code, added before any order fact existed, and the swapped code fails "x − N borrows" on every row.
    In the bare recipe x − N alone did not keep the swapped code out (§4); the wrap-around fact does.
- **One fact about size**, for the one entry that every fact reading a final borrow is blind to: (0, 0, borrow 1), the
  run of 9s. The trapped entries keep their borrows right, so a final borrow cannot tell. "A wrapped x − N is large,
  so (x − N) − N does not borrow" reads the wrapped value.

**Rule 3. A lesson must train only the entries it uses: soft lessons leak.**
- A lesson computed on blurry digits and borrows mixes, at every place, the entry it uses with its borrow-in-1
  neighbour, and it teaches the neighbour the same target.
- For R − 0 = R and R − R = 0 those neighbours, (r, 0, 1) and (r, r, 1), are never used by the lesson at all. The leak
  is pure bias, and it builds exactly the two trap families (§6.2).
- Every "does not borrow" fact leaks the same way onto the borrows of those entries: x − 0, x − x, N − 0, N − N, even
  N − x push them toward swallowing the borrow while the table is blurry. "Borrows" facts (x − N, 0 − N, 0 − x) push
  them the right way. That is why "x − N borrows" is the useful one of the pair, and why the general order rule of §8
  is slower than the two facts it generalizes.
- The same rule is behind training along the machine's own hard path (LEARNED_REMAINDER.md §3.3; here soft answers
  break a right entry for about 250 steps). It is behind scoring only the places below N's width
  (LEARNED_REMAINDER.md §3.2), and behind COMBINED.md's joint-borrow soft adder for the ladder (§4.2 there).
- The general fix: ripple a lesson with the hard machine's own borrows (`HARD_CIN`), so that its gradient reaches only
  the entries it uses. That removes the push exactly (0.0 in the probe at every sharpness) (§8).

**Rule 4. Several ingredients were patches for problems other ingredients created.** In the order they were added:
- R − 0 = R and R − R = 0 planted the traps.
- x mod N = x (LEARNED_REMAINDER.md, for (r, r, 1)) and the wrap-around fact (ADDER.md, for (r, 0, 1)) were added to
  repair them.
- The complete identity repaired the same (0, 0, 1) entry in CHAIN.md, under the hand-written adder.
- The lowest-place weight guarded against a code that the order facts later excluded anyway.
- The phase switch protected against x mod N = x's early bias. The Adam reset protected the switch.
- Remove the anchors and the two families never form (§6.2). The root entry (0, 0, 1) is another matter: it is the
  hardest entry for the hand-written adder too (its soft check has a shallow minimum there, CHAIN.md §5.2; at
  sharpness 1.5 even the designed check pushes it toward 0, +60 in §6.2's units). The wrap-around fact acts from step
  1, and at step 50, inside the warm start, it decides that entry. Without the leaky anchors, (0, 0, 1) is still wrong
  at step 50 in all 12 runs without the wrap-around fact, and right in 7 of the 8 with it. (Without R − 0 = R alone,
  it is the one entry of its family still wrong at step 50 on every seed.) Without the wrap-around fact (and x mod N =
  x) the learned check then holds it and drags the m = 0 column along (§5). So the wrap-around fact is the one repair
  that has a cause of its own.

## 8. General replacements

Three general rules were tested in place of special cases (`submission_ablate.py`):
- **The order rule** (`ANCHOR_ORDER`): for every ordered pair of different numbers among 0, x and N, and for a − a, the
  final borrow of a − b is [a < b]. That is one rule in place of x − N, N − x, 0 − N and the borrow halves of R − 0 = R
  and R − R = 0 (whose digit halves leak, Rule 3).
- **The chain rule** (`ANCHOR_CHAIN`): a link that borrows wraps around to 10⁹ + D − N, which is at least N, so on every
  link of the machine's own chains, the link after a borrowing one does not borrow. It is the wrap-around fact on
  the machine's own numbers instead of on the prompt's x.
- **The hard-borrow ripple** (`HARD_CIN`): R − 0 = R and R − R = 0 ripple with the hard machine's own borrow into each
  place, so their gradient reaches only the entries they use (Rule 3).

<!-- TABLE:general -->
| general replacement | certified, frozen at, failures |
|---|---|
| x − N, N − x, 0 − N, R − 0 = R, R − R = 0 all off | 4/4, frozen 349–859 |
| … replaced by the order rule | 4/4, frozen 329–431 |
| wrap replaced by the chain rule | 4/4, frozen 338–350 |
| both replacements | 4/4, frozen 714–945 |
| both, and nothing else after the warm start but the check | 0/4; failed: 190/200 (borrow-free 91), 190/200 (borrow-free 91), 188/200 (borrow-free 91), 190/200 (borrow-free 91) |
| R − 0 = R, R − R = 0 rippled with the hard borrows | 4/4, frozen 252–278 |
| the same, without wrap | 3/4, frozen 288–295; failed: 190/200 (borrow-free 91) |
| the bare check + x − N + N − x + the two anchors rippled with the hard borrows, no wrap | 0/4; failed: 190/200 (borrow-free 91), 191/200 (borrow-free 92), 190/200 (borrow-free 91), 191/200 (borrow-free 92) |
| the same, no schedule | 0/4; failed: 190/200 (borrow-free 91), 191/200 (borrow-free 92), 190/200 (borrow-free 91), 191/200 (borrow-free 92) |
<!-- /TABLE -->

What it says:
- **The chain rule does the wrap-around fact's job in full** (4/4, frozen 338–350, as fast as the original). It reads
  every chain the machine computes, not the prompt's x, so it needs N but not x. It still relies on the register width.
- **The order rule works, but more slowly** (329–431). Its "does not borrow" pairs leak onto the borrow-in-1 entries
  (Rule 3): x − 0, x − x, N − 0 and N − N push the borrows of (r, 0, 1) and (r, r, 1) the wrong way while the table is
  blurry, by +1,000 to +3,000 (×10⁻⁴) per family at sharpness 2.5, while x − N, 0 − N and 0 − x push them the right
  way by about −5,000. So a general rule has to respect Rule 3 to beat the special case it replaces.
- **Together they fail without help.** The two replacements together are slower still (714–945): the (r, 0, 1) family
  waits for phase B's answers and x mod N = x. Without those (the check, the order rule and the chain rule, nothing
  else), all four seeds stop at 188–190 of 200 for 2,700 steps. The (r, 0, 1) family is wrong, and row (0, 0, 1)
  has lost its borrow too: the order rule's no-borrow pairs (x − 0, N − 0, x − x, N − N) pull that borrow to 0 through
  the soft ripple. A chain that no longer borrows at the top gives the chain rule nothing to read. So the general
  order rule should keep only the pairs whose hard ripple uses the entries it trains (x − N, 0 − N, 0 − x, N − x), or
  ripple with the hard borrows; as built here it is Rule 3's counterexample.
- **The hard-borrow ripple turns the two anchors from harmful to helpful.** With R − 0 = R and R − R = 0 rippled
  with the hard machine's borrows, the full recipe freezes at 252–278, the fastest of all (the original: 327–343).
  It also certifies without the wrap-around fact on 3 of 4 seeds, which no other variant of the full recipe did. The
  anchors then pin the borrow-in-0 half of their columns without leaking into the borrow-in-1 half. That keeps the
  m = 0 column from settling into the "ignore the borrow" block that the lone (0, 0, 1) error otherwise drags it into.
  The learned check and R + N = N + R then pull (0, 0, 1) in by step 100 (it was wrong at step 50 on all four seeds).
  On seed 1 it was still wrong at step 100 and the block formed by step 150 anyway (190 of 200). So Rule 3's fix does
  most of the wrap-around fact's job without any register width: the nearest thing here to a general replacement for
  a wiring-tied ingredient, but not a full one.
- **Nor is it standalone.** The same two anchors, rippled with the hard borrows, on top of the bare check with x − N
  and N − x and no wrap-around fact, fail on every seed (0/8 with and without the schedule; 190–191 of 200, the
  (r, 0, 1) family stuck). In the full recipe the anchors' fix works together with the learned adder's own rules and
  the phase-B lessons; on its own it only stops the leak.

## 9. Which ingredients assume today's hand-made wiring

The next step learns the wiring too (cpu_first/tinyrouting: which register feeds which port, the shift, the stop rule,
the direction N's digits are read in). An ingredient survives that if it is a statement about the subtraction cell or
about the row's numbers. It breaks if it names a part of the hand-made circuit. Sorted that way:
- **Statements about the cell, on whatever operands it is given:**
  - the warm start's check, (D − N) + N = D with a trusted adder. It needs to know which operand was subtracted (the
    cell's port convention) and nothing else. TINYROUTING found that a cell check of this kind is what makes learned
    routing work at all;
  - R − 0 = R and R − R = 0, and the order rule.
- **Statements about the row's numbers:** x − N borrows, N − x does not, 0 − N borrows, the answers. They need the
  reader to know which field is x and which is N (in the combined model the facts wait for the reading for that
  reason), and the cell's port convention. They do not care how the division is wired.
- **Tied to the hand-made circuit:**
  - the lowest-place weight: "place 0" is the ripple's start, the one place whose borrow-in is fixed;
  - the complete identity: the final borrow matters because the hand-made stop rule reads it;
  - the wrap-around fact and the chain rule: "a wrapped number is at least N" needs 9 register places for 8-digit
    moduli (the layout);
  - "x mod N = x": one division step through the chain and the stop rule, with the soft own-borrow choice;
  - the learned-adder check, 0 + N = N and R + N = N + R: the adder A − (0 − B) is a hand-wired composition of two
    subtractions, and the check runs on the chain's link inputs;
  - "every link is checked, also after the first borrow": the chain;
  - scoring only the places below N's width: the layout.
- **The schedule** (phase switch, Adam reset, freeze, which checks the freeze reads) assumes no wiring.

The simplified recipe keeps the warm start (cell), the learned check (tied: the adder composition), x − N and N − x
(row facts) and the wrap-around fact (tied: the register width).
- The learned check is the one to replace first: a learned router would need its own adder wiring. R + N = N + R
  could stand in for it in the simplified recipe (§4: 2/2), but it is a learned-adder rule too.
- For the wrap-around fact, the direction to look is Rule 3. Within the full recipe, R − 0 = R and R − R = 0 rippled
  with the hard borrows (statements about the cell) did its job on 3 of 4 seeds (§8). On their own they did not.
- The answers are the label-side fallback, but §4 shows they need a self-check beside them.

## 10. The simplified recipe, certified

[`submission_simple.py`](submission_simple.py) is `submission_adder.py`'s machine with the training recipe of §4
(bare + x − N borrows + N − x does not + the wrap-around fact) and no schedule. Its whole training is:
- every link of the machine's own division of each row's x² checks itself, (D − N) + N = D, digits only, every place
  alike: with the hand-written adder for steps 1–70, then with the learned adder (D − N) − (0 − N);
- x − N borrows, and N − x does not (x < N);
- (x − N) − N does not borrow (x − N wraps around to 10⁹ + x − N ≥ N);
- 3,000 steps of Adam at 0.05, nothing else. No answer is used; no phase, reset or freeze.

It is 363 lines against the original's 696: the learned adder's own rules, route B, the variants, the soft division,
the answers and the schedule are gone. [`local_equiv_simple.py`](local_equiv_simple.py) checks that it trains exactly as
`submission_ablate.py` with the settings of `simple_LMWnosched` (identical losses and parameters at every step for
90 steps, across the warm start's end).

| run | what ran | max T (seen / unseen moduli) | rungs | test, ood_t, ood_n_t | table | training s (steps) | evaluation s |
|---|---|---|---|---|---|---|---|
| **`simple_official`** | `submission_simple.py` byte for byte (md5 `f374ec70…`), the official manifest untouched, H100 | **64 / 64** | 768/768 at every rung, both profiles | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200; AUDIT `eval_rows_used=200/200 used_but_never_trained=0` (every row trained ≥ 2.9 million times) | 300.7 (3,000) | 23.1 |
| `simple_s0_official` | the same file with the exact table (`MODE = "designed"`, `MAX_STEPS = 100`): the plumbing check | 64 / 64 | 768/768 everywhere | all exact | designed | 3.4 (100) | 15.8 |
| for comparison: `adder_warm_official` | `submission_adder.py` | 64 / 64 | 768/768 everywhere | all exact | 200/200 | 77.8 (frozen at 343 of 3,000) | 12.8 |

The official run's log (one line per 250 steps): the check and the three facts are exact on every row from step 250
to step 3,000, and the loss falls from 0.026 to 0.0004 with no event. Without a freeze the training takes 301 s
instead of 78 s, because every step does work; the budget is 3,600 s.

On CPU through the same evaluator (manifest device cpu, the seed changed), the same training (`simple_LMWnosched_s*`,
the ablation copy with these settings, bit-identical to the file) certified on all 8 seeds, 74 and 1–7: T = 64 on
both profiles, every rung 768/768, the table 200/200, `used_but_never_trained=0` on every seed, training 668–954 s
for 3,000 steps, evaluation 17–25 s. With the old schedule kept (`simple_LMW_s*`) the same three lessons froze at
260–305 on the same 8 seeds. The file itself (`simple_file_cpu_s1`–`s4`) certified on seeds 1–4 as well, T = 64 on
both profiles with clean audits, and through the evaluator it reproduces its twin exactly: the same loss at every
logged step (13 per run) and the same final table on every seed.

**How short can the warm start be now?** ADDER.md needed 50–70 steps of the hand-written adder, with R − 0 = R and
R − R = 0 planting traps that only it could clear. Without them:

<!-- TABLE:warm -->
| warm start | certified, frozen at, failures |
|---|---|
| hand-written adder for the first 0 steps | 0/2; failed: 21/200 (copy the minuend 70), 21/200 (copy the minuend 36) |
| hand-written adder for the first 10 steps | 0/2; failed: 91/200 (borrow-free 46), 89/200 (borrow-free 42) |
| hand-written adder for the first 20 steps | 0/2; failed: 97/200 (borrow-free 49), 100/200 (borrow-free 51) |
| hand-written adder for the first 30 steps | 0/2, frozen 561; failed: 98/200 (borrow-free 49), 100/200 (borrow-free 50) |
| hand-written adder for the first 50 steps | 2/2, frozen 285–328 |
| 70 steps (the recipe) | 2/2, frozen 280–305 |
<!-- /TABLE -->

So the warm start cannot be shortened by much: 50 steps are enough on both seeds, 30 or fewer on neither (ADDER.md's
full recipe: 50 on 2/2, 30 on 1/2). One failure is worth spelling out. With 30 warm steps, seed 1 froze at step 561
with every check exact for 200 steps, on a table with only 100 of 200 entries right. Its digit is r − m − [m even],
the borrow coming in is ignored, and all 200 borrows are true. On its own numbers this table passes the learned-adder
check on every link and all three facts on every row, while it fails the hand-written adder's check (0.5%), the
answers (0%) and R − 0 = R (0%). It is a consistent wrong table that none of the simplified recipe's later lessons
can see (Rule 1), so the warm start has to bring the table past it. `submission_simple.py` has no freeze, but a run
stuck there would stay stuck: nothing would pull it out. The full recipe's R − 0 = R rejects it, which is the one
real use of that anchor (§8: rippled with the hard borrows it keeps that use and loses its harm).

## 11. The combined model

[`submission_combined.py`](../learned_remainder/submission_combined.py) (md5 `4f5cecc7…`, certified as
`combined_official`, COMBINED.md) learns every working part at once: reader and writer, times table, subtraction table,
commit gate. Its recipe carries the core's ingredients for the subtraction table, with the timing changed where the
pieces meet, plus the other pieces' own.

**Inventory** (constants of the file; "why" from the write-up that added it):

| part | ingredient | constant | why (source) |
|---|---|---|---|
| table | the core's check, complete identity, lowest place ×5, R − 0 = R, R − R = 0, 0 + N = N, R + N = N + R, x − N borrows, N − x does not, (x − N) − N does not, 0 − N borrows | as in §1 | §1 |
| table | the hand-written adder until the reading is chosen (and at least 70 steps) | `ADDER_WARM`, `WARM_TO_FACTS` | COMBINED §4.1: the learned adder drifted for the 20–155 steps it ran before the facts could join |
| table | the facts about x and N join the step the reading is chosen | `FACTS_FROM = "reading"` | COMBINED §2: they are true only of the true reading |
| table | phase B (Adam reset) at 80% of the checks, after the reading | `ID_SWITCH`, `STAGE_WIN`, `ID_STEPS` | LEARNED_REMAINDER §4.2 |
| gate | x mod N = x and N mod N = 0 on a detached table, trial mixture | `SELF_LESSON`, `NMOD_LESSON`, `FACT_TABLE`, `OWN_GATE` | CONTROL §5–6: the facts must teach only the gate |
| gate | phase C once x − N borrows on 99% | `CMP_SWITCH`, `CMP_STEPS` | CONTROL §6 (d1–d4) |
| times table | the ladder X·(b + 1) = X·b + X along the verified frontier, joining once the table's checks are exact | `LADDER_W`, `LADDER_WIN`, `LADDER_STEPS` | SQUARE §5; COMBINED §2: wrong early targets leave the carry blur |
| times table | the learned adder's two ripples as one chain over the joint borrow (soft rows) | (code) | COMBINED §4.2: two independent ripples stall the ladder at 29–59/100 |
| answers | rows with T ≤ 2 whose count-down, squarings and ladder check out, along the own hard path | `REAL_MAX_T` | SQUARE §5 (per-row filter); LEARNED_REMAINDER §3.3 |
| reader | five stages: digit-ness, ranked codes (Bradley–Terry on x < N), the reading mixture, tiling, T's direction by speculation | `BT_TAU`, `PICK_PASS`, `PICK_WIN`, `CHOSEN`, `T_AFTER`, `T_ROWS`, `T_DONE` | IO §7 |
| all | freeze after 200 exact steps in phase C, T's direction settled | `FREEZE_WIN` | LEARNED_REMAINDER §3.1 |
| prediction | gradient damped 10× per link (rule 8, not training) | `GRAD_DAMP` | COMBINED §3 |

**The decisive ablations repeated** (seeds 74, 1, 2, 3 or 74, 1; the same evaluator path, `combined/
submission_combined_ablate.py`, which trains exactly as the certified file at its defaults: `local_equiv_combined.py`):

<!-- TABLE:combined -->
| combined model: removed | certified, frozen at, failures |
|---|---|
| R − 0 = R, R − R = 0 off | 4/4, frozen 515–594 |
| wrap off | 4/4, frozen 561–582 |
| the simplified table recipe (also 0 − N, 0 + N = N, R + N = N + R off; digits-only check, places alike) | 3/4, frozen 573–2287; failed: 91/200 (mod-5 borrows 60) |
| the learned check off after the warm start (its hard version still gates) | 0/2; failed: 183/200 (borrow-free 91), 192/200 (borrow-free 100) |
| complete identity and lowest-place weight off | 2/2, frozen 564–601 |
| R − 0 = R, R − R = 0, wrap off | 2/2, frozen 518–617 |
<!-- /TABLE -->

For comparison, the certified file itself froze at 545, 568, 567 and 567 on seeds 74, 1, 2, 3 (`combined_cpu_seed*`,
COMBINED.md's CPU plan; 8/8 seeds, 535–714).

What carries over from the core, and what does not:
- **R − 0 = R and R − R = 0 can go here too** (4/4, frozen 515–594), and so can the complete identity with the
  lowest-place weight (2/2). They are not needed in either model.
- **The wrap-around fact is not needed in the combined model** (4/4, frozen 561–582), unlike in the core (0/4). The
  combined model keeps the hand-written adder until the reading is chosen, steps 90–225 across seeds (COMBINED.md
  §4.1), instead of 70. ADDER.md §5.1 already found that 200 warm steps fix the run-of-9s family with the hand-written
  adder. So the wrap-around fact and the length of the warm start cover for each other: a pair that only shows up when
  two pieces meet.
- **The core's simplified table recipe does not carry over.** With no anchors on the registers, no 0 − N, 0 + N = N
  or R + N = N + R, and the plain digits check, the combined model certifies on 3 of 4 seeds, and one of those took
  2,287 steps instead of 568. Seed 3 settled by step 350 nearest the "borrows through the mod-5 part of each digit"
  table of ADDER.md §3 (60–75 of that table's entries matching, 91–105 of 200 right) and stayed there until the
  3,600 s training budget ran out (step 6,995).
  The reason is timing. In the combined model the facts about x and N can join only once the reader has chosen a
  reading (steps 90–225). Until then the hand-written-adder check runs alone, and alone it is the core's bare check,
  which the add-one-in runs of §4 showed falling into such wrong number systems. In the full combined recipe, the
  lessons that need no reading (R − 0 = R, R − R = 0, the complete identity, the lowest-place weight) hold the early
  table. Each can go on its own (above), but not all together. So ingredients that are dead weight in one piece carry
  weight where pieces meet, and here the reason is that the reader delays the facts.
- **The learned-adder check is load-bearing in the combined model**, though dead weight in the core. Without it,
  both seeds fail: the (r, r, borrow 1) family that R − R = 0 plants (§6.2) stays wrong, digit 0 instead of 9, from step
  200 until the training budget runs out (192 and 183 of 200). In the core, x mod N = x and the answers repaired that
  family in phase B (§5). In the combined model x mod N = x teaches only the commit gate, on a detached copy of the
  table (CONTROL.md), and the answers wait for the times table's ladder, which waits for the table's checks to be
  exact. So the learned check is the only lesson left that pulls the family in.

The reader's, the gate's and the ladder's own ingredients were not ablated here: their write-ups each tested them
(IO.md §7, CONTROL.md §6, SQUARE.md §5), and the core's findings say nothing new about them. One of them is an instance
of Rule 3 already: COMBINED.md §4.2's joint-borrow soft adder fixes a soft lesson that treated tied digits as
independent.

## 12. Reproduce

Everything heavy runs on Modal. Deploy the runner once from the one-layer-deeper repository root (it builds the
competition image from there and regenerates H1 inside it with the official generator arguments; the image is the
pieces' own, so the build is cached):

```bash
~/github/sotaku/venv/bin/python -m modal deploy /path/to/cpu_first/ablation/modal_ablate.py
```

Write the plans, launch one, and wait for its records (the waiter polls the Volume `ablation` every 60 s and exits
when every record is in):

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python make_plans.py
```

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python launch.py plans/w1_loo.json
```

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python fetch.py plans/w1_loo.json
```

Tables and trajectories:

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python analyze.py variants loo
```

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python analyze.py traj loo_noWrap_s74
```

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python table.py
```

The local checks (each under 2 s or a minute, under 1 GB; they need torch and the H1 jsonl files, `H1_DATA` as in
`cpu_first/h1.py`):

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python local_equiv.py
```

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python probe_trap.py 001
```

```bash
cd cpu_first/ablation && ~/github/sotaku/venv/bin/python probe_grad.py 2.5
```

When done, stop the deployed app (it idles at zero containers, but costs nothing to keep or to stop):

```bash
~/github/sotaku/venv/bin/python -m modal app stop ablation
```

Files in `cpu_first/ablation/`:
- **`submission_simple.py`**: the simplified recipe, certified byte for byte as `simple_official`.
- **`submission_ablate.py`**: `submission_adder.py` with an off-switch per ingredient, the general rules, and table
  logging; at its defaults it trains exactly as the certified file (`local_equiv.py`).
- **`combined/submission_combined_ablate.py`**: `submission_combined.py` with `ID_AFTER_WARM` and table logging; at its
  defaults it trains exactly as the certified file (`combined/local_equiv_combined.py`).
- **`modal_ablate.py`, `launch.py`, `fetch.py`**: the runner (deployed app `ablation`, Volume `ablation`), the launcher,
  the waiter.
- **`make_plans.py`, `plans/*.json`**: every wave: `w1_loo` (leave one out), `w2_aoi` (add one in), `w3_pairs_small`,
  `w3b_wrap`, `w5_zeroself` (pairs), `w4_general`, `w6_hardcin`, `w12_general_simple` (general rules), `w7_simple`,
  `w8_facts`, `w9_nowrap`, `w11_simple_loo` (small recipes), `w10_warm` (warm-start length), `h1_simple` (the H100
  runs), `h2_simple_cpu` (the simplified file on CPU), `c1_combined` (the combined model).
- **`runs/<name>.json`**: one record per run (RESULT_JSON, the full log with the table every 50 steps, the exact
  source, the manifest, the job's settings); `runs/launch_*.log`: the call ids; `runs/fetch_*.out`: one line per record.
- **`analyze.py`, `table.py`**: the tables of this file (`table.py fill ABLATION.md` rewrites them from the records).
- **`probe_trap.py`, `probe_grad.py`**: §6.
- **`local_equiv.py`, `local_equiv_simple.py`, `combined/local_equiv_combined.py`**: the equivalence checks.

The certified files are untouched: `submission_adder.py` (md5 `44e658ab…`), `submission_chain.py` (`ed360cb1…`),
`submission.py` (`bd477b9a…`), `submission_square.py` (`37765051…`), `submission_control.py` (`ec79a312…`),
`submission_io.py` (`45cc6e1b…`), `submission_combined.py` (`4f5cecc7…`), `modal_chain.py` (`ad5fbd9c…`).
