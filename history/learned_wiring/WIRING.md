# The H1 machine learns its own wiring

## In plain words

**The task.** The model reads three numbers written as digits in a prompt: a modulus N, a starting number x and a count
T. It must square x, keep only the remainder after dividing by N, repeat that T times, and write the result's digits.
To be certified it must get every one of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, for moduli it saw in
training and for larger moduli it never saw (up to 8 digits). Training only shows T = 1, 2, 4, 8 and 16.

**Where this started.** [COMBINED.md](../../learned_remainder/COMBINED.md) built a small calculator for this whose every
computing part is a learned table: digit products, digit subtractions (which also do every addition), a two-entry
keep-or-discard switch, a reader and a writer. What stayed hand-written was the **wiring**: which part feeds which, in
what order. That is the paper layout of long division and schoolbook squaring, and the shape of the reader. Under a
strict reading of the rules ("no hard-coded algorithm in the forward pass") the wiring is the algorithm.

**What the machine now learns about its own wiring.** The hand-written connections of the long division, the
squaring and the reader are now switches. A switch is a small menu of options, one of which is the right one. Each
switch starts at random and is learned in training, alongside the tables. In three stages:
1. **The long division** (stage 1). Seven switches:
   - which end of the square the division starts from;
   - how the running remainder makes room for the next digit (move up, move down, overwrite the lowest place, ignore);
   - what each subtraction subtracts from (the running number, the step's starting number, or N);
   - what it subtracts (N, the running number, or the step's start);
   - what a subtraction's result does (always replace the running number, replace it when the keep-or-discard switch
     says so, or never);
   - whether N's digits are used as read or reversed;
   - which places of the register are the answer.

   Together that is 864 wirings.
2. **The schoolbook squaring** (stage 2). Where each digit product goes: one placement rule for the products' low
   digits and one for their high digits, each chosen from 24 rules. The column of the product of digits i and j is
   "i or its mirror image, plus j, its mirror image or nothing, plus a shift of −1 to 2". That is 576 layouts.
3. **The reader and the writer** (stage 3). The shape of the pointer that picks a number's digits out of the prompt,
   and the answer's places: one anchor and a shared step, or an anchor and a separate offset for every digit, or a
   separate fixed position for every digit. One switch serves all four fields.

**How a switch learns.** Each switch is taught by a check that the right wiring passes and wrong ones fail. The check
runs on the training questions' own numbers and needs no answers:
- **the division's inner wiring**: start one division step from "x plus q copies of N" (q = 0 to 9). It must give
  back x. This is the division taken one step at a time;
- **the division's start, shift and readout**: dividing x by N must leave x, and dividing N by N must leave 0, through
  the whole division. The training answers of the T = 1 and 2 questions (2% of training) join in, but are not needed:
  in the small test machines the switches were found without any answer;
- **the squaring's placement**: X times digit b + 1 must be X times b plus X, row by row (the "ladder");
- **the reader's shape**: the reading under which "x is smaller than N" holds on every question (the reader's existing
  lesson), scored for each shape. Which way T's digits are read is settled last, by the reader's existing lesson too:
  the finished machine runs from both readings, and the one that reproduces the training answers wins.

For the division, every combination of the switches' options is tried on the machine itself (with its tables
snapped to their best entries), and the combinations that pass gain weight. For the squaring and the reader, the
candidate rules and shapes are blended, and the blend's probability of a right row or a right reading is what training
raises. The switches start learning once the tables they use are exact. Then they are found in a few steps: the
division's inner switches within 1 to 4 training steps in every run.

**What is still hand-written.**
- **The menus.** Each switch chooses among options that I wrote, and the right wiring is one of them. The algorithm
  now lives in the menus, the grammar, not in one fixed layout. The machine does not invent long division; it picks
  long division's connections out of a few hundred to a few thousand alternatives.
- **The sizes**: 8 division steps of up to 9 subtractions, a 9-place register, 8 squaring rows, 8, 8 and 2 digits read,
  8 written.
- **The loop around it**: the count-down of T and "run rounds until no question wants another".
- **The adder's own wiring**: a + b is computed as a − (0 − b) with the subtraction table.
- **The lessons and their schedule**: what each table and switch is checked against, and in which order they join.

**Did it certify?** Yes, all three stages. Each certified on the untouched official H1 manifest on an H100, the file
run byte for byte, with T = 64 on seen and unseen moduli and every rung 768/768. Each was repeated on 8 CPU seeds
through the same evaluator.

| stage | what is learned on top of COMBINED.md's tables | official H100 run | CPU seeds certified | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|
| 1 | the division's 7 switches | certified | 8 of 8 (first version: 7 of 8) | 314 (615) | 37 |
| 2 | + the squaring's layout | certified | 8 of 8 | 378 (479) | 49 |
| 3 | + the reader's and writer's shape | certified | 8 of 8 (first version: 6 of 8) | 729 (532) | 52 |

Every stage fits the budget (3,600 s of training, 1,800 s of evaluation), so none needed a relaxed run. Stage 3 takes
longer because T's lesson runs whole count-downs while it settles T's reading direction (§6). Two files have a
first version, kept with its records:
- stage 1's first version (`submission_div_v1.py`) certified on the official run but failed one CPU seed: its
  untrained reader read every N as 0, a fault already present in the combined recipe (below);
- stage 3's first version (`submission_io3_v1.py`) certified on the official run but failed two CPU seeds: its T
  lesson pulled the shared shape switch off the right shape, and the machine ended up reading one-digit T wrong
  (below).

**What broke along the way.** In the order the fixes were found:
- **The table needed checks that do not depend on the wiring.** Checking "every subtraction any option might make"
  sounds safe but fails: the wrong options make lopsided numbers (x − x, 0 − x) and the table settles on a wrong
  answer for them. The table is now checked on the question's own numbers (x and pieces of x²) minus N, again and
  again, whatever the wiring is.
- **Switches taught by blending options fail.** If every switch passes on a blend of its options' digits (the usual
  "soft" switch), training settles on a machine that does nothing (it never subtracts), even with every answer given.
  Trying whole combinations and scoring each with a pass/fail check does not.
- **A latent fault in the old recipe**: before the reader has learned anything, it can read every N as 0. The table's
  warm-up check then only ever subtracts 0, and the warm-up ends before it has taught anything. Two of four seeds hit
  it in stage 2's first runs, and one of eight CPU seeds of the certified stage-1 file (seed 7). The warm-up now runs
  70 steps from the moment the reader works (stage 1's file was re-certified with it; the first version is kept).
- **A placement learned column by column does not generalize.** Training numbers have at most 7 digits, so the
  columns that hold the 8th digit's products are never exercised. The squaring then fails on 8-digit numbers. A
  placement rule shared by every digit (stage 2's grammar) covers the 8th digit by construction.
- **Rules whose parts are weighed apart co-adapt.** A rule's three parts (digit order, row order, shift) learned as
  three separate switches stall in a wrong combination; one weight per whole rule does not.
- **Shapes chosen field by field** let the short T field (mostly one digit) settle on a per-digit shape that cannot
  read two-digit T. One shape switch shared by all four fields is learned right every time.
- **A one-digit number has no direction.** Generalizing "which run does this pointer read" to every shape first
  checked only that the slots sat on the run's digits. A one-digit x then matched both reading directions, and
  those rare rows (x < 10) picked a reading hypothesis with an untrained digit code. The writer's code got conflicting
  lessons, wavered on rare steps, and training never reached its last stage. Now the slot right past the number must
  sit on the reading direction's side, as the whole-reading lesson already assumed.
- **A lesson's side effects matter.** When the writer's lesson was rewritten so that all three shapes are scored
  alike, it lost a side effect of the old one: a strong push for the answer's symbols to count as digits. That push
  is now its own lesson.
- **A check that compares two options can reward a third that has not decided.** T's lesson asks which of the two
  directions of reading T reproduces the answers, and raises that one's share. Through the shape switch, shared by
  all four fields, it could also raise the share by moving toward an untrained shape, which leans neither way. In one
  diagnostic run the whole reader moved to that shape, stopped reading for a while, then stuck halfway between the
  two shapes with T never settled. Two CPU seeds of stage 3's first version ended on the other shape and read every
  one-digit T wrong (that version still certified on the official run; it is kept with its records). T's lesson now
  leaves the shape switch alone.

## Rule by rule: 7, 8 and 14

The rules are [one-layer-deeper/README.md](../../../one-layer-deeper/README.md) §Rules. COMBINED.md's table covers rules
1–16 for the combined machine. Here only the three rules this work changes. "Gray" means a reasonable reader could
object; the point is flagged, not argued away.

| rule | before (COMBINED.md) | now | still gray |
|---|---|---|---|
| **7. no hard-coded algorithm in the forward pass** | gray, the main one: every number is computed by learned tables, but the forward pass is a designed CPU | **less gray**: which register feeds which port, the shift, the write-back, the readout, the squaring's placement and the reader's shape are learned switches. The forward pass at evaluation is the machine those switches select. | the menus are designed and contain the true wiring; the sizes, the T loop, the adder's composition a − (0 − b) and the order of the squaring's summation are fixed. A strict reader can say the algorithm moved into the grammar. |
| **8. logits inside autograd, unbroken gradient path** | yes, with gray edges (straight-through tables, discounted gradient) | **the same, extended to every switch**: on the prediction path every switch is a straight-through one-hot, so the forward pass is exactly the snapped machine and the final logits' gradient reaches every switch's logits (tested at the truth and on trained models, §5) | the same edges as before. And: a switch's gradient arrives only through the places where its options differ. On some batches those are only deep places, whose 10× per-link discounted gradient underflows to 0 in fp32: in a first form of stage 2's layout (a rule's three parts as three switches) one switch got exactly 0 on both test batches. With one weight per whole rule (the files) every tensor gets a gradient (§5). |
| **14. no data inspection, task-specific solvers, custom loops, participant backward** | gray: the lessons carry designed knowledge | **gray in the same way, with new items**: the division ladder states the definition of division with remainder (x + qN leaves x); "x mod N = x", "N mod N = 0" through the whole division; the routed ladder states what a schoolbook row is (row j adds X·10^j); the switches' pass/fail verdicts come from running the hard machine on candidate wirings inside the loss, under no_grad (no backward or grad call; the evaluator does the one backward pass) | as before; and the candidate wirings scored in the loss are a search over the menu, done by gradient on mixture weights, not by an outer loop |

## Details

### 1. Results

Every official run is the untouched official manifest `benchmark/manifests/h100_hard_h1.json` on a Modal H100 (seed 74,
3,600 s training and 1,800 s evaluation budgets), with the submission file sent byte for byte (`modal_wiring.py`, a
copy of the pieces' launcher). Every CPU seed is the same evaluator with the manifest's device set to cpu and the seed
changed (`plans/<stage>_cpu.json`). The records are `cpu_first/history/runs/learned_wiring/<name>.json`: RESULT_JSON,
the full log, the exact source and the manifest. The tables of each stage (§2.5, §3.5, §4.5) are generated from them by
`fill_wiring.py`.

<!-- RESULTS -->
| stage | file (md5) | official H100 run: max T seen / unseen | training s (frozen at step) | evaluation s | CPU seeds certified (T = 64 on both profiles) | CPU seeds: frozen at step | CPU training s | CPU evaluation s |
|---|---|---|---|---|---|---|---|---|
| 1 (first version) | `submission_div_v1.py` (`a0050a2f`) | 64 / 64 | 393.4 (640) | 48.5 | 7/8 (seed 7 failed) | 610-741 | 554-876 | 192-550 |
| 1 | `submission_div.py` (`11cccf13`) | 64 / 64 | 313.6 (615) | 36.8 | 8/8 | 606-743 | 517-890 | 337-542 |
| 2 | `submission_sq.py` (`781928ea`) | 64 / 64 | 377.7 (479) | 49.0 | 8/8 | 471-543 | 656-928 | 374-546 |
| 3 (first version) | `submission_io3_v1.py` (`070d2ee5`) | 64 / 64 | 729.4 (532) | 48.9 | 6/8 (seed 2, 5 failed) | 489-543 | 645-937 | 416-585 |
| 3 | `submission_io3.py` (`c2331362`) | 64 / 64 | 728.6 (532) | 51.8 | 8/8 | 489-543 | 580-1060 | 380-559 |
<!-- /RESULTS -->

In every record that certified:
- every rung of both profiles is 768/768, and test (9,999 rows), ood_t and ood_n_t (10,002 each) are exact;
- the snapped tables are the true ones: 200/200 subtraction rows, 100/100 product cells, the gate committing on "no
  borrow";
- the switches chose the true wiring: the division's 7 switches, the squaring's two placement rules (stage 2 on), the
  step shape (stage 3);
- `AUDIT ... used_but_never_trained=0`: every table row, product cell and gate row used at evaluation was used by a
  training lesson.

### 2. Stage 1: the division's wiring

#### 2.1 The switches

The combined machine's division ([CHAIN.md](../../learned_remainder/CHAIN.md),
[CONTROL.md](../../learned_remainder/CONTROL.md)) takes the 16 digits of s², starts its 9-place register with the top 8,
and runs 8 steps. Each step brings in the next digit and runs a chain of 9 subtractions of N with the learned table; the
learned 2×2 gate reads each subtraction's final borrow and keeps or discards its result. The answer is the register's
low 8 places.

Its wiring as tied switches, one per kind of connection, shared by every step, link and round (TINYROUTING.md's
grammar on the gated chain; `DIV_MENU` in `submission_div.py`):

| switch | options (in the file's order) | true |
|---|---|---|
| `vread` | the dividend's digits as given / reversed | given |
| `shift` | the step's register from R and the incoming digit d: down (R / 10 + d·10⁸) / stay (d replaces place 0) / up (10 R + d) / skip (R) | up |
| `read` | the remainder: the register's high 8 places / low 8 places | low |
| `nread` | the subtrahend register: N's read digits reversed / as given | given |
| `a` | each link's minuend: the step's start / N / the chain register | the chain register |
| `b` | each link's subtrahend: the chain register / the step's start / N | N |
| `keep` | what a link writes into the chain register: always / its result when the gate commits / never | when the gate commits |

864 wirings, times the gate's 4 tables. No option list starts with the true option, and every lesson treats the
options alike; the true wiring is used only by the plumbing check (`MODE = "designed"`) and the reports.

#### 2.2 What teaches them, and why that way

**H1 has answers for 2% of its training questions.** Only the T = 1 and T = 2 questions (1,000 each of 101,999) have
answers that one or two rounds of the machine reproduce; the others need 4 to 16 rounds. TINYROUTING.md's tiny cores
learned their switches from the answers of every question, through a soft machine, one quotient digit at a time.
H1 cannot do that: the questions whose square has few quotient digits are rarer still, and making up dividends would be
a new kind of data. So the switches are taught by checks that need no answer:
- **The division ladder** (the chain's switches `nread`, `a`, `b`, `keep` and the gate's two rows). From the register
  x + N·q, q = 0..9, one division step must give back x. N·q is the machine's own ladder row (stage 2's routed row 0)
  and the sum is the learned adder's. This is the division one step at a time: a one-step problem whose quotient digit
  is q and whose remainder is known. It is TINYROUTING's condition 3, "a law that reads the switch directly".
- **The I/O laws** (`vread`, `shift`, `read`). x mod N = x and N mod N = 0 through the whole 8-step division (x and N
  padded with zeros to 16 digits), plus the answers of the T ≤ 2 rows through the whole machine.

**The verdicts are blended as a mixture of whole wirings, not on the wire.** Every combination of the chain's
switches (3·3·3·2·2·2 = 216 with the gate's rows) runs the step on the hard machine (snapped table), and each
combination's pass/fail verdict is weighted by the product of the switches' probabilities. The loss is
−log Σ weight·verdict. The I/O switches do the same over their 16 combinations, with the chain at its argmax. That is
TINYROUTING's factorized mixture with exact verdicts. TINYROUTING §6 found that a law is a crisp test of a whole wiring,
but in a blend on the wire it rewards the half-and-half do-nothing wiring. The tiny core (§2.4) confirms both halves.

**The switches wait for the tables.** The chain lesson joins once the table and the product cell have been exact for 20
steps (`WIRE_WIN`), and the I/O lesson once the argmax chain has passed the division ladder for 20 steps. Before that
every verdict is 0, so the mixture's gradient is exactly 0: the lesson is harmless early (`notablefirst` in §2.4).

**The gate is part of the chain's mixture.** COMBINED.md taught the gate with "x mod N = x" and "N mod N = 0" through the
hand-wired chain on a detached table. Those facts are the q = 0 and q = 1 rungs of the division ladder, which now
teaches the gate through the switched chain. The answers along the hand-wired own path are gone too (ABLATION.md found
them dead weight).

#### 2.3 The table under a wiring it does not know

COMBINED.md checks the table on the links of the machine's own division, (D − N) + N = D. Under a learned wiring that
division is whatever the random switches make at first:
- **Every option's subtractions (the union).** In the tiny core (W = 3, base 10), checking the union of all nine
  (a, b) options' chains leaves the table at 315–330/400 entries (digit + borrow) with the hand-written adder, against
  397/400 for the true chain's links alone. The wrong options flood the check with x − x and 0 − x subtractions, and
  the table settles on "a borrowing digit is 9". All ordered pairs of the row's own numbers fail the same way
  (365/400).
- **The row's numbers minus N, again and again** (`CHECK_REGS = "rows"`): x and the P-place windows of the machine's
  own square of x, each minus N repeatedly for 9 links. These are numbers the wiring cannot change, and the chains
  subtract N like the true chain does. In the tiny core it teaches the table as well as the true chain (397/400). At
  H1 scale it does as well as the machine's own registers (`"own"`, d1 vs d2: 3 of 4 seeds froze in both, below).
- **Straight-through borrows.** The ablation (ABLATION.md rule 3) found that soft lessons leak through blurry borrows,
  and that rippling R − 0 = R and R − R = 0 with the hard machine's borrows (HARD_CIN) fixes their harm. Applied to every
  table lesson (`STE_CIN = "all"`), the tiny core improved (400/400), but at H1 scale seed 74's table collapsed after the
  hand-written adder handed over (≤ 180/200 at the handover, 90/200 after) in both register variants. Applied only to the two anchors
  (`"anchors"`, the ablation's scope) or to none, seed 74 trains: the file uses `"anchors"`.

**Why the combined recipe's table lessons and not the simplified one.** ABLATION.md's simplified recipe
(`history/ablation/submission_simple.py`: the check, x − N borrows, N − x does not, the wrap fact) certifies the core
machine. In the combined model it failed 1 of 4 seeds (ABLATION.md §11). Before the reader has chosen a reading, the
facts about x and N cannot join, and only the lessons that need no reading hold the early table: R − 0 = R, R − R = 0,
the complete identity, the lowest-place weight. The stage files keep them, R − 0 = R and R − R = 0 rippled with the hard
borrows (ABLATION.md §8's fastest full recipe), and drop what the switches replace: the gate's facts and the answers
along the hand-wired path. The reader's delay bites in a second way too (§3.4): before the reading, the reader can read
every N as 0.

| H1 diagnostic (CPU, harness `local/local_wiring.py`, 2,500 steps max) | seeds 74 / 1 / 2 / 4 |
|---|---|
| d1: own registers, straight-through on every lesson | collapsed (91/200) / froze 648 / 702 / 603 |
| d2: the row's numbers, straight-through on every lesson | collapsed (90/200) / froze 705 / 693 / 616 |
| d3: the row's numbers, straight-through on none | froze 616 / – / 673 / – |
| d3: the row's numbers, straight-through on R − 0, R − R (the file) | froze 639 / – / 692 / – |

Every run that froze has the true wiring and both tables exact. It reproduces every one of 128 prompts of all 14 depth
rungs (both profiles) on both prediction paths with identical logits, and passes the gradient test.

#### 2.4 The tiny core

`tiny/tiny_div.py`: the same machine at base 10 with N of 2 and 3 digits in 4 slots (P = 5 places, 4 division steps,
9 links), x < N, dividend x², 2% of rows labeled, certified on the whole training pool and on 4-digit N (never trained).
The table's lessons are the H1 recipe in miniature (the check on the row's numbers minus N, the facts, the
hand-written adder for 70 steps then the learned one). Suite `s1b`, 6 or 4 seeds per row (2 for the diagnostics):

<!-- TINY_S1 -->
| variant | certified | true wiring | table exact at step | chain switches right after (steps) | I/O switches right after (steps) | failures (table entries right / 400) |
|---|---|---|---|---|---|---|
| the recipe (mixtures with crisp verdicts) | 5/6 | 5/6 | 57–163 | 22 | 51–54 | 385 |
| I/O switches as one weight per combination (16) | 3/4 | 3/4 | 62–131 | 22 | 51–54 | 379 |
| the chain's switches soft on the wire (division ladder) | 4/4 | 4/4 | 57–163 | 53–81 | 51–130 | – |
| no labels at all (the laws only) | 4/4 | 4/4 | 57–163 | 22 | 51–53 | – |
| the I/O switches from the 2% answers only | 4/4 | 4/4 | 57–163 | 22 | 51–54 | – |
| the hand-written adder throughout | 2/2 | 2/2 | 57–166 | 22 | 51 | – |
| the chain lesson from step 1 (no waiting) | 2/2 | 2/2 | 57–137 | 20 | 51 | – |
| no straight-through borrows | 1/2 | 1/2 | 83 | 22 | 51 | 374 |
| the check on every option's subtractions | 1/2 | 1/2 | 61 | 22 | 51 | 261 |
| every switch soft on the wire, through the whole division (2% labeled) | 0/4 | 0/4 | 57–163 | – | – | 400 (table right; wiring wrong) |
| the same with every answer labeled | 0/4 | 0/4 | 57–163 | – | – | 400 (table right; wiring wrong) |
| the I/O switches soft on the wire (the chain's mixture right) | 0/4 | 0/4 | 57–163 | 22 | – | 400 (table right; wiring wrong) |
<!-- /TINY_S1 -->

- **Every failure of the mixtures is a table that never became exact.** The tiny core's table recipe (70 steps of the
  hand-written adder from step 1, P = 5) is weaker than H1's. With a right table, the chain's switches were right 22
  steps after their lesson joined (20 of them the waiting window), and the I/O switches 51–54 steps later.
- **The answers are optional.** With no labeled row the laws alone find every switch, in the same number of steps; with
  the 2% answers alone (no laws) the I/O switches are found too.
- **Soft on the wire, through the whole division, fails in every run**, with the table exact, including with every
  answer labeled. The end states are the do-nothing machine: `keep = never` or a gate that never commits (10.5–10.9% of
  answers right: the questions whose square is already below N). With only the I/O switches soft, the end states read
  the high places, or shift wrong (TINYROUTING's co-adaptation of reading, shift and readout).
- **Soft on the wire works for the chain's switches when the lesson is one step** (the division ladder): 4/4, a little
  slower. TINYROUTING's condition 3 again: the problem is depth and co-adaptation, not softness as such.

#### 2.5 The H1 runs

<!-- STAGE1 -->
| run | seed | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | subtractor, product cell, gate | division wiring | layout | shape | reading chosen | ladder joins | chain lesson joins | I/O lesson joins | T settled | frozen | never trained | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| div_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 151 | 242 | 414 | 437 | 461 | 640 | 0 | 393.4 | 48.5 |
| div_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 151 | 242 | 414 | 437 | 461 | 640 | 0 | 554.2 | 430.9 |
| div_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 133 | 234 | 417 | 440 | 463 | 642 | 0 | 566.7 | 192.5 |
| div_cpu_seed2 (CPU) | 2 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 214 | 318 | 515 | 539 | 605 | 741 | 0 | 865.3 | 530.0 |
| div_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 118 | 213 | 384 | 407 | 472 | 610 | 0 | 562.1 | 345.3 |
| div_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 175 | 272 | 457 | 479 | 545 | 681 | 0 | 875.6 | 549.7 |
| div_cpu_seed5 (CPU) | 5 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 170 | 268 | 438 | 459 | 482 | 661 | 0 | 737.9 | 446.9 |
| div_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 232 | 331 | 514 | 536 | 606 | 740 | 0 | 869.2 | 420.2 |
| div_cpu_seed7 (CPU) | 7 | None / None | 0 / 0 | 0.0, 0.0, 0.0 | 100/200, 39/100, 10 | a=N b=N keep=none nread=reversed read=low shift=skip vread=given gate=CH true=False | - | step (designed) | - | - | - | - | - | - | 0 | 3600.3 | 1826.3 |
| div2_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 151 | 201 | 389 | 412 | 436 | 615 | 0 | 313.6 | 36.8 |
| div2_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 151 | 201 | 389 | 412 | 436 | 615 | 0 | 516.9 | 422.9 |
| div2_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 133 | 218 | 381 | 404 | 427 | 606 | 0 | 755.3 | 541.7 |
| div2_cpu_seed2 (CPU) | 2 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 214 | 275 | 472 | 496 | 564 | 698 | 0 | 552.6 | 337.1 |
| div2_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 118 | 230 | 409 | 432 | 496 | 635 | 0 | 847.4 | 538.8 |
| div2_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 175 | 278 | 464 | 486 | 552 | 688 | 0 | 890.5 | 465.7 |
| div2_cpu_seed5 (CPU) | 5 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 170 | 224 | 399 | 420 | 443 | 622 | 0 | 655.2 | 536.7 |
| div2_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 232 | 332 | 517 | 539 | 608 | 743 | 0 | 862.9 | 443.3 |
| div2_cpu_seed7 (CPU) | 7 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | - | step (designed) | 136 | 223 | 419 | 443 | 467 | 646 | 0 | 741.7 | 507.8 |
<!-- /STAGE1 -->

- **The first version (`div`) certified on the official run and on 7 of its 8 CPU seeds.** Seed 7 ran out of training
  time (3,600 s, at step 4,620) with the subtraction table at 100/200 rows and the product cell at 39/100. The
  switches' lessons wait for exact tables, so they never joined (the wiring in its row is where the switches drifted
  without them).
  Its untrained reader had read N as 0 on every row while the hand-written adder taught the table, so the table
  learned little more than "minus 0" (§3.4).
- **The fixed version (`div2`, the file) certified on the official run and on all 8 CPU seeds**, seed 7 included
  (frozen at step 646). It differs from the first only in when the hand-written adder hands over (`facts_from`).
- In every certified run the chain's switches were right 1–4 steps after their lesson joined (the I/O lesson joins
  once the chain has passed the division ladder for 20 steps: 21–24 steps later). The I/O switches followed within a
  few steps.
- The official run and CPU seed 74 follow the same schedule step for step: the evaluator's training is deterministic
  across the two devices here (in every stage).

### 3. Stage 2: the squaring's layout

#### 3.1 The switches

The combined machine's square ([SQUARE.md](../../learned_remainder/SQUARE.md)): row j holds the products of every digit
s_i with the multiplier digit s_j from the learned product cell (a, b) → (lo, hi). The low digit goes to place i, the
high digit to place i + 1, the two are added by the learned adder, and the 8 rows, shifted by j, are summed one by one.
Now each row is two numbers of C = 18 columns, its low-digit number and its high-digit number, added by the learned
adder; the 8 rows are summed the same way. The layout says which digit each column holds: column c of row j's
low-digit number holds the low digit of one pair (s_i, s_j), or nothing; the high-digit number likewise.

The grammar (`LAYOUT = "rule"`, `LAY_MENU`): one placement rule per output kind, shared by every pair. The column of
pair (i, j) is f(i) + g(j) + d with f ∈ {W − 1 − i, i}, g ∈ {0, j, W − 1 − j}, d ∈ {−1, 0, 1, 2}. That is 24 rules
per kind, with one weight per whole rule, and 576 layouts. The truth puts the low digit in column i + j and the high
digit in column i + j + 1. The rule decides "which digit pairs combine, and into which column": the pairs whose
digits share a column are the ones the rule sends there.

#### 3.2 What teaches it

- **The product cell's own ladder** (`cell_ladder`): a · 0 = 0 and a · (b + 1) = a · b + a on single digits, as
  two-place numbers [lo, hi], with the learned adder, along the verified frontier. It involves no layout
  (TINYROUTING's condition 1: every table taught by a check that ignores the wiring).
- **The routed ladder** (TINYROUTING §5, SQUARE.md's verified frontier): for X in the rows' x and N, every row j and rung
  b, the soft row_j(X, b + 1) against the hard row_j(X, b) + X·10^j. The soft row uses the soft cell and the soft
  layout, blended on the wire; the target uses the snapped cell, the argmax layout and the learned adder. It is trained
  where row j's rungs 0..b check out, 2 rows j per number per step, cycling (`LAYOUT_ROWS`). It reads the layout
  directly: a column that holds the wrong pair's digit breaks the row.

Both join when the table's checks are exact (COMBINED.md's ladder), and the division's lessons wait for the ladder to
pass every rung: the division ladder uses row 0 of N·q.

#### 3.3 Three grammars

| grammar | tiny core (`tiny_sq.py`, the cell learned too) | H1 (diagnostics) |
|---|---|---|
| a source switch for every column of every row (2·W·(2W + 2) switches of W + 1 options; at H1 288 of 9) | trained on every W-digit number: 12/12 certified (W = 3 and 4, `s2`). Trained on numbers one digit shorter (`s2b`, W = 4): 0/4. Each fits every training rung, then squares 9.5–10% of 4-digit numbers right | d5: 256/288 columns right on 4 of 4 seeds. The 32 columns that should hold pair 7's digits never matter in training (x and N have at most 7 digits), and the routed ladder stays at 99.91% of rung checks, so the division's lessons never start |
| one offset per output kind for every pair (TINYROUTING's tied rule) | 8/8 (`s2`); with numbers one digit shorter 4/4 (`s2b`) | (not run) |
| one rule per kind, its three parts as three switches | W = 3, numbers one digit shorter: stalled at 37/48 columns (local); W = 4: 2/4 (`s2b`; the other two square 5% of 4-digit numbers right) | (not run) |
| one rule per kind, one weight per whole rule (the file) | 8/8 (`s2b`: 4/4 on every 4-digit number, 4/4 on numbers one digit shorter) | d8: the true layout on 4 of 4 seeds, frozen at 477–551; the official run (§3.5) |

The tiny suites (`tiny/tiny_sq.py`: base 10, the subtraction table given exactly, the product cell and the layout
learned together from random, certified on 20,000 random W-digit squares; 4 seeds per row):

<!-- TINY_S2 -->
| layout | W | cell ladder | certified (square exact) | true layout | stopped at step |
|---|---|---|---|---|---|
| soft | 3 | no | 0/4 | 0/4 | - |
| soft | 3 | yes | 4/4 | 4/4 | 130-133 |
| tied | 3 | no | 0/4 | 0/4 | - |
| tied | 3 | yes | 4/4 | 4/4 | 93-95 |
| soft | 4 | no | 0/4 | 0/4 | - |
| soft | 4 | yes | 4/4 | 4/4 | 133-135 |
| soft (one row per number per step) | 4 | yes | 4/4 | 4/4 | 149-155 |
| tied | 4 | no | 0/4 | 0/4 | - |
| tied | 4 | yes | 4/4 | 4/4 | 93-95 |

Trained on numbers one digit shorter than at certification (`s2b`):

| layout | trained on numbers below | certified (W = 4 squares exact) | true layout | stopped at step | square exact (failures) |
|---|---|---|---|---|---|
| rule_mix | 10000 | 4/4 | 4/4 | 101-110 | - |
| rule_mix | 1000 | 4/4 | 4/4 | 146-149 | - |
| rule | 1000 | 2/4 | 2/4 | 120-168 | 0.05, 0.046 |
| soft | 1000 | 0/4 | 0/4 | 132-133 | 0.095, 0.098, 0.1, 0.1 |
| tied | 1000 | 4/4 | 4/4 | 101-104 | - |
<!-- /TINY_S2 -->

- **Without the cell's own ladder nothing works** (`s2`: every layout without it fails; with it every one certifies): with the routed ladder alone, the cell and the
  layout co-adapt. One fixed point is TINYROUTING §5.1's twin: low and high digits swapped in both the cell and the
  layout.
- **A layout learned column by column does not generalize to 8 digits.** H1's training numbers have at most 7 digits,
  so the columns that hold pair 7's digits are never exercised. IO.md §6 found the same for a pointer with an own offset
  per slot. A rule shared by every pair covers pair 7 by construction.
- **A rule's parts depend on each other.** As three switches (digit order, row order, shift) they lock into a wrong
  combination (tiny: 37/48 columns, stalled). Weighed together, as one weight per whole rule, they do not. That is
  TINYROUTING §2.1's grouping rule again.

#### 3.4 A latent fault: the reader reads N as 0

In stage 2's first H1 runs (d4) two of four seeds kept the subtraction table at 24–26/200 for 100 steps of the
hand-written adder, with every check passing. The untrained reader read N as 0 on every row (seeds 1 and 4: 100% of
rows). The check then subtracts 0, which any table with r − 0 = r passes. The warm start ends when the reading is
chosen (COMBINED.md's schedule), so the learned adder took over an untrained table, which then stalled near 100/200.
Stage 1's seeds read some nonzero N by chance (31–78% of rows read N = 0); stage 2's extra parameters shifted the
reader's random start. The fix (stages 2 and 3, and stage 1's second version): the hand-written adder runs for 70
steps counted from the moment the reading is chosen (`facts_from`). After it all four seeds' tables were exact (d5).
Stage 1's first version, already certified, then failed CPU seed 7 the same way (§2.5).

#### 3.5 The H1 runs

<!-- STAGE2 -->
| run | seed | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | subtractor, product cell, gate | division wiring | layout | shape | reading chosen | ladder joins | chain lesson joins | I/O lesson joins | T settled | frozen | never trained | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sq_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 123 | 173 | 253 | 276 | 300 | 479 | 0 | 377.7 | 49.0 |
| sq_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 123 | 173 | 253 | 276 | 300 | 479 | 0 | 747.6 | 445.8 |
| sq_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 109 | 226 | 308 | 331 | 401 | 533 | 0 | 928.5 | 538.3 |
| sq_cpu_seed2 (CPU) | 2 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 108 | 218 | 294 | 318 | 414 | 520 | 0 | 892.8 | 473.6 |
| sq_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 115 | 226 | 301 | 324 | 348 | 527 | 0 | 793.3 | 440.9 |
| sq_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 114 | 234 | 314 | 336 | 359 | 538 | 0 | 826.5 | 545.6 |
| sq_cpu_seed5 (CPU) | 5 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 118 | 171 | 248 | 269 | 361 | 471 | 0 | 890.4 | 495.2 |
| sq_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 175 | 232 | 317 | 339 | 420 | 543 | 0 | 656.0 | 373.8 |
| sq_cpu_seed7 (CPU) | 7 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step (designed) | 96 | 188 | 266 | 290 | 314 | 493 | 0 | 800.2 | 524.7 |
<!-- /STAGE2 -->

- **All nine runs certified** (the official run and 8 CPU seeds): the true layout (288/288 columns, low digits at
  i + j and high digits at i + j + 1), the true division wiring, both tables exact, frozen at steps 471–543.
- T's direction started the wrong way in 4 of the 8 CPU seeds (1, 2, 5 and 6) and settled 70–96 steps after the I/O
  lesson joined, instead of 24 (§6). Stage 2 has no shape switch for T's lesson to pull on.

### 4. Stage 3: the reader's and writer's shape

#### 4.1 The switch

IO.md's reader reads a field with a pointer of one shape: an anchor (a token type or the prompt's end) and one shared
step, slot j at anchor + (j + 1)·step, live until the first non-digit. IO.md §6 contrasted it, under supervision, with
an own offset per slot and an own absolute position per slot. Only the shared step read 8-digit moduli there, because
slot 7 is never live in training. (Under this file's whole-reading lesson, slot 7 is trained anyway as the stop slot of
7-digit moduli, one place past their last digit, which is where an 8-digit modulus has its 8th digit. Two runs that
ended on the offsets shape read every 8-digit modulus right, §4.5.) Stage 3 offers all three shapes and lets training
choose (`SHAPES`):

| shape | parameters per field | slot j's position |
|---|---|---|
| step | anchor, step ±1 | anchor + (j + 1)·step |
| offsets | anchor, an offset per slot in −9..9 | anchor + offset_j |
| absolute | a position per slot | position_j |

**One shape switch serves all four fields** (N, x, T and the writer), as one kind of connection. Each field keeps every
shape's own parameters.

#### 4.2 What teaches it

IO.md's lessons, with every pointer probability a mixture over the shapes:
- **the reading mixture and the tiling**: a field "reads run r in direction d" when its slots 0..len−1 sit on the
  run's digits in reading order and its next slot right past the run (where the read stops). Each shape gives that
  whole reading a probability; the shape switch weights them. The verdicts are IO.md's (x < N under each hypothesis's
  code);
- **the writer**: the same whole-reading probability on the answer's own positions (the evaluator's target positions:
  IO.md's answer round trip read the labels placed there), plus IO.md's pointer-free code round trip and the answer's
  tokens being digits;
- **T's direction**: IO.md's speculation, its two whole-run programs weighted by the shape mixture, with the shape
  switch held fixed in this lesson (§4.4, third bug).

#### 4.3 The lab

`tiny/tiny_io.py`: H1's real prompts, the digit values given (the lab isolates the shape). The reading lesson uses the
verdicts x < N and T's value; the writer lesson uses the answer's positions. Certification: every field of every
prompt of the T = 1, 16, 32, 64 rungs, both profiles, the 8-digit moduli included.

<!-- TINY_S3 -->
One shape per field (`s3`):

| shapes offered | certified (every field exact on the T = 1, 16, 32, 64 rungs, both profiles) | shapes chosen (N, x, T, writer) | N exact | x exact | T exact | writer exact | every field exact on the 8-digit-N prompts | stopped at step |
|---|---|---|---|---|---|---|---|---|
| step,offsets,absolute | 4/6 | step/step/step/step x4; step/step/offsets/step x2 | 1.0 | 1.0 | 0.049-1.0 | 1.0 | 0.02-1.0 | 152-161 |
| step,offsets | 3/4 | step/step/step/step x3; step/step/offsets/step x1 | 1.0 | 1.0 | 0.049-1.0 | 1.0 | 0.02-1.0 | 107-115 |
| step | 2/2 | step/step/step/step x2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 104-105 |
| offsets | 0/2 | offsets/offsets/offsets/offsets x2 | 0.0-0.08 | 0.0-0.095 | 0.012-0.049 | 0.989 | 0.0 | - |
| absolute | 0/2 | absolute/absolute/absolute/absolute x2 | 0.0-1.0 | 0.0-0.001 | 0.005-0.046 | 0.043-0.075 | 0.0 | - |

One shape switch for all four fields (`s3b`, the file):

| shapes offered | certified (every field exact on the T = 1, 16, 32, 64 rungs, both profiles) | shapes chosen (N, x, T, writer) | N exact | x exact | T exact | writer exact | every field exact on the 8-digit-N prompts | stopped at step |
|---|---|---|---|---|---|---|---|---|
| step,offsets,absolute | 8/8 | step/step/step/step x8 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 102-129 |
| step,offsets | 4/4 | step/step/step/step x4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 102-152 |
<!-- /TINY_S3 -->

- **One shape per field picks the shared step for N, x and the writer every time, but not for T.** In 3 of 10
  mixture runs T, mostly one digit in training (two thirds of the rows), took the offsets shape and stuck. An anchor at T's marker with slot 0
  one place right reads every one-digit T and no two-digit one. The step shape's lesson for T is no stronger than the
  offsets shape's.
- **One shape switch for all four fields** (the file): 12 of 12 runs pick the step shape and read every field exactly,
  8-digit moduli included. The long fields' evidence for the shared step carries the short one. TINYROUTING's
  "one shared switch per kind of connection" again.
- **A single shape alone**: the step shape certifies (2/2). Offsets alone and absolute positions alone do not even fit
  the training prompts in 1,500 steps (0/2 each).

#### 4.4 Three bugs on the way to H1

- **The direction of a one-digit field.** The reader decides which reading hypothesis a row's snapped pointers make
  (`which_run`). IO.md's version knew the pointer's step and so its direction. The version for every shape first
  matched the slots on the run's digits and required only "the next slot is a non-digit". A one-digit run then
  matches both directions. Rows with x < 10 (about 0.1–0.5% of training rows) selected hypothesis 0 (x read the other
  way) instead of 1, with that hypothesis's untrained code (checked on 64 such rows: all 64 on hypothesis 0 before the
  fix, all on 1 after). Their code round trips taught the writer conflicting values. In the H1 diagnostics d9–d11
  the I/O's exactness check then failed on 25–40% of the logged steps in phase C. The failing part, logged in d10, was
  the writer's code not inverting the reader's code. So the 20 exact steps that start T's lesson never came. The
  fix: the stop slot must sit right past the run in the reading direction, the same whole reading `shape_probs`
  scores.
- **The writer's liveness term.** IO.md's answer round trip reads the answer at the writer's slots, and its liveness
  (slot j counts only if slots 0..j read digits) pushes the answer's tokens to be digits with a weight that grows
  with the slot. Replaced by the shape-neutral positions lesson, that push was lost. It is now its own term on the
  answer's own positions (d11 on).
- **T's lesson moved the shared shape switch** (`submission_io3_v1.py`, fixed in `submission_io3.py`). T's digits
  never appear in an answer, so until T's lesson starts every lesson sums over T's two directions. T's direction then
  drifts to whichever side starts ahead, to near certainty: in the diagnostics d12 the wrong side in 3 of 4 seeds
  (weight 0.99 or more). T's lesson is normalized over the two directions: it raises the share of the reading that
  reproduces the answers. Under the trained step shape with the wrong direction that share is about 0.01. Under an
  untrained shape, which has not chosen a direction, it is about one half. So the lesson also pushed the shape
  switch toward the untrained shapes, and that switch serves all four fields. In d12:
  - seed 1's direction flipped first (T settled at step 384, the switch at 0.998);
  - seed 74's switch fell to 0.73, then T's direction flipped (step 439) and the switch went back to step;
  - seed 2's switch went from step (0.995 at step 375) to offsets (0.62 at step 400), and for 14 steps the snapped
    pointers read no row. The offsets shape's own pointers then learned the training prompts, and the switch stalled
    at 0.40 step and 0.60 offsets: T read the right way under offsets and the wrong way under step, so T's share
    stayed at 0.60, and 300 steps later (step 700) its lesson had still not ended. Two CPU seeds of this version
    ended that way under the evaluator (§4.5).

  This version's official run certified (§4.5): its T settled at step 442, and its switch ended at 0.9957. The fix:
  T's lesson holds the shape switch fixed (its weights detached), so it teaches each shape's own T pointer and leaves
  the choice of shape to the reading lessons. In d13 (the same seeds, the runs identical until T's lesson) the switch
  stays at 1.000, and T settles at steps 384, 423 and 422 in seeds 1, 2 and 74 (the first version: 384, never and
  439), and at 305 in seed 4 as before. It is TINYROUTING §6's failure in a new form: a verdict normalized over two
  options rewards a third option that has not decided.

#### 4.5 The H1 runs

<!-- STAGE3 -->
| run | seed | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | subtractor, product cell, gate | division wiring | layout | shape | reading chosen | ladder joins | chain lesson joins | I/O lesson joins | T settled | frozen | never trained | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| io3_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | 226 | 306 | 329 | 442 | 532 | 0 | 729.4 | 48.9 |
| io3_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | 226 | 306 | 329 | 442 | 532 | 0 | 762.4 | 443.2 |
| io3_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 109 | 227 | 312 | 335 | 404 | 537 | 0 | 645.0 | 416.1 |
| io3_cpu_seed2 (CPU) | 2 | None / None | 0 / 0 | 0.33383339643478394, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | offsets | - | - | - | - | - | - | 0 | 3601.0 | 450.4 |
| io3_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 115 | 224 | 298 | 321 | 345 | 524 | 0 | 795.4 | 459.7 |
| io3_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 114 | 188 | 267 | 289 | 312 | 491 | 0 | 812.6 | 585.4 |
| io3_cpu_seed5 (CPU) | 5 | None / None | 0 / 0 | 0.33383339643478394, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | offsets | - | - | - | - | - | - | 0 | 3603.3 | 545.7 |
| io3_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 175 | 232 | 317 | 339 | 420 | 543 | 0 | 937.2 | 459.4 |
| io3_cpu_seed7 (CPU) | 7 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 96 | 184 | 262 | 286 | 310 | 489 | 0 | 808.6 | 500.0 |
| io3v2_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | 226 | 306 | 329 | 420 | 532 | 0 | 728.6 | 51.8 |
| io3v2_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 123 | 226 | 306 | 329 | 420 | 532 | 0 | 904.9 | 472.3 |
| io3v2_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 109 | 227 | 312 | 335 | 404 | 537 | 0 | 708.4 | 379.9 |
| io3v2_cpu_seed2 (CPU) | 2 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 108 | 235 | 310 | 334 | 427 | 536 | 0 | 1060.0 | 543.3 |
| io3v2_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 115 | 224 | 298 | 321 | 345 | 524 | 0 | 770.4 | 444.7 |
| io3v2_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 114 | 188 | 267 | 289 | 312 | 491 | 0 | 579.5 | 415.1 |
| io3v2_cpu_seed5 (CPU) | 5 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 118 | 203 | 280 | 301 | 386 | 503 | 0 | 805.2 | 459.0 |
| io3v2_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 175 | 232 | 317 | 339 | 420 | 543 | 0 | 647.9 | 398.3 |
| io3v2_cpu_seed7 (CPU) | 7 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | true | 288/288 | step | 96 | 184 | 262 | 286 | 310 | 489 | 0 | 898.7 | 558.6 |
<!-- /STAGE3 -->

- **The file (`io3v2`) certified on the official run and on all 8 CPU seeds**: the step shape, the true division
  wiring, the true layout, both tables exact, frozen at steps 489–543.
- T's direction started the wrong way in 5 of the 8 seeds (74, 1, 2, 5 and 6). There it settled 69–93 steps after
  the I/O lesson joined, against 23–24 when it started the right way.
- **The first version (`io3`) certified on the official run and on 6 of its 8 CPU seeds.** The two versions
  run identically, step for step, until T's lesson, and after it too where T started the right way (seeds 3, 4 and
  7). Where it started the wrong way, the first version settled T at the same step in seeds 1 and 6, 22 steps later in
  seed 74 (442 against 420), and never in seeds 2 and 5. Those two ended on the offsets shape and never froze: every
  step of T's unfinished lesson runs whole count-downs, so their training ran out of time (3,600 s) at steps 1,644 and
  1,288. Their tables, division wiring and layout were right. They read every two-digit T right and every one-digit T
  wrong: the T = 16, 32 and 64 rungs 768/768 on both profiles (8-digit moduli included), the T = 1, 2, 4 and 8 rungs
  0–2 of 768. Their final pointers show why (the `IO field=T` line): the switch at 0.40 step and 0.60 offsets, and
  T's anchor, which the two shapes share, on the separator before T. The step shape read from there forward (slot 0
  one place past the anchor): the wrong order for a two-digit T. The offsets shape put slot 0 two places past the
  anchor and slot 1 one place past it: the right order for a two-digit T, but for a one-digit T slot 0 sits on the
  separator after it. T's lesson only sees two-digit rows (rows whose two readings differ). It reached the right
  order through the shape switch and T's offsets instead of moving the anchor to the prompt's end and turning the
  step around, which is where T's pointer ends in every run of the fixed version (and in the first version's runs
  that certified). Meanwhile the step shape's share still read one-digit T for the reading lesson, while the
  snapped reader followed the larger share, offsets.

### 5. The prediction path (rule 8)

COMBINED.md §3's path (`PATH = "ste"`): one-hot digits through straight-through tables. Every table is
one-hot(argmax) + (softmax − softmax.detach()), so its value is exactly 0 or 1, and the state's gradient is discounted
10× after every link, addition and round (`GRAD_DAMP`; undiscounted it overflows within one division). The switches join
the same way:
- **the division's switches** (`ste_switches`): each a straight-through one-hot over its options. With autograd on, an
  option set is the blend Σ_o w_o·option_o, whose value is the chosen option exactly and whose gradient reaches every
  option's logit. Without autograd, as at evaluation, the chosen option itself is used: the same value, less work
  (`pick`). The keep-or-discard write uses the straight-through gate as before;
- **the squaring's layout** (`ste_sources`): the argmax rule's source map as a one-hot [W, C, W + 1] plus (soft map −
  soft map.detach()), the soft map being every rule's map weighted by its probability;
- **the pointer's shape** (`slot`): a straight-through one-hot over the three shapes, each shape's position a
  straight-through one-hot from its own parameters.

`PATH = "index"` runs the same argmax machine on integer digits.

**The test** (`local/local_wiring.py grad`, and at the end of every harness `train` run). The model is in eval mode
with autograd on. The cross-entropy of the final logits at the answer's target positions is divided by 20 (the logits
are 0 and −20, so a right answer's plain cross-entropy is exactly 0 in fp32), followed by one backward pass. Every
parameter tensor must get a finite gradient that is not all zero, and the rows must stay exact.

| model | rows | every tensor gets a gradient |
|---|---|---|
| stage 1 at the truth (`MODE = "designed"`) | 16 (T = 1, 2) | yes, all 7 switches included |
| stage 1 trained (d1–d3, 10 frozen runs) | 32 (T = 1, 2) | yes, 10/10 |
| stage 2 at the truth | 32 (T = 1, 2) and 32 (T = 8, OOD-N T = 4) | yes |
| stage 3 at the truth | the same two batches | yes, the shape switch and both other shapes' parameters included |
| stage 2 trained (d8, 4 frozen runs) | 32 (T = 1, 2) | yes, 4/4 |
| stage 3 trained (d12: 3 frozen runs of the first version; d13: 4 of the file) | 32 (T = 1, 2) | yes, 7/7 |

**A gradient that underflows.** A switch's gradient arrives only through the places where its options differ. The
squaring's high columns reach the answer only through up to 72 links of the division, each discounted 10×, so their
share underflows to 0 in fp32. With the layout's three parts as three separate switches, the high-digit order switch
got exactly 0 on these batches: its two options differed only in such columns or in zero digits. With one weight per
whole rule (the file) the gradient reaches both rule tensors. As in COMBINED.md, the gradient is the straight-through
surrogate, discounted. It is a path, not the derivative of anything exact.

Identical logits: with every table and switch at the truth, `predict_ste` and `predict_index` give `torch.equal` logits
on 16 prompts of the T = 1, 2, 8, 16, 64 rungs of both profiles. Every frozen harness run gives them on 128 prompts of
all 14 rungs (§2.3).

### 6. Cost

The switches' lessons are cheap next to the tables'. Each step adds:
- the division ladder: 216 combinations × 64 rows × 10 registers, one hard division step each (9 links of 9 places);
- the I/O lesson: 16 combinations × 128 rows × two whole hard divisions, and the T ≤ 2 rows through the whole machine;
- the routed ladder: 2 rows j per number instead of 1.

Training times on the H100 (official runs, seed 74), against COMBINED.md's 667 s (frozen at step 545):

| file | training s | frozen at step | seconds to step 300 / 400 / 500 / 700 | evaluation s |
|---|---|---|---|---|
| stage 1, first version | 393 | 640 | | 49 |
| stage 1 (`submission_div.py`) | 314 | 615 | 46 / 67 / 113 / 166 | 37 |
| stage 2 (`submission_sq.py`) | 378 | 479 | 84 / 151 / 202 / 204 | 49 |
| stage 3, first version | 729 | 532 | 65 / 295 / 519 / 541 | 49 |
| stage 3 (`submission_io3.py`) | 729 | 532 | 77 / 335 / 485 / 509 | 52 |

After the freeze every stage runs its remaining ~19,500 steps in about 150–220 s. Stage 3's extra ~350 s are T's
lesson. On up to 64 rows it runs the exact machine's whole count-down from both readings of T (a 16 read backwards is
61 rounds), at every step until T settles: about 4.5 s a step on the H100, launch-bound. In stage 3 seed 74's T
started the wrong way and settled 91 steps after the I/O lesson joined (329 → 420); the lesson ran for about 70 of
them. In stages 1 and 2 it started the right way and settled 24 steps after the I/O lesson joined (20 of them the
waiting window). On the CPU seeds of stage 2, where T started the wrong way in 4 of 8, it settled 70–96 steps after.

No stage needed a relaxed budget; the official budgets are 3,600 s and 1,800 s. On CPU the evaluation with the
prediction path inside autograd takes 190–550 s.

### 7. What is still hand-written

The menus are the grammar. The algorithm lives in them, spread over many small choices. Item by item:

| part | learned now | still written by hand |
|---|---|---|
| the division | which end of the dividend is read first; how the register makes room for a digit; each link's minuend and subtrahend; whether N's digits are reversed; what a link writes; which places are the answer; the keep-or-discard table (CONTROL.md) | the menus; 8 steps of up to 9 links; a 9-place register starting with 8 digits; the gate reading a link's final borrow; the subtractor rippling from place 0 |
| the squaring | the placement rule of the low digits and of the high digits; the product table (SQUARE.md) | the menu of rules (f(i) + g(j) + d); 8 rows of pairs (s_i, s_j), each two numbers summed, rows summed in order; 18 columns |
| the count-down and the loop | the count's subtraction and its commit (CONTROL.md) | the 2-place count register, the number 1, "rounds until no row commits", the cap |
| the adder | its table (the subtraction table) | a + b = a − (0 − b) |
| the reader and writer | the pointer's shape (one of three); anchors, steps, offsets or positions; digit-ness; the codes; the reading among 24 hypotheses (IO.md) | the menu of shapes; 8, 8 and 2 slots read, 8 written; "a slot reads 0 once a slot before it met a non-digit"; the prompt holds three digit runs; ranks as codes |
| training | — | every lesson (which law each switch and table is checked against), the menus' factorization (which switches are weighed together), the schedule (what waits for what), the freeze |

The checks are themselves designed knowledge (rule 14 above): the definition of division with remainder (the division
ladder), "x mod N = x", "N mod N = 0", what a schoolbook row is (the routed ladder), the order of x and N.

### 8. Reproduce

Everything heavy runs on Modal. Launch from the one-layer-deeper repository root.
- `modal_wiring.py` is a copy of `learned_remainder/modal_chain.py` with its own app and Volume (`learned-wiring`,
  folder `records/`). It builds the competition image, regenerates H1 inside it with the official generator arguments,
  and writes each record there.
- `wait_fetch.py` downloads the records to `cpu_first/history/runs/learned_wiring/<name>.json`.

```bash
cd ~/github/one-layer/one-layer-deeper
M=~/github/one-layer/cpu_first/history/learned_wiring/modal_wiring.py
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode official --name div2_official --file submission_div.py
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode plan --plan ~/github/one-layer/cpu_first/history/learned_wiring/plans/div2_cpu.json --file submission_div.py
cd ../cpu_first/history/learned_wiring
~/github/sotaku/venv/bin/python wait_fetch.py div2_official div2_cpu_seed74 div2_cpu_seed1
~/github/sotaku/venv/bin/python summarize_wiring.py --stage div2
```

The same with `sq` / `submission_sq.py` (stage 2) and `io3v2` / `submission_io3.py` (stage 3). The first versions'
records are `div` (`submission_div_v1.py`) and `io3` (`submission_io3_v1.py`). `--set '{"MODE": "\"designed\""}'`
runs a file with every table and switch at the truth (the plumbing check).

Local checks, each well under a minute and 1 GB (`local/`, with `learned_remainder/chain_local/stub` standing in for
the evaluator's API):

```bash
cd ~/github/one-layer/cpu_first/history/learned_wiring/local
~/github/sotaku/venv/bin/python local_wiring.py submission_io3.py designed N=16   # both paths exact, identical logits
~/github/sotaku/venv/bin/python local_wiring.py submission_io3.py grad N=32       # the gradient test at the truth
```

The training diagnostics (`local/modal_diag.py`, app `learned-wiring-diag`, logs to `local/logs/`; plans `d1`–`d13`
in `local/plans/`):

```bash
cd ~/github/one-layer/one-layer-deeper
~/github/sotaku/venv/bin/python -m modal run ~/github/one-layer/cpu_first/history/learned_wiring/local/modal_diag.py --plan ~/github/one-layer/cpu_first/history/learned_wiring/local/plans/d8.json
```

The tiny cores (`tiny/`, app `learned-wiring-tiny`, rows to `tiny/runs/<suite>.jsonl`; suites in `tiny_suites.py`):

```bash
cd ~/github/one-layer/cpu_first/history
~/github/sotaku/venv/bin/python -m modal run --detach learned_wiring/tiny/modal_tiny.py --suite s1b,s2,s2b,s3,s3b
~/github/sotaku/venv/bin/python -m modal run learned_wiring/tiny/modal_tiny.py --suite s1b,s2,s2b,s3,s3b --fetch
~/github/sotaku/venv/bin/python -m learned_wiring.tiny.summarize_tiny
```

A single tiny run locally: `python -m learned_wiring.tiny.tiny_div seed=0 max_steps=300` (or `tiny_sq`, `tiny_io`).

Files in `cpu_first/history/learned_wiring/`:
- `submission_div.py`, `submission_sq.py`, `submission_io3.py`: stages 1, 2 and 3, each certified byte for byte (§1);
- `submission_div_v1.py`, `submission_io3_v1.py`: the first versions of stages 1 and 3, kept byte for byte with their
  records (§1, §3.4, §4.4);
- `modal_wiring.py`, `wait_fetch.py`, `summarize_wiring.py`, `plans/*.json`: the official and CPU-seed runs;
- `local/`: `local_wiring.py` (harness), `modal_diag.py`, `plans/`, `logs/`;
- `tiny/`: `tiny_div.py` (stage 1), `tiny_sq.py` (stage 2), `tiny_io.py` (stage 3), `tiny_suites.py`,
  `modal_tiny.py`, `summarize_tiny.py`, `runs/`;
- records: `cpu_first/history/runs/learned_wiring/<name>.json` (RESULT_JSON, the full log, the exact source, the
  manifest).

The certified files of the other pieces are untouched: `learned_remainder/submission_combined.py` (md5 `4f5cecc7…`),
`history/ablation/submission_simple.py` (`f374ec70…`), `learned_remainder/modal_chain.py` (`ad5fbd9c…`).
