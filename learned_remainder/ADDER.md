# H1 with learned addition in the self-check: a 70-step warm start, then only the learned table

## In plain words

**The task.** The model reads three numbers written as digits: a modulus N, a starting number x and a count T. It must
square x, keep only the remainder after dividing by N, and repeat that T times. To be certified it must get every one
of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, both for moduli it saw in training and for larger moduli
it never saw (up to 8 digits).

**Where this starts.** [CHAIN.md](CHAIN.md)'s calculator does long division with one learned rule, a 200-entry table:
"this digit minus that digit, minus a borrow, gives this digit and maybe a new borrow". Every subtraction in the
remainder is that table. Training teaches it by checking each subtraction the calculator does: subtract N, add N back,
and see whether the starting number comes back. The adding-back was done by a hand-written adder, the last piece of
arithmetic in training that nobody had to learn.

**What I tried.** Taking that adder out, in the two ways the plan named:
- **A. Addition from the same table.** In ten's complement, 0 − b is "minus b", so a + b = a − (0 − b). The check
  becomes: subtract N, subtract (0 − N), compare. Every step is the learned table.
- **B. A second learned table for addition**, trained alongside the subtraction table.

**Did it work?**
- **From a random start, no.** Neither route learned the table, in any seed or variant: 46 training runs on Modal,
  about 25 short local ones, and the official H100 run `adder_official` (34 of 200 entries right, no rung certified).
  Both routes stall with at most 43 of the 200 entries right.
- **With the hand-written adder kept for the first 70 training steps only, yes** (route A). By then the table is about
  185/200 right. From step 71 on, every addition and subtraction in training is the learned table, and the learned
  adder finishes the table. On the untouched official H1 manifest on an H100 (`adder_warm_official`, the final file byte
  for byte): all 768 questions right at every T up to 64, on seen and unseen moduli; 200/200 entries right, every
  entry used at evaluation trained; training 78 s (the table is done after 343 steps, then it holds still; the
  chain's recipe took 272 s and 417 steps), evaluation 13 s. The same file certified on 8 of 8 further seeds on CPU
  (frozen at steps 319–354) and on 4 of 4 seeds that never see an answer (326–336).
- With the exact table the file's wiring certifies (`adder_s0_official`), and the certified
  [`submission_chain.py`](submission_chain.py) is unchanged.

**What is still hand-written** in [`submission_adder.py`](submission_adder.py):
- reading and writing the digits;
- squaring the number;
- the long-division layout and the chain's wiring;
- the stop rule ("keep the last result before the first borrow");
- the loop over T;
- for the first 70 training steps only, the adder in the self-check (before: for the whole of training).

**What broke from a random start.** With a hand-written adder the check can only be passed by subtracting right: every
failed check points at exactly what to fix. With the adder made of the same table, the check only asks the table to
agree with itself, and many wrong tables do:
- The first one training finds is lazy: "copy the first number, write 0 when both digits are equal, and raise a borrow
  that never goes away". It passes 99% of the checks. The answers cannot correct it: with every subtraction
  "negative", the calculator never uses a subtraction, so the answers have nothing to push on.
- The other rules of addition (a + b = b + a, 0 + a = a, a − a = 0, …) knock out the lazy table, but training then
  stalls on other wrong tables, with 20–43 of the 200 entries right.
- Some wrong tables are perfectly consistent. One subtracts correctly but with the digits relabeled; another carries
  only the "odd/even" part of each digit. They pass every check made of the table alone.

**What the data can still teach.** Two kinds of facts are not the table agreeing with itself:
- **x is smaller than N.** So x − N must end in a borrow, and N − x must not. These two facts alone teach all 200
  borrow outputs exactly, in every run. They say nothing about the digits.
- **The answers.** Only 1% of the training rows have T = 1 and 1% T = 2, about 5 + 5 per batch of 512. That is too
  few to teach the digits from scratch. Rows with larger T only help once the table is nearly right.

**Why the warm start works.** With the borrows right, the learned-adder check finishes the table by itself once about
half of the digit entries are right; a random start has about 10%. Seventy steps of the hand-written adder get it
there. One trap remains near the truth: "0 minus 0 with a borrow" and its nine neighbours ignoring the borrow, the same
run of 9s that stalled CHAIN.md's first recipe. A plain fact breaks it: x − N wraps around to a large number, so
subtracting N once more must not borrow. Without that fact the warm start stalls at 168–170 of 200.

**The learned adder itself is exact.** Built from a correct table, a + b = a − (0 − b) gets every sum of two numbers
up to 3 digits right (10⁶ pairs) and 440,000 random sums of 9 to 72 digits, with the table `adder_warm_official`
learned (and with `chain_v2_official`'s). It is in `submission_adder.py` as `add_learned` (integers) and
`add_learned_soft` (digit distributions), for the squaring work.

## 1. Results

### 1.1 Official runs
Official manifest `benchmark/manifests/h100_hard_h1.json`, untouched (cuda:0, bf16 autocast, the submission's batch,
3,600 s training and 1,800 s evaluation budgets, seed 74), on a Modal H100 80GB through CHAIN.md's runner
(`modal_chain.py --file submission_adder.py`, same image, H1 regenerated with the official generator arguments). Every
rung is 768 prompts.

| run | what ran | max T (seen moduli) | OOD-N max T | rungs T = 1…64, seen / OOD-N | test, ood_t, ood_n_t | subtractor rows right | training s (steps) | evaluation s |
|---|---|---|---|---|---|---|---|---|
| `adder_s0_official` | the file with the exact table (`MODE = "designed"`, `MAX_STEPS = 100`): the plumbing check (the earlier version `56af1e12…`; same datapath, and training is frozen from step 1) | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 (designed) | 2.9 (100) | 13.7 |
| **`adder_warm_official`** | **`submission_adder.py` byte for byte** (70-step warm start), md5 `44e658ab2dfd270baa1d7625d734c3da` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 77.8 (frozen at 343) | 12.8 |
| `adder_official` | from a random start: the file with `ADDER_WARM = 0` (an earlier version, md5 `56af1e12…`; its code differs from the final file only in that default and one equivalent refactor) | none | none | 3, 0, 0, 0, 0, 0, 1 / 3, 0, 0, 0, 0, 0, 0 | 6/9,999, 7/10,002, 8/10,002 | 34/200 | 765.8 (3,000, never frozen) | 13.1 |
| for comparison: `chain_v2_official` | `submission_chain.py` (hand-written adder in the check) | 64 | 64 | 768 everywhere | all exact | 200/200 | 271.8 (frozen at 417) | 15.4 |

The same file through the same unmodified evaluator on Modal CPU containers, the manifest's device set to cpu and
everything else official, one seed per run (`plans/a1_cpu.json`):

| recipe | seeds | certified T = 64 on both profiles | table rows right | frozen at step | training s / evaluation s (CPU) |
|---|---|---|---|---|---|
| **final file**, with the rows' answers | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** (every rung 768/768) | 200/200 | 343, 331, 329, 327, 349, 319, 342, 354 | 128–242 / 19–27 |
| **final file, never sees an answer** (`REAL_MAX_T = 0`) | 74, 1, 2, 3 | **4/4** (every rung 768/768) | 200/200 | 336, 327, 329, 326 | 134–216 / 18–30 |

Every run's audit reads `AUDIT eval_rows_used=200/200 used_but_never_trained=0`, with every row used at least 246,744
times by training's hard checks. The learned adder built from each of the 12 tables is exact on every test of §7. CPU
seed 74 took the same path as the H100 run (phase B at step 134, frozen at 343): training has no random numbers in it
once the rows are drawn, as in CHAIN.md §1. For comparison, CHAIN.md's recipe froze at 417–481 on its 8 CPU seeds and
408–544 without answers.

Every record holds the exact source that ran.
- `adder_warm_official`: hand-over to the learned adder after step 70, phase B at step 134 (80.5% of the checks
  passing), frozen at step 343, then still for the rest of the 3,000 steps. Audit: `AUDIT eval_rows_used=200/200
  used_but_never_trained=0`; training's hard checks used every row at least 273,412 times (median 601,255).
- `adder_official`: training used every row at least 206,546 times and evaluation used all 200
  (`used_but_never_trained=0`), so its 166 wrong rows were trained, not missed.

### 1.2 Training runs
Every row below trains the file's lessons through a small harness (`adder_local/local_adder.py`, the same lessons
and schedule without the evaluator) that prints the snapped table against true subtraction every 25 steps, on Modal:
batch 64 on CPU containers, batch 512 (the official one) on L4 GPUs. 1,500 steps each (the last eight rows: 1,000), past
the point where every unfrozen run had stopped changing. "Digits" and "borrows" count the 200 digit outputs and the
200 borrow outputs that are right at the end (a row is right when both are). "Checks passed" is the share of the hard
checks the schedule reads. The last column names the wrong table of §3 that the end state is closest to.

| route | batch | lessons | seeds | rows right | digits right | borrows right | checks passed | frozen at | closest wrong table (entries matching) |
|---|---|---|---|---|---|---|---|---|---|
| A | 64 | the plan's lessons: check, R − 0 = R, R − R = 0, 0 − N borrows; phase B: x mod N = x, answers | 1, 2, 3 | 27, 28, 27 | 27, 28, 27 | 112, 121, 124 | 99%, 71%, 16% | –, –, – | lazy 185; lazy 150; lazy 159 |
| A | 64 | + 0 + N = N | 1, 2, 3 | 29, 26, 29 | 29, 29, 29 | 158, 147, 139 | 24%, 30%, 25% | –, –, – | lazy 123; lazy 127; lazy 140 |
| A | 64 | + R + N = N + R | 1, 2, 3 | 22, 23, 24 | 23, 32, 27 | 110, 117, 115 | 2%, 38%, 21% | –, –, – | lazy 152; lazy 72; lazy 105 |
| A | 64 | + both | 1, 2, 3 | 25, 22, 20 | 33, 34, 29 | 120, 118, 120 | 13%, 3%, 9% | –, –, – | lazy 50; lazy 53; lazy 52 |
| A | 64 | the plan's lessons + x − N borrows, N − x does not | 1, 2, 3 | 27, 28, 27 | 27, 28, 27 | 200, 200, 200 | 38%, 23%, 21% | –, –, – | lazy 116; lazy 101; lazy 101 |
| A | 64 | everything above | 1, 2, 3 | 30, 38, 36 | 30, 38, 36 | 198, 196, 197 | 9%, 10%, 7% | –, –, – | copy the minuend 46; lazy 50; lazy 52 |
| A | 64 | everything, x mod N = x from step 1 | 1, 2, 3 | 34, 35, 36 | 34, 35, 36 | 196, 196, 195 | 8%, 15%, 7% | –, –, – | lazy 56; lazy 64; lazy 72 |
| A | 64 | everything but R − 0 = R and R − R = 0 | 1, 2, 3 | 26, 12, 8 | 26, 34, 26 | 199, 101, 47 | 39%, 0%, 16% | –, –, – | copy the minuend 45; lazy 43; lazy 33 |
| designed | 64 | CHAIN.md's recipe (hand-written adder), for comparison | 1, 2, 3 | 200, 200, 200 | 200, 200, 200 | 200, 200, 200 | 100%, 100%, 100% | 643, 624, 597 | –; –; – |
| A | 512 | everything | 1, 2 | 39, 37 | 39, 37 | 196, 197 | 12%, 9% | –, – | lazy 72; lazy 46 |
| A | 512 | everything, answers from step 1 | 1, 2 | 38, 36 | 38, 36 | 196, 196 | 5%, 12% | –, – | lazy 79; lazy 70 |
| A | 512 | everything, answers from step 1 through the soft own-borrow choice (T = 1) | 1, 2 | 38, 38 | 38, 38 | 194, 198 | 8%, 7% | –, – | lazy 73; lazy 65 |
| A | 512 | everything, answers from step 1, borrow table held true (diagnostic) | 1, 2 | 43, 28 | 43, 28 | 200, 200 | 19%, 2% | –, – | lazy 76; lazy 76 |
| A | 512 | everything, answers through the soft choice from step 1, borrow table held true (diagnostic) | 1, 2 | 36, 30 | 36, 30 | 200, 200 | 19%, 14% | –, – | lazy 77; lazy 69 |
| A | 512 | no check: R − 0 = R, R − R = 0, order facts, answers from step 1 | 1, 2 | 42, 40 | 42, 40 | 200, 200 | 1%, 0% | –, – | true except (0, 0, 1) 43; true except (0, 0, 1) 41 |
| A | 512 | no check, answers through the soft choice (T = 1) | 1, 2 | 39, 39 | 39, 39 | 200, 200 | 1%, 1% | –, – | true except (0, 0, 1) 40; true except (0, 0, 1) 40 |
| B | 512 | check through the adder table, (R + N) − N = R, 0 + N = N, R + N = N + R, R − 0 = R, R − R = 0 | 1, 2 | 28, 32 (adder table: 28, 34) | 36, 39 | 135, 133 | 31%, 31% | –, – | lazy 50; lazy 53 |
| B | 512 | + order facts | 1, 2 | 36, 42 (adder table: 33, 39) | 36, 42 | 200, 199 | 23%, 36% | –, – | true except (0, 0, 1) 37; true except (0, 0, 1) 43 |
| B | 512 | + order facts, answers from step 1 | 1, 2 | 30, 33 (adder table: 36, 38) | 30, 33 | 200, 200 | 31%, 27% | –, – | lazy 58; copy the minuend 47 |
| designed | 512 | CHAIN.md's recipe, for comparison | 1, 2 | 200, 200 | 200, 200 | 200, 200 | 100%, 100% | 614, 606 | –; – |
| A | 512 | everything + (x − N) − N no borrow, from a random start (1,000 steps) | 1, 2 | 26, 31 | 26, 31 | 196, 200 | 8%, 15% | –, – | lazy 79; lazy 68 |
| A | 512 | the same, hand-written adder for the first 70 steps: the final file | 1, 2 | 200, 200 | 200, 200 | 200, 200 | 100%, 100% | 328, 343 | –; – |
| A | 512 | the same, hand-written adder for the first 100 steps | 1, 2 | 200, 200 | 200, 200 | 200, 200 | 100%, 100% | 329, 324 | –; – |
| A | 512 | hand-written adder for the first 10 steps | 1, 2 | 112, 99 | 112, 100 | 198, 197 | 12%, 29% | –, – | true except (0, 0, 1) 113; true except (0, 0, 1) 100 |
| A | 512 | hand-written adder for the first 30 steps | 1, 2 | 150, 200 | 150, 200 | 198, 200 | 9%, 100% | –, 897 | true except (0, 0, 1) 151; – |
| A | 512 | hand-written adder for the first 50 steps | 1, 2 | 200, 200 | 200, 200 | 200, 200 | 100%, 100% | 338, 339 | –; – |
| A | 512 | 70 warm steps, never sees an answer (REAL_MAX_T = 0) | 1, 2 | 200, 200 | 200, 200 | 200, 200 | 100%, 100% | 328, 329 | –; – |
| A | 512 | 70 warm steps, without (x − N) − N no borrow | 1, 2 | 168, 170 | 168, 170 | 197, 198 | 9%, 9% | –, – | true except (0, 0, 1) 169; true except (0, 0, 1) 171 |

## 2. Route A: addition from the subtraction table
The datapath is `submission_chain.py`'s, unchanged (CHAIN.md §3). Only the training check changes.

**The learned adder.** On P places, 0 − B with true subtraction is 10^P − B (B > 0), so
A − (0 − B) = A + B mod 10^P. Write the three results as integers with their final borrows:
0 − B = C − b_c·10^P and A − C = Y − b_y·10^P. Then Y = A + B + (b_y − b_c)·10^P, so the sum's final carry is
b_c − b_y (0 or 1 with the true table; b_c = 0 only for B = 0). `add_learned(A, B, Td, Tb)` returns (Y, b_c − b_y) for
any width; `add_learned_soft` does the same on digit distributions (the cell applied multilinearly, as `link_soft`).

**The check.** For every link input D of the machine's own division of each row's x² (512 rows × 8 steps × 9 links =
36,864 per training step, as in CHAIN.md §4):
- E = D − N (the link itself, soft table);
- C = 0 − N (once per row, soft table);
- F = E − C (soft table on both soft operands, `sub_soft`);
- F's digits must be D's (lowest place weighted `ID_PLACE0` = 5), and, completing the identity, E's final borrow must
  equal the learned adder's final carry: b_e = b_c − b_f, that is b_e + b_f = b_c.

**The anchors**, each with a hard version that joins the checks the schedule reads (`ANCHOR_*` weights):
- R − 0 = R and R − R = 0 on the registers Rp (digits and no final borrow);
- 0 − N borrows (N > 0), which makes the learned adder's carry 1 − b_f;
- x − N borrows and N − x does not (x < N);
- x − N wraps around to 10⁹ + x − N, which is at least N, so (x − N) − N does not borrow (`ANCHOR_WRAP`, §5.1);
- the learned adder's own rules: 0 + N = N, that is 0 − (0 − N) = N (complete: the two borrows agree), and
  R + N = N + R on the registers.

Phase B, the freeze and the own hard path for the answers are CHAIN.md's. The final file turns every anchor on and
keeps the hand-written adder for the first `ADDER_WARM = 70` steps (§5.2); 0 + N = N, commutativity and 0 − N borrows
join after the warm start, since they need the learned adder. `adder_official` ran it with `ADDER_WARM = 0`. Other
options in the file are the variants of §4–§6: `ANCHOR_INV` (R − (R − N) = N), `ANCHOR_PAIRS` (the check on pairs of
registers), `ID_CURRIC` (the check mod 10^k, growing k), `ANSWER_PATH = "soft"` (answers through the soft own-borrow
choice), `ID_WEIGHT`, `SELF_FROM`, and `ADDER = "table"` (route B, §6) or `"designed"` (CHAIN.md's adder).

With the exact table every one of these checks passes with loss 0 (`adder_local/local_adder.py designed`).

## 3. Which wrong tables pass which checks
`adder_local/landscape.py` builds complete 200-row tables and runs each check on the table's own numbers, as training
would: the link inputs of its own hard division of 1,024 random training rows' x², and the rows' N, x and answers.
Share of checks passed:

| table | (D − N) + N = D, digits | complete | complete, 0 − N borrows | R − 0 = R | R − R = 0 | 0 + N = N | R − (R − N) = N | R + N = N + R | x − N borrows | N − x no borrow | (x − N) − N no borrow | x mod N = x | answers T = 1 | CHAIN.md's designed-adder check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| true subtraction | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| digits relabeled (d → −d mod 10, borrows in that order) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.224 | 0.224 | 0.350 | 0.224 | 0 | 0 |
| borrows only in the mod-5 part of each digit | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.687 | 0.689 | 0.884 | 0.687 | 0 | 0.013 |
| borrows only in the mod-2 part of each digit | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.485 | 0.522 | 0.745 | 0.485 | 0 | 0.014 |
| no borrows at all (digit by digit mod 10) | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0.042 |
| copy the minuend, never borrow | 1 | 1 | 0 | 1 | 0.042 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| **lazy**: copy the minuend, 0 on the diagonal, a borrow that sticks (§4) | 0.990 | 0.990 | 0.990 | 1 | 1 | 0 | 0 | 0 | 1 | 0.003 | 0 | 1 | 0 | 0 |
| swapped borrow code (LEARNED_REMAINDER.md §3.4) | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| reversed (m − r − c) | 1 | 0.125 | 0 | 0.016 | 1 | 1 | 0.016 | 0 | 0 | 0 | 1 | 0 | 0 | 0.013 |
| true except (0, 0, 1) → 0 (CHAIN.md §5.2) | 0.673 | 0.673 | 0.673 | 1 | 1 | 0.625 | 0.969 | 0.142 | 1 | 1 | 0.739 | 1 | 0.857 | 0.508 |

What it shows:
- **Identities made only of the table cannot single out the truth.** The first three wrong rows are exact subtractions
  of other number systems of the same size: true subtraction with the digit symbols relabeled, or a carry that runs
  only through the mod-5 or the mod-2 part of each digit (10 = 2 × 5). They pass every table-only check exactly. Any
  relabeling that keeps 0 fixed gives another one.
- **Only facts tied to the true order of numbers separate them:** x − N borrows, N − x does not, (x − N) − N does
  not, "x mod N = x", and the answers. That matches TINYCORE.md §7: label-free checks need something trusted.
- **The swapped borrow code** (LEARNED_REMAINDER.md §3.4) fails the learned-adder check on every link, off by one at
  the lowest place as with the designed adder, so the lowest-place weight stays. No run ended near it.
- The lazy table, the one training actually finds (§4), passes the learned-adder check on 99% of the links, every
  check of the original plan (R − 0 = R, R − R = 0, 0 − N borrows, x mod N = x), and x − N borrows. It fails 0 + N = N,
  R − (R − N) = N, commutativity, N − x no borrow, and every answer.

## 4. What training settles on
The rows of §1.2, by what was added:
- **The plan's lessons** (the check, R − 0 = R, R − R = 0, 0 − N borrows, then "x mod N = x" and the answers): the lazy
  table of §3. At the end 150–185 of the 200 entries match it and 27–28 match the truth. Seed 1 passes 99% of its
  checks there.
  - Why it comes first: copying the first operand satisfies (D − N) − (0 − N) = D for any input, even a blurry one, so
    every entry gains by copying on its own. The truth needs the entries to agree with each other.
  - Why the answers cannot correct it: every link borrows at once (N's last digit is never 0) and the borrow sticks.
    So the chain stops at trial 0 at every step, the own hard path runs no link, and the answers send no gradient to
    the table (their loss stays near 22–28 nats).
- **+ 0 + N = N and/or commutativity**: the lazy table is knocked out (50–140 entries still match it), but the checks
  drop to 2–38% and 20–29 entries are right.
- **+ the order facts** (x − N borrows, N − x does not): the borrow outputs end exactly right, 200/200 in all three
  seeds. The digits stay at 27–28/200. A ripple's final borrow never reads the digit outputs (each place's borrow
  depends only on its inputs and the borrow coming in), so these facts teach the borrow column and nothing else.
- **Everything on** (the final file without its warm start): borrows 195–198/200, digits 30–38/200, checks 7–10%. A
  typical end state (`s1_v5_all_b64_seed1`): the column m = 0 and the diagonal are right (the anchors), and "0 − m" is
  a wrong pairing of digits, 1↔3, 4↔5, 6↔8, 7↔9 (the truth pairs 1↔9, 2↔8, 3↔7, 4↔6). Pairs are what 0 + N = N asks
  for.
- **At batch 512 with the answers from step 1**, through the own hard path or the soft own-borrow choice: 36–38/200.
- **With the borrow table held at the truth** (a diagnostic, not a recipe): 28–43/200. Getting the borrows right is
  not what is missing.

Other things tried, each with the borrows held true, from random digits (batch 64, seed 1, 300–600 steps, all
stalled at 30–50 entries right): lowest-place weight 1 or 0.2 instead of 5; a nearly hard random start (init scale 1 or
3); R − (R − N) = N; the check on pairs of registers (R − R′) + R′ = R (back to the lazy table); and the check grown
from the lowest place up, mod 10^k for k = 1, 2, …, 9. The last one has a reason to fail. With true borrows, one place
of the check only says whether e < 10 − n. So column 1 can be any one-to-one map from 1…9 onto 0…8, with column 9 its
inverse: 9! solutions for that pair of columns alone. Only many places together pin the digits.

## 5. How big the truth's basin is
Diagnostic only: the borrow table held at the truth, some digit entries started at the truth (logit 4 on the true
digit), the rest random. The learned-adder check with every anchor (with the borrows held, the order facts change
nothing), batch 64, seed 1 (`FIX=b TRUE_FRAC=…` or `TRUE_ROWS=…` in the harness):

| digit entries started true | entries right at the end | steps |
|---|---|---|
| none (random start) | ~40 | 600 |
| 10% at random | 45 | 500 |
| 20% at random | 65 | 500 |
| 30% at random | 122 | 500 |
| **50% at random** | **200** | 175 |
| 80% / 95% at random | 200 | under 100 |
| the "0 − m" rows (r = 0, 20 entries) | 107 | 300 |
| column m = 1, "subtract 1" (20 entries) | 80 | 300 |
| columns m ≤ 1 (40 entries) | 85 | 300 |
| columns m ≤ 2 (60 entries) | 157, still rising | 300 |
| r = 0, m = 0 or r = m (the anchors' rows and "0 − m") | 153, still rising | 300 |
| the borrow-in-0 half (100 entries) | 83 | 300 |
| **the borrow-in-1 half (100 entries)** | **200** (frozen at step 273) | 273 |
| 50% at random, borrows learned instead of held | 99 | 500 |

So the learned-adder check does pin the truth, and pulls it in from far away, once the borrows are right and about
half the digits are. Nothing in the data gets a random table there.

### 5.1 A trap ten entries from the truth
The basin depends on which entries are wrong, not only how many. CHAIN.md's recipe (hand-written adder) for 100 steps,
then the learned adder alone, every anchor on (batch 64, seed 1): the table is 189/200 right at the switch and stays
at 189–190 for the next 600 steps, passing 12% of its checks. The 10–11 wrong entries are one family, "subtract 0 with a
borrow coming in", (r, 0, 1): each writes r instead of r − 1 (and (0, 0, 1) writes 0 instead of 9). These are the
entries above N's top digit when a borrow runs up, the run of 9s of CHAIN.md §5.2. There the hand-written adder's carry
priced the run correctly and broke the stall. Here the adder that should price it is built from the same wrong
entries, so the stall holds. After 200 warm steps (197/200 right, that family fixed by the hand-written adder) the
learned adder finishes the table.

A plain fact about numbers breaks this trap: x − N wraps around to 10⁹ + x − N, which is at least N (N < 10⁸), so
(x − N) − N must not borrow. With (0, 0, 1) writing 0 instead of 9 the wrapped number is small and the second link
borrows. With that lesson (`ANCHOR_WRAP`) the same 100-step warm start finishes: 199/200 at step 175 and 200/200 by
step 275. From a random start it changes nothing (37/200 after 300 steps with the borrows held true). The final file
has it on.

### 5.2 The warm start (`ADDER_WARM`)
So the recipe that works keeps the hand-written adder for the first steps only. For steps 1…`ADDER_WARM` the check
uses it (and the learned adder's own rules are off); the order facts, R − 0 = R, R − R = 0 and the wrap-around fact
run throughout. From then on the check is the learned adder's, with every anchor, and phase B, the answers and the
freeze are CHAIN.md's. Harness runs at the official batch (512, L4, up to 1,000 steps):

| warm steps | seeds | frozen at step (rows right at the end) |
|---|---|---|
| 10 | 1, 2 | never (112, 99) |
| 30 | 1, 2 | never (150); 897 (200) |
| 50 | 1, 2 | 338, 339 (200, 200) |
| **70** | 1, 2 | **328, 343** (200, 200) |
| 70, never sees an answer (`REAL_MAX_T = 0`) | 1, 2 | 328, 329 (200, 200) |
| 70, without the wrap-around fact | 1, 2 | never (168, 170) |
| 100 | 1, 2 | 329, 324 (200, 200) |
| for comparison: CHAIN.md's recipe, hand-written adder throughout | 1, 2 | 614, 606 |

A typical run (70 warm steps, seed 1): 180/200 right at step 50 and about 185 at the hand-over; five steps later the
wrong entries are the (r, r, 1) family (CHAIN.md's old borrow saddle) and a few (r, 0, 1). The learned adder fixes
them: 193/200 at step 100, phase B at step 125 (199/200), 200/200 at step 150, frozen at 328, earlier than CHAIN.md's
recipe at the same batch (606–614 in this harness, 417 in `chain_v2_official`). The final file uses 70 steps, the
length of CHAIN.md's phase A at this batch (`chain_v2_official` handed off at step 71), with margin over the 50 that
was enough on both seeds.

## 6. Route B: a second table
`ADDER = "table"`: a second cell (r, m, carry in) → (digit, carry out), 2,400 more logits, trained with the subtraction
table. Its lessons:
- the check (D − N) + N = D runs through it, complete (the link's final borrow equals the adder's final carry);
- (R + N) − N = R (`ANCHOR_INV2`), 0 + N = N and R + N = N + R on the registers;
- the subtraction table's own anchors (R − 0 = R, R − R = 0, the order facts) and, in phase B, "x mod N = x" and the
  answers.

Results (§1.2, batch 512, 1,500 steps): without the order facts, 28 and 32 subtraction rows right and 28 and 34 adder
rows (true addition r + m + c); with them, the subtraction borrows end right (200 and 199 of 200) and 36 and 42
subtraction rows, 33 and 39 adder rows; with the answers from step 1 as well, 30 and 33 subtraction rows, 36 and 38
adder rows. Locally (batch 64), with the subtraction table's borrows held true: 31 and 33 after 375 steps.

The wrong solutions are of the same kind as route A's. The two inverse identities only ask the two tables to undo each
other: any one-to-one "subtract m" paired with its inverse as "add m" passes them, digit by digit. Commutativity and
0 + N = N narrow that down, but every relabeled number system of §3 still passes with both tables relabeled together.
This is the mutually inverse wrong pair that TINYCORE.md §7 warned about. The subtraction table's order facts fix its
borrows as in route A, and nothing fixes the digits.

## 7. The learned adder, for squaring work
In [`submission_adder.py`](submission_adder.py):
- `add_learned(A, B, Td, Tb) -> (digits, carry)`: A, B are `[X, width]` digit tensors, LSB first, any width; `Td, Tb`
  the snapped table (`model.tables()`, flat over (r, m, c) → r·20 + m·2 + c). It runs two ripples of the table,
  C = 0 − B and A − C, and returns A + B mod 10^width with the final carry borrow(0 − B) − borrow(A − C).
- `add_learned_soft(A, B, TDC) -> (digit distributions, P(carry = 1))`: the same on digit distributions
  `[X, width, 10]`, the cell applied multilinearly; `TDC` is the softmaxed table laid out as in `lessons` (`[20, 120]`).
- `sub_soft(D, M, TDC)`: a subtraction with both operands soft (`link_soft` needs exact digits for the subtrahend).
- `link_hard` and `link_soft` now run over any width.

How exact the addition is on its own (`adder_local/adder_exact.py NAME`, checked against Python integers outside the
submission):

| table | table rows right | every pair, widths 1 / 2 / 3 | random pairs, widths 9 and 18 | widths 36 and 72 |
|---|---|---|---|---|
| true subtraction (reference) | 200/200 | 100/100, 10⁴/10⁴, 10⁶/10⁶ | 200,000/200,000 each | 20,000/20,000 each |
| `chain_v2_official`'s learned table (hand-written adder throughout) | 200/200 | 100/100, 10⁴/10⁴, 10⁶/10⁶ | 200,000/200,000 each | 20,000/20,000 each |
| **`adder_warm_official`'s learned table** (this file) | 200/200 | 100/100, 10⁴/10⁴, 10⁶/10⁶ | 200,000/200,000 each | 20,000/20,000 each |

Every pair of 3-digit numbers covers every (digit, digit, carry in) case at every place. Route A has no addition table
of its own: its addition is exact exactly when the subtraction table is, so any certified table gives squaring work an
exact learned adder. The one this file learns needed the hand-written adder for its first 70 training steps only.

## 8. Side finding: the soft division overflows over 8 steps
`divide_soft` (the soft own-borrow choice over the chain) was only ever run for one division step ("x mod N = x").
Over the 8 steps of a full division it overflows in float32 within a few training steps: the mass of the digit
distributions drifts by about 10⁻³ after 9 links and goes to infinity by the third of the 8 division steps. Each
softmaxed table row sums to 1 only to within about 2·10⁻⁷, and the multilinear ripple multiplies those errors along
every path through the places and links. `submission_adder.py` renormalizes the digits and the borrow after each
link; the one-step use is unchanged in effect. `submission_chain.py` is unaffected (one step only).

## 9. Gray areas against the rules
Same as CHAIN.md §6, except:
- **Rule 14, "no data augmentation, task-specific solvers."**
  - The training check still uses the designed adder, but for the first 70 steps only (CHAIN.md: the whole of
    training, including the 200 checked steps before the freeze). From step 71 on the check's adder is the learned
    table, and so is every other subtraction and addition in training.
  - The comparison "is D below N?" that CHAIN.md's designed carry delivered (its §6) now comes from facts about the
    rows: x − N borrows and N − x does not (x < N, the fact "x mod N = x" already used), and 0 − N borrows (N > 0).
  - (x − N) − N does not borrow uses the register width: 9 places and N < 10⁸, so a wrapped result is at least N. That
    is knowledge of the designed representation, like the long-division layout itself.
  - The learned adder uses the constant 0 (as 0 − B), as R − 0 = R and R − R = 0 do.
- **Rule 7, the schedule, the freeze, Adam's reset, argmax snapping, rule 8.** Unchanged: the datapath is CHAIN.md's.

## 10. Next steps
1. **Shrink the warm start to nothing.** What is left of the hand-written adder is the first 70 training steps (50
   were enough on both seeds tried, 30 on one of two, 10 on none). §5 says what the warm start supplies: about half the
   digit entries, with the borrows (which the order facts already teach). Untested ideas, roughly from most to least in
   the spirit of the rules:
   - more answers per step, since only 2% of the rows carry a usable one (a larger batch, or more of the T ≥ 4 rows
     once the table is partly right);
   - a lesson that targets the borrow-in-1 half, the half that pulls in the rest (§5); I found none;
   - structure in the cell, for instance reading (r, m + c) instead of (r, m, c): fewer entries to find, but that is
     designing part of the rule.
2. **Route B** brought no new handle: two tables have the same wrong solutions (pairs of mutually inverse maps, the
   relabeled systems), as TINYCORE.md §7 warned.
3. **For the squaring work**, `add_learned` is ready: exact with the table this file learns (§7), and a learned
   multiplier can be built on it.

## 11. Reproduce
Everything heavy runs on Modal, launched from the one-layer-deeper repository root. The official runs use CHAIN.md's
runner with its `--file` option (no change to `modal_chain.py`); records land on the Volume `learned-remainder` and
`wait_fetch.py` downloads them to `cpu_first/history/runs/learned_remainder/` (full log, RESULT_JSON, the exact source).

The final file as it is (70-step warm start):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --file submission_adder.py --name adder_warm_official
```

From a random start (`adder_official` ran the earlier version of the file, whose default this was):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --file submission_adder.py --name adder_official_rerun --set '{"ADDER_WARM": "0"}'
```

The plumbing check with the exact table:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --file submission_adder.py --name adder_s0_official --set '{"MODE": "\"designed\"", "MAX_STEPS": "100"}'
```

The CPU seeds (`plans/a1_cpu.json`: 8 seeds with the answers, 4 that never see one; one container each):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode plan --file submission_adder.py --plan /path/to/cpu_first/learned_remainder/plans/a1_cpu.json
```

Fetch, then tabulate with the table audit:

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/wait_fetch.py adder_warm_official adder_official adder_s0_official
```

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/summarize.py --table adder_s0_official adder_warm_official adder_official
```

The training runs of §1.2 (`adder_local/modal_diag.py`: Modal app `learned-adder-diag`, CPU containers or L4 GPUs, one
job per run; the client waits, so launch it in the background). Logs land in `adder_local/logs/`:

```bash
~/github/sotaku/venv/bin/python -m modal run /path/to/cpu_first/learned_remainder/adder_local/modal_diag.py --plan /path/to/cpu_first/learned_remainder/adder_local/plans/sweep4.json
```

```bash
cd cpu_first/learned_remainder/adder_local && ~/github/sotaku/venv/bin/python table_rows.py
```

Files:
- **`submission_adder.py`.** The final file (route A, every anchor on, 70-step warm start, `MAX_STEPS = 3000`), run byte
  for byte as `adder_warm_official`. `ADDER_WARM = 0` gives the from-scratch recipe; the other variants of §4–§6 are its
  constants (`ADDER`, `ANCHOR_*`, `ID_CURRIC`, `ANSWER_PATH`, …).
- **`plans/a1_cpu.json`.** The CPU seed plan.
- **`adder_local/`**, each local run under a minute and well under 1 GB:
  - `landscape.py`: the table of §3;
  - `local_adder.py designed`: the exact table passes every check, and the adder is exact;
  - `local_adder.py train STEPS B [SEED] [K=V …]`: a training run that prints the snapped table against the truth
    and the closest wrong table of §3 every 25 steps. Diagnostics through the environment: `FIX=b` holds the borrow
    table at the truth, `TRUE_FRAC=f` or `TRUE_ROWS='<expression in r, m, c>'` starts digit entries at the truth
    (§5), `DEVICE=cuda`;
  - `adder_exact.py NAME`: §7's check of a run's snapped table;
  - `modal_diag.py`; `plans/sweep1.json` (its first launch stopped after the three runs of CHAIN.md's recipe) and
    `plans/sweep1b.json` (the 24 route-A runs; batch 64, CPU), `plans/sweep2.json` (batch 512, L4), `plans/sweep3.json`
    and `plans/sweep4.json` (the final file and its warm-start lengths); `summarize_logs.py`, `table_rows.py`; `logs/`
    and the sweeps' one-line summaries `logs_sweep*.txt`.

  They need torch and the H1 jsonl files (`H1_DATA`, as in `cpu_first/history/h1.py`); `../chain_local/stub/` stands in
  for the evaluator's `benchmark` API.
- **`submission.py`, `submission_chain.py`, `modal_chain.py`.** Untouched (md5 `bd477b9a…`, `ed360cb1…`, `ad5fbd9c…`).
