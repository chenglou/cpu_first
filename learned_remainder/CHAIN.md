# H1 with the multiples of N built from the learned subtraction table

## In plain words

**The task.** The model reads three numbers written as digits: a modulus N, a starting number x and a count T. It must
square x, keep only the remainder after dividing by N, and repeat that T times. To be certified it must get every one
of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, both for moduli it saw in training and for larger moduli
it never saw (up to 8 digits).

**What changed.** The calculator of [LEARNED_REMAINDER.md](LEARNED_REMAINDER.md) does long division the way a
schoolchild does. For each digit of the quotient it tries the ten candidates 0·N, 1·N, …, 9·N and keeps the largest one
that fits. Its subtractions already used a learned rule: a 200-entry table, "this digit minus that digit, minus a borrow,
gives this digit and maybe a new borrow". But the ten multiples of N were computed by hand-written multiplication.

Those multiples are gone. The calculator now only knows N's digits, as the prompt writes them. To try "3·N" it subtracts
N, then N again, then N again, each time with the same learned table. It stops at the first subtraction that goes below
zero and keeps the result just before it (the CPU textbook's "restoring division"). So every piece of arithmetic in the
remainder is now the learned table.

**What is still designed:**
- reading and writing the digits;
- squaring the number (integer multiplication);
- the long-division layout;
- the chain's wiring (up to nine subtractions of N per quotient digit);
- the stop rule ("keep the last result before the first borrow");
- the loop over T;
- for training only, a hand-written adder for the self-check.

**Did it certify?** Yes, on the untouched official H1 manifest on an H100:
- all 768 questions right at every T up to 64, on seen and on unseen moduli;
- the learned table is exactly right (200 of 200 entries), and every entry used at evaluation was trained;
- training: 272 s (the table is done after 417 steps, then it holds still);
- evaluation: 15 s, up from 5 s, because the chain is nine subtractions deep where the ten multiples used to be tried
  side by side. The budget is 1,800 s.

The same recipe certified on 8 of 8 further seeds on CPU, and on 4 of 4 seeds that never see an answer.

**What broke.** The first recipe moved the previous version's lessons onto single subtractions. It certified on the
official run and on 8 of 8 CPU seeds, but:
- three of those seeds sat stuck for roughly 350 to 1,150 steps;
- one seed that never sees an answer never got out. When its hour of training ran out it still had one entry wrong,
  and it failed certification.

One table entry, "0 minus 0 with a borrow", settled on a blur that the self-check prefers. The self-check subtracts N,
adds N back and compares. Writing 0 where the truth is 9 costs one wrong digit in it, and every step toward 9 first
costs more at the digits above it.

The fix completes the self-check. It also compares the subtraction's final borrow with the adder's final carry. That is
the one output the digits alone did not check, and it is exactly what the stop rule reads. With it, every seed was done
in 408 to 544 steps.

## 1. Results

Official manifest `benchmark/manifests/h100_hard_h1.json`, untouched:
- cuda:0, bf16 autocast, the batch from the submission;
- 3,600 s training budget, 1,800 s evaluation budget;
- Modal H100 80GB, with the image and H1 data regenerated with the official generator arguments (`modal_chain.py`, the
  same image as `modal_learned_remainder.py`);
- seed 74. Every rung is 768 prompts.

| run | what ran | max T (seen moduli) | OOD-N max T | rungs T=1…64, seen / OOD-N | test, ood_t, ood_n_t | subtractor rows right | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|---|---|---|
| `chain_s0_official` | the chain with the exact table (`MODE = "designed"`, 100 steps): a plumbing check | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 (designed) | 3.9 (–) | 16.9 |
| `chain_official` | first recipe (`ID_BORROW = 0`), md5 `8bb9d146…` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 245.7 (516) | 15.7 |
| **`chain_v2_official`** | **`submission_chain.py` byte for byte**, md5 `ed360cb107e644615e30c910ba6f6a2d` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | 271.8 (417) | 15.4 |
| for comparison: `final_s2_official` | `submission.py`, designed multiples (LEARNED_REMAINDER.md) | 64 | 64 | 768 everywhere | all exact | 200/200 | 181.9 (499) | 4.9 |

Both learned runs train every weight: 2,400 logits, 200 rows × (10 digit + 2 borrow).

Most of each training run is the frozen remainder of the 20,000 steps, at about 8 ms each:
- `chain_official` reached step 600 at 85 s;
- `chain_v2_official` reached step 500 at 104 s.

The same submissions through the same unmodified evaluator on Modal CPU containers, with the manifest's device set to
cpu and everything else official, one seed per run:

| recipe | seeds | certified T = 64 on both profiles | table rows right | frozen at step | training s / evaluation s (CPU) |
|---|---|---|---|---|---|
| first recipe (`ID_BORROW = 0`), with the rows' answers | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** | 200/200 | 516, 545, 524, **824**, 411, 497, **1,489**, **1,467** | 324–830 / 17–25 |
| first recipe, identity only (`REAL_MAX_T = 0`: never sees a label) | 74, 1, 2, 3 | 3/4: seed 3 **failed** (no rung certified: 640→570 of 768 seen, 530→413 OOD-N; test 5,673/9,999) | 200/200; seed 3: 199/200, only (0, 0, borrow 1) wrong | 641, 518, 493; seed 3 never (51% of the checks when the 3,600 s ran out, step 15,150) | 278–3,600 / 17–25 |
| **final** (complete identity), with the rows' answers | 74, 1, 2, 3, 4, 5, 6, 7 | **8/8** | 200/200 | 417, 481, 457, 446, 464, 434, 456, 474 | 306–502 / 18–31 |
| **final**, identity only | 74, 1, 2, 3 | **4/4** | 200/200 | 408, 466, 443, 544 | 282–355 / 19–24 |

Every certified run's audit reads `AUDIT eval_rows_used=200/200 used_but_never_trained=0`. The final recipe's
training checks used every row at least 161,137 times. The CPU run of seed 74 took the same path as the H100 run
(frozen at 417, and at 516 under the first recipe): identity-mode training has no random numbers in it, as in
LEARNED_REMAINDER.md §4.2. For comparison, `submission.py` froze at steps 499–815 on its 8 CPU seeds and at 661–715
identity-only (LEARNED_REMAINDER.md §1).

## 2. What is designed and what is learned

| part | `submission.py` (Stage 2) | `submission_chain.py` |
|---|---|---|
| read N (LSB first), x and T from the prompt; write the answer's digits right-aligned | designed | designed |
| loop T times | designed | designed |
| square the state | designed (integer product) | designed (integer product) |
| long-division wiring: start from v // 10^8 (below N), shift in one digit per step, 8 steps, 9 register places | designed | designed |
| N inside the reduction | its integer value, multiplied by s = 0…9 (designed), then cut into digits (designed) | **only its digits, as the prompt gives them**; no integer N anywhere in the datapath |
| the trials of a step | ten side by side: trial s = Rp − s·N | **a chain**: trial 0 = Rp, trial s = trial (s − 1) − N, s = 1…9 (designed wiring) |
| **the subtractor cell** (r, m, borrow in) → (digit, borrow out), one 200-row table shared by every place, link, step and width | **learned** (2,400 logits) | **learned** (the same kind of table, 2,400 logits); now also does the work of the multiples |
| which trial is kept | designed rule: the largest s whose trial does not borrow | designed stop rule: the last trial before the chain first borrows |
| what teaches the cell | label-free checks on the rows' own numbers ((Rp − s·N) + s·N = Rp, "x mod N = x"), then the rows' answers | the same lessons on single links ((D − N) + N = D, completed with the final borrow), "x mod N = x", the rows' answers |
| adder in the training check | designed (adds s·N) | designed (adds N; its final carry is now checked too) |

Evaluation snaps the learned table to its argmax and runs the whole machine on integers.

## 3. The datapath: a chain of subtractions instead of ten multiples

Each long-division step forms the register Rp = 10 R + v_t, as before. `submission.py` then subtracted the exact digits
of 0·N, 1·N, …, 9·N from Rp, all ten at once. `submission_chain.py` instead runs a chain (`step_hard`):
- trial 0 is Rp itself;
- link s subtracts N's digits from trial s − 1, place by place from the lowest, with the learned table (`link_hard`);
- the stop rule keeps the last trial before the first link that ends in a borrow. If none of the nine links borrows, it
  keeps trial 9.

With the exact table this is restoring division:
- R < N, so Rp ≤ 10N − 1, and at most nine subtractions bring it below N;
- trial s equals Rp − s·N for as long as nothing has borrowed;
- the kept trial is Rp mod N and its index is the quotient digit.

`chain_s0_official` checks the plumbing with the exact table. It certified T = 64 on both profiles through the official
manifest, and a local run reproduced every answer of all 14 depth rungs.

**What it costs.** One reduction:
- before: 8 steps × 9 places = 72 lookups in sequence, ten trials side by side (720 lookups in all);
- now: 8 steps × 9 links × 9 places = 648 lookups in sequence (648 in all).

A T = 64 answer is 41,472 table lookups in a row. Evaluation went from 5 s to 15–17 s on the H100, and from 9–14 s to
17–31 s on CPU. Either is far inside the 1,800 s budget. Every lookup is a few small kernels over the batch, so the
time presumably goes to launches rather than arithmetic (not profiled).

**The coverage audit.** At its first evaluation batch the submission prints:
- its snapped table;
- how often training's hard checks used each row;
- which rows evaluation used.

`summarize.py` compares the table with true subtraction outside the submission. Evaluation counts every link it
computes, including links after the chain has stopped, whose results it throws away. That count is stricter than
needed. In `chain_v2_official`:
- `AUDIT eval_rows_used=200/200 used_but_never_trained=0`;
- training's hard checks used every row at least 167,473 times (median 390,524).

The table is position-generic, so the unseen 8-digit moduli use the same 200 rows as the 4–7-digit training moduli.

## 4. How the table learns

These are the lessons of `submission.py`'s Stage 2 (LEARNED_REMAINDER.md §4.2), moved from the ten trials onto the
chain's single links:

- **Phase A: every link checks itself.**
  - **The check.** For each training row's x², the hard machine (argmax table) runs its own division and records the
    input D of every link at every step: 512 rows × 8 steps × 9 links = 36,864 checks per training step. Each check
    runs D − N with the soft table (a distribution over digits and borrows, the cell applied multilinearly), adds N
    back with a designed adder, and asks for D's digits.
  - **The final borrow (`ID_BORROW = 1`).** The check also asks that the subtraction's final borrow equal the adder's
    final carry. As whole numbers, D − N = digits − borrow·10⁹ and digits + N = D + carry·10⁹, so (D − N) + N = D
    holds exactly when the digits come back and borrow = carry. The first recipe checked only the digits, that is,
    the identity mod 10⁹ (§5).
  - **The lowest place weighs 5×** (`ID_PLACE0`), as before. It is the one place where the true borrow code and the
    swapped one differ (LEARNED_REMAINDER.md §3.4).
  - **Every link is checked**, including those after the chain first borrowed. Their inputs are wrapped-around numbers
    the table itself made: still valid checks, since the identity holds for any D.
  - **No label, no made-up number.** The inputs are the rows' x and N and what the machine does with them.
- **Hand-off** once the hard checks pass on 80% of the links over 20 steps, or after 300 steps. Adam's state is reset.
- **Phase B: two more lessons.**
  - **"x mod N = x".** x < N, so one division step from x // 10 must stop the chain at trial 0, which means the first
    link, x − N, must borrow. It uses the soft own-borrow choice over the chain: trial s is kept with the probability
    that links 1…s do not borrow and link s + 1 does. For x < N this is dominated by the first link.
  - **The rows' answers along the model's own hard path,** for the batch's rows with T ≤ 2. The hard machine picks
    each step's chain length (the quotient digit). The soft table then runs that many links from the soft register,
    and the soft remainder is scored against the label.
- **Freeze** (learning rate 0) after 200 consecutive steps in which every check and every "x mod N = x" is exact on the
  hard machine.

**Why TINYCORE's blurring did not bite.** TINYCORE found that soft chains of repeated subtractions blur as they get
longer: four chained soft subtractions at base 5 reached only 0.30–0.46 ([TINYCORE.md](../history/tinycore/TINYCORE.md)
§4). Here no lesson needs a long soft chain while the table is still vague:
- phase A's checks are single links on hard inputs;
- phase B's soft chains follow the hard machine's own chain length, on a table that phase A has already made nearly
  right;
- "x mod N = x" is dominated by its first link.

## 5. What broke and why

### 5.1 The first recipe: certified, but some seeds sit at half the checks
The first recipe was Stage 2's lessons moved onto single links unchanged, with the check covering the digits only
(`ID_BORROW = 0`). It certified on the official run (`chain_official`, frozen at step 516) and on 8/8 CPU seeds. But:
- seeds 3, 6 and 7 spent roughly 350, 1,000 and 1,150 steps of phase B at 49–52% of the checks passing (hard exact
  0.50–0.52 at every log line from step 250 to 1,250 for seeds 6 and 7) before they froze at 824, 1,489 and 1,467;
- the identity-only run of seed 3 sat at 51% until training hit the 3,600 s cap (step 15,150) and failed
  certification. Its table ended 199/200 right, wrong only at (0, 0, borrow 1), the row of §5.2. Test: 5,673/9,999
  exact. Rungs from 640 down to 570 of 768 on seen moduli, and from 530 down to 413 on unseen ones.

### 5.2 The stalled state
Reproduced locally (`chain_local/diag_chain.py 700 64 6`: seed 6, batch 64, digits-only check), printing the snapped table against true subtraction every
25 steps:
- **Entering phase B.** Rows (0, 0, borrow 1) → digit 0, borrow 1 (truth: digit 9, borrow 1) and (9, 0, borrow 1) →
  digit 8, borrow 1 (truth: digit 8, borrow 0) were wrong together.
- **Later.** Only (0, 0, borrow 1) was wrong, from step 550 to the end of the 700 steps, with the digits-only check
  passing on 48.8% of the links.

Its digit was a blur: P(0) = 0.67–0.70, P(9) = 0.30–0.32, borrow right (0.997–0.999). At the official batch the
failed identity-only seed 3 ended with exactly this row wrong.

That row is the cell at the zeros above N's top digit when a link goes negative (D < N). The true result there is a run
of 9s (the wrap-around): adding N back carries through all of them and every place comes back 0. Written as 0s
instead, the first place above N comes back 1 (wrong), and every place above it comes back 0 (right, because no carry
reaches it). Since the 9s and 0s all come from the same row, the soft check prices a move of that row toward 9 at
every one of those places at once:
- the first place gains;
- every place above it loses, because a partial carry now spoils its 0.

**The chain feeds the stall.** With the true 9s, a wrapped result is a huge number, so the links after it do not go
negative again: one negative link per step. With the wrong 0s on top, the wrapped result is small, so the chain keeps
going below zero, and every such link fails the check at one place. Measured with designed tables on the division of
1,024 training rows' x² (`chain_local/neg_links.py`):

| table | links that go negative | per step (of 9) | digits-only check passes |
|---|---|---|---|
| exact | 10.8% | 0.97 | 100% |
| exact except (0, 0, borrow 1) → digit 0 | 49.3% | 4.44 | 48.9% |

The second row is the plateau the stalled seeds sat on (49–52%).

Measured on that stalled table (`chain_local/probe_barrier.py`), sliding only that row's digit logits
from 0 toward 9 and computing the check's soft loss on fixed link inputs from 1,024 training rows:

| P(digit 9) of row (0, 0, 1) | 0.10 | 0.19 | 0.32 (stalled) | 0.50 | 0.67 | 0.81 | 0.90 | 0.97 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| digits-only check (first recipe) | 0.133 | **0.126** | 0.136 | 0.156 | **0.158** | 0.126 | 0.088 | 0.049 | 0.035 |
| complete check (digits + final borrow) | 0.289 | 0.282 | 0.283 | 0.268 | 0.227 | 0.166 | 0.113 | 0.063 | 0.046 |

- **Digits-only check.** Along this line it has a local minimum near P(9) ≈ 0.2, where the argmax digit is 0 (wrong).
  Its barrier peaks near 0.67, about 0.03 nats per check above that minimum.
- **Complete check.** It has no such barrier: a bump of 0.001, then downhill all the way to the truth.

This is the saddle that LEARNED_REMAINDER.md §4.2 met in the parallel version: rows (r, r, borrow 1) "sit at digit 0
or 9 50/50", because "the adder's carry makes up for a wrong digit". There, "x mod N = x" moved the stuck borrow. Here
the borrow is already right and the digit is stuck. "x mod N = x" only asks whether x − N borrows, so it never looks
at that digit. Why the parallel version stalled less here (8/8 seeds frozen by step 815) was not investigated.

### 5.3 The fix: the complete identity
The final borrow of a link is the stop rule's signal. The digits-only check never looked at it: the identity held only
mod 10⁹. Checking it against the designed adder's final carry completes the identity. It also prices the run of 9s
correctly: a carry out of the top needs every place in the run to be 9. Locally (batch 64) with `ID_BORROW = 1`:

| seed | 3 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| frozen at step | 453 | 465 | 513 | 491 | 511 |

The digits-only check had seed 6 stuck and seed 3 stalled for about 400 steps at the same batch.

On Modal (batch 512), the final file:
- certified on the official run: frozen at 417 instead of 516;
- on CPU: 8/8 seeds frozen at 417–481 with the answers (first recipe: 411–1,489), and 4/4 at 408–544 identity-only,
  including seed 3, on which the first recipe failed.

### 5.4 Side notes
- **Checking only the live links does not fix it.** These are the links up to and including the first that borrows
  (`ID_LINKS = "live"`). With the digits-only check, seed 6 stayed stuck locally with the same two rows. Combined with
  the complete identity it froze at step 436.
- **"x mod N = x" still helps.** With the complete identity but without it, seeds 6 and 3 froze locally at 580 and 795
  steps, with the (0, 0, 1) borrow flickering. With it they froze at 465 and 453.
- **A transient after "x mod N = x" joins.** Locally, seed 1 under the first recipe:
  - phase A ends with only (0, 0, borrow 1) wrong, so x − N never borrows at the top and "x mod N = x" fails on every
    row;
  - "x mod N = x" fixes that row within about 60 steps;
  - its soft ripple then pushes (0, 0, borrow 0)'s borrow to 1 for about 200 steps (the check at 69%), until the check
    pulls it back.

  The likely mechanism is the soft-lesson bias of LEARNED_REMAINDER.md §3.3: while the borrow coming into the top
  places is still soft, "the final borrow must be 1" also lands on the borrow-0 row. In `submission.py` the same lesson
  also asked trial 0, x − 0·N computed by the table, not to borrow, which held that row at no borrow. In the chain,
  trial 0 is Rp itself, with no subtraction in it.

## 6. Gray areas against the rules
Same as LEARNED_REMAINDER.md §5, except:
- **Rule 7, "no hard-coded algorithm in the forward pass."** The multiples s·N are no longer designed arithmetic, and
  the datapath never forms N's integer value. Still designed:
  - the digit I/O and the integer square;
  - the long-division wiring and the chain's wiring;
  - the stop rule;
  - the T loop.

  It is still a designed CPU with one learned gate table, now doing all of the reduction's arithmetic.
- **Rule 14, "no data augmentation, task-specific solvers."**
  - The training check still uses a designed exact adder.
  - It now also compares the adder's final carry with the learned final borrow. The carry says whether D − N wrapped
    around, which is the comparison "is D below N?" that the stop rule needs. So that knowledge reaches the table from
    the designed adder, by way of the identity.
  - The first recipe got the same knowledge only from "x mod N = x" (the fact x < N) and the rows' answers, and
    certified that way too (`chain_official`, 8/8 CPU seeds), just more slowly on some seeds.
- **Schedule, freeze, Adam reset, argmax snapping, rule 8.** Unchanged.

## 7. Next pieces to remove
Following LEARNED_REMAINDER.md §6, with item 2 done:
1. **The designed adder in the check.** TINYCORE warns that a learned adder pinned by its own identities admits
   mutually inverse wrong pairs unless something anchors them. The rows' answers and "x mod N = x" are candidate
   anchors. Here the adder now carries the comparison too (§6), so removing it removes that lesson's source.
2. **The stop rule** (and before it, in `submission.py`, the "largest trial without a borrow" rule). Restoring division's
   rule reads one borrow per link, a simpler thing to learn than a choice among ten trials.
3. **The squaring.** The chain's subtractor cannot do it. A learned multiplier would bring its own table and lessons,
   and the labels cannot teach arithmetic upstream of a reduction (REPORT.md §2).
4. **The long-division wiring, the chain's length, the loop counter and the I/O.**

## 8. Reproduce
Everything heavy runs on Modal. Launch the runner from the one-layer-deeper repository root: it builds the competition
image from it and regenerates H1 inside with the official generator arguments. Results land on the Modal Volume
`learned-remainder`, and `wait_fetch.py` downloads them to `cpu_first/history/runs/learned_remainder/<name>.json` (full
log, RESULT_JSON, the exact submission source).

The final file as it is:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name chain_v2_official
```

The first recipe:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name chain_official_rerun --set '{"ID_BORROW": "0.0"}'
```

The plumbing check with the exact table:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name chain_s0_official --set '{"MODE": "\"designed\"", "MAX_STEPS": "100"}'
```

The CPU seed plans (one container each):
- `plans/c1_cpu.json` ran the first recipe, which was the file's default when it launched; with the final file, add
  `"ID_BORROW": "0.0"` to each entry's `set`;
- `plans/c2_cpu.json` runs the final recipe.

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode plan --plan cpu_first/learned_remainder/plans/c2_cpu.json
```

Fetch, then tabulate with the table audit:

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/wait_fetch.py chain_v2_official chain_official chain_s0_official
```

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/summarize.py --table chain_s0_official chain_official chain_v2_official
```

Files:
- **`submission_chain.py`.** The submission, certified byte for byte as `chain_v2_official`. `ID_BORROW = 0.0` gives
  the first recipe (`chain_official` ran that recipe's file, whose exact source is in its record). `MODE = "designed"`
  gives the plumbing check.
- **`modal_chain.py`.** A copy of `modal_learned_remainder.py` with a `--file` option (default `submission_chain.py`)
  and its own app name, `learned-chain`. Same image, same Volume.
- **`plans/c1_cpu.json`, `plans/c2_cpu.json`.** The CPU seed plans.
- **`chain_local/`.** The small local checks behind §3 and §5, each under a minute and well under 1 GB:
  - `local_chain.py designed`: the chain with the exact table against every depth rung's answers;
  - `local_chain.py train STEPS B [SEED]`: a short training run;
  - `diag_chain.py`: the wrong rows during a run;
  - `probe_barrier.py`: the loss along the move of row (0, 0, 1);
  - `neg_links.py`: negative links under the exact and the stalled table.

  They need torch and the H1 jsonl files (`H1_DATA`, as in `cpu_first/history/h1.py`). `stub/` stands in for the
  evaluator's `benchmark` API.
- **`submission.py`, `modal_learned_remainder.py`.** Untouched (md5 of `submission.py`:
  `bd477b9aadc9fcf34f428f728b5bea20`).
