# Litmus: the best H1 model on tasks it was not built for

## In plain words

<!-- PLAIN -->
**The question.** The best H1 model ([`submission_wide_anchored.py`](../wider_menus/submission_wide_anchored.py)) is a
small paper-arithmetic calculator. It learns its own digit tables (what 7 − 5 and 7 × 5 are) and chooses its own
wiring from menus of about 4·10^17 possible machines. What teaches it is a set of lessons it can check on each
question's own numbers, such as "dividing x by N leaves x when x is smaller than N"; the answers in the data only
pick among a few reading switches at the very end. Did the project build a learning *method*, or a well-engineered
*solution* to H1? The test: run the same file, with the same menus, lessons and sizes, on tasks it was not built for,
changing only the data. 26 tasks, closest first:
- the competition's 20 practice datasets (Easy E1–E10, Medium M1–M10): the same recurrence, x^(2^T) mod N, with
  other data (mostly a single fixed modulus, short T, N written the usual way round);
- H1 with the prompt's fields in another order;
- two other formulas the menus can already compute: Montgomery's squaring (y² · 10^−8 mod N) and 10·y² mod N;
- three they cannot: y² + 1 mod N, x^(3^T) mod N, and x·y^T mod N with a fourth number in the prompt.

**What it learned unchanged.**
- **The same task in another layout.** H1 with its prompt's fields reordered certified exactly as H1 does: every
  question right at every T up to 64, on seen and unseen moduli.
- **The practice datasets whose moduli vary** (6 of the 20). Given H1's training time, five scored 100% and
  certified T = 64 on both ladders, and one (M5) scored 50.5% because it read T backwards (16 as 61), as it did on 2
  of 4 seeds: with T always a single digit in its training data, nothing tells it which way a two-digit T runs.
  Each one learned exact tables, the meaning of every digit, the reading this data needs (N written the usual way
  round, the opposite of H1), and long division, from data that differs from H1's in every respect but the
  recurrence: 27 to 6,074 moduli, 3,200 to 81,000 rows, and one of them with no answers it could use at all. Two of
  the Medium ones certified within Medium's own 600 s. No Easy one fits in Easy's 60 s: they became exact after
  270–520 s.

**What it could not learn, and why.**
- **The 14 practice datasets with one fixed modulus: nothing (at most 2.4%).** The machine learns what each digit token
  means, and which way each number is written, from "x is smaller than N" across many moduli. With one modulus that
  fact only ever compares a few digits with N's own (N = 323 never orders 4 to 9), so several readings and digit
  meanings fit every row: half the runs never chose a reading, and the other half chose one with scrambled digit
  meanings (six of them also read N backwards). The subtraction table, which gets its meaning from the same fact,
  never became exact either.
- **The two other formulas its menus hold, and the two one step outside them:** every unchanged run built a perfect
  x² mod N machine, the one its lessons describe, and answered the new question wrong. Its scores equal an exact
  x² mod N machine's on every split and rung. The answers could not override the lessons: they only choose among a
  few reading switches, and no combination of those switches reproduces them.
- **The four-number task:** the reader takes exactly three numbers and never started.

**What the others needed.**
- **y² + 1, cubing, ×10: one new option each.** Pollard's map and the ×10 task each needed one new switch, and cubing
  one new round option (multiply by the round's own input and reduce again) plus a second operand for the multiply:
  13–21 lines of code each. The answers the file already uses teach the new option, through the same mixture as its
  reading switches. Each certified T = 64 on both ladders, and each changed file still certifies H1 itself, choosing
  the old option there.
- **Montgomery's squaring: its lessons rewritten.** The menu already holds Montgomery's reduction, but the three
  lessons that teach the division describe long division. Restated for Montgomery, it certified on 2 of 4 seeds. On
  the other two the search found another Montgomery form whose commit gate breaks the final correction; a fourth
  change (the division lesson also checks the gate's role there) made it 4 of 4. Two of the four changes were found
  only by running it.
- **T's direction: one tie.** On data whose T always has one digit nothing tells "16" from "61", and the unchanged
  file read T backwards on 2 of 4 seeds of M5. Reading T the way the answer is written (one tie, 7 lines) fixed all 4.
- **One fixed modulus: no small change.** Even with the reader handed the right answer and held there, and even with
  hand-written arithmetic checking the table, the subtraction table did not learn from one modulus: every lesson that
  teaches it pairs a digit with one of N's own, so with one N some digit pairs never meet in any lesson and the rest
  are not held. It would take lessons built on other numbers, or the answers, which this design keeps away from its
  reader and its tables.
- **Four numbers: a redesign** of the reader and of the round (not tried).

**What this says.** A method within its family; an engineered solution across families.
- **What transfers is the learning core**: exact digit tables taught by lessons, and wiring chosen by trying whole
  machines. Unchanged, it handled a reordered prompt, numbers written the other way round, small moduli, 3% of H1's
  rows, and no usable answers at all; given the right lessons, it found Montgomery's reduction among the same billions
  of wirings, in two forms.
- **What does not transfer is the knowledge in the lessons.** They describe H1: squaring, then long division by N;
  many different moduli; T sometimes written with two digits. Where a task fits that description, the file learns it
  without help. Where it does not, it learns H1 anyway and ignores the answers, or stalls because the data cannot tell
  it what the lessons take for granted.
- **Every failing task needed a designer** to say what was different: one switch for an extra step, a prior for what
  the data cannot say (which way T is written), a rewritten set of lessons for another algorithm (with traps that
  showed up only when it ran), and for one fixed modulus, a different source of meaning altogether.

So the project built a well-engineered solution with a general search core, not a method that learns a new task from
its data.
<!-- /PLAIN -->

## Every task at a glance

"Unchanged" is the file byte for byte (for the tiers, with the relaxed budget of §1; the official budgets are in §3).
Scores are the evaluator's mean exact accuracy; "max T" is the certified T on seen / unseen moduli (0: not even T = 1).

<!-- SUMMARY -->
| task | unchanged: score, max T seen / unseen | unchanged: where it stopped | smallest change that made it work | runs with that change (fixed moduli: the diagnostics of §4) |
|---|---|---|---|---|
| E1 | 1.7%, max T 0 / 0 | subtraction table never exact (15.1% of its checks pass) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | 4.7%, max T 0 / 0 (e1_reader_frozen); 1.7%, max T 0 / 0 (e1_warm_frozen) |
| E2 | 0.1%, max T 0 / 0 | reading never chosen (posterior 0.953 < 0.99) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | 0.1%, max T 0 / 0 (e2_reader_given); 1.0%, max T 0 / 0 (e2_reader_frozen); 0.6%, max T 0 / 0 (e2_warm_frozen) |
| E3 | 100.0%, max T 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | none (more time than the tier's budget) | - |
| E4 | 100.0%, max T 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | none (more time than the tier's budget) | - |
| E5 | 100.0%, max T 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | none (more time than the tier's budget) | - |
| E6 | 0.7%, max T 0 / 0 | reading never chosen (posterior 0.965 < 0.99) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| E7 | 0.4%, max T 0 / 0 | reading never chosen (posterior 0.958 < 0.99) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| E8 | 0.4%, max T 0 / 0 | reading never chosen (posterior 0.958 < 0.99) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| E9 | 2.0%, max T 0 / 0 | reading never chosen (posterior 0.856 < 0.99) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| E10 | 1.7%, max T 0 / 0 | reading never chosen (posterior 0.898 < 0.99) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | 0.7%, max T 0 / 0 (e10_reader_frozen) |
| M1 | 0.0%, max T 0 / 0 | subtraction table never exact (28.4% of its checks pass) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | 0.2%, max T 0 / 0 (m1_reader_given); 2.3%, max T 0 / 0 (m1_reader_frozen); 2.3%, max T 0 / 0 (m1_warm_frozen) |
| M2 | 0.2%, max T 0 / 0 | reading never chosen (posterior 0.578 < 0.99) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| M3 | 100.0%, max T 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | none (more time than the tier's budget) | - |
| M4 | 100.0%, max T 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | none (more time than the tier's budget) | - |
| M5 | 50.5%, max T 8 / 8 | exact on every check and answer; T's direction never settled, so no freeze | one tie: T's pointer reads the way the writer writes, its anchor and step (variants/submission_tie_writer.py), 4 of 4 seeds; tying only its step to x's (variants/submission_tie.py) fixed 3 of 4 | 100.0%, max T 64 / 64 (m5_tiew_s74); 100.0%, max T 64 / 64 (m5_tiew_s1); 100.0%, max T 64 / 64 (m5_tiew_s2); 100.0%, max T 64 / 64 (m5_tiew_s3); 100.0%, max T 64 / 64 (tiew_on_h1) |
| M6 | 2.1%, max T 0 / 0 | subtraction table never exact (71.0% of its checks pass) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | 1.1%, max T 0 / 0 (m6_reader_frozen); 1.0%, max T 0 / 0 (m6_warm_frozen) |
| M7 | 0.0%, max T 0 / 0 | subtraction table never exact (62.1% of its checks pass) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| M8 | 0.3%, max T 0 / 0 | subtraction table never exact (39.2% of its checks pass) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| M9 | 0.2%, max T 0 / 0 | subtraction table never exact (70.0% of its checks pass) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| M10 | 0.5%, max T 0 / 0 | subtraction table never exact (31.0% of its checks pass) | none found: not even a given, frozen reader with hand-written arithmetic (§4) | - |
| xfirst | 100.0%, max T 64 / 64 | froze at step 631 | none (seeds 1-3 too) | 100.0%, max T 64 / 64 (xfirst_s1); 100.0%, max T 64 / 64 (xfirst_s2); 100.0%, max T 64 / 64 (xfirst_s3) |
| mont | 0.0%, max T 0 / 0 | exact on its own checks; 0.0% of the T <= 2 answers right | its three division lessons restated for Montgomery's reduction (variants/submission_mont.py: 2 of 4 seeds), plus a fourth: the division ladder also checks the gate's role in the final correction (variants/submission_mont_gate.py) | 0.1%, max T 0 / 0 (mont_fix2_s74); 0.1%, max T 0 / 0 (mont_fix2_s1); 100.0%, max T 64 / 64 (mont_fix2_s2); 100.0%, max T 64 / 64 (mont_fix2_s3); 100.0%, max T 64 / 64 (mont_gate_s74); 100.0%, max T 64 / 64 (mont_gate_s1); 100.0%, max T 64 / 64 (mont_gate_s2); 100.0%, max T 64 / 64 (mont_gate_s3) |
| ten | 0.0%, max T 0 / 0 | exact on its own checks; 0.0% of the T <= 2 answers right | one switch: the square times 10 before the reduction (variants/submission_ten.py) | 100.0%, max T 64 / 64 (ten_fix_s74); 100.0%, max T 64 / 64 (ten_fix_on_h1) |
| rho | 0.0%, max T 0 / 0 | exact on its own checks; 0.0% of the T <= 2 answers right | one switch: the square plus 1 before the reduction (variants/submission_rho.py) | 100.0%, max T 64 / 64 (rho_fix_s74); 100.0%, max T 64 / 64 (rho_fix_on_h1) |
| cube | 0.1%, max T 0 / 0 | exact on its own checks; 0.0% of the T <= 2 answers right | one round option: times the round's input, reduced again; the multiply takes a second operand (variants/submission_cube.py) | 100.0%, max T 64 / 64 (cube_fix_s74); 100.0%, max T 64 / 64 (cube_fix_on_h1) |
| xy | 0.0%, max T 0 / 0 | the reader never found exactly three numbers in a prompt (its first stage) | a redesign, not tried: a reader for four numbers and a round that multiplies by y | - |
<!-- /SUMMARY -->

## Details

### 1. The tasks

Every "unchanged" run uses [`wider_menus/submission_wide_anchored.py`](../wider_menus/submission_wide_anchored.py) (md5
`c82b11fa…`, the current best model, [WIDE_ANCHORED.md](../wider_menus/WIDE_ANCHORED.md)) byte for byte, through the
unmodified evaluator, on a Modal H100 with seed 74 (the manifests' seed). Only the data changes. §4's changed files
ran the same way, on H100s or, when H100s were scarce, on 8-CPU containers (the same evaluator with the device set
to cpu; the H1 file's CPU seeds follow its H100 schedule step for step, WIDE_ANCHORED.md). Three kinds of task,
closest first:

**(a) The practice tiers: the same recurrence, other data.** The competition's own Easy (E1–E10) and Medium (M1–M10)
datasets, generated by `scripts/generate_datasets.sh` with the official generator, run through their official
manifests. They square modulo N as H1 does, but:
- 14 of the 20 use one fixed modulus (E1: N = 323) and the other 6 (E3–E5, M3–M5) a few dozen to a few thousand
  sampled moduli of 10–22 bits (H1: 430 moduli of 13–20 bits);
- T is short: 1–4 in Easy, at most 16 in Medium, and in 18 of the 20 always a single digit;
- N's digits are written in the usual order (H1 writes them reversed);
- Easy trains for 60 s and Medium for 600 s (H1: 3,600 s), and both score mean exact accuracy, with H1's depth
  ladders reported as diagnostics. E1 trains on N = 323 alone, and its "unseen moduli" ladder uses 10- and 11-bit
  moduli.

Each was run twice: with its official manifest untouched, and with the budget relaxed to H1's 3,600 s and at most
1,500 steps (the best model froze at step 588 on H1), so that a failure is not just the clock.

**(b) Other tasks inside the machine's grammar.** H1's own generator (`data.reversed_n_squaring_mod`, H1's
arguments) with another tokenizer or another answer, written as [`tasks.py`](tasks.py). The generator draws (N, x, T)
from its seeded random stream before it computes the answer, so these tasks have H1's prompts row for row (checked);
only the answers, or the layout, differ:
- `xfirst`: x^(2^T) mod N, H1's answers, with the prompt's fields reordered: X x N n T t;
- `mont`: Montgomery squaring, y ← y² · 10^−8 mod N, T times. The division menu holds this reduction (REDC with
  R = 10^8; MENUS.md §1): set by hand, the unchanged file computes it exactly;
- `ten`: y ← 10 · y² mod N, T times. The multiply's menu holds it as a column offset (every product one column up):
  set by hand, the unchanged file computes it exactly. Its unseen moduli stop at 22 bits so that 10 y² keeps a top
  half below N, which long division needs.

**(c) Tasks outside the grammar.**
- `rho`: Pollard's map, y ← y² + 1 mod N: nothing in the menus adds a constant;
- `cube`: x^(3^T) mod N: a round squares once, and the multiply has one operand;
- `xy`: x · y^T mod N, with a fourth field Y y in the prompt: the reader takes exactly three numbers.

The answers of `tasks.py` were checked by recomputing every row of a small version step by step, and its prompts
against H1's generator run with the same arguments.

| task | kind | what differs from H1 | expressible by the file's datapath? |
|---|---|---|---|
| E1–E10 | (a) practice tier, Easy | the data: one fixed modulus (E3–E5: 27–35 sampled 10–12-bit moduli), T of 1–4, N written forwards, 60 s | yes: the same recurrence |
| M1–M10 | (a) practice tier, Medium | the data: one fixed modulus (M3–M5: 204–6,074 sampled 11–22-bit moduli), T of 1–16, N written forwards, 600 s | yes: the same recurrence |
| `xfirst` | (b) | the prompt's layout: X x N n T t | yes: the reader chooses which run is which number |
| `mont` | (b) | the answer: y ← y² · 10^−8 mod N, T times | yes: a wiring of the division menu (Montgomery's REDC) |
| `ten` | (b) | the answer: y ← 10 y² mod N, T times | yes: a layout of the multiply's menu (every column one up) |
| `rho` | (c) | the answer: y ← y² + 1 mod N, T times | no: nothing adds a constant |
| `cube` | (c) | the answer: x^(3^T) mod N | no: one square per round, one operand |
| `xy` | (c) | the answer: x · y^T mod N, and a fourth field Y y | no: the reader takes three numbers, the multiply one operand |

### 2. What the data shows the lessons

The file learns almost everything from lessons that need no answer (WIDE_ANCHORED.md, MENUS.md §5): x < N for the
reader's digit code and reading direction, x < N on every top part for the subtraction table's borrows, laws of
subtraction for its digits, the multiplication ladder, long division's laws on the rows' own numbers. The answers'
values teach only two things: the reading's division switches (through the rows with T = 1 or 2) and T's direction
(through the rows with a two-digit T); the answer tokens also show the writer where and how to write. Whether a lesson
can pin down what it teaches depends on the data. Counted on each task's training rows
([`audit_data.py`](audit_data.py), `audit.json`):

<!-- AUDIT -->
| task | training rows | moduli | their digits (digits: moduli) | layout | digit order x < N pins (adjacent pairs) | N's direction pinned | x's direction pinned | T in training | rows with answers used (T = 1, 2) | rows with a two-digit T |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 600 | 1 (fixed) | 3: 1 | NXT | 2/9 | no | yes | 1, 2, 3 | 400 | 0 |
| E2 | 1,920 | 1 (fixed) | 3: 1 | NXT | 2/9 | no | yes | 1, 2, 4 | 1,280 | 0 |
| E3 | 3,200 | 27 | 3: 6, 4: 21 | NXT | 9/9 | yes | yes | 2 | 3,200 | 0 |
| E4 | 6,400 | 35 | 4: 35 | NXT | 9/9 | yes | yes | 2 | 6,400 | 0 |
| E5 | 4,800 | 27 | 3: 6, 4: 21 | NXT | 9/9 | yes | yes | 1, 2, 3 | 3,200 | 0 |
| E6 | 588 | 1 (fixed) | 3: 1 | NXT | 2/9 | no | yes | 1, 2, 3, 4 | 294 | 0 |
| E7 | 528 | 1 (fixed) | 3: 1 | NXT | 3/9 | no | yes | 1, 2, 3 | 352 | 0 |
| E8 | 528 | 1 (fixed) | 3: 1 | NXT | 3/9 | no | yes | 1, 2, 3 | 352 | 0 |
| E9 | 600 | 1 (fixed) | 3: 1 | NXT | 2/9 | no | yes | 1, 2, 3 | 400 | 0 |
| E10 | 792 | 1 (fixed) | 3: 1 | NXT | 2/9 | yes | yes | 1, 2, 3 | 528 | 0 |
| M1 | 27,000 | 1 (fixed) | 5: 1 | NXT | 2/9 | no | yes | 4, 8, 16 | 0 | 9,000 |
| M2 | 81,000 | 1 (fixed) | 5: 1 | NXT | 4/9 | yes | yes | 4, 8, 16 | 0 | 27,000 |
| M3 | 21,600 | 247 | 4: 78, 5: 169 | NXT | 9/9 | yes | yes | 2 | 21,600 | 0 |
| M4 | 81,000 | 6,074 | 4: 19, 5: 23, 6: 543, 7: 5489 | NXT | 9/9 | yes | yes | 8 | 0 | 0 |
| M5 | 81,000 | 204 | 4: 33, 5: 171 | NXT | 9/9 | yes | yes | 2, 4, 8 | 27,000 | 0 |
| M6 | 3,510 | 1 (fixed) | 4: 1 | NXT | 3/9 | no | yes | 1, 2, 4 | 2,340 | 0 |
| M7 | 4,050 | 1 (fixed) | 4: 1 | NXT | 3/9 | no | yes | 1, 2, 4 | 2,700 | 0 |
| M8 | 2,970 | 1 (fixed) | 4: 1 | NXT | 1/9 | no | yes | 1, 2, 4 | 1,980 | 0 |
| M9 | 4,185 | 1 (fixed) | 4: 1 | NXT | 3/9 | no | yes | 1, 2, 4 | 2,790 | 0 |
| M10 | 3,780 | 1 (fixed) | 4: 1 | NXT | 3/9 | no | yes | 1, 2, 4 | 2,520 | 0 |
| h1 | 101,999 | 430 | 4: 11, 5: 42, 6: 375, 7: 2 | NXT | 9/9 | yes | yes | 1, 2, 4, 8, 16 | 2,000 | 33,333 |
| xfirst | 101,999 | 430 | 4: 11, 5: 42, 6: 375, 7: 2 | XNT | 9/9 | yes | yes | 1, 2, 4, 8, 16 | 2,000 | 33,333 |
| mont | 101,999 | 430 | 4: 11, 5: 42, 6: 375, 7: 2 | NXT | 9/9 | yes | yes | 1, 2, 4, 8, 16 | 2,000 | 33,333 |
| ten | 101,999 | 430 | 4: 11, 5: 42, 6: 375, 7: 2 | NXT | 9/9 | yes | yes | 1, 2, 4, 8, 16 | 2,000 | 33,333 |
| rho | 101,999 | 430 | 4: 11, 5: 42, 6: 375, 7: 2 | NXT | 9/9 | yes | yes | 1, 2, 4, 8, 16 | 2,000 | 33,333 |
| cube | 101,999 | 430 | 4: 11, 5: 42, 6: 375, 7: 2 | NXT | 9/9 | yes | yes | 1, 2, 4, 8, 16 | 2,000 | 33,333 |
| xy | 101,999 | 430 | 4: 11, 5: 42, 6: 375, 7: 2 | NXYT | 9/9 | yes | yes | 1, 2, 4, 8, 16 | 2,000 | 33,333 |
<!-- /AUDIT -->

- **The digit code.** The reader learns which token is which digit from x < N alone: at the highest place where x's
  and N's digits differ, x's digit is the smaller (IO.md §5). That orders two digits only when they meet there, so
  the code is pinned only if every adjacent pair (k, k + 1) meets in some row. With 430 moduli (H1) or any sampled
  modulus (E3–E5, M3–M5), all 9 do. With one fixed modulus, only pairs involving N's own digits ever meet: 1 to 4 of
  the 9. N = 323 orders 0 and 1 below 2, and 2 below 3, and says nothing about 4–9.
- **N's direction.** With one fixed modulus, reading N backwards (998 for 899) passes x < N on every row whenever the
  reversed number is the larger one, and then the reader cannot choose: in 12 of the 14 (E1's 323 reads the same both
  ways). For E10 (403) and M2 (38021) no digit code makes the backwards reading pass every row.
- **T's direction.** Only M1 and M2, among the tiers, ever train on a two-digit T. On the others nothing can tell
  T = 16 from T = 61 (IO.md §4); the reader keeps whichever direction its random start gave it.
- **The answers.** M1, M2 and M4 have no row with T ≤ 2, so the reading's switches are taught by the laws alone
  (which works on H1: MENUS.md §2.3).

### 3. Results: the practice tiers

<!-- TIERS_NOTES -->
**The 6 tiers with sampled moduli learned the machine unchanged.** Given H1's budget (relaxed: 3,600 s, at most 1,500
steps), E3, E4, E5, M3 and M4 scored 100% and certified T = 64 on both depth ladders; M5 scored 50.5% with max T 8
(below). Every one ended with exact tables (200/200 subtraction rows, 100/100 product cells), the identity digit code,
the reading this data needs (N read from the right end of its run, the opposite of H1), long division by repeated
subtraction and the schoolbook rows summed in reverse, the same choices as H1's official run. None froze: their T
always has one digit, so T's direction never settles (§2), and they ran to the step cap, exact from roughly step
600–800 on.

**M5: T read backwards.** Its machine is exact, but its T pointer reads T from the run's left end, so T = 16, 32, 64
count down 61, 23, 46 rounds. Every one-digit rung is 768/768, the rungs of T ≥ 16 hold only where both powers agree
(11, 11 and 101 of 768), and its scored OOD split (T = 16) is 1% right: 50.5% overall. E3–E5, M3 and M4 happened to
start their T pointer the right way. Nothing in their data can decide it; §4 tries one tie that does.

**The 14 tiers with one modulus learned nothing.** Their best score is 2.4%. Two stages fail, both for the reasons in
§2:
- 7 (E2, E6–E10, M2) never chose a reading: several readings each find a digit code under which x < N holds on every
  row, so the reader's posterior on its favourite stays at 0.58–0.97, below the 0.99 every arithmetic lesson waits for,
  for all 1,500 steps. That includes E10 and M2, where x < N rules out reading N backwards under the true code;
- 7 (E1, M1, M6–M10) chose one. E1's N reads the same both ways (323); the six Medium ones read N from its left end, the
  wrong way, and M1 and M8 x too, each with a scrambled digit code that still satisfies x < N (0–4 of the 10 tokens
  right). Then the subtraction table never became exact: 26–47 of its 200 entries right, 15–71% of its checks passing.
  The borrow facts that give the table its meaning reach only the rows and columns of N's own digits.

**Under the official budgets.** Easy's 60 s is about 80–160 steps: every Easy run was still in the reader's first
stages at the deadline. E1 and E3–E5 then ran out of their 30 s evaluation budget (the evaluator counts a failed run);
the others scored 0–2.4%. Medium's 600 s is 800–1,800 steps: M3 and M4 certified T = 64 on both ladders at 100% (818
and 880 steps), M5 scored 50.5% as relaxed, and the fixed-modulus Medium runs failed as relaxed.
<!-- /TIERS_NOTES -->

<!-- TIERS -->
| task | modulus | T in training | official manifest: score | relaxed: score | relaxed: splits | relaxed: max T seen / unseen | relaxed: where it stopped | relaxed: chose |
|---|---|---|---|---|---|---|---|---|
| E1 | fixed | 1, 2, 3 | failed: TimeoutError (77 steps) | 1.7% | ood 0.0%, test 3.3% | 0 / 0 | subtraction table never exact (15.1% of its checks pass) | - |
| E2 | fixed | 1, 2, 4 | 0.1% (136 steps) | 0.1% | ood 0.0%, test 0.2% | 0 / 0 | reading never chosen (posterior 0.953 < 0.99) | - |
| E3 | 27 moduli | 2 | failed: TimeoutError (152 steps) | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed |
| E4 | 35 moduli | 2 | failed: TimeoutError (158 steps) | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed |
| E5 | 27 moduli | 1, 2, 3 | failed: TimeoutError (110 steps) | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed |
| E6 | fixed | 1, 2, 3, 4 | 0.0% (116 steps) | 0.7% | ood 0.0%, test 1.4% | 0 / 0 | reading never chosen (posterior 0.965 < 0.99) | - |
| E7 | fixed | 1, 2, 3 | 0.4% (105 steps) | 0.4% | ood 0.0%, test 0.8% | 0 / 0 | reading never chosen (posterior 0.958 < 0.99) | - |
| E8 | fixed | 1, 2, 3 | 1.7% (82 steps) | 0.4% | ood 0.0%, test 0.8% | 0 / 0 | reading never chosen (posterior 0.958 < 0.99) | - |
| E9 | fixed | 1, 2, 3 | 2.4% (121 steps) | 2.0% | ood 3.3%, test 0.7% | 0 / 0 | reading never chosen (posterior 0.856 < 0.99) | - |
| E10 | fixed | 1, 2, 3 | 1.7% (105 steps) | 1.7% | ood 2.4%, test 1.0% | 0 / 0 | reading never chosen (posterior 0.898 < 0.99) | - |
| M1 | fixed | 4, 8, 16 | 0.1% (843 steps) | 0.0% | ood 0.0%, test 0.1% | 0 / 0 | subtraction table never exact (28.4% of its checks pass) | - |
| M2 | fixed | 4, 8, 16 | 0.2% (1818 steps) | 0.2% | ood 0.1%, test 0.2% | 0 / 0 | reading never chosen (posterior 0.578 < 0.99) | - |
| M3 | 247 moduli | 2 | 100.0% (818 steps) | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed |
| M4 | 6074 moduli | 8 | 100.0% (880 steps) | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed |
| M5 | 204 moduli | 2, 4, 8 | 50.5% (876 steps) | 50.5% | ood 1.0%, test 100.0% | 8 / 8 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed |
| M6 | fixed | 1, 2, 4 | 2.1% (1196 steps) | 2.1% | ood 2.4%, test 1.8% | 0 / 0 | subtraction table never exact (71.0% of its checks pass) | - |
| M7 | fixed | 1, 2, 4 | 0.0% (1175 steps) | 0.0% | ood 0.0%, test 0.0% | 0 / 0 | subtraction table never exact (62.1% of its checks pass) | - |
| M8 | fixed | 1, 2, 4 | 0.3% (785 steps) | 0.3% | ood 0.2%, test 0.3% | 0 / 0 | subtraction table never exact (39.2% of its checks pass) | - |
| M9 | fixed | 1, 2, 4 | 0.2% (824 steps) | 0.2% | ood 0.5%, test 0.0% | 0 / 0 | subtraction table never exact (70.0% of its checks pass) | - |
| M10 | fixed | 1, 2, 4 | 0.2% (1093 steps) | 0.5% | ood 0.5%, test 0.5% | 0 / 0 | subtraction table never exact (31.0% of its checks pass) | - |
<!-- /TIERS -->

### 4. Results: the other tasks, and the smallest changes

**Unchanged, every task of (b) and (c) but `xfirst` learned H1.** `mont`, `ten`, `rho` and `cube` have H1's prompts,
and every lesson but the answers' looks only at the prompts, so their runs repeat the official H1 run step for step
until the answers first matter: the reading chosen at step 97, the digit lessons at 134, the ladder at 199, the
chain lesson at 352, the reading's division switches at 385 (`wide_anchored_official`: the same steps). There the
answers of the T ≤ 2 rows disagree with every combination of the reading's switches, the answers' mixture gets no
gradient, and the lessons keep the machine they describe: long division of the square. Each run ended exact on all of
its own checks, with 0% of the T ≤ 2 answers right. Its scores equal, split by split and rung by rung, what an exact
x^(2^T) mod N machine scores on that task (`agreement.json`: the few rows it gets right are coincidences, counted the
evaluator's way, which compares only the answer's own digit positions). Without a freeze (it needs the answers right)
they ran to the 1,500-step cap.

**`xy`** never got past the reader's first stage: every prompt has four digit runs, and the reader takes rows with
exactly three. Nothing else ever gets a lesson.

**The smallest changes**, each a file in [`variants/`](variants/), checked locally first (every table and switch at
the truth: both prediction paths exact on the task with identical logits, and still exact on H1 with the new option
off; the gradient test passes with every parameter tensor, the new switch included, getting a gradient):
- `rho`: one switch, `plus1` (none | one): the round adds the number 1 to its square before the reduction, with the
  learned adder. It joins the reading's division switches, so the laws (which test the division alone and pass either
  way) and the answers teach it. 17 lines of code added or rewritten;
- `ten`: one switch, `times10`: the square moved up one place before the reduction. The menu already held this
  product as a column offset, but the multiply's ladder pins the true columns, so the switch goes where only the answers
  reach it. 13 lines;
- `cube`: one round option, `cube` (none | times_x): after the reduction, multiply by the round's own input and reduce
  again. The multiply gains a second operand: the pair (a_i, b_j) instead of (s_i, s_j), with the same product cell,
  column rules and number rule. 21 lines;
- `mont`: no new option; its three division lessons restated for Montgomery's reduction (REDC with R = 10^8),
  `variants/submission_mont.py`:
  - the division ladder: from 10x + qN (q = 0..9) one chain gives 10x if q = 0, else 10(x + N); long division's says
    that from x + qN it gives x;
  - the step law: the same register, part e of its lowest digit left in place and the rest brought in (no move: the
    digit is added there), then the chain and the post-move give x or x + N; long division's moves the upper places
    up and writes the digit in;
  - the whole-division laws: REDC(x·10^8) = x, REDC(N·10^8) = 0, REDC(x·10^8 + Q·N) = x; long division's are
    x mod N = x, N mod N = 0, x + QN mod N = x.

  On exact tables, the long-division wiring passes the file's three lessons and fails all three restated ones, and
  the Montgomery wiring the reverse; only the Montgomery wiring reproduces the task's answers. 22 lines.

**Montgomery needed more than restating, and the runs found what.**
- The first restatement (`mont_v1`) left a hole: its step law emptied the register's lowest place before bringing
  the digit in, so writing the digit there passed as well as adding it there. The search kept writing, which loses a
  live digit in a real Montgomery step, and no reading could then pass the whole-division laws (`mont_fix_s74`, 0.1%,
  on CPU).
- With that closed (part of the lowest digit stays in place, so writing over it fails), the chain lesson found a
  correct Montgomery chain on every seed, in one of two forms, as long division has two: repeated addition (add N
  while the lowest digit is not 0) or trial multiples (keep the multiple Rp + sN whose sum ends in 0). Seeds 2 and 3
  chose repeated addition and certified T = 64 on both ladders. Seeds 74 and 1 chose trial multiples and deadlocked
  again (0.1%). That form commits when the sum's lowest digit is 0, the commit gate's flag 1; the same gate makes the
  final correction, which must commit on "no borrow", flag 0. With the gate the other way round every final correction
  in the menu goes wrong, so no reading passes the laws. Long division never meets this: both its forms use the same
  gate.
- A fourth change closes it (`variants/submission_mont_gate.py`, 12 more lines): the division ladder also asks the
  gate to do the final correction's job on the row's own numbers (from x + N it subtracts N, from x it holds). With it
  all four seeds certified T = 64 on both ladders, three with repeated addition and one (seed 1) with a third form,
  trial multiples that test the register's own lowest digit, which uses the usual gate. They froze at steps 582–625,
  like H1.

So Montgomery's squaring, which the unchanged menu can compute, needed its three division lessons rewritten and a
fourth added, and two of the four changes were found only by running it.

**T's direction (M5): one tie, the second one tried.** Over four seeds the unchanged file read M5's T backwards on two
(74 and 3: 50.5%, max T 8) and the right way on two (1 and 2: 100%, max T 64). Any fix is a designer's prior, since
the data cannot tell:
- `variants/submission_tie.py` makes T's pointer use x's step parameter (5 lines): T is written the same way as x. It
  certified on seeds 74, 1 and 2, but not on seed 3 (5.7%): there T's pointer kept its anchor on the T marker and,
  stepping the right way from it, read x's last two digits as T. Its training checks still all passed: read that way,
  no row's T was a single 1 or 2, so no answer was ever compared;
- `variants/submission_tie_writer.py` makes T's pointer use the writer's anchor and step (7 lines): T is read the way
  the answer is written, leftwards from the prompt's end (the writer's pointer is pinned on every row by the answers'
  positions, and the competition writes each answer over the prompt's tail, where T is). It certified T = 64 on both
  ladders on all four seeds, and still certifies H1 itself (T = 64 on both ladders, frozen at step 590).

**One fixed modulus: what three diagnostics show.** Three variants that are not submissions (they set weights by
hand, breaking rule 6) ask whether the rest of the machine would learn from one modulus if the reader were not the
problem:
- `reader_given` starts the reader and writer at the truth for this layout (the digit code, the pointers, T's
  direction) with the reader's stages done, and lets everything learn as in the file. On E2 and M1 the reader's own
  ranking lesson then pushed the given code wrong: x < N keeps pushing N's few digits away from x's, past digits it
  never compares, and at the end E2 read the token for 0 as 6 and M1 the token for 4 as 9. On one modulus the
  reader's lesson does not just fail to find the code; it moves a right code away;
- `reader_frozen` also switches the reader's label-free lessons off, so the given code stays. The table still never
  became exact on any of the five tasks tried (E1, E2, E10, M1, M6: 0.7–4.7%; 27–54 of the table's 200 entries right,
  wrong in every column);
- `warm_frozen` is `submission_wide.py` (step 10 of the ladder, with the 70-step hand-written-adder warm start that the
  anchors later replaced) with the same frozen reader. Still no exact table on E1, E2, M1 or M6 (0.6–2.3%; 55–82 of 200
  entries right).

Every lesson that teaches the table pairs a digit with one of N's own: a register minus N (or minus 0 − N), N minus
a register, minus 0 or minus itself; the hand-written adder checks the same subtractions. With N = 323 no lesson ever
pairs 8 with 4, for example: 36 of the table's 200 entries (18 digit pairs, each with and without a borrow in) appear
in no lesson at all, and in E1's warm-start run none of the 36 came out right. The other 164 are reached, but only
through N's three digits, and the learned lessons held just 58 of them once the hand-written adder stopped. One
modulus starves the reader and the table alike. Teaching them would take lessons built on other numbers than N, or
the answers, which this design keeps away from both.

<!-- RUNS -->
| run | task | file | score | splits | max T seen / unseen | steps | training s | evaluation s | where it stopped | chose | added switch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cube_fix_on_h1 | h1 | cube | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 2131 | 640 | froze at step 605 | repeated subtraction, rows by multiplier digit | cube=none |
| cube_fix_s74 | cube | cube | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1527 | 879 | froze at step 605 | repeated subtraction, rows by multiplier digit | cube=times_x |
| cube_s74 | cube | the file | 0.1% | ood_n_t 0.0%, ood_t 0.1%, test 0.0% | 0 / 0 | 1500 | 1637 | 48 | exact on its own checks; 0.0% of the T <= 2 answers right | repeated subtraction, rows by multiplier digit, reversed | - |
| e10_reader_frozen | e10 | reader_frozen | 0.7% | ood 0.8%, test 0.5% | 0 / 0 | 1500 | 2133 | 7 | subtraction table never exact (64.2% of its checks pass) | - | - |
| e1_reader_frozen | e1 | reader_frozen | 4.7% | ood 6.0%, test 3.3% | 0 / 0 | 1500 | 2082 | 4 | subtraction table never exact (54.9% of its checks pass) | - | - |
| e1_warm_frozen | e1 | warm_frozen | 1.7% | ood 0.0%, test 3.3% | 0 / 0 | 1500 | 1184 | 104 | subtraction table never exact (84.3% of its checks pass) | - | - |
| e2_reader_frozen | e2 | reader_frozen | 1.0% | ood 0.7%, test 1.3% | 0 / 0 | 1500 | 2272 | 9 | subtraction table never exact (13.1% of its checks pass) | - | - |
| e2_reader_given | e2 | reader_given | 0.1% | ood 0.0%, test 0.2% | 0 / 0 | 1500 | 1571 | 8 | subtraction table never exact (29.8% of its checks pass) | - | - |
| e2_warm_frozen | e2 | warm_frozen | 0.6% | ood 0.0%, test 1.3% | 0 / 0 | 1500 | 1281 | 4 | subtraction table never exact (84.5% of its checks pass) | - | - |
| m1_reader_frozen | m1 | reader_frozen | 2.3% | ood 2.6%, test 1.9% | 0 / 0 | 1500 | 1310 | 5 | subtraction table never exact (71.9% of its checks pass) | - | - |
| m1_reader_given | m1 | reader_given | 0.2% | ood 0.3%, test 0.1% | 0 / 0 | 1500 | 1469 | 11 | subtraction table never exact (57.4% of its checks pass) | - | - |
| m1_warm_frozen | m1 | warm_frozen | 2.3% | ood 2.6%, test 1.9% | 0 / 0 | 1500 | 878 | 4 | subtraction table never exact (63.4% of its checks pass) | - | - |
| m5_relaxed_s1 | m5 | the file | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1991 | 71 | exact on every check and answer; T's direction never settled, so no freeze | trial multiples, rows by multiplier digit | - |
| m5_relaxed_s2 | m5 | the file | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 2005 | 71 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit | - |
| m5_relaxed_s3 | m5 | the file | 50.5% | ood 1.0%, test 100.0% | 8 / 8 | 1500 | 1923 | 94 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit | - |
| m5_tie_s1 | m5 | tie | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1953 | 71 | exact on every check and answer; T's direction never settled, so no freeze | trial multiples, rows by multiplier digit | - |
| m5_tie_s2 | m5 | tie | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1528 | 42 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit | - |
| m5_tie_s3 | m5 | tie | 5.7% | ood 6.2%, test 5.3% | 0 / 0 | 1500 | 1231 | 227 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit | - |
| m5_tie_s74 | m5 | tie | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 2061 | 80 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed | - |
| m5_tiew_s1 | m5 | tie_writer | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1434 | 38 | exact on every check and answer; T's direction never settled, so no freeze | trial multiples, rows by multiplier digit | - |
| m5_tiew_s2 | m5 | tie_writer | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1573 | 42 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit | - |
| m5_tiew_s3 | m5 | tie_writer | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1460 | 46 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit | - |
| m5_tiew_s74 | m5 | tie_writer | 100.0% | ood 100.0%, test 100.0% | 64 / 64 | 1500 | 1256 | 36 | exact on every check and answer; T's direction never settled, so no freeze | repeated subtraction, rows by multiplier digit, reversed | - |
| m6_reader_frozen | m6 | reader_frozen | 1.1% | ood 1.0%, test 1.3% | 0 / 0 | 1500 | 1469 | 4 | subtraction table never exact (62.3% of its checks pass) | - | - |
| m6_warm_frozen | m6 | warm_frozen | 1.0% | ood 1.0%, test 1.0% | 0 / 0 | 1500 | 1312 | 6 | subtraction table never exact (63.2% of its checks pass) | - | - |
| mont_fix2_s1 | mont | mont | 0.1% | ood_n_t 0.1%, ood_t 0.2%, test 0.1% | 0 / 0 | 1500 | 1539 | 4 | wiring lessons on, checks 0.0% exact | Montgomery, trial multiples (the sum's digit tested), rows by multiplier digit | - |
| mont_fix2_s2 | mont | mont | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1114 | 91 | froze at step 585 | Montgomery, repeated addition, rows by multiplier digit | - |
| mont_fix2_s3 | mont | mont | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 564 | 76 | froze at step 621 | Montgomery, repeated addition, rows by multiplier digit | - |
| mont_fix2_s74 | mont | mont | 0.1% | ood_n_t 0.1%, ood_t 0.2%, test 0.1% | 0 / 0 | 1500 | 1564 | 4 | wiring lessons on, checks 0.0% exact | Montgomery, trial multiples (the sum's digit tested), rows by multiplier digit, reversed | - |
| mont_fix_s74 | mont | mont_v1 | 0.1% | ood_n_t 0.1%, ood_t 0.2%, test 0.1% | 0 / 0 | 871 | 3605 | 10 | wiring lessons on, checks 0.0% exact | Montgomery, trial multiples (the sum's digit tested), rows by multiplier digit, reversed | - |
| mont_gate_s1 | mont | mont_gate | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 645 | 84 | froze at step 625 | Montgomery, trial multiples (the register's digit tested), rows by multiplier digit | - |
| mont_gate_s2 | mont | mont_gate | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1033 | 78 | froze at step 582 | Montgomery, repeated addition, rows by multiplier digit | - |
| mont_gate_s3 | mont | mont_gate | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 684 | 99 | froze at step 622 | Montgomery, repeated addition, rows by multiplier digit | - |
| mont_gate_s74 | mont | mont_gate | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 680 | 96 | froze at step 622 | Montgomery, repeated addition, rows by multiplier digit, reversed | - |
| mont_s74 | mont | the file | 0.0% | ood_n_t 0.0%, ood_t 0.0%, test 0.0% | 0 / 0 | 1500 | 1450 | 43 | exact on its own checks; 0.0% of the T <= 2 answers right | repeated subtraction, rows by multiplier digit, reversed | - |
| rho_fix_on_h1 | h1 | rho | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1556 | 514 | froze at step 605 | repeated subtraction, rows by multiplier digit | plus1=none |
| rho_fix_s74 | rho | rho | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 2060 | 586 | froze at step 605 | repeated subtraction, rows by multiplier digit | plus1=one |
| rho_s74 | rho | the file | 0.0% | ood_n_t 0.0%, ood_t 0.0%, test 0.0% | 0 / 0 | 1500 | 2338 | 74 | exact on its own checks; 0.0% of the T <= 2 answers right | repeated subtraction, rows by multiplier digit, reversed | - |
| ten_fix_on_h1 | h1 | ten | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1831 | 582 | froze at step 605 | repeated subtraction, rows by multiplier digit | times10=none |
| ten_fix_s74 | ten | ten | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1360 | 409 | froze at step 605 | repeated subtraction, rows by multiplier digit | times10=ten |
| ten_s74 | ten | the file | 0.0% | ood_n_t 0.0%, ood_t 0.0%, test 0.0% | 0 / 0 | 1500 | 2437 | 100 | exact on its own checks; 0.0% of the T <= 2 answers right | repeated subtraction, rows by multiplier digit, reversed | - |
| tiew_on_h1 | h1 | tie_writer | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 537 | 53 | froze at step 590 | repeated subtraction, rows by multiplier digit, reversed | - |
| xfirst_s1 | xfirst | the file | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1216 | 88 | froze at step 631 | trial multiples, rows by multiplier digit | - |
| xfirst_s2 | xfirst | the file | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 737 | 85 | froze at step 671 | repeated subtraction, rows by multiplier digit | - |
| xfirst_s3 | xfirst | the file | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 1170 | 87 | froze at step 698 | repeated subtraction, rows by multiplier digit | - |
| xfirst_s74 | xfirst | the file | 100.0% | ood_n_t 100.0%, ood_t 100.0%, test 100.0% | 64 / 64 | 1500 | 456 | 45 | froze at step 631 | repeated subtraction, rows by multiplier digit, reversed | - |
| xy_s74 | xy | the file | 0.0% | ood_n_t 0.0%, ood_t 0.0%, test 0.0% | 0 / 0 | 1500 | 776 | 666 | the reader never found exactly three numbers in a prompt (its first stage) | - | - |
<!-- /RUNS -->

### 5. Which lessons still make sense

Every lesson of the file, against each task: **true and pins it down** (as on H1), **true but does not pin it down**
on this data (a fact that holds, but leaves more than one answer), **true but silent** (the task has a step no lesson
describes), **false** (the task breaks the fact). Nothing was switched off: inapplicable lessons ran as they are.

| lesson (what it teaches) | tiers, sampled moduli | tiers, one fixed modulus | `xfirst` | `mont` | `ten` | `rho` | `cube` | `xy` |
|---|---|---|---|---|---|---|---|---|
| the reader: three digit runs, x < N ranks the digit tokens and picks the reading, round trips (the reader, the writer) | pins it | true, does not pin the code (1–4 of 9 digit orders) or, in 12 of 14, N's direction | pins it | pins it | pins it | pins it | pins it | false: four numbers, no row is read |
| T's direction by the answers of rows with a two-digit T | no such rows: T's direction is left to chance | no such rows, except M1 and M2 | pins it | needs the machine to reproduce the answers, which it never does | the same | the same | the same | - |
| the subtraction table: x < N on every top part (borrows), the order laws and bijectivity (digits), the pieces' own checks | pins it | true, does not pin it: every lesson pairs a digit with one of N's own, so some entries are in no lesson and the rest were not held | pins it | pins it | pins it | pins it | pins it | - |
| the product cell and the layout: the cell's ladder, the routed ladder, distributivity | pins it | true | pins it | pins it | true but silent: they pin the true product, and the task multiplies by 10 too | pins it | true but silent about the second product | - |
| the division: the division ladder (from x + qN one step gives x), the step law, x mod N = x, N mod N = 0, x + QN mod N = x | pins it | true | pins it | **false**: they state long division and the remainder; Montgomery's reduction fails all three | true | true | true | - |
| the answers of the rows with T = 1, 2 (through the reading's switches only) | pin it (M4: none) | never reached: the reader or the table stalls first (M1, M2: none) | pin it | disagree with every combination the reading's switches can make | disagree | disagree | disagree | - |

So the lessons are true statements on every task but `mont` (and `xy`'s reader). What breaks elsewhere is that they
either cannot pin down what they teach on this data (one modulus, one-digit T), or describe only x² mod N, so that
a task with one more step (×10, +1, ×x) has nothing that teaches that step except answers that reach only the reading's
switches.

### 6. Budgets

The file was built for H1's 3,600 s (it froze at step 588 after 558 s on H1). On the tiers' data a step costs about
what it costs on H1 (0.4–0.5 s on an H100 while the reader and the table learn, about 1 s once the wiring lessons run;
the evaluator's two data workers restart every epoch, and the smallest Easy sets are one batch an epoch), so:
- **Easy (60 s)** bought 77–158 steps. The reader had not chosen its reading by then, and nothing else had a
  lesson yet. E1's official run made 77 steps and then ran out of its 30 s evaluation budget on its first batch,
  which the evaluator counts as a failed run. The likely reason: with an untrained table and commit gate (commit
  probabilities 0.45 and 0.50) the count-down need not reach its stop, so the rounds run to their cap of 100.
- **Medium (600 s)** bought 785–1,818 steps, about what H1 needs: M3 and M4 certified within it.
- The relaxed runs give every tier H1's 3,600 s and stop at 1,500 steps, so they measure what the machinery learns
  from the data, not the clock.

### 7. Reproduce

Everything heavy ran on Modal through [`modal_litmus.py`](modal_litmus.py) (app and Volume `litmus`, at most 10
H100 and 10 CPU containers per launch; a copy of `wider_menus/modal_wide.py` whose image also generates every dataset
of `scripts/generate_datasets.sh` and the tasks of `tasks.py`). [`wait_fetch_litmus.py`](wait_fetch_litmus.py)
downloads the records to `litmus/records/<name>.json` (each holds the full log, the result, the manifest and the exact
source); [`summarize_litmus.py`](summarize_litmus.py) `fill LITMUS.md` regenerates the tables here. The plans, in the
order they ran (each record's name is in its plan):
- `smoke.json` (E1 official) and `sweep1.json`: the 6 new tasks, the 20 tiers relaxed and the other 19 official;
- `sweep2_fixes_cpu.json`: `rho`, `ten`, `cube`, `mont_v1` on their tasks, and `rho`, `ten`, `cube` on H1 (CPU; its
  H100 twin `sweep2_fixes.json` never got an H100 and was stopped);
- `sweep3_reader_given_cpu.json` (stopped after E2 and M1, once the given code was seen to move), `sweep5_reader_frozen.json`,
  `sweep8_warm_frozen.json`: the diagnostics;
- `sweep4a_tie_xfirst.json`, `sweep4b_m5_seeds.json`, `sweep10_m5_tie_writer.json`, `sweep11_tie_writer_h1.json`: T's
  direction and `xfirst`'s seeds;
- `sweep6_mont_seeds.json`, `sweep7_mont_gate.json`, `sweep9_mont_gate_seeds.json`: Montgomery.

`new_tasks.json`, `tiers_relaxed.json` and `tiers_official.json` are the parts `sweep1.json` was merged from, and
`sweep3_xfirst_seeds.json` part of `sweep4a_tie_xfirst.json`.

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run ../cpu_first/litmus/modal_litmus.py --mode audit
```

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run ../cpu_first/litmus/modal_litmus.py --mode agreement
```

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/litmus/modal_litmus.py --mode plan --plan ../cpu_first/litmus/plans/sweep1.json
```

```bash
cd ~/github/one-layer/cpu_first/litmus && ~/github/sotaku/venv/bin/python summarize_litmus.py fill LITMUS.md
```

Local checks, each under a minute and well under 1 GB. The competition's data package needs Python 3.10's
`typing.TypeAlias`, which the local Python 3.9 lacks, so the small datasets are made with a one-line shim:

```bash
cd ~/github/one-layer/one-layer-deeper && PYTHONPATH=$PWD:../cpu_first/litmus ~/github/sotaku/venv/bin/python -c "import typing; typing.TypeAlias = typing.Any; import sys, runpy; sys.argv = ['tasks', '/tmp/litmus_small', '--small']; runpy.run_module('tasks', run_name='__main__')"
```

```bash
cd ~/github/one-layer/cpu_first/litmus && ~/github/sotaku/venv/bin/python local_check.py litmus/variants/submission_mont.py /tmp/litmus_small/litmus_mont lessons N=376 op=add flag=zD start=zero order=up "bring=(0,0,'add')" post=-1 final=addhigh
```

`local_check.py FILE DATA` runs both prediction paths with every table and switch at the truth (`switch=option`
moves a division switch), `... lessons` the division lessons' hard checks, `... grad` the gradient test.

### 8. Files

In `cpu_first/litmus/`:
- `LITMUS.md`: this write-up;
- `tasks.py`: the generators of §1 (b) and (c); `audit_data.py` and `audit.json`: §2; `agreement.json`: per task and
  split, how many answers an exact x^(2^T) mod N machine gets right, scored the evaluator's way (§4);
- `modal_litmus.py`, `wait_fetch_litmus.py`, `summarize_litmus.py`, `local_check.py`;
- `plans/`: every launch; `records/`: every run; `logs/`: the waiters' logs, and the md5s of the certified files at
  the start;
- `variants/`: the changed files of §4, each `submission_wide_anchored.py` plus its change, with a docstring that says
  what changed: `rho`, `ten`, `cube`, `tie`, `tie_writer`, `mont` (with `mont_v1`, the first version, which has the
  hole in its step law, and `mont_gate`, which adds the fourth change); and the diagnostics `reader_given`,
  `reader_frozen` and `warm_frozen` (`submission_wide.py` plus the frozen reader), which are not submissions: they set
  the reader by hand.

No certified file was edited.
