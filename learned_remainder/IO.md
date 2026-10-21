# H1 with learned prompt reading and answer writing

## In plain words

**The task.** The model reads three numbers written as digits: a modulus N, a starting number x and a count T. It must
square x, keep only the remainder after dividing by N, and repeat that T times. To be certified it must get every one
of 768 fresh questions exactly right at T = 1, 2, 4, …, 64, both for moduli it saw in training and for larger moduli
it never saw (up to 8 digits).

**What changed.** The calculator of [CHAIN.md](CHAIN.md) was handed its inputs and had its answer printed for it:
- a hand-written routine found the three numbers in the question. It knew which symbol marks which number, that N's
  digits are written lowest first while x's and T's are written highest first, and that token number 7 means the
  digit 0, token 8 the digit 1, and so on;
- another hand-written routine wrote the answer's digits into the last places of the line.

Both routines are gone. The calculator now learns:
- where each number starts, and in which direction to read its digits;
- which symbols are digits at all, and which digit each symbol stands for;
- where to write the answer, in which order, and with which symbols.

**What is still hand-written:**
- the shape of a reader: start next to some symbol (or at the end of the line), step one place at a time in one
  direction, and stop at the first symbol that is not a digit;
- the calculator's needs: three numbers in, with room for 8, 8 and 2 digits, and one number out, 8 digits;
- that the ten digit symbols stand for the values 0 to 9 in some order (which order is learned);
- the lessons below;
- everything CHAIN.md lists for the calculator itself: the squaring (which also turns x's digits into a number), the
  long-division layout, the chain of subtractions, the stop rule, the loop over T (which turns T's digits into a
  count), and the adder in its training self-check.

**How it learns to read and write.** No lesson says "this symbol means 3". Instead:
1. **Copy the answer back.** The calculator reads the expected answer where its writer would write it and writes it
   back. That only comes out right if the writer writes at the end of the line, lowest digit last, and if all ten
   digit symbols count as digits.
2. **Round trips.** A symbol read and written back must come back the same. A number the calculator computes, written
   and read back, must come back the same. Then the writer is exactly the reader run backwards. But any consistent
   relabeling of the digits would pass too: 3,628,800 of them.
3. **"x is smaller than N."** Each question's x is already reduced by N, so x < N. At the highest place where x's and
   N's digits differ, x's digit must be the smaller one. In training every one of the 45 pairs of digits shows up
   that way. So only the true digit order passes on every question, and only if N's digits are taken as written
   lowest first and x's as written highest first, which is how the question writes them.
4. **Try every way of reading at once.** There are 24 ways to assign the digit runs to N and x and to pick
   directions. Each way gets its own digit order. The way whose order makes x < N hold on every question wins.
5. **T's direction.** Nothing above can tell T = 16 from T = 61. Once the calculator is exact, it works out the
   T = 16 questions both ways, and the expected answers keep the right one.

**Did it certify?** Yes, on the untouched official H1 manifest on an H100 (`io_official`):
- all 768 questions right at every T up to 64, on seen and on unseen moduli, plus every row of the test sets;
- the learned reader is exactly the true one: it reads every N, x and T of the certification questions exactly,
  including the 110 eight-digit moduli per unseen-moduli rung, and its writer reproduces every answer;
- training: 220 s; everything is done at step 440 and then holds still. The budget is 3,600 s;
- evaluation: 12.8 s (CHAIN.md's calculator: 15.4 s). The budget is 1,800 s.

The same file certified on 8 of 8 seeds on CPU (§1), and every one of them learned exactly the same reader and writer.

**What broke along the way.** In the order the fixes were found (§7):
- Lessons scored one position at a time were happy with readers smeared over the whole line that never committed to
  anything. Every lesson now scores a whole reading per question.
- Copying the answer back gave credit for empty slots, so any reader that merely covered the answer's places looked
  as good as the right one.
- Learning each symbol's digit by trial and error stalled with two symbols on one digit and another digit unused. The
  digit meanings are now an order (a rank), and the order comes from "x < N".
- "x < N" has a trap: read both N and x backwards and rank the odd digits high, and it holds on 85–90% of questions,
  because N always ends in 1, 3, 7 or 9. Giving each way of reading its own ranking, and letting the best one win,
  avoids it.
- The choice of reading direction got settled by early noise before the evidence came in. Making the choice compare
  the readings with each other first fixed that.
- My first experiments used only the first 30,000 training rows. The file is sorted by modulus size, so those rows had
  no 20-bit moduli and no N starting with a 9; part of the early trouble with the digit order came from that.

## 1. Results

Official manifest `benchmark/manifests/h100_hard_h1.json`, untouched:
- cuda:0, bf16 autocast, the batch from the submission (512);
- 3,600 s training budget, 1,800 s evaluation budget;
- Modal H100 80GB, the image and H1 data regenerated with the official generator arguments (`modal_chain.py
  --file submission_io.py`; the launcher is unchanged);
- seed 74. Every rung is 768 prompts.

| run | what ran | max T (seen moduli) | OOD-N max T | rungs T=1…64, seen / OOD-N | test, ood_t, ood_n_t | subtractor rows right | reader read right (offline) | training s (frozen at step) | evaluation s |
|---|---|---|---|---|---|---|---|---|---|
| `io_s0_official` | exact table, I/O parameters set at the true reading (`MODE = IO_MODE = "designed"`, 100 steps): the plumbing check | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 (designed) | all (set) | 3.6 (–) | 20.6 |
| **`io_official`** | **`submission_io.py` byte for byte**, md5 `45cc6e1b66865797d9b1a38505175a0a` | **64** | **64** | 768 at every rung / 768 at every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 200/200 | N, x, T and the written answer exact on every depth prompt, both profiles | 220.0 (440) | 12.8 |
| for comparison: `chain_v2_official` | `submission_chain.py`, designed I/O (CHAIN.md) | 64 | 64 | 768 everywhere | all exact | 200/200 | (designed) | 271.8 (417) | 15.4 |

`io_official`'s schedule (the `LR` lines of its log):
- step 4: three digit runs per prompt (stage 2);
- step 101: a reading passes x < N on the whole batch for 20 steps: (N's run, read from its left; x's run, read from
  its right), the true one (stage 3);
- step 121: the mixture puts 0.99 on that reading (stage 4); the machine's phase B starts with it;
- step 261: T's direction: the 64 rows tried both ways; the snapped reading ("16") reproduced all 64 answers and
  carried all the weight (stage 5);
- step 440: frozen (200 consecutive steps with every check exact).

The snapped I/O it printed at its first evaluation batch:
- N: anchor token 2 (the N marker), step +1;
- x: anchor token 4 (the T marker), step −1;
- T and the writer: anchor END (the prompt's end), step −1;
- digit tokens 7…16, valued 0…9; the writer writes value v as token 7 + v.

`io_local/summarize_io.py` rebuilds this reader outside the submission from those lines. It reads N, x and T exactly
on all 768 prompts of `depth_t_1`, `depth_ood_n_t_1` and `depth_ood_n_t_64`, the 110 eight-digit moduli included, and
writes every true answer at its target positions.

The same submission through the same unmodified evaluator on Modal CPU containers (the manifest's device set to cpu,
everything else official, one seed per run, `plans/io1_cpu.json`):

| seed | certified T = 64, both profiles | subtractor rows right | reading chosen (stage 4) at step | T's direction settled at step | frozen at step | training s / evaluation s (CPU) |
|---|---|---|---|---|---|---|
| 74 | yes | 200/200 | 121 | 261 | 440 | 454.7 / 29.2 |
| 1 | yes | 200/200 | 122 | 341 | 457 | 377.3 / 18.9 |
| 2 | yes | 200/200 | 104 | 304 | 483 | 451.0 / 26.3 |
| 3 | yes | 200/200 | 123 | 269 | 448 | 416.8 / 25.7 |
| 4 | yes | 200/200 | 168 | 355 | 460 | 508.5 / 25.6 |
| 5 | yes | 200/200 | 90 | 326 | 505 | 441.5 / 23.0 |
| 6 | yes | 200/200 | 110 | 265 | 444 | 353.0 / 19.0 |
| 7 | yes | 200/200 | 101 | 260 | 439 | 439.9 / 28.1 |

- **Every run:** test 9,999/9,999, ood_t and ood_n_t 10,002/10,002, `AUDIT eval_rows_used=200/200
  used_but_never_trained=0`, and the same snapped I/O as `io_official`. The offline check reads every prompt of
  `depth_t_1`, `depth_ood_n_t_1` and `depth_ood_n_t_64` exactly.
- **Seed 74** took the same path on CPU as on the H100 (stages at steps 4, 101, 121 and 261, frozen at 440). As in
  CHAIN.md, nothing random enters this training after the initialization.
- **T's direction:** only seed 6 entered T's lesson on the wrong reading ("61", with all the weight on it); the
  answers turned it by step 265. In the other seven the snapped direction was already right and the lesson only
  confirmed it.
- **For comparison,** CHAIN.md's final recipe froze at steps 417–481 on its 8 CPU seeds, with 306–502 s of training
  and 18–31 s of evaluation.

Locally (`io_local/local_io.py train 2000 128 SEED`, batch 128, CPU): seeds 1, 2 and 3 froze at steps 433, 509 and
565, and certified locally (seed 1 checked on all 14 rungs, seeds 2 and 3 at T = 64 on both profiles).

## 2. What is designed and what is learned

| part | `submission_chain.py` | `submission_io.py` |
|---|---|---|
| find N's, x's and T's digits in the prompt | designed `parse`: marker ids 2, 3, 4, field extents, directions | **learned** pointers: per field an anchor (a token type, or the prompt's end) and a step (+1/−1) |
| the reader's grammar | (designed) | designed: slot j at anchor + (j + 1) · step, stop at the first non-digit, later slots read 0 |
| which tokens are digits | designed (7…16) | **learned** (one logit per token) |
| a digit token's value | designed (token − 7) | **learned**: per reading hypothesis a score per token; the value is the rank among the digit tokens |
| the slot counts (8, 8, 2; 8 out) | designed | designed |
| digits → integers x and T, and the square's digits | designed (place value) | designed (unchanged: part of the designed squaring) |
| write the answer right-aligned | designed (position end − 1 − j, token 7 + digit) | **learned**: the writer's own pointer and value → token logits |
| read the rows' answers for the machine's lesson | designed (label − 7) | **learned**: read at the writer's positions with the reader's code |
| the machine (squaring, long division, chain, stop rule, loop over T) and its lessons | as CHAIN.md | unchanged; phase B waits for the reading |

The model state is 3,075 numbers: the subtractor's 2,400 logits, and the I/O's 675 (17 digit-ness logits, 24 × 17
scores, 10 × 17 writer logits, 4 × 18 anchor logits, 4 × 2 step logits).

## 3. The reader and the writer

The prompt is `N <N, lowest digit first> X <x, highest first> T <T, highest first>`, and the answer's digits go on its
last positions, highest first. The reader sees the token ids and the padding mask, and adds two virtual tokens: END
at the position just past the prompt, OUT beyond it.

**A field's pointer.** Each of the four fields (N, x, T, the writer) has:
- an anchor: a learned distribution over token types (17 ids and END). At evaluation, the anchor is the first
  position holding the most likely type;
- a step: a learned choice of +1 or −1.

Slot j sits at anchor + (j + 1) · step. A reading field takes the token there while every token from the anchor's
neighbour up to it is a digit (the learned digit-ness), and reads 0 after the first non-digit. The writer writes slot
j of the machine's final state at its slot j's position.

**Why one shared step matters.** In training N never has 8 digits, x has 7 digits in only 2 of 101,999 rows, and the
answer never has 8. With one step per field, slot 7 is the same rule applied once more, so it needs no training of
its own. §6 checks this against the alternatives.

**The code.** The machine needs digit values: the integer square reads x as a number, and the subtractor table is
indexed by N's digits. For each of 24 reading hypotheses (which of the prompt's three digit runs the N and x fields
read, in which direction), the model keeps one score per token. A digit token's value is its rank among the digit
tokens' scores, so a code is always a one-to-one assignment of the ten digit tokens to 0…9. The reader uses the code
of the hypothesis its snapped N and x pointers make.

**The writer's code.** Value v is written as the token maximizing `wc[v, t] + log sigmoid(digit-ness[t])`, so only
tokens believed to be digits can be written.

## 4. How the I/O learns

Five stages, each started by a check on the snapped model (Python-side schedule, like the machine's phases):

| stage | lessons added | starts when |
|---|---|---|
| 1 | the answer round trip; the token-side code round trip | at once |
| 2 | Bradley-Terry ranking, every hypothesis | the snapped digit-ness splits every prompt into three digit runs (99% of the batch) |
| 3 | the reading: a normalized mixture over complete readings; the value-side code round trip | a hypothesis passes x < N on ≥ 99.9% of the batch, 20 steps in a row |
| 4 | the tiling | the mixture's mean posterior on one reading ≥ 0.99 |
| 5 | (T's direction is settled) | the snapped T reading reproduces the answers of the rows tried both ways and carries ≥ 0.999 of the weight |

The machine's lessons are CHAIN.md's, fed by the snapped reader: phase A from the start, phase B once the link checks
pass and stage 4 is reached. The model freezes after 200 consecutive steps in which:
- every link check and "x mod N = x" is exact;
- the answer round trip and the code round trips are exact on the batch;
- every row is read as two whole, distinct runs for N and x;
- stage 5 is reached.

**The answer round trip.** The answer's tokens are placed at their target positions (given by the evaluator). Per
row, the loss is a mixture over the writer's programs (every anchor position, both steps). Each program reads the
answer at its slots, writes it back there and must reproduce every answer token. A slot that reads a non-digit earns
nothing. This teaches:
- the writer's pointer;
- that the answer's tokens are digits;
- through the writer's `log digit-ness` term, that the other tokens are not: a marker that counted as a digit would
  take writing probability away from the answer's own token.

**The code round trips.** Both are taught against the reader's code, held fixed:
- read an answer token and write it back;
- write the digits of the machine's own square of x and read them back.

Together they teach the writer's code.

**Bradley-Terry on x < N.** For each hypothesis and each row, both readings are compared slot by slot from the top,
and the decisive place is the highest one where their tokens differ. There, x's token must rank below N's: the loss
is −log σ((u_N − u_x)/τ) with τ = 1. If x's reading ends first, the decisive "pair" is x's padding against N's digit,
and N's digit must not be the lowest-ranked token. This is the place where x − N sets its final borrow.

**The reading.** Per row, a mixture over complete readings (which run each of the three fields takes, and the
directions). Each reading's weight is the product of the fields' pointer probabilities for those whole-run programs.
Its likelihood is the exact verdict of its hypothesis's code on x < N (1 or 10⁻⁶). The mixture is normalized over
these readings, so its gradient is "posterior minus prior": weight moves only towards readings that explain more
rows.

**The tiling.** From stage 4, −log of the readings' total weight: each field must read one whole run, the three
fields different runs, and T's two slots can only hold a run of at most two digits.

**T's direction.** Every lesson so far is indifferent to it: one-digit T reads the same both ways, and x < N does not
involve T. Once the machine has been exact for 20 steps in a row, each step takes up to 64 rows whose two readings of
T differ (in training, the T = 16 rows). The exact machine runs to the larger count (61), and the state after 16 and
after 61 squarings is compared with the answer, read by the learned answer reader. The loss is the normalized mixture
over T's two whole-run programs, scored by those verdicts.

## 5. Pinning the digit meanings: which wrong-but-consistent mappings each lesson leaves

Counted on the training rows by `io_local/pinning.py` (20,000 random rows for (a), (c), (d); all rows for (e)):

| lesson | pins | wrong-but-consistent mappings it still allows | excluded by |
|---|---|---|---|
| answer round trip | the writer's pointer: only (END, step −1) reproduces every answer (100% of rows; every other program ≤ 6.5%); the ten answer tokens are digits | any code shared consistently by reading and writing; any way of reading the prompt | the lessons below |
| code round trips | the writer's code = the reader's code run backwards | all 10! = 3,628,800 relabelings of the values applied to both | x < N |
| tiling | each field reads a whole run; T's two slots take T's run (any other assignment fits ≤ 0.65% of rows) | N's and x's runs swapped; each of the three fields read in either direction: 16 readings | x < N; T's direction by the answers |
| x < N (ranking + speculation) | the code is the identity: every one of the 45 digit pairs is decisive somewhere (each in ≥ 133 rows), so a code passing on every row is strictly increasing on all of them; N read from its left end, x from its right end (both lowest digit first into slot 0) | T's direction | the answers of the T = 16 rows through the exact machine |
| answers at T = 16 | T read from its right end (the prompt's end) | none | – |

The readings that come closest under x < N, each with its best ranking (Bradley-Terry fit, `pinning.py` (d)):

| reading (N's run, N's direction, x's run, x's direction; direction 0 = step +1) | share of rows where x < N holds and T fits | its best code (digits 0…9 → values) |
|---|---|---|
| (0, 0, 1, 1): the true reading | 1.0000 | 0123456789 |
| (0, 1, 1, 1): N read from the wrong end | 0.8985 | 5706124839 |
| (0, 1, 1, 0): N and x both read from the wrong end | 0.8501 | 0937425618 |
| (0, 0, 1, 0): x read from the wrong end | 0.8143 | 2105987643 |
| (1, 1, 0, 0): N and x swapped | 0.8033 | 9876543210 |
| the other 19 | three more N/x readings at 0.7017, 0.6535, 0.6176; the 16 that put x's or N's run into T's two slots at most 0.0065 | |

The swapped reading with the reversed code 9876543210 shows the symmetry x < N cannot break on equal-width rows. It
fails where x is shorter than N, because the padding is a fixed 0 that cannot be ranked above a digit.

T read as "61" instead of "16": the answer would also be x^(2^61) mod N on 78 of the 33,333 training rows with T = 16
(elements of small order), so the answers reject it on 99.8% of them.

**Why the order lesson is needed here at all.** In a machine whose arithmetic were all learned, a relabeling applied
consistently to every table, the reader and the writer would be harmless: the round trips would be enough. Here the
squaring and the place values are designed integer arithmetic, so the digit values must be the true ones. Only the
answers through that arithmetic pin them directly, and that path is chaotic. "x < N" is the short path: one comparison
of two read numbers, using a fact the data guarantees.

## 6. Does the reader work on 8 digits?

**On the certification prompts.** The snapped reader of `io_official` reads all 110 eight-digit moduli per OOD-N rung
exactly, and every 8-digit x and every 8-digit answer (offline check, §1). The certification itself confirms it: both
profiles certified at T = 64.

**Against the alternatives** (`io_local/contrast.py`). Three pointer mechanisms for the N field, the x field and the
writer. Each is trained with the strongest possible teacher, supervised on the true slot positions of 300 batches of
256 training prompts:
- a live slot is taught its true position;
- the first slot past a number's end is taught to land on a non-digit.

Then each reads all 768 prompts of `depth_ood_n_t_1`. Three seeds each:

| mechanism | N: all prompts / the 110 with 8 digits | x: all / the 24 with 8 digits | writer: all / the 17 8-digit answers |
|---|---|---|---|
| **one anchor, one shared step** (`submission_io.py`) | **100% / 100%** (3/3 seeds) | **100% / 100%** | **100% / 100%** |
| one anchor, an own offset per slot | 85.7% / **0%** | 60.8% / **0%** | 97.8% / **0%** |
| an own absolute position per slot | 85.7–96.0% / 0–71.8% | 0–1.4% / 0% | 0% / 0% |

- **An own offset per slot** reads every number of up to 7 digits and no 8-digit one. Training never gives slot 7 a
  live digit, so its offset is only taught to land somewhere dead, and it does. For x it also fails the 277 seven-digit
  ones: slot 6 is live in 2 training rows.
- **Absolute positions** cannot follow x and the answer, whose positions move with N's width. N's field always starts
  at position 1, so for N they work up to 7 digits and then depend on where slot 7's training left it.
- **One shared step** has no slot-specific parameter: slot 7 is one more step.

## 7. What broke and why

Most of this happened in a reader-only lab, before the submission was written. There, the exact subtraction table is
the comparator, so only the I/O learns. The numbers are from those runs, batch 128–256, CPU. `io_local/lab/` keeps the
final states of the lab scripts:
- `reader_lab5.py`: item 7;
- `bt_only.py`: item 6;
- `code_only2.py`: item 5;
- `reader_lab6.py`: items 6–8.

Items 1–4 come from earlier states of the same lab that were edited in place and are not kept.

1. **Per-position lessons reward smeared readers.** The first version scored the round trip and "every digit is read"
   position by position. A uniform anchor plus both steps covers every position a little, so those losses fell while
   the snapped reader read nothing right (0% after 300 steps). The x field settled on T's run (T < N always) and N on
   x's run. Fix: score whole readings per row: a mixture over programs, each of which must produce the whole answer
   or the whole tiling.
2. **Dead slots earned credit.** A slot past the answer's start reads nothing and writes "0"; at random start that
   matches an answer digit as often as a live slot does. The writer's anchor drifted to the X marker with step +1.
   Fix: only live slots that write the answer's token back count. The writer then found (END, −1) by step ~100.
3. **Marker digit-ness.** "Whatever is read as a digit must write back as itself" pushed every token's digit-ness down
   while the code was still random: nothing round-trips at first, and "not a digit" is always safe. Fix: the writer can
   only write digits (`+ log digit-ness`), so the answer round trip pushes non-answer tokens out ("explaining away").
4. **The code by round trips alone.** With a free 17 × 10 code table, the round trips stalled with two tokens on one
   value and another value unused. One case: digit tokens 0…9 coded 0,1,2,7,3,6,7,4,5,9 (two tokens on 7, none on 8),
   with the answer round trip stuck at 53–61% of rows. Adding the value-side round trip, or writing by Bayes' rule
   from the reader's code, did not unstick it: a token holding two values half-and-half passes soft round trips, and
   moving out of a collision needs several tokens to move at once.
5. **The code by "x < N" with a free table** (true reading fixed): an ordered code with ties, 0123567899, which passes
   x < N on 99.6–100% of rows. Strictness is invisible when a tie only affects one rare pair. Soft ranks squeezed into
   the middle (2223456778). Round trips added after the order fixed most but stopped one swap short (0123456798 in 4
   of 6 seeds). Sinkhorn normalization gave wrong orders. These runs also had the data bug below.
6. **Bradley-Terry on the decisive pair** (true reading fixed, all rows): the identity in 6 of 6 seeds by step 600,
   with brief flicker on the rarest pair (1, 2), decisive in 133 of 82,268 equal-width rows.
7. **The trap of reading both backwards.** With the pointers and the code learned together, 2 of 3 seeds settled on N
   and x both read from the wrong end. Their code ranked the odd digits high, and x < N held on ~85% of rows: when
   both are reversed, the comparison is decided by the last digits, and N's last digit is always 1, 3, 7 or 9. Fix: one
   ranking per reading hypothesis (speculation), so a wrong reading cannot borrow the true reading's progress, and the
   readings compete on their own best codes.
8. **Early commitment.** With per-hypothesis codes but the mixture scored by soft Bradley-Terry probabilities, then by
   exact verdicts, 3 of 3 and then 3 of 6 seeds still picked a wrong reading. The tiling term pushes up every whole-run
   program in proportion to its current weight, so whichever direction random initialization favoured got amplified
   before the 10–20% likelihood gap could act. Fix: the normalized mixture first (only relative evidence moves the
   weights), the tiling only once one reading carries 0.99. 8 of 8 lab seeds then found the true reading and the
   identity code.
9. **T's direction.** In the first local training run (seed 1), the tiling had pushed T's pointer to 99.6% on the
   wrong direction ("61") by the time T's lesson started. The answers reversed it in about 75 steps. In `io_official`
   the snapped direction was already right and the lesson only confirmed it (20 steps).
10. **The data bug.** The early lab runs (items 4 and 5) used the first 30,000 rows of `train.jsonl`. The file is
    ordered by modulus size: those rows hold only 13- and 16-bit moduli, so N never started with 9 there. All later
    runs use all 101,999 rows.

## 8. Gray areas against the rules

Same as CHAIN.md §6, plus:
- **The answers as data in a loss.** The answer round trip places the label tokens at their target positions and reads
  them with the model's own reader. The forward pass never sees labels; the loss does.
- **Speculation.** The reading and T's direction are chosen by likelihood mixtures over complete hypotheses (24
  readings; T's two directions), each scored by an exact verdict: x < N under the hypothesis's code, or the exact
  machine's answer. The gradient reaches the pointers through the mixture weights, not through the verdicts. This is
  REPORT.md's "speculation over complete machines" applied to the I/O.
- **Designed knowledge in the lessons.** The decisive-place rule (where x − N sets its final borrow) is how decimal
  comparison works. That x is a reduced residue (x < N) is a guarantee of the task's inputs. The machine's own
  "x mod N = x" check, with the learned subtractor, re-checks the comparison inside the freeze condition.
- **The 24 hypotheses are defined over digit runs.** The lesson knows the prompt has digit runs separated by
  non-digits (from the learned digit-ness), but not which run is which number, which direction, or what any token
  means.
- **The grammar of a reader** (anchor, one step, stop at a non-digit) and the slot counts are designed. So are the
  integer conversions around the designed square: the reader delivers digits, and the squaring turns x's digits into
  an integer as before.

## 9. Reproduce

Everything heavy runs on Modal. Launch from the one-layer-deeper repository root; records land on the Volume
`learned-remainder`, and `wait_fetch.py` downloads them to `cpu_first/runs/learned_remainder/<name>.json`.

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name io_official --file submission_io.py
```

The plumbing check:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode official --name io_s0_official --file submission_io.py --set '{"MODE": "\"designed\"", "IO_MODE": "\"designed\"", "MAX_STEPS": "100"}'
```

The CPU seeds (one container each):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach /path/to/cpu_first/learned_remainder/modal_chain.py --mode plan --plan cpu_first/learned_remainder/plans/io1_cpu.json --file submission_io.py
```

Fetch, then summarize with the subtractor audit and the offline check of the snapped reader:

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/wait_fetch.py io_official io_s0_official
```

```bash
~/github/sotaku/venv/bin/python cpu_first/learned_remainder/io_local/summarize_io.py io_official io_s0_official
```

Files:
- **`submission_io.py`**: the submission, certified byte for byte as `io_official` (md5 `45cc6e1b…`).
  `IO_MODE = "designed"` sets the I/O at the true reading; `MODE = "designed"` uses the exact subtraction table (both:
  the plumbing check).
- **`plans/io1_cpu.json`**: the CPU seed plan.
- **`io_local/`**: the small local checks, each under a minute and under 1 GB (they need torch and the H1 jsonl files,
  `H1_DATA` as in `cpu_first/h1.py`; `IO_CACHE` optionally caches the parsed splits outside the repository):
  - `local_io.py designed`: the plumbing check against every depth rung;
  - `local_io.py train STEPS B [SEED]`: a short training run, then every depth rung;
  - `contrast.py`: §6's three pointer mechanisms;
  - `pinning.py`: §5's counts;
  - `summarize_io.py`: run records, with the offline check of the snapped reader;
  - `lab/`: the reader-only lab of §7 (the exact subtraction table as comparator; `reader_lab6.py STEPS B SEED`,
    `bt_only.py SEED STEPS B`, `code_only2.py SEED STEPS B` with `CODE=free|rank` and friends);
  - `stub/`: stands in for the evaluator's `benchmark` API.
- **`submission_chain.py`, `submission.py`, `modal_chain.py`**: untouched (md5 `ed360cb1…`, `bd477b9a…`,
  `ad5fbd9c…`).
